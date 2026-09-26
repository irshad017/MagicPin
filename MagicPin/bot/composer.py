"""compose() — turn (category, merchant, trigger, customer?) into a grounded WhatsApp message.

Pipeline: resolve trigger references -> build a focused context slice -> LLM (strict JSON)
-> validate (CTA shape, send_as, taboo words, non-empty) -> one corrective re-prompt on failure
-> deterministic template fallback if the LLM is unavailable or keeps failing.
"""
import hashlib
import json
import re
from datetime import datetime, timezone
from typing import Any, Optional

import config
import llm_client
from prompts import COMPOSE_SYSTEM

VALID_CTA = {"binary", "open_ended", "none"}
VALID_SEND_AS = {"vera", "merchant_on_behalf"}

# Compose cache: content-addressed, so it auto-invalidates when any context version changes.
_CACHE: dict[str, dict] = {}
_CACHE_MAX = 2000


# ---------------------------------------------------------------- reference resolution
def _iter_id_values(payload: dict) -> list[str]:
    """Collect id-looking string values from a trigger payload (top_item_id, content_id, ...)."""
    ids: list[str] = []
    for k, v in (payload or {}).items():
        if isinstance(v, str) and (k == "id" or k.endswith("_id")):
            ids.append(v)
        elif isinstance(v, list):
            ids.extend(x for x in v if isinstance(x, str) and x.count("_") >= 1)
    return ids


def _find_by_id(collections: list[list], wanted: str) -> Optional[dict]:
    for coll in collections:
        for item in coll or []:
            if isinstance(item, dict) and item.get("id") == wanted:
                return item
    return None


def resolve_trigger_refs(trigger: dict, category: dict, merchant: dict) -> list[dict]:
    """Return the actual category/merchant objects a trigger points at, so the model has real facts."""
    payload = trigger.get("payload", {}) or {}
    cat_collections = [
        category.get("digest", []),
        category.get("patient_content_library", []),
        category.get("offer_catalog", []),
    ]
    mer_collections = [merchant.get("offers", [])]
    resolved: list[dict] = []
    seen = set()
    for wanted in _iter_id_values(payload):
        hit = _find_by_id(cat_collections + mer_collections, wanted)
        if hit and id(hit) not in seen:
            resolved.append(hit)
            seen.add(id(hit))
    return resolved


# ---------------------------------------------------------------- prompt building
def _compact(obj: Any, limit: int = 1500) -> str:
    s = json.dumps(obj, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + "…"


def _parse_dt(s: str) -> Optional[datetime]:
    if not s or not isinstance(s, str):
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def days_until_event(trigger: dict, now: Optional[str]) -> Optional[int]:
    """Days from `now` until the trigger's event, if a date is derivable; else None."""
    payload = trigger.get("payload", {}) or {}
    days = payload.get("days_until")
    if days is not None:
        try:
            return int(days)
        except (TypeError, ValueError):
            return None
    now_dt = _parse_dt(now) if now else None
    event_dt = _parse_dt(payload.get("date", "")) or _parse_dt(trigger.get("expires_at", ""))
    if event_dt and now_dt:
        return (event_dt - now_dt).days
    return None


def _time_context(trigger: dict, now: Optional[str]) -> str:
    """A one-line time hint so the model doesn't call far-off events 'imminent'."""
    days = days_until_event(trigger, now)
    if days is None:
        return f"today is {now[:10]}" if now else ""
    if days <= 2:
        horizon = "IMMINENT — urgency is appropriate"
    elif days <= 14:
        horizon = "near — mild time-sensitivity is fine"
    else:
        horizon = "FAR OFF — do NOT frame as urgent/coming-soon; frame as planning ahead"
    return f"today is {(now or '')[:10]}; the event is ~{days} days away ({horizon})"


def build_prompt(category: dict, merchant: dict, trigger: dict,
                 customer: Optional[dict], resolved_refs: list[dict], now: Optional[str] = None) -> str:
    ident = merchant.get("identity", {})
    voice = category.get("voice", {})
    tctx = _time_context(trigger, now)
    parts = [
        "Compose the next WhatsApp message for this exact situation.\n",
        (f"=== TIME CONTEXT ===\n{tctx}\n" if tctx else ""),
        "=== CATEGORY ===",
        f"slug: {category.get('slug')}",
        f"voice.tone: {voice.get('tone')} | code_mix: {voice.get('code_mix')}",
        f"vocab_allowed (sample): {voice.get('vocab_allowed', [])[:8]}",
        f"vocab_taboo (NEVER use): {voice.get('vocab_taboo', [])}",
        f"peer_stats: {_compact(category.get('peer_stats', {}))}",
        f"offer_catalog: {[o.get('title') for o in category.get('offer_catalog', [])]}",
        "",
        "=== MERCHANT ===",
        f"name: {ident.get('name')} | owner_first_name: {ident.get('owner_first_name')}",
        f"city/locality: {ident.get('city')} / {ident.get('locality')}",
        f"languages: {ident.get('languages')}  (if it includes 'hi', prefer Hindi-English code-mix)",
        f"subscription: {_compact(merchant.get('subscription', {}))}",
        f"performance: {_compact(merchant.get('performance', {}))}",
        f"active_offers: {[o.get('title') for o in merchant.get('offers', []) if o.get('status') == 'active']}",
        f"signals: {merchant.get('signals', [])}",
        f"customer_aggregate: {_compact(merchant.get('customer_aggregate', {}))}",
        f"review_themes: {_compact(merchant.get('review_themes', []))}",
        f"recent_conversation (last turns): {_compact(merchant.get('conversation_history', [])[-3:])}",
        "",
        "=== TRIGGER (why now) ===",
        f"kind: {trigger.get('kind')} | scope: {trigger.get('scope')} | source: {trigger.get('source')} | urgency: {trigger.get('urgency')}",
        f"payload: {_compact(trigger.get('payload', {}))}",
        f"suppression_key: {trigger.get('suppression_key')}",
    ]
    if resolved_refs:
        parts += ["", "=== RESOLVED FACTS the trigger points to (use these verbatim, don't invent) ===",
                  _compact(resolved_refs, limit=2200)]
    if customer:
        parts += [
            "", "=== CUSTOMER (message is sent on behalf of the merchant TO this customer) ===",
            f"identity: {_compact(customer.get('identity', {}))}",
            f"relationship: {_compact(customer.get('relationship', {}))}",
            f"state: {customer.get('state')}",
            f"preferences: {_compact(customer.get('preferences', {}))}",
            f"consent.scope: {customer.get('consent', {}).get('scope')}",
            "=> send_as MUST be 'merchant_on_behalf'. Write as the merchant, warm, no internal jargon.",
        ]
    else:
        parts += ["", "=> send_as MUST be 'vera' (merchant-facing)."]
    parts += ["", "Return ONLY the JSON object described in the system message."]
    return "\n".join(parts)


# ---------------------------------------------------------------- parsing + validation
def _extract_json(text: str) -> Optional[dict]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _validate(result: dict, category: dict) -> list[str]:
    problems = []
    body = (result.get("body") or "").strip()
    if not body:
        problems.append("body is empty")
    if result.get("cta") not in VALID_CTA:
        problems.append(f"cta must be one of {VALID_CTA}")
    # send_as is authoritative from customer presence — we correct rather than reject.
    taboos = [t.lower() for t in category.get("voice", {}).get("vocab_taboo", [])]
    low = body.lower()
    hit = [t for t in taboos if t and t.split(" (")[0] in low]
    if hit:
        problems.append(f"uses taboo words {hit}")
    # multiple-CTA smell
    if low.count("reply ") >= 2 or ("yes" in low and "no" in low and "reply" in low and result.get("cta") != "binary"):
        problems.append("looks like multiple CTAs; use a single primary CTA")
    return problems


def _finalize(result: dict, trigger: dict, customer: Optional[dict]) -> dict:
    result["send_as"] = "merchant_on_behalf" if customer else "vera"
    if not result.get("suppression_key"):
        result["suppression_key"] = trigger.get("suppression_key", "")
    if result.get("cta") not in VALID_CTA:
        result["cta"] = "open_ended"
    result["body"] = (result.get("body") or "").strip()
    result["rationale"] = (result.get("rationale") or "").strip()
    return result


# ---------------------------------------------------------------- fallback (keyless / on error)
def _fallback(category: dict, merchant: dict, trigger: dict,
              customer: Optional[dict], resolved_refs: list[dict]) -> dict:
    ident = merchant.get("identity", {})
    owner = ident.get("owner_first_name") or ident.get("name", "there")
    is_dentist = category.get("slug") == "dentists"
    salut = f"Dr. {owner}" if is_dentist else owner
    fact = ""
    if resolved_refs:
        r = resolved_refs[0]
        fact = r.get("title") or r.get("summary") or ""
    if customer:
        cname = customer.get("identity", {}).get("name", "there")
        active = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
        offer = active[0] if active else ""
        body = (f"Hi {cname}, {ident.get('name')} here. "
                f"{'It has been a while since your last visit — worth a quick check-in. ' if trigger.get('kind','').startswith('recall') else ''}"
                f"{('We have ' + offer + ' available. ') if offer else ''}Want me to hold a slot for you? Reply YES.")
        return _finalize({"body": body.strip(), "cta": "binary", "rationale": "Template fallback (no LLM)."},
                         trigger, customer)
    hook = fact or f"quick note on your {category.get('slug','business')} listing"
    body = f"{salut}, {hook}. Want me to take it from here? Reply YES."
    return _finalize({"body": body, "cta": "binary", "rationale": "Template fallback (no LLM)."},
                     trigger, customer)


# ---------------------------------------------------------------- public API
def _cache_key(category: dict, merchant: dict, trigger: dict, customer: Optional[dict], now: Optional[str]) -> str:
    # Content-addressed: any change to the pushed context (new version) changes the key.
    day = (now or "")[:10]
    blob = json.dumps([category, merchant, trigger, customer, day], sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()


def compose(category: dict, merchant: dict, trigger: dict,
            customer: Optional[dict] = None, now: Optional[str] = None) -> dict:
    """Return {body, cta, send_as, suppression_key, rationale}. Never raises."""
    category = category or {}
    merchant = merchant or {}
    trigger = trigger or {}
    resolved = resolve_trigger_refs(trigger, category, merchant)

    if not config.LLM_ENABLED:
        return _fallback(category, merchant, trigger, customer, resolved)

    key = _cache_key(category, merchant, trigger, customer, now)
    cached = _CACHE.get(key)
    if cached is not None:
        return dict(cached)   # copy so callers can't mutate the cache

    prompt = build_prompt(category, merchant, trigger, customer, resolved, now)
    try:
        raw = llm_client.complete(COMPOSE_SYSTEM, prompt)
        result = _extract_json(raw)
        if result is None:
            return _fallback(category, merchant, trigger, customer, resolved)
        problems = _validate(result, category)
        if problems:
            correction = (prompt + "\n\nYour previous attempt had these problems: "
                          + "; ".join(problems) + ". Fix them and return ONLY the JSON object.")
            raw2 = llm_client.complete(COMPOSE_SYSTEM, correction)
            fixed = _extract_json(raw2)
            if fixed and not _validate(fixed, category):
                result = fixed
            elif fixed:
                result = fixed  # accept best-effort; _finalize cleans obvious issues
        final = _finalize(result, trigger, customer)
        if len(_CACHE) < _CACHE_MAX:
            _CACHE[key] = dict(final)   # cache successful LLM composition only
        return final
    except Exception:
        return _fallback(category, merchant, trigger, customer, resolved)
