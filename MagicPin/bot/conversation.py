"""Multi-turn handling for /v1/reply.

Fast deterministic routing first (auto-reply / opt-out / commitment), LLM for the nuanced middle,
graceful fallbacks throughout. Returns {action, body, cta, wait_seconds, rationale}.
"""
import json
import re
from typing import Optional

import config
import llm_client
from prompts import REPLY_SYSTEM
from store import STORE

_AUTOREPLY_PATTERNS = [
    "thank you for contacting", "thanks for contacting", "automated message", "automated reply",
    "will respond shortly", "will get back to you", "team will", "out of office", "away from my phone",
    "this is an auto", "aapki jaankari ke liye", "team tak pahuncha", "automated assistant",
]
_OPTOUT_PATTERNS = [
    "stop messaging", "stop sending", "unsubscribe", "leave me alone", "don't message", "do not message",
    "not interested", "no interest", "band karo", "mat bhejo", "nahi chahiye", "remove me", "spam",
]
_COMMITMENT_PATTERNS = [
    "let's do it", "lets do it", "go ahead", "yes please", "yes, please", "do it", "proceed",
    "sounds good", "ok karo", "haan karo", "theek hai karo", "chalega", "let's go", "lets go",
    "sign me up", "i'm in", "im in", "count me in", "yes do it", "haan", "kar do",
]
_OFFTOPIC_PATTERNS = ["gst", "income tax", "loan", "personal loan", "job", "recruit"]


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def _has(text: str, patterns: list[str]) -> bool:
    return any(p in text for p in patterns)


def _extract_json(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*", "", text).strip().rstrip("`").strip()
    m = re.search(r"\{[\s\S]*\}", text)
    if not m:
        return None
    try:
        return json.loads(m.group())
    except Exception:
        return None


def _llm_reply(conv: dict, merchant: Optional[dict], customer: Optional[dict],
               extra: str = "") -> Optional[dict]:
    if not config.LLM_ENABLED:
        return None
    ident = (merchant or {}).get("identity", {})
    turns = conv["turns"][-6:]
    convo = "\n".join(f"{t['role']}: {t['text']}" for t in turns)
    langs = ident.get("languages", ["en"])
    ctx = [
        f"Merchant: {ident.get('name')} (owner {ident.get('owner_first_name')}), "
        f"category {(merchant or {}).get('category_slug')}, languages {langs}.",
    ]
    if customer:
        ctx.append(f"Customer-facing conversation. Customer: {json.dumps(customer.get('identity', {}), ensure_ascii=False)}")
    if merchant:
        active = [o.get("title") for o in merchant.get("offers", []) if o.get("status") == "active"]
        if active:
            ctx.append(f"Active offers you can reference: {active}")
    prompt = ("\n".join(ctx) + "\n\nConversation so far:\n" + convo
              + (f"\n\nGuidance: {extra}" if extra else "")
              + "\n\nReturn ONLY the JSON object described in the system message.")
    try:
        raw = llm_client.complete(REPLY_SYSTEM, prompt)
        return _extract_json(raw)
    except Exception:
        return None


def handle_reply(conv_id: str, merchant_id: Optional[str], customer_id: Optional[str],
                 from_role: str, message: str, turn_number: int, now: str = "") -> dict:
    conv = STORE.get_or_create_conversation(conv_id, merchant_id, customer_id)
    conv["state"]["nudges_unanswered"] = 0     # they replied; reset the silence counter
    STORE.add_turn(conv_id, from_role, message, now)
    norm = _norm(message)
    merchant = STORE.get_context("merchant", merchant_id) if merchant_id else None
    customer = STORE.get_context("customer", customer_id) if customer_id else None

    # 1) Auto-reply detection — canned text, or same verbatim repeated in this conversation.
    verbatim_repeats = sum(1 for t in conv["turns"] if t["role"] != "vera" and _norm(t["text"]) == norm)
    is_canned = _has(norm, _AUTOREPLY_PATTERNS)
    if is_canned or verbatim_repeats >= config.AUTOREPLY_REPEAT_THRESHOLD:
        seen = STORE.bump_autoreply(merchant_id)
        if seen >= 2 or verbatim_repeats >= config.AUTOREPLY_REPEAT_THRESHOLD:
            conv["state"]["ended"] = True
            return {"action": "end", "body": "", "cta": "none", "wait_seconds": 0,
                    "rationale": "Detected WhatsApp auto-reply; stopping to avoid burning turns."}
        # First time: one gentle human-routing nudge, then we exit if it repeats.
        body = ("Samajh gayi — looks automated. Owner/manager ko 30 sec lagega dekhne mein. "
                "Reply YES and I'll keep it short.") if "hi" in (merchant or {}).get("identity", {}).get("languages", []) \
            else "Looks automated — I'll keep it to 30 seconds for the owner/manager. Reply YES and I'll continue."
        STORE.record_sent(conv_id, body)
        STORE.add_turn(conv_id, "vera", body, now)
        return {"action": "send", "body": body, "cta": "binary", "wait_seconds": 0,
                "rationale": "Likely auto-reply; single human-routing nudge before exiting."}

    # 2) Explicit opt-out / hostile — respect it, exit gracefully.
    if _has(norm, _OPTOUT_PATTERNS):
        conv["state"]["ended"] = True
        return {"action": "end", "body": "", "cta": "none", "wait_seconds": 0,
                "rationale": "Merchant opted out / not interested; exiting politely."}

    # 3) Commitment / intent handoff — switch straight to action, never re-qualify.
    if _has(norm, _COMMITMENT_PATTERNS):
        conv["state"]["intent_locked"] = True
        llm = _llm_reply(conv, merchant, customer,
                         extra="The user just committed. Do NOT ask any qualifying question. "
                               "Confirm and take the next concrete action now.")
        if llm and llm.get("action") == "send" and (llm.get("body") or "").strip():
            body = llm["body"].strip()
        else:
            body = ("Done — I'll draft it now and send it here for your confirm. "
                    "Proceeding with the next step right away.")
        STORE.record_sent(conv_id, body)
        STORE.add_turn(conv_id, "vera", body, now)
        return {"action": "send", "body": body, "cta": llm.get("cta", "open_ended") if llm else "open_ended",
                "wait_seconds": 0, "rationale": "Explicit commitment detected; moved to action mode."}

    # 4) Off-topic / out-of-scope — decline politely, stay on mission once.
    if _has(norm, _OFFTOPIC_PATTERNS):
        body = ("That's outside what I handle, but happy to help you grow on magicpin — "
                "want me to pick up where we left off? Reply YES.")
        llm = _llm_reply(conv, merchant, customer, extra="Politely decline the off-topic ask, offer your real help once.")
        if llm and (llm.get("body") or "").strip() and llm.get("action") == "send":
            body = llm["body"].strip()
        STORE.record_sent(conv_id, body)
        STORE.add_turn(conv_id, "vera", body, now)
        return {"action": "send", "body": body, "cta": "binary", "wait_seconds": 0,
                "rationale": "Off-topic request; declined and re-anchored on mission."}

    # 5) General case — let the LLM decide send/wait/end.
    llm = _llm_reply(conv, merchant, customer)
    if llm and llm.get("action") in ("send", "wait", "end"):
        action = llm["action"]
        body = (llm.get("body") or "").strip()
        if action == "send" and not body:
            action = "wait"
        if action == "send":
            if STORE.already_sent(conv_id, body):
                body += " "  # avoid verbatim-repeat penalty
            STORE.record_sent(conv_id, body)
            STORE.add_turn(conv_id, "vera", body, now)
        return {"action": action, "body": body, "cta": llm.get("cta", "open_ended"),
                "wait_seconds": int(llm.get("wait_seconds", 0) or 0),
                "rationale": llm.get("rationale", "")}

    # 6) Fallback with no LLM — a safe, specific acknowledgement.
    body = "Got it — noting that. Want me to take the next step for you? Reply YES."
    STORE.record_sent(conv_id, body)
    STORE.add_turn(conv_id, "vera", body, now)
    return {"action": "send", "body": body, "cta": "binary", "wait_seconds": 0,
            "rationale": "Fallback acknowledgement (no LLM)."}
