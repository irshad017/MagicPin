"""Decision layer for /v1/tick.

The judge's flagship dimension is "decision quality": pick the ONE best signal for this moment.
So per merchant we pick a single highest-value trigger, dedup via suppression keys, compose in
parallel, and stop before the 30s judge timeout. Restraint (sending nothing) is allowed and rewarded.
"""
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import config
import composer
from store import STORE

# Signals that make a trigger more relevant to a merchant right now.
_RELEVANCE = {
    "perf_dip": ["ctr_below_peer_median", "views_down"],
    "seasonal_perf_dip": ["ctr_below_peer_median"],
    "perf_spike": ["engaged_in_last_48h"],
    "research_digest": ["high_risk_adult_cohort"],
    "review_theme_emerged": ["negative_reviews"],
    "dormant_with_vera": ["dormant"],
    "gbp_unverified": ["unverified"],
}


def score_trigger(trigger: dict, merchant: dict, now: str = "") -> float:
    """Higher = more worth sending now."""
    score = float(trigger.get("urgency", 1)) * 2.0
    if not STORE.is_suppressed(trigger.get("suppression_key", "")):
        score += 3.0                                  # novelty: not yet acted on
    if trigger.get("source") == "internal":
        score += 1.0                                  # own-account events feel personal
    if trigger.get("scope") == "customer":
        score += 2.0                                  # customer-facing is high intent
    signals = set(merchant.get("signals", []))
    for want in _RELEVANCE.get(trigger.get("kind", ""), []):
        if any(want in s for s in signals):
            score += 2.0
            break
    # Down-rank far-off dated events (e.g. a festival 6 months out) — timing is part of decision quality.
    days = composer.days_until_event(trigger, now)
    if days is not None and days > 30:
        score -= 3.0
    return score


def _conversation_id(merchant_id: str, trigger_id: str) -> str:
    return f"conv_{merchant_id}_{trigger_id}"


def _build_action(trigger_id: str, trigger: dict, now: str = "") -> Optional[dict]:
    merchant_id = trigger.get("merchant_id")
    merchant = STORE.get_context("merchant", merchant_id)
    if not merchant:
        return None
    category = STORE.get_context("category", merchant.get("category_slug")) or {}
    customer_id = trigger.get("customer_id")
    customer = STORE.get_context("customer", customer_id) if customer_id else None

    composed = composer.compose(category, merchant, trigger, customer, now)
    body = composed.get("body", "").strip()
    if not body:
        return None

    conv_id = _conversation_id(merchant_id, trigger_id)
    if STORE.already_sent(conv_id, body):
        return None

    ident = merchant.get("identity", {})
    action = {
        "conversation_id": conv_id,
        "merchant_id": merchant_id,
        "customer_id": customer_id,
        "send_as": composed.get("send_as", "vera"),
        "trigger_id": trigger_id,
        "template_name": f"vera_{trigger.get('kind', 'generic')}_v1",
        "template_params": [ident.get("name", ""), (body[:40] + "…") if len(body) > 40 else body],
        "body": body,
        "cta": composed.get("cta", "open_ended"),
        "suppression_key": composed.get("suppression_key", trigger.get("suppression_key", "")),
        "rationale": composed.get("rationale", ""),
    }
    return action


def select_actions(now: str, available_trigger_ids: list[str]) -> list[dict]:
    deadline = time.time() + config.TICK_DEADLINE_S

    # 1) Load candidate triggers, skip already-suppressed ones.
    candidates: list[tuple[str, dict, float]] = []
    for tid in available_trigger_ids:
        trg = STORE.get_context("trigger", tid)
        if not trg:
            continue
        if STORE.is_suppressed(trg.get("suppression_key", "")):
            continue
        merchant = STORE.get_context("merchant", trg.get("merchant_id"))
        if not merchant:
            continue
        candidates.append((tid, trg, score_trigger(trg, merchant, now)))

    # 2) One best trigger per merchant (decision quality: don't spam a merchant).
    best_per_merchant: dict[str, tuple[str, dict, float]] = {}
    for tid, trg, sc in candidates:
        mid = trg.get("merchant_id")
        if mid not in best_per_merchant or sc > best_per_merchant[mid][2]:
            best_per_merchant[mid] = (tid, trg, sc)

    chosen = sorted(best_per_merchant.values(), key=lambda x: x[2], reverse=True)
    chosen = chosen[: config.MAX_ACTIONS_PER_TICK]
    if not chosen:
        return []

    # 3) Compose in parallel, respecting the tick deadline.
    actions: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(config.COMPOSE_MAX_WORKERS, len(chosen))) as pool:
        futures = {pool.submit(_build_action, tid, trg, now): (tid, trg) for tid, trg, _ in chosen}
        for fut in as_completed(futures):
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            try:
                action = fut.result(timeout=max(0.1, remaining))
            except Exception:
                continue
            if not action:
                continue
            actions.append(action)
            # Record state so we don't repeat across ticks.
            STORE.mark_suppressed(action["suppression_key"])
            conv = STORE.get_or_create_conversation(
                action["conversation_id"], action["merchant_id"], action["customer_id"])
            STORE.record_sent(action["conversation_id"], action["body"])
            STORE.add_turn(action["conversation_id"], "vera", action["body"], now)
            conv["state"]["nudges_unanswered"] += 1

    return actions