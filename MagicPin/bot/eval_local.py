"""Paced local evaluation: compose + judge-score a representative set of triggers, one at a time.

Groq free tier rate-limits bursts, and the judge_simulator's scoring competes with the bot's
composing for the same key. So we pace: one compose, one score, a short sleep between, which
yields TRUE per-dimension scores instead of 429-driven fallbacks.

Usage (from magicpin-ai-challenge/):  bot/.venv/Scripts/python.exe bot/eval_local.py
"""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parent
sys.path.insert(0, str(HERE))       # for composer/config
sys.path.insert(0, str(CHALLENGE))  # for judge_simulator
import judge_simulator as J  # noqa: E402
import composer               # noqa: E402

PACE_S = 12.0   # sleep between LLM calls to stay under free-tier per-minute RPM

# A representative spread: multiple categories + at least one customer-facing trigger.
DEFAULT_PICKS = [
    "trg_001_research_digest_dentists",       # dentist / research (merchant)
    "trg_003_recall_due_priya",               # dentist / recall (customer-facing)
    "trg_006_festival_diwali",                # salon / festival (merchant)
    "trg_011_review_theme_late_delivery",     # restaurant / review theme (merchant)
    "trg_015_winback_rashmi",                 # gym / winback (customer-facing)
    "trg_018_supply_atorvastatin_recall",     # pharmacy / supply alert (merchant)
    "trg_023_competitor_opened_dentist",      # dentist / competitor (merchant)
]


def main():
    llm = J.create_provider()
    ds = J.DatasetLoader(J.DATASET_DIR)
    ds.load()
    print(f"Loaded {len(ds.categories)} cats, {len(ds.merchants)} merchants, {len(ds.triggers)} triggers")

    scorer = J.LLMScorer(llm, ds)
    picks = [p for p in DEFAULT_PICKS if p in ds.triggers] or list(ds.triggers.keys())[:6]
    NOW = "2026-04-26T10:00:00Z"

    scores = []
    for tid in picks:
        trg = ds.triggers.get(tid, {})
        mer = ds.merchants.get(trg.get("merchant_id"), {})
        cat = ds.categories.get(mer.get("category_slug", ""), {})
        cust = ds.customers.get(trg.get("customer_id")) if trg.get("customer_id") else None
        # Compose directly (retry past rate-limit fallbacks so the score reflects real quality).
        a = composer.compose(cat, mer, trg, cust, NOW)
        for _ in range(3):
            if "Template fallback" not in a["rationale"]:
                break
            time.sleep(10)
            composer._CACHE.clear()
            a = composer.compose(cat, mer, trg, cust, NOW)
        time.sleep(PACE_S)
        s = scorer.score(a, cat, mer, trg, cust)
        scores.append(s)
        print(f"\n=== {tid}  ->  {a['send_as']} | cta={a['cta']} ===")
        print("BODY:", a["body"])
        print(f"  spec={s.specificity} cat={s.category_fit} mer={s.merchant_fit} "
              f"dec={s.decision_quality} eng={s.engagement_compulsion}  TOTAL={s.total}/50")
        if s.hint:
            print("  hint:", s.hint)
        time.sleep(PACE_S)

    if scores:
        n = len(scores)
        avg = lambda f: sum(getattr(s, f) for s in scores) / n
        dims = ["specificity", "category_fit", "merchant_fit", "decision_quality", "engagement_compulsion"]
        print("\n" + "=" * 60)
        print(f"AVERAGES over {n} messages:")
        for d in dims:
            print(f"  {d:22} {avg(d):.1f}/10")
        print(f"  {'TOTAL':22} {sum(avg(d) for d in dims):.1f}/50")


if __name__ == "__main__":
    main()
