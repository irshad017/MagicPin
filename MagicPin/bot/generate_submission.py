"""Generate submission.jsonl — one composed message per (merchant, trigger) pair in the dataset.

Uses the same composer the live bot uses, so the static artifact matches runtime behavior.
Paced to stay under free-tier rate limits. Output: bot/submission.jsonl

Usage (from magicpin-ai-challenge/):  bot/.venv/Scripts/python.exe bot/generate_submission.py
"""
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHALLENGE = HERE.parent
sys.path.insert(0, str(HERE))         # for composer/config
sys.path.insert(0, str(CHALLENGE))    # for judge_simulator dataset loader

import judge_simulator as J  # noqa: E402
import composer               # noqa: E402

PACE_S = 3.0
NOW = "2026-04-26T10:00:00Z"   # the dataset's implied "current time" (matches trigger day-counts)
OUT = HERE / "submission.jsonl"


def main():
    ds = J.DatasetLoader(J.DATASET_DIR)
    ds.load()
    triggers = list(ds.triggers.values())
    print(f"Composing {len(triggers)} messages -> {OUT.name}")

    lines = []
    for i, trg in enumerate(triggers, 1):
        mer = ds.merchants.get(trg.get("merchant_id"), {})
        cat = ds.categories.get(mer.get("category_slug", ""), {})
        cust = ds.customers.get(trg.get("customer_id")) if trg.get("customer_id") else None
        # Retry if a rate-limit forced a template fallback, so no line is left generic.
        c = composer.compose(cat, mer, trg, cust, NOW)
        for _ in range(2):
            if "Template fallback" not in c["rationale"]:
                break
            time.sleep(8)
            c = composer.compose(cat, mer, trg, cust, NOW)
        line = {
            "test_id": f"T{i:02d}",
            "merchant_id": trg.get("merchant_id"),
            "trigger_id": trg.get("id"),
            "body": c["body"],
            "cta": c["cta"],
            "send_as": c["send_as"],
            "suppression_key": c["suppression_key"],
            "rationale": c["rationale"],
        }
        lines.append(line)
        print(f"  [{i:02d}/{len(triggers)}] {trg.get('id')[:38]:38} -> {c['send_as']:18} cta={c['cta']}")
        time.sleep(PACE_S)

    with open(OUT, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(json.dumps(ln, ensure_ascii=False) + "\n")
    print(f"\nWrote {len(lines)} lines to {OUT}")


if __name__ == "__main__":
    main()
