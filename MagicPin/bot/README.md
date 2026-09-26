# Vera Reforged — magicpin AI Challenge submission

A stateful HTTP bot that plays magicpin's merchant-growth assistant **Vera**: it receives the
4-layer context (category / merchant / trigger / customer), decides the single best moment to speak,
and composes a grounded, single-CTA WhatsApp message — then handles the multi-turn reply.

## Approach

**Decision-first, then compose.** The judge's flagship dimension is *decision quality*, so the tick
handler (`decision.py`) scores candidate triggers per merchant (`urgency × novelty × signal-relevance`),
picks **one** best trigger per merchant, dedups via `suppression_key`, and stays silent when nothing is
worth sending. It never fires two messages at one merchant in a tick.

**Grounded composition.** `composer.py` resolves the trigger's references (e.g. `payload.top_item_id`)
into the real category/merchant objects, builds a focused context slice, and asks the LLM for strict
JSON. A validation pass checks CTA shape, category taboo words, and single-CTA discipline, with one
corrective re-prompt. `send_as` is derived authoritatively from customer presence. Everything degrades
to a deterministic template if the LLM is unavailable, so the bot never returns malformed output.

**Multi-turn routing** (`conversation.py`): fast deterministic detection of auto-replies (canned phrases
or verbatim repeats → one human-routing nudge, then graceful `end`), explicit opt-out/hostile (→ `end`),
and intent commitment (→ switch straight to action mode, never re-qualify). The nuanced middle is handled
by the LLM with a rubric-encoded system prompt.

**Robustness for the harness:** idempotent versioned context store, `contexts_loaded` health counts,
bounded 429/5xx backoff+retry, a hard per-tick deadline that returns `{"actions": []}` rather than time
out, anti-repetition, and `/v1/teardown` to wipe state.

**Model:** provider-agnostic client (`llm_client.py`). Configured here for Groq `openai/gpt-oss-120b`
at `temperature=0` for determinism; switch provider/model in `.env` (one line) to run on
OpenAI / Gemini / OpenRouter / Anthropic-via-OpenRouter.

## Results (local `judge_simulator`)

Paced 7-trigger evaluation across all 5 categories (2 customer-facing): **≈39.9/50 (80%)**, no
hallucinated data. Per dimension: specificity 7.6, category-fit 8.3, merchant-fit 8.1,
decision-quality 8.1, engagement 7.7.

## Tradeoffs

- **Free-tier rate limits.** On a shared free key the composer and the judge compete for RPM; the bot
  uses backoff+retry and low tick-concurrency. On paid infra, concurrency can be raised in `config.py`.
- **One message per merchant per tick** — deliberately conservative for decision-quality/anti-spam;
  follow-ups come on later ticks.
- **Determinism vs. flair** — `temperature=0` for reproducible scoring; slightly less linguistic variety.

## What extra context would have helped most

1. **Peer-cohort deltas per merchant** (their CTR vs *their* locality's median), to make social-proof
   levers concrete ("3 dentists in Lajpat Nagar did X this week").
2. **Real open appointment slots** for customer-facing booking messages (currently inferred from prefs).
3. **A canonical suppression/cooldown policy** per trigger kind, to plan multi-touch cadence within 24h.

## Run

```bash
cp .env.example .env            # add your LLM_API_KEY
pip install -r requirements.txt
uvicorn bot:app --host 0.0.0.0 --port 8080
```

Endpoints: `GET /v1/healthz`, `GET /v1/metadata`, `POST /v1/context`, `POST /v1/tick`,
`POST /v1/reply`, `POST /v1/teardown`.

Local scoring: `python ../judge_simulator.py` (reads the same `.env`), or the paced
`python eval_local.py`. Regenerate the artifact with `python generate_submission.py`.
