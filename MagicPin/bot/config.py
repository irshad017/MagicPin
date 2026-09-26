"""Central configuration. All tunables live here so the rest of the code stays clean."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env next to this file, not the current working directory — so the key loads
# no matter which directory the server (or a grader script) is launched from.
load_dotenv(Path(__file__).resolve().parent / ".env")


def _get(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


# --- LLM ---
LLM_PROVIDER = _get("LLM_PROVIDER", "groq").lower()
LLM_API_KEY = _get("LLM_API_KEY")
LLM_MODEL = _get("LLM_MODEL", "llama-3.3-70b-versatile")
OLLAMA_URL = _get("OLLAMA_URL", "http://localhost:11434")

# Composition determinism + budget
LLM_TEMPERATURE = 0.0
LLM_SEED = 7
LLM_MAX_TOKENS = 900
LLM_TIMEOUT_S = 20          # per-call ceiling; harness gives us 30s total per request
LLM_MAX_RETRIES = 3         # backoff retries on 429/5xx
LLM_MAX_BACKOFF_S = 6.0     # cap per-retry wait so we stay within the tick budget

# Compose concurrency per tick. Free-tier rate limits favor low concurrency + retry.
COMPOSE_MAX_WORKERS = 3

# --- Decision layer ---
MAX_ACTIONS_PER_TICK = 20   # hard cap from the testing brief
TICK_DEADLINE_S = 26        # stop composing and return what we have before the 30s judge timeout
REPLY_DEADLINE_S = 26

# Auto-reply detection
AUTOREPLY_REPEAT_THRESHOLD = 3   # same verbatim text N times => auto-reply
MAX_UNANSWERED_NUDGES = 3        # stop after this many one-sided nudges

# --- Identity (/v1/metadata) ---
TEAM_NAME = _get("TEAM_NAME", "Vera Reforged")
TEAM_MEMBERS = [m.strip() for m in _get("TEAM_MEMBERS", "Ansh").split(",") if m.strip()]
CONTACT_EMAIL = _get("CONTACT_EMAIL", "team@example.com")
BOT_VERSION = _get("BOT_VERSION", "1.0.0")

# Human-readable approach string for /v1/metadata
APPROACH = (
    "Grounded single-signal composer: per-tick decision layer picks the one best trigger "
    "per merchant, resolves trigger references into category/merchant facts, and composes a "
    "single-CTA WhatsApp message via a provider-agnostic LLM with strict JSON output, "
    "post-hoc validation, and multi-turn reply routing (auto-reply detection, intent handoff)."
)

LLM_ENABLED = bool(LLM_API_KEY) or LLM_PROVIDER == "ollama"
