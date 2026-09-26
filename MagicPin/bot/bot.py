"""Vera Reforged — merchant engagement bot for the magicpin AI Challenge.

Exposes the 5-endpoint contract the judge harness drives:
  GET  /v1/healthz    liveness + contexts_loaded counts
  GET  /v1/metadata   bot identity
  POST /v1/context    idempotent, versioned context push
  POST /v1/tick       proactive: decide + compose zero or more sends
  POST /v1/reply      multi-turn: respond to a merchant/customer reply (send/wait/end)
  POST /v1/teardown   optional: wipe state at end of test

Run: uvicorn bot:app --host 0.0.0.0 --port 8080
"""
import time
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import config
import decision
import conversation
from store import STORE

# Re-export the composition entrypoint so `from bot import compose` works too — the main
# brief (§7.1) frames the submission as a compose() in bot.py, while the testing brief drives
# the HTTP endpoints below. Exposing both keeps every evaluation path working.
from composer import compose  # noqa: F401

app = FastAPI(title="Vera Reforged")
START = time.time()

VALID_SCOPES = {"category", "merchant", "customer", "trigger"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


# --------------------------------------------------------------------- health / identity
@app.get("/v1/healthz")
def healthz():
    return {"status": "ok", "uptime_seconds": int(time.time() - START),
            "contexts_loaded": STORE.counts()}


@app.get("/v1/metadata")
def metadata():
    return {
        "team_name": config.TEAM_NAME,
        "team_members": config.TEAM_MEMBERS,
        "model": config.LLM_MODEL if config.LLM_ENABLED else "template-fallback",
        "approach": config.APPROACH,
        "contact_email": config.CONTACT_EMAIL,
        "version": config.BOT_VERSION,
        "submitted_at": _now_iso(),
    }


# --------------------------------------------------------------------- context push
class CtxBody(BaseModel):
    scope: str
    context_id: str
    version: int
    payload: dict[str, Any]
    delivered_at: Optional[str] = None


@app.post("/v1/context")
def push_context(body: CtxBody):
    if body.scope not in VALID_SCOPES:
        return JSONResponse(status_code=400,
                            content={"accepted": False, "reason": "invalid_scope",
                                     "details": f"scope must be one of {sorted(VALID_SCOPES)}"})
    accepted, current = STORE.put_context(body.scope, body.context_id, body.version, body.payload)
    if not accepted:
        return JSONResponse(status_code=409,
                            content={"accepted": False, "reason": "stale_version",
                                     "current_version": current})
    return {"accepted": True,
            "ack_id": f"ack_{body.context_id}_v{body.version}",
            "stored_at": _now_iso()}


# --------------------------------------------------------------------- tick (proactive)
class TickBody(BaseModel):
    now: Optional[str] = None
    available_triggers: list[str] = []


@app.post("/v1/tick")
def tick(body: TickBody):
    try:
        actions = decision.select_actions(body.now or _now_iso(), body.available_triggers)
    except Exception:
        actions = []          # never blow the tick budget; empty is always valid
    return {"actions": actions}


# --------------------------------------------------------------------- reply (multi-turn)
class ReplyBody(BaseModel):
    conversation_id: str
    merchant_id: Optional[str] = None
    customer_id: Optional[str] = None
    from_role: str = "merchant"
    message: str = ""
    received_at: Optional[str] = None
    turn_number: int = 1


@app.post("/v1/reply")
def reply(body: ReplyBody):
    try:
        return conversation.handle_reply(
            body.conversation_id, body.merchant_id, body.customer_id,
            body.from_role, body.message, body.turn_number, body.received_at or _now_iso())
    except Exception:
        # Safe, non-malformed default.
        return {"action": "wait", "body": "", "cta": "none", "wait_seconds": 600,
                "rationale": "Internal error; backing off rather than sending a bad message."}


# --------------------------------------------------------------------- teardown (optional)
@app.post("/v1/teardown")
def teardown():
    STORE.reset()
    return {"status": "wiped"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)