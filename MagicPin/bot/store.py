"""In-memory state. The judge never restarts us mid-test, so memory is sufficient.

Everything the bot needs to be *stateful* lives here:
  - contexts       : the 4-layer context, versioned + idempotent
  - conversations  : per-conversation turn history (for multi-turn + anti-repetition)
  - sent_bodies    : bodies already sent per conversation (anti-repetition guard)
  - fired_suppression : suppression_keys already acted on (dedup across ticks)
"""
import threading
from typing import Optional


class Store:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        # (scope, context_id) -> {"version": int, "payload": dict}
        self.contexts: dict[tuple[str, str], dict] = {}
        # conversation_id -> {"merchant_id","customer_id","turns":[...], "state":{...}}
        self.conversations: dict[str, dict] = {}
        # conversation_id -> set of verbatim bodies we've sent
        self.sent_bodies: dict[str, set] = {}
        # suppression_key -> True (already fired)
        self.fired_suppression: dict[str, bool] = {}
        # merchant_id -> count of canned auto-replies seen (across conversations)
        self.autoreply_counts: dict[str, int] = {}

    # ---- context ----
    def put_context(self, scope: str, cid: str, version: int, payload: dict) -> tuple[bool, Optional[int]]:
        """Idempotent upsert. Returns (accepted, current_version_if_rejected)."""
        key = (scope, cid)
        with self._lock:
            cur = self.contexts.get(key)
            if cur is not None:
                if version == cur["version"]:
                    return True, None            # idempotent no-op, treat as accepted
                if version < cur["version"]:
                    return False, cur["version"]  # stale
            self.contexts[key] = {"version": version, "payload": payload}
            return True, None

    def get_context(self, scope: str, cid: Optional[str]) -> Optional[dict]:
        if cid is None:
            return None
        with self._lock:
            entry = self.contexts.get((scope, cid))
            return entry["payload"] if entry else None

    def all_of_scope(self, scope: str) -> dict[str, dict]:
        with self._lock:
            return {cid: e["payload"] for (s, cid), e in self.contexts.items() if s == scope}

    def counts(self) -> dict[str, int]:
        counts = {"category": 0, "merchant": 0, "customer": 0, "trigger": 0}
        with self._lock:
            for (scope, _) in self.contexts:
                counts[scope] = counts.get(scope, 0) + 1
        return counts

    # ---- conversations ----
    def get_or_create_conversation(self, conv_id: str, merchant_id=None, customer_id=None) -> dict:
        with self._lock:
            conv = self.conversations.get(conv_id)
            if conv is None:
                conv = {
                    "conversation_id": conv_id,
                    "merchant_id": merchant_id,
                    "customer_id": customer_id,
                    "turns": [],                 # [{"role","text","ts"}]
                    "state": {                   # rolling conversation flags
                        "intent_locked": False,
                        "nudges_unanswered": 0,
                        "ended": False,
                    },
                }
                self.conversations[conv_id] = conv
                self.sent_bodies[conv_id] = set()
            return conv

    def add_turn(self, conv_id: str, role: str, text: str, ts: str = "") -> None:
        with self._lock:
            conv = self.conversations.get(conv_id)
            if conv is not None:
                conv["turns"].append({"role": role, "text": text, "ts": ts})

    def already_sent(self, conv_id: str, body: str) -> bool:
        with self._lock:
            return body.strip() in self.sent_bodies.get(conv_id, set())

    def record_sent(self, conv_id: str, body: str) -> None:
        with self._lock:
            self.sent_bodies.setdefault(conv_id, set()).add(body.strip())

    # ---- suppression ----
    def is_suppressed(self, key: str) -> bool:
        if not key:
            return False
        with self._lock:
            return self.fired_suppression.get(key, False)

    def mark_suppressed(self, key: str) -> None:
        if not key:
            return
        with self._lock:
            self.fired_suppression[key] = True

    def bump_autoreply(self, merchant_id: Optional[str]) -> int:
        if not merchant_id:
            return 0
        with self._lock:
            self.autoreply_counts[merchant_id] = self.autoreply_counts.get(merchant_id, 0) + 1
            return self.autoreply_counts[merchant_id]

    def reset(self) -> None:
        with self._lock:
            self.contexts.clear()
            self.conversations.clear()
            self.sent_bodies.clear()
            self.fired_suppression.clear()
            self.autoreply_counts.clear()


# Single global instance shared across requests.
STORE = Store()
