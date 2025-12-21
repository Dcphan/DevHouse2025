import time
from collections import deque
from typing import Deque, Dict, List, Optional, TypedDict, Union


class Utterance(TypedDict):
    speaker: str  # "user" | "other"
    text: str
    ts: float


class Topic(TypedDict):
    label: str
    confidence: float


class CoachAction(TypedDict, total=False):
    type: str
    payload: dict


class SessionState:
    """
    In-memory STM state container for a single session.
    """

    def __init__(self, max_utterances: int, ttl_seconds: int) -> None:
        self.active_person_id: Optional[str] = None
        self.utterances: Deque[Utterance] = deque(maxlen=max_utterances)
        self.topic: Optional[Topic] = None
        self.last_coach_action: Optional[CoachAction] = None
        self.expires_at: float = time.time() + ttl_seconds
        self.ttl_seconds = ttl_seconds

    def touch(self) -> None:
        self.expires_at = time.time() + self.ttl_seconds


class ShortTermMemory:
    """
    Deterministic, in-memory STM keyed by session_id.

    - No database, no vector store, no LLM calls.
    - Appends only FINAL utterances.
    - TTL-based expiry (default 5 minutes).
    - Size bounded via deque (default last 80 utterances).
    """

    def __init__(self, ttl_seconds: int = 300, max_utterances: int = 80) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_utterances = max_utterances
        self._store: Dict[str, SessionState] = {}

    def _get_state(self, session_id: str) -> SessionState:
        now = time.time()
        state = self._store.get(session_id)
        if state is None or state.expires_at <= now:
            state = SessionState(self.max_utterances, self.ttl_seconds)
            self._store[session_id] = state
        else:
            state.touch()
        return state

    def _reset_on_person_change(self, state: SessionState) -> None:
        state.utterances.clear()
        state.topic = None
        state.last_coach_action = None

    def append_utterance(self, session_id: str, speaker: str, text: str, ts: Optional[float] = None) -> None:
        """
        Append a FINAL utterance. Speaker must be "user" or "other".
        """
        if speaker not in ("user", "other"):
            return
        state = self._get_state(session_id)
        state.utterances.append(
            {
                "speaker": speaker,
                "text": text,
                "ts": float(ts) if ts is not None else time.time(),
            }
        )

    def set_active_person(self, session_id: str, person_id: Optional[str]) -> None:
        state = self._get_state(session_id)
        if state.active_person_id != person_id:
            self._reset_on_person_change(state)
            state.active_person_id = person_id

    def set_topic(self, session_id: str, label: str, confidence: Union[int, float]) -> None:
        state = self._get_state(session_id)
        state.topic = {"label": label, "confidence": float(confidence)}

    def set_last_coach_action(self, session_id: str, action_type: str, payload: Optional[dict] = None) -> None:
        state = self._get_state(session_id)
        state.last_coach_action = {"type": action_type, "payload": payload or {}}

    def summary(self, session_id: str) -> Dict[str, object]:
        """
        Deterministic summary: last 6 utterances (in order), topic, last coach action.
        No inference or rewriting.
        """
        state = self._get_state(session_id)
        utterances: List[Utterance] = list(state.utterances)[-6:]
        return {
            "active_person_id": state.active_person_id,
            "utterances": utterances,
            "topic": state.topic,
            "last_coach_action": state.last_coach_action,
        }
