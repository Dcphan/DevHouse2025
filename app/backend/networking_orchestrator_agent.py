from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal
import time

EventType = Literal["utterance_final", "conversation_end", "person_detected"]

@dataclass
class OrchestratorState:
    active_person_id: Optional[str] = None
    stm_dialogue: Dict[str, Any] = field(default_factory=lambda: {"utterances": []})

    # cached context
    person_context: Dict[str, Any] = field(default_factory=dict)
    recent_topics: Dict[str, Any] = field(default_factory=lambda: {"name": "No conversation yet", "ideas": []})

    # throttling / timing
    last_coach_ts: float = 0.0
    last_topic_ts: float = 0.0
    last_speaker: str = ""
    turn_count_since_coach: int = 0

    # optional: track if name already recognized
    recognized_name: Optional[str] = None


class ConversationOrchestrator:
    def __init__(
        self,
        tools,
        *,
        coach_min_interval_sec: float = 1.2,
        coach_turns_interval: int = 2,
        topic_refresh_turns: int = 6,
        context_refresh_turns: int = 6,
        fact_limit: int = 8,
        threshold: float = 0.4,
    ):
        self.tools = tools
        self.coach_min_interval_sec = coach_min_interval_sec
        self.coach_turns_interval = coach_turns_interval
        self.topic_refresh_turns = topic_refresh_turns
        self.context_refresh_turns = context_refresh_turns
        self.fact_limit = fact_limit
        self.threshold = threshold

        # internal counters
        self._turns_since_topic = 0
        self._turns_since_context = 0

    # ----------------------------
    # Public entry point
    # ----------------------------
    def handle_event(self, state: OrchestratorState, event: Dict[str, Any]) -> Dict[str, Any]:
        etype: EventType = event.get("type", "utterance_final")
        signals: List[Dict[str, Any]] = []

        # 1) update active person if provided
        if etype == "person_detected":
            new_person_id = event.get("person_id")
            if new_person_id and new_person_id != state.active_person_id:
                state = self._switch_person(state, new_person_id)
                signals.append({"type": "person_switched", "person_id": new_person_id})

            return self._build_payload(state, signals=signals)

        # 2) utterance_final -> append STM
        if etype == "utterance_final":
            utter = event.get("utterance") or {}
            speaker = str(utter.get("speaker") or "Speaker")
            text = str(utter.get("utterance") or "").strip()
            if text:
                state.stm_dialogue["utterances"].append({"speaker": speaker, "utterance": text})
                state.last_speaker = speaker
                state.turn_count_since_coach += 1
                self._turns_since_topic += 1
                self._turns_since_context += 1

        # 3) conversation_end -> store meeting summary and (optional) last topic recap
        if etype == "conversation_end":
            if state.active_person_id:
                try:
                    self.tools.store_meeting_summary(state.active_person_id, state.stm_dialogue)
                    signals.append({"type": "meeting_summary_saved"})
                except Exception as e:
                    signals.append({"type": "error", "where": "store_meeting_summary", "msg": str(e)[:180]})

            # refresh recent topics one last time (nice for next time you meet)
            try:
                state.recent_topics = self.tools.summarize_recent_topics(state.stm_dialogue)
                signals.append({"type": "recent_topics_updated"})
            except Exception as e:
                signals.append({"type": "error", "where": "summarize_recent_topics", "msg": str(e)[:180]})

            return self._build_payload(state, signals=signals)

        # 4) ensure we have an active person id (if not, still can coach but no memory)
        if not state.active_person_id:
            # still can provide coach hint without person context
            state.person_context = {}
        else:
            # refresh person context occasionally
            if self._should_refresh_context(state):
                state.person_context = self._safe_retrieve_context(state.active_person_id, signals)
                self._turns_since_context = 0

        # 5) refresh recent topics occasionally
        if self._should_refresh_topics(state):
            try:
                state.recent_topics = self.tools.summarize_recent_topics(state.stm_dialogue)
                signals.append({"type": "recent_topics_updated"})
            except Exception as e:
                signals.append({"type": "error", "where": "summarize_recent_topics", "msg": str(e)[:180]})
            self._turns_since_topic = 0

        # 6) maybe run coach
        coach_out: Dict[str, Any] = {"hint": "", "followups": []}
        mem_out: Dict[str, Any] = {"stored": False, "items": []}

        if self._should_call_coach(state, event):
            coach_out, mem_out = self._run_coach(state, signals)
            state.last_coach_ts = time.time()
            state.turn_count_since_coach = 0

        # 7) build payload
        payload = self._build_payload(
            state,
            coach=coach_out,
            memory=mem_out,
            signals=signals,
        )
        return payload

    # ----------------------------
    # Helpers
    # ----------------------------
    def _switch_person(self, state: OrchestratorState, person_id: str) -> OrchestratorState:
        # new conversation context for a new person
        state.active_person_id = person_id
        state.stm_dialogue = {"utterances": []}
        state.recent_topics = {"name": "No conversation yet", "ideas": []}
        state.person_context = self._safe_retrieve_context(person_id, signals=[])
        state.last_coach_ts = 0.0
        state.turn_count_since_coach = 0
        state.recognized_name = None
        self._turns_since_topic = 0
        self._turns_since_context = 0
        return state

    def _safe_retrieve_context(self, person_id: str, signals: List[Dict[str, Any]]) -> Dict[str, Any]:
        try:
            return self.tools.retrieve_person_context(
                person_id,
                fact_limit=self.fact_limit,
                threshold=self.threshold,
            )
        except Exception as e:
            signals.append({"type": "error", "where": "retrieve_person_context", "msg": str(e)[:180]})
            return {"person_id": person_id, "person_name": None, "last_conversation_summary": None, "shared_interest": None, "facts": []}

    def _should_refresh_topics(self, state: OrchestratorState) -> bool:
        # refresh on a schedule, not every utterance
        return self._turns_since_topic >= self.topic_refresh_turns

    def _should_refresh_context(self, state: OrchestratorState) -> bool:
        return self._turns_since_context >= self.context_refresh_turns

    def _should_call_coach(self, state: OrchestratorState, event: Dict[str, Any]) -> bool:
        if event.get("type") != "utterance_final":
            return False

        now = time.time()
        if (now - state.last_coach_ts) < self.coach_min_interval_sec:
            return False

        if state.turn_count_since_coach < self.coach_turns_interval:
            # allow earlier if trigger is strong
            return self._has_trigger_phrase(state)

        return True

    def _has_trigger_phrase(self, state: OrchestratorState) -> bool:
        # lightweight heuristic; keep it simple for demo
        utts = state.stm_dialogue.get("utterances") or []
        if not utts:
            return False
        last = str(utts[-1].get("utterance") or "").lower()

        triggers = [
            "what do you think",
            "any advice",
            "recommend",
            "i'm not sure",
            "help me",
            "how should i",
            "tell me about",
            "do you know",
            "?",
        ]
        return any(t in last for t in triggers)

    def _run_coach(self, state: OrchestratorState, signals: List[Dict[str, Any]]) -> (Dict[str, Any], Dict[str, Any]):
        # Optionally: recognize name once early in convo (not every time)
        if state.active_person_id and state.recognized_name is None and len(state.stm_dialogue.get("utterances", [])) >= 2:
            try:
                name_res = self.tools.recognize_person_name(state.stm_dialogue)
                if name_res.get("confidence", 0.0) >= 0.7 and name_res.get("name"):
                    state.recognized_name = name_res["name"]
                    signals.append({"type": "recognized_name", "name": state.recognized_name})
            except Exception as e:
                signals.append({"type": "error", "where": "recognize_person_name", "msg": str(e)[:180]})

        try:
            decision = self.tools.coach_and_decide(
                person_context=state.person_context or {},
                stm_dialogue=state.stm_dialogue,
                recent_topics=state.recent_topics or {},
                mode="default",
                person_id=state.active_person_id,
            )
        except Exception as e:
            signals.append({"type": "error", "where": "coach_and_decide", "msg": str(e)[:180]})
            return {"hint": "", "followups": []}, {"stored": False, "items": []}

        coach_out = {"hint": decision.get("hint", ""), "followups": decision.get("followups", [])}

        # store memory candidates if should_store
        mem_out = {"stored": False, "items": []}
        if decision.get("should_store") and state.active_person_id:
            candidates = decision.get("memory_candidates") or []
            stored_items: List[Dict[str, Any]] = []

            # IMPORTANT: Your AgentTools doesn't show a "store_person_fact" function,
            # so here we call MemoryService through tools.memory_service directly.
            # If you already have a wrapper, replace this block with that wrapper.
            for c in candidates[:3]:
                try:
                    ctype = str(c.get("type") or "other")
                    key = str(c.get("key") or "").strip()
                    value = str(c.get("value") or "").strip()
                    conf = float(c.get("confidence") or 0.5)
                    evidence = str(c.get("evidence") or "").strip()

                    if not key or not value:
                        continue

                    # Save as person_fact in Supermemory via MemoryService
                    saved = self.tools.memory_service.store_person_fact(
                        person_id=state.active_person_id,
                        key=key,
                        value=value,
                        confidence=conf,
                        evidence=evidence,
                        fact_type=ctype,
                        tags=decision.get("tags") or [],
                    )
                    stored_items.append({"key": key, "value": value, "type": ctype, "ok": True, "result": saved})
                except Exception as e:
                    signals.append({"type": "error", "where": "store_person_fact", "msg": str(e)[:180]})
                    stored_items.append({"key": c.get("key"), "value": c.get("value"), "type": c.get("type"), "ok": False})

            mem_out = {"stored": any(x.get("ok") for x in stored_items), "items": stored_items}

        return coach_out, mem_out

    def _build_payload(
        self,
        state: OrchestratorState,
        *,
        coach: Optional[Dict[str, Any]] = None,
        memory: Optional[Dict[str, Any]] = None,
        signals: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        return {
            "active_person_id": state.active_person_id,
            "stm": state.stm_dialogue,
            "recent_topics": state.recent_topics,
            "person_context": state.person_context,
            "coach": coach or {"hint": "", "followups": []},
            "memory": memory or {"stored": False, "items": []},
            "signals": signals or [],
        }
