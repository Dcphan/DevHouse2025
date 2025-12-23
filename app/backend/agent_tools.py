from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from openai import OpenAI

from memory_service import MemoryService
from supermemory_client import SupermemoryError


class AgentTools:
    def __init__(self, memory_service: Optional[MemoryService] = None, openai_client: Optional[OpenAI] = None):
        self.memory_service = memory_service or MemoryService()
        self._openai_client = openai_client

    def summarize_recent_topics(self, stm_summary: Optional[object]) -> dict:
        """
        Create a compact recap title and short list of follow-up ideas.
        """
        transcript_text = self._normalize_utterances(stm_summary)
        if not transcript_text:
            return {"name": "No conversation yet", "ideas": []}

        client = self._openai_client or OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=180,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Return JSON ONLY with keys: "
                        '{"name": <short title <=12 words>, '
                        '"ideas": [<<=12 word suggestions>]} '
                        "Keep ideas grounded in the transcript; omit if unsure."
                    ),
                },
                {"role": "user", "content": transcript_text},
            ],
        )
        raw = (completion.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return {"name": "No conversation yet", "ideas": []}

        name = str(data.get("name") or "Conversation recap").strip()
        ideas_raw = data.get("ideas")
        ideas: List[str] = []
        if isinstance(ideas_raw, list):
            for idea in ideas_raw:
                if isinstance(idea, str) and idea.strip():
                    ideas.append(idea.strip())
                    if len(ideas) >= 2:
                        break
        elif isinstance(ideas_raw, str) and ideas_raw.strip():
            ideas.append(ideas_raw.strip())

        return {"name": name[:120], "ideas": ideas}

    # Compatibility alias
    def summ(self, stm_summary: Optional[object]) -> dict:
        return self.summarize_recent_topics(stm_summary)

    def _normalize_utterances(self, utterances: object) -> str:
        """
        Flatten STM utterances (or similar shapes) into readable text lines.
        """
        if utterances is None:
            return ""
        if isinstance(utterances, str):
            return utterances.strip()

        records: Optional[List[Any]] = None
        if isinstance(utterances, dict):
            records = utterances.get("utterances") or utterances.get("conversation") or utterances.get("dialogue")
        elif isinstance(utterances, list):
            records = utterances

        lines: List[str] = []
        if isinstance(records, list):
            for item in records:
                if isinstance(item, dict):
                    speaker = item.get("speaker") or item.get("role") or "Speaker"
                    text = item.get("utterance") or item.get("text") or item.get("content") or ""
                    if text:
                        lines.append(f"{speaker}: {text}")
                else:
                    lines.append(str(item))

        return "\n".join(lines)

    def retrieve_person_context(self, person_id: str, *, fact_limit: int = 8, threshold: float = 0.4) -> dict:
        '''
            Fetch last meeting summary + Fetch Key Interests Fact and Preferences
        '''
        container_tag = self.memory_service._container_tag  # type: ignore[attr-defined]
        client = self.memory_service.client

        def _safe_search(**kwargs):
            try:
                return client.search_memories(**kwargs)
            except SupermemoryError:
                return {}

        def _person_card(person_id: str) -> dict:
            return self.memory_service.fetch_user_card(person_id, limit=3, threshold=threshold)

        def _person_facts(person_id: str, limit: int = 8, threshold: float = 0.4) -> List[dict]:
            resp = _safe_search(
                q="facts, interests, preferences, hobbies, favorites, work, school, company, role",
                container_tag=container_tag(person_id),
                threshold=threshold,
                limit=limit,
                filters={"metadata.kind": "person_fact"},
            )
            facts: List[dict] = []
            for result in resp.get("results", []) or []:
                meta = result.get("metadata") or {}
                if meta.get("kind") != "person_fact":
                    continue
                facts.append(
                    {
                        "key": meta.get("key") or meta.get("tag"),
                        "value": result.get("memory") or result.get("content"),
                        "confidence": meta.get("confidence") or result.get("score"),
                    }
                )
            return facts

        card = _person_card(person_id)
        return {
            "person_id": person_id,
            "person_name": card.get("person_name"),
            "last_conversation_summary": card.get("Short Summary of last conversation"),
            "shared_interest": card.get("Shared Interest"),
                "facts": _person_facts(person_id, limit=fact_limit, threshold=threshold),
        }

    def _empty_coach_decision(self) -> dict:
        return {"hint": "", "followups": [], "should_store": False, "store_reason": "", "memory_candidates": [], "tags": []}

    def _empty_name_result(self) -> Dict[str, object]:
        return {"name": "", "confidence": 0.0, "evidence": ""}

    def recognize_person_name(self, stm_dialogue: object) -> Dict[str, object]:
        """
        Identify the other person's name from the conversation transcript.
        """
        transcript_text = self._normalize_utterances(stm_dialogue)
        if not transcript_text:
            return self._empty_name_result()

        client = self._openai_client or OpenAI()
        messages = [
            {
                "role": "system",
                "content": (
                    "Return JSON ONLY with keys {\"name\": string, \"confidence\": number 0-1, \"evidence\": string}. "
                    "Infer the counterparty's name introduced in the transcript. "
                    "If unsure, leave name empty and set confidence to 0. Avoid guessing or inventing names."
                ),
            },
            {"role": "user", "content": transcript_text},
        ]

        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=120,
            messages=messages,
        )
        raw = (completion.choices[0].message.content or "").strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return self._empty_name_result()

        name = str(data.get("name") or "").strip()
        evidence = str(data.get("evidence") or "").strip()
        try:
            conf_val = float(data.get("confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            conf_val = 0.0
        confidence = max(0.0, min(1.0, conf_val))

        if not name:
            confidence = 0.0
            evidence = ""

        return {"name": name[:80], "confidence": confidence, "evidence": evidence[:200]}

    def coach_and_decide(
        self,
        person_context: dict,
        stm_dialogue: object,
        recent_topics: Optional[dict] = None,
        mode: str = "default",
        person_id: Optional[str] = None,
    ) -> dict:
        """
        Use a single OpenAI call to suggest the next move and potential memories to store.
        Returns a strict JSON dict per the expected schema.
        """
        transcript_text = self._normalize_utterances(stm_dialogue)
        if not transcript_text:
            return self._empty_coach_decision()

        client = self._openai_client or OpenAI()

        def _build_messages(strict: bool = False) -> List[dict]:
            system_content = (
                "Return valid JSON ONLY. No prose, no markdown. "
                "Schema: {"
                '"hint": string (<=22 words), '
                '"followups": [string <=14 words] (0-2 items), '
                '"should_store": boolean, '
                '"store_reason": string (1-6 words), '
                '"memory_candidates": [ {'
                '"type": "identity"|"work"|"school"|"interest"|"preference"|"project"|"product"|"plan"|"other", '
                '"key": string, "value": string, "confidence": number 0-1, "evidence": string'
                "} ] (max 3), "
                '"tags": [string] (max 4)'
                "}. "
                "Do NOT guess traits. Only use transcript or provided context. "
                "If unsure, lower confidence and set should_store=false. "
                "Be concise and safe."
            )
            if strict:
                system_content = "Return valid JSON only. No text. " + system_content

            user_payload = {
                "mode": mode,
                "person_id": person_id,
                "person_context": person_context or {},
                "recent_topics": recent_topics or {},
                "transcript": transcript_text,
            }
            return [
                {"role": "system", "content": system_content},
                {"role": "user", "content": json.dumps(user_payload, indent=2)},
            ]

        def _call(messages: List[dict]) -> str:
            completion = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.3,
                max_tokens=320,
                messages=messages,
            )
            return (completion.choices[0].message.content or "").strip()

        raw = _call(_build_messages(strict=False))
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            raw = _call(_build_messages(strict=True))
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                return self._empty_coach_decision()

        decision: dict = parsed if isinstance(parsed, dict) else {}

        hint = str(decision.get("hint") or "").strip()
        followups_raw = decision.get("followups")
        followups: List[str] = []
        if isinstance(followups_raw, list):
            for item in followups_raw:
                if isinstance(item, str) and item.strip():
                    followups.append(item.strip())
        elif isinstance(followups_raw, str) and followups_raw.strip():
            followups.append(followups_raw.strip())
        followups = followups[:2]

        should_store = bool(decision.get("should_store"))
        store_reason = str(decision.get("store_reason") or "").strip()

        memory_candidates_raw = decision.get("memory_candidates")
        memory_candidates: List[dict] = []
        if isinstance(memory_candidates_raw, list):
            for item in memory_candidates_raw:
                if not isinstance(item, dict):
                    continue
                memory_candidates.append(
                    {
                        "type": str(item.get("type") or "other"),
                        "key": str(item.get("key") or "").strip(),
                        "value": str(item.get("value") or "").strip(),
                        "confidence": float(item.get("confidence", 0.5) or 0.5),
                        "evidence": str(item.get("evidence") or "").strip(),
                    }
                )
                if len(memory_candidates) >= 3:
                    break
        if not should_store:
            memory_candidates = []

        tags_raw = decision.get("tags")
        tags: List[str] = []
        if isinstance(tags_raw, list):
            for t in tags_raw:
                if isinstance(t, str) and t.strip():
                    tags.append(t.strip())
        elif isinstance(tags_raw, str) and tags_raw.strip():
            tags.append(tags_raw.strip())
        tags = tags[:4]

        return {
            "hint": hint[:256],
            "followups": followups,
            "should_store": should_store,
            "store_reason": store_reason[:64],
            "memory_candidates": memory_candidates,
            "tags": tags,
        }


    def store_meeting_summary(self, person_id: str, stm_dialogue: object, *, person_name: Optional[str] = None):
        """
        Persist a concise last-conversation title for the person.
        """
        return self.memory_service.save_meeting_summary(
            person_id=person_id,
            stm_dialogue=stm_dialogue,
            person_name=person_name,
        )

    def web_browse_context(self):
        pass
