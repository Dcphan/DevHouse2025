from __future__ import annotations

import os
import json
from typing import Any, Dict, List, Optional

from dotenv import find_dotenv, load_dotenv
from openai import OpenAI

load_dotenv(find_dotenv())

try:
    from .supermemory_client import SupermemoryClient, SupermemoryError
except ImportError:  # pragma: no cover - fallback when running as script
    from supermemory_client import SupermemoryClient, SupermemoryError


class MemoryService:
    """
    Higher-level helpers built on top of SupermemoryClient.
    """

    def __init__(self, supermemory_client: Optional[SupermemoryClient] = None, openai_client: Optional[OpenAI] = None) -> None:
        self.client = supermemory_client or SupermemoryClient()
        self._openai_client = openai_client

    def _container_tag(self, person_id: str) -> str:
        return f"user_{person_id}"

    def _pick_metadata(self, results: List[Dict[str, Any]], keys: List[str]) -> Optional[Any]:
        for result in results:
            metadata = result.get("metadata") or {}
            for key in keys:
                if metadata.get(key) is not None:
                    return metadata[key]
        return None

    def write_memory(self, person_id: str, key: str, value: object, confidence: float, *, source: str = "conversation") -> Dict[str, Any]:
        container_tag = self._container_tag(person_id)
        content_block = f"key: {key}\nvalue: {value}\n"
        metadata = {
            "key": key,
            "source": source,
            "kind": "atomic_memory",
        }
        return self.client.add_document(content=content_block, container_tag=container_tag, metadata=metadata)

    def fetch_user_card(self, person_id: str, *, limit: int = 5, threshold: float = 0.6) -> Dict[str, Any]:
        container_tag = self._container_tag(person_id)
        search_resp = self.client.search_memories(
            q="Shared Interest and last conversatin summary with that person",
            container_tag=container_tag,
            threshold=threshold,
            limit=limit,
        )
        results = search_resp.get("results", []) or []
        return {
            "person_name": self._pick_metadata(results, ["person_name", "name"]) or person_id,
            "First Met": self._pick_metadata(results, ["first_met", "met_on", "met_date"]),
            "Short Summary of last conversation": self._pick_metadata(results, ["last_conversation_title", "last_summary", "summary"])
            or (results[0].get("memory") if results else None),
            "Shared Interest": self._pick_metadata(results, ["shared_interest", "common_interest"]),
        }

    def _normalize_dialogue(self, dialogue: object) -> str:
        """
        Accepts raw text, a list, or a dict (e.g., {"conversation": [...]}) and flattens to text.
        """
        if dialogue is None:
            return ""
        if isinstance(dialogue, str):
            return dialogue
        lines: List[str] = []
        records = None
        if isinstance(dialogue, dict) and "conversation" in dialogue:
            records = dialogue.get("conversation")
        elif isinstance(dialogue, list):
            records = dialogue
        if isinstance(records, list):
            for item in records:
                if isinstance(item, dict):
                    speaker = item.get("speaker") or item.get("role") or "Speaker"
                    text = item.get("utterance") or item.get("text") or item.get("content") or ""
                    lines.append(f"{speaker}: {text}")
                else:
                    lines.append(str(item))
        return "\n".join(lines) if lines else str(dialogue)

    def _summarize_dialogue(self, person_ref: str, stm_dialogue: str) -> str:
        if not stm_dialogue:
            raise ValueError("stm_dialogue is required for summarization.")
        client = self._openai_client or OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=32,
            messages=[
                {"role": "system", "content": "You write concise meeting titles in 4-7 words without punctuation."},
                {
                    "role": "user",
                    "content": f"Your last conversation with {person_ref} is about {stm_dialogue}. Provide a 4-7 word title without punctuation or quotes.",
                },
            ],
        )
        summary = (completion.choices[0].message.content or "").strip()
        return summary.strip("\"' ").rstrip(".!?,;:")

    def _extract_person_facts(self, stm_dialogue: str) -> List[Dict[str, Any]]:
        """
        Extract simple key/value facts about the person from dialogue text.
        """
        if not stm_dialogue:
            return []
        client = self._openai_client or OpenAI()
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=256,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract concise person facts from the conversation. "
                        "Return a JSON array of objects with fields: "
                        '{"key": <one of ["interest","sport","drink","school","company","role","hobby","location"]>, '
                        '"value": <short text>, "confidence": <0-1 float>}. '
                        "Only include facts stated or strongly implied; keep values short."
                    ),
                },
                {"role": "user", "content": stm_dialogue},
            ],
        )
        raw = completion.choices[0].message.content or "[]"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []

        facts: List[Dict[str, Any]] = []
        if isinstance(data, list):
            for item in data:
                if not isinstance(item, dict):
                    continue
                key = item.get("key") or item.get("tag") or item.get("type")
                value = item.get("value") or item.get("fact") or item.get("detail")
                if key and value:
                    facts.append(
                        {
                            "key": str(key),
                            "value": str(value),
                            "confidence": float(item.get("confidence", 0.8)),
                        }
                    )
        return facts

    def save_meeting_summary(self, person_id: str, stm_dialogue: object) -> Dict[str, Any]:
        container_tag = self._container_tag(person_id)
        normalized_dialogue = self._normalize_dialogue(stm_dialogue)
        summary = self._summarize_dialogue(person_id, normalized_dialogue)
        formatted_content = f"Your last conversation with {person_id} is about {summary}"
        metadata = {"kind": "last_conversation_title", "source": "meeting_summary"}
        search_resp = self.client.search_memories(
            q="last conversation title",
            container_tag=container_tag,
            limit=5,
            threshold=0.3,
        )
        results = search_resp.get("results", []) or []
        existing_id: Optional[str] = None
        for result in results:
            metadata_result = result.get("metadata") or {}
            if metadata_result.get("kind") == "last_conversation_title":
                existing_id = result.get("id")
                if existing_id:
                    break
        if existing_id:
            return self.client.update_memory(
                container_tag=container_tag,
                memory_id=existing_id,
                new_content=formatted_content,
                metadata=metadata,
            )
        return self.client.add_document(content=formatted_content, container_tag=container_tag, metadata=metadata)

    def save_person_facts(self, person_id: str, stm_dialogue: object, *, default_confidence: float = 0.8) -> List[Dict[str, Any]]:
        """
        Extract person facts (school, interests, etc.) from dialogue and store them in Supermemory.
        """
        container_tag = self._container_tag(person_id)
        normalized_dialogue = self._normalize_dialogue(stm_dialogue)
        facts = self._extract_person_facts(normalized_dialogue)
        responses: List[Dict[str, Any]] = []
        for fact in facts:
            key = fact.get("key")
            value = fact.get("value")
            if not key or not value:
                continue
            content = f"{key}: {value}"
            metadata = {
                "key": key,
                "kind": "person_fact",
                "tag": key,
                "source": "conversation_end",
            }
            responses.append(
                self.client.add_document(
                    content=content,
                    container_tag=container_tag,
                    metadata=metadata,
                )
            )
        return responses


if __name__ == "__main__":
    if not os.getenv("SUPERMEMORY_API_KEY"):
        print("Set SUPERMEMORY_API_KEY to run the example.")
    else:
        client = SupermemoryClient()
        service = MemoryService(client)
        try:
            write_response = service.write_memory("demo_person", "favorite_food", "sushi", 0.92)
            print("Write response:", write_response)
            card = service.fetch_user_card("demo_person", limit=3)
            print("User card:", card)
        except SupermemoryError as exc:
            print(f"Supermemory error: {exc}")
