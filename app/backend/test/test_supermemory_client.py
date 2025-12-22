import os
from types import SimpleNamespace

import httpx
import pytest

from backend.supermemory_client import SupermemoryClient, SupermemoryError
from backend.memory_service import MemoryService


@pytest.fixture(autouse=True)
def set_api_key(monkeypatch):
    monkeypatch.setenv("SUPERMEMORY_API_KEY", "test-key")


def _make_response(status: int, data: dict, method: str = "POST", path: str = "/v3/documents") -> httpx.Response:
    req = httpx.Request(method, f"https://api.supermemory.ai{path}")
    return httpx.Response(status, json=data, request=req)


def test_add_document_builds_payload(monkeypatch):
    captured = {}

    def fake_request(method, url, headers=None, json=None, timeout=None):
        captured.update({"method": method, "url": url, "headers": headers, "json": json, "timeout": timeout})
        return _make_response(200, {"id": "doc1", "status": "ok"}, method=method, path="/v3/documents")

    monkeypatch.setattr(httpx, "request", fake_request)
    client = SupermemoryClient()
    resp = client.add_document("hello", container_tag="user_1", custom_id="cid", metadata={"a": "b"})
    assert resp["id"] == "doc1"
    assert captured["method"] == "POST"
    assert captured["url"].endswith("/v3/documents")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["content"] == "hello"
    assert captured["json"]["containerTag"] == "user_1"
    assert captured["json"]["customId"] == "cid"
    assert captured["json"]["metadata"] == {"a": "b"}
    assert captured["timeout"] == 10


def test_non_2xx_raises(monkeypatch):
    def fake_request(method, url, headers=None, json=None, timeout=None):
        return _make_response(500, {"error": "fail"}, method=method, path="/v4/search")

    monkeypatch.setattr(httpx, "request", fake_request)
    client = SupermemoryClient()
    with pytest.raises(SupermemoryError):
        client.search_memories("hi")


def test_nested_metadata_rejected():
    client = SupermemoryClient()
    with pytest.raises(ValueError):
        client.add_document("hi", metadata={"nested": {"bad": True}})


def test_update_memory_requires_identifier():
    client = SupermemoryClient()
    with pytest.raises(ValueError):
        client.update_memory(container_tag="user_1", new_content="x")


def test_forget_memory_requires_identifier():
    client = SupermemoryClient()
    with pytest.raises(ValueError):
        client.forget_memory(container_tag="user_1")


class DummySupermemoryClient:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []
        self.update_calls = []
        self.search_results = {"results": []}

    def add_document(self, **kwargs):
        self.add_calls.append(kwargs)
        return {"id": "new", "status": "ok"}

    def search_memories(self, **kwargs):
        self.search_calls.append(kwargs)
        return self.search_results

    def update_memory(self, **kwargs):
        self.update_calls.append(kwargs)
        return {"id": "updated"}


class DummyOpenAI:
    class DummyChat:
        class DummyCompletions:
            def create(self, **kwargs):
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="Meeting recap title"))])

        def __init__(self):
            self.completions = DummyOpenAI.DummyChat.DummyCompletions()

    def __init__(self):
        self.chat = DummyOpenAI.DummyChat()


def test_write_memory_formats_and_sends():
    dummy = DummySupermemoryClient()
    service = MemoryService(supermemory_client=dummy)
    service.write_memory("p1", "favorite_food", "sushi", 0.9)
    assert dummy.add_calls
    sent = dummy.add_calls[0]
    assert sent["container_tag"] == "user_p1"
    assert "favorite_food" in sent["content"]
    assert sent["metadata"]["kind"] == "atomic_memory"


def test_fetch_user_card_picks_metadata():
    dummy = DummySupermemoryClient()
    dummy.search_results = {
        "results": [
            {"metadata": {"person_name": "Ada", "first_met": "2024-01-01", "last_conversation_title": "Catch up", "shared_interest": "ML"}, "memory": "fallback"}
        ]
    }
    service = MemoryService(supermemory_client=dummy)
    card = service.fetch_user_card("p2")
    assert card["person_name"] == "Ada"
    assert card["First Met"] == "2024-01-01"
    assert card["Short Summary of last conversation"] == "Catch up"
    assert card["Shared Interest"] == "ML"


def test_save_meeting_summary_updates_existing(monkeypatch):
    dummy = DummySupermemoryClient()
    dummy.search_results = {"results": [{"id": "m1", "metadata": {"kind": "last_conversation_title"}}]}
    service = MemoryService(supermemory_client=dummy, openai_client=DummyOpenAI())
    resp = service.save_meeting_summary("p3", "We talked about product plans.")
    assert resp["id"] == "updated"
    assert dummy.update_calls
    update = dummy.update_calls[0]
    assert update["container_tag"] == "user_p3"
    assert update["memory_id"] == "m1"
    assert update["new_content"] == "Meeting recap title"


def test_save_meeting_summary_adds_when_missing(monkeypatch):
    dummy = DummySupermemoryClient()
    dummy.search_results = {"results": []}
    service = MemoryService(supermemory_client=dummy, openai_client=DummyOpenAI())
    resp = service.save_meeting_summary("p4", "Discussed roadmap.")
    assert resp["id"] == "new"
    assert dummy.add_calls
    added = dummy.add_calls[0]
    assert added["container_tag"] == "user_p4"
    assert "Meeting recap title" in added["content"]
