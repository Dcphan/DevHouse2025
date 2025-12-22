# tests/test_stm.py
import time
import pytest

# Update this import to your actual module filename/path
# Example: from app.stm import ShortTermMemory
from backend.stm import ShortTermMemory


@pytest.fixture
def stm():
    # short TTL to make expiry tests easy
    return ShortTermMemory(ttl_seconds=2, max_utterances=80)


def test_new_session_summary_is_empty(stm):
    s = stm.summary("s1")
    assert s["active_person_id"] is None
    assert s["utterances"] == []
    assert s["topic"] is None
    assert s["last_coach_action"] is None


def test_append_utterance_keeps_order_and_fields(stm):
    stm.append_utterance("s1", "other", "hi", ts=10.0)
    stm.append_utterance("s1", "user", "hello", ts=10.5)
    

    s = stm.summary("s1")
    print(s)
    assert len(s["utterances"]) == 2
    assert s["utterances"][0]["speaker"] == "other"
    assert s["utterances"][0]["text"] == "hi"
    assert s["utterances"][0]["ts"] == 10.0
    assert s["utterances"][1]["speaker"] == "user"
    assert s["utterances"][1]["text"] == "hello"
    assert s["utterances"][1]["ts"] == 10.5


def test_append_utterance_invalid_speaker_is_ignored(stm):
    stm.append_utterance("s1", "bot", "should be ignored", ts=1.0)

    s = stm.summary("s1")
    assert s["utterances"] == []


def test_topic_is_set(stm):
    stm.set_topic("s1", "career", 0.7)

    s = stm.summary("s1")
    assert s["topic"] == {"label": "career", "confidence": pytest.approx(0.7)}


def test_last_coach_action_is_set(stm):
    stm.set_last_coach_action("s1", "prompt", {"text": "Ask a follow-up"})

    s = stm.summary("s1")
    assert s["last_coach_action"]["type"] == "prompt"
    assert s["last_coach_action"]["payload"]["text"] == "Ask a follow-up"
