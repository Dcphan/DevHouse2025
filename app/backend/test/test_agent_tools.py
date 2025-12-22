from types import SimpleNamespace

from backend.agent_tools import AgentTools


class DummyOpenAI:
    """
    Minimal OpenAI stub that captures the last chat.completions.create call.
    """

    class DummyChat:
        class DummyCompletions:
            def __init__(self, content: str):
                self._content = content
                self.last_kwargs = None

            def create(self, **kwargs):
                self.last_kwargs = kwargs
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self._content))])

        def __init__(self, content: str):
            self.completions = DummyOpenAI.DummyChat.DummyCompletions(content)

    def __init__(self, content: str):
        self.chat = DummyOpenAI.DummyChat(content)


def _sample_stm():
    return {
        "active_person_id": "user_Duc",
        "utterances": [
            {"speaker": "Me", "utterance": "Man, I just finished my CS exam today."},
            {"speaker": "Duc", "utterance": "Oh nice, how did it go?"},
            {"speaker": "Me", "utterance": "Honestly, tougher than I expected. A lot of algorithm questions."},
            {"speaker": "Duc", "utterance": "Algorithms again? What kind this time?"},
        ],
        "last_coach_action": {"type": "light_pivot", "topic": "career direction"},
    }


def test_summarize_recent_topics_parses_json_and_limits_ideas():
    dummy = DummyOpenAI(
        content='{"name":"Exam debrief and AI goals","ideas":["Congratulate finishing exam","Ask about ML internships","Extra idea"]}'
    )
    tools = AgentTools(openai_client=dummy)

    result = tools.summarize_recent_topics(_sample_stm())

    assert result["name"] == "Exam debrief and AI goals"
    assert result["ideas"] == ["Congratulate finishing exam", "Ask about ML internships"]  # capped at n=2


def test_summarize_recent_topics_sends_transcript_to_openai():
    dummy = DummyOpenAI(content='{"name":"Exam recap","ideas":["One","Two"]}')
    tools = AgentTools(openai_client=dummy)

    tools.summarize_recent_topics(_sample_stm())

    sent = dummy.chat.completions.last_kwargs
    assert sent is not None
    user_msg = [m for m in sent["messages"] if m["role"] == "user"][0]
    assert "finished my CS exam" in user_msg["content"]
    assert "Algorithms again?" in user_msg["content"]


def test_summarize_recent_topics_handles_empty_input():
    dummy = DummyOpenAI(content="")
    tools = AgentTools(openai_client=dummy)

    result = tools.summarize_recent_topics(None)

    assert result["name"] == "No conversation yet"
    assert result["ideas"] == []


def test_recognize_person_name_reads_transcript_and_parses_json():
    dummy = DummyOpenAI(content='{"name":"Duc Tran","confidence":0.82,"evidence":"He said he is Duc."}')
    tools = AgentTools(openai_client=dummy)

    result = tools.recognize_person_name(_sample_stm())

    assert result["name"] == "Duc Tran"
    assert abs(result["confidence"] - 0.82) < 1e-6
    assert "Duc" in result["evidence"]
    sent = dummy.chat.completions.last_kwargs
    assert sent is not None
    user_msg = [m for m in sent["messages"] if m["role"] == "user"][0]
    assert "finished my CS exam" in user_msg["content"]
    assert "Algorithms again?" in user_msg["content"]


def test_recognize_person_name_handles_empty_dialogue():
    dummy = DummyOpenAI(content='{"name":"Someone","confidence":0.5,"evidence":"Placeholder"}')
    tools = AgentTools(openai_client=dummy)

    result = tools.recognize_person_name(None)

    assert result == {"name": "", "confidence": 0.0, "evidence": ""}
    assert dummy.chat.completions.last_kwargs is None
