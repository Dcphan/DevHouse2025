"""
Quick harness to exercise AgentTools.summarize_recent_topics
with a fixed STM-style conversation using your real OpenAI API key.
"""
from __future__ import annotations

from backend.agent_tools import AgentTools


STM_CONVERSATION = {
    "active_person_id": "user_Duc",
    "utterances": [
        {"speaker": "Me", "utterance": "Man, I just finished my CS exam today."},
        {"speaker": "Duc", "utterance": "Oh nice, how did it go?"},
        {"speaker": "Me", "utterance": "Honestly, tougher than I expected. A lot of algorithm questions."},
        {"speaker": "Duc", "utterance": "Algorithms again? What kind this time?"},
        {"speaker": "Me", "utterance": "Mostly problem solving. There was a dynamic programming question that really threw me off."},
        {"speaker": "Duc", "utterance": "DP under time pressure is brutal. Was it the state or the transition that got you?"},
        {"speaker": "Me", "utterance": "The transition. I knew what I wanted to do, but I couldn’t formalize it fast enough."},
        {"speaker": "Duc", "utterance": "Do you usually prepare a lot for exams like this?"},
        {"speaker": "Me", "utterance": "I study, but I think I focus too much on understanding instead of drilling patterns."},
        {"speaker": "Duc", "utterance": "Yeah, exams reward speed more than deep understanding sometimes."},
        {"speaker": "Me", "utterance": "Exactly. After the exam, everything suddenly feels obvious."},
        {"speaker": "Duc", "utterance": "That post-exam clarity hits hard."},
        {"speaker": "Me", "utterance": "For real. It’s also kind of exhausting. This semester feels heavier than usual."},
        {"speaker": "Duc", "utterance": "Burnout?"},
        {"speaker": "Me", "utterance": "A bit. Juggling classes, projects, and thinking about internships at the same time."},
        {"speaker": "Duc", "utterance": "Are you applying for internships right now?"},
        {"speaker": "Me", "utterance": "Yeah, mostly software and some ML roles. The market feels rough though."},
        {"speaker": "Duc", "utterance": "True. Everyone I know is stressing about it."},
        {"speaker": "Me", "utterance": "It feels like you need classes, projects, and experience all at once."},
        {"speaker": "Duc", "utterance": "What kind of roles are you leaning toward?"},
        {"speaker": "Me", "utterance": "I’m really interested in machine learning and applied AI."},
        {"speaker": "Duc", "utterance": "Nice. More research-y or industry-focused?"},
        {"speaker": "Me", "utterance": "Industry, but I like building systems that feel intelligent, not just models."},
        {"speaker": "Duc", "utterance": "That makes sense. Like end-to-end stuff?"},
        {"speaker": "Me", "utterance": "Exactly. Combining ML with real products, not just notebooks."},
        {"speaker": "Duc", "utterance": "That’s a solid direction honestly."},
        {"speaker": "Me", "utterance": "I just hope I’m not spreading myself too thin."},
        {"speaker": "Duc", "utterance": "That’s normal though. You kind of figure it out as you go."},
        {"speaker": "Me", "utterance": "Yeah. I guess finishing this exam is already a small win."},
        {"speaker": "Duc", "utterance": "Definitely. What are you doing after this?"},
        {"speaker": "Me", "utterance": "Probably just taking a break tonight. Clear my head."},
        {"speaker": "Duc", "utterance": "Sounds deserved."},
    ],
    "last_coach_action": {
        "type": "light_pivot",
        "topic": "career direction",
        "suggested_say": "What kind of projects make you feel most motivated lately?",
    },
}


def main() -> None:
    # Uses your real OpenAI API key from environment; ensure OPENAI_API_KEY is set.
    tools = AgentTools()

    print("=== STM CONVERSATION ===")
    print(f"Active person: {STM_CONVERSATION['active_person_id']}")
    print(f"Utterances: {len(STM_CONVERSATION['utterances'])}")
    print("\n=== SUMMARIZE RECENT TOPICS ===")
    result = tools.summ(STM_CONVERSATION)
    print(result)


if __name__ == "__main__":
    main()
