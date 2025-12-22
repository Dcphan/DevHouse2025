from __future__ import annotations

import json
import sys

from dotenv import find_dotenv, load_dotenv

load_dotenv(find_dotenv())

try:
    from .supermemory_client import SupermemoryClient, SupermemoryError
    from .memory_service import MemoryService
except ImportError:  # fallback for direct script execution
    from supermemory_client import SupermemoryClient, SupermemoryError
    from memory_service import MemoryService

CONVERSATION_DEMO = {
    "conversation": [
        {"speaker": "Me", "utterance": "Did you watch the Real vs Barca game last night?"},
        {"speaker": "Duc", "utterance": "Yeah, I did. That match was intense from the first minute."},
        {"speaker": "Me", "utterance": "I know, the pace was crazy. Real Madrid pressed so aggressively."},
        {"speaker": "Duc", "utterance": "True, but Barca's midfield control was impressive, especially in the first half."},
        {"speaker": "Me", "utterance": "That goal from Barca really shifted the momentum."},
        {"speaker": "Duc", "utterance": "Yeah, but Real's counterattacks were deadly. They always look dangerous in big games."},
        {"speaker": "Me", "utterance": "Honestly, El Clasico always delivers. It doesn't matter the season."},
        {"speaker": "Duc", "utterance": "Agreed. No matter who wins, it's always a great football match."},
    ]
}


def p(obj: object) -> None:
    print(json.dumps(obj, indent=2))


def prompt(msg: str) -> str:
    return input(msg).strip()


def main() -> int:
    client = SupermemoryClient()
    service = MemoryService(client)

    menu = """
1) Add document
2) Search memories
3) Update document
4) Update memory
5) Forget memory
6) Write structured memory (person/key/value)
7) Fetch user card
8) Save meeting summary (creates/updates last conversation title)
0) Exit
Choose: """

    while True:
        choice = prompt(menu)
        try:
            if choice == "1":
                resp = client.add_document(
                    content=prompt("Content: "),
                    container_tag=prompt("Container tag (blank ok): ") or None,
                    custom_id=prompt("Custom ID (blank ok): ") or None,
                )
                p(resp)

            elif choice == "2":
                resp = client.search_memories(
                    q=prompt("Query: "),
                    container_tag=prompt("Container tag (blank ok): ") or None,
                    threshold=float(prompt("Threshold 0-1 (default 0.6): ") or 0.6),
                    limit=int(prompt("Limit 1-100 (default 5): ") or 5),
                )
                p(resp)

            elif choice == "3":
                resp = client.update_document(
                    prompt("Document ID: "),
                    content=prompt("New content (blank skip): ") or None,
                    container_tag=prompt("Container tag (blank ok): ") or None,
                    custom_id=prompt("Custom ID (blank ok): ") or None,
                )
                p(resp)

            elif choice == "4":
                resp = client.update_memory(
                    container_tag=prompt("Container tag (required): "),
                    memory_id=(mid := prompt("Memory ID (blank if matching content): ") or None),
                    content_exact=None if mid else prompt("Exact old content (if no ID): "),
                    new_content=prompt("New content (required): "),
                )
                p(resp)

            elif choice == "5":
                resp = client.forget_memory(
                    container_tag=prompt("Container tag (required): "),
                    memory_id=(mid := prompt("Memory ID (blank if matching content): ") or None),
                    content_exact=None if mid else prompt("Exact content (if no ID): "),
                    reason=prompt("Reason (blank ok): ") or None,
                )
                p(resp)

            elif choice == "6":
                resp = service.write_memory(
                    person_id=prompt("Person ID: "),
                    key=prompt("Key: "),
                    value=prompt("Value: "),
                    confidence=float(prompt("Confidence (0-1): ") or 0.8),
                    source=prompt("Source (default conversation): ") or "conversation",
                )
                p(resp)

            elif choice == "7":
                resp = service.fetch_user_card(
                    person_id=prompt("Person ID: "),
                    limit=int(prompt("Limit (default 5): ") or 5),
                    threshold=float(prompt("Threshold (default 0.6): ") or 0.6),
                )
                p(resp)

            elif choice == "8":
                person_id = prompt("Person ID: ")
                dialogue_raw = prompt("Dialogue (paste text or JSON; blank uses demo JSON): ")
                if dialogue_raw:
                    try:
                        stm_dialogue = json.loads(dialogue_raw)
                    except json.JSONDecodeError:
                        stm_dialogue = dialogue_raw  # treat as plain text
                else:
                    stm_dialogue = CONVERSATION_DEMO
                resp = service.save_meeting_summary(person_id=person_id, stm_dialogue=stm_dialogue)
                p(resp)

            elif choice == "0":
                print("Bye.")
                return 0

            else:
                print("Invalid choice.")

        except (SupermemoryError, ValueError) as exc:
            print(f"Error: {exc}")
        except KeyboardInterrupt:
            print("\nBye.")
            return 0


if __name__ == "__main__":
    sys.exit(main())
