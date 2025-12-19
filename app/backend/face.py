from fastapi import FastAPI, WebSocket
import base64, cv2, uuid, os, json
import numpy as np
from deepface import DeepFace
from collections import deque
from typing import Dict
from starlette.websockets import WebSocketDisconnect

import asyncpg
from facedb import FaceEmbeddingDB
from voice import AudioStreamProcessor, load_whisper_model

app = FastAPI()

# --- put credentials here (or env vars later) ---
PG_USER = "postgres"
PG_PASSWORD = "Tinphan1711@"   # <-- @ is safe here
PG_DB = "devhouse2025"
PG_HOST = "127.0.0.1"
PG_PORT = 5432

SCORE_WINDOW = 8
STABLE_FRAMES = 3
ENROLL_FRAMES = 12
MATCH_THRESHOLD = 0.45

tracks: Dict[str, dict] = {}
KNOWN_EMBEDDINGS: Dict[str, np.ndarray] = {}
CONTACT_NAMES: Dict[str, str] = {}
SESSION_ACTIVE: Dict[str, str] = {}  # track_id -> session_id
SESSION_CREATED: set[str] = set()
WHISPER_MODEL = None

db = FaceEmbeddingDB()

def b64_to_bgr(image_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img

def l2_normalize(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-10)

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))

def find_best_match(embedding: np.ndarray):
    best_id, best_score = None, -1.0
    for pid, ref in KNOWN_EMBEDDINGS.items():
        score = cosine_similarity(embedding, ref)
        if score > best_score:
            best_score, best_id = score, pid
    return best_id, best_score

def next_person_id() -> str:
    return str(uuid.uuid4())

def default_contact_name(person_id: str) -> str:
    short = person_id.split("-")[0]
    return f"new_person_{short}"

@app.on_event("startup")
async def startup():
    global KNOWN_EMBEDDINGS, CONTACT_NAMES, WHISPER_MODEL

    # ✅ create db_pool (NO URL parsing)
    app.state.db_pool = await asyncpg.create_pool(
        user=PG_USER,
        password=PG_PASSWORD,
        database=PG_DB,
        host=PG_HOST,
        port=PG_PORT,
        min_size=1,
        max_size=5,
    )

    # bind pool to db class
    await db.bind_pool(app.state.db_pool)

    embeddings, names = await db.load_all()
    KNOWN_EMBEDDINGS = {k: l2_normalize(v.astype(np.float32)) for k, v in embeddings.items()}
    CONTACT_NAMES = {k: names.get(k) or default_contact_name(k) for k in embeddings.keys()}

    print(f"Loaded {len(KNOWN_EMBEDDINGS)} embeddings from DB")

    try:
        model_size = os.getenv("WHISPER_MODEL", "tiny")
        WHISPER_MODEL = load_whisper_model(model_size)
        if WHISPER_MODEL:
            print(f"Loaded Faster-Whisper model: {model_size}")
        else:
            print("Whisper model not loaded (library missing or init failed).")
    except Exception as e:
        print(f"Whisper model load error: {e}")

@app.on_event("shutdown")
async def shutdown():
    # close pool owned by app
    pool = getattr(app.state, "db_pool", None)
    if pool:
        await pool.close()

@app.websocket("/ws/embedding")
async def ws_embedding(ws: WebSocket):
    await ws.accept()

    while True:
        # 1) If client disconnects, receive_json throws WebSocketDisconnect
        try:
            msg = await ws.receive_json()
        except WebSocketDisconnect:
            break

        track_id = msg.get("track_id", "unknown")
        session_id = msg.get("session_id")
        image_b64 = msg.get("image_b64")  # 2) don't use msg["image_b64"] (can KeyError)

        if not image_b64:
            # client might have disconnected already; guard send
            try:
                await ws.send_json({"track_id": track_id, "error": "Missing image_b64"})
            except WebSocketDisconnect:
                break
            continue

        # Create a meeting entry once per session (person_id may be null at start)
        if session_id and session_id not in SESSION_CREATED:
            await db.insert_meeting_session(session_id=session_id, person_id=None)
            SESSION_CREATED.add(session_id)

        try:

            # Face Embedding
            img = b64_to_bgr(image_b64)

            rep = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                enforce_detection=False,
            )[0]

            embedding = l2_normalize(np.array(rep["embedding"], dtype=np.float32))

            # Smoothing and Locking Identity
            if track_id not in tracks:
                tracks[track_id] = {
                    "score_buffer": deque(maxlen=SCORE_WINDOW),
                    "id_buffer": deque(maxlen=STABLE_FRAMES),
                    "enroll_buffer": [],
                    "locked_id": None,
                }
            track = tracks[track_id]
                # Track which session this track belong to
            if session_id:
                SESSION_ACTIVE[track_id] = session_id

            # Matching Embedding to Known People
            best_id, score = find_best_match(embedding)
            is_known = best_id is not None and score >= MATCH_THRESHOLD

            track["score_buffer"].append(score)
            track["id_buffer"].append(best_id if is_known else "UNKNOWN")
            smooth_score = float(np.mean(track["score_buffer"]))

            # Locking Identity Logic, preventing bouncing name
            if len(track["id_buffer"]) == STABLE_FRAMES:
                unique = set(track["id_buffer"])
                if len(unique) == 1:
                    track["locked_id"] = list(unique)[0]

            if track["locked_id"] == "UNKNOWN":
                track["enroll_buffer"].append(embedding)

            # Auto Enrolled Person if Unknown is Long Enough
            enrolled = False
            new_person_id = None
            new_person_name = None

            if track["locked_id"] == "UNKNOWN" and len(track["enroll_buffer"]) >= ENROLL_FRAMES:
                E = l2_normalize(np.mean(track["enroll_buffer"], axis=0).astype(np.float32))

                pid, final_score = find_best_match(E)
                if pid is None or final_score < MATCH_THRESHOLD:
                    new_person_id = next_person_id()
                    new_person_name = default_contact_name(new_person_id)
                    await db.insert_contact_with_embedding(new_person_id, new_person_name, E)
                    KNOWN_EMBEDDINGS[new_person_id] = E
                    CONTACT_NAMES[new_person_id] = new_person_name
                    enrolled = True
                    track["locked_id"] = new_person_id

                track["enroll_buffer"].clear()

            # Update meeting/contact for known identities (or newly enrolled)
            resolved_id = track["locked_id"] if track["locked_id"] != "UNKNOWN" else None
            if resolved_id:
                await db.touch_contact(resolved_id)
                if session_id:
                    # attach the session to this person (using meetings table)
                    await db.update_meeting_person(session_id, resolved_id)
            elif session_id:
                # Ensure a meeting exists for this session even if unknown yet
                # context carries session_id; person_id null for now
                await db.insert_meeting_session(session_id=session_id, person_id=None)

            # 3) client can disconnect between compute and send
            try:
                await ws.send_json({
                    "track_id": track_id,
                    "best_match": best_id,
                    "best_match_name": CONTACT_NAMES.get(best_id) if best_id else None,
                    "score": score,
                    "smooth_score": smooth_score,
                    "identity": track["locked_id"],
                    "identity_name": CONTACT_NAMES.get(track["locked_id"])
                    if track["locked_id"] and track["locked_id"] != "UNKNOWN"
                    else None,
                    "enrolled": enrolled,
                    "new_person_id": new_person_id,
                    "new_person_name": new_person_name,
                    "session_id": session_id,
                })
            except WebSocketDisconnect:
                break

        except Exception as e:
            # 4) don't crash if disconnected while trying to send error
            try:
                await ws.send_json({"track_id": track_id, "error": str(e)})
            except WebSocketDisconnect:
                break


@app.websocket("/ws/audio")
async def ws_audio(ws: WebSocket):
    await ws.accept()
    processor = AudioStreamProcessor(lambda: WHISPER_MODEL)
    session_id = None

    try:
        while True:
            message = await ws.receive()

            if "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                    session_id = data.get("session_id", session_id)
                    await ws.send_json({"type": "ACK", "session_id": session_id})
                except Exception as e:
                    await ws.send_json(
                        {
                            "type": "ERROR",
                            "session_id": session_id,
                            "error": f"Invalid control message: {e}",
                        }
                    )
                continue

            if "bytes" in message and message["bytes"] is not None:
                events = processor.process_chunk(message["bytes"], session_id)
                for ev in events:
                    await ws.send_json(ev)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await ws.send_json({"type": "ERROR", "session_id": session_id, "error": str(e)})
        except WebSocketDisconnect:
            pass
