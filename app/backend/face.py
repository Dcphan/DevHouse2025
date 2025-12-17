from fastapi import FastAPI, WebSocket
import base64
import cv2
import numpy as np
from deepface import DeepFace

app = FastAPI()

def b64_to_bgr(image_b64: str) -> np.ndarray:
    img_bytes = base64.b64decode(image_b64)
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image")
    return img

@app.get("/")
def hi():
    print("hi")

@app.websocket("/ws/embedding")
async def ws_embedding(ws: WebSocket):
    await ws.accept()
    while True:
        msg = await ws.receive_json()

        track_id = msg.get("track_id", "unknown")
        image_b64 = msg["image_b64"]  # base64 string (no data: prefix)

        try:
            img = b64_to_bgr(image_b64)

            # Returns list[dict], we take the first face (since you're sending a cropped face)
            rep = DeepFace.represent(
                img_path=img,
                model_name="ArcFace",
                enforce_detection=False,  # cropped face, so keep this false
            )[0]

            embedding = rep["embedding"]  # python list of floats

            await ws.send_json({
                "track_id": track_id,
                "model": "ArcFace",
                "embedding_dim": len(embedding),
                "embedding": embedding,  # big payload, but fine for step 1
            })

        except Exception as e:
            await ws.send_json({
                "track_id": track_id,
                "error": str(e),
            })
