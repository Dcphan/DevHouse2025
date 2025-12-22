import os
import time
import uuid
from typing import Callable, List, Optional, Dict, Any

import numpy as np

try:
    import torch
except Exception:
    torch = None  # type: ignore

try:
    from resemblyzer import VoiceEncoder
except Exception:
    VoiceEncoder = None  # type: ignore

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # type: ignore


class AudioStreamProcessor:
    """
    Streaming VAD + endpointing.
    - Speaker ID: Resemblyzer embeddings (YOU/OTHER/UNCERTAIN) with cosine + smoothing.
    - ASR: Faster-Whisper (if available).

    Expected audio: mono, 16-bit PCM, 16 kHz, 200 ms chunks.
    """

    def __init__(
        self,
        model_provider: Callable[[], Optional["WhisperModel"]],
        sample_rate: int = 16_000,
        frame_ms: int = 200,
        endpoint_ms: int = 1000,
        vad_threshold_db: float = 35.0,
        max_buffer_ms: int = 10_000,
    ) -> None:
        self.model_provider = model_provider
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.samples_per_frame = int(sample_rate * frame_ms / 1000)
        self.endpoint_ms = endpoint_ms
        self.vad_threshold_db = vad_threshold_db
        self.max_buffer_ms = max_buffer_ms

        # Speaker identification state
        self.spk_model: Optional[VoiceEncoder] = None
        self.user_embedding: Optional[np.ndarray] = None
        self.sim_high = float(os.getenv("SPEAKER_SIM_HIGH", "0.7"))
        self.sim_low = float(os.getenv("SPEAKER_SIM_LOW", "0.55"))

        self.in_speech = False
        self.silence_ms = 0
        self.buffer = bytearray()
        self.last_speech_ts = 0.0
        self._model_error = None

    def _rms_db(self, chunk: bytes) -> float:
        if not chunk:
            return -120.0
        data = np.frombuffer(chunk, dtype=np.int16).astype(np.float32)
        if data.size == 0:
            return -120.0
        rms = np.sqrt(np.mean(np.square(data))) + 1e-8
        return 20 * np.log10(rms / 32768.0 + 1e-8)

    def _ensure_speaker_model(self) -> Optional[VoiceEncoder]:
        if self.spk_model:
            return self.spk_model
        if VoiceEncoder is None:
            self._model_error = "Resemblyzer not available"
            print("[AUDIO] Resemblyzer import failed; install resemblyzer", flush=True)
            return None
        try:
            self.spk_model = VoiceEncoder()
            return self.spk_model
        except Exception as e:
            self._model_error = f"Speaker model load failed: {e}"
            print(f"[AUDIO] speaker model load failed: {e}", flush=True)
            return None

    def _speaker_embed(self, audio: np.ndarray) -> Optional[np.ndarray]:
        model = self._ensure_speaker_model()
        if model is None:
            if self._model_error:
                print(f"[AUDIO] speaker model unavailable: {self._model_error}", flush=True)
            return None
        try:
            # Resemblyzer expects float32, -1..1, 16k
            vec = model.embed_utterance(audio.astype(np.float32))
            vec = vec / (np.linalg.norm(vec) + 1e-9)
            return vec
        except Exception as e:
            self._model_error = f"Speaker embed failed: {e}"
            print(f"[AUDIO] speaker embed failed: {e}", flush=True)
            return None

    def _label_speaker(self, audio: np.ndarray) -> tuple[str, float]:
        emb = self._speaker_embed(audio)
        if emb is None:
            return "Unknown", 0.0

        if self.user_embedding is None:
            self.user_embedding = emb
            return "YOU", 1.0

        sim = float(np.dot(self.user_embedding, emb))

        if sim >= self.sim_high:
            self.user_embedding = 0.9 * self.user_embedding + 0.1 * emb
            self.user_embedding = self.user_embedding / (np.linalg.norm(self.user_embedding) + 1e-9)
            return "YOU", sim

        if sim < self.sim_low:
            return "OTHER", sim

        return "UNCERTAIN", sim

    def _transcribe(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        audio = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32) / 32768.0
        speaker_label, speaker_similarity = self._label_speaker(audio)

        model = None
        try:
            model = self.model_provider()
        except Exception as e:  # pragma: no cover - defensive
            self._model_error = str(e)
            model = None

        if model is None:
            return {
                "type": "ERROR",
                "session_id": session_id,
                "error": self._model_error or "Whisper model not loaded",
                "speaker_label": speaker_label,
                "speaker_similarity": speaker_similarity,
            }

        segments, _ = model.transcribe(audio, language="en")
        text_parts: List[str] = []
        for seg in segments:
            t = getattr(seg, "text", "")
            if t:
                text_parts.append(t.strip())
        text = " ".join(text_parts).strip()
        if not text:
            return None

        # Debug print for backend visibility
        try:
            print(
                f"[AUDIO] speaker={speaker_label} sim={speaker_similarity:.3f} text={text}",
                flush=True,
            )
        except Exception:
            pass

        return {
            "type": "TRANSCRIPT_FINAL",
            "session_id": session_id,
            "utterance_id": str(uuid.uuid4()),
            "text": text,
            "speaker_label": speaker_label,
            "speaker_similarity": speaker_similarity,
        }

    def process_chunk(self, chunk: bytes, session_id: Optional[str]) -> List[Dict[str, Any]]:
        events: List[Dict[str, Any]] = []
        level_db = self._rms_db(chunk)
        is_speech = level_db >= -self.vad_threshold_db  # simple energy gate

        # Track silence
        if not is_speech:
            self.silence_ms += self.frame_ms
            events.append(
                {
                    "type": "SIGNAL",
                    "signal": "silence_ms",
                    "value": self.silence_ms,
                    "session_id": session_id,
                }
            )
        else:
            self.silence_ms = 0

        # State transitions
        if is_speech and not self.in_speech:
            self.in_speech = True
            self.last_speech_ts = time.time()
            events.append(
                {
                    "type": "SIGNAL",
                    "signal": "speech_started",
                    "session_id": session_id,
                }
            )

        if self.in_speech and is_speech:
            self.buffer.extend(chunk)

        if self.in_speech and not is_speech and self.silence_ms >= self.endpoint_ms:
            # Endpoint hit
            events.append(
                {
                    "type": "SIGNAL",
                    "signal": "speech_ended",
                    "session_id": session_id,
                }
            )
            transcript = self._transcribe(session_id)
            if transcript:
                events.append(transcript)
            # reset
            self.in_speech = False
            self.silence_ms = 0
            self.buffer = bytearray()

        # Safety: flush overly long buffers to avoid memory bloat
        if len(self.buffer) > int(self.sample_rate * (self.max_buffer_ms / 1000) * 2):
            self.buffer = bytearray()
            self.in_speech = False
            self.silence_ms = 0

        return events


def load_whisper_model(model_size: str = "base") -> Optional["WhisperModel"]:
    if WhisperModel is None:
        return None
    return WhisperModel(model_size, device="cpu")