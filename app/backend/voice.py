import time
import uuid
from typing import Callable, List, Optional, Dict, Any

import numpy as np

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None  # type: ignore


class AudioStreamProcessor:
    """
    Simple streaming VAD + endpointing + Faster-Whisper transcription.

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

    def _transcribe(self, session_id: Optional[str]) -> Optional[Dict[str, Any]]:
        model = None
        try:
            model = self.model_provider()
        except Exception as e:  # pragma: no cover - defensive
            self._model_error = str(e)
            return {
                "type": "ERROR",
                "session_id": session_id,
                "error": f"Whisper model unavailable: {e}",
            }

        if model is None:
            return {
                "type": "ERROR",
                "session_id": session_id,
                "error": "Whisper model not loaded",
            }

        audio = np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32) / 32768.0
        segments, _ = model.transcribe(audio, language="en")
        text_parts: List[str] = []
        for seg in segments:
            t = getattr(seg, "text", "")
            if t:
                text_parts.append(t.strip())
        text = " ".join(text_parts).strip()
        if not text:
            return None
        return {
            "type": "TRANSCRIPT_FINAL",
            "session_id": session_id,
            "utterance_id": str(uuid.uuid4()),
            "text": text,
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


def load_whisper_model(model_size: str = "tiny") -> Optional["WhisperModel"]:
    if WhisperModel is None:
        return None
    return WhisperModel(model_size, device="cpu")
