from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

VOICE_PROFILE_STORE = Path(__file__).resolve().parent / "voice_profiles.json"
_STORE_LOCK = threading.Lock()


def _read_store() -> Dict[str, Any]:
    if not VOICE_PROFILE_STORE.exists():
        return {}
    try:
        raw = VOICE_PROFILE_STORE.read_text()
        data = json.loads(raw)
    except (json.JSONDecodeError, OSError):
        return {}
    if not isinstance(data, dict):
        return {}
    return data


def _write_store(data: Dict[str, Any]) -> None:
    VOICE_PROFILE_STORE.parent.mkdir(parents=True, exist_ok=True)
    VOICE_PROFILE_STORE.write_text(json.dumps(data, indent=2))


def _sanitize(profile: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in profile.items() if k != "embedding"}


def get_profile() -> Optional[Dict[str, Any]]:
    """Return the current voice profile metadata, without the embedding."""
    with _STORE_LOCK:
        data = _read_store()
        profile = data.get("profile")
        if not isinstance(profile, dict):
            return None
        return _sanitize(profile)


def load_full_profile() -> Optional[Dict[str, Any]]:
    """Return the raw stored profile including the embedding."""
    with _STORE_LOCK:
        data = _read_store()
        profile = data.get("profile")
        if not isinstance(profile, dict):
            return None
        return profile


def save_profile(name: str, embedding: np.ndarray) -> Dict[str, Any]:
    """Persist a single voice profile, replacing any existing profile."""
    sanitized_name = name.strip() or "Unnamed"
    profile = {
        "id": str(uuid.uuid4()),
        "name": sanitized_name,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "embedding_length": len(embedding),
        "embedding": embedding.astype(np.float32).tolist(),
    }
    with _STORE_LOCK:
        data = _read_store()
        data["profile"] = profile
        _write_store(data)
    return _sanitize(profile)
