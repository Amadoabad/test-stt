import threading
from dataclasses import dataclass, field
from enum import Enum


class ModelStatus(str, Enum):
    NOT_DOWNLOADED = "not_downloaded"
    DOWNLOADING    = "downloading"
    DOWNLOADED     = "downloaded"     # on disk, not in VRAM
    LOADING        = "loading"        # being moved to GPU
    LOADED         = "loaded"         # in VRAM, ready
    ERROR          = "error"


@dataclass
class ModelState:
    status: ModelStatus = ModelStatus.NOT_DOWNLOADED
    download_progress: float = 0.0    # 0–100
    download_size_gb: float = 0.0
    error_msg: str = ""


# Global state — one entry per model key
_states: dict[str, ModelState] = {}
_lock = threading.Lock()


def get_state(key: str) -> ModelState:
    """Get the state object for a model key."""
    with _lock:
        if key not in _states:
            _states[key] = ModelState()
        return _states[key]


def set_state(key: str, **kwargs):
    """Update state fields for a model key."""
    with _lock:
        s = _states.setdefault(key, ModelState())
        for k, v in kwargs.items():
            setattr(s, k, v)


def get_all_states() -> dict[str, ModelState]:
    """Get a snapshot of all states."""
    with _lock:
        return {k: ModelState(
            status=v.status,
            download_progress=v.download_progress,
            download_size_gb=v.download_size_gb,
            error_msg=v.error_msg
        ) for k, v in _states.items()}
