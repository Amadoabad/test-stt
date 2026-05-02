from abc import ABC, abstractmethod
import numpy as np


class BaseLoader(ABC):
    """Abstract base class for all STT model loaders."""

    def __init__(self, model_id: str):
        self.model_id = model_id
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Download weights and move to GPU."""
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        """Return transcription string for 16kHz mono float32 audio."""
        ...

    def unload(self) -> None:
        """Free GPU memory (optional override)."""
        import torch
        import gc

        self._loaded = False
        torch.cuda.empty_cache()
        gc.collect()
