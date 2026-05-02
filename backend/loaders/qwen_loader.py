import torch
import numpy as np
import tempfile
import os
import soundfile as sf
from .base import BaseLoader


class QwenLoader(BaseLoader):
    """Qwen3 ASR model using custom qwen_asr library."""

    def load(self):
        from qwen_asr import Qwen3ASRModel

        self.model = Qwen3ASRModel.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            max_inference_batch_size=32,
            max_new_tokens=256,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        """Transcribe audio using Qwen3ASRModel.
        
        Args:
            audio: 16kHz mono float32 audio array
            lang: Language code (e.g., "English", "Arabic", None for auto-detect)
            
        Returns:
            Transcription text string
        """
        # Create a temporary WAV file for Qwen's interface
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            tmp_path = f.name
        
        try:
            # Qwen expects file path and returns list of results
            results = self.model.transcribe(
                audio=tmp_path,
                language=lang if lang and lang != "ar" else None,
            )
            
            # Extract text from first result
            return results[0].text if results else ""
        finally:
            os.unlink(tmp_path)
