import torch
import numpy as np
import tempfile
import os
import soundfile as sf
from .base import BaseLoader


class OmniLoader(BaseLoader):
    """Omni ASR model using custom omnilingual-asr library."""

    def load(self):
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        self.model = ASRInferencePipeline(
            model_card=self.model_id
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
