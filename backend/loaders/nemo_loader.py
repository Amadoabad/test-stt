import numpy as np
import tempfile
import os
import soundfile as sf
from .base import BaseLoader


class NeMoLoader(BaseLoader):
    """NVIDIA NeMo ASR models (FastConformer, etc.)."""

    def load(self):
        import nemo.collections.asr as nemo_asr

        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_id)
        self.model.eval()
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        # NeMo requires a file path, so write a temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            tmp_path = f.name
        try:
            result = self.model.transcribe([tmp_path])
            # result is a list; for hybrid models it may be (hyps, scores)
            if isinstance(result[0], list):
                return result[0][0]
            return result[0]
        finally:
            os.unlink(tmp_path)
