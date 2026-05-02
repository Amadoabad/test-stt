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
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, audio, 16000)
            tmp_path = f.name
        try:
            result = self.model.transcribe([tmp_path])
            # result[0] is a Hypothesis object for hybrid RNNT/CTC models
            # extract .text, falling back through list nesting
            hyp = result[0]
            if isinstance(hyp, list):
                hyp = hyp[0]
            # Hypothesis object has a .text attribute; plain string passes through
            return hyp.text if hasattr(hyp, "text") else str(hyp)
        finally:
            os.unlink(tmp_path)
