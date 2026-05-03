import requests
import numpy as np
import os
from pathlib import Path
import tempfile
from .base import BaseLoader

# vLLM server endpoint
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")


class VLLMLoader(BaseLoader):
    """Loader for models served via vLLM server."""

    def load(self) -> None:
        """Verify vLLM server is running and accessible."""
        try:
            response = requests.get(f"{VLLM_BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                self._loaded = True
                print(f"[vLLM] Connected to server at {VLLM_BASE_URL}")
            else:
                raise RuntimeError(f"vLLM server returned status {response.status_code}")
        except requests.exceptions.RequestException as e:
            raise RuntimeError(f"Cannot connect to vLLM server at {VLLM_BASE_URL}: {e}")

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        """Send audio to vLLM server for transcription."""
        if not self._loaded:
            raise RuntimeError("vLLM loader not loaded. Call load() first.")

        # Write audio to temporary WAV file
        import soundfile as sf

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
            try:
                # Audio is 16kHz mono float32, write it directly
                sf.write(tmp_path, audio, samplerate=16000)

                # Send request to vLLM audio transcription endpoint
                headers = {}
                if VLLM_API_KEY:
                    headers["Authorization"] = f"Bearer {VLLM_API_KEY}"

                with open(tmp_path, "rb") as audio_file:
                    files = {"file": audio_file}
                    data = {"model": self.model_id}

                    response = requests.post(
                        f"{VLLM_BASE_URL}/v1/audio/transcriptions",
                        headers=headers,
                        files=files,
                        data=data,
                        timeout=60,
                    )

                if response.status_code != 200:
                    raise RuntimeError(
                        f"vLLM transcription failed: {response.status_code} {response.text}"
                    )

                result = response.json()
                return result.get("text", "")

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
