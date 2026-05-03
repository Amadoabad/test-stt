import requests
import numpy as np
import os
from pathlib import Path
import tempfile
from .base import BaseLoader

# vLLM server endpoint
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000").rstrip("/")
VLLM_API_KEY = os.environ.get("VLLM_API_KEY", "")
VLLM_AUDIO_ENDPOINT = os.environ.get("VLLM_AUDIO_ENDPOINT", "/v1/audio/transcriptions")


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

                    # Try the configured endpoint first
                    endpoints_to_try = [VLLM_AUDIO_ENDPOINT]
                    # Add alternative path if not already included
                    if VLLM_AUDIO_ENDPOINT != "/openai/v1/audio/transcriptions":
                        endpoints_to_try.append("/openai/v1/audio/transcriptions")

                    response = None
                    last_error = None

                    for endpoint in endpoints_to_try:
                        try:
                            url = f"{VLLM_BASE_URL}{endpoint}"
                            print(f"[vLLM] Trying endpoint: {url}")
                            
                            response = requests.post(
                                url,
                                headers=headers,
                                files=files,
                                data=data,
                                timeout=60,
                            )

                            if response.status_code == 200:
                                result = response.json()
                                return result.get("text", "")
                            else:
                                last_error = f"{response.status_code} {response.text}"
                                print(f"[vLLM] Endpoint returned {response.status_code}")
                        except requests.exceptions.RequestException as e:
                            last_error = str(e)
                            print(f"[vLLM] Request failed: {e}")

                    # If we get here, all endpoints failed
                    error_msg = f"All vLLM endpoints failed. Last error: {last_error}"
                    print(f"[vLLM] {error_msg}")
                    raise RuntimeError(error_msg)

            finally:
                # Clean up temporary file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
