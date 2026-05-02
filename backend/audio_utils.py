import io
import subprocess
import tempfile
import os
from pathlib import Path
import numpy as np
import soundfile as sf
import scipy.signal as sps


def decode_to_16k_mono(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw audio bytes to 16kHz mono float32.
    Uses ffmpeg for WebM/Opus/OGG/MP4; falls back to soundfile for WAV/FLAC.
    """
    # Try soundfile first (fast path for WAV/FLAC)
    try:
        buf = io.BytesIO(raw_bytes)
        audio, sr = sf.read(buf, dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sr != 16000:
            n = int(len(audio) * 16000 / sr)
            audio = sps.resample(audio, n)
        return audio.astype(np.float32)
    except Exception:
        pass  # fall through to ffmpeg

    # ffmpeg path — handles WebM, Opus, MP4, OGG, etc.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".input") as inp:
        inp.write(raw_bytes)
        inp_path = inp.name
    out_path = inp_path + ".wav"
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-i", inp_path,
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                out_path,
            ],
            check=True,
        )
        audio, _ = sf.read(out_path, dtype="float32")
        return audio
    finally:
        os.unlink(inp_path)
        if os.path.exists(out_path):
            os.unlink(out_path)
