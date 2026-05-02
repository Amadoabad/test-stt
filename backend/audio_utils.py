import io
import numpy as np
import soundfile as sf
import scipy.signal as sps


def decode_to_16k_mono(raw_bytes: bytes) -> np.ndarray:
    """
    Decode raw audio bytes (any format soundfile can read) to
    16kHz mono float32 numpy array.
    Returns array of shape (N,).
    """
    buf = io.BytesIO(raw_bytes)
    audio, sr = sf.read(buf, dtype="float32", always_2d=True)
    # Mix down to mono
    audio = audio.mean(axis=1)
    # Resample to 16kHz if needed
    if sr != 16000:
        num_samples = int(len(audio) * 16000 / sr)
        audio = sps.resample(audio, num_samples)
    return audio.astype(np.float32)
