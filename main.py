import time
import traceback
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.config import MODELS
from backend.audio_utils import decode_to_16k_mono
from backend.loaders.hf_loader import HFPipelineLoader, HFSeq2SeqLoader, SeamlessLoader, MMSLoader
from backend.loaders.nemo_loader import NeMoLoader

app = FastAPI(title="STT Evaluation API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # tighten in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Loader dispatch ──────────────────────────────────────────────
LOADER_MAP = {
    "hf_pipeline": HFPipelineLoader,
    "hf_seq2seq":  HFSeq2SeqLoader,
    "hf_ctc":      HFPipelineLoader,
    "seamless":    SeamlessLoader,
    "mms":         MMSLoader,
    "nemo":        NeMoLoader,
}

# Warm cache: key → loader instance (loaded on first use)
_cache: dict[str, object] = {}

def get_loader(model_key: str):
    if model_key not in MODELS:
        raise HTTPException(404, f"Unknown model key: {model_key}")
    if model_key not in _cache:
        cfg = MODELS[model_key]
        loader_cls = LOADER_MAP[cfg.loader]
        loader = loader_cls(cfg.id)
        loader.load()
        _cache[model_key] = loader
    return _cache[model_key]

# ── Routes ───────────────────────────────────────────────────────
class TranscribeResponse(BaseModel):
    model_key: str
    model_name: str
    text: str
    latency_ms: float

@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(_cache.keys()),
    }

@app.get("/models")
def list_models():
    return [
        {"key": k, "name": v.name, "loader": v.loader, "notes": v.notes}
        for k, v in MODELS.items()
    ]

@app.post("/transcribe/{model_key}", response_model=TranscribeResponse)
async def transcribe(model_key: str, audio: UploadFile = File(...)):
    raw = await audio.read()
    try:
        wav = decode_to_16k_mono(raw)
    except Exception as e:
        raise HTTPException(422, f"Audio decode failed: {e}")

    try:
        loader = get_loader(model_key)
        t0 = time.perf_counter()
        text = loader.transcribe(wav, lang=MODELS[model_key].lang)
        latency_ms = (time.perf_counter() - t0) * 1000
    except Exception:
        raise HTTPException(500, traceback.format_exc())

    return TranscribeResponse(
        model_key=model_key,
        model_name=MODELS[model_key].name,
        text=text,
        latency_ms=round(latency_ms, 1),
    )

@app.delete("/unload/{model_key}")
def unload_model(model_key: str):
    if model_key in _cache:
        _cache[model_key].unload()
        del _cache[model_key]
    return {"unloaded": model_key}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
