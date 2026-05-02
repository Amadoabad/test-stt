import os
import time
import asyncio
import traceback
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import MODELS
from audio_utils import decode_to_16k_mono
from loaders.hf_loader import HFPipelineLoader, HFSeq2SeqLoader, SeamlessLoader, MMSLoader
from loaders.nemo_loader import NeMoLoader
from model_manager import ModelStatus, get_state, set_state, get_all_states

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

# Thread pool for blocking inference (one at a time)
_executor = ThreadPoolExecutor(max_workers=1)

def get_loader(model_key: str):
    if model_key not in MODELS:
        raise HTTPException(404, f"Unknown model key: {model_key}")
    if model_key not in _cache:
        cfg = MODELS[model_key]
        loader_cls = LOADER_MAP[cfg.loader]
        loader = loader_cls(cfg.id, trust_remote_code=cfg.trust_remote_code)
        loader.load()
        _cache[model_key] = loader
    return _cache[model_key]


# ── Background tasks ──────────────────────────────────────────────

def _download_task(model_key: str):
    """Download model to disk without loading into VRAM."""
    import threading
    import time
    from huggingface_hub import snapshot_download, model_info as hf_model_info
    
    cfg = MODELS[model_key]
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    local_dir = os.path.join(hf_home, "hub", model_key)
    os.makedirs(local_dir, exist_ok=True)
    
    # Get expected total size
    try:
        info = hf_model_info(cfg.id)
        total_bytes = sum(s.size for s in info.siblings if s.size)
    except Exception:
        total_bytes = 0
    
    set_state(model_key, download_size_gb=round(total_bytes / 1e9, 2))
    
    # Poll directory size in background thread
    done = threading.Event()
    def size_poller():
        while not done.is_set():
            try:
                current = sum(
                    f.stat().st_size
                    for f in Path(local_dir).rglob("*")
                    if f.is_file()
                )
                pct = min(99, current / max(total_bytes, 1) * 100)
                set_state(model_key, download_progress=pct)
            except Exception:
                pass
            time.sleep(1.5)
    
    t = threading.Thread(target=size_poller, daemon=True)
    t.start()
    
    try:
        snapshot_download(
            repo_id=cfg.id,
            local_dir=local_dir,
            ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
            token=os.environ.get("HF_TOKEN") or None,  # None = unauthenticated
        )
        done.set()
        set_state(model_key, status=ModelStatus.DOWNLOADED, download_progress=100)
    except Exception as e:
        done.set()
        set_state(model_key, status=ModelStatus.ERROR, error_msg=str(e))


def _load_task(model_key: str):
    """Load model from disk into VRAM."""
    set_state(model_key, status=ModelStatus.LOADING)
    try:
        get_loader(model_key)
        set_state(model_key, status=ModelStatus.LOADED)
    except Exception as e:
        set_state(model_key, status=ModelStatus.ERROR, error_msg=str(e))


# ── Routes ───────────────────────────────────────────────────────
class TranscribeResponse(BaseModel):
    model_key: str
    model_name: str
    text: str
    latency_ms: float

class TokenRequest(BaseModel):
    token: str


@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "cuda": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "loaded_models": list(_cache.keys()),
    }

@app.post("/set-token")
def set_token(req: TokenRequest):
    import loaders.hf_loader as hf
    token = req.token.strip() or None
    os.environ["HF_TOKEN"] = token or ""
    hf.HF_TOKEN = token          # update the module-level var live
    return {"set": bool(token)}

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
        loop = asyncio.get_event_loop()
        t0 = time.perf_counter()
        # Run blocking inference in thread pool so it doesn't block the event loop
        text = await asyncio.wait_for(
            loop.run_in_executor(_executor, loader.transcribe, wav, MODELS[model_key].lang),
            timeout=180.0,  # 3 minutes max
        )
        latency_ms = (time.perf_counter() - t0) * 1000
    except asyncio.TimeoutError:
        raise HTTPException(504, "Inference timed out after 3 minutes")
    except Exception:
        raise HTTPException(500, traceback.format_exc())

    return TranscribeResponse(
        model_key=model_key,
        model_name=MODELS[model_key].name,
        text=text,
        latency_ms=round(latency_ms, 1),
    )


@app.get("/system")
def system_info():
    """Get GPU VRAM + disk info."""
    import torch
    import shutil
    info = {
        "cuda": torch.cuda.is_available(),
        "gpu": None,
        "vram_total_gb": 0,
        "vram_used_gb": 0,
        "vram_free_gb": 0,
        "disk_free_gb": 0
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        used = torch.cuda.memory_allocated(0)
        total = props.total_memory
        info.update({
            "gpu": props.name,
            "vram_total_gb": round(total / 1e9, 1),
            "vram_used_gb": round(used / 1e9, 2),
            "vram_free_gb": round((total - used) / 1e9, 2),
        })
    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    try:
        _, _, free_d = shutil.disk_usage(hf_home)
        info["disk_free_gb"] = round(free_d / 1e9, 1)
    except Exception:
        info["disk_free_gb"] = 0
    return info


@app.get("/model-states")
def all_model_states():
    """Get status of all models at once."""
    states = get_all_states()
    return {
        key: {
            "status": states.get(key, get_state(key)).status,
            "progress": states.get(key, get_state(key)).download_progress,
            "size_gb": states.get(key, get_state(key)).download_size_gb,
            "error": states.get(key, get_state(key)).error_msg,
            "loaded": key in _cache,
        }
        for key in MODELS
    }


@app.post("/download/{model_key}")
def download_model(model_key: str, background_tasks: BackgroundTasks):
    """Download model to disk (not load to VRAM)."""
    if model_key not in MODELS:
        raise HTTPException(404, "Unknown model")
    set_state(model_key, status=ModelStatus.DOWNLOADING, download_progress=0)
    background_tasks.add_task(_download_task, model_key)
    return {"started": model_key}


@app.post("/load/{model_key}")
def load_model_route(model_key: str, background_tasks: BackgroundTasks):
    """Load model from disk into VRAM."""
    if model_key not in MODELS:
        raise HTTPException(404, "Unknown model")
    background_tasks.add_task(_load_task, model_key)
    return {"loading": model_key}


@app.delete("/unload/{model_key}")
def unload_model(model_key: str):
    """Unload model from VRAM (keep on disk)."""
    if model_key in _cache:
        _cache[model_key].unload()
        del _cache[model_key]
    # Set state back to DOWNLOADED if it was LOADED
    current_state = get_state(model_key)
    if current_state.status == ModelStatus.LOADED:
        set_state(model_key, status=ModelStatus.DOWNLOADED)
    return {"unloaded": model_key}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
