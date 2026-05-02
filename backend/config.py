from dataclasses import dataclass
from typing import Literal


@dataclass
class ModelConfig:
    id: str  # HuggingFace model ID or NGC path
    name: str  # display name shown in UI
    loader: Literal[
        "hf_pipeline", "hf_seq2seq", "hf_ctc", "nemo", "seamless", "mms", "cohere", "qwen", "omni"
    ]
    lang: str = "ar"
    notes: str = ""
    trust_remote_code: bool = False


MODELS: dict[str, ModelConfig] = {
    "qwen3-asr": ModelConfig(
        id="Qwen/Qwen3-ASR-1.7B",
        name="Qwen3 ASR 1.7B",
        loader="qwen",
        notes="Uses AutoModelForSpeechSeq2Seq",
    ),
    "cohere-transcribe": ModelConfig(
        id="CohereLabs/cohere-transcribe-03-2026",
        name="Cohere Transcribe",
        loader="cohere",
        trust_remote_code=True,
    ),
    "omni-300m": ModelConfig(
        id="omniASR-LLM-300M",
        name="OmniASR 300M",
        loader="omni",
        trust_remote_code=True,
    ),
    "omni-1b": ModelConfig(
        id="omniASR-LLM-1B",
        name="OmniASR 1B",
        loader="omni",
        trust_remote_code=True,
    ),
    "omni-3b": ModelConfig(
        id="omniASR-LLM-3B",
        name="OmniASR 3B",
        loader="omni",
        trust_remote_code=True,
    ),
    "omni-7b": ModelConfig(
        id="omniASR-LLM-7B",
        name="OmniASR 7B",
        loader="omni",
        trust_remote_code=True,
    ),
    "omni-7b-zs": ModelConfig(
        id="omniASR-LLM-7B-ZS",
        name="OmniASR 7B ZS",
        loader="omni",
        trust_remote_code=True,
    ),
    "nvidia-pcd": ModelConfig(
        id="nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0",
        name="NVIDIA FastConformer PCD",
        loader="nemo",
    ),
    "nvidia-pc": ModelConfig(
        id="nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0",
        name="NVIDIA FastConformer PC",
        loader="nemo",
    ),
    "seamless-v2": ModelConfig(
        id="facebook/seamless-m4t-v2-large",
        name="SeamlessM4T v2 Large",
        loader="seamless",
        notes="Uses SeamlessM4TForSpeechToText",
    ),
    "mms-1b": ModelConfig(
        id="facebook/mms-1b-all",
        name="MMS 1B All",
        loader="mms",
        notes="Uses Wav2Vec2ForCTC with mms-300m processor",
    ),
}
