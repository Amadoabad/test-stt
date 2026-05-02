import torch
import numpy as np
import os
from transformers import (
    pipeline,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    SeamlessM4TForSpeechToText,
    Wav2Vec2ForCTC,
)
from .base import BaseLoader

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
HF_TOKEN = os.environ.get("HF_TOKEN") or None   # None = unauthenticated

class HFPipelineLoader(BaseLoader):
    """Generic ASR pipeline (covers omniASR, cohere-transcribe)."""

    def load(self):
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,
            device=DEVICE,
            torch_dtype=DTYPE,
            token=HF_TOKEN,
            trust_remote_code=self.trust_remote_code,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        result = self.pipe(
            {"array": audio, "sampling_rate": 16000},
            generate_kwargs={"language": lang},
        )
        return result["text"]


class HFSeq2SeqLoader(BaseLoader):
    """For models like Qwen3-ASR that need explicit generate() call."""

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=HF_TOKEN, trust_remote_code=self.trust_remote_code)
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=DTYPE,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=self.trust_remote_code,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                language=lang,
                task="transcribe",
                max_new_tokens=512,
            )
        return self.processor.batch_decode(ids, skip_special_tokens=True)[0]


class SeamlessLoader(BaseLoader):
    """Seamless M4T models for multilingual STT."""

    def load(self):
        self.processor = AutoProcessor.from_pretrained(self.model_id, token=HF_TOKEN, trust_remote_code=self.trust_remote_code)
        self.model = SeamlessM4TForSpeechToText.from_pretrained(
            self.model_id,
            torch_dtype=DTYPE,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=self.trust_remote_code,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        inputs = self.processor(
            audios=audio, sampling_rate=16000, return_tensors="pt"
        ).to(DEVICE)
        with torch.no_grad():
            ids = self.model.generate(**inputs, tgt_lang="arb")  # arb = Modern Standard Arabic
        return self.processor.decode(ids[0].tolist(), skip_special_tokens=True)


class MMSLoader(BaseLoader):
    """facebook/mms-1b-all — CTC model with language adapters."""

    def load(self):
        from transformers import Wav2Vec2Processor

        self.processor = Wav2Vec2Processor.from_pretrained(self.model_id, token=HF_TOKEN, trust_remote_code=self.trust_remote_code)
        self.model = Wav2Vec2ForCTC.from_pretrained(self.model_id, token=HF_TOKEN, trust_remote_code=self.trust_remote_code).to(DEVICE)
        if DTYPE == torch.float16:
            self.model = self.model.to(DTYPE)
        # Set Arabic adapter
        self.model.load_adapter("ara")
        self.processor.tokenizer.set_target_lang("ara")
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt").to(
            DEVICE
        )
        with torch.no_grad():
            logits = self.model(**inputs).logits
        ids = torch.argmax(logits, dim=-1)
        return self.processor.batch_decode(ids)[0]
    
class CohereASRLoader(BaseLoader):
    def load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            token=HF_TOKEN,
            trust_remote_code=self.trust_remote_code,
        )
        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            device_map="auto",
            token=HF_TOKEN,
            trust_remote_code=self.trust_remote_code,
        )
        self._loaded = True

    def transcribe(self, audio: np.ndarray, lang: str = "ar") -> str:
        inputs = self.processor(
            audio, sampling_rate=16000, return_tensors="pt", language=lang
        )
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        
        # Seed the decoder with the start token
        tok = self.processor.tokenizer
        start_id = tok.bos_token_id or tok.pad_token_id or 0
        print(f"[Cohere] bos={tok.bos_token_id} pad={tok.pad_token_id} eos={tok.eos_token_id} vocab={tok.vocab_size}")
        decoder_input_ids = torch.tensor(
            [[start_id]],
            device=self.model.device,
            dtype=torch.long
        )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                decoder_input_ids=decoder_input_ids,
                max_new_tokens=256)
            
        return self.processor.decode(outputs[0], skip_special_tokens=True)