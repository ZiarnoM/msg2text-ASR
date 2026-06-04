import time

import numpy as np
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

from ..asr_interface import ASRModel, ASRResult, ASRSegment, register_model

WINDOW_S = 20
OVERLAP_S = 3


class Wav2Vec2XLSR(ASRModel):
    def __init__(self, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        model_id = "jonatasgrosman/wav2vec2-large-xlsr-53-polish"
        self.processor = Wav2Vec2Processor.from_pretrained(model_id)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_id).to(device)

    def transcribe(self, audio: np.ndarray, sr: int) -> ASRResult:
        start = time.perf_counter()

        window_samples = WINDOW_S * sr
        stride_samples = (WINDOW_S - OVERLAP_S) * sr

        segments: list[ASRSegment] = []

        for pos in range(0, max(1, len(audio)), stride_samples):
            chunk = audio[pos:pos + window_samples]

            inputs = self.processor(
                chunk, sampling_rate=sr, return_tensors="pt", padding=True
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            text = self.processor.batch_decode(predicted_ids)[0].strip()

            if text:
                chunk_start = pos / sr
                chunk_end = min((pos + len(chunk)) / sr, len(audio) / sr)
                segments.append(ASRSegment(start=round(chunk_start, 1), end=round(chunk_end, 1), text=text))

        elapsed = time.perf_counter() - start
        full_text = " ".join(s.text for s in segments)

        return ASRResult(
            text=full_text,
            segments=segments,
            inference_time_s=round(elapsed, 2),
        )


register_model("wav2vec2-xlsr", Wav2Vec2XLSR)
