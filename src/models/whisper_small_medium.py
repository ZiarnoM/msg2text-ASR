import time

import numpy as np
from faster_whisper import WhisperModel as FWModel

from ..asr_interface import ASRModel, ASRResult, ASRSegment, register_model


class _FasterWhisperBase(ASRModel):
    model_size: str

    def __init__(self, device: str = "auto", compute_type: str = "auto"):
        self.model = FWModel(self.model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sr: int) -> ASRResult:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start = time.perf_counter()
        fw_segments, info = self.model.transcribe(audio, beam_size=5)
        elapsed = time.perf_counter() - start

        segments = [ASRSegment(start=s.start, end=s.end, text=s.text.strip()) for s in fw_segments]
        full_text = " ".join(s.text for s in segments)
        conf = info.language_probability if info else 0.0

        return ASRResult(
            text=full_text,
            segments=segments,
            confidence=round(conf, 4),
            inference_time_s=round(elapsed, 2),
        )


class WhisperSmall(_FasterWhisperBase):
    model_size = "small"


class WhisperMedium(_FasterWhisperBase):
    model_size = "medium"


register_model("whisper-small", WhisperSmall)
register_model("whisper-medium", WhisperMedium)
