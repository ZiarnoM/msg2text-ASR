import time

import numpy as np
from faster_whisper import WhisperModel as FWModel

from ..asr_interface import ASRModel, ASRResult, ASRSegment, register_model


class WhisperLargeV3(ASRModel):
    def __init__(self, device: str = "auto", compute_type: str = "auto"):
        self.model = FWModel("large-v3", device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, sr: int) -> ASRResult:
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start = time.perf_counter()
        fw_segments, info = self.model.transcribe(audio, beam_size=5)
        elapsed = time.perf_counter() - start

        segments = []
        for seg in fw_segments:
            segments.append(ASRSegment(start=seg.start, end=seg.end, text=seg.text.strip()))

        full_text = " ".join(seg.text for seg in segments)

        conf = info.language_probability if info else 0.0

        return ASRResult(
            text=full_text,
            segments=segments,
            confidence=round(conf, 4),
            inference_time_s=round(elapsed, 2),
        )


register_model("whisper-large-v3", WhisperLargeV3)
