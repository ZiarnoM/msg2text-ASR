from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


@dataclass
class ASRSegment:
    start: float
    end: float
    text: str


@dataclass
class ASRResult:
    text: str
    segments: list[ASRSegment] = field(default_factory=list)
    confidence: float = 0.0
    inference_time_s: float = 0.0


class ASRModel(ABC):
    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int) -> ASRResult:
        ...


_model_registry: dict[str, type[ASRModel]] = {}


def register_model(name: str, model_cls: type[ASRModel]):
    _model_registry[name] = model_cls


def get_model(name: str, **kwargs) -> ASRModel:
    cls = _model_registry.get(name)
    if cls is None:
        raise ValueError(f"Unknown model: {name}. Available: {list(_model_registry.keys())}")
    return cls(**kwargs)


def list_models() -> list[str]:
    return list(_model_registry.keys())
