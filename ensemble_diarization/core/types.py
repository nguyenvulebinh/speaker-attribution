from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, TypedDict


class Turn(TypedDict):
    start: float
    end: float
    speaker: str


@dataclass(frozen=True)
class SegmentInput:
    audio_bytes: bytes
    transcript: str
    language: str


@dataclass(frozen=True)
class ExclusiveRegion:
    start: float
    end: float
    speaker: str


@dataclass
class WordInfo:
    segment_index: int
    word_index: int
    word: str
    t0: float  # session time, seconds
    t1: float  # session time, seconds
    speaker: Optional[str] = None
    confidence: str = "low"  # "high" | "low" | "sent"


@dataclass
class Window:
    speaker: str
    start: float
    end: float
    words: List[Tuple[int, int, str]]  # (segment_index, word_index, word)

