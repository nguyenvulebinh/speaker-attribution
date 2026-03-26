from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import time

import torch

from .audio_io import SAMPLE_RATE, decode_audio_bytes_to_waveform_16k_mono


@dataclass(frozen=True)
class AssembledSession:
    waveform: torch.Tensor  # (1, N) at 16kHz
    segment_offsets_sec: List[float]
    segment_lengths_sec: List[float]


def assemble_session(
    audio_bytes_list: List[bytes],
    silence_gap_sec: float = 0.3,
    timing: Optional[Dict[str, float]] = None,
) -> AssembledSession:
    def _add_time(key: str, dt: float) -> None:
        if timing is None:
            return
        timing[key] = timing.get(key, 0.0) + float(dt)

    if silence_gap_sec < 0:
        raise ValueError("silence_gap_sec must be >= 0")

    gap_samples = int(silence_gap_sec * SAMPLE_RATE)
    gap = torch.zeros(1, gap_samples, dtype=torch.float32)

    waveforms: List[torch.Tensor] = []
    offsets: List[float] = []
    lengths: List[float] = []

    cur_samples = 0
    for idx, b in enumerate(audio_bytes_list):
        offsets.append(cur_samples / SAMPLE_RATE)
        try:
            t0 = time.perf_counter()
            w = decode_audio_bytes_to_waveform_16k_mono(b)
            _add_time("assembler_decode_total", time.perf_counter() - t0)
        except Exception:
            # Some segments may be empty header-only WAVs; treat as short silence.
            w = torch.zeros(1, max(1, int(0.01 * SAMPLE_RATE)), dtype=torch.float32)
        waveforms.append(w)
        seg_len_sec = w.shape[1] / SAMPLE_RATE
        lengths.append(seg_len_sec)
        cur_samples += w.shape[1]
        if idx != len(audio_bytes_list) - 1 and gap_samples > 0:
            waveforms.append(gap)
            cur_samples += gap_samples

    if not waveforms:
        raise ValueError("No segments provided.")
    t0 = time.perf_counter()
    session = torch.cat(waveforms, dim=1)
    _add_time("assembler_cat_total", time.perf_counter() - t0)
    return AssembledSession(waveform=session, segment_offsets_sec=offsets, segment_lengths_sec=lengths)

