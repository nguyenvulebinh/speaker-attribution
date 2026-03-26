from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..core.types import ExclusiveRegion, Turn


def build_exclusive_regions(turns: Sequence[Turn]) -> List[ExclusiveRegion]:
    """
    Convert (possibly overlapping) diarization turns into exclusive single-speaker regions.
    Conservative: keep only atomic intervals where exactly one speaker is active.
    """
    if not turns:
        return []

    boundaries: List[float] = []
    for t in turns:
        boundaries.append(float(t["start"]))
        boundaries.append(float(t["end"]))
    boundaries = sorted(set(boundaries))
    if len(boundaries) < 2:
        return []

    # Pre-group turns per speaker for faster overlap checks.
    by_speaker: Dict[str, List[Tuple[float, float]]] = {}
    for t in turns:
        s = str(t["speaker"])
        by_speaker.setdefault(s, []).append((float(t["start"]), float(t["end"])))

    regions: List[ExclusiveRegion] = []
    for b0, b1 in zip(boundaries[:-1], boundaries[1:]):
        if b1 <= b0:
            continue
        active = []
        for s, spans in by_speaker.items():
            # any overlap with [b0, b1)
            for a, b in spans:
                if not (b0 >= b or b1 <= a):
                    active.append(s)
                    break
        if len(active) == 1:
            regions.append(ExclusiveRegion(start=b0, end=b1, speaker=active[0]))

    # Merge adjacent regions of same speaker
    merged: List[ExclusiveRegion] = []
    for r in regions:
        if not merged:
            merged.append(r)
            continue
        last = merged[-1]
        if r.speaker == last.speaker and abs(r.start - last.end) < 1e-6:
            merged[-1] = ExclusiveRegion(start=last.start, end=r.end, speaker=last.speaker)
        else:
            merged.append(r)
    return merged


def lookup_exclusive_speaker(regions: Sequence[ExclusiveRegion], t0: float, t1: float) -> str | None:
    """
    Return speaker label if [t0,t1] is fully contained in one exclusive region.
    """
    if t1 <= t0:
        return None
    for r in regions:
        if t0 >= r.start and t1 <= r.end:
            return r.speaker
    return None

