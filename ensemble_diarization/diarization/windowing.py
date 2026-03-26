from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

from ..core.types import ExclusiveRegion, Window, WordInfo


def build_windows_from_exclusive_regions(
    regions: Sequence[ExclusiveRegion],
    words: Sequence[WordInfo],
    min_words: int = 1,
    merge_gap_sec: float = 0.2,
) -> List[Window]:
    """
    Build single-speaker windows based on exclusive diarization regions.
    Collect words fully contained in each region, then merge adjacent windows for same speaker
    when the gap is small.
    """
    by_speaker: Dict[str, List[Window]] = {}

    # Index words by time to avoid O(N*M) scanning could be added later; keep simple first.
    for r in regions:
        wlist: List[Tuple[int, int, str]] = []
        for w in words:
            if w.t0 >= r.start and w.t1 <= r.end and w.speaker == r.speaker:
                wlist.append((w.segment_index, w.word_index, w.word))
        if len(wlist) < min_words:
            continue
        by_speaker.setdefault(r.speaker, []).append(Window(speaker=r.speaker, start=r.start, end=r.end, words=wlist))

    merged_all: List[Window] = []
    for speaker, ws in by_speaker.items():
        ws.sort(key=lambda x: x.start)
        merged: List[Window] = []
        for w in ws:
            if not merged:
                merged.append(w)
                continue
            last = merged[-1]
            if w.start - last.end <= merge_gap_sec:
                merged[-1] = Window(
                    speaker=speaker,
                    start=last.start,
                    end=w.end,
                    words=last.words + w.words,
                )
            else:
                merged.append(w)
        merged_all.extend(merged)

    merged_all.sort(key=lambda x: (x.speaker, x.start))
    return merged_all


def pick_longest_window_per_speaker(windows: Sequence[Window]) -> Dict[str, Window]:
    best: Dict[str, Window] = {}
    for w in windows:
        dur = w.end - w.start
        if w.speaker not in best:
            best[w.speaker] = w
            continue
        if dur > (best[w.speaker].end - best[w.speaker].start):
            best[w.speaker] = w
    return best

