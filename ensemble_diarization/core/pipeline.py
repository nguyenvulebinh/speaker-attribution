from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..alignment.alignment_client import align_words_http
from ..diarization.exclusive_regions import build_exclusive_regions, lookup_exclusive_speaker
from ..diarization.diarization_client import diarize
from ..diarization.windowing import build_windows_from_exclusive_regions, pick_longest_window_per_speaker
from ..embedding.repair import RepairResult, compute_stable_speaker_pools, repair_embeddings, repair_embeddings_central_pool
from ..embedding.reference_embedding import build_speaker_reference_embeddings
from ..embedding.speaker_attribute_client import infer_word_speaker_embeddings_http
from ..io.audio_io import slice_waveform, waveform_to_wav_bytes_pcm16
from ..io.session_assembler import assemble_session
from .types import ExclusiveRegion, SegmentInput, Turn, Window, WordInfo

_SENT_END_PUNCT = (".", "?", "!")


def _split_words(text: str) -> List[str]:
    text = text.strip()
    return text.split() if text else []


def _split_sentence_ranges(words: Sequence[str]) -> List[Tuple[int, int]]:
    if not words:
        return []
    ranges: List[Tuple[int, int]] = []
    start = 0
    for i, w in enumerate(words):
        if w.endswith(_SENT_END_PUNCT):
            ranges.append((start, i))
            start = i + 1
    if start <= len(words) - 1:
        ranges.append((start, len(words) - 1))
    return ranges


def _speaker_by_overlap(turns: Sequence[Turn], t0: float, t1: float) -> Optional[str]:
    if t1 <= t0:
        return None
    overlap_by_spk: Dict[str, float] = {}
    for tr in turns:
        s = float(tr["start"])
        e = float(tr["end"])
        if e <= t0 or s >= t1:
            continue
        ov = min(e, t1) - max(s, t0)
        if ov > 0:
            spk = str(tr["speaker"])
            overlap_by_spk[spk] = overlap_by_spk.get(spk, 0.0) + ov
    if not overlap_by_spk:
        return None
    return max(overlap_by_spk.items(), key=lambda kv: kv[1])[0]


def _propagate_sentence_speakers(
    words: Sequence[str],
    spans_sec: Sequence[Tuple[float, float]],
    speakers: List[Optional[str]],
    confidences: List[str],
    turns: Sequence[Turn],
) -> None:
    if not words:
        return
    if not (len(words) == len(spans_sec) == len(speakers) == len(confidences)):
        raise ValueError("words/spans/speakers/confidences lengths must match")

    for a, b in _split_sentence_ranges(words):
        counts: Dict[str, int] = {}
        for i in range(a, b + 1):
            spk = speakers[i]
            if spk is None:
                continue
            counts[str(spk)] = counts.get(str(spk), 0) + 1

        if counts:
            sent_spk = max(counts.items(), key=lambda kv: kv[1])[0]
        else:
            t0 = min(spans_sec[i][0] for i in range(a, b + 1))
            t1 = max(spans_sec[i][1] for i in range(a, b + 1))
            sent_spk = _speaker_by_overlap(turns, t0, t1)

        if sent_spk is None:
            continue

        for i in range(a, b + 1):
            if speakers[i] is None:
                speakers[i] = sent_spk
                confidences[i] = "sent"


@dataclass(frozen=True)
class PipelineConfig:
    diarization_url: str = "http://0.0.0.0:5022/diarize"
    alignment_url: str = "http://0.0.0.0:5023/align_words"
    speaker_attribute_base_url: str = "http://0.0.0.0:5024"
    silence_gap_sec: float = 0.3
    min_words_per_window: int = 2
    merge_gap_sec: float = 0.2
    repair_policy: str = "central_pool"  # "central_pool" | "reference"
    central_similarity_threshold: float = 0.9
    central_center_top_k: int = 10
    central_max_pool_per_speaker: int = 50
    central_replace_mode: str = "centroid"  # "random" | "best" | "centroid"
    central_random_seed: int = 0
    sentence_level_speaker: bool = True
    segment_level_embeddings: bool = True


@dataclass(frozen=True)
class PipelineOutput:
    segments: List[List[object]]
    speaker_pools: Dict[str, List[List[float]]]


def _compute_base(
    segments: Sequence[SegmentInput],
    cfg: PipelineConfig = PipelineConfig(),
    timing: Optional[Dict[str, float]] = None,
    timing_meta: Optional[Dict[str, Any]] = None,
) -> PipelineOutput:
    if not segments:
        return PipelineOutput(segments=[], speaker_pools={})

    def _add_time(key: str, dt: float) -> None:
        if timing is None:
            return
        timing[key] = timing.get(key, 0.0) + float(dt)

    def _inc_meta(key: str, by: int = 1) -> None:
        if timing_meta is None:
            return
        timing_meta[key] = int(timing_meta.get(key, 0)) + int(by)

    audio_bytes_list = [s.audio_bytes for s in segments]
    t0 = time.perf_counter()
    assembled = assemble_session(audio_bytes_list, silence_gap_sec=cfg.silence_gap_sec, timing=timing)
    _add_time("assemble_session_total", time.perf_counter() - t0)
    session_waveform = assembled.waveform

    t0 = time.perf_counter()
    session_wav_bytes = waveform_to_wav_bytes_pcm16(session_waveform)
    _add_time("session_encode_for_diarize", time.perf_counter() - t0)

    t0 = time.perf_counter()
    turns: List[Turn] = diarize(session_wav_bytes, service_url=cfg.diarization_url)
    _add_time("diarize_request", time.perf_counter() - t0)
    _inc_meta("diarize_requests", 1)

    t0 = time.perf_counter()
    exclusive_regions: List[ExclusiveRegion] = build_exclusive_regions(turns)
    _add_time("build_exclusive_regions", time.perf_counter() - t0)

    all_words: List[WordInfo] = []
    per_segment_words: List[List[str]] = []
    per_segment_spans: List[List[Tuple[float, float]]] = []

    for seg_idx, seg in enumerate(segments):
        orig_words = _split_words(seg.transcript)
        per_segment_words.append(orig_words)
        if not orig_words:
            per_segment_spans.append([])
            continue

        t0 = time.perf_counter()
        seg_wav = slice_waveform(
            session_waveform,
            assembled.segment_offsets_sec[seg_idx],
            assembled.segment_offsets_sec[seg_idx] + assembled.segment_lengths_sec[seg_idx],
        )
        seg_wav_bytes = waveform_to_wav_bytes_pcm16(seg_wav)
        _add_time("align_prep_total", time.perf_counter() - t0)

        t0 = time.perf_counter()
        spans = align_words_http(
            audio_wav_bytes=seg_wav_bytes,
            transcript=seg.transcript,
            service_url=cfg.alignment_url,
        )
        _add_time("align_request_total", time.perf_counter() - t0)
        _inc_meta("align_words_requests", 1)
        spans_sec = [
            (sp.t0 + assembled.segment_offsets_sec[seg_idx], sp.t1 + assembled.segment_offsets_sec[seg_idx])
            for sp in spans
        ]
        per_segment_spans.append(spans_sec)

        speakers_seg: List[Optional[str]] = []
        confs_seg: List[str] = []
        for (t0, t1) in spans_sec[: len(orig_words)]:
            spk = lookup_exclusive_speaker(exclusive_regions, t0, t1)
            speakers_seg.append(spk)
            confs_seg.append("high" if spk is not None else "low")

        if cfg.sentence_level_speaker and speakers_seg:
            _propagate_sentence_speakers(orig_words, spans_sec[: len(orig_words)], speakers_seg, confs_seg, turns)

        for widx, (w, (t0, t1)) in enumerate(zip(orig_words, spans_sec)):
            if widx >= len(speakers_seg):
                break
            all_words.append(
                WordInfo(
                    segment_index=seg_idx,
                    word_index=widx,
                    word=w,
                    t0=t0,
                    t1=t1,
                    speaker=speakers_seg[widx],
                    confidence=confs_seg[widx],
                )
            )

    t0 = time.perf_counter()
    windows: List[Window] = build_windows_from_exclusive_regions(
        exclusive_regions,
        all_words,
        min_words=cfg.min_words_per_window,
        merge_gap_sec=cfg.merge_gap_sec,
    )
    best_windows = pick_longest_window_per_speaker(windows)
    _add_time("build_windows_total", time.perf_counter() - t0)
    lang_by_seg = {i: s.language for i, s in enumerate(segments)}

    _inc_meta("speaker_attribute_reference_requests", len(best_windows))
    t0 = time.perf_counter()
    speaker_ref = build_speaker_reference_embeddings(
        session_waveform,
        best_windows,
        lang_by_seg,
        speaker_attribute_base_url=cfg.speaker_attribute_base_url,
    )
    _add_time("speaker_attribute_reference_total", time.perf_counter() - t0)

    per_word_embedding: Dict[Tuple[int, int], List[float]] = {}

    if cfg.segment_level_embeddings:
        for seg_idx, words in enumerate(per_segment_words):
            if not words:
                continue
            seg_audio = slice_waveform(
                session_waveform,
                assembled.segment_offsets_sec[seg_idx],
                assembled.segment_offsets_sec[seg_idx] + assembled.segment_lengths_sec[seg_idx],
            )
            lang = lang_by_seg.get(seg_idx, "en")
            t0 = time.perf_counter()
            out = infer_word_speaker_embeddings_http(
                audio_bytes=waveform_to_wav_bytes_pcm16(seg_audio),
                transcript=" ".join(words),
                language=lang,
                base_url=cfg.speaker_attribute_base_url,
            )
            _add_time("speaker_attribute_segment_embeddings_total", time.perf_counter() - t0)
            _inc_meta("speaker_attribute_segment_embeddings_requests", 1)
            n = min(len(out.embeddings), len(words))
            for widx in range(n):
                per_word_embedding[(seg_idx, widx)] = out.embeddings[widx]

    for w in windows:
        audio = slice_waveform(session_waveform, w.start, w.end)
        words = [x[2] for x in w.words]
        lang = lang_by_seg.get(w.words[0][0], "en")
        t0 = time.perf_counter()
        out = infer_word_speaker_embeddings_http(
            audio_bytes=waveform_to_wav_bytes_pcm16(audio),
            transcript=" ".join(words),
            language=lang,
            base_url=cfg.speaker_attribute_base_url,
        )
        _add_time("speaker_attribute_window_embeddings_total", time.perf_counter() - t0)
        _inc_meta("speaker_attribute_window_embeddings_requests", 1)
        n = min(len(out.embeddings), len(w.words))
        for j in range(n):
            seg_idx, word_idx, _ = w.words[j]
            per_word_embedding[(seg_idx, word_idx)] = out.embeddings[j]

    # -- collect all pre-repair embeddings & speakers for session-wide pool --
    all_embeds_flat: List[Optional[List[float]]] = []
    all_speakers_flat: List[Optional[str]] = []
    per_seg_embeds: List[List[Optional[List[float]]]] = []
    per_seg_speakers: List[List[Optional[str]]] = []
    per_seg_confs: List[List[str]] = []

    for seg_idx, words in enumerate(per_segment_words):
        embeds: List[Optional[List[float]]] = []
        speakers: List[Optional[str]] = []
        confs: List[str] = []

        for widx in range(len(words)):
            wi = next((x for x in all_words if x.segment_index == seg_idx and x.word_index == widx), None)
            spk = wi.speaker if wi else None
            conf = wi.confidence if wi else "low"
            emb = per_word_embedding.get((seg_idx, widx))
            embeds.append(emb)
            speakers.append(spk)
            confs.append(conf)

        per_seg_embeds.append(embeds)
        per_seg_speakers.append(speakers)
        per_seg_confs.append(confs)
        all_embeds_flat.extend(embeds)
        all_speakers_flat.extend(speakers)

    # -- compute session-wide stable pools once --
    if cfg.repair_policy == "central_pool":
        session_pools = compute_stable_speaker_pools(
            all_embeds_flat,
            all_speakers_flat,
            similarity_threshold=cfg.central_similarity_threshold,
            center_top_k=cfg.central_center_top_k,
            max_pool_per_speaker=cfg.central_max_pool_per_speaker,
        )
    else:
        session_pools = {spk: [ref] for spk, ref in speaker_ref.items()}

    # -- per-segment repair --
    output: List[List[object]] = []
    for seg_idx, words in enumerate(per_segment_words):
        t0 = time.perf_counter()
        if cfg.repair_policy == "central_pool":
            repair_result: RepairResult = repair_embeddings_central_pool(
                words,
                per_seg_embeds[seg_idx],
                per_seg_speakers[seg_idx],
                per_seg_confs[seg_idx],
                similarity_threshold=cfg.central_similarity_threshold,
                center_top_k=cfg.central_center_top_k,
                max_pool_per_speaker=cfg.central_max_pool_per_speaker,
                replace_mode=cfg.central_replace_mode,
                random_seed=cfg.central_random_seed,
            )
        else:
            repair_result = repair_embeddings(
                words, per_seg_embeds[seg_idx], per_seg_speakers[seg_idx], per_seg_confs[seg_idx], speaker_ref,
            )
        _add_time("repair_total", time.perf_counter() - t0)
        _inc_meta("repair_segments", 1)
        # Keep speaker + confidence aligned with `words` so `server._format_response`
        # can zip `words/embeddings/speakers/confidences` safely.
        output.append([words, repair_result.embeddings, per_seg_speakers[seg_idx], per_seg_confs[seg_idx]])

    return PipelineOutput(segments=output, speaker_pools=session_pools)


def compute_word_speaker_embeddings(
    segments: Sequence[SegmentInput],
    cfg: PipelineConfig = PipelineConfig(),
    timing: Optional[Dict[str, float]] = None,
    timing_meta: Optional[Dict[str, Any]] = None,
) -> PipelineOutput:
    # `_compute_base` already runs diarization + alignment once and produces the
    # final response shape expected by `server._format_response`.
    return _compute_base(segments, cfg=cfg, timing=timing, timing_meta=timing_meta)

