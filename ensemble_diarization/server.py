"""
Ensemble diarization service — orchestrates diarization, alignment, and
speaker-attribute services behind a single API endpoint.

Run:
  python -m ensemble_diarization.server

Infer:
  curl -fsS -X POST "http://localhost:5025/ensemble_diarize" \
    -F "audio_wav=@session.wav" \
    -F "segments_json=@segments.json"

Env vars:
  - HOST, PORT: bind address (default 0.0.0.0:5025)
  - DIARIZATION_URL: diarization service URL
  - ALIGNMENT_URL: alignment service URL
  - SPEAKER_ATTRIBUTE_BASE_URL: speaker attribute service base URL
"""

from __future__ import annotations

import json
import os
import queue
import threading
import traceback
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from loguru import logger

from .core.pipeline import PipelineConfig, PipelineOutput, compute_word_speaker_embeddings
from .core.types import SegmentInput
from .io.audio_io import (
    SAMPLE_RATE,
    decode_audio_bytes_to_waveform_16k_mono,
    slice_waveform,
    waveform_to_wav_bytes_pcm16,
)

DEFAULT_DIARIZATION_URL = os.environ.get("DIARIZATION_URL", "http://0.0.0.0:5022/diarize")
DEFAULT_ALIGNMENT_URL = os.environ.get("ALIGNMENT_URL", "http://0.0.0.0:5023/align_words")
DEFAULT_SPEAKER_ATTRIBUTE_BASE_URL = os.environ.get("SPEAKER_ATTRIBUTE_BASE_URL", "http://0.0.0.0:5024")


# ---------------------------------------------------------------------------
# Request queue (serialise pipeline calls)
# ---------------------------------------------------------------------------

@dataclass
class _Request:
    segments: List[SegmentInput]
    cfg: PipelineConfig
    segments_meta: List[Dict[str, Any]]
    condition: threading.Condition
    timing: Dict[str, float]
    timing_meta: Dict[str, Any]
    result: Optional[Dict[str, Any]] = field(default=None)
    error: Optional[Exception] = field(default=None)


_queue: queue.Queue[_Request] = queue.Queue()


def _format_response(
    pipeline_output: PipelineOutput,
    segments_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    segments_out: List[Dict[str, Any]] = []

    for seg_idx, item in enumerate(pipeline_output.segments):
        words, embeddings, speakers, confidences = item
        meta = segments_meta[seg_idx] if seg_idx < len(segments_meta) else {}

        word_entries: List[Dict[str, Any]] = []
        for w, emb, spk, conf in zip(words, embeddings, speakers, confidences):
            word_entries.append({
                "word": w,
                "speaker": spk,
                "confidence": conf,
                "embedding": emb,
            })

        segments_out.append({
            "start": meta.get("start"),
            "duration": meta.get("duration"),
            "text": meta.get("text"),
            "language": meta.get("language", "en"),
            "words": word_entries,
        })

    speakers_out: Dict[str, Any] = {
        spk: {"embeddings": embs}
        for spk, embs in pipeline_output.speaker_pools.items()
    }

    return {"segments": segments_out, "speakers": speakers_out}


def _format_audio_duration_hm(audio_duration_sec: float) -> str:
    total_seconds = max(0, int(audio_duration_sec))
    h, rem = divmod(total_seconds, 3600)
    m, _s = divmod(rem, 60)
    return f"{h}h{m:02d}m"


def _log_timing_breakdown(
    *,
    timing: Dict[str, float],
    timing_meta: Dict[str, Any],
    total_wall_s: float,
    audio_duration_sec: float,
) -> None:
    duration_hm = _format_audio_duration_hm(audio_duration_sec)
    seg_count = int(timing_meta.get("segments_count", 0))
    diarize_calls = int(timing_meta.get("diarize_requests", 0))
    align_calls = int(timing_meta.get("align_words_requests", 0))

    logger.info(
        f"[ensemble][timing] duration={duration_hm} segments={seg_count} "
        f"total_wall={total_wall_s:.2f}s diarize_requests={diarize_calls} align_words_requests={align_calls}"
    )

    if total_wall_s <= 0:
        return

    for k, v in sorted(timing.items(), key=lambda kv: kv[1], reverse=True):
        pct = (v / total_wall_s) * 100.0
        logger.info(f"[ensemble][timing] {k} {v:.2f}s ({pct:.1f}%)")

    # Summary of request counts per service.
    sa_seg = int(timing_meta.get("speaker_attribute_segment_embeddings_requests", 0))
    sa_win = int(timing_meta.get("speaker_attribute_window_embeddings_requests", 0))
    sa_ref = int(timing_meta.get("speaker_attribute_reference_requests", 0))
    repair_calls = int(timing_meta.get("repair_segments", 0))
    logger.info(
        f"[ensemble][timing][calls] speaker_attribute segments={sa_seg} windows={sa_win} reference={sa_ref} "
        f"repair_segments={repair_calls}"
    )


def _worker() -> None:
    while True:
        req = _queue.get()
        logger.info(f"[ensemble] processing request  backlog={_queue.qsize()}")
        try:
            pipeline_output = compute_word_speaker_embeddings(
                req.segments,
                cfg=req.cfg,
                timing=req.timing,
                timing_meta=req.timing_meta,
            )
            req.result = _format_response(pipeline_output, req.segments_meta)
        except Exception as e:
            traceback.print_exc()
            req.error = e
        finally:
            with req.condition:
                req.condition.notify()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[ensemble] starting worker thread")
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    yield


app = FastAPI(title="ensemble_diarization", version="0.1.0", lifespan=lifespan)


@app.get("/live")
def live() -> Dict[str, str]:
    return {"status": "live"}


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    return {"ok": "true"}


@app.post("/ensemble_diarize")
def ensemble_diarize(
    audio_wav: UploadFile = File(...),
    segments_json: UploadFile = File(...),
    diarization_url: Optional[str] = Query(None),
    speaker_attribute_base_url: Optional[str] = Query(None),
    alignment_url: Optional[str] = Query(None),
    repair_policy: Optional[str] = Query(None),
    central_similarity_threshold: Optional[float] = Query(None),
    central_replace_mode: Optional[str] = Query(None),
) -> Dict[str, Any]:
    # --- parse inputs ---
    req_wall_start = time.perf_counter()
    timing: Dict[str, float] = {}
    timing_meta: Dict[str, Any] = {}

    try:
        wav_bytes = audio_wav.file.read()
        t0 = time.perf_counter()
        waveform = decode_audio_bytes_to_waveform_16k_mono(wav_bytes)
        timing["decode_full_audio"] = time.perf_counter() - t0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to decode audio: {e}") from e

    try:
        raw = segments_json.file.read()
        t0 = time.perf_counter()
        segments_meta: List[Dict[str, Any]] = json.loads(raw)
        timing["parse_segments_json"] = time.perf_counter() - t0
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid segments_json: {e}") from e

    if not isinstance(segments_meta, list) or not segments_meta:
        raise HTTPException(status_code=400, detail="segments_json must be a non-empty JSON array")

    timing_meta["segments_count"] = len(segments_meta)

    # --- slice audio per segment ---
    t0_slice = time.perf_counter()
    segment_inputs: List[SegmentInput] = []
    for seg in segments_meta:
        start = float(seg.get("start", 0))
        duration = float(seg.get("duration", 0))
        text = str(seg.get("text", ""))
        language = str(seg.get("language", "en"))

        seg_waveform = slice_waveform(waveform, start, start + duration)
        seg_wav_bytes = waveform_to_wav_bytes_pcm16(seg_waveform)
        segment_inputs.append(SegmentInput(audio_bytes=seg_wav_bytes, transcript=text, language=language))
    timing["slice_and_encode_segments"] = time.perf_counter() - t0_slice

    audio_duration_sec = float(waveform.shape[1]) / float(SAMPLE_RATE)

    # --- build pipeline config ---
    cfg_kwargs: Dict[str, Any] = {}
    if diarization_url is not None:
        cfg_kwargs["diarization_url"] = diarization_url
    else:
        cfg_kwargs["diarization_url"] = DEFAULT_DIARIZATION_URL
    if alignment_url is not None:
        cfg_kwargs["alignment_url"] = alignment_url
    else:
        cfg_kwargs["alignment_url"] = DEFAULT_ALIGNMENT_URL
    if speaker_attribute_base_url is not None:
        cfg_kwargs["speaker_attribute_base_url"] = speaker_attribute_base_url
    else:
        cfg_kwargs["speaker_attribute_base_url"] = DEFAULT_SPEAKER_ATTRIBUTE_BASE_URL
    if repair_policy is not None:
        cfg_kwargs["repair_policy"] = repair_policy
    if central_similarity_threshold is not None:
        cfg_kwargs["central_similarity_threshold"] = central_similarity_threshold
    if central_replace_mode is not None:
        cfg_kwargs["central_replace_mode"] = central_replace_mode

    cfg = PipelineConfig(**cfg_kwargs)

    # --- enqueue and wait ---
    condition = threading.Condition()
    req = _Request(
        segments=segment_inputs,
        cfg=cfg,
        segments_meta=segments_meta,
        condition=condition,
        timing=timing,
        timing_meta=timing_meta,
    )

    with condition:
        _queue.put(req)
        t0_wait = time.perf_counter()
        condition.wait()
        timing["queue_wait"] = time.perf_counter() - t0_wait

    if req.error is not None:
        raise HTTPException(status_code=500, detail=str(req.error))

    total_wall_s = time.perf_counter() - req_wall_start
    _log_timing_breakdown(
        timing=timing,
        timing_meta=timing_meta,
        total_wall_s=float(total_wall_s),
        audio_duration_sec=audio_duration_sec,
    )

    return req.result


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5025"))
    uvicorn.run("ensemble_diarization.server:app", host=host, port=port)
