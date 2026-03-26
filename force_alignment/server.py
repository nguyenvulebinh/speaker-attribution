from __future__ import annotations

import queue
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import List, Optional, TypedDict

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from .alignment import WordSpan, align_transcript_batch, warmup

MAX_BATCH_SIZE = 32
BATCH_TIMEOUT_S = 0.05  # wait up to 50 ms to accumulate more items


class SpanOut(TypedDict):
    word: str
    t0: float
    t1: float


class AlignResponse(TypedDict):
    spans: List[SpanOut]


@dataclass
class _Request:
    wav_bytes: bytes
    transcript: str
    condition: threading.Condition
    result: Optional[List[WordSpan]] = field(default=None)
    error: Optional[Exception] = field(default=None)


_queue: queue.Queue = queue.Queue()


def _batch_worker():
    """Background thread: collect requests from the queue and process in batches."""
    while True:
        req = _queue.get()
        batch: List[_Request] = [req]

        while len(batch) < MAX_BATCH_SIZE:
            try:
                req = _queue.get(timeout=BATCH_TIMEOUT_S)
                batch.append(req)
            except queue.Empty:
                break

        print(f"[force_alignment] batch={len(batch)}  backlog={_queue.qsize()}")

        try:
            wav_list = [r.wav_bytes for r in batch]
            transcript_list = [r.transcript for r in batch]
            all_spans = align_transcript_batch(wav_list, transcript_list)
            for r, spans in zip(batch, all_spans):
                r.result = spans
                with r.condition:
                    r.condition.notify()
        except Exception as e:
            traceback.print_exc()
            for r in batch:
                r.error = e
                with r.condition:
                    r.condition.notify()


@asynccontextmanager
async def lifespan(app: FastAPI):
    print("[force_alignment] warming up model + uroman ...")
    warmup()
    print("[force_alignment] model ready — starting batch worker")
    worker = threading.Thread(target=_batch_worker, daemon=True)
    worker.start()
    yield


app = FastAPI(title="force_alignment", version="0.1.0", lifespan=lifespan)


@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.post("/align_words", response_model=None)
def align_words(
    audio_wav: UploadFile = File(...),
    transcript: str = Form(...),
) -> AlignResponse:
    """
    Align a transcript to audio.

    - **audio_wav**: WAV file (any sample rate / channels; converted internally)
    - **transcript**: raw transcript text (splitting + normalization done server-side)

    Returns one span per word in the transcript.
    """
    wav_bytes = audio_wav.file.read()

    condition = threading.Condition()
    req = _Request(wav_bytes=wav_bytes, transcript=transcript, condition=condition)

    with condition:
        _queue.put(req)
        condition.wait()

    if req.error is not None:
        raise HTTPException(status_code=500, detail=str(req.error))

    return {"spans": [{"word": s.word, "t0": float(s.t0), "t1": float(s.t1)} for s in req.result]}
