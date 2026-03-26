"""
Diarization service with a request queue to serialize GPU access.

Follows the same queue pattern as force_alignment/server.py:
a background worker thread pulls one request at a time from a queue,
ensuring only one inference runs on the GPU concurrently.

Run:
  python -m diarization.server

Infer:
  curl -fsS -X POST "http://localhost:5022/diarize?min_speakers=0&max_speakers=20" \
    -F "audio_wav=@your_audio.wav"

Env vars:
  - CUDA_VISIBLE_DEVICES: GPU index (default: "7")
  - PIPELINE_PATH: path to local pipeline directory
  - DEVICE: "cuda" | "cpu" (default: cuda if available)
  - EMBEDDING_BATCH_SIZE: integer (default: 32)
  - SEGMENTATION_BATCH_SIZE: integer (default: 32)
  - HOST, PORT: bind address (default 0.0.0.0:5022)
"""

from __future__ import annotations

import io
import os
import queue
import threading
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from loguru import logger
from pyannote.audio import Pipeline
from pyannote import audio
print(audio.__version__)

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")

try:
    import soundfile as sf  # type: ignore
except Exception:  # pragma: no cover
    sf = None

try:
    import torchaudio  # type: ignore
except Exception:  # pragma: no cover
    torchaudio = None

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None


SAMPLE_RATE = 16000
DEFAULT_PIPELINE_PATH = os.environ.get(
    "PIPELINE_PATH",
    "nguyenvulebinh/speaker-diarization-community",
)
EMBEDDING_BATCH_SIZE = int(os.environ.get("EMBEDDING_BATCH_SIZE", "16"))
SEGMENTATION_BATCH_SIZE = int(os.environ.get("SEGMENTATION_BATCH_SIZE", "16"))


# ---------------------------------------------------------------------------
# Audio helpers (identical to simple_service.py)
# ---------------------------------------------------------------------------

def _pick_device() -> torch.device:
    device_env = os.environ.get("DEVICE")
    if device_env:
        return torch.device(device_env)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes into mono float32 waveform (num_samples,)."""
    if sf is not None:
        try:
            data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes with soundfile: {e}") from e
    elif torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), channels_first=True, normalize=True)
            data = waveform.mean(dim=0).cpu().numpy() if waveform.shape[0] > 1 else waveform.squeeze(0).cpu().numpy()
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes with torchaudio: {e}") from e
    else:
        raise ValueError("No audio decoder available. Install 'soundfile' or 'torchaudio'.")

    if data is None:
        raise ValueError("Decoded audio is empty.")

    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=-1)

    return np.asarray(data, dtype=np.float32), int(sr)


def resample_if_needed(waveform: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return waveform, sr
    if torchaudio is not None:
        try:
            wave_t = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            wave_rs = torchaudio.functional.resample(wave_t, orig_freq=sr, new_freq=target_sr)
            return wave_rs.squeeze(0).cpu().numpy().astype(np.float32), target_sr
        except Exception as e:
            raise ValueError(f"Resampling failed with torchaudio: {e}") from e
    if librosa is not None:
        try:
            wave_rs = librosa.resample(np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=target_sr)
            return np.asarray(wave_rs, dtype=np.float32), target_sr
        except Exception as e:
            raise ValueError(f"Resampling failed with librosa: {e}") from e
    raise ValueError(
        f"Input sample rate is {sr}, but no resampler is available. "
        "Install 'torchaudio' or 'librosa', or send 16kHz mono audio."
    )


# ---------------------------------------------------------------------------
# Pipeline loader
# ---------------------------------------------------------------------------

def load_pipeline() -> Pipeline:
    device = _pick_device()
    logger.info(f"Loading diarization pipeline from {DEFAULT_PIPELINE_PATH!r}")
    pipeline = Pipeline.from_pretrained(DEFAULT_PIPELINE_PATH)

    if hasattr(pipeline, "embedding_batch_size"):
        pipeline.embedding_batch_size = EMBEDDING_BATCH_SIZE
        logger.info(f"pipeline.embedding_batch_size={EMBEDDING_BATCH_SIZE}")
    if hasattr(pipeline, "segmentation_batch_size"):
        pipeline.segmentation_batch_size = SEGMENTATION_BATCH_SIZE
        logger.info(f"pipeline.segmentation_batch_size={SEGMENTATION_BATCH_SIZE}")

    pipeline.to(device)
    logger.info(f"Pipeline ready on device={device}")
    return pipeline


# ---------------------------------------------------------------------------
# Request queue (single-sample, serialized GPU access)
# ---------------------------------------------------------------------------

@dataclass
class _Request:
    waveform: torch.Tensor
    sample_rate: int
    kwargs: Dict[str, Any]
    condition: threading.Condition
    result: Optional[List[Dict[str, Any]]] = field(default=None)
    error: Optional[Exception] = field(default=None)


_queue: queue.Queue[_Request] = queue.Queue()
_pipeline: Optional[Pipeline] = None


def _worker():
    """Background thread: pull one request at a time and run inference."""
    global _pipeline
    assert _pipeline is not None
    while True:
        req = _queue.get()
        logger.info(f"[diarization] processing request  backlog={_queue.qsize()}")
        try:
            with torch.inference_mode():
                output = _pipeline(
                    {"waveform": req.waveform, "sample_rate": req.sample_rate},
                    **req.kwargs,
                )
            segments: List[Dict[str, Any]] = []
            for turn, speaker in output.speaker_diarization:
                segments.append({
                    "start": round(float(turn.start), 3),
                    "end": round(float(turn.end), 3),
                    "speaker": speaker,
                })
            req.result = segments
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
    global _pipeline
    logger.info("[diarization] loading pipeline ...")
    _pipeline = load_pipeline()
    logger.info("[diarization] pipeline ready — starting worker")
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    yield


app = FastAPI(title="diarization", version="0.1.0", lifespan=lifespan)


@app.get("/live")
def live() -> Dict[str, str]:
    return {"status": "live"}


@app.post("/diarize")
def diarize(
    audio_wav: UploadFile = File(...),
    num_speakers: Optional[int] = Query(None, ge=0),
    min_speakers: Optional[int] = Query(None, ge=0),
    max_speakers: Optional[int] = Query(None, ge=0),
) -> List[Dict[str, Any]]:
    try:
        wav_bytes = audio_wav.file.read()
        waveform_np, sr = decode_audio_bytes(wav_bytes)
        waveform_np, sr = resample_if_needed(waveform_np, sr, target_sr=SAMPLE_RATE)
        waveform = torch.from_numpy(waveform_np).float()
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    kwargs: Dict[str, Any] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if min_speakers is not None:
        kwargs["min_speakers"] = min_speakers
    if max_speakers is not None:
        kwargs["max_speakers"] = max_speakers

    condition = threading.Condition()
    req = _Request(
        waveform=waveform,
        sample_rate=sr,
        kwargs=kwargs,
        condition=condition,
    )

    with condition:
        _queue.put(req)
        condition.wait()

    if req.error is not None:
        raise HTTPException(status_code=500, detail=str(req.error))

    return req.result


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5022"))
    uvicorn.run("diarization.server:app", host=host, port=port)
