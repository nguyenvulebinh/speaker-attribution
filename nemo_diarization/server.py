from __future__ import annotations

import io
import os
import queue
import threading
import traceback
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from loguru import logger

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

from .diarization import DiarizationService, Segment

SAMPLE_RATE = 16000

_PACKAGE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_CONFIG_PATH = _PACKAGE_DIR / "diar_infer_general.yaml"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "7")


def _pick_device() -> str:
    device_env = os.environ.get("DEVICE")
    if device_env:
        return device_env
    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_audio_bytes(audio_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode audio bytes into mono float32 waveform (num_samples,)."""
    if sf is not None:
        try:
            data, sr = sf.read(io.BytesIO(audio_bytes), always_2d=False)
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes with soundfile: {e}") from e
    elif torchaudio is not None:
        try:
            waveform, sr = torchaudio.load(
                io.BytesIO(audio_bytes), channels_first=True, normalize=True
            )
            data = (
                waveform.mean(dim=0).cpu().numpy()
                if waveform.shape[0] > 1
                else waveform.squeeze(0).cpu().numpy()
            )
        except Exception as e:
            raise ValueError(f"Failed to decode audio bytes with torchaudio: {e}") from e
    else:
        raise ValueError("No audio decoder available. Install 'soundfile' or 'torchaudio'.")

    if data is None:
        raise ValueError("Decoded audio is empty.")

    if isinstance(data, np.ndarray) and data.ndim > 1:
        data = np.mean(data, axis=-1)

    return np.asarray(data, dtype=np.float32), int(sr)


def resample_if_needed(
    waveform: np.ndarray, sr: int, target_sr: int = SAMPLE_RATE
) -> tuple[np.ndarray, int]:
    if sr == target_sr:
        return waveform, sr

    if torchaudio is not None:
        try:
            wave_t = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0)
            wave_rs = torchaudio.functional.resample(
                wave_t, orig_freq=sr, new_freq=target_sr
            )
            return wave_rs.squeeze(0).cpu().numpy().astype(np.float32), target_sr
        except Exception as e:
            raise ValueError(f"Resampling failed with torchaudio: {e}") from e

    if librosa is not None:
        try:
            wave_rs = librosa.resample(
                np.asarray(waveform, dtype=np.float32), orig_sr=sr, target_sr=target_sr
            )
            return np.asarray(wave_rs, dtype=np.float32), target_sr
        except Exception as e:
            raise ValueError(f"Resampling failed with librosa: {e}") from e

    raise ValueError(
        f"Input sample rate is {sr}, but no resampler is available. "
        "Install 'torchaudio' or 'librosa', or send 16kHz mono audio."
    )


def _write_wav_temp(mono_waveform: np.ndarray, sr: int, suffix: str = ".wav") -> str:
    """Write mono waveform to a temp WAV file and return its path."""
    if mono_waveform.ndim != 1:
        mono_waveform = np.mean(mono_waveform, axis=-1)
    mono_waveform = np.asarray(mono_waveform, dtype=np.float32)

    # Keep file on disk for NeMo.
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp_path = tmp.name
    tmp.close()

    try:
        if sf is not None:
            sf.write(tmp_path, mono_waveform, sr)
            return tmp_path
        if torchaudio is not None:
            wav_t = torch.from_numpy(mono_waveform).unsqueeze(0)  # [1, T]
            torchaudio.save(tmp_path, wav_t, sample_rate=sr)
            return tmp_path
    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        raise ValueError(f"Failed to write temp wav: {e}") from e

    raise ValueError("No audio writer available. Install 'soundfile' or 'torchaudio'.")


@dataclass
class _Request:
    waveform: torch.Tensor  # [1, T] float32
    num_speakers: Optional[int]
    min_speakers: Optional[int]
    max_speakers: Optional[int]
    condition: threading.Condition
    result: Optional[List[Dict[str, Any]]] = field(default=None)
    error: Optional[Exception] = field(default=None)


_queue: queue.Queue[_Request] = queue.Queue()
_service: Optional[DiarizationService] = None


def _normalize_speaker_param(v: Optional[int]) -> Optional[int]:
    # The diarization service treats `0` as "unset" for constraints.
    if v is None:
        return None
    if v == 0:
        return None
    return int(v)


def _worker() -> None:
    global _service
    assert _service is not None

    while True:
        req = _queue.get()
        logger.info(
            f"[nemo_diarization] processing request backlog={_queue.qsize()}"
        )
        try:
            mono = req.waveform.squeeze(0).cpu().numpy().astype(np.float32)
            wav_path = _write_wav_temp(mono, SAMPLE_RATE)
            try:
                with torch.inference_mode():
                    segments: List[Segment] = _service.diarize(
                        wav_path,
                        num_speakers=req.num_speakers,
                        min_speakers=req.min_speakers,
                        max_speakers=req.max_speakers,
                        cleanup_rttm=True,
                    )
            finally:
                try:
                    os.unlink(wav_path)
                except Exception:
                    pass

            req.result = [
                {
                    "start": round(float(seg.start), 3),
                    "end": round(float(seg.end), 3),
                    "speaker": seg.speaker,
                }
                for seg in segments
            ]
        except Exception as e:
            traceback.print_exc()
            req.error = e
        finally:
            with req.condition:
                req.condition.notify()


def _load_service() -> DiarizationService:
    device = _pick_device()

    model_config_path = os.environ.get("MODEL_CONFIG_PATH", str(DEFAULT_MODEL_CONFIG_PATH))
    output_dir = os.environ.get("OUTPUT_DIR", "/tmp/nemo_diarizer_output")
    oracle_vad = os.environ.get("ORACLE_VAD", "false").strip().lower() in {"1", "true", "yes", "y"}

    # Speaker counts will be supplied per request via API params.
    return DiarizationService(
        model_config_path=model_config_path,
        output_dir=output_dir,
        num_speakers=None,
        max_speakers=None,
        oracle_vad=oracle_vad,
        device=device,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _service
    logger.info("[nemo_diarization] loading NeMo diarization service ...")
    _service = _load_service()
    logger.info("[nemo_diarization] starting worker queue thread ...")
    worker = threading.Thread(target=_worker, daemon=True)
    worker.start()
    yield


app = FastAPI(title="nemo_diarization", version="0.1.0", lifespan=lifespan)


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
            waveform = waveform.unsqueeze(0)  # [1, T]
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    req = _Request(
        waveform=waveform,
        num_speakers=_normalize_speaker_param(num_speakers),
        min_speakers=_normalize_speaker_param(min_speakers),
        max_speakers=_normalize_speaker_param(max_speakers),
        condition=threading.Condition(),
    )

    with req.condition:
        _queue.put(req)
        req.condition.wait()

    if req.error is not None:
        raise HTTPException(status_code=500, detail=str(req.error))

    assert req.result is not None
    return req.result


if __name__ == "__main__":
    import uvicorn

    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "5022"))
    uvicorn.run("nemo_diarization.server:app", host=host, port=port)

