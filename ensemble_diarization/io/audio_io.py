from __future__ import annotations

import io
import wave

import numpy as np
import torch
import torchaudio

SAMPLE_RATE = 16000


def decode_audio_bytes_to_waveform_16k_mono(audio_bytes: bytes) -> torch.Tensor:
    """
    Decode arbitrary audio bytes to mono waveform at 16kHz.
    Returns waveform shaped (1, num_samples) float32 in [-1, 1].
    """
    # Prefer stdlib WAV decode first (avoids torchcodec decode issues for some files).
    try:
        with wave.open(io.BytesIO(audio_bytes), "rb") as wf:
            n_channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            sr = wf.getframerate()
            n_frames = wf.getnframes()
            pcm = wf.readframes(n_frames)

        if n_frames == 0:
            raise ValueError("WAV contains 0 frames.")

        if sampwidth == 2:
            data = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(pcm, dtype=np.int32).astype(np.float32) / 2147483648.0
        else:
            raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

        if n_channels > 1:
            data = data.reshape(-1, n_channels).mean(axis=1)

        waveform = torch.from_numpy(data).unsqueeze(0)  # (1, N)
    except wave.Error:
        # Not a WAV (or malformed) — fall back to torchaudio decoder.
        waveform, sr = torchaudio.load(io.BytesIO(audio_bytes), channels_first=True, normalize=True)
        if waveform.dim() != 2:
            raise ValueError(f"Unexpected waveform dims: {tuple(waveform.shape)}")
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=SAMPLE_RATE)
    return waveform.to(dtype=torch.float32)


def waveform_to_wav_bytes_pcm16(waveform_16k_mono: torch.Tensor) -> bytes:
    """
    Convert waveform (1, N) float32 [-1,1] to WAV PCM16 bytes.
    This avoids dependencies like soundfile and matches what the diarization service can decode.
    """
    if waveform_16k_mono.dim() != 2 or waveform_16k_mono.shape[0] != 1:
        raise ValueError(f"Expected waveform shape (1, N). Got {tuple(waveform_16k_mono.shape)}")
    x = waveform_16k_mono.squeeze(0).detach().cpu().numpy()
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767.0).astype(np.int16).tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm)
    return buf.getvalue()


def slice_waveform(waveform_16k_mono: torch.Tensor, start_sec: float, end_sec: float) -> torch.Tensor:
    start = int(max(0.0, start_sec) * SAMPLE_RATE)
    end = int(max(0.0, end_sec) * SAMPLE_RATE)
    end = max(end, start)
    return waveform_16k_mono[:, start:end]

