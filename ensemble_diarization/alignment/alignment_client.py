from __future__ import annotations

import json
import time
import urllib.request
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class WordSpan:
    t0: float
    t1: float


def align_words_http(
    *,
    audio_wav_bytes: bytes,
    transcript: str,
    service_url: str = "http://0.0.0.0:5010/align_words",
    timeout_s: float = 600000.0,
) -> List[WordSpan]:
    """
    Call the forced-alignment service.

    Service contract:
    - POST multipart form with fields:
        - audio_wav (file): WAV bytes
        - transcript (text): raw transcript (splitting + normalization done server-side)
    - response: {"spans": [{"word":..,"t0":..,"t1":..}, ...]}  (one per word)
    """
    boundary = "----ensemble-align-boundary"

    parts: List[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="transcript"\r\n')
    parts.append(b"Content-Type: text/plain; charset=utf-8\r\n\r\n")
    parts.append(transcript.encode("utf-8"))
    parts.append(b"\r\n")

    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="audio_wav"; filename="audio.wav"\r\n')
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(audio_wav_bytes)
    parts.append(b"\r\n")

    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url=service_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = resp.read()
    _ = time.perf_counter() - t0

    payload = json.loads(raw.decode("utf-8"))
    spans = payload.get("spans", [])
    return [WordSpan(t0=float(s["t0"]), t1=float(s["t1"])) for s in spans]
