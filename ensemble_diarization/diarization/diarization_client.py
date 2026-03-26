from __future__ import annotations

import json
import time
import urllib.request
from typing import List, Optional

from ..core.types import Turn


def diarize(
    audio_bytes: bytes,
    service_url: str = "http://0.0.0.0:5000/diarize",
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
    timeout_s: float = 600000.0,
) -> List[Turn]:
    """
    Generic diarization HTTP client.

    Sends the audio as a multipart file upload (field ``audio_wav``) to a
    diarization service endpoint that returns a JSON list of turns:
    [{start, end, speaker}, ...].
    """
    qs = []
    if num_speakers is not None:
        qs.append(f"num_speakers={num_speakers}")
    if min_speakers is not None:
        qs.append(f"min_speakers={min_speakers}")
    if max_speakers is not None:
        qs.append(f"max_speakers={max_speakers}")
    url = service_url + ("?" + "&".join(qs) if qs else "")

    boundary = "----diarize-boundary"
    parts: List[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="audio_wav"; filename="audio.wav"\r\n')
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(audio_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        payload = resp.read()
    _ = time.perf_counter() - t0
    return json.loads(payload.decode("utf-8"))
