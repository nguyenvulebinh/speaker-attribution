"""
Test the diarization service with session_1 audio files.

Start the service first:
    python -m diarization.server

Then run this test:
    python -m ensemble_diarization.diarization.test_service
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List


def _send_diarize_request(
    wav_bytes: bytes,
    url: str = "http://0.0.0.0:5000/diarize",
    min_speakers: int | None = None,
    max_speakers: int | None = None,
) -> List[dict]:
    qs = []
    if min_speakers is not None:
        qs.append(f"min_speakers={min_speakers}")
    if max_speakers is not None:
        qs.append(f"max_speakers={max_speakers}")
    full_url = url + ("?" + "&".join(qs) if qs else "")

    boundary = "----test-boundary"
    parts: List[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="audio_wav"; filename="audio.wav"\r\n')
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(wav_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url=full_url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600000.0) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload


def main() -> None:
    session_dir = Path("examples/session_1")
    wav_files = sorted(session_dir.glob("*.wav"))

    if not wav_files:
        print(f"No .wav files found in {session_dir}")
        return

    print(f"Found {len(wav_files)} audio files in {session_dir}\n")

    # ---------- Sequential ----------
    print("--- Sequential ---")
    t0 = time.perf_counter()
    for wav_path in wav_files:
        wav_bytes = wav_path.read_bytes()
        segments = _send_diarize_request(wav_bytes)
        speakers = set(s["speaker"] for s in segments)
        total_dur = sum(s["end"] - s["start"] for s in segments)
        print(
            f"  {wav_path.name}: {len(segments)} segments, "
            f"{len(speakers)} speakers, {total_dur:.1f}s speech"
        )
        if segments:
            first = segments[0]
            print(f"    first: start={first['start']:.3f}  end={first['end']:.3f}  speaker={first['speaker']}")
    seq_time = time.perf_counter() - t0
    print(f"Sequential total: {seq_time:.2f}s\n")

    # ---------- Concurrent ----------
    print("--- Concurrent (all at once) ---")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(wav_files)) as pool:
        futures = {
            pool.submit(_send_diarize_request, wav_path.read_bytes()): wav_path.name
            for wav_path in wav_files
        }
        for fut in as_completed(futures):
            name = futures[fut]
            segments = fut.result()
            speakers = set(s["speaker"] for s in segments)
            print(f"  {name}: {len(segments)} segments, {len(speakers)} speakers")
    conc_time = time.perf_counter() - t0
    print(f"Concurrent total: {conc_time:.2f}s")

    if conc_time > 0:
        print(f"\nSpeedup: {seq_time / conc_time:.1f}x")


if __name__ == "__main__":
    main()
