"""
Test the alignment service with session_1 utterances.

Start the service first:
    CUDA_VISIBLE_DEVICES=7 FORCE_ALIGN_DEVICE=cuda \
        uvicorn force_alignment.server:app --host 0.0.0.0 --port 5010

Then run this test:
    python -m force_alignment.test_service
"""
from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List


def _send_align_request(
    wav_bytes: bytes,
    transcript: str,
    url: str = "http://0.0.0.0:5010/align_words",
) -> List[dict]:
    boundary = "----test-boundary"
    parts: List[bytes] = []
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="transcript"\r\n')
    parts.append(b"Content-Type: text/plain; charset=utf-8\r\n\r\n")
    parts.append(transcript.encode("utf-8"))
    parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append(b'Content-Disposition: form-data; name="audio_wav"; filename="audio.wav"\r\n')
    parts.append(b"Content-Type: audio/wav\r\n\r\n")
    parts.append(wav_bytes)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        url=url,
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    return payload["spans"]


def main() -> None:
    session_dir = Path("examples/session_1")
    wav_files = sorted(session_dir.glob("*.wav"))

    items = []
    for wav_path in wav_files:
        txt_path = wav_path.with_suffix(".txt")
        if not txt_path.exists():
            continue
        transcript = txt_path.read_text(encoding="utf-8").strip()
        if transcript:
            items.append((wav_path, transcript))

    if not items:
        print(f"No (wav, txt) pairs found in {session_dir}")
        return

    print(f"Found {len(items)} segments in {session_dir}\n")

    # ---------- Sequential ----------
    print("--- Sequential ---")
    t0 = time.perf_counter()
    for wav_path, transcript in items:
        spans = _send_align_request(wav_path.read_bytes(), transcript)
        words = transcript.split()
        ok = len(spans) == len(words)
        tag = "OK" if ok else f"MISMATCH (words={len(words)} spans={len(spans)})"
        print(f"  {wav_path.name}: {len(words)} words -> {len(spans)} spans  [{tag}]")
        if spans:
            print(f"    first: t0={spans[0]['t0']:.4f}  t1={spans[0]['t1']:.4f}")
    seq_time = time.perf_counter() - t0
    print(f"Sequential total: {seq_time:.2f}s\n")

    # ---------- Concurrent ----------
    print("--- Concurrent (all at once) ---")
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=len(items)) as pool:
        futures = {
            pool.submit(_send_align_request, wav_path.read_bytes(), transcript): wav_path.name
            for wav_path, transcript in items
        }
        for fut in as_completed(futures):
            name = futures[fut]
            spans = fut.result()
            print(f"  {name}: {len(spans)} spans")
    conc_time = time.perf_counter() - t0
    print(f"Concurrent total: {conc_time:.2f}s")

    if conc_time > 0:
        print(f"\nSpeedup: {seq_time / conc_time:.1f}x")


if __name__ == "__main__":
    main()
