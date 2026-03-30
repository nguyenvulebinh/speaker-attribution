from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional
from typing import Any, Dict

import requests


@dataclass(frozen=True)
class SpeakerAttributeOutput:
    words: List[str]
    embeddings: List[List[float]]
    hypo: Optional[str] = None
    lid: Optional[str] = None


def infer_word_speaker_embeddings_http(
    audio_bytes: bytes,
    transcript: str,
    language: str,
    base_url: str = "http://192.168.0.60:5024",
    timeout_s: float = 600000.0,
) -> SpeakerAttributeOutput:
    """
    Call speaker attribution service exactly like speaker_attribute/test_call_api.py:
      POST {base_url}/asr/infer/{language},None with multipart files:
        - pcm_s16le: bytes
        - transcript: utf-8 string

    Returns per-word embeddings from response['saasr'] = [words, embeddings].
    """
    if audio_bytes is None:
        raise ValueError("audio_bytes must not be None")

    # Flask/Werkzeug's request.files is sensitive to how multipart parts are
    # encoded. Send `pcm_s16le` explicitly as a multipart *file* part.
    files: Dict[str, Any] = {
        "pcm_s16le": ("pcm_s16le.wav", audio_bytes, "audio/wav"),
    }
    # Send transcript as a normal form field (not a file part).
    data: Dict[str, Any] = {"transcript": transcript}

    url = f"{base_url}/asr/infer/{language},None"
    print(
        f"[speaker_attribute_client] POST {url} "
        f"pcm_s16le_bytes={len(audio_bytes)} transcript_chars={len(transcript)}"
    )

    resp = requests.post(url, files=files, data=data, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    saasr = data.get("saasr")
    if not saasr or len(saasr) != 2:
        raise ValueError(f"Invalid speaker_attribute response: missing/invalid 'saasr' in {data.keys()}")
    words, embeds = saasr
    return SpeakerAttributeOutput(words=words, embeddings=embeds, hypo=data.get("hypo"), lid=data.get("lid"))

