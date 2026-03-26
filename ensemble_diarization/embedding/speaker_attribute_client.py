from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

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
    files = {
        "pcm_s16le": audio_bytes,
        "transcript": transcript,
    }
    url = f"{base_url}/asr/infer/{language},None"
    resp = requests.post(url, files=files, timeout=timeout_s)
    resp.raise_for_status()
    data = resp.json()
    saasr = data.get("saasr")
    if not saasr or len(saasr) != 2:
        raise ValueError(f"Invalid speaker_attribute response: missing/invalid 'saasr' in {data.keys()}")
    words, embeds = saasr
    return SpeakerAttributeOutput(words=words, embeddings=embeds, hypo=data.get("hypo"), lid=data.get("lid"))

