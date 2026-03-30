from __future__ import annotations

from typing import Dict, List

from ..core.types import Window
from ..io.audio_io import SAMPLE_RATE, slice_waveform, waveform_to_wav_bytes_pcm16
from .repair import choose_reference_medoid_near_mean
from .speaker_attribute_client import SpeakerAttributeOutput, infer_word_speaker_embeddings_http


def build_speaker_reference_embeddings(
    session_waveform,
    windows_by_speaker: Dict[str, Window],
    language_by_segment: Dict[int, str],
    speaker_attribute_base_url: str = "http://0.0.0.0:5024",
) -> Dict[str, List[float]]:
    """
    For each speaker, choose a reference embedding E_s from the longest window:
    E_s = embedding of word closest to mean embedding of that window.
    """
    refs: Dict[str, List[float]] = {}
    for spk, w in windows_by_speaker.items():
        words = [x[2] for x in w.words]
        if not words:
            continue
        # Avoid degenerate reference windows (can produce empty audio blobs).
        if w.end <= w.start:
            continue
        lang = language_by_segment.get(w.words[0][0], "en")

        start_samp = int(max(0.0, w.start) * SAMPLE_RATE)
        end_samp = int(max(0.0, w.end) * SAMPLE_RATE)
        if end_samp <= start_samp:
            continue

        audio = slice_waveform(session_waveform, w.start, w.end)
        audio_bytes = waveform_to_wav_bytes_pcm16(audio)
        if not audio_bytes:
            continue
        out: SpeakerAttributeOutput = infer_word_speaker_embeddings_http(
            audio_bytes=audio_bytes,
            transcript=" ".join(words),
            language=lang,
            base_url=speaker_attribute_base_url,
        )
        if not out.embeddings:
            continue
        refs[spk] = choose_reference_medoid_near_mean(out.embeddings)
    return refs

