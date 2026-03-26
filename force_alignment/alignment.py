from __future__ import annotations

import io
import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import List, Optional

import torch
import torchaudio

SAMPLE_RATE = 16000

_NON_ALPHA_RE = re.compile(r"([^a-z' ])")
_SPACES_RE = re.compile(r" +")


@dataclass(frozen=True)
class WordSpan:
    word: str
    t0: float
    t1: float


# ---------------------------------------------------------------------------
# uroman normalization
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_uroman():
    try:
        import uroman as ur  # type: ignore
    except Exception as e:
        raise ImportError(
            "uroman is required. Install with `pip install uroman`."
        ) from e
    return ur.Uroman()


def split_original_words(text: str) -> List[str]:
    text = text.strip()
    return text.split() if text else []


def normalize_words_preserve_length(original_words: List[str]) -> List[str]:
    """
    Produce a normalized word list with the SAME length as `original_words`,
    suitable for MMS_FA alignment.
    """
    uroman = _get_uroman()
    out: List[str] = []
    for w in original_words:
        t = uroman.romanize_string(w)
        t = t.lower().replace("\u2019", "'")
        t = _NON_ALPHA_RE.sub(" ", t)
        t = _SPACES_RE.sub(" ", t).strip().replace(" ", "")
        out.append(t if t else "a")
    return out


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def _decode_wav_bytes_to_16k_mono(wav_bytes: bytes) -> torch.Tensor:
    with io.BytesIO(wav_bytes) as f:
        wav, sr = torchaudio.load(f)
    if wav.numel() == 0:
        return torch.zeros((1, 0), dtype=torch.float32)
    if wav.dim() != 2:
        raise ValueError(f"Expected waveform (C, N). Got {tuple(wav.shape)}")
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if int(sr) != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, int(sr), SAMPLE_RATE)
    return wav


# ---------------------------------------------------------------------------
# MMS_FA model loading
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _load_mms_fa():
    from torchaudio.pipelines import MMS_FA as bundle  # type: ignore

    pref = os.environ.get("FORCE_ALIGN_DEVICE", "cpu").strip().lower()
    use_cuda = pref in {"cuda", "gpu"} and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = bundle.get_model()
    model.to(device)
    tokenizer = bundle.get_tokenizer()
    aligner = bundle.get_aligner()
    return bundle, model, tokenizer, aligner, device


def warmup() -> None:
    """Eagerly load MMS_FA model + uroman so the first request is fast."""
    _load_mms_fa()
    _get_uroman()


# ---------------------------------------------------------------------------
# Single-item alignment (kept for reference / direct calls)
# ---------------------------------------------------------------------------

def align_transcript(wav_bytes: bytes, transcript: str) -> List[WordSpan]:
    orig_words = split_original_words(transcript)
    if not orig_words:
        return []
    norm_words = normalize_words_preserve_length(orig_words)
    waveform = _decode_wav_bytes_to_16k_mono(wav_bytes)
    return _align_waveform_single(waveform, orig_words, norm_words)


def _align_waveform_single(
    waveform_16k_mono: torch.Tensor,
    orig_words: List[str],
    norm_words: List[str],
) -> List[WordSpan]:
    if not norm_words:
        return []
    if waveform_16k_mono.dim() != 2 or waveform_16k_mono.shape[0] != 1:
        raise ValueError(f"Expected waveform shape (1, N). Got {tuple(waveform_16k_mono.shape)}")
    if waveform_16k_mono.size(1) < 4000:
        return [WordSpan(word=w, t0=0.0, t1=0.0) for w in orig_words]

    _bundle, model, tokenizer, aligner, device = _load_mms_fa()

    with torch.inference_mode():
        emission, _ = model(waveform_16k_mono.to(device))
        token_spans = aligner(emission[0], tokenizer(norm_words))

    num_samples = waveform_16k_mono.size(1)
    num_frames = emission.size(1)
    ratio = num_samples / num_frames / SAMPLE_RATE

    return _token_spans_to_word_spans(token_spans, orig_words, ratio)


# ---------------------------------------------------------------------------
# Batch alignment – batched model forward, per-sample CTC aligner
# ---------------------------------------------------------------------------

def align_transcript_batch(
    wav_bytes_list: List[bytes],
    transcript_list: List[str],
) -> List[List[WordSpan]]:
    """
    Process multiple (wav_bytes, transcript) pairs collected by the batch worker.

    Items are processed individually on the GPU (torchaudio's MMS_FA wrapper
    doesn't support true batch-dim > 1).  Collecting them in one batch call
    still helps: it serialises GPU access, avoids lock contention, and
    amortises Python/CUDA overhead.
    """
    if len(wav_bytes_list) != len(transcript_list):
        raise ValueError("wav_bytes_list and transcript_list must have same length")
    return [
        align_transcript(wav, txt)
        for wav, txt in zip(wav_bytes_list, transcript_list)
    ]


# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

def _token_spans_to_word_spans(
    token_spans, orig_words: List[str], ratio: float,
) -> List[WordSpan]:
    spans: List[WordSpan] = []
    for i, t_spans in enumerate(token_spans):
        w = orig_words[i] if i < len(orig_words) else ""
        if not t_spans:
            spans.append(WordSpan(word=w, t0=0.0, t1=0.0))
            continue
        t0 = float(t_spans[0].start) * ratio
        t1 = float(t_spans[-1].end) * ratio
        spans.append(WordSpan(word=w, t0=t0, t1=t1))
    return spans
