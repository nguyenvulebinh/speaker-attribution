from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _to_unit(x: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(x) + 1e-12)
    return x / n


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    Cosine distance in [0, 2]. 0 means identical direction.
    """
    a_u = _to_unit(a)
    b_u = _to_unit(b)
    return float(1.0 - np.dot(a_u, b_u))


@dataclass(frozen=True)
class Prototype:
    speaker: str
    mean_all: np.ndarray
    mean_topk: np.ndarray
    topk_indices: List[int]  # indices into the flattened word list


@dataclass(frozen=True)
class VerificationResult:
    total: int
    correct: int
    error_rate: float
    speakers: List[str]
    confusion: Dict[Tuple[str, str], int]  # (true, pred) -> count
    prototypes: Dict[str, Prototype]


def build_prototypes_topk_near_mean(
    embeddings: Sequence[Optional[Sequence[float]]],
    speakers: Sequence[Optional[str]],
    top_k: int = 10,
) -> Dict[str, Prototype]:
    """
    For each speaker cluster, compute mean embedding over all available embeddings, then
    take top-k words closest to that mean (cosine distance), and compute a refined mean over top-k.
    """
    if len(embeddings) != len(speakers):
        raise ValueError("embeddings and speakers must have same length")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    by_spk: Dict[str, List[int]] = {}
    for i, spk in enumerate(speakers):
        if spk is None:
            continue
        if embeddings[i] is None:
            continue
        by_spk.setdefault(str(spk), []).append(i)

    prototypes: Dict[str, Prototype] = {}
    for spk, idxs in by_spk.items():
        mat = np.stack([np.asarray(embeddings[i], dtype=np.float32) for i in idxs], axis=0)
        mean_all = mat.mean(axis=0)

        dists = [cosine_distance(np.asarray(embeddings[i], dtype=np.float32), mean_all) for i in idxs]
        order = sorted(range(len(idxs)), key=lambda j: dists[j])
        k = min(top_k, len(order))
        topk_indices = [idxs[order[j]] for j in range(k)]
        mat_top = np.stack([np.asarray(embeddings[i], dtype=np.float32) for i in topk_indices], axis=0)
        mean_topk = mat_top.mean(axis=0)

        prototypes[spk] = Prototype(
            speaker=spk,
            mean_all=mean_all.astype(np.float32),
            mean_topk=mean_topk.astype(np.float32),
            topk_indices=topk_indices,
        )

    return prototypes


def predict_speaker_by_nearest_prototype(
    emb: Sequence[float],
    prototypes: Dict[str, Prototype],
    use: str = "topk_mean",
) -> str:
    if not prototypes:
        raise ValueError("No prototypes provided")
    v = np.asarray(emb, dtype=np.float32)
    best_spk = None
    best_dist = None
    for spk, proto in prototypes.items():
        p = proto.mean_topk if use == "topk_mean" else proto.mean_all
        d = cosine_distance(v, p)
        if best_dist is None or d < best_dist:
            best_dist = d
            best_spk = spk
    assert best_spk is not None
    return best_spk


def verify_cluster_consistency(
    words: Sequence[str],
    embeddings: Sequence[Optional[Sequence[float]]],
    diarization_speakers: Sequence[Optional[str]],
    top_k: int = 10,
    prototype_use: str = "topk_mean",
) -> VerificationResult:
    """
    Verification:
    - build per-cluster prototypes using top-k closest-to-mean strategy
    - for each word, assign cluster by nearest prototype
    - compute error rate using diarization speaker as reference

    Only evaluates words with (speaker != None) and (embedding != None).
    """
    if not (len(words) == len(embeddings) == len(diarization_speakers)):
        raise ValueError("words, embeddings, diarization_speakers must have same length")

    prototypes = build_prototypes_topk_near_mean(embeddings, diarization_speakers, top_k=top_k)
    speakers_sorted = sorted(prototypes.keys())

    total = 0
    correct = 0
    confusion: Dict[Tuple[str, str], int] = {}

    for w, emb, true_spk in zip(words, embeddings, diarization_speakers):
        _ = w
        if true_spk is None or emb is None:
            continue
        true_spk = str(true_spk)
        pred = predict_speaker_by_nearest_prototype(emb, prototypes, use=prototype_use)
        total += 1
        if pred == true_spk:
            correct += 1
        confusion[(true_spk, pred)] = confusion.get((true_spk, pred), 0) + 1

    err = 1.0 - (correct / total) if total else 0.0
    return VerificationResult(
        total=total,
        correct=correct,
        error_rate=err,
        speakers=speakers_sorted,
        confusion=confusion,
        prototypes=prototypes,
    )


def flatten_output(
    segments: Sequence[List[object]],
) -> Tuple[List[str], List[Optional[List[float]]], List[Optional[str]]]:
    """
    Flatten per-segment pipeline output to flat lists.

    Each item is: [words, embeddings, speakers, confidences]
    """
    words_flat: List[str] = []
    emb_flat: List[Optional[List[float]]] = []
    spk_flat: List[Optional[str]] = []

    for item in segments:
        w, e, s, _c = item
        words_flat.extend(list(w))
        emb_flat.extend(list(e))
        spk_flat.extend(list(s))

    return words_flat, emb_flat, spk_flat


def format_confusion_matrix(
    confusion: Dict[Tuple[str, str], int],
    speakers: Sequence[str],
) -> str:
    """
    Return a compact text table. Rows=true, cols=pred.
    """
    header = ["true\\pred"] + list(speakers)
    rows = [header]
    for t in speakers:
        row = [t]
        for p in speakers:
            row.append(str(confusion.get((t, p), 0)))
        rows.append(row)

    widths = [max(len(rows[r][c]) for r in range(len(rows))) for c in range(len(rows[0]))]
    lines = []
    for r, row in enumerate(rows):
        line = "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))
        lines.append(line)
        if r == 0:
            lines.append("  ".join("-" * w for w in widths))
    return "\n".join(lines)

