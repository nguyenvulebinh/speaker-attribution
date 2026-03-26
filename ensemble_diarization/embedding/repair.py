from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class RepairResult:
    embeddings: List[Optional[List[float]]]
    speaker_pools: Dict[str, List[List[float]]] = field(default_factory=dict)


def choose_reference_medoid_near_mean(embeddings: List[List[float]]) -> List[float]:
    """
    Choose the word embedding closest to the mean embedding.
    Returns a vector from the input list (not the mean itself).
    """
    if not embeddings:
        raise ValueError("No embeddings to choose reference from.")

    import numpy as np

    E = np.asarray(embeddings, dtype=np.float32)
    mu = E.mean(axis=0, keepdims=True)
    d2 = ((E - mu) ** 2).sum(axis=1)
    idx = int(np.argmin(d2))
    return embeddings[idx]


def _cosine_similarity(a, b) -> float:
    import numpy as np

    av = np.asarray(a, dtype=np.float32)
    bv = np.asarray(b, dtype=np.float32)
    na = float(np.linalg.norm(av) + 1e-12)
    nb = float(np.linalg.norm(bv) + 1e-12)
    return float(np.dot(av, bv) / (na * nb))


def _robust_center_topk_near_mean(embeddings: List[List[float]], top_k: int) -> List[float]:
    """
    Compute a robust center:
    - mean over all embeddings
    - pick top-k closest to that mean (cosine distance)
    - return mean of the selected top-k
    """
    if not embeddings:
        raise ValueError("No embeddings for center.")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")

    import numpy as np

    E = np.asarray(embeddings, dtype=np.float32)
    mu = E.mean(axis=0)
    sims = [float(np.dot(e, mu) / ((np.linalg.norm(e) + 1e-12) * (np.linalg.norm(mu) + 1e-12))) for e in E]
    order = sorted(range(len(embeddings)), key=lambda i: sims[i], reverse=True)
    k = min(top_k, len(order))
    mu2 = E[order[:k]].mean(axis=0)
    return mu2.astype(np.float32).tolist()


def compute_stable_speaker_pools(
    embeddings: List[Optional[List[float]]],
    speakers: List[Optional[str]],
    *,
    similarity_threshold: float = 0.8,
    center_top_k: int = 10,
    max_pool_per_speaker: int = 50,
) -> Dict[str, List[List[float]]]:
    """
    Compute stable embedding pools per speaker from session-wide data.

    For each speaker: compute robust center, keep embeddings with
    cosine sim >= threshold to that center, cap at max_pool_per_speaker.
    """
    by_spk: Dict[str, List[List[float]]] = {}
    for emb, spk in zip(embeddings, speakers):
        if spk is None or emb is None:
            continue
        by_spk.setdefault(str(spk), []).append(emb)

    pools: Dict[str, List[List[float]]] = {}
    for spk, embs in by_spk.items():
        center = _robust_center_topk_near_mean(embs, top_k=center_top_k)
        scored = []
        for e in embs:
            sim = _cosine_similarity(e, center)
            if sim >= similarity_threshold:
                scored.append((sim, e))
        if not scored:
            scored = [(_cosine_similarity(e, center), e) for e in embs]
        scored.sort(key=lambda x: x[0], reverse=True)
        pools[spk] = [e for _sim, e in scored[: min(max_pool_per_speaker, len(scored))]]
    return pools


def repair_embeddings_central_pool(
    words: List[str],
    embeddings: List[Optional[List[float]]],
    speakers: List[Optional[str]],
    confidences: Optional[List[str]] = None,
    *,
    similarity_threshold: float = 0.8,
    center_top_k: int = 10,
    max_pool_per_speaker: int = 50,
    replace_mode: str = "random",
    random_seed: int = 0,
) -> RepairResult:
    """
    Use diarization speakers as pseudo-label clusters.

    For each speaker cluster:
    - compute robust center embedding (mean of top-k closest-to-mean)
    - keep embeddings close to the center (cosine sim >= threshold), cap to max_pool
    - for each word embedding: if missing or far from center, replace with one from the pool

    Returns RepairResult with repaired embeddings and the stable speaker pools.

    Notes:
    - Only words with `speaker != None` are considered for repair.
    - If a speaker has no pool, we fall back to its center.
    - `replace_mode`:
        - "random": sample from pool
        - "best": most central in pool
        - "centroid": always use the cluster center
    """
    _ = words  # words are not used, but kept for interface consistency
    if not (len(embeddings) == len(speakers)):
        raise ValueError("embeddings and speakers must have same length.")
    if confidences is not None and len(confidences) != len(embeddings):
        raise ValueError("confidences length must match embeddings when provided.")
    if not (0.0 <= similarity_threshold <= 1.0):
        raise ValueError("similarity_threshold must be in [0, 1].")
    if max_pool_per_speaker <= 0:
        raise ValueError("max_pool_per_speaker must be > 0.")
    if replace_mode not in {"random", "best", "centroid"}:
        raise ValueError("replace_mode must be 'random', 'best', or 'centroid'.")

    import random

    rng = random.Random(random_seed)

    # collect embeddings per speaker
    by_spk: Dict[str, List[List[float]]] = {}
    for emb, spk in zip(embeddings, speakers):
        if spk is None or emb is None:
            continue
        by_spk.setdefault(str(spk), []).append(emb)

    centers: Dict[str, List[float]] = {}
    pools: Dict[str, List[List[float]]] = {}

    for spk, embs in by_spk.items():
        center = _robust_center_topk_near_mean(embs, top_k=center_top_k)
        centers[spk] = center

        # filter by similarity to center, then cap by most central
        scored = []
        for e in embs:
            sim = _cosine_similarity(e, center)
            if sim >= similarity_threshold:
                scored.append((sim, e))
        if not scored:
            # if nothing passes threshold, keep the most central few anyway
            scored = [(_cosine_similarity(e, center), e) for e in embs]
        scored.sort(key=lambda x: x[0], reverse=True)
        pools[spk] = [e for _sim, e in scored[: min(max_pool_per_speaker, len(scored))]]

    out: List[Optional[List[float]]] = []
    for emb, spk in zip(embeddings, speakers):
        if spk is None:
            out.append(None if emb is None else emb)
            continue

        spk = str(spk)
        center = centers.get(spk)
        pool = pools.get(spk, [])

        if center is None:
            out.append(emb)
            continue

        needs_replace = emb is None or (_cosine_similarity(emb, center) < similarity_threshold)
        if not needs_replace:
            out.append(emb)
            continue

        if replace_mode == "centroid":
            out.append(center)
        elif pool:
            if replace_mode == "best":
                out.append(pool[0])
            else:
                out.append(rng.choice(pool))
        else:
            out.append(center)

    return RepairResult(embeddings=out, speaker_pools=pools)


def repair_embeddings(
    words: List[str],
    embeddings: List[Optional[List[float]]],
    speakers: List[Optional[str]],
    confidences: List[str],
    speaker_reference: Dict[str, List[float]],
) -> RepairResult:
    """
    Replace low-confidence embeddings with per-speaker reference embedding when possible.
    Returns RepairResult with repaired embeddings and speaker_reference as pools.
    """
    if not (len(words) == len(embeddings) == len(speakers) == len(confidences)):
        raise ValueError("Input lengths must match.")

    out: List[Optional[List[float]]] = []
    for emb, spk, conf in zip(embeddings, speakers, confidences):
        if conf == "high" and emb is not None:
            out.append(emb)
            continue
        if spk is not None and spk in speaker_reference:
            out.append(speaker_reference[spk])
        else:
            out.append(None)

    pools: Dict[str, List[List[float]]] = {
        spk: [ref] for spk, ref in speaker_reference.items()
    }
    return RepairResult(embeddings=out, speaker_pools=pools)

