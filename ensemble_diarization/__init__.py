"""Ensemble diarization + word-level speaker embeddings (offline orchestrator)."""

# Public API re-exports (preferred imports)
from .core.pipeline import PipelineConfig, PipelineOutput, compute_word_speaker_embeddings
from .core.types import SegmentInput


