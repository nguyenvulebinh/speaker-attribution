import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from nemo.collections.asr.models import ClusteringDiarizer
from omegaconf import OmegaConf


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    duration: float


class DiarizationService:
    """
    Wraps ClusteringDiarizer as a stateless service.
    No manifest file management needed by the caller.
    """

    def __init__(
        self,
        model_config_path: str = "diar_infer_general.yaml",
        output_dir: str = "/tmp/diarizer_output",
        num_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        oracle_vad: bool = False,
        device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        config = OmegaConf.load(model_config_path)
        config.diarizer.out_dir = str(self.output_dir)
        config.diarizer.oracle_vad = oracle_vad

        # Optional device override (e.g. "cuda" / "cuda:1" / "cpu").
        if device is not None:
            config.device = device

        # Defaults used when caller doesn't provide speaker count params.
        self._num_speakers_default = num_speakers
        self._max_speakers_default = max_speakers

        config.diarizer.clustering.parameters.oracle_num_speakers = num_speakers is not None

        if num_speakers is not None:
            config.diarizer.clustering.parameters.num_speakers = num_speakers

        if max_speakers is not None:
            config.diarizer.clustering.parameters.max_num_speakers = max_speakers

        # Model is loaded once, reused across calls
        self._model = ClusteringDiarizer(cfg=config)
        self._config = config

    def _apply_speaker_params(
        self,
        *,
        num_speakers: Optional[int],
        max_speakers: Optional[int],
    ) -> None:
        """
        Best-effort update of speaker count parameters.

        Note: NeMo's internal parameter plumbing varies across versions;
        we try both the held config and model-private params.
        """
        oracle_num = num_speakers is not None

        # Update config used to construct the diarizer.
        self._config.diarizer.clustering.parameters.oracle_num_speakers = oracle_num
        self._config.diarizer.clustering.parameters.num_speakers = num_speakers
        if max_speakers is not None:
            self._config.diarizer.clustering.parameters.max_num_speakers = max_speakers

        # Also update model-private params if they exist.
        private = getattr(self._model, "_diarizer_params", None)
        try:
            if private is not None:
                # The structure may differ, but this covers the common OmegaConf-like case.
                private.clustering.parameters.oracle_num_speakers = oracle_num
                private.clustering.parameters.num_speakers = num_speakers
                if max_speakers is not None:
                    private.clustering.parameters.max_num_speakers = max_speakers
        except Exception:
            # If it fails, we'll rely on the manifest + config values.
            pass

    def diarize(
        self,
        audio_path: str,
        *,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        cleanup_rttm: bool = True,
    ) -> list[Segment]:
        # NeMo ClusterDiarizer does not support `min_speakers` directly.
        _ = min_speakers

        audio_path = str(Path(audio_path).resolve())
        resolved_num_speakers = (
            num_speakers if num_speakers is not None else self._num_speakers_default
        )
        resolved_max_speakers = (
            max_speakers if max_speakers is not None else self._max_speakers_default
        )

        self._apply_speaker_params(
            num_speakers=resolved_num_speakers,
            max_speakers=resolved_max_speakers,
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
            manifest_path = tmp.name
            json.dump({
                "audio_filepath": audio_path,
                "offset": 0,
                "duration": None,
                "label": "infer",
                "text": "-",
                # When oracle_num_speakers=True, NeMo reads this from the manifest.
                "num_speakers": resolved_num_speakers,
                "rttm_filepath": None,
                "uem_filepath": None,
            }, tmp)
            tmp.write("\n")

        try:
            self._model._diarizer_params.manifest_filepath = manifest_path
            self._model.diarize()
        finally:
            os.unlink(manifest_path)

        segments = self._parse_rttm(audio_path)
        if cleanup_rttm:
            try:
                stem = Path(audio_path).stem
                rttm_path = self.output_dir / "pred_rttms" / f"{stem}.rttm"
                if rttm_path.exists():
                    rttm_path.unlink()
            except Exception:
                # Non-fatal: the next run can still succeed.
                pass
        return segments

    def _parse_rttm(self, audio_path: str) -> list[Segment]:
        """Read the RTTM output NeMo writes to output_dir."""
        stem = Path(audio_path).stem
        rttm_path = self.output_dir / "pred_rttms" / f"{stem}.rttm"

        if not rttm_path.exists():
            raise FileNotFoundError(f"RTTM not found: {rttm_path}")

        segments = []
        with open(rttm_path) as f:
            for line in f:
                parts = line.strip().split()
                if not parts or parts[0] != "SPEAKER":
                    continue
                start = float(parts[3])
                duration = float(parts[4])
                speaker = parts[7]
                segments.append(
                    Segment(
                        speaker=speaker,
                        start=start,
                        end=round(start + duration, 3),
                        duration=duration,
                    )
                )

        return sorted(segments, key=lambda s: s.start)

if __name__ == "__main__":
    # One-time setup (loads the model)
    service = DiarizationService()
    print("Diarizing audio...")
    # Call as many times as needed — no manifest management
    import time
    start_time = time.time()
    segments = service.diarize("/home/tbnguyen/workspaces/nemo/try_sortformer_diarizer/alex-waibel-full.wav")
    end_time = time.time()
    for seg in segments:
        print(f"[{seg.start:.2f}s → {seg.end:.2f}s] {seg.speaker}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")