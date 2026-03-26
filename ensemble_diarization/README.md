## `ensemble_diarization/`

Offline pipeline that combines:

- **Diarization service (HTTP)** to produce session-level speaker turns (backend-agnostic)
- **Alignment service (HTTP)** to produce word-level timestamps (normalization + MMS_FA done server-side)
- **Speaker-attribute service (HTTP)** to produce per-word speaker embeddings
- **Repair** to make embeddings more consistent within each diarization speaker cluster

The main output is, for each input segment, the original words plus a "best effort" speaker embedding for each word.

### Components

- **core/**
  - `core/pipeline.py`: end-to-end pipeline + `PipelineConfig`
  - `core/types.py`: shared dataclasses/typed dicts
- **io/**
  - `io/audio_io.py`: decode/resample/slice + WAV encoding helpers
  - `io/session_assembler.py`: concatenate segments with fixed silence gaps + offsets
- **diarization/**
  - `diarization/diarization_client.py`: HTTP client for `POST /diarize` (service-agnostic)
  - `diarization/exclusive_regions.py`: converts turns into "exclusive" single-speaker regions
  - `diarization/windowing.py`: builds single-speaker windows for higher-quality embedding calls
- **alignment/**
  - `alignment/alignment_client.py`: HTTP client for the forced-alignment service (sends audio + raw transcript)
- **embedding/**
  - `embedding/speaker_attribute_client.py`: HTTP client for speaker-attribute service
  - `embedding/reference_embedding.py`: legacy reference embedding builder (`repair_policy="reference"`)
  - `embedding/repair.py`: central-pool repair and helpers
  - `embedding/verification.py`: nearest-prototype verification vs diarization labels
- **scripts/**
  - `scripts/scripts_utils.py`: helpers for scripts (timestamp sort, loading session directories)
  - `scripts/run_example.py`: single-file example runner
  - `scripts/run_session_1.py`: runs the full `/examples/session_1` dataset + prints verification
  - `scripts/export_session_1_output.py`: exports concatenated session audio + RTTM + diarization-turn text

### External services required

1) **Diarization service**
- URL used by default: `http://0.0.0.0:5000/diarize`

2) **Alignment service**
- URL used by default: `http://0.0.0.0:5010/align_words`
- Endpoint: `POST /align_words` multipart form with `audio_wav` (WAV bytes) + `transcript` (raw text)
- Normalization (uroman) and MMS_FA alignment happen server-side

3) **Speaker-attribute service**
- Base URL used by default: `http://192.168.0.60:5024`
- Endpoint called: `POST {base_url}/asr/infer/{language},None` with multipart `pcm_s16le` + `transcript`

### How to run

All examples below assume you run with a Python that has the dependencies installed:

#### 1) Single example (one segment)

```bash
python -m ensemble_diarization.scripts.run_example
```

#### 2) Run `examples/session_1` (sorted by timestamp) + verification

```bash
python -m ensemble_diarization.scripts.run_session_1
```

This prints per-word:
- original word
- assigned diarization speaker label
- confidence tag (`high` / `sent`)
- embedding head

Then it prints a simple verification summary (nearest-prototype reassignment vs diarization labels).
Diarization speaker labels are treated as *pseudo labels*.

#### 3) Export concatenated session audio + RTTM + diarization-turn text

```bash
python -m ensemble_diarization.scripts.export_session_1_output
```

Outputs are written to:
- `examples/session_1_output/session_1_concat.wav`
- `examples/session_1_output/session_1_concat.rttm`
- `examples/session_1_output/session_1_concat.turns_with_text.jsonl`
- `examples/session_1_output/session_1_concat.segments.txt`

### Notes on "best embedding per word"

The pipeline calls speaker-attribute at **two levels**:
- **Segment-level**: always call on the whole segment to ensure every word gets an embedding.
- **Window-level (single speaker)**: also call on single-speaker windows and override embeddings for words inside those windows.

Then `repair_policy="central_pool"` enforces per-speaker consistency by replacing "bad" embeddings (below threshold / missing) using cluster-level statistics.
