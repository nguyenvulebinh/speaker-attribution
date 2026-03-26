## Speaker Attribution

This project provides a diarization pipeline that turns audio into per-word speaker attribution.

It runs three model services (diarization, forced alignment, speaker-embedding inference) and an optional orchestrator that merges them into a single per-segment/per-word JSON result.

This repository runs a 4-service pipeline via `docker compose`:

- `nemo_diarization`: speaker turn diarization (HTTP)
- `force_alignment`: forced word alignment (HTTP)
- `speaker_attribute`: per-word speaker embedding inference (HTTP)
- `ensemble_diarization`: end-to-end pipeline that combines the 3 services above

## Start the stack

From the repo root:

```bash
docker compose up --build
```

GPU selection (optional). Defaults match `docker-compose.yaml`:

- `CUDA_DIARIZATION` (default `7`) -> `nemo_diarization`
- `CUDA_FORCE_ALIGNMENT` (default `7`) -> `force_alignment`
- `CUDA_ASR` (default `7`) -> `speaker_attribute`
- `CUDA_ENSEMBLE` (default `7`) -> `ensemble_diarization`

## Endpoints

- `nemo_diarization`: `http://localhost:5022/live` and `/diarize`
- `force_alignment`: `http://localhost:5023/healthz` and `/align_words`
- `speaker_attribute`: `http://localhost:5024/asr/infer/<language>,None`
- `ensemble_diarization`: `http://localhost:5025/healthz` and `/ensemble_diarize`


## Quick usage (all services)

### `ensemble_diarization`

Exposes:

- `POST /ensemble_diarize`
- `Content-Type: multipart/form-data`
- Fields:
  - `audio_wav`: WAV audio bytes
  - `segments_json`: JSON array describing the segments to process
    - each item should include: `start` (seconds), `duration` (seconds), `text` (transcript), and optional `language` (default `en`)

Example:

```bash
cat > segments.json <<'JSON'
[
  {"start": 0.0, "duration": 5.0, "text": "hello world", "language": "en"}
]
JSON

curl -s -X POST "http://localhost:5025/ensemble_diarize" \
  -F "audio_wav=@test_audio.wav" \
  -F "segments_json=@segments.json" | jq .
```

Output format (JSON):

```json
{
  "segments": [
    {
      "start": 0.0,
      "duration": 5.0,
      "text": "hello world",
      "language": "en",
      "words": [
        {
          "word": "hello",
          "speaker": "SPEAKER_0",
          "confidence": "high",
          "embedding": [0.01, -0.02, "..."]
        }
      ]
    }
  ],
  "speakers": {
    "SPEAKER_0": {"embeddings": [[0.01, -0.02, "..."], [0.03, 0.04, "..."]]}
  }
}
```

### `nemo_diarization`

Exposes:

- `POST /diarize` (multipart form)

Example:

```bash
curl -s -X POST "http://localhost:5022/diarize" \
  -F "audio_wav=@test_audio.wav" | jq .
```

Output format (JSON):

```json
[
  {"start": 0.123, "end": 1.234, "speaker": "SPEAKER_0"},
  {"start": 1.234, "end": 2.345, "speaker": "SPEAKER_1"}
]
```

### `force_alignment`

Exposes:

- `POST /align_words` (multipart form)

Example:

```bash
curl -s -X POST "http://localhost:5023/align_words" \
  -F "audio_wav=@test_audio.wav" \
  -F "transcript=@transcript.txt" | jq .
```

Output format (JSON):

```json
{
  "spans": [
    {"word": "hello", "t0": 0.12, "t1": 0.34},
    {"word": "world", "t0": 0.35, "t1": 0.56}
  ]
}
```

### `speaker_attribute`

Exposes:

- `POST /asr/infer/<language>,None`

Example:

```bash
curl -s -X POST "http://localhost:5024/asr/infer/en,None" \
  -F "pcm_s16le=@test_audio.wav" \
  -F "transcript=@transcript.txt" | jq .
```

Note: `pcm_s16le` input should be 16 kHz, little-endian PCM with `s16le` codec (the service will not correct codec issues).

Output format (JSON):

```json
{
  "hypo": "<transcript string>",
  "lid": "en",
  "saasr": [
    ["word0", "word1", "..."],
    [[/* embedding floats for word0 */], [/* embedding floats for word1 */], "..."]
  ]
}
```

`saasr[1]` is a list of per-word speaker embedding vectors (each is a list of numbers).
