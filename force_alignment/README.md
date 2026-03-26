## `force_alignment/`

HTTP service for **forced word alignment** using torchaudio `MMS_FA`.

The service handles everything internally: word splitting, uroman normalization,
resampling to 16 kHz mono, and MMS_FA alignment.

### Features

- **Eager model loading** — MMS_FA model + uroman are loaded at startup (not on
  the first request).
- **Queue-based batch inference** — concurrent requests are collected into
  batches (up to 32) and processed together, giving significant throughput
  improvement under concurrent load (~13x on 9 segments).

### Endpoint

- `POST /align_words`
  - multipart form fields:
    - `audio_wav`: WAV audio bytes (any sample rate / channels; converted internally)
    - `transcript`: raw transcript text (words separated by spaces)
  - response: `{"spans": [{"t0": <sec>, "t1": <sec>}, ...]}` (one span per word)

- `GET /healthz`

### Run

```bash
pip install fastapi uvicorn torch torchaudio uroman python-multipart
FORCE_ALIGN_DEVICE=cuda uvicorn force_alignment.server:app --host 0.0.0.0 --port 5010
```

### Device selection

Set `FORCE_ALIGN_DEVICE=cuda` to run MMS_FA on GPU (default: `cpu`).

```bash
CUDA_VISIBLE_DEVICES=7 FORCE_ALIGN_DEVICE=cuda \
    uvicorn force_alignment.server:app --host 0.0.0.0 --port 5010
```

### Test

```bash
python -m force_alignment.test_service
```

Sends all segments from `examples/session_1` both sequentially and concurrently,
and reports span counts + timing.
