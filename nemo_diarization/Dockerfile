# Use official PyTorch Docker image with CUDA 12.1 support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime AS builder

# Install system dependencies (build tools, git, ffmpeg, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create virtual environment inside /opt/venv
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Upgrade pip and install dependencies
COPY nemo_diarization/requirements.txt ./nemo_diarization/requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir 'uvicorn[standard]' \
    && pip install --no-cache-dir -r nemo_diarization/requirements.txt \
    && pip install --no-cache-dir 'nemo_toolkit[asr]==2.5.3' \
    && pip cache purge

# ================================
# Stage 2: Runtime
# ================================
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg gcc g++ curl libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application code
COPY nemo_diarization/ ./nemo_diarization/

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv

# Set env vars
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

ENV HOST=0.0.0.0
ENV PORT=5022

EXPOSE 5022

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5022/live || exit 1

CMD ["python", "-m", "nemo_diarization.server"]