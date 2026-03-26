# Use official PyTorch Docker image with CUDA 12.1 support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    wget \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install (build context is workspace root)
COPY diarization/requirements.txt ./diarization/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r diarization/requirements.txt

# Copy application and model files
COPY diarization/ ./diarization/

ENV HOST=0.0.0.0
ENV PORT=5022
ENV PIPELINE_PATH=nguyenvulebinh/speaker-diarization-community

EXPOSE 5022

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5022/live || exit 1

CMD ["python", "-m", "diarization.server"]
