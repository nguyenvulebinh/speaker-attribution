# Use PyTorch base (torch 2.4 from image)
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install (build context is workspace root)
COPY ensemble_diarization/requirements.txt ./ensemble_diarization/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r ensemble_diarization/requirements.txt

# Copy application files
COPY ensemble_diarization/ ./ensemble_diarization/

ENV HOST=0.0.0.0
ENV PORT=5025

EXPOSE 5025

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5025/healthz || exit 1

CMD ["python", "-m", "ensemble_diarization.server"]
