# Use official PyTorch Docker image with CUDA 12.1 support
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# Copy requirements and install (build context is workspace root)
COPY force_alignment/requirements.txt ./force_alignment/
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r force_alignment/requirements.txt

# Copy application files
COPY force_alignment/ ./force_alignment/

ENV HOST=0.0.0.0
ENV PORT=5023
ENV FORCE_ALIGN_DEVICE=cuda

EXPOSE 5023

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5023/healthz || exit 1

CMD ["uvicorn", "force_alignment.server:app", "--host", "0.0.0.0", "--port", "5023"]
