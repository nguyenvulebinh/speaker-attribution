# Use official PyTorch Docker image with CUDA 12.1 support (compatible with CUDA 12.2.0)
# This ensures PyTorch, TorchAudio, and CUDA compatibility are pre-tested
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

# Set environment variables
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

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install additional Python dependencies (PyTorch and TorchAudio are already installed)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create directory for model cache if needed
RUN mkdir -p /app/model-cache

# Environment variables with defaults (can be overridden at runtime)
ENV HOST=0.0.0.0
ENV PORT=5024

# Expose the port
EXPOSE 5024

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:5024/asr/version || exit 1

# Start the Flask application directly
CMD ["python", "server.py"]

# Labels for better image management
LABEL maintainer="msa-asr"
LABEL description="msa-asr-service"
LABEL version="1.0"