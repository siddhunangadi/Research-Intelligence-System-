# Production Dockerfile for Research Intelligence System
# Multi-purpose image: can run FastAPI backend or Streamlit frontend

# Use python-slim for a smaller production image footprint
FROM python:3.10-slim

# System setup
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    HF_HOME=/app/data/embeddings/hf_cache \
    CHROMA_DB_PATH=/app/data/chroma_db

WORKDIR /app

# Install system dependencies needed for compiling psycopg2 and general utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -U pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download HuggingFace and SentenceTransformer model weights to cache them in the Docker layer
# This prevents network fetch issues and latency spikes on service startup in production
RUN python -c " \
import os; \
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification; \
models = [ \
    'BAAI/bge-large-en-v1.5', \
    'cross-encoder/ms-marco-MiniLM-L-6-v2', \
    'cross-encoder/nli-deberta-v3-base' \
]; \
for m in models: \
    print(f'Caching model: {m}'); \
    AutoTokenizer.from_pretrained(m); \
    try: \
        AutoModelForSequenceClassification.from_pretrained(m); \
    except Exception: \
        AutoModel.from_pretrained(m); \
"

# Copy application code
COPY . .

# Create directories for data and logs, configure permissions for non-root execution
RUN mkdir -p data/raw_papers data/processed_papers data/extracted_tables data/embeddings/hf_cache data/chroma_db logs && \
    chmod -R 777 data logs

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000
EXPOSE 8501

# Default command (can be overridden in docker-compose.yml or docker run)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
