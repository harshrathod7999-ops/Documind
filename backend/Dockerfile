# DocuMind backend — self-contained image (deploys to Render / HF Spaces, free tier).
#
# Build from THIS backend directory (everything it needs is vendored here):
#   docker build -t documind-api .
#
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/hf_cache \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download embedding + reranker models so cold starts are fast.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
    SentenceTransformer('BAAI/bge-large-en-v1.5'); \
    CrossEncoder('BAAI/bge-reranker-v2-m3')"

# App code + the vendored shared router/tracing (both local to this folder).
COPY app ./app
COPY shared ./shared

EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
