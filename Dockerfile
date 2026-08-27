# ============================================================
# TamilGuard AI — Multi-stage Docker Build
# ============================================================
# Stage 1: Build the React frontend
# Stage 2: Python production image with model + built frontend
# ============================================================

# ── Stage 1: Build React Frontend ─────────────────────────────
FROM node:20 AS frontend-builder

WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install --include=optional
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python Production Image ──────────────────────────
FROM python:3.11-slim

# System deps
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY inference.py normalize.py redact.py serve.py torchcrf.py ./
COPY models/ models/
COPY lexicon/ lexicon/
COPY scripts/generate_weak_bio.py scripts/generate_weak_bio.py

# Copy pre-built frontend from Stage 1
COPY --from=frontend-builder /app/frontend/dist frontend/dist/

# Port
ENV PORT=5000
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health')" || exit 1

# Run with Gunicorn (1 worker to keep memory low with PyTorch)
# We use sh -c to ensure the environment variable $PORT is expanded
CMD ["sh", "-c", "gunicorn serve:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 120"]
