# =============================================================
# Multi-stage production Dockerfile
# App: Sentiment Analysis Streamlit App (Python / Streamlit / VADER)
# Runtime target: EC2 Docker (port 8501)
# =============================================================

# -----------------------------
# Stage 1: Builder - dependency installation
# -----------------------------
FROM python:3.11-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg

WORKDIR /app

# Install build-time OS deps required by numpy/scikit-learn/wordcloud(matplotlib) wheels
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first to maximize layer caching
COPY requirements.txt .

RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

# -----------------------------
# Stage 2: Runtime - minimal production image
# -----------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    NLTK_DATA=/usr/local/share/nltk_data \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Minimal runtime OS libs (font support for wordcloud/matplotlib)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Pre-download NLTK assets at build time so container start never depends on network
RUN python -c "import nltk; \
nltk.download('punkt', download_dir='/usr/local/share/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/usr/local/share/nltk_data', quiet=True); \
nltk.download('vader_lexicon', download_dir='/usr/local/share/nltk_data', quiet=True)"

# Copy application source (preserves layer cache: deps installed before source copy)
COPY app.py ./
COPY requirements.txt ./
COPY src ./src
COPY models ./models

# Non-root user for production hardening
RUN useradd --create-home --shell /usr/sbin/nologin appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://127.0.0.1:8501/_stcore/health || exit 1

ENTRYPOINT []
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]