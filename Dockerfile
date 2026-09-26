# Syntax=docker/dockerfile:1

FROM python:3.11-slim AS builder

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install build-time system dependencies (wheel/toolchain for numeric libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifest first to maximize layer caching
COPY requirements.txt .

RUN pip install --prefix=/install --no-warn-script-location -r requirements.txt

# ---------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    NLTK_DATA=/usr/share/nltk_data

# Minimal runtime system libs (matplotlib/wordcloud/font rendering)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    fontconfig \
    fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*

# Copy installed python packages from builder
COPY --from=builder /install /usr/local

# Pre-download NLTK assets at build time so runtime needs no network access
RUN python -c "import nltk; \
nltk.download('punkt', download_dir='/usr/share/nltk_data', quiet=True); \
nltk.download('stopwords', download_dir='/usr/share/nltk_data', quiet=True); \
nltk.download('vader_lexicon', download_dir='/usr/share/nltk_data', quiet=True)"

# Create non-root user for production hardening
RUN groupadd --system appuser && useradd --system --gid appuser --create-home appuser

# Copy application source tree
COPY app.py ./
COPY src ./src
COPY models ./models

RUN mkdir -p /app/models && chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]