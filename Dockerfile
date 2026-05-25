# Two stages so the runtime image does not carry build toolchains.
FROM python:3.11-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml README.md ./
COPY src ./src

RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --upgrade pip && \
    /opt/venv/bin/pip install ".[ml,llm,docs]"

# ─────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# ffmpeg is needed for audio assembly; libgomp for the sklearn/torch runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgomp1 curl && \
    rm -rf /var/lib/apt/lists/*

# Never run as root.
RUN useradd --create-home --uid 10001 voicebrief
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/home/voicebrief/.cache/huggingface

WORKDIR /app
COPY --chown=voicebrief:voicebrief src ./src
COPY --chown=voicebrief:voicebrief migrations ./migrations
COPY --chown=voicebrief:voicebrief alembic.ini config ./
COPY --chown=voicebrief:voicebrief config ./config

USER voicebrief
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "voicebrief.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
