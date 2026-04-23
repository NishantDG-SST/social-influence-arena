FROM python:3.11-slim

WORKDIR /app/env

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir uv && \
    uv pip install --system --no-cache -e .

ENV PYTHONUNBUFFERED=1
ENV ARENA_DEFAULT_TASK=resist_pressure
ENV ARENA_DIFFICULTY=1
ENV ARENA_PORT=8000
ENV ENABLE_WEB_INTERFACE=true

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "social_influence_env.server.app"]
