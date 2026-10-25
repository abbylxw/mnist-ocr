FROM python:3.13-slim-bookworm

# Supply the same uv version used by the development environment.
COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /usr/local/bin/uv

WORKDIR /usr/local/app

# Use the image's Python and the project environment for all runtime commands.
ENV UV_PYTHON_DOWNLOADS=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/usr/local/app/.venv/bin:$PATH"

# OpenMP runtime for CPU numerical libraries; discard apt's package lists.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Keep dependency installation cached when only application code changes.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-default-groups --no-cache

# Include everything needed for inference, including the default checkpoint.
COPY src ./src
COPY models/checkpoint.pt ./models/checkpoint.pt

# Inference only needs read access to the application and mounted inputs.
RUN useradd --uid 10001 --user-group --no-create-home app
USER app

# docker run arguments become arguments to the existing CLI.
ENTRYPOINT ["python", "-m", "src.cli"]
