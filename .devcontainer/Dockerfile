FROM ghcr.io/astral-sh/uv:0.13.5 AS uv
FROM mcr.microsoft.com/devcontainers/python:3.13-bookworm

COPY --from=uv /uv /uvx /usr/local/bin/

# Yarn is unused here; its inherited apt repository has an obsolete signing key.
RUN rm -f /etc/apt/sources.list.d/yarn.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
