FROM python:3.10.17-slim

# Define USER
ARG USER=seunguk
ARG UID=1000
ARG GID=1000

# Install cuda and cudnn
WORKDIR /usr/local/cuda
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/local/cuda-12.4 .
WORKDIR /usr/lib/x86_64-linux-gnu
COPY --from=nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04 /usr/lib/x86_64-linux-gnu/libcudnn* .

# Install UV
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Create non root user, add it to custom group and setup environment.
RUN groupadd --gid $GID $USER \
    && useradd --uid $UID --gid $GID -m $USER -d /home/${USER} --shell /usr/bin/bash

# Install required apt packages and clear cache afterwards.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    # For opencv
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install the project into `/app`
WORKDIR /data
WORKDIR /app

# Set env variables for uv
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY --chown=$USER:$USER . /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv pip install -e .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

RUN chown $UID:$GID -R /root
RUN chown $UID:$GID -R .venv
RUN chown $UID:$GID /data

USER $USER
ENV PATH="/app/.venv/bin:$PATH"
