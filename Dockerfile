FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev \
        libegl1 libgl1 libgles2 libosmesa6 \
        git build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/local/bin/python \
    && ln -sf /usr/bin/python3 /usr/local/bin/python3

WORKDIR /workspace/Humanoid-GPT

COPY pyproject.toml README.md LICENSE ./
COPY tracking/ tracking/
COPY scripts/ scripts/
COPY utils/ utils/
COPY gmr/ gmr/
COPY projects/ projects/
COPY storage/assets/ storage/assets/
COPY storage/configs/ storage/configs/
COPY storage/ckpts/ storage/ckpts/
COPY storage/test/ storage/test/

RUN pip install torch==2.8.0 \
    && pip install -e ".[cuda]"
