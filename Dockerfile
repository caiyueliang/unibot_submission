# syntax=docker/dockerfile:1

ARG BASE_IMAGE=pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime
FROM ${BASE_IMAGE}

LABEL org.opencontainers.image.title="unibot_submission"
LABEL org.opencontainers.image.ref.name="twr.wair.ac.cn/taichu-studio/unibot_submission:1.0.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV APP_HOME=/app/unibot_submission
ENV UNIBOT_SERVER_PORT=8765
ENV UNIBOT_CONTROL_SPACE=joint
ENV LEROBOT_SRC=/opt/lerobot/src
ENV UNITREE_LEROBOT_SRC=/opt/unitree_lerobot
ENV TELEIMAGER_SRC=/opt/teleimager/src
ENV HF_HOME=/cache/huggingface
ENV TORCH_HOME=/cache/torch

WORKDIR ${APP_HOME}

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_DEFAULT_TIMEOUT=120
ENV PIP_RETRIES=10

COPY requirements.txt requirements-inference.txt ./
ARG INSTALL_INFERENCE_DEPS=false
RUN python -m pip install --no-cache-dir --retries ${PIP_RETRIES} --timeout ${PIP_DEFAULT_TIMEOUT} \
    -r requirements.txt \
    && if [ "${INSTALL_INFERENCE_DEPS}" = "true" ]; then \
        python -m pip install --no-cache-dir --retries ${PIP_RETRIES} --timeout ${PIP_DEFAULT_TIMEOUT} \
            -r requirements-inference.txt; \
    fi

COPY README.md README.zh.md client.py ./
COPY policy ./policy
COPY example ./example

EXPOSE 8765

ENTRYPOINT ["/bin/bash", "-lc", "export PYTHONPATH=\"${APP_HOME}:${LEROBOT_SRC}:${UNITREE_LEROBOT_SRC}:${TELEIMAGER_SRC}:${PYTHONPATH}\"; exec \"$@\"", "--"]
CMD ["python", "example/run_server.py"]
