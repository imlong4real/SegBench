# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ARG PIXI_VERSION=0.80.0
ARG SEGGER_COMMIT=ca7bf1caaf9a177dd1f3f9051f020d2e7d3937ac
ENV DEBIAN_FRONTEND=noninteractive \
    PATH=/opt/segger/.pixi/envs/cuda121/bin:/usr/local/bin:${PATH} \
    PIXI_HOME=/opt/pixi \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git bzip2 procps \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL "https://github.com/prefix-dev/pixi/releases/download/v${PIXI_VERSION}/pixi-x86_64-unknown-linux-musl.tar.gz" \
       | tar -xz -C /usr/local/bin

RUN git clone https://github.com/dpeerlab/segger.git /opt/segger \
    && cd /opt/segger \
    && git checkout "${SEGGER_COMMIT}" \
    && pixi install --frozen --environment cuda121 \
    && pixi clean cache --yes \
    && mkdir -p /opt/container-manifest \
    && git rev-parse HEAD > /opt/container-manifest/segger-commit.txt \
    && cp pixi.lock /opt/container-manifest/segger-pixi.lock

WORKDIR /work
CMD ["segger", "--help"]
