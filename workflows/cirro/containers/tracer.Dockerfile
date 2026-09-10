# syntax=docker/dockerfile:1.7

# TRACER itself remains byte-for-byte pinned to the released v0.1.1 image.
# procps supplies `ps`, which Nextflow requires to collect Cirro task metrics.
FROM ghcr.io/imlong4real/tracer@sha256:1ab4e0b2704fad56237fb4a9099c4ecbd1d83c71a52b3738386e5fafca90d282

RUN apt-get update \
    && apt-get install -y --no-install-recommends procps \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
CMD ["python", "-c", "import tracer; print(tracer.__version__)"]
