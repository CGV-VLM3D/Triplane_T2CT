# VLM3D 2026 Task 4 submission docker — Phase A stub.
# Phase D will graduate this to a production container that bakes in `ours/final/best.ckpt`.
# Contract follows third_party/vlm3d_dockers/ctgen_example_docker/Dockerfile.

FROM python:3.12-slim

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
RUN mkdir -p /opt/app /input /output && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app
ENV PATH="/home/user/.local/bin:${PATH}"

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --user -U pip && python -m pip install --user --no-cache-dir -r requirements.txt

COPY --chown=user:user process.py /opt/app/

ENTRYPOINT ["python", "/opt/app/process.py"]
