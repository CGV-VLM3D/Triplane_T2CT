# ctgen submission image — report2ct_wan (thin: no weights baked in).
#
# Build from the REPO ROOT so `src/` is in context:
#   docker build -f vlm3d_challenge_eval/Dockerfile -t vlm3d_challenge_eval_test:latest .
#
# Weights ship separately via `forithmus submit --weights weights.zip` (built by
# stage_weights.sh) and are mounted at /weights, which is where HF_HUB_CACHE and the checkpoint
# path point — so nothing needs symlinking at runtime.
#
# CUDA 12.8 runtime base: torch 2.7.0+cu128 needs those libs, and a plain python: image silently
# reports torch.cuda.is_available() == False (README §9). python3.11 comes from deadsnakes to
# match the interpreter the pipeline was validated on (Ubuntu 22.04 ships 3.10).
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        software-properties-common ca-certificates && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3.11-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3.11 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1

COPY vlm3d_challenge_eval/requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir -r /opt/app/requirements.txt && \
    pip install --no-cache-dir --no-deps monai==1.4.0

# Our package (720 KB, python only) plus the one vendored file it reaches outside it:
# src/baselines/rflow.py re-exports RFlowScheduler from the MAISI bundle and looks for it at this
# absolute path, so mirror it rather than patching read-only third_party code.
COPY src /opt/app/src
COPY third_party/maisi_bundle/scripts/rectified_flow.py \
     /workspace/third_party/maisi_bundle/scripts/rectified_flow.py
COPY vlm3d_challenge_eval/process.py vlm3d_challenge_eval/report_parse.py /opt/app/
COPY vlm3d_challenge_eval/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# No network egress at runtime — every weight comes from the /weights mount.
# /weights is mounted READ-ONLY (README §6), so only HF_HUB_CACHE (pure reads) points there;
# HF_HOME stays on writable scratch because huggingface_hub still wants to create lock/scratch
# dirs under it even offline.
ENV HF_HUB_CACHE=/weights/hf_cache/hub \
    HF_HOME=/tmp/hf \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    HF_HUB_DISABLE_TELEMETRY=1

# The validator rejects an image whose effective USER is root (README §9).
RUN useradd --create-home --uid 1000 user && \
    mkdir -p /input /output && \
    chown -R user:user /opt/app /input /output
USER user
WORKDIR /opt/app

ENTRYPOINT ["/entrypoint.sh"]
