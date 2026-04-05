# ──────────────────────────────────────────────────────────────────────────────
# Base: PyTorch 2.3.1 + CUDA 12.1 (matches setup_gpu.sh / requirements)
# ──────────────────────────────────────────────────────────────────────────────
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# System dependencies: ffmpeg (for ffprobe rotation fix), git (mim needs it)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ──────────────────────────────────────────────────────────────────────────────
# Python dependencies — install order matters for OpenMMLab
# ──────────────────────────────────────────────────────────────────────────────

# 1. Packaging tools
RUN pip install --no-cache-dir -U pip wheel "setuptools<70"

# 2. OpenMMLab stack (torch already in base image)
RUN pip install --no-cache-dir openmim
RUN python -m mim install "mmengine>=0.10.0"
RUN python -m mim install "mmcv>=2.1.0"
RUN python -m mim install "mmdet>=3.1.0"

# 3. Remaining runtime deps (including xtcocotools binary wheel)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Queue-service-specific deps (not in requirements.txt)
RUN pip install --no-cache-dir aio-pika

# 5. Force numpy<2.0 as the VERY LAST pip step.
#    Every prior install (mim, mmcv, requirements.txt) can silently pull in
#    numpy 2.x. The xtcocotools prebuilt wheel was compiled against numpy 1.x
#    (dtype struct = 96 bytes); numpy 2.x changed it to 88 bytes, causing:
#    "ValueError: numpy.dtype size changed, may indicate binary incompatibility"
#    Nothing runs after this, so numpy cannot be upgraded again.
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0"

# ──────────────────────────────────────────────────────────────────────────────
# Application code
# ──────────────────────────────────────────────────────────────────────────────
COPY . .

# Make mmpose-main importable as a top-level package (vendored source)
ENV PYTHONPATH="/app/mmpose-main"

# Unbuffered output so logs stream to docker logs immediately
ENV PYTHONUNBUFFERED=1

CMD ["python", "analysis_queue_service.py"]
