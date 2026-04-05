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

# 3. Pin numpy<2.0 AFTER all mim installs — mim/mmcv pull numpy 2.x as a
#    dependency, overriding any earlier pin. The xtcocotools prebuilt wheel
#    was compiled against numpy 1.x (dtype struct = 96 bytes); numpy 2.x
#    changed it to 88 bytes, causing a runtime ABI crash. Force-reinstall
#    to downgrade numpy back to 1.x before installing xtcocotools.
RUN pip install --no-cache-dir --force-reinstall "numpy<2.0"

# 4. Remaining runtime deps (including xtcocotools) — numpy<2.0 is now locked
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Queue-service-specific deps (not in requirements.txt)
RUN pip install --no-cache-dir aio-pika

# ──────────────────────────────────────────────────────────────────────────────
# Application code
# ──────────────────────────────────────────────────────────────────────────────
COPY . .

# Make mmpose-main importable as a top-level package (vendored source)
ENV PYTHONPATH="/app/mmpose-main"

# Unbuffered output so logs stream to docker logs immediately
ENV PYTHONUNBUFFERED=1

CMD ["python", "analysis_queue_service.py"]
