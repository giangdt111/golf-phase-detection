#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-golf_swing}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
TORCH_VERSION="${TORCH_VERSION:-2.3.1}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.18.1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Install Miniconda/Anaconda first." >&2
  exit 1
fi

# Load conda shell helpers in non-interactive shells.
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "Using existing conda env: $ENV_NAME"
else
  echo "Creating conda env: $ENV_NAME (python=$PYTHON_VERSION)"
  conda create -n "$ENV_NAME" "python=$PYTHON_VERSION" -y
fi

conda activate "$ENV_NAME"

echo "Python in use: $(which python)"
python -V
python -m pip -V

python -m pip install -U pip wheel "setuptools<70"
python -m pip install "torch==${TORCH_VERSION}" "torchvision==${TORCHVISION_VERSION}" --index-url "$TORCH_INDEX_URL"
python -m pip install openmim
python -m mim install "mmengine>=0.6.0,<1.0.0"
python -m mim install "mmcv>=2.0.0rc4,<2.1.0"
python -m mim install "mmdet>=3.0.0,<3.1.0"
python -m pip install -r requirements.txt

python -c "import torch, mmengine, mmcv, ultralytics; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('mmengine', mmengine.__version__); print('mmcv', mmcv.__version__); print('ultralytics', ultralytics.__version__)"

echo
echo "Setup completed."
echo "Activate env with: conda activate $ENV_NAME"
echo "Run example: python app.py --video video/A001_03220730_C001.mov --device cuda:0 --seg-device cuda:0 --force-yolo-person"
