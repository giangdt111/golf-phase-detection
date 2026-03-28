# Setup On Ubuntu GPU

This repo runs inference from `app.py` and loads the vendored `mmpose-main/` source directly from the repository.

The main setup risk is environment mismatch:
- using `pip` from a different Python than `python`
- using a too-new PyTorch/CUDA combo that has no compatible `mmcv` wheel
- trying to install OpenMMLab packages with plain `pip` instead of `openmim`

## Recommended environment

Use a dedicated Conda env and keep all install commands tied to that env's Python.

```bash
cd /home/savvycom/AI_team/giangdt/ai_golf_swing

conda create -n golf_swing python=3.10 -y
conda activate golf_swing

which python
python -V
python -m pip -V
```

Expected outcome:
- `python` points to `.../anaconda3/envs/golf_swing/bin/python`
- `python -m pip -V` also points to the same env

Do not use bare `pip` or bare `mim` if they resolve to `~/.local/bin/...`.
Always use:
- `python -m pip ...`
- `python -m mim ...`

## Install steps

### 1. Base packaging tools

```bash
python -m pip install -U pip wheel "setuptools<70"
```

`setuptools<70` avoids the `pkg_resources` issue seen when building package metadata for older OpenMMLab dependencies.

### 2. Install PyTorch first

Use a PyTorch/CUDA combination that OpenMMLab has wheels for.

Recommended:

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

### 3. Install OpenMMLab runtime with `openmim`

```bash
python -m pip install openmim
python -m mim install "mmengine>=0.6.0,<1.0.0"
python -m mim install "mmcv>=2.0.0rc4,<2.1.0"
python -m mim install "mmdet>=3.0.0,<3.1.0"
```

Notes:
- `mmdet` is optional for this project. The app can run with YOLO person detection by passing `--force-yolo-person`.
- `mmcv` is the sensitive package. If it does not find a prebuilt wheel for your PyTorch/CUDA combo, it will try to build from source and often fail or take a long time.

### 4. Install the remaining Python packages

```bash
python -m pip install -r requirements.txt
```

If you want to avoid reinstalling `torch/mmcv/mmengine/mmdet`, you can also install only the runtime packages directly:

```bash
python -m pip install ultralytics numpy opencv-python pillow scipy matplotlib json_tricks munkres xtcocotools rich
```

## Repo weights

This repository already includes the weights used by the default config:
- `model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth`
- `model/golf_segment.pt`
- `model/yolov8n_person_detection.pt`

No extra download is required for the default local run.

## Configure `.env`

```bash
cp .env.example .env
```

Typical GPU config:

```env
DEVICE=cuda:0
SEG_DEVICE=cuda:0
VIDEO_PATH=video/A001_03220730_C001.mov
OUTPUT_ROOT=output

POSE2D_CONFIG=mmpose-main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py
POSE2D_WEIGHTS=model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
SEG_MODEL=model/golf_segment.pt
PERSON_DET_MODEL=model/yolov8n_person_detection.pt
```

## Run

Recommended command on Ubuntu GPU:

```bash
python app.py \
  --video video/A001_03220730_C001.mov \
  --device cuda:0 \
  --seg-device cuda:0 \
  --force-yolo-person
```

Outputs are written to:
- `output/<video_name>/swing_result.json`
- `output/<video_name>/swing_overlay_slow4x.mp4`
- `output/<video_name>/phase_frames/`

## Quick validation

```bash
python -c "import torch, mmengine, mmcv, ultralytics; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('mmengine', mmengine.__version__); print('mmcv', mmcv.__version__); print('ultralytics', ultralytics.__version__)"
```

Then run a short inference:

```bash
python app.py --video video/A001_03220730_C001.mov --device cuda:0 --seg-device cuda:0 --force-yolo-person --max-frames 30
```

## Troubleshooting

### `pip` and `python` point to different environments

Symptoms:
- `python -V` shows one version, but `pip -V` shows another
- packages install successfully but imports still fail

Fix:

```bash
which python
python -V
python -m pip -V
which pip
which mim
```

Use only:
- `python -m pip ...`
- `python -m mim ...`

### `mim install mmcv` tries to build from source

This usually means your current PyTorch/CUDA combo has no matching prebuilt `mmcv` wheel.

Fix:
- recreate the env
- install a supported PyTorch build first, for example `torch==2.3.1` with `cu121`
- rerun `python -m mim install ...`

### `ModuleNotFoundError: pkg_resources`

Fix:

```bash
python -m pip install "setuptools<70"
```

### `ImportError: libcudnn.so.9`

This usually means you are importing `torch` from a different Python environment than the one you intended.

Fix:
- check `which python`
- check `python -m pip -V`
- avoid `~/.local/bin/pip` and `~/.local/bin/mim`

## Notes on `requirements.txt`

`requirements.txt` is provided as a package list for this repo, but the most reliable order on Ubuntu GPU is still:
1. create env
2. install PyTorch
3. install `openmim`
4. install `mmengine/mmcv/mmdet` with `python -m mim`
5. install the remaining packages
