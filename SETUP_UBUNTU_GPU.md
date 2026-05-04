# Setup On Ubuntu / WSL GPU

This repo runs inference from `app.py` and loads the vendored `mmpose-main/`
source directly from the repository.

This guide reflects the environment that was verified to run successfully with:

- Ubuntu on WSL
- Python `3.10`
- `venv` instead of Conda
- CUDA-enabled PyTorch on `cuda:0`
- YOLO person detection via `--force-yolo-person`

The main setup risks for this repo are:

- using `pip` from a different Python than `python`
- installing an `mmcv` version that has no prebuilt wheel for your Torch/CUDA
- letting `numpy` get upgraded to `2.x`, which breaks `xtcocotools`
- letting `opencv-python` upgrade to a build that requires `numpy>=2`

## Recommended environment

Use a dedicated virtualenv and keep every install tied to that env's Python.

```bash
cd /home/dinhthang26/projects/golf-phase-detection

python3.10 -m venv .venv
source .venv/bin/activate

which python
python -V
python -m pip -V
```

Expected outcome:

- `python` points to `.../golf-phase-detection/.venv/bin/python`
- `python -m pip -V` points to the same `.venv`

Do not use bare `pip` or bare `mim` if they resolve outside the venv.
Always use:

- `python -m pip ...`
- `python -m mim ...`

## Verified install steps

### 1. Base packaging tools

```bash
python -m pip install -U pip wheel
python -m pip install "setuptools==69.5.1" "packaging==24.2"
```

Why:

- `setuptools==69.5.1` avoids the `pkg_resources` failure seen when `mmcv`
  falls back to build logic.
- `packaging==24.2` avoids drifting too far from what the OpenMMLab helper
  packages expect.

### 2. Install PyTorch first

Use a Torch/CUDA combination that has a compatible `mmcv` wheel.

Verified working combo:

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
```

Verify:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

Expected output is similar to:

```text
2.3.1+cu121 12.1 True
```

### 3. Install OpenMMLab runtime

Install `mmengine` with `mim`, but install `mmcv` explicitly from the OpenMMLab
wheel index.

```bash
python -m pip install openmim
python -m mim install "mmengine>=0.6.0,<1.0.0"
python -m pip install --no-cache-dir \
  "mmcv==2.2.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
```

Notes:

- The vendored `mmpose-main` accepts `mmcv<=2.2.0`.
- The old `mmcv<2.1.0` path tends to fall back to source build and fail.
- `mmdet` is not required for this repo if you run with `--force-yolo-person`.

Optional:

```bash
python -m pip uninstall -y mmdet
```

### 4. Install runtime packages with pinned NumPy/OpenCV

Install the repo runtime packages, but keep `numpy` and `opencv-python` pinned
 to versions that work with `xtcocotools`.

```bash
python -m pip install \
  "numpy==1.26.4" \
  "opencv-python==4.10.0.84" \
  pillow \
  scipy \
  matplotlib \
  json_tricks \
  munkres \
  rich \
  ultralytics \
  aio_pika
```

Then install `xtcocotools` without dependencies so it does not pull `numpy 2.x`
back into the environment:

```bash
python -m pip install --no-deps --force-reinstall "xtcocotools==1.14.3"
```

Why this matters:

- `numpy 2.x` caused `ValueError: numpy.dtype size changed`
- newer `opencv-python` builds required `numpy>=2`
- `xtcocotools` without `--no-deps` may re-upgrade `numpy`

### 5. Verify the environment

```bash
python -c "import torch, mmcv, mmengine; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available()); print('mmcv', mmcv.__version__); print('mmengine', mmengine.__version__)"
python -c "import numpy, cv2; print('numpy', numpy.__version__); print('cv2', cv2.__version__)"
python -c "from xtcocotools.coco import COCO; print('xtcocotools ok')"
```

Expected shape:

```text
torch 2.3.1+cu121 12.1 True
mmcv 2.2.0
mmengine 0.10.x
numpy 1.26.4
cv2 4.10.0.84
xtcocotools ok
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
HEIGHT_MM=1750

VIDEO_FACE_ON_PATH=video/face-on/1.mov
VIDEO_DOWN_THE_LINE_PATH=video/down-the-line/1.mov

POSE2D_CONFIG=mmpose-main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py
POSE2D_WEIGHTS=model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
SEG_MODEL=model/golf_segment.pt
PERSON_DET_MODEL=model/yolov8n_person_detection.pt
```

If the video is outside the repo, use an absolute path. On WSL, Windows files
usually live under `/mnt/c/...`.

Examples:

```env
VIDEO_FACE_ON_PATH=/mnt/c/Users/<user>/Videos/face_on.mov
VIDEO_DOWN_THE_LINE_PATH=/mnt/c/Users/<user>/Videos/down_the_line.mov
```

## Run

Recommended command:

```bash
python app.py --force-yolo-person
```

Useful variants:

```bash
python app.py --force-yolo-person --max-frames 30
python app.py --video-face-on /abs/path/face_on.mov --force-yolo-person
python app.py --video-face-on /abs/path/face_on.mov --video-down-the-line /abs/path/dtl.mov --force-yolo-person
```

Outputs are written to:

- `output/<session_name>/swing_result.json`
- `output/<session_name>/face_on/raw_result.json`
- `output/<session_name>/face_on/overlay_payload.json`
- `output/<session_name>/face_on/swing_overlay_slow4x.mp4`
- `output/<session_name>/face_on/phase_frames/`

Dual-view runs also write the same structure under `down_the_line/`.

## WSL notes

- If VS Code opens the repo in WSL, make sure the terminal is also using the
  same WSL Python and the same `.venv`.
- If `python app.py` says the video is not found, the path in `.env` is wrong.
- Prefer storing videos in the Linux filesystem or use explicit `/mnt/c/...`
  paths.

## Known warnings

These warnings were seen in a successful run and are not blockers by
themselves:

- `dataset_meta are not saved in the checkpoint's meta data, load via config`
- `Failed to search registry with scope "mmpose"...`
- `Failed to add LocalVisBackend, please provide the save_dir argument`
- `Ultralytics settings reset to default values`

## Troubleshooting

### `Face-on video not found`

The path in `.env` or CLI does not exist.

Check:

```bash
ls -l "$VIDEO_FACE_ON_PATH"
```

Or pass the file explicitly:

```bash
python app.py --video-face-on /abs/path/to/file.mov --force-yolo-person
```

### `ModuleNotFoundError: No module named 'mmcv'`

`mmcv` is missing or installed in a different environment.

Fix:

```bash
source .venv/bin/activate
python -c "import mmcv; print(mmcv.__version__, mmcv.__file__)"
```

If it is missing, reinstall:

```bash
python -m pip install --no-cache-dir \
  "mmcv==2.2.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
```

### `ValueError: numpy.dtype size changed`

This is an ABI mismatch between `numpy` and `xtcocotools`.

Fix:

```bash
python -m pip uninstall -y xtcocotools numpy opencv-python pycocotools mmdet
python -m pip install "setuptools==69.5.1" "packaging==24.2"
python -m pip install "numpy==1.26.4" "opencv-python==4.10.0.84"
python -m pip install --no-deps --force-reinstall "xtcocotools==1.14.3"
```

### `mim install mmcv` tries to build from source

That usually means the selected `mmcv` version has no prebuilt wheel for your
Torch/CUDA combination.

Use the verified wheel directly instead:

```bash
python -m pip install --no-cache-dir \
  "mmcv==2.2.0" \
  -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.3.0/index.html
```

### `python` works but `pip` installed packages are not found

This means `pip` and `python` are not using the same environment.

Check:

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

### CPU-only fallback

If WSL cannot access CUDA, use CPU:

```bash
python -m pip install torch==2.3.1 torchvision==0.18.1
```

Then set:

```env
DEVICE=cpu
SEG_DEVICE=cpu
```

And run:

```bash
python app.py --force-yolo-person
```
