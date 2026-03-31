# AI Golf Swing Inference Service

Refactored into a small entrypoint (`app.py`) plus reusable package `golf_swing/` for BE/FE integration.

Setup guide for Ubuntu GPU: [SETUP_UBUNTU_GPU.md](SETUP_UBUNTU_GPU.md)

## Quick start
```bash
cp .env.example .env   # optional: edit paths/devices
pip install -r requirements.local.txt  # plus ultralytics for YOLOv8
python app.py
```
Required env/input:
- `HEIGHT_MM`
- `VIDEO_FACE_ON_PATH`

Optional for dual-view mode:
- `VIDEO_DOWN_THE_LINE_PATH`
- Provide DTL trimmed to start near the same moment as face-on; DTL phases are mapped from face-on rather than detected independently.

Outputs are written to `$OUTPUT_ROOT/<session_name>/`:
- `swing_result.json` — combined single-view / dual-view result
- `face_on/` — raw result, overlay payload, slow overlay video, 9 phase frames
- `down_the_line/` — same structure, only when DTL video is provided

Key flags:
- `--seg-disable` to skip club segmentation
- `--device cuda:0` to force GPU
- `--force-yolo-person` to bypass MMDetection and use YOLOv8 person detector
- `--height-mm 1750` required for mm-based output
- `--video-face-on path/to/face_on.mov`
- `--video-down-the-line path/to/dtl.mov`

Environment overrides (all have CLI equivalents):
- DEVICE, SEG_DEVICE, VIDEO_FACE_ON_PATH, VIDEO_DOWN_THE_LINE_PATH, OUTPUT_ROOT
- POSE2D_CONFIG / POSE2D_WEIGHTS, DET_MODEL / DET_WEIGHTS, SEG_MODEL, PERSON_DET_MODEL
- STRIDE, MAX_FRAMES, HEIGHT_MM, OVERLAY_SCORE_THR
- SEG_IMGSZ, SEG_CONF, SEG_IOU, SEG_ALPHA, SEG_DISABLE, DET_DEBUG
- PERSON_DET_CONF, PERSON_DET_IOU, PERSON_DET_IMGSZ, FORCE_PERSON_YOLO
`.env` is auto-loaded on startup (keys are ignored if already present in the process env).

## Package layout (for service embedding)
- `golf_swing/pipeline.py` — `SwingInferenceService.run()` orchestrates detection → pose → events
- `golf_swing/overlay.py` — render overlay videos and phase frames
- `golf_swing/detection.py` — person detectors (RTMDet via MMDetection, fallback YOLOv8)
- `golf_swing/segmentation.py` — club shaft/head segmentation (YOLOv8)
- `golf_swing/events.py` + `events_logic.py` — swing feature extraction & 9-phase event detector
- `golf_swing/utils.py` — small IO helpers

## Weights
Default paths assume weights are downloaded to `model/` and `golf_segment/best.pt`.

## Dependencies
- PyTorch
- MMEngine/MMCV/MMDetection (optional: better person bbox). Without it, YOLOv8 person model is used.
- MMPose
- ultralytics (YOLOv8) for segmentation and fallback person detector
- OpenCV, NumPy
