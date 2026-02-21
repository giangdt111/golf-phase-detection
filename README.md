# AI Golf Swing Inference Service

Refactored into a small entrypoint (`app.py`) plus reusable package `golf_swing/` for BE/FE integration.

## Quick start
```bash
cp .env.example .env   # optional: edit paths/devices
pip install -r requirements.local.txt  # plus ultralytics for YOLOv8
python app.py --video video_raw/IMG_6850_1.MOV
```
Outputs are written to `$OUTPUT_ROOT/<video_name>/` (default `output/<video_name>/`):
- `swing_result.json` — all frames (COCO keypoints) + detected swing events
- `swing_overlay_slow4x.mp4` — overlay video at 4× slow motion
- `phase_frames/` — stills for each detected phase

Key flags:
- `--event-mode [rule|rule9]` (default rule9 uses club segmentation cues)
- `--seg-disable` to skip club segmentation
- `--device cuda:0` to force GPU
- `--force-yolo-person` to bypass MMDetection and use YOLOv8 person detector

Environment overrides (all have CLI equivalents):
- DEVICE, SEG_DEVICE, VIDEO_PATH, OUTPUT_ROOT
- POSE2D_CONFIG / POSE2D_WEIGHTS, DET_MODEL / DET_WEIGHTS, SEG_MODEL, PERSON_DET_MODEL
- STRIDE, MAX_FRAMES, EVENT_MODE, OVERLAY_SCORE_THR
- SEG_IMGSZ, SEG_CONF, SEG_IOU, SEG_ALPHA, SEG_DISABLE, DET_DEBUG
- PERSON_DET_CONF, PERSON_DET_IOU, PERSON_DET_IMGSZ, FORCE_PERSON_YOLO
`.env` is auto-loaded on startup (keys are ignored if already present in the process env).

## Package layout (for service embedding)
- `golf_swing/pipeline.py` — `SwingInferenceService.run()` orchestrates detection → pose → events
- `golf_swing/overlay.py` — render overlay videos and phase frames
- `golf_swing/detection.py` — person detectors (RTMDet via MMDetection, fallback YOLOv8)
- `golf_swing/segmentation.py` — club shaft/head segmentation (YOLOv8)
- `golf_swing/events.py` + `events_logic.py` — swing feature extraction & rule-based event detectors
- `golf_swing/utils.py` — small IO helpers

## Weights
Default paths assume weights are downloaded to `model/` and `golf_segment/best.pt`.
Original research weights (SwingNet, classifier) remain in the repo for reference.

## Dependencies
- PyTorch
- MMEngine/MMCV/MMDetection (optional: better person bbox). Without it, YOLOv8 person model is used.
- MMPose
- ultralytics (YOLOv8) for segmentation and fallback person detector
- OpenCV, NumPy

