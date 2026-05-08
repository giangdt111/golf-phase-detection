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
- `phase_metrics.xlsx` — workbook summary of overlay metrics by P1-P9 phase
- `face_on/` — raw result, overlay payload, slow overlay video, 9 phase frames
- `down_the_line/` — same structure, only when DTL video is provided

## Coordinate convention
All body-point displacements are referenced to the detected `Address` phase.

Current signed displacement convention:
- `x`: positive = move toward the lead side, negative = move toward the trail side
- `y`: positive = move upward, negative = move downward
- `z`: positive = move toward the ball side, negative = move away from the ball side

Notes:
- `x` and `y` are derived from the face-on view.
- `z` is derived from the down-the-line view when DTL video is available.
- In face-on-only mode, `z_mm` is `null` because depth cannot be measured from a single face-on view.
- `z` sign is inferred from the Address setup geometry in DTL: the horizontal image direction from the body core toward the grip is treated as the positive ball-side direction.
- Displacements are `null` before the detected `Address` frame because `Address` is the reference pose.

Where this appears in outputs:
- `face_on/raw_result.json`: `frames[*].body_points.{hip,chest,shoulder,grip}.dx`, `.dy`, `.dx_mm_est`, `.dy_mm_est`
- `swing_result.json`: `frames[*].translations.{hip,chest,grip}.x_mm`, `.y_mm`, `.z_mm`
- Legacy combined fields remain available at `frames[*].points.{hip,chest,grip}.dx_mm`, `.dy_mm`, `.dz_mm`
- `overlay_payload.json`: mirrored display values used by the overlay stats panel

Each output JSON also includes a `coordinate_system` block describing the active sign convention and the reference Address frame.

## Angle convention
Current segment rotation outputs are grouped under `frames[*].rotations.{hip,chest}`:
- `x_deg`: unsigned proxy for rotation around the `x` axis, measured from DTL segment angle change in the `Y-Z` plane relative to `Address`. This is `null` in face-on-only mode.
- `y_deg`: unsigned proxy for rotation around the vertical `y` axis, measured from face-on foreshortening relative to `Address`.
- `z_deg`: unsigned 2D proxy for rotation around the `z` axis, measured as the segment's visible tilt against the horizontal image axis.

Y-axis angle formula:
- `hip_y_angle` uses the horizontal width between left/right hip keypoints.
- `chest_y_angle` uses the horizontal width between left/right shoulder keypoints.
- At `Address`, the reference width is `L0`.
- For each frame, current horizontal width is `L`.
- `angle_y = arccos(clamp(L / L0, 0, 1))`.

These Y-axis angles are unsigned: they estimate how much the hip/chest has turned, but not yet whether it turned open or closed.

X-axis angle proxy:
- Requires DTL video.
- `hip.x_deg` uses the DTL angle change of segment `hip -> chest`.
- `chest.x_deg` uses the DTL angle change of segment `chest -> shoulder`, falling back to `hip -> chest` if shoulder is missing.
- The value is unsigned: it estimates how much the segment bends/rotates around X, not yet the positive/negative direction.

Legacy angle fields remain available:
- `hip_angle` / `chest_angle` map to `rotations.{hip,chest}.z_deg`
- `hip_y_angle` / `chest_y_angle` map to `rotations.{hip,chest}.y_deg`

Overlay videos render the same translation and rotation groups in the stats panel. A small coordinate-axis legend is also drawn on the video:
- `+X lead`: toward the lead side
- `+Y up`: upward
- `+Z ball`: toward the ball side

Key flags:
- `--seg-disable` to skip club segmentation
- `--render-disable` to skip overlay video + phase frame rendering
- `--device cuda:0` to force GPU
- `--force-yolo-person` to bypass MMDetection and use YOLOv8 person detector
- `--height-mm 1750` required for mm-based output
- `--video-face-on path/to/face_on.mov`
- `--video-down-the-line path/to/dtl.mov`

Environment overrides (all have CLI equivalents):
- DEVICE, SEG_DEVICE, VIDEO_FACE_ON_PATH, VIDEO_DOWN_THE_LINE_PATH, OUTPUT_ROOT
- POSE2D_CONFIG / POSE2D_WEIGHTS, DET_MODEL / DET_WEIGHTS, SEG_MODEL, PERSON_DET_MODEL
- STRIDE, MAX_FRAMES, HEIGHT_MM, OVERLAY_SCORE_THR
- SEG_IMGSZ, SEG_CONF, SEG_IOU, SEG_ALPHA, SEG_DISABLE, RENDER_OUTPUT, DET_DEBUG
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
