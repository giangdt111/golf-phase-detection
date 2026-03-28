#!/usr/bin/env python3
"""
Entry point for AI Golf swing inference service.
Keeps CLI thin; heavy logic lives in golf_swing package.
"""
import argparse
import json
import os

from golf_swing.pipeline import SwingInferenceService
from golf_swing.segmentation import init_seg_model
from golf_swing.overlay import render_overlay


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        pass


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y", "on")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer 2D skeleton keypoints and swing events from a video."
    )
    parser.add_argument(
        "--video",
        default=os.getenv("VIDEO_PATH", "video_raw/IMG_6850_1.MOV"),
        help="Path to input video (env: VIDEO_PATH).",
    )
    parser.add_argument(
        "--pose2d",
        default=os.getenv(
            "POSE2D_CONFIG",
            "mmpose-main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
        ),
        help="MMPose pose2d config path or model alias.",
    )
    parser.add_argument(
        "--pose2d-weights",
        default=os.getenv("POSE2D_WEIGHTS", "model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth"),
        help="Checkpoint path for pose2d model.",
    )
    parser.add_argument(
        "--det-model",
        default=os.getenv("DET_MODEL", "whole_image"),
        help="Detector config or alias. Use 'whole_image' to skip detection.",
    )
    parser.add_argument(
        "--det-weights",
        default=os.getenv("DET_WEIGHTS", None),
        help="Detector checkpoint path (optional).",
    )
    parser.add_argument(
        "--device",
        default=os.getenv("DEVICE", None),
        help="Inference device, e.g. cuda:0 (env: DEVICE).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=int(os.getenv("STRIDE", "1")),
        help="Frame sampling stride (env: STRIDE).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=int(os.getenv("MAX_FRAMES", "0")) or None,
        help="Max frames to run (env: MAX_FRAMES). 0 means no limit.",
    )
    parser.add_argument("--out", default=None, help="Output JSON path.")
    parser.add_argument("--overlay-out", default=None, help="Output overlay video path.")
    parser.add_argument(
        "--overlay-score-thr",
        type=float,
        default=float(os.getenv("OVERLAY_SCORE_THR", "0.2")),
        help="Keypoint score threshold for overlay (env: OVERLAY_SCORE_THR).",
    )
    parser.add_argument("--phase-frames-out", default=None, help="Directory to write phase frames as images.")
    parser.add_argument(
        "--swing-direction",
        choices=["clockwise", "counterclockwise"],
        default=None,
        help="Swing direction in front view to resolve lead/trail arms.",
    )
    parser.add_argument(
        "--seg-model",
        default=os.getenv("SEG_MODEL", "golf_segment/best.pt"),
        help="YOLOv8 segmentation model path (shaft/head).",
    )
    parser.add_argument(
        "--seg-imgsz",
        type=int,
        default=int(os.getenv("SEG_IMGSZ", "960")),
        help="Segmentation inference image size (env: SEG_IMGSZ).",
    )
    parser.add_argument(
        "--seg-conf",
        type=float,
        default=float(os.getenv("SEG_CONF", "0.25")),
        help="Segmentation confidence threshold (env: SEG_CONF).",
    )
    parser.add_argument(
        "--seg-iou",
        type=float,
        default=float(os.getenv("SEG_IOU", "0.7")),
        help="Segmentation IoU threshold (env: SEG_IOU).",
    )
    parser.add_argument(
        "--seg-device",
        default=os.getenv("SEG_DEVICE", "cpu"),
        help="Segmentation device (e.g. 0 or cpu) (env: SEG_DEVICE).",
    )
    parser.add_argument(
        "--seg-alpha",
        type=float,
        default=float(os.getenv("SEG_ALPHA", "0.45")),
        help="Segmentation mask alpha (env: SEG_ALPHA).",
    )
    parser.add_argument(
        "--seg-disable",
        action="store_true",
        default=_env_bool("SEG_DISABLE", False),
        help="Disable segmentation overlay & features (env: SEG_DISABLE).",
    )
    parser.add_argument(
        "--det-debug",
        action="store_true",
        default=_env_bool("DET_DEBUG", False),
        help="Enable detector debug logging and bbox scores (env: DET_DEBUG).",
    )
    parser.add_argument(
        "--person-det-model",
        default=os.getenv("PERSON_DET_MODEL", "model/yolov8n.pt"),
        help="YOLOv8 person detector model path (env: PERSON_DET_MODEL).",
    )
    parser.add_argument(
        "--person-det-conf",
        type=float,
        default=float(os.getenv("PERSON_DET_CONF", "0.25")),
        help="YOLOv8 person detector confidence threshold (env: PERSON_DET_CONF).",
    )
    parser.add_argument(
        "--person-det-iou",
        type=float,
        default=float(os.getenv("PERSON_DET_IOU", "0.7")),
        help="YOLOv8 person detector IoU threshold (env: PERSON_DET_IOU).",
    )
    parser.add_argument(
        "--person-det-imgsz",
        type=int,
        default=int(os.getenv("PERSON_DET_IMGSZ", "640")),
        help="YOLOv8 person detector inference image size (env: PERSON_DET_IMGSZ).",
    )
    parser.add_argument(
        "--force-yolo-person",
        action="store_true",
        default=_env_bool("FORCE_PERSON_YOLO", False),
        help="Skip MMDetection detector and force YOLOv8 person (env: FORCE_PERSON_YOLO).",
    )
    return parser.parse_args()


def main() -> None:
    _load_env_file()
    args = parse_args()
    args.video = os.path.abspath(args.video)
    # Optional hard-coded override (set to a path string to force a specific video).
    input_video_path = None
    if input_video_path is not None:
        args.video = os.path.abspath(input_video_path)
    # Auto-pick GPU if available and user did not force a device.
    if args.device is None:
        try:
            import torch

            if torch.cuda.is_available():
                args.device = "cuda:0"
        except Exception:
            pass
    else:
        # Validate requested device; fallback to cpu if CUDA build not available.
        if args.device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[WARN] CUDA requested but not available/compiled. Falling back to cpu.")
                    args.device = "cpu"
            except Exception:
                print("[WARN] CUDA requested but torch is missing/unusable. Falling back to cpu.")
                args.device = "cpu"
    if args.seg_device == "cpu" and args.device and args.device.startswith("cuda"):
        args.seg_device = args.device
    else:
        if args.seg_device.startswith("cuda"):
            try:
                import torch

                if not torch.cuda.is_available():
                    print("[WARN] Seg device CUDA requested but not available. Using cpu.")
                    args.seg_device = "cpu"
            except Exception:
                print("[WARN] Seg device CUDA requested but torch is missing/unusable. Using cpu.")
                args.seg_device = "cpu"
    if not os.path.exists(args.video):
        raise SystemExit(f"Video not found: {args.video}")

    video_base = os.path.splitext(os.path.basename(args.video))[0]
    output_root = os.path.abspath(os.getenv("OUTPUT_ROOT", "output"))
    output_dir = os.path.join(output_root, video_base)
    os.makedirs(output_dir, exist_ok=True)
    if args.out is None:
        args.out = os.path.join(output_dir, "swing_result.json")
    else:
        args.out = os.path.abspath(args.out)
    if args.overlay_out is None:
        args.overlay_out = os.path.join(output_dir, "swing_overlay.mp4")
    else:
        args.overlay_out = os.path.abspath(args.overlay_out)
    service = SwingInferenceService()
    seg_model_path = None if args.seg_disable else args.seg_model

    result = service.run(
        video_path=args.video,
        pose2d=args.pose2d,
        pose2d_weights=args.pose2d_weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
        swing_direction=args.swing_direction,
        seg_model_path=seg_model_path,
        seg_imgsz=args.seg_imgsz,
        seg_conf=args.seg_conf,
        seg_iou=args.seg_iou,
        seg_device=args.seg_device,
        person_det_model=args.person_det_model,
        person_det_conf=args.person_det_conf,
        person_det_iou=args.person_det_iou,
        person_det_imgsz=args.person_det_imgsz,
        force_yolo_person=args.force_yolo_person,
        # Enable debug P9 by env flag
        debug_p9=True,
        debug_p9_path=os.path.join(output_dir, "phase_debug.csv"),
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {args.out}")

    seg_model = None if args.seg_disable else init_seg_model(args.seg_model)
    render_video_path = os.path.abspath(result.get("video", {}).get("path", args.video))
    slow_out = os.path.splitext(args.overlay_out)[0] + "_slow4x.mp4"
    phase_frames_dir = os.path.join(output_dir, "phase_frames")
    render_overlay(
        render_video_path,
        args.out,
        slow_out,
        args.overlay_score_thr,
        slow_factor=4,
        seg_model=seg_model,
        seg_imgsz=args.seg_imgsz,
        seg_conf=args.seg_conf,
        seg_iou=args.seg_iou,
        seg_device=args.seg_device,
        seg_alpha=args.seg_alpha,
        det_model=args.det_model,
        det_weights=args.det_weights,
        det_device=args.device,
        det_debug=args.det_debug,
        person_det_model=args.person_det_model,
        person_det_conf=args.person_det_conf,
        person_det_iou=args.person_det_iou,
        person_det_imgsz=args.person_det_imgsz,
        phase_frames_out=phase_frames_dir,
    )
    print(f"Wrote {slow_out}")
    print(f"Phase frames saved to {phase_frames_dir}")


if __name__ == "__main__":
    main()
