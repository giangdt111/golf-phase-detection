#!/usr/bin/env python3
"""
Entrypoint for single-view and dual-view golf swing inference.
"""
import argparse
import json
import os
from typing import Dict, List, Optional

from golf_swing.multiview import (
    build_combined_output,
    build_overlay_payload,
    build_phase_pairs,
    build_sync,
)
from golf_swing.overlay import render_overlay
from golf_swing.pipeline import SwingInferenceService
from golf_swing.segmentation import init_seg_model


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


def _env_float(name: str):
    val = os.getenv(name)
    if val in (None, ""):
        return None
    try:
        return float(val)
    except ValueError:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Infer golf swing metrics from face-on video, with optional down-the-line fusion."
    )
    parser.add_argument(
        "--video-face-on",
        default=os.getenv("VIDEO_FACE_ON_PATH") or os.getenv("VIDEO_PATH"),
        help="Path to required face-on input video (env: VIDEO_FACE_ON_PATH or VIDEO_PATH).",
    )
    parser.add_argument(
        "--video-down-the-line",
        default=os.getenv("VIDEO_DOWN_THE_LINE_PATH"),
        help="Path to optional down-the-line input video (env: VIDEO_DOWN_THE_LINE_PATH).",
    )
    parser.add_argument(
        "--video",
        dest="legacy_video",
        default=None,
        help=argparse.SUPPRESS,
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
    parser.add_argument(
        "--height-mm",
        type=float,
        default=_env_float("HEIGHT_MM"),
        help="Required player height in millimeters (env: HEIGHT_MM).",
    )
    parser.add_argument("--out", default=None, help="Optional output JSON path for the combined result.")
    parser.add_argument(
        "--overlay-score-thr",
        type=float,
        default=float(os.getenv("OVERLAY_SCORE_THR", "0.2")),
        help="Keypoint score threshold for overlay (env: OVERLAY_SCORE_THR).",
    )
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


def _resolve_video_path(path: Optional[str]) -> Optional[str]:
    if path in (None, ""):
        return None
    return os.path.abspath(path)


def _ensure_video_path(path: Optional[str], label: str) -> str:
    if not path:
        raise SystemExit(f"{label} video is required.")
    if not os.path.exists(path):
        raise SystemExit(f"{label} video not found: {path}")
    return path


def _configure_devices(args: argparse.Namespace) -> None:
    if args.device is None:
        try:
            import torch

            if torch.cuda.is_available():
                args.device = "cuda:0"
        except Exception:
            pass
    elif args.device.startswith("cuda"):
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
    elif args.seg_device.startswith("cuda"):
        try:
            import torch

            if not torch.cuda.is_available():
                print("[WARN] Seg device CUDA requested but not available. Using cpu.")
                args.seg_device = "cpu"
        except Exception:
            print("[WARN] Seg device CUDA requested but torch is missing/unusable. Using cpu.")
            args.seg_device = "cpu"


def _dump_json(path: str, payload: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _phase_events_from_pairs(phase_pairs: List[Dict], view_key: str) -> List[Dict]:
    events = []
    for pair in phase_pairs:
        data = pair.get(view_key, {})
        events.append(
            {
                "name": pair.get("phase"),
                "frame": int(data.get("frame", 0)),
                "t": float(data.get("t", 0.0)),
            }
        )
    return events


def _phase_image_name(frame_id: int, phase_num: int, phase_name: str, t_value: float) -> str:
    if frame_id <= 0 and abs(t_value) <= 1e-9:
        return ""
    return f"{frame_id:06d}_P{phase_num}_{phase_name.replace(' ', '_')}_{t_value:.3f}.jpg"


def _attach_output_paths(combined_result: Dict, has_dtl: bool) -> None:
    for pair in combined_result.get("phase_frames", []) or []:
        idx = int(pair.get("phase_index", 0))
        name = pair.get("phase", f"P{idx}")
        face_info = pair.get("face_on", {})
        face_image = _phase_image_name(int(face_info.get("frame", 0)), idx, name, float(face_info.get("t", 0.0)))
        face_info["image"] = os.path.join("face_on", "phase_frames", face_image) if face_image else ""

        dtl_info = pair.get("down_the_line", {})
        dtl_image = _phase_image_name(int(dtl_info.get("frame", 0)), idx, name, float(dtl_info.get("t", 0.0)))
        dtl_info["image"] = os.path.join("down_the_line", "phase_frames", dtl_image) if has_dtl and dtl_image else ""


def _run_view(
    service: SwingInferenceService,
    args: argparse.Namespace,
    video_path: str,
    output_dir: str,
) -> Dict:
    return service.run(
        video_path=video_path,
        pose2d=args.pose2d,
        pose2d_weights=args.pose2d_weights,
        det_model=args.det_model,
        det_weights=args.det_weights,
        device=args.device,
        stride=args.stride,
        max_frames=args.max_frames,
        height_mm=args.height_mm,
        swing_direction=args.swing_direction,
        seg_model_path=None if args.seg_disable else args.seg_model,
        seg_imgsz=args.seg_imgsz,
        seg_conf=args.seg_conf,
        seg_iou=args.seg_iou,
        seg_device=args.seg_device,
        person_det_model=args.person_det_model,
        person_det_conf=args.person_det_conf,
        person_det_iou=args.person_det_iou,
        person_det_imgsz=args.person_det_imgsz,
        force_yolo_person=args.force_yolo_person,
        debug_p9=True,
        debug_p9_path=os.path.join(output_dir, "phase_debug.csv"),
    )


def main() -> None:
    _load_env_file()
    args = parse_args()
    if not args.video_face_on and args.legacy_video:
        args.video_face_on = args.legacy_video

    face_video_path = _ensure_video_path(_resolve_video_path(args.video_face_on), "Face-on")
    dtl_video_path = _resolve_video_path(args.video_down_the_line)
    if dtl_video_path:
        dtl_video_path = _ensure_video_path(dtl_video_path, "Down-the-line")

    if args.height_mm is None or float(args.height_mm) <= 0.0:
        raise SystemExit("HEIGHT_MM is required and must be > 0.")

    _configure_devices(args)

    face_base = os.path.splitext(os.path.basename(face_video_path))[0]
    dtl_base = os.path.splitext(os.path.basename(dtl_video_path))[0] if dtl_video_path else ""
    mode = "dual_view" if dtl_video_path else "face_on_only"

    if args.out:
        combined_json_path = os.path.abspath(args.out)
        output_dir = os.path.dirname(combined_json_path)
    else:
        output_root = os.path.abspath(os.getenv("OUTPUT_ROOT", "output"))
        session_name = face_base if not dtl_video_path else f"{face_base}__{dtl_base}"
        output_dir = os.path.join(output_root, session_name)
        combined_json_path = os.path.join(output_dir, "swing_result.json")

    face_dir = os.path.join(output_dir, "face_on")
    dtl_dir = os.path.join(output_dir, "down_the_line")
    os.makedirs(face_dir, exist_ok=True)
    if dtl_video_path:
        os.makedirs(dtl_dir, exist_ok=True)

    service = SwingInferenceService()
    face_raw = _run_view(service, args, face_video_path, face_dir)
    face_raw_json_path = os.path.join(face_dir, "raw_result.json")
    _dump_json(face_raw_json_path, face_raw)

    dtl_raw = None
    dtl_raw_json_path = ""
    if dtl_video_path:
        dtl_raw = _run_view(service, args, dtl_video_path, dtl_dir)
        dtl_raw["events"] = []
        dtl_raw["events_raw"] = None
        dtl_raw_json_path = os.path.join(dtl_dir, "raw_result.json")
        _dump_json(dtl_raw_json_path, dtl_raw)

    sync = build_sync(face_raw, dtl_raw)
    combined_result = build_combined_output(mode, float(args.height_mm), face_raw, dtl_raw, sync)
    phase_pairs = build_phase_pairs(face_raw, dtl_raw, sync)
    combined_result["phase_frames"] = phase_pairs

    face_events = _phase_events_from_pairs(phase_pairs, "face_on")
    dtl_events = _phase_events_from_pairs(phase_pairs, "down_the_line") if dtl_raw else []

    face_overlay_payload = build_overlay_payload("face_on", face_raw, combined_result, sync, face_events)
    face_overlay_payload_path = os.path.join(face_dir, "overlay_payload.json")
    _dump_json(face_overlay_payload_path, face_overlay_payload)

    dtl_overlay_payload_path = ""
    if dtl_raw:
        dtl_overlay_payload = build_overlay_payload("down_the_line", dtl_raw, combined_result, sync, dtl_events)
        dtl_overlay_payload_path = os.path.join(dtl_dir, "overlay_payload.json")
        _dump_json(dtl_overlay_payload_path, dtl_overlay_payload)

    seg_model = None if args.seg_disable else init_seg_model(args.seg_model)

    face_overlay_video = os.path.join(face_dir, "swing_overlay_slow4x.mp4")
    face_phase_frames_dir = os.path.join(face_dir, "phase_frames")
    render_overlay(
        face_video_path,
        face_overlay_payload_path,
        face_overlay_video,
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
        phase_frames_out=face_phase_frames_dir,
    )
    print(f"Wrote {face_overlay_video}")

    dtl_overlay_video = ""
    dtl_phase_frames_dir = ""
    if dtl_raw and dtl_video_path:
        dtl_overlay_video = os.path.join(dtl_dir, "swing_overlay_slow4x.mp4")
        dtl_phase_frames_dir = os.path.join(dtl_dir, "phase_frames")
        render_overlay(
            dtl_video_path,
            dtl_overlay_payload_path,
            dtl_overlay_video,
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
            phase_frames_out=dtl_phase_frames_dir,
        )
        print(f"Wrote {dtl_overlay_video}")

    combined_result["views"]["face_on"].update(
        {
            "raw_result_json": os.path.join("face_on", "raw_result.json"),
            "overlay_payload_json": os.path.join("face_on", "overlay_payload.json"),
            "overlay_video": os.path.join("face_on", "swing_overlay_slow4x.mp4"),
            "phase_frames_dir": os.path.join("face_on", "phase_frames"),
        }
    )
    combined_result["views"]["down_the_line"].update(
        {
            "raw_result_json": os.path.join("down_the_line", "raw_result.json") if dtl_raw else "",
            "overlay_payload_json": os.path.join("down_the_line", "overlay_payload.json") if dtl_raw else "",
            "overlay_video": os.path.join("down_the_line", "swing_overlay_slow4x.mp4") if dtl_raw else "",
            "phase_frames_dir": os.path.join("down_the_line", "phase_frames") if dtl_raw else "",
        }
    )
    _attach_output_paths(combined_result, has_dtl=bool(dtl_raw))
    _dump_json(combined_json_path, combined_result)
    print(f"Wrote {combined_json_path}")


if __name__ == "__main__":
    main()
