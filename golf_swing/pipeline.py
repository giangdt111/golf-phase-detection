from typing import Dict, List, Optional, Tuple
import os
import json
import torch
import numpy as np
import cv2

from .constants import COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES
from .events import FramePose, extract_keypoints, infer_swing_direction
from .events import detect_events_rule, detect_events_rule9
from .pose import import_mmpose
from .detection import (
    init_det_inferencer,
    init_person_yolo,
    select_person_bbox_yolo,
    select_person_bbox,
    expand_bbox,
)
from .segmentation import init_seg_model, segment_frame_features
from .utils import temporary_cwd, load_video_meta


class SwingInferenceService:
    """Encapsulates the full swing inference pipeline."""

    def __init__(self, mmpose_root: Optional[str] = None) -> None:
        self.mmpose_root = mmpose_root or os.path.join(os.path.dirname(__file__), "..", "mmpose-main")
        self.MMPoseInferencer = import_mmpose()

    def _build_pose_inferencer(
        self,
        pose2d: str,
        pose2d_weights: str,
        det_model: Optional[str],
        det_weights: Optional[str],
        device: Optional[str],
    ):
        det_model_pose = det_model
        det_weights_pose = det_weights
        detector = init_det_inferencer(det_model, det_weights, device)
        if detector is not None:
            det_model_pose = "whole_image"
            det_weights_pose = None
        with temporary_cwd(self.mmpose_root):
            try:
                try:
                    import mmdet  # noqa: F401
                except Exception:
                    pass
                inferencer = self.MMPoseInferencer(
                    pose2d=pose2d,
                    pose2d_weights=pose2d_weights,
                    det_model=det_model_pose,
                    det_weights=det_weights_pose,
                    device=device,
                )
            except RuntimeError as exc:
                if "MMDetection" not in str(exc):
                    raise
                print(
                    "MMDetection not available; using whole_image pose only. Install mmdet for better cropping."
                )
                inferencer = self.MMPoseInferencer(
                    pose2d=pose2d,
                    pose2d_weights=pose2d_weights,
                    det_model="whole_image",
                    det_weights=None,
                    device=device,
                )
        return inferencer, detector

    def run(
        self,
        video_path: str,
        pose2d: str,
        pose2d_weights: str,
        det_model: Optional[str],
        det_weights: Optional[str],
        device: Optional[str],
        stride: int,
        max_frames: Optional[int],
        event_mode: str,
        swing_direction: Optional[str],
        seg_model_path: Optional[str],
        seg_imgsz: int,
        seg_conf: float,
        seg_iou: float,
        seg_device: str,
        person_det_model: Optional[str],
        person_det_conf: float,
        person_det_iou: float,
        person_det_imgsz: int,
        force_yolo_person: bool,
    ) -> Dict:
        video_path = os.path.abspath(video_path)
        pose2d = os.path.abspath(pose2d) if pose2d else pose2d
        pose2d_weights = os.path.abspath(pose2d_weights) if pose2d_weights else pose2d_weights
        if det_model and det_model not in ("whole_image", "whole-image"):
            det_model = os.path.abspath(det_model)
        if det_weights:
            det_weights = os.path.abspath(det_weights)

        cap, meta = load_video_meta(video_path)
        fps = meta["fps"]

        detector = None
        inferencer = None
        if not force_yolo_person:
            inferencer, detector = self._build_pose_inferencer(
                pose2d=pose2d,
                pose2d_weights=pose2d_weights,
                det_model=det_model,
                det_weights=det_weights,
                device=device,
            )
        else:
            inferencer, detector = self._build_pose_inferencer(
                pose2d=pose2d,
                pose2d_weights=pose2d_weights,
                det_model="whole_image",
                det_weights=None,
                device=device,
            )
        yolo_person = None
        if detector is None:
            yolo_person = init_person_yolo(person_det_model, device)

        seg_model = init_seg_model(seg_model_path) if seg_model_path else None

        print("[INFO] Models:")
        print(f"  Pose2D: {pose2d} | weights: {pose2d_weights}")
        if detector is not None:
            print(f"  Detector: MMDetection ({det_model})")
        elif yolo_person is not None:
            print(f"  Detector: YOLOv8 person ({person_det_model})")
        else:
            print("  Detector: whole_image (no person detector)")
        if seg_model_path:
            print(f"  Segmentation: {seg_model_path}")
        else:
            print("  Segmentation: disabled")
        print(f"  Device pose/det: {device or 'auto'} | seg: {seg_device}")

        frames: List[FramePose] = []
        shaft_angles: List[Optional[float]] = []
        club_centers: List[Optional[Tuple[float, float]]] = []
        shaft_centers: List[Optional[Tuple[float, float]]] = []
        frame_id = 0
        kept = 0
        last_bbox = None

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_id % stride != 0:
                frame_id += 1
                continue
            if max_frames is not None and kept >= max_frames:
                break
            pose_frame = frame
            offset_x = 0
            offset_y = 0
            bbox = None
            if detector is not None:
                det_out = detector(frame, return_datasample=True).get("predictions", [])
                det_sample = det_out[0] if det_out else None
                bbox = select_person_bbox(det_sample)
            elif yolo_person is not None:
                bbox = select_person_bbox_yolo(
                    yolo_person,
                    frame,
                    conf=person_det_conf,
                    iou=person_det_iou,
                    imgsz=person_det_imgsz,
                )
            if bbox is None:
                bbox = last_bbox
            if bbox is not None:
                bbox = expand_bbox(bbox, frame.shape[1], frame.shape[0], scale=1.2)
                x1, y1, x2, y2 = bbox
                if x2 > x1 and y2 > y1:
                    pose_frame = frame[y1:y2, x1:x2]
                    offset_x, offset_y = x1, y1
                last_bbox = bbox

            result = inferencer(pose_frame, return_datasample=False, show=False)
            if hasattr(result, "__iter__") and not isinstance(result, dict):
                result = next(result)
            prediction = result.get("predictions", []) if isinstance(result, dict) else []
            if prediction:
                prediction = prediction[0]
            keypoints = extract_keypoints(prediction)
            if keypoints and (offset_x or offset_y):
                for kp in keypoints:
                    kp["x"] += float(offset_x)
                    kp["y"] += float(offset_y)
            frames.append(FramePose(frame_id=frame_id, t=frame_id / fps, keypoints=keypoints))
            if seg_model is not None:
                angle, center, shaft_center = segment_frame_features(
                    frame,
                    model=seg_model,
                    imgsz=seg_imgsz,
                    conf=seg_conf,
                    iou=seg_iou,
                    device=seg_device,
                    classes=[0, 1],
                )
                shaft_angles.append(angle)
                club_centers.append(center)
                shaft_centers.append(shaft_center)
            else:
                shaft_angles.append(None)
                club_centers.append(None)
                shaft_centers.append(None)
            frame_id += 1
            kept += 1

        cap.release()
        inferred_direction = swing_direction or infer_swing_direction(frames)
        if event_mode == "rule9":
            events = detect_events_rule9(
                frames,
                fps,
                inferred_direction,
                shaft_angles,
                club_centers,
                shaft_centers,
            )
        else:
            events = detect_events_rule(frames, fps)

        return {
            "video": {
                "path": video_path,
                "fps": fps,
                "width": meta["width"],
                "height": meta["height"],
                "frame_count": meta["frame_count"],
                "stride": stride,
            },
            "skeleton": {
                "format": "coco",
                "keypoint_names": COCO_KEYPOINT_NAMES,
                "edges": COCO_SKELETON_EDGES,
            },
            "frames": [
                {
                    "frame": f.frame_id,
                    "t": f.t,
                    "keypoints": f.keypoints,
                }
                for f in frames
            ],
            "events": events,
            "events_raw": None,
            "swing_direction": inferred_direction,
        }


__all__ = ["SwingInferenceService"]
