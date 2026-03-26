from typing import Any, Dict, List, Optional, Tuple
import os
import json
import math
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


def _model_cache_key(**kwargs) -> tuple:
    """Stable key for model cache from run/warmup kwargs (paths normalized)."""
    pose2d = kwargs.get("pose2d") or ""
    pose2d_weights = kwargs.get("pose2d_weights") or ""
    det_model = kwargs.get("det_model") or ""
    det_weights = kwargs.get("det_weights") or ""
    device = kwargs.get("device")
    person_det_model = kwargs.get("person_det_model")
    force_yolo_person = kwargs.get("force_yolo_person", False)
    seg_model_path = kwargs.get("seg_model_path")
    seg_device = kwargs.get("seg_device")
    return (
        os.path.abspath(pose2d) if pose2d else "",
        os.path.abspath(pose2d_weights) if pose2d_weights else "",
        os.path.abspath(det_model) if det_model and det_model not in ("whole_image", "whole-image") else (det_model or ""),
        os.path.abspath(det_weights) if det_weights else "",
        device,
        os.path.abspath(person_det_model) if person_det_model else None,
        force_yolo_person,
        os.path.abspath(seg_model_path) if seg_model_path else None,
        seg_device,
    )


class SwingInferenceService:
    """Encapsulates the full swing inference pipeline."""

    def __init__(self, mmpose_root: Optional[str] = None) -> None:
        self.mmpose_root = mmpose_root or os.path.join(os.path.dirname(__file__), "..", "mmpose-main")
        self.MMPoseInferencer = import_mmpose()
        self._cache_key: Optional[tuple] = None
        self._cached_inferencer = None
        self._cached_detector = None
        self._cached_yolo_person = None
        self._cached_seg_model = None

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

    def _get_or_build_models(
        self,
        pose2d: str,
        pose2d_weights: str,
        det_model: Optional[str],
        det_weights: Optional[str],
        device: Optional[str],
        person_det_model: Optional[str],
        force_yolo_person: bool,
        seg_model_path: Optional[str],
        seg_device: str,
    ) -> Tuple[Any, Optional[Any], Optional[Any], Optional[Any]]:
        """Return (inferencer, detector, yolo_person, seg_model), building and caching if needed."""
        pose2d = os.path.abspath(pose2d) if pose2d else pose2d
        pose2d_weights = os.path.abspath(pose2d_weights) if pose2d_weights else pose2d_weights
        if det_model and det_model not in ("whole_image", "whole-image"):
            det_model = os.path.abspath(det_model)
        if det_weights:
            det_weights = os.path.abspath(det_weights)
        if person_det_model:
            person_det_model = os.path.abspath(person_det_model)
        if seg_model_path:
            seg_model_path = os.path.abspath(seg_model_path)

        key = _model_cache_key(
            pose2d=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_weights=det_weights,
            device=device,
            person_det_model=person_det_model,
            force_yolo_person=force_yolo_person,
            seg_model_path=seg_model_path,
            seg_device=seg_device,
        )
        if self._cache_key == key and self._cached_inferencer is not None:
            return (
                self._cached_inferencer,
                self._cached_detector,
                self._cached_yolo_person,
                self._cached_seg_model,
            )

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

        self._cache_key = key
        self._cached_inferencer = inferencer
        self._cached_detector = detector
        self._cached_yolo_person = yolo_person
        self._cached_seg_model = seg_model
        print("[INFO] Models loaded:")
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
        return inferencer, detector, yolo_person, seg_model

    def warmup(
        self,
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
    ) -> None:
        """Load and cache models before processing. Call with same kwargs as run() (no video_path)."""
        self._get_or_build_models(
            pose2d=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_weights=det_weights,
            device=device,
            person_det_model=person_det_model,
            force_yolo_person=force_yolo_person,
            seg_model_path=seg_model_path,
            seg_device=seg_device,
        )

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

        inferencer, detector, yolo_person, seg_model = self._get_or_build_models(
            pose2d=pose2d,
            pose2d_weights=pose2d_weights,
            det_model=det_model,
            det_weights=det_weights,
            device=device,
            person_det_model=person_det_model,
            force_yolo_person=force_yolo_person,
            seg_model_path=seg_model_path,
            seg_device=seg_device,
        )

        frames: List[FramePose] = []
        shaft_angles: List[Optional[float]] = []
        club_centers: List[Optional[Tuple[float, float]]] = []
        shaft_centers: List[Optional[Tuple[float, float]]] = []
        person_bboxes: List[Optional[Tuple[float, float, float, float]]] = []
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
            person_bboxes.append(tuple(bbox) if bbox is not None else None)

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

        video_width = float(meta["width"])
        video_height = float(meta["height"])

        def _sanitize_seg_point(
            point: Optional[Tuple[float, float]],
        ) -> Optional[Tuple[float, float]]:
            if point is None:
                return None
            x, y = point
            if not (math.isfinite(x) and math.isfinite(y)):
                return None
            if x < 0.0 or x > video_width or y < 0.0 or y > video_height:
                return None
            return x, y

        def _seg_point_json(
            point: Optional[Tuple[float, float]],
        ) -> Optional[Dict[str, float]]:
            sanitized = _sanitize_seg_point(point)
            if sanitized is None:
                return None
            return {"x": sanitized[0], "y": sanitized[1]}

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
                    "shaft_angle": shaft_angles[i] if shaft_angles[i] is not None else None,
                    "club_center": _seg_point_json(club_centers[i]),
                    "shaft_center": _seg_point_json(shaft_centers[i]),
                    "person_bbox": (
                        {"x1": person_bboxes[i][0], "y1": person_bboxes[i][1], "x2": person_bboxes[i][2], "y2": person_bboxes[i][3]}
                        if person_bboxes[i] is not None
                        else None
                    ),
                }
                for i, f in enumerate(frames)
            ],
            "events": events,
            "events_raw": None,
            "swing_direction": inferred_direction,
        }


__all__ = ["SwingInferenceService"]
