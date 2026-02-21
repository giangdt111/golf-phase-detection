from typing import Dict, List, Optional, Tuple
import json
import cv2
import os

from .constants import COCO_SKELETON_EDGES
from .segmentation import apply_segmentation
from .detection import init_det_inferencer, init_person_yolo, select_person_bbox_yolo, expand_bbox
from .events import person_bbox_from_keypoints


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_overlay(
    video_path: str,
    json_path: str,
    output_path: str,
    score_thr: float = 0.2,
    slow_factor: int = 1,
    events_override: Optional[List[Dict]] = None,
    seg_model=None,
    seg_imgsz: int = 960,
    seg_conf: float = 0.25,
    seg_iou: float = 0.7,
    seg_device: str = "cpu",
    seg_classes: Optional[List[int]] = None,
    seg_alpha: float = 0.45,
    det_model: Optional[str] = None,
    det_weights: Optional[str] = None,
    det_device: Optional[str] = None,
    det_debug: bool = False,
    person_det_model: Optional[str] = None,
    person_det_conf: float = 0.25,
    person_det_iou: float = 0.7,
    person_det_imgsz: int = 640,
) -> None:
    data = _load_json(json_path)
    frame_map = {int(item["frame"]): item.get("keypoints") for item in data.get("frames", [])}
    edges = data.get("skeleton", {}).get("edges", COCO_SKELETON_EDGES)
    events_src = events_override if events_override is not None else data.get("events", [])
    event_map = {int(event["frame"]): event["name"] for event in events_src}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_fps = fps / max(1, slow_factor)
    writer = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")

    detector = init_det_inferencer(det_model, det_weights, det_device)
    yolo_person = None
    if detector is None:
        yolo_person = init_person_yolo(person_det_model, det_device)
    last_det_bbox = None
    frame_id = 0
    current_phase = ""
    sorted_events = sorted(event_map.items())
    event_idx = 0
    det_total = det_found = det_fallback = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if seg_model is not None:
            frame = apply_segmentation(
                frame,
                model=seg_model,
                imgsz=seg_imgsz,
                conf=seg_conf,
                iou=seg_iou,
                device=seg_device,
                classes=seg_classes or [0, 1],
                alpha=seg_alpha,
            )
        if event_idx < len(sorted_events) and frame_id >= sorted_events[event_idx][0]:
            current_phase = sorted_events[event_idx][1]
            event_idx += 1

        keypoints = frame_map.get(frame_id)
        det_bbox = None
        det_score = None
        if detector is not None:
            det_out = detector(frame, return_datasample=True).get("predictions", [])
            det_sample = det_out[0] if det_out else None
            if det_sample is not None:
                pred = det_sample.pred_instances
                bboxes = pred.bboxes.cpu().numpy()
                scores = pred.scores.cpu().numpy()
                labels = pred.labels.cpu().numpy()
                if bboxes.size > 0:
                    mask = labels == 0
                    if mask.any():
                        bboxes = bboxes[mask]
                        scores = scores[mask]
                    if bboxes.size > 0:
                        best = int(scores.argmax())
                        x1, y1, x2, y2 = bboxes[best].astype(int).tolist()
                        det_bbox = (x1, y1, x2, y2)
                        det_score = float(scores[best])
        elif yolo_person is not None:
            det_bbox = select_person_bbox_yolo(
                yolo_person,
                frame,
                conf=person_det_conf,
                iou=person_det_iou,
                imgsz=person_det_imgsz,
            )
            det_score = None
        if det_bbox is None:
            det_bbox = last_det_bbox
        if det_bbox is not None:
            det_bbox = expand_bbox(det_bbox, width, height, scale=1.15)
            last_det_bbox = det_bbox
            x1, y1, x2, y2 = det_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            if det_score is not None:
                cv2.putText(
                    frame,
                    f"det {det_score:.2f}",
                    (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 165, 0),
                    2,
                    cv2.LINE_AA,
                )
            det_found += 1
        else:
            det_fallback += 1
        det_total += 1
        if det_debug and det_total <= 5:
            print(
                f"Det debug frame {frame_id}: bbox={det_bbox} score={det_score} fallback={det_bbox is None}"
            )

        if keypoints:
            pts = []
            for kp in keypoints:
                if kp.get("score", 0.0) >= score_thr:
                    pts.append((int(kp["x"]), int(kp["y"])))
                else:
                    pts.append(None)
            for a, b in edges:
                pa = pts[a] if a < len(pts) else None
                pb = pts[b] if b < len(pts) else None
                if pa is None or pb is None:
                    continue
                cv2.line(frame, pa, pb, (0, 255, 0), 2)
            for p in pts:
                if p is None:
                    continue
                cv2.circle(frame, p, 3, (0, 0, 255), -1)

            if det_bbox is None:
                kp_bbox = person_bbox_from_keypoints(keypoints, score_thr=score_thr)
                if kp_bbox is not None:
                    x1, y1, x2, y2 = kp_bbox
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)

        label = f"Phase: {current_phase}" if current_phase else "Phase: -"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        pad = 6
        x0, y0 = 20, 70
        cv2.rectangle(
            frame,
            (x0 - pad, y0 - text_h - pad),
            (x0 + text_w + pad, y0 + pad),
            (0, 0, 0),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x0, y0),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    if det_debug and det_total > 0:
        print(
            f"Det debug summary: frames={det_total}, found={det_found}, fallback={det_fallback}"
        )


def write_phase_frames(
    video_path: str,
    events: List[Dict],
    out_dir: str,
    frame_map: Optional[Dict[int, Optional[List[Dict[str, float]]]]] = None,
    edges: Optional[List[List[int]]] = None,
    score_thr: float = 0.2,
    seg_model=None,
    seg_imgsz: int = 960,
    seg_conf: float = 0.25,
    seg_iou: float = 0.7,
    seg_device: str = "cpu",
    seg_classes: Optional[List[int]] = None,
    seg_alpha: float = 0.45,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    if edges is None:
        edges = COCO_SKELETON_EDGES
    for event in events:
        frame_id = int(event["frame"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
        ok, frame = cap.read()
        if not ok:
            continue
        if seg_model is not None:
            frame = apply_segmentation(
                frame,
                model=seg_model,
                imgsz=seg_imgsz,
                conf=seg_conf,
                iou=seg_iou,
                device=seg_device,
                classes=seg_classes or [0, 1],
                alpha=seg_alpha,
            )
        keypoints = frame_map.get(frame_id) if frame_map is not None else None
        if keypoints:
            pts = []
            for kp in keypoints:
                if kp.get("score", 0.0) >= score_thr:
                    pts.append((int(kp["x"]), int(kp["y"])))
                else:
                    pts.append(None)
            for a, b in edges:
                pa = pts[a] if a < len(pts) else None
                pb = pts[b] if b is not None and b < len(pts) else None
                if pa is None or pb is None:
                    continue
                cv2.line(frame, pa, pb, (0, 255, 0), 2)
            for p in pts:
                if p is None:
                    continue
                cv2.circle(frame, p, 3, (0, 0, 255), -1)

        name = event["name"].replace(" ", "_")
        conf = float(event.get("confidence", 0.0))
        fname = f"{frame_id:06d}_{name}_{conf:.3f}.jpg"
        cv2.imwrite(os.path.join(out_dir, fname), frame)
    cap.release()
