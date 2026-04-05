from typing import Optional, Tuple
import os


def init_det_inferencer(det_model: Optional[str], det_weights: Optional[str], device: Optional[str]):
    if not det_model or det_model in ("whole_image", "whole-image"):
        return None
    try:
        # mmdet >= 3.1 exports DetInferencer directly from mmdet.apis
        try:
            from mmdet.apis import DetInferencer  # type: ignore
        except ImportError:
            from mmdet.apis.det_inferencer import DetInferencer  # type: ignore
    except Exception:
        return None
    return DetInferencer(det_model, det_weights, device=device)


def init_person_yolo(model_path: Optional[str], device: Optional[str]):
    if not model_path:
        return None
    try:
        from ultralytics import YOLO  # type: ignore
        import numpy as np
    except Exception:
        return None
    try:
        model = YOLO(model_path)
        if device:
            model.to(device)
        # Force Ultralytics' lazy internal warmup (cuDNN init) here so the
        # first real predict() never hits an uninitialized cuDNN context.
        try:
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            model.predict(dummy, imgsz=64, verbose=False)
        except Exception as exc:
            if device and "cuda" in str(device).lower():
                print(
                    f"[WARN] YOLO CUDA warmup failed ({exc}); falling back to CPU."
                )
                model.to("cpu")
                dummy = np.zeros((64, 64, 3), dtype=np.uint8)
                model.predict(dummy, imgsz=64, verbose=False)
        return model
    except Exception:
        return None


def select_person_bbox_yolo(model, frame, conf: float, iou: float, imgsz: int):
    if model is None:
        return None
    res = model.predict(
        frame,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[0],
        verbose=False,
    )
    if not res:
        return None
    r0 = res[0]
    if r0.boxes is None or len(r0.boxes) == 0:
        return None
    boxes = r0.boxes.xyxy.cpu().numpy()
    confs = r0.boxes.conf.cpu().numpy()
    best = int(confs.argmax())
    x1, y1, x2, y2 = boxes[best].astype(int).tolist()
    return x1, y1, x2, y2


def select_person_bbox(det_result) -> Optional[Tuple[int, int, int, int]]:
    if det_result is None:
        return None
    pred = det_result.pred_instances
    bboxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    labels = pred.labels.cpu().numpy()
    if bboxes.size == 0:
        return None
    mask = labels == 0
    if mask.any():
        bboxes = bboxes[mask]
        scores = scores[mask]
    if bboxes.size == 0:
        return None
    best = int(scores.argmax())
    x1, y1, x2, y2 = bboxes[best].astype(int).tolist()
    return x1, y1, x2, y2


def expand_bbox(bbox: Tuple[int, int, int, int], width: int, height: int, scale: float = 1.2):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * scale
    h = (y2 - y1) * scale
    nx1 = max(0, int(cx - w / 2))
    ny1 = max(0, int(cy - h / 2))
    nx2 = min(width - 1, int(cx + w / 2))
    ny2 = min(height - 1, int(cy + h / 2))
    return nx1, ny1, nx2, ny2
