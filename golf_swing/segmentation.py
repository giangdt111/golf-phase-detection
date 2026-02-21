from typing import List, Optional, Tuple
import math
import os
import cv2
import numpy as np


def init_seg_model(model_path: str):
    try:
        from ultralytics import YOLO
    except Exception as exc:  # pragma: no cover - ultralytics optional
        print(f"Ultralytics not available for segmentation: {exc}")
        return None
    if not os.path.exists(model_path):
        print(f"Segmentation model not found: {model_path}")
        return None
    return YOLO(model_path)


def apply_segmentation(frame, model, imgsz: int, conf: float, iou: float, device: str, classes: List[int], alpha: float):
    if model is None:
        return frame
    res = model.predict(
        frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        classes=classes,
        verbose=False,
    )
    if not res:
        return frame
    res = res[0]
    if res.masks is None:
        return frame
    masks = res.masks.data.cpu().numpy()
    clss = None
    if res.boxes is not None and res.boxes.cls is not None:
        clss = res.boxes.cls.cpu().numpy().astype(int)

    overlay = frame.copy()
    h, w = frame.shape[:2]
    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(frame, dtype=np.uint8)
        class_id = clss[i] if clss is not None and i < len(clss) else 0
        if class_id == 0:
            color[:, :, 1] = 255  # shaft = green
        else:
            color[:, :, 2] = 255  # head = red
        overlay[m == 1] = cv2.addWeighted(overlay, 1.0, color, alpha, 0)[m == 1]
    return overlay


def segment_frame_features(
    frame,
    model,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    classes: List[int],
) -> Tuple[Optional[float], Optional[Tuple[float, float]], Optional[Tuple[float, float]]]:
    if model is None:
        return None, None, None
    res = model.predict(
        frame,
        imgsz=imgsz,
        conf=conf,
        iou=iou,
        device=device,
        classes=classes,
        verbose=False,
    )
    if not res:
        return None, None, None
    res = res[0]
    if res.masks is None or res.boxes is None or res.boxes.cls is None:
        return None, None, None
    masks = res.masks.data.cpu().numpy()
    clss = res.boxes.cls.cpu().numpy().astype(int)
    shaft_idx = [i for i, c in enumerate(clss) if c == 0]
    head_idx = [i for i, c in enumerate(clss) if c == 1]

    shaft_angle = None
    shaft_center = None
    if shaft_idx:
        areas = [float(masks[i].sum()) for i in shaft_idx]
        sel = shaft_idx[int(np.argmax(areas))]
        m = (masks[sel] > 0.5)
        ys, xs = np.where(m)
        if ys.size > 0:
            shaft_center = (float(xs.mean()), float(ys.mean()))
        if ys.size > 10:
            coords = np.column_stack([xs, ys]).astype(np.float32)
            coords -= coords.mean(axis=0, keepdims=True)
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            v = eigvecs[:, int(np.argmax(eigvals))]
            angle = math.degrees(math.atan2(v[1], v[0]))
            angle = abs(angle) % 180.0
            shaft_angle = angle

    head_center = None
    if head_idx:
        areas = [float(masks[i].sum()) for i in head_idx]
        sel = head_idx[int(np.argmax(areas))]
        m = (masks[sel] > 0.5)
        ys, xs = np.where(m)
        if ys.size > 0:
            head_center = (float(xs.mean()), float(ys.mean()))

    return shaft_angle, head_center, shaft_center
