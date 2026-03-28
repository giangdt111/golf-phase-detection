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


def apply_segmentation_with_line(
    frame,
    model,
    imgsz: int,
    conf: float,
    iou: float,
    device: str,
    classes: List[int],
    alpha: float,
) -> Tuple[np.ndarray, Optional[Tuple[float, float]], Optional[float]]:
    """Run segmentation once: draw colored masks, fit PCA line through best shaft mask, draw line.

    Returns:
        (frame_with_overlay, shaft_center_xy, shaft_angle_deg_from_horizontal)
        shaft_center_xy and shaft_angle_deg are None when no shaft detected.
    """
    if model is None:
        return frame, None, None
    res = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, classes=classes, verbose=False)
    if not res:
        return frame, None, None
    res = res[0]
    if res.masks is None:
        return frame, None, None

    masks = res.masks.data.cpu().numpy()
    clss = None
    if res.boxes is not None and res.boxes.cls is not None:
        clss = res.boxes.cls.cpu().numpy().astype(int)

    overlay = frame.copy()
    h, w = frame.shape[:2]

    # Collect all shaft mask pixels to find the best (largest) shaft mask
    best_shaft: Optional[Tuple[np.ndarray, np.ndarray]] = None  # (xs, ys) pixel arrays
    best_shaft_area = 0

    for i in range(masks.shape[0]):
        m = (masks[i] > 0.5).astype(np.uint8)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        color = np.zeros_like(frame, dtype=np.uint8)
        class_id = clss[i] if clss is not None and i < len(clss) else 0
        if class_id == 0:
            color[:, :, 1] = 255  # shaft = green
            ys, xs = np.where(m)
            if ys.size > best_shaft_area:
                best_shaft_area = int(ys.size)
                best_shaft = (xs, ys)
        else:
            color[:, :, 2] = 255  # head = red
        overlay[m == 1] = cv2.addWeighted(overlay, 1.0, color, alpha, 0)[m == 1]

    shaft_center: Optional[Tuple[float, float]] = None
    shaft_angle: Optional[float] = None

    if best_shaft is not None and best_shaft_area >= 12:
        xs, ys = best_shaft
        cx, cy = float(xs.mean()), float(ys.mean())
        coords = np.column_stack([xs, ys]).astype(np.float32)
        coords -= coords.mean(axis=0, keepdims=True)
        cov = np.cov(coords, rowvar=False)
        eigvals, eigvecs = np.linalg.eigh(cov)
        if eigvals.max() > 1e-6:
            v = eigvecs[:, int(np.argmax(eigvals))]
            # Normalize direction: flip so vy >= 0 (angle in [0°,180°) consistently).
            # abs(atan2) is wrong — it can map (-0.6,-0.8) to direction (-0.6,+0.8)
            # which is a DIFFERENT line (different slope). Canonical form: vy >= 0.
            if v[1] < 0:
                v = -v
            angle_deg = math.degrees(math.atan2(v[1], v[0]))  # in [0°, 180°)
            shaft_center = (cx, cy)
            shaft_angle = angle_deg

            # Extend line to image boundaries
            vx, vy = float(v[0]), float(v[1])
            ts = []
            if abs(vx) > 1e-6:
                ts.extend([(0 - cx) / vx, (w - 1 - cx) / vx])
            if abs(vy) > 1e-6:
                ts.extend([(0 - cy) / vy, (h - 1 - cy) / vy])
            if ts:
                t1, t2 = min(ts), max(ts)
                x1 = int(cx + t1 * vx)
                y1 = int(cy + t1 * vy)
                x2 = int(cx + t2 * vx)
                y2 = int(cy + t2 * vy)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 255), 3, cv2.LINE_AA)

    return overlay, shaft_center, shaft_angle


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
    roi: Optional[Tuple[int, int, int, int]] = None,
    grip_point: Optional[Tuple[float, float]] = None,
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
    shaft_idx_all = [i for i, c in enumerate(clss) if c == 0]
    head_idx_all = [i for i, c in enumerate(clss) if c == 1]

    def _center(idx: int) -> Optional[Tuple[float, float]]:
        m = (masks[idx] > 0.5)
        ys, xs = np.where(m)
        if ys.size == 0:
            return None
        return float(xs.mean()), float(ys.mean())

    def _filter_roi(idxs: List[int]) -> List[int]:
        if roi is None:
            return idxs
        x1, y1, x2, y2 = roi
        kept = []
        for i in idxs:
            ctr = _center(i)
            if ctr is None:
                continue
            cx, cy = ctr
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                kept.append(i)
        return kept

    shaft_idx = _filter_roi(shaft_idx_all)
    head_idx = _filter_roi(head_idx_all)
    if not shaft_idx and shaft_idx_all:
        shaft_idx = shaft_idx_all
    if not head_idx and head_idx_all:
        head_idx = head_idx_all

    h, w = frame.shape[:2]

    head_center = None
    if head_idx:
        areas = [float(masks[i].sum()) for i in head_idx]
        sel = head_idx[int(np.argmax(areas))]
        m = (masks[sel] > 0.5).astype(np.uint8)
        if m.shape[0] != h or m.shape[1] != w:
            m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
        ys, xs = np.where(m)
        if ys.size > 0:
            head_center = (float(xs.mean()), float(ys.mean()))

    shaft_angle = None
    shaft_center = None
    if shaft_idx:
        # --- Shaft angle: always from the LARGEST mask (same logic as apply_segmentation_with_line) ---
        best_area = 0
        for idx in shaft_idx:
            m = (masks[idx] > 0.5).astype(np.uint8)
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m)
            if ys.size < 12 or ys.size <= best_area:
                continue
            coords = np.column_stack([xs, ys]).astype(np.float32)
            coords -= coords.mean(axis=0, keepdims=True)
            cov = np.cov(coords, rowvar=False)
            eigvals_a, eigvecs_a = np.linalg.eigh(cov)
            if eigvals_a.max() <= 1e-6:
                continue
            best_area = int(ys.size)
            v_a = eigvecs_a[:, int(np.argmax(eigvals_a))]
            if v_a[1] < 0:
                v_a = -v_a
            shaft_angle = math.degrees(math.atan2(v_a[1], v_a[0]))  # [0°, 180°)

        # --- Shaft center: filtered by proximity to grip/head (for Kalman position tracking) ---
        best_score = float("inf")
        best_sel = None
        for idx in shaft_idx:
            m = (masks[idx] > 0.5).astype(np.uint8)
            if m.shape[0] != h or m.shape[1] != w:
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            ys, xs = np.where(m)
            if ys.size < 12:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            coords = np.column_stack([xs, ys]).astype(np.float32)
            coords -= coords.mean(axis=0, keepdims=True)
            cov = np.cov(coords, rowvar=False)
            eigvals, eigvecs = np.linalg.eigh(cov)
            if eigvals.min() <= 1e-6:
                continue
            area = float(m.sum())
            dist = 0.0
            if grip_point is not None:
                dist = math.hypot(cx - grip_point[0], cy - grip_point[1])
            if grip_point is not None and dist > 160.0:
                continue
            head_dist = 0.0
            if head_center is not None:
                head_dist = math.hypot(cx - head_center[0], cy - head_center[1])
                if head_dist > 220.0:
                    continue
            ratio = float(np.sqrt(max(eigvals) / max(1e-6, min(eigvals))))
            if not (dist <= 100.0 or head_dist <= 150.0):
                if ratio > 40.0:
                    continue
            score = dist + 0.4 * head_dist + 0.0025 * max(1.0, area)
            if score < best_score:
                best_score = score
                best_sel = (cx, cy)
        if best_sel is not None:
            shaft_center = best_sel

    return shaft_angle, head_center, shaft_center
