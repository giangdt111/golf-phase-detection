from typing import Dict, List, Optional, Tuple
import json
import math
import cv2
import os
import numpy as np

from .constants import COCO_SKELETON_EDGES
from .segmentation import apply_segmentation_with_line
from .detection import init_det_inferencer, init_person_yolo, select_person_bbox_yolo, expand_bbox
from .events import person_bbox_from_keypoints


def _load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)



def _safe_kp(keypoints: List[Dict], idx: int, score_thr: float = 0.2) -> Optional[Tuple[float, float]]:
    """Return (x, y) for keypoint idx if score >= score_thr, else None."""
    if not keypoints or idx >= len(keypoints):
        return None
    kp = keypoints[idx]
    if kp.get("score", 0.0) < score_thr:
        return None
    return (kp["x"], kp["y"])


def _midpoint(p1: Optional[Tuple], p2: Optional[Tuple]) -> Optional[Tuple[float, float]]:
    if p1 is None or p2 is None:
        return None
    return ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)


def _get_lead_shoulder(
    keypoints: Optional[List[Dict]],
    score_thr: float = 0.2,
    lead_is_right: Optional[bool] = None,
) -> Optional[Tuple[float, float]]:
    """Return lead-side shoulder point.

    If lead_is_right is provided (globally determined from swing_direction),
    use that. Otherwise fall back to per-frame arm score comparison.
    """
    if not keypoints:
        return None
    if lead_is_right is None:
        left_score = (keypoints[5].get("score", 0) + keypoints[7].get("score", 0)
                      + keypoints[9].get("score", 0))
        right_score = (keypoints[6].get("score", 0) + keypoints[8].get("score", 0)
                       + keypoints[10].get("score", 0))
        lead_is_right = right_score >= left_score
    lead_idx = 6 if lead_is_right else 5
    return _safe_kp(keypoints, lead_idx, score_thr)


def _draw_stats_panel(
    frame: np.ndarray,
    body_points: Optional[dict],
    display_angles: Optional[dict],
    calibration: Optional[dict],
    display_meta: Optional[dict],
    panel_x: int = 10,
    panel_y: int = 200,
) -> None:
    """Draw a unified stats panel for both face-on and DTL overlays."""
    FONT       = cv2.FONT_HERSHEY_SIMPLEX
    FS         = 0.5
    THICKNESS  = 1
    LINE_H     = 28
    PAD        = 8
    COL_W      = 420
    use_mm = True

    rows = [("Stats (mm)", (255, 255, 255), True)]
    if display_meta:
        view_name = str(display_meta.get("view_name", "")).replace("_", " ").title()
        face_on_fps = float(display_meta.get("face_on_fps", 0.0))
        dtl_fps = float(display_meta.get("down_the_line_fps", 0.0))
        sync_method = str(display_meta.get("sync_method", ""))
        rows.append((f"View: {view_name}", (200, 200, 255), False))
        rows.append((f"FPS    FO:{face_on_fps:>6.2f}  DTL:{dtl_fps:>6.2f}", (180, 255, 180), False))
        if sync_method:
            rows.append((f"Sync:  {sync_method}", (180, 220, 255), False))
    point_cfg = [
        ("Hip",      "hip"),
        ("Chest",    "chest"),
        ("Grip",     "grip"),
    ]
    for label, key in point_cfg:
        bp = (body_points or {}).get(key)
        if bp is not None:
            dx = bp.get("dx_mm_est") if use_mm else bp.get("dx")
            dy = bp.get("dy_mm_est") if use_mm else bp.get("dy")
            dz = bp.get("dz_mm_est") if use_mm else 0.0
            dx_str = f"{dx:+.1f}" if dx is not None else "  --"
            dy_str = f"{dy:+.1f}" if dy is not None else "  --"
            dz_str = f"{dz:+.1f}" if dz is not None else "  --"
            text = f"{label:<7} dx:{dx_str:>7}  dy:{dy_str:>7}  dz:{dz_str:>7}"
        else:
            text = f"{label:<7} dx:  --      dy:  --      dz:  --"
        rows.append((text, (220, 220, 220), False))

    angle_cfg = [
        ("Shaft", "shaft", (0, 255, 255)),
        ("Chest", "chest", (120, 220, 255)),
        ("Hip", "hip", (255, 200, 80)),
    ]
    for label, key, color in angle_cfg:
        angle_pair = (display_angles or {}).get(key) or {}
        fo = angle_pair.get("face_on_deg")
        dtl = angle_pair.get("down_the_line_deg")
        fo_str = f"{fo:.1f}" if fo is not None else "--"
        dtl_str = f"{dtl:.1f}" if dtl is not None else "--"
        rows.append((f"{label:<7} FO:{fo_str:>6}  DTL:{dtl_str:>6} deg", color, False))

    n_rows   = len(rows)
    box_h    = n_rows * LINE_H + PAD * 2
    box_w    = COL_W
    x1, y1   = panel_x, panel_y
    x2, y2   = x1 + box_w, y1 + box_h

    # Semi-transparent background
    h_f, w_f = frame.shape[:2]
    x2 = min(x2, w_f - 1);  y2 = min(y2, h_f - 1)
    roi = frame[y1:y2, x1:x2]
    black = np.zeros_like(roi)
    cv2.addWeighted(black, 0.55, roi, 0.45, 0, roi)
    frame[y1:y2, x1:x2] = roi

    # Text lines
    for i, (text, color, is_header) in enumerate(rows):
        ty = y1 + PAD + (i + 1) * LINE_H - 6
        fs = FS * 1.1 if is_header else FS
        tw = THICKNESS + 1 if is_header else THICKNESS
        cv2.putText(frame, text, (x1 + PAD, ty), FONT, fs, color, tw, cv2.LINE_AA)


def _compute_body_points(
    keypoints: Optional[List[Dict]],
    shaft_center: Optional[Tuple[float, float]],
    shaft_angle_deg: Optional[float],
    score_thr: float = 0.2,
) -> Tuple[Optional[Tuple], Optional[Tuple], Optional[Tuple], Optional[Tuple]]:
    """Return (hip_center, chest_center, shoulder_center, grip_point).

    hip_center      : mid(left_hip, right_hip)           — KP 11, 12
    shoulder_center : mid(left_shoulder, right_shoulder)  — KP 5, 6
    chest_center    : mid(shoulder_center, hip_center)    — ước tính xương ức
    grip_point:
        d1 = đường thẳng left_elbow (KP 7) → left_wrist (KP 9)
        d2 = đường thẳng right_elbow (KP 8) → right_wrist (KP 10)
        H1 = giao điểm d1 với shaft line
        H2 = giao điểm d2 với shaft line
        grip = (H1 + H2) / 2  (hoặc Hi nếu chỉ 1 tay detect được)
        Fallback về trung điểm wrist nếu shaft không detect được.
    """
    kp = lambda idx: _safe_kp(keypoints, idx, score_thr)

    shoulder_center = _midpoint(kp(5), kp(6))
    hip_center = _midpoint(kp(11), kp(12))
    # P chia MN (shoulder→hip) sao cho NP = 2·PM  →  P = (2·M + N) / 3
    if shoulder_center is not None and hip_center is not None:
        chest_center = ((2*shoulder_center[0] + hip_center[0]) / 3,
                        (2*shoulder_center[1] + hip_center[1]) / 3)
    else:
        chest_center = shoulder_center or hip_center

    le, lw = kp(7), kp(9)   # left_elbow, left_wrist
    re, rw = kp(8), kp(10)  # right_elbow, right_wrist
    l_ok = le is not None and lw is not None
    r_ok = re is not None and rw is not None

    # Ước tính body_scale từ torso (shoulder→hip) để làm ngưỡng khoảng cách
    _torso = _midpoint(shoulder_center, hip_center)
    if shoulder_center is not None and hip_center is not None:
        _torso_len = math.hypot(shoulder_center[0] - hip_center[0],
                                shoulder_center[1] - hip_center[1])
        _max_dist = max(30.0, _torso_len * 0.6)  # ≈ 0.2 * body_scale
    else:
        _max_dist = 120.0

    def _intersect(elbow, wrist, cx, cy, vx, vy):
        """Trả về None nếu giao điểm cách wrist > _max_dist."""
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        det = vx * dy - vy * dx
        if abs(det) < 1e-6:
            t = (wrist[0] - cx) * vx + (wrist[1] - cy) * vy
        else:
            t = (dx * (cy - elbow[1]) - dy * (cx - elbow[0])) / det
        H = (cx + t * vx, cy + t * vy)
        if math.hypot(H[0] - wrist[0], H[1] - wrist[1]) > _max_dist:
            return None
        return H

    grip = None
    if shaft_center is not None and shaft_angle_deg is not None and (l_ok or r_ok):
        rad = math.radians(shaft_angle_deg)
        vx, vy = math.cos(rad), math.sin(rad)
        cx, cy = shaft_center
        H1 = _intersect(le, lw, cx, cy, vx, vy) if l_ok else None
        H2 = _intersect(re, rw, cx, cy, vx, vy) if r_ok else None
        if H1 is not None and H2 is not None:
            grip = ((H1[0] + H2[0]) / 2, (H1[1] + H2[1]) / 2)
        elif H1 is not None:
            grip = H1
        elif H2 is not None:
            grip = H2
        # else: cả 2 tay không hướng về gậy → grip = None
    else:
        # Fallback khi không có shaft
        A = _midpoint(lw, rw) or lw or rw
        grip = A

    return hip_center, chest_center, shoulder_center, grip


def _compute_direction(y_history: List[float], threshold: float = 2.0) -> Optional[str]:
    """Return 'UP', 'DOWN', or None based on linear regression slope of Y values."""
    n = len(y_history)
    if n < 3:
        return None
    xs = list(range(n))
    x_mean = sum(xs) / n
    y_mean = sum(y_history) / n
    num = sum((xs[i] - x_mean) * (y_history[i] - y_mean) for i in range(n))
    den = sum((xs[i] - x_mean) ** 2 for i in range(n))
    slope = num / (den + 1e-6)
    if slope < -threshold:
        return "UP"
    if slope > threshold:
        return "DOWN"
    return None


def render_overlay(
    video_path: str,
    json_path: str,
    output_path: str,
    score_thr: float = 0.2,
    slow_factor: int = 1,
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
    phase_frames_out: Optional[str] = None,
) -> None:
    video_path = os.path.abspath(video_path)
    json_path = os.path.abspath(json_path)
    output_path = os.path.abspath(output_path)
    data = _load_json(json_path)
    calibration = data.get("calibration")
    display_meta = data.get("display_meta")
    frame_map = {
        int(item["frame"]): {
            "keypoints":          item.get("keypoints"),
            "shaft_angle_smooth": item.get("shaft_angle_smooth"),
            "shaft_smooth":       item.get("shaft_smooth"),
            "body_points":        item.get("body_points"),
            "hip_angle":          item.get("hip_angle"),
            "chest_angle":        item.get("chest_angle"),
            "display_angles":     item.get("display_angles"),
        }
        for item in data.get("frames", [])
    }
    edges = data.get("skeleton", {}).get("edges", COCO_SKELETON_EDGES)

    # Determine lead side globally (same logic as pipeline)
    swing_direction = data.get("swing_direction")
    if swing_direction == "clockwise":
        lead_is_right = True
    elif swing_direction == "counterclockwise":
        lead_is_right = False
    else:
        lead_is_right = None  # unknown — fall back to per-frame

    # Build phase structures from detected events
    events = data.get("events", []) or []
    # phase_milestones: sorted list of (frame_id, label) for continuous display
    # phase_exact_map: frame_id -> list[(phase_num, name, t)] for exact-frame actions
    phase_milestones: List[Tuple[int, str]] = []
    phase_exact_map: Dict[int, List[Tuple[int, str, float]]] = {}
    if events:
        sorted_events = sorted(events, key=lambda e: int(e["frame"]))
        for i, ev in enumerate(sorted_events):
            phase_num = i + 1
            fid_ev = int(ev["frame"])
            label = f"P{phase_num} - {ev['name']}"
            phase_milestones.append((fid_ev, label))
            phase_exact_map.setdefault(fid_ev, []).append((phase_num, ev["name"], float(ev.get("t", 0.0))))

    def _get_phase_label(fid: int) -> Optional[str]:
        """Return the active phase label for frame fid (O(n_phases))."""
        label = None
        for milestone_fid, milestone_label in phase_milestones:
            if milestone_fid <= fid:
                label = milestone_label
            else:
                break
        return label

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path} (cwd={os.getcwd()})")
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
    det_total = det_found = det_fallback = 0

    # UP/DOWN tracking state
    shaft_y_history: List[float] = []
    current_direction: Optional[str] = None


    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Draw segmentation mask + shaft line (single YOLO call)
        live_shaft_center: Optional[Tuple[float, float]] = None
        live_shaft_angle: Optional[float] = None
        if seg_model is not None:
            frame, live_shaft_center, live_shaft_angle = apply_segmentation_with_line(
                frame,
                model=seg_model,
                imgsz=seg_imgsz,
                conf=seg_conf,
                iou=seg_iou,
                device=seg_device,
                classes=seg_classes or [0, 1],
                alpha=seg_alpha,
            )

        # Person detection bbox
        frame_data = frame_map.get(frame_id) or {}
        keypoints            = frame_data.get("keypoints")
        stored_shaft_angle   = frame_data.get("shaft_angle_smooth")
        stored_shaft_center  = frame_data.get("shaft_smooth")
        if stored_shaft_center is not None:
            stored_shaft_center = tuple(stored_shaft_center)
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
                yolo_person, frame, conf=person_det_conf, iou=person_det_iou, imgsz=person_det_imgsz,
            )
        if det_bbox is None:
            det_bbox = last_det_bbox
        if det_bbox is not None:
            det_bbox = expand_bbox(det_bbox, width, height, scale=1.15)
            last_det_bbox = det_bbox
            x1, y1, x2, y2 = det_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
            if det_score is not None and det_debug:
                cv2.putText(frame, f"det {det_score:.2f}", (x1, max(0, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 2, cv2.LINE_AA)
            det_found += 1
        else:
            det_fallback += 1
        det_total += 1

        # Draw skeleton keypoints
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

        # Draw body center points + grip
        # Use stored shaft data from JSON (same source as phase detection pipeline)
        # so that grip_c is consistent with what the pipeline computed.
        # Dùng live_shaft (cùng nguồn với đường vàng đang vẽ) để grip luôn nằm trên shaft
        _shaft_center_for_grip = live_shaft_center if live_shaft_center is not None else stored_shaft_center
        _shaft_angle_for_grip  = live_shaft_angle  if live_shaft_angle  is not None else stored_shaft_angle
        hip_c, chest_c, shoulder_c, grip_c = _compute_body_points(
            keypoints, _shaft_center_for_grip, _shaft_angle_for_grip, score_thr
        )
        for pt, label, color in [
            (hip_c,      "Hip",      (255, 128,   0)),
            (chest_c,    "Chest",    (255, 220,   0)),
            (shoulder_c, "Shoulder", (180,  80, 255)),
            (grip_c,     "Grip",     (  0, 220, 255)),
        ]:
            if pt is None:
                continue
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 10, color, -1)
            cv2.circle(frame, (x, y), 12, (255, 255, 255), 2)  # white border
            cv2.putText(frame, label, (x + 15, y + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)

        # Draw lead-shoulder → grip line
        lead_sho = _get_lead_shoulder(keypoints, score_thr, lead_is_right)
        if lead_sho is not None and grip_c is not None:
            ls_pt = (int(lead_sho[0]), int(lead_sho[1]))
            gr_pt = (int(grip_c[0]),   int(grip_c[1]))
            cv2.line(frame, ls_pt, gr_pt, (0, 165, 255), 2)   # orange line
            cv2.circle(frame, ls_pt, 8, (0, 165, 255), -1)    # orange dot on lead shoulder

        # Track shaft Y from live segmentation if needed later
        if live_shaft_center is not None:
            shaft_y_history.append(live_shaft_center[1])
            if len(shaft_y_history) > 7:
                shaft_y_history.pop(0)
            direction = _compute_direction(shaft_y_history)
            if direction is not None:
                current_direction = direction

        # Phase label (continuous, active from detected frame until next phase)
        phase_label = _get_phase_label(frame_id)
        if phase_label:
            cv2.putText(frame, phase_label, (20, height - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3, cv2.LINE_AA)

        # Exact phase frame: draw border + save image
        if frame_id in phase_exact_map:
            cv2.rectangle(frame, (0, 0), (width - 1, height - 1), (0, 255, 255), 6)
            if phase_frames_out is not None:
                os.makedirs(phase_frames_out, exist_ok=True)
            for idx_exact, (phase_num, phase_name_exact, t_ev) in enumerate(phase_exact_map[frame_id]):
                cv2.putText(frame, f"P{phase_num} - {phase_name_exact}", (20, 50 + idx_exact * 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3, cv2.LINE_AA)
                if phase_frames_out is not None:
                    fname = f"{frame_id:06d}_P{phase_num}_{phase_name_exact.replace(' ', '_')}_{t_ev:.3f}.jpg"
                    cv2.imwrite(os.path.join(phase_frames_out, fname), frame)

        # Stats panel (bottom-left)
        _bp = frame_data.get("body_points") if frame_data else None
        _display_angles = frame_data.get("display_angles") if frame_data else None
        if _display_angles is None:
            _display_angles = {
                "shaft": {
                    "face_on_deg": float(min(live_shaft_angle % 180.0, 180.0 - live_shaft_angle % 180.0))
                    if live_shaft_angle is not None
                    else (stored_shaft_angle if stored_shaft_angle is not None else 0.0),
                    "down_the_line_deg": 0.0,
                },
                "chest": {
                    "face_on_deg": frame_data.get("chest_angle") if frame_data else 0.0,
                    "down_the_line_deg": 0.0,
                },
                "hip": {
                    "face_on_deg": frame_data.get("hip_angle") if frame_data else 0.0,
                    "down_the_line_deg": 0.0,
                },
            }
        _draw_stats_panel(frame, _bp, _display_angles, calibration, display_meta,
                          panel_x=10, panel_y=height - 330)

        # Frame number (top-right corner)
        fn_text = f"#{frame_id}"
        (tw, th), _ = cv2.getTextSize(fn_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(frame, fn_text, (width - tw - 10, th + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2, cv2.LINE_AA)

        writer.write(frame)
        frame_id += 1

    cap.release()
    writer.release()
    if det_debug and det_total > 0:
        print(f"Det debug summary: frames={det_total}, found={det_found}, fallback={det_fallback}")
