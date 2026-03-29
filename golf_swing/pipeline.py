from typing import Dict, List, Optional, Tuple
import os
import numpy as np

from .constants import COCO_KEYPOINT_NAMES, COCO_SKELETON_EDGES
from .events import FramePose, extract_keypoints, infer_swing_direction, safe_point
from .events_logic import detect_events_rule9
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
        height_mm: Optional[float],
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
        debug_p9: bool = False,
        debug_p9_path: str = "p9_debug.txt",
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
        head_pred_traj: List[Optional[Tuple[float, float]]] = []
        shaft_pred_traj: List[Optional[Tuple[float, float]]] = []
        head_meas_traj: List[Optional[Tuple[float, float]]] = []
        shaft_meas_traj: List[Optional[Tuple[float, float]]] = []
        shaft_angle_pred_traj: List[Optional[float]] = []
        frame_id = 0
        kept = 0
        last_bbox = None

        # Simple Kalman filters for head/shaft position (x,y) and shaft angle (deg)
        def _kf_init_pos(meas: np.ndarray, std: float = 12.0):
            state = np.array([meas[0], meas[1], 0.0, 0.0], dtype=float)
            P = np.eye(4) * (std ** 2)
            return state, P

        def _kf_predict_pos(state: np.ndarray, P: np.ndarray, dt: float, q: float = 10.0):
            F = np.array(
                [[1, 0, dt, 0],
                 [0, 1, 0, dt],
                 [0, 0, 1, 0],
                 [0, 0, 0, 1]],
                dtype=float,
            )
            Q = np.eye(4) * (q ** 2)
            state = F @ state
            P = F @ P @ F.T + Q
            return state, P

        def _kf_update_pos(state: np.ndarray, P: np.ndarray, meas: np.ndarray, r: float = 6.0):
            H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
            R = np.eye(2) * (r ** 2)
            y = meas - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + K @ y
            I = np.eye(len(P))
            P = (I - K @ H) @ P
            return state, P

        def _kf_predict_ang(state: np.ndarray, P: np.ndarray, dt: float, q: float = 9.0):
            F = np.array([[1, dt], [0, 1]], dtype=float)
            Q = np.eye(2) * (q ** 2)
            state = F @ state
            P = F @ P @ F.T + Q
            return state, P

        def _kf_update_ang(state: np.ndarray, P: np.ndarray, meas: float, r: float = 6.0):
            H = np.array([[1, 0]], dtype=float)
            R = np.array([[r ** 2]], dtype=float)
            y = meas - H @ state
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            state = state + (K * y).flatten()
            P = (np.eye(2) - K @ H) @ P
            return state, P

        head_state = None
        head_P = None
        ang_state = None  # [angle, ang_vel]
        ang_P = None
        shaft_state = None
        shaft_P = None
        prev_head_meas = None
        prev_shaft_meas = None
        dt = 1.0 / max(fps, 1e-6)

        # Gating & damping parameters
        gate_px = 220.0  # reject only extreme jumps
        vel_damp = 0.92
        max_step = 90.0  # px per frame cap on displacement

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
            # Grip midpoint (for segmentation scoring)
            grip_mid = None
            if keypoints:
                lw = safe_point(keypoints, 9)
                rw = safe_point(keypoints, 10)
                if lw is not None and rw is not None:
                    grip_mid = ((lw[0] + rw[0]) / 2.0, (lw[1] + rw[1]) / 2.0)

            if seg_model is not None:
                # Predict step for current frame to keep motion when detections drop
                if head_state is not None and head_P is not None:
                    head_state, head_P = _kf_predict_pos(head_state, head_P, dt=dt)
                if shaft_state is not None and shaft_P is not None:
                    shaft_state, shaft_P = _kf_predict_pos(shaft_state, shaft_P, dt=dt)
                if ang_state is not None and ang_P is not None:
                    ang_state, ang_P = _kf_predict_ang(ang_state, ang_P, dt=dt)

                angle, center, shaft_center = segment_frame_features(
                    frame,
                    model=seg_model,
                    imgsz=seg_imgsz,
                    conf=seg_conf,
                    iou=seg_iou,
                    device=seg_device,
                    classes=[0, 1],
                    roi=bbox if bbox is not None else None,
                    grip_point=grip_mid,
                )
                shaft_angles.append(angle)
                club_centers.append(center)
                shaft_centers.append(shaft_center)

                # Kalman head update (with mild gating + damping)
                head_meas_used = False
                if center is not None:
                    meas = np.array([center[0], center[1]], dtype=float)
                    if head_state is not None:
                        dist = np.linalg.norm(meas - head_state[:2])
                        if dist <= gate_px:
                            head_state, head_P = _kf_update_pos(head_state, head_P, meas, r=6.0)
                            head_meas_used = True
                        else:
                            center = None  # drop extreme outlier
                    if head_state is None and center is not None:
                        head_state, head_P = _kf_init_pos(meas, std=6.0)
                        head_meas_used = True
                head_meas_traj.append((float(center[0]), float(center[1])) if center is not None else None)

                if head_state is not None:
                    # If we used a measurement and have a previous one, update velocity estimate.
                    if head_meas_used:
                        if prev_head_meas is not None:
                            meas_prev = np.array(prev_head_meas, dtype=float)
                            vel_est = (meas - meas_prev) / max(dt, 1e-6)
                            head_state[2:] = 0.5 * head_state[2:] + 0.5 * vel_est
                        prev_head_meas = (float(meas[0]), float(meas[1])) if center is not None else prev_head_meas
                    # If no measurement this frame, keep coasting but gently damp/clamp.
                    if not head_meas_used:
                        head_state[2:] *= vel_damp
                    step = float(np.linalg.norm(head_state[2:] * dt))
                    if step > max_step:
                        scale = max_step / (step + 1e-6)
                        head_state[2:] *= scale
                    head_pred_traj.append((float(head_state[0]), float(head_state[1])))
                else:
                    head_pred_traj.append(center)

                # Kalman shaft center (with mild gating + damping)
                shaft_meas_used = False
                if shaft_center is not None:
                    meas_s = np.array([shaft_center[0], shaft_center[1]], dtype=float)
                    if shaft_state is not None:
                        dist = np.linalg.norm(meas_s - shaft_state[:2])
                        if dist <= gate_px:
                            shaft_state, shaft_P = _kf_update_pos(shaft_state, shaft_P, meas_s, r=6.0)
                            shaft_meas_used = True
                        else:
                            shaft_center = None
                    if shaft_state is None and shaft_center is not None:
                        shaft_state, shaft_P = _kf_init_pos(meas_s, std=6.0)
                        shaft_meas_used = True
                shaft_meas_traj.append((float(shaft_center[0]), float(shaft_center[1])) if shaft_center is not None else None)

                if shaft_state is not None:
                    if shaft_meas_used:
                        if prev_shaft_meas is not None:
                            meas_prev_s = np.array(prev_shaft_meas, dtype=float)
                            vel_est_s = (meas_s - meas_prev_s) / max(dt, 1e-6)
                            shaft_state[2:] = 0.5 * shaft_state[2:] + 0.5 * vel_est_s
                        prev_shaft_meas = (float(meas_s[0]), float(meas_s[1])) if shaft_center is not None else prev_shaft_meas
                    if not shaft_meas_used:
                        shaft_state[2:] *= vel_damp
                    step_s = float(np.linalg.norm(shaft_state[2:] * dt))
                    if step_s > max_step:
                        scale = max_step / (step_s + 1e-6)
                        shaft_state[2:] *= scale
                    shaft_pred_traj.append((float(shaft_state[0]), float(shaft_state[1])))
                else:
                    shaft_pred_traj.append(shaft_center)

                # Kalman angle update
                if angle is not None:
                    ang_meas = float(angle)
                    if ang_state is None or ang_P is None:
                        ang_state = np.array([ang_meas, 0.0], dtype=float)
                        ang_P = np.eye(2) * 25.0
                    else:
                        # Pick the equivalent measurement closest to the predicted angle
                        # so horizontal states around 0/180 deg do not jump.
                        pred_ang = float(ang_state[0])
                        ang_meas = min(
                            (ang_meas - 180.0, ang_meas, ang_meas + 180.0),
                            key=lambda cand: abs(cand - pred_ang),
                        )
                        ang_state, ang_P = _kf_update_ang(ang_state, ang_P, ang_meas, r=6.0)
                if ang_state is not None:
                    shaft_angle_pred_traj.append(float(ang_state[0] % 180.0))
                else:
                    shaft_angle_pred_traj.append(angle)
            else:
                shaft_angles.append(None)
                club_centers.append(None)
                shaft_centers.append(None)
                head_pred_traj.append(None)
                shaft_pred_traj.append(None)
                shaft_angle_pred_traj.append(None)
            frame_id += 1
            kept += 1

        cap.release()

        def _blend_smooth_positions(
            measured: List[Optional[Tuple[float, float]]],
            predicted: List[Optional[Tuple[float, float]]],
            max_gap: int = 4,
        ) -> List[Optional[Tuple[float, float]]]:
            out: List[Optional[Tuple[float, float]]] = []
            seen_measurement = False
            gap = max_gap + 1
            for meas, pred in zip(measured, predicted):
                if meas is not None:
                    seen_measurement = True
                    gap = 0
                    out.append(pred if pred is not None else meas)
                    continue
                if seen_measurement:
                    gap += 1
                if seen_measurement and gap <= max_gap and pred is not None:
                    out.append(pred)
                else:
                    out.append(None)
            return out

        def _blend_smooth_angles(
            measured: List[Optional[float]],
            predicted: List[Optional[float]],
            max_gap: int = 4,
        ) -> List[Optional[float]]:
            out: List[Optional[float]] = []
            seen_measurement = False
            gap = max_gap + 1
            for meas, pred in zip(measured, predicted):
                if meas is not None:
                    seen_measurement = True
                    gap = 0
                    out.append(pred if pred is not None else float(meas))
                    continue
                if seen_measurement:
                    gap += 1
                if seen_measurement and gap <= max_gap and pred is not None:
                    out.append(pred)
                else:
                    out.append(None)
            return out

        club_centers_smooth = _blend_smooth_positions(head_meas_traj, head_pred_traj)
        shaft_centers_smooth = _blend_smooth_positions(shaft_meas_traj, shaft_pred_traj)
        shaft_angles_smooth = _blend_smooth_angles(shaft_angles, shaft_angle_pred_traj)

        inferred_direction = swing_direction or infer_swing_direction(frames)
        events, body_signals = detect_events_rule9(
            frames,
            fps,
            inferred_direction,
            shaft_angles_smooth,
            club_centers_smooth,
            shaft_centers_smooth,
            debug=debug_p9,
            debug_path=debug_p9_path,
        )

        # --- Body points displacement (relative to Address / P1) ---
        frame_id_to_idx = {f.frame_id: i for i, f in enumerate(frames)}
        p1_frame_id = next((e['frame'] for e in events if e['name'] == 'Address'), None)
        p1_idx = frame_id_to_idx.get(p1_frame_id)

        def _midpoint(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if a is None or b is None:
                return None
            return (a + b) / 2.0

        def _segment_length(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
            if a is None or b is None:
                return None
            return float(np.linalg.norm(a - b))

        def _mean_valid(values: List[Optional[float]]) -> Optional[float]:
            finite = [float(v) for v in values if v is not None and np.isfinite(v)]
            if not finite:
                return None
            return float(sum(finite) / len(finite))

        def _estimate_height_scale(
            keypoints: Optional[List[Dict[str, float]]],
            player_height_mm: Optional[float],
        ) -> Optional[Dict[str, float]]:
            if player_height_mm is None or player_height_mm <= 0 or not keypoints:
                return None

            shoulder_mid = _midpoint(safe_point(keypoints, 5), safe_point(keypoints, 6))
            hip_mid = _midpoint(safe_point(keypoints, 11), safe_point(keypoints, 12))
            torso_len = _segment_length(shoulder_mid, hip_mid)
            left_leg = _mean_valid([
                _segment_length(safe_point(keypoints, 11), safe_point(keypoints, 13)),
                _segment_length(safe_point(keypoints, 13), safe_point(keypoints, 15)),
            ])
            right_leg = _mean_valid([
                _segment_length(safe_point(keypoints, 12), safe_point(keypoints, 14)),
                _segment_length(safe_point(keypoints, 14), safe_point(keypoints, 16)),
            ])
            leg_len = _mean_valid([left_leg, right_leg])

            head_anchor_dist = _mean_valid([
                _segment_length(safe_point(keypoints, idx), shoulder_mid) for idx in (0, 1, 2, 3, 4)
            ])
            if head_anchor_dist is not None:
                head_len = head_anchor_dist * 1.18
            elif torso_len is not None:
                head_len = torso_len * 0.42
            else:
                head_len = None

            if torso_len is None or leg_len is None or head_len is None:
                return None

            height_proxy_px = torso_len + leg_len + head_len
            if not np.isfinite(height_proxy_px) or height_proxy_px <= 1e-6:
                return None

            return {
                "height_mm": float(player_height_mm),
                "height_proxy_px": round(float(height_proxy_px), 2),
                "mm_per_px": round(float(player_height_mm / height_proxy_px), 6),
                "source": "address_pose_height_proxy",
            }

        scale_info = None
        mm_per_px = None
        if p1_idx is not None and 0 <= p1_idx < len(frames):
            scale_info = _estimate_height_scale(frames[p1_idx].keypoints, height_mm)
            if scale_info is not None:
                mm_per_px = float(scale_info["mm_per_px"])

        _BP_KEYS = ('hip_xy', 'chest_xy', 'shoulder_xy', 'grip_xy')
        _BP_NAMES = ('hip', 'chest', 'shoulder', 'grip')

        # Reference coords at P1 (NaN → None)
        ref = {}
        if p1_idx is not None:
            for key in _BP_KEYS:
                v = body_signals[key][p1_idx]
                ref[key] = v if np.isfinite(v).all() else None

        def _body_points_for_frame(arr_idx):
            """Return body_points dict for one frame (None nếu trước Address)."""
            if p1_idx is None or arr_idx < p1_idx:
                return None
            result = {}
            for key, name in zip(_BP_KEYS, _BP_NAMES):
                v = body_signals[key][arr_idx]
                if not np.isfinite(v).all():
                    result[name] = None
                    continue
                x, y = round(float(v[0]), 1), round(float(v[1]), 1)
                r = ref.get(key)
                dx = round(float(v[0] - r[0]), 1) if r is not None else None
                dy = round(float(v[1] - r[1]), 1) if r is not None else None
                item = {"x": x, "y": y, "dx": dx, "dy": dy}
                if mm_per_px is not None:
                    item["dx_mm_est"] = round(dx * mm_per_px, 1) if dx is not None else None
                    item["dy_mm_est"] = round(dy * mm_per_px, 1) if dy is not None else None
                result[name] = item
            return result

        return {
            "video": {
                "path": video_path,
                "fps": fps,
                "width": meta["width"],
                "height": meta["height"],
                "frame_count": meta["frame_count"],
                "stride": stride,
            },
            "calibration": scale_info,
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
                    "shaft_angle": shaft_angles[idx] if idx < len(shaft_angles) else None,
                    "shaft_angle_smooth": shaft_angles_smooth[idx] if idx < len(shaft_angles_smooth) else None,
                    "head_pred": head_pred_traj[idx] if idx < len(head_pred_traj) else None,
                    "head_meas": head_meas_traj[idx] if idx < len(head_meas_traj) else None,
                    "head_smooth": club_centers_smooth[idx] if idx < len(club_centers_smooth) else None,
                    "shaft_pred": shaft_pred_traj[idx] if idx < len(shaft_pred_traj) else None,
                    "shaft_meas": shaft_meas_traj[idx] if idx < len(shaft_meas_traj) else None,
                    "shaft_smooth": shaft_centers_smooth[idx] if idx < len(shaft_centers_smooth) else None,
                    "body_points": _body_points_for_frame(idx),
                    "hip_angle": round(float(body_signals['hip_angle'][idx]), 2) if idx < len(body_signals['hip_angle']) and np.isfinite(body_signals['hip_angle'][idx]) else None,
                    "chest_angle": round(float(body_signals['chest_angle'][idx]), 2) if idx < len(body_signals['chest_angle']) and np.isfinite(body_signals['chest_angle'][idx]) else None,
                }
                for idx, f in enumerate(frames)
            ],
            "events": events,
            "events_raw": None,
            "swing_direction": inferred_direction,
        }


__all__ = ["SwingInferenceService"]
