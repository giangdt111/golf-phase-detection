"""Rule-based 9-phase swing event detector."""
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .events import FramePose, compute_swing_features


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _estimate_body_scale(features: Dict[str, np.ndarray]) -> float:
    """Median shoulder/ankle span in pixels — used to normalize thresholds."""
    spans = []
    for la, ra in zip(features["left_ankle"], features["right_ankle"]):
        if not (np.any(np.isnan(la)) or np.any(np.isnan(ra))):
            spans.append(float(np.linalg.norm(la - ra)))
    for ls, rs in zip(features["left_shoulder"], features["right_shoulder"]):
        if not (np.any(np.isnan(ls)) or np.any(np.isnan(rs))):
            spans.append(float(np.linalg.norm(ls - rs) * 1.8))
    return max(1.0, float(np.median(spans))) if spans else 120.0


def _normalize_cost(values: List[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.inf, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return out
    lo = float(np.percentile(arr[finite], 10))
    hi = float(np.percentile(arr[finite], 90))
    out[finite] = np.clip((arr[finite] - lo) / max(hi - lo, 1e-6), 0.0, 8.0)
    return out


def _argmin_range(arr: np.ndarray, i0: int, i1: int) -> int:
    """Index of minimum finite value in arr[i0:i1+1]. Returns i0 if none finite."""
    if i1 < i0:
        return i0
    seg = arr[i0:i1 + 1]
    return i0 + (int(np.nanargmin(seg)) if np.isfinite(seg).any() else 0)


def _find_extrema(arr: np.ndarray, min_delta: float = 10.0):
    """
    Duyệt toàn bộ tín hiệu, tìm tất cả đỉnh (peak) và đáy (trough) theo thứ tự.
    Chỉ tính extremum khi tín hiệu đã đổi chiều ít nhất min_delta độ.
    Trả về list of (index, 'peak'|'trough').
    """
    filled = _fill(np.where(np.isfinite(arr), arr, np.nan))
    n = len(filled)
    extrema = []
    direction = 0          # +1 đang tăng, -1 đang giảm, 0 chưa xác định
    ref_idx = 0
    ref_val = float(filled[0])

    for i in range(1, n):
        v = float(filled[i])
        if direction == 0:
            if v - ref_val >= min_delta:
                direction = 1;  ref_idx, ref_val = i, v
            elif ref_val - v >= min_delta:
                direction = -1; ref_idx, ref_val = i, v
        elif direction == 1:          # đang tăng, theo dõi max
            if v >= ref_val:
                ref_idx, ref_val = i, v
            elif ref_val - v >= min_delta:
                extrema.append((ref_idx, 'peak'))
                direction = -1; ref_idx, ref_val = i, v
        else:                         # đang giảm, theo dõi min
            if v <= ref_val:
                ref_idx, ref_val = i, v
            elif v - ref_val >= min_delta:
                extrema.append((ref_idx, 'trough'))
                direction = 1;  ref_idx, ref_val = i, v
    return extrema


def _next_extremum(extrema, after: int, kind: str, fallback: int) -> int:
    """Trả về index của extremum đầu tiên có type=kind xuất hiện sau `after`."""
    for idx, t in extrema:
        if idx > after and t == kind:
            return idx
    return fallback


def _fill(arr: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN gaps; edge-fills beyond detected range."""
    arr = arr.copy().astype(float)
    finite = np.isfinite(arr)
    if not finite.any():
        return arr
    idx = np.arange(arr.size)
    arr[~finite] = np.interp(idx[~finite], idx[finite], arr[finite])
    return arr


def _smooth(arr: np.ndarray, w: int = 5) -> np.ndarray:
    """Median smoothing with window w."""
    out = arr.copy()
    half = max(1, w // 2)
    for i in range(arr.size):
        seg = arr[max(0, i - half):min(arr.size, i + half + 1)]
        seg = seg[np.isfinite(seg)]
        if seg.size:
            out[i] = float(np.median(seg))
    return out


def _midpoint_xy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise midpoint; NaN when either point is NaN."""
    out = np.full_like(a, np.nan)
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    out[valid] = (a[valid] + b[valid]) / 2.0
    return out


# ---------------------------------------------------------------------------
# Signal computation  ←  the 4 things we track
# ---------------------------------------------------------------------------

def _compute_signals(
    shaft_angles: List[Optional[float]],
    shaft_centers: List[Optional[Tuple[float, float]]],
    features: Dict[str, np.ndarray],
    n: int,
    body_scale: float,
) -> Dict[str, np.ndarray]:
    """Compute the four per-frame signals that drive phase detection.

    Signals (match exactly what is visible on the overlay video):

    1. shaft_angle  [0°, 90°]
           Angle of the club from horizontal.
           0° = shaft lies flat / horizontal.
           90° = shaft stands straight up / vertical.

    2. direction  (encoded as dir_up / dir_dn bool arrays)
           Whether the grip (yellow dot) is moving UP or DOWN.
           UP  = grip_y decreasing (hands rising).
           DOWN = grip_y increasing (hands falling).

    3. hip_xy, chest_xy, shoulder_xy  [n, 2]
           Coordinates of the three body-center points drawn on the video.

    4. grip_xy  [n, 2]
           Coordinates of the yellow grip dot (mid-wrist projected on shaft).
    """

    # ---- 1. Shaft angle ----
    def _to_horiz(a) -> float:
        v = float(abs(a) % 180.0)
        return min(v, 180.0 - v)   # [0°,180°) → [0°,90°]

    sa_raw = np.array([
        _to_horiz(a) if (a is not None and np.isfinite(float(a))) else np.nan
        for a in shaft_angles
    ], dtype=float)
    shaft_angle = _smooth(_fill(sa_raw), w=5)

    # ---- 2. Body center points ----
    left_shoulder  = features['left_shoulder']   # [n,2]
    right_shoulder = features['right_shoulder']
    left_hip       = features['left_hip']
    right_hip      = features['right_hip']
    left_wrist     = features['left_wrist']
    right_wrist    = features['right_wrist']

    shoulder_xy = _midpoint_xy(left_shoulder, right_shoulder)
    hip_xy      = _midpoint_xy(left_hip,      right_hip)
    chest_xy    = (2 * shoulder_xy + hip_xy) / 3  # P = (2M + N)/3, NP = 2·PM

    # ---- 3. Grip point = trung điểm H1, H2 ----
    # d1 = đường thẳng qua left_elbow → left_wrist
    # d2 = đường thẳng qua right_elbow → right_wrist
    # H1 = giao điểm d1 với shaft line
    # H2 = giao điểm d2 với shaft line
    # grip = (H1 + H2) / 2  (hoặc Hi nếu chỉ 1 tay detect được)
    def _intersect_forearm_shaft(elbow, wrist, cx, cy, vx, vy, max_dist):
        """Giao điểm của đường thẳng (elbow→wrist) với shaft line qua (cx,cy) hướng (vx,vy).
        Trả về None nếu giao điểm cách wrist > max_dist (tay không hướng về gậy).
        Nếu 2 đường song song, fallback chiếu vuông góc wrist lên shaft."""
        dx = wrist[0] - elbow[0]
        dy = wrist[1] - elbow[1]
        det = vx * dy - vy * dx
        if abs(det) < 1e-6:
            t = (wrist[0] - cx) * vx + (wrist[1] - cy) * vy
        else:
            t = (dx * (cy - elbow[1]) - dy * (cx - elbow[0])) / det
        H = np.array([cx + t * vx, cy + t * vy])
        if float(np.linalg.norm(H - wrist)) > max_dist:
            return None
        return H

    left_elbow  = features['left_elbow']
    right_elbow = features['right_elbow']
    _grip_max_dist = 0.2 * body_scale

    grip_xy = np.full((n, 2), np.nan)
    for i in range(n):
        lw = left_wrist[i];  le = left_elbow[i]
        rw = right_wrist[i]; re = right_elbow[i]
        l_ok = np.isfinite(lw).all() and np.isfinite(le).all()
        r_ok = np.isfinite(rw).all() and np.isfinite(re).all()
        if not l_ok and not r_ok:
            continue
        sc = shaft_centers[i]
        sa = float(shaft_angle[i])
        if sc is not None and np.isfinite(sa):
            cx, cy = float(sc[0]), float(sc[1])
            rad = math.radians(sa)
            vx, vy = math.cos(rad), math.sin(rad)
            H1 = _intersect_forearm_shaft(le, lw, cx, cy, vx, vy, _grip_max_dist) if l_ok else None
            H2 = _intersect_forearm_shaft(re, rw, cx, cy, vx, vy, _grip_max_dist) if r_ok else None
            if H1 is not None and H2 is not None:
                grip_xy[i] = (H1 + H2) / 2.0
            elif H1 is not None:
                grip_xy[i] = H1
            elif H2 is not None:
                grip_xy[i] = H2
            # else: cả 2 tay đều không hướng về gậy → NaN
        else:
            # Fallback khi không có shaft: dùng trung điểm wrist thô
            if l_ok and r_ok:
                grip_xy[i] = (lw + rw) / 2.0
            elif l_ok:
                grip_xy[i] = lw.copy()
            else:
                grip_xy[i] = rw.copy()

    grip_x = _smooth(_fill(grip_xy[:, 0]), w=5)
    grip_y = _smooth(_fill(grip_xy[:, 1]), w=5)
    grip_xy_smooth = np.column_stack([grip_x, grip_y])

    # ---- 4. Direction — UP/DOWN from grip Y movement (LAG-based) ----
    LAG = max(5, n // 15)
    grip_dy = np.zeros(n, dtype=float)
    for i in range(n):
        lo = max(0, i - LAG)
        hi = min(n - 1, i + LAG)
        if hi > lo:
            grip_dy[i] = (grip_y[hi] - grip_y[lo]) / (hi - lo)
    thr = max(0.3, 0.004 * body_scale)
    dir_up = grip_dy < -thr
    dir_dn = grip_dy > thr

    # ---- Hip angle: góc nhọn giữa đường hông và phương ngang [0°, 90°] ----
    hip_angle = np.full(n, np.nan)
    for i in range(n):
        lh = left_hip[i]
        rh = right_hip[i]
        if np.isfinite(lh).all() and np.isfinite(rh).all():
            dx = abs(rh[0] - lh[0])
            dy = abs(rh[1] - lh[1])
            hip_angle[i] = math.degrees(math.atan2(dy, dx))
    hip_angle = _smooth(_fill(hip_angle), w=5)

    # ---- Chest angle: góc nhọn giữa trục shoulder->chest và phương ngang [0°, 90°] ----
    chest_angle = np.full(n, np.nan)
    for i in range(n):
        sho = shoulder_xy[i]
        che = chest_xy[i]
        if np.isfinite(sho).all() and np.isfinite(che).all():
            dx = abs(che[0] - sho[0])
            dy = abs(che[1] - sho[1])
            chest_angle[i] = math.degrees(math.atan2(dy, dx))
    chest_angle = _smooth(_fill(chest_angle), w=5)

    return dict(
        shaft_angle=shaft_angle,
        shaft_angle_raw=sa_raw,
        dir_up=dir_up,
        dir_dn=dir_dn,
        grip_dy=grip_dy,
        hip_xy=hip_xy,
        chest_xy=chest_xy,
        shoulder_xy=shoulder_xy,
        grip_xy=grip_xy_smooth,
        hip_angle=hip_angle,
        chest_angle=chest_angle,
    )


def _axis_y_turn_angle_from_width(
    point_a: np.ndarray,
    point_b: np.ndarray,
    ref_idx: int,
    n: int,
) -> np.ndarray:
    """Unsigned Y-axis turn proxy from horizontal foreshortening.

    The reference width is measured at Address. As the segment turns around the
    vertical axis in a face-on view, its visible horizontal width shortens:
        angle_y = arccos(current_width_x / address_width_x)
    """
    out = np.full(n, np.nan)
    if ref_idx < 0 or ref_idx >= n:
        return out

    ref_a = point_a[ref_idx]
    ref_b = point_b[ref_idx]
    if not (np.isfinite(ref_a).all() and np.isfinite(ref_b).all()):
        return out

    ref_width = abs(float(ref_b[0] - ref_a[0]))
    if ref_width <= 1e-6:
        return out

    for i in range(n):
        a = point_a[i]
        b = point_b[i]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        curr_width = abs(float(b[0] - a[0]))
        ratio = max(0.0, min(1.0, curr_width / ref_width))
        out[i] = math.degrees(math.acos(ratio))
    out = _smooth(_fill(out), w=5)
    if 0 <= ref_idx < n and np.isfinite(out[ref_idx]):
        out[ref_idx] = 0.0
    return out


def _compute_y_axis_turn_angles(
    features: Dict[str, np.ndarray],
    n: int,
    ref_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hip_y_angle, chest_y_angle), both unsigned in degrees."""
    hip_y_angle = _axis_y_turn_angle_from_width(
        features["left_hip"],
        features["right_hip"],
        ref_idx,
        n,
    )
    chest_y_angle = _axis_y_turn_angle_from_width(
        features["left_shoulder"],
        features["right_shoulder"],
        ref_idx,
        n,
    )
    return hip_y_angle, chest_y_angle


def _apply_signed_turn_stage(
    angles: np.ndarray,
    n: int,
    p1: int,
    p5: int,
    p6: int,
) -> np.ndarray:
    """Convert unsigned face-on turn magnitude to a signed proxy.

    Width foreshortening gives only turn magnitude. We add sign from the swing
    stage: backswing and early downswing are negative, while the motion from
    mid-downswing onward is positive. The sign flips smoothly between P5 and P6.
    """
    out = angles.copy()
    if n <= 0:
        return out

    p1 = max(0, min(n - 1, int(p1)))
    p5 = max(p1, min(n - 1, int(p5)))
    p6 = max(p5, min(n - 1, int(p6)))

    control_idx = [0, p1, p5, p6, n - 1]
    control_sign = [0.0, 0.0, -1.0, 1.0, 1.0]
    sign = np.interp(np.arange(n, dtype=float), control_idx, control_sign)

    finite = np.isfinite(out)
    out[finite] = out[finite] * sign[finite]
    if 0 <= p1 < n and np.isfinite(out[p1]):
        out[p1] = 0.0
    return out


def _angle_delta_signed_deg(current: float, reference: float) -> float:
    delta = (float(current) - float(reference) + 180.0) % 360.0 - 180.0
    if delta > 90.0:
        delta -= 180.0
    elif delta < -90.0:
        delta += 180.0
    return delta


def _signed_segment_orientation_angles(
    point_a: np.ndarray,
    point_b: np.ndarray,
    ref_idx: int,
    n: int,
    lead_is_right: bool,
) -> np.ndarray:
    """Signed face-on rotation proxy from left-right segment orientation.

    This is an alternative to width-only foreshortening. We compare the
    left->right segment angle against Address and normalize the sign by the
    inferred lead side so backswing tends negative and follow-through positive.
    """
    out = np.full(n, np.nan)
    if ref_idx < 0 or ref_idx >= n:
        return out

    ref_a = point_a[ref_idx]
    ref_b = point_b[ref_idx]
    if not (np.isfinite(ref_a).all() and np.isfinite(ref_b).all()):
        return out

    ref_angle = math.degrees(math.atan2(float(ref_b[1] - ref_a[1]), float(ref_b[0] - ref_a[0])))
    sign_norm = -1.0 if lead_is_right else 1.0

    for i in range(n):
        a = point_a[i]
        b = point_b[i]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            continue
        curr_angle = math.degrees(math.atan2(float(b[1] - a[1]), float(b[0] - a[0])))
        out[i] = _angle_delta_signed_deg(curr_angle, ref_angle) * sign_norm

    out = _smooth(_fill(out), w=5)
    if 0 <= ref_idx < n and np.isfinite(out[ref_idx]):
        out[ref_idx] = 0.0
    return out


def _compute_signed_orientation_turn_angles(
    features: Dict[str, np.ndarray],
    n: int,
    ref_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    lead_is_right = bool(features.get("lead_is_right", np.array([True]))[0])
    hip_y_angle = _signed_segment_orientation_angles(
        features["left_hip"],
        features["right_hip"],
        ref_idx,
        n,
        lead_is_right,
    )
    chest_y_angle = _signed_segment_orientation_angles(
        features["left_shoulder"],
        features["right_shoulder"],
        ref_idx,
        n,
        lead_is_right,
    )
    return hip_y_angle, chest_y_angle


def _build_toggle_direction_labels(
    n: int,
    turn_points: List[int],
    initial_label: str = "UP",
) -> np.ndarray:
    labels = np.full(n, initial_label, dtype=object)
    current_label = initial_label
    start_idx = 0
    for turn_idx in turn_points:
        turn_idx = int(max(0, min(n - 1, turn_idx)))
        labels[start_idx:turn_idx + 1] = current_label
        current_label = "DOWN" if current_label == "UP" else "UP"
        start_idx = turn_idx + 1
    if start_idx < n:
        labels[start_idx:] = current_label
    return labels


def _labels_from_slope(
    values: np.ndarray,
    threshold: float,
    up_when_decreasing: bool,
) -> np.ndarray:
    n = len(values)
    labels = np.full(n, "", dtype=object)
    if n <= 1:
        return labels
    lag = max(2, n // 20)
    for i in range(n):
        lo = max(0, i - lag)
        hi = min(n - 1, i + lag)
        if hi <= lo:
            continue
        delta = float(values[hi] - values[lo])
        if abs(delta) < threshold:
            continue
        if up_when_decreasing:
            labels[i] = "UP" if delta < 0.0 else "DOWN"
        else:
            labels[i] = "DOWN" if delta > 0.0 else "UP"
    return labels


def _stabilize_direction_labels(labels: np.ndarray, min_run: int = 3) -> np.ndarray:
    out = labels.copy()
    n = len(out)
    if n == 0:
        return out
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if j - i < min_run:
            left = out[i - 1] if i > 0 else None
            right = out[j] if j < n else None
            fill = left or right or out[i]
            out[i:j] = fill
        i = j
    return out


def _build_fused_direction_labels(
    sa_raw: np.ndarray,
    grip_y: np.ndarray,
    lead_grip_angle: np.ndarray,
    p5: int,
    threshold: float,
) -> np.ndarray:
    n = len(sa_raw)
    shaft_signal = _smooth(_fill(np.where(np.isfinite(sa_raw), sa_raw, np.nan)), w=7)
    shaft_extrema = _find_extrema(shaft_signal, min_delta=max(8.0, threshold * 0.5))
    shaft_turns = [int(idx) for idx, _kind in shaft_extrema if 0 <= int(idx) < n]
    shaft_labels = _build_toggle_direction_labels(n, shaft_turns, initial_label="UP")

    grip_labels = _labels_from_slope(grip_y, threshold=max(6.0, np.nanstd(grip_y) * 0.05), up_when_decreasing=True)

    fused = np.full(n, "UP", dtype=object)
    prev = "UP"
    for i in range(n):
        votes = [shaft_labels[i]]
        if grip_labels[i]:
            votes.append(grip_labels[i])
        up_votes = sum(1 for item in votes if item == "UP")
        down_votes = sum(1 for item in votes if item == "DOWN")
        if up_votes > down_votes:
            prev = "UP"
        elif down_votes > up_votes:
            prev = "DOWN"
        fused[i] = prev
    return _stabilize_direction_labels(fused, min_run=3)


def _next_extremum_with_direction(
    extrema,
    after: int,
    kind: str,
    fallback: int,
    direction_labels: np.ndarray,
    required_direction: str,
    radius: int = 1,
) -> int:
    for idx, t in extrema:
        if idx <= after or t != kind:
            continue
        lo = max(0, idx - radius)
        hi = min(len(direction_labels), idx + radius + 1)
        window = direction_labels[lo:hi]
        if any(label == required_direction for label in window):
            return idx
    return fallback


# ---------------------------------------------------------------------------
# Phase detection
# ---------------------------------------------------------------------------

PHASE_NAMES = [
    'Address', 'Mid-backswing', 'Late-backswing', 'Top-backswing',
    'Early-downswing', 'Mid-downswing', 'Ball-impact',
    'Mid-followthrough', 'Late-followthrough',
]


def compute_body_signals(
    frames: List[FramePose],
    swing_direction: Optional[str],
    shaft_angles: List[Optional[float]],
    shaft_centers: List[Optional[Tuple[float, float]]],
    ref_idx: int = 0,
) -> Dict[str, np.ndarray]:
    """Compute per-frame body/club signals without running phase detection."""
    if not frames:
        return {}

    features = compute_swing_features(frames, swing_direction=swing_direction)
    n = min(
        len(frames),
        len(shaft_angles),
        len(shaft_centers),
        *(len(v) for v in features.values() if hasattr(v, "__len__")),
    )
    if n <= 0:
        return {}

    shaft_angles = shaft_angles[:n]
    shaft_centers = shaft_centers[:n]
    features = {k: v[:n] for k, v in features.items() if hasattr(v, "__len__")}

    body_scale = _estimate_body_scale(features)
    sig = _compute_signals(shaft_angles, shaft_centers, features, n, body_scale)
    body_signals = {
        "hip_xy": sig["hip_xy"],
        "chest_xy": sig["chest_xy"],
        "shoulder_xy": sig["shoulder_xy"],
        "grip_xy": sig["grip_xy"],
        "hip_angle": sig["hip_angle"],
        "chest_angle": sig["chest_angle"],
        "lead_is_right": bool(features.get("lead_is_right", np.array([True]))[0]),
    }

    ref_idx = max(0, min(n - 1, int(ref_idx)))
    hip_y_angle_legacy, chest_y_angle_legacy = _compute_y_axis_turn_angles(features, n, ref_idx)
    hip_y_angle_experimental, chest_y_angle_experimental = _compute_signed_orientation_turn_angles(
        features,
        n,
        ref_idx,
    )
    body_signals["hip_y_angle"] = hip_y_angle_legacy
    body_signals["chest_y_angle"] = chest_y_angle_legacy
    body_signals["hip_y_angle_legacy"] = hip_y_angle_legacy
    body_signals["chest_y_angle_legacy"] = chest_y_angle_legacy
    body_signals["hip_y_angle_experimental"] = hip_y_angle_experimental
    body_signals["chest_y_angle_experimental"] = chest_y_angle_experimental
    return body_signals


def detect_swing_phases(
    frames: List[FramePose],
    fps: float,
    swing_direction: Optional[str],
    shaft_angles: List[Optional[float]],
    club_centers: List[Optional[Tuple[float, float]]],
    shaft_centers: List[Optional[Tuple[float, float]]],
    debug: bool = False,
    debug_path: str = "phase_debug.csv",
) -> List[Dict[str, float]]:
    """Detect golf swing phases sequentially."""
    if not frames:
        return [], {}

    features = compute_swing_features(frames, swing_direction=swing_direction)
    frame_ids = [f.frame_id for f in frames]
    times     = [f.t       for f in frames]

    n = min(len(frames), len(shaft_angles), len(shaft_centers),
            *(len(v) for v in features.values() if hasattr(v, '__len__')))
    if n <= 0:
        return [], {}

    frames        = frames[:n]
    frame_ids     = frame_ids[:n]
    times         = times[:n]
    shaft_angles  = shaft_angles[:n]
    shaft_centers = shaft_centers[:n]
    features      = {k: v[:n] for k, v in features.items() if hasattr(v, '__len__')}

    body_scale = _estimate_body_scale(features)
    sig = _compute_signals(shaft_angles, shaft_centers, features, n, body_scale)

    shaft_angle = sig['shaft_angle']      # [0°,90°]
    sa_raw      = sig['shaft_angle_raw']
    dir_up      = sig['dir_up']
    dir_dn      = sig['dir_dn']
    grip_dy     = sig['grip_dy']
    grip_xy     = sig['grip_xy']
    hip_xy      = sig['hip_xy']
    chest_xy    = sig['chest_xy']
    shoulder_xy = sig['shoulder_xy']
    hip_angle   = sig['hip_angle']
    chest_angle = sig['chest_angle']

    body_signals = compute_body_signals(
        frames,
        swing_direction=swing_direction,
        shaft_angles=shaft_angles,
        shaft_centers=shaft_centers,
        ref_idx=0,
    )

    # Short clips are not meaningful for 9-phase detection, but the pipeline
    # should still return a stable shape for smoke tests / truncated runs.
    if n < 3:
        hip_y_angle_legacy, chest_y_angle_legacy = _compute_y_axis_turn_angles(features, n, 0)
        hip_y_angle_experimental, chest_y_angle_experimental = _compute_signed_orientation_turn_angles(features, n, 0)
        hip_y_angle = hip_y_angle_legacy
        chest_y_angle = chest_y_angle_legacy
        body_signals['hip_y_angle'] = hip_y_angle
        body_signals['chest_y_angle'] = chest_y_angle
        body_signals['hip_y_angle_legacy'] = hip_y_angle_legacy
        body_signals['chest_y_angle_legacy'] = chest_y_angle_legacy
        body_signals['hip_y_angle_experimental'] = hip_y_angle_experimental
        body_signals['chest_y_angle_experimental'] = chest_y_angle_experimental
        events = []
        for k, name in enumerate(PHASE_NAMES):
            idx = min(k, n - 1)
            events.append({'name': name, 'frame': int(frame_ids[idx]), 't': float(times[idx])})
        return events, body_signals

    # Derived: distance from horizontal / vertical — use RAW angle for phase detection
    # (smoothing can suppress true peaks/troughs; raw values reflect actual shaft position)
    horiz_sc = np.where(np.isfinite(sa_raw), sa_raw,        np.inf)
    vert_sc  = np.where(np.isfinite(sa_raw), 90.0 - sa_raw, np.inf)

    # Grip speed — derived from grip_xy, used only for P1
    grip_spd = np.zeros(n, dtype=float)
    for i in range(1, n):
        grip_spd[i] = float(np.linalg.norm(grip_xy[i] - grip_xy[i - 1]))
    grip_spd = _smooth(grip_spd, w=5)

    # Two-hand grip geometry for Address detection
    _lw = features['left_wrist']
    _rw = features['right_wrist']
    wrist_grip_dist_l = np.full(n, np.inf)
    wrist_grip_dist_r = np.full(n, np.inf)
    wrist_shaft_perp_l = np.full(n, np.inf)
    wrist_shaft_perp_r = np.full(n, np.inf)
    wrist_shaft_along_l = np.full(n, np.inf)
    wrist_shaft_along_r = np.full(n, np.inf)
    grip_below_hip = np.zeros(n, dtype=bool)
    for i in range(n):
        gp = grip_xy[i]
        hp = hip_xy[i]
        lw = _lw[i]
        rw = _rw[i]
        if np.isfinite(gp).all() and np.isfinite(hp).all():
            grip_below_hip[i] = bool(gp[1] >= hp[1] + 0.08 * body_scale)
        if not np.isfinite(gp).all():
            continue
        if np.isfinite(lw).all():
            wrist_grip_dist_l[i] = float(np.linalg.norm(lw - gp))
        if np.isfinite(rw).all():
            wrist_grip_dist_r[i] = float(np.linalg.norm(rw - gp))

        sc = shaft_centers[i]
        if sc is None:
            continue
        shaft_ctr = np.asarray(sc, dtype=float)
        if not np.isfinite(shaft_ctr).all():
            continue
        shaft_vec = shaft_ctr - gp
        shaft_norm = float(np.linalg.norm(shaft_vec))
        if shaft_norm <= 1e-6:
            continue
        shaft_unit = shaft_vec / shaft_norm
        normal_unit = np.array([-shaft_unit[1], shaft_unit[0]], dtype=float)
        if np.isfinite(lw).all():
            rel_l = lw - gp
            wrist_shaft_perp_l[i] = float(abs(np.dot(rel_l, normal_unit)))
            wrist_shaft_along_l[i] = float(abs(np.dot(rel_l, shaft_unit)))
        if np.isfinite(rw).all():
            rel_r = rw - gp
            wrist_shaft_perp_r[i] = float(abs(np.dot(rel_r, normal_unit)))
            wrist_shaft_along_r[i] = float(abs(np.dot(rel_r, shaft_unit)))

    # ===== P1: Address — last stable frame before swing starts =====
    s_end  = max(3, min(n - 1, int(n * 0.35)))
    grip_y = grip_xy[:, 1]
    base_y = np.nanmedian(grip_y[:max(3, s_end // 5)])
    disp   = np.abs(grip_y - base_y)
    spd_thr = float(np.nanpercentile(grip_spd[:s_end + 1], 70)) \
              if np.isfinite(grip_spd[:s_end + 1]).any() else 0.0
    onset = s_end
    for i in range(1, s_end + 1):
        fast = sum(1 for j in range(i, min(s_end + 1, i + 3)) if grip_spd[j] >= spd_thr)
        if fast >= 2 and disp[i] >= max(5.0, 0.05 * body_scale):
            onset = i
            break
    p1_cost = (0.6 * _normalize_cost(disp.tolist()) +
               0.4 * _normalize_cost(grip_spd.tolist()))
    p1 = _argmin_range(p1_cost, 0, max(1, onset))

    # Address candidates: grip below hip, both wrists near the shaft, and both wrists near grip.
    _TH_WRIST_TO_SHAFT_PERP = 0.08 * body_scale
    _TH_WRIST_TO_SHAFT_ALONG = 0.22 * body_scale
    _TH_WRIST_TO_GRIP = 0.22 * body_scale
    address_mask = np.zeros(n, dtype=bool)
    for i in range(min(onset + 1, n)):
        dl = wrist_grip_dist_l[i]
        dr = wrist_grip_dist_r[i]
        if not grip_below_hip[i]:
            continue
        if not np.isfinite(dl) or not np.isfinite(dr):
            continue
        if dl > _TH_WRIST_TO_GRIP or dr > _TH_WRIST_TO_GRIP:
            continue
        ratio = max(dl, dr) / max(min(dl, dr), 1e-6)
        if ratio > 1.35:
            continue
        if wrist_shaft_perp_l[i] > _TH_WRIST_TO_SHAFT_PERP or wrist_shaft_perp_r[i] > _TH_WRIST_TO_SHAFT_PERP:
            continue
        if wrist_shaft_along_l[i] > _TH_WRIST_TO_SHAFT_ALONG or wrist_shaft_along_r[i] > _TH_WRIST_TO_SHAFT_ALONG:
            continue
        address_mask[i] = True

    # Require a short consecutive stable run; choose the last stable frame before onset.
    best_run = None
    run_start = None
    _MIN_ADDRESS_RUN = 4
    for i in range(min(onset + 1, n)):
        if address_mask[i]:
            if run_start is None:
                run_start = i
        elif run_start is not None:
            run_end = i - 1
            if run_end - run_start + 1 >= _MIN_ADDRESS_RUN:
                best_run = (run_start, run_end)
            run_start = None
    if run_start is not None:
        run_end = min(onset, n - 1)
        if run_end - run_start + 1 >= _MIN_ADDRESS_RUN:
            best_run = (run_start, run_end)

    if best_run is not None:
        run_start, run_end = best_run
        stable_idxs = list(range(run_start, run_end + 1))
        stable_cost = (
            0.55 * p1_cost[stable_idxs]
            + 0.20 * _normalize_cost(wrist_grip_dist_l[stable_idxs].tolist())
            + 0.20 * _normalize_cost(wrist_grip_dist_r[stable_idxs].tolist())
            + 0.05 * _normalize_cost(grip_spd[stable_idxs].tolist())
        )
        best_local = int(np.argmin(stable_cost))
        p1 = stable_idxs[best_local]
    else:
        # If the strict rule finds no stable cluster, keep the original onset-based fallback
        # instead of snapping to a weak single-frame candidate.
        pass

    hip_y_angle_unsigned, chest_y_angle_unsigned = _compute_y_axis_turn_angles(features, n, p1)
    body_signals['hip_y_angle_legacy'] = hip_y_angle_unsigned
    body_signals['chest_y_angle_legacy'] = chest_y_angle_unsigned

    # Duyệt toàn bộ tín hiệu một lần, tìm tất cả đỉnh/đáy theo thứ tự
    _THR = 15.0
    sa_extrema = _find_extrema(sa_raw, min_delta=_THR)

    # ===== P2: Mid-backswing — đáy đầu tiên của sa_raw sau P1 =====
    p2 = max(_next_extremum(sa_extrema, p1, 'trough', p1 + 1), p1 + 1)

    # ===== P3: Late-backswing — đỉnh đầu tiên của sa_raw sau P2 =====
    p3 = max(_next_extremum(sa_extrema, p2, 'peak', p2 + 1), p2 + 1)

    # ===== P4: Top-backswing — đáy đầu tiên của grip_y sau P3 =====
    # grip_y dùng pixel nên dùng riêng (không phải độ)
    _Y_THR = max(15.0, 0.05 * body_scale)
    gy_extrema = _find_extrema(grip_y, min_delta=_Y_THR)
    p4 = max(_next_extremum(gy_extrema, p3, 'trough', p3 + 1), p3 + 1)

    # ===== Lead shoulder (cho P5) =====
    lead_is_right = bool(features.get('lead_is_right', np.array([True]))[0])
    _ls_raw = features['right_shoulder'] if lead_is_right else features['left_shoulder']
    lead_shoulder_xy = np.column_stack([
        _smooth(_fill(_ls_raw[:, 0]), w=5),
        _smooth(_fill(_ls_raw[:, 1]), w=5),
    ])

    # Lead-grip angle per frame [0°=horizontal, 90°=vertical]
    lead_grip_angle = np.full(n, np.nan)
    for i in range(n):
        ls = lead_shoulder_xy[i]
        gp = grip_xy[i]
        if np.isfinite(ls).all() and np.isfinite(gp).all():
            dx = abs(gp[0] - ls[0])
            dy = abs(gp[1] - ls[1])
            lead_grip_angle[i] = math.degrees(math.atan2(dy, dx + 1e-6))

    lg_extrema = _find_extrema(lead_grip_angle, min_delta=_THR)

    # ===== P5: Early-downswing — đáy đầu tiên của lead_grip_angle sau P4 =====
    p5 = max(_next_extremum(lg_extrema, p4, 'trough', p4 + 1), p4 + 1)

    direction_labels = _build_fused_direction_labels(sa_raw, grip_y, lead_grip_angle, p5, _THR)

    # ===== P6: Mid-downswing — đáy đầu tiên của sa_raw sau đỉnh đầu tiên sau P5 =====
    # Sau P5, gậy tăng lên đỉnh (đỉnh này nằm trước P6), rồi mới giảm xuống nằm ngang (P6).
    # Vì vậy cần bỏ qua đỉnh đầu tiên sau P5, rồi lấy đáy tiếp theo.
    _p6_peak = _next_extremum_with_direction(
        sa_extrema, p5, 'peak', p5, direction_labels, required_direction='DOWN'
    )
    p6 = max(
        _next_extremum_with_direction(
            sa_extrema, _p6_peak, 'trough', _p6_peak + 1, direction_labels, required_direction='DOWN'
        ),
        p5 + 1,
    )

    # ===== P7: Ball-impact — đỉnh đầu tiên của sa_raw sau P6 =====
    p7 = max(
        _next_extremum_with_direction(
            sa_extrema, p6, 'peak', p6 + 1, direction_labels, required_direction='DOWN'
        ),
        p6 + 1,
    )

    # ===== P8: Mid-followthrough — đáy đầu tiên của sa_raw sau P7 =====
    p8 = max(
        _next_extremum_with_direction(
            sa_extrema, p7, 'trough', p7 + 1, direction_labels, required_direction='UP'
        ),
        p7 + 1,
    )

    # ===== P9: Late-followthrough — đỉnh đầu tiên của sa_raw sau P8 =====
    p9 = max(
        _next_extremum_with_direction(
            sa_extrema, p8, 'peak', p8 + 1, direction_labels, required_direction='UP'
        ),
        p8 + 1,
    )

    hip_y_angle_legacy = _apply_signed_turn_stage(hip_y_angle_unsigned, n, p1, p5, p6)
    chest_y_angle_legacy = _apply_signed_turn_stage(chest_y_angle_unsigned, n, p1, p5, p6)
    hip_y_angle_experimental, chest_y_angle_experimental = _compute_signed_orientation_turn_angles(features, n, p1)
    hip_y_angle = hip_y_angle_legacy
    chest_y_angle = chest_y_angle_legacy
    body_signals['hip_y_angle'] = hip_y_angle
    body_signals['chest_y_angle'] = chest_y_angle
    body_signals['hip_y_angle_legacy'] = hip_y_angle_legacy
    body_signals['chest_y_angle_legacy'] = chest_y_angle_legacy
    body_signals['hip_y_angle_experimental'] = hip_y_angle_experimental
    body_signals['chest_y_angle_experimental'] = chest_y_angle_experimental

    idxs = [p1, p2, p3, p4, p5, p6, p7, p8, p9]

    # ===== Debug CSV =====
    if debug:
        def _xy(arr, i):
            x, y = arr[i]
            if np.isfinite(x) and np.isfinite(y):
                return f"{x:.1f}", f"{y:.1f}"
            return '', ''

        try:
            import os as _os
            _os.makedirs(_os.path.dirname(_os.path.abspath(debug_path)), exist_ok=True)
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write('frame_id,'
                        'shaft_angle_raw,shaft_angle,'
                        'direction,'
                        'hip_x,hip_y,'
                        'chest_x,chest_y,'
                        'shoulder_x,shoulder_y,'
                        'grip_x,grip_y,'
                        'lead_grip_angle,'
                        'hip_y_angle,chest_y_angle,'
                        'phase\n')
                phase_at = {idxs[k]: f"P{k+1}-{PHASE_NAMES[k]}" for k in range(len(idxs))}
                for i in range(n):
                    d   = str(direction_labels[i])
                    ph  = phase_at.get(i, '')
                    raw = f"{sa_raw[i]:.1f}" if np.isfinite(sa_raw[i]) else ''
                    sa  = f"{shaft_angle[i]:.1f}" if np.isfinite(shaft_angle[i]) else ''
                    hx, hy = _xy(hip_xy,      i)
                    cx, cy = _xy(chest_xy,    i)
                    sx, sy = _xy(shoulder_xy, i)
                    gx, gy = _xy(grip_xy,     i)
                    lg = f"{lead_grip_angle[i]:.1f}" if np.isfinite(lead_grip_angle[i]) else ''
                    hya = f"{hip_y_angle[i]:.1f}" if np.isfinite(hip_y_angle[i]) else ''
                    cya = f"{chest_y_angle[i]:.1f}" if np.isfinite(chest_y_angle[i]) else ''
                    f.write(f"{frame_ids[i]},{raw},{sa},{d},{hx},{hy},{cx},{cy},{sx},{sy},{gx},{gy},{lg},{hya},{cya},{ph}\n")
        except Exception as e:
            print(f"[WARN] phase debug write failed: {e}")

    events = []
    for k, idx in enumerate(idxs):
        idx = int(max(0, min(int(idx), n - 1)))
        events.append({'name': PHASE_NAMES[k], 'frame': int(frame_ids[idx]), 't': float(times[idx])})

<<<<<<< HEAD
    body_signals = {
        'hip_xy':      hip_xy,
        'chest_xy':    chest_xy,
        'shoulder_xy': shoulder_xy,
        'grip_xy':     grip_xy,
        'hip_angle':   hip_angle,
        'chest_angle': chest_angle,
    }
=======
>>>>>>> f21225886234271dcf8b7a859d5518995d89d006
    return events, body_signals


# Alias for backward compatibility with pipeline.py import
detect_events_rule9 = detect_swing_phases
