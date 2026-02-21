"""Rule-based swing event detectors extracted from original script."""
from typing import Dict, List, Optional, Tuple
import math
import numpy as np

from .events import FramePose, safe_point, moving_average, compute_swing_features


def detect_events_rule(frames: List[FramePose], fps: float) -> List[Dict[str, float]]:
    if not frames:
        return []

    wrist_y = []
    hip_y = []
    for frame in frames:
        kpts = frame.keypoints
        lw = safe_point(kpts, 9)
        rw = safe_point(kpts, 10)
        lh = safe_point(kpts, 11)
        rh = safe_point(kpts, 12)
        if lw is None and rw is None:
            wrist_y.append(None)
        else:
            y = []
            if lw is not None:
                y.append(lw[1])
            if rw is not None:
                y.append(rw[1])
            wrist_y.append(float(sum(y) / len(y)))
        if lh is None and rh is None:
            hip_y.append(None)
        else:
            y = []
            if lh is not None:
                y.append(lh[1])
            if rh is not None:
                y.append(rh[1])
            hip_y.append(float(sum(y) / len(y)))

    filled_wrist = [v for v in wrist_y if v is not None]
    if not filled_wrist:
        return []
    fallback = sum(filled_wrist) / len(filled_wrist)
    wrist_y = [fallback if v is None else v for v in wrist_y]
    hip_filled = [v for v in hip_y if v is not None]
    hip_fallback = sum(hip_filled) / len(hip_filled) if hip_filled else fallback
    hip_y = [hip_fallback if v is None else v for v in hip_y]

    wrist_y = moving_average(wrist_y, window=7)
    speeds = [0.0]
    for i in range(1, len(wrist_y)):
        speeds.append(abs(wrist_y[i] - wrist_y[i - 1]))

    # Address: first low-motion frame in the first 20%
    addr_idx = 0
    first_window = max(1, int(len(frames) * 0.2))
    speed_sorted = sorted(speeds[:first_window])
    speed_thr = speed_sorted[max(0, len(speed_sorted) // 3)]
    for i in range(first_window):
        if speeds[i] <= speed_thr:
            addr_idx = i
            break

    # Top: highest hands (min y)
    top_idx = int(np.argmin(wrist_y))
    if top_idx < addr_idx:
        top_idx = max(addr_idx, top_idx)

    # Impact: after top, hands near hip height while descending
    impact_idx = top_idx
    best = math.inf
    for i in range(top_idx + 1, len(frames)):
        if wrist_y[i] >= wrist_y[i - 1]:
            diff = abs(wrist_y[i] - hip_y[i])
            if diff < best:
                best = diff
                impact_idx = i

    # Finish: last low-motion frame in final 20%
    finish_idx = len(frames) - 1
    last_window = max(1, int(len(frames) * 0.2))
    speed_sorted = sorted(speeds[-last_window:])
    speed_thr = speed_sorted[max(0, len(speed_sorted) // 3)]
    for i in range(len(frames) - last_window, len(frames)):
        if speeds[i] <= speed_thr:
            finish_idx = i
            break

    def _interp(a: int, b: int, ratio: float) -> int:
        return int(round(a + (b - a) * ratio))

    toe_up_idx = _interp(addr_idx, top_idx, 0.3)
    mid_backswing_idx = _interp(addr_idx, top_idx, 0.6)
    mid_downswing_idx = _interp(top_idx, impact_idx, 0.5)
    mid_follow_idx = _interp(impact_idx, finish_idx, 0.5)

    events = [
        ("address", addr_idx),
        ("toe-up", toe_up_idx),
        ("mid-backswing", mid_backswing_idx),
        ("top", top_idx),
        ("mid-downswing", mid_downswing_idx),
        ("impact", impact_idx),
        ("mid-follow-through", mid_follow_idx),
        ("finish", finish_idx),
    ]

    out = []
    for name, idx in events:
        idx = max(0, min(len(frames) - 1, idx))
        out.append({"name": name, "frame": idx, "t": idx / fps})
    return out


# The long-form rule9 detector remains mostly verbatim for correctness.

def detect_events_rule9(
    frames: List[FramePose],
    fps: float,
    swing_direction: Optional[str],
    shaft_angles: List[Optional[float]],
    club_centers: List[Optional[Tuple[float, float]]],
    shaft_centers: List[Optional[Tuple[float, float]]],
) -> List[Dict[str, float]]:
    if not frames:
        return []
    n = len(frames)
    features = compute_swing_features(frames, swing_direction=swing_direction)
    frame_ids = [f.frame_id for f in frames]
    times = [f.t for f in frames]

    def _interpolate_angles(angles: List[Optional[float]]) -> List[float]:
        out = [float("nan")] * n
        for i in range(n):
            ang = angles[i]
            if ang is None or not np.isfinite(ang):
                continue
            out[i] = float(abs(ang) % 180.0)
        valid = [i for i, v in enumerate(out) if np.isfinite(v)]
        if not valid:
            return out
        for i in range(n):
            if np.isfinite(out[i]):
                continue
            prev = max([j for j in valid if j < i], default=None)
            nxt = min([j for j in valid if j > i], default=None)
            if prev is None:
                out[i] = out[nxt]
            elif nxt is None:
                out[i] = out[prev]
            else:
                ratio = (i - prev) / (nxt - prev)
                out[i] = out[prev] + (out[nxt] - out[prev]) * ratio
        return out

    shaft_angles_interp = _interpolate_angles(shaft_angles)

    def _horiz_score(i: int) -> float:
        ang = shaft_angles_interp[i]
        if not np.isfinite(ang):
            return float(features["forearm_horiz"][i])
        return min(ang, 180.0 - ang)

    def _horiz_score_from_angle(ang: float) -> float:
        return min(ang, 180.0 - ang)

    def _speed(coords: List[Optional[Tuple[float, float]]]) -> List[float]:
        speeds = [0.0] * n
        prev = None
        for i in range(n):
            cur = coords[i]
            if cur is None:
                cur = prev
            if prev is None or cur is None:
                speeds[i] = speeds[i - 1] if i > 0 else 0.0
            else:
                speeds[i] = float(np.linalg.norm(np.array(cur) - np.array(prev)))
            prev = cur
        return speeds

    club_speed = _speed(club_centers)
    wrist_coords = []
    for i in range(n):
        w = features["wrist"][i]
        if np.any(np.isnan(w)):
            wrist_coords.append(None)
        else:
            wrist_coords.append((float(w[0]), float(w[1])))
    wrist_speed = _speed(wrist_coords)
    speed = [0.7 * cs + 0.3 * ws for cs, ws in zip(club_speed, wrist_speed)]

    first_window = max(1, n // 5)
    speed_thr = np.quantile(speed[:first_window], 0.3)
    p1 = 0
    for i in range(first_window):
        if speed[i] <= speed_thr:
            p1 = i
            break

    def _pick_min(start: int, end: int, values: List[float]) -> int:
        if end <= start:
            return start
        segment = values[start:end]
        return int(start + np.argmin(segment))

    def _pick_max(start: int, end: int, values: List[float]) -> int:
        if end <= start:
            return start
        segment = values[start:end]
        return int(start + np.argmax(segment))

    wrist_y = []
    for i in range(n):
        lw = features["left_wrist"][i]
        rw = features["right_wrist"][i]
        if np.any(np.isnan(lw)) or np.any(np.isnan(rw)):
            if np.any(np.isnan(features["wrist"][i])):
                wrist_y.append(float("inf"))
            else:
                wrist_y.append(float(features["wrist"][i][1]))
        else:
            wrist_y.append((float(lw[1]) + float(rw[1])) / 2.0)

    coarse_top = _pick_min(p1 + 1, max(p1 + 2, n // 2), wrist_y)

    horiz_scores = [_horiz_score(i) for i in range(n)]
    p2 = _pick_min(p1 + 1, coarse_top, horiz_scores)

    vert_scores = []
    for i in range(n):
        ang = shaft_angles_interp[i]
        if not np.isfinite(ang):
            vert_scores.append(float("inf"))
        else:
            vert_scores.append(abs(ang - 90.0))
    p3 = _pick_min(p2 + 1, coarse_top, vert_scores)

    top_start = min(n - 2, p3 + 1)
    top_end = max(top_start + 1, n // 2)
    p4 = _pick_min(top_start, top_end, wrist_y)

    club_x = []
    club_y = []
    for i in range(n):
        c = club_centers[i]
        if c is None:
            club_x.append(np.nan)
            club_y.append(np.nan)
        else:
            club_x.append(c[0])
            club_y.append(c[1])
    if not np.all(np.isnan(club_x)):
        window = max(2, n // 20)
        start = max(top_start, p4 - window)
        end = min(top_end, p4 + window)
        vx = np.diff(club_x)
        sign = np.sign(vx)
        cand = None
        for i in range(start, end):
            if i <= 0 or i >= len(sign):
                continue
            if sign[i - 1] == 0 or sign[i] == 0:
                continue
            if sign[i - 1] != sign[i]:
                cand = i
                break
        if cand is not None:
            p4 = cand
    elif not np.all(np.isnan(club_y)):
        p4 = int(np.nanargmin(club_y))
    if p4 <= p1:
        p4 = min(n - 1, p1 + n // 4)

    wrist_hip = [
        abs(float(features["wrist"][i][1]) - float(features["hip_y"][i]))
        if not np.any(np.isnan(features["wrist"][i])) and not np.isnan(features["hip_y"][i])
        else float("inf")
        for i in range(n)
    ]

    address_club = club_centers[p1]
    address_wrist = features["wrist"][p1]
    if np.any(np.isnan(address_wrist)):
        address_wrist = None
    ankle_mid = features["ankle_mid"]
    left_ankle = features["left_ankle"]
    right_ankle = features["right_ankle"]
    ankle_target = None
    if not np.all(np.isnan(ankle_mid)):
        ankle_target = np.nanmedian(ankle_mid, axis=0)

    def _ankle_band(i: int) -> Optional[Tuple[float, float]]:
        la = left_ankle[i]
        ra = right_ankle[i]
        if np.any(np.isnan(la)) or np.any(np.isnan(ra)):
            return None
        xmin = float(min(la[0], ra[0]))
        xmax = float(max(la[0], ra[0]))
        stance = max(1.0, xmax - xmin)
        pad = 0.12 * stance
        return xmin - pad, xmax + pad

    def _between_ankle_penalty(i: int, point: np.ndarray) -> float:
        band = _ankle_band(i)
        if band is None:
            return 0.0
        xmin, xmax = band
        x = float(point[0])
        if x < xmin:
            return 220.0 + (xmin - x) * 4.0
        if x > xmax:
            return 220.0 + (x - xmax) * 4.0
        return 0.0

    def _wrists_in_ankle_band(i: int) -> bool:
        band = _ankle_band(i)
        if band is None:
            return True
        lw = features["left_wrist"][i]
        rw = features["right_wrist"][i]
        if np.any(np.isnan(lw)) or np.any(np.isnan(rw)):
            return False
        xmin, xmax = band
        lx = float(lw[0])
        rx = float(rw[0])
        return xmin <= lx <= xmax and xmin <= rx <= xmax

    def _wrists_in_impact_zone(i: int) -> bool:
        if not _wrists_in_ankle_band(i):
            return False

        lw = features["left_wrist"][i]
        rw = features["right_wrist"][i]
        if np.any(np.isnan(lw)) or np.any(np.isnan(rw)):
            return False

        ls = features["left_shoulder"][i]
        rs = features["right_shoulder"][i]
        y_top = None
        shoulder_ys = []
        if not np.any(np.isnan(ls)):
            shoulder_ys.append(float(ls[1]))
        if not np.any(np.isnan(rs)):
            shoulder_ys.append(float(rs[1]))
        if shoulder_ys:
            y_top = min(shoulder_ys)

        la = left_ankle[i]
        ra = right_ankle[i]
        y_bottom = None
        ankle_ys = []
        if not np.any(np.isnan(la)):
            ankle_ys.append(float(la[1]))
        if not np.any(np.isnan(ra)):
            ankle_ys.append(float(ra[1]))
        if ankle_ys:
            y_bottom = max(ankle_ys)
        elif ankle_target is not None:
            y_bottom = float(ankle_target[1])

        if y_top is None or y_bottom is None or y_bottom <= y_top:
            return True

        margin = 0.08 * (y_bottom - y_top)
        for wy in (float(lw[1]), float(rw[1])):
            if wy < y_top - margin or wy > y_bottom + margin:
                return False
        return True

    def _impact_score(i: int) -> float:
        c = club_centers[i]
        if c is not None:
            score = 0.0
            if address_club is not None:
                score += float(np.linalg.norm(np.array(c) - np.array(address_club)))
            am = ankle_mid[i]
            if not np.any(np.isnan(am)):
                score += 1.8 * float(np.linalg.norm(np.array(c) - np.array(am)))
            elif ankle_target is not None:
                score += 1.8 * float(np.linalg.norm(np.array(c) - np.array(ankle_target)))
            score += _between_ankle_penalty(i, np.array(c))
            if address_club is None and ankle_target is None:
                return float("inf")
            return score

        lw = features["left_wrist"][i]
        rw = features["right_wrist"][i]
        if np.any(np.isnan(lw)) or np.any(np.isnan(rw)):
            return float("inf")
        proxy = (lw + rw) / 2.0
        if np.any(np.isnan(proxy)):
            return float("inf")
        score = 0.0
        has_term = False
        if address_wrist is not None:
            score += float(np.linalg.norm(np.array(proxy) - np.array(address_wrist)))
            has_term = True
        am = ankle_mid[i]
        if not np.any(np.isnan(am)):
            score += 2.0 * float(np.linalg.norm(np.array(proxy) - np.array(am)))
            has_term = True
        elif ankle_target is not None:
            score += 2.0 * float(np.linalg.norm(np.array(proxy) - np.array(ankle_target)))
            has_term = True
        score += _between_ankle_penalty(i, np.array(proxy))
        if not _wrists_in_impact_zone(i):
            score += 1200.0
        if has_term:
            return score + 900.0
        return float("inf")

    impact_scores = [_impact_score(i) for i in range(n)]
    p7 = _pick_min(p4 + 1, n - 1, impact_scores)
    if p7 <= p4:
        p7 = min(n - 1, p4 + n // 4)

    lead_arm_scores = list(features["lead_arm_horiz"])
    p5 = _pick_min(p4 + 1, p7, lead_arm_scores)

    p6_start = min(n - 1, p5 + 1)
    p6_end = max(p6_start + 1, p7)
    p6_end = min(n, p6_end)
    p6_raw_horiz_scores = [float("inf")] * n
    for i in range(p6_start, p6_end):
        ang = shaft_angles_interp[i]
        if np.isfinite(ang):
            p6_raw_horiz_scores[i] = _horiz_score_from_angle(float(ang))

    p6_parallel_thr = 10.0
    p6 = None
    for i in range(p6_start, p6_end):
        if p6_raw_horiz_scores[i] <= p6_parallel_thr:
            p6 = i
            break
    if p6 is None:
        if np.isfinite(min(p6_raw_horiz_scores[p6_start:p6_end])):
            p6 = _pick_min(p6_start, p6_end, p6_raw_horiz_scores)
        else:
            p6 = _pick_min(p6_start, p6_end, horiz_scores)

    p7_start = max(p6 + 1, p4 + 1)
    p7_window = max(3, n // 20)
    p7_stop = min(n, p7_start + p7_window)
    if p7_stop <= p7_start:
        p7_stop = min(n, p7_start + 1)
    p7 = _pick_min(p7_start, p7_stop, impact_scores)
    if np.isfinite(impact_scores[p7]):
        p7_best = float(impact_scores[p7])
        p7_tol = max(8.0, 0.06 * p7_best)
        for i in range(p7_start, p7_stop):
            if np.isfinite(impact_scores[i]) and impact_scores[i] <= p7_best + p7_tol:
                p7 = i
                break
    if p7 <= p6:
        p7 = min(n - 1, p6 + 1)

    horiz_scores = [_horiz_score(i) for i in range(n)]

    p8_window = int(max(5, min(24, round(0.14 * fps))))
    p8_start = min(n - 1, p7 + 1)
    p8_end = min(n - 1, p7 + p8_window)
    if p8_end < p8_start:
        p8_end = p8_start
    grip_angles = []
    grip_horiz_scores = []
    grip_mid = []
    grip_dist = []
    for i in range(n):
        lw = features["left_wrist"][i]
        rw = features["right_wrist"][i]
        if np.any(np.isnan(lw)) or np.any(np.isnan(rw)):
            grip_angles.append(float("nan"))
            grip_horiz_scores.append(float("inf"))
            grip_mid.append(None)
            grip_dist.append(float("nan"))
        else:
            v = rw - lw
            ang = math.degrees(math.atan2(v[1], v[0]))
            ang = abs(ang) % 180.0
            grip_angles.append(ang)
            grip_horiz_scores.append(_horiz_score_from_angle(ang))
            mid = (float((lw[0] + rw[0]) / 2.0), float((lw[1] + rw[1]) / 2.0))
            grip_mid.append(mid)
            grip_dist.append(float(np.linalg.norm(v)))

    def _shaft_is_reliable(i: int) -> bool:
        if shaft_centers[i] is None or grip_mid[i] is None or not np.isfinite(grip_dist[i]):
            return False
        d = float(np.linalg.norm(np.array(shaft_centers[i]) - np.array(grip_mid[i])))
        return d <= max(40.0, 2.5 * grip_dist[i])

    shaft_reliable = [_shaft_is_reliable(i) for i in range(n)]

    head_horiz_scores = [float("inf")] * n
    head_angles = [float("nan")] * n
    head_dist = [float("nan")] * n
    for i in range(n):
        gm = grip_mid[i]
        hc = club_centers[i]
        if gm is None or hc is None:
            continue
        v = np.array([float(hc[0] - gm[0]), float(hc[1] - gm[1])], dtype=np.float32)
        d = float(np.linalg.norm(v))
        head_dist[i] = d
        if d < 1e-6:
            continue
        ang = float(abs(math.degrees(math.atan2(float(v[1]), float(v[0])))) % 180.0)
        head_angles[i] = ang
        head_horiz_scores[i] = _horiz_score_from_angle(ang)

    def _angle_diff(a: float, b: float) -> float:
        d = abs(a - b) % 180.0
        return min(d, 180.0 - d)

    def _head_is_reliable(i: int) -> bool:
        if grip_mid[i] is None or club_centers[i] is None:
            return False
        if not np.isfinite(head_dist[i]) or not np.isfinite(head_angles[i]):
            return False
        gw = float(grip_dist[i]) if np.isfinite(grip_dist[i]) else float("nan")
        d = float(head_dist[i])
        if np.isfinite(gw):
            min_d = max(14.0, 0.65 * gw)
            max_d = max(120.0, 14.0 * gw)
        else:
            min_d = 14.0
            max_d = 520.0
        if d < min_d or d > max_d:
            return False
        if shaft_reliable[i] and np.isfinite(shaft_angles_interp[i]):
            if _angle_diff(float(head_angles[i]), float(shaft_angles_interp[i])) > 70.0:
                return False
        return True

    head_reliable = [_head_is_reliable(i) for i in range(n)]

    p8_scores = [float("inf")] * n
    p8_high_penalty = [0.0] * n
    for i in range(p8_start, p8_end + 1):
        shaft_score = float("inf")
        if shaft_reliable[i] and np.isfinite(shaft_angles_interp[i]):
            shaft_score = _horiz_score_from_angle(float(shaft_angles_interp[i]))
        grip_score = float(grip_horiz_scores[i]) if np.isfinite(grip_horiz_scores[i]) else float("inf")
        if np.isfinite(shaft_score) and np.isfinite(grip_score):
            shaft_ang = float(abs(shaft_angles_interp[i]) % 180.0)
            grip_ang = float(abs(grip_angles[i]) % 180.0) if np.isfinite(grip_angles[i]) else shaft_ang
            if abs(shaft_ang - grip_ang) > 35.0:
                p8_scores[i] = grip_score + 1.0
            else:
                p8_scores[i] = 0.8 * shaft_score + 0.2 * grip_score
        elif np.isfinite(shaft_score):
            p8_scores[i] = shaft_score
        elif np.isfinite(grip_score):
            p8_scores[i] = grip_score + 2.0
        else:
            p8_scores[i] = float(features["lead_forearm_horiz"][i]) + 5.0

        if head_reliable[i] and np.isfinite(head_horiz_scores[i]):
            head_score = float(head_horiz_scores[i])
            if np.isfinite(shaft_score):
                p8_scores[i] = min(p8_scores[i], 0.75 * head_score + 0.25 * shaft_score)
            elif np.isfinite(grip_score):
                p8_scores[i] = min(p8_scores[i], 0.85 * head_score + 0.15 * grip_score)
            else:
                p8_scores[i] = min(p8_scores[i], head_score)

        gm = grip_mid[i]
        hy = features["hip_y"][i]
        ls = features["left_shoulder"][i]
        rs = features["right_shoulder"][i]
        if gm is not None and not np.isnan(hy):
            shoulder_ys = []
            if not np.any(np.isnan(ls)):
                shoulder_ys.append(float(ls[1]))
            if not np.any(np.isnan(rs)):
                shoulder_ys.append(float(rs[1]))
            if shoulder_ys:
                shy = min(shoulder_ys)
                torso = max(1.0, float(hy) - shy)
                y_upper = shy + 0.32 * torso
                if float(gm[1]) < y_upper:
                    p8_high_penalty[i] = 8.0 + (y_upper - float(gm[1])) / torso * 20.0
                    p8_scores[i] += p8_high_penalty[i]

        p8_scores[i] += 0.65 * float(i - p8_start)

    p8 = _pick_min(p8_start, p8_end + 1, p8_scores)
    p8_head_thr = 14.0
    p8_head_candidates = [
        i
        for i in range(p8_start, p8_end + 1)
        if head_reliable[i]
        and np.isfinite(head_horiz_scores[i])
        and float(head_horiz_scores[i]) <= p8_head_thr
        and p8_high_penalty[i] <= 14.0
    ]
    if p8_head_candidates:
        p8 = p8_head_candidates[0]
    p8_shaft_thr = 14.0
    p8_grip_thr = 12.0
    p8_candidates = []
    if not p8_head_candidates:
        for i in range(p8_start, p8_end + 1):
            shaft_ok = (
                shaft_reliable[i]
                and np.isfinite(shaft_angles_interp[i])
                and _horiz_score_from_angle(float(shaft_angles_interp[i])) <= p8_shaft_thr
            )
            grip_ok = np.isfinite(grip_horiz_scores[i]) and float(grip_horiz_scores[i]) <= p8_grip_thr
            ok = shaft_ok or (not shaft_reliable[i] and grip_ok)
            if ok and p8_high_penalty[i] <= 12.0:
                p8_candidates.append(i)
        if p8_candidates:
            p8 = p8_candidates[0]
        else:
            p8_loose_candidates = []
            for i in range(p8_start, p8_end + 1):
                shaft_ok = (
                    shaft_reliable[i]
                    and np.isfinite(shaft_angles_interp[i])
                    and _horiz_score_from_angle(float(shaft_angles_interp[i])) <= 22.0
                )
                grip_ok = np.isfinite(grip_horiz_scores[i]) and float(grip_horiz_scores[i]) <= 18.0
                if (shaft_ok or grip_ok) and p8_high_penalty[i] <= 18.0:
                    p8_loose_candidates.append(i)
            if p8_loose_candidates:
                p8 = p8_loose_candidates[0]
            else:
                finite_idxs = [i for i in range(p8_start, p8_end + 1) if np.isfinite(p8_scores[i])]
                if finite_idxs:
                    best = min(float(p8_scores[i]) for i in finite_idxs)
                    tol = max(1.5, min(8.0, 0.2 * best + 2.0))
                    near_best = [i for i in finite_idxs if float(p8_scores[i]) <= best + tol]
                    if near_best:
                        p8 = near_best[0]

    p9_start = min(n - 1, p8 + 2)
    p9_window = int(max(10, min(50, round(0.35 * fps))))
    p9_end = min(n - 1, p8 + p9_window)
    if p9_end < p9_start:
        p9_end = p9_start
    lead_arm_scores = list(features["lead_arm_horiz"])
    lead_arm_thr = 12.0
    lead_is_right = bool(features["lead_is_right"][0])
    lead_shoulder_arr = features["right_shoulder"] if lead_is_right else features["left_shoulder"]
    lead_wrist_arr = features["right_wrist"] if lead_is_right else features["left_wrist"]

    p9_scores = [float("inf")] * n
    for i in range(p9_start, p9_end + 1):
        if shaft_reliable[i] and np.isfinite(shaft_angles_interp[i]):
            shaft_vert = abs(float(shaft_angles_interp[i]) - 90.0)
        elif np.isfinite(grip_angles[i]):
            shaft_vert = abs(float(grip_angles[i]) - 90.0)
        else:
            shaft_vert = 30.0
        p9_scores[i] = 0.7 * float(lead_arm_scores[i]) + 0.3 * float(shaft_vert)

        ls = lead_shoulder_arr[i]
        lw = lead_wrist_arr[i]
        if not np.any(np.isnan(ls)) and not np.any(np.isnan(lw)):
            dy = float(lw[1] - ls[1])
            if dy > 8.0:
                p9_scores[i] += min(26.0, (dy - 8.0) * 0.7)
            elif dy < -30.0:
                p9_scores[i] += min(10.0, (-30.0 - dy) * 0.25)
            p9_scores[i] += 0.2 * float(i - p9_start)

    p9 = _pick_min(p9_start, p9_end + 1, p9_scores)
    p9_good = []
    for i in range(p9_start, p9_end + 1):
        ls = lead_shoulder_arr[i]
        lw = lead_wrist_arr[i]
        dy_ok = True
        if not np.any(np.isnan(ls)) and not np.any(np.isnan(lw)):
            dy = float(lw[1] - ls[1])
            dy_ok = (-20.0 <= dy <= 10.0)

        if shaft_reliable[i] and np.isfinite(shaft_angles_interp[i]):
            vert_ok = abs(float(shaft_angles_interp[i]) - 90.0) <= 20.0
        elif np.isfinite(grip_angles[i]):
            vert_ok = abs(float(grip_angles[i]) - 90.0) <= 28.0
        else:
            vert_ok = True

        if np.isfinite(lead_arm_scores[i]) and lead_arm_scores[i] <= lead_arm_thr and dy_ok and vert_ok:
            p9_good.append(i)
    if p9_good:
        p9 = min(p9_end, p9_good[0] + 1)

    idxs = [p1, p2, p3, p4, p5, p6, p7, p8, p9]
    default_gap = max(1, n // 25)
    gaps = [
        default_gap,
        default_gap,
        default_gap,
        default_gap,
        1,
        1,
        1,
        1,
    ]
    for i in range(1, len(idxs)):
        min_gap = gaps[i - 1]
        if idxs[i] <= idxs[i - 1] + min_gap:
            idxs[i] = min(n - 1, idxs[i - 1] + min_gap)

    names = [
        "Address",
        "Mid-backswing",
        "Late-backswing",
        "Top-backswing",
        "Early-downswing",
        "Mid-downswing",
        "Ball-impact",
        "Mid-followthrough",
        "Late-followthrough",
    ]
    events = []
    for name, idx in zip(names, idxs):
        events.append({"name": name, "frame": int(frame_ids[idx]), "t": float(times[idx])})
    return events
