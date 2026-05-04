from bisect import bisect_left
import math
from typing import Dict, List, Optional, Tuple


POINT_NAMES = ("grip", "chest", "hip")
ANGLE_NAMES = ("shaft", "chest", "hip", "chest_y", "hip_y")
ROTATION_SEGMENT_NAMES = ("chest", "hip")


def _round1(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return round(float(value), 1)


def _round1_optional(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), 1)


def _round3(value: Optional[float]) -> float:
    if value is None:
        return 0.0
    return round(float(value), 3)


def _safe_event(result: Dict, name: str) -> Optional[Dict]:
    for event in result.get("events", []) or []:
        if event.get("name") == name:
            return event
    return None


def _body_point_value(frame: Optional[Dict], point_name: str, axis: str) -> Optional[float]:
    if not frame:
        return None
    body_points = frame.get("body_points") or {}
    point = body_points.get(point_name) or {}
    value = point.get(axis)
    if value is None:
        return None
    return float(value)


def _frame_map(result: Dict) -> Dict[int, Dict]:
    return {int(frame["frame"]): frame for frame in result.get("frames", []) or []}


def _sorted_frame_ids(result: Dict) -> List[int]:
    return sorted(_frame_map(result).keys())


def _nearest_frame(frame_lookup: Dict[int, Dict], sorted_ids: List[int], target_frame: int) -> Optional[Dict]:
    if not sorted_ids:
        return None
    pos = bisect_left(sorted_ids, target_frame)
    candidates: List[int] = []
    if pos < len(sorted_ids):
        candidates.append(sorted_ids[pos])
    if pos > 0:
        candidates.append(sorted_ids[pos - 1])
    best_id = min(candidates, key=lambda fid: abs(fid - target_frame))
    return frame_lookup.get(best_id)


def _body_point_axis(frame: Optional[Dict], point_name: str, axis: str) -> float:
    value = _body_point_value(frame, point_name, axis)
    if value is None:
        return 0.0
    return value


def _translation_axis_value(frame: Optional[Dict], point_name: str, axis: str) -> Optional[float]:
    raw_axis = {
        "x": "dx_in_est",
        "y": "dy_in_est",
    }.get(axis)
    if raw_axis is None:
        return None
    return _body_point_value(frame, point_name, raw_axis)


def _frame_time(frame: Optional[Dict]) -> float:
    if not frame:
        return 0.0
    value = frame.get("t")
    if value is None:
        return 0.0
    return float(value)


def _frame_angle(frame: Optional[Dict], key: str) -> float:
    if not frame:
        return 0.0
    value = frame.get(key)
    if value is None:
        return 0.0
    return float(value)


def _frame_angle_optional(frame: Optional[Dict], key: str) -> Optional[float]:
    if not frame:
        return None
    value = frame.get(key)
    if value is None:
        return None
    return float(value)


def _find_first_frame_with_point(result: Dict, point_name: str = "grip") -> Optional[Dict]:
    for frame in result.get("frames", []) or []:
        body_points = frame.get("body_points") or {}
        point = body_points.get(point_name)
        if point is not None:
            return frame
    return None


def _find_top_anchor_frame(result: Dict) -> Optional[Dict]:
    address_event = _safe_event(result, "Address")
    address_frame = int(address_event["frame"]) if address_event is not None else None
    best_frame = None
    best_y = None
    for frame in result.get("frames", []) or []:
        frame_id = int(frame["frame"])
        if address_frame is not None and frame_id < address_frame:
            continue
        body_points = frame.get("body_points") or {}
        grip = body_points.get("grip")
        if grip is None:
            continue
        y = grip.get("y")
        if y is None:
            continue
        if best_y is None or float(y) < best_y:
            best_y = float(y)
            best_frame = frame
    return best_frame


def _infer_dtl_ball_axis_sign(reference_frame: Optional[Dict]) -> Tuple[float, str]:
    """Infer the positive DTL depth direction from Address geometry.

    We use the horizontal image-x direction from the body core toward the grip at
    Address as the "toward ball" direction. Positive z then means movement
    toward the ball side of the setup; negative z means movement away from it.
    """
    if not reference_frame:
        return 1.0, "fallback_image_x"

    grip_x = _body_point_value(reference_frame, "grip", "x")
    if grip_x is None:
        return 1.0, "fallback_image_x"

    for anchor_name in ("hip", "chest", "shoulder"):
        anchor_x = _body_point_value(reference_frame, anchor_name, "x")
        if anchor_x is None:
            continue
        delta = grip_x - anchor_x
        if abs(delta) > 1e-6:
            return (1.0 if delta > 0.0 else -1.0), f"dtl_address_grip_minus_{anchor_name}_x"
    return 1.0, "fallback_image_x"


def _angle_delta_deg(current: float, reference: float) -> float:
    delta = abs(float(current) - float(reference)) % 360.0
    if delta > 180.0:
        delta = 360.0 - delta
    if delta > 90.0:
        delta = 180.0 - delta
    return delta


def _dtl_segment_x_angle(
    frame: Optional[Dict],
    reference_frame: Optional[Dict],
    start_point: str,
    end_point: str,
    z_axis_sign: float,
) -> Optional[float]:
    """Unsigned X-axis rotation proxy from a DTL segment in the Y-Z plane."""

    def _segment_angle(item: Optional[Dict]) -> Optional[float]:
        if not item:
            return None
        z0 = _body_point_value(item, start_point, "x")
        y0 = _body_point_value(item, start_point, "y")
        z1 = _body_point_value(item, end_point, "x")
        y1 = _body_point_value(item, end_point, "y")
        if z0 is None or y0 is None or z1 is None or y1 is None:
            return None
        dz = (z1 - z0) * z_axis_sign
        dy = -(y1 - y0)  # image-y grows downward; positive body Y is upward.
        if abs(dz) <= 1e-6 and abs(dy) <= 1e-6:
            return None
        return math.degrees(math.atan2(dz, dy))

    ref_angle = _segment_angle(reference_frame)
    curr_angle = _segment_angle(frame)
    if ref_angle is None or curr_angle is None:
        return None
    return _angle_delta_deg(curr_angle, ref_angle)


def _dtl_x_rotations(
    frame: Optional[Dict],
    reference_frame: Optional[Dict],
    z_axis_sign: float,
) -> Dict[str, Optional[float]]:
    hip_x = _dtl_segment_x_angle(frame, reference_frame, "hip", "chest", z_axis_sign)
    chest_x = _dtl_segment_x_angle(frame, reference_frame, "chest", "shoulder", z_axis_sign)
    if chest_x is None:
        chest_x = _dtl_segment_x_angle(frame, reference_frame, "hip", "chest", z_axis_sign)
    return {
        "hip": hip_x,
        "chest": chest_x,
    }


def build_sync(face_result: Dict, dtl_result: Optional[Dict]) -> Dict[str, float]:
    face_start_frame = int((face_result.get("frames") or [{}])[0].get("frame", 0)) if face_result.get("frames") else 0
    face_start_time = _frame_time((face_result.get("frames") or [None])[0]) if face_result.get("frames") else 0.0
    if not dtl_result:
        face_address = _safe_event(face_result, "Address")
        if face_address is None:
            face_anchor = _find_first_frame_with_point(face_result)
            face_address = {
                "frame": int(face_anchor["frame"]) if face_anchor else 0,
                "t": _frame_time(face_anchor),
            }
        return {
            "enabled": False,
            "face_on_address_frame": int(face_address["frame"]) if face_address else 0,
            "down_the_line_address_frame": 0,
            "anchor_frame_face_on": 0,
            "anchor_frame_down_the_line": 0,
            "offset_frames": 0,
            "confidence": 0.0,
            "time_scale": 1.0,
            "time_offset_sec": 0.0,
            "method": "single_view",
            "face_on_start_frame": face_start_frame,
            "down_the_line_start_frame": 0,
        }

    face_address = _safe_event(face_result, "Address")
    if face_address is None:
        face_anchor = _find_first_frame_with_point(face_result)
        face_address = {
            "frame": int(face_anchor["frame"]) if face_anchor else 0,
            "t": _frame_time(face_anchor),
        }
    dtl_frames = dtl_result.get("frames", []) or []
    dtl_start_frame = int(dtl_frames[0].get("frame", 0)) if dtl_frames else 0
    face_frame_count = int(face_result.get("video", {}).get("frame_count", 0))
    dtl_frame_count = int(dtl_result.get("video", {}).get("frame_count", 0))
    frame_ratio = 1.0
    if face_frame_count > 1 and dtl_frame_count > 1:
        frame_ratio = float(dtl_frame_count - 1) / float(face_frame_count - 1)
    confidence = 1.0
    mapped_address_frame = int(round(int(face_address["frame"]) * frame_ratio)) if face_address else 0
    return {
        "enabled": True,
        "face_on_address_frame": int(face_address["frame"]) if face_address else 0,
        "down_the_line_address_frame": mapped_address_frame,
        "anchor_frame_face_on": 0,
        "anchor_frame_down_the_line": 0,
        "offset_frames": 0,
        "confidence": round(confidence, 3),
        "time_scale": 1.0,
        "time_offset_sec": 0.0,
        "method": "frame_ratio_aligned",
        "face_on_start_frame": face_start_frame,
        "down_the_line_start_frame": dtl_start_frame,
        "face_on_frame_count": face_frame_count,
        "down_the_line_frame_count": dtl_frame_count,
        "frame_ratio": round(frame_ratio, 9),
    }


def map_face_frame_to_dtl_frame(face_frame: int, dtl_result: Optional[Dict], sync: Optional[Dict[str, float]] = None) -> int:
    if not dtl_result:
        return 0
    face_start_frame = int((sync or {}).get("face_on_start_frame", 0))
    dtl_start_frame = int((sync or {}).get("down_the_line_start_frame", 0))
    frame_ratio = float((sync or {}).get("frame_ratio", 1.0))
    frame_count = int(dtl_result.get("video", {}).get("frame_count", 0))
    frame_id = dtl_start_frame + int(round((int(face_frame) - face_start_frame) * frame_ratio))
    if frame_count > 0:
        frame_id = max(0, min(frame_count - 1, frame_id))
    return frame_id


def map_face_time_to_dtl_frame(face_time: float, dtl_result: Optional[Dict], sync: Dict[str, float]) -> int:
    if sync.get("method") == "frame_ratio_aligned":
        fps = float(sync.get("face_on_fps", 0.0))
        if fps > 0.0:
            return map_face_frame_to_dtl_frame(int(round(float(face_time) * fps)), dtl_result, sync)
    if not dtl_result:
        return 0
    fps = float(dtl_result.get("video", {}).get("fps", 30.0))
    frame_count = int(dtl_result.get("video", {}).get("frame_count", 0))
    t_dtl = sync.get("time_scale", 1.0) * float(face_time) + sync.get("time_offset_sec", 0.0)
    if t_dtl < 0.0:
        t_dtl = 0.0
    frame_id = int(round(t_dtl * fps))
    if frame_count > 0:
        frame_id = max(0, min(frame_count - 1, frame_id))
    return frame_id


def map_dtl_time_to_face_frame(dtl_time: float, face_result: Dict, sync: Dict[str, float]) -> int:
    if sync.get("method") == "frame_ratio_aligned":
        fps = float(sync.get("down_the_line_fps", 0.0))
        frame_count = int(face_result.get("video", {}).get("frame_count", 0))
        if fps > 0.0:
            dtl_frame = int(round(float(dtl_time) * fps))
            face_start_frame = int(sync.get("face_on_start_frame", 0))
            dtl_start_frame = int(sync.get("down_the_line_start_frame", 0))
            frame_ratio = float(sync.get("frame_ratio", 1.0))
            if abs(frame_ratio) <= 1e-9:
                frame_id = face_start_frame
            else:
                frame_id = face_start_frame + int(round((dtl_frame - dtl_start_frame) / frame_ratio))
            if frame_count > 0:
                frame_id = max(0, min(frame_count - 1, frame_id))
            return frame_id
    fps = float(face_result.get("video", {}).get("fps", 30.0))
    frame_count = int(face_result.get("video", {}).get("frame_count", 0))
    a = sync.get("time_scale", 1.0)
    b = sync.get("time_offset_sec", 0.0)
    if abs(a) <= 1e-6:
        t_face = float(dtl_time)
    else:
        t_face = (float(dtl_time) - b) / a
    if t_face < 0.0:
        t_face = 0.0
    frame_id = int(round(t_face * fps))
    if frame_count > 0:
        frame_id = max(0, min(frame_count - 1, frame_id))
    return frame_id


def build_phase_pairs(face_result: Dict, dtl_result: Optional[Dict], sync: Dict[str, float]) -> List[Dict]:
    pairs: List[Dict] = []
    dtl_fps = float(dtl_result.get("video", {}).get("fps", 0.0)) if dtl_result else 0.0
    for idx, event in enumerate(face_result.get("events", []) or [], start=1):
        face_frame = int(event.get("frame", 0))
        face_time = float(event.get("t", 0.0))
        dtl_frame = map_face_frame_to_dtl_frame(face_frame, dtl_result, sync) if dtl_result else 0
        dtl_time = round(float(dtl_frame / dtl_fps), 3) if dtl_result and dtl_fps > 0 else 0.0
        pairs.append(
            {
                "phase": event.get("name", f"P{idx}"),
                "phase_index": idx,
                "face_on": {
                    "frame": face_frame,
                    "t": _round3(face_time),
                },
                "down_the_line": {
                    "frame": dtl_frame,
                    "t": dtl_time,
                },
            }
        )
    return pairs


def build_combined_output(
    mode: str,
    height_mm: float,
    face_result: Dict,
    dtl_result: Optional[Dict],
    sync: Dict[str, float],
) -> Dict:
    face_lookup = _frame_map(face_result)
    face_ids = sorted(face_lookup.keys())
    dtl_lookup = _frame_map(dtl_result) if dtl_result else {}
    dtl_ids = sorted(dtl_lookup.keys())
    face_address = _safe_event(face_result, "Address")
    face_address_frame = int(face_address["frame"]) if face_address else 0
    dtl_reference_frame_id = map_face_frame_to_dtl_frame(face_address_frame, dtl_result, sync) if dtl_result else 0
    dtl_reference_frame = _nearest_frame(dtl_lookup, dtl_ids, dtl_reference_frame_id) if dtl_result else None
    dtl_in_per_px = float((dtl_result or {}).get("calibration", {}).get("in_per_px", 0.0)) if dtl_result else 0.0
    dtl_ball_axis_sign, dtl_ball_axis_source = _infer_dtl_ball_axis_sign(dtl_reference_frame)
    face_coordinate_system = face_result.get("coordinate_system") or {}

    frames: List[Dict] = []
    for face_frame in face_result.get("frames", []) or []:
        face_frame_id = int(face_frame["frame"])
        face_time = float(face_frame.get("t", 0.0))
        dtl_frame_id = map_face_frame_to_dtl_frame(face_frame_id, dtl_result, sync) if dtl_result else 0
        dtl_frame = _nearest_frame(dtl_lookup, dtl_ids, dtl_frame_id) if dtl_result else None
        actual_dtl_frame_id = int(dtl_frame["frame"]) if dtl_frame is not None else 0
        actual_dtl_time = _round3(_frame_time(dtl_frame)) if dtl_frame is not None else 0.0
        x_rotations = _dtl_x_rotations(dtl_frame, dtl_reference_frame, dtl_ball_axis_sign) if dtl_result else {
            "hip": None,
            "chest": None,
        }

        points: Dict[str, Dict[str, float]] = {}
        for point_name in POINT_NAMES:
            dz_in = None
            if dtl_frame is not None and dtl_reference_frame is not None and dtl_in_per_px > 0.0:
                curr_x = _body_point_axis(dtl_frame, point_name, "x")
                ref_x = _body_point_axis(dtl_reference_frame, point_name, "x")
                dz_in = (curr_x - ref_x) * dtl_in_per_px * dtl_ball_axis_sign
            points[point_name] = {
                "dx_in": _round1_optional(_translation_axis_value(face_frame, point_name, "x")),
                "dy_in": _round1_optional(_translation_axis_value(face_frame, point_name, "y")),
                "dz_in": _round1_optional(dz_in),
            }
        translations = {
            point_name: {
                "x_in": points[point_name]["dx_in"],
                "y_in": points[point_name]["dy_in"],
                "z_in": points[point_name]["dz_in"],
            }
            for point_name in POINT_NAMES
        }
        rotations = {
            "chest": {
                "x_deg": _round1_optional(x_rotations.get("chest")),
                "y_deg": _round1_optional(_frame_angle_optional(face_frame, "chest_y_angle")),
                "y_deg_legacy": _round1_optional(_frame_angle_optional(face_frame, "chest_y_angle_legacy")),
                "y_deg_experimental": _round1_optional(_frame_angle_optional(face_frame, "chest_y_angle_experimental")),
                "z_deg": _round1_optional(_frame_angle_optional(face_frame, "chest_angle")),
            },
            "hip": {
                "x_deg": _round1_optional(x_rotations.get("hip")),
                "y_deg": _round1_optional(_frame_angle_optional(face_frame, "hip_y_angle")),
                "y_deg_legacy": _round1_optional(_frame_angle_optional(face_frame, "hip_y_angle_legacy")),
                "y_deg_experimental": _round1_optional(_frame_angle_optional(face_frame, "hip_y_angle_experimental")),
                "z_deg": _round1_optional(_frame_angle_optional(face_frame, "hip_angle")),
            },
        }

        angles = {
            "shaft": {
                "face_on_deg": _round1(_frame_angle(face_frame, "shaft_angle_smooth")),
                "down_the_line_deg": _round1(_frame_angle(dtl_frame, "shaft_angle_smooth")),
            },
            "chest": {
                "face_on_deg": _round1(_frame_angle(face_frame, "chest_angle")),
                "down_the_line_deg": _round1(_frame_angle(dtl_frame, "chest_angle")),
            },
            "hip": {
                "face_on_deg": _round1(_frame_angle(face_frame, "hip_angle")),
                "down_the_line_deg": _round1(_frame_angle(dtl_frame, "hip_angle")),
            },
            "chest_y": {
                "face_on_deg": _round1(_frame_angle(face_frame, "chest_y_angle")),
                "down_the_line_deg": 0.0,
            },
            "chest_y_legacy": {
                "face_on_deg": _round1_optional(_frame_angle_optional(face_frame, "chest_y_angle_legacy")),
                "down_the_line_deg": 0.0,
            },
            "chest_y_experimental": {
                "face_on_deg": _round1_optional(_frame_angle_optional(face_frame, "chest_y_angle_experimental")),
                "down_the_line_deg": 0.0,
            },
            "hip_y": {
                "face_on_deg": _round1(_frame_angle(face_frame, "hip_y_angle")),
                "down_the_line_deg": 0.0,
            },
            "hip_y_legacy": {
                "face_on_deg": _round1_optional(_frame_angle_optional(face_frame, "hip_y_angle_legacy")),
                "down_the_line_deg": 0.0,
            },
            "hip_y_experimental": {
                "face_on_deg": _round1_optional(_frame_angle_optional(face_frame, "hip_y_angle_experimental")),
                "down_the_line_deg": 0.0,
            },
            "chest_x": {
                "face_on_deg": None,
                "down_the_line_deg": _round1_optional(x_rotations.get("chest")),
            },
            "hip_x": {
                "face_on_deg": None,
                "down_the_line_deg": _round1_optional(x_rotations.get("hip")),
            },
        }

        frames.append(
            {
                "face_on_frame": face_frame_id,
                "face_on_t": _round3(face_time),
                "down_the_line_frame": actual_dtl_frame_id,
                "down_the_line_t": actual_dtl_time,
                "points": points,
                "translations": translations,
                "angles": angles,
                "rotations": rotations,
            }
        )

    phase_pairs = build_phase_pairs(face_result, dtl_result, sync)
    return {
        "mode": mode,
        "height_mm": float(height_mm),
        "height_in": round(float(height_mm / 25.4), 3),
        "events": face_result.get("events", []) or [],
        "sync": sync,
        "coordinate_system": {
            "reference_phase": "Address",
            "reference_frames": {
                "face_on": face_coordinate_system.get("reference_frame"),
                "down_the_line": int(dtl_reference_frame["frame"]) if dtl_reference_frame is not None else None,
            },
            "axes": {
                "x": {
                    "source_view": "face_on",
                    "positive": "toward lead side",
                    "negative": "toward trail side",
                    "note": "Inherited from face-on Address-referenced displacement.",
                },
                "y": {
                    "source_view": "face_on",
                    "positive": "upward",
                    "negative": "downward",
                    "note": "Inherited from face-on Address-referenced displacement.",
                },
                "z": {
                    "source_view": "down_the_line",
                    "positive": "toward ball side",
                    "negative": "away from ball side",
                    "note": "Signed from DTL image-x delta relative to Address; positive direction is inferred from the Address setup geometry (body core -> grip).",
                    "sign_source": dtl_ball_axis_source,
                    "image_x_sign": dtl_ball_axis_sign,
                },
            },
        },
        "views": {
            "face_on": {
                "video_path": face_result.get("video", {}).get("path", ""),
                "fps": float(face_result.get("video", {}).get("fps", 0.0)),
            },
            "down_the_line": {
                "video_path": dtl_result.get("video", {}).get("path", "") if dtl_result else "",
                "fps": float(dtl_result.get("video", {}).get("fps", 0.0)) if dtl_result else 0.0,
            },
        },
        "phase_frames": phase_pairs,
        "frames": frames,
    }


def _combined_frame_lookup(combined_result: Dict) -> Dict[int, Dict]:
    return {int(frame["face_on_frame"]): frame for frame in combined_result.get("frames", []) or []}


def _display_points(frame: Optional[Dict]) -> Dict[str, Dict[str, float]]:
    body_points: Dict[str, Dict[str, float]] = {}
    points = (frame or {}).get("points", {})
    for point_name in POINT_NAMES:
        point = points.get(point_name, {})
        body_points[point_name] = {
            "dx_in_est": _round1_optional(point.get("dx_in")),
            "dy_in_est": _round1_optional(point.get("dy_in")),
            "dz_in_est": _round1_optional(point.get("dz_in")),
        }
    return body_points


def _display_translations(frame: Optional[Dict]) -> Dict[str, Dict[str, float]]:
    translations = (frame or {}).get("translations") or {}
    if not translations:
        points = (frame or {}).get("points") or {}
        translations = {
            point_name: {
                "x_in": (points.get(point_name) or {}).get("dx_in"),
                "y_in": (points.get(point_name) or {}).get("dy_in"),
                "z_in": (points.get(point_name) or {}).get("dz_in"),
            }
            for point_name in POINT_NAMES
        }
    return {
        point_name: {
            "x_in": _round1_optional((translations.get(point_name) or {}).get("x_in")),
            "y_in": _round1_optional((translations.get(point_name) or {}).get("y_in")),
            "z_in": _round1_optional((translations.get(point_name) or {}).get("z_in")),
        }
        for point_name in POINT_NAMES
    }


def _display_rotations(frame: Optional[Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    rotations = (frame or {}).get("rotations") or {}
    if not rotations:
        angles = (frame or {}).get("angles") or {}
        rotations = {
            "chest": {
                "x_deg": None,
                "y_deg": (angles.get("chest_y") or {}).get("face_on_deg"),
                "z_deg": (angles.get("chest") or {}).get("face_on_deg"),
            },
            "hip": {
                "x_deg": None,
                "y_deg": (angles.get("hip_y") or {}).get("face_on_deg"),
                "z_deg": (angles.get("hip") or {}).get("face_on_deg"),
            },
        }
    return {
        segment_name: {
            "x_deg": _round1_optional((rotations.get(segment_name) or {}).get("x_deg")),
            "y_deg": _round1_optional((rotations.get(segment_name) or {}).get("y_deg")),
            "z_deg": _round1_optional((rotations.get(segment_name) or {}).get("z_deg")),
        }
        for segment_name in ROTATION_SEGMENT_NAMES
    }


def _display_angles(frame: Optional[Dict]) -> Dict[str, Dict[str, float]]:
    angles = (frame or {}).get("angles", {})
    out: Dict[str, Dict[str, float]] = {}
    for angle_name in ANGLE_NAMES:
        angle = angles.get(angle_name, {})
        out[angle_name] = {
            "face_on_deg": _round1_optional(angle.get("face_on_deg")),
            "down_the_line_deg": _round1_optional(angle.get("down_the_line_deg")),
        }
    return out


def build_overlay_payload(
    view_name: str,
    raw_result: Dict,
    combined_result: Dict,
    sync: Dict[str, float],
    mapped_events: List[Dict],
) -> Dict:
    combined_lookup = _combined_frame_lookup(combined_result)
    combined_ids = sorted(combined_lookup.keys())
    face_result_meta = combined_result.get("views", {}).get("face_on", {})
    dtl_result_meta = combined_result.get("views", {}).get("down_the_line", {})

    frames = []
    for raw_frame in raw_result.get("frames", []) or []:
        if view_name == "face_on":
            combined_frame = combined_lookup.get(int(raw_frame["frame"]))
        else:
            mapped_face_frame = int(raw_frame["frame"])
            combined_frame = _nearest_frame(combined_lookup, combined_ids, mapped_face_frame)
        frames.append(
            {
                "frame": raw_frame.get("frame"),
                "t": raw_frame.get("t"),
                "keypoints": raw_frame.get("keypoints"),
                "shaft_angle_smooth": raw_frame.get("shaft_angle_smooth"),
                "shaft_smooth": raw_frame.get("shaft_smooth"),
                "body_points": _display_points(combined_frame),
                "translations": _display_translations(combined_frame),
                "display_angles": _display_angles(combined_frame),
                "display_rotations": _display_rotations(combined_frame),
                "hip_angle": raw_frame.get("hip_angle"),
                "chest_angle": raw_frame.get("chest_angle"),
                "hip_y_angle": raw_frame.get("hip_y_angle"),
                "chest_y_angle": raw_frame.get("chest_y_angle"),
            }
        )

    calibration = raw_result.get("calibration") or {"height_mm": combined_result.get("height_mm", 0.0)}
    return {
        "video": raw_result.get("video", {}),
        "calibration": calibration,
        "coordinate_system": combined_result.get("coordinate_system") or raw_result.get("coordinate_system"),
        "display_meta": {
            "view_name": view_name,
            "face_on_fps": float(face_result_meta.get("fps", 0.0)),
            "down_the_line_fps": float(dtl_result_meta.get("fps", 0.0)),
            "sync_method": sync.get("method", ""),
        },
        "skeleton": raw_result.get("skeleton", {}),
        "frames": frames,
        "events": mapped_events,
        "events_raw": None,
        "swing_direction": raw_result.get("swing_direction"),
        "lead_is_right": raw_result.get("lead_is_right"),
    }
