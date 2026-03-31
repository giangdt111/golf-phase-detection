from bisect import bisect_left
from typing import Dict, List, Optional, Tuple


POINT_NAMES = ("grip", "chest", "hip")
ANGLE_NAMES = ("shaft", "chest", "hip")


def _round1(value: Optional[float]) -> float:
    if value is None:
        return 0.0
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
    if not frame:
        return 0.0
    body_points = frame.get("body_points") or {}
    point = body_points.get(point_name) or {}
    value = point.get(axis)
    if value is None:
        return 0.0
    return float(value)


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
    dtl_start_time = _frame_time(dtl_frames[0]) if dtl_frames else 0.0
    face_end_time = _frame_time((face_result.get("frames") or [None])[-1]) if face_result.get("frames") else face_start_time
    dtl_end_time = _frame_time(dtl_frames[-1]) if dtl_frames else dtl_start_time

    face_duration = max(face_end_time - face_start_time, 1e-6)
    dtl_duration = max(dtl_end_time - dtl_start_time, 1e-6)
    a = dtl_duration / face_duration
    b = dtl_start_time - a * face_start_time
    confidence = 0.8

    face_fps = float(face_result.get("video", {}).get("fps", 30.0))
    offset_frames = int(round(b * face_fps))
    mapped_address_frame = map_face_time_to_dtl_frame(float(face_address["t"]) if face_address else 0.0, dtl_result, {
        "time_scale": round(float(a), 6),
        "time_offset_sec": round(float(b), 6),
    })
    return {
        "enabled": True,
        "face_on_address_frame": int(face_address["frame"]) if face_address else 0,
        "down_the_line_address_frame": mapped_address_frame,
        "anchor_frame_face_on": 0,
        "anchor_frame_down_the_line": 0,
        "offset_frames": offset_frames,
        "confidence": round(confidence, 3),
        "time_scale": round(float(a), 6),
        "time_offset_sec": round(float(b), 6),
        "method": "prealigned_progress",
        "face_on_start_frame": face_start_frame,
        "down_the_line_start_frame": dtl_start_frame,
    }


def map_face_time_to_dtl_frame(face_time: float, dtl_result: Optional[Dict], sync: Dict[str, float]) -> int:
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
        face_time = float(event.get("t", 0.0))
        dtl_frame = map_face_time_to_dtl_frame(face_time, dtl_result, sync) if dtl_result else 0
        dtl_time = round(float(dtl_frame / dtl_fps), 3) if dtl_result and dtl_fps > 0 else 0.0
        pairs.append(
            {
                "phase": event.get("name", f"P{idx}"),
                "phase_index": idx,
                "face_on": {
                    "frame": int(event.get("frame", 0)),
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
    face_address_time = float(face_address["t"]) if face_address else 0.0
    dtl_reference_frame_id = map_face_time_to_dtl_frame(face_address_time, dtl_result, sync) if dtl_result else 0
    dtl_reference_frame = _nearest_frame(dtl_lookup, dtl_ids, dtl_reference_frame_id) if dtl_result else None
    dtl_mm_per_px = float((dtl_result or {}).get("calibration", {}).get("mm_per_px", 0.0)) if dtl_result else 0.0

    frames: List[Dict] = []
    for face_frame in face_result.get("frames", []) or []:
        face_frame_id = int(face_frame["frame"])
        face_time = float(face_frame.get("t", 0.0))
        dtl_frame_id = map_face_time_to_dtl_frame(face_time, dtl_result, sync) if dtl_result else 0
        dtl_frame = _nearest_frame(dtl_lookup, dtl_ids, dtl_frame_id) if dtl_result else None
        actual_dtl_frame_id = int(dtl_frame["frame"]) if dtl_frame is not None else 0
        actual_dtl_time = _round3(_frame_time(dtl_frame)) if dtl_frame is not None else 0.0

        points: Dict[str, Dict[str, float]] = {}
        for point_name in POINT_NAMES:
            dz_mm = 0.0
            if dtl_frame is not None and dtl_reference_frame is not None and dtl_mm_per_px > 0.0:
                curr_x = _body_point_axis(dtl_frame, point_name, "x")
                ref_x = _body_point_axis(dtl_reference_frame, point_name, "x")
                dz_mm = (curr_x - ref_x) * dtl_mm_per_px
            points[point_name] = {
                "dx_mm": _round1(_body_point_axis(face_frame, point_name, "dx_mm_est")),
                "dy_mm": _round1(_body_point_axis(face_frame, point_name, "dy_mm_est")),
                "dz_mm": _round1(dz_mm),
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
        }

        frames.append(
            {
                "face_on_frame": face_frame_id,
                "face_on_t": _round3(face_time),
                "down_the_line_frame": actual_dtl_frame_id,
                "down_the_line_t": actual_dtl_time,
                "points": points,
                "angles": angles,
            }
        )

    phase_pairs = build_phase_pairs(face_result, dtl_result, sync)
    return {
        "mode": mode,
        "height_mm": float(height_mm),
        "events": face_result.get("events", []) or [],
        "sync": sync,
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
            "dx_mm_est": _round1(point.get("dx_mm")),
            "dy_mm_est": _round1(point.get("dy_mm")),
            "dz_mm_est": _round1(point.get("dz_mm")),
        }
    return body_points


def _display_angles(frame: Optional[Dict]) -> Dict[str, Dict[str, float]]:
    angles = (frame or {}).get("angles", {})
    out: Dict[str, Dict[str, float]] = {}
    for angle_name in ANGLE_NAMES:
        angle = angles.get(angle_name, {})
        out[angle_name] = {
            "face_on_deg": _round1(angle.get("face_on_deg")),
            "down_the_line_deg": _round1(angle.get("down_the_line_deg")),
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
            mapped_face_frame = map_dtl_time_to_face_frame(
                float(raw_frame.get("t", 0.0)),
                {"video": face_result_meta},
                sync,
            )
            combined_frame = _nearest_frame(combined_lookup, combined_ids, mapped_face_frame)
        frames.append(
            {
                "frame": raw_frame.get("frame"),
                "t": raw_frame.get("t"),
                "keypoints": raw_frame.get("keypoints"),
                "shaft_angle_smooth": raw_frame.get("shaft_angle_smooth"),
                "shaft_smooth": raw_frame.get("shaft_smooth"),
                "body_points": _display_points(combined_frame),
                "display_angles": _display_angles(combined_frame),
                "hip_angle": raw_frame.get("hip_angle"),
                "chest_angle": raw_frame.get("chest_angle"),
            }
        )

    calibration = raw_result.get("calibration") or {"height_mm": combined_result.get("height_mm", 0.0)}
    return {
        "video": raw_result.get("video", {}),
        "calibration": calibration,
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
    }
