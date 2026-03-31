import json
import os
import subprocess
from contextlib import contextmanager
from typing import Any, Dict, Tuple
import cv2


@contextmanager
def temporary_cwd(path: str):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def load_video_meta(video_path: str) -> Tuple[cv2.VideoCapture, Dict[str, float]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0:
        fps = 30.0
    return cap, {
        "fps": float(fps),
        "width": width,
        "height": height,
        "frame_count": frame_count,
    }


def normalize_rotation_degrees(value: object) -> int:
    """Normalize raw rotation metadata into one of {0, 90, 180, 270}."""
    try:
        deg = int(round(float(value)))
    except (TypeError, ValueError):
        return 0
    deg %= 360
    # Keep nearest right-angle; avoids odd values like 89.999 or -270.
    return min((0, 90, 180, 270), key=lambda c: abs(c - deg))


def probe_video_rotation_with_details(video_path: str) -> Tuple[int, Dict[str, Any]]:
    """Read rotation via ffprobe and return (normalized_degrees, diagnostics dict)."""
    details: Dict[str, Any] = {
        "video_path": video_path,
        "ffprobe_cmd": None,
        "ffprobe_not_found": False,
        "ffprobe_subprocess_error": None,
        "ffprobe_returncode": None,
        "ffprobe_stderr_tail": "",
        "ffprobe_json_preview": "",
        "stream_count": 0,
        "stream_tags": {},
        "side_data_summary": [],
        "rotation_source": None,
        "rotation_raw_value": None,
        "normalized_degrees": 0,
    }
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,codec_name,side_data_list:stream_tags=rotate,rotation",
        "-of",
        "json",
        video_path,
    ]
    details["ffprobe_cmd"] = " ".join(cmd)
    try:
        proc = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=8,
        )
    except FileNotFoundError:
        details["ffprobe_not_found"] = True
        return 0, details
    except (OSError, subprocess.SubprocessError) as exc:
        details["ffprobe_subprocess_error"] = repr(exc)
        return 0, details

    details["ffprobe_returncode"] = proc.returncode
    if proc.stderr:
        details["ffprobe_stderr_tail"] = proc.stderr.strip()[-800:]

    stdout = (proc.stdout or "").strip()
    if proc.returncode != 0 or not stdout:
        return 0, details

    details["ffprobe_json_preview"] = stdout[:2500] + ("..." if len(stdout) > 2500 else "")
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        details["ffprobe_json_error"] = str(exc)
        return 0, details

    streams = payload.get("streams") or []
    details["stream_count"] = len(streams)
    if not streams:
        return 0, details

    stream = streams[0] if isinstance(streams[0], dict) else {}
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    details["stream_tags"] = dict(tags) if tags else {}
    if "width" in stream:
        details["coded_width"] = stream.get("width")
    if "height" in stream:
        details["coded_height"] = stream.get("height")
    if "codec_name" in stream:
        details["codec_name"] = stream.get("codec_name")

    for key in ("rotate", "rotation"):
        if key in tags:
            raw = tags.get(key)
            details["rotation_source"] = f"tags.{key}"
            details["rotation_raw_value"] = raw
            deg = normalize_rotation_degrees(raw)
            details["normalized_degrees"] = deg
            return deg, details

    side_data_list = stream.get("side_data_list")
    if isinstance(side_data_list, list):
        for i, item in enumerate(side_data_list):
            if not isinstance(item, dict):
                continue
            sdt = item.get("side_data_type")
            summary = {"index": i, "side_data_type": sdt}
            if "displaymatrix" in item:
                dm = item.get("displaymatrix")
                if isinstance(dm, str):
                    summary["displaymatrix_preview"] = dm.strip()[:120]
            details["side_data_summary"].append(summary)
            for key in ("rotation", "rotate"):
                if key in item:
                    raw = item.get(key)
                    details["rotation_source"] = f"side_data[{i}].{key}"
                    details["rotation_raw_value"] = raw
                    deg = normalize_rotation_degrees(raw)
                    details["normalized_degrees"] = deg
                    return deg, details

    return 0, details


def probe_video_rotation_degrees(video_path: str) -> int:
    """Read video rotation metadata via ffprobe, defaulting to 0."""
    deg, _ = probe_video_rotation_with_details(video_path)
    return deg
