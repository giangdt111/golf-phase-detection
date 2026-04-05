#!/usr/bin/env python3
"""
RabbitMQ-backed golf analysis service.

Consumes analysis jobs from the backend, runs swing phase + pose detection on
the source video, and responds with phases, pose_frames, and video duration.
"""

import asyncio
import concurrent.futures
import json
import os
import random
import re
import shlex
import shutil
import subprocess
import tempfile
import traceback
import urllib.request
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import aio_pika
from aiormq.exceptions import ProbableAuthenticationError

from golf_swing.pipeline import SwingInferenceService
from golf_swing.utils import probe_video_rotation_with_details


REQUEST_QUEUE = "request_golf_analysis"
RESPONSE_QUEUE = "response_golf_analysis"
OUTPUT_JSON_FILE = "analysis_queue_service.json"


def _redact_rabbitmq_url(url: str) -> str:
    """Redact password in amqp URL for safe logging."""
    return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", url)


class GolfPhase(str, Enum):
    P1_BREAK_AWAY = "P1: Break-away"
    P2_MID_BACKSWING = "P2: Mid Backswing"
    P3_LATE_BACKSWING = "P3: Late Backswing"
    P4_TOP_OF_BACKSWING = "P4: Top of Backswing"
    P5_EARLY_DOWNSWING = "P5: Early Downswing"
    P6_MID_DOWNSWING = "P6: Mid downswing"
    P7_BALL_IMPACT = "P7: Ball impact"
    P8_MID_FOLLOWTHROUGH = "P8: Mid followthrough"
    P9_LATE_FOLLOWTHROUGH = "P9: Late followthrough"


class AnalysisMessage(str, Enum):
    GOOD = "Good"
    WARNING = "Warning"
    INCORRECT = "Incorrect"


MOCK_DESCRIPTIONS: Dict[GolfPhase, Dict[AnalysisMessage, str]] = {
    GolfPhase.P1_BREAK_AWAY: {
        AnalysisMessage.GOOD: "Excellent break-away! Club is moving away from the ball smoothly with proper wrist hinge.",
        AnalysisMessage.WARNING: "Break-away is slightly inside. Try to keep the club on the target line initially.",
        AnalysisMessage.INCORRECT: "Poor break-away position. The club should move straight back, not around your body.",
    },
    GolfPhase.P2_MID_BACKSWING: {
        AnalysisMessage.GOOD: "Great mid backswing position! Arms are extended and club is on plane.",
        AnalysisMessage.WARNING: "Club is slightly off-plane. Focus on keeping your left arm straight.",
        AnalysisMessage.INCORRECT: "Club is too steep at mid backswing. Work on maintaining proper swing plane.",
    },
    GolfPhase.P3_LATE_BACKSWING: {
        AnalysisMessage.GOOD: "Excellent late backswing! Shoulders are turning well with good hip resistance.",
        AnalysisMessage.WARNING: "Slight over-rotation in late backswing. Maintain hip resistance for power.",
        AnalysisMessage.INCORRECT: "Too much sway in late backswing. Keep your head centered over the ball.",
    },
    GolfPhase.P4_TOP_OF_BACKSWING: {
        AnalysisMessage.GOOD: "Perfect position at the top! Club is parallel with excellent wrist hinge.",
        AnalysisMessage.WARNING: "Club crosses the line slightly at the top. Stop when parallel.",
        AnalysisMessage.INCORRECT: "Club is too short at the top. You need more shoulder turn for power.",
    },
    GolfPhase.P5_EARLY_DOWNSWING: {
        AnalysisMessage.GOOD: "Excellent early downswing! Hips are clearing ahead of the shoulders.",
        AnalysisMessage.WARNING: "Transition is slightly rushed. Start with the lower body to generate power.",
        AnalysisMessage.INCORRECT: "Casting the club from the top. Maintain wrist angle in early downswing.",
    },
    GolfPhase.P6_MID_DOWNSWING: {
        AnalysisMessage.GOOD: "Great mid downswing! Lag is maintained perfectly with hips leading.",
        AnalysisMessage.WARNING: "Losing some lag at mid downswing. Keep wrists hinged until impact.",
        AnalysisMessage.INCORRECT: "Early release of the club. This causes loss of power and accuracy.",
    },
    GolfPhase.P7_BALL_IMPACT: {
        AnalysisMessage.GOOD: "Perfect impact position! Hands ahead of ball with proper shaft lean.",
        AnalysisMessage.WARNING: "Slight flip at impact. Hands should lead the clubhead through contact.",
        AnalysisMessage.INCORRECT: "Scooping at impact. Hands must be ahead of the clubhead for solid contact.",
    },
    GolfPhase.P8_MID_FOLLOWTHROUGH: {
        AnalysisMessage.GOOD: "Excellent mid follow-through! Arms are extending toward the target.",
        AnalysisMessage.WARNING: "Chicken-wing forming in follow-through. Let your arms extend fully.",
        AnalysisMessage.INCORRECT: "Arms are collapsing. Extend through the ball for better distance.",
    },
    GolfPhase.P9_LATE_FOLLOWTHROUGH: {
        AnalysisMessage.GOOD: "Balanced finish with proper weight transfer! Great athletic position.",
        AnalysisMessage.WARNING: "Finish is slightly off-balance. Focus on posting up on your front leg.",
        AnalysisMessage.INCORRECT: "Incomplete finish. Chest should face target with belt buckle pointing left.",
    },
}


EVENT_TO_PHASE: Dict[str, GolfPhase] = {
    "address": GolfPhase.P1_BREAK_AWAY,
    "mid-backswing": GolfPhase.P2_MID_BACKSWING,
    "late-backswing": GolfPhase.P3_LATE_BACKSWING,
    "top-backswing": GolfPhase.P4_TOP_OF_BACKSWING,
    "early-downswing": GolfPhase.P5_EARLY_DOWNSWING,
    "mid-downswing": GolfPhase.P6_MID_DOWNSWING,
    "ball-impact": GolfPhase.P7_BALL_IMPACT,
    "mid-followthrough": GolfPhase.P8_MID_FOLLOWTHROUGH,
    "late-followthrough": GolfPhase.P9_LATE_FOLLOWTHROUGH,
}


def _load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("\"'")
                if key and key not in os.environ:
                    os.environ[key] = val
    except Exception:
        return


def _env_bool(name: str, default: bool = False) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes", "y", "on")


class AnalysisQueueService:
    def __init__(self) -> None:
        self._channel: Optional[aio_pika.abc.AbstractChannel] = None
        self._service = SwingInferenceService()
        # Single-thread executor so warmup and all inference share the same
        # CUDA context.  asyncio.to_thread uses the default pool which can
        # assign a different OS thread each call, losing the cuDNN context.
        self._gpu_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def _inference_kwargs(self) -> Dict[str, Any]:
        return {
            "pose2d": os.getenv(
                "POSE2D_CONFIG",
                "mmpose-main/configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w48_8xb32-210e_coco-256x192.py",
            ),
            "pose2d_weights": os.getenv(
                "POSE2D_WEIGHTS",
                "model/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth",
            ),
            "det_model": os.getenv("DET_MODEL", "whole_image"),
            "det_weights": os.getenv("DET_WEIGHTS"),
            "device": os.getenv("DEVICE"),
            "stride": int(os.getenv("STRIDE", "1")),
            "max_frames": int(os.getenv("MAX_FRAMES", "0")) or None,
            "swing_direction": os.getenv("SWING_DIRECTION"),
            "seg_model_path": None if _env_bool("SEG_DISABLE", False) else os.getenv("SEG_MODEL", "golf_segment/best.pt"),
            "seg_imgsz": int(os.getenv("SEG_IMGSZ", "960")),
            "seg_conf": float(os.getenv("SEG_CONF", "0.25")),
            "seg_iou": float(os.getenv("SEG_IOU", "0.7")),
            "seg_device": os.getenv("SEG_DEVICE", "cpu"),
            "person_det_model": os.getenv("PERSON_DET_MODEL", "model/yolov8n.pt"),
            "person_det_conf": float(os.getenv("PERSON_DET_CONF", "0.25")),
            "person_det_iou": float(os.getenv("PERSON_DET_IOU", "0.7")),
            "person_det_imgsz": int(os.getenv("PERSON_DET_IMGSZ", "640")),
            "force_yolo_person": _env_bool("FORCE_PERSON_YOLO", False),
        }

    @staticmethod
    def _download_video(video_url: str) -> str:
        suffix = os.path.splitext(video_url.split("?", 1)[0])[1] or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            with urllib.request.urlopen(video_url, timeout=90) as response:
                shutil.copyfileobj(response, temp_file)
            return temp_file.name

    @staticmethod
    def _build_pose_frames(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        pose_frames: List[Dict[str, Any]] = []
        for frame in result.get("frames", []):
            t = float(frame.get("t", 0.0))
            keypoints = frame.get("keypoints") or []
            landmarks = []
            for kp in keypoints:
                landmarks.append(
                    {
                        "type": kp.get("name", ""),
                        "x": float(kp.get("x", 0.0)),
                        "y": float(kp.get("y", 0.0)),
                        "z": 0.0,
                        "likelihood": float(kp.get("score", 0.0)),
                    }
                )
            pose_frames.append(
                {
                    "timestamp_ms": int(round(t * 1000.0)),
                    "landmarks": landmarks,
                }
            )
        return pose_frames

    @staticmethod
    def _video_duration_ms(result: Dict[str, Any], fallback_duration_ms: int) -> int:
        video_data = result.get("video", {})
        fps = float(video_data.get("fps", 0.0) or 0.0)
        frame_count = int(video_data.get("frame_count", 0) or 0)
        if fps > 0.0 and frame_count > 0:
            return int(round((frame_count / fps) * 1000.0))
        return int(fallback_duration_ms)

    @staticmethod
    def _build_phases(result: Dict[str, Any], video_duration_ms: int) -> List[Dict[str, Any]]:
        events = result.get("events") or []
        normalized_events = []
        for ev in events:
            ev_name = str(ev.get("name", "")).strip().lower()
            phase = EVENT_TO_PHASE.get(ev_name)
            if not phase:
                continue
            normalized_events.append(
                {
                    "phase": phase,
                    "t_ms": int(round(float(ev.get("t", 0.0)) * 1000.0)),
                }
            )

        phases: List[Dict[str, Any]] = []
        for i, ev in enumerate(normalized_events):
            phase: GolfPhase = ev["phase"]
            start_time_ms = ev["t_ms"]
            if i + 1 < len(normalized_events):
                end_time_ms = normalized_events[i + 1]["t_ms"]
            else:
                end_time_ms = video_duration_ms
            end_time_ms = max(end_time_ms, start_time_ms)

            message = random.choice(list(AnalysisMessage))
            phases.append(
                {
                    "phase_name": phase.value,
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                    "message": message.value,
                    "description": MOCK_DESCRIPTIONS[phase][message],
                }
            )
        return phases

    @staticmethod
    def _point_to_dict(point) -> Optional[Dict[str, float]]:
        """Convert a [x, y] list/tuple to {"x": ..., "y": ...}, or None."""
        if point is None:
            return None
        try:
            x, y = float(point[0]), float(point[1])
            return {"x": x, "y": y}
        except (TypeError, IndexError, ValueError):
            return None

    @staticmethod
    def _build_segmentation_frames(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build segmentation_frames from pipeline result (per-frame shaft/club data)."""
        frames = result.get("frames") or []
        out: List[Dict[str, Any]] = []
        to_pt = AnalysisQueueService._point_to_dict
        for frame in frames:
            t = float(frame.get("t", 0.0))
            timestamp_ms = int(round(t * 1000.0))
            out.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "shaft_angle": frame.get("shaft_angle"),
                    "shaft_angle_smooth": frame.get("shaft_angle_smooth"),
                    "club_center": to_pt(frame.get("head_smooth")),
                    "shaft_center": to_pt(frame.get("shaft_smooth")),
                }
            )
        return out

    @staticmethod
    def _build_body_frames(result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Build body_frames from pipeline result (per-frame body_points + body angles)."""
        frames = result.get("frames") or []
        out: List[Dict[str, Any]] = []
        for frame in frames:
            t = float(frame.get("t", 0.0))
            timestamp_ms = int(round(t * 1000.0))
            out.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "hip_angle": frame.get("hip_angle"),
                    "chest_angle": frame.get("chest_angle"),
                    "body_points": frame.get("body_points"),
                }
            )
        return out

    @staticmethod
    def _build_video_info(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Extract video metadata from pipeline result."""
        video = result.get("video")
        if not isinstance(video, dict):
            return None
        return {
            "fps": video.get("fps"),
            "width": video.get("width"),
            "height": video.get("height"),
            "frame_count": video.get("frame_count"),
            "stride": video.get("stride"),
        }

    @staticmethod
    def _parse_height_mm(payload: Dict[str, Any]) -> Optional[float]:
        """Extract player height in millimeters from payload."""
        # Accept both snake_case and camelCase keys from different clients.
        mm_candidates = (
            "height_mm",
            "heightMm",
            "player_height_mm",
            "playerHeightMm",
        )
        cm_candidates = (
            "height_cm",
            "heightCm",
            "player_height_cm",
            "playerHeightCm",
        )
        m_candidates = (
            "height_m",
            "heightM",
            "player_height_m",
            "playerHeightM",
        )

        def _to_positive_float(val: Any) -> Optional[float]:
            if val is None:
                return None
            try:
                n = float(val)
            except (TypeError, ValueError):
                return None
            if n <= 0:
                return None
            return n

        for key in mm_candidates:
            n = _to_positive_float(payload.get(key))
            if n is not None:
                return n
        for key in cm_candidates:
            n = _to_positive_float(payload.get(key))
            if n is not None:
                return n * 10.0
        for key in m_candidates:
            n = _to_positive_float(payload.get(key))
            if n is not None:
                return n * 1000.0
        return None

    def _run_inference(self, video_path: str, height_mm: Optional[float]) -> Dict[str, Any]:
        kwargs = self._inference_kwargs()
        return self._service.run(video_path=video_path, height_mm=height_mm, **kwargs)

    @staticmethod
    def _rotation_vf(rotation_degrees: int) -> Optional[str]:
        if rotation_degrees == 90:
            return "transpose=1"
        if rotation_degrees == 180:
            return "hflip,vflip"
        if rotation_degrees == 270:
            return "transpose=2"
        return None

    @staticmethod
    def _log_rotation_probe(prefix: str, details: Dict[str, Any]) -> None:
        """Print ffprobe diagnostics for video rotation (container metadata)."""
        print(f"{prefix} --- rotation probe (ffprobe) ---")
        cmd = details.get("ffprobe_cmd")
        if cmd:
            print(f"{prefix} cmd: {cmd}")
        if details.get("ffprobe_not_found"):
            print(
                f"{prefix} ffprobe not found on PATH — install ffmpeg package; "
                "cannot read or fix rotation metadata."
            )
            return
        if details.get("ffprobe_subprocess_error"):
            print(f"{prefix} ffprobe subprocess error: {details['ffprobe_subprocess_error']}")
            return
        rc = details.get("ffprobe_returncode")
        print(f"{prefix} ffprobe returncode={rc}")
        if details.get("ffprobe_stderr_tail"):
            print(f"{prefix} ffprobe stderr (tail): {details['ffprobe_stderr_tail']!r}")
        cw = details.get("coded_width")
        ch = details.get("coded_height")
        codec = details.get("codec_name")
        if cw is not None or ch is not None or codec:
            print(f"{prefix} video stream: codec={codec!r} coded_size={cw}x{ch}")
        tags = details.get("stream_tags") or {}
        if tags:
            # Most relevant for orientation debugging
            rot_hint = {k: tags[k] for k in ("rotate", "rotation") if k in tags}
            print(f"{prefix} stream tags (subset): {rot_hint or tags}")
        side = details.get("side_data_summary") or []
        if side:
            print(f"{prefix} side_data_list summary: {json.dumps(side, default=str)}")
        src = details.get("rotation_source")
        raw = details.get("rotation_raw_value")
        norm = details.get("normalized_degrees", 0)
        print(
            f"{prefix} parsed rotation: source={src!r} raw={raw!r} normalized_degrees={norm}"
        )
        if norm == 0 and not src:
            print(
                f"{prefix} no rotate/rotation in tags or side_data — "
                "file may truly be landscape pixels, or uses a format ffprobe did not expose here."
            )
        preview = details.get("ffprobe_json_preview")
        if preview:
            print(f"{prefix} ffprobe json preview:\n{preview}")

    @staticmethod
    def _normalize_video_orientation(
        video_path: str,
        log_prefix: str = "[orientation]",
    ) -> Tuple[str, int, Dict[str, Any]]:
        """
        Return (path_for_inference, applied_rotation_degrees, probe_details).
        If rotation metadata exists, produce an upright temp MP4 via ffmpeg.
        """
        rotation_degrees, probe_details = probe_video_rotation_with_details(video_path)
        AnalysisQueueService._log_rotation_probe(log_prefix, probe_details)

        vf = AnalysisQueueService._rotation_vf(rotation_degrees)
        if vf is None:
            print(
                f"{log_prefix} skip transcode: normalized rotation is 0 (no ffmpeg filter needed)."
            )
            return video_path, 0, probe_details

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as out_file:
            out_path = out_file.name

        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vf",
            vf,
            "-metadata:s:v:0",
            "rotate=0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            out_path,
        ]
        print(f"{log_prefix} running ffmpeg normalize: filter={vf!r}")
        print(f"{log_prefix} ffmpeg cmd: {shlex.join(cmd)}")
        print(f"{log_prefix} ffmpeg output path: {out_path}")
        try:
            proc = subprocess.run(
                cmd,
                check=False,
                capture_output=True,
                text=True,
                timeout=300,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError) as exc:
            try:
                os.remove(out_path)
            except OSError:
                pass
            print(
                f"{log_prefix} ffmpeg unavailable/failed ({exc!r}). "
                "Using original source video for inference."
            )
            return video_path, 0, probe_details

        if proc.returncode != 0:
            try:
                os.remove(out_path)
            except OSError:
                pass
            err_full = (proc.stderr or "").strip()
            err_tail = err_full[-1200:] if err_full else ""
            print(
                f"{log_prefix} ffmpeg failed exit={proc.returncode}. "
                f"Using original source video. stderr_tail:\n{err_tail}"
            )
            return video_path, 0, probe_details

        try:
            out_sz = os.path.getsize(out_path)
        except OSError:
            out_sz = -1
        print(
            f"{log_prefix} ffmpeg OK: normalized file size_bytes={out_sz} "
            f"(rotation applied in pixels={rotation_degrees})"
        )
        _, out_probe = probe_video_rotation_with_details(out_path)
        print(
            f"{log_prefix} post-transcode probe: coded="
            f"{out_probe.get('coded_width')}x{out_probe.get('coded_height')} "
            f"normalized_degrees={out_probe.get('normalized_degrees')} "
            f"tags={out_probe.get('stream_tags')}"
        )

        return out_path, rotation_degrees, probe_details

    @staticmethod
    def _save_output_json(data: Dict[str, Any]) -> None:
        output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def process_request(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process():
            request_id: Optional[str] = None
            temp_video_path: Optional[str] = None
            normalized_video_path: Optional[str] = None
            try:
                data = json.loads(message.body.decode("utf-8"))
                request_id = data.get("id")
                payload = data.get("payload", {})
                video_url = payload.get("video_url")
                input_duration_ms = int(payload.get("video_duration_ms", 0) or 0)
                height_mm = self._parse_height_mm(payload)

                if not request_id:
                    raise ValueError("Missing request id")
                if not video_url:
                    raise ValueError("Missing payload.video_url")

                print(f"Received request id={request_id} video_url={video_url}")
                temp_video_path = await asyncio.to_thread(self._download_video, video_url)
                try:
                    dl_sz = os.path.getsize(temp_video_path)
                except OSError:
                    dl_sz = -1
                print(
                    f"[request id={request_id}] downloaded path={temp_video_path} "
                    f"size_bytes={dl_sz}"
                )
                orientation_prefix = f"[orientation request_id={request_id}]"
                inference_video_path, applied_rotation, _probe_details = await asyncio.to_thread(
                    self._normalize_video_orientation, temp_video_path, orientation_prefix
                )
                print(
                    f"[request id={request_id}] inference_input_path={inference_video_path} "
                    f"applied_rotation_metadata_degrees={applied_rotation} "
                    f"(same_as_download={inference_video_path == temp_video_path})"
                )
                if inference_video_path != temp_video_path:
                    normalized_video_path = inference_video_path
                    print(
                        f"[request id={request_id}] using normalized temp file for inference: "
                        f"{normalized_video_path}"
                    )

                print(f"[request id={request_id}] starting SwingInferenceService.run ...")
                _loop = asyncio.get_running_loop()
                _vp, _hm = inference_video_path, height_mm
                result = await _loop.run_in_executor(
                    self._gpu_executor,
                    lambda: self._run_inference(_vp, _hm),
                )
                print(f"[request id={request_id}] inference finished")
                video_duration_ms = self._video_duration_ms(result, input_duration_ms)
                phases = self._build_phases(result, video_duration_ms)
                pose_frames = self._build_pose_frames(result)
                segmentation_frames = self._build_segmentation_frames(result)
                body_frames = self._build_body_frames(result)
                video_info = self._build_video_info(result)
                print(
                    f"[request id={request_id}] result video_info={video_info!r} "
                    f"pose_frames={len(pose_frames)}"
                )

                response = {
                    "id": request_id,
                    "payload": {
                        "phases": phases,
                        "pose_frames": pose_frames,
                        "video_duration_ms": video_duration_ms,
                        "segmentation_frames": segmentation_frames,
                        "body_frames": body_frames,
                        "swing_direction": result.get("swing_direction"),
                        "video_info": video_info,
                    },
                }
                self._save_output_json(response)
                print(f"Saved response JSON to {OUTPUT_JSON_FILE}")

                if not self._channel:
                    raise RuntimeError("Channel not initialized")
                await self._channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(response).encode("utf-8"),
                        content_type="application/json",
                        delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
                    ),
                    routing_key=RESPONSE_QUEUE,
                )
                print(
                    f"Responded id={request_id} phases={len(phases)} "
                    f"pose_frames={len(pose_frames)} duration_ms={video_duration_ms}"
                )
            except Exception as exc:
                print(f"Request processing failed id={request_id}: {exc}")
                traceback.print_exc()
            finally:
                if normalized_video_path and os.path.exists(normalized_video_path):
                    try:
                        os.remove(normalized_video_path)
                    except Exception:
                        pass
                if temp_video_path and os.path.exists(temp_video_path):
                    try:
                        os.remove(temp_video_path)
                    except Exception:
                        pass

    async def run(self) -> None:
        rabbitmq_url = os.getenv("RABBITMQ_URL", "amqp://guest:guest@localhost:5672/")
        try:
            connection = await aio_pika.connect_robust(rabbitmq_url)
        except ProbableAuthenticationError as e:
            print(
                "RabbitMQ login refused. Set RABBITMQ_URL with valid credentials "
                "(e.g. RABBITMQ_URL=amqp://user:password@localhost:5672/). "
                f"Attempted: {_redact_rabbitmq_url(rabbitmq_url)}"
            )
            raise SystemExit(1) from e
        except (OSError, ConnectionError) as e:
            print(
                "Could not connect to RabbitMQ. Ensure RabbitMQ is running and "
                "RABBITMQ_URL is correct. "
                f"Attempted: {_redact_rabbitmq_url(rabbitmq_url)}"
            )
            raise SystemExit(1) from e

        self._channel = await connection.channel()
        await self._channel.set_qos(prefetch_count=1)
        request_queue = await self._channel.declare_queue(REQUEST_QUEUE, durable=True)
        await self._channel.declare_queue(RESPONSE_QUEUE, durable=True)

        print(f"Connected to RabbitMQ: {_redact_rabbitmq_url(rabbitmq_url)}")
        print(f"Listening on queue: {REQUEST_QUEUE}")
        print(f"Responding on queue: {RESPONSE_QUEUE}")

        print("Loading and caching inference models...")
        loop = asyncio.get_running_loop()
        kwargs = self._inference_kwargs()
        await loop.run_in_executor(
            self._gpu_executor,
            lambda: self._service.warmup(**kwargs),
        )
        print("Models ready. Starting consumer.")

        await request_queue.consume(self.process_request)
        await asyncio.Future()


async def main() -> None:
    _load_env_file()
    service = AnalysisQueueService()
    await service.run()


if __name__ == "__main__":
    asyncio.run(main())
