#!/usr/bin/env python3
"""
RabbitMQ-backed golf analysis service.

Consumes analysis jobs from the backend, runs swing phase + pose detection on
the source video, and responds with phases, pose_frames, and video duration.
"""

import asyncio
import json
import os
import random
import re
import shutil
import tempfile
import traceback
import urllib.request
from enum import Enum
from typing import Any, Dict, List, Optional

import aio_pika
from aiormq.exceptions import ProbableAuthenticationError

from golf_swing.pipeline import SwingInferenceService


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
        """Build body_frames from pipeline result (per-frame body_points + hip_angle)."""
        frames = result.get("frames") or []
        out: List[Dict[str, Any]] = []
        for frame in frames:
            t = float(frame.get("t", 0.0))
            timestamp_ms = int(round(t * 1000.0))
            out.append(
                {
                    "timestamp_ms": timestamp_ms,
                    "hip_angle": frame.get("hip_angle"),
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

    def _run_inference(self, video_path: str) -> Dict[str, Any]:
        kwargs = self._inference_kwargs()
        return self._service.run(video_path=video_path, **kwargs)

    @staticmethod
    def _save_output_json(data: Dict[str, Any]) -> None:
        output_path = os.path.join(os.path.dirname(__file__), OUTPUT_JSON_FILE)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    async def process_request(self, message: aio_pika.abc.AbstractIncomingMessage) -> None:
        async with message.process():
            request_id: Optional[str] = None
            temp_video_path: Optional[str] = None
            try:
                data = json.loads(message.body.decode("utf-8"))
                request_id = data.get("id")
                payload = data.get("payload", {})
                video_url = payload.get("video_url")
                input_duration_ms = int(payload.get("video_duration_ms", 0) or 0)

                if not request_id:
                    raise ValueError("Missing request id")
                if not video_url:
                    raise ValueError("Missing payload.video_url")

                print(f"Received request id={request_id} video_url={video_url}")
                temp_video_path = await asyncio.to_thread(self._download_video, video_url)
                print(f"Downloaded video to {temp_video_path}")

                result = await asyncio.to_thread(self._run_inference, temp_video_path)
                video_duration_ms = self._video_duration_ms(result, input_duration_ms)
                phases = self._build_phases(result, video_duration_ms)
                pose_frames = self._build_pose_frames(result)
                segmentation_frames = self._build_segmentation_frames(result)
                body_frames = self._build_body_frames(result)
                video_info = self._build_video_info(result)

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
        await asyncio.to_thread(
            self._service.warmup,
            **self._inference_kwargs(),
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
