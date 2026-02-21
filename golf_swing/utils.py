import os
from contextlib import contextmanager
from typing import Dict, Tuple
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
