import os
import sys


def import_mmpose():
    try:
        mmpose_path = os.path.join(os.path.dirname(__file__), "..", "mmpose-main")
        mmpose_path = os.path.abspath(mmpose_path)
        if mmpose_path not in sys.path:
            sys.path.insert(0, mmpose_path)
        from mmpose.apis import MMPoseInferencer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mmpose. Install dependencies or add mmpose to PYTHONPATH."
        ) from exc
    return MMPoseInferencer
