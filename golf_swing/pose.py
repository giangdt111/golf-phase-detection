import os
import sys


def import_mmpose():
    # Try installed mmpose (PyPI / conda) first; it may be a newer version
    # than the vendored mmpose-main/ (v1.0.0) bundled in this repo.
    try:
        from mmpose.apis import MMPoseInferencer  # type: ignore
        return MMPoseInferencer
    except Exception:
        pass

    # Fall back to the vendored mmpose-main/ source tree.
    try:
        mmpose_path = os.path.join(os.path.dirname(__file__), "..", "mmpose-main")
        mmpose_path = os.path.abspath(mmpose_path)
        if mmpose_path not in sys.path:
            sys.path.insert(0, mmpose_path)
        from mmpose.apis import MMPoseInferencer  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "Failed to import mmpose. Install mmpose (pip install mmpose) or add mmpose to PYTHONPATH."
        ) from exc
    return MMPoseInferencer
