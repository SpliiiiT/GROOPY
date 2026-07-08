"""MediaPipe Holistic landmark extraction for the dynamic word module.

Turns a single BGR frame into a fixed-length landmark feature vector (pose + both hands).
The SAME function is used for offline training-sequence extraction and live inference, so
there is no train/serve skew — the same principle the static pipeline follows in
preprocess.py. Feature layout matches shared.config.FRAME_FEATURES (258).

mediapipe is imported lazily (optional at import time), mirroring preprocess._get_hands.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import mediapipe as mp
    _mp_holistic = mp.solutions.holistic
except Exception:  # pragma: no cover - mediapipe optional at import time
    _mp_holistic = None

from shared.config import FRAME_FEATURES, HAND_FEATURES, POSE_FEATURES

_HOLISTIC = None


def _get_holistic(static: bool = False):
    global _HOLISTIC
    if _mp_holistic is None:
        raise RuntimeError("mediapipe is not installed. pip install mediapipe")
    if _HOLISTIC is None:
        _HOLISTIC = _mp_holistic.Holistic(
            static_image_mode=static,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _HOLISTIC


def _pose_vec(landmarks) -> np.ndarray:
    if landmarks is None:
        return np.zeros(POSE_FEATURES, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark], dtype=np.float32
    ).flatten()


def _hand_vec(landmarks) -> np.ndarray:
    if landmarks is None:
        return np.zeros(HAND_FEATURES, dtype=np.float32)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
    ).flatten()


def landmarks_from_results(results) -> np.ndarray:
    """Concatenate pose + left-hand + right-hand landmarks into one (FRAME_FEATURES,) vector."""
    return np.concatenate(
        [
            _pose_vec(results.pose_landmarks),
            _hand_vec(results.left_hand_landmarks),
            _hand_vec(results.right_hand_landmarks),
        ]
    ).astype(np.float32)


def landmarks(bgr_frame: np.ndarray, static: bool = False) -> np.ndarray:
    """Extract the (FRAME_FEATURES,) landmark vector from one BGR frame.

    Missing pose/hands are zero-filled, so the output length is always FRAME_FEATURES.
    """
    import cv2  # lazy

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    results = _get_holistic(static).process(rgb)
    vec = landmarks_from_results(results)
    assert vec.shape[0] == FRAME_FEATURES, (vec.shape, FRAME_FEATURES)
    return vec


def empty_frame() -> np.ndarray:
    """A zero landmark vector (used to pad short sequences)."""
    return np.zeros(FRAME_FEATURES, dtype=np.float32)


def video_to_sequence(path: str, seq_len: int, static: bool = True) -> Optional[np.ndarray]:
    """Extract a (seq_len, FRAME_FEATURES) landmark sequence from a video file.

    Uniformly samples seq_len frames across the clip (pad with zeros if shorter). Returns
    None if the video can't be opened / has no frames. Used by sequence_data.py.
    """
    import cv2  # lazy

    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return None

    # Uniformly sample seq_len indices across the available frames.
    idxs = np.linspace(0, len(frames) - 1, num=seq_len).astype(int)
    seq = np.stack([landmarks(frames[i], static=static) for i in idxs])
    return seq.astype(np.float32)
