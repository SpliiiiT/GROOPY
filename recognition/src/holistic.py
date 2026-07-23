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
    import mediapipe as mp                       # optional heavy dep — only needed to EXTRACT landmarks
    _mp_holistic = mp.solutions.holistic
except Exception:  # pragma: no cover - mediapipe optional at import time
    _mp_holistic = None                          # module still imports without mediapipe (tests, headless use)

from shared.config import FRAME_FEATURES, HAND_FEATURES, POSE_FEATURES

_HOLISTIC = None                                 # cached detector (building the graph is expensive)


def _get_holistic(static: bool = False):
    global _HOLISTIC
    if _mp_holistic is None:                     # guard: fail clearly if mediapipe isn't installed
        raise RuntimeError("mediapipe is not installed. pip install mediapipe")
    if _HOLISTIC is None:                         # build once, reuse afterwards (singleton)
        _HOLISTIC = _mp_holistic.Holistic(
            static_image_mode=static,             # False = video mode (tracks between frames, faster)
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    return _HOLISTIC


def _pose_vec(landmarks) -> np.ndarray:
    if landmarks is None:                         # no pose detected this frame -> zeros (keeps length fixed)
        return np.zeros(POSE_FEATURES, dtype=np.float32)   # 33*4 = 132
    # each pose point contributes x, y, z, visibility -> flatten to a 1-D vector
    return np.array(
        [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark], dtype=np.float32
    ).flatten()


def _hand_vec(landmarks) -> np.ndarray:
    if landmarks is None:                         # hand not visible -> zeros
        return np.zeros(HAND_FEATURES, dtype=np.float32)   # 21*3 = 63
    # each hand point contributes x, y, z (no visibility for hands)
    return np.array(
        [[lm.x, lm.y, lm.z] for lm in landmarks.landmark], dtype=np.float32
    ).flatten()


def landmarks_from_results(results) -> np.ndarray:
    """Concatenate pose + left-hand + right-hand landmarks into one (FRAME_FEATURES,) vector."""
    return np.concatenate(
        [
            _pose_vec(results.pose_landmarks),        # [0:132]   body pose
            _hand_vec(results.left_hand_landmarks),   # [132:195] left hand
            _hand_vec(results.right_hand_landmarks),  # [195:258] right hand
        ]
    ).astype(np.float32)                              # fixed layout: pose | LH | RH = 258


def landmarks(bgr_frame: np.ndarray, static: bool = False) -> np.ndarray:
    """Extract the (FRAME_FEATURES,) landmark vector from one BGR frame.

    Missing pose/hands are zero-filled, so the output length is always FRAME_FEATURES.
    """
    import cv2  # lazy

    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)  # MediaPipe wants RGB; OpenCV gives BGR
    results = _get_holistic(static).process(rgb)      # run the detector -> pose + hands
    vec = landmarks_from_results(results)             # pack into the 258-vector
    assert vec.shape[0] == FRAME_FEATURES, (vec.shape, FRAME_FEATURES)  # sanity-check the length
    return vec


def empty_frame() -> np.ndarray:
    """A zero landmark vector (used to pad short sequences)."""
    return np.zeros(FRAME_FEATURES, dtype=np.float32)


# Feature-vector layout (see landmarks_from_results): pose 33*(x,y,z,vis) | LH 21*(x,y,z) |
# RH 21*(x,y,z). Pose shoulders are landmarks 11 (left) and 12 (right).
_POSE = 33 * 4                                    # index where the hands start (=132)
_SHOULDER_L = 11 * 4   # x at this index, y at +1   (left shoulder in the pose block)
_SHOULDER_R = 12 * 4   #                             (right shoulder)


def normalize_sequence(seq: np.ndarray) -> np.ndarray:
    """Make a landmark sequence translation- and scale-invariant, per frame.

    Recenters x,y on the shoulder midpoint and divides by shoulder width, so the model
    learns hand motion RELATIVE to the body rather than absolute image position/distance.
    Only x,y are normalized (z and pose visibility are noisy and left as-is). Absent/zero
    landmarks stay zero; frames without a detected pose are left unchanged.

    Applied identically at training (sequence_data) and live inference (word_stream) — no
    train/serve skew. Input/return shape: (T, FRAME_FEATURES).
    """
    seq = seq.astype(np.float32).copy()          # don't mutate the caller's array
    T = seq.shape[0]                             # number of frames

    pose = seq[:, :_POSE].reshape(T, 33, 4)          # (T,33, x/y/z/vis)  — reshape flat block into points
    lh = seq[:, _POSE:_POSE + 63].reshape(T, 21, 3)  # left hand points
    rh = seq[:, _POSE + 63:].reshape(T, 21, 3)       # right hand points

    Lx, Ly = seq[:, _SHOULDER_L], seq[:, _SHOULDER_L + 1]   # left-shoulder x,y per frame
    Rx, Ry = seq[:, _SHOULDER_R], seq[:, _SHOULDER_R + 1]   # right-shoulder x,y per frame
    valid = ~((Lx == 0) & (Ly == 0)) & ~((Rx == 0) & (Ry == 0))   # (T,) frames where BOTH shoulders were seen

    cx = (Lx + Rx) / 2.0                          # shoulder-midpoint x  -> our translation origin
    cy = (Ly + Ry) / 2.0                          # shoulder-midpoint y
    scale = np.hypot(Lx - Rx, Ly - Ry)            # shoulder width       -> our scale unit (distance-invariant)
    scale = np.where(scale < 1e-6, 1.0, scale)    # avoid divide-by-zero
    # Frames without a reliable reference: no-op (center 0, scale 1).
    cx = np.where(valid, cx, 0.0)                 # invalid frame -> don't shift
    cy = np.where(valid, cy, 0.0)
    scale = np.where(valid, scale, 1.0)           # invalid frame -> don't rescale

    def _apply_xy(block: np.ndarray) -> None:
        present = ~((block[:, :, 0] == 0) & (block[:, :, 1] == 0))   # (T,N) which landmarks actually exist
        nx = (block[:, :, 0] - cx[:, None]) / scale[:, None]         # recenter + rescale x
        ny = (block[:, :, 1] - cy[:, None]) / scale[:, None]         # recenter + rescale y
        block[:, :, 0] = np.where(present, nx, block[:, :, 0])       # only touch real landmarks (keep zeros as zeros)
        block[:, :, 1] = np.where(present, ny, block[:, :, 1])

    _apply_xy(pose)   # z (col 2) and visibility (col 3) untouched
    _apply_xy(lh)     # normalise left-hand x,y
    _apply_xy(rh)     # normalise right-hand x,y

    seq[:, :_POSE] = pose.reshape(T, _POSE)          # write the normalised points back into the flat vector
    seq[:, _POSE:_POSE + 63] = lh.reshape(T, 63)
    seq[:, _POSE + 63:] = rh.reshape(T, 63)
    return seq


def video_to_sequence(
    path: str,
    seq_len: int,
    static: bool = True,
    frame_start: int = 1,
    frame_end: int = -1,
) -> Optional[np.ndarray]:
    """Extract a (seq_len, FRAME_FEATURES) landmark sequence from a video file.

    Uniformly samples seq_len frames across the clip (pad with zeros if shorter). Returns
    None if the video can't be opened / has no frames. Used by sequence_data.py and the
    WLASL preparer.

    frame_start/frame_end restrict extraction to a sub-range (1-indexed, inclusive), matching
    the WLASL instance annotations. frame_end=-1 means "to the end"; frame_start=1 means
    "from the beginning".
    """
    import cv2  # lazy

    cap = cv2.VideoCapture(path)                 # open the video file
    frames = []
    while True:                                  # read every frame into memory
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:                               # unreadable / empty clip
        return None

    # Apply the WLASL frame range (1-indexed, inclusive).
    lo = max(0, frame_start - 1)                 # convert 1-indexed start -> 0-indexed
    hi = len(frames) if frame_end is None or frame_end < 0 else min(len(frames), frame_end)
    frames = frames[lo:hi] or frames  # fall back to all frames if the range is empty
    if not frames:
        return None

    # Uniformly sample seq_len indices across the selected frames.
    idxs = np.linspace(0, len(frames) - 1, num=seq_len).astype(int)   # evenly spaced frame indices
    seq = np.stack([landmarks(frames[i], static=static) for i in idxs])  # extract landmarks at each -> (seq_len,258)
    return seq.astype(np.float32)
