"""Hand-region preprocessing with MediaPipe.

The single most important robustness trick: we crop to the hand *before* classifying,
so the model learns the hand shape rather than the background. The same crop function is
used everywhere — training-time offline cropping, the desktop app, and live inference —
so there is no train/serve skew.

Two entry points:
  - crop_hand(bgr_image)      -> cropped BGR image around the detected hand (or None)
  - preprocess_for_model(img) -> float32 [0,1] RGB tensor of shape (IMG_SIZE, IMG_SIZE, 3)
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
except Exception:  # pragma: no cover - mediapipe optional at import time
    _mp_hands = None

from .config import IMG_SIZE

# A single reusable detector. static_image_mode=True is best for offline dataset
# cropping; the live app should build its own with static_image_mode=False.
_HANDS = None


def _get_hands(static: bool = True):
    global _HANDS
    if _mp_hands is None:
        raise RuntimeError("mediapipe is not installed. pip install mediapipe")
    if _HANDS is None:
        _HANDS = _mp_hands.Hands(
            static_image_mode=static,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )
    return _HANDS


def crop_hand(
    bgr_image: np.ndarray,
    margin: float = 0.25,
    static: bool = True,
) -> Optional[np.ndarray]:
    """Return a square-ish BGR crop around the detected hand, or None if no hand.

    margin expands the bounding box by this fraction on each side so we don't clip
    fingertips.
    """
    h, w = bgr_image.shape[:2]
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    results = _get_hands(static).process(rgb)
    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]
    ys = [p.y for p in lm]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    bw, bh = (x_max - x_min), (y_max - y_min)
    x_min -= bw * margin
    x_max += bw * margin
    y_min -= bh * margin
    y_max += bh * margin

    x1 = max(0, int(x_min * w))
    y1 = max(0, int(y_min * h))
    x2 = min(w, int(x_max * w))
    y2 = min(h, int(y_max * h))
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr_image[y1:y2, x1:x2]


def preprocess_for_model(bgr_image: np.ndarray) -> np.ndarray:
    """Resize + normalise a BGR image to a model-ready RGB float32 tensor in [0,1].

    Note: this returns [0,1] floats. The pre-trained backbones apply their own
    Keras preprocessing layer internally (see models/), so all candidates receive
    the same [0,1] input and each backbone rescales as it expects.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return (resized.astype(np.float32) / 255.0)


def crop_and_preprocess(bgr_image: np.ndarray, static: bool = True) -> Optional[np.ndarray]:
    """Convenience: crop to the hand then preprocess. Falls back to the full frame
    if no hand is detected (useful so the live demo never goes blank)."""
    crop = crop_hand(bgr_image, static=static)
    if crop is None:
        crop = bgr_image
    return preprocess_for_model(crop)
