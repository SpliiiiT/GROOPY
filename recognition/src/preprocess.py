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
    import mediapipe as mp                       # optional dep — only needed to detect/crop the hand
    _mp_hands = mp.solutions.hands
except Exception:  # pragma: no cover - mediapipe optional at import time
    _mp_hands = None                             # module still imports without it

from .config import IMG_SIZE

# A single reusable detector. static_image_mode=True is best for offline dataset
# cropping; the live app should build its own with static_image_mode=False.
_HANDS = None                                    # cached detector (built once)


def _get_hands(static: bool = True):
    global _HANDS
    if _mp_hands is None:                         # guard: clear error if mediapipe is missing
        raise RuntimeError("mediapipe is not installed. pip install mediapipe")
    if _HANDS is None:                            # build the detector once, then reuse
        _HANDS = _mp_hands.Hands(
            static_image_mode=static,
            max_num_hands=1,                      # fingerspelling is one hand
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
    h, w = bgr_image.shape[:2]                    # frame height/width in pixels
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)   # MediaPipe expects RGB
    results = _get_hands(static).process(rgb)     # run hand detection
    if not results.multi_hand_landmarks:          # HAND-PRESENCE GATE: no hand -> caller skips this frame
        return None

    lm = results.multi_hand_landmarks[0].landmark  # the 21 landmarks of the first hand
    xs = [p.x for p in lm]                        # normalised x coords (0..1)
    ys = [p.y for p in lm]                        # normalised y coords (0..1)
    x_min, x_max = min(xs), max(xs)               # tight bounding box around the hand
    y_min, y_max = min(ys), max(ys)

    bw, bh = (x_max - x_min), (y_max - y_min)     # box width/height (normalised)
    x_min -= bw * margin                          # expand the box by `margin` so fingertips aren't clipped
    x_max += bw * margin
    y_min -= bh * margin
    y_max += bh * margin

    x1 = max(0, int(x_min * w))                   # normalised -> pixels, clamped to the frame
    y1 = max(0, int(y_min * h))
    x2 = min(w, int(x_max * w))
    y2 = min(h, int(y_max * h))
    if x2 <= x1 or y2 <= y1:                       # degenerate box -> treat as "no hand"
        return None
    return bgr_image[y1:y2, x1:x2]                # the cropped hand region (BGR)


def preprocess_for_model(bgr_image: np.ndarray) -> np.ndarray:
    """Resize + normalise a BGR image to a model-ready RGB float32 tensor in [0,1].

    Note: this returns [0,1] floats. The pre-trained backbones apply their own
    Keras preprocessing layer internally (see models/), so all candidates receive
    the same [0,1] input and each backbone rescales as it expects.
    """
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)   # BGR -> RGB
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)  # -> 224x224
    return (resized.astype(np.float32) / 255.0)   # 0..255 ints -> 0..1 floats


def crop_and_preprocess(bgr_image: np.ndarray, static: bool = True) -> Optional[np.ndarray]:
    """Convenience: crop to the hand then preprocess. Falls back to the full frame
    if no hand is detected (useful so the live demo never goes blank)."""
    crop = crop_hand(bgr_image, static=static)    # try to isolate the hand
    if crop is None:                              # no hand -> use the whole frame (gate downstream suppresses weak guess)
        crop = bgr_image
    return preprocess_for_model(crop)             # resize + normalise for the model
