"""Render a SignPlan to the screen (OpenCV).

This is the I/O layer — verified manually (it opens a window), while everything that builds
the plan above it stays headless-testable. cv2 is imported lazily so this module can be
imported (and the pipeline tested) without OpenCV installed.

  - WordClip     -> play the .mp4 clip frame by frame
  - Fingerspell  -> show one ASL-alphabet image per letter, hold each briefly
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from shared.config import LETTERS_DIR

from .gloss_to_signplan import Fingerspell, SignPlan, WordClip

_IMG_EXTS = (".jpg", ".jpeg", ".png")            # image types we accept for letter frames


def _first_image(dir_path: Path) -> Optional[Path]:
    if not dir_path.is_dir():                     # letter folder missing
        return None
    for p in sorted(dir_path.iterdir()):          # pick the first image file (a representative frame)
        if p.suffix.lower() in _IMG_EXTS:
            return p
    return None


def play_sign_plan(
    plan: SignPlan,
    fps: int = 25,                                # clip playback speed
    letter_hold_ms: int = 700,                    # how long each fingerspelled letter stays on screen
    letters_dir: Path = LETTERS_DIR,
    window: str = "GROOPY — Sign",
    on_missing: Optional[Callable[[str], None]] = None,   # callback for missing assets (defaults to print)
) -> None:
    """Play a plan in an OpenCV window. Missing assets are skipped (reported via on_missing).

    Blocks until the plan finishes or the user presses 'q'/Esc.
    """
    import cv2  # lazy: only needed for playback

    def _report(msg: str) -> None:
        (on_missing or print)(msg)                # report a missing asset without crashing

    delay = max(1, int(1000 / fps))               # ms to wait between frames (=> the fps)
    for step in plan.steps:                        # render each step in order
        if isinstance(step, WordClip):
            if not step.clip_path.is_file():        # clip file missing -> skip, don't crash
                _report(f"[missing clip] {step.gloss} -> {step.clip_path}")
                continue
            for _ in range(max(1, step.repeat)):    # sentiment emphasis (Decision A2): replay N times
                cap = cv2.VideoCapture(str(step.clip_path))   # open the clip
                while True:
                    ok, frame = cap.read()          # read next frame
                    if not ok:                      # end of clip
                        break
                    cv2.imshow(window, frame)       # show it
                    if cv2.waitKey(delay) & 0xFF in (ord("q"), 27):   # 'q' or Esc -> quit playback
                        cap.release()
                        cv2.destroyAllWindows()
                        return
                cap.release()                       # release this playthrough before repeating
            if step.hold_ms and cv2.waitKey(step.hold_ms) & 0xFF in (ord("q"), 27):   # A2 held pause
                cv2.destroyAllWindows()
                return
        elif isinstance(step, Fingerspell):
            for _ in range(max(1, step.repeat)):    # replay support for symmetry
                for ltr, ldir in zip(step.letters, step.letter_dirs(letters_dir)):   # each letter + its image folder
                    img_path = _first_image(ldir)   # pick a representative image for this letter
                    if img_path is None:            # no image for this letter -> skip
                        _report(f"[missing letter image] {ltr} -> {ldir}")
                        continue
                    img = cv2.imread(str(img_path))
                    if img is None:                 # unreadable file -> skip
                        _report(f"[unreadable image] {img_path}")
                        continue
                    cv2.imshow(window, img)         # show the letter
                    if cv2.waitKey(letter_hold_ms + step.hold_ms) & 0xFF in (ord("q"), 27):   # hold, allow quit
                        cv2.destroyAllWindows()
                        return
    cv2.destroyAllWindows()                         # clean up the window when the plan finishes
