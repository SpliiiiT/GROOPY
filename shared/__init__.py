"""GROOPY shared layer — the single source of truth across both tracks.

Recognition (Sign -> Text/Speech) and Synthesis (Text/Speech -> Sign) both pivot on the
same Token contract and the same curated vocabulary defined here, so the two directions
stay in lockstep. Nothing in here imports tensorflow/mediapipe — it's pure, importable
from any component (Python tracks, tests, tooling).
"""
