"""GROOPY Synthesis track — Text/Speech -> Sign.

Turns text (typed, or transcribed from speech by asr.py) into a sequence of sign-video
clips, fingerspelling any word not in the curated vocabulary. Consumes the same shared
contract/vocabulary the Recognition track emits, so the two directions stay in lockstep.
"""
