"""Headless smoke tests for the bidirectional pipeline (no camera/GUI/heavy training).

Covers the pure-logic layers of both tracks + the shared contract, so a regression in
gloss rules, the sign-plan fallback, sentiment, or the v1/v2 contract is caught fast.
Runnable two ways:

    python tests/test_smoke.py          # standalone (prints PASS/FAIL, exit code)
    pytest tests/test_smoke.py          # if pytest is installed

Does NOT require clips, trained models, mediapipe, or PyQt — those paths are exercised by
the make_stub_* generators + the desktop apps (manual).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from shared.contract import CONTRACT_VERSION, Sentiment, Token
from shared import vocabulary as vocab
from synthesis.src.gloss_to_signplan import Fingerspell, WordClip, build_sign_plan
from synthesis.src.pipeline import apply_sentiment, synthesize
from synthesis.src.text_to_gloss import normalise, text_to_gloss


def test_contract_v2_roundtrip():
    t = Token("hello", 0.941, 1730812345678, "word", Sentiment("positive", 0.879))
    d = t.to_dict()
    assert d["sentiment"] == {"label": "positive", "score": 0.879}
    assert Token.from_dict(d).sentiment.label == "positive"
    # v1 payloads (no sentiment) still parse -> backward compatible
    assert Token.from_dict({"token": "a", "confidence": 0.9, "timestamp": 1, "kind": "letter"}).sentiment is None
    assert CONTRACT_VERSION == "v2"


def test_vocabulary_single_source():
    # LSTM class indices and clip keys come from the same list.
    assert vocab.NUM_WORDS == len(vocab.GLOSSES) == len(vocab.CLIP_MAP)
    assert set(vocab.CLIP_MAP) == set(vocab.GLOSSES)
    assert vocab.has_clip("hello") and not vocab.has_clip("qwerty")
    assert vocab.INDEX_TO_GLOSS[vocab.GLOSS_TO_INDEX["hello"]] == "hello"


def test_text_to_gloss_rules():
    assert text_to_gloss("Hello, how are you?") == ["hello", "how", "you"]  # punct + stopword
    assert text_to_gloss("Hi thank you") == ["hello", "thanks", "you"]      # synonyms
    assert normalise("Ça va!") == "ca va"                                    # accents + punct


def test_sign_plan_fallback():
    plan = build_sign_plan(text_to_gloss("hello my name is Oussama"), source_text="x")
    assert isinstance(plan.steps[0], WordClip)                # 'hello' in vocab
    assert plan.fingerspelled_glosses == ["my", "oussama"]    # OOV -> fingerspell
    spell = [s for s in plan.steps if isinstance(s, Fingerspell)][-1]
    assert spell.letters == list("OUSSAMA")


def test_landmark_normalization_invariance():
    import numpy as np

    from recognition.src.holistic import normalize_sequence
    from shared.config import FRAME_FEATURES, SEQ_LEN

    seq = np.zeros((SEQ_LEN, FRAME_FEATURES), np.float32)
    seq[:, 44], seq[:, 45] = 0.4, 0.5     # left shoulder x,y
    seq[:, 48], seq[:, 49] = 0.6, 0.5     # right shoulder x,y
    seq[:, 132], seq[:, 133] = 0.55, 0.7  # a left-hand landmark
    base = normalize_sequence(seq)

    # translation invariance
    shifted = seq.copy()
    for i in (44, 48, 132):
        shifted[:, i] += 0.1
    for i in (45, 49, 133):
        shifted[:, i] += 0.1
    assert np.allclose(normalize_sequence(shifted), base, atol=1e-5)

    # scale invariance + shoulder midpoint -> origin, width -> 1
    scaled = seq.copy()
    for i in (44, 48, 132, 45, 49, 133):
        scaled[:, i] = (seq[:, i] - 0.5) * 2 + 0.5
    assert np.allclose(normalize_sequence(scaled), base, atol=1e-5)
    assert abs(base[0, 44] + 0.5) < 1e-5 and abs(base[0, 48] - 0.5) < 1e-5
    # absent landmarks stay zero
    assert base[0, 200] == 0.0


def test_sentiment_and_seam():
    from sentiment import analyze

    assert analyze("i am so happy and good").label == "positive"
    assert analyze("this is bad and sad").label == "negative"
    assert analyze("not good").label == "negative"            # negation
    assert analyze("the box is here").label == "neutral"
    r = synthesize(text="hello friend I am happy", with_sentiment=True)
    assert r.sentiment.label == "positive"
    # apply_sentiment is a no-op seam for now
    assert apply_sentiment(r.plan, r.sentiment).summary() == r.plan.summary()


def _run() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run())
