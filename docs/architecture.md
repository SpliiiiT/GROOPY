# GROOPY architecture — bidirectional Sign ↔ Text/Speech

GROOPY is a two-way communication PoC between a Deaf/HoH signer and a hearing non-signer.
Two tracks meet at one shared contract + vocabulary.

```
                         ┌───────────────────────────────┐
   Deaf person  ─────▶   │  TRACK A · Recognition        │  ─────▶  text on screen
   (camera)              │  Sign → Text/Speech           │          / spoken (TTS)
                         │                               │
                         │  fingerspell → static CNN     │
                         │  whole words → Holistic+LSTM  │
                         └──────────────┬────────────────┘
                                        │  Token {token, confidence, ts, kind, sentiment?}
                                        ▼
                         ┌──────────────────────────────┐
                         │  shared/  (single source)     │
                         │  contract.py   vocabulary.py  │
                         └──────────────┬────────────────┘
                                        │
                         ┌──────────────▼────────────────┐
   Hearing person ────▶  │  TRACK B · Synthesis          │  ─────▶  sign-video clips
   (type / mic)          │  Text/Speech → Sign           │          (fingerspell fallback)
                         │                               │
                         │  ASR → text_to_gloss →        │
                         │  gloss_to_signplan → player   │
                         └───────────────────────────────┘

   sentiment/  analyzes the text on either side → Sentiment(label, score)
               carried on the Token; behavioural effect = OPEN SEAM (apply_sentiment).
```

## The pivot: shared vocabulary + contract

- **`shared/contract.py`** — the `Token` (and optional `Sentiment`) that flows between
  tracks. One definition, mirrored in Dart (`app/lib/contract/token.dart`). Version **v2**.
- **`shared/vocabulary.py`** — the curated word list. It is *simultaneously* the LSTM's
  output classes (`GLOSS_TO_INDEX`) and the Synthesis clip keys (`CLIP_MAP`). Change it once,
  both directions update. Any word not in it is **fingerspelled** on both sides, so nothing
  dead-ends.
- **`shared/config.py`** — shared paths (clips, letters, WLASL, landmarks) and the LSTM
  input geometry (`SEQ_LEN`, `FRAME_FEATURES`).

## Track A · Recognition (Sign → Text/Speech)

| Path | Module | Model |
|------|--------|-------|
| Fingerspelling (letters) | `preprocess.py` → CNN | static-image bake-off winner |
| Whole-word signs | `holistic.py` → `word_stream.py` | `models/lstm_word.py` (landmark sequence LSTM) |

Both emit onto the same `TokenStream` (`letter` vs `word` kind). The desktop app runs
fingerspelling always, and words too when `--word-model` is supplied.

## Track B · Synthesis (Text/Speech → Sign)

`asr.py` (optional speech→text) → `text_to_gloss.py` (rule-based PoC gloss) →
`gloss_to_signplan.py` (in-vocab → `WordClip`; else `Fingerspell` using the ASL-alphabet
letter images) → `player.py` (OpenCV clip playback). Orchestrated by `pipeline.synthesize`.

## Sentiment (shared, partner-owned)

`sentiment/analyze.py` returns `Sentiment(label, score)` (dependency-free lexicon stub,
swappable backend). It is attached to the contract today; **what it drives is not yet
decided** — the single seam is `synthesis/src/pipeline.apply_sentiment`, currently a no-op.

## Data (one download, two uses)

`data/download_sign_clips.py` turns pre-downloaded WLASL videos into **both** synthesis
playback clips (`synthesis/clips/`) **and** LSTM training sequences
(`data/wlasl_landmarks/`). `data/download_asl_alphabet.py` supplies the CNN training data +
the fingerspelling fallback images.

## Verification without real data

Every logic layer is headless-testable on synthetic stubs:
`data/make_stub_data.py` (CNN), `synthesis/make_stub_clips.py` (clips),
`data/make_stub_sequences.py` (LSTM), and `tests/test_smoke.py` (contract, gloss, plan,
sentiment). Camera/GUI paths are verified manually via the two desktop apps.
