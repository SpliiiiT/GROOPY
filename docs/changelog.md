# GROOPY — Changelog & Session Notes (2026-07-20 → 2026-07-21)

*A narrative explanation of a large batch of work: the Keras 3 upgrade, a sentiment bake-off,
a live-demo bug fix, and a packaged desktop app. Complements `docs/results.md` (the polished
CRISP-DM write-up) — this is the "what happened and why" behind the commits.*

---

## 1. Local environment upgrade: Keras 2 → Keras 3

**The problem:** the fingerspelling CNN models were trained on Colab, whose runtime saves
`.keras` files in **Keras 3** format. The local `.venv` ran **TF 2.15/Keras 2.15**, whose loader
looks for the model class at the old internal path (`keras.src.engine.functional.Functional`) —
Keras 3 moved it (`keras.src.models.functional.Functional`) — so loading a Colab-trained model
locally failed with `"Could not deserialize class 'Functional'"`. Word recognition worked live
(trained locally, already Keras 2), but **fingerspelling couldn't run in the live desktop demo at
all** — only Colab screenshots could be shown.

**The fix, done carefully:**
1. Built an isolated `.venv-keras3` first (never touched the real environment blind) and
   installed `tensorflow==2.17.1` (ships Keras 3, still `numpy<2`-compatible — mediapipe's
   official wheels hard-require `numpy<2`, and jumping straight to TF's latest would have
   forced `numpy>=2`, breaking mediapipe outright).
2. Verified the Colab-trained CNN models load and predict correctly there — confirmed the fix.
3. **Then checked the currently-working word models** (`word_transformer.keras` etc., trained
   locally under Keras 2) — they **failed to load** under Keras 3 (the exact mirror-image error).
   This was the real risk: fixing fingerspelling could have silently broken the word demo.
4. Retrained the word bake-off fresh under Keras 3 (`train_word.py --model all`, same 791-sequence
   dataset, ~68–78% test accuracy — same band as before). Re-verified Grad-CAM, both desktop apps
   live, and the full test suite before promoting the verified stack into the real `.venv`.
5. Fixed the root cause so this doesn't happen again on any future *local* retrain:
   `requirements.txt`/`desktop/requirements.txt` now pin `tensorflow==2.17.1`/`keras==3.15.0`
   explicitly (previously unpinned, which is what let local drift from Colab in the first place).

**Docs reconciliation:** the word bake-off's retrain shifted the numbers slightly (this run's
scorecard narrowly favours GRU, not Transformer) — `docs/results.md`, `docs/presentation.md`, and
`README.md` were updated to the actual current numbers instead of the stale "Transformer wins 78%"
claim, with the same "scorecard winner ≠ deployment pick" framing already used for the CNN track
(Transformer still ships: same accuracy within noise, ~7× faster live).

*Commits: `b055562`, `ad689d7`*

---

## 2. CNN bake-off: real robustness scores, training-curve plots

`recognition/src/evaluate.py` had hardcoded `"robustness": 0.5` for every model, regardless of
the Grad-CAM review already written up in `docs/results.md` (efficientnetb0 0.85, cnn_scratch
0.65, resnet50/mobilenetv2 estimated 0.80/0.70). Any future re-run of the scorecard would have
silently produced numbers contradicting the report. Fixed with a `ROBUSTNESS_SCORES` lookup
(falls back to 0.5 "unreviewed" for anything not in it, rather than guessing) — verified it
reproduces the exact re-scored totals already published (0.681/0.611/0.498/0.399).

Added `recognition/src/plot_history.py` — reads the `history_*.json` files `train.py`/
`train_word.py` already write and plots validation accuracy/loss per epoch across a bake-off's
candidates. Generated `word_training_curves.png` immediately (data already existed locally); the
CNN track's curves need a Colab re-run (see §4) since those history files were never downloaded
before.

*Commit: `ad689d7`*

---

## 3. The Colab notebook saga

Three separate problems surfaced in sequence while getting `02_bakeoff.ipynb` working again —
worth understanding as a chain, not three unrelated bugs:

1. **Git divergence:** while pushing local fixes, a real CNN bake-off run had happened on Colab
   in parallel, and Colab's autosave pushed a "Created using Colab" commit that diverged on the
   same notebook file. That Colab runtime had already disconnected before its outputs were
   downloaded, so the run's real data (`history_*.json`, `bakeoff.json`, retrained models) was
   unrecoverable — only stale cached execution metadata survived. Resolved with a real merge
   commit (not an automatic JSON merge, which risks corrupting notebook structure): kept the
   correct local notebook content, added a **download-zip cell** (cell 7) so a finished Colab run
   can be saved back in one click instead of grabbing files one by one from the Files panel.
   *Commit: `5acdc2f`*

2. **The `tensorflow==2.17.1` pin, tried and reverted:** to keep Colab and local producing the
   same Keras 3.x line, the notebook was pinned to match local's TF version exactly. This forced
   `numpy<2` on Colab, which broke ABI compatibility with Colab's own preinstalled numpy-2.x-
   compiled packages — first `jax` (`import tensorflow`'s lazy `from jax import xla_computation`
   raised an uncaught `ValueError`, not the `ImportError` the code expected), then `pandas`
   ("numpy.dtype size changed, may indicate binary incompatibility") after the jax fix.
   **Root-cause realisation:** the pin's actual job — avoiding a Keras-2-vs-3 load failure — was
   only relevant while local was still on Keras 2. Local is on Keras 3 now, so that risk is gone;
   what's left is only a same-major-version Keras-3.x-to-3.x question, and Keras 3's save format
   is designed to be compatible within the 3.x line. **Dropped the pin entirely** rather than keep
   patching Colab's environment package-by-package — Colab now just uses its own default TF.
   *Commits: `8be08d5` (the jax patch, superseded), `efaffe1` (the actual fix)*

3. **Stale browser tab / stale VM:** even after the fixes were pushed, Colab kept showing the old
   behaviour twice — once because "Restart session" only restarts the Python process, not the
   underlying VM's installed packages (a genuinely fresh VM needs **Runtime → Disconnect and
   delete runtime**, not just restart), and once because the browser tab had the notebook's
   *content* cached from before the latest push (Colab doesn't re-fetch from GitHub on
   reconnect — the page itself needs a hard refresh). Both are worth remembering for next time.

**Current state:** Colab now reports its own native `TF 2.20.0` with the GPU correctly detected,
training at the expected pace. The bake-off is running as of this writing; once it finishes and
the download-zip cell is run, the CNN training-curve plot (mirroring `word_training_curves.png`)
can be built the same way.

---

## 4. Fingerspelling hand-presence gate

`desktop/app.py`'s `Recognizer.predict()` used `crop_and_preprocess()`, which silently **fell
back to classifying the full frame** when MediaPipe found no hand — so with no hand in view, the
CNN still confidently argmaxed some letter out of background noise. This is the exact bug class
`word_stream.py` had already fixed for whole-word signs (the LSTM has no "idle" class either).
Fixed by bypassing that fallback and gating on `crop_hand()` directly: no hand → a distinct
`NO_HAND` sentinel, never emitted as a token (0.0 confidence never clears `CONFIDENCE_GATE`), and
the UI shows "no hand detected — show a letter" instead of a bogus guess. Verified live.

*Commit: `004209a`*

---

## 5. Sentiment: a real bake-off, and a benchmark that picked the wrong model

Built a 3-way bake-off for the sentiment model, mirroring the CNN/word tracks:

| Candidate | What it is |
|---|---|
| `scratch` | TF-IDF + Logistic Regression, trained from scratch on IMDB — trains in ~9s on CPU, no Colab needed |
| `distilbert` | Pretrained, general-purpose, binary (SST-2 movie reviews) |
| `twitter_roberta` | Pretrained, natively 3-class, tuned on short informal text |

One real correction along the way: HuggingFace `transformers` dropped TensorFlow model support
in recent versions — the pretrained candidates needed **PyTorch (CPU-only)**, a new dependency
isolated to `sentiment/requirements.txt`, not the TF-only path originally assumed.

**Bake-off #1** (IMDB accuracy/latency/size scorecard) picked `distilbert` (0.744 vs. scratch's
0.662 vs. twitter_roberta's 0.0 — it was worst on all three axes simultaneously). **This pick was
wrong for actual deployment.** IMDB has zero neutral reviews, so the metric can't reward or
punish neutral-detection quality, and `distilbert` — trained only on movie-review polarity — has
no real concept of "neutral" at all. A second check on 20 hand-labeled, app-realistic sentences
(greetings, everyday statements, clear emotion, negation) caught it:

| Model | Overall | Positive | Negative | Neutral |
|---|---|---|---|---|
| **twitter_roberta** | **100%** | 6/6 | 8/8 | **6/6** |
| scratch | 90% | 5/6 | 7/8 | 6/6 |
| distilbert | 70% | 6/6 | 8/8 | **0/6** |

`distilbert` gets **every neutral sentence wrong** ("hello my name is Oussama" → positive, 100%
confidence) — it was never trained to hedge. `twitter_roberta` is perfect. The 500MB/80ms cost
that looked bad in bake-off #1 turns out not to matter: sentiment runs once per typed/spoken
sentence (a button click), not per-frame like the vision models, so even 80ms is imperceptible.

**Recommendation shipped:** `twitter_roberta`, not the raw scorecard winner — opt in via
`sentiment.src.models.load_recommended_backend()`. The dependency-free `LexiconBackend` stays the
default; the ML backends are optional. Full numbers in `docs/sentiment_options.md`.

*Commit: `4754229`*

---

## 6. Sentiment Decision A2: emphasis instead of just a label

`docs/sentiment_options.md` had split sentiment into two questions: **B** (how good is the
model — done, above) and **A** (what should sentiment *behaviorally drive*). A was deliberately
left for a discussion with the sentiment module's owner (Oussama's partner) — a displayed label
(A1, already done) vs. signing emphasis/speed (A2) vs. avatar facial expression (A3, needs a 3D
avatar the project doesn't have — out of scope). Oussama chose to build A2 himself rather than
wait.

`apply_sentiment(plan, sentiment)` in `synthesis/src/pipeline.py` was a pass-through no-op; now,
confident non-neutral sentiment (`label != "neutral"` and `score >= 0.75`) sets `hold_ms=400`/
`repeat=2` on the `SignPlan`'s `WordClip` steps — known-vocabulary signs replay with an extra
held pause, like tone of voice. Fingerspelled/out-of-vocabulary words (mostly names, not
emotional content) are untouched. Verified live: "hello friend I am happy" (score 1.0) visibly
replays and holds each clip; a neutral sentence plays exactly as before.

*Commit: `bc56fd0`*

---

## 7. Packaged desktop app (in progress)

The natural "make this a real app, not a script" step, scoped to a **packaged desktop
executable** (not mobile/web — those would need a different framework/export path entirely).

- **Fixed a real fragility found along the way:** `Fingerspell` letter images came from
  `shared.config.LETTERS_DIR`, which pointed at the **full 1.3GB, 87k-image ASL Alphabet training
  set** just to show one representative letter image. Curated a small `synthesis/assets/letters/`
  (26 images, ~344KB, committed to git) and repointed `LETTERS_DIR` there — fixes an existing
  dependency on the full dataset even outside packaging, and makes bundling practical.
- **One app, not two scripts:** added `desktop/launcher.py` — a small window with "Sign → Text"
  and "Text → Sign" buttons that open the existing `MainWindow`/`SynthWindow` unmodified. Safer
  than merging both stateful windows (camera+timer vs. video playback) into one, since it reuses
  working code untouched. Verified both buttons live.
- **Path resolution fix for a frozen build:** `recognition/src/config.py`, `sentiment/src/
  config.py`, and `shared/config.py` all computed their "repo root" as `Path(__file__).resolve().
  parents[N]` — correct from source, wrong once PyInstaller bundles the code. Added
  `shared/paths.py`'s `app_root()` (checks `sys.frozen`/`sys._MEIPASS`) and pointed all three at
  it, so the same code works from source and from a packaged build.
- **PyInstaller spec** (`packaging/groopy.spec`): bundles `cnn_scratch.keras` +
  `word_transformer.keras` (not the other 3 CNN bake-off models, or sentiment's optional
  transformers/torch backends — `LexiconBackend` ships, the ML sentiment models stay a dev
  opt-in), the 20 sign clips, and the new small letters folder. `--onedir` (more reliable than
  `--onefile` for TensorFlow/mediapipe's dynamic loading).
- **Status as of this doc:** build in progress (TensorFlow's dependency analysis is slow, several
  minutes). Next: verify the built `.exe` actually launches and runs outside the dev venv.

---

## Commit reference

| Commit | Summary |
|---|---|
| `b055562` | Upgrade to Keras 3, reconcile docs with retrained word models |
| `1613caa` | (Colab autosave — superseded by the merge below) |
| `ad689d7` | Real Grad-CAM robustness scores in evaluate.py; add training-curve plots |
| `5acdc2f` | Resolve notebook merge conflict, keep local content, add download-zip cell |
| `004209a` | Gate fingerspelling on hand presence |
| `8be08d5` | Remove jax/jaxlib on Colab (superseded by `efaffe1`) |
| `4754229` | Sentiment scratch-vs-pretrained bake-off, with a course-correction |
| `efaffe1` | Drop the Colab tensorflow pin entirely |
| `bc56fd0` | Implement sentiment Decision A2 (emphasis/replay) |
| *(pending)* | Packaged desktop app (launcher, frozen-path fix, PyInstaller spec) |

---

## What's still open

- **Kaggle token rotation** — a live token has been pasted in chat 3× across sessions; still not
  rotated. Pure housekeeping.
- **CNN training-curve plot** — waiting on the Colab re-run (§3) to finish and its download-zip
  cell to be run.
- **Packaged app** — build + launch verification in progress (§7).
- Nothing else sentiment-related is open — both Decision A and B are done.
