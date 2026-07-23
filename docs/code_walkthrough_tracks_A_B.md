# GROOPY — Line-by-line Code Walkthrough (Track A + Track B)

> **What this is:** a defense-ready reading of *your* code — Track A (Recognition, `recognition/`)
> and Track B (Synthesis, `synthesis/`), plus the `shared/` contract that ties them together.
> Every file is walked block by block: what each line does and *why it's there*.
>
> **Reading order for the oral:** Part 1 (the 4 cross-cutting ideas) → Part 2 (Track A files) →
> Part 3 (Track B files) → Part 4–6 (architectures / metrics / XAI, the professor's other axes).

---

# PART 1 — How the code works (the 4 cross-cutting ideas)

## 1.1 Architecture at a glance

```
                    ┌─────────────────────────────────────────────┐
                    │  shared/  (single source of truth)          │
                    │  contract.py · vocabulary.py · config.py    │
                    └───────────────┬──────────────┬──────────────┘
        Sign → Text                 │              │              Text → Sign
   ┌────────────────────────────────┘              └────────────────────────────────┐
   │  TRACK A — recognition/                          TRACK B — synthesis/           │
   │  camera → preprocess/holistic → CNN/seq model    text/speech → gloss → SignPlan │
   │  → TokenStream (gate/debounce) → Token           → player (clips / letters)     │
   └──────────────────────────────────────────────────────────────────────────────┘
```

**The `Token`** (`shared/contract.py`) is the one object that flows between the two tracks.
Recognition **emits** Tokens; Synthesis **consumes** the same shape. Define it once → both
directions agree by construction.

## 1.2 The live recognition pipeline — gates, debounce, capture-to-commit

The raw model is never trusted directly. Four guards sit between a prediction and a committed
character. Here is where each one lives:

| Guard | Purpose | Where in code |
|-------|---------|---------------|
| **Hand-presence gate** | don't predict when no hand is on screen | `preprocess.crop_hand` returns `None`; `word_stream` checks `hand_activity < 0.3` |
| **Confidence gate** | emit only if `p ≥ 0.80` | `token_stream.TokenStream.update` (`confidence < self.gate`) |
| **Debounce** | at most one emit per 500 ms window | `token_stream.update` (`now_ms - self._last_emit_ms < debounce_ms`) |
| **No-repeat window** | same label can't re-fire for 2× the window | `token_stream.update` (the `label == self._last_token` check) |
| **Capture-to-commit** | letter is added only on a deliberate keypress | the desktop app calls `.update()` on demand, not every frame |

The **key architectural decision**: the CNN/sequence model produces a *guess every frame*, but
`TokenStream` is a **stateful filter** that converts a noisy stream of guesses into a small number
of clean, committed `Token`s. This is what stops the demo spewing 30 letters per second.

## 1.3 The shared "contract"

Two files make the two tracks agree:

- **`shared/contract.py`** — the `Token` and `Sentiment` dataclasses + the control-token set. Both
  tracks import it, so the data shape can never drift between them (there's even a Dart mirror for
  a future mobile app).
- **`shared/vocabulary.py`** — the 20 curated words. This *same list* is simultaneously the word
  model's output classes **and** the synthesis clip keys. Change it once, both directions update.
  Anything not in the list is fingerspelled, so the system never dead-ends.

## 1.4 Reproducibility

`SEED = 42` appears in `shared/config.py` and `recognition/src/config.py`. Every dataset split,
every shuffle, and every training run seeds from it, so the bake-off is **fair**: no candidate
gets a luckier split or a different augmentation. `paths.app_root()` makes every path repo-relative
so the exact same code runs locally, on Colab, and inside the packaged `.exe`.

---

# PART 2 — TRACK A: Recognition, file by file

## 2.1 `shared/contract.py` — the data contract

```python
CONTRACT_VERSION = "v2"
CONTROL_TOKENS = {"del", "nothing", "space"}
KIND_LETTER = "letter"; KIND_WORD = "word"; KIND_CONTROL = "control"
```
- **`CONTRACT_VERSION`** — a version string so a consumer can tell which shape it's getting. v2
  added the optional `sentiment` field.
- **`CONTROL_TOKENS`** — the three non-letter classes of the ASL Alphabet. Note it's `"del"` (not
  `"delete"`) to match the dataset's folder names exactly.
- **`KIND_*`** — the three token kinds. A `Token` carries one of these so Track B knows whether to
  play a word clip, a letter, or act on a control.

```python
@dataclass
class Sentiment:
    label: str
    score: float
    def to_dict(self): return {"label": self.label, "score": round(float(self.score), 3)}
    @staticmethod
    def from_dict(d): ...  # None-safe: returns None if d is falsy
```
- A tiny value object: a `label` ("positive"/"neutral"/"negative") and a `[0,1]` `score`.
- `to_dict` / `from_dict` are the **JSON serialization seam** — how the token crosses a process or
  language boundary. `from_dict(None)` returns `None`, so a token without sentiment round-trips
  cleanly.

```python
@dataclass
class Token:
    token: str
    confidence: float
    timestamp: int
    kind: str
    sentiment: Optional[Sentiment] = field(default=None)
```
- **`token`** — the normalized payload: an ASL letter (`"a"`), a word gloss (`"hello"`), or a
  control (`"space"`).
- **`confidence`** — 0–1; only emitted at/above the gate (so a stored token is always "confident").
- **`timestamp`** — epoch ms, stamped at prediction time (used by the debounce logic downstream).
- **`sentiment`** — optional; `default=None` makes v2 **backward compatible** with v1 consumers.
- `to_dict` overrides `asdict`'s recursion so the nested `sentiment` is normalized via *its own*
  `to_dict`, and rounds confidence to 3 places.

## 2.2 `recognition/src/config.py` — the fixed protocol

This file is the **fairness guarantee**: every candidate imports the same constants, so no model
gets a different image size, split, or learning rate.

```python
CLASS_NAMES = [chr(c) for c in range(ord("A"), ord("Z")+1)] + ["del", "nothing", "space"]  # 29
IMG_SIZE = 224; INPUT_SHAPE = (224, 224, 3)
BATCH_SIZE = 32; EPOCHS = 20; LEARNING_RATE = 1e-3
VAL_SPLIT = 0.15; SEED = 42
EARLY_STOPPING_PATIENCE = 4; REDUCE_LR_PATIENCE = 2
CONFIDENCE_GATE = 0.80; DEBOUNCE_MS = 500
```
- **`CLASS_NAMES`** — A–Z built with `chr()` + the 3 controls = 29 classes.
- **`IMG_SIZE = 224`** — the size the pretrained backbones expect.
- **`LEARNING_RATE = 1e-3`** — the head LR; backbones fine-tune at LR/10 (see `train.py`).
- **`CONFIDENCE_GATE` / `DEBOUNCE_MS`** — the live-inference guards, defined here so both the CNN
  and word paths share them.

```python
SCORECARD_WEIGHTS = {"accuracy":0.40,"latency":0.20,"size":0.15,"robustness":0.15,"stability":0.10}
WORD_SCORECARD_WEIGHTS = {"accuracy":0.60,"latency":0.20,"size":0.20}
```
- The two weight vectors that turn raw metrics into a single winner (see Part 5). They **must sum
  to 1.0**. The word set is 100% automatic (no manual robustness/stability terms).

## 2.3 `recognition/src/preprocess.py` — static hand cropping (fingerspelling input)

> The single most important robustness trick: **crop to the hand before classifying**, so the CNN
> learns the *hand shape*, not the background. The same function is used offline and live → no
> train/serve skew.

```python
try:
    import mediapipe as mp
    _mp_hands = mp.solutions.hands
except Exception:
    _mp_hands = None
```
- MediaPipe is imported **lazily/defensively**: if it isn't installed, the module still imports
  (so tests and the headless pipeline run), and only *using* the cropper raises.

```python
def _get_hands(static=True):
    global _HANDS
    if _HANDS is None:
        _HANDS = _mp_hands.Hands(static_image_mode=static, max_num_hands=1, min_detection_confidence=0.5)
    return _HANDS
```
- A **cached singleton** detector — building a MediaPipe graph is expensive, so we build it once.
- `static_image_mode=True` for offline dataset cropping; the live app passes `static=False` (video
  mode tracks between frames, faster).
- `max_num_hands=1` — fingerspelling is one hand.

```python
def crop_hand(bgr_image, margin=0.25, static=True):
    results = _get_hands(static).process(rgb)
    if not results.multi_hand_landmarks:
        return None                     # ← the hand-presence gate
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    x_min -= bw*margin; ...             # expand the box by 25% each side
    x1 = max(0, int(x_min*w)); ...      # normalized coords → pixels, clamped
    if x2 <= x1 or y2 <= y1: return None
    return bgr_image[y1:y2, x1:x2]
```
- Runs hand detection; **returns `None` when no hand** — this is the hand-presence gate for the
  static path.
- Landmarks are normalized (0–1); we take the bounding box, **expand it 25%** so fingertips aren't
  clipped, convert to pixels, and clamp to the frame.
- Returns the cropped BGR sub-image (or `None`).

```python
def preprocess_for_model(bgr_image):
    rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0
```
- BGR→RGB (OpenCV loads BGR; Keras expects RGB), resize to 224², normalize to **`[0,1]`**.
- Returns `[0,1]` floats — every backbone then applies *its own* rescaling internally, so all
  candidates receive identical input (this is the contract between `preprocess` and `models/`).

```python
def crop_and_preprocess(bgr_image, static=True):
    crop = crop_hand(bgr_image, static=static)
    if crop is None: crop = bgr_image   # fall back to full frame so the demo never goes blank
    return preprocess_for_model(crop)
```
- Convenience combiner. If no hand, it falls back to the **full frame** so live playback never
  freezes on a blank — the confidence gate downstream will suppress a low-confidence guess anyway.

## 2.4 `recognition/src/data.py` — the CNN dataset (tf.data)

> Same split, batch size, augmentation for every candidate (imported from `config`).

```python
def _augment_layer():
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomTranslation(0.08, 0.08),
        tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
        tf.keras.layers.RandomContrast(0.15),
    ], name="augment")
```
- Augmentation applied **only at train time**. Deliberately **no horizontal flip** — flipping can
  turn a valid ASL handshape into a different/invalid one.
- `RandomBrightness`/`RandomContrast` help the model generalize **across skin tones** (a fairness
  consideration, not just accuracy).

```python
def make_datasets(..., test_split=0.10, cache=False):
    train_ds = image_dataset_from_directory(data_dir, validation_split=val_split+test_split,
                                             subset="training", seed=SEED, ...)
    holdout_ds = image_dataset_from_directory(..., subset="validation", seed=SEED, ...)
    val_batches = int(holdout_batches * (val_split/(val_split+test_split)))
    val_ds = holdout_ds.take(val_batches); test_ds = holdout_ds.skip(val_batches)
```
- Keras gives us `training` / `validation` by fraction with a **fixed seed**. We then **peel a test
  set off the validation stream** (`take`/`skip`) so test images are never seen in training. All
  three splits are deterministic given `SEED`.

```python
    train_ds = train_ds.map(_normalise, ...)
    if cache: train_ds = train_ds.cache()
    train_ds = (train_ds.shuffle(32, seed=SEED, reshuffle_each_iteration=True)
                        .map(lambda x,y: (augment(x, training=True), y), ...)
                        .prefetch(AUTOTUNE))
```
- **Two OOM lessons are baked into these lines** (found running on Colab's 12.7 GB):
  1. `cache=False` by default — caching all 87k images at 224² would need ~40 GB.
  2. `shuffle(32)` not `shuffle(1000)` — the dataset is already **batched**, so the buffer counts
     *batches*; 1000 batches × 32 ≈ 19 GB → crash. 32 is plenty since files are already shuffled.
- Augmentation runs **after** normalize/cache so randomness is resampled every epoch.

## 2.5 `recognition/src/models/` — the CNN candidates

All four expose `build(num_classes, input_shape)` and share the head
`GlobalAveragePooling2D → Dropout → Dense(softmax)`. Layer-by-layer tables are in **Part 4**. The
code mechanics worth knowing:

- **`cnn_scratch.py`** — a `_conv_block` helper = `[Conv2D(3×3, use_bias=False) → BN → ReLU]×2 →
  MaxPool`, stacked at 32/64/128/256 filters. `use_bias=False` because the following BatchNorm
  cancels any bias. The 256-filter block is the **Grad-CAM target**.
- **`efficientnet.py` / `mobilenetv2.py` / `resnet50.py`** — each wraps a Keras backbone with
  `input_tensor=x` (so the backbone's layers are **inlined** into the outer graph — that's what
  makes Grad-CAM reach them), `base.trainable = False` initially, and stores
  `model.base_layer_names = [l.name for l in base.layers]`. That last line is subtle: storing the
  `base` *object* would double-track its inlined layers and **corrupt `.keras` save/load**; storing
  the *names* lets `train.py` re-resolve the layers for fine-tuning without that bug.
- Each backbone bakes its **own preprocessing** in (`Rescaling`, `preprocess_input`) so callers
  always pass `[0,1]`.

## 2.6 `recognition/src/train.py` — the unified CNN trainer

> Same protocol for every candidate. Pretrained models get a two-phase transfer-learning schedule.

```python
def _callbacks():
    return [EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6)]
```
- **EarlyStopping** with `restore_best_weights` — stop when val accuracy plateaus and **roll back to
  the best epoch** (a core overfitting guard).
- **ReduceLROnPlateau** — when val loss stalls, cut the LR ×0.3 so the model can settle into a
  finer minimum.

```python
def train_one(name, epochs, train_ds, val_ds):
    _set_seeds()                                  # reproducibility
    model = model_zoo.build(name, NUM_CLASSES, INPUT_SHAPE)
    _compile(model, LEARNING_RATE)                # Phase 1: LR 1e-3
    hist1 = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=_callbacks())
```
- **Phase 1** — train the new head (for pretrained, base is frozen; for scratch, the whole net).

```python
    if is_pretrained:
        for layer in backbone: layer.trainable = True
        for layer in backbone[:-30]: layer.trainable = False     # keep only last ~30 trainable
        for layer in backbone:
            if isinstance(layer, BatchNormalization): layer.trainable = False
        _compile(model, LEARNING_RATE/10.0)       # Phase 2: LR 1e-4
        hist2 = model.fit(..., epochs=max(5, epochs//2), ...)
```
- **Phase 2 (pretrained only)** — the textbook fine-tune: unfreeze **only the last ~30 layers**
  (stability on a small GPU), keep **BatchNorm frozen** (moving stats shouldn't shift on a small
  dataset), recompile at **LR/10**. `cnn_scratch` skips this entirely — nothing to unfreeze.
- Every run saves `{name}.keras` + a `history_{name}.json` so `evaluate.py` can build the bake-off.

## 2.7 `recognition/src/holistic.py` — landmark extraction (word input)

> Turns one BGR frame into a **258-feature** vector (pose + both hands). Same function offline and
> live → no skew. This is the most important file in the word track.

```python
def _pose_vec(landmarks):
    if landmarks is None: return np.zeros(POSE_FEATURES)   # 33*4
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks.landmark]).flatten()
def _hand_vec(landmarks):
    if landmarks is None: return np.zeros(HAND_FEATURES)   # 21*3
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()
```
- Pose keeps **4 values/point** (x,y,z + visibility); hands keep **3** (x,y,z).
- **Missing parts are zero-filled** — so the output length is *always* 258, whether or not a hand
  is visible (critical: the model's input shape must be fixed).

```python
def landmarks_from_results(results):
    return np.concatenate([_pose_vec(results.pose_landmarks),
                           _hand_vec(results.left_hand_landmarks),
                           _hand_vec(results.right_hand_landmarks)]).astype(np.float32)
```
- The **feature layout**: `[pose 132 | left hand 63 | right hand 63] = 258`. This ordering is fixed
  and shared with `shared/config.FRAME_FEATURES`.

```python
def normalize_sequence(seq):                       # ← THE decisive fix (5% → 70%)
    Lx,Ly = seq[:,_SHOULDER_L], seq[:,_SHOULDER_L+1]   # pose landmark 11
    Rx,Ry = seq[:,_SHOULDER_R], seq[:,_SHOULDER_R+1]   # pose landmark 12
    valid = ~((Lx==0)&(Ly==0)) & ~((Rx==0)&(Ry==0))    # both shoulders detected?
    cx = (Lx+Rx)/2; cy = (Ly+Ry)/2                     # shoulder midpoint (translation ref)
    scale = np.hypot(Lx-Rx, Ly-Ry)                     # shoulder width (scale ref)
    def _apply_xy(block):
        present = ~((block[...,0]==0)&(block[...,1]==0))
        nx = (block[...,0]-cx[:,None])/scale[:,None]   # recenter + rescale x
        ny = (block[...,1]-cy[:,None])/scale[:,None]   # recenter + rescale y
        block[...,0] = np.where(present, nx, block[...,0])
        block[...,1] = np.where(present, ny, block[...,1])
```
- **Why this matters:** without it the model sees *absolute* image positions and confuses "where
  the signer stands" with "what they sign". Recentering on the shoulder midpoint and dividing by
  shoulder width makes the features **translation- and scale-invariant** — hand motion *relative to
  the body*. This single change moved test accuracy from ~5% (chance) → 37% → 70%.
- Only **x,y** are normalized (z and visibility are noisy → left as-is). Zero/absent landmarks and
  frames with no detected pose are **left untouched** (`np.where(present, ...)` / the `valid` mask),
  so padding stays padding.

```python
def video_to_sequence(path, seq_len, ..., frame_start=1, frame_end=-1):
    frames = [read all frames]
    frames = frames[lo:hi] or frames                   # apply WLASL sub-range
    idxs = np.linspace(0, len(frames)-1, num=seq_len).astype(int)   # uniform sampling
    return np.stack([landmarks(frames[i], static=static) for i in idxs])
```
- Turns a **training video** into a `(seq_len, 258)` array by **uniformly sampling 30 frames**
  across the clip. `frame_start/end` honor the WLASL annotations (a sign may occupy only part of a
  clip).

## 2.8 `recognition/src/sequence_data.py` — the word dataset

```python
def _pad_or_truncate(seq, seq_len):
    if t > seq_len: return seq[:seq_len]
    pad = np.zeros((seq_len - t, seq.shape[1])); return np.concatenate([seq, pad])
```
- Forces every sample to exactly `(30, 258)` — **truncate if longer, zero-pad if shorter**. The
  model's `Masking` layer later ignores those zero-pad rows.

```python
def augment_sequence(seq, rng):
    present = seq != 0.0
    out = np.where(present, out*scale + shift, 0.0)              # signer distance/position
    out = np.where(present, out + rng.normal(0, 0.012, ...), 0.0) # light landmark jitter
    src = np.clip(np.linspace(0, (seq_len-1)*speed, seq_len), ...).astype(int)
    out = out[src]                                               # temporal speed-warp
```
- Three physically-meaningful augmentations: **global scale+shift** (how far/where the signer is),
  **small noise** (landmark jitter), **temporal resample** (signing speed 0.8–1.2×).
- Every op is masked with `present` so **zero-padding never gets perturbed** into fake data.
- **No handedness mirroring** — flipping would change some signs.

```python
def load_dataset(..., augment_factor=0):
    for gloss in GLOSSES:
        for npy in sorted(cls_dir.glob("*.npy")):
            seq = normalize_sequence(np.load(npy))     # SAME normalization as live
            X.append(_pad_or_truncate(seq, seq_len)); y.append(GLOSS_TO_INDEX[gloss])
    perm = rng.permutation(len(X_arr))                 # deterministic shuffle (SEED)
    X_test = ...[:n_test]; X_val = ...; X_train = ...  # test/val peeled off first
    if augment_factor > 0:
        for _ in range(augment_factor):
            aug_X.append(np.stack([augment_sequence(s, aug_rng) for s in X_train]))
```
- Loads the `.npy` cache class-by-class (classes come from the shared vocabulary → labels line up
  with the clip dictionary).
- Applies the **same `normalize_sequence`** used live — this is the anti-skew guarantee.
- Splits deterministically, then **augments the training set only** (`×augment_factor`, default 8
  from `train_word.py`), leaving val/test clean. Essential given ~40 samples/class.

## 2.9 `recognition/src/word_models.py` — the 4 sequence candidates

`build(name, num_classes, timesteps, features)` dispatches to `lstm/gru/bilstm/transformer`. All
start with `Input(30,258) → Masking(0.0)`. Layer tables + the pre-norm Transformer block are in
**Part 4**. Mechanics worth stating aloud:
- **`Masking(mask_value=0.0)`** — tells the RNN/attention to **skip zero-padded frames**, so a
  short sign isn't diluted by empty timesteps.
- The **transformer** adds a **learned positional embedding** (`Embedding(30, d_model)`) because
  self-attention is order-invariant on its own — it needs to be told *which frame* is which.

## 2.10 `recognition/src/train_word.py` — the unified word trainer

Mirror of `train.py` but single-phase (these models train from scratch). Same callbacks
(EarlyStopping + ReduceLROnPlateau), same Adam/1e-3. Key call:
```python
(X_tr,y_tr),(X_val,y_val),(X_te,y_te),class_names = load_dataset(augment_factor=args.augment)  # ×8
model = word_models.build(name, NUM_WORDS, SEQ_LEN, FRAME_FEATURES)
test_acc = model.evaluate(X_te, y_te)[1]
model.save(MODELS_DIR / f"word_{name}.keras")
```
- Trains each candidate under one protocol, records `test_accuracy` + `params` + `latency` seeds
  into `word_train_summary.json`, saves `word_{name}.keras` for `evaluate_word.py` to rank.

## 2.11 `recognition/src/token_stream.py` — the gate/debounce (the live "brain")

> Wraps raw predictions with the contract rules. **Feed it `(label, confidence)`; it returns a
> `Token` or `None`.** This is where the four live guards actually execute.

```python
class TokenStream:
    def __init__(self, gate=CONFIDENCE_GATE, debounce_ms=DEBOUNCE_MS, kind=KIND_LETTER):
        self._last_token = None; self._last_emit_ms = 0.0
```
- Stateful: it remembers the **last emitted label** and **when** it fired. Both the CNN path and
  `WordStream` create one (with `kind="letter"` / `kind="word"`).

```python
    def update(self, label, confidence, sentiment=None):
        now_ms = time.time() * 1000.0
        if confidence < self.gate: return None                          # ① confidence gate
        if now_ms - self._last_emit_ms < self.debounce_ms: return None  # ② debounce
        if label == self._last_token and (now_ms-self._last_emit_ms) < self.debounce_ms*2:
            return None                                                 # ③ no-repeat window
        self._last_token = label; self._last_emit_ms = now_ms
        kind = KIND_CONTROL if label in CONTROL_TOKENS else self.kind
        return Token(token=label if label in CONTROL_TOKENS else label.lower(),
                     confidence=round(float(confidence),3), timestamp=int(now_ms),
                     kind=kind, sentiment=sentiment)
```
- **① Confidence gate** — below 0.80 → nothing.
- **② Debounce** — within 500 ms of the last emit → nothing (caps the emit rate).
- **③ No-repeat window** — the *same* label can't re-fire within 2× the window, so holding a letter
  steady produces **one** token, not a stream.
- On success it updates state and returns a fully-formed `Token`, promoting control labels to
  `kind="control"` and lowercasing letter/word payloads.

## 2.12 `recognition/src/word_stream.py` — live word recognition

> Rolling-window buffer → sequence model → word `Token`, through the same `TokenStream`.

```python
class WordStream:
    def __init__(self, model, seq_len=SEQ_LEN, gate=CONFIDENCE_GATE):
        self._buf = deque(maxlen=seq_len)          # keeps only the last 30 frames
        self._stream = TokenStream(gate=gate, kind=KIND_WORD)
```
- A `deque(maxlen=30)` is the **rolling window**: push a frame, the oldest drops off automatically.

```python
    def push(self, landmark_vec):
        self._buf.append(landmark_vec.astype(np.float32))
        if not self.ready: return None             # wait until 30 frames buffered
        stacked = np.stack(self._buf)
        hand_activity = float(np.mean(np.any(stacked[:, FRAME_FEATURES-126:] != 0.0, axis=1)))
        if hand_activity < 0.3:                     # ← hand-presence gate for words
            self.last_gloss, self.last_conf = None, 0.0
            return None
        seq = normalize_sequence(stacked)           # SAME normalization as training
        probs = self.model.predict(np.expand_dims(seq, 0))[0]
        idx = int(np.argmax(probs))
        return self._stream.update(INDEX_TO_GLOSS[idx], float(probs[idx]))
```
- Buffers frames until 30 are ready, then predicts **every frame** on the sliding window.
- **The hand-presence gate** is clever: the model has no "not signing" class, so it would always
  argmax *some* word. We look at the last 126 features (both hands) and, if hands are essentially
  absent across the window (`< 0.3`), **suppress the guess** — idle noise never surfaces a spurious
  word.
- Normalizes identically to training, then hands `(gloss, confidence)` to the shared `TokenStream`
  so words get the same gate/debounce treatment as letters.

## 2.13 `recognition/src/xai_gradcam.py` — explainability

Covered step-by-step in **Part 6**. Code highlights:
- `_last_conv_layer_name` **descends into nested backbones** to find the last 4-D (conv) layer, and
  handles the Keras-2→3 API change (`layer.output_shape` → `layer.output.shape`).
- `make_gradcam_heatmap` builds a `grad_model` that outputs *(last conv activations, predictions)*,
  uses `tf.GradientTape` to get the gradient of the top class w.r.t. those activations, pools the
  gradients into per-channel weights, and produces a ReLU'd, normalized heatmap.

## 2.14 `recognition/src/evaluate.py` & `evaluate_word.py` — the bake-off scorers

Covered numerically in **Part 5**. Code mechanics:
- Loops every `*.keras` in `models/`, runs it on the **held-out test set**, computes accuracy +
  **macro** precision/recall/F1, a confusion matrix (PNG), **latency** (via direct `model(x)`
  calls, *not* `.predict`, to avoid dispatch overhead — it feeds 20% of the score), and **size** on
  disk.
- Pins metrics to `labels=list(range(29))` so the report/matrix span **all** classes even if a
  small test split misses some (otherwise sklearn crashes/misaligns).
- Feeds the rows to `scorecard.score(...)`, writes `bakeoff.json/.md` + `winner.json`.
- **Robustness** is a *manual* dict (`ROBUSTNESS_SCORES`) filled from the Grad-CAM review;
  **stability** is still a 0.5 placeholder (needs a live-webcam pass) — the code prints exactly
  which models are unreviewed, so nothing is silently faked.

---

# PART 3 — TRACK B: Synthesis, file by file

The flow: **`synthesize()` → `text_to_gloss` → `build_sign_plan` → `apply_sentiment` → `play_sign_plan`**.

## 3.1 `synthesis/src/text_to_gloss.py` — English text → ASL gloss

> Deliberately simple, deterministic, dependency-free. Real ASL grammar is far richer; this is a
> PoC rule set good enough to drive the player.

```python
_STOPWORDS = {"a","an","the","is","are","am","to","of","do","does"}
_SYNONYMS  = {"hi":"hello","hey":"hello","thank":"thanks","thankyou":"thanks","wanna":"want","gonna":"want"}
_PUNCT_RE  = re.compile(r"[^\w\s]", flags=re.UNICODE)
```
- **Stopwords** — function words ASL usually omits. Kept *small on purpose*: over-dropping hurts
  meaning.
- **Synonyms** — normalize a few common words toward glosses that actually exist in the vocabulary
  (so "hi" plays the `hello` clip instead of being fingerspelled).

```python
def normalise(text):
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))  # strip accents
    text = text.casefold()                                           # aggressive lowercase
    text = _PUNCT_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()
```
- Casefold + **accent stripping** (NFKD decomposition then drop combining marks) + punctuation
  removal + whitespace collapse. This makes matching against the vocabulary robust to messy input.

```python
def text_to_gloss(text, drop_stopwords=True):
    for w in normalise(text).split():
        if drop_stopwords and w in _STOPWORDS: continue
        glosses.append(_SYNONYMS.get(w, w))     # synonym-map, else keep the word
    return glosses
```
- Tokenize → drop stopwords → synonym-map → return an ordered gloss list. Words with no clip
  survive here and get **fingerspelled** downstream — the system never dead-ends.

## 3.2 `synthesis/src/gloss_to_signplan.py` — glosses → a renderable plan

> Pure/headless: it builds the plan; `player.py` does the I/O. For each gloss: known word → a
> `WordClip` step; unknown → a `Fingerspell` step.

```python
@dataclass
class WordClip:
    gloss: str; clip_path: Path
    kind: str = field(default="word", init=False)
    hold_ms: int = 0; repeat: int = 1        # sentiment emphasis (Decision A2), default = no effect
@dataclass
class Fingerspell:
    word: str; letters: list[str]            # uppercase A-Z, in order
    kind: str = field(default="fingerspell", init=False)
    hold_ms: int = 0; repeat: int = 1
    def letter_dirs(self, letters_dir=LETTERS_DIR): return [letters_dir/ltr for ltr in self.letters]
```
- Two step types. `hold_ms`/`repeat` default to **"no effect"** so an unmodified plan behaves
  exactly as before — the sentiment layer is an *opt-in* mutation (see 3.4).
- `Fingerspell.letter_dirs` maps each letter to the folder of ASL-alphabet images that the
  **Recognition track already trains on** — one asset set, reused both ways.

```python
@dataclass
class SignPlan:
    source_text: Optional[str]; glosses: list[str]; steps: list[Step]
    def summary(self):   # e.g. "hello [O-U-S-S-A-M-A] thanks"
        return " ".join(s.gloss if isinstance(s,WordClip) else f"[{'-'.join(s.letters)}]" for s in self.steps)
```
- The plan is just an ordered list of steps + a human-readable `summary()` (handy for the UI/logs
  and for showing the professor what the planner decided).

```python
def build_sign_plan(glosses, ...):
    for g in glosses:
        clip = vocab.resolve(g, clips_dir)                 # is there a clip for this gloss?
        if clip is not None: steps.append(WordClip(gloss=g, clip_path=clip))
        else:
            letters = _letters_of(g)                       # A-Z only, uppercased
            if letters: steps.append(Fingerspell(word=g, letters=letters))
    return SignPlan(source_text=source_text, glosses=list(glosses), steps=steps)
```
- `vocab.resolve` is the **vocabulary lookup** — the exact same list the word model classifies.
  This is where "known word vs fingerspell" is decided. Empty results (stray punctuation) are
  skipped.

## 3.3 `synthesis/src/asr.py` — speech → text (optional)

> ASR only produces **text**, which then flows through the identical `text_to_gloss` path — speech
> adds zero downstream complexity.

```python
def _get_whisper(model_size="base"):
    global _WHISPER
    if _WHISPER is not None: return _WHISPER
    try: from faster_whisper import WhisperModel
    except Exception: return None
    _WHISPER = WhisperModel(model_size, device="cpu", compute_type="int8")
    return _WHISPER
```
- **Lazy, cached, optional** — mirrors the MediaPipe pattern. If `faster-whisper` isn't installed,
  returns `None` and the caller falls back; nothing here is imported unless you actually call it.
- `compute_type="int8"` — quantized for CPU speed.

```python
def transcribe(wav_path, model_size="base"):
    model = _get_whisper(model_size)
    if model is not None:
        segments, _ = model.transcribe(wav_path)
        return " ".join(seg.text for seg in segments).strip()
    # else fall back to SpeechRecognition (online Google)
```
- Two backends behind one function: **offline whisper preferred**, online `SpeechRecognition` as
  fallback. `listen_mic` captures from the microphone (SpeechRecognition for capture, whisper for
  the actual transcription if present). `available_backend()` lets the UI show status.

## 3.4 `synthesis/src/pipeline.py` — the orchestrator (your Track B entry point)

> The single function a UI calls: `synthesize(text=... | wav=...)` → a `Result` the player renders.

```python
EMPHASIS_SCORE_THRESHOLD = 0.75   # only emphasise CONFIDENT sentiment
EMPHASIS_HOLD_MS = 400            # extra held pause after the clip
EMPHASIS_REPEAT = 2              # replay the clip twice
```
- The **Decision A2** tuning constants — sentiment's *behavioral* effect on signing.

```python
def apply_sentiment(plan, sentiment):
    if sentiment is None or sentiment.label == "neutral": return plan   # no-op cases
    if sentiment.score < EMPHASIS_SCORE_THRESHOLD: return plan          # weak → no-op
    for step in plan.steps:
        if isinstance(step, WordClip):
            step.hold_ms = EMPHASIS_HOLD_MS; step.repeat = EMPHASIS_REPEAT
    return plan
```
- **The sentiment seam.** Only **confident, non-neutral** sentiment changes playback: known-word
  signs get a **held pause + replay**, like tone of voice. Neutral/weak/missing sentiment leaves the
  plan **untouched** — so this is a true no-op in exactly those cases. Fingerspelled steps are left
  alone (they're usually names/rare words, not emotional content).

```python
def _analyze(text):
    try: from sentiment import analyze
    except Exception: return None
    try: return analyze(text)
    except Exception: return None
```
- Sentiment is called **defensively** — if the sentiment module (your partner's) is missing or
  throws, synthesis still works. It can never break the pipeline.

```python
def synthesize(text=None, wav=None, with_sentiment=True, drop_stopwords=True, ...):
    if wav is not None:
        from .asr import transcribe; text = transcribe(wav)     # speech → text (lazy import)
    if text is None: raise ValueError("synthesize() needs either text= or wav=.")
    glosses = text_to_gloss(text, drop_stopwords=drop_stopwords)
    plan = build_sign_plan(glosses, source_text=text, ...)
    sentiment = _analyze(text) if with_sentiment else None
    plan = apply_sentiment(plan, sentiment)
    return Result(source_text=text, glosses=glosses, plan=plan, sentiment=sentiment)
```
- The whole Track B in one function: (optionally transcribe speech) → glosses → plan → annotate with
  sentiment → return a `Result`. Speech and text converge on the **same** downstream path.

## 3.5 `synthesis/src/player.py` — render the plan (OpenCV I/O)

> The I/O layer (verified manually — it opens a window). Everything above it stays headless-testable.

```python
def play_sign_plan(plan, fps=25, letter_hold_ms=700, ...):
    delay = max(1, int(1000/fps))
    for step in plan.steps:
        if isinstance(step, WordClip):
            if not step.clip_path.is_file(): _report(...); continue     # missing clip → skip
            for _ in range(max(1, step.repeat)):                        # ← A2 replay
                cap = cv2.VideoCapture(str(step.clip_path))
                while True:
                    ok, frame = cap.read()
                    if not ok: break
                    cv2.imshow(window, frame)
                    if cv2.waitKey(delay) & 0xFF in (ord("q"), 27): return   # q/Esc quits
                cap.release()
            if step.hold_ms and cv2.waitKey(step.hold_ms) ...:          # ← A2 held pause
```
- Plays each `WordClip` frame-by-frame at `fps`. **`repeat` and `hold_ms`** (set by `apply_sentiment`)
  are honored here — that's how emphasis becomes visible motion. Missing clips are **skipped with a
  report**, never a crash.

```python
        elif isinstance(step, Fingerspell):
            for _ in range(max(1, step.repeat)):
                for ltr, ldir in zip(step.letters, step.letter_dirs(letters_dir)):
                    img = cv2.imread(str(_first_image(ldir)))
                    cv2.imshow(window, img)
                    cv2.waitKey(letter_hold_ms + step.hold_ms)      # hold each letter ~700ms
```
- Fingerspelling shows **one image per letter**, held ~700 ms each (`_first_image` picks a
  representative frame from the letter's folder). The player is fully **defensive** — every missing
  or unreadable asset is reported and skipped so a demo never hard-fails.

## 3.6 `shared/paths.py` — one path resolver for source *and* frozen builds

```python
def app_root():
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).resolve().parent))
    return Path(__file__).resolve().parents[1]
```
- Every config locates data/models relative to "the repo root". When packaged with PyInstaller,
  `__file__` points *inside the bundle*, so this returns `sys._MEIPASS` (the bundle's data root)
  instead. Centralizing it here is why the **same code runs from source, on Colab, and as a `.exe`**.

---

# PART 4 — Architectures (layers + hyperparameters)

## 4.A Fingerspelling CNN — 29 classes, input 224×224×3

**`cnn_scratch`** (VGG-style, ~1.25M params, Grad-CAM target = the 256-filter block):

| Stage | Layers | Filters |
|-------|--------|---------|
| Block 1 | Conv-BN-ReLU ×2 + MaxPool | 32 |
| Block 2 | Conv-BN-ReLU ×2 + MaxPool | 64 |
| Block 3 | Conv-BN-ReLU ×2 + MaxPool | 128 |
| Block 4 | Conv-BN-ReLU ×2 + MaxPool | 256 |
| Head | GAP → Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3) → Dense(29, softmax) |

**Transfer-learning candidates** (shared head: `GAP → Dropout(0.3) → Dense(29, softmax)`):

| Model | Preprocessing (built-in) | Params | Frozen at start |
|-------|--------------------------|--------|-----------------|
| EfficientNetB0 | Rescaling `[0,1]→[0,255]` | ~4.09M | yes |
| MobileNetV2 | `preprocess_input` → `[-1,1]` | ~2.30M | yes |
| ResNet50 | `preprocess_input` (caffe) | ~23.6M | yes |

**Fixed hyperparameters:** 224², batch 32, Adam, LR 1e-3 (head) / 1e-4 (fine-tune), loss
`sparse_categorical_crossentropy`, val split 15%, EarlyStopping patience 4, ReduceLROnPlateau
(0.3, patience 2), seed 42. **Two-phase fine-tuning:** Phase 1 head only; Phase 2 unfreeze last ~30
backbone layers (BatchNorm frozen) at LR/10.

## 4.B Word-sign sequence models — 20 classes, input (30, 258)

All: `Input(30,258) → Masking(0.0) → … → Dense(20, softmax)`.

| Candidate | Core | Params |
|-----------|------|--------|
| lstm | LSTM(128, seq) → Drop → LSTM(64) → Drop → Dense(64) | 253,012 |
| gru | GRU(128, seq) → Drop → GRU(64) → Drop → Dense(64) | 191,700 |
| bilstm | Bi-LSTM(96, seq) → Drop → Bi-LSTM(48) → Drop → Dense(64) | 372,692 |
| transformer | Dense(128)+pos-emb → 2× block → GAP1D → Dense(64) | 241,876 |

**Pre-norm Transformer block:**
```
LayerNorm → MultiHeadAttention(4 heads, key_dim=32) → + residual
LayerNorm → Dense(128, ReLU) → Dropout → Dense(d_model) → + residual
```
`d_model=128`, learned `Embedding(30,128)` positional encoding, 2 stacked blocks.
**Hyperparameters:** batch 16, epochs 60 (EarlyStopping), Adam 1e-3, augmentation ×8, Masking 0.0.

## 4.C Sentiment (partner-owned) — 3 candidates

| Candidate | Architecture | Size |
|-----------|--------------|------|
| scratch | TF-IDF + Logistic Regression (IMDB) | 0.84 MB |
| distilbert | DistilBERT SST-2 (binary), PyTorch | 268 MB |
| twitter_roberta | RoBERTa Twitter (3-class), PyTorch | 499 MB |

Binary models derive "neutral" from a probability band around 0.5 (`NEUTRAL_BAND = 0.15`).

---

# PART 5 — Metrics (weighted scorecard) + the 3 stories

**Principle (`scorecard.py`):** each metric is **min-max normalized to [0,1]** across candidates
(lower-is-better metrics inverted), then **weighted**; the total ranks them.

**CNN results (fingerspelling, ~99–100% — easy dataset):**

| Model | Test acc | Latency | Size | Verdict |
|-------|----------|---------|------|---------|
| EfficientNetB0 | 0.9994 | ~297 ms (slowest) | int8 4.94 MB | 🏆 scorecard winner (0.629) |
| resnet50 | 1.0000 | heavy | largest | accuracy ceiling |
| cnn_scratch | 0.9932 | ~26 ms | 15 MB | real-time / deployable |
| mobilenetv2 | 0.9932 | fast | light | — |

> **Story 1 — "EfficientNet wins but is slow."** It tops accuracy but is the slowest; `cnn_scratch`
> is what you'd deploy live (26 ms). Re-weight toward latency and the in-house baseline wins — *the
> scorecard encodes the priorities*.

**Word results (20 classes, chance 5%):**

| Rank | Model | Acc | Macro-F1 | Latency | Score |
|------|-------|-----|----------|---------|-------|
| 1 | gru | 0.7542 | 0.7472 | 142 ms | 0.724 🏆 |
| 2 | transformer | 0.7458 | 0.7494 | 21 ms | 0.711 |
| 3 | bilstm | 0.7797 | 0.7914 | 237 ms | 0.600 |
| 4 | lstm | 0.6864 | 0.6487 | 118 ms | 0.244 |

> **Story 2 — "gru vs transformer."** The scorecard picks **gru**, but we keep the **transformer**
> live: within noise on accuracy (74.6 vs 75.4%) and **~7× faster** (21 vs 142 ms), which matters
> more for the capture-to-commit UX. We report **Macro-F1** (not just accuracy) because classes are
> small and equal-weighted.

**Sentiment results:**

| Model | IMDB acc | Raw score | Realistic eval |
|-------|----------|-----------|----------------|
| distilbert | 0.86 | 0.744 (raw winner) | 0.70 — 0/6 neutrals! |
| scratch | 0.768 | 0.662 | 0.90 |
| twitter_roberta | 0.724 | 0.000 | 1.00 ✓ recommended |

> **Story 3 — the metric trap.** The IMDB scorecard picks distilbert, but IMDB has **no neutral
> examples**, so it can't test neutral detection. On realistic sentences distilbert misses **all**
> neutrals while twitter_roberta scores 100%. **The wrong benchmark picks the wrong model.**

---

# PART 6 — XAI (explainability)

## 6.1 Grad-CAM on the CNN, step by step (`xai_gradcam.py`)

**Question it answers:** *which pixels drove this prediction?*

1. Take the **last conv layer** (256-filter block for `cnn_scratch`) — its activation maps still
   carry spatial resolution.
2. Build a `grad_model` outputting *(those activations, the predictions)*.
3. With `tf.GradientTape`, compute the **gradient of the predicted class score** w.r.t. those
   activations.
4. **Global-average the gradients** → one importance weight per feature map.
5. **Weighted sum** of maps → **ReLU** (keep what pushes *toward* the class) → normalize to [0,1] →
   resize to 224² → overlay in `jet`.

**What it revealed:**
- **EfficientNetB0** attends **tightly to the hand** → robust.
- **cnn_scratch** sometimes drifts to the **background** on hard cases → less robust.
- Hard pairs **M/A, M/E, Q/G** (predicted at EDA) are confirmed → **closes the CRISP-DM loop**.
- These feed the manual **robustness** score in the scorecard (EfficientNet 0.85, cnn_scratch 0.65).

**Why it matters here:** a model that "cheats" on background or skin tone would be **biased**. In
sign language that's an equity issue, so Grad-CAM is a **trust/bias check**, not decoration.

## 6.2 Interpretability of the other models

- **Word model:** no Grad-CAM (no image), but interpretable via **landmark normalization** (it sees
  invariant hand/body geometry, not pixels) and **Masking** (we know which frames count); the
  **confusion matrix** shows which words blur together.
- **Sentiment:** the `scratch` TF-IDF+LogReg model is **intrinsically interpretable** — you can read
  the per-word weights. The neutral-band analysis itself explains model behavior.

---

## Appendix — one-line map of every file you own

**Track A (recognition/src/):** `config.py` protocol · `preprocess.py` static crop · `data.py` CNN
dataset · `models/*` CNN candidates · `train.py` CNN trainer · `holistic.py` landmarks+normalize ·
`sequence_data.py` word dataset · `word_models.py` seq candidates · `train_word.py` seq trainer ·
`token_stream.py` gate/debounce · `word_stream.py` live word buffer · `evaluate*.py` scorers ·
`scorecard.py` weighting · `xai_gradcam.py` Grad-CAM.

**Track B (synthesis/src/):** `text_to_gloss.py` text→gloss · `gloss_to_signplan.py` plan builder ·
`asr.py` speech→text · `pipeline.py` orchestrator+sentiment seam · `player.py` OpenCV renderer.

**Shared (shared/):** `contract.py` Token/Sentiment · `vocabulary.py` the 20 words · `config.py`
seq geometry+paths · `paths.py` frozen-safe root.
