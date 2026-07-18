# GROOPY — Presentation & Demo Runbook

Everything you need for the soutenance: a slide outline, a step-by-step live-demo script, and
fallbacks if something misbehaves.

---

## Part 1 — Slide outline (≈ 6–7 slides)

**1. Title & problem**
- GROOPY: two-way sign-language communication between a Deaf/HoH signer and a hearing person.
- The gap: no interpreter always available. *A proof of concept, built with a community-centered
  ethos — not a replacement for human interpreters.*

**2. The system (one diagram)**
- **Recognition** (Sign → Text/Speech) and **Synthesis** (Text/Speech → Sign), over a **shared
  contract + vocabulary**.
- One 20-word vocabulary = the word model's classes *and* the synthesis clip keys.

**3. Methodology — CRISP-DM + the "bake-off"**
- Data Understanding → Preparation → Modeling → Evaluation → Deployment.
- Key idea: don't assume a model — **compare several under one fixed protocol** and let a
  **weighted scorecard** pick the most *shippable* one.

**4. Recognition results — two bake-offs**
- **Fingerspelling (image CNN):** EfficientNetB0 wins (≈ 99.9 %); but cnn_scratch is 11× faster/
  smallest → the *deployable* model. *"The scorecard's weights encode the deployment priority."*
- **Words (sequence models):** Transformer wins — best accuracy (78 %) **and** fastest (20 ms).

**5. The hard part — data & trust**
- Word recognition: **chance (5 %) → 78 %**, via **landmark normalization** + a **data
  investigation** (WLASL was 62 % dead → recovered clips + switched to ASL Citizen).
- **Grad-CAM**: models look at the **hand**, not the background; and they confuse the *exact*
  letter pairs (M/A, Q/G) we predicted in the EDA — the CRISP-DM loop closes.

**6. Live demo** (see Part 2) — synthesis, then word recognition.

**7. Limitations & conclusion**
- PoC; 20-word vocab; naive gloss; fixed clip dictionary; mobile/web not built.
- Strongest outcomes: two rigorous bake-offs, a normalization fix (chance→78 %), an
  evidence-based data investigation.

---

## Part 2 — Live-demo script

### Before you start (checklist)
- [ ] Terminal in the project folder, venv active: `.venv\Scripts\activate`
- [ ] Windows **camera access enabled** (Settings → Privacy → Camera → desktop apps ON)
- [ ] Close other apps using the webcam (Zoom/Teams/browser)
- [ ] Models present locally: `recognition\models\word_transformer.keras`
- [ ] Good lighting, plain background, hands visible

### Demo A — Synthesis (Text/Speech → Sign)  *[most reliable — start here]*
```powershell
python desktop\synthesis_app.py
```
1. Type **`hello my name is oussama`** → click **Sign it**.
2. Narrate: *"Known words play a sign clip; unknown words — like my name — are automatically
   fingerspelled letter by letter."* → it plays `hello`, `name`, then spells `O-U-S-S-A-M-A`.
3. Point out the **gloss line** and the **sentiment label**.
4. *(Optional)* Click **🎤 Speak** and say a short phrase → it transcribes and signs it.
- Press **q/Esc** in the video window to stop playback.

### Demo B — Word recognition (Sign → Text)
```powershell
python desktop\app.py --word-model recognition\models\word_transformer.keras
```
1. Wait for your webcam feed to appear.
2. Perform one of the **20 words** (easy ones: **hello, yes, no, thanks**), **hold it ~1 second**.
3. Watch the **live guess** (`word %`) settle, then press **Space** to commit it to the sentence.
4. Narrate: *"It only guesses when hands are visible, and I confirm the word with Space — so it
   doesn't spew noise."*

### Fingerspelling
- **Pending the professor's decision** (Colab vs. local model). Have the **Colab results ready as
  a backup**: the scorecard table + the two Grad-CAM heatmaps (`recognition/results/gradcam_*.png`).

---

## Part 3 — If something goes wrong (fallbacks)

| Problem | Fix on the spot |
|---|---|
| Webcam feed blank / "no camera feed" | Check Windows camera privacy; close other apps; it's a system setting, not the app. |
| Word never commits (guess stuck < 80 %) | Lower `CONFIDENCE_GATE` in `recognition/src/config.py` to ~0.5 **before** the demo. |
| A word keeps misreading | Retry — retries are free; or switch to an easier word (hello/yes/no). |
| Synthesis: a word shows as fingerspelled | Expected if it's outside the 20-word vocab — that's the fallback feature, say so. |
| 🎤 Speak does nothing | It needs `faster-whisper` (installed) + a working mic; skip to typed input if unsure. |

**Golden rule:** lead with **Demo A (synthesis)** — it needs no camera and always works. Keep the
Colab fingerspelling results open in a tab as a safety net.

---

## Part 4 — One-line summary to open/close with
> *"GROOPY translates sign language both ways. We didn't just build models — we compared
> architectures rigorously under CRISP-DM, diagnosed and fixed a real data-scarcity problem, and
> verified with Grad-CAM that the models look at the hand. It's a proof of concept, built with the
> deaf community in mind."*
