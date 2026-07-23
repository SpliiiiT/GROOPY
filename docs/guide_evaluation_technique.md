# GROOPY — Guide d'évaluation technique (phase ACT)

> **But de ce document** : te préparer à défendre le projet devant le prof.
> La note ACT porte sur **4 axes** :
> 1. **Fonctionnement du code** — comment le système marche de bout en bout
> 2. **Architectures des modèles** — les couches + les hyperparamètres
> 3. **Métriques d'évaluation** — comment on a mesuré et comparé
> 4. **XAI** — l'explicabilité (Grad-CAM et interprétabilité)
>
> Ce guide couvre les **3 modèles maison** (bake-offs) du projet : le **CNN** (dactylologie /
> fingerspelling), le **modèle de séquence** (signes de mots), et le **sentiment**.

---

## 0. Vue d'ensemble — c'est quoi GROOPY ?

GROOPY est une **preuve de concept de communication bidirectionnelle** en langue des signes,
entre une personne sourde/malentendante (le signeur) et une personne entendante non-signeuse.

Deux directions + une couche partagée, dans un seul repo :

| Piste | Dossier | Sens | Ce qui la porte |
|-------|---------|------|-----------------|
| **A — Recognition** | `recognition/` | Signe → Texte/Parole | CNN (lettres) + modèle de séquence (mots) |
| **B — Synthesis** | `synthesis/` | Texte/Parole → Signe | ASR → gloss → plan de signes → lecture de clips |
| **shared/** | `shared/` | source de vérité | `contract.py`, `vocabulary.py`, `config.py` |
| **sentiment/** | `sentiment/` | analyse du ton | bake-off 3 modèles |

**Le mot-clé du projet, c'est « bake-off »** : pour chaque tâche on n'entraîne pas UN modèle,
on en met **plusieurs en compétition** sous un **protocole identique**, puis un **scorecard
pondéré** désigne le gagnant. C'est ça qui montre au prof qu'on a compris le trade-off
(précision / vitesse / taille), pas juste qu'on sait appeler `.fit()`.

**Méthodologie** : CRISP-DM (Business understanding → Data → Modeling → Evaluation → Deployment).
Chaque bake-off suit ces étapes.

---

## 1. FONCTIONNEMENT DU CODE (axe 1)

### 1.1 Architecture logicielle globale

```
shared/           ← contrat + vocabulaire + chemins (importé par TOUT le monde)
  contract.py       Token(text, kind, confidence, sentiment)  — le format d'échange (v2)
  vocabulary.py     les 20 mots curés = classes du modèle mots = clés des clips
  config.py         SEQ_LEN=30, FRAME_FEATURES=258, chemins repo-relatifs

recognition/
  src/config.py     protocole d'entraînement FIXE (mêmes hyperparamètres pour tous)
  src/models/       CNN candidats : cnn_scratch, efficientnet, mobilenetv2, resnet50
  src/word_models.py candidats séquence : lstm, gru, bilstm, transformer
  src/train.py      entraîneur unifié CNN (protocole identique pour chaque candidat)
  src/train_word.py entraîneur unifié séquence
  src/evaluate*.py  calcule les métriques du bake-off
  src/scorecard.py  normalise les métriques + pondère → désigne le gagnant
  src/xai_gradcam.py Grad-CAM (explicabilité)
  src/holistic.py   MediaPipe Holistic + normalisation des landmarks
  src/word_stream.py inférence live (webcam) pour les mots

synthesis/
  src/pipeline.py   asr → text_to_gloss → gloss_to_signplan → player
sentiment/
  src/models.py     3 backends (scratch / distilbert / twitter_roberta)
desktop/            apps de démo (fingerspelling + mots + synthèse) + launcher packagé
```

### 1.2 Le pipeline Recognition (Signe → Texte), en direct

**Dactylologie (lettres), `desktop/app.py` :**
1. La webcam envoie une image → recadrée en **224×224**, normalisée en `[0,1]`.
2. Un **gate de présence de main** vérifie qu'une main est visible (sinon on ne prédit rien).
3. Le CNN sort une **softmax sur 29 classes** (A–Z + `del`, `nothing`, `space`).
4. **Gate de confiance** (`CONFIDENCE_GATE = 0.80`) : on n'émet un token que si la proba ≥ 0,80.
5. **Debounce** (`DEBOUNCE_MS = 500`) : au plus une prédiction par fenêtre de 500 ms.
6. **Capture-to-commit** : la lettre n'est ajoutée au texte que sur action (pas de « spew »
   automatique) → évite de spammer 30 lettres/seconde.

**Signes de mots, `word_stream.py` :**
1. **MediaPipe Holistic** extrait à chaque frame les landmarks : **pose 33 + main gauche 21 +
   main droite 21**.
2. Chaque frame → un vecteur de **258 features** = `pose 33×(x,y,z,visibilité) + LH 21×(x,y,z) +
   RH 21×(x,y,z)`.
3. On empile **30 frames** (`SEQ_LEN=30`) → une séquence `(30, 258)`.
4. **Normalisation des landmarks** (`holistic.normalize_sequence`) : on recentre x,y sur le
   milieu des épaules et on divise par la largeur d'épaules (pose 11/12). ⚠️ **C'est LE fix
   décisif** : la précision est passée de ~5 % (hasard) à 37 % puis 70 %. Elle est appliquée
   **à l'identique** à l'entraînement et en live → pas de « train/serve skew ».
5. Le modèle de séquence sort une **softmax sur 20 classes**.

### 1.3 La couche partagée (le « contrat »)

`shared/vocabulary.py` définit **20 mots** (`hello, thanks, please, yes, no, sorry, help, want,
need, name, you, me, good, bad, happy, sad, eat, drink, friend, love`). Cette liste est
**simultanément** : (a) les classes de sortie du modèle de mots, et (b) les clés des clips de
synthèse. Un seul endroit à changer → les deux directions se mettent à jour. Tout mot **hors de
cette liste** est géré par **dactylologie** (lettre par lettre) → le système ne se bloque jamais.

### 1.4 Reproductibilité

`SEED = 42` partout, mêmes splits, mêmes hyperparamètres pour tous les candidats d'un bake-off.
C'est ce qui rend la comparaison **juste** (fair).

---

## 2. ARCHITECTURES DES MODÈLES (axe 2 : couches + hyperparamètres)

### 2.A — CNN de dactylologie (fingerspelling)

**Tâche** : classer une image de main en une des **29 classes** (A–Z + del/nothing/space).
**Données** : Kaggle *ASL Alphabet*, 87 000 images, 3000/classe. **Entrée** : `224×224×3`, `[0,1]`.

#### Candidat 1 — `cnn_scratch` (le baseline « fait maison »)
> Fichier : [cnn_scratch.py](../recognition/src/models/cnn_scratch.py). C'est LE modèle « on a
> vraiment compris les CNN » — on conçoit chaque couche.

Bloc convolutif répété (style VGG) : `[Conv2D → BatchNorm → ReLU] ×2 → MaxPooling2D`.

| Étage | Couches | Filtres |
|-------|---------|---------|
| Bloc 1 | Conv-BN-ReLU ×2 + MaxPool | 32 |
| Bloc 2 | Conv-BN-ReLU ×2 + MaxPool | 64 |
| Bloc 3 | Conv-BN-ReLU ×2 + MaxPool | 128 |
| Bloc 4 | Conv-BN-ReLU ×2 + MaxPool | 256 ← **couche cible du Grad-CAM** |
| Tête | GlobalAveragePooling2D → Dropout(0.4) → Dense(256, ReLU) → Dropout(0.3) → Dense(29, softmax) |

- **Conv2D** : noyaux 3×3, `padding="same"`, `use_bias=False` (le biais est inutile car la
  BatchNorm juste après le neutralise).
- **BatchNormalization** : stabilise et accélère l'entraînement.
- **GlobalAveragePooling** (au lieu de Flatten) : beaucoup moins de paramètres → modèle petit
  (~1,25 M params, ~15 MB) et **Grad-CAM propre**.
- **Dropout 0.4 / 0.3** : régularisation contre le sur-apprentissage.
- **~1 248 381 paramètres**.

#### Candidats 2–4 — Transfer learning (backbones pré-entraînés ImageNet)
> `efficientnet.py`, `mobilenetv2.py`, `resnet50.py`. Même **tête** pour tous :
> `GlobalAveragePooling2D → Dropout(0.3) → Dense(29, softmax)`.

| Modèle | Backbone gelé au départ | Prétraitement intégré | Params | Rôle dans le bake-off |
|--------|------------------------|-----------------------|--------|----------------------|
| **EfficientNetB0** | oui | Rescaling `[0,1]→[0,255]` (norm interne) | ~4,09 M | meilleur ratio précision/FLOP |
| **MobileNetV2** | oui | `preprocess_input` → `[-1,1]` | ~2,30 M | léger, orienté mobile |
| **ResNet50** | oui | `preprocess_input` (caffe, soustraction de moyenne) | ~23,6 M | plafond de précision, le plus lourd |

**Point clé à savoir expliquer** : le prétraitement est **inclus dans le modèle** (couche
`Rescaling` / `preprocess_input`), donc l'appelant passe toujours du `[0,1]` sans se soucier de
la convention de chaque backbone.

#### Hyperparamètres CNN (protocole FIXE, `recognition/src/config.py`)
| Hyperparamètre | Valeur |
|---|---|
| Image | 224×224×3 |
| Batch size | 32 |
| Epochs | 20 (max) |
| Optimiseur | **Adam** |
| Learning rate (tête) | **1e-3** ; backbones fine-tunés à **LR/10 = 1e-4** |
| Loss | `sparse_categorical_crossentropy` |
| Validation split | 15 % |
| EarlyStopping | patience 4 sur `val_accuracy`, `restore_best_weights` |
| ReduceLROnPlateau | factor 0.3, patience 2, min_lr 1e-6 |
| Seed | 42 |

**Fine-tuning en 2 phases (pour les pré-entraînés uniquement)** — c'est du transfer learning
« by the book » :
- **Phase 1** : backbone gelé, on entraîne seulement la nouvelle tête (LR 1e-3).
- **Phase 2** : on dégèle **les ~30 dernières couches** du backbone, on garde les
  **BatchNorm gelées** (pratique standard), et on ré-entraîne à **LR/10**.
- Le `cnn_scratch` s'entraîne en **une seule phase** (rien à dégeler).

---

### 2.B — Modèle de séquence pour les signes de mots

**Tâche** : classer une **séquence** `(30 frames, 258 features)` en un des **20 mots**.
Un mot signé est un **mouvement**, donc les candidats sont des modèles **de séquence** (pas des
CNN). Tous partagent : `Input(30,258) → Masking(0.0)` (ignore les frames de padding) → … →
`Dense(20, softmax)`.

| Candidat | Cœur de l'architecture | Params | Idée |
|----------|------------------------|--------|------|
| **lstm** | LSTM(128, seq) → Drop(0.3) → LSTM(64) → Drop(0.3) → Dense(64) | 253 012 | baseline récurrent |
| **gru** | GRU(128, seq) → Drop → GRU(64) → Drop → Dense(64) | 191 700 | grille plus légère, entraîne plus vite |
| **bilstm** | Bi-LSTM(96, seq) → Drop → Bi-LSTM(48) → Drop → Dense(64) | 372 692 | lit le signe dans les 2 sens |
| **transformer** | Dense(128)+pos-embedding → 2× bloc attention → GAP1D → Dense(64) | 241 876 | attention, non-récurrent, parallélisable |

**Le bloc Transformer** (pré-norm, avec résidus) :
```
LayerNorm → MultiHeadAttention(4 têtes, key_dim=32) → + résidu
LayerNorm → Dense(128, ReLU) → Dropout → Dense(d_model) → + résidu
```
- **Positional embedding appris** (`Embedding(30, 128)`) : dit au modèle *quelle frame* c'est
  (l'attention seule est invariante à l'ordre).
- `d_model = 128`, 2 blocs empilés, `GlobalAveragePooling1D` avant la tête.

#### Hyperparamètres séquence (`train_word.py`)
| Hyperparamètre | Valeur |
|---|---|
| SEQ_LEN | 30 frames |
| Features/frame | 258 |
| Batch size | 16 |
| Epochs | 60 (max, avec EarlyStopping) |
| Optimiseur / LR | Adam / 1e-3 |
| Loss | `sparse_categorical_crossentropy` |
| **Augmentation** | ×8 copies par échantillon (bruit/scale/décalage) — **vital** vu le peu de données (~40/classe) |
| Masking | valeur 0.0 (frames padées ignorées) |

**À savoir défendre** : pourquoi le Masking ? Parce que les séquences font < 30 frames et sont
**padées à 0** ; le Masking empêche le modèle d'apprendre sur du vide. Pourquoi l'augmentation
×8 ? Parce que le dataset est **minuscule** (791 séquences, ~40/classe) — c'est le vrai mur du
projet, résolu en combinant **WLASL + ASL Citizen**.

---

### 2.C — Modèle de sentiment (analyse du ton)

**Tâche** : classer un texte en `positive / negative / neutral`. 3 candidats :

| Candidat | Architecture | Taille | Note |
|----------|--------------|--------|------|
| **scratch** | **TF-IDF + Régression Logistique** (from scratch, sur IMDB) | 0,84 MB | s'entraîne en ~9 s CPU |
| **distilbert** | Transformer pré-entraîné (DistilBERT SST-2, **binaire**) | 268 MB | PyTorch CPU |
| **twitter_roberta** | Transformer pré-entraîné (RoBERTa Twitter, **3 classes natif**) | 499 MB | PyTorch CPU |

- Les modèles **binaires** (scratch, distilbert) dérivent le « neutral » d'une **bande de proba
  autour de 0,5** (`NEUTRAL_BAND = 0.15`), car IMDB n'a pas de label neutre.
- ⚠️ **Note technique importante** : HuggingFace `transformers` a **retiré le support TensorFlow**
  dans ses versions récentes → les 2 pré-entraînés tournent sur **PyTorch (CPU only)**, isolés
  dans `sentiment/requirements.txt` (le reste du projet est en TensorFlow/Keras).

---

## 3. MÉTRIQUES D'ÉVALUATION (axe 3)

### 3.1 Le principe : scorecard pondéré (`scorecard.py`)

On ne juge PAS un modèle sur la seule précision. Chaque métrique est **normalisée min-max en
[0,1]** sur l'ensemble des candidats (les « plus petit = mieux » comme la latence sont inversées),
puis **pondérée**. Le total désigne le gagnant.

**Poids CNN** : accuracy 40 % · latency 20 % · size 15 % · robustness 15 % · stability 10 %
*(robustness & stability sont des scores manuels 0–1 : la vérif humaine « le modèle regarde bien
la main et reste stable en live » fait partie de la décision).*

**Poids mots** : accuracy 60 % · latency 20 % · size 20 % *(100 % automatique).*

**Poids sentiment** : accuracy 50 % · latency 30 % · size 20 %.

### 3.2 Résultats CNN (dactylologie) — dataset « facile », tous à ~99–100 %

| Modèle | Test acc | Params | Latence | Taille | Verdict |
|--------|----------|--------|---------|--------|---------|
| **EfficientNetB0** | **0.9994** | 4,09 M | ~297 ms (le + lent) | export int8 4,94 MB | **🏆 gagnant du scorecard (0.629)** |
| resnet50 | 1.0000 | 23,6 M | lourd | le + gros | plafond de précision |
| cnn_scratch | 0.9932 | 1,25 M | **~26 ms** | 15 MB | **le vrai modèle temps-réel/déployable** |
| mobilenetv2 | 0.9932 | 2,30 M | rapide | léger | — |

**L'histoire à raconter** : EfficientNet gagne sur la précision **mais est le plus lent**.
Le `cnn_scratch` est celui qu'on déploierait en temps réel (26 ms, 15 MB, 99,3 %). *« Le
scorecard encode les priorités »* — si on donnait plus de poids à la latence, le baseline maison
gagnerait. C'est exactement le genre de nuance que le prof veut entendre.

### 3.3 Résultats séquence (mots) — 20 classes, hasard = 5 %

| Rang | Modèle | Acc | Macro-F1 | Latence | Taille | Score |
|------|--------|-----|----------|---------|--------|-------|
| 1 | **gru** | 0.7542 | 0.7472 | 142 ms | 2,35 MB | **0.724** 🏆 |
| 2 | transformer | 0.7458 | 0.7494 | **21 ms** | 3,13 MB | 0.711 |
| 3 | bilstm | **0.7797** | 0.7914 | 237 ms | 4,54 MB | 0.600 |
| 4 | lstm | 0.6864 | 0.6487 | 118 ms | 3,08 MB | 0.244 |

**Nuance à connaître** : le scorecard désigne **gru**, mais on garde le **transformer** comme
modèle live par défaut — il est dans le bruit du gru (74,6 % vs 75,4 %) et **~7× plus rapide**
(21 ms vs 142 ms), ce qui compte plus pour l'UX capture-to-commit. Le **bilstm** a la meilleure
précision brute (77,97 %) mais est pénalisé par sa lenteur/taille. On mesure le **Macro-F1** (et
pas juste l'accuracy) parce que les classes sont petites → la F1 macro traite chaque classe à
poids égal.

### 3.4 Résultats sentiment — LE bon exemple de « métrique piège »

| Modèle | Acc IMDB | Latence | Taille | Score brut | **Éval réaliste (20 phrases app)** |
|--------|----------|---------|--------|-----------|-----------------------------------|
| distilbert | 0.86 | 40 ms | 268 MB | **0.744** (gagnant brut) | **0.70** — rate **0/6** les neutres ! |
| scratch | 0.768 | 0,6 ms | 0,84 MB | 0.662 | 0.90 |
| twitter_roberta | 0.724 | 81 ms | 499 MB | 0.000 | **1.00** ✅ **← le vrai bon choix** |

**L'histoire (très valorisée en soutenance)** : le scorecard basé sur l'accuracy IMDB choisit
**distilbert**. Mais **IMDB n'a AUCUN exemple neutre** → la métrique ne peut pas tester la
détection du neutre. Sur un test réaliste (`eval_realistic.py`), distilbert rate **tous** les
neutres (entraîné uniquement à forcer pos/neg), alors que **twitter_roberta** fait 100 %.
Conclusion : `RECOMMENDED_MODEL = "twitter_roberta"`. **Le mauvais benchmark choisit le mauvais
modèle** — c'est une vraie leçon de data science, pas juste un chiffre.

---

## 4. XAI — EXPLICABILITÉ (axe 4)

### 4.1 Grad-CAM sur le CNN (`xai_gradcam.py`) — l'artefact XAI principal

**Grad-CAM** (Gradient-weighted Class Activation Mapping) répond à : *« sur quelles zones de
l'image le CNN s'appuie-t-il pour sa prédiction ? »* On produit une **heatmap** superposée à
l'image.

**Comment ça marche (à savoir expliquer)** :
1. On prend la **dernière couche convolutive** (pour `cnn_scratch`, le bloc à 256 filtres) —
   ses cartes d'activation gardent une résolution spatiale.
2. On calcule le **gradient de la classe prédite** par rapport à ces cartes d'activation
   (`tf.GradientTape`).
3. On fait la **moyenne globale des gradients** (pooling) → un poids par carte = « importance ».
4. **Combinaison pondérée** des cartes → ReLU (on garde ce qui pousse *vers* la classe) →
   normalisation [0,1] → on redimensionne en 224×224 et on superpose en `jet`.

**Ce qu'on a trouvé (résultats dans `recognition/results/gradcam_*.png`)** :
- **EfficientNetB0** concentre son attention **serrée sur la main** → robuste.
- **cnn_scratch** part parfois sur **l'arrière-plan** sur les cas durs → moins robuste.
- Les **paires difficiles M/A, M/E, Q/G** (prédites dès l'EDA) sont confirmées → **ça ferme la
  boucle CRISP-DM** : ce qu'on avait supposé à l'exploration est vérifié par l'explicabilité.
- Ces observations **alimentent le score « robustness »** du scorecard (0–1 manuel). Après
  re-scoring avec la robustesse, EfficientNet gagne **encore plus large** (0.681).

**Pourquoi c'est important ici** : en langue des signes, un modèle qui « triche » sur
l'arrière-plan ou la couleur de peau serait **biaisé et inéquitable**. Grad-CAM est donc un
**contrôle de confiance et de biais**, pas juste une jolie image.

### 4.2 Interprétabilité des autres modèles

- **Modèle de séquence (mots)** : pas de Grad-CAM (pas d'image), mais l'explicabilité vient de
  la **normalisation des landmarks** (le modèle voit une géométrie main/corps invariante à la
  position dans le cadre, pas des pixels bruts) et du **Masking** (on sait exactement quelles
  frames comptent). On peut aussi inspecter la **matrice de confusion** pour voir quels mots se
  confondent.
- **Sentiment** : le modèle `scratch` (TF-IDF + LogReg) est **intrinsèquement interprétable** —
  on peut lire les **poids des mots** (quels tokens poussent vers positif/négatif). L'analyse de
  la **bande neutre** (§3.4) est elle-même une forme d'explicabilité du comportement du modèle.

---

## 5. Antisèche — questions probables du prof

| Question | Réponse courte |
|----------|----------------|
| Pourquoi un CNN pour les lettres et un RNN/Transformer pour les mots ? | Une lettre est une **image statique** (CNN) ; un mot est un **mouvement dans le temps** (séquence). |
| Pourquoi `use_bias=False` dans les Conv ? | La **BatchNorm** qui suit annule tout biais → paramètre inutile. |
| Pourquoi GlobalAveragePooling plutôt que Flatten ? | Beaucoup moins de params (modèle petit) **et** Grad-CAM propre. |
| Rôle du Dropout ? | Régularisation, évite le sur-apprentissage (crucial avec peu de données). |
| Qu'est-ce qui a débloqué le modèle de mots ? | La **normalisation des landmarks** (5 %→37 %→70 %), + fusion WLASL/ASL Citizen (791 séquences). |
| Pourquoi fine-tuning en 2 phases ? | On protège les poids ImageNet : on entraîne d'abord la tête, puis on dégèle **doucement** (LR/10, BatchNorm gelées). |
| C'est quoi le scorecard ? | Une décision **multi-critères** (précision + latence + taille + robustesse) normalisée et pondérée — pas juste l'accuracy. |
| Meilleur exemple de rigueur d'évaluation ? | Sentiment : le benchmark IMDB choisit distilbert, mais il **rate tous les neutres** ; twitter_roberta gagne sur un test réaliste. |
| C'est quoi Grad-CAM et à quoi ça sert ici ? | Une heatmap des zones qui déclenchent la prédiction → vérifier que le modèle **regarde la main** (confiance + anti-biais). |
| Overfitting, comment vous le contrôlez ? | EarlyStopping (`restore_best_weights`), Dropout, augmentation, ReduceLROnPlateau, split validation 15 %. |

---

## 6. Comment faire tourner le code (démo)

```bash
# CNN dactylologie — entraînement du bake-off complet
python -m recognition.src.train --model all
python -m recognition.src.evaluate            # métriques + scorecard
python -m recognition.src.xai_gradcam --model recognition/models/cnn_scratch.keras --n 12

# Mots (séquence)
python -m recognition.src.train_word --model all --epochs 60 --augment 8
python -m recognition.src.evaluate_word

# Sentiment
python -m sentiment.src.train_scratch
python -m sentiment.src.evaluate
python -m sentiment.src.eval_realistic

# Démos desktop (webcam live)
python desktop/app.py             # fingerspelling
python desktop/synthesis_app.py   # texte → signes

# Tests
pytest tests/test_smoke.py
```

**Environnement** : `.venv` local, **TensorFlow 2.17.1 / Keras 3.15.0** (pins dans
`requirements.txt`). Le sentiment pré-entraîné tourne en **PyTorch CPU** (`sentiment/requirements.txt`).
