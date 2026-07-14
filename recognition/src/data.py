"""tf.data loaders + augmentation for the ASL Alphabet dataset.

The SAME split, batch size, and augmentation are used for every candidate in the
bake-off (imported from config), so no model gets an unfair data advantage.

We load directly from the class-subfolder layout produced by download_asl_alphabet.py:

    data/asl_alphabet_train/<CLASS>/<image>.jpg

and derive train/val from it with a fixed seed. A held-out test set is built from the
same tree (Kaggle's official test set is tiny — 1 image/class — so we carve our own).
"""
from __future__ import annotations

import tensorflow as tf

from .config import (
    ASL_TRAIN_DIR,
    BATCH_SIZE,
    IMG_SIZE,
    SEED,
    VAL_SPLIT,
)

AUTOTUNE = tf.data.AUTOTUNE


def _augment_layer() -> tf.keras.Sequential:
    """Shared augmentation. NOTE: no horizontal flip — flipping changes some ASL
    letters (e.g. it can turn a valid handshape into a different/invalid one)."""
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.08),          # ~ +/-29 deg
            tf.keras.layers.RandomZoom(0.10),
            tf.keras.layers.RandomTranslation(0.08, 0.08),
            # value_range=(0,1) because our images are normalised to [0,1] before augment
            tf.keras.layers.RandomBrightness(0.15, value_range=(0.0, 1.0)),
            tf.keras.layers.RandomContrast(0.15),          # helps across skin tones
        ],
        name="augment",
    )


def _normalise(image, label):
    # directory loader yields uint8 [0,255]; models expect [0,1] (see preprocess.py)
    return tf.cast(image, tf.float32) / 255.0, label


def make_datasets(
    img_size: int = IMG_SIZE,
    batch_size: int = BATCH_SIZE,
    val_split: float = VAL_SPLIT,
    test_split: float = 0.10,
    data_dir=ASL_TRAIN_DIR,
):
    """Return (train_ds, val_ds, test_ds, class_names).

    Split logic: image_dataset_from_directory gives us train/val by fraction with a
    fixed seed. We then peel a test set off the *validation* stream so test images are
    never seen in training. All three are deterministic given SEED.

    data_dir overrides the class-subfolder root (default ASL_TRAIN_DIR) — handy for a
    quick subset proof-run locally, or a different path on Colab.
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split + test_split,
        subset="training",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )
    holdout_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split + test_split,
        subset="validation",
        seed=SEED,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        label_mode="int",
    )
    class_names = train_ds.class_names

    # Split holdout -> val + test (roughly val_split : test_split)
    holdout_batches = holdout_ds.cardinality().numpy()
    val_batches = int(holdout_batches * (val_split / (val_split + test_split)))
    val_ds = holdout_ds.take(val_batches)
    test_ds = holdout_ds.skip(val_batches)

    augment = _augment_layer()
    # Cache the *normalised* images, then augment AFTER the cache so the random
    # augmentation is resampled every epoch. (Augmenting before .cache() would
    # freeze one augmented version per image for the whole run.)
    train_ds = (
        train_ds.map(_normalise, num_parallel_calls=AUTOTUNE)
        .cache()
        .shuffle(1000, seed=SEED)
        .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )
    val_ds = val_ds.map(_normalise, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)
    test_ds = test_ds.map(_normalise, num_parallel_calls=AUTOTUNE).cache().prefetch(AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names
