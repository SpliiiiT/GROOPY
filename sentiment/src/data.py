"""IMDB train/test text + binary labels for the sentiment bake-off.

Uses tf.keras.datasets.imdb (already bundled with TensorFlow — no new download dependency)
rather than raw-text sources like tensorflow-datasets, which would add a second data-loading
stack. imdb.load_data() returns integer-encoded sequences; this decodes them back to raw text
via the dataset's own word index so both the scratch TF-IDF model and the pretrained
transformer tokenizers can consume the same plain-text inputs.

Binary only (positive/negative) — IMDB has no "neutral" reviews, which is exactly why it's a
clean, standard benchmark. The neutral label the app's Sentiment contract supports is derived
at inference time (see models.py's probability-band logic), not learned from this data.
"""
from __future__ import annotations

from typing import List, Tuple

from .config import IMDB_NUM_WORDS, TEST_SIZE


def _decode(sequences, reverse_index) -> List[str]:
    # Keras reserves indices 0-3 (padding/start/unknown/unused); real words start at 4, hence
    # the -3 offset back into the raw word_index (which itself starts at 1).
    return [
        " ".join(reverse_index.get(i - 3, "?") for i in seq if i >= 3)
        for seq in sequences
    ]


def load_imdb_text(test_size: int = TEST_SIZE) -> Tuple[List[str], List[int], List[str], List[int]]:
    """Returns (train_texts, train_labels, test_texts, test_labels). Labels: 1=positive, 0=negative."""
    from tensorflow.keras.datasets import imdb

    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=IMDB_NUM_WORDS)
    word_index = imdb.get_word_index()
    reverse_index = {v: k for k, v in word_index.items()}

    train_texts = _decode(x_train, reverse_index)
    test_texts = _decode(x_test[:test_size], reverse_index)
    test_labels = y_test[:test_size].tolist()
    return train_texts, y_train.tolist(), test_texts, test_labels
