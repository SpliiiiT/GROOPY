"""Grad-CAM explainability (CRISP-DM: Evaluation — trust & bias check).

Produces heatmaps over sample images to verify the model attends to the HAND, not the
background or skin tone. Also drives the manual 'robustness' score in the scorecard.

Usage:
  python -m recognition.src.xai_gradcam --model recognition/models/cnn_scratch.keras --n 12
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf

from .config import IMG_SIZE, RESULTS_DIR
from .data import make_datasets


def _last_conv_layer_name(model: tf.keras.Model) -> str:
    """Find the last 4D (conv) output layer, descending into a nested base model."""
    def _is_4d(layer) -> bool:
        # Keras 3 exposes layer.output.shape; Keras 2 used layer.output_shape.
        for get in (lambda l: l.output.shape, lambda l: l.output_shape):   # try both APIs
            try:
                return len(get(layer)) == 4   # a conv feature map is 4-D: (batch, H, W, channels)
            except Exception:
                continue
        return False

    def search(m):
        for layer in reversed(m.layers):      # scan from the output backwards -> the LAST conv layer
            if isinstance(layer, tf.keras.Model):
                found = search(layer)          # recurse into a nested backbone model
                if found:
                    return found
            if _is_4d(layer):
                return layer.name              # first 4-D layer from the end = target
        return None

    name = search(model)
    if name is None:
        raise RuntimeError("Could not locate a conv layer for Grad-CAM.")
    return name


def make_gradcam_heatmap(img_array, model, last_conv_name) -> np.ndarray:
    grad_model = tf.keras.models.Model(          # a model that ALSO outputs the last conv activations
        model.inputs,
        [model.get_layer(last_conv_name).output, model.output],
    )
    with tf.GradientTape() as tape:              # record ops so we can differentiate
        conv_out, preds = grad_model(img_array)  # forward pass -> (activations, class probs)
        class_idx = tf.argmax(preds[0])          # the predicted class
        class_channel = preds[:, class_idx]      # its score (what we explain)

    grads = tape.gradient(class_channel, conv_out)   # d(class score)/d(activations) — importance signal
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))   # average over H,W -> one weight per feature map
    conv_out = conv_out[0]                            # drop the batch dim
    heatmap = conv_out @ pooled[..., tf.newaxis]      # weighted sum of feature maps
    heatmap = tf.squeeze(heatmap)                     # -> 2-D map
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)   # ReLU + normalise to [0,1]
    return heatmap.numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Grad-CAM heatmap panel.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--n", type=int, default=12)   # how many sample images to show
    args = parser.parse_args()

    import matplotlib

    matplotlib.use("Agg")                        # headless backend (no display needed)
    import matplotlib.pyplot as plt

    model = tf.keras.models.load_model(args.model)     # load the trained model
    last_conv = _last_conv_layer_name(model)           # find its Grad-CAM target layer
    print("Grad-CAM target layer:", last_conv)

    _, _, test_ds, class_names = make_datasets()       # grab test images to explain
    images, labels = next(iter(test_ds))               # one batch
    images = images[: args.n]
    labels = labels[: args.n]

    cols = 4                                            # grid layout
    rows = (args.n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for i in range(args.n):                            # one heatmap per sample image
        img = images[i].numpy()
        arr = np.expand_dims(img, 0)                   # add batch dim
        heat = make_gradcam_heatmap(arr, model, last_conv)          # compute the heatmap
        heat = tf.image.resize(heat[..., None], (IMG_SIZE, IMG_SIZE)).numpy().squeeze()  # upscale to image size
        pred = class_names[int(np.argmax(model.predict(arr, verbose=0)[0]))]             # predicted label
        axes[i].imshow(img)                            # the original image
        axes[i].imshow(heat, cmap="jet", alpha=0.45)   # the heatmap overlaid (red = most important)
        axes[i].set_title(f"true={class_names[int(labels[i])]}  pred={pred}", fontsize=8)
        axes[i].axis("off")
    for j in range(args.n, len(axes)):
        axes[j].axis("off")                            # hide any unused grid cells

    out = RESULTS_DIR / f"gradcam_{Path(args.model).stem}.png"
    fig.tight_layout()
    fig.savefig(out, dpi=120)                          # save the panel
    print("Saved", out)
    print(
        "Review: heatmaps should concentrate on the hand. If they light up the "
        "background/arm, lower this model's 'robustness' score in the bake-off."
    )


if __name__ == "__main__":
    main()
