"""EfficientNetB0 transfer-learning candidate.

Strong accuracy-per-FLOP; a good middle ground on the size/latency/accuracy trade-off.

Note: keras EfficientNet models expect raw [0,255] inputs — preprocessing is built into
the model as a normalization layer. So we rescale [0,1] -> [0,255] and pass straight in.
"""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0


def build_efficientnetb0(num_classes: int, input_shape) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape, name="input")   # (224,224,3) in [0,1], same contract as every candidate
    x = layers.Rescaling(255.0)(inputs)            # [0,1] -> [0,255]; EfficientNet does its own internal normalisation
    # Load the ImageNet-pretrained convolutional base WITHOUT its 1000-class top.
    # input_tensor=x inlines the backbone into THIS graph (so Grad-CAM can reach its conv layers).
    base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=x)
    base.trainable = False                          # freeze the backbone for phase-1 (train only the new head)
    y = layers.GlobalAveragePooling2D(name="gap")(base.output)  # pool the backbone's feature maps -> one vector
    y = layers.Dropout(0.3)(y)                      # regularise the head
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(y)  # our 29-class head
    model = models.Model(inputs, outputs, name="efficientnetb0")
    # Record backbone layer NAMES for fine-tuning; storing `base` itself would
    # double-track its inlined layers and corrupt .keras save/load. See mobilenetv2.py.
    model.base_layer_names = [layer.name for layer in base.layers]  # names only -> train.py re-resolves them in phase 2
    return model
