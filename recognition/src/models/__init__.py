"""Bake-off model registry.

Every candidate is a function build_*(num_classes, input_shape) -> tf.keras.Model.
They all accept [0,1] float inputs of the same shape; each pre-trained backbone applies
its own preprocessing internally so the input contract is identical across candidates.
"""
from __future__ import annotations

from typing import Callable, Dict

import tensorflow as tf

from .cnn_scratch import build_cnn_scratch
from .mobilenetv2 import build_mobilenetv2
from .efficientnet import build_efficientnetb0
from .resnet50 import build_resnet50

# name -> builder. "cnn_scratch" is our required from-scratch baseline.
REGISTRY: Dict[str, Callable[..., tf.keras.Model]] = {
    "cnn_scratch": build_cnn_scratch,
    "mobilenetv2": build_mobilenetv2,
    "efficientnetb0": build_efficientnetb0,
    "resnet50": build_resnet50,
}

# Which candidates use ImageNet transfer learning (fine-tuned at a lower LR).
PRETRAINED = {"mobilenetv2", "efficientnetb0", "resnet50"}


def build(name: str, num_classes: int, input_shape) -> tf.keras.Model:
    if name not in REGISTRY:
        raise KeyError(f"Unknown model '{name}'. Options: {list(REGISTRY)}")
    return REGISTRY[name](num_classes=num_classes, input_shape=input_shape)
