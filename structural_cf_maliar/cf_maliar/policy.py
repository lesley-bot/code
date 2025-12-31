"""Policy networks with constraint-respecting parameterizations."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

import tensorflow as tf

DTYPE = tf.float32


@dataclass(frozen=True)
class PolicyBounds:
    """Bounds for baseline policy output k'."""
    k_min: float
    k_max: float


class BaselinePolicyNet(tf.keras.Model):
    """Neural policy k' = h(k,z) with bounded output in [k_min, k_max]."""

    def __init__(self, bounds: PolicyBounds, hidden: Tuple[int, ...] = (128, 128, 128), act: str = "tanh"):
        super().__init__()
        self.bounds = bounds
        self.net = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(2,))]
            + [tf.keras.layers.Dense(h, activation=act) for h in hidden]
            + [tf.keras.layers.Dense(1)]
        )

    def call(self, k: tf.Tensor, z: tf.Tensor) -> tf.Tensor:
        k = tf.reshape(tf.cast(k, DTYPE), (-1, 1))
        z = tf.reshape(tf.cast(z, DTYPE), (-1, 1))
        x = tf.concat([tf.math.log(tf.maximum(k, 1e-12)), tf.math.log(tf.maximum(z, 1e-12))], axis=1)
        raw = self.net(x)
        s = tf.sigmoid(raw)
        k_min = tf.cast(self.bounds.k_min, DTYPE)
        k_max = tf.cast(self.bounds.k_max, DTYPE)
        return k_min + (k_max - k_min) * s


@dataclass(frozen=True)
class RiskyPolicyBounds:
    """Bounds for risky-debt policy outputs."""
    k_min: float
    k_max: float
    b_min: float
    b_max: float


class RiskyDebtPolicyNet(tf.keras.Model):
    """Neural policy for (k', b') = h(k,b,z) with bounded outputs."""

    def __init__(self, bounds: RiskyPolicyBounds, hidden: Tuple[int, ...] = (192, 192, 192), act: str = "tanh"):
        super().__init__()
        self.bounds = bounds
        self.net = tf.keras.Sequential(
            [tf.keras.layers.InputLayer(input_shape=(3,))]
            + [tf.keras.layers.Dense(h, activation=act) for h in hidden]
            + [tf.keras.layers.Dense(2)]
        )

    def call(self, k: tf.Tensor, b: tf.Tensor, z: tf.Tensor):
        k = tf.reshape(tf.cast(k, DTYPE), (-1, 1))
        b = tf.reshape(tf.cast(b, DTYPE), (-1, 1))
        z = tf.reshape(tf.cast(z, DTYPE), (-1, 1))
        x = tf.concat([tf.math.log(tf.maximum(k, 1e-12)), b, tf.math.log(tf.maximum(z, 1e-12))], axis=1)
        raw = self.net(x)
        s = tf.sigmoid(raw)

        k_min = tf.cast(self.bounds.k_min, DTYPE)
        k_max = tf.cast(self.bounds.k_max, DTYPE)
        b_min = tf.cast(self.bounds.b_min, DTYPE)
        b_max = tf.cast(self.bounds.b_max, DTYPE)

        k1 = k_min + (k_max - k_min) * s[:, 0:1]
        b1 = b_min + (b_max - b_min) * s[:, 1:2]
        return k1, b1
