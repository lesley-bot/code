"""Utility functions for reproducible randomness and JSON-friendly artifacts."""

from __future__ import annotations
from dataclasses import asdict
from typing import Any, Dict, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions


def make_tf_rng(seed: int) -> tf.random.Generator:
    """Create a TensorFlow RNG generator with a fixed seed."""
    return tf.random.Generator.from_seed(int(seed))


def truncated_normal(
    rng: tf.random.Generator,
    shape: Tuple[int, ...],
    loc: float,
    scale: tf.Tensor,
    low: float,
    high: float,
    dtype: tf.dtypes.DType = tf.float32,
) -> tf.Tensor:
    """Sample from a truncated normal distribution using TensorFlow Probability."""
    dist = tfd.TruncatedNormal(loc=tf.cast(loc, dtype), scale=tf.cast(scale, dtype), low=low, high=high)
    seed = rng.make_seeds(2)[0]
    return dist.sample(sample_shape=shape, seed=seed)


def params_to_dict(params: Any) -> Dict[str, Any]:
    """Convert a dataclass parameter object into a JSON-serializable dictionary."""
    return asdict(params)
