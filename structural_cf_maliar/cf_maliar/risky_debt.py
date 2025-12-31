"""Simplified risky-debt ingredients used for Part 1(d).

This module is a lightweight demonstrator of how the baseline approach can be adapted
when default risk creates nonlinearities. It is not a full replication of Section 3.6.
"""

from __future__ import annotations
import tensorflow as tf

from .config import RiskyDebtParams
from .model import gross_risky_rate, profit

DTYPE = tf.float32


def default_probability_smooth(
    k_next: tf.Tensor,
    b_next: tf.Tensor,
    z_next_mc: tf.Tensor,
    params: RiskyDebtParams,
    temp: float = 0.05,
) -> tf.Tensor:
    """Smooth approximation to Pr(default | k', b', z').

    Default is approximated as equity < 0, using cashflow proxy pi(k',z') and promised repayment b'.
    The hard indicator 1{b' > pi(k',z')} is replaced by sigmoid((b' - pi)/temp).
    """
    k_next = tf.reshape(tf.cast(k_next, DTYPE), (-1, 1, 1))
    b_next = tf.reshape(tf.cast(b_next, DTYPE), (-1, 1, 1))
    z_next_mc = tf.cast(z_next_mc, DTYPE)
    cash = profit(k_next, z_next_mc, params.theta)
    score = (b_next - cash) / tf.cast(temp, DTYPE)
    return tf.sigmoid(score)


def risky_rate_from_default_prob(pdef: tf.Tensor, params: RiskyDebtParams) -> tf.Tensor:
    """Return gross promised rate implied by risk-neutral pricing with recovery."""
    p = tf.reduce_mean(tf.cast(pdef, DTYPE), axis=1, keepdims=True)
    return gross_risky_rate(p, params)


def risky_debt_penalty(pdef: tf.Tensor, default_weight: float = 0.5) -> tf.Tensor:
    """Penalty used in the demonstrator objective to discourage extreme default probabilities."""
    p = tf.reduce_mean(tf.cast(pdef, DTYPE), axis=1)
    return tf.cast(default_weight, DTYPE) * tf.reduce_mean(tf.square(p))
