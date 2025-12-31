"""Model primitives for the baseline and risky-debt variants."""

from __future__ import annotations
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from .config import BaselineParams, RiskyDebtParams
from .utils import truncated_normal

tfd = tfp.distributions
DTYPE = tf.float32


def discount_factor(r: float) -> tf.Tensor:
    """Return beta = 1 / (1 + r)."""
    return tf.constant(1.0 / (1.0 + float(r)), dtype=DTYPE)


def k_steady_state(params: BaselineParams) -> float:
    """Closed-form steady-state k under z=1 and ignoring adjustment costs."""
    theta, r, delta = params.theta, params.r, params.delta
    return float((theta / (r + delta)) ** (1.0 / (1.0 - theta)))


def profit(k: tf.Tensor, z: tf.Tensor, theta: float) -> tf.Tensor:
    """Operating profit pi(k,z) = z * k^theta."""
    return z * tf.pow(k, tf.cast(theta, DTYPE))


def profit_k(k: tf.Tensor, z: tf.Tensor, theta: float) -> tf.Tensor:
    """Marginal profit d pi / d k."""
    th = tf.cast(theta, DTYPE)
    return th * z * tf.pow(k, th - 1.0)


def adj_cost_I(I: tf.Tensor, k: tf.Tensor, psi0: float) -> tf.Tensor:
    """Partial derivative d psi / d I for psi(I,k)=psi0*I^2/(2k)."""
    return tf.cast(psi0, DTYPE) * I / (k + 1e-12)


def adj_cost_k(I: tf.Tensor, k: tf.Tensor, psi0: float) -> tf.Tensor:
    """Partial derivative d psi / d k."""
    return -tf.cast(psi0, DTYPE) * tf.square(I) / (2.0 * tf.square(k + 1e-12))


def shock_next_z(rng: tf.random.Generator, z: tf.Tensor, params: BaselineParams, n_mc: int) -> tf.Tensor:
    """Draw MC samples of next-period z' given current z using TFP truncated normals."""
    z = tf.reshape(tf.cast(z, DTYPE), (-1, 1))
    sigma = tf.cast(params.sigma_eps, DTYPE)
    lo = -float(params.trunc_m) * float(params.sigma_eps)
    hi = +float(params.trunc_m) * float(params.sigma_eps)
    eps = truncated_normal(rng, (tf.shape(z)[0], n_mc, 1), 0.0, sigma, lo, hi, dtype=DTYPE)
    logz = tf.math.log(tf.maximum(z, 1e-12))
    logz1 = tf.cast(params.rho, DTYPE) * tf.expand_dims(logz, 1) + eps
    return tf.exp(logz1)


def tauchen_logz_grid(params: BaselineParams, n: int = 7) -> Tuple[tf.Tensor, tf.Tensor]:
    """Tauchen discretization for log z using TFP Normal CDF.

    Returns
    -------
    grid : tf.Tensor
        Grid for log z (shape n).
    P : tf.Tensor
        Transition matrix for log z (shape n x n) with rows summing to 1.
    """
    if n < 3:
        raise ValueError("Tauchen grid size n must be >= 3.")

    rho = float(params.rho)
    sigma = float(params.sigma_eps)

    std = sigma / ((1.0 - rho**2) ** 0.5 + 1e-12)
    m = 3.0
    zmax = m * std
    zmin = -m * std
    grid = tf.linspace(zmin, zmax, n)
    step = (zmax - zmin) / (n - 1)

    normal = tfd.Normal(loc=0.0, scale=sigma)
    rows = []
    for i in range(n):
        mu = rho * grid[i]
        cols = []
        for j in range(n):
            if j == 0:
                p = normal.cdf(grid[0] - mu + step / 2.0)
            elif j == n - 1:
                p = 1.0 - normal.cdf(grid[-1] - mu - step / 2.0)
            else:
                hi = normal.cdf(grid[j] - mu + step / 2.0)
                lo = normal.cdf(grid[j] - mu - step / 2.0)
                p = hi - lo
            cols.append(p)
        row = tf.stack(cols)
        rows.append(row)
    P = tf.stack(rows)
    P = P / tf.reduce_sum(P, axis=1, keepdims=True)
    return grid, P


def gross_risky_rate(default_prob: tf.Tensor, params: RiskyDebtParams) -> tf.Tensor:
    """Gross risky rate implied by risk-neutral pricing with recovery.

    If lenders lose a fraction lambda_default upon default, then:
      1 = (1/(1+r)) * E[ (1 - lambda_default * 1{default}) * R_tilde ]
    Solving for R_tilde (using default probability p) gives:
      R_tilde = (1+r) / (1 - p*lambda_default)
    """
    lam = tf.cast(params.lambda_default, DTYPE)
    r = tf.cast(params.r, DTYPE)
    denom = tf.maximum(1.0 - default_prob * lam, 1e-6)
    return (1.0 + r) / denom
