"""Synthetic data generation for baseline and risky-debt variants."""

from __future__ import annotations
from typing import Dict, Optional

import tensorflow as tf

from .config import BaselineParams, RiskyDebtParams
from .model import k_steady_state
from .policy import BaselinePolicyNet, RiskyDebtPolicyNet
from .risky_debt import default_probability_smooth, risky_rate_from_default_prob
from .utils import make_tf_rng, truncated_normal

DTYPE = tf.float32


def simulate_baseline_panel(
    policy: BaselinePolicyNet,
    params: BaselineParams,
    T: int,
    N: int,
    burn: int = 50,
    seed: int = 123,
    k0: Optional[float] = None,
    z0: float = 1.0,
) -> Dict[str, tf.Tensor]:
    """Simulate a panel of (k_t, z_t, k_{t+1}) under a trained policy."""
    if k0 is None:
        k0 = k_steady_state(params)

    rng = make_tf_rng(seed)
    k = tf.fill((N, 1), tf.cast(k0, DTYPE))
    z = tf.fill((N, 1), tf.cast(z0, DTYPE))

    ks, zs, kps, iok = [], [], [], []
    delta = tf.cast(params.delta, DTYPE)

    for t in range(T):
        kp = policy(k, z)
        I = kp - (1.0 - delta) * k

        if t >= burn:
            ks.append(k)
            zs.append(z)
            kps.append(kp)
            iok.append(I / tf.maximum(k, 1e-8))

        sigma = tf.cast(params.sigma_eps, DTYPE)
        lo = -float(params.trunc_m) * float(params.sigma_eps)
        hi = +float(params.trunc_m) * float(params.sigma_eps)
        eps = truncated_normal(rng, (N, 1), 0.0, sigma, lo, hi, dtype=DTYPE)

        z = tf.exp(tf.cast(params.rho, DTYPE) * tf.math.log(tf.maximum(z, 1e-12)) + eps)
        k = kp

    out = {"k": tf.concat(ks, 0), "z": tf.concat(zs, 0), "kp": tf.concat(kps, 0), "i_over_k": tf.concat(iok, 0)}
    out["logk"] = tf.math.log(tf.maximum(out["k"], 1e-12))
    out["logz"] = tf.math.log(tf.maximum(out["z"], 1e-12))
    return out


def simulate_risky_debt_panel(
    policy: RiskyDebtPolicyNet,
    params: RiskyDebtParams,
    T: int,
    N: int,
    burn: int = 50,
    seed: int = 123,
    k0: Optional[float] = None,
    b0: float = 0.5,
    z0: float = 1.0,
    default_temp: float = 0.05,
    n_mc_rate: int = 16,
) -> Dict[str, tf.Tensor]:
    """Simulate a panel for the risky-debt demonstrator, including default probs and risky rates."""
    if k0 is None:
        k0 = k_steady_state(params)

    rng = make_tf_rng(seed)
    k = tf.fill((N, 1), tf.cast(k0, DTYPE))
    b = tf.fill((N, 1), tf.cast(b0, DTYPE))
    z = tf.fill((N, 1), tf.cast(z0, DTYPE))

    ks, bs, zs, kps, bps, pdefs, rtils = [], [], [], [], [], [], []

    for t in range(T):
        kp, bp = policy(k, b, z)

        sigma = tf.cast(params.sigma_eps, DTYPE)
        lo = -float(params.trunc_m) * float(params.sigma_eps)
        hi = +float(params.trunc_m) * float(params.sigma_eps)

        eps_mc = truncated_normal(rng, (N, n_mc_rate, 1), 0.0, sigma, lo, hi, dtype=DTYPE)
        z_mc = tf.exp(tf.cast(params.rho, DTYPE) * tf.math.log(tf.maximum(tf.expand_dims(z, 1), 1e-12)) + eps_mc)

        pdef = default_probability_smooth(kp, bp, z_mc, params, temp=default_temp)
        rtil = risky_rate_from_default_prob(pdef, params)

        if t >= burn:
            ks.append(k)
            bs.append(b)
            zs.append(z)
            kps.append(kp)
            bps.append(bp)
            pdefs.append(tf.reduce_mean(pdef, axis=1))  # (N,1)
            rtils.append(rtil)                          # (N,1)

        eps = truncated_normal(rng, (N, 1), 0.0, sigma, lo, hi, dtype=DTYPE)
        z = tf.exp(tf.cast(params.rho, DTYPE) * tf.math.log(tf.maximum(z, 1e-12)) + eps)
        k = kp
        b = bp

    return {
        "k": tf.concat(ks, 0),
        "b": tf.concat(bs, 0),
        "z": tf.concat(zs, 0),
        "kp": tf.concat(kps, 0),
        "bp": tf.concat(bps, 0),
        "pdef": tf.concat(pdefs, 0),
        "r_tilde": tf.concat(rtils, 0),
    }
