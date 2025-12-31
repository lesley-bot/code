"""Solvers for structural models using a Maliar-style Euler-residual method."""

from __future__ import annotations
from typing import Dict, Tuple

import tensorflow as tf

from .config import BaselineParams, RiskyDebtParams, TrainConfig
from .model import (
    adj_cost_I,
    adj_cost_k,
    discount_factor,
    k_steady_state,
    profit_k,
    shock_next_z,
    tauchen_logz_grid,
)
from .policy import BaselinePolicyNet, PolicyBounds, RiskyDebtPolicyNet, RiskyPolicyBounds
from .risky_debt import default_probability_smooth, risky_debt_penalty
from .utils import make_tf_rng

DTYPE = tf.float32


@tf.function
def euler_residual_baseline(
    policy: BaselinePolicyNet,
    k: tf.Tensor,
    z: tf.Tensor,
    params: BaselineParams,
    rng_state: tf.Tensor,
    n_mc: int,
) -> tf.Tensor:
    """Compute Euler residuals for the baseline model at given states."""
    rng = tf.random.Generator.from_state(rng_state)
    k = tf.reshape(tf.cast(k, DTYPE), (-1, 1))
    z = tf.reshape(tf.cast(z, DTYPE), (-1, 1))

    beta = discount_factor(params.r)
    delta = tf.cast(params.delta, DTYPE)

    k1 = policy(k, z)
    I0 = k1 - (1.0 - delta) * k
    mc0 = 1.0 + adj_cost_I(I0, k, params.psi0)

    z1 = shock_next_z(rng, z, params, n_mc=n_mc)  # (B, mc, 1)
    k1_rep = tf.repeat(k1, repeats=n_mc, axis=0)
    z1_flat = tf.reshape(z1, (-1, 1))

    k2 = policy(k1_rep, z1_flat)
    I1 = k2 - (1.0 - delta) * k1_rep

    term = (
        profit_k(k1_rep, z1_flat, params.theta)
        + (1.0 - delta) * (1.0 + adj_cost_I(I1, k1_rep, params.psi0))
        - adj_cost_k(I1, k1_rep, params.psi0)
    )
    term = tf.reshape(term, (-1, n_mc, 1))
    rhs = beta * tf.reduce_mean(term, axis=1)

    return mc0 - rhs


def train_baseline_policy(
    params: BaselineParams,
    cfg: TrainConfig,
    hidden: Tuple[int, ...] = (128, 128, 128),
    act: str = "tanh",
) -> BaselinePolicyNet:
    """Train a baseline policy network by minimizing Euler residual MSE."""
    k_ss = k_steady_state(params)
    bounds = PolicyBounds(k_min=1e-4, k_max=3.0 * k_ss)
    policy = BaselinePolicyNet(bounds=bounds, hidden=hidden, act=act)

    rng = make_tf_rng(cfg.seed)
    opt = tf.keras.optimizers.Adam(cfg.lr)

    for step in range(1, cfg.steps + 1):
        logk = rng.uniform(
            (cfg.batch_size, 1),
            minval=tf.math.log(0.3 * k_ss),
            maxval=tf.math.log(3.0 * k_ss),
            dtype=DTYPE,
        )
        logz = rng.uniform(
            (cfg.batch_size, 1),
            minval=tf.math.log(0.5),
            maxval=tf.math.log(1.5),
            dtype=DTYPE,
        )
        k = tf.exp(logk)
        z = tf.exp(logz)
        rng_state = rng.state

        with tf.GradientTape() as tape:
            resid = euler_residual_baseline(policy, k, z, params, rng_state, cfg.n_mc)
            loss = tf.reduce_mean(tf.square(resid))
            loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(1e6, DTYPE))

        grads = tape.gradient(loss, policy.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, policy.trainable_variables))

        if step % 250 == 0 or step == 1:
            tf.print("[baseline] step", step, "euler_mse", loss)

    return policy


def solve_vfi_baseline(
    params: BaselineParams,
    n_k: int = 60,
    n_z: int = 7,
    max_iter: int = 700,
    tol: float = 1e-5,
) -> Dict[str, tf.Tensor]:
    """Value function iteration (VFI) benchmark on a small grid (evaluation only)."""
    k_ss = k_steady_state(params)
    k_grid = tf.linspace(0.3 * k_ss, 3.0 * k_ss, n_k)
    logz_grid, P = tauchen_logz_grid(params, n=n_z)
    z_grid = tf.exp(logz_grid)

    beta = discount_factor(params.r)
    delta = tf.cast(params.delta, DTYPE)

    V = tf.zeros((n_k, n_z), dtype=DTYPE)
    kp_grid = k_grid

    for _ in range(max_iter):
        V_new = tf.TensorArray(DTYPE, size=n_z)
        policy_kp = tf.TensorArray(DTYPE, size=n_z)

        for j in range(n_z):
            z = z_grid[j]
            k = tf.reshape(k_grid, (-1, 1))
            kp = tf.reshape(kp_grid, (1, -1))
            I = kp - (1.0 - delta) * k
            payoff = z * tf.pow(k, params.theta) - I - (params.psi0 * tf.square(I) / (2.0 * (k + 1e-12)))
            cont = tf.reduce_sum(V * tf.reshape(P[j, :], (1, n_z)), axis=1)
            cont2 = tf.reshape(cont, (1, -1))
            total = payoff + beta * cont2
            best_idx = tf.argmax(total, axis=1, output_type=tf.int32)
            best_val = tf.reduce_max(total, axis=1)
            best_kp = tf.gather(kp_grid, best_idx)

            V_new = V_new.write(j, best_val)
            policy_kp = policy_kp.write(j, best_kp)

        Vn = tf.transpose(V_new.stack())
        diff = tf.reduce_max(tf.abs(Vn - V))
        V = Vn
        if diff < tol:
            break

    return {
        "k_grid": k_grid,
        "z_grid": z_grid,
        "V": V,
        "kp_policy": tf.transpose(policy_kp.stack()),
    }


@tf.function
def risky_debt_training_loss(
    policy: RiskyDebtPolicyNet,
    k: tf.Tensor,
    b: tf.Tensor,
    z: tf.Tensor,
    params: RiskyDebtParams,
    rng_state: tf.Tensor,
    n_mc: int,
    default_temp: float,
    default_weight: float,
) -> tf.Tensor:
    """Training loss for the risky-debt demonstrator (Euler anchor + default penalty)."""
    rng = tf.random.Generator.from_state(rng_state)
    k = tf.reshape(tf.cast(k, DTYPE), (-1, 1))
    b = tf.reshape(tf.cast(b, DTYPE), (-1, 1))
    z = tf.reshape(tf.cast(z, DTYPE), (-1, 1))

    beta = discount_factor(params.r)
    delta = tf.cast(params.delta, DTYPE)

    k1, b1 = policy(k, b, z)
    I0 = k1 - (1.0 - delta) * k
    mc0 = 1.0 + adj_cost_I(I0, k, params.psi0)

    z1 = shock_next_z(rng, z, params, n_mc=n_mc)  # (B, mc, 1)
    k1_rep = tf.repeat(k1, repeats=n_mc, axis=0)
    z1_flat = tf.reshape(z1, (-1, 1))
    b1_rep = tf.repeat(b1, repeats=n_mc, axis=0)

    k2, _ = policy(k1_rep, b1_rep, z1_flat)
    I1 = k2 - (1.0 - delta) * k1_rep

    term = (
        profit_k(k1_rep, z1_flat, params.theta)
        + (1.0 - delta) * (1.0 + adj_cost_I(I1, k1_rep, params.psi0))
        - adj_cost_k(I1, k1_rep, params.psi0)
    )
    term = tf.reshape(term, (-1, n_mc, 1))
    rhs = beta * tf.reduce_mean(term, axis=1)

    resid = mc0 - rhs
    euler_mse = tf.reduce_mean(tf.square(resid))

    pdef = default_probability_smooth(k1, b1, z1, params, temp=default_temp)
    penalty = risky_debt_penalty(pdef, default_weight=default_weight)

    return euler_mse + penalty


def train_risky_debt_policy(
    params: RiskyDebtParams,
    cfg: TrainConfig,
    hidden: Tuple[int, ...] = (192, 192, 192),
    act: str = "tanh",
    default_temp: float = 0.05,
    default_weight: float = 0.5,
) -> RiskyDebtPolicyNet:
    """Train a risky-debt demonstrator policy network."""
    k_ss = k_steady_state(params)
    bounds = RiskyPolicyBounds(k_min=1e-4, k_max=3.0 * k_ss, b_min=0.0, b_max=params.b_max)
    policy = RiskyDebtPolicyNet(bounds=bounds, hidden=hidden, act=act)

    rng = make_tf_rng(cfg.seed)
    opt = tf.keras.optimizers.Adam(cfg.lr)

    for step in range(1, cfg.steps + 1):
        logk = rng.uniform(
            (cfg.batch_size, 1),
            minval=tf.math.log(0.3 * k_ss),
            maxval=tf.math.log(3.0 * k_ss),
            dtype=DTYPE,
        )
        logz = rng.uniform(
            (cfg.batch_size, 1),
            minval=tf.math.log(0.5),
            maxval=tf.math.log(1.5),
            dtype=DTYPE,
        )
        b = rng.uniform((cfg.batch_size, 1), minval=0.0, maxval=float(params.b_max), dtype=DTYPE)
        k = tf.exp(logk)
        z = tf.exp(logz)
        rng_state = rng.state

        with tf.GradientTape() as tape:
            loss = risky_debt_training_loss(policy, k, b, z, params, rng_state, cfg.n_mc, default_temp, default_weight)
            loss = tf.where(tf.math.is_finite(loss), loss, tf.constant(1e6, DTYPE))

        grads = tape.gradient(loss, policy.trainable_variables)
        grads = [tf.clip_by_norm(g, 1.0) if g is not None else None for g in grads]
        opt.apply_gradients(zip(grads, policy.trainable_variables))

        if step % 250 == 0 or step == 1:
            tf.print("[risky] step", step, "loss", loss)

    return policy
