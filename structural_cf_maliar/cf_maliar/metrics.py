"""Effectiveness metrics for solver evaluation on synthetic data."""

from __future__ import annotations
from typing import Any, Dict

import numpy as np
import tensorflow as tf

from .config import BaselineParams, EvalConfig, RiskyDebtParams
from .model import k_steady_state
from .simulate import simulate_baseline_panel, simulate_risky_debt_panel
from .solver import euler_residual_baseline, solve_vfi_baseline
from .utils import make_tf_rng, params_to_dict

DTYPE = tf.float32


def baseline_effectiveness(policy, params: BaselineParams, cfg: EvalConfig, include_vfi_rmse: bool = True) -> Dict[str, Any]:
    """Compute baseline effectiveness metrics."""
    rng = make_tf_rng(cfg.seed)
    k_ss = k_steady_state(params)

    logk = rng.uniform((cfg.n_states, 1), minval=tf.math.log(0.3 * k_ss), maxval=tf.math.log(3.0 * k_ss), dtype=DTYPE)
    logz = rng.uniform((cfg.n_states, 1), minval=tf.math.log(0.5), maxval=tf.math.log(1.5), dtype=DTYPE)
    k = tf.exp(logk)
    z = tf.exp(logz)

    resid = euler_residual_baseline(policy, k, z, params, rng_state=rng.state, n_mc=32)
    euler_rms = float(tf.sqrt(tf.reduce_mean(tf.square(resid))).numpy())

    panel = simulate_baseline_panel(policy, params, T=cfg.panel_T, N=cfg.panel_N, burn=cfg.burn, seed=cfg.seed)
    iok = panel["i_over_k"].numpy().reshape(-1)
    logk_p = panel["logk"].numpy().reshape(-1)
    logz_p = panel["logz"].numpy().reshape(-1)

    z_lag, z_now = logz_p[:-1], logz_p[1:]
    rho_hat = float(np.dot(z_lag, z_now) / (np.dot(z_lag, z_lag) + 1e-12))
    eps = z_now - rho_hat * z_lag
    sig_hat = float(np.sqrt(np.mean(eps**2)))

    out: Dict[str, Any] = {
        "params": params_to_dict(params),
        "euler_rms": euler_rms,
        "panel_moments": {
            "mean_i_over_k": float(np.mean(iok)),
            "var_i_over_k": float(np.var(iok)),
            "mean_logk": float(np.mean(logk_p)),
            "var_logk": float(np.var(logk_p)),
            "rho_logz": rho_hat,
            "sigma_eps_logz": sig_hat,
        },
    }

    if include_vfi_rmse:
        vfi = solve_vfi_baseline(params, n_k=50, n_z=7, max_iter=400, tol=1e-5)
        k_grid = vfi["k_grid"].numpy()
        z_grid = vfi["z_grid"].numpy()
        kp_vfi = vfi["kp_policy"].numpy()

        KK, ZZ = np.meshgrid(k_grid, z_grid, indexing="ij")
        kp_nn = policy(
            tf.constant(KK.reshape(-1, 1), DTYPE),
            tf.constant(ZZ.reshape(-1, 1), DTYPE),
        ).numpy().reshape(KK.shape)

        out["policy_rmse_vs_vfi"] = float(np.sqrt(np.mean((kp_nn - kp_vfi) ** 2)))

    return out


def risky_debt_effectiveness(policy, params: RiskyDebtParams, cfg: EvalConfig) -> Dict[str, Any]:
    """Compute effectiveness metrics for the risky-debt demonstrator."""
    panel = simulate_risky_debt_panel(policy, params, T=cfg.panel_T, N=cfg.panel_N, burn=cfg.burn, seed=cfg.seed)
    pdef = panel["pdef"].numpy().reshape(-1)
    rtil = panel["r_tilde"].numpy().reshape(-1)
    spread = rtil - (1.0 + params.r)

    return {
        "params": params_to_dict(params),
        "mean_default_prob": float(np.mean(pdef)),
        "p95_default_prob": float(np.quantile(pdef, 0.95)),
        "mean_spread": float(np.mean(spread)),
        "p95_spread": float(np.quantile(spread, 0.95)),
    }
