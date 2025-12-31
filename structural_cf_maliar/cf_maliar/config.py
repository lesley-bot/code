"""Configuration and parameter containers.

All parameters are kept in dataclasses to improve readability, testability, and reproducibility.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class BaselineParams:
    """Parameters for the baseline investment model (Strebulaev 3.1-style).

    Attributes
    ----------
    r : float
        Risk-free interest rate.
    delta : float
        Depreciation rate.
    theta : float
        Curvature of production: pi(k,z) = z k^theta.
    psi0 : float
        Adjustment cost scale: psi(I,k) = psi0 * I^2 / (2k).
    rho : float
        Persistence of log productivity (AR(1)).
    sigma_eps : float
        Innovation volatility of log productivity.
    trunc_m : float
        Truncation multiplier for innovations: eps in [-m*sigma, m*sigma].
    """

    r: float = 0.04
    delta: float = 0.15
    theta: float = 0.70
    psi0: float = 0.01
    rho: float = 0.70
    sigma_eps: float = 0.15
    trunc_m: float = 3.0


@dataclass(frozen=True)
class RiskyDebtParams(BaselineParams):
    """Parameters for a simplified risky-debt demonstrator.

    The extension adds one-period debt b and a default probability object that maps (k', b', z')
    into a smooth default probability. A risky gross interest rate is implied by risk-neutral
    pricing with recovery.

    Attributes
    ----------
    lambda_default : float
        Fraction of firm value lost in default (recovery = 1 - lambda_default).
    b_max : float
        Borrowing limit used for bounded policy output and stable training.
    """

    lambda_default: float = 0.40
    b_max: float = 2.0


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters shared by solvers."""

    steps: int = 2500
    batch_size: int = 4096
    n_mc: int = 16
    lr: float = 3e-4
    seed: int = 123


@dataclass(frozen=True)
class EvalConfig:
    """Evaluation configuration."""

    n_states: int = 50000
    panel_T: int = 300
    panel_N: int = 2000
    burn: int = 50
    seed: int = 123
