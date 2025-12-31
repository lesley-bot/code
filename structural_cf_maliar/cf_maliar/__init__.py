"""cf_maliar: Deep-learning solvers for structural corporate finance models."""

from .config import BaselineParams, RiskyDebtParams, TrainConfig, EvalConfig
from .solver import train_baseline_policy, train_risky_debt_policy, solve_vfi_baseline
from .simulate import simulate_baseline_panel, simulate_risky_debt_panel
from .metrics import baseline_effectiveness, risky_debt_effectiveness
