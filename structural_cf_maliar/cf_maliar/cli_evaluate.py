"""Command-line entry point for evaluating trained policies."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import tensorflow as tf

from .config import BaselineParams, RiskyDebtParams, EvalConfig
from .metrics import baseline_effectiveness, risky_debt_effectiveness
from .model import k_steady_state
from .policy import BaselinePolicyNet, PolicyBounds, RiskyDebtPolicyNet, RiskyPolicyBounds


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate trained policies and write JSON results.")
    p.add_argument("--model", choices=["baseline", "risky"], default="baseline")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--out", type=str, default="results.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(seed=args.seed)

    if args.model == "baseline":
        params = BaselineParams()
        k_ss = k_steady_state(params)
        policy = BaselinePolicyNet(bounds=PolicyBounds(1e-4, 3.0 * k_ss))
        weights = args.weights or "artifacts/baseline_policy.weights.h5"
        policy.load_weights(weights)
        res = baseline_effectiveness(policy, params, cfg, include_vfi_rmse=True)
    else:
        params = RiskyDebtParams()
        k_ss = k_steady_state(params)
        policy = RiskyDebtPolicyNet(bounds=RiskyPolicyBounds(1e-4, 3.0 * k_ss, 0.0, params.b_max))
        weights = args.weights or "artifacts/risky_policy.weights.h5"
        policy.load_weights(weights)
        res = risky_debt_effectiveness(policy, params, cfg)

    res["tensorflow_version"] = tf.__version__
    out_path.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
