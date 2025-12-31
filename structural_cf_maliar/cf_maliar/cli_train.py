"""Command-line entry point for training policies."""

from __future__ import annotations
import argparse
import json
from pathlib import Path

import tensorflow as tf

from .config import BaselineParams, RiskyDebtParams, TrainConfig
from .solver import train_baseline_policy, train_risky_debt_policy
from .utils import params_to_dict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Maliar-style policy networks.")
    p.add_argument("--model", choices=["baseline", "risky"], default="baseline")
    p.add_argument("--steps", type=int, default=2500)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--n_mc", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--outdir", type=str, default="artifacts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig(steps=args.steps, batch_size=args.batch_size, n_mc=args.n_mc, lr=args.lr, seed=args.seed)

    if args.model == "baseline":
        params = BaselineParams()
        policy = train_baseline_policy(params, cfg)
        weights_path = outdir / "baseline_policy.weights.h5"
    else:
        params = RiskyDebtParams()
        policy = train_risky_debt_policy(params, cfg)
        weights_path = outdir / "risky_policy.weights.h5"

    policy.save_weights(str(weights_path))

    meta = {
        "model": args.model,
        "train_config": vars(cfg),
        "params": params_to_dict(params),
        "weights": str(weights_path),
        "tensorflow_version": tf.__version__,
    }
    (outdir / f"{args.model}_train_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Saved weights to {weights_path}")


if __name__ == "__main__":
    main()
