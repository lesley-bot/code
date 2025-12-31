# Structural Corporate Finance with Maliar-Style Deep Learning (TensorFlow + TFP)

This repository implements Part 1 of the assignment:

1. Implement the basic structural investment model (Strebulaev, Section 3.1) with an AR(1) shock.
2. Solve it using a Maliar-style deep learning method that minimizes Euler-equation residuals in TensorFlow.
3. Generate synthetic data and report effectiveness metrics (Euler RMS, synthetic moments, optional VFI policy RMSE).
4. Discuss and demonstrate how the same approach can be adapted to a more sophisticated setting (Section 3.6-style risky debt)
   via a lightweight demonstrator (joint policy and default-probability/risky-rate objects).

Mandatory tools are used:
- Python 3
- TensorFlow
- TensorFlow Probability (TFP)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train (baseline)

```bash
python -m cf_maliar.cli_train --model baseline --steps 2000 --seed 123 --outdir artifacts
```

## Evaluate (baseline)

```bash
python -m cf_maliar.cli_evaluate --model baseline --seed 123 --out results_baseline.json
```

## Train (risky debt demonstrator)

```bash
python -m cf_maliar.cli_train --model risky --steps 2000 --seed 123 --outdir artifacts
```

## Evaluate (risky debt demonstrator)

```bash
python -m cf_maliar.cli_evaluate --model risky --seed 123 --out results_risky.json
```

## Run tests (unit + integration)

```bash
pytest
```

## Notes on the risky-debt demonstrator

The risky-debt code is intentionally minimal: it is designed to show how the Maliar-style approach can be extended when default
creates nonlinearities. A fully faithful Section 3.6 implementation would replace the demonstrator penalty with the exact pricing
and optimality conditions of the risky-debt model (more states, additional FOCs, careful treatment of non-smooth default sets).
