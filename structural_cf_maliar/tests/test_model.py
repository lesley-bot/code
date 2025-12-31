import numpy as np
import tensorflow as tf

from cf_maliar.config import BaselineParams, RiskyDebtParams
from cf_maliar.model import tauchen_logz_grid, gross_risky_rate


def test_tauchen_rows_sum_to_one():
    params = BaselineParams(rho=0.8, sigma_eps=0.1)
    _, P = tauchen_logz_grid(params, n=7)
    rows = tf.reduce_sum(P, axis=1).numpy()
    assert np.allclose(rows, 1.0, atol=1e-6)


def test_gross_risky_rate_increases_with_default_prob():
    params = RiskyDebtParams(r=0.04, lambda_default=0.4)
    p = tf.constant([[0.0], [0.2], [0.5]], tf.float32)
    R = gross_risky_rate(p, params).numpy().reshape(-1)
    assert R[0] < R[1] < R[2]
