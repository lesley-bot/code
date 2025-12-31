import numpy as np
import tensorflow as tf

from cf_maliar.config import RiskyDebtParams
from cf_maliar.model import k_steady_state
from cf_maliar.policy import RiskyDebtPolicyNet, RiskyPolicyBounds
from cf_maliar.risky_debt import default_probability_smooth, risky_rate_from_default_prob
from cf_maliar.utils import make_tf_rng, truncated_normal


def test_default_probability_and_risky_rate_are_well_behaved():
    params = RiskyDebtParams()
    k_ss = k_steady_state(params)
    policy = RiskyDebtPolicyNet(bounds=RiskyPolicyBounds(1e-4, 3.0 * k_ss, 0.0, params.b_max))

    N = 64
    k = tf.fill((N, 1), tf.constant(k_ss, tf.float32))
    b = tf.fill((N, 1), tf.constant(0.8, tf.float32))
    z = tf.fill((N, 1), tf.constant(1.0, tf.float32))

    kp, bp = policy(k, b, z)

    rng = make_tf_rng(0)
    sigma = tf.constant(params.sigma_eps, tf.float32)
    lo = -float(params.trunc_m) * float(params.sigma_eps)
    hi = +float(params.trunc_m) * float(params.sigma_eps)
    eps_mc = truncated_normal(rng, (N, 16, 1), 0.0, sigma, lo, hi, dtype=tf.float32)
    z_mc = tf.exp(tf.constant(params.rho, tf.float32) * tf.math.log(tf.maximum(tf.expand_dims(z, 1), 1e-12)) + eps_mc)

    pdef = default_probability_smooth(kp, bp, z_mc, params, temp=0.05).numpy()
    assert np.isfinite(pdef).all()
    assert (pdef >= 0.0).all() and (pdef <= 1.0).all()

    R = risky_rate_from_default_prob(tf.constant(pdef, tf.float32), params).numpy().reshape(-1)
    assert np.isfinite(R).all()
    assert (R >= 1.0 + params.r - 1e-6).all()
