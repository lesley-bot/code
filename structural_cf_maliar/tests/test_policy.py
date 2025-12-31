import tensorflow as tf

from cf_maliar.config import BaselineParams, RiskyDebtParams
from cf_maliar.model import k_steady_state
from cf_maliar.policy import BaselinePolicyNet, PolicyBounds, RiskyDebtPolicyNet, RiskyPolicyBounds


def test_baseline_policy_bounds():
    params = BaselineParams()
    k_ss = k_steady_state(params)
    net = BaselinePolicyNet(bounds=PolicyBounds(1e-4, 3.0 * k_ss))
    k = tf.constant([[k_ss], [0.5 * k_ss]], tf.float32)
    z = tf.constant([[1.0], [1.2]], tf.float32)
    kp = net(k, z).numpy()
    assert (kp >= 1e-4).all()
    assert (kp <= 3.0 * k_ss + 1e-6).all()


def test_risky_policy_bounds():
    params = RiskyDebtParams()
    k_ss = k_steady_state(params)
    net = RiskyDebtPolicyNet(bounds=RiskyPolicyBounds(1e-4, 3.0 * k_ss, 0.0, params.b_max))
    k = tf.constant([[k_ss], [0.7 * k_ss]], tf.float32)
    b = tf.constant([[0.3], [1.0]], tf.float32)
    z = tf.constant([[1.0], [1.1]], tf.float32)
    kp, bp = net(k, b, z)
    kp = kp.numpy()
    bp = bp.numpy()
    assert (kp >= 1e-4).all()
    assert (kp <= 3.0 * k_ss + 1e-6).all()
    assert (bp >= 0.0).all()
    assert (bp <= params.b_max + 1e-6).all()
