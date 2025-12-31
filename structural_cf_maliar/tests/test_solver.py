import tensorflow as tf

from cf_maliar.config import BaselineParams
from cf_maliar.model import k_steady_state
from cf_maliar.policy import BaselinePolicyNet, PolicyBounds
from cf_maliar.solver import euler_residual_baseline
from cf_maliar.utils import make_tf_rng


def test_euler_loss_decreases_with_training_steps():
    params = BaselineParams(theta=0.7, psi0=0.01, rho=0.7, sigma_eps=0.15)
    k_ss = k_steady_state(params)
    policy = BaselinePolicyNet(bounds=PolicyBounds(1e-4, 3.0 * k_ss))
    opt = tf.keras.optimizers.Adam(1e-3)

    rng = make_tf_rng(0)
    logk = rng.uniform((1024, 1), minval=tf.math.log(0.5 * k_ss), maxval=tf.math.log(1.5 * k_ss), dtype=tf.float32)
    logz = rng.uniform((1024, 1), minval=tf.math.log(0.8), maxval=tf.math.log(1.2), dtype=tf.float32)
    k = tf.exp(logk)
    z = tf.exp(logz)

    rng_state = rng.state  # fixed MC shocks

    def loss_value():
        resid = euler_residual_baseline(policy, k, z, params, rng_state=rng_state, n_mc=16)
        return tf.reduce_mean(tf.square(resid))

    loss0 = float(loss_value().numpy())
    for _ in range(35):
        with tf.GradientTape() as tape:
            loss = loss_value()
        grads = tape.gradient(loss, policy.trainable_variables)
        opt.apply_gradients(zip(grads, policy.trainable_variables))
    loss1 = float(loss_value().numpy())

    assert loss1 < 0.9 * loss0
