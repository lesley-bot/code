import tensorflow as tf

from cf_maliar.config import BaselineParams, TrainConfig, EvalConfig
from cf_maliar.solver import train_baseline_policy
from cf_maliar.metrics import baseline_effectiveness


def test_end_to_end_baseline_train_and_eval_runs():
    params = BaselineParams()
    train_cfg = TrainConfig(steps=120, batch_size=512, n_mc=8, lr=5e-4, seed=123)
    policy = train_baseline_policy(params, train_cfg)

    eval_cfg = EvalConfig(n_states=2000, panel_T=120, panel_N=256, burn=20, seed=123)
    res = baseline_effectiveness(policy, params, eval_cfg, include_vfi_rmse=False)

    assert "euler_rms" in res
    assert tf.math.is_finite(tf.constant(res["euler_rms"], tf.float32))
    assert res["euler_rms"] < 5.0
    assert res["panel_moments"]["var_i_over_k"] >= 0.0
