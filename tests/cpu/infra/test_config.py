"""
Test invariants, not implementation to flag parameter choices that silently
break PPO

Examples:
--------
- gamma > 1
- negative entropy coefficient
- invalid warmup percentages
- TBPTT chunk > rollout length
- mismatched buffer/trainer/env dimensions
"""
from lstmppo.types import Config


def test_ppo_hyperparams_valid():
    cfg = Config().ppo

    assert 0 < cfg.gamma <= 1
    assert 0 < cfg.gae_lambda <= 1
    assert cfg.initial_clip_range > 0
    assert cfg.update_epochs > 0
    assert cfg.vf_coef >= 0
    assert cfg.max_grad_norm > 0
    assert cfg.target_kl > 0
    assert cfg.early_stopping_kl_factor >= 1.0


def test_lstm_config_valid():
    cfg = Config().lstm

    assert cfg.enc_hidden_size > 0
    assert cfg.lstm_hidden_size > 0
    assert 0 <= cfg.dropconnect_p < 1
    assert cfg.lstm_ar_coef >= 0
    assert cfg.lstm_tar_coef >= 0


def test_schedule_config_valid():
    cfg = Config().sched

    assert cfg.base_lr > 0
    assert 0 <= cfg.lr_warmup_pct < 100
    assert 0 <= cfg.lr_final_pct <= 100
    assert 0 < cfg.start_entropy_coef <= 1
    assert 0 <= cfg.end_entropy_coef < cfg.start_entropy_coef


def test_trainer_config_valid():
    cfg = Config().trainer

    assert cfg.rollout_steps > 0
    assert cfg.mini_batch_envs > 0
    assert cfg.tbptt_chunk_len > 0
    assert cfg.max_sparkline_history > 0
    assert cfg.avg_ep_stat_ema_alpha > 0
    assert cfg.gate_sat_eps > 0
    assert cfg.gate_ent_eps > 0


def test_environment_config_valid():
    cfg = Config().env

    assert cfg.num_envs > 0
    assert cfg.max_env_history > 0
    assert cfg.ep_len_reward_bonus >= 0


def test_buffer_config_consistency():
    cfg = Config()
    buf = cfg.buffer_config

    assert buf.rollout_steps == cfg.trainer.rollout_steps
    assert buf.num_envs == cfg.env.num_envs
    assert buf.mini_batch_envs == cfg.trainer.mini_batch_envs
    assert buf.gamma == cfg.ppo.gamma
    assert buf.lam == cfg.ppo.gae_lambda
    assert buf.lstm_hidden_size == cfg.lstm.lstm_hidden_size


"""
Check for subtle bugs that break TBPTT or batching.
"""
def test_cross_section_invariants():

    cfg = Config()

    # TBPTT chunk must divide rollout or be smaller
    assert cfg.trainer.tbptt_chunk_len <= cfg.trainer.rollout_steps

    # minibatch envs must divide num_envs
    assert cfg.env.num_envs % cfg.trainer.mini_batch_envs == 0

    # hidden sizes must match
    assert cfg.lstm.lstm_hidden_size == cfg.buffer_config.lstm_hidden_size