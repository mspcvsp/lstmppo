def test_last_lstm_state_shape(deterministic_trainer):
    trainer = deterministic_trainer
    trainer.collect_rollout()

    states = trainer.buffer.get_last_lstm_states()
    hxs, cxs = states.hxs, states.cxs

    assert hxs.dim() == 2
    assert cxs.dim() == 2
    assert hxs.shape[0] == trainer.num_envs
    assert cxs.shape[0] == trainer.num_envs


def test_state_flow_initialization(deterministic_trainer):
    trainer = deterministic_trainer

    # Before any rollout, buffer should not have states
    assert trainer.buffer.last_hxs is None
    assert trainer.buffer.last_cxs is None

    # validate_lstm_state_flow must initialize them
    trainer.validate_lstm_state_flow()

    assert trainer.buffer.last_hxs is not None
    assert trainer.buffer.last_cxs is not None
