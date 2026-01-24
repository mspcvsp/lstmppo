"""
This test catches:

- config drift
- invalid defaults
"""
def test_config_defaults():
    from lstmppo.config import PPOConfig

    cfg = PPOConfig()
    assert cfg.learning_rate > 0
    assert cfg.gamma <= 1.0