import crafter
import gym


def _make_crafter_env(**kwargs):
    return crafter.Env(**kwargs)


gym.register(
    id="Crafter-v1",
    entry_point=_make_crafter_env,
)
