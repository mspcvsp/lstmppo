import torch


def make_fake_rollout(T, B, obs_dim, *, pattern="range"):
    if pattern == "range":
        obs = torch.arange(T * B * obs_dim, dtype=torch.float32).reshape(T, B, obs_dim)
    elif pattern == "zeros":
        obs = torch.zeros(T, B, obs_dim)
    else:
        obs = torch.randn(T, B, obs_dim)

    next_obs = obs.roll(-1, dims=0)
    next_obs[-1] = 0

    actions = torch.zeros(T, B, dtype=torch.long)
    rewards = torch.arange(T, dtype=torch.float32).unsqueeze(1).expand(T, B)
    masks = torch.ones(T, B)

    return obs, next_obs, actions, rewards, masks
