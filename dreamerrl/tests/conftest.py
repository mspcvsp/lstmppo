import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.spaces import Box

from dreamerrl.models.aux_objectives import AUX_OBJECTIVES
from dreamerrl.models.world_model import WorldModel
from dreamerrl.models.world_model_core import RSSMCore as RSSM
from dreamerrl.utils.types import LatentConfig, NetworkConfig


@pytest.fixture
def dummy_batch():
    B, L, obs_dim, action_dim = 4, 5, 8, 5

    return {
        "obs": torch.rand(B, L, obs_dim),
        "action": torch.nn.functional.one_hot(torch.randint(0, action_dim, (B, L)), num_classes=action_dim).float(),
        "reward": torch.randn(B, L),
        "is_terminal": torch.zeros(B, L),
        "is_first": torch.zeros(B, L),
        "is_last": torch.zeros(B, L),
    }


@pytest.fixture
def world_model():
    """
    Minimal world model fixture for invariants testing.
    Matches the latent + network config used in your tests.
    """

    obs_space = gym.spaces.Box(
        low=0.0,
        high=1.0,
        shape=(8,),  # matches dummy_batch obs_dim
        dtype=np.float32,
    )

    latent = LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )

    net = NetworkConfig(
        hidden_size=256,
        action_dim=5,  # matches dummy_batch action_dim
        value_bins=41,
    )

    wm = WorldModel(
        obs_space=obs_space,
        latent=latent,
        net=net,
        device=torch.device("cpu"),
    )

    return wm


@pytest.fixture
def latent():
    return LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )


@pytest.fixture
def net():
    return NetworkConfig(
        hidden_size=256,
        action_dim=5,
        value_bins=41,
        bin_min=-20.0,
        bin_max=20.0,
    )


class DummyActor(torch.nn.Module):
    """
    Minimal actor for invariants tests.
    Consumes V3 latents (B, K, C) and returns logits (B, action_dim).
    """

    def __init__(self, latent: LatentConfig, net: NetworkConfig) -> None:
        super().__init__()
        assert net.action_dim is not None
        in_features = latent.deter_size + latent.stoch_size * latent.num_classes
        self.fc = torch.nn.Linear(in_features, net.action_dim)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # Flatten z from (B, K, C) → (B, K*C)
        z_flat = z.view(z.size(0), -1)
        x = torch.cat([h, z_flat], dim=-1)
        return self.fc(x)


@pytest.fixture
def dummy_actor(latent, net):
    return DummyActor(latent, net)


@pytest.fixture
def actor(dummy_actor):
    return dummy_actor


class DummyObsSpace(Box):
    """
    A minimal Box-like observation space for testing DreamerV3.
    Fully compatible with get_flat_obs_dim and ObsEncoder.
    """

    def __init__(self, obs_dim: int):
        # low/high must be arrays, not scalars
        low = -np.inf * np.ones((obs_dim,), dtype=np.float32)
        high = np.inf * np.ones((obs_dim,), dtype=np.float32)

        super().__init__(low=low, high=high, dtype=np.float32)

    def sample(self):
        # Pylance wants explicit ints, not a tuple
        return torch.randn(self.shape[0], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Deterministic global setup
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session", autouse=True)
def _set_determinism():
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)


# ---------------------------------------------------------------------------
# RSSM fixture (matches your WorldModel latent config)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def rssm(latent, net):
    """
    Minimal RSSM instance matching the latent + network config.
    """
    return RSSM(latent=latent, net=net).eval()


# ---------------------------------------------------------------------------
# Actor input fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_h(latent):
    """
    Deterministic RSSM deterministic state (B, deter_size).
    """
    return torch.randn(8, latent.deter_size)


@pytest.fixture
def dummy_z_actor(latent):
    """
    Stochastic latent z for actor (B, K, C).
    """
    return torch.randn(8, latent.num_classes, latent.stoch_size)


# ---------------------------------------------------------------------------
# Posterior invariance fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_obs():
    """
    Observations for posterior invariance tests.
    """
    return torch.randn(8, 8)  # matches obs_dim=8 in your world_model fixture


@pytest.fixture
def dummy_action():
    """
    One-hot actions for posterior invariance tests.
    """
    A = 5
    return torch.nn.functional.one_hot(torch.randint(0, A, (8,)), num_classes=A).float()


@pytest.fixture
def dummy_reward():
    """
    Reward tensor for posterior invariance tests.
    """
    return torch.randn(8, 1)


# ---------------------------------------------------------------------------
# KL test logits
# ---------------------------------------------------------------------------


@pytest.fixture
def kl_logits(latent):
    """
    Random logits for KL free-bits monotonicity tests.
    """
    B = 32
    K = latent.num_classes
    return (
        torch.randn(B, K),  # logits_p
        torch.randn(B, K),  # logits_q
    )


# ---------------------------------------------------------------------------
# Symlog/symexp round-trip obs
# ---------------------------------------------------------------------------


@pytest.fixture
def roundtrip_obs():
    """
    Observations for symlog/symexp round-trip tests.
    """
    return torch.randn(8, 8)  # matches obs_dim=8


# ---------------------------------------------------------------------------
# CPU/GPU determinism helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def cpu_gpu_available():
    """
    Skip GPU determinism tests if CUDA is not available.
    """
    return torch.cuda.is_available()


# ---------------------------------------------------------------------------
# Manual test control (disable unless --run-manual is passed)
# ---------------------------------------------------------------------------
def pytest_addoption(parser):
    parser.addoption(
        "--run-manual",
        action="store_true",
        default=False,
        help="Run tests marked as manual",
    )

    parser.addoption(
        "--env",
        action="store",
        default="popgym-RepeatFirstEasy-v0",
        help="PopGym environment ID",
    )

    parser.addoption(
        "--steps",
        action="store",
        default=5000,
        type=int,
        help="Number of training steps",
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-manual"):
        # User explicitly asked to run manual tests → do nothing
        return

    skip_manual = pytest.mark.skip(reason="Use --run-manual to run this test")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)


# ---------------------------------------------------------------------------
# Aux-loss fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def aux_world_model():
    """
    WorldModel with aux losses enabled for structural + numerical tests.
    """
    obs_space = DummyObsSpace(obs_dim=8)

    latent = LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )

    net = NetworkConfig(
        hidden_size=256,
        action_dim=5,
        value_bins=41,
        disable_aux_losses=False,
    )

    wm = WorldModel(
        obs_space=obs_space,
        latent=latent,
        net=net,
        aux_objectives=[
            AUX_OBJECTIVES["novelty"],
            AUX_OBJECTIVES["skill"],
        ],
        device=torch.device("cpu"),
    )

    return wm


@pytest.fixture
def world_model_aux_losses_disabled():
    """
    WorldModel with aux losses disabled for invariants tests.
    """
    obs_space = DummyObsSpace(obs_dim=8)

    latent = LatentConfig(
        deter_size=200,
        stoch_size=30,
        num_classes=32,
    )

    net = NetworkConfig(
        hidden_size=256,
        action_dim=5,
        value_bins=41,
        disable_aux_losses=True,
    )

    wm = WorldModel(
        obs_space=obs_space,
        latent=latent,
        net=net,
        aux_objectives=[
            AUX_OBJECTIVES["novelty"],
            AUX_OBJECTIVES["skill"],
        ],
        device=torch.device("cpu"),
    )

    return wm


@pytest.fixture
def latent_cluster(latent):
    """
    Synthetic latent cluster generator for aux-loss tests.
    Produces latents shaped exactly like RSSM stochastic state:
        (B, L, num_classes, stoch_size)
    """

    def _make(cluster_id, batch=4, L=1):
        # Create zeros
        z = torch.zeros(batch, L, latent.num_classes, latent.stoch_size)

        # Activate one cluster
        z[:, :, cluster_id, :] = 1.0

        return z.view(batch * L, latent.num_classes * latent.stoch_size)

    return _make
