"""
✔ Trainer seeding stays untouched. Your trainer’s reset() ensures:
- deterministic rollout 0
- deterministic env reset
- deterministic LSTM state flow
This is essential for your drift, correlation, and determinism tests.

✔ Global seeding is now cleanly separated. seed.py handles:
- model weight initialization
- optimizer initialization
- any randomness before trainer construction
This is the correct SRP boundary.

✔ Tyro handles multi‑seed CLI elegantly.
You now run: python run_experiment.py --seeds 1 2 3 --total-updates 5000

✔ Each seed gets its own W&B run
Because cfg.init_run_name() is called per seed.
"""

import copy
from typing import List

import tyro

import wandb
from lstmppo.seed import set_global_seeds
from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import Config


def run_single_seed(
    base_cfg: Config,
    total_updates: int,
    seed: int,
    wandb_project: str,
    wandb_group: str,
):
    # Create a fresh config instance for this seed
    cfg = copy.deepcopy(base_cfg)

    # Apply seed to config
    cfg.trainer.seed = seed

    # Global seeding BEFORE trainer construction
    set_global_seeds(seed)

    # Build run name (timestamp + seed)
    cfg.init_run_name()

    # Initialize W&B
    wandb.init(
        project=wandb_project,
        group=wandb_group,
        name=cfg.log.run_name,
        config={
            "seed": seed,
            "total_updates": total_updates,
            **cfg.__dict__,
        },
    )

    # Create trainer
    trainer = LSTMPPOTrainer(cfg)

    # Train
    trainer.train(total_updates=total_updates)

    wandb.finish()


def main(
    cfg: Config,
    total_updates: int = 2000,
    seeds: List[int] = [0],  # Multi-seed support
    wandb_project: str = "lstmppo",
    wandb_group: str = "position-only-cartpole",
):
    """
    - supports multi‑seed via Tyro (List[int])
    - uses seed.py for global seeding
    - keeps trainer‑internal seeding untouched
    - creates a fresh config per seed
    - creates a separate W&B run per seed
    """
    for seed in seeds:
        print(f"\n=== Running seed {seed} ===")
        run_single_seed(
            base_cfg=cfg,
            total_updates=total_updates,
            seed=seed,
            wandb_project=wandb_project,
            wandb_group=wandb_group,
        )


if __name__ == "__main__":
    tyro.cli(main)
