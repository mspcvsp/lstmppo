import tyro
import wandb

from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.types import Config


def main(
    cfg: Config,
    total_updates: int = 2000,
    seed: int = 0,
    wandb_project: str = "lstmppo",
    wandb_group: str = "position-only-cartpole",
):
    # Override seed
    cfg.trainer.seed = seed

    # Initialize run name
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


if __name__ == "__main__":
    tyro.cli(main)
