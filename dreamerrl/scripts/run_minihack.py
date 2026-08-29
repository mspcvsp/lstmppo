import torch

from dreamerrl.models.aux_objectives import AUX_OBJECTIVES
from dreamerrl.training.trainer import DreamerTrainer
from dreamerrl.utils.types import make_default_config


def main():
    cfg = make_default_config()

    # -----------------------------------------------------
    # Environment: MiniHack symbolic mode
    # -----------------------------------------------------
    cfg.env.env_id = "MiniHack-LockedDoor-v0"
    cfg.env.num_envs = 8
    cfg.env.max_episode_steps = 200
    cfg.env.parallel = False
    cfg.env.deterministic = False
    cfg.env.seed = 0

    # -----------------------------------------------------
    # World Model: enable auxiliary losses
    # -----------------------------------------------------
    cfg.train.disable_aux_losses = False

    # Recommended aux losses for MiniHack symbolic mode
    cfg.world.aux_objectives = [
        AUX_OBJECTIVES["novelty"],
        AUX_OBJECTIVES["reachability"],
        AUX_OBJECTIVES["resource"],
        AUX_OBJECTIVES["affordance"],  # optional but helpful
    ]

    # DreamerV3 latent sizes (moderate)
    cfg.world.deter_size = 128
    cfg.world.stoch_size = 32
    cfg.world.num_classes = 32
    cfg.world.hidden_size = 256
    cfg.world.imagination_horizon = 5

    # -----------------------------------------------------
    # Training hyperparameters
    # -----------------------------------------------------
    cfg.train.seed = 0
    cfg.train.cuda = torch.cuda.is_available()
    cfg.train.batch_size = 16
    cfg.train.collect_steps = 50
    cfg.train.seq_len = 50

    cfg.train.model_lr = 3e-4
    cfg.train.actor_lr = 3e-4
    cfg.train.critic_lr = 3e-4

    cfg.train.random_exploration_steps = 1000
    cfg.train.freeze_actor_critic_steps = 0

    # -----------------------------------------------------
    # Logging (TensorBoard)
    # -----------------------------------------------------
    cfg.log.enable_wandb = False
    cfg.log.tb_logdir = "./logs"
    cfg.log.run_name = "minihack_lockeddoor_aux"

    # -----------------------------------------------------
    # Trainer
    # -----------------------------------------------------
    trainer = DreamerTrainer(cfg)

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------
    trainer.train(total_updates=5000)


if __name__ == "__main__":
    main()
