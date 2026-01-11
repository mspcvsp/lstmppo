from rich.progress import Progress, BarColumn, TimeElapsedColumn, TimeRemainingColumn, TextColumn
from rich.console import Console
from lstmppo.trainer import LSTMPPOTrainer, Config, initialize_config

def train_rich(total_updates=300):
    cfg = Config()
    cfg = initialize_config(cfg)
    cfg.env.env_id = "popgym-PositionOnlyCartPoleEasy-v0"
    trainer = LSTMPPOTrainer(cfg)
    trainer.state.reset(total_updates)

    console = Console()
    with Progress(
        TextColumn("[bold blue]Update {task.fields[update]:04d}"),
        BarColumn(),
        TextColumn("loss={task.fields[loss]:.3f}"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("training", total=total_updates, update=0, loss=0.0)

        for trainer.state.update_idx in range(total_updates):
            trainer.state.init_stats()
            trainer.state.apply_schedules(trainer.optimizer)

            last_value = trainer.collect_rollout()
            trainer.buffer.compute_returns_and_advantages(last_value)
            trainer.optimize_policy()

            loss = float(trainer.state.stats.get("policy_loss", 0.0))
            progress.update(task, advance=1, update=trainer.state.update_idx, loss=loss)

            if trainer.state.update_idx % 10 == 0:
                ep_ret = trainer.state.stats.get("avg_ep_returns", 0.0)
                ep_len = trainer.state.stats.get("avg_ep_len", 0.0)
                ev = trainer.state.stats.get("explained_var", 0.0)
                console.print(f"[green]Return={ep_ret:.1f}  Length={ep_len:.1f}  EV={ev:.2f}")

if __name__ == "__main__":
    train_rich()