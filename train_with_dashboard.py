# run_with_dashboard.py
import threading

from lstmppo.trainer import LSTMPPOTrainer, Config, initialize_config
from lstmppo.rl_dashboard import RLDashboard


def main(total_updates: int = 1000):
    cfg = Config()
    cfg = initialize_config(cfg)

    trainer = LSTMPPOTrainer(cfg)

    # Run training in a background thread so the dashboard stays responsive
    train_thread = threading.Thread(
        target=trainer.train,
        args=(total_updates,),
        daemon=True,
    )
    train_thread.start()

    # Launch Textual dashboard
    app = RLDashboard(trainer)
    app.run()


if __name__ == "__main__":
    main()