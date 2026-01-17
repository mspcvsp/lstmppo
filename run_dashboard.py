from lstmppo.trainer import LSTMPPOTrainer
from lstmppo.rl_dashboard import RLDashboard

def main():
    trainer = LSTMPPOTrainer.from_preset("cartpole_easy")
    dashboard = RLDashboard(trainer)
    dashboard.run()

if __name__ == "__main__":
    main()

