import threading
from typing import Sequence

import numpy as np
import streamlit as st

from lstmppo.trainer import Config, LSTMPPOTrainer, initialize_config


@st.cache_resource
def get_trainer():
    cfg = Config()
    cfg = initialize_config(cfg)
    return LSTMPPOTrainer(cfg)


def sparkline(data: Sequence[float], width=80):
    if not data:
        return ""
    arr = np.array(data[-width:])
    mn, mx = arr.min(), arr.max()
    rng = mx - mn if mx != mn else 1.0
    norm = (arr - mn) / rng
    blocks = "▁▂▃▄▅▆▇█"
    chars = [blocks[int(v * (len(blocks) - 1))] for v in norm]
    return "".join(chars)


def main():
    trainer = get_trainer()

    if "trainer_thread_started" not in st.session_state:
        train_thread = threading.Thread(target=trainer.train, args=(1000,), daemon=True)
        train_thread.start()

        st.session_state["trainer_thread_started"] = True

    st.title("PositionOnlyCartPole Dashboard")

    # PPO Metrics
    st.header("PPO Metrics")
    stats = trainer.state.metrics.to_dict()
    if stats:
        cols = st.columns(3)
        for i, k in enumerate(["policy_loss", "value_loss", "entropy", "approx_kl", "clip_frac", "explained_var"]):
            if k in stats:
                cols[i % 3].metric(k, f"{stats[k]:.4f}")
    else:
        st.write("Waiting for data...")

    # Episode Stats
    st.header("Episode Stats")
    if stats:
        cols = st.columns(2)
        for i, k in enumerate(["avg_ep_len", "max_ep_len", "avg_ep_returns", "max_ep_returns"]):
            if k in stats:
                cols[i % 2].metric(k, f"{stats[k]:.3f}")

    # Episode Trends
    st.header("Episode Trends")
    st.text("Episode Length")

    st.text(sparkline(trainer.state.history.ep_len))
    st.text("Episode Returns")
    st.text(sparkline(trainer.state.history.ep_return))

    # Histogram
    st.header("Recent Returns Histogram")
    if trainer.env.completed_ep_returns:
        hist, _ = np.histogram(trainer.env.completed_ep_returns, bins=10)
        st.bar_chart(hist)
    else:
        st.write("Waiting for data...")


if __name__ == "__main__":
    main()
