# trainer_renderers.py
import numpy as np

from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns


def sparkline(data, width=30, style="cyan"):
    if not data:
        return Text(" " * width)

    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx != mn else 1.0
    norm = [(x - mn) / rng for x in data]

    blocks = "▁▂▃▄▅▆▇█"
    chars = [blocks[int(v * (len(blocks) - 1))] for v in norm]

    if len(chars) < width:
        chars = [" "] * (width - len(chars)) + chars
    else:
        chars = chars[-width:]

    return Text("".join(chars), style=style)


def histogram(data, bins=10, width=30, style="cyan"):
    if not data:
        return Text(" " * width)

    hist = np.histogram(data, bins=bins)[0]
    max_count = max(hist) if max(hist) > 0 else 1
    blocks = " ▁▂▃▄▅▆▇█"
    norm = [int((count / max_count) * (len(blocks) - 1)) for count in hist]
    chars = [blocks[n] for n in norm]

    if len(chars) < width:
        chars = [" "] * (width - len(chars)) + chars
    else:
        chars = chars[:width]

    return Text("".join(chars), style=style)


def render_ppo_table(self):
    table = Table(title="PPO Metrics")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for k in [
        "policy_loss",
        "value_loss",
        "entropy",
        "approx_kl",
        "clip_frac",
        "explained_var",
        "grad_norm",
    ]:
        if k in self.state.stats:
            table.add_row(k, f"{float(self.state.stats[k]):.4f}")

    return table


def render_episode_table(self):
    table = Table(title="Episode Stats")
    table.add_column("Metric")
    table.add_column("Value", justify="right")

    for k in [
        "episodes",
        "alive_envs",
        "max_ep_len",
        "avg_ep_len",
        "max_ep_returns",
        "avg_ep_returns",
        "avg_ep_len_ema",
        "avg_ep_returns_ema",
    ]:
        if k in self.state.stats:
            table.add_row(k, f"{float(self.state.stats[k]):.3f}")

    return table


def render_episode_trends(self):
    table = Table.grid()
    table.add_column(width=12)
    table.add_column(width=32)

    table.add_row(
        "avg_ep_len",
        sparkline(self.state.ep_len_history, width=30, style="green"),
    )
    table.add_row(
        "avg_ep_returns",
        sparkline(self.state.ep_return_history, width=30, style="magenta"),
    )

    return Panel(table, title="Episode Trends")


def render_policy_stability(self):
    table = Table.grid()
    table.add_column(width=12)
    table.add_column(width=32)

    table.add_row(
        "KL", sparkline(self.state.kl_history, width=30, style="cyan")
    )
    table.add_row(
        "Entropy",
        sparkline(self.state.entropy_history, width=30, style="magenta"),
    )

    return Panel(table, title="Policy Stability")


def render_value_drift(self):
    table = Table.grid()
    table.add_column(width=20)
    table.add_column(width=32)

    table.add_row(
        "Explained Var",
        sparkline(self.state.ev_history, width=30, style="green"),
    )

    return Panel(table, title="Value Function Drift")


def render_histogram(self):
    return Panel(
        histogram(
            self.env.completed_ep_returns,
            bins=10,
            width=30,
            style="green",
        ),
        title="Recent Returns",
    )


def render_env_timelines(self):
    colors = ["cyan", "green", "magenta", "yellow", "red", "blue"]
    env_panels = []

    for i, hist in enumerate(self.env.ep_len_history):
        style = colors[i % len(colors)]
        spark = sparkline(hist, width=30, style=style)
        current = int(self.env.ep_len[i].cpu().item())

        t = Table.grid()
        t.add_column(justify="right", width=4)
        t.add_column(justify="left", width=32)
        t.add_column(justify="right", width=4)
        t.add_row(str(i), spark, str(current))

        env_panels.append(Panel(t, padding=0))

    return Panel(
        Columns(env_panels, equal=True),
        title="Per‑Env Episode Timelines",
    )