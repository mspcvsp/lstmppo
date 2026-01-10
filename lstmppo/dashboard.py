from rich import box
from rich.columns import Columns
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def render_dashboard(trainer):

        target_kl = trainer.state.target_kl
        stats = trainer.state.stats

        # ---------------- PPO METRICS ----------------
        ppo = Table(title="PPO Metrics",
                    box=box.SIMPLE)
        
        ppo.add_column("Metric")
        ppo.add_column("Value", justify="right")

        def colorize(name, value):

            if name == "approx_kl":
            
                if value > 2 * target_kl:
                    return f"[red]{value:.4f}[/red]"
            
                elif value > target_kl:
                    return f"[yellow]{value:.4f}[/yellow]"
            
                else:
                    return f"[green]{value:.4f}[/green]"
            
            if name == "entropy":

                return (f"[green]{value:.3f}[/green]"
                        if value > 1.0 else f"[yellow]{value:.3f}[/yellow]")
            
            if name == "clip_frac":

                if value > 0.5:
                    return f"[red]{value:.3f}[/red]"
                elif value > 0.3:
                    return f"[yellow]{value:.3f}[/yellow]"
                else:
                    return f"[green]{value:.3f}[/green]"

            return f"{value:.3f}"

        for key in ["policy_loss",
                    "value_loss",
                    "entropy",
                    "approx_kl",
                    "clip_frac",
                    "explained_var",
                    "grad_norm"]:
            
            if key in stats:
                ppo.add_row(key, colorize(key, float(stats[key])))

        # ---------------- EPISODE METRICS ----------------
        ep = Table(title="Episode Stats", box=box.SIMPLE)

        ep.add_column("Metric")
        ep.add_column("Value", justify="right")

        for key in ["episodes",
                    "alive_envs",
                    "max_ep_len",
                    "avg_ep_len",
                    "max_ep_returns",
                    "avg_ep_returns",
                    "avg_ep_len_ema",
                    "avg_ep_returns_ema"]:
            
            if key in stats:
                ep.add_row(key, f"{float(stats[key]):.3f}")

        # ---------------- EPISODE TRENDS ----------------
        spark = Table.grid()

        spark.add_column(width=12)
        spark.add_column(width=32)
        
        spark.add_row("avg_ep_len",
                      sparkline(trainer.state.ep_len_history,
                                width=30,
                                style="green"))
        
        spark.add_row("avg_ep_returns",
                      sparkline(trainer.state.ep_return_history,
                                width=30,
                                style="magenta"))

        spark_panel = Panel(spark,
                            title="Episode Trends",
                            border_style="yellow",
                            padding=0)

        # ---------------- POLICY STABILITY ----------------
        policy = Table.grid()
        policy.add_column(width=12)
        policy.add_column(width=32)
        
        policy.add_row("KL",
                       sparkline(trainer.state.kl_history,
                                 width=30,
                                 style="cyan"))
        
        policy.add_row("Entropy",
                       sparkline(trainer.state.entropy_history,
                                 width=30,
                                 style="magenta"))

        policy_panel = Panel(policy,
                             title="Policy Stability",
                             border_style="cyan",
                             padding=0)

        # ---------------- VALUE FUNCTION DRIFT ----------------
        vf = Table.grid()
        vf.add_column(width=20)
        vf.add_column(width=32)
        vf.add_row("Explained Var",
                   sparkline(trainer.state.ev_history,
                             width=30,
                             style="green"))
        
        vf_panel = Panel(vf,
                         title="Value Function Drift",
                         border_style="green",
                         padding=0)

        # ---------------- RECENT RETURNS HISTOGRAM ----------------
        hist_panel =\
            Panel(histogram(trainer.env.completed_ep_returns,
                            bins=10,
                            width=30,
                            style="green"),
                            title="Recent Returns Histogram",
                            border_style="green",
                            padding=0)

        # ---------------- PER-ENV TIMELINES ----------------
        colors = ["cyan", "green", "magenta", "yellow", "red", "blue"]
        env_panels = []

        for row, history in enumerate(trainer.env.ep_len_history):
            style = colors[row % len(colors)]
            spark = sparkline(history, width=30, style=style)
            current = int(trainer.env.ep_len[row].cpu().item())

            table = Table.grid(padding=0)
            table.add_column(justify="right", width=4)
            table.add_column(justify="left", width=32)
            table.add_column(justify="right", width=4)
            table.add_row(f"{row}", spark, f"{current}")

            env_panels.append(Panel(table, padding=0))

        env_sparks_panel = Panel(
            Columns(env_panels, equal=True, expand=True),
            title="Per‑Env Episode Timelines",
            border_style="blue"
        )

        # ---------------- FINAL DASHBOARD ----------------
        dashboard = Table.grid(expand=True)
        dashboard.add_row(
            Panel(ppo, title="PPO", border_style="cyan"),
            Panel(ep, title="Episodes", border_style="green"),
            spark_panel,
        )
        dashboard.add_row(policy_panel, vf_panel, hist_panel)
        dashboard.add_row(env_sparks_panel)

        return dashboard


def sparkline(data,
              width=30,
              style="cyan"):

    if not data:
        return Text(" " * width)

    # Normalize data to 0–1
    mn = min(data)
    mx = max(data)
    rng = mx - mn if mx != mn else 1.0
    norm = [(x - mn) / rng for x in data]

    # Unicode sparkline blocks
    blocks = "▁▂▃▄▅▆▇█"
    chars = [blocks[int(v * (len(blocks) - 1))] for v in norm]

    # Fit to width
    if len(chars) < width:
        chars = [" "] * (width - len(chars)) + chars
    else:
        chars = chars[-width:]

    return Text("".join(chars), style=style)


def histogram(data, bins=10, width=30, style="cyan"):
    if not data:
        return Text(" " * width)
    hist = np.histogram(data, bins=bins)[0]
    max_count = max(hist)
    blocks = " ▁▂▃▄▅▆▇█"
    norm = [int((count / max_count) * (len(blocks) - 1)) for count in hist]
    chars = [blocks[n] for n in norm]
    return Text("".join(chars), style=style)
