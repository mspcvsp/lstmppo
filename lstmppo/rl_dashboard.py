# rl_dashboard.py
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, VerticalScroll
from textual.widgets import Static
from textual.reactive import reactive

class MetricPanel(Static):
    """Wrapper for Rich-rendered panels/tables."""
    data = reactive(None)

    def render(self):
        return self.data


class RLDashboard(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #top-row {
        height: 25%;
        layout: horizontal;
    }

    #middle-row {
        height: 25%;
        layout: horizontal;
    }

    #bottom-row {
        height: 50%;
        overflow: auto;
    }

    Static {
        border: solid green;
        padding: 1;
    }
    """

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def compose(self) -> ComposeResult:
        yield Container(
            Horizontal(
                MetricPanel(id="ppo"),
                MetricPanel(id="episodes"),
                MetricPanel(id="trends"),
                id="top-row"
            ),
            Horizontal(
                MetricPanel(id="policy"),
                MetricPanel(id="value"),
                MetricPanel(id="hist"),
                id="middle-row"
            ),
            VerticalScroll(
                MetricPanel(id="envs"),
                id="bottom-row"
            )
        )

    def on_mount(self):
        self.set_interval(0.25, self.refresh_panels)

    def refresh_panels(self):
        # PPO Metrics
        self.query_one("#ppo", MetricPanel).data = self.trainer.render_ppo_table()

        # Episode Stats
        self.query_one("#episodes", MetricPanel).data = self.trainer.render_episode_table()

        # Episode Trends
        self.query_one("#trends", MetricPanel).data = self.trainer.render_episode_trends()

        # Policy Stability
        self.query_one("#policy", MetricPanel).data = self.trainer.render_policy_stability()

        # Value Drift
        self.query_one("#value", MetricPanel).data = self.trainer.render_value_drift()

        # Histogram
        self.query_one("#hist", MetricPanel).data = self.trainer.render_histogram()

        # Per-env timelines
        self.query_one("#envs", MetricPanel).data = self.trainer.render_env_timelines()