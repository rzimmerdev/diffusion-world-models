from dataclasses import dataclass
from io import BytesIO


@dataclass
class PSNRBarChart:
    models: list[str]
    colors: list[str]
    ctx16_avg: list[float]
    ctx16_fin: list[float]
    ctx50_avg: list[float]
    ctx50_fin: list[float]
    title: str = "MiniGrid Quantitative Results"
    bar_width: float = 0.35

    _THEME = {
        "font.family": "monospace",
        "axes.facecolor": "#0f0f1a",
        "figure.facecolor": "#0f0f1a",
        "axes.edgecolor": "#334",
        "text.color": "#dde",
        "axes.labelcolor": "#dde",
        "xtick.color": "#889",
        "ytick.color": "#889",
        "axes.grid": True,
        "grid.color": "#223",
        "grid.linestyle": "--",
    }

    def render(self, fmt: str = "png", dpi: int = 150) -> BytesIO:
        """Render to a BytesIO image. Safe for WASM — no top-level native imports."""
        import numpy as np
        import matplotlib
        import matplotlib.pyplot as plt

        matplotlib.rcParams.update(self._THEME)
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

        panels = [
            (self.ctx16_avg, self.ctx16_fin, "Context Length 16"),
            (self.ctx50_avg, self.ctx50_fin, "Context Length 50"),
        ]
        for ax, (avg, fin, panel_title) in zip(axes, panels):
            self._draw_panel(ax, avg, fin, panel_title, np)

        fig.suptitle(self.title, color="#cce", fontsize=11, y=1.02)
        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf

    def _draw_panel(self, ax, avg, fin, panel_title, np):
        x = np.arange(len(self.models))
        w = self.bar_width
        bars_avg = ax.bar(x - w / 2, avg, w, label="Avg PSNR",   color=self.colors, alpha=0.65)
        bars_fin = ax.bar(x + w / 2, fin, w, label="Final PSNR", color=self.colors, alpha=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(self.models, fontsize=8)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(panel_title, color="#aac", fontsize=10)
        ax.set_ylim(0, 50)
        ax.legend(fontsize=7, framealpha=0.2)
        self._annotate_bars(ax, list(bars_avg) + list(bars_fin))

    def _annotate_bars(self, ax, bars):
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", va="bottom", fontsize=7, color="#dde",
            )
