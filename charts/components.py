from dataclasses import dataclass
from io import BytesIO


@dataclass
class ComponentBreakdownChart:
    components: list[str]
    sizes: list[float]
    colors: list[str]
    title: str = ""
    box_width: float = 0.85
    box_height: float = 0.6

    def render(self, fmt: str = "png", dpi: int = 150) -> BytesIO:
        """Render to a BytesIO image. Safe for WASM — no top-level native imports."""
        import matplotlib
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        fig, ax = plt.subplots(figsize=(7, 2.5))
        fig.patch.set_facecolor("#0f0f1a")
        ax.set_facecolor("#0f0f1a")
        ax.set_xlim(0, len(self.components))
        ax.set_ylim(0, 1)
        ax.axis("off")

        y = 0.2
        for i, (name, val, col) in enumerate(zip(self.components, self.sizes, self.colors)):
            x = i + 0.075
            ax.add_patch(FancyBboxPatch(
                (x, y), self.box_width, self.box_height,
                boxstyle="round,pad=0.02,rounding_size=0.05",
                linewidth=2, edgecolor=col, facecolor="#141426",
            ))
            ax.text(
                x + self.box_width / 2, y + self.box_height * 0.65,
                name, ha="center", va="center", color="#cce", fontsize=9,
            )
            ax.text(
                x + self.box_width / 2, y + self.box_height * 0.30,
                f"{val}%", ha="center", va="center",
                color="#ffffff", fontsize=11, fontweight="bold",
            )

        if self.title:
            fig.suptitle(self.title, color="#cce", fontsize=11)

        fig.tight_layout()

        buf = BytesIO()
        fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return buf
