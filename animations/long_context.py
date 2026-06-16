"""
LongContextComparison — Manim Community Edition scene
======================================================

Side-by-side comparison showing forward → reverse trajectory consistency.
  LEFT   – Diffusion Baseline (low PSNR on reverse, highlighted red)
  RIGHT  – StateSpaceDiffuser (high PSNR on reverse, highlighted green)

The further back in the sequence from the U-turn point, the bigger
the quality gap between the two methods.
"""

import numpy as np

from manim import Scene, VGroup
from manim import (
    Write,
    FadeIn,
    Create,
    GrowArrow,
    FadeOut,
    LaggedStart,
    there_and_back,
    CurvedArrow,
)
from manim import (
    Square,
    Arrow,
    DashedLine,
    Dot,
    RoundedRectangle,
    Line,
    Text,
    Tex,
    SurroundingRectangle,
)
from manim import UP, DOWN, LEFT, RIGHT, PI
from manim import BOLD, ITALIC
from manim import (
    WHITE,
    GREY_A,
    GREY_B,
    GREY_C,
    GREY_D,
    BLUE_D,
    GREEN_C,
    RED,
    RED_B,
    RED_C,
    YELLOW,
    YELLOW_A,
    YELLOW_B,
)

# ── Constants ────────────────────────────────────────────────────────────────
PANEL_BG = "#1a1a2e"
PANEL_STROKE = "#3a3a5c"
MAZE_BG = "#16213e"
MAZE_WALL = "#1a3a5c"
MAZE_GRID = "#0a1a3a"
DIFFUSION_COLOR = "#f97316"  # orange
SSM_COLOR = BLUE_D
PSNR_GOOD = GREEN_C
PSNR_BAD = RED_C
AGENT_COLOR = YELLOW

N_STEPS = 5  # forward steps (and symmetric reverse steps)

# Forward agent positions in mini-grid coords (col, row) ∈ [0,2]×[0,2]
FORWARD_POS = [(0, 2), (1, 2), (2, 2), (2, 1), (2, 0)]

# Reverse (correct) — retracing the path
REVERSE_POS = [(2, 0), (2, 1), (2, 2), (1, 2), (0, 2)]

# Diffusion baseline reverse positions (increasing error further from U-turn)
DIFFUSION_REVERSE = [
    (2.0, 0.0),  # step 0 — at U-turn, correct
    (2.1, 1.1),  # step 1 — slight drift
    (2.3, 1.6),  # step 2 — moderate drift
    (1.6, 1.2),  # step 3 — wrong cell
    (0.3, 1.7),  # step 4 — fully wrong
]

# PSNR values (dB) — higher = better reconstruction
PSNR_DIFFUSION = [33.1, 27.4, 22.6, 15.8, 10.3]
PSNR_SSM = [35.2, 34.8, 33.9, 33.5, 32.7]


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════


def make_panel(center: np.ndarray, width=4.6, height=4.2) -> RoundedRectangle:
    """Dark rounded panel background."""
    return RoundedRectangle(
        corner_radius=0.2,
        width=width,
        height=height,
        fill_color=PANEL_BG,
        fill_opacity=0.7,
        stroke_color=PANEL_STROKE,
        stroke_width=2,
    ).move_to(center)


def make_mini_frame(agent_pos, noise=(0.0, 0.0), cell_size=0.17, n_cells=3):
    """
    Small maze thumbnail showing a n_cells×n_cells grid with the agent as a
    yellow dot.  `agent_pos` = (col, row) in [0, n_cells-1]; `noise` is an
    (dx, dy) offset in cell-size units added to the agent position.
    """
    total = n_cells * cell_size
    frame = Square(
        side_length=total,
        fill_color=MAZE_BG,
        fill_opacity=1.0,
        stroke_color=GREY_B,
        stroke_width=1.2,
    )

    grid = VGroup()
    for i in range(1, n_cells):
        grid.add(
            Line(
                frame.get_left() + RIGHT * i * cell_size,
                frame.get_left() + RIGHT * i * cell_size + UP * total,
                stroke_color=MAZE_GRID,
                stroke_width=0.6,
            )
        )
        grid.add(
            Line(
                frame.get_bottom() + UP * i * cell_size,
                frame.get_bottom() + UP * i * cell_size + RIGHT * total,
                stroke_color=MAZE_GRID,
                stroke_width=0.6,
            )
        )

    # A few "walls" to make positions visually distinct
    walls = VGroup()
    wall_cfg = [
        ((0.5, 0.3), (0.5, 0.7)),  # vertical wall segment, col 1, rows 1-2
        ((0.3, 0.5), (0.7, 0.5)),  # horizontal wall segment
    ]
    for (x1, y1), (x2, y2) in wall_cfg:
        walls.add(
            Line(
                frame.get_left() + RIGHT * x1 * total + UP * y1 * total,
                frame.get_left() + RIGHT * x2 * total + UP * y2 * total,
                stroke_color=MAZE_WALL,
                stroke_width=2.5,
            )
        )

    # Agent dot
    cx = frame.get_left()[0] + (agent_pos[0] + 0.5 + noise[0]) * cell_size
    cy = frame.get_bottom()[1] + (agent_pos[1] + 0.5 + noise[1]) * cell_size
    agent = Dot(
        point=[cx, cy, 0],
        radius=cell_size * 0.32,
        color=AGENT_COLOR,
        stroke_width=1,
        stroke_color=WHITE,
    )

    return VGroup(frame, grid, walls, agent)


def make_psnr_label(value: float, color) -> Tex:
    """PSNR score label: 'XX.X dB'."""
    return Tex(f"{value:.1f}\\,dB", font_size=12, color=color)


def make_dotted_connector(top_center, bottom_center, color=GREY_C):
    """Dashed vertical line connecting matching forward ↔ reverse frames."""
    return DashedLine(
        top_center,
        bottom_center,
        dash_length=0.08,
        dashed_ratio=0.55,
        color=color,
        stroke_width=1.8,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Scene
# ═════════════════════════════════════════════════════════════════════════════


class LongContextComparison(Scene):
    def construct(self):
        self.camera.background_color = "#0d0d1a"

        # ── Title ─────────────────────────────────────────────────────────
        title = Text(
            "Long Context: Forward → Reverse Consistency",
            font_size=30,
            color=WHITE,
            weight=BOLD,
        )
        title.to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.wait(0.2)

        # ── Panels ────────────────────────────────────────────────────────
        panel_L = make_panel(LEFT * 3.0 + DOWN * 0.15)
        panel_R = make_panel(RIGHT * 3.0 + DOWN * 0.15)

        lbl_L = Text(
            "Diffusion Baseline", font_size=20, color=DIFFUSION_COLOR, weight=BOLD
        )
        lbl_L.next_to(panel_L, UP, buff=0.15)

        lbl_R = Text("StateSpaceDiffuser", font_size=20, color=SSM_COLOR, weight=BOLD)
        lbl_R.next_to(panel_R, UP, buff=0.15)

        self.play(
            FadeIn(panel_L),
            FadeIn(panel_R),
            Write(lbl_L),
            Write(lbl_R),
            run_time=0.8,
        )

        # ── Common forward frames (both panels share the same forward path) ──
        fwd_label_L = Text("Forward", font_size=14, color=GREY_A, slant=ITALIC)
        fwd_label_R = Text("Forward", font_size=14, color=GREY_A, slant=ITALIC)
        fwd_label_L.move_to(panel_L).shift(UP * 1.95 + LEFT * 1.5)
        fwd_label_R.move_to(panel_R).shift(UP * 1.95 + LEFT * 1.5)

        fwd_frames_L = self._build_frame_row(
            FORWARD_POS, panel_L.get_center(), y_off=1.45, noise=0.0
        )
        fwd_frames_R = self._build_frame_row(
            FORWARD_POS, panel_R.get_center(), y_off=1.45, noise=0.0
        )

        self.play(
            Write(fwd_label_L),
            Write(fwd_label_R),
            LaggedStart(*[FadeIn(f) for f in fwd_frames_L], lag_ratio=0.12),
            LaggedStart(*[FadeIn(f) for f in fwd_frames_R], lag_ratio=0.12),
            run_time=1.2,
        )
        self.wait(0.3)

        # ── U-turn arrow ──────────────────────────────────────────────────
        rev_y_L = panel_L.get_center()[1] - 1.28
        rev_y_R = panel_R.get_center()[1] - 1.28
        u_turn_L = self._make_uturn(fwd_frames_L[-1], rev_y_L, panel_L)
        u_turn_R = self._make_uturn(fwd_frames_R[-1], rev_y_R, panel_R)
        self.play(FadeIn(u_turn_L), FadeIn(u_turn_R), run_time=0.5)
        self.wait(0.2)

        # ── Reverse frames ────────────────────────────────────────────────
        rev_label_L = Text("Reverse", font_size=14, color=GREY_A, slant=ITALIC)
        rev_label_R = Text("Reverse", font_size=14, color=GREY_A, slant=ITALIC)
        rev_label_L.move_to(panel_L).shift(DOWN * 2.35 + LEFT * 1.6)
        rev_label_R.move_to(panel_R).shift(DOWN * 2.35 + LEFT * 1.6)

        # Diffusion reverse: wrong positions
        rev_frames_L = self._build_frame_row(
            DIFFUSION_REVERSE,
            panel_L.get_center(),
            y_off=-1.55,
            noise=0.0,
            positions_given_as_absolute=True,
        )
        # SSM reverse: correct positions
        rev_frames_R = self._build_frame_row(
            REVERSE_POS, panel_R.get_center(), y_off=-1.55, noise=0.0
        )

        self.play(
            Write(rev_label_L),
            Write(rev_label_R),
            LaggedStart(*[FadeIn(f) for f in rev_frames_L], lag_ratio=0.12),
            LaggedStart(*[FadeIn(f) for f in rev_frames_R], lag_ratio=0.12),
            run_time=1.2,
        )
        self.wait(0.4)

        # ── Dotted connectors + PSNR labels ───────────────────────────────
        connectors_L, psnr_labels_L = self._build_connectors(
            fwd_frames_L, rev_frames_L, PSNR_DIFFUSION, PSNR_BAD
        )
        connectors_R, psnr_labels_R = self._build_connectors(
            fwd_frames_R, rev_frames_R, PSNR_SSM, PSNR_GOOD
        )

        self.play(
            LaggedStart(*[Create(c) for c in connectors_L], lag_ratio=0.1),
            LaggedStart(*[Create(c) for c in connectors_R], lag_ratio=0.1),
            LaggedStart(*[Write(p) for p in psnr_labels_L], lag_ratio=0.1),
            LaggedStart(*[Write(p) for p in psnr_labels_R], lag_ratio=0.1),
            run_time=1.5,
        )
        self.wait(0.5)

        # ── Highlight the growing PSNR gap ────────────────────────────────
        # Pulse the worst PSNR values on the diffusion side
        self.play(
            psnr_labels_L[-1].animate.scale(1.3).set_color(RED),
            psnr_labels_L[-2].animate.scale(1.15).set_color(RED_B),
            rate_func=there_and_back,
            run_time=0.7,
        )
        self.wait(0.3)

        # ── Circumscribe the SSM side to show it stays consistent ────────
        highlight_box = SurroundingRectangle(
            VGroup(rev_frames_R, psnr_labels_R),
            color=SSM_COLOR,
            buff=0.15,
            corner_radius=0.1,
        )
        self.play(Create(highlight_box), run_time=0.8)
        self.bring_to_front(u_turn_R)
        self.wait(0.3)

        # ── Final message ─────────────────────────────────────────────────
        msg = Tex(
            r"\text{StateSpaceDiffuser preserves fidelity far from the U-turn, while Diffusion collapses}",
            font_size=20,
            color=WHITE,
        )
        msg.arrange(DOWN, buff=0.15)
        msg.to_edge(DOWN, buff=0.6)
        self.play(Write(msg), run_time=1.2)
        self.wait(2.5)

        self.play(FadeOut(*self.mobjects), run_time=0.8)

    # ── Internal builders ─────────────────────────────────────────────────

    def _build_frame_row(
        self,
        positions,
        panel_center,
        y_off,
        noise=0.0,
        positions_given_as_absolute=False,
    ):
        """Build a horizontal row of mini-frames centred under the panel."""
        frames = VGroup()
        # Horizontal span
        total_w = N_STEPS * 0.64  # frame side ≈0.55 + spacing
        start_x = panel_center[0] - total_w / 2 + 0.32

        for i, pos in enumerate(positions):
            if positions_given_as_absolute:
                # pos is already (col, row) in grid coords
                col, row = pos[0], pos[1]
                noise_xy = (0, 0)
            else:
                col, row = pos[0], pos[1]
                noise_xy = (noise, noise) if noise else (0, 0)

            frame = make_mini_frame((col, row), noise=noise_xy, cell_size=0.18)
            frame.move_to([start_x + i * 0.64, panel_center[1] + y_off, 0])
            frames.add(frame)
        return frames

    def _make_uturn(self, last_fwd_frame, target_y, panel):
        """Curved U-turn arrow from last forward frame down toward reverse row."""
        start = last_fwd_frame.get_right() + RIGHT * 0.10 + DOWN * 0.08
        end = np.array([start[0], target_y + 0.25, 0])
        arrow = CurvedArrow(
            start,
            end,
            angle=-PI / 2,
            color=YELLOW_B,
            stroke_width=3,
            tip_length=0.15,
        )
        label = Tex(r"\text{U-turn}", font_size=12, color=YELLOW_B)
        label.next_to(panel, RIGHT, buff=0.15)
        return VGroup(arrow, label)

    def _build_connectors(self, fwd_frames, rev_frames, psnr_values, psnr_color):
        """Dotted lines and PSNR labels connecting forward ↔ reverse pairs."""
        connectors = VGroup()
        labels = VGroup()
        for i, (fw, rv) in enumerate(zip(fwd_frames, rev_frames)):
            top = fw.get_bottom()
            bot = rv.get_top()
            conn = make_dotted_connector(
                top, bot, color=GREY_C if psnr_color == PSNR_BAD else GREY_D
            )
            connectors.add(conn)

            # PSNR label placed on the connector line, slight right offset
            mid = (top + bot) / 2
            lbl = make_psnr_label(abs(psnr_values[i]), psnr_color)
            lbl.next_to(mid, RIGHT, buff=0.06)
            labels.add(lbl)
        return connectors, labels
