"""
WorldModelAnimation — Manim Community Edition scene
=====================================================

SCENE PLAN
──────────────────────────────────────────────────────────────────────
LAYOUT (approximate screen regions)
  TOP            : title "World Models"
  LEFT           : 5×5 environment grid + agent
  BOTTOM-CENTRE  : history rectangle (mini-frames accumulate inside)
  CENTRE-RIGHT   : F block (world model)
  FAR RIGHT      : prediction box  ô_{t+1}
  BOTTOM         : final learning message

Step-by-step narrative
──────────────────────────────────────────────────────────────────────
STEP 1 — Title + Grid World
  • Write() the title at the top.
  • Create() the 5×5 grid on the left (border + cells + "Environment" label).
  • GrowFromCenter() a yellow circle agent at cell (row=2, col=1).
  • Short pause so the viewer can orient.

STEP 2 — History box accumulates observations
  • FadeIn() the history rectangle (+ "History" title + math notation below it).
  • Three coloured mini-frames (BLUE_C, TEAL_C, GREEN_C) appear one-by-one
    inside the box; each arrival also flashes the corresponding grid cell
    (row 2, cols 1-3) to show the agent "sensing" its environment.

STEP 3 — World Model F block appears
  • DrawBorderThenFill() the orange rounded rectangle;
    Write() the calligraphic 𝓕 simultaneously.
  • FadeIn() the "World Model" sub-label below the block.

STEP 4 — Data flow: history → F → prediction
  • FadeIn() the purple prediction box to the right of F.
  • GrowArrow() history → F (green arrow).
  • GrowArrow() F → prediction (orange arrow).
  • A green Dot packet slides along the first arrow, then disappears
    as the F block briefly pulses brighter (simulates computation).
  • An orange Dot packet slides along the second arrow, then disappears
    as ô_{t+1} flashes yellow (prediction produced).

STEP 5 — Agent takes an action and moves
  • FadeIn() the action label  "aₜ = →"  above the agent.
  • The agent slides one cell to the right (col 1 → col 2).
  • A fourth yellow mini-frame appears in the history box;
    the destination cell flashes.
  • Action label fades out.

STEP 6 — Final learning message
  • Circumscribe() the F block to draw attention.
  • Write() the closing message at the bottom.
  • Hold 2 s, then FadeOut() everything.

Object inventory (all built by helper methods)
  create_title()            → Text
  create_grid_world()       → (VGroup, VGroup[cells])
  create_agent()            → Circle
  create_history_box()      → (VGroup, Rectangle)
  create_mini_frame()       → Square
  create_world_model_block()→ (VGroup, RoundedRectangle)
  create_prediction_box()   → (VGroup, Rectangle)
──────────────────────────────────────────────────────────────────────
"""
import numpy as np

from manim import Scene, VGroup, Group, ManimColor
from manim import (
    Write, 
    FadeIn, 
    Create, 
    GrowFromCenter, 
    DrawBorderThenFill,
    GrowArrow,
    FadeOut,
    smooth,
    Circumscribe,
)


from manim import (
    Square, Text, Rectangle, Circle, Arrow, RoundedRectangle,
    SurroundingRectangle, MathTex, Tex, Dot,
    UP, RIGHT, DOWN, LEFT, ORIGIN,
    BLACK, WHITE,
    GRAY, GRAY_A, GRAY_B, GRAY_C, GRAY_D, GRAY_E,
    DARKER_GRAY, DARK_GRAY, LIGHT_GRAY,
    BLUE, BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_E,
    DARK_BLUE,
    TEAL_A, TEAL_B, TEAL_C, TEAL_D, TEAL_E,
    GREEN, GREEN_A, GREEN_B, GREEN_C, GREEN_D, GREEN_E,
    RED, RED_A, RED_B, RED_C, RED_D, RED_E,
    YELLOW, YELLOW_A, YELLOW_B, YELLOW_C, YELLOW_D, YELLOW_E,
    ORANGE,
    PURPLE, PURPLE_A, PURPLE_B, PURPLE_C, PURPLE_D, PURPLE_E,
)


class WorldModelAnimation(Scene):

    # ══════════════════════════════════════════════════════════════
    # Component builders
    # ══════════════════════════════════════════════════════════════

    def create_title(self) -> Tex:
        """
        Bold title 'World Models' centred at the top of the frame.
        """
        title = Tex(r"\textbf{World Models}", font_size=46, color=WHITE)
        title.to_edge(UP, buff=0.25)
        return title

    def create_grid_world(
        self, rows: int = 5, cols: int = 5, cell_size: float = 0.52
    ) -> tuple[VGroup, VGroup]:
        """
        5×5 environment grid.

        Cells are laid out on a regular grid, then the whole group is
        re-centred.  A rounded border and an 'Environment' label are
        added around the cells.

        Returns
        -------
        full_group : VGroup   – border + cells + label  (animate / position this)
        grid_cells : VGroup   – the 25 individual Square cells
                                indexed as  row * cols + col
        """
        grid_cells = VGroup()
        for r in range(rows):
            for c in range(cols):
                cell = Square(side_length=cell_size)
                cell.set_stroke(color=BLUE_D, width=1.5)
                cell.set_fill(color=DARK_BLUE, opacity=0.35)
                cell.move_to(np.array([c * cell_size, -r * cell_size, 0]))
                grid_cells.add(cell)
        grid_cells.move_to(ORIGIN)

        border = SurroundingRectangle(
            grid_cells, color=BLUE_B, buff=0.13, corner_radius=0.09
        )
        env_label = Tex(r"\text{Environment}", font_size=20, color=BLUE_C)
        env_label.next_to(grid_cells, DOWN, buff=0.18)

        full_group = VGroup(border, grid_cells, env_label)
        return full_group, grid_cells

    def create_agent(
        self, grid_cells: VGroup, row: int, col: int, cols: int = 5
    ) -> Circle:
        """
        Yellow filled circle that represents the agent.
        Placed at cell (row, col) of grid_cells.
        The returned Circle is *not* part of any group —
        add it to the scene separately.
        """
        cell = grid_cells[row * cols + col]
        agent = Circle(radius=0.17, fill_opacity=0.95)
        agent.set_fill(YELLOW).set_stroke(WHITE, width=2)
        agent.move_to(cell.get_center())
        return agent

    def create_history_box(self) -> tuple[VGroup, Rectangle]:
        """
        Rectangle that accumulates observation mini-frames.

        Layout
        ------
          [  History  ]    ← title above box
          ┌───────────┐
          │           │    ← mini-frames will be placed here dynamically
          └───────────┘
          (o₁,a₁,…,oₜ)    ← math notation below box

        Returns
        -------
        full_group   : VGroup     – position / animate this
        history_box  : Rectangle  – use .get_right() / .get_left() for arrows
        """
        history_box = Rectangle(width=3.0, height=0.88, color=GREEN_D)
        history_box.set_fill(GREEN_E, opacity=0.18)

        title = Tex(r"\textbf{History}", font_size=21, color=GREEN_B)
        title.next_to(history_box, UP, buff=0.12)

        notation = MathTex(
            r"(o_1,\, a_1,\, \ldots,\, o_t)", font_size=19, color=GREEN_C
        )
        notation.next_to(history_box, DOWN, buff=0.10)

        full_group = VGroup(history_box, title, notation)
        return full_group, history_box

    def create_mini_frame(self, color: ManimColor = BLUE_C) -> Square:
        """
        Small filled square representing one stored observation frame.
        Frames are positioned inside the history box by the caller.
        """
        frame = Square(side_length=0.26)
        frame.set_fill(color, opacity=0.78).set_stroke(WHITE, width=1)
        return frame

    def create_world_model_block(self) -> tuple[VGroup, RoundedRectangle]:
        """
        Central orange block containing the calligraphic 𝓕 symbol.

        Layout
        ------
          ┌──────────┐
          │    𝓕     │   ← RoundedRectangle + MathTex
          └──────────┘
           World Model    ← sub-label below

        Returns
        -------
        full_group : VGroup             – indices [0]=box, [1]=label, [2]=sub-label
        wm_box     : RoundedRectangle   – for arrow attachment & pulsing
        """
        wm_box = RoundedRectangle(
            width=1.55, height=1.55, corner_radius=0.18, color=ORANGE
        )
        wm_box.set_fill("#2E1000", opacity=0.88)

        f_label = MathTex(r"\mathcal{F}", font_size=54, color=ORANGE)
        f_label.move_to(wm_box.get_center())

        sub_label = Tex(r"\text{World Model}", font_size=18, color=ORANGE)
        sub_label.next_to(wm_box, DOWN, buff=0.14)

        full_group = VGroup(wm_box, f_label, sub_label)
        return full_group, wm_box

    def create_prediction_box(self) -> tuple[VGroup, Rectangle]:
        """
        Purple box that displays the predicted next observation ô_{t+1}.

        Returns
        -------
        full_group : VGroup     – indices [0]=box, [1]=title, [2]=pred_tex
        pred_box   : Rectangle  – for arrow attachment
        """
        pred_box = Rectangle(width=1.7, height=1.1, color=PURPLE_B)
        pred_box.set_fill(PURPLE_E, opacity=0.30)

        title = Tex(r"\textbf{Prediction}", font_size=21, color=PURPLE_B)
        title.next_to(pred_box, UP, buff=0.12)

        pred_tex = MathTex(r"\hat{o}_{t+1}", font_size=34, color=WHITE)
        pred_tex.move_to(pred_box.get_center())

        full_group = VGroup(pred_box, title, pred_tex)
        return full_group, pred_box

    # ══════════════════════════════════════════════════════════════
    # Main construct
    # ══════════════════════════════════════════════════════════════

    def construct(self):

        # ──────────────────────────────────────────────────────────
        # STEP 1 — Title + Grid World + Agent
        # ──────────────────────────────────────────────────────────
        title = self.create_title()
        self.play(Write(title), run_time=0.8)

        grid_group, grid_cells = self.create_grid_world()
        grid_group.scale(0.92).to_edge(LEFT, buff=0.5).shift(UP * 0.2)

        agent = self.create_agent(grid_cells, row=2, col=1)

        self.play(Create(grid_group), run_time=1.0)
        self.play(GrowFromCenter(agent), run_time=0.5)
        self.wait(0.6)

        # ──────────────────────────────────────────────────────────
        # STEP 2 — History box accumulates observation frames
        # ──────────────────────────────────────────────────────────
        history_group, history_box = self.create_history_box()
        # Place to the right of the grid world, vertically aligned
        history_group.next_to(grid_group, RIGHT, buff=0.85)

        self.play(FadeIn(history_group), run_time=0.6)

        # Three mini-frames appear one by one; each one also flashes
        # the matching grid cell (row 2, cols 1 → 3) to show "sensing"
        frame_colors = [ORANGE, YELLOW_C, RED_C]       # History mini-frame colors
        flash_colors = [BLUE_C, TEAL_C, GREEN_C]        # Environment cell flash colors
        mini_frames = VGroup()

        for i, (frame_color, flash_color) in enumerate(zip(frame_colors, flash_colors)):
            mf = self.create_mini_frame(color=frame_color)
            if i == 0:
                # First frame: anchor to left interior of the history box
                mf.move_to(history_box.get_left() + RIGHT * 0.32)
            else:
                mf.next_to(mini_frames[-1], RIGHT, buff=0.10)
            mini_frames.add(mf)

            # Flash the cell the agent is "reading"
            cell = grid_cells[2 * 5 + (1 + i)]   # row=2, col=1,2,3
            self.play(
                FadeIn(mf, shift=UP * 0.14),
                cell.animate.set_fill(flash_color, opacity=0.55),
                run_time=0.45,
            )

        self.wait(0.5)

        # ──────────────────────────────────────────────────────────
        # STEP 3 — World Model F block
        # ──────────────────────────────────────────────────────────
        wm_group, wm_box = self.create_world_model_block()
        wm_group.next_to(history_group, RIGHT, buff=1.1)

        self.play(
            DrawBorderThenFill(wm_group[0]),   # orange box
            Write(wm_group[1]),                 # 𝓕 label
            run_time=0.9,
        )
        self.play(FadeIn(wm_group[2]), run_time=0.4)   # "World Model"
        self.wait(0.4)

        # ──────────────────────────────────────────────────────────
        # STEP 4 — Data flow: history → F → prediction
        # ──────────────────────────────────────────────────────────
        pred_group, pred_box = self.create_prediction_box()
        pred_group.next_to(wm_group, RIGHT, buff=1.05)

        # Build arrows (positions computed after all blocks are placed)
        arrow_h_to_f = Arrow(
            start=history_box.get_right(),
            end=wm_box.get_left(),
            buff=0.12,
            color=GREEN_B,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.18,
        )
        arrow_f_to_p = Arrow(
            start=wm_box.get_right(),
            end=pred_box.get_left(),
            buff=0.12,
            color=ORANGE,
            stroke_width=2.5,
            max_tip_length_to_length_ratio=0.18,
        )

        # Reveal arrows and prediction box
        self.play(GrowArrow(arrow_h_to_f), run_time=0.65)
        self.play(FadeIn(pred_group), run_time=0.5)
        self.play(GrowArrow(arrow_f_to_p), run_time=0.65)

        # --- Packet 1: history → F --------------------------------
        packet1 = Dot(radius=0.09, color=GREEN_B)
        packet1.move_to(arrow_h_to_f.get_start())
        self.add(packet1)
        self.play(
            packet1.animate.move_to(arrow_h_to_f.get_end()),
            run_time=0.75, rate_func=smooth,
        )
        # F block pulses on arrival (simulates computation)
        self.play(
            wm_group[0].animate.set_fill(ORANGE, opacity=0.65),
            FadeOut(packet1),
            run_time=0.35,
        )
        self.play(
            wm_group[0].animate.set_fill("#2E1000", opacity=0.88),
            run_time=0.30,
        )

        # --- Packet 2: F → prediction -----------------------------
        packet2 = Dot(radius=0.09, color=ORANGE)
        packet2.move_to(arrow_f_to_p.get_start())
        self.add(packet2)
        self.play(
            packet2.animate.move_to(arrow_f_to_p.get_end()),
            run_time=0.75, rate_func=smooth,
        )
        # Prediction text flashes to signal a new prediction is ready
        self.play(
            pred_group[2].animate.set_color(YELLOW),
            FadeOut(packet2),
            run_time=0.35,
        )
        self.play(pred_group[2].animate.set_color(WHITE), run_time=0.30)
        self.wait(0.5)

        # ──────────────────────────────────────────────────────────
        # STEP 5 — Agent takes an action and moves
        # ──────────────────────────────────────────────────────────
        action_label = MathTex(r"a_t = \rightarrow", font_size=28, color=YELLOW)
        action_label.next_to(agent, UP, buff=0.22)
        self.play(FadeIn(action_label, shift=UP * 0.10), run_time=0.40)
        self.wait(0.25)

        # Slide agent one cell to the right: (row=2, col=1) → (row=2, col=2)
        target_cell = grid_cells[2 * 5 + 2]
        self.play(
            agent.animate.move_to(target_cell.get_center()),
            action_label.animate.shift(RIGHT * 0.52),
            run_time=0.65,
            rate_func=smooth,
        )

        # New yellow frame added to history; destination cell flashes
        new_frame = self.create_mini_frame(color=YELLOW_D)
        new_frame.next_to(mini_frames[-1], RIGHT, buff=0.10)
        mini_frames.add(new_frame)

        self.play(
            target_cell.animate.set_fill(YELLOW, opacity=0.45),
            FadeIn(new_frame, shift=UP * 0.12),
            FadeOut(action_label),
            run_time=0.55,
        )
        self.wait(0.5)

        # ──────────────────────────────────────────────────────────
        # STEP 6 — Final learning message
        # ──────────────────────────────────────────────────────────
        # Draw attention ring around the F block
        self.play(Circumscribe(wm_group[0], color=ORANGE), run_time=1.20)

        final_msg = Tex(
            r"\text{F learns to predict the future from experience}",
            font_size=25,
            color=WHITE,
        )
        final_msg.to_edge(DOWN, buff=0.60)
        self.play(Write(final_msg), run_time=1.40)
        self.wait(2.0)

        # Clean exit — fade the entire scene
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)