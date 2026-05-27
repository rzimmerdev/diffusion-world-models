from manim import *

PANEL_BG      = "#1a1a2e"
SSM_COLOR     = BLUE_D
SSM_FROZEN    = GREY
DIFFUSION_COLOR = "#f97316"   # orange
LOSS_COLOR    = RED_B
GRAD_COLOR    = GREEN_C
BORDER_ACTIVE = WHITE
BORDER_FROZEN = GREY_B
LOCK_COLOR    = "#e2b96a"


# ── Helper: build an SSM block ──────────────────────────────────────────────
def make_ssm_block(label="SSM\n(Mamba)", color=SSM_COLOR, border=BORDER_ACTIVE):
    box = RoundedRectangle(
        corner_radius=0.15, width=2.4, height=1.2,
        fill_color=color, fill_opacity=0.85,
        stroke_color=border, stroke_width=3,
    )
    lbl = Text(label, font_size=22, color=WHITE, weight=BOLD).move_to(box)
    return VGroup(box, lbl)


# ── Helper: sequence of input tokens ───────────────────────────────────────
def make_sequence(n=7, long=True):
    colors = [BLUE_B, TEAL_B, GREEN_B, YELLOW_B, ORANGE, RED_B, PURPLE_B]
    squares = VGroup(*[
        Square(side_length=0.3,
               fill_color=colors[i % len(colors)],
               fill_opacity=0.9, stroke_width=1)
        for i in range(n)
    ]).arrange(RIGHT, buff=0.06)
    if long:
        dots = Text("…", font_size=20).next_to(squares, RIGHT, buff=0.06)
        return VGroup(squares, dots)
    return squares


# ── Helper: padlock SVG-like shape (made from Manim primitives) ─────────────
def make_padlock(scale=0.5):
    body = RoundedRectangle(
        corner_radius=0.08, width=0.7, height=0.55,
        fill_color=LOCK_COLOR, fill_opacity=1.0,
        stroke_color=LOCK_COLOR, stroke_width=1,
    )
    shackle = Arc(
        radius=0.22, start_angle=0, angle=PI,
        stroke_color=LOCK_COLOR, stroke_width=5,
    ).next_to(body, UP, buff=-0.04)
    keyhole = Circle(
        radius=0.07,
        fill_color=PANEL_BG, fill_opacity=1.0,
        stroke_width=0,
    ).move_to(body).shift(UP * 0.05)
    lock = VGroup(body, shackle, keyhole).scale(scale)
    return lock


# ── Helper: UNet diffusion block ────────────────────────────────────────────
def make_unet_block(active=False):
    color = DIFFUSION_COLOR if active else GREY_C
    box = RoundedRectangle(
        corner_radius=0.15, width=2.4, height=1.2,
        fill_color=color, fill_opacity=0.85,
        stroke_color=color if active else GREY_B,
        stroke_width=4 if active else 2,
    )
    lbl = Text("Diffusion\nUNet", font_size=22, color=WHITE, weight=BOLD).move_to(box)
    return VGroup(box, lbl)


# ── Main Scene ───────────────────────────────────────────────────────────────
class TwoStageTraining(Scene):
    """
    Animates a two-panel timeline showing:
      Stage 1 – SSM trained on long sequences with a loss arrow.
      Stage 2 – SSM frozen (grey + padlock); diffusion UNet highlighted
                and trained; gradient arrows stop at the frozen SSM boundary.
    """

    def construct(self):
        self._show_title()
        self._build_stage1()
        self._transition_to_stage2()
        self.wait(2)

    # ── Title ────────────────────────────────────────────────────────────────
    def _show_title(self):
        title = Text(
            "Two-Stage World-Model Training",
            font_size=32, color=WHITE, weight=BOLD,
        ).to_edge(UP, buff=0.3)
        self.play(Write(title), run_time=0.8)
        self.wait(0.3)
        self.title = title

    # ── Stage 1 ──────────────────────────────────────────────────────────────
    def _build_stage1(self):
        # --- Panel background ---
        panel = self._make_panel(LEFT * 3.3)

        stage_lbl = Text("Stage 1", font_size=26, color=BLUE_B, weight=BOLD)\
            .move_to(panel).align_to(panel, UP).shift(DOWN * 0.25)

        subtitle = Text(
            "Train the Long-Context Branch",
            font_size=16, color=GREY_A,
        ).next_to(stage_lbl, DOWN, buff=0.12)

        # --- Input sequence ---
        seq = make_sequence(n=7, long=True)\
            .scale(0.9)\
            .move_to(panel).shift(UP * 1.0)

        seq_lbl = Tex(r"\text{long sequence }(T = 50)", font_size=14, color=GREY_A)\
            .next_to(seq, UP, buff=0.12)

        # --- Arrow: sequence → SSM ---
        ssm = make_ssm_block(label="SSM\n(Mamba)", color=SSM_COLOR)\
            .move_to(panel).shift(DOWN * 0.15)

        in_arrow = Arrow(
            seq.get_bottom(), ssm.get_top(),
            buff=0.1, color=WHITE, stroke_width=3,
        )

        # --- SSM output node ---
        out_dot = Dot(radius=0.1, color=YELLOW_B)\
            .next_to(ssm, DOWN, buff=0.5)

        out_arrow = Arrow(
            ssm.get_bottom(), out_dot.get_top(),
            buff=0.05, color=YELLOW_B, stroke_width=3,
        )

        out_lbl = Text("f̂_{t+1}", font_size=18, color=YELLOW_B)\
            .next_to(out_dot, RIGHT, buff=0.15)

        # --- Loss arrow (downward) ---
        loss_arrow = Arrow(
            out_dot.get_bottom(),
            out_dot.get_bottom() + DOWN * 1.2,
            buff=0.05, color=LOSS_COLOR, stroke_width=3,
        )
        loss_lbl = MathTex(r"\mathcal{L}_{\text{SSM}}", font_size=28, color=LOSS_COLOR)\
            .next_to(loss_arrow, DOWN, buff=0.1)

        # --- Animate Stage 1 ---
        self.play(FadeIn(panel), run_time=0.4)
        self.play(Write(stage_lbl), Write(subtitle), run_time=0.6)
        self.play(FadeIn(seq, seq_lbl), run_time=0.6)
        self.play(GrowArrow(in_arrow), run_time=0.5)
        self.play(FadeIn(ssm), run_time=0.6)
        self.play(GrowArrow(out_arrow), FadeIn(out_dot, out_lbl), run_time=0.5)
        self.play(GrowArrow(loss_arrow), Write(loss_lbl), run_time=0.7)

        # Pulse loss arrow to emphasise training
        self.play(
            loss_arrow.animate.set_color(RED),
            loss_lbl.animate.set_color(RED),
            rate_func=there_and_back, run_time=0.6,
        )
        self.wait(0.5)

        # Store for transition
        self.s1_group = VGroup(
            panel, stage_lbl, subtitle, seq, seq_lbl,
            in_arrow, ssm, out_arrow, out_dot, out_lbl,
            loss_arrow, loss_lbl,
        )
        self.s1_ssm = ssm          # keep SSM ref for freeze transition

    # ── Transition to Stage 2 ────────────────────────────────────────────────
    def _transition_to_stage2(self):
        # ---- Build Stage-2 panel (right side) first, then reveal ----
        panel2 = self._make_panel(RIGHT * 3.3)

        stage_lbl2 = Text("Stage 2", font_size=26, color=ORANGE, weight=BOLD)\
            .move_to(panel2).align_to(panel2, UP).shift(DOWN * 0.25)

        subtitle2 = Text(
            "Train the Generative Branch",
            font_size=16, color=GREY_A,
        ).next_to(stage_lbl2, DOWN, buff=0.12)

        # Short input sequence (length 4)
        seq2 = make_sequence(n=4, long=False)\
            .scale(0.9)\
            .move_to(panel2).shift(UP * 1.15)

        seq2_lbl = Tex(r"\text{short sequence }(T = 4)", font_size=14, color=GREY_A)\
            .next_to(seq2, UP, buff=0.12)

        # Frozen SSM block
        ssm_frozen = make_ssm_block(
            label="SSM\n(Mamba)", color=SSM_FROZEN, border=BORDER_FROZEN,
        ).move_to(panel2).shift(DOWN * 0.05 + LEFT * 0.0)

        # Padlock on frozen SSM
        lock = make_padlock(scale=0.55)\
            .next_to(ssm_frozen, UR, buff=-0.15)

        frozen_lbl = Text("frozen", font_size=14, color=GREY_B, slant=ITALIC)\
            .next_to(ssm_frozen, LEFT, buff=0.12)

        # SSM → feature output
        feat_dot = Dot(radius=0.1, color=GREY_B)\
            .next_to(ssm_frozen, DOWN, buff=0.45)

        feat_arrow_down = Arrow(
            ssm_frozen.get_bottom(), feat_dot.get_top(),
            buff=0.05, color=GREY_B, stroke_width=2.5,
        )

        feat_lbl = Text("context features", font_size=13, color=GREY_B)\
            .next_to(feat_dot, RIGHT, buff=0.12)

        # Diffusion UNet (highlighted)
        unet = make_unet_block(active=True)\
            .next_to(feat_dot, DOWN, buff=0.45)

        feat_to_unet = Arrow(
            feat_dot.get_bottom(), unet.get_top(),
            buff=0.05, color=ORANGE, stroke_width=3,
        )

        # Gradient flow arrows (going UP, stopping at SSM boundary)
        def grad_arrow(start, end, dashed=False):
            if dashed:
                return DashedLine(
                    start, end,
                    dash_length=0.12, dashed_ratio=0.6,
                    color=GRAD_COLOR, stroke_width=2.5,
                ).add_tip(tip_length=0.18)
            return Arrow(start, end,
                         buff=0.05, color=GRAD_COLOR, stroke_width=2.5)

        # Solid gradient: inside UNet upward
        grad1 = grad_arrow(
            unet.get_top() + LEFT * 0.5,
            feat_dot.get_bottom() + LEFT * 0.5,
        )
        # Dashed gradient: trying to cross into frozen SSM – stops at border
        grad_blocked_start = feat_dot.get_top() + LEFT * 0.5
        grad_blocked_end   = ssm_frozen.get_bottom() + LEFT * 0.5 + DOWN * 0.05
        grad2 = DashedLine(
            grad_blocked_start, grad_blocked_end,
            dash_length=0.12, dashed_ratio=0.6,
            color=GRAD_COLOR, stroke_width=2.5,
        )  # no tip – it fades out / is blocked

        # Red "stop" cross at frozen SSM boundary
        stop_pos = grad_blocked_end + UP * 0.05
        stop_cross = VGroup(
            Line(stop_pos + UL * 0.12, stop_pos + DR * 0.12,
                 color=RED, stroke_width=4),
            Line(stop_pos + UR * 0.12, stop_pos + DL * 0.12,
                 color=RED, stroke_width=4),
        )

        grad_lbl = Text("∇ gradient flow", font_size=13, color=GRAD_COLOR)\
            .next_to(grad1, LEFT, buff=0.1)

        # ---- Animate Stage 2 ----
        self.play(FadeIn(panel2), run_time=0.4)
        self.play(Write(stage_lbl2), Write(subtitle2), run_time=0.6)
        self.play(FadeIn(seq2, seq2_lbl), run_time=0.5)

        # Short input → frozen SSM
        in_arr2 = Arrow(
            seq2.get_bottom(), ssm_frozen.get_top(),
            buff=0.1, color=GREY_B, stroke_width=2.5,
        )
        self.play(GrowArrow(in_arr2), run_time=0.4)

        # Freeze the SSM: fade to grey + lock appears
        self.play(
            FadeIn(ssm_frozen),
            run_time=0.6,
        )
        self.play(
            FadeIn(lock, scale=1.3),
            FadeIn(frozen_lbl),
            run_time=0.5,
        )

        # Feature flow
        self.play(
            GrowArrow(feat_arrow_down),
            FadeIn(feat_dot, feat_lbl),
            run_time=0.5,
        )

        # Highlight and show UNet
        self.play(GrowArrow(feat_to_unet), run_time=0.4)
        self.play(FadeIn(unet), run_time=0.5)

        # Pulse UNet to emphasise it's being trained
        self.play(
            unet[0].animate.set_stroke(color=YELLOW, width=6),
            rate_func=there_and_back, run_time=0.5,
        )

        # Gradient flow arrows
        self.play(GrowArrow(grad1), Write(grad_lbl), run_time=0.6)

        # Dashed gradient approaching frozen SSM
        self.play(Create(grad2), run_time=0.8)

        # Stop-cross at SSM boundary
        self.play(Create(stop_cross), run_time=0.4)

        # Emphasise the block
        self.play(
            stop_cross.animate.scale(1.4).set_color(RED),
            rate_func=there_and_back, run_time=0.4,
        )

        # Final callout label
        blocked_note = Text(
            "gradients\nblocked", font_size=13, color=GRAD_COLOR,
        ).next_to(stop_cross, DL, buff=0.15)
        self.play(FadeIn(blocked_note), run_time=0.4)

        self.s2_group = VGroup(
            panel2, stage_lbl2, subtitle2, seq2, seq2_lbl,
            in_arr2, ssm_frozen, lock, frozen_lbl,
            feat_arrow_down, feat_dot, feat_lbl,
            feat_to_unet, unet,
            grad1, grad2, stop_cross, grad_lbl, blocked_note,
        )

    # ── Utility ──────────────────────────────────────────────────────────────
    def _make_panel(self, center_shift) -> RoundedRectangle:
        panel = RoundedRectangle(
            corner_radius=0.2, width=5.8, height=6.2,
            fill_color=PANEL_BG, fill_opacity=0.6,
            stroke_color="#3a3a5c", stroke_width=2,
        ).shift(center_shift + DOWN * 0.2)
        return panel