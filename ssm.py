from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

BG          = "#0f0f1a"

# Palette for the three section acts
C_H         = "#7C83FD"   # hidden state   – periwinkle blue
C_F         = "#FD7C83"   # input token    – coral
C_OUT       = "#7CFD9A"   # output         – mint green
C_MATRIX    = "#FFD97D"   # A, B, C labels – amber
C_MAMBA     = "#CF9FFF"   # Mamba accent   – soft purple
C_TRANSF    = "#FF6B6B"   # Transformer    – red
C_SSM_CMP   = "#7C83FD"   # SSM in compare – periwinkle

T           = 5           # number of time steps shown in the chain


# ---------------------------------------------------------------------------
# Reusable sub-builders
# ---------------------------------------------------------------------------

def h_box(label: str, width=0.55, height=0.38) -> VGroup:
    """A coloured rectangle + label representing h_t."""
    rect = Rectangle(
        width=width, height=height,
        fill_color=C_H, fill_opacity=0.85,
        stroke_color=WHITE, stroke_width=1.4,
    )
    lbl = MathTex(label, font_size=18, color=WHITE)
    lbl.move_to(rect)
    return VGroup(rect, lbl)


def f_box(label: str, width=0.45, height=0.32) -> VGroup:
    """A coloured rectangle + label representing f_t (input)."""
    rect = Rectangle(
        width=width, height=height,
        fill_color=C_F, fill_opacity=0.85,
        stroke_color=WHITE, stroke_width=1.2,
    )
    lbl = MathTex(label, font_size=17, color=WHITE)
    lbl.move_to(rect)
    return VGroup(rect, lbl)


def section_title(text: str, color=WHITE) -> Text:
    return Text(text, font_size=24, color=color, weight=BOLD)


# ---------------------------------------------------------------------------
# Main Scene
# ---------------------------------------------------------------------------

class SSMAnimation(Scene):
    """
    Three-act animation explaining State-Space Models (SSMs / Mamba).

    Act 1 – Recurrence chain
        Draw t = 1 … T as h boxes connected by arrows.
        f_t inputs drop in from above at each step.
        Annotate the recurrence  h_t = A h_{t-1} + B f_t.

    Act 2 – Mamba equations
        Show the selective-gating equations in full.
        Highlight Δ, B, C = Linear(f_t) and  Ā = e^{ΔA}.

    Act 3 – Comparison column
        Left:  Transformer — re-reads ALL previous tokens (growing cost).
        Right: SSM         — only needs h_{t-1} + f_t   (constant cost).
        Highlight the constant memory footprint with a brace.
    """

    def setup(self):
        self.camera.background_color = BG

    # -----------------------------------------------------------------------
    # Act 1 – Recurrence chain
    # -----------------------------------------------------------------------

    def _act1(self):
        # ── section label ──────────────────────────────────────────────────
        lbl = section_title("State-Space Model: Recurrence", color=C_H)
        lbl.to_edge(UP, buff=0.30)
        self.play(Write(lbl), run_time=0.7)

        # ── chain layout ───────────────────────────────────────────────────
        STEP_X  = 1.85      # horizontal spacing between h boxes
        CHAIN_Y = 0.35      # vertical centre of the h chain
        F_DY    = 0.92      # f boxes sit this far ABOVE h boxes

        # h_0 … h_T
        h_boxes: list[VGroup] = []
        for t in range(T + 1):
            hb = h_box(rf"h_{t}" if t > 0 else r"h_0")
            hb.move_to([STEP_X * t - STEP_X * T / 2, CHAIN_Y, 0])
            h_boxes.append(hb)

        # f_1 … f_T  (one per transition)
        f_boxes: list[VGroup] = []
        for t in range(1, T + 1):
            fb = f_box(rf"f_{t}")
            fb.move_to([
                (h_boxes[t - 1].get_center()[0] + h_boxes[t].get_center()[0]) / 2,
                CHAIN_Y + F_DY,
                0,
            ])
            f_boxes.append(fb)

        # ── animate h_0 ────────────────────────────────────────────────────
        self.play(FadeIn(h_boxes[0], scale=0.8), run_time=0.55)

        # ── grow chain step by step ────────────────────────────────────────
        h_arrows:  list[Arrow]  = []
        f_arrows:  list[Arrow]  = []

        for i in range(T):
            fb  = f_boxes[i]
            hb  = h_boxes[i + 1]
            src = h_boxes[i]

            # Arrow: h_{t-1} → h_t
            h_arr = Arrow(
                src.get_right(), hb.get_left(),
                buff=0.07, stroke_width=2.2, color=C_H,
                max_tip_length_to_length_ratio=0.22,
            )
            # Arrow: f_t ↓ into h_t
            f_arr = Arrow(
                fb.get_bottom(), hb.get_top(),
                buff=0.07, stroke_width=2.0, color=C_F,
                max_tip_length_to_length_ratio=0.22,
            )

            h_arrows.append(h_arr)
            f_arrows.append(f_arr)

            self.play(
                FadeIn(fb, shift=DOWN * 0.15),
                GrowArrow(h_arr),
                run_time=0.38,
            )
            self.play(
                GrowArrow(f_arr),
                FadeIn(hb, scale=0.8),
                run_time=0.38,
            )

        self.wait(0.4)

        # ── recurrence equation annotation ────────────────────────────────
        eq_recur = MathTex(
            r"h_t", r"=", r"A", r"\,h_{t-1}", r"+", r"B", r"\,f_t",
            font_size=32,
        )
        eq_recur.set_color_by_tex(r"h_t",      C_H)
        eq_recur.set_color_by_tex(r"h_{t-1}",  C_H)
        eq_recur.set_color_by_tex(r"f_t",       C_F)
        eq_recur.set_color_by_tex(r"A",         C_MATRIX)
        eq_recur.set_color_by_tex(r"B",         C_MATRIX)
        eq_recur.move_to([0, CHAIN_Y - 1.15, 0])

        eq_out = MathTex(
            r"m_t", r"=", r"C", r"\,h_t",
            font_size=32,
        )
        eq_out.set_color_by_tex(r"m_t", C_OUT)
        eq_out.set_color_by_tex(r"h_t", C_H)
        eq_out.set_color_by_tex(r"C",   C_MATRIX)
        eq_out.next_to(eq_recur, RIGHT, buff=0.9)

        comma = MathTex(r",", font_size=32, color=WHITE)
        comma.move_to(
            (eq_recur.get_right() + eq_out.get_left()) / 2
        )

        abc_note = Tex(
            r"\textit{A, B, C} are learned matrices",
            font_size=20, color=C_MATRIX,
        )
        abc_note.next_to(eq_recur, DOWN, buff=0.32)

        self.play(Write(eq_recur), run_time=0.9)
        self.play(Write(comma), Write(eq_out), run_time=0.6)
        self.play(FadeIn(abc_note, shift=UP * 0.1), run_time=0.55)
        self.wait(0.6)

        # Pulse the A, B, C labels in the equation
        self.play(
            eq_recur.animate.set_color_by_tex("A", YELLOW),
            eq_recur.animate.set_color_by_tex("B", YELLOW),
            eq_out.animate.set_color_by_tex("C",   YELLOW),
            run_time=0.4,
        )
        self.play(
            eq_recur.animate.set_color_by_tex("A", C_MATRIX),
            eq_recur.animate.set_color_by_tex("B", C_MATRIX),
            eq_out.animate.set_color_by_tex("C",   C_MATRIX),
            run_time=0.4,
        )
        self.wait(0.5)

        # pack for cleanup
        chain_group = Group(
            *h_boxes, *f_boxes, *h_arrows, *f_arrows,
            eq_recur, eq_out, comma, abc_note,
        )
        return lbl, chain_group

    # -----------------------------------------------------------------------
    # Act 2 – Mamba selective gating
    # -----------------------------------------------------------------------

    def _act2(self, prev_lbl, prev_group):
        self.play(
            FadeOut(prev_group),
            prev_lbl.animate.become(
                section_title("Mamba: Selective Gating", color=C_MAMBA)
                .to_edge(UP, buff=0.30)
            ),
            run_time=0.6,
        )

        # Main equations stacked
        eq1 = MathTex(
            r"\Delta, B, C",
            r"=",
            r"\text{Linear}(f_t)",
            font_size=34,
        )
        eq1.set_color_by_tex(r"\Delta, B, C", C_MATRIX)
        eq1.set_color_by_tex(r"f_t",           C_F)

        eq2 = MathTex(
            r"\bar{A}",
            r"=",
            r"e^{\Delta A}",
            font_size=34,
        )
        eq2.set_color_by_tex(r"\bar{A}", C_MAMBA)
        eq2.set_color_by_tex(r"A",       C_MATRIX)

        eq3 = MathTex(
            r"h_t",
            r"=",
            r"\bar{A}\,h_{t-1}",
            r"+",
            r"\bar{B}\,f_t",
            font_size=34,
        )
        eq3.set_color_by_tex(r"h_t",    C_H)
        eq3.set_color_by_tex(r"h_{t-1}", C_H)
        eq3.set_color_by_tex(r"f_t",     C_F)
        eq3.set_color_by_tex(r"\bar{A}", C_MAMBA)
        eq3.set_color_by_tex(r"\bar{B}", C_MAMBA)

        eqs = VGroup(eq1, eq2, eq3).arrange(DOWN, buff=0.55, aligned_edge=LEFT)
        eqs.move_to([0, 0.2, 0])

        # Annotation bullets
        note1 = Tex(
            r"$\Delta$ controls \textit{how much} to advance the state",
            font_size=20, color=WHITE,
        )
        note2 = Tex(
            r"$B, C$ are input-dependent i.e. dynamic per token",
            font_size=20, color=WHITE,
        )
        note3 = Tex(
            r"Expressive advantage over LSTMs / GRUs",
            font_size=20, color=C_MAMBA,
        )
        notes = VGroup(note1, note2, note3).arrange(DOWN, buff=0.22, aligned_edge=LEFT)
        notes.next_to(eqs, DOWN, buff=0.48)

        self.play(Write(eq1), run_time=0.8)
        self.play(Write(eq2), run_time=0.7)
        self.play(Write(eq3), run_time=0.8)
        self.wait(0.3)
        self.play(LaggedStart(
            FadeIn(note1, shift=RIGHT * 0.12),
            FadeIn(note2, shift=RIGHT * 0.12),
            FadeIn(note3, shift=RIGHT * 0.12),
            lag_ratio=0.35,
        ), run_time=1.0)
        self.wait(0.8)

        mamba_group = VGroup(eqs, notes)
        return prev_lbl, mamba_group

    # -----------------------------------------------------------------------
    # Act 3 – Transformer vs SSM cost comparison
    # -----------------------------------------------------------------------

    def _act3(self, prev_lbl, prev_group):
        self.play(
            FadeOut(prev_group),
            prev_lbl.animate.become(
                section_title("Memory Cost: Transformer vs SSM", color=WHITE)
                .to_edge(UP, buff=0.30)
            ),
            run_time=0.6,
        )

        # ── Column headers ─────────────────────────────────────────────────
        LEFT_X  = -3.2
        RIGHT_X =  3.2
        COL_Y   =  2.6   # just below title

        transf_hdr = Text("Transformer", font_size=26, color=C_TRANSF, weight=BOLD)
        ssm_hdr    = Text("SSM  (Mamba)", font_size=26, color=C_SSM_CMP, weight=BOLD)
        transf_hdr.move_to([LEFT_X,  COL_Y, 0])
        ssm_hdr.move_to([RIGHT_X, COL_Y, 0])

        divider = Line(
            [0, COL_Y + 0.35, 0], [0, -2.0, 0],
            stroke_color=GREY_C, stroke_width=1.2,
        )

        self.play(
            Write(transf_hdr), Write(ssm_hdr),
            Create(divider),
            run_time=0.65,
        )

        # ── Transformer side – growing token sequence ──────────────────────
        # Show t = 1 … 5: at each step ALL previous tokens are re-read
        T_CMP  = 5
        BOX_W  = 0.38
        BOX_H  = 0.30
        ROW_DY = 0.68     # vertical spacing between rows
        ROW_Y0 = 1.80     # top row y

        # We'll show each "attention over all tokens" as a row of coloured boxes
        # with a brace spanning them all
        transf_rows: list[VGroup] = []
        transf_braces: list[Brace] = []
        transf_costs: list[MathTex] = []

        for step in range(1, T_CMP + 1):
            row_y = ROW_Y0 - (step - 1) * ROW_DY
            row_boxes = VGroup()
            for k in range(step):
                color = interpolate_color(ManimColor(C_TRANSF), ManimColor("#440000"), k / max(T_CMP - 1, 1))
                b = Rectangle(
                    width=BOX_W, height=BOX_H,
                    fill_color=color, fill_opacity=0.85,
                    stroke_color=WHITE, stroke_width=0.8,
                )
                b.move_to([LEFT_X - (step - 1) * (BOX_W + 0.06) / 2
                            + k * (BOX_W + 0.06), row_y, 0])
                row_boxes.add(b)

            # brace underneath spanning the full row
            brace = Brace(row_boxes, RIGHT, buff=0.06, color=C_TRANSF)
            cost  = MathTex(rf"O({step})", font_size=18, color=C_TRANSF)
            cost.next_to(brace, RIGHT, buff=0.05)

            transf_rows.append(row_boxes)
            transf_braces.append(brace)
            transf_costs.append(cost)

        # ── SSM side – constant h_{t-1} + f_t ─────────────────────────────
        ssm_rows:  list[VGroup]   = []
        ssm_costs: list[MathTex]  = []

        for step in range(1, T_CMP + 1):
            row_y = ROW_Y0 - (step - 1) * ROW_DY

            # always just: [h_{t-1}]  +  [f_t]  →  [h_t]
            hb_prev = Rectangle(
                width=BOX_W + 0.05, height=BOX_H,
                fill_color=C_H, fill_opacity=0.85,
                stroke_color=WHITE, stroke_width=0.8,
            )
            hb_prev.move_to([RIGHT_X - 0.58, row_y, 0])

            plus = MathTex(r"+", font_size=20, color=WHITE)
            plus.move_to([RIGHT_X, row_y, 0])

            fb = Rectangle(
                width=BOX_W, height=BOX_H,
                fill_color=C_F, fill_opacity=0.85,
                stroke_color=WHITE, stroke_width=0.8,
            )
            fb.move_to([RIGHT_X + 0.52, row_y, 0])

            cost = MathTex(r"O(1)", font_size=18, color=C_SSM_CMP)
            cost.move_to([RIGHT_X + 1.3, row_y, 0])

            row_grp = VGroup(hb_prev, plus, fb)
            ssm_rows.append(row_grp)
            ssm_costs.append(cost)

        # ── Animate both columns in lockstep ──────────────────────────────
        for step in range(T_CMP):
            self.play(
                LaggedStart(
                    AnimationGroup(
                        FadeIn(transf_rows[step]),
                        FadeIn(transf_braces[step]),
                        Write(transf_costs[step]),
                    ),
                    AnimationGroup(
                        FadeIn(ssm_rows[step]),
                        Write(ssm_costs[step]),
                    ),
                    lag_ratio=0.15,
                ),
                run_time=0.52,
            )

        self.wait(0.5)

        # ── "Constant memory" highlight brace on SSM side ─────────────────
        ssm_all_rows = Group(*ssm_rows)
        ssm_all_costs = Group(*ssm_costs)
        mem_brace = Brace(ssm_all_costs, RIGHT, buff=0.12, color=GREEN)
        
        mem_lbl   = Text("Constant\nmemory", font_size=19, color=GREEN)
        mem_lbl.next_to(mem_brace, RIGHT, buff=0.12)

        self.play(
            GrowFromCenter(mem_brace),
            FadeIn(mem_lbl, shift=LEFT * 0.1),
            run_time=0.7,
        )

        # ── Growing-cost highlight on Transformer side ────────────────────
        grow_brace = Brace(Group(*transf_rows), LEFT, buff=0.12, color=C_TRANSF)
        grow_lbl   = Text("Growing\ncost", font_size=19, color=C_TRANSF)
        grow_lbl.next_to(grow_brace, LEFT, buff=0.12)

        self.play(
            GrowFromCenter(grow_brace),
            FadeIn(grow_lbl, shift=RIGHT * 0.1),
            run_time=0.7,
        )

        self.wait(0.5)

        # ── Bottom summary line ────────────────────────────────────────────
        summary = MathTex(
            r"\text{SSM:}\quad |h_t| = \text{const}",
            r"\quad\Rightarrow\quad",
            r"\text{scales to arbitrarily long sequences}",
            font_size=24,
        )
        summary[0].set_color(C_SSM_CMP)
        summary[2].set_color(GREEN_B)
        summary.to_edge(DOWN, buff=1.0)

        self.play(Write(summary), run_time=1.0)
        self.wait(2.2)

        # ── Fade out ──────────────────────────────────────────────────────
        self.play(
            FadeOut(Group(
                prev_lbl,
                transf_hdr, ssm_hdr, divider,
                *transf_rows, *transf_braces, *transf_costs,
                *ssm_rows, *ssm_costs,
                mem_brace, mem_lbl,
                grow_brace, grow_lbl,
                summary,
            )),
            run_time=1.0,
        )

    # -----------------------------------------------------------------------
    # construct
    # -----------------------------------------------------------------------

    def construct(self):
        # Act 1
        lbl, chain = self._act1()
        self.wait(0.3)

        # Act 2
        lbl, mamba = self._act2(lbl, chain)
        self.wait(0.3)

        # Act 3
        self._act3(lbl, mamba)
        self.wait(0.3)