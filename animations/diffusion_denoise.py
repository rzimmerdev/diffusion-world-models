from manim import *
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

T_STEPS   = 5          # t = 0 … T  → 6 frames total (fits comfortably)
IMG_PATH  = "media/images/cat.jpg"  # placed next to the script at render time

# Physical size of each image thumbnail on screen (Manim units)
FRAME_W   = 1.6
FRAME_H   = 1.6

# Gap between frames reserved for arrows
ARROW_GAP = 0.60

# Vertical centre of the thumbnail strip
STRIP_Y   = 0.15


# ---------------------------------------------------------------------------
# Noise helper
# ---------------------------------------------------------------------------

def _noisy_image_array(base_array: np.ndarray, noise_level: int) -> np.ndarray:
    """
    Returns a uint8 RGB(A) array with Gaussian noise blended in.
      noise_level = 0  → pure original
      noise_level = T  → almost pure white-noise
    The RNG is seeded deterministically so every call at the same level is
    identical (important: Manim calls this during construction, not just once).
    """
    alpha = noise_level / T_STEPS          # 0.0 … 1.0
    rng   = np.random.default_rng(seed=noise_level * 31 + 7)

    arr   = base_array.astype(np.float32)
    noise = rng.normal(loc=128.0, scale=60.0, size=arr.shape).astype(np.float32)

    blended = (1.0 - alpha) * arr + alpha * noise
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

def frame_label(t: int) -> MathTex:
    if t == 0:
        tex = r"x_0"
    elif t == T_STEPS:
        tex = r"x_T"
    else:
        tex = rf"x_{{{t}}}"
    return MathTex(tex, font_size=28, color=WHITE)


def beta_label(t: int) -> MathTex:
    return MathTex(rf"\beta_{{{t}}}", font_size=22, color=YELLOW_B)


def eps_label() -> MathTex:
    return MathTex(r"\epsilon_\theta", font_size=22, color=TEAL_B)


# ---------------------------------------------------------------------------
# Main Scene
# ---------------------------------------------------------------------------

class DiffusionProcessAnimation(Scene):
    """
    Visualises the diffusion forward / reverse process using cat.png.

    Strip layout  (6 thumbnails, T_STEPS = 5)
    ──────────────────────────────────────────
      β_1 ↗  β_2 ↗  β_3 ↗  β_4 ↗  β_5 ↗        ← arrows above
    [x_0] [x_1] [x_2] [x_3] [x_4] [x_T]          ← image strip
      ε_θ ↙  ε_θ ↙  ε_θ ↙  ε_θ ↙  ε_θ ↙         ← arrows below (reverse)

    Sequence
    ────────
    1.  Title + direction labels appear.
    2.  x_0 (clean cat) fades in.
    3.  Forward corruption: arrow + β_t label → noisy frame, left→right.
    4.  "Pure Gaussian noise" callout at x_T.
    5.  Reverse denoising: ε_θ arrow → cleaner frame, right→left.
        Each step cross-dissolves the noisy thumbnail to the cleaner one.
    6.  "Generation complete" replaces direction labels; x_0 border pulses green.
    7.  Context-window brace on the last 2 frames (red).
    8.  Fade-out.
    """

    BG = "#0f0f1a"

    def setup(self):
        self.camera.background_color = self.BG

    # ── layout ──────────────────────────────────────────────────────────────

    @staticmethod
    def _frame_x(idx: int) -> float:
        """Horizontal centre of the idx-th frame (0-based)."""
        n       = T_STEPS + 1
        step    = FRAME_W + ARROW_GAP
        total_w = step * (n - 1)
        return -total_w / 2 + idx * step

    # ── construction ────────────────────────────────────────────────────────

    def construct(self):
        n_frames = T_STEPS + 1   # 6

        # ── Load base image & pre-compute noisy arrays ───────────────────────
        # ImageMobject reads the file; we grab its pixel array immediately.
        base_img = ImageMobject(IMG_PATH)
        base_arr = base_img.get_pixel_array().copy()   # uint8, shape (H, W, 4)

        def make_frame(t: int) -> ImageMobject:
            """Return a positioned, sized ImageMobject at noise level t."""
            if t == 0:
                img = ImageMobject(IMG_PATH)
            else:
                noisy_arr = _noisy_image_array(base_arr, t)
                img = ImageMobject(noisy_arr)
            img.set_resampling_algorithm(RESAMPLING_ALGORITHMS["nearest"])
            img.width  = FRAME_W
            img.height = FRAME_H
            img.move_to([self._frame_x(t), STRIP_Y, 0])
            return img

        def make_border(t: int, color=GREY_C) -> Rectangle:
            b = Rectangle(
                width=FRAME_W + 0.06, height=FRAME_H + 0.06,
                stroke_color=color, stroke_width=1.8,
                fill_opacity=0,
            )
            b.move_to([self._frame_x(t), STRIP_Y, 0])
            return b

        # Pre-build all forward frames up-front (deterministic noise)
        fwd_frames  = [make_frame(t) for t in range(n_frames)]
        borders     = [make_border(t) for t in range(n_frames)]
        borders_vg  = VGroup(*borders)

        t_labels = VGroup(*[
            frame_label(t).move_to([
                self._frame_x(t),
                STRIP_Y - FRAME_H / 2 - 0.34,
                0,
            ])
            for t in range(n_frames)
        ])

        # ── Title ────────────────────────────────────────────────────────────
        title = Text(
            "Diffusion Models: Forward & Reverse Process",
            font_size=30, color=WHITE,
        ).to_edge(UP, buff=0.28)
        self.play(Write(title), run_time=0.8)

        # ── Direction labels ─────────────────────────────────────────────────
        fwd_lbl = Text("Forward  (corruption)", font_size=20, color=YELLOW_B)
        rev_lbl = Text("Reverse  (generation)", font_size=20, color=TEAL_B)
        fwd_lbl.next_to(title, DOWN, buff=0.18).shift(LEFT * 2.8)
        rev_lbl.next_to(title, DOWN, buff=0.18).shift(RIGHT * 2.8)
        self.play(FadeIn(fwd_lbl), FadeIn(rev_lbl), run_time=0.5)

        # ── Show x_0 ─────────────────────────────────────────────────────────
        self.play(
            FadeIn(fwd_frames[0]),
            Create(borders[0]),
            Write(t_labels[0]),
            run_time=0.8,
        )
        self.wait(0.35)

        # ── FORWARD PASS ─────────────────────────────────────────────────────
        arrow_y_fwd = STRIP_Y + FRAME_H / 2 + 0.38   # above the strip

        fwd_arrows: list[Arrow]     = []
        fwd_blabels: list[MathTex]  = []

        for t in range(1, n_frames):
            x0 = self._frame_x(t - 1)
            x1 = self._frame_x(t)

            arrow = Arrow(
                start=[x0 + FRAME_W / 2 + 0.05, arrow_y_fwd, 0],
                end  =[x1 - FRAME_W / 2 - 0.05, arrow_y_fwd, 0],
                buff=0,
                stroke_width=2.5,
                color=YELLOW_B,
                max_tip_length_to_length_ratio=0.28,
            )
            blbl = beta_label(t)
            blbl.next_to(arrow, UP, buff=0.07)

            fwd_arrows.append(arrow)
            fwd_blabels.append(blbl)

            self.play(GrowArrow(arrow), Write(blbl), run_time=0.32)
            self.play(
                FadeIn(fwd_frames[t]),
                Create(borders[t]),
                Write(t_labels[t]),
                run_time=0.42,
            )

        self.wait(0.5)

        # ── Pure-noise callout ────────────────────────────────────────────────
        noise_call = Text("Pure Gaussian noise", font_size=18, color=GREY_A)
        noise_call.next_to(fwd_frames[-1], DOWN, buff=0.52)
        self.play(FadeIn(noise_call, shift=UP * 0.12), run_time=0.45)
        self.wait(0.45)
        self.play(FadeOut(noise_call), run_time=0.28)

        # ── REVERSE PASS ─────────────────────────────────────────────────────
        arrow_y_rev = STRIP_Y - FRAME_H / 2 - 0.52   # below the strip

        # Highlight x_T to mark the start of reverse
        self.play(
            borders[-1].animate.set_stroke(TEAL_B, width=2.8),
            run_time=0.38,
        )

        rev_frames:  list[ImageMobject] = []
        rev_arrows:  list[Arrow]        = []
        rev_elabels: list[MathTex]      = []

        for t in range(T_STEPS - 1, -1, -1):
            x_src = self._frame_x(t + 1)
            x_tgt = self._frame_x(t)

            rev_arrow = Arrow(
                start=[x_src - FRAME_W / 2 - 0.05, arrow_y_rev, 0],
                end  =[x_tgt + FRAME_W / 2 + 0.05, arrow_y_rev, 0],
                buff=0,
                stroke_width=2.5,
                color=TEAL_B,
                max_tip_length_to_length_ratio=0.28,
            )
            elbl = eps_label()
            elbl.next_to(rev_arrow, DOWN, buff=0.07)

            rev_arrows.append(rev_arrow)
            rev_elabels.append(elbl)

            # Denoised frame at level t (same as forward frame[t])
            denoised = make_frame(t)
            rev_frames.append(denoised)

            border_color = GREEN if t == 0 else TEAL_C

            self.play(GrowArrow(rev_arrow), Write(elbl), run_time=0.32)
            self.play(
                FadeOut(fwd_frames[t]),   # noisy version fades out …
                FadeIn(denoised),         # … cleaner version fades in
                borders[t].animate.set_stroke(border_color, width=2.0),
                run_time=0.42,
            )

        self.wait(0.45)

        # ── Generation complete ───────────────────────────────────────────────
        gen_lbl = Text(
            "Generation complete  —  high-fidelity image recovered",
            font_size=21, color=GREEN_B,
        ).next_to(title, DOWN, buff=0.18)

        self.play(FadeOut(fwd_lbl), FadeOut(rev_lbl), run_time=0.28)
        self.play(Write(gen_lbl), run_time=0.75)

        # Pulse x_0 border green
        self.play(borders[0].animate.set_stroke(GREEN, width=4.0), run_time=0.35)
        self.play(borders[0].animate.set_stroke(GREEN_B, width=2.2), run_time=0.35)

        self.wait(0.8)

        # ── Context-window brace (last 2 frames) ─────────────────────────────
        brace_y   = STRIP_Y + FRAME_H / 2 + 0.88
        brace_x0  = self._frame_x(T_STEPS - 1) - FRAME_W / 2
        brace_x1  = self._frame_x(T_STEPS)     + FRAME_W / 2

        brace = BraceBetweenPoints(
            [brace_x0, brace_y - 0.08, 0],
            [brace_x1, brace_y - 0.08, 0],
            direction=UP, color=RED_B,
        )
        brace_lbl = Text("Short context window only", font_size=17, color=RED_B)
        brace_lbl.next_to(brace, UP, buff=0.07)

        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_lbl, shift=UP * 0.1),
            run_time=0.65,
        )

        self.wait(2.0)

        # ── Fade out ─────────────────────────────────────────────────────────
        self.play(
            FadeOut(Group(
                title, gen_lbl,
                brace, brace_lbl,
                *fwd_arrows, *fwd_blabels,
                *rev_arrows, *rev_elabels,
                *rev_frames,
                *fwd_frames,
                borders_vg, t_labels,
            )),
            run_time=1.1,
        )
        self.wait(0.3)