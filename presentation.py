import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="full",
    layout_file="layouts/presentation.slides.json",
    css_file="custom.css",
)


@app.cell
def _():
    import logging

    logging.getLogger("manim").setLevel(logging.ERROR)
    return


@app.cell
def _():
    from world_model_animation import WorldModelAnimation

    scene_world_model = WorldModelAnimation()
    _scene_world_model = scene_world_model.render()
    return (scene_world_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("""
        # StateSpaceDiffuser
        ## Bringing Long Context to Diffusion World Models

        **Nedko Savov · Naser Kazemi · Deheng Zhang · Danda Pani Paudel · Xi Wang · Luc Van Gool**

        INSAIT, Sofia University · ETH Zurich · TU Munich

        *arXiv:2505.22246 — October 2025*
        """),
            mo.callout(
                mo.md(
                    "This seminar walks through the problem, the math, the key idea, and what it achieves, more or less in that order."
                ),
                kind="info",
            ),
        ],
        gap=0.2,
        align="center",
    )
    return


@app.cell(hide_code=True)
def _(mo, mvideo, scene_world_model):
    mo.vstack(
        [
            mo.md("## What Is a World Model?"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                As most  of you will already know from both: 1. reading your own papers; 2. the previous presentations; 
                a **world model** is a learned generative system that can predict future observations given past observations and actions.
            
                I will summarize the formal definition of a world model briefly so we are all on the same page, and then go on the more interesting aspects of the StateSpaceDiffuser model: 
                given a trajectory of interactions $a_1, a_2, \\ldots, a_{T-1}$ producing observations $I_1, I_2, \\ldots, I_T,$ the world model $\\mathcal{F}$ predicts the next frame:
                $$I_{T+1} = \\mathcal{F}\\bigl([I_1,\\ldots,I_T],\\,[a_1,\\ldots,a_T]\\bigr).$$

                As the vast range of topics in the assigned papers show, they can be used for a multitude of tasks: 
                autonomous driving, game playing, robotics planning, and virtual interaction. 
                These models are usually trained purely from experience i.e. no hand-coded physics.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mvideo(scene_world_model)
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.55, 0.45],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## The Core Problem: Temporal Drift"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 3**

                *Animation: Show a long corridor in a video game. An agent walks
                forward 20 steps (frames appear left to right). Then the agent
                turns around and walks back. The diffusion baseline generates
                frames for the return trip that look like a completely different
                corridor — colours, textures, and layout have drifted. Then replay
                with StateSpaceDiffuser: the return trip frames correctly match
                the outbound frames. Use a side-by-side split to make the contrast
                vivid.*
                """),
                                kind="neutral",
                            ),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                State-of-the-art world models are **diffusion models** — powerful
                generative models that produce high-fidelity images. But they have
                a hard architectural constraint:

                $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$

                They only see the last $K$ frames — typically $K = 4$ or $K = 16$.
                Everything before that window is simply **forgotten**.

                As the agent explores and later revisits a location, the model has
                no memory of what it looked like. The generated scene **drifts**,
                breaking temporal coherence.
                """),
                            mo.callout(
                                mo.md("""
                **Why not just increase $K$?**
                Transformers — the dominant backbone — have $O(T^2)$ attention
                complexity. Doubling the context quadruples the compute. Long
                contexts quickly become computationally prohibitive.
                """),
                                kind="warn",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(
    Arrow,
    DOWN,
    FadeIn,
    GRAY,
    GREEN,
    LEFT,
    ORIGIN,
    RED,
    RIGHT,
    Scene,
    Square,
    Text,
    UP,
    VGroup,
    WHITE,
    YELLOW,
    config,
    ipython_magic,
):
    config.pixel_height = 720
    config.pixel_width = 1280
    config.frame_rate = 30

    class DiffusionAnimation(Scene):
        """Animation showing the diffusion forward and reverse process"""

        def construct(self):
            title = Text(
                "Diffusion: Forward (Noising) & Reverse (Denoising)", font_size=28
            ).to_edge(UP)
            self.add(title)

            # Create a grid to represent image pixels
            grid_size = 4
            pixel_size = 0.15

            def create_pixel_grid(color, opacity=0.9):
                """Create a simple representation of an image"""
                pixels = VGroup()
                for i in range(grid_size):
                    for j in range(grid_size):
                        pixel = Square(
                            side_length=pixel_size, color=color, fill_opacity=opacity
                        )
                        pixel.move_to(
                            ORIGIN
                            + RIGHT * (i * pixel_size * 1.2)
                            + DOWN * (j * pixel_size * 1.2)
                        )
                        pixels.add(pixel)
                return pixels

            # Forward process (top half)
            forward_label = (
                Text("Forward Process (Add Noise)", font_size=20, color=RED)
                .to_edge(LEFT)
                .shift(UP * 2)
            )
            self.add(forward_label)

            # Start with clean image
            clean = create_pixel_grid(GREEN)
            clean.move_to(ORIGIN + LEFT * 4 + UP * 1)
            self.add(clean)
            clean_label = Text("x0", font_size=18, color=GREEN).next_to(
                clean, DOWN, buff=0.1
            )
            self.add(clean_label)

            # Arrow and progressively noisier images
            positions = [LEFT * 1.5, LEFT * 0.5, RIGHT * 0.5, RIGHT * 1.5]
            noise_levels = [0.3, 0.5, 0.7, 0.9]

            for i, (pos, noise) in enumerate(zip(positions, noise_levels)):
                # Arrow from previous
                if i == 0:
                    arrow = Arrow(
                        start=clean.get_right(), end=pos + LEFT * 0.5, color=WHITE
                    )
                else:
                    prev_pos = positions[i - 1]
                    arrow = Arrow(
                        start=prev_pos + RIGHT * 0.3, end=pos + LEFT * 0.3, color=WHITE
                    )
                self.add(arrow)

                # Noisy image
                noisy = create_pixel_grid(GRAY, opacity=noise)
                noisy.move_to(pos)
                self.add(noisy)

                label = Text(f"x{i + 1}", font_size=16, color=GRAY).next_to(
                    noisy, DOWN, buff=0.1
                )
                self.add(label)

                # Beta label on arrow
                beta_label = Text(f"β{i + 1}", font_size=14, color=RED).move_to(
                    arrow.get_center() + UP * 0.2
                )
                self.add(beta_label)

            # Final pure noise
            final_noise = create_pixel_grid(GRAY, opacity=0.95)
            final_noise.move_to(RIGHT * 2.5 + UP * 1)
            self.add(final_noise)
            final_label = Text("xT ~ N(0,I)", font_size=16, color=GRAY).next_to(
                final_noise, DOWN, buff=0.1
            )
            self.add(final_label)

            arrow_to_final = Arrow(
                start=positions[-1] + RIGHT * 0.3,
                end=final_noise.get_left(),
                color=WHITE,
            )
            self.add(arrow_to_final)

            # Reverse process (bottom half)
            reverse_label = (
                Text("Reverse Process (Denoise)", font_size=20, color=GREEN)
                .to_edge(LEFT)
                .shift(DOWN * 1.5)
            )
            self.add(reverse_label)

            # Start from noise
            start_noise = create_pixel_grid(GRAY, opacity=0.95)
            start_noise.move_to(ORIGIN + LEFT * 4 + DOWN * 1.5)
            self.add(start_noise)
            start_label = Text("xT", font_size=18, color=GRAY).next_to(
                start_noise, DOWN, buff=0.1
            )
            self.add(start_label)

            # Arrows and progressively cleaner images
            rev_positions = [LEFT * 1.5, LEFT * 0.5, RIGHT * 0.5, RIGHT * 1.5]
            clean_levels = [0.7, 0.5, 0.3, 0.1]

            T = len(rev_positions)

            for i, (pos, clean_op) in enumerate(zip(rev_positions, clean_levels)):
                if i == 0:
                    arrow = Arrow(
                        start=start_noise.get_right(), end=pos + LEFT * 0.5, color=WHITE
                    )
                else:
                    prev_pos = rev_positions[i - 1]
                    arrow = Arrow(
                        start=prev_pos + RIGHT * 0.3, end=pos + LEFT * 0.3, color=WHITE
                    )
                self.add(arrow)

                denoised = create_pixel_grid(YELLOW, opacity=1 - clean_op)
                denoised.move_to(pos)
                self.add(denoised)

                label = Text(f"x{T - i}", font_size=16, color=YELLOW).next_to(
                    denoised, DOWN, buff=0.1
                )
                self.add(label)

                # Epsilon theta label
                eps_label = Text("εθ", font_size=14, color=GREEN).move_to(
                    arrow.get_center() + DOWN * 0.2
                )
                self.add(eps_label)

            # Final clean image
            final_clean = create_pixel_grid(GREEN)
            final_clean.move_to(RIGHT * 2.5 + DOWN * 1.5)
            self.add(final_clean)
            final_clean_label = Text(
                "x0 (generated)", font_size=16, color=GREEN
            ).next_to(final_clean, DOWN, buff=0.1)
            self.add(final_clean_label)

            arrow_to_clean = Arrow(
                start=rev_positions[-1] + RIGHT * 0.3,
                end=final_clean.get_left(),
                color=WHITE,
            )
            self.add(arrow_to_clean)

            # Summary text
            summary = Text(
                "Diffusion models transform noise into structured images",
                font_size=18,
                color=WHITE,
            )
            summary.to_edge(DOWN, buff=0.5)
            self.play(FadeIn(summary), run_time=0.5)

            self.wait(2)

    ipython_magic.ManimMagic({}).manim(
        "DiffusionAnimation",
        None,
        {"DiffusionAnimation": DiffusionAnimation, "config": {"media_embed": True}},
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Mathematical Background: Diffusion Models"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                Diffusion models learn to reverse a **noise process**.

                **Forward process** — gradually corrupt a clean image $x_0$:
                $$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(x_t;\\,\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t I\\right)$$

                After $T$ steps, $x_T \\sim \\mathcal{N}(0, I)$ — pure noise.

                **Reverse process** — a neural network $\\epsilon_\\theta$ learns to
                undo one step of noise:
                $$p_\\theta(x_{t-1}\\mid x_t) = \\mathcal{N}\\!\\left(x_{t-1};\\,\\mu_\\theta(x_t, t),\\,\\Sigma_\\theta(x_t, t)\\right)$$

                Sampling: start from noise, apply the learned reverse process
                iteratively to arrive at a clean, high-quality image.

                The paper uses **DIAMOND** [Alonso et al., 2024] as its diffusion
                backbone — a UNet-based EDM model designed for sequential visual
                prediction, generating frames with only **3 denoising iterations**.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 4**

                *Animation: A clean image $x_0$ (e.g., a maze frame) is
                progressively corrupted across $t = 0 \\to T$ — show Gaussian
                noise being added step by step until the image is unrecognisable.
                Then run the reverse direction: starting from pure noise, the
                network iteratively denoises, revealing the image. Label each
                arrow with $\\beta_t$ (forward) and $\\epsilon_\\theta$ (reverse).
                Keep it clean and schematic.*
                """),
                                kind="neutral",
                            ),
                            mo.callout(
                                mo.md("""
                **Key insight:** Diffusion models are excellent at generating
                high-fidelity images — but they carry **no persistent state**
                between frames beyond their short context window.
                """),
                                kind="info",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.55, 0.45],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Mathematical Background: State-Space Models"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 5**

                *Animation: Draw a linear chain of time steps $t=1,2,\\ldots,T$.
                At each step, a compact state vector $h_t$ (shown as a small
                coloured rectangle) is updated from $h_{t-1}$ and the new input
                $f_t$. Annotate with the recurrence equations. Then show a
                comparison column: a Transformer re-reads all previous tokens at
                every step (expensive), while the SSM only needs $h_{t-1}$ and
                $f_t$ (constant cost). Highlight the constant memory footprint.*
                """),
                                kind="neutral",
                            ),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                A **State-Space Model (SSM)** maintains a hidden state $h_t$
                updated at every step of a sequence $f_1, \\ldots, f_T$:

                $$h_t = A\\,h_{t-1} + B\\,f_t, \\qquad m_t = C\\,h_t$$

                where $A, B, C$ are **learned** parameter matrices and $h_t$
                is a compact summary of the entire history so far.

                The paper uses **Mamba** [Gu & Dao, 2023] — an SSM variant
                with a *selective gating* mechanism that dynamically decides
                what information to retain or discard:

                $$\\Delta, B, C = \\text{Linear}(f_t), \\quad
                  \\bar{A} = e^{\\Delta A}$$

                This gives Mamba its expressive advantage over simpler RNNs
                like LSTMs or GRUs.
                """),
                            mo.callout(
                                mo.md("""
                **Complexity comparison**

                | Model | Training | Inference memory |
                |---|---|---|
                | Transformer | $O(T^2)$ | $O(T)$ |
                | CNN | $O(T)$ | $O(K)$ fixed |
                | **SSM (Mamba)** | **$O(T)$** | **$O(1)$ constant** |

                At inference, Mamba processes each new frame in **constant time**
                and **constant memory**, regardless of how long the sequence is.
                """),
                                kind="success",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## How Others Have Tried to Solve This"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                The long-context consistency problem is well-known. The field
                has responded in three main directions:

                **1. Transformer variants with reduced attention**
                Models like Swin and MViT use local or hierarchical attention
                to reduce the $O(T^2)$ cost — but they still struggle to
                scale to the very long video horizons needed here.

                **2. Sampling historical observations**
                Concurrent work [DFoT, etc.] samples a fixed number of past
                frames to use as conditioning. This helps, but it is a
                *manual heuristic* — it cannot guarantee that the most
                important historical frames are selected, and it still has
                a fixed context budget.

                **3. LSTM/GRU world models (DreamerV2/V3)**
                Recurrent models like DreamerV3 do maintain a persistent
                state — but they operate on **discrete latent tokens**
                and have limited generative capacity. They cannot produce
                the high-fidelity continuous images that diffusion models
                deliver.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **The gap no one has filled**

                High-fidelity image generation (diffusion) and persistent
                long-term memory (state-space) have never been combined
                in a world model — until now.
                """),
                                kind="warn",
                            ),
                            mo.md("""
                The key observation is that these two capabilities are
                **complementary**, not competing:

                The SSM is *not generative* — its low-dimensional state
                cannot directly render sharp images.

                The diffusion model is *not persistent* — it forgets
                anything beyond $K$ frames.

                **Combine them and you get both.**
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.55, 0.45],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## The Proposal: StateSpaceDiffuser"),
            mo.callout(
                mo.md("""
        **Core idea:** A Mamba SSM processes the *entire* history $O(T)$ and
        compresses it into a state. That state is injected into a diffusion
        model that generates each frame at high fidelity.
        """),
                kind="info",
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### Long-Context Branch (SSM)

                Each frame $I_t$ is encoded by the
                **Cosmos tokenizer** (scale 16) into a compact feature vector
                $f_t \\in \\mathbb{R}^d$.
                The discrete action $a_t$ indexes a learnable embedding of
                dimension 16, concatenated with $f_t$.

                The Mamba SSM then processes the full sequence:
                $$\\hat{f}_2, \\ldots, \\hat{f}_{T+1} = \\mathcal{M}([f_1, a_1], \\ldots, [f_T, a_T])$$

                It is trained with an **MSE loss** to predict the next
                Cosmos features — forcing the state to encode long-range context.

                At inference, only the final state $h_T$ matters.
                It is updated in **O(1)** per new frame.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                ### Generative Branch (Diffusion)

                The DIAMOND diffusion model generates the next frame
                conditioned on a **short window** of 4 frames *plus*
                the SSM's long-context features $\\hat{f}_t$.

                ### Fusion Module

                The two streams are merged via a two-layer MLP with SiLU:

                $$\\text{cond} = \\text{concat}\\!\\left[\\text{MLP}_{\\text{mem}}(\\hat{f}_t),\\;\\text{MLP}_{\\text{act}}(e_t + \\varepsilon)\\right]$$

                where $e_t$ is the action embedding and $\\varepsilon$ is injected
                noise for robustness. Processing memory and action independently
                before concatenation was found empirically to work best.
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.5, 0.5],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Architecture at a Glance"),
            mo.callout(
                mo.md("""
        **MANIM PLACEHOLDER — Slide 8 (Main Architecture Animation)**

        *This is the centrepiece animation of the talk. Build the architecture
        diagram step by step:*

        *Step 1: Show the input stream — a long sequence of frames
        $I_1, I_2, \\ldots, I_T$ with actions $a_t$ below each.*

        *Step 2: Animate the Cosmos tokenizer encoding each frame into a compact
        feature vector $f_t$ (shrink the frame into a small vector icon).*

        *Step 3: Show the Mamba SSM reading the features left-to-right,
        maintaining a rolling state $h_t$ (a glowing rectangle that updates
        at each step). Label the recurrence $h_t = Ah_{t-1} + Bf_t$.*

        *Step 4: Branch off: the last 4 frames $I_{T-3},\\ldots,I_T$ also feed
        directly into the DIAMOND diffusion UNet (show the UNet icon).*

        *Step 5: The SSM output $\\hat{f}_t$ and the action embedding $e_t$
        pass through the Fusion MLP and merge into the diffusion conditioning
        vector.*

        *Step 6: The UNet denoises (show 3 iterations of the noise schedule)
        and emits $\\hat{I}_{T+1}$. Compare side-by-side with ground truth.*

        *Use colour coding: blue for the Long-Context Branch, red/pink for the
        Generative Branch, cyan for the Fusion module.*
        """),
                kind="neutral",
            ),
            mo.md("""
        The architecture has two key design choices worth pausing on:

        **Two-stage training** — The SSM is trained first (frozen), then the
        diffusion model is trained on top of its frozen features.
        Direct end-to-end training is unstable: diffusion gives noisy gradients
        to the SSM, so the SSM's features never stabilise, and diffusion learns
        to ignore them entirely.

        **Frame-level (not patch-level) SSM** — Unlike prior work that applies
        SSMs at the image token level, each *entire frame* is one SSM step.
        This avoids conflating spatial and temporal dependencies.
        """),
        ],
        gap=2,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Training Protocol"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### Stage 1 — Train the Long-Context Branch

                The Mamba SSM is trained on long sequences
                (length 50 or 16) to predict next-frame Cosmos features
                from the full history:

                $$\\mathcal{L}_{\\text{SSM}} = \\sum_{t=1}^{T} \\|\\hat{f}_{t+1} - f_{t+1}\\|^2$$

                The produced features decode to images with artifacts
                but carry important long-range context cues.

                ### Stage 2 — Train the Generative Branch

                With the SSM **frozen**, the DIAMOND diffusion model
                is trained on short sequences of length 4,
                conditioned on the frozen SSM features.
                It learns to render high-quality images *given*
                reliable long-context cues from Stage 1.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **Why freeze the SSM?**

                Direct end-to-end training is unstable:

                Diffusion's noisy gradients destabilise the SSM.
                The SSM returns constantly shifting features.
                Diffusion learns to ignore the SSM entirely.

                Decoupling the two stages solves this. It also means the
                SSM branch can be swapped independently at test time
                without retraining the (heavier) diffusion model.
                """),
                                kind="warn",
                            ),
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 9**

                *Animation: A two-panel timeline. Left panel: Stage 1 —
                the SSM reads a long sequence and a loss arrow points at
                its output. Right panel: Stage 2 — the SSM is shown
                "frozen" (grey, padlocked), while the diffusion UNet
                is highlighted in orange and trained. Show the gradient
                flow arrows stopping at the frozen SSM boundary.*
                """),
                                kind="neutral",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.5, 0.5],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Evaluating Long-Context Memory"),
            mo.md("""
        The authors design a clever evaluation protocol specifically to stress-test
        long-term memory — something standard video metrics do not capture.
        """),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 10**

                *Animation: Show an agent walking through a maze for $n$ steps
                (forward trajectory, frames shown left to right). At the midpoint,
                a "U-turn" arrow appears and the agent retraces its steps for
                $n$ steps (reverse trajectory). For each reverse frame, draw a
                dotted line back to the corresponding forward frame and show the
                PSNR score. For the diffusion baseline, the reverse frames look
                nothing like the forward ones (low PSNR, highlighted red). For
                StateSpaceDiffuser, they match closely (high PSNR, highlighted
                green). The further back in the sequence, the bigger the gap.*
                """),
                                kind="neutral",
                            ),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                ### The Forward-Backward Protocol

                The agent takes $n$ actions **forward**, then $n$ mirrored
                actions **backward** (turning left becomes turning right, etc.).

                The second half of the sequence should be *identical* in content
                to the first half — the model must recall what it saw up to
                $n$ steps ago.

                Evaluation metric: **PSNR** on each reverse frame, especially
                the final frame, which requires recalling frame $I_1$ — the
                oldest possible memory.

                Two environments:

                **MiniGrid** — a 2D maze with partial observations.
                The agent sees a $7\\times7$ window of an $85\\times85$ grid.
                Sequences of 100 steps (50 forward + 50 back).

                **CSGO** — a 3D first-person shooter with 51 action types.
                Visually complex, continuous motion, compounding effects.
                Evaluated with a **user study** (12 participants) due to the
                known mismatch between PSNR and perceptual quality in video.
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(matplotlib, mo, np, plt):
    matplotlib.rcParams.update(
        {
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
    )

    # Data from Table 1
    models = ["DIAMOND", "SSVM", "SSD (ours)"]
    colors = ["#e05c5c", "#5cb8e0", "#5ce08a"]

    ctx16_avg = [27.13, 33.40, 41.01]
    ctx16_fin = [25.44, 33.17, 40.55]
    ctx50_avg = [26.13, 32.64, 39.68]
    ctx50_fin = [25.15, 32.44, 39.32]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

    x = np.arange(len(models))
    w = 0.35

    for ax, avg, fin, title in [
        (axes[0], ctx16_avg, ctx16_fin, "Context Length 16"),
        (axes[1], ctx50_avg, ctx50_fin, "Context Length 50"),
    ]:
        bars_avg = ax.bar(x - w / 2, avg, w, label="Avg PSNR", color=colors, alpha=0.65)
        bars_fin = ax.bar(
            x + w / 2, fin, w, label="Final PSNR", color=colors, alpha=1.0
        )
        ax.set_xticks(x)
        ax.set_xticklabels(models, fontsize=8)
        ax.set_ylabel("PSNR (dB)")
        ax.set_title(title, color="#aac", fontsize=10)
        ax.set_ylim(0, 50)
        ax.legend(fontsize=7, framealpha=0.2)
        for bar in list(bars_avg) + list(bars_fin):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="#dde",
            )

    fig.suptitle(
        "MiniGrid Quantitative Results — Table 1", color="#cce", fontsize=11, y=1.02
    )
    fig.tight_layout()

    mo.vstack(
        [
            mo.md("## Results: MiniGrid Quantitative Evaluation"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.as_html(fig),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **StateSpaceDiffuser achieves +51.9% average PSNR improvement
                over the DIAMOND baseline at context length 50.**

                The gain is even larger (+56.3%) on the hardest case — the
                *final* frame, which requires recalling content from 50 steps
                ago. This is precisely where the SSM's persistent state matters
                most.
                """),
                                kind="success",
                            ),
                            mo.md("""
                Three baselines are compared:

                **DIAMOND** — pure diffusion, no long-term memory.

                **State-Space World Model (SSVM)** — the SSM branch alone,
                decoded directly to images. Good memory, blurry images.

                **SSD w/o state** — StateSpaceDiffuser with the SSM features
                zeroed out. Performance drops *below* baseline, confirming the
                features are actively used, not ignored.

                The combination (SSD) outperforms both components individually,
                showing the two branches are genuinely complementary.
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.55, 0.45],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Results: CSGO & Generalisation to Longer Contexts"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### CSGO User Study

                For the 3D environment, PSNR is a poor proxy for perceptual
                quality (known issue in continuous video). Instead, 12
                participants rated whether StateSpaceDiffuser or DIAMOND
                generated frames closer to the ground truth.

                Rating scale: $[-1, 1]$, where
                $-1$ = prefer baseline, $+1$ = prefer SSD, $0$ = borderline.

                **Results:**

                Frame 15 (second-to-last): **+0.20**

                Frame 17 (final, hardest): **+0.24**

                A clear human preference for SSD, growing stronger the
                further back in the sequence the recall must reach.
                """),
                            mo.callout(
                                mo.md("""
                **Generalisation without finetuning**

                The model trained on context length 50 was evaluated on
                sequences of length 100 and 150.

                At length 100: SSD achieves 37.99 Avg PSNR vs 26.39 for
                DIAMOND — a **+44%** improvement.

                At length 150: SSD achieves 30.75 vs 24.35 — still a
                **+26%** improvement, with zero additional training.
                """),
                                kind="success",
                            ),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 12**

                *Animation: Show a timeline with context length on the
                x-axis (16, 50, 100, 150). For each length, plot PSNR
                bars for DIAMOND vs SSD. Animate the bars growing as
                context length increases — but show DIAMOND's bar
                flattening or shrinking (it cannot generalise) while
                SSD's bar stays high. Annotate with "trained here" arrow
                at length 50 and "evaluated here (zero-shot)" arrows at
                100 and 150. This visually communicates the SSM's
                key advantage: it naturally handles sequences longer
                than it was trained on.*
                """),
                                kind="neutral",
                            ),
                            mo.md("""
                This generalisation is **not a coincidence**. It follows
                directly from the SSM's computational structure: the
                recurrence $h_t = Ah_{t-1} + Bf_t$ applies identically
                regardless of how long the sequence is. The state simply
                keeps accumulating. There is no architectural limit on $T$.

                The diffusion model, by contrast, was designed for a fixed
                short context and cannot benefit from a longer history
                even if one were provided.
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.5, 0.5],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Ablation: Do the SSM Features Actually Matter?"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                The authors replace the SSM output features $\\hat{f}_t$
                with **zeros** before passing them to the diffusion model.
                Everything else stays the same.

                The result: performance drops below the DIAMOND baseline
                (23.68 vs 27.13 Avg PSNR at context 16). The diffusion
                model was *relying* on those features — removing them
                actively hurts it.

                This rules out the possibility that the diffusion model
                simply learned to ignore the SSM signal. The features
                are integrated and depended upon.

                A parallel experiment on CSGO shows the same pattern:
                without SSM features the model **hallucinates** content
                (generates plausible-looking but entirely wrong scenes).
                With SSM features it correctly recalls what the scene
                looked like.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 13**

                *Animation: Show three columns of generated frames for the
                same CSGO sequence — ground truth, DIAMOND baseline, and
                SSD. Zoom into the return-trip frames. Highlight with red
                circles the hallucinated content in DIAMOND (wrong wall
                colours, missing objects) and show SSD correctly reproducing
                the original scene. Then fade in a fourth column: SSD without
                SSM features — it hallucinates just like DIAMOND, confirming
                the SSM is load-bearing.*
                """),
                                kind="neutral",
                            ),
                            mo.callout(
                                mo.md("""
                **The complementarity picture**

                | Capability | SSM alone | Diffusion alone | **SSD** |
                |---|---|---|---|
                | Long-term memory | ✅ | ❌ | ✅ |
                | High visual fidelity | ❌ | ✅ | ✅ |
                | Scales with $T$ | ✅ | ❌ | ✅ |
                | Constant inference cost | ✅ | — | ✅ |
                """),
                                kind="info",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.5, 0.5],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _(mo, plt):
    fig2, ax2 = plt.subplots(figsize=(7, 3))
    fig2.patch.set_facecolor("#0f0f1a")
    ax2.set_facecolor("#0f0f1a")

    components = ["DIAMOND\n(diffusion)", "SSM\n(Mamba)", "Fusion\nMLP"]
    sizes = [98, 1.5, 0.5]
    colors2 = ["#e05c5c", "#5ce08a", "#5cb8e0"]

    wedges, texts, autotexts = ax2.pie(
        sizes,
        labels=components,
        colors=colors2,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"color": "#cce", "fontsize": 9},
        wedgeprops={"edgecolor": "#0f0f1a", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("#fff")

    ax2.set_title("Inference compute breakdown", color="#cce", fontsize=10, pad=12)
    fig2.tight_layout()

    mo.vstack(
        [
            mo.md("## Computational Cost: Almost Free"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.as_html(fig2),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                One of the most striking claims in the paper: the SSM branch
                contributes **less than 2%** of total inference compute.

                The reason is architectural. At inference time, the Mamba SSM
                processes each new frame in a single recurrent step — no
                attention over the full history, no growing KV cache:

                $$h_t = Ah_{t-1} + Bf_t \\quad \\text{(one matrix multiply)}$$

                The diffusion model (DIAMOND) does the heavy lifting: it runs
                a full UNet denoising pass for each generated frame.

                This means StateSpaceDiffuser is essentially a **free upgrade**
                over the diffusion-only baseline — you get long-term memory
                at negligible additional cost.
                """),
                            mo.callout(
                                mo.md("""
                The SSM contributes < 2% of inference compute
                while enabling memory across an order of magnitude
                more steps than the baseline.
                """),
                                kind="success",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


app._unparsable_cell(
    """
    mo.vstack(
        [
            mo.md(\"## Limitations & Open Questions\"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md(\"\"\"
                The paper is candid about what does not yet work well.

                **Low-dimensional bottleneck.** The SSM state has dimension 256.
                In extended rollouts — especially in visually complex environments
                like CSGO — fine-grained detail can be lost as the state is
                compressed over many steps. Scaling the SSM (more heads, layers,
                larger state) is expected to help but was not explored within
                this compute budget.

                **Lightweight diffusion backbone.** DIAMOND is a relatively
                small UNet. Replacing it with a larger pretrained backbone
                (e.g., a video diffusion transformer) could dramatically
                improve visual sharpness, without changing the method.

                **Sensitivity to current-step noise.** While the model recovers
                from noisy *future* inputs, it is sensitive to noise in the
                *current* step. This is a known limitation of the sliding-window
                generation strategy.

                **No active memory management.** The SSM compresses everything
                equally — there is no mechanism to explicitly prioritise
                task-relevant memories over irrelevant ones. Selective
                attention over the state is a natural extension.
                \"\"\"),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md(\"\"\"
                **What this means for the field**

                The result opens a new design axis for world models:
                *how should memory and generation be decoupled?*

                The two-stage training trick (freeze memory, train generation)
                may generalise well beyond this specific architecture.
                SSMs are a natural fit for any sequential generative task
                where context windows are the binding constraint.
                \"\"\"),
                                kind=\"info\",
    literature 
                            ),
                            mo.callout(
                                mo.md(\"\"\"
                **Future directions suggested by the authors**

                Scaling the SSM state dimension and depth.
                Replacing DIAMOND with a larger diffusion backbone.
                Applying the approach to real-world ego-centric video.
                Combining with agent planning — the persistent state
                could serve as a planning memory for RL agents.
                \"\"\"),
                                kind=\"neutral\",
                            ),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.55, 0.45],
                gap=2,
            ),
        ],
        gap=1,
    )
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.vstack(
        [
            mo.md("## Conclusion"),
            mo.callout(
                mo.md("""
        **StateSpaceDiffuser** shows that you do not have to choose between
        *remembering the past* and *generating the present beautifully*.
        A lightweight SSM branch — costing < 2% of inference — gives a
        diffusion world model persistent long-term memory, enabling it to
        maintain temporal coherence for an order of magnitude more steps
        than a diffusion-only baseline.
        """),
                kind="success",
            ),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### The story in three acts

                **Problem.** Diffusion world models forget everything beyond
                a short window of $K$ frames, causing temporal drift on
                long interactions.

                **Idea.** A Mamba SSM processes the full history in $O(T)$
                time and constant memory, compressing it into a persistent
                state that is injected into the diffusion model via a
                learned fusion module.

                **Result.** +51.9% PSNR on MiniGrid at horizon 50.
                Positive user preference on CSGO.
                Zero-shot generalisation to 3× longer contexts.
                All for less than 2% extra inference cost.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **MANIM PLACEHOLDER — Slide 16 (Closing)**

                *Animation: A final, satisfying replay of the opening scene
                (Slide 3) — the agent walks forward and returns. This time,
                show the SSM state $h_t$ as a glowing orb that grows and
                "remembers" the corridor as the agent walks through it.
                On the return trip, the orb lights up the correct memory
                and the generated frames match the ground truth perfectly.
                Fade to the paper title and authors.*
                """),
                                kind="neutral",
                            ),
                            mo.md("""
                **Paper:** arXiv:2505.22246

                **Project page:**
                https://insait-institute.github.io/StateSpaceDiffuser/

                **Key references**

                Gu & Dao (2023) — Mamba

                Alonso et al. (2024) — DIAMOND

                Hafner et al. (2023) — DreamerV3
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.5, 0.5],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np

    return matplotlib, np, plt


@app.cell
def _():
    from manim import Scene, Square, Text, VGroup, Rectangle, FadeIn, Arrow
    from manim import (
        UP,
        RIGHT,
        DOWN,
        LEFT,
        ORIGIN,
        GRAY,
        RED,
        WHITE,
        YELLOW,
        GREEN,
    )
    from manim import config
    from manim.utils import ipython_magic

    return (
        Arrow,
        DOWN,
        FadeIn,
        GRAY,
        GREEN,
        LEFT,
        ORIGIN,
        RED,
        RIGHT,
        Scene,
        Square,
        Text,
        UP,
        VGroup,
        WHITE,
        YELLOW,
        config,
        ipython_magic,
    )


@app.cell
def _(Scene, mo):
    def mvideo(scene: Scene, width=900, height=900 * 9 / 16):
        path = scene.renderer.file_writer.movie_file_path
        return mo.video(src=str(path), width=width, height=height)

    return (mvideo,)


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
