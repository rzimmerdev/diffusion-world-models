import marimo

__generated_with = "0.23.4"
app = marimo.App(
    width="full",
    layout_file="layouts/presentation.slides.json",
    css_file="custom.css",
)

with app.setup:
    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib.patches import FancyBboxPatch

    import marimo as mo
    import numpy as np

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
    from manim import config, tempconfig

    import logging
    import importlib
    from pathlib import Path

    logging.getLogger("manim").setLevel(logging.WARNING)


@app.cell
def _():
    mo.vstack(
        [
            mo.md("""
        # StateSpaceDiffuser
        ## Bringing Long Context to Diffusion World Models

        **Nedko Savov · Naser Kazemi · Deheng Zhang · Danda Pani Paudel · Xi Wang · Luc Van Gool**

        INSAIT, Sofia University · ETH Zurich · TU Munich

        *arXiv:2505.22246  October 2025*
        """),
            mo.callout(
                mo.md(
                    "This seminar walks through the problem, the math, the key idea, and what it achieves, more or less in that order."
                ),
                kind="info",
            ),
            mo.md("Seminar by Rafael Zimmer"),
        ],
        gap=0.2,
        align="center",
    )
    return


@app.cell
def _():
    mo.md("""
    ### Slide Index

    1. **Title & Overview**
    2. **What Is a World Model?**
    3. **The Core Problem: Temporal Drift**
    4. **Mathematical Background: Diffusion Models**
    5. **Mathematical Background: State-Space Models**
    6. **How Others Have Tried to Solve This**
    7. **The Proposal: StateSpaceDiffuser**
    8. **Architecture at a Glance**
    9. **Training Protocol**
    10. **Evaluating Long-Context Memory**
    11. **Results: MiniGrid Quantitative Evaluation**
    12. **Results: CSGO & Generalisation**
    13. **Ablation: Do the SSM Features Matter?**
    14. **Computational Cost: Almost Free**
    15. **Limitations & Open Questions**
    16. **Criticism & Gaps Not Addressed**
    17. **Conclusion**
    """)
    return


app._unparsable_cell(
    r"""
    import world_model

    importlib.reload(world_model)

    scene_world_model = world_model.WorldModelAnimation()
    _scene_world_model = scene_world_model.render()

    video_world_model = 
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(video_world_model):
    mo.vstack(
        [
            mo.md("## What Is a World Model?"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                As most  of you will already know from both reading your assigned paper and the previous presentations,
                a **world model** is a learned generative system that can predict future observations given past observations and actions.

                The seminar will cover shortly the formal definition of a world model and move on to more interesting aspects of the StateSpaceDiffuser model.
                Given a trajectory of interactions $a_1, a_2, \\ldots, a_{T-1}$ producing observations $I_1, I_2, \\ldots, I_T,$ the world model $\\mathcal{F}$ predicts the next frame:
                $$I_{T+1} = \\mathcal{F}\\bigl([I_1,\\ldots,I_T],\\,[a_1,\\ldots,a_T]\\bigr).$$

                As the vast range of topics in the assigned papers show, they can be used for a multitude of tasks: 
                autonomous driving, game playing, robotics planning, and virtual interaction. 
                These models are usually trained purely from experience i.e. no hand-coded physics.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [video_world_model],
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


@app.cell
def _():
    import compare_diffusion

    importlib.reload(compare_diffusion)

    scene_compare_diffusion = compare_diffusion.CompareDiffusionAnimation()
    _scene_compare_diffusion = scene_compare_diffusion.render()

    video_compare_diffusion = mo.video(
        src=str(scene_compare_diffusion.renderer.file_writer.movie_file_path)
    )
    return (video_compare_diffusion,)


@app.cell(hide_code=True)
def _(video_compare_diffusion):
    mo.vstack(
        [
            mo.md("## The Core Problem: Temporal Drift"),
            mo.hstack(
                [
                    video_compare_diffusion,
                    mo.vstack(
                        [
                            mo.md("""
                State-of-the-art world models are diffusion models, as most of you know already, powerful
                generative models that produce high-fidelity images. But they have
                a hard architectural constraint:

                $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$

                This means that the information they see is a filtration on the past $K$ frames typically $K = 4$ or $K = 16$.
                This means the model does not consider anything before that window and it is simply forgotten.

                As the agent explores and later revisits a location, the model has
                no memory of what it looked like. The key term is that the scene **drifts**, and the model stops being temporally coherent anymore.
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


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## The Core Problem: Temporal Drift"),
            mo.hstack(
                [
                    mo.callout(
                        mo.md("""
                **Why not just increase $K$?**
                The dominant backbone models (such as transformers) are usually quadratic in $T$ (e.g., transformers have $O(T^2)$ attention complexity). 
                Doubling the context quadruples the compute. This means that long contexts quickly become computationally prohibitive for strong models.
                """),
                        kind="warn",
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                State-of-the-art world models are diffusion models, as most of you know already, powerful
                generative models that produce high-fidelity images. But they have
                a hard architectural constraint:

                $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$

                This means that the information they see is a filtration on the past $K$ frames typically $K = 4$ or $K = 16$.
                This means the model does not consider anything before that window and it is simply forgotten.

                As the agent explores and later revisits a location, the model has
                no memory of what it looked like. The key term is that the scene **drifts**, and the model stops being temporally coherent anymore.
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


@app.cell
def _():
    import diffusion_denoise

    importlib.reload(diffusion_denoise)

    scene_diffusion_denoise = diffusion_denoise.DiffusionProcessAnimation()
    _scene_diffusion_denoise = scene_diffusion_denoise.render()

    video_diffusion_denoise = mo.video(
        src=str(scene_diffusion_denoise.renderer.file_writer.movie_file_path)
    )
    return (video_diffusion_denoise,)


@app.cell(hide_code=True)
def _(video_diffusion_denoise):
    mo.vstack(
        [
            mo.md("## Mathematical Background: Diffusion Models"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                **Forward process**: repeatedly corrupt a clean image $x_0$ by applying gaussian noise:
                $$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(x_t;\\,\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t I\\right)$$

                **Reverse process**: a neural network $\\epsilon_\\theta$ learns to
                undo one step of noise:
                $$p_\\theta(x_{t-1}\\mid x_t) = \\mathcal{N}\\!\\left(x_{t-1};\\,\\mu_\\theta(x_t, t),\\,\\Sigma_\\theta(x_t, t)\\right)$$

                The paper uses a UNet-based EDM, called **DIAMOND** [Alonso et al., 2024] as its diffusion
                backbone model designed for sequential visual prediction, 
                generating frames with only **3 denoising iterations**.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            video_diffusion_denoise,
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


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Mathematical Background: Diffusion Models"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                **Forward process**: repeatedly corrupt a clean image $x_0$ by applying gaussian noise:
                $$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(x_t;\\,\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t I\\right)$$

                **Reverse process**: a neural network $\\epsilon_\\theta$ learns to
                undo one step of noise:
                $$p_\\theta(x_{t-1}\\mid x_t) = \\mathcal{N}\\!\\left(x_{t-1};\\,\\mu_\\theta(x_t, t),\\,\\Sigma_\\theta(x_t, t)\\right)$$

                The paper uses a UNet-based EDM, called **DIAMOND** [Alonso et al., 2024] as its diffusion
                backbone model designed for sequential visual prediction, 
                generating frames with only **3 denoising iterations**.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **Key insight:** Diffusion models are excellent at generating
                high-fidelity images but carry no persistent state
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


@app.cell
def _():
    import ssm

    importlib.reload(ssm)

    scene_ssm = ssm.SSMAnimation()
    _scene_ssm = scene_ssm.render()

    video_ssm = mo.video(src=str(scene_ssm.renderer.file_writer.movie_file_path))
    return (video_ssm,)


@app.cell(hide_code=True)
def _(video_ssm):
    mo.vstack(
        [
            mo.md("## Mathematical Background: State-Space Models"),
            mo.hstack(
                [
                    video_ssm,
                    mo.vstack(
                        [
                            mo.md("""
                A **State-Space Model (SSM)** maintains a hidden state a compact summary $h_t$ of the entire history so far,
                updated at every step of a sequence $f_1, \\ldots, f_T$

                $$h_t = A\\,h_{t-1} + B\\,f_t, \\qquad m_t = C\\,h_t$$

                The paper uses **Mamba** [Gu & Dao, 2023] which is a SSM variant
                with *selective gating* that dynamically decides what information to retain or discard, which allows Mamba to be more expressive over simpler RNNs like LSTMs or GRUs.

                $$\\Delta, B, C = \\text{Linear}(f_t), \\quad
                  \\bar{A} = e^{\\Delta A}$$            
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


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Mathematical Background: State-Space Models"),
            mo.hstack(
                [
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
                    mo.vstack(
                        [
                            mo.md("""
                A **State-Space Model (SSM)** maintains a hidden state a compact summary $h_t$ of the entire history so far,
                updated at every step of a sequence $f_1, \\ldots, f_T$

                $$h_t = A\\,h_{t-1} + B\\,f_t, \\qquad m_t = C\\,h_t$$

                The paper uses **Mamba** [Gu & Dao, 2023] which is a SSM variant
                with *selective gating* that dynamically decides what information to retain or discard, which allows Mamba to be more expressive over simpler RNNs like LSTMs or GRUs.

                $$\\Delta, B, C = \\text{Linear}(f_t), \\quad
                  \\bar{A} = e^{\\Delta A}$$            
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


@app.cell
def _():
    mo.vstack(
        [
            # mo.md("## How Others Have Tried to Solve This"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            # mo.md("""
                            #     The long-context consistency problem is well-known. The field
                            #     has responded in three main directions:
                            #     **1. Transformer variants with reduced attention**
                            #     Models like Swin and MViT use local or hierarchical attention
                            #     to reduce the $O(T^2)$ cost  but they still struggle to
                            #     scale to the very long video horizons needed here.
                            #     **2. Sampling historical observations**
                            #     Concurrent work [DFoT, etc.] samples a fixed number of past
                            #     frames to use as conditioning. This helps, but it is a
                            #     *manual heuristic*  it cannot guarantee that the most
                            #     important historical frames are selected, and it still has
                            #     a fixed context budget.
                            #     **3. LSTM/GRU world models (DreamerV2/V3)**
                            #     Recurrent models like DreamerV3 do maintain a persistent
                            #     state  but they operate on **discrete latent tokens**
                            #     and have limited generative capacity. They cannot produce
                            #     the high-fidelity continuous images that diffusion models
                            #     deliver.
                            #     """),
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
                                    in a world model until now.
                                    """),
                                kind="warn",
                            ),
                            # mo.md("""
                            #     The key observation is that these two capabilities are
                            #     **complementary**, not competing:
                            #     The SSM is *not generative* its low-dimensional state
                            #     cannot directly render sharp images.
                            #     The diffusion model is *not persistent* it forgets
                            #     anything beyond $K$ frames.
                            #     **Combine them and you get both.**
                            #     """),
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


@app.cell
def _():
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
                Cosmos features thus forcing the state to encode long-range context.

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


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Architecture at a Glance"),
            mo.image("media/images/architecture.png", width=800),
            mo.md("""
        The architecture has two key design choices worth pausing on:

        **Two-stage training** - The SSM is trained first (frozen), then the
        diffusion model is trained on top of its frozen features.
        Direct end-to-end training is unstable: diffusion gives noisy gradients
        to the SSM, so the SSM's features never stabilise, and diffusion learns
        to ignore them entirely.

        **Frame-level (not patch-level) SSM** - Unlike prior work that applies
        SSMs at the image token level, each *entire frame* is one SSM step.
        This avoids conflating spatial and temporal dependencies.
        """),
        ],
        gap=2,
        align="center"
    )
    return


@app.cell
def _():
    import two_stage

    importlib.reload(two_stage)

    scene_two_stage = two_stage.TwoStageTraining()
    _scene_two_stage = scene_two_stage.render()

    video_two_stage = mo.video(
        src=str(scene_two_stage.renderer.file_writer.movie_file_path)
    )
    return (video_two_stage,)


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Training Protocol"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### Stage 1 - Train the Long-Context Branch

                The Mamba SSM is trained on long sequences
                (length 50 or 16) to predict next-frame Cosmos features
                from the full history, 
                and produced features decode to images with artifacts
                but carry important long-range context cues:

                $$\\mathcal{L}_{\\text{SSM}} = \\sum_{t=1}^{T} \\|\\hat{f}_{t+1} - f_{t+1}\\|^2$$


                ### Stage 2 - Train the Generative Branch

                With the SSM **frozen**, the DIAMOND diffusion model
                is trained on short sequences of length 4,
                conditioned on frozen SSM features.
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


@app.cell
def _(video_two_stage):
    video_two_stage
    return


@app.cell
def _():
    import long_context

    importlib.reload(long_context)

    scene_long_context = long_context.LongContextComparison()
    _scene_long_context = scene_long_context.render()

    video_long_context = mo.video(
        src=str(scene_long_context.renderer.file_writer.movie_file_path)
    )
    return (video_long_context,)


@app.cell
def _(video_long_context):
    mo.vstack(
        [
            mo.md("## Evaluating Long-Context Memory"),
            mo.md("""
        The authors design an evaluation protocol to stress-test long-term memory.
        """),
            mo.hstack(
                [
                    mo.vstack(
                        [video_long_context],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                ### The Forward-Backward Protocol

                The agent takes $n$ actions **forward**, then $n$ mirrored
                actions **backward** (turning left becomes turning right, etc.).

                The second half of the sequence should be *identical* in content
                to the first half, so the model must recall what it saw up to
                $n$ steps ago.

                Evaluation metric: PSNR on each reverse frame, especially
                the final frame, which requires recalling the
                oldest possible memory in frame $I_1$.
                """),
                        ],
                        gap=0,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Evaluating Long-Context Memory"),
            mo.md("""
        The authors stress-test the long-term memory as follows using the following environments:
        """),
            mo.hstack(
                [
                    mo.vstack(
                        [mo.image("media/images/csgo.png")],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                **MiniGrid** is a 2D maze with partial observations.
                The agent sees a $7\\times7$ window of an $85\\times85$ grid.
                Sequences of 100 steps (50 forward + 50 back).

                **CSGO** is a 3D first-person shooter with 51 action types.
                Visually complex, continuous motion, compounding effects.
                Evaluated with a **user study** (12 participants) due to the
                known mismatch between PSNR and perceptual quality in video.
                """),
                        ],
                        gap=0,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _():
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
        "MiniGrid Quantitative Results", color="#cce", fontsize=11, y=1.02
    )
    fig.tight_layout()
    return (fig,)


@app.cell(hide_code=True)
def _(fig):
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

                The gain is even larger (+56.3%) on the hardest case, where the
                final frame requires recalling content from 50 steps
                ago, thus with the results we see where SSM's persistent state matters most.
                """),
                                kind="success",
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


@app.cell
def _(fig):
    mo.vstack(
        [
            mo.md("## Results: MiniGrid Quantitative Evaluation"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.as_html(fig),
                            mo.md(
                                "The combination (SSD) outperforms both components individually, i.e. they are complementary."
                            ),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                Three baselines are compared:

                **DIAMOND**: pure diffusion with no long-term memory.

                **State-Space World Model (SSVM)**: SSM branch alone,
                decoded directly to images (shows good long-term memory but bad quality/performance).

                **SSD w/o state**: StateSpaceDiffuser with SSM features
                zeroed out. Performance drops *below* baseline, which confirms the
                features are actively used.


                """),
                        ],
                        gap=0,
                    ),
                ],
                widths=[0.55, 0.45],
                gap=1,
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
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
                generated frames closer to the labels.
                Rating scale: $[-1, 1]$, where
                $-1$ = prefer baseline, $+1$ = prefer SSD, $0$ = borderline.

                **Results:**

                - Frame 15 (second-to-last): **+0.20**
                - Frame 17 (final, hardest): **+0.24**

                """),
                        ],
                        gap=0,
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                This generalisation is **not a coincidence**. It follows
                directly from the SSM's computational structure of the
                recurrence $h_t = Ah_{t-1} + Bf_t$, by applying it identically
                regardless of how long the sequence is. The state simply
                keeps accumulating and there is no architectural limit on $T$.

                The diffusion model, by contrast, was designed for a fixed
                short context and cannot benefit from a longer history
                even if one were provided.


                A clear human preference for SSD, growing stronger the
                further back in the sequence the recall must reach.

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


@app.cell
def _():
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
                generated frames closer to the labels.
                Rating scale: $[-1, 1]$, where
                $-1$ = prefer baseline, $+1$ = prefer SSD, $0$ = borderline.

                **Results:**

                - Frame 15 (second-to-last): **+0.20**
                - Frame 17 (final, hardest): **+0.24**
                """),
                        ],
                        gap=0,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **Generalisation without finetuning**

                The model trained on context length 50 was evaluated on
                sequences of length 100 and 150.

                At length 100: SSD achieves 37.99 Avg PSNR vs 26.39 for
                DIAMOND (**+44%** improvement).

                At length 150: SSD achieves 30.75 vs 24.35 
                (**+26%** improvement), with zero additional training.
                """),
                                kind="success",
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


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Ablation: Do the SSM Features Actually Matter?"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                The authors replace the SSM output features $\\hat{f}_t$
                with **zeros** before passing them to the diffusion model,
                and maintain the rest of the model the same.\
                Performance drops below the DIAMOND baseline
                (23.68 vs 27.13 Avg PSNR at context 16), showing the diffusion
                model was *relying* on those features, and removing them
                actively decreases performance.

                This rules out the possibility that the diffusion model
                simply learned to ignore the SSM signal, as the features
                are integrated and depended upon. 
                On CSGO the model without SSM features **hallucinates** content
                (generates plausible-looking but entirely wrong scenes).
                """),
                        ],
                        gap=1,
                    ),
                    mo.image("media/images/ablation.png"),
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
    mo.vstack(
        [
            mo.md("## Ablation: Do the SSM Features Actually Matter?"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""

                | Capability | Diffusion alone | SSM alone | **SSD** |
                |---|---|---|---|
                | Long-term memory | No | Yes | Yes |
                | High visual fidelity | Yes | No | No |
                | Scales with $T$ | No | Yes | Yes |
                | Constant inference cost | - | Yes | Yes |
                """),
                                kind="info",
                            ),
                        ],
                        gap=1,
                    ),
                    mo.image("media/images/ablation.png"),
                ],
                widths=[0.5, 0.5],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _():
    fig2, ax2 = plt.subplots(figsize=(7, 2.5))
    fig2.patch.set_facecolor("#0f0f1a")
    ax2.set_facecolor("#0f0f1a")
    ax2.set_xlim(0, 3)
    ax2.set_ylim(0, 1)
    ax2.axis("off")

    components = ["DIAMOND\n(diffusion)", "SSM\n(Mamba)", "Fusion\nMLP"]
    sizes = [98, 1.5, 0.5]
    colors_2 = ["#e05c5c", "#5ce08a", "#5cb8e0"]

    box_w, box_h = 0.85, 0.6
    y = 0.2

    for i, (name, val, col) in enumerate(zip(components, sizes, colors_2)):
        x_2 = i + 0.075

        rect = FancyBboxPatch(
            (x_2, y),
            box_w,
            box_h,
            boxstyle="round,pad=0.02,rounding_size=0.05",
            linewidth=2,
            edgecolor=col,
            facecolor="#141426",
        )
        ax2.add_patch(rect)

        ax2.text(
            x_2 + box_w / 2,
            y + box_h * 0.65,
            name,
            ha="center",
            va="center",
            color="#cce",
            fontsize=9,
        )

        ax2.text(
            x_2 + box_w / 2,
            y + box_h * 0.30,
            f"{val}%",
            ha="center",
            va="center",
            color="#ffffff",
            fontsize=11,
            fontweight="bold",
        )

    plt.tight_layout()
    return (fig2,)


@app.cell(hide_code=True)
def _(fig2):
    mo.vstack(
        [
            mo.md("## Computational Cost: Almost Free"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.as_html(fig2),
                            mo.md("""
                            An important result is that the SSM branch contributes **less than 2%** of total inference compute."""),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                At inference time, the Mamba SSM
                processes each new frame in a single recurrent step and doesn't require
                iterating over the full history:
                $$h_t = Ah_{t-1} + Bf_t \\quad \\text{(one matrix multiply)}$$

                Compare to a transformer approach, which has growing KV cache.
                The diffusion model (DIAMOND) does the heavy lifting with the
                full UNet denoising pass for each generated frame.
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


@app.cell
def _(fig2):
    mo.vstack(
        [
            mo.md("## Computational Cost: Almost Free"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.as_html(fig2),
                            mo.md("""
                            An important result is that the SSM branch contributes **less than 2%** of total inference compute."""),
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.md("""
                At inference time, the Mamba SSM
                processes each new frame in a single recurrent step and doesn't require
                iterating over the full history:
                $$h_t = Ah_{t-1} + Bf_t \\quad \\text{(one matrix multiply)}$$

                Compare to a transformer approach, which has growing KV cache.
                The diffusion model (DIAMOND) does the heavy lifting with the
                full UNet denoising pass for each generated frame.
                """),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            ),
            mo.hstack(
                [
                    mo.callout(
                        """StateSpaceDiffuser is essentially a very cheap
                upgrade over the diffusion-only baseline, i.e. you get long-term memory
                at negligible additional cost.""",
                        kind="success",
                    )
                ],
                justify="center",
            ),
        ],
        gap=1,
    )
    return


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md("## Limitations & Open Questions"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                The paper puts out very explicitly what does not yet work well.

                **Low-dimensional bottleneck.** The SSM state has dimension 256.
                In extended rollouts, especially in visually complex environments
                like CSGO. 
                In these, fine-grained detail is frequently lost as the state is
                compressed over many steps so that scaling the SSM (more heads, layers,
                larger state) is expected to help but was not explored within
                this compute budget.

                **Lightweight diffusion network.** DIAMOND is a relatively
                small UNet. Replacing it with a larger pretrained model
                (e.g., a video diffusion transformer) could dramatically
                improve visual sharpness, without changing the method.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **What this means for the field**

                The results propose a very interesting discussion for world models:
                *how should memory and generation be decoupled?*

                State models are a natural fit and very well understood in the math literature,
                even more so for sequential generative task where context windows are usually the compute source.
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
def _():
    mo.vstack(
        [
            mo.md("## Limitations & Open Questions"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                The paper puts out very explicitly what does not yet work well.

                **Sensitivity to current-step noise.** While the model recovers
                from noisy *future* inputs, it is sensitive to noise in the
                *current* step. This is a known limitation of uusing sliding windows
                for the generation strategy.

                **No active memory management.** The SSM compresses everything
                equally, meaning there is no mechanism to explicitly prioritise
                task-relevant (or just overall more relevant in the sense of more information) memories over irrelevant ones. 
                Selective attention over the state can be a promising extension.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **Future directions suggested by the authors**

                Scaling the SSM state dimension and depth.
                Replacing DIAMOND with a larger diffusion backbone.
                Applying the approach to real-world ego-centric video (point of view).
                Combining with agent planning where the persistent state
                could serve as a planning memory for RL agents.
                """),
                                kind="neutral",
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
def _():
    mo.vstack(
        [
            mo.md("## Conclusion"),
            mo.callout(
                mo.md("""
                    **StateSpaceDiffuser** shows that its possible to both
                    *remember the past* and *generate performantly*.
                    A lightweight SSM branch costing < 2% of inference gives a
                    diffusion world model persistent long-term memory, enabling it to
                    maintain temporal coherence for an order of magnitude more steps
                    than a diffusion-only baseline.
        """),
                kind="success",
            ),
            mo.vstack(
                [
                    mo.md("### Three main takeaways of the paper:"),
                    mo.hstack(
                        [
                            mo.md("""
                **Problem.** Diffusion world models forget everything beyond
                a short window of $K$ frames, causing temporal drift on
                long interactions.
                """),
                            mo.md("""

                **Idea.** A Mamba SSM processes the full history in $O(T)$
                time and constant memory, compressing it into a persistent
                state that is injected into the diffusion model via a
                learned fusion module.
    """),
                            mo.md("""
                **Result.** +51.9% PSNR on MiniGrid at horizon 50.
                Positive user preference on CSGO.
                Zero-shot generalisation to 3× longer contexts.
                All for less than 2% extra inference cost.
                """),
                        ],
                        gap=1,
                    ),
                ],
                gap=2,
            ),
        ],
        gap=1,
    )
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Conclusion"),
            mo.callout(
                mo.md("""
                    **StateSpaceDiffuser** shows that its possible to both
                    *remember the past* and *generate performantly*.
                    A lightweight SSM branch costing < 2% of inference gives a
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
                            mo.md("Additional references made"),
                            mo.hstack(
                                [
                                    mo.md("- Gu & Dao (2023): Mamba"),
                                    mo.md("- Alonso et al. (2024): DIAMOND"),
                                    mo.md("- Hafner et al. (2023): DreamerV3"),
                                ],
                                gap=1,
                            ),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.md("Paper references"),
                            mo.md("""
                - **Paper:** arXiv:2505.22246
                - **Project page:**
                https://insait-institute.github.io/StateSpaceDiffuser/

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
    mo.md(r"""
    # Appendix
    """)
    return


@app.cell
def _():
    mo.vstack(
        [
            mo.md("## Criticism & Gaps Not Addressed"),
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
    **1. Two-stage training is a workaround, not a feature.**

    The paper frames decoupled training as a design choice, but it is
    really a response to failure: end-to-end training *collapses*
    because the diffusion model learns to ignore the SSM features.
    The frozen SSM never benefits from the generative objective, so
    the features it produces are *not* optimised for the diffusion
    model's needs meaning they are a fixed, suboptimal interface.
                                """),
                                kind="warn",
                            ),
                            mo.callout(
                                mo.md("""
    **2. Weak baselines, they use only DIAMOND and their own SSM.**

    The paper cites concurrent work (e.g., [[51]](https://arxiv.org/abs/2505.18236),
    [[74]](https://arxiv.org/abs/2407.07764), [[81]](https://arxiv.org/abs/2409.01720))
    that also address long-context consistency by sampling historical
    frames, but does not benchmark against any of them. The only
    comparison target is a 4-frame diffusion model, making the
    "order of magnitude" claim less impressive than it sounds.
                                """),
                                kind="warn",
                            ),
                            mo.callout(
                                mo.md("""
    **3. Tiny user study (N=12) for the only real-world domain.**

    PSNR/SSIM are used for MiniGrid, but for CSGO which is a visually complex
    environment that matters for practical deployment the results use a
    subjective study with just 12 participants. The preference ratings
    (0.20–0.24) are close to borderline (0). Statistical significance
    is not reported.
                                """),
                                kind="warn",
                            ),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
    **4. Severe information bottleneck.**

    The entire history is compressed into a 256-dimensional state.
    For CSGO (51 action types, complex 3D geometry), this is
    extremely lossy. The paper acknowledges detail decay but does
    not quantify *what* information is lost or provide ablations
    at different state sizes.
                                """),
                                kind="warn",
                            ),
                            mo.callout(
                                mo.md("""
    **5. Flattened spatial features lose locality.**

    The SSM processes full-frame Cosmos features flattened into a
    single vector per timestep. The paper argues this avoids
    "conflating spatial and temporal dependencies," but it also
    means the SSM cannot provide spatially-localised memory
    (*"what was in the top-left corner 50 steps ago?"*).
                                """),
                                kind="warn",
                            ),
                            mo.callout(
                                mo.md("""
    **6. Context lengths are still modest.**

    "Long context" means 50–150 steps. For real-world deployment
    (autonomous driving at 30 fps = 5 seconds), this is very short.
    The PSNR drops from 39.7 (ctx 50) to 30.8 (ctx 150), suggesting
    degradation under genuine scaling, not the flat curve one would
    hope for from a truly persistent memory.
                                """),
                                kind="warn",
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


@app.cell
def _():
    mo.vstack(
        [
            mo.md("# Additional Results"),
            mo.vstack(
                [
                    mo.image("media/images/recall_performance.png", width=900),
                    mo.image("media/images/results.png", width=620),
                ],
                gap=2,
                align="center",
            ),
        ]
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
