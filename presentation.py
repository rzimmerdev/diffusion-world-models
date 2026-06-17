import marimo

__generated_with = "0.23.8"
app = marimo.App(
    width="full",
    layout_file="layouts/presentation.slides.json",
    css_file="custom.css",
)

async with app.setup(hide_code=True):
    import marimo as mo
    import sys

    if sys.platform == "emscripten":
        import micropip
        await micropip.install("bibtexparser")

    def render_scene(module_name: str, class_name: str, quality="high_quality", height=None, width=None):
        import sys, pathlib, importlib
        if sys.platform == "emscripten":
            return mo.video(f"media/videos/1080p60/{class_name}.mp4", width=800)
        from manim import config
        root = pathlib.Path(__file__).parent
        out = root / "media" / "videos" / "1080p60" / f"{class_name}.mp4"
        if not out.exists():
            mod = importlib.import_module(module_name)
            scene_cls = getattr(mod, class_name)
            config.output_file = class_name
            scene_cls().render()
        return mo.video(src=str(out), height=height, width=width)

    def display_chart(module_name: str, class_name: str, name: str, fmt: str = "png", dpi: int = 150, **kwargs):
        import sys, pathlib
        out = pathlib.Path("media") / "images" / f"{name}.{fmt}"

        if sys.platform == "emscripten":
            return mo.image(src=str(out))

        import importlib
        mod = importlib.import_module(module_name)
        chart = getattr(mod, class_name)(**kwargs)

        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(chart.render(fmt=fmt, dpi=dpi).read())

        return mo.image(src=str(out))

    def make_slide(content, tag=None, title=None):
        tag_map = {
            "intro": '<span class="tag-d tag-intro-d">01 intro</span>',
            "theory": '<span class="tag-d tag-theory-d">02 theory</span>',
            "proposal": '<span class="tag-d tag-method-d">03 Proposal</span>',
            "results": '<span class="tag-d tag-results-d">04 results</span>',
            "discussion": '<span class="tag-d tag-critique-d">05 Discussion</span>',
        }

        tag_html = tag_map[tag] if tag else ""

        # Use custom title or default
        slide_title = title if title else "StateSpaceDiffuser - Bringing Long Context to Diffusion World Models"
        header_footer = mo.Html(f"""
    <style>
      .slide-header {{
        position: fixed;
        top: -0.5vw;
        left: -1vw;
        right: 1vw;
        box-sizing: border-box;
        z-index: 99999;
        padding: 8px 16px;
        border-bottom: 1px solid transparent;
        background-image: linear-gradient(to right, #f0f4f8, #f0f4f8, #ffffff),
                          linear-gradient(to right, #0a2a3a 70%, #178ad1);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        font-weight: bold;
        color: #1a1a18;
        display: flex;
        justify-content: space-between;
        align-items: center;
      }}
      .slide-footer {{
        position: fixed;
        display: flex;
        justify-content: space-between;
        align-items: center;
        bottom: -0.5vw;
        left: -1vw;
        right: 1vw;
        box-sizing: border-box;
        z-index: 99999;
        padding: 8px 16px;
        border-top: 1px solid transparent;
        background-image: linear-gradient(to right, #f0f4f8, #f0f4f8, #ffffff),
                          linear-gradient(to right, #0a2a3a 70%, #178ad1);
        background-origin: border-box;
        background-clip: padding-box, border-box;
        font-size: 0.85em;
        color: #444;
      }}
      .footer-center {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        white-space: nowrap;
      }}
      .footer-center img {{
        height: 16px;
        width: auto;
        vertical-align: middle;
        display: inline-block;
      }}
      .slide-tag {{
        margin-left: 16px;
      }}
    </style>
    <div class="slide-header">
      <span>{slide_title}</span>
      <span class="slide-tag">{tag_html}</span>
    </div>
    <div class="slide-footer">
      <span>Seminar by Rafael Zimmer</span>
      <div class="footer-center" style="padding-right: 30px">
        <img width="16" src="https://upload.wikimedia.org/wikipedia/commons/thumb/c/c8/Logo_of_the_Technical_University_of_Munich.svg/500px-Logo_of_the_Technical_University_of_Munich.svg.png" alt="TUM Logo">
        <span> Master-Seminar on World Models (IN2107, IN45153)</span>
      </div>
    </div>""")

        top_spacer = mo.Html('<div style="margin-top: 70px;"></div>')
        bottom_spacer = mo.Html('<div style="margin-bottom: 60px;"></div>')

        return mo.vstack([header_footer, top_spacer, content, bottom_spacer])


@app.cell(hide_code=True)
def _():
    mo.vstack(
        [
            mo.md("""
        # StateSpaceDiffuser
        ## Bringing Long Context to Diffusion World Models

        **Nedko Savov · Naser Kazemi · Deheng Zhang · Danda Pani Paudel · Xi Wang · Luc Van Gool**

        INSAIT, Sofia University · ETH Zurich · TU Munich

        *NeurIPS 2025 (10.48550/arXiv.2505.22246)*
        """),
            mo.callout(
                mo.md(
                    "This seminar walks through the problem, the math, the key idea, and what it achieves, more or less in that order."
                ),
                kind="info",
            ),
            mo.md("Seminar by Rafael Zimmer @ [rzimmerdev.github.io/diffusion-world-models](https://rzimmerdev.github.io/diffusion-world-models)"),
        ],
        gap=0.2,
        align="center",
    )
    return


@app.cell(hide_code=True)
def _():
    def create_slide_index(disabled_slides=None):
        """
        Create an interactive slide index HTML.

        Args:
            disabled_slides: List of slide numbers (as strings or ints) to grey out.
                            Example: ["03", "07", "12"] or [3, 7, 12]

        Returns:
            HTML string for the slide index
        """
        if disabled_slides is None:
            disabled_slides = []

        # Convert to strings for consistent comparison
        disabled_slides = [str(slide).zfill(2) if isinstance(slide, int) else str(slide) for slide in disabled_slides]

        # Define slide data structure
        slides_data = {
            "intro": [
                {"num": "00", "title": "Title & Overview", "tag": "intro"},
                {"num": "01", "title": "What Is a World Model?", "tag": "intro"},
                {"num": "02", "title": "Core Problem: Temporal Drift", "tag": "intro"}
            ],
            "theory": [
                {"num": "03", "title": "Diffusion Models", "tag": "theory"},
                {"num": "04", "title": "State-Space Models", "tag": "theory"},
                {"num": "05", "title": "Other approaches", "tag": "theory"}
            ],
            "proposal": [
                {"num": "06", "title": "StateSpaceDiffuser", "tag": "method"},
                {"num": "07", "title": "Architecture", "tag": "method"},
                {"num": "08", "title": "Training Procedure", "tag": "method"},
                {"num": "09", "title": "Evaluating Long-Context Memory", "tag": "method"}
            ],
            "results": [
                {"num": "10", "title": "Quantitative Evaluation", "tag": "results"},
                {"num": "11", "title": "Quantitative Evaluation (Minigrid & CSGO)", "tag": "results"},
                {"num": "12", "title": "Ablation Study & Analysis", "tag": "results"}
            ],
            "discussion": [
                {"num": "13", "title": "Limitations & Open Questions", "tag": "critique"},
                {"num": "14", "title": "Criticism & Gaps", "tag": "critique"},
                {"num": "15", "title": "Conclusion", "tag": "critique"}
            ]
        }

        # Helper function to generate card HTML
        def generate_card(slide):
            is_disabled = slide["num"] in disabled_slides
            disabled_class = 'slide-card-d-disabled' if is_disabled else 'slide-card-d'

            return f"""
            <div class="{disabled_class}" data-slide="{slide['num']}">
              <span class="slide-num-d">{slide['num']}</span>
              <span class="slide-title-d">{slide['title']}</span>
            </div>
            """

        # Build the HTML sections
        sections = []

        # Intro section
        intro_cards = '\n'.join([generate_card(slide) for slide in slides_data["intro"]])
        sections.append(f"""
        <div>
          <p class="section-label-d">
            <span class="tag-d tag-intro-d">01 intro</span>
          </p>
          <div style="display:flex;flex-direction:column;gap:10px;">
            {intro_cards}
          </div>
        </div>
        """)

        # Theory section
        theory_cards = '\n'.join([generate_card(slide) for slide in slides_data["theory"]])
        sections.append(f"""
        <div>
          <p class="section-label-d">
            <span class="tag-d tag-theory-d">02 theory</span>
          </p>
          <div style="display:flex;flex-direction:column;gap:10px;">
            {theory_cards}
          </div>
        </div>
        """)

        # Proposal section
        proposal_cards = '\n'.join([generate_card(slide) for slide in slides_data["proposal"]])
        sections.append(f"""
        <div>
          <p class="section-label-d">
            <span class="tag-d tag-method-d">03 Proposal</span>
          </p>
          <div style="display:flex;flex-direction:column;gap:10px;">
            {proposal_cards}
          </div>
        </div>
        """)

        # Results section
        results_cards = '\n'.join([generate_card(slide) for slide in slides_data["results"]])
        sections.append(f"""
        <div>
          <p class="section-label-d">
            <span class="tag-d tag-results-d">04 results</span>
          </p>
          <div style="display:flex;flex-direction:column;gap:10px;">
            {results_cards}
          </div>
        </div>
        """)

        # Discussion section
        discussion_cards = '\n'.join([generate_card(slide) for slide in slides_data["discussion"]])
        sections.append(f"""
        <div>
          <p class="section-label-d">
            <span class="tag-d tag-critique-d">05 Discussion</span>
          </p>
          <div style="display:flex;flex-direction:column;gap:10px;">
            {discussion_cards}
          </div>
        </div>
        """)

        # Complete HTML with CSS
        html = f"""
    <style>
    .dark-wrap {{
      border-radius: 16px;
      padding: 1.5rem;
      background: #f5f5f3;
    }}
    .index-grid-d {{
      display: grid;
      grid-template-columns: repeat(auto-fit, 200px);
      justify-content: space-around;
      gap: 6px;
    }}
    .slide-card-d {{
      background: #ffffff;
      border: 0.5px solid rgba(0,0,0,0.1);
      border-radius: 12px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      transition: border-color 0.15s, background 0.15s;
      cursor: pointer;
    }}
    .slide-card-d:hover {{
      border-color: rgba(0,0,0,0.2);
      background: #f0f0ee;
    }}
    .slide-card-d-disabled {{
      background: #ebebea;
      border: 0.5px solid rgba(0,0,0,0.06);
      border-radius: 12px;
      padding: 14px 16px;
      display: flex;
      flex-direction: column;
      gap: 6px;
      cursor: not-allowed;
      opacity: 0.4;
      filter: grayscale(0.3);
    }}
    .slide-card-d-disabled .slide-num-d,
    .slide-card-d-disabled .slide-title-d {{
      color: #aaa;
    }}
    .slide-num-d {{
      font-size: 16px;
      font-weight: 500;
      letter-spacing: 0.04em;
      color: #a0a0a0;
    }}
    .slide-title-d {{
      font-size: 18px;
      font-weight: 500;
      color: #1a1a18;
      line-height: 1.4;
    }}
    .section-label-d {{
      font-size: 16px;
      font-weight: 500;
      letter-spacing: 0.07em;
      color: #b0b0b0;
      text-transform: uppercase;
      margin: 1.25rem 0 0.4rem;
      padding-left: 2px;
    }}
    .tag-d {{
      display: inline-block;
      font-size: 10px;
      font-weight: 500;
      padding: 2px 7px;
      border-radius: 999px;
      margin-top: 2px;
      width: fit-content;
    }}
    .tag-intro-d    {{ background: #ECEAF9; color: #4A44A0; }}
    .tag-theory-d   {{ background: #D6F3EC; color: #0A6B52; }}
    .tag-method-d   {{ background: #D6EAFA; color: #1A5A8A; }}
    .tag-results-d  {{ background: #E8F4D0; color: #3A6A08; }}
    .tag-critique-d {{ background: #FAE5DC; color: #8A3010; }}
    </style>
    <div class="dark-wrap">
      <div class="index-grid-d">
        {''.join(sections)}
      </div>
    </div>
        """

        return mo.Html(html)

    make_slide(create_slide_index())
    return (create_slide_index,)


@app.cell(hide_code=True)
def _():
    video_world_model = render_scene("animations.world_model", "WorldModelAnimation")
    return (video_world_model,)


@app.cell(hide_code=True)
def _(video_world_model):
    world_model = mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
    So, what is a world model?  
    How is it different from just a generative model?

    Given a trajectory of interactions $a_1, a_2, \\ldots, a_{T-1}$ producing observations $I_1, I_2, \\ldots, I_T,$ the world model $\\mathcal{F}$ predicts the next frame $I_{T+1}$:
    $$I_{T+1} = \\mathcal{F}\\bigl([I_1,\\ldots,I_T],\\,[a_1,\\ldots,a_T]\\bigr).$$"""),
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
            )
    make_slide(world_model, tag="intro", title="What Is a World Model?")
    return


@app.cell(hide_code=True)
def _():
    temporal_drift_0 = mo.hstack(
                [
                    mo.md(""),
                    mo.vstack(
                        [
                            mo.md("""
    Computation costs for diffusion approaches **increase exponentially** with $T$, thus context windows are limited to size $K$:
    $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$

    As $T \\gg K$ (after the agent explores a lot), the model loses memory of previous observations, called **temporal drift**."""),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            )

    make_slide(temporal_drift_0, "intro", title="Core Problem: Temporal Drift")
    return


@app.cell(hide_code=True)
def _():
    temporal_drift_2 = mo.hstack(
                [
                    mo.callout(
                        mo.md("""
    **Why not just increase $K$?**

    **Doubling** context **quadruples** compute.
                """),
                        kind="warn",
                    ),
                    mo.vstack(
                        [
                            mo.md("""
    Computation costs for diffusion approaches **increase exponentially** with $T$, thus context windows are limited to size $K$:
    $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$

    As $T \\gg K$ (after the agent explores a lot), the model loses memory of previous observations, called **temporal drift**."""),
                        ],
                        gap=1,
                    ),
                ],
                widths=[0.45, 0.55],
                gap=2,
            )
    make_slide(temporal_drift_2, "intro", "Core Problem: Temporal Drift")
    return


@app.cell
def _():
    video_compare_diffusion = render_scene("animations.compare_diffusion", "CompareDiffusionAnimation")
    mo.center(video_compare_diffusion)
    return


@app.cell
def _(create_slide_index):
    make_slide(create_slide_index(["00", "01", "02"]))
    return


@app.cell(hide_code=True)
def _():
    diffusion_models = mo.md("""
                **Forward process**: repeatedly corrupt a clean image $x_0$ by applying gaussian noise:
                $$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(x_t;\\,\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t I\\right)$$

                **Reverse process**: a neural network $\\epsilon_\\theta$ learns to
                undo one step of noise:
                $$p_\\theta(x_{t-1}\\mid x_t) = \\mathcal{N}\\!\\left(x_{t-1};\\,\\mu_\\theta(x_t, t),\\,\\Sigma_\\theta(x_t, t)\\right)$$
                """)

    make_slide(diffusion_models, "theory", title="Background: Diffusion Models")
    return


@app.cell
def _():
    mo.callout(mo.md("""
    Don't let the r.h.s. scare you, it's just a Gaussian distribution with mean and variance predicted by the neural network."""))
    return


@app.cell
def _():
    video_diffusion_denoise = render_scene("animations.diffusion_denoise", "DiffusionProcessAnimation")
    mo.center(video_diffusion_denoise)
    return


@app.cell
def _():
    diffusion_problem = mo.callout(
        mo.center(mo.md("""
    **Main point:** Diffusion models are excellent at generating
    high-fidelity images but carry no persistent state.""")
        ), 
    kind="info")

    make_slide(diffusion_problem, tag="theory", title="Background: Why Not Just Diffusion Models?")
    return


@app.cell
def _():
    state_space_models_0 = mo.center(mo.md("""
    A **State-Space Model (SSM)** maintains a hidden state $h_t$, simply a compact summary of the entire history so far, updated at every step of a sequence $f_1, \\ldots, f_T$

    $$h_t = A\\,h_{t-1} + B\\,f_t, \\qquad m_t = C\\,h_t$$"""))

    make_slide(state_space_models_0, "theory", "Background: State-Space Models")
    return


@app.cell
def _():
    mo.center(mo.md("""
    - $f_t$ are the input features/frames at time $t$.
    - $h_t$ is the hidden state.
    - $A, B, C$ are learnable matrices that weigh the hidden state and features.
    - $m_t$ is the output memory at time $t$.
    """))
    return


@app.cell
def _():
    mo.vstack([mo.center(mo.image("https://substackcdn.com/image/fetch/$s_!SFe0!,w_1456,c_limit,f_webp,q_auto:good,fl_progressive:steep/https%3A%2F%2Fsubstack-post-media.s3.amazonaws.com%2Fpublic%2Fimages%2F0876819d-8a46-4187-9826-14391bfd47b9_1796x624.png")),
    mo.center(mo.md("`A Visual Guide to Mamba and State-Space Models: An Alternative to Transformers for Language Modeling` (Maarten Grootendorst, 2024)"))
               ])
    return


@app.cell
def _():
    state_space_models_2 = mo.md("""\n
    The authors use a SSM with **Selective gating**. This *upgrade* dynamically decides what information to retain or discard.  
    Think of the **memory cells** in LSTM's and GRU's gates, but applied to a decoupled latent space.

    $$\\Delta, B, C = \\text{Linear}(f_t), \\quad \\bar{A} = e^{\\Delta A}$$""")

    make_slide(state_space_models_2, "theory", title="Background: State-Space Models")
    return


@app.cell(hide_code=True)
def _():
    video_ssm = render_scene("animations.ssm", "SSMAnimation")
    mo.center(video_ssm)
    return


@app.cell(hide_code=True)
def _():
    complexity_comparison = mo.vstack([
        mo.center(
            mo.callout(
                mo.Html("""
    <table style="width: 600px; border-collapse: collapse; font-family: monospace; font-size: 14px;">
      <thead>
        <tr style="background-color: #006bad; color: white; border-bottom: 2px solid #1d3a35;">
          <th style="padding: 12px; text-align: left; width: 25%;">Model</th>
          <th style="padding: 12px; text-align: left; width: 37.5%;">Training</th>
          <th style="padding: 12px; text-align: left; width: 37.5%;">Inference memory</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom: 0px solid #1d3a35;">
          <td style="padding: 10px;">Transformer</td>
          <td style="padding: 10px;">O(T²)</td>
          <td style="padding: 10px;">O(T)</td>
         </tr>
        <tr style="border-bottom: 0px solid #1d3a35;">
          <td style="padding: 10px;">CNN</td>
          <td style="padding: 10px;">O(T)</td>
          <td style="padding: 10px;">O(K) fixed</td>
         </tr>
        <tr style="border-bottom: 0px solid #1d3a35; background-color: #c8e7e8;">
          <td style="padding: 10px; font-weight: bold;">SSM (Mamba)</td>
          <td style="padding: 10px; font-weight: bold;">O(T)</td>
          <td style="padding: 10px; font-weight: bold;">O(1) constant</td>
         </tr>
      </tbody>
    </table>"""),
            )
        )
    ], align="center")

    make_slide(complexity_comparison, tag="theory", title="Background: State-Space Models compared to other approaches")
    return


@app.cell
def _():
    gap = mo.hstack([
        mo.md(""), 
        mo.vstack([
            mo.center(mo.md("**The gap the StateSpaceDiffuser fills**")),
            mo.md("""High-fidelity image generation using diffusion and persistent long-term memory (state-space) have not been combined in a world model.""")
        ]),
        mo.md("")
    ], widths=[0.1, 0.8, 0.1])

    make_slide(gap, tag="theory", title="Background: Gap in Literature")
    return


@app.cell
def _(create_slide_index):
    make_slide(create_slide_index(["00", "01", "02", "03", "04", "05"]))
    return


@app.cell
def _():
    proposal_0 = mo.vstack(
        [
            mo.center(mo.md("## The Proposal: StateSpaceDiffuser")),
            mo.callout(
                mo.md("""
    **Core idea:** A SSM, namely Mamba, processes the *entire* history $O(T)$ and compresses it into a state that is injected into a diffusion model.
        """),
                kind="info",
            )
        ],
        gap=1,
    )

    make_slide(proposal_0, tag="proposal", title=" ")
    return


@app.cell
def _():
    proposal_1 = mo.vstack(
        [
            mo.callout(
                mo.md("""
    **Core idea:** A SSM, namely Mamba, processes the *entire* history $O(T)$ and compresses it into a state that is injected into a diffusion model.
        """),
                kind="info",
            ),
            mo.md("""
    ## Long-Context Branch (SSM)

    Each frame $I_t$ is encoded by the
    **Cosmos tokenizer** into a compact feature vector $f_t \\in \\mathbb{R}^d$.  
    The Mamba SSM then processes the full sequence:
    $$\\hat{f}_2, \\ldots, \\hat{f}_{T+1} = \\mathcal{M}([f_1, a_1], \\ldots, [f_T, a_T])$$
                """),
        ],
        gap=1,
    )

    make_slide(proposal_1, tag="proposal", title="The Proposal: StateSpaceDiffuser")
    return


@app.cell
def _():
    proposal_2 = mo.vstack(
        [
            mo.callout(
                mo.md("""
    **Core idea:** A SSM, namely Mamba, processes the *entire* history $O(T)$ and compresses it into a state that is injected into a diffusion model.
        """),
                kind="info",
            ),
            mo.md("""
    ## Generative Branch (Diffusion)

    The DIAMOND diffusion model generates the next frame conditioned on a **short window** of 4 frames *plus* the SSM's long-context features $\\hat{f}_t$.  
    The SSM's features are then fused and passed to the diffusion model.
                """),
        ],
        gap=1,
    )

    make_slide(proposal_2, tag="proposal", title="The Proposal: StateSpaceDiffuser")
    return


@app.cell
def _():
    architecture = mo.image("media/images/architecture.png", width=800)

    make_slide(mo.center(architecture), tag="proposal", title="Detailed Architecture Diagram")
    return


@app.cell
def _():
    training_protocol_0 = mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### Stage 1 -  **Train the Long-Context Branch**

                The SSM is trained on long sequences to predict next-frame features from the full history.

                $$\\mathcal{L}_{\\text{SSM}} = \\sum_{t=1}^{T} \\|\\hat{f}_{t+1} - f_{t+1}\\|^2$$


                ### Stage 2 -  **Train the Generative Branch**

                With the SSM *frozen*, the diffusion model
                is trained on short sequences,
                conditioned on frozen SSM features.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [

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

    make_slide(training_protocol_0, "proposal", "Training Protocol")
    return


@app.cell
def _():
    training_protocol_1 = mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                ### Stage 1 - **Train the Long-Context Branch**

                The SSM is trained on long sequences to predict next-frame features from the full history.

                $$\\mathcal{L}_{\\text{SSM}} = \\sum_{t=1}^{T} \\|\\hat{f}_{t+1} - f_{t+1}\\|^2$$


                ### Stage 2 - **Train the Generative Branch**

                With the SSM *frozen*, the diffusion model
                is trained on short sequences,
                conditioned on frozen SSM features.
                """),
                        ],
                        gap=1,
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **Why freeze the SSM?**

                Direct **end-to-end training is unstable**! Diffusion's noisy gradients destabilise the SSM $\\rightarrow$ Diffusion learns to ignore the SSM entirely.
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

    make_slide(training_protocol_1, "proposal", "Training Protocol")
    return


@app.cell
def _():
    video_two_stage = render_scene("animations.two_stage", "TwoStageTraining")
    mo.center(video_two_stage)
    return


@app.cell
def _():
    long_context_0 = mo.md("""
    ## The Forward-Backward Protocol
    The agent takes $n$ actions **forward**, then $n$ mirrored actions **backward**. The second half of the sequence should be *identical*.""")

    make_slide(long_context_0, tag="proposal", title="Evaluating Long-Context Memory")
    return


@app.cell
def _():
    video_long_context = render_scene("animations.long_context", "LongContextComparison")
    mo.hstack([mo.md(""), video_long_context, mo.md("")], widths=[0.1, 0.6, 0.1])
    return


@app.cell
def _(create_slide_index):
    make_slide(create_slide_index(["00", "01", "02", "03", "04", "05", "06", "07", "08", "09"]))
    return


@app.cell
def _():
    quantitative_evaluation = mo.vstack(
        [
            mo.md("""
        The authors stress-test the long-term memory using the following environments:
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

    **CSGO** is a 3D first-person shooter with 51 action types.
    Performance is evaluated with a **user study**.
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

    make_slide(quantitative_evaluation, tag="results", title="Quantitative Evaluation")
    return


@app.cell
def _():
    mo.center(mo.video("media/videos/paper_video_short.mp4"))
    return


@app.cell
def _():
    results_comparison = mo.center(mo.md("""
    Three approaches are compared:

    **Pure diffusion** (DIAMOND, with no long-term memory),

    **SSM branch alone** (SS-WM, State-Space World Model),

    **StateSpaceDiffuser** (the authors approach)."""))

    make_slide(results_comparison, tag="results", title="Results: MiniGrid Quantitative Evaluation")
    return


@app.cell
def _():
    psnr = display_chart(
        "charts.psnr", "PSNRBarChart", "psnr",
        models=["DIAMOND", "SSVM", "SSD (ours)"],
        colors=["#e05c5c", "#5cb8e0", "#5ce08a"],
        ctx16_avg=[27.13, 33.40, 41.01],
        ctx16_fin=[25.44, 33.17, 40.55],
        ctx50_avg=[26.13, 32.64, 39.68],
        ctx50_fin=[25.15, 32.44, 39.32],
    )

    results_minigrid = mo.hstack([mo.md(""), psnr, mo.md("")], widths=[0.1, 0.7, 0.1])

    make_slide(results_minigrid, tag="results", title="Results: MiniGrid Quantitative Evaluation")
    return


@app.cell(hide_code=True)
def _():
    mo.callout(
        mo.center(mo.md("StateSpaceDiffuser achieves **+51.9% average PSNR improvement** over the DIAMOND baseline.")),
    kind="success")
    return


@app.cell(hide_code=True)
def _():
    results_csgo_0 = mo.hstack([
            mo.md("""
    ### CSGO User Study

    12 participants rate between StateSpaceDiffuser or DIAMOND baseline.  Rating scale: $[-1, 1]$

    **Results:**

    - Frame 15 (second-to-last): **+0.20**
    - Frame 17 (final, hardest): **+0.24**"""),
            mo.callout(
                mo.md("""
    **Generalisation without finetuning**

    While performance scales with context size $K$, the DIAMOND model has to be retrained but SSD can be used as is."""),
            kind="info")
                ], widths=[0.4, 0.6], gap=2)

    make_slide(results_csgo_0, tag="results", title="Results: CSGO & Generalisation to Longer Contexts")
    return


@app.cell
def _():
    mo.callout(mo.center(mo.md("""This generalisation follows directly from the SSM's recurrence depending always only on $h_{t-1}$ and $f_t$.""")))
    return


@app.cell
def _():
    ablation_0 = mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
    The authors additionally perform an ablation study on the SSM features.  
    On CSGO the model without SSM features **hallucinates** content (plausible-looking but entirely wrong scenes).

    **Performance drops below** the DIAMOND baseline (23.68 vs 27.13 Avg PSNR at context 16)
                """),
                        ],
                        gap=1,
                    ),
                    mo.image("media/images/ablation.png"),
                ],
                widths=[0.35, 0.75],
                gap=2,
            ),
        ],
        gap=1,
    )

    make_slide(ablation_0, tag="results", title="Ablation: Do the SSM Features Actually Matter?")
    return


@app.cell(hide_code=True)
def _():
    ablation_1 = capability_comparison = mo.vstack([
        mo.center(
            mo.callout(
                mo.Html("""
    <table style="width: 800px; border-collapse: collapse; font-family: monospace; font-size: 14px;">
      <thead>
        <tr style="background-color: #006bad; color: white; border-bottom: 2px solid #1d3a35;">
          <th style="padding: 12px; text-align: left; width: 25%;">Capability</th>
          <th style="padding: 12px; text-align: left; width: 25%;">Diffusion alone</th>
          <th style="padding: 12px; text-align: left; width: 25%;">SSM alone</th>
          <th style="padding: 12px; text-align: left; width: 25%;">SSD</th>
        </tr>
      </thead>
      <tbody>
        <tr style="border-bottom: 0px solid #1d3a35;">
          <td style="padding: 10px;">Long-term memory</td>
          <td style="padding: 10px;">No</td>
          <td style="padding: 10px;">Yes</td>
          <td style="padding: 10px;">Yes</td>
        </tr>
        <tr style="border-bottom: 0px solid #1d3a35;">
          <td style="padding: 10px;">High visual fidelity</td>
          <td style="padding: 10px;">Yes</td>
          <td style="padding: 10px;">No</td>
          <td style="padding: 10px;">No</td>
        </tr>
        <tr style="border-bottom: 0px solid #1d3a35;">
          <td style="padding: 10px;">Scales with T</td>
          <td style="padding: 10px;">No</td>
          <td style="padding: 10px;">Yes</td>
          <td style="padding: 10px;">Yes</td>
        </tr>
        <tr style="border-bottom: 0px solid #1d3a35; background-color: #c8e7e8;">
          <td style="padding: 10px; font-weight: bold;">O(1) inference cost</td>
          <td style="padding: 10px; font-weight: bold;">-</td>
          <td style="padding: 10px; font-weight: bold;">Yes</td>
          <td style="padding: 10px; font-weight: bold;">Yes</td>
        </tr>
      </tbody>
    </table>"""),
            )
        )
    ], align="center")

    make_slide(ablation_1, tag="results", title="Ablation: Do the SSM Features Actually Matter?")
    return


@app.cell
def _():
    components = display_chart(
        "charts.components", "ComponentBreakdownChart", "components",
        components=["DIAMOND\n(diffusion)", "SSM\n(Mamba)", "Fusion\nMLP"],
        sizes=[98, 1.5, 0.5],
        colors=["#e05c5c", "#5ce08a", "#5cb8e0"],
    )
    return (components,)


@app.cell(hide_code=True)
def _(components):
    cost =  mo.vstack([
                mo.hstack([mo.md(""), mo.center(components), mo.md("")], widths=[0.1, 0.7, 0.1]),
            
            ])

    make_slide(cost, tag="results", title="Computational Cost (Per each component)")
    return


@app.cell
def _():
    mo.center(mo.callout(mo.md("""StateSpaceDiffuser is essentially a **very cheap upgrade** over the diffusion-only baseline, i.e. you get long-term memory at negligible additional cost."""), kind="success"))
    return


@app.cell
def _(create_slide_index):
    make_slide(create_slide_index(["00", "01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]))
    return


@app.cell(hide_code=True)
def _():
    limitations_0 = mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("""
    The paper puts out very explicitly what does not yet work well.

    **Low-dimensional bottleneck.**
    Detail is frequently lost as state is compressed, but scaling the SMM was not explored.

    **Lightweight diffusion network.** 
    Replacing DIAMOND with a larger model could dramatically improve visual sharpness."""),
            ], gap=1),
            mo.vstack([
                mo.callout(
                    mo.md("""
    **What this means for the field**

    The results show that **decoupling memory and generation** is a valid approach."""),
                kind="info"),
            ], gap=1),
        ], widths=[0.55, 0.45], gap=2),
    ], gap=1)

    make_slide(limitations_0, tag="discussion", title="Limitations & Open Questions")
    return


@app.cell(hide_code=True)
def _():
    limitations_1 = mo.vstack([
        mo.hstack([
            mo.vstack([
                mo.md("""
    The paper puts out very explicitly what does not yet work well.

    **Sensitivity to current-step noise.** 
    While model can deal with *future* inputs, it is still sensitive to noise in *current* step.

    **No active memory management.** 
    The SSM compresses everything equally, meaning there is no mechanism to explicitly prioritise task-relevant memories over irrelevant ones."""),
            ], gap=1),
            mo.vstack([
                mo.callout(
                    mo.md("""
    **Future directions suggested by the authors**

    - Scaling the SSM state dimension and depth.  
    - Replacing DIAMOND with a larger diffusion backbone."""),
                kind="neutral"),
            ], gap=1),
        ], widths=[0.55, 0.45], gap=2),
    ], gap=1)

    make_slide(limitations_1, tag="discussion", title="Limitations & Open Questions")
    return


@app.cell(hide_code=True)
def _():
    conclusion_0 = mo.vstack([
        mo.callout(
            mo.vstack([
                mo.md("""
    **StateSpaceDiffuser** shows that its possible to both ***remember the past*** and ***generate performantly***.  
    The lightweight SSM branch costs < 2% of inference and maintains temporal coherence for an order of magnitude more steps than a diffusion-only baseline."""),
                mo.hstack([
                    mo.image("media/images/qrcode.png", width=64),
                    mo.md("[Interactive 2D Grid-Maze running a StateSpaceDiffuser World Model](https://rzimmerdev.github.io/diffusion-world-models/game)"),
                ], gap=4, justify="start")
            ]), 
            kind="success"
        ),
        mo.vstack([
            mo.md("### Three main takeaways of the paper:"),
            mo.hstack([
                mo.md("""
    **Problem.** 
    Diffusion world models forget observations beyond a short window of $K$ frames."""),
                mo.md("""
    **Idea.** 
    A state-space model can process full history in $O(1)$ memory, compressing context into a persistent state."""),
                mo.md("""
    **Result.** 
    +51.9% PSNR on MiniGrid. Zero-shot generalisation to 3× longer contexts."""),
            ], gap=1, widths=[0.28, 0.36, 0.36]),
        ], gap=2),
    ], gap=1)

    make_slide(conclusion_0, tag="discussion", title="Conclusion")
    return


@app.cell(hide_code=True)
def _():
    conclusion_1 = mo.vstack([
        mo.callout(
            mo.vstack([
                mo.md("""
    **StateSpaceDiffuser** shows that its possible to both ***remember the past*** and ***generate performantly***.  
    A lightweight SSM branch costing < 2% of inference gives a diffusion world model persistent long-term memory, 
    enabling it to maintain temporal coherence for an order of magnitude more steps than a diffusion-only baseline."""),
                mo.hstack([
                    mo.image("media/images/qrcode.png", width=64),
                    mo.md("[Interactive 2D Grid-Maze running a StateSpaceDiffuser World Model](https://rzimmerdev.github.io/diffusion-world-models/game)"),
                ], gap=4, justify="start")
            ])
        , kind="success"),
        mo.hstack([
            mo.vstack([
                mo.md("Principal references"),
                mo.hstack([
                    mo.md("Gu & Dao (2023)  Mamba"),
                    mo.md("Alonso et al. (2024)  DIAMOND"),
                    mo.md("Hafner et al. (2023)  DreamerV3"),
                ], gap=1),
            ], gap=1),
            mo.vstack([
                mo.md("Paper references"),
                mo.md("""
    - **Paper:** NeurIPS 2025 (arXiv:2505.22246)
    - **Project page:**
    https://insait-institute.github.io/StateSpaceDiffuser/
    - /EOS/ (end of seminar!!)
    """),
            ], gap=1),
        ], widths=[0.5, 0.5], gap=2)
    ], gap=1)

    make_slide(conclusion_1, tag="discussion", title="Conclusion")
    return


@app.cell(hide_code=True)
def bibliography_slide_1():
    def _():
        import bibtexparser

        ENTRIES_PER_PAGE = 7

        def clean(s):
            return s.replace("{", "").replace("}", "").replace("\n", " ")

        import sys, io
        if sys.platform == "emscripten":
            from pyodide.http import open_url
            _bib_text = open_url("references.bib").read()
        else:
            with open("references.bib") as f:
                _bib_text = f.read()
        db = bibtexparser.load(io.StringIO(_bib_text))

        all_entries = list(enumerate(db.entries, 1))
        lines = ["# References\n"]
        for i, entry in all_entries[:ENTRIES_PER_PAGE]:
            authors = clean(entry.get("author", ""))
            year = clean(entry.get("year", ""))
            title = clean(entry.get("title", ""))
            venue = clean(entry.get("journal") or entry.get("booktitle") or "")
            lines.append(f"**[{i}]** {authors} ({year}). *{title}*. {venue}.\n")

        return mo.vstack([mo.md("\n".join(lines))], align="start")

    _()
    return


@app.cell(hide_code=True)
def bibliography_slide_2():
    def _():
        import bibtexparser

        ENTRIES_PER_PAGE = 7

        def clean(s):
            return s.replace("{", "").replace("}", "").replace("\n", " ")

        import sys, io
        if sys.platform == "emscripten":
            from pyodide.http import open_url
            _bib_text = open_url("references.bib").read()
        else:
            with open("references.bib") as f:
                _bib_text = f.read()
        db = bibtexparser.load(io.StringIO(_bib_text))

        all_entries = list(enumerate(db.entries, 1))
        page_entries = all_entries[ENTRIES_PER_PAGE : ENTRIES_PER_PAGE * 2]
        mo.stop(not page_entries)

        lines = ["# References (cont.)\n"]
        for i, entry in page_entries:
            authors = clean(entry.get("author", ""))
            year = clean(entry.get("year", ""))
            title = clean(entry.get("title", ""))
            venue = clean(entry.get("journal") or entry.get("booktitle") or "")
            lines.append(f"**[{i}]** {authors} ({year}). *{title}*. {venue}.\n")

        return mo.vstack([mo.md("\n".join(lines))], align="start")

    _()
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Appendix
    """)
    return


@app.cell(hide_code=True)
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

    Decoupled training as a design choice is rather a response to end-to-end training *collapsing*
    because the diffusion model learns to ignore the SSM features.
    The frozen SSM does not use the generative objective, i.e. the features it produces are *suboptimal* for the diffusion model's needs.
                                """),
                                kind="warn",
                            ),
                            mo.callout(
                                mo.md("""
    **3. Weak baselines, they use only DIAMOND and their own SSM.**

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
    **5. Tiny user study (N=12) for the only real-world domain.**

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
    **2. Severe information bottleneck.**

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
    **4. Flattened spatial features lose locality.**

    The SSM processes full-frame Cosmos features flattened into a
    single vector per timestep. The paper argues this avoids
    "conflating spatial and temporal dependencies," but it also
    means the SSM cannot provide spatially-localised memory
    (*"what was in the top-left corner 50 steps ago?"* This is also why in the tests we can that when regions are forgotten it happens completely instead of just pixel-wise).
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
    mo.md("""
    # Possible Questions
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    # RNNs vs. Diffusion Models for Image Generation
    RNNs can generate images autoregressively (pixel-by-pixel), but are slow, struggle with long-range spatial dependencies, and don't scale to high resolutions.
    Diffusion models are superior (in this sense) because they

    - **Process the whole image at once, not sequentially**
    - Capture global structure naturally
    - Train stably with a simple noise-prediction loss
    - **Scale** to high-resolution outputs
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
