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

        *arXiv:2505.22246  October 2025*
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
                This section will shortly cover the formal definition of a world model and move on to more interesting aspects of the StateSpaceDiffuser model.

                Given a trajectory of interactions $a_1, a_2, \\ldots, a_{T-1}$ producing observations $I_1, I_2, \\ldots, I_T,$ the world model $\\mathcal{F}$ predicts the next frame:
                $$I_{T+1} = \\mathcal{F}\\bigl([I_1,\\ldots,I_T],\\,[a_1,\\ldots,a_T]\\bigr).$$
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
    Diffusion models have a hard architectural constraint due to computation costs 
    $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$
    (The input of $I_t$ and $a_t$ are thus restricted up to only $K$ steps in the past).

    As the agent explores and later revisits a location, the model has no memory of what it looked like, that is, the scene *drifts*, also called **temporal drift**.
                """),
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
    Diffusion models have a hard architectural constraint due to computation costs
    $$I_{T+1} = \\mathcal{F}\\bigl([I_{T-K+1},\\ldots,I_T],\\,[a_{T-K+1},\\ldots,a_T]\\bigr)$$
    (The input of $I_t$ and $a_t$ are thus restricted up to only $K$ steps in the past).

    As the agent explores and later revisits a location, the model has no memory of what it looked like, that is, the scene *drifts*, also called **temporal drift**.
                """),
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
    video_compare_diffusion
    return


@app.cell
def _(create_slide_index):
    make_slide(create_slide_index(["00", "01", "02"]))
    return


@app.cell(hide_code=True)
def _():
    diffusion_models = mo.hstack(
                [
                    mo.vstack(
                        [
                            mo.md("""
                **Forward process**: repeatedly corrupt a clean image $x_0$ by applying gaussian noise:
                $$q(x_t \\mid x_{t-1}) = \\mathcal{N}\\!\\left(x_t;\\,\\sqrt{1-\\beta_t}\\,x_{t-1},\\,\\beta_t I\\right)$$

                **Reverse process**: a neural network $\\epsilon_\\theta$ learns to
                undo one step of noise:
                $$p_\\theta(x_{t-1}\\mid x_t) = \\mathcal{N}\\!\\left(x_{t-1};\\,\\mu_\\theta(x_t, t),\\,\\Sigma_\\theta(x_t, t)\\right)$$
                """),
                        ],
                        gap=1,
                    ),

                ],
    #             widths=[0.55, 0.45],
                gap=2,
            )

    make_slide(diffusion_models, "theory", title="Background: Diffusion Models")
    return


@app.cell
def _():
    video_diffusion_denoise = render_scene("animations.diffusion_denoise", "DiffusionProcessAnimation")
    video_diffusion_denoise
    return


@app.cell
def _():
    mo.callout(
                                mo.center(mo.md("""
                **Main point:** Diffusion models are excellent at generating
                high-fidelity images but carry no persistent state.
                """)),
                                kind="info",
                            )
    return


@app.cell(hide_code=True)
def _():
    video_ssm = render_scene("animations.ssm", "SSMAnimation")
    return (video_ssm,)


@app.cell
def _():
    state_space_models_0 = mo.center(mo.md("""
    A **State-Space Model (SSM)** maintains a hidden state a compact summary $h_t$ of the entire history so far, updated at every step of a sequence $f_1, \\ldots, f_T$

    $$h_t = A\\,h_{t-1} + B\\,f_t, \\qquad m_t = C\\,h_t$$"""))

    make_slide(state_space_models_0, "theory", "Background: State-Space Models")
    return


@app.cell
def _():
    mo.center(mo.md("""
    - $f_t$ are the input features at time $t$ (e.g., encoded video frames).
    - $h_t$ is the hidden state, which is updated recurrently.
    - $A, B, C$ are learnable matrices that govern the state update and output.
    - $m_t$ is the output memory at time $t$, derived from the hidden state.
    """))
    return


@app.cell
def _():
    state_space_models_2 = mo.md("""\n
    *Selective gating* dynamically decides what information to retain or discard. Think of LSTM's and GRU's gates, but here we apply it to the state-space model's parameters.

    $$\\Delta, B, C = \\text{Linear}(f_t), \\quad \\bar{A} = e^{\\Delta A}$$""")

    make_slide(state_space_models_2, "theory", title="Background: State-Space Models")
    return


@app.cell(hide_code=True)
def _(video_ssm):
    video_ssm
    return


@app.cell
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
    gap = mo.hstack([mo.md(""), mo.md("""
                                    **The gap the StateSpaceDiffuser fills**

                                    High-fidelity image generation using diffusion and persistent
                                    long-term memory (state-space) have not been combined
                                    in a world model.
                                    """), mo.md("")], widths=[0.15, 0.7, 0.15])

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
    **Cosmos tokenizer** into a compact feature vector
    $f_t \\in \\mathbb{R}^d$.
    The discrete action $a_t$ indexes a learnable embedding of
    dimension 16, concatenated with $f_t$. 

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

    The DIAMOND diffusion model generates the next frame conditioned on a **short window** of 4 frames *plus* the SSM's long-context features $\\hat{f}_t$. The SSM's features are fused into the diffusion model, allowing the model to use long-term dependencies when generating the next frame.

    **Fusion Module**: The two streams are merged via a two-layer MLP with SiLU:

    $$\\text{cond} = \\text{concat}\\!\\left[\\text{MLP}_{\\text{mem}}(\\hat{f}_t),\\;\\text{MLP}_{\\text{act}}(e_t + \\varepsilon)\\right]$$
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

                The SSM is trained on long sequences to predict next-frame features from the full history, 
                where produced features carry important long-range context cues. The loss used is then a simple L2 regression on the predicted features:

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

                Direct end-to-end training is unstable! Diffusion's noisy gradients destabilise the SSM and diffusion learns to ignore the SSM entirely, while decoupling the two stages solves this.
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
    video_two_stage
    return


@app.cell
def _():
    video_long_context = render_scene("animations.long_context", "LongContextComparison")
    return (video_long_context,)


@app.cell
def _():
    long_context_0 = mo.vstack([
        mo.hstack([
            mo.md("""
    ## The Forward-Backward Protocol
    The agent takes $n$ actions **forward**, then $n$ mirrored actions **backward**.

    The second half of the sequence should be *identical* in content to the first half."""),
        ], widths=[0.45, 0.55], gap=2)
    ], gap=1)

    make_slide(long_context_0, tag="proposal", title="Evaluating Long-Context Memory")
    return


@app.cell
def _(video_long_context):
    long_context_1 = mo.vstack([
        mo.hstack([
            mo.vstack([mo.md("""
    ## The Forward-Backward Protocol
    The agent takes $n$ actions **forward**, then $n$ mirrored actions **backward**.

    The second half of the sequence should be *identical* in content to the first half,
    so the model must recall what it saw up to $n$ steps ago.""")]),
            video_long_context
        ], widths=[0.45, 0.55], gap=2)
    ], gap=1)

    make_slide(long_context_1, tag="proposal", title="Evaluating Long-Context Memory")
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
    Sequences of 100 steps (50 forward + 50 back).

    **CSGO** is a 3D first-person shooter with 51 action types.
    Performance is evaluated with a user study
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
    mo.video("media/videos/paper_video_short.mp4")
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
    return (psnr,)


@app.cell(hide_code=True)
def _(psnr):
    results_minigrid_0 = mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            psnr,
                        ]
                    ),
                    mo.vstack(
                        [
                            mo.callout(
                                mo.md("""
                **StateSpaceDiffuser achieves +51.9% average PSNR improvement
                over the DIAMOND baseline at context length 50.**
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

    make_slide(results_minigrid_0, tag="results", title="Results: MiniGrid Quantitative Evaluation")
    return


@app.cell
def _(psnr):
    results_minigrid_1 = mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                           psnr,
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

                **SSD w/o state**: StateSpaceDiffuser with SSM features
                zeroed out.


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

    make_slide(results_minigrid_1, tag="results", title="Results: MiniGrid Quantitative Evaluation")
    return


@app.cell(hide_code=True)
def _():
    results_csgo_0 = mo.vstack([
        mo.hstack([
            mo.md("""
    ### CSGO User Study

    12 participants rated whether StateSpaceDiffuser or DIAMOND generated frames closer to the labels.  
    Rating scale: $[-1, 1]$

    **Results:**

    - Frame 15 (second-to-last): **+0.20**
    - Frame 17 (final, hardest): **+0.24**"""),
            mo.md("""
    This generalisation is **not a coincidence**. 
    It follows directly from the SSM's computational structure of the recurrence $h_t = Ah_{t-1} + Bf_t$.
    The diffusion model, by contrast, was designed for a fixed short context.""")
                ], widths=[0.48, 0.48], gap=2)
    ], gap=1)

    make_slide(results_csgo_0, tag="results", title="Results: CSGO & Generalisation to Longer Contexts")
    return


@app.cell
def _():
    mo.callout(
                mo.md("""
    **Generalisation without finetuning**

    - At length 100: SSD achieves 37.99 Avg PSNR vs 26.39 for DIAMOND (**+44%** improvement).  
    - At length 150: SSD achieves 30.75 vs 24.35  (**+26%** improvement), with zero additional training."""),
            kind="success")
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
    **Performance drops below** the DIAMOND baseline (23.68 vs 27.13 Avg PSNR at context 16)

    On CSGO the model without SSM features **hallucinates** content (generates plausible-looking but entirely wrong scenes).
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

    make_slide(ablation_0, tag="results", title="Ablation: Do the SSM Features Actually Matter?")
    return


@app.cell(hide_code=True)
def _():
    ablation_1 = mo.vstack(
        [
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
    cost = mo.vstack([
        mo.hstack([
            mo.vstack([
                components,
                mo.md("""
    An additional important result is that the SSM branch contributes **less than 2%** of total inference compute."""),
            ]),
            mo.vstack([
                mo.md("""
    Mamba SSM processes each new frame in a single recurrent step and doesn't require iterating over the full history:
    $$h_t = Ah_{t-1} + Bf_t \\quad \\text{(one matrix multiply)}$$
    """),
            ], gap=1),
        ], widths=[0.45, 0.55], gap=2),
    ], gap=1)

    make_slide(cost, tag="results", title="Computational Cost")
    return


@app.cell
def _():
    mo.callout(mo.md("""StateSpaceDiffuser is essentially a **very cheap upgrade** over the diffusion-only baseline, i.e. you get long-term memory at negligible additional cost."""), kind="success")
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
    In extended rollouts, detail is frequently lost as state is compressed, so that scaling the SSM is expected to help but was not explored within the compute budget.

    **Lightweight diffusion network.** Replacing DIAMOND with a larger pretrained model (e.g., a video diffusion transformer) could dramatically improve visual sharpness."""),
            ], gap=1),
            mo.vstack([
                mo.callout(
                    mo.md("""
    **What this means for the field**

    The results propose a very interesting discussion for world models: *should memory and generation be decoupled, and if so, how?*"""),
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
    A lightweight SSM branch costing < 2% of inference gives a diffusion world model persistent long-term memory, 
    enabling it to maintain temporal coherence for an order of magnitude more steps than a diffusion-only baseline."""),
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
    Diffusion world models forget everything beyond a short window of $K$ frames, causing temporal drift on long interactions."""),
                mo.md("""
    **Idea.** 
    Use a state-space model to process full history in $O(T)$ time and $O(1)$ (constant) memory, compressing the context into a persistent state that for the diffusion model to use via fusion module."""),
                mo.md("""
    **Result.** 
    +51.9% PSNR on MiniGrid at horizon 50. Positive user preference on complex environment (CSGO) vs Baseline.  
    Zero-shot generalisation to 3× longer contexts. Less than 2% extra inference cost."""),
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
    - **Paper:** arXiv:2505.22246
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
