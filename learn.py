import marimo

__generated_with = "0.23.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
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

    def mvideo(scene: Scene, width=900, height=900 * 9 / 16):
        path = scene.renderer.file_writer.movie_file_path
        return mo.video(src=str(path), width=width, height=height)

    return (mvideo,)


@app.cell
def _(mvideo):
    import importlib
    import compare_diffusion

    importlib.reload(compare_diffusion)

    from compare_diffusion import CompareDiffusionAnimation

    scene = CompareDiffusionAnimation()
    _scene = scene.render()

    mvideo(scene)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
