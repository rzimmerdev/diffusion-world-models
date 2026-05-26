from manim import *
import random
import numpy as np

# ---------------------------------------------------------------------------
# World definition
# ---------------------------------------------------------------------------

GRID_ROWS = 6
GRID_COLS = 6

# State types
EMPTY = "empty"
REWARD = "reward"
ENEMY = "enemy"
OBSTACLE = "obstacle"
UNKNOWN = "unknown"

STATE_COLOR = {
    EMPTY: WHITE,
    REWARD: GREEN,
    ENEMY: ORANGE,
    OBSTACLE: BLACK,
    UNKNOWN: DARK_GRAY,  # not yet discovered
}

# True world layout  (row 0 = top)
TRUE_WORLD = [
    [EMPTY, EMPTY, EMPTY, REWARD, EMPTY, EMPTY],
    [EMPTY, OBSTACLE, EMPTY, EMPTY, OBSTACLE, EMPTY],
    [EMPTY, EMPTY, ENEMY, EMPTY, EMPTY, EMPTY],
    [EMPTY, OBSTACLE, EMPTY, EMPTY, ENEMY, EMPTY],
    [EMPTY, EMPTY, EMPTY, OBSTACLE, EMPTY, EMPTY],
    [REWARD, EMPTY, EMPTY, EMPTY, EMPTY, REWARD],
]

# Agent trajectory (row, col) – 20 steps starting at (0,0)
TRAJECTORY = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (0, 5),
    (1, 5),
    (2, 5),
    (3, 5),
    (4, 5),
    (5, 5),
    (5, 4),
    (5, 3),
    (5, 2),
    (5, 1),
    (5, 0),
    (4, 0),
    (3, 0),
    (2, 0),
    (1, 0),
    (0, 0),
]
# forward: first 20 steps; return: replay backwards (indices 20 down)
FORWARD_TRAJ = TRAJECTORY[:21]  # 21 positions = 20 moves
# For return, we rebuild from index 20 back to index 0
RETURN_TRAJ = TRAJECTORY[20::-1]


# ---------------------------------------------------------------------------
# Helper: observation rule
# ---------------------------------------------------------------------------


def get_observations(prev_pos, new_pos):
    """
    Returns list of (row, col) states discovered when moving from prev_pos to new_pos.
    The new_pos itself + 2 neighbors perpendicular to the movement direction.
    """
    discovered = [new_pos]
    dr = new_pos[0] - prev_pos[0]
    dc = new_pos[1] - prev_pos[1]

    if dr == 0:  # horizontal move → observe up/down neighbours of new_pos
        for offset in [-1, 1]:
            nr, nc = new_pos[0] + offset, new_pos[1]
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                discovered.append((nr, nc))
    elif dc == 0:  # vertical move → observe left/right neighbours
        for offset in [-1, 1]:
            nr, nc = new_pos[0], new_pos[1] + offset
            if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                discovered.append((nr, nc))

    return discovered


# ---------------------------------------------------------------------------
# Helper: build wrong return map (baseline hallucinations)
# ---------------------------------------------------------------------------


def build_baseline_return_map(true_observed):
    """Produces a wrong version of the observed map for the baseline agent."""
    wrong = {}
    all_types = [EMPTY, REWARD, ENEMY, OBSTACLE]
    rng = random.Random(42)
    for pos, true_type in true_observed.items():
        # ~60% chance of getting it wrong
        if rng.random() < 0.6:
            choices = [t for t in all_types if t != true_type]
            wrong[pos] = rng.choice(choices)
        else:
            wrong[pos] = true_type
    return wrong


# ---------------------------------------------------------------------------
# Main Scene
# ---------------------------------------------------------------------------


class CompareDiffusionAnimation(Scene):
    """
    Side-by-side comparison of:
      LEFT   – Baseline Diffusion agent's perceived grid
      CENTER – True grid world (revealed as both agents explore)
      RIGHT  – StateSpaceDiffuser agent's perceived grid

    Forward trip:   both agents walk the same trajectory, discovering states.
    Return trip:    individual agent grids are cleared; baseline makes mistakes,
                    StateSpaceDiffuser reproduces the true world. Each cell gets
                    a ✓ or ✗ overlay compared to the true world.
    """

    # -----------------------------------------------------------------------
    # Scene constants
    # -----------------------------------------------------------------------
    CELL_SIZE = 0.55
    GRID_ORIGIN = UP * 1.5  # vertical offset for grid centres
    LABEL_BUFF = 0.25

    # horizontal centres for the three grids
    LEFT_X = -4.5
    CENTER_X = 0.0
    RIGHT_X = 4.5

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------

    def setup(self):
        self.camera.background_color = "#1a1a2e"

    # -----------------------------------------------------------------------
    # Grid-building utilities
    # -----------------------------------------------------------------------

    def _make_grid(self, cx, label_text):
        """
        Creates a 6×6 grid of Rectangles (all UNKNOWN initially) and
        a label below it.  Returns (VGroup of cells[row][col], label).
        """
        cells = []
        group = VGroup()
        for r in range(GRID_ROWS):
            row_cells = []
            for c in range(GRID_COLS):
                rect = Rectangle(
                    width=self.CELL_SIZE,
                    height=self.CELL_SIZE,
                    fill_color=STATE_COLOR[UNKNOWN],
                    fill_opacity=1.0,
                    stroke_color=GRAY,
                    stroke_width=1.5,
                )
                # Position
                x = cx + (c - (GRID_COLS - 1) / 2) * self.CELL_SIZE
                y = self.GRID_ORIGIN[1] - r * self.CELL_SIZE
                rect.move_to([x, y, 0])
                row_cells.append(rect)
                group.add(rect)
            cells.append(row_cells)

        label = Text(label_text, font_size=22, color=WHITE)
        label.next_to(group, DOWN, buff=self.LABEL_BUFF)

        return cells, group, label

    def _cell(self, cells, row, col):
        return cells[row][col]

    # -----------------------------------------------------------------------
    # Animation helpers
    # -----------------------------------------------------------------------

    def _reveal_cells(self, positions, cells, run_time=0.25):
        """Animate filling a list of (row,col) positions with the true color."""
        anims = []
        for r, c in positions:
            color = STATE_COLOR[TRUE_WORLD[r][c]]
            anims.append(self._cell(cells, r, c).animate.set_fill(color, opacity=1.0))
        if anims:
            self.play(*anims, run_time=run_time)

    def _make_agent_dot(self, cx, row, col, color=YELLOW):
        x = cx + (col - (GRID_COLS - 1) / 2) * self.CELL_SIZE
        y = self.GRID_ORIGIN[1] - row * self.CELL_SIZE
        dot = Dot(radius=self.CELL_SIZE * 0.28, color=color)
        dot.move_to([x, y, 0])
        return dot

    def _cell_center(self, cx, row, col):
        x = cx + (col - (GRID_COLS - 1) / 2) * self.CELL_SIZE
        y = self.GRID_ORIGIN[1] - row * self.CELL_SIZE
        return np.array([x, y, 0])

    def _make_checkmark(self, center, correct: bool):
        sym = "✓" if correct else "✗"
        color = GREEN if correct else RED
        t = Text(sym, font_size=int(self.CELL_SIZE * 38), color=color)
        t.move_to(center + OUT * 0.01)
        return t

    # -----------------------------------------------------------------------
    # construct
    # -----------------------------------------------------------------------

    def construct(self):

        # ── Titles ──────────────────────────────────────────────────────────
        title = Text("Diffusion Agent Comparison", font_size=32, color=YELLOW_A)
        title.to_edge(UP, buff=0.2)
        self.play(Write(title), run_time=0.8)

        # ── Build three grids ───────────────────────────────────────────────
        cells_L, group_L, lbl_L = self._make_grid(self.LEFT_X, "Baseline Diffusion")
        cells_C, group_C, lbl_C = self._make_grid(self.CENTER_X, "True World")
        cells_R, group_R, lbl_R = self._make_grid(self.RIGHT_X, "StateSpaceDiffuser")

        self.play(
            LaggedStart(
                AnimationGroup(*[FadeIn(r) for r in group_L], lag_ratio=0.02),
                AnimationGroup(*[FadeIn(r) for r in group_C], lag_ratio=0.02),
                AnimationGroup(*[FadeIn(r) for r in group_R], lag_ratio=0.02),
                lag_ratio=0.3,
            ),
            FadeIn(lbl_L),
            FadeIn(lbl_C),
            FadeIn(lbl_R),
            run_time=1.2,
        )
        self.wait(0.3)

        # Phase label
        phase_lbl = Text("Forward Trip", font_size=26, color=BLUE_B)
        phase_lbl.next_to(title, DOWN, buff=0.15)
        self.play(FadeIn(phase_lbl))

        # ── Agent dots ──────────────────────────────────────────────────────
        r0, c0 = FORWARD_TRAJ[0]
        dot_L = self._make_agent_dot(self.LEFT_X, r0, c0, YELLOW)
        dot_C = self._make_agent_dot(self.CENTER_X, r0, c0, YELLOW)
        dot_R = self._make_agent_dot(self.RIGHT_X, r0, c0, YELLOW)
        self.add(dot_L, dot_C, dot_R)

        # ── Tracking structures ─────────────────────────────────────────────
        true_observed: dict[tuple, str] = {}  # pos → true type

        def reveal_forward(positions):
            """Reveal cells in all three grids and update true_observed."""
            l_anims, c_anims, r_anims = [], [], []
            for r, c in positions:
                if (r, c) not in true_observed:
                    true_observed[(r, c)] = TRUE_WORLD[r][c]
                color = STATE_COLOR[TRUE_WORLD[r][c]]
                l_anims.append(self._cell(cells_L, r, c).animate.set_fill(color, 1.0))
                c_anims.append(self._cell(cells_C, r, c).animate.set_fill(color, 1.0))
                r_anims.append(self._cell(cells_R, r, c).animate.set_fill(color, 1.0))
            if l_anims:
                self.play(*l_anims, *c_anims, *r_anims, run_time=0.3)

        # ── FORWARD TRIP ────────────────────────────────────────────────────
        for step in range(len(FORWARD_TRAJ) - 1):
            prev = FORWARD_TRAJ[step]
            curr = FORWARD_TRAJ[step + 1]

            obs = get_observations(prev, curr)
            reveal_forward(obs)

            # Move all three agent dots simultaneously
            self.play(
                dot_L.animate.move_to(self._cell_center(self.LEFT_X, curr[0], curr[1])),
                dot_C.animate.move_to(
                    self._cell_center(self.CENTER_X, curr[0], curr[1])
                ),
                dot_R.animate.move_to(
                    self._cell_center(self.RIGHT_X, curr[0], curr[1])
                ),
                run_time=0.22,
            )

        self.wait(0.5)

        # ── Transition to Return Trip ────────────────────────────────────────
        self.play(
            phase_lbl.animate.become(
                Text("Return Trip – Memory Replay", font_size=26, color=RED_B).next_to(
                    title, DOWN, buff=0.15
                )
            ),
            run_time=0.5,
        )

        # Clear individual agent grids (not the true center one)
        fade_anims = []
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                fade_anims.append(
                    self._cell(cells_L, r, c).animate.set_fill(
                        STATE_COLOR[UNKNOWN], 1.0
                    )
                )
                fade_anims.append(
                    self._cell(cells_R, r, c).animate.set_fill(
                        STATE_COLOR[UNKNOWN], 1.0
                    )
                )
        self.play(*fade_anims, run_time=0.8)

        # Reset agents to end of forward trip
        end_pos = FORWARD_TRAJ[-1]
        self.play(
            dot_L.animate.move_to(
                self._cell_center(self.LEFT_X, end_pos[0], end_pos[1])
            ),
            dot_C.animate.move_to(
                self._cell_center(self.CENTER_X, end_pos[0], end_pos[1])
            ),
            dot_R.animate.move_to(
                self._cell_center(self.RIGHT_X, end_pos[0], end_pos[1])
            ),
            run_time=0.4,
        )

        # Build baseline wrong map from everything that was observed
        baseline_wrong = build_baseline_return_map(true_observed)

        # Accumulators for check/cross overlays
        overlay_L: list[Mobject] = []
        overlay_R: list[Mobject] = []

        # ── RETURN TRIP ─────────────────────────────────────────────────────
        for step in range(len(RETURN_TRAJ) - 1):
            prev = RETURN_TRAJ[step]
            curr = RETURN_TRAJ[step + 1]

            if (curr[0], curr[1]) not in true_observed:
                self.play(
                    dot_L.animate.move_to(
                        self._cell_center(self.LEFT_X, curr[0], curr[1])
                    ),
                    dot_C.animate.move_to(
                        self._cell_center(self.CENTER_X, curr[0], curr[1])
                    ),
                    dot_R.animate.move_to(
                        self._cell_center(self.RIGHT_X, curr[0], curr[1])
                    ),
                    run_time=0.22,
                )
                continue

            # Determine what baseline generates vs what StateSpaceDiffuser generates
            baseline_type = baseline_wrong.get((curr[0], curr[1]), EMPTY)
            ssd_type = TRUE_WORLD[curr[0]][curr[1]]  # perfect recall
            true_type = TRUE_WORLD[curr[0]][curr[1]]

            l_color = STATE_COLOR[baseline_type]
            r_color = STATE_COLOR[ssd_type]

            anims = [
                self._cell(cells_L, curr[0], curr[1]).animate.set_fill(l_color, 1.0),
                self._cell(cells_R, curr[0], curr[1]).animate.set_fill(r_color, 1.0),
                dot_L.animate.move_to(self._cell_center(self.LEFT_X, curr[0], curr[1])),
                dot_C.animate.move_to(
                    self._cell_center(self.CENTER_X, curr[0], curr[1])
                ),
                dot_R.animate.move_to(
                    self._cell_center(self.RIGHT_X, curr[0], curr[1])
                ),
            ]
            self.play(*anims, run_time=0.28)

            # Check/cross overlays
            center_L = self._cell_center(self.LEFT_X, curr[0], curr[1])
            center_R = self._cell_center(self.RIGHT_X, curr[0], curr[1])

            mark_l = self._make_checkmark(center_L, baseline_type == true_type)
            mark_r = self._make_checkmark(center_R, ssd_type == true_type)

            overlay_L.append(mark_l)
            overlay_R.append(mark_r)

            self.play(
                FadeIn(mark_l, scale=0.7),
                FadeIn(mark_r, scale=0.7),
                run_time=0.18,
            )

        # ── Final pause & summary ────────────────────────────────────────────
        self.wait(0.5)

        # Count correct
        correct_baseline = sum(
            1
            for pos in true_observed
            if baseline_wrong.get(pos, EMPTY) == TRUE_WORLD[pos[0]][pos[1]]
        )
        correct_ssd = len(true_observed)  # StateSpaceDiffuser is perfect

        summary = Text(
            f"Baseline: {correct_baseline}/{len(true_observed)} correct   |   "
            f"StateSpaceDiffuser: {correct_ssd}/{len(true_observed)} correct",
            font_size=20,
            color=WHITE,
        )
        summary.to_edge(DOWN, buff=0.25)
        self.play(Write(summary), run_time=1.0)
        self.wait(2.0)
