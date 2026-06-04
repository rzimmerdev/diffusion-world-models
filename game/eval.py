"""Visual DDPM inference preview — run against any checkpoint.

Usage:
    python src/eval.py                          # uses model.pt by default
    python src/eval.py --checkpoint model.pt    # explicit checkpoint
    python src/eval.py --steps 10               # more denoising steps
"""

import argparse
import torch
import pygame

from diffusion import (
    DiffusionWorldModel, generate_maze, maze_to_tensor,
    ddpm_infer, CONTEXT_LEN, COLS, ROWS, W, H, DIFF_STEPS,
)


def main():
    parser = argparse.ArgumentParser(description="DDPM visual preview")
    parser.add_argument("--checkpoint", default="model.pt")
    parser.add_argument("--scale", type=float, default=2.0,
                        help="initial window scale factor")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--steps", type=int, default=DIFF_STEPS,
                        help="denoising steps per frame")
    args = parser.parse_args()

    device = torch.device(args.device)

    # ── load model ────────────────────────────────────────────────────────
    model = DiffusionWorldModel(CONTEXT_LEN).to(device)
    state = torch.load(args.checkpoint, map_location=device)
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    model.load_state_dict(state)
    model.eval()
    print(f"Loaded {args.checkpoint}  |  denoising steps: {args.steps}")

    # ── pygame window ─────────────────────────────────────────────────────
    pygame.init()
    base_w = W * 2 + 16
    base_h = H + 36
    screen = pygame.display.set_mode(
        (int(base_w * args.scale), int(base_h * args.scale)),
        pygame.RESIZABLE,
    )
    pygame.display.set_caption("DDPM preview — arrow keys to move, ESC to quit")

    # ── setup maze + context ──────────────────────────────────────────────
    maze = generate_maze(COLS, ROWS)
    pos = [1, 1]
    goal = (ROWS - 2, COLS - 2)
    ctx_win = [maze_to_tensor(maze, pos, goal) for _ in range(CONTEXT_LEN)]

    DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    KEY_TO_DIR = {
        pygame.K_UP: 0, pygame.K_DOWN: 1,
        pygame.K_LEFT: 2, pygame.K_RIGHT: 3,
    }
    clock = pygame.time.Clock()
    running = True
    step = 0

    def tensor_to_surface(t):
        """[3,H,W] float in [0,1] → pygame Surface."""
        arr = (t.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
        return pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    # ── initial draw ──────────────────────────────────────────────────────
    real_01 = maze_to_tensor(maze, pos, goal)
    win_w, win_h = screen.get_size()
    gap = max(4, win_w // 60)
    pad = max(2, win_w // 160)
    label_h = max(16, win_h // 30)
    avail_w = win_w - gap - pad * 2
    avail_h = win_h - label_h - pad * 2
    panel_w = avail_w // 2
    panel_h = avail_h
    if panel_w * H > panel_h * W:
        panel_w = panel_h * W // H
    else:
        panel_h = panel_w * H // W
    real_scaled = pygame.transform.scale(tensor_to_surface(real_01), (panel_w, panel_h))
    screen.fill((10, 10, 20))
    screen.blit(real_scaled, (pad, pad))
    font = pygame.font.SysFont("monospace", max(10, label_h // 2))
    label = font.render(
        "step 0    real                          (press arrow key)",
        True, (180, 180, 200),
    )
    screen.blit(label, (pad, pad + panel_h + 4))
    pygame.display.flip()
    while running:
        # wait for a valid arrow-key move
        a_idx = None
        while a_idx is None:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        break
                    if event.key in KEY_TO_DIR:
                        candidate = KEY_TO_DIR[event.key]
                        dr, dc = DIRS[candidate]
                        nr, nc = pos[0] + dr, pos[1] + dc
                        if not maze[nr][nc]:
                            a_idx = candidate
            if not running:
                break
            clock.tick(60)
        if not running:
            break
        pos = [nr, nc]

        # ── DDPM inference ────────────────────────────────────────────────
        ctx = torch.stack(list(ctx_win)).view(1, CONTEXT_LEN * 3, H, W).to(device)
        act = torch.zeros(1, 4, device=device)
        act[0, a_idx] = 1.0
        with torch.no_grad():
            pred = ddpm_infer(model, ctx, act, steps=args.steps, device=device)
        pred_01 = (pred.squeeze(0).cpu() * 0.5 + 0.5).clamp(0, 1)
        real_01 = maze_to_tensor(maze, pos, goal)

        # ── draw ──────────────────────────────────────────────────────────
        win_w, win_h = screen.get_size()
        gap = max(4, win_w // 60)
        pad = max(2, win_w // 160)
        label_h = max(16, win_h // 30)
        avail_w = win_w - gap - pad * 2
        avail_h = win_h - label_h - pad * 2
        panel_w = avail_w // 2
        panel_h = avail_h
        if panel_w * H > panel_h * W:
            panel_w = panel_h * W // H
        else:
            panel_h = panel_w * H // W

        real_scaled = pygame.transform.scale(tensor_to_surface(real_01), (panel_w, panel_h))
        pred_scaled = pygame.transform.scale(tensor_to_surface(pred_01), (panel_w, panel_h))

        screen.fill((10, 10, 20))
        screen.blit(real_scaled, (pad, pad))
        screen.blit(pred_scaled, (pad + panel_w + gap, pad))
        font = pygame.font.SysFont("monospace", max(10, label_h // 2))
        label = font.render(
            f"step {step+1:<3}  real                          predicted",
            True, (180, 180, 200),
        )
        screen.blit(label, (pad, pad + panel_h + 4))
        pygame.display.flip()

        # advance context
        ctx_win.append(real_01)
        if len(ctx_win) > CONTEXT_LEN:
            ctx_win.pop(0)
        step += 1
        clock.tick(5)

    pygame.quit()


if __name__ == "__main__":
    main()
