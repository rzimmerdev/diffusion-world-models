"""
Diffusion World Model — Training & Export
==========================================
Tiny U-Net diffusion model for maze-frame prediction.
Provides: model architecture, DDPM utilities, training loop, and ONNX export.

Usage:
    python diffusion.py train --out model.pt --steps 800
    python diffusion.py export --checkpoint model.pt --out world_model.onnx
"""

import argparse
import math
import random
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F

# ─────────────────────────── constants ────────────────────────────────────────
CELL = 16          # pixels per maze cell
COLS = 19          # maze grid columns (odd for walls)
ROWS = 19          # maze grid rows
W = COLS * CELL    # frame width
H = ROWS * CELL    # frame height

CONTEXT_LEN = 4    # how many past frames the model sees
DIFF_STEPS = 6     # DDPM denoising steps at inference (fast)

# ─────────────────────────── Maze generation ──────────────────────────────────


def generate_maze(cols, rows):
    """Recursive backtracker. Returns 2-D bool grid: True = wall."""
    grid = [[True] * cols for _ in range(rows)]

    def carve(r, c):
        grid[r][c] = False
        dirs = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc]:
                grid[r + dr // 2][c + dc // 2] = False
                carve(nr, nc)

    carve(1, 1)
    for i in range(rows):
        grid[i][0] = grid[i][-1] = True
    for j in range(cols):
        grid[0][j] = grid[-1][j] = True
    return grid


def maze_to_tensor(maze, player_pos, goal_pos):
    """Render maze + player + goal as a float32 tensor [3, H, W] in [0,1].
    No pygame dependency — builds the pixel grid directly."""
    t = torch.zeros(3, ROWS * CELL, COLS * CELL)
    for r in range(ROWS):
        for c in range(COLS):
            y0, x0 = r * CELL, c * CELL
            if maze[r][c]:
                t[:, y0 : y0 + CELL, x0 : x0 + CELL] = torch.tensor([30, 30, 50]).view(3, 1, 1) / 255.0
            else:
                t[:, y0 : y0 + CELL, x0 : x0 + CELL] = torch.tensor([200, 195, 175]).view(3, 1, 1) / 255.0
    # goal
    gr, gc = goal_pos
    y0, x0 = gr * CELL + 2, gc * CELL + 2
    t[:, y0 : y0 + CELL - 4, x0 : x0 + CELL - 4] = torch.tensor([240, 180, 60]).view(3, 1, 1) / 255.0
    # player
    pr, pc = player_pos
    cy, cx = pr * CELL + CELL // 2, pc * CELL + CELL // 2
    rr = CELL // 2 - 1
    for dy in range(-rr, rr + 1):
        for dx in range(-rr, rr + 1):
            if dy * dy + dx * dx <= rr * rr:
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < H and 0 <= nx < W:
                    t[:, ny, nx] = torch.tensor([80, 160, 240]) / 255.0
    return t




class GroupNorm4D(nn.Module):
    """GroupNorm that decomposes to WebGL-compatible 4-D ONNX ops.

    Standard GroupNorm exported to ONNX (opset < 21) uses a 3-D Reshape
    [N, C, H, W] → [N, G, C//G*H*W] before InstanceNormalization, which
    the onnxruntime-web WebGL backend rejects (only supports 4-D).

    This implementation reshapes to [N, G, C//G, H*W] (always 4-D) and
    normalizes with explicit ReduceMean/Var — no InstanceNorm in the graph."""

    def __init__(self, num_groups, num_channels, eps=1e-5):
        super().__init__()
        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))

    def forward(self, x):
        N, C, H, W = x.shape
        G = self.num_groups
        # [N, C, H, W] → [N, G, C//G, H*W]   stays 4-D (WebGL-safe)
        x = x.reshape(N, G, C // G, -1)
        mean = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        # [N, G, C//G, H*W] → [N, C, H, W]
        x = x.reshape(N, C, H, W)
        return x * self.weight.view(1, C, 1, 1) + self.bias.view(1, C, 1, 1)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            GroupNorm4D(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            GroupNorm4D(min(8, ch), ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class DiffusionWorldModel(nn.Module):
    """
    Input:  context (CONTEXT_LEN × 3 channels) + noisy frame (3 ch) + action (4 ch one-hot) + timestep emb
    Output: denoised frame (3 ch)

    Tiny U-Net: enc → bottleneck → dec, skip connections.
    """

    def __init__(self, context_len=CONTEXT_LEN, base_ch=32):
        super().__init__()
        in_ch = context_len * 3 + 3 + 4 + 1  # context + noisy + action + t_emb broadcast

        # encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, base_ch, 3, padding=1), ResBlock(base_ch)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1),
            ResBlock(base_ch * 2),
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1),
            ResBlock(base_ch * 4),
        )

        # bottleneck
        self.bot = nn.Sequential(ResBlock(base_ch * 4), ResBlock(base_ch * 4))

        # decoder
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 4, stride=2, padding=1),
            ResBlock(base_ch * 2),
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch, 4, stride=2, padding=1),
            ResBlock(base_ch),
        )
        self.out = nn.Conv2d(base_ch * 2, 3, 1)

    def forward(self, context, noisy, action, t_frac):
        """
        context : (B, CL*3, H, W)
        noisy   : (B, 3, H, W)
        action  : (B, 4)   one-hot
        t_frac  : (B,)     in [0,1]
        """
        B, _, Hh, Ww = noisy.shape
        act_ch = action.view(B, 4, 1, 1).expand(B, 4, Hh, Ww)
        t_ch = t_frac.view(B, 1, 1, 1).expand(B, 1, Hh, Ww)
        x = torch.cat([context, noisy, act_ch, t_ch], dim=1)

        e1 = self.enc1(x)   # (B, ch,   H,   W)
        e2 = self.enc2(e1)  # (B, ch*2, H/2, W/2)
        e3 = self.enc3(e2)  # (B, ch*4, H/4, W/4)
        b = self.bot(e3)
        d3 = self.dec3(b)                            # (B, ch*2, H/2, W/2)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))    # (B, ch,   H,   W)
        out = self.out(torch.cat([d2, e1], dim=1))    # (B, 3,   H,   W)
        return out

    def forward_concat(self, x):
        """x: (B, CL*3+3+4+1, H, W) — pre-concatenated input (no Expand ops)."""
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        b = self.bot(e3)
        d3 = self.dec3(b)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))
        return self.out(torch.cat([d2, e1], dim=1))


# ─────────────────────────── ONNX export wrapper ──────────────────────────────


class ExportWrapper(nn.Module):
    """Takes pre-expanded action_map (B,4,H,W) and t_map (B,1,H,W) —
    expansion happens in JS, so the ONNX graph has no Expand/ConstantOfShape."""

    def __init__(self, model: DiffusionWorldModel):
        super().__init__()
        self.model = model

    def forward(self, context, noisy, action, t_frac):
        x = torch.cat([context, noisy, action, t_frac], dim=1)
        return self.model.forward_concat(x)


# ─────────────────────────── DDPM noise schedule ──────────────────────────────


def cosine_betas(T=1000):
    s = 0.008
    steps = torch.arange(T + 1, dtype=torch.float32)
    alphas_bar = torch.cos((steps / T + s) / (1 + s) * math.pi / 2) ** 2
    alphas_bar = alphas_bar / alphas_bar[0]
    betas = 1 - alphas_bar[1:] / alphas_bar[:-1]
    return betas.clamp(0, 0.999)


T_TOTAL = 1000
betas = cosine_betas(T_TOTAL)
alphas = 1.0 - betas
alphas_cumprod = torch.cumprod(alphas, 0)


def q_sample(x0, t_idx):
    """Add noise to x0 at timestep t_idx (int tensor B)."""
    a = alphas_cumprod[t_idx].view(-1, 1, 1, 1).to(x0.device)
    noise = torch.randn_like(x0)
    return a.sqrt() * x0 + (1 - a).sqrt() * noise, noise


def p_sample_step(model, x_t, context, action, t_idx, t_next_idx):
    """One DDPM denoising step."""
    B = x_t.shape[0]
    device = x_t.device
    t_frac = (t_idx / T_TOTAL).float().to(device).expand(B)

    with torch.no_grad():
        pred_noise = model(context, x_t, action, t_frac)

    at = alphas_cumprod[t_idx].to(device)
    at1 = (
        alphas_cumprod[t_next_idx].to(device) if t_next_idx >= 0 else torch.tensor(1.0)
    )

    # predict x0
    x0_pred = (x_t - (1 - at).sqrt() * pred_noise) / at.sqrt()
    x0_pred = x0_pred.clamp(-1, 1)

    if t_next_idx < 0:
        return x0_pred

    # compute x_{t-1}
    coef1 = at1.sqrt() * (1 - at / at1) / (1 - at)
    coef2 = (at / at1).sqrt() * (1 - at1) / (1 - at)
    mean = coef1 * x0_pred + coef2 * x_t
    var = betas[t_idx].to(device) * (1 - at1) / (1 - at)
    return mean + var.sqrt() * torch.randn_like(x_t)


def ddpm_infer(model, context, action, steps=DIFF_STEPS, device="cpu"):
    """Fast DDPM sampling with `steps` denoising steps."""
    B = context.shape[0]
    x = torch.randn(B, 3, H, W, device=device)
    t_seq = torch.linspace(T_TOTAL - 1, 0, steps).long()
    for i, t in enumerate(t_seq):
        t_next = t_seq[i + 1] if i + 1 < len(t_seq) else -1
        t_b = t.expand(B)
        x = p_sample_step(
            model, x, context, action, t_b, t_next if t_next == -1 else t_next.expand(B)
        )
    return x


# ─────────────────────────── Training loop helper ─────────────────────────────


def train_on_replay(model, optimizer, replay, device, steps=50):
    """Train model on stored (context, frame, action) tuples."""
    if len(replay) < 4:
        return 0.0
    model.train()
    total_loss = 0.0
    for _ in range(steps):
        batch = random.choices(replay, k=4)
        ctx_b = torch.stack([b[0] for b in batch]).to(device)
        frame_b = torch.stack([b[1] for b in batch]).to(device)
        act_b = torch.stack([b[2] for b in batch]).to(device)

        t_idx = torch.randint(0, T_TOTAL, (4,))
        noisy, noise = q_sample(frame_b * 2 - 1, t_idx)  # scale to [-1,1]
        t_frac = (t_idx.float() / T_TOTAL).to(device)

        pred = model(ctx_b, noisy.to(device), act_b.to(device), t_frac)
        loss = F.mse_loss(pred, noise.to(device))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    model.eval()
    return total_loss / steps


# ─────────────────────────── ONNX export ──────────────────────────────────────



def train(checkpoint_path: str | None, out_path: str, device_str: str, steps: int, lr: float):
    """Train DiffusionWorldModel via random maze exploration (headless)."""
    device = torch.device(device_str)
    print(f"Using device: {device}")

    model = DiffusionWorldModel(CONTEXT_LEN).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"Loaded weights from {checkpoint_path}")

    DIR_MAP = {0: (-1, 0), 1: (1, 0), 2: (0, -1), 3: (0, 1)}
    replay = collections.deque(maxlen=2000)
    start_pos = (1, 1)
    goal_pos = (ROWS - 2, COLS - 2)

    print(f"Training for {steps} exploration steps...")
    ctx_window = collections.deque(maxlen=CONTEXT_LEN)

    for epoch in range(max(1, steps // 800)):
        maze = generate_maze(COLS, ROWS)
        player = list(start_pos)
        ctx_window.clear()
        for _ in range(CONTEXT_LEN):
            ctx_window.append(maze_to_tensor(maze, player, goal_pos))

        for s in range(800):
            a_idx = random.randint(0, 3)
            dr, dc = DIR_MAP[a_idx]
            nr, nc = player[0] + dr, player[1] + dc
            if not maze[nr][nc]:
                player = [nr, nc]

            frame = maze_to_tensor(maze, player, goal_pos)
            ctx = torch.cat(list(ctx_window), dim=0)  # (CL*3, H, W)
            act = torch.zeros(4); act[a_idx] = 1.0
            replay.append((ctx, frame, act))
            ctx_window.append(frame)

            if len(replay) % 200 == 0:
                loss = train_on_replay(model, optimizer, replay, device, steps=80)
                print(f"  step {len(replay)}  loss={loss:.4f}")

            if player == list(goal_pos):
                player = list(start_pos)
                ctx_window.clear()
                for _ in range(CONTEXT_LEN):
                    ctx_window.append(maze_to_tensor(maze, player, goal_pos))

    if out_path:
        torch.save({"model": model.state_dict(), "steps": steps}, out_path)
        print(f"Saved checkpoint to {out_path}")

    return model


# ─────────────────────────── ONNX export ──────────────────────────────────────
def export(checkpoint_path: str | None, out_path: str):
    device = torch.device("cpu")  # export on CPU for portability
    model = DiffusionWorldModel(CONTEXT_LEN).to(device)

    if checkpoint_path:
        state = torch.load(checkpoint_path, map_location=device)
        # handle plain state_dict or {"model": ...} checkpoints
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
        print(f"Loaded weights from {checkpoint_path}")
    else:
        print("No checkpoint supplied — exporting with random weights (for wiring test).")

    model.eval()
    wrapper = ExportWrapper(model)

    # dummy inputs — action/t_frac are pre-expanded spatial maps by JS
    context = torch.zeros(1, CONTEXT_LEN * 3, H, W)
    noisy   = torch.randn(1, 3, H, W)
    action  = torch.zeros(1, 4, H, W)
    t_frac  = torch.zeros(1, 1, H, W)

    print(f"Exporting to {out_path} …")
    torch.onnx.export(
        wrapper,
        (context, noisy, action, t_frac),
        out_path,
        input_names=["context", "noisy", "action_map", "t_map"],
        output_names=["pred_noise"],
        opset_version=14,
        do_constant_folding=True,
        dynamo=False,
    )
    print("Done.")

    # ── quick validation ──────────────────────────────────────────────────────
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
        feeds = {
            "context":     context.numpy(),
            "noisy":       noisy.numpy(),
            "action_map":  action.numpy(),
            "t_map":       t_frac.numpy(),
        }
        out = sess.run(["pred_noise"], feeds)[0]
        print(f"Validation OK — output shape: {out.shape}")
    except ImportError:
        print("onnxruntime not installed; skipping validation.")


# ─────────────────────────── CLI ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diffusion World Model")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--checkpoint", type=str, default=None,
                         help="Resume from checkpoint (.pt)")
    p_train.add_argument("--out", type=str, default="model.pt",
                         help="Output checkpoint path")
    p_train.add_argument("--device", type=str, default="cuda",
                         help="Device (cuda / cpu)")
    p_train.add_argument("--steps", type=int, default=800,
                         help="Exploration steps")
    p_train.add_argument("--lr", type=float, default=3e-4,
                         help="Learning rate")

    # export
    p_export = sub.add_parser("export", help="Export to ONNX")
    p_export.add_argument("--checkpoint", type=str, default=None,
                          help="Path to model checkpoint (.pt)")
    p_export.add_argument("--out", type=str, default="world_model.onnx",
                          help="Output .onnx path")

    args = parser.parse_args()
    if args.cmd == "train":
        train(args.checkpoint, args.out, args.device, args.steps, args.lr)
    elif args.cmd == "export":
        export(args.checkpoint, args.out)
