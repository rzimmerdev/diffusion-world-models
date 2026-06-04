// ═══════════════════════════════════════════════════════════════════════════
// Constants — must match Python training code
// ═══════════════════════════════════════════════════════════════════════════
const CELL        = 8;
const COLS        = 19;
const ROWS        = 19;
const W           = COLS * CELL;   // 152
const H           = ROWS * CELL;   // 152
const CONTEXT_LEN = 4;
const T_TOTAL     = 1000;

// Colours
const C_WALL   = [30,  30,  50 ];
const C_FLOOR  = [200, 195, 175];
const C_PLAYER = [80,  160, 240];
const C_GOAL   = [240, 180, 60 ];
const C_FOG    = [15,  15,  25 ];

// ═══════════════════════════════════════════════════════════════════════════
// Maze generation — recursive backtracker
// ═══════════════════════════════════════════════════════════════════════════
function generateMaze(cols, rows) {
  const grid = Array.from({length: rows}, () => Array(cols).fill(true));

  function carve(r, c) {
    grid[r][c] = false;
    const dirs = [[0,2],[0,-2],[2,0],[-2,0]].sort(() => Math.random() - 0.5);
    for (const [dr, dc] of dirs) {
      const nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < rows && nc >= 0 && nc < cols && grid[nr][nc]) {
        grid[r + dr/2|0][c + dc/2|0] = false;
        carve(nr, nc);
      }
    }
  }
  carve(1, 1);
  for (let i = 0; i < rows; i++) grid[i][0] = grid[i][cols-1] = true;
  for (let j = 0; j < cols; j++) grid[0][j] = grid[rows-1][j] = true;
  return grid;
}

// ═══════════════════════════════════════════════════════════════════════════
// Rendering helpers
// ═══════════════════════════════════════════════════════════════════════════
function renderMaze(ctx, maze, playerPos, goalPos) {
  const img = ctx.createImageData(W, H);
  const d   = img.data;

  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const col = maze[r][c] ? C_WALL : C_FLOOR;
      for (let py = r*CELL; py < (r+1)*CELL; py++) {
        for (let px = c*CELL; px < (c+1)*CELL; px++) {
          const i = (py * W + px) * 4;
          d[i]=col[0]; d[i+1]=col[1]; d[i+2]=col[2]; d[i+3]=255;
        }
      }
    }
  }

  // goal (filled square)
  const [gr, gc] = goalPos;
  for (let py = gr*CELL+2; py < (gr+1)*CELL-2; py++) {
    for (let px = gc*CELL+2; px < (gc+1)*CELL-2; px++) {
      const i = (py * W + px) * 4;
      d[i]=C_GOAL[0]; d[i+1]=C_GOAL[1]; d[i+2]=C_GOAL[2]; d[i+3]=255;
    }
  }

  ctx.putImageData(img, 0, 0);

  // player (circle — easier with canvas API)
  const [pr, pc] = playerPos;
  const cx = pc*CELL + CELL/2, cy = pr*CELL + CELL/2;
  ctx.beginPath();
  ctx.arc(cx, cy, CELL/2 - 1, 0, Math.PI*2);
  ctx.fillStyle = `rgb(${C_PLAYER.join(',')})`;
  ctx.fill();
}

// Canvas → Float32Array [3, H, W] in [0,1]
function canvasToTensor(canvas) {
  const ctx  = canvas.getContext('2d');
  const img  = ctx.getImageData(0, 0, W, H);
  const data = img.data;
  const t    = new Float32Array(3 * H * W);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const src = (y * W + x) * 4;
      const dst = y * W + x;
      t[0*H*W + dst] = data[src]   / 255;
      t[1*H*W + dst] = data[src+1] / 255;
      t[2*H*W + dst] = data[src+2] / 255;
    }
  }
  return t;
}

// Float32Array [3, H, W] in [0,1] → draw onto canvas
function tensorToCanvas(t, canvas) {
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(W, H);
  const d   = img.data;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const src = y * W + x;
      const dst = (y * W + x) * 4;
      d[dst]   = Math.min(255, Math.max(0, t[0*H*W + src] * 255));
      d[dst+1] = Math.min(255, Math.max(0, t[1*H*W + src] * 255));
      d[dst+2] = Math.min(255, Math.max(0, t[2*H*W + src] * 255));
      d[dst+3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// DDPM noise schedule (cosine) — matches Python
// ═══════════════════════════════════════════════════════════════════════════
function buildAlphasCumprod(T = T_TOTAL) {
  // Match Python: compute alphas_bar at T+1 points (0..T), then
  // betas[i] = 1 - alphas_bar[i+1]/alphas_bar[i], then cumprod.
  const s     = 0.008;
  const ac    = new Float32Array(T + 1);       // T+1 points, like Python
  const base  = Math.cos((s / (1 + s)) * Math.PI / 2) ** 2;
  for (let i = 0; i <= T; i++) {
    const frac  = (i / T + s) / (1 + s);
    ac[i] = (Math.cos(frac * Math.PI / 2) ** 2) / base;
  }
  const betas  = new Float32Array(T);
  for (let i = 0; i < T; i++) {
    betas[i] = Math.min(0.999, 1 - ac[i+1] / ac[i]);
  }
  const alphasCumprod = new Float32Array(T);
  let cp = 1.0;
  for (let i = 0; i < T; i++) {
    cp *= (1 - betas[i]);
    alphasCumprod[i] = cp;
  }
  return { alphasCumprod, betas };
}

const { alphasCumprod, betas } = buildAlphasCumprod();

function randn(n) {
  const out = new Float32Array(n);
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.random(), u2 = Math.random();
    const mag = Math.sqrt(-2 * Math.log(u1 + 1e-10));
    out[i]   = mag * Math.cos(2 * Math.PI * u2);
    if (i+1 < n) out[i+1] = mag * Math.sin(2 * Math.PI * u2);
  }
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// ONNX inference wrapper
// ═══════════════════════════════════════════════════════════════════════════
let ortSession = null;

function detectGPU() {
  const el = document.getElementById('st-gpu');
  if (!ortSession) { el.textContent = '—'; return; }

  // Check what backend the session actually uses
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
    if (gl) {
      // RENDERER is standard WebGL — no extension needed (avoids
      // WEBGL_debug_renderer_info deprecation in Firefox).
      const renderer = gl.getParameter(gl.RENDERER);
      el.textContent = renderer;
      el.className = 'ok';
      return;
    }
  } catch(e) {}
  el.textContent = 'cpu (wasm)';
  el.className = '';
}


async function loadModel(arrayBuffer) {
  if (typeof ort === 'undefined') {
    throw new Error('ONNX Runtime Web failed to load.');
  }
  ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.1/dist/';

  let backend = 'wasm';
  try {
    ortSession = await ort.InferenceSession.create(arrayBuffer, {
      executionProviders: ['wasm'],
    });
  } catch (e) {
    console.warn('WASM backend failed, falling back to WebGL:', e.message);
    backend = 'webgl';
    ortSession = await ort.InferenceSession.create(arrayBuffer, {
      executionProviders: ['webgl', 'wasm'],
    });
  }

  document.getElementById('st-backend').textContent = backend;
  detectGPU();
  document.getElementById('pred-panel').style.opacity = '1';
  document.getElementById('st-model').textContent = 'loaded';
  document.getElementById('st-model').className   = 'ok';
}

async function modelForward(contextFlat, noisyFlat, actionVec, tFrac) {
  if (!ortSession) return null;

  const N = H * W;
  // expand action (4,) → (4, H, W)
  const actionMap = new Float32Array(4 * N);
  for (let c = 0; c < 4; c++) {
    actionMap.fill(actionVec[c], c * N, (c + 1) * N);
  }
  // expand t_frac scalar → (1, H, W)
  const tMap = new Float32Array(N);
  tMap.fill(tFrac);

  const feeds = {
    context:     new ort.Tensor('float32', contextFlat, [1, CONTEXT_LEN * 3, H, W]),
    noisy:       new ort.Tensor('float32', noisyFlat,   [1, 3, H, W]),
    action_map:  new ort.Tensor('float32', actionMap,   [1, 4, H, W]),
    t_map:       new ort.Tensor('float32', tMap,        [1, 1, H, W]),
  };
  const result = await ortSession.run(feeds);
  return result['pred_noise'].data;
}

// Fast DDPM sampling — `steps` denoising steps
async function ddpmInfer(contextFlat, actionVec, steps) {
  const N   = 3 * H * W;
  let x     = randn(N);

  const tSeq = [];
  for (let i = 0; i < steps; i++) {
    tSeq.push(Math.trunc((T_TOTAL - 1) * (1 - i / (steps - 1 || 1))));
  }

  for (let i = 0; i < tSeq.length; i++) {
    const t      = tSeq[i];
    const tNext  = i + 1 < tSeq.length ? tSeq[i+1] : -1;
    const tFrac  = t / T_TOTAL;

    const predNoise = await modelForward(contextFlat, x, actionVec, tFrac);
    if (!predNoise) return null;

    const at  = alphasCumprod[t];
    const at1 = tNext >= 0 ? alphasCumprod[tNext] : 1.0;
    const beta_t = betas[t];

    const sqrtAt  = Math.sqrt(at);
    const sqrtOm  = Math.sqrt(1 - at);
    const x0pred  = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      x0pred[k] = Math.max(-1, Math.min(1, (x[k] - sqrtOm * predNoise[k]) / sqrtAt));
    }

    if (tNext < 0) { x = x0pred; break; }

    const coef1 = Math.sqrt(at1) * (1 - at / at1) / (1 - at);
    const coef2 = Math.sqrt(at / at1) * (1 - at1) / (1 - at);
    const var_t  = Math.sqrt(beta_t * (1 - at1) / (1 - at));
    const noise  = randn(N);
    const xPrev  = new Float32Array(N);
    for (let k = 0; k < N; k++) {
      xPrev[k] = coef1 * x0pred[k] + coef2 * x[k] + var_t * noise[k];
    }
    x = xPrev;
  }

  const out = new Float32Array(N);
  for (let k = 0; k < N; k++) out[k] = x[k] * 0.5 + 0.5;
  return out;
}

// ═══════════════════════════════════════════════════════════════════════════
// Context window
// ═══════════════════════════════════════════════════════════════════════════
const contextFrames = [];

function pushFrame(t) {
  contextFrames.push(t);
  if (contextFrames.length > CONTEXT_LEN) contextFrames.shift();
}

function getContextFlat() {
  const out = new Float32Array(CONTEXT_LEN * 3 * H * W);
  const empty = new Float32Array(3 * H * W);
  for (let i = 0; i < CONTEXT_LEN; i++) {
    const srcIdx = Math.max(0, contextFrames.length - CONTEXT_LEN + i);
    const s = contextFrames[srcIdx] ?? empty;
    out.set(s, i * 3 * H * W);
  }
  return out;
}

function encodeAction(idx) {
  const v = new Float32Array(4);
  v[idx] = 1.0;
  return v;
}

// ═══════════════════════════════════════════════════════════════════════════
// Game state
// ═══════════════════════════════════════════════════════════════════════════
const realCanvas = document.getElementById('real');
const predCanvas = document.getElementById('pred');
const realCtx    = realCanvas.getContext('2d');

let maze     = generateMaze(COLS, ROWS);
let player   = [1, 1];
const goal   = [ROWS-2, COLS-2];
let wins     = 0;
let stepCount = 0;

const DIR_MAP = { 0: [-1,0], 1: [1,0], 2: [0,-1], 3: [0,1] };
let inferRunning = false;

function drawReal() {
  renderMaze(realCtx, maze, player, goal);
}

function resetPlayer() {
  player = [1, 1];
  contextFrames.length = 0;
  drawReal();
  // Mirror initial state to prediction panel so it's not blank before first move
  const predCtx = predCanvas.getContext('2d');
  predCtx.drawImage(realCanvas, 0, 0);
  for (let i = 0; i < CONTEXT_LEN; i++) pushFrame(canvasToTensor(realCanvas));
}

async function move(dirIdx) {
  if (inferRunning) return;

  const [dr, dc] = DIR_MAP[dirIdx];
  const nr = player[0] + dr, nc = player[1] + dc;
  if (maze[nr]?.[nc] === false) {
    if (ortSession) {
      inferRunning = true;
      try {
        const t0     = performance.now();
        const ctx    = getContextFlat();
        const action = encodeAction(dirIdx);
        const steps  = parseInt(document.getElementById('steps-slider').value);
        const pred   = await ddpmInfer(ctx, action, steps);
        const dt     = (performance.now() - t0).toFixed(0);
        document.getElementById('st-infer').textContent = `${dt}ms`;
        if (pred) tensorToCanvas(pred, predCanvas);
      } catch (err) {
        console.error('Inference failed:', err);
        document.getElementById('st-infer').textContent = 'err';
      } finally {
        inferRunning = false;
      }
    }

    player = [nr, nc];
    stepCount++;
    drawReal();
    pushFrame(canvasToTensor(realCanvas));

    document.getElementById('st-steps').textContent = stepCount;

    if (player[0] === goal[0] && player[1] === goal[1]) {
      wins++;
      document.getElementById('st-wins').textContent = wins;
      maze   = generateMaze(COLS, ROWS);
      resetPlayer();
    }
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// Input handling
// ═══════════════════════════════════════════════════════════════════════════
const KEY_MAP = {
  ArrowUp: 0, ArrowDown: 1, ArrowLeft: 2, ArrowRight: 3,
  w: 0, s: 1, a: 2, d: 3,
};

document.addEventListener('keydown', e => {
  if (e.key in KEY_MAP) { e.preventDefault(); move(KEY_MAP[e.key]); }
});

document.querySelectorAll('#controls button').forEach(btn => {
  btn.addEventListener('click', () => move(parseInt(btn.dataset.dir)));
});

// ── ONNX file loader ────────────────────────────────────────────────────────
document.getElementById('onnx-file').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  document.getElementById('st-model').textContent = 'loading…';
  document.getElementById('st-model').className   = 'loading';
  try {
    const buf = await file.arrayBuffer();
    await loadModel(buf);
  } catch (err) {
    document.getElementById('st-model').textContent = 'error';
    document.getElementById('st-model').className   = 'err';
    console.error(err);
  }
});

// ── steps slider ────────────────────────────────────────────────────────────
document.getElementById('steps-slider').addEventListener('input', e => {
  document.getElementById('steps-val').textContent = e.target.value;
});

// ═══════════════════════════════════════════════════════════════════════════
// Boot
// ═══════════════════════════════════════════════════════════════════════════
resetPlayer();
