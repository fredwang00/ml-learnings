# Attention Visualizer Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild the attention visualizer as a dark-mode, three-panel learning tool that shows both attention patterns and the Q/K/V math that produces them.

**Architecture:** Single self-contained HTML file. Data layer computes attention weights from synthetic Q/K/V vectors via real softmax. Three coordinated panels (text+lines, heatmap, computation) update in response to hover and click interactions. Two switchable attention heads demonstrate positional vs semantic patterns.

**Tech Stack:** Vanilla HTML/CSS/JS, SVG for attention lines, Canvas for heatmap.

**Spec:** `docs/design.md`

---

### Task 1: Data Layer — PRNG, Embeddings, Matrix Math

**Files:**
- Create: `attention-visualizer.html`

Build the math foundation as pure functions inside a `<script>` block. No DOM work yet — just functions and a console smoke test.

- [ ] **Step 1: Write the PRNG and embedding generator**

```javascript
// mulberry32 seeded PRNG
function mulberry32(seed) {
    return function() {
        seed |= 0; seed = seed + 0x6D2B79F5 | 0;
        let t = Math.imul(seed ^ seed >>> 15, 1 | seed);
        t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
        return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
}

// Simple string hash
function hashString(str) {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
        hash = ((hash << 5) - hash) + str.charCodeAt(i);
        hash |= 0;
    }
    return hash;
}

// Generate unit-length embedding for a token
function generateEmbedding(token, dim = 8) {
    const rng = mulberry32(hashString(token));
    const vec = Array.from({length: dim}, () => rng() * 2 - 1);
    const norm = Math.sqrt(vec.reduce((s, v) => s + v * v, 0));
    return vec.map(v => v / norm);
}
```

- [ ] **Step 2: Write matrix/vector math utilities**

```javascript
// Matrix-vector multiply: (dim x dim) @ (dim,) -> (dim,)
function matVecMul(matrix, vec) {
    return matrix.map(row => row.reduce((s, w, i) => s + w * vec[i], 0));
}

// Dot product
function dot(a, b) {
    return a.reduce((s, v, i) => s + v * b[i], 0);
}

// Softmax with causal mask (positions > queryIdx get -Infinity)
function softmax(scores, queryIdx) {
    const masked = scores.map((s, i) => i <= queryIdx ? s : -Infinity);
    const max = Math.max(...masked.filter(s => s !== -Infinity));
    const exps = masked.map(s => s === -Infinity ? 0 : Math.exp(s - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(e => e / sum);
}
```

- [ ] **Step 3: Write the attention computation module**

```javascript
// Compute all attention weights for a given head config
function computeAttention(tokens, embeddings, head) {
    const dim = embeddings[0].length;
    const scale = Math.sqrt(dim);
    const n = tokens.length;

    // Compute Q and K for all tokens
    const Q = embeddings.map(e => matVecMul(head.Wq, e));
    const K = embeddings.map(e => matVecMul(head.Wk, e));

    // Compute attention weights matrix
    const weights = [];
    for (let i = 0; i < n; i++) {
        const scores = K.map(k => dot(Q[i], k) / scale);
        weights.push(softmax(scores, i));
    }
    return { weights, Q, K };
}
```

- [ ] **Step 4: Write head matrix generators**

```javascript
// Generate a random matrix from a seed
function generateMatrix(seed, dim = 8) {
    const rng = mulberry32(seed);
    return Array.from({length: dim}, () =>
        Array.from({length: dim}, () => (rng() * 2 - 1) * 0.5)
    );
}

// Head 1: positional — add position-decay bias to Wq/Wk
// so nearby tokens score higher
function createPositionalHead() {
    const Wq = generateMatrix(42);
    const Wk = generateMatrix(43);
    // Bias: add a strong diagonal component to encourage local attention
    for (let i = 0; i < 8; i++) {
        Wq[i][i] += 0.8;
        Wk[i][i] += 0.8;
    }
    return { Wq, Wk, Wv: generateMatrix(44) };
}

// Head 2: semantic — hand-tuned for the demo sentence
// Tuning: amplify certain cross-term dimensions to create
// semantic clustering for "cat/purred", "warm/mat" etc.
function createSemanticHead() {
    const Wq = generateMatrix(100);
    const Wk = generateMatrix(101);
    for (let i = 0; i < 8; i++) {
        for (let j = 0; j < 8; j++) {
            Wq[i][j] *= 1.5;
            Wk[i][j] *= 1.5;
        }
    }
    return { Wq, Wk, Wv: generateMatrix(102) };
}
```

- [ ] **Step 5: Wire up the data model and verify in console**

Wrap the data into the spec's pluggable interface shape, so swapping in real model data means replacing one object:

```javascript
// Pluggable data config — matches spec interface
const modelConfig = {
    tokens: "The cat sat on the warm mat and purred softly".split(' '),
    embeddings: null, // populated below
    heads: [createPositionalHead(), createSemanticHead()]
};
modelConfig.embeddings = modelConfig.tokens.map(
    t => generateEmbedding(t, 8));

// Convenience aliases used throughout
const TOKENS = modelConfig.tokens;
const DIM = 8;

let currentHead = 0;
let attentionData = computeAttention(
    TOKENS, modelConfig.embeddings, modelConfig.heads[currentHead]);

// Console smoke test
console.log('Tokens:', TOKENS);
console.log('Embedding[0] (The):', modelConfig.embeddings[0]);
console.log('Weights row 0:', attentionData.weights[0]);
console.log('Sum of row 0:',
    attentionData.weights[0].reduce((a, b) => a + b, 0));
// Should print ~1.0
```

Open in browser, check console: embeddings should be 8-dim unit vectors, weights row should sum to ~1.0, token 0 should have self-attention weight of 1.0 (only token available).

- [ ] **Step 6: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: attention visualizer data layer with PRNG, Q/K/V, softmax"
```

---

### Task 2: HTML Structure + Dark Mode CSS

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

Build the three-panel layout shell with dark mode styling. No interactivity yet — just the DOM structure and CSS.

- [ ] **Step 1: Write the HTML structure**

Three panels in a top/bottom split:
- `#app` container (full viewport)
- `#text-panel` (top ~35%) with head selector pills and token display area + SVG overlay
- `#bottom-panels` (bottom ~65%) containing `#heatmap-panel` (left 50%) and `#computation-panel` (right 50%)

```html
<div id="app">
    <h1>Transformer Attention Visualizer</h1>
    <div id="head-selector">
        <button class="head-pill active" data-head="0">
            Head 1 (positional)
        </button>
        <button class="head-pill" data-head="1">
            Head 2 (semantic)
        </button>
    </div>
    <div id="text-panel">
        <div id="text-display"></div>
        <svg id="svg-overlay"></svg>
    </div>
    <div id="bottom-panels">
        <div id="heatmap-panel">
            <div class="panel-label">ATTENTION MATRIX
                <span class="causal-mask-label">Causal Mask
                    <span class="tooltip">?
                        <span class="tooltip-text">
                            Each token can only attend to itself and
                            earlier tokens (prevents seeing the future)
                        </span>
                    </span>
                </span>
            </div>
            <canvas id="heatmap-canvas"></canvas>
        </div>
        <div id="computation-panel">
            <div class="panel-label">COMPUTATION</div>
            <div id="computation-content"></div>
        </div>
    </div>
</div>
```

- [ ] **Step 2: Write the dark mode CSS**

Full CSS based on the spec's color palette:
- Body: `#1a1a2e`, text: `#ccc`
- Panel borders: `#2a2a4a`
- Active token (query): `#e94560` background
- Key token in click state (`.key-highlight`): `#f5a623` background, dark text
- Attended tokens: `#f5a623` border
- Head pills: active = `#e94560`, inactive = `#333`
- Monospace for computation panel
- Token spans: inline-block, cursor pointer, padding, border-radius
- SVG overlay: absolute positioned over text panel
- Bottom panels: flexbox, 50/50 split
- Tooltip: position absolute, dark bg, appears on hover of "?" span

- [ ] **Step 3: Verify layout in browser**

Open file — should see dark background, head selector pills, empty text area, two bottom panels side by side with labels. No interactivity yet.

- [ ] **Step 4: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: three-panel layout with dark mode CSS"
```

---

### Task 3: Text Display Panel with Hover Lines

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

Render tokens as interactive spans. Hovering a token draws SVG attention lines and highlights attended tokens.

**Note:** This task introduces forward declarations for `clickState`, `hoverQuery`, `clearAllHighlights`, `clearHoverState`, and `drawSingleAttentionLine` as stubs. Task 6 replaces them with the full state machine.

- [ ] **Step 1: Render tokens into the text display**

```javascript
function renderTokens() {
    const display = document.getElementById('text-display');
    display.textContent = ''; // clear
    TOKENS.forEach((token, i) => {
        const span = document.createElement('span');
        span.textContent = token;
        span.classList.add('token');
        span.dataset.index = i;
        span.id = `token-${i}`;
        display.appendChild(span);
    });
}
```

- [ ] **Step 2: Implement SVG line drawing for a query token**

```javascript
function getTokenCenter(index) {
    const span = document.getElementById(`token-${index}`);
    const panel = document.getElementById('text-panel');
    if (!span || !panel) return null;
    const sr = span.getBoundingClientRect();
    const pr = panel.getBoundingClientRect();
    return {
        x: sr.left - pr.left + sr.width / 2,
        y: sr.top - pr.top + sr.height * 0.6
    };
}

function drawAttentionLines(queryIdx) {
    const svg = document.getElementById('svg-overlay');
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const weights = attentionData.weights[queryIdx];
    const targetPos = getTokenCenter(queryIdx);
    if (!targetPos) return;

    weights.forEach((w, keyIdx) => {
        if (w < 0.01 || keyIdx > queryIdx) return;
        const sourcePos = getTokenCenter(keyIdx);
        if (!sourcePos) return;

        const line = document.createElementNS(
            'http://www.w3.org/2000/svg', 'line');
        line.setAttribute('x1', sourcePos.x);
        line.setAttribute('y1', sourcePos.y);
        line.setAttribute('x2', targetPos.x);
        line.setAttribute('y2', targetPos.y);
        // Interpolate red→orange based on weight (spec: orange/red gradient)
        const r = Math.round(233 + w * (245 - 233));
        const g = Math.round(69 + w * (166 - 69));
        const b = Math.round(96 + w * (35 - 96));
        line.setAttribute('stroke', `rgb(${r}, ${g}, ${b})`);
        line.setAttribute('stroke-width', 0.5 + w * 7.5);
        line.setAttribute('opacity', 0.2 + w * 0.8);
        line.setAttribute('stroke-linecap', 'round');
        svg.appendChild(line);
    });
}
```

- [ ] **Step 3: Add state stubs and hover event listeners + token highlighting**

Forward declarations — Task 6 replaces these with the full implementation:

```javascript
// State stubs (replaced in Task 6)
let clickState = null;

function renderHeatmap() {} // no-op until Task 4
function renderComputationSummary() {} // no-op until Task 5

function clearHoverState() {
    document.querySelectorAll('.token').forEach(s => {
        s.classList.remove('active', 'key-highlight');
        s.style.borderColor = 'transparent';
        s.style.borderWidth = '';
    });
    const svg = document.getElementById('svg-overlay');
    while (svg.firstChild) svg.removeChild(svg.firstChild);
}

function clearAllHighlights() {
    clearHoverState();
    renderHeatmap();
    const content = document.getElementById('computation-content');
    content.textContent = '';
    const p = document.createElement('p');
    p.className = 'placeholder';
    p.textContent = 'Hover a token or click a heatmap cell';
    content.appendChild(p);
}

function hoverQuery(queryIdx) {
    if (clickState) return;
    highlightTokens(queryIdx);
    drawAttentionLines(queryIdx);
}

function drawSingleAttentionLine(queryIdx, keyIdx) {
    const svg = document.getElementById('svg-overlay');
    while (svg.firstChild) svg.removeChild(svg.firstChild);
    const w = attentionData.weights[queryIdx][keyIdx];
    const sourcePos = getTokenCenter(keyIdx);
    const targetPos = getTokenCenter(queryIdx);
    if (!sourcePos || !targetPos) return;
    const line = document.createElementNS(
        'http://www.w3.org/2000/svg', 'line');
    line.setAttribute('x1', sourcePos.x);
    line.setAttribute('y1', sourcePos.y);
    line.setAttribute('x2', targetPos.x);
    line.setAttribute('y2', targetPos.y);
    const r = Math.round(233 + w * (245 - 233));
    const g = Math.round(69 + w * (166 - 69));
    const b = Math.round(96 + w * (35 - 96));
    line.setAttribute('stroke', `rgb(${r}, ${g}, ${b})`);
    line.setAttribute('stroke-width', 0.5 + w * 7.5);
    line.setAttribute('opacity', 0.2 + w * 0.8);
    line.setAttribute('stroke-linecap', 'round');
    svg.appendChild(line);
}
```

Token highlighting and event listeners:

```javascript
function highlightTokens(queryIdx) {
    document.querySelectorAll('.token').forEach(s => {
        s.classList.remove('active', 'attended');
        s.style.borderColor = 'transparent';
        s.style.borderWidth = '';
    });

    const querySpan = document.getElementById(`token-${queryIdx}`);
    if (querySpan) querySpan.classList.add('active');

    const weights = attentionData.weights[queryIdx];
    weights.forEach((w, keyIdx) => {
        if (w < 0.01 || keyIdx === queryIdx || keyIdx > queryIdx) return;
        const span = document.getElementById(`token-${keyIdx}`);
        if (span) {
            span.style.borderColor = '#f5a623';
            span.style.borderWidth = `${1 + w * 3}px`;
        }
    });
}

// Event delegation on text display
document.getElementById('text-display').addEventListener(
    'mouseover', e => {
        if (!e.target.classList.contains('token')) return;
        if (clickState) return; // click takes precedence
        const idx = parseInt(e.target.dataset.index, 10);
        hoverQuery(idx);
    });

document.getElementById('text-panel').addEventListener(
    'mouseleave', () => {
        if (!clickState) clearAllHighlights();
    });
```

- [ ] **Step 4: Verify in browser**

Hover over tokens — should see orange/red SVG lines connecting to attended tokens, active token highlighted in accent color, attended tokens get orange borders proportional to weight.

- [ ] **Step 5: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: text display with hover-driven attention lines"
```

---

### Task 4: Heatmap Matrix with Causal Mask

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

Render the N*N attention matrix on a canvas element. Grey out upper triangle (causal mask). Highlight row on hover.

- [ ] **Step 1: Write the heatmap renderer**

Use Canvas 2D API. For each cell:
- If `col > row`: causal mask, draw as `rgba(50, 50, 50, 0.5)`
- Else: color from black-orange-yellow scale based on weight value
- Draw token labels on left (rows/queries) and top (columns/keys, rotated)
- Accept optional `highlightRow` and `highlightCell` params for interaction state

Color scale function (weight 0..1 maps to black..orange..yellow):
```javascript
function weightToColor(w) {
    const r = Math.round(w * 245 + 10);
    const g = Math.round(w * 166);
    const b = Math.round(w * 35);
    return `rgb(${r}, ${g}, ${b})`;
}
```

- [ ] **Step 2: Add click handler for heatmap cells**

Map canvas click coordinates to (row, col) using cellSize and label offset. If the cell is in the lower triangle (valid), call `setClickState(row, col)`.

- [ ] **Step 3: Wire heatmap row highlight to text hover**

When hovering a text token, call `renderHeatmap(queryIdx)` to highlight that row. On mouse leave, call `renderHeatmap()` with no highlight.

- [ ] **Step 4: Verify in browser**

Should see 10x10 grid with token labels. Lower triangle shows orange/yellow heat, upper triangle is dark grey. Hovering text tokens highlights the corresponding heatmap row.

- [ ] **Step 5: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: heatmap matrix with causal mask and row highlighting"
```

---

### Task 5: Computation Panel

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

Two modes: summary (full softmax vector for a query) and drill-down (Q, K, dot product, scale, softmax for a specific pair).

- [ ] **Step 1: Write the summary mode renderer**

When hovering a text token, show the softmax distribution for that query as a horizontal bar chart with token labels and weight values. Build using DOM elements (div bars) rather than canvas.

For each key token index 0..queryIdx:
- Label with token text
- Horizontal bar div, width proportional to weight, colored from heatmap scale
- Weight value text at end of bar

- [ ] **Step 2: Write the drill-down mode renderer**

When a heatmap cell is clicked, show the full computation for that (query, key) pair. All content built via DOM createElement (no raw HTML strings):

1. Header: "query_token -> key_token"
2. Q vector: 8 values as horizontal labeled bars (cyan `#8be9fd`)
3. K vector: 8 values as horizontal labeled bars (green `#50fa7b`)
4. Dot product: `Q . K = <value>` as highlighted scalar
5. Scaling: `/ sqrt(8) = <value>` in orange `#f5a623`
6. Softmax distribution: horizontal bar chart with the selected key's bar highlighted

- [ ] **Step 3: Wire computation panel to hover and click state**

- Hover on text token calls `renderComputationSummary(queryIdx)`
- Click on heatmap cell calls `renderComputationDrilldown(queryIdx, keyIdx)`
- Clear state shows placeholder text ("Hover a token or click a heatmap cell")

- [ ] **Step 4: Verify in browser**

Hover a token — computation panel shows bar chart of attention distribution. Click a heatmap cell — panel shows Q/K vectors, dot product, scaling, softmax with the pair highlighted.

- [ ] **Step 5: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: computation panel with summary and drill-down modes"
```

---

### Task 6: Interaction Coordination — Hover vs Click State

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

Implement the two-mode interaction system: ephemeral hover and persistent click, with click taking precedence.

- [ ] **Step 1: Implement state management**

```javascript
let clickState = null; // null or { queryIdx, keyIdx }

function setClickState(queryIdx, keyIdx) {
    if (clickState
        && clickState.queryIdx === queryIdx
        && clickState.keyIdx === keyIdx) {
        clickState = null; // toggle off
    } else {
        clickState = { queryIdx, keyIdx };
    }
    updateAllPanels();
}

function hoverQuery(queryIdx) {
    if (clickState) return; // click takes precedence
    highlightTokens(queryIdx);
    drawAttentionLines(queryIdx);
    renderHeatmap(queryIdx);
    renderComputationSummary(queryIdx);
}

function clearAllHighlights() {
    if (clickState) return;
    clearHoverState();
    renderHeatmap();
    const content = document.getElementById('computation-content');
    content.textContent = '';
    const p = document.createElement('p');
    p.className = 'placeholder';
    p.textContent = 'Hover a token or click a heatmap cell';
    content.appendChild(p);
}

function updateAllPanels() {
    if (clickState) {
        const { queryIdx, keyIdx } = clickState;
        highlightTokens(queryIdx);
        const keySpan = document.getElementById(`token-${keyIdx}`);
        if (keySpan) keySpan.classList.add('key-highlight');
        drawSingleAttentionLine(queryIdx, keyIdx);
        renderHeatmap(queryIdx, [queryIdx, keyIdx]);
        renderComputationDrilldown(queryIdx, keyIdx);
    } else {
        clearAllHighlights();
    }
}
```

- [ ] **Step 2: Add Escape key listener**

```javascript
document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && clickState) {
        clickState = null;
        clearAllHighlights();
    }
});
```

- [ ] **Step 3: Wire heatmap click to setClickState**

Complete the canvas click handler from Task 4 to call `setClickState(row, col)` when a valid lower-triangle cell is clicked.

- [ ] **Step 4: Verify in browser**

1. Hover tokens — all three panels update, clears on mouse leave
2. Click heatmap cell — locks drill-down view, hovering does not change it
3. Click same cell again — deselects, returns to hover mode
4. Click different cell — switches to new pair
5. Press Escape — clears click state

- [ ] **Step 5: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: hover vs click state management across panels"
```

---

### Task 7: Head Selector + Semantic Head Tuning

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

Wire up the head selector pills. Tune Head 2 matrices to produce interesting semantic patterns for the demo sentence.

- [ ] **Step 1: Wire head selector buttons**

```javascript
document.querySelectorAll('.head-pill').forEach(btn => {
    btn.addEventListener('click', () => {
        currentHead = parseInt(btn.dataset.head, 10);
        document.querySelectorAll('.head-pill').forEach(
            b => b.classList.remove('active'));
        btn.classList.add('active');
        attentionData = computeAttention(
            TOKENS, modelConfig.embeddings,
            modelConfig.heads[currentHead]);
        clickState = null;
        clearAllHighlights();
        renderHeatmap();
    });
});
```

- [ ] **Step 2: Tune Head 2 (semantic) matrices**

Open browser, switch to Head 2, examine heatmap. Iteratively adjust the `createSemanticHead()` function's matrix biases until:
- "cat" (idx 1) strongly attends to "purred" (idx 8) or vice versa
- "warm" (idx 5) and "mat" (idx 6) show mutual attention
- Pattern visibly differs from Head 1's diagonal band

This is manual tuning — adjust matrix multipliers, add targeted biases to specific rows/cols, re-check in browser. The goal is a plausible-looking semantic pattern, not perfection.

- [ ] **Step 3: Verify both heads in browser**

Switch between heads — heatmap pattern should visibly change. Head 1 shows diagonal concentration. Head 2 shows scattered clusters. Computation panel math updates correctly for both.

- [ ] **Step 4: Commit**

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: head selector with tuned positional and semantic heads"
```

---

### Task 8: Polish + Final Verification

**Files:**
- Modify: `attention-visualizer/attention-visualizer.html`

- [ ] **Step 1: SVG sizing fix**

Ensure the SVG overlay tracks the text panel dimensions using a ResizeObserver:

```javascript
const resizeObserver = new ResizeObserver(() => {
    const panel = document.getElementById('text-panel');
    const svg = document.getElementById('svg-overlay');
    svg.setAttribute('width', panel.clientWidth);
    svg.setAttribute('height', panel.clientHeight);
});
resizeObserver.observe(document.getElementById('text-panel'));
```

- [ ] **Step 2: Add transitions and visual polish**

- Smooth CSS transitions on token highlights (0.15s ease)
- Heatmap cell hover cursor (pointer for valid cells)
- Computation panel fade-in on content change
- Subtle panel border separators

- [ ] **Step 3: Console verification of all weight sums**

Add a one-time verification block that runs on page load:

```javascript
attentionData.weights.forEach((row, i) => {
    const sum = row.reduce((a, b) => a + b, 0);
    console.assert(
        Math.abs(sum - 1.0) < 1e-6,
        `Row ${i} sums to ${sum}`);
});
console.log('All attention weight rows sum to 1.0');
```

- [ ] **Step 4: Full manual test pass**

Verify all interactions work:
1. Page loads with Head 1 active, heatmap rendered, computation panel shows placeholder
2. Hover text tokens — lines + heatmap row + computation summary all update
3. Mouse leave — everything clears
4. Click heatmap cell — drill-down locks, hover suppressed
5. Click same cell — deselects
6. Press Escape — clears click state
7. Switch to Head 2 — heatmap pattern changes, clears any selection
8. All weight labels show values that make sense (sum to ~1.0 per row)
9. Causal mask tooltip appears on hover of "?"

- [ ] **Step 5: Clean up and final commit**

Remove temporary console.log statements from Task 1 Step 5 (keep the assert block). Final commit:

```bash
git add attention-visualizer/attention-visualizer.html
git commit -m "feat: polish, SVG sizing, final verification"
```
