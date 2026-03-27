# Pixelator

A pixel art sprite animation pipeline that generates pixel art characters, detects pixel grids, generates animation strips via OpenAI GPT Image API, normalizes frames, and exports animated GIFs.

## Quick Start

```bash
./setup.sh                    # creates .venv, installs deps
source .venv/bin/activate
export OPENAI_API_KEY=sk-...
python server.py              # web UI at http://localhost:8765
```

## Pipeline Stages

The web UI (`app.html` + `server.py`) has four stages, each independently navigable. Every stage shows previously generated artifacts so you can pick from existing work without re-running earlier stages.

### Stage 1: Generate Base Sprite

Generates pixel art character sprites from text prompts using `gpt-image-1.5`.

- **Preset characters**: ~30 D&D-style presets (heroes, enemies, monsters, NPCs) auto-populate name and description
- **Resolution**: 8x8 through 128x128 (default 16x16)
- **Variants**: Generate 1-4 candidates in parallel
- **Output**: `output/<character>/base/candidate_<uid>.png`

### Stage 2: Confirm Pixel Grid

Detects the logical pixel grid in a generated image and downscales it to a clean pixel art sprite.

#### Reconstruction Error Scoring

The core algorithm scores each candidate grid size (8-128) by measuring how well that grid explains the image:

1. **Downscale** the image to NxN using block averaging (PIL BOX filter)
2. **Upscale** back to original size with nearest-neighbor
3. **MSE** between original and reconstructed = the score

Lower MSE = better grid fit. A true pixel art image at 32x32 will reconstruct almost perfectly at grid size 32, producing a sharp minimum in the score curve.

This is fully vectorized with numpy (`np.ix_` for index mapping) and runs ~27ms per grid size. The full 8-128 sweep (121 sizes) completes in ~3.3 seconds.

#### Peak-Based Edge Detection (`_find_edges_from_peak`)

For per-grid-size boundary detection (used in the scored grid overlays), edges are found starting from the **strongest signal region** rather than from position 0:

1. Convolve the color-difference signal with a box filter to find the densest edge region
2. Anchor on the strongest edge in that region
3. Expand outward in both directions, snapping to real edges where they exist
4. Fall back to uniform spacing where the signal is flat (e.g., white background areas)

This handles AI-generated images where the sprite is centered and the edges/corners are uniform background with no pixel grid signal.

#### Local Minima Detection (Plateau-Aware)

The score curve often has multiple valleys at different grid sizes (e.g., a true minimum at 32x32, but also harmonics at 16x16 or 64x64). The system finds all **local minima** — points where the score is lower than both neighbors.

**Plateau handling**: When adjacent grid sizes have identical scores (e.g., 56, 57, 58 all scoring 55), a strict `<` comparison misses them all. Instead, for each point, the algorithm looks past equal-score neighbors to find the first *different* score on each side. If both sides are higher, the plateau is a local minimum. Only the center of the plateau is marked to keep the display clean.

- Green border (bright, star): global best
- Green border (dim, diamond): other local minima
- "Best only" toggle: shows only local minima (checked by default)
- Arrow keys: step through grid sizes one at a time
- Cmd/Ctrl+Arrow: jump between local minima

#### Output

- Grid preview overlays: `output/<character>/grids/<candidate_name>/`
- Confirmed pixel seeds: `output/<character>/seeds/`

### Stage 3: Generate Animation

Creates animation strips from a confirmed pixel seed using the OpenAI `images.edit` API.

#### Layout Modes

- **Linear** (default): All frames in a single horizontal row (cols=N, rows=1). The seed sprite is placed vertically centered on the left. This produces sequential left-to-right frames which may give the AI better spatial continuity for animation.
- **Grid**: Auto-computed optimal grid layout that maximizes slot size (e.g., 4 frames = 2x2 at 512px slots vs 4x1 at 256px slots). Larger slots give the AI more detail resolution per frame.

#### Masking

The canvas uses the image's own transparency to guide editing. A separate mask is also generated:
- Slot 0 (seed sprite): opaque in mask = preserve
- All other slots: transparent in mask = generate

Per OpenAI docs, the mask is advisory — the model may still modify masked areas, but it biases toward preserving the seed.

#### Animation Types

idle, walk, run, attack, hurt, jump, death — each with specific frame-by-frame action descriptions in the prompt.

#### Variants

Generate 1-4 animation strips in parallel. All candidates are saved and displayed in a picker strip. Each strip is saved with a unique ID:
- `output/<character>/animations/<type>/canvas_<uid>.png`
- `output/<character>/animations/<type>/strip_raw_<uid>.png`

### Stage 4: Align & Export

Normalizes raw animation strips into game-ready frames and exports GIFs/spritesheets.

#### Frame Resolution Auto-Detection

Same reconstruction error scoring as Stage 2, applied to the first frame slot of the animation strip. Sweeps grid sizes 4-128, finds local minima, lets you pick the best pixel density.

This is important because the AI-generated animation frames may have a different effective pixel density than the original seed sprite.

#### Normalization Pipeline (`sprite_normalizer.py`)

1. **Extract frames** from the strip grid (respects cols, rows, slot_size, y_offset for linear layouts)
2. **Remove background** — detects and removes opaque backgrounds the AI sometimes adds
3. **Detect bounding boxes** for each frame's sprite
4. **Compute shared scale factor** across all frames (prevents per-frame size inconsistency)
5. **Align by bottom-center anchor** (character feet) using center-of-mass calculation
6. **Output** individual frame PNGs + spritesheet

#### Alignment Controls

- Arrow keys: step through frames
- Shift+Arrow: nudge current frame by 1 pixel (in sprite space, regardless of display scale)
- Scale slider: 1x-32x (default 8x)
- FPS slider: 1-24 (default 8)
- Onion skinning toggle
- Reset all offsets

#### Export

- Animated GIF with transparency
- Spritesheet PNG

## Architecture

```
app.html          Single-page web UI
server.py         FastAPI backend (all endpoints)
ai_refiner.py     OpenAI API wrapper (generate, refine, animate)
canvas_builder.py Canvas + mask construction for edit API
grid_detector.py  Grid detection, scoring, edge detection
edge_detector.py  Color difference calculation
sprite_normalizer.py  Frame extraction, scaling, alignment
gif_exporter.py   GIF assembly with transparency
pipeline.py       Interactive CLI (alternative to web UI)
```

### Key API Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /api/generate-base` | Generate base sprites from prompt |
| `POST /api/batch-grid-preview` | Score grid sizes 8-128, generate previews |
| `POST /api/confirm-grid` | Downscale to pixel art, save as seed |
| `POST /api/generate-animation` | Build canvas + mask, call edit API |
| `POST /api/score-strip-grids` | Score pixel density on animation frames |
| `POST /api/normalize` | Extract, scale, align frames |
| `POST /api/export-gif` | Assemble aligned frames into GIF |
| `GET /api/list-bases` | List all candidates and seeds |
| `GET /api/list-all-strips` | List all animation strips |
| `GET /api/list-animation-candidates` | List strips for a character+type |

### Output Directory Structure

```
output/
  <character_name>/
    base/                          Raw AI-generated candidates
      candidate_<uid>.png
    grids/
      <candidate_name>/            Grid preview overlays per candidate
        grid_<size>.png
        grid_<size>_pixel.png
    seeds/                         Confirmed pixel art seeds
      <candidate_name>_seed.png
    animations/
      <type>/                      Per animation type (idle, attack, etc.)
        canvas_<uid>.png           Edit API input canvas
        canvas_<uid>_mask.png      Edit API mask
        strip_raw_<uid>.png        Raw AI output
        normalized_<uid>/          Normalized frames
          *_frame_01.png
          *_normalized_sheet.png
      <type>/
        <type>.gif                 Exported animated GIF
```

## Algorithms Reference

### Reconstruction Error (MSE)

```python
small = img.resize((grid_size, grid_size), Image.BOX)
reconstructed = small.resize((w, h), Image.NEAREST)
mse = mean((original - reconstructed)^2)
```

The BOX filter computes block averages (equivalent to the "mode" of a perfect pixel art cell). NEAREST upscaling paints each cell with a single color. MSE measures how much information was lost — for the correct grid size, this approaches zero.

### Edge Detection from Peak Signal

```python
# 1. Find strongest signal region via convolution
cumsum = np.cumsum(differences)
windowed = cumsum[kernel_size:] - cumsum[:-kernel_size]
peak_center = np.argmax(windowed)

# 2. Anchor on strongest edge near peak
# 3. Expand outward, snapping to edges or falling back to uniform spacing
```

### Plateau-Aware Local Minima

```python
for each point i:
    look left past equal scores → find first different score
    look right past equal scores → find first different score
    if both sides are higher → this plateau is a local minimum
    mark only the center of the plateau
```

## Known Limitations & Future Work

- **Mask is advisory**: OpenAI's edit API treats the mask as a hint. The seed sprite in slot 0 may still be modified by the AI. A stronger approach might composite the original seed back onto slot 0 after generation.
- **Linear vs grid quality**: Linear layout gives smaller slots (256px for 4 frames vs 512px for grid). The AI has less resolution to work with but may produce better sequential flow. Worth A/B testing per animation type.
- **Background removal heuristic**: The normalizer removes opaque backgrounds by detecting corner colors. This can fail on sprites that extend to the frame edges.
- **Scale factor is global**: All frames share one scale factor to prevent size jitter, but this means one oversized frame constrains all others.
- **No undo in alignment**: Frame offset changes are immediate with no undo stack.
- **Server doesn't hot-reload**: Must kill and restart `python server.py` after any backend Python changes.
