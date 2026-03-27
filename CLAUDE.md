# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

Pixelator is a pixel art sprite animation pipeline that converts images into pixel art, generates animation strips using the OpenAI GPT Image API (gpt-image-1.5), normalizes frames, and exports animated GIFs. It has both a CLI (pipeline.py) and a web UI (server.py + app.html).

## Setup & Running

```bash
# Initial setup (creates .venv, installs deps via uv)
./setup.sh

# Activate venv
source .venv/bin/activate

# Run the interactive CLI pipeline
python pipeline.py                          # fully interactive
python pipeline.py -i hero.png -a idle      # partially specified

# Run the web server (serves app.html at http://localhost:8765)
python server.py
```

Requires `OPENAI_API_KEY` in the environment for AI features.

**IMPORTANT:** After modifying any Python backend file (server.py, grid_detector.py, ai_refiner.py, etc.), you MUST restart the server for changes to take effect. Kill the running server and re-run `python server.py`. The web UI (app.html) reloads automatically in the browser, but the Python backend does not hot-reload.

## Architecture

The pipeline has 7 phases, each handled by a dedicated module:

1. **Pixelate** (`pixelator_grid.py` + `grid_detector.py` + `edge_detector.py`) — Detects grids in source images and extracts pixel art variants with different spacing/color parameters. CLI-only path.
2. **AI Refine** (`ai_refiner.py`) — Calls OpenAI GPT Image 1.5 API to refine pixel art, generate sprites from prompts/references, and generate animation strips. Central AI module.
3. **Build Canvas** (`canvas_builder.py`) — Upscales a seed sprite (nearest-neighbor) onto a 1024x1024 transparent canvas with a grid layout for the edit API. Computes optimal grid layout (cols x rows x slot_size) to maximize slot size.
4. **Normalize** (`sprite_normalizer.py`) — Splits raw AI strips into frames, detects sprite bounding boxes, computes shared scale factor across all frames, and aligns by bottom-center anchor (feet). Produces individual frame PNGs + spritesheet.
5. **Export GIF** (`gif_exporter.py`) — Converts RGBA frames to palette mode with transparency and assembles animated GIF.

**Two entry points:**
- `pipeline.py` — Interactive Rich CLI that orchestrates all phases sequentially with user prompts between steps
- `server.py` — FastAPI backend exposing each phase as a POST endpoint; serves `app.html` as the single-page web UI

**Key design details:**
- All AI calls go through `ai_refiner.py` which wraps the OpenAI `images.edit` and `images.generate` APIs
- Canvas grid layout is auto-computed by `canvas_builder.compute_grid_layout()` — shared between canvas building and normalization
- Sprite normalization uses a single shared scale factor (not per-frame) to prevent size inconsistency across animation frames
- Frame alignment uses bottom-center anchor point (character feet) via center-of-mass calculation
- Output goes to `output/<character_name>/base/` and `output/<character_name>/animations/<type>/`

## Key Modules as Standalone CLIs

Each processing module can also run independently:

```bash
python ai_refiner.py generate -d "a knight" -f 32
python ai_refiner.py refine input.png -d "a knight"
python ai_refiner.py animate canvas.png -t idle -n 4
python canvas_builder.py seed.png -n 6
python sprite_normalizer.py strip.png -n 4 -t 64
python gif_exporter.py --frames-dir ./normalized/ -s 4 -d 100
```
