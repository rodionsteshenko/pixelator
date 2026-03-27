#!/usr/bin/env python3
"""
Pixelator Web Server — FastAPI backend for the unified sprite pipeline UI.

Endpoints:
  POST /api/generate-base     — Generate base character sprites from prompt
  POST /api/generate-animation — Generate animation strip from base sprite
  POST /api/normalize          — Normalize a raw strip into aligned frames
  POST /api/export-gif         — Export frames as animated GIF
  GET  /api/images/<path>      — Serve generated images
"""

import os
import uuid
import shutil
import base64
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

import ai_refiner
import canvas_builder
import sprite_normalizer
import gif_exporter
import grid_detector

app = FastAPI(title="Pixelator")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

OUTPUT_ROOT = Path("output")
OUTPUT_ROOT.mkdir(exist_ok=True)

_executor = ThreadPoolExecutor(max_workers=8)

# Serve generated images
app.mount("/output", StaticFiles(directory="output"), name="output")
# Serve examples
if Path("examples").exists():
    app.mount("/examples", StaticFiles(directory="examples"), name="examples")


def _unique_id():
    return uuid.uuid4().hex[:8]


def _image_to_data_url(path: str) -> str:
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = Path(path).suffix.lower()
    mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext.lstrip("."), "image/png")
    return f"data:{mime};base64,{b64}"


def _mark_local_minima(results):
    """Mark local minima in a scored results list, handling plateaus."""
    if not results:
        return
    scores = [r["score"] for r in results]
    n = len(scores)
    local_min_indices = set()

    # For each point, find the nearest neighbor with a DIFFERENT score
    # on each side. If both are higher, this point (or plateau) is a local min.
    for i in range(n):
        # Look left for a different score
        left_higher = False
        for j in range(i - 1, -1, -1):
            if scores[j] != scores[i]:
                left_higher = scores[j] > scores[i]
                break
        else:
            left_higher = True  # at edge, treat as higher

        # Look right for a different score
        right_higher = False
        for j in range(i + 1, n):
            if scores[j] != scores[i]:
                right_higher = scores[j] > scores[i]
                break
        else:
            right_higher = True  # at edge, treat as higher

        if left_higher and right_higher:
            local_min_indices.add(i)

    # For plateaus, only mark the center point(s)
    # Group consecutive local min indices with equal scores
    groups = []
    current_group = []
    for i in sorted(local_min_indices):
        if current_group and (i != current_group[-1] + 1 or scores[i] != scores[current_group[-1]]):
            groups.append(current_group)
            current_group = [i]
        else:
            current_group.append(i)
    if current_group:
        groups.append(current_group)

    # Keep only the center of each plateau
    final_indices = set()
    for group in groups:
        mid = group[len(group) // 2]
        final_indices.add(mid)

    # Always include global best
    global_best_idx = min(range(n), key=lambda i: scores[i])
    final_indices.add(global_best_idx)

    sorted_by_score = sorted(results, key=lambda r: r["score"])
    for i, r in enumerate(results):
        r["is_local_min"] = i in final_indices
        r["is_top"] = i in final_indices
        r["rank"] = next(j + 1 for j, s in enumerate(sorted_by_score) if s["grid_size"] == r["grid_size"])


@app.get("/")
async def index():
    return FileResponse("app.html")


@app.get("/api/list-bases")
async def list_bases():
    """List all generated base sprites and seeds across all characters."""
    bases = []
    if not OUTPUT_ROOT.exists():
        return {"bases": []}

    for char_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not char_dir.is_dir():
            continue
        # Candidates from base/
        base_dir = char_dir / "base"
        if base_dir.exists():
            for f in sorted(base_dir.iterdir()):
                if f.suffix.lower() == ".png" and f.stem.startswith("candidate_"):
                    rel_path = str(f.relative_to(Path(".")))
                    bases.append({
                        "character": char_dir.name,
                        "name": f.stem,
                        "path": rel_path,
                        "url": f"/{rel_path}",
                        "type": "candidate",
                    })
        # Seeds from seeds/
        seeds_dir = char_dir / "seeds"
        if seeds_dir.exists():
            for f in sorted(seeds_dir.iterdir()):
                if f.suffix.lower() == ".png" and f.stem.endswith("_seed") and "_preview" not in f.stem:
                    rel_path = str(f.relative_to(Path(".")))
                    bases.append({
                        "character": char_dir.name,
                        "name": f.stem,
                        "path": rel_path,
                        "url": f"/{rel_path}",
                        "type": "seed",
                    })
        # Legacy: seeds in base/ (old layout)
        if base_dir.exists():
            for f in sorted(base_dir.iterdir()):
                if f.suffix.lower() == ".png" and "seed" in f.stem and "preview" not in f.stem:
                    rel_path = str(f.relative_to(Path(".")))
                    if not any(b["path"] == rel_path for b in bases):
                        bases.append({
                            "character": char_dir.name,
                            "name": f.stem,
                            "path": rel_path,
                            "url": f"/{rel_path}",
                            "type": "seed",
                        })
    return {"bases": bases}


@app.get("/api/list-examples")
async def list_examples():
    """List all image files in the examples directory."""
    examples_dir = Path("examples")
    if not examples_dir.exists():
        return {"examples": []}

    image_exts = {".png", ".jpg", ".jpeg", ".gif", ".webp"}
    examples = []
    for f in sorted(examples_dir.iterdir()):
        if f.suffix.lower() in image_exts:
            examples.append({
                "name": f.stem,
                "path": str(f),
                "url": f"/examples/{f.name}",
            })
    return {"examples": examples}


@app.post("/api/preview-prompt")
async def preview_prompt(
    prompt_type: str = Form(...),
    prompt: str = Form(""),
    frame_size: int = Form(32),
    animation_type: str = Form("idle"),
    character_description: str = Form("a game character"),
    game_style: str = Form("platformer"),
    num_frames: int = Form(4),
    layout: str = Form("linear"),
    seed_path: str = Form(""),
    character_name: str = Form("character"),
    transparent: str = Form("false"),
    draw_slots: str = Form("true"),
    prefill_slots: str = Form("false"),
):
    """Return the exact prompt that would be sent to the AI, without calling the API."""
    if prompt_type == "generate":
        use_transparent = transparent.lower() in ("true", "1", "yes")
        if use_transparent:
            bg_instruction = "TRANSPARENT BACKGROUND ONLY — no scenery, no floor, no labels, no UI, no text, no glow."
        else:
            bg_instruction = "SOLID SINGLE-COLOR BACKGROUND (e.g. solid black, solid dark blue, or solid grey) — no scenery, no floor, no labels, no UI, no text, no glow. The background must be one uniform flat color."
        text = f"""Create a pixel art character sprite that FILLS THE ENTIRE IMAGE.
The character should be large and take up most of the canvas.
The image must look like {frame_size}x{frame_size} pixel art upscaled with nearest-neighbor scaling — each logical pixel is a uniform square block of solid color with perfectly hard edges. NO anti-aliasing, NO gradients, NO blending between blocks.
The entire image should be a clean grid of exactly {frame_size} columns and {frame_size} rows of crisp, solid-color square blocks.
Restrained color palette (16-32 colors max). Stepped shading only.
{bg_instruction}
IMPORTANT: The character MUST face RIGHT (toward the right side of the image).
Character: {character_description}"""
        return {"prompt": text, "model": ai_refiner.IMAGE_MODEL, "api": "images.generate"}

    elif prompt_type == "animate":
        import canvas_builder as cb
        if layout == "linear":
            cols = num_frames
            rows = 1
            slot_size = 1024 // cols
        else:
            cols, rows, slot_size = cb.compute_grid_layout(num_frames, 1024)

        action_template = ai_refiner.ANIMATION_ACTIONS.get(
            animation_type,
            "frames 1-{n} show a smooth " + animation_type + " animation",
        )
        action_description = action_template.format(n=num_frames)
        template = ai_refiner.ANIMATION_PROMPT_LINEAR if layout == "linear" else ai_refiner.ANIMATION_PROMPT_GRID
        text = template.format(
            game_style=game_style,
            animation_type=animation_type,
            num_frames=num_frames,
            slot_size=slot_size,
            canvas_size=1024,
            cols=cols,
            rows=rows,
            action_description=action_description,
            character_description=character_description,
        )

        # Build preview canvas if a seed is available
        canvas_url = None
        if seed_path and os.path.exists(seed_path):
            char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
            preview_dir = char_dir / "previews"
            preview_dir.mkdir(parents=True, exist_ok=True)
            canvas_path = str(preview_dir / "preview_canvas.png")
            force_cols = num_frames if layout == "linear" else None
            use_draw = transparent.lower() not in ("false", "0", "no")  # reuse transparent param for draw_slots
            # Parse draw_slots and prefill_slots from form — they come as extra fields
            cb.build_edit_canvas(
                seed_path, canvas_path, num_frames=num_frames, force_cols=force_cols,
                draw_slots=draw_slots.lower() in ("true", "1", "yes") if isinstance(draw_slots, str) else True,
                prefill_slots=prefill_slots.lower() in ("true", "1", "yes") if isinstance(prefill_slots, str) else False,
            )
            canvas_url = f"/{canvas_path}"

        return {"prompt": text, "model": ai_refiner.IMAGE_MODEL, "api": "images.edit",
                "layout": f"{cols}x{rows}", "slot_size": slot_size,
                "canvas_url": canvas_url}

    return {"error": "Unknown prompt_type"}


@app.post("/api/generate-base")
async def generate_base(
    prompt: str = Form(...),
    frame_size: int = Form(32),
    count: int = Form(3),
    character_name: str = Form("character"),
    transparent: str = Form("false"),
):
    """Generate multiple base character sprites from a text prompt."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    base_dir = char_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    use_transparent = transparent.lower() in ("true", "1", "yes")

    async def _gen(i):
        uid = _unique_id()
        out_path = str(base_dir / f"candidate_{uid}.png")
        try:
            await loop.run_in_executor(
                _executor, ai_refiner.generate_pixel_art, out_path, prompt, frame_size, use_transparent,
            )
            rel_path = str(Path(out_path).relative_to(Path(".")))
            return {"id": uid, "path": rel_path, "url": f"/{rel_path}"}
        except Exception as e:
            return {"id": uid, "error": str(e)}

    results = await asyncio.gather(*[_gen(i) for i in range(count)])
    return {"character_name": character_name, "results": list(results)}


@app.post("/api/generate-from-reference")
async def generate_from_reference(
    prompt: str = Form(...),
    reference_path: str = Form(...),
    frame_size: int = Form(32),
    count: int = Form(3),
    character_name: str = Form("character"),
):
    """Generate base character sprites using a reference image."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    base_dir = char_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    async def _gen(i):
        uid = _unique_id()
        out_path = str(base_dir / f"candidate_{uid}.png")
        try:
            await loop.run_in_executor(
                _executor, ai_refiner.generate_from_reference,
                reference_path, out_path, prompt, frame_size,
            )
            rel_path = str(Path(out_path).relative_to(Path(".")))
            return {"id": uid, "path": rel_path, "url": f"/{rel_path}"}
        except Exception as e:
            return {"id": uid, "error": str(e)}

    results = await asyncio.gather(*[_gen(i) for i in range(count)])
    return {"character_name": character_name, "results": list(results)}


@app.post("/api/detect-grid")
async def detect_grid(
    image_path: str = Form(...),
    character_name: str = Form("character"),
    suggested_size: int = Form(32),
):
    """Detect the pixel grid in an AI-generated sprite and return candidates."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    base_dir = char_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()
    candidates = await loop.run_in_executor(
        _executor, grid_detector.detect_pixel_grid, image_path,
    )

    # Always include the user's suggested size and common pixel art sizes
    existing_sizes = {c["grid_size"] for c in candidates}
    common_sizes = [16, 24, 32, 48, 64, 96, 128]
    extras = [suggested_size] + common_sizes
    for size in extras:
        if size not in existing_sizes:
            img = Image.open(image_path)
            pixel_block = img.width / size
            candidates.append({
                "grid_size": size,
                "pixel_block": round(pixel_block, 2),
                "confidence": 0.5 if size == suggested_size else 0.1,
            })
            existing_sizes.add(size)

    # Sort: user's suggested first, then by confidence
    candidates.sort(key=lambda c: (
        -(2.0 if c["grid_size"] == suggested_size else c["confidence"]),
    ))

    # Generate preview for the top candidates
    previews = []
    for i, c in enumerate(candidates[:8]):
        preview_path = str(base_dir / f"grid_preview_{c['grid_size']}.png")
        result = await loop.run_in_executor(
            _executor,
            grid_detector.create_grid_preview,
            image_path, c["grid_size"], preview_path,
        )
        rel_overlay = str(Path(result["grid_overlay"]).relative_to(Path(".")))
        rel_pixel = str(Path(result["pixel_preview"]).relative_to(Path(".")))
        previews.append({
            **c,
            "preview_url": f"/{rel_overlay}",
            "pixel_preview_url": f"/{rel_pixel}",
        })

    return {
        "candidates": previews,
        "image_path": image_path,
    }


@app.post("/api/grid-preview")
async def grid_preview(
    image_path: str = Form(...),
    character_name: str = Form("character"),
    grid_size: int = Form(32),
    offset_x: int = Form(0),
    offset_y: int = Form(0),
):
    """Generate a grid preview using edge-based detection (find_edges_with_window).

    Uses the grid_size to compute approx_spacing, then snaps grid lines to
    actual color transitions in the image rather than forcing uniform spacing.
    """
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    candidate_name = Path(image_path).stem
    grid_dir = char_dir / "grids" / candidate_name
    grid_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    # Compute approximate spacing from grid_size
    img = Image.open(image_path)
    approx_spacing = img.width / grid_size

    # Detect edges that snap to color transitions
    detection = await loop.run_in_executor(
        _executor,
        lambda: grid_detector.detect_grid_edges(
            image_path, approx_spacing, window_size=5,
            offset_x=offset_x, offset_y=offset_y,
        ),
    )

    col_bounds = detection["col_boundaries"]
    row_bounds = detection["row_boundaries"]

    # Generate preview using detected boundaries
    preview_path = str(grid_dir / f"grid_{grid_size}.png")
    result = await loop.run_in_executor(
        _executor,
        lambda: grid_detector.create_seed_grid_preview(
            image_path, col_bounds, row_bounds, preview_path,
        ),
    )

    rel_overlay = str(Path(result["grid_overlay"]).relative_to(Path(".")))
    rel_pixel = str(Path(result["pixel_preview"]).relative_to(Path(".")))
    return {
        "preview_url": f"/{rel_overlay}",
        "pixel_preview_url": f"/{rel_pixel}",
        "grid_size": grid_size,
        "col_boundaries": col_bounds,
        "row_boundaries": row_bounds,
    }


@app.post("/api/batch-grid-preview")
async def batch_grid_preview(
    image_path: str = Form(...),
    character_name: str = Form("character"),
    size_min: int = Form(8),
    size_max: int = Form(64),
):
    """Generate edge-detected grid previews for all sizes in a range, with quality scores.

    Phase 1: Batch edge detection + scoring (single executor call, shares image load).
    Phase 2: Parallel preview generation for all sizes.
    """
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    # Store grid previews in a subfolder named after the candidate image
    candidate_name = Path(image_path).stem
    grid_dir = char_dir / "grids" / candidate_name
    grid_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()
    grid_sizes = list(range(size_min, size_max + 1))

    # Phase 1: batch detection + scoring in one executor call
    scored = await loop.run_in_executor(
        _executor,
        grid_detector.score_grid_batch,
        image_path, grid_sizes, 5,
    )

    # Phase 2: generate previews in parallel
    async def _make_preview(grid_size, score, col_bounds, row_bounds):
        preview_path = str(grid_dir / f"grid_{grid_size}.png")
        result = await loop.run_in_executor(
            _executor,
            lambda cb=col_bounds, rb=row_bounds, pp=preview_path: grid_detector.create_seed_grid_preview(
                image_path, cb, rb, pp,
            ),
        )
        rel_overlay = str(Path(result["grid_overlay"]).relative_to(Path(".")))
        rel_pixel = str(Path(result["pixel_preview"]).relative_to(Path(".")))
        return {
            "grid_size": grid_size,
            "preview_url": f"/{rel_overlay}",
            "pixel_preview_url": f"/{rel_pixel}",
            "col_boundaries": col_bounds,
            "row_boundaries": row_bounds,
            "score": round(score, 2),
        }

    results = await asyncio.gather(*[
        _make_preview(gs, sc, cb, rb) for gs, sc, cb, rb in scored
    ])
    results = list(results)

    _mark_local_minima(results)
    return {"results": results}


@app.post("/api/seed-grid-detect")
async def seed_grid_detect_endpoint(
    image_path: str = Form(...),
    character_name: str = Form("character"),
    click_x: int = Form(...),
    click_y: int = Form(...),
):
    """Detect pixel grid by clicking a pixel — finds boundaries automatically."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    candidate_name = Path(image_path).stem
    grid_dir = char_dir / "grids" / candidate_name
    grid_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    # Run detection from click point
    detection = await loop.run_in_executor(
        _executor,
        grid_detector.click_detect_grid,
        image_path, click_x, click_y,
    )

    # Generate preview
    preview_path = str(grid_dir / "grid_seed.png")
    result = await loop.run_in_executor(
        _executor,
        grid_detector.create_seed_grid_preview,
        image_path, detection["col_boundaries"], detection["row_boundaries"],
        preview_path,
    )

    rel_overlay = str(Path(result["grid_overlay"]).relative_to(Path(".")))
    rel_pixel = str(Path(result["pixel_preview"]).relative_to(Path(".")))

    return {
        "grid_w": detection["grid_w"],
        "grid_h": detection["grid_h"],
        "pixel_w": detection["pixel_w"],
        "pixel_h": detection["pixel_h"],
        "seed_x": detection["seed_x"],
        "seed_y": detection["seed_y"],
        "col_boundaries": detection["col_boundaries"],
        "row_boundaries": detection["row_boundaries"],
        "preview_url": f"/{rel_overlay}",
        "pixel_preview_url": f"/{rel_pixel}",
    }


@app.post("/api/confirm-grid")
async def confirm_grid(
    character_name: str = Form(...),
    image_path: str = Form(...),
    grid_size: int = Form(None),
    col_boundaries: str = Form(None),
    row_boundaries: str = Form(None),
    offset_x: int = Form(0),
    offset_y: int = Form(0),
):
    """Confirm the grid and downscale to the final pixel sprite."""
    import json
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    seeds_dir = char_dir / "seeds"
    seeds_dir.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    if col_boundaries and row_boundaries:
        col_b = json.loads(col_boundaries)
        row_b = json.loads(row_boundaries)
        seed_img = await loop.run_in_executor(
            _executor,
            grid_detector.downscale_with_boundaries,
            image_path, col_b, row_b, "mode",
        )
        final_grid_size = seed_img.width
    else:
        seed_img = await loop.run_in_executor(
            _executor,
            lambda: grid_detector.downscale_to_grid(
                image_path, grid_size, "mode",
                offset_x=offset_x, offset_y=offset_y,
            ),
        )
        final_grid_size = seed_img.width

    candidate_name = Path(image_path).stem
    seed_path = seeds_dir / f"{candidate_name}_seed.png"
    seed_img.save(str(seed_path), "PNG")

    # Also save a preview (upscaled for display)
    preview_size = 256
    preview = seed_img.resize((preview_size, preview_size), Image.NEAREST)
    preview_path = seeds_dir / f"{candidate_name}_seed_preview.png"
    preview.save(str(preview_path), "PNG")

    rel_seed = str(seed_path.relative_to(Path(".")))
    rel_preview = str(preview_path.relative_to(Path(".")))

    return {
        "seed_path": rel_seed,
        "seed_url": f"/{rel_seed}",
        "preview_url": f"/{rel_preview}",
        "grid_size": final_grid_size,
    }


@app.post("/api/upload-base")
async def upload_base(
    character_name: str = Form(...),
    file: UploadFile = File(...),
):
    """Upload an existing image as the base sprite."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    base_dir = char_dir / "base"
    base_dir.mkdir(parents=True, exist_ok=True)

    uid = _unique_id()
    out_path = base_dir / f"uploaded_{uid}.png"
    with open(out_path, "wb") as f:
        content = await file.read()
        f.write(content)

    return {"path": str(out_path), "url": f"/{out_path}"}


@app.post("/api/generate-animation")
async def generate_animation(
    character_name: str = Form(...),
    seed_path: str = Form(...),
    animation_type: str = Form("idle"),
    character_description: str = Form("a game character"),
    game_style: str = Form("platformer"),
    num_frames: int = Form(6),
    layout: str = Form("linear"),
    draw_slots: str = Form("true"),
    prefill_slots: str = Form("false"),
):
    """Generate an animation strip from a base sprite."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    anim_dir = char_dir / "animations" / animation_type
    anim_dir.mkdir(parents=True, exist_ok=True)

    use_draw_slots = draw_slots.lower() in ("true", "1", "yes")
    use_prefill = prefill_slots.lower() in ("true", "1", "yes")

    # Build canvas — linear forces single row, grid auto-computes
    uid = _unique_id()
    canvas_path = str(anim_dir / f"canvas_{uid}.png")
    force_cols = num_frames if layout == "linear" else None
    canvas_result = canvas_builder.build_edit_canvas(
        seed_path, canvas_path, num_frames=num_frames,
        force_cols=force_cols,
        draw_slots=use_draw_slots,
        prefill_slots=use_prefill,
    )

    # Generate strip with mask to preserve seed sprite in slot 0
    strip_path = str(anim_dir / f"strip_raw_{uid}.png")
    ai_refiner.generate_animation_strip(
        canvas_path, strip_path,
        animation_type=animation_type,
        character_description=character_description,
        game_style=game_style,
        num_frames=num_frames,
        cols=canvas_result["cols"],
        rows=canvas_result["rows"],
        slot_size=canvas_result["slot_size"],
        mask_path=canvas_result.get("mask_path"),
        layout=layout,
    )

    # Force-composite the original seed back onto slot 0.
    # The mask is advisory — the AI may still modify the seed area.
    # This guarantees frame 1 matches the approved seed exactly.
    canvas_img = Image.open(canvas_path).convert("RGBA")
    strip_img = Image.open(strip_path).convert("RGBA")
    slot_size = canvas_result["slot_size"]
    seed_slot = canvas_img.crop((0, 0, slot_size, slot_size))
    strip_img.paste(seed_slot, (0, 0), seed_slot)
    strip_img.save(strip_path, "PNG")

    return {
        "strip_path": strip_path,
        "strip_url": f"/{strip_path}",
        "canvas_path": canvas_path,
        "canvas_url": f"/{canvas_path}",
        "cols": canvas_result["cols"],
        "rows": canvas_result["rows"],
        "slot_size": canvas_result["slot_size"],
        "y_offset": canvas_result.get("y_offset", 0),
        "num_frames": num_frames,
    }


@app.get("/api/list-animation-candidates")
async def list_animation_candidates(
    character_name: str,
    animation_type: str = "idle",
):
    """List all generated animation strip candidates for a character + anim type."""
    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    anim_dir = char_dir / "animations" / animation_type
    candidates = []
    if anim_dir.exists():
        for f in sorted(anim_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
            if f.name.startswith("strip_raw_") and f.suffix.lower() == ".png":
                uid = f.stem.replace("strip_raw_", "")
                canvas_file = anim_dir / f"canvas_{uid}.png"
                rel_strip = str(f.relative_to(Path(".")))
                rel_canvas = str(canvas_file.relative_to(Path("."))) if canvas_file.exists() else None
                candidates.append({
                    "uid": uid,
                    "strip_path": rel_strip,
                    "strip_url": f"/{rel_strip}",
                    "canvas_url": f"/{rel_canvas}" if rel_canvas else None,
                })
    return {"candidates": candidates}


@app.get("/api/list-all-strips")
async def list_all_strips():
    """List all animation strip candidates across all characters and types."""
    strips = []
    if not OUTPUT_ROOT.exists():
        return {"strips": []}
    for char_dir in sorted(OUTPUT_ROOT.iterdir()):
        if not char_dir.is_dir():
            continue
        anim_root = char_dir / "animations"
        if not anim_root.exists():
            continue
        for anim_dir in sorted(anim_root.iterdir()):
            if not anim_dir.is_dir():
                continue
            for f in sorted(anim_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.name.startswith("strip_raw_") and f.suffix.lower() == ".png":
                    uid = f.stem.replace("strip_raw_", "")
                    canvas_file = anim_dir / f"canvas_{uid}.png"
                    rel_strip = str(f.relative_to(Path(".")))
                    rel_canvas = str(canvas_file.relative_to(Path("."))) if canvas_file.exists() else None
                    strips.append({
                        "character": char_dir.name,
                        "animation_type": anim_dir.name,
                        "uid": uid,
                        "strip_path": rel_strip,
                        "strip_url": f"/{rel_strip}",
                        "canvas_url": f"/{rel_canvas}" if rel_canvas else None,
                    })
    return {"strips": strips}


@app.post("/api/score-strip-grids")
async def score_strip_grids(
    strip_path: str = Form(...),
    num_frames: int = Form(4),
    cols: int = Form(2),
    rows: int = Form(2),
    slot_size: int = Form(512),
    y_offset: int = Form(0),
    size_min: int = Form(4),
    size_max: int = Form(128),
):
    """Score grid sizes on the first frame of an animation strip, like Stage 2."""
    loop = asyncio.get_event_loop()

    def _score():
        strip = Image.open(strip_path).convert("RGBA")
        # Extract first frame slot (accounting for y_offset in linear layout)
        slot = strip.crop((0, y_offset, slot_size, y_offset + slot_size))
        slot_rgb = slot.convert("RGB")
        arr = np.array(slot_rgb, dtype=np.float32)
        s = slot_size

        results = []
        for gs in range(size_min, size_max + 1):
            mse = grid_detector._reconstruction_error(np.array(slot_rgb), gs)
            results.append({"grid_size": gs, "score": round(float(mse), 2)})

        _mark_local_minima(results)
        return results

    results = await loop.run_in_executor(_executor, _score)
    return {"results": results}


@app.post("/api/normalize")
async def normalize(
    strip_path: str = Form(...),
    num_frames: int = Form(6),
    target_size: int = Form(64),
    cols: int = Form(3),
    rows: int = Form(2),
    slot_size: int = Form(341),
    y_offset: int = Form(0),
):
    """Normalize a raw animation strip into individual frames."""
    strip_dir = str(Path(strip_path).parent)
    uid = _unique_id()
    norm_dir = os.path.join(strip_dir, f"normalized_{uid}")

    result = sprite_normalizer.normalize_strip(
        strip_path,
        num_frames=num_frames,
        target_size=target_size,
        output_dir=norm_dir,
        cols=cols, rows=rows, slot_size=slot_size,
        y_offset=y_offset,
    )

    # Return frame data as data URLs for the frontend
    frame_data = []
    for i, fp in enumerate(result["frames"]):
        frame_data.append({
            "index": i,
            "path": fp,
            "url": f"/{fp}",
            "data_url": _image_to_data_url(fp),
        })

    return {
        "frames": frame_data,
        "spritesheet_path": result["spritesheet"],
        "spritesheet_url": f"/{result['spritesheet']}",
    }


@app.post("/api/export-gif")
async def export_gif_endpoint(
    frames_json: str = Form(...),
    animation_type: str = Form("idle"),
    frame_delay: int = Form(120),
    scale: int = Form(4),
    character_name: str = Form("character"),
):
    """Export aligned frames as an animated GIF."""
    import json
    frame_paths = json.loads(frames_json)

    char_dir = OUTPUT_ROOT / character_name.replace(" ", "_").lower()
    anim_dir = char_dir / "animations" / animation_type
    anim_dir.mkdir(parents=True, exist_ok=True)

    frames = [Image.open(p).convert("RGBA") for p in frame_paths]
    gif_path = str(anim_dir / f"{animation_type}.gif")

    gif_exporter.export_gif(
        frames, gif_path,
        frame_delay_ms=frame_delay,
        scale=scale,
    )

    return {
        "gif_path": gif_path,
        "gif_url": f"/{gif_path}",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)
