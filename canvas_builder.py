#!/usr/bin/env python3
"""
Canvas Builder: Prepares a seed pixel art frame for the GPT Image Edit API.

Upscales the seed frame using nearest-neighbor interpolation (preserves sharp
pixel edges) and places it into a transparent 1024x1024 canvas with reserved
frame slots arranged in a grid — ready to be sent to the image edit API.

Grid layout allows many frames (8, 12, 16+) while keeping each slot large
enough for the AI to generate quality sprites (256x256 minimum).
"""

import os
import math
import argparse
from PIL import Image


def compute_grid_layout(num_frames: int, canvas_size: int = 1024, min_slot: int = 128):
    """
    Compute optimal grid layout (cols, rows, slot_size) for a given frame count.

    Strategy: maximize slot size while fitting all frames in the canvas.
    Prefers wider grids (more cols than rows) for spritesheet convention.

    Returns:
        (cols, rows, slot_size)
    """
    best = None
    for cols in range(1, num_frames + 1):
        rows = math.ceil(num_frames / cols)
        slot_w = canvas_size // cols
        slot_h = canvas_size // rows
        slot_size = min(slot_w, slot_h)
        if slot_size < min_slot:
            continue
        # Prefer layouts that maximize slot size, then prefer wider (more cols)
        if best is None or slot_size > best[2] or (slot_size == best[2] and cols > best[0]):
            best = (cols, rows, slot_size)

    if best is None:
        # Fallback: just pack them, even if slots are small
        cols = math.ceil(math.sqrt(num_frames))
        rows = math.ceil(num_frames / cols)
        slot_size = canvas_size // max(cols, rows)
        best = (cols, rows, slot_size)

    return best


def crop_to_sprite(image: Image.Image, padding: int = 2) -> Image.Image:
    """
    Crop image to the bounding box of the sprite.
    Handles both transparent backgrounds (alpha-based) and solid backgrounds
    (detects the most common corner color as background).
    """
    import numpy as np
    rgba = image.convert("RGBA")
    arr = np.array(rgba)
    alpha = arr[:, :, 3]

    has_transparency = np.any(alpha < 240)

    if has_transparency:
        mask = alpha > 10
    else:
        corners = [
            arr[0, 0, :3], arr[0, -1, :3],
            arr[-1, 0, :3], arr[-1, -1, :3],
        ]
        bg_color = np.median(corners, axis=0).astype(np.uint8)
        diff = np.sqrt(np.sum((arr[:, :, :3].astype(float) - bg_color.astype(float)) ** 2, axis=2))
        mask = diff > 30

        bg_mask = ~mask
        arr[bg_mask, 3] = 0
        rgba = Image.fromarray(arr)

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return rgba

    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - np.argmax(cols[::-1]))

    top = max(0, top - padding)
    bottom = min(arr.shape[0], bottom + padding)
    left = max(0, left - padding)
    right = min(arr.shape[1], right + padding)

    return rgba.crop((left, top, right, bottom))


def upscale_nearest_fit(image: Image.Image, target_size: int) -> Image.Image:
    """
    Upscale image with nearest-neighbor to fit inside target_size x target_size,
    preserving aspect ratio. Centers on transparent canvas.
    """
    w, h = image.size
    scale = target_size / max(w, h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))
    scaled = image.resize((new_w, new_h), Image.NEAREST)

    result = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))
    x = (target_size - new_w) // 2
    y = (target_size - new_h) // 2
    result.paste(scaled, (x, y), scaled)
    return result


def build_edit_canvas(
    frame_path: str,
    output_path: str,
    num_frames: int = 4,
    canvas_size: int = 1024,
    force_cols: int | None = None,
    draw_slots: bool = True,
    prefill_slots: bool = False,
) -> dict:
    """
    Upscale seed frame and place it in slot (0,0) of a grid-layout transparent canvas.

    If force_cols is set, uses that many columns (e.g. force_cols=num_frames for linear).
    draw_slots: draw grey guide boxes around each frame slot
    prefill_slots: copy the seed sprite into all slots (AI modifies each)

    Returns:
        Dict with canvas_path, mask_path, cols, rows, slot_size, y_offset
    """
    if force_cols is not None:
        cols = force_cols
        rows = math.ceil(num_frames / cols)
        slot_size = min(canvas_size // cols, canvas_size // rows)
    else:
        cols, rows, slot_size = compute_grid_layout(num_frames, canvas_size)

    # Load seed frame, ensure RGBA
    seed = Image.open(frame_path).convert("RGBA")

    # Crop to sprite bounds (remove excess transparency)
    cropped = crop_to_sprite(seed)

    # Upscale with nearest-neighbor to fit the frame slot
    upscaled = upscale_nearest_fit(cropped, slot_size)

    # For linear layout, center the strip vertically so the AI
    # has no vertical space to create a grid
    if rows == 1 and cols > 1:
        y_offset = (canvas_size - slot_size) // 2
    else:
        y_offset = 0

    # Create transparent canvas
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    # Draw slot guide boxes (grey outlines showing where each frame goes)
    if draw_slots:
        from PIL import ImageDraw
        draw = ImageDraw.Draw(canvas)
        guide_color = (80, 80, 80, 120)  # semi-transparent grey
        for frame_idx in range(num_frames):
            col = frame_idx % cols
            row = frame_idx // cols
            x0 = col * slot_size
            y0 = row * slot_size + y_offset
            # Draw rectangle outline (2px border)
            for t in range(2):
                draw.rectangle(
                    [x0 + t, y0 + t, x0 + slot_size - 1 - t, y0 + slot_size - 1 - t],
                    outline=guide_color,
                )

    # Pre-fill all slots with the seed sprite if requested
    if prefill_slots:
        for frame_idx in range(num_frames):
            col = frame_idx % cols
            row = frame_idx // cols
            x0 = col * slot_size
            y0 = row * slot_size + y_offset
            canvas.paste(upscaled, (x0, y0), upscaled)
    else:
        # Just place seed in slot 0
        canvas.paste(upscaled, (0, y_offset), upscaled)

    # Save
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    canvas.save(output_path, "PNG")

    # Create mask for OpenAI images.edit API:
    # The mask must be the same size as the image.
    # Fully transparent areas (alpha=0) = regions to EDIT/GENERATE
    # Opaque areas (alpha=255) = regions to PRESERVE
    mask = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    slot_block = Image.new("RGBA", (slot_size, slot_size), (0, 0, 0, 255))
    mask.paste(slot_block, (0, y_offset))

    mask_path = output_path.replace(".png", "_mask.png")
    mask.save(mask_path, "PNG")

    print(f"Canvas saved to {output_path}")
    print(f"Mask saved to {mask_path}")
    print(f"  Canvas: {canvas_size}x{canvas_size}")
    print(f"  Grid: {cols} cols x {rows} rows = {cols * rows} slots ({num_frames} used)")
    print(f"  Slot size: {slot_size}x{slot_size}")

    return {
        "canvas_path": output_path,
        "mask_path": mask_path,
        "cols": cols,
        "rows": rows,
        "slot_size": slot_size,
        "y_offset": y_offset,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build a 1024x1024 edit canvas from a seed pixel art frame"
    )
    parser.add_argument("frame", help="Path to the seed pixel art frame")
    parser.add_argument("-o", "--output", help="Output path for the canvas PNG")
    parser.add_argument(
        "-n", "--num-frames", type=int, default=4, help="Number of animation frame slots"
    )
    parser.add_argument(
        "-c", "--canvas-size", type=int, default=1024, help="Canvas size in pixels"
    )
    args = parser.parse_args()

    base = os.path.splitext(os.path.basename(args.frame))[0]
    output = args.output or f"{os.path.dirname(args.frame) or '.'}/{base}_canvas.png"

    build_edit_canvas(
        args.frame,
        output,
        num_frames=args.num_frames,
        canvas_size=args.canvas_size,
    )


if __name__ == "__main__":
    main()
