#!/usr/bin/env python3
"""
Sprite Normalizer: Processes a raw AI-generated animation strip into game-ready frames.

Key steps (from chongdashu's workflow):
  1. Split the strip into equal frame slots
  2. For each frame, detect the ACTUAL sprite bounding box (robust alpha threshold)
  3. Find the character body anchor (center-of-mass of opaque pixels) for alignment
  4. Compute ONE shared scale across all frames (prevents per-frame squashing)
  5. Align all frames to a common anchor point (feet/bottom-center)
  6. Scale + pad each frame to a fixed target size
  7. Optionally lock frame 0 to the exact seed sprite
"""

import os
import argparse
import numpy as np
from PIL import Image


def remove_background(frame: Image.Image, bg_threshold: float = 30.0) -> Image.Image:
    """
    Detect and remove the background from a frame.

    If the frame has real transparency (alpha < 255 anywhere), uses alpha.
    Otherwise, detects background color from corners and makes it transparent.

    Returns:
        RGBA image with background removed
    """
    arr = np.array(frame.convert("RGBA")).copy()
    alpha = arr[:, :, 3]

    if np.all(alpha == 255):
        # Fully opaque — detect background from corners
        h, w = arr.shape[:2]
        corner_size = max(4, min(h, w) // 10)
        corners = np.concatenate([
            arr[:corner_size, :corner_size, :3].reshape(-1, 3),
            arr[:corner_size, -corner_size:, :3].reshape(-1, 3),
            arr[-corner_size:, :corner_size, :3].reshape(-1, 3),
            arr[-corner_size:, -corner_size:, :3].reshape(-1, 3),
        ])
        bg_color = np.median(corners, axis=0).astype(float)

        # Color distance from background
        diff = np.sqrt(np.sum((arr[:, :, :3].astype(float) - bg_color) ** 2, axis=2))
        bg_mask = diff < bg_threshold
        arr[bg_mask, 3] = 0

    return Image.fromarray(arr)


def detect_sprite_bbox(frame: Image.Image, alpha_threshold: int = 32) -> tuple[int, int, int, int] | None:
    """
    Find the bounding box of the sprite in an RGBA frame.

    Expects the frame to already have background removed (transparent).

    Returns:
        (left, top, right, bottom) or None if frame is fully transparent
    """
    arr = np.array(frame.convert("RGBA"))
    alpha = arr[:, :, 3]

    mask = alpha > alpha_threshold

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any():
        return None

    top = int(np.argmax(rows))
    bottom = int(len(rows) - np.argmax(rows[::-1]))
    left = int(np.argmax(cols))
    right = int(len(cols) - np.argmax(cols[::-1]))
    return (left, top, right, bottom)


def find_anchor_point(frame: Image.Image, bbox: tuple[int, int, int, int], alpha_threshold: int = 32) -> tuple[int, int]:
    """
    Find the anchor point for alignment: bottom-center of the character body.

    This is the character's "feet" position — the horizontal center of mass
    and the bottom of the sprite. Using this as anchor keeps characters
    grounded in the same spot across frames even when poses differ in height.

    Returns:
        (anchor_x, anchor_y) relative to the frame
    """
    arr = np.array(frame.convert("RGBA"))
    alpha = arr[:, :, 3]
    mask = alpha > alpha_threshold

    left, top, right, bottom = bbox

    # Horizontal center of mass (weighted by alpha)
    col_weights = np.sum(alpha[top:bottom, left:right].astype(float), axis=0)
    if col_weights.sum() > 0:
        col_indices = np.arange(left, right)
        center_x = int(np.average(col_indices, weights=col_weights))
    else:
        center_x = (left + right) // 2

    # Vertical anchor = bottom of sprite (feet)
    anchor_y = bottom

    return (center_x, anchor_y)


def compute_shared_scale(
    bboxes: list[tuple[int, int, int, int] | None],
    target_size: int,
    padding: int = 4,
) -> float:
    """
    Compute one scale factor for the entire strip based on the largest sprite.

    Using a single shared scale (not per-frame) prevents height/width
    inconsistencies when one pose is naturally taller than another.
    """
    max_w = max_h = 0
    for bbox in bboxes:
        if bbox is None:
            continue
        left, top, right, bottom = bbox
        max_w = max(max_w, right - left)
        max_h = max(max_h, bottom - top)

    if max_w == 0 or max_h == 0:
        return 1.0

    usable = target_size - padding * 2
    scale = usable / max(max_w, max_h)
    return scale


def normalize_frame_anchored(
    frame: Image.Image,
    bbox: tuple[int, int, int, int] | None,
    anchor: tuple[int, int],
    scale: float,
    target_size: int,
    target_anchor_x: int,
    target_anchor_y: int,
) -> Image.Image:
    """
    Crop sprite to bbox, scale by shared factor, then place in target frame
    aligned by the anchor point (bottom-center of character).

    This ensures all frames are aligned at the feet, regardless of how
    wide or tall each individual pose is.
    """
    out = Image.new("RGBA", (target_size, target_size), (0, 0, 0, 0))

    if bbox is None:
        return out

    left, top, right, bottom = bbox
    cropped = frame.crop((left, top, right, bottom))

    new_w = max(1, round((right - left) * scale))
    new_h = max(1, round((bottom - top) * scale))
    scaled = cropped.resize((new_w, new_h), Image.NEAREST)

    # Position: align the anchor point
    # The anchor in the original frame is at (anchor_x, anchor_y)
    # After cropping, it's at (anchor_x - left, anchor_y - top)
    # After scaling, it's at ((anchor_x - left) * scale, (anchor_y - top) * scale)
    scaled_anchor_x = round((anchor[0] - left) * scale)
    scaled_anchor_y = round((anchor[1] - top) * scale)

    # Place so that the scaled anchor lands at the target anchor
    paste_x = target_anchor_x - scaled_anchor_x
    paste_y = target_anchor_y - scaled_anchor_y

    # Clamp to canvas bounds
    paste_x = max(0, min(target_size - new_w, paste_x))
    paste_y = max(0, min(target_size - new_h, paste_y))

    out.paste(scaled, (paste_x, paste_y), scaled)
    return out


def normalize_strip(
    strip_path: str,
    num_frames: int = 4,
    target_size: int = 64,
    output_dir: str | None = None,
    seed_frame_path: str | None = None,
    canvas_size: int = 1024,
    padding: int = 4,
    cols: int | None = None,
    rows: int | None = None,
    slot_size: int | None = None,
    y_offset: int = 0,
) -> dict:
    """
    Full normalization pipeline for a raw AI animation strip/grid.

    Supports both single-row strips and multi-row grids.
    Frames are read left-to-right, top-to-bottom.
    All frames are aligned by their bottom-center anchor point (feet).
    """
    if output_dir is None:
        output_dir = os.path.dirname(strip_path) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Auto-compute grid layout if not provided
    if cols is None or rows is None or slot_size is None:
        import canvas_builder
        cols, rows, slot_size = canvas_builder.compute_grid_layout(num_frames, canvas_size)

    base = os.path.splitext(os.path.basename(strip_path))[0]
    strip = Image.open(strip_path).convert("RGBA")
    strip_w, strip_h = strip.size

    print(f"Strip size: {strip_w}x{strip_h}, grid: {cols}x{rows}, slot: {slot_size}x{slot_size}")

    # 1. Extract raw frame slots (left-to-right, top-to-bottom)
    raw_frames: list[Image.Image] = []
    for frame_idx in range(num_frames):
        col = frame_idx % cols
        row = frame_idx // cols
        x0 = col * slot_size
        y0 = row * slot_size + y_offset
        slot = strip.crop((x0, y0, x0 + slot_size, y0 + slot_size))
        raw_frames.append(slot)

    # 1b. Skip background removal — keep frames as-is (white bg or transparent)
    # The downstream pixelation and alignment should work on the raw frames.

    # 2. Detect bounding boxes with proper alpha threshold
    bboxes = [detect_sprite_bbox(f) for f in raw_frames]
    print("Sprite bounding boxes:")
    for i, bb in enumerate(bboxes):
        if bb:
            w, h = bb[2] - bb[0], bb[3] - bb[1]
            print(f"  Frame {i+1}: {bb}  ({w}x{h})")
        else:
            print(f"  Frame {i+1}: empty")

    # 3. Find anchor points (bottom-center of each sprite)
    anchors = []
    for frame, bbox in zip(raw_frames, bboxes):
        if bbox:
            anchors.append(find_anchor_point(frame, bbox))
        else:
            anchors.append((slot_size // 2, slot_size // 2))

    print("Anchor points (center-x, bottom-y):")
    for i, a in enumerate(anchors):
        print(f"  Frame {i+1}: {a}")

    # 4. Shared scale
    scale = compute_shared_scale(bboxes, target_size, padding)
    print(f"Shared scale factor: {scale:.4f}")

    # Target anchor position: horizontal center, near bottom with padding
    target_anchor_x = target_size // 2
    target_anchor_y = target_size - padding

    # 5. Normalize each frame with anchor alignment
    normalized: list[Image.Image] = []
    frame_paths: list[str] = []
    for i, (frame, bbox, anchor) in enumerate(zip(raw_frames, bboxes, anchors)):
        norm = normalize_frame_anchored(
            frame, bbox, anchor, scale, target_size,
            target_anchor_x, target_anchor_y,
        )
        normalized.append(norm)

        frame_path = os.path.join(output_dir, f"{base}_frame_{i+1:02d}.png")
        norm.save(frame_path, "PNG")
        frame_paths.append(frame_path)
        print(f"Saved frame {i+1}: {frame_path}")

    # 6. Optionally replace frame 0 with the exact seed sprite
    if seed_frame_path and os.path.exists(seed_frame_path):
        seed = Image.open(seed_frame_path).convert("RGBA")
        seed_resized = seed.resize((target_size, target_size), Image.NEAREST)
        normalized[0] = seed_resized
        normalized[0].save(frame_paths[0], "PNG")
        print(f"Locked frame 01 to seed sprite: {seed_frame_path}")

    # 7. Build normalized spritesheet
    sheet_w = target_size * num_frames
    sheet = Image.new("RGBA", (sheet_w, target_size), (0, 0, 0, 0))
    for i, frame in enumerate(normalized):
        sheet.paste(frame, (i * target_size, 0), frame)

    sheet_path = os.path.join(output_dir, f"{base}_normalized_sheet.png")
    sheet.save(sheet_path, "PNG")
    print(f"Saved normalized spritesheet: {sheet_path}")

    return {
        "frames": frame_paths,
        "spritesheet": sheet_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Normalize a raw AI animation strip into game frames")
    parser.add_argument("strip", help="Raw animation strip PNG from AI")
    parser.add_argument("-n", "--num-frames", type=int, default=4, help="Number of frames in strip")
    parser.add_argument("-t", "--target-size", type=int, default=64, help="Output frame size in pixels")
    parser.add_argument("-o", "--output-dir", help="Output directory")
    parser.add_argument("-s", "--seed", help="Seed frame to lock frame 01 to (exact shipped sprite)")
    parser.add_argument("-p", "--padding", type=int, default=4, help="Padding around sprite in frame")

    args = parser.parse_args()

    result = normalize_strip(
        args.strip,
        num_frames=args.num_frames,
        target_size=args.target_size,
        output_dir=args.output_dir,
        seed_frame_path=args.seed,
        padding=args.padding,
    )

    print("\nSummary:")
    for path in result["frames"]:
        print(f"  {path}")
    print(f"  Spritesheet: {result['spritesheet']}")


if __name__ == "__main__":
    main()
