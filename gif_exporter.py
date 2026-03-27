#!/usr/bin/env python3
"""
GIF Exporter: Converts normalized sprite frames into an animated GIF.

Can take either:
  - A directory of frame PNGs (frame_01.png, frame_02.png, ...)
  - A normalized spritesheet PNG + frame count
"""

import os
import glob
import argparse
from PIL import Image


def frames_from_dir(frames_dir: str, pattern: str = "*_frame_*.png") -> list[Image.Image]:
    """Load all frame PNGs from a directory, sorted by filename."""
    paths = sorted(glob.glob(os.path.join(frames_dir, pattern)))
    if not paths:
        # Fall back to any PNG files sorted alphabetically
        paths = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
    frames = [Image.open(p).convert("RGBA") for p in paths]
    print(f"Loaded {len(frames)} frames from {frames_dir}")
    return frames, paths


def frames_from_sheet(
    sheet_path: str, num_frames: int, target_size: int | None = None
) -> list[Image.Image]:
    """Split a horizontal spritesheet into individual frames."""
    sheet = Image.open(sheet_path).convert("RGBA")
    w, h = sheet.size
    frame_w = w // num_frames
    frames = []
    for i in range(num_frames):
        frame = sheet.crop((i * frame_w, 0, (i + 1) * frame_w, h))
        if target_size and frame.size != (target_size, target_size):
            frame = frame.resize((target_size, target_size), Image.NEAREST)
        frames.append(frame)
    print(f"Split {num_frames} frames from sheet: {sheet_path}")
    return frames


def rgba_to_p_with_transparency(frame: Image.Image) -> Image.Image:
    """
    Convert RGBA frame to palette mode with transparency for GIF export.
    GIF only supports palette (P) mode with one transparent color index.
    """
    # Convert to RGBA to ensure alpha is present
    frame = frame.convert("RGBA")

    # Create palette image with transparency
    converted = frame.convert("P", palette=Image.ADAPTIVE, colors=255)

    # Map fully transparent pixels to index 255
    arr = frame.split()[3]  # alpha channel
    mask = Image.eval(arr, lambda a: 255 if a < 128 else 0)
    converted.paste(255, mask)
    converted.info["transparency"] = 255
    return converted


def export_gif(
    frames: list[Image.Image],
    output_path: str,
    frame_delay_ms: int = 100,
    loop: int = 0,
    scale: int = 1,
    background_color: tuple | None = None,
) -> str:
    """
    Compose frames into an animated GIF.

    Args:
        frames: List of PIL Images (RGBA)
        output_path: Output .gif path
        frame_delay_ms: Milliseconds per frame (default 100 = 10 fps)
        loop: Loop count (0 = infinite)
        scale: Upscale factor applied with nearest-neighbor (for visibility)
        background_color: Optional RGB tuple for background (default transparent)

    Returns:
        Path to saved GIF
    """
    if not frames:
        raise ValueError("No frames provided")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    processed = []
    for frame in frames:
        img = frame.convert("RGBA")

        if scale > 1:
            w, h = img.size
            img = img.resize((w * scale, h * scale), Image.NEAREST)

        if background_color:
            bg = Image.new("RGBA", img.size, background_color + (255,))
            bg.paste(img, mask=img.split()[3])
            img = bg.convert("RGB").convert("P", palette=Image.ADAPTIVE)
        else:
            img = rgba_to_p_with_transparency(img)

        processed.append(img)

    duration = frame_delay_ms  # PIL uses ms for GIF duration

    processed[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=processed[1:],
        duration=duration,
        loop=loop,
        transparency=255 if background_color is None else None,
        disposal=2,  # restore to background between frames
    )

    print(f"Animated GIF saved to {output_path}")
    print(f"  Frames: {len(frames)}, delay: {frame_delay_ms}ms, scale: {scale}x")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Export sprite frames as animated GIF")

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--frames-dir", help="Directory containing frame PNGs")
    source.add_argument("--sheet", help="Normalized spritesheet PNG")

    parser.add_argument("-n", "--num-frames", type=int, default=4, help="Frame count (for --sheet mode)")
    parser.add_argument("-o", "--output", help="Output GIF path")
    parser.add_argument("-d", "--delay", type=int, default=100, help="Frame delay in ms (default 100)")
    parser.add_argument("-s", "--scale", type=int, default=4, help="Upscale factor for visibility (default 4)")
    parser.add_argument("--no-loop", action="store_true", help="Play GIF once then stop")
    parser.add_argument(
        "--bg",
        nargs=3,
        type=int,
        metavar=("R", "G", "B"),
        help="Background color RGB (default: transparent)",
    )

    args = parser.parse_args()

    if args.frames_dir:
        frames, paths = frames_from_dir(args.frames_dir)
        default_base = os.path.basename(args.frames_dir.rstrip("/"))
        default_out = os.path.join(args.frames_dir, f"{default_base}_animation.gif")
    else:
        frames = frames_from_sheet(args.sheet, args.num_frames)
        base = os.path.splitext(os.path.basename(args.sheet))[0]
        default_out = os.path.join(os.path.dirname(args.sheet) or ".", f"{base}_animation.gif")

    output = args.output or default_out
    bg = tuple(args.bg) if args.bg else None

    export_gif(
        frames,
        output,
        frame_delay_ms=args.delay,
        loop=0 if not args.no_loop else 1,
        scale=args.scale,
        background_color=bg,
    )


if __name__ == "__main__":
    main()
