#!/usr/bin/env python3
"""
AI Refiner: Uses the OpenAI GPT Image 1.5 API to refine pixel art and generate animation strips.

Two main operations:
  1. refine_pixel_art   — takes a rough pixelated image, returns a cleaner pixel art version
  2. generate_animation — takes an edit canvas and returns a full multi-frame animation strip
"""

import os
import base64
import argparse
import urllib.request
from pathlib import Path
from openai import OpenAI


IMAGE_MODEL = "gpt-image-1.5"

REFINE_PROMPT_TEMPLATE = """
Refine this pixel art image. Keep the same subject, pose, and general colors.
Make it cleaner and more authentic-looking 16-bit pixel art:
- crisp, hard-edged pixel clusters (no anti-aliasing, no blur)
- stepped shading with a restrained palette
- clear, readable silhouette
- transparent background (RGBA PNG)
- same proportions and facing direction as the input
- no background scenery, no labels, no UI elements
Character description: {description}
""".strip()

ANIMATION_PROMPT_LINEAR = """
Intended use: candidate production spritesheet for a 2D side-view {game_style} {animation_type} animation.
Edit the provided transparent reference-canvas image into a horizontal strip of {num_frames} {animation_type} animation frames arranged LEFT TO RIGHT in a single row ACROSS THE MIDDLE of the canvas.
The leftmost frame (frame 1) is the exact approved seed frame and must remain unchanged as the starting frame.
Composition: keep the image fully transparent outside sprites. The {canvas_size}x{canvas_size} canvas contains {num_frames} frames side by side in one horizontal row across the vertical center of the canvas, each frame is {slot_size}x{slot_size} pixels. Frame 1 is on the far left, frame {num_frames} is on the far right. DO NOT stack frames vertically — all frames must be in a single horizontal line across the middle of the canvas. No overlap between frames, no extra characters, no labels, no UI.
Action: {action_description}
Keep body size, head size, and outfit proportions consistent across all {num_frames} frames.
Style: authentic 16-bit pixel art with CRISP, HARD EDGES. Every pixel must be a clean solid-color square block — NO anti-aliasing, NO soft edges, NO gradients, NO blending between pixels. Stepped shading only, restrained palette (16-32 colors max). This must look like real pixel art, not a filtered or smoothed illustration.
Constraints: no scenery, no floor, no glow, no atmospheric haze, no shadows outside the sprite contours, no collage, no poster layout, no blurry details, no sub-pixel rendering.
Keep wide transparent empty space outside the sprite in each frame.
IMPORTANT: The character MUST face RIGHT (toward the right side of the image) in ALL frames. This is standard for 2D game sprites.
Character description: {character_description}
""".strip()

ANIMATION_PROMPT_GRID = """
Intended use: candidate production spritesheet for a 2D side-view {game_style} {animation_type} animation.
Edit the provided transparent reference-canvas image into a {cols}x{rows} grid of {num_frames} {animation_type} animation frames.
The existing sprite in the top-left slot is the exact approved seed frame and must remain unchanged as the starting frame.
Composition: keep the image fully transparent outside sprites. The {canvas_size}x{canvas_size} canvas is divided into a grid of {cols} columns and {rows} rows, each cell is {slot_size}x{slot_size} pixels. Frames are read left-to-right, top-to-bottom (slot 1 is top-left, slot {num_frames} is the last filled cell). No overlap between cells, no extra characters, no labels, no UI.
Action: {action_description}
Keep body size, head size, and outfit proportions consistent across all {num_frames} frames.
Style: authentic 16-bit pixel art with CRISP, HARD EDGES. Every pixel must be a clean solid-color square block — NO anti-aliasing, NO soft edges, NO gradients, NO blending between pixels. Stepped shading only, restrained palette (16-32 colors max). This must look like real pixel art, not a filtered or smoothed illustration.
Constraints: no scenery, no floor, no glow, no atmospheric haze, no shadows outside the sprite contours, no collage, no poster layout, no blurry details, no sub-pixel rendering.
Keep wide transparent empty space outside the sprite in each cell.
IMPORTANT: The character MUST face RIGHT (toward the right side of the image) in ALL frames. This is standard for 2D game sprites.
Character description: {character_description}
""".strip()

ANIMATION_ACTIONS = {
    "idle":   "frames 1-{n} show a gentle idle breathing loop, slight up-down bob, subtle eye blink mid-sequence",
    "walk":   "frames 1-{n} show a full walk cycle, alternating legs and arms in smooth steps",
    "run":    "frames 1-{n} show a fast run cycle, exaggerated limb motion, slight forward lean",
    "hurt":   "frame 1 stays as the calm idle starting pose, frames 2-{n} show a hurt reaction: torso pulled back, head jolted, brief pain expression, slight recovery",
    "attack": "frame 1 stays as the idle starting pose, frames 2-{n} show a forward attack swing with follow-through and recovery",
    "jump":   "frames 1-{n} show a jump arc: crouch wind-up, launch, peak, landing impact",
    "death":  "frames 1-{n} show a death sequence: stagger, fall, final lying pose",
}


def _save_image_response(image_data, output_path: str):
    """Save an image from an OpenAI API response (handles both URL and b64_json)."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if hasattr(image_data, "b64_json") and image_data.b64_json:
        raw = base64.b64decode(image_data.b64_json)
        with open(output_path, "wb") as f:
            f.write(raw)
    elif hasattr(image_data, "url") and image_data.url:
        urllib.request.urlretrieve(image_data.url, output_path)
    else:
        raise ValueError("No image data in API response")


REFERENCE_PROMPT_TEMPLATE = """
Use the provided image ONLY as an ART STYLE reference — match its pixel art style, color palette approach, shading technique, and level of detail. Do NOT copy the character, pose, or subject from the reference image.
Create a COMPLETELY DIFFERENT character based on the description below, rendered in the same art style as the reference.
Output a single {frame_size}x{frame_size} pixel art character sprite.
TRANSPARENT BACKGROUND ONLY — no scenery, no floor, no labels, no UI, no text, no borders, no frames, no decoration.
Just the sprite on a completely clear transparent background, centered.
IMPORTANT: The character MUST face RIGHT (toward the right side of the image). This is standard for 2D game sprites.
This is a production game sprite at {frame_size}x{frame_size} pixels.
Character to create (NOT the reference image subject): {description}
""".strip()


def generate_from_reference(
    reference_path: str,
    output_path: str,
    character_description: str = "a game character",
    frame_size: int = 32,
) -> str:
    """
    Generate pixel art from a reference image using GPT Image 1.5 edit API.
    Extracts just the character, ignoring borders/backgrounds/text.
    """
    client = OpenAI()
    prompt = REFERENCE_PROMPT_TEMPLATE.format(
        frame_size=frame_size,
        description=character_description,
    )

    print(f"Generating {frame_size}x{frame_size} pixel art from reference: {reference_path}")
    with open(reference_path, "rb") as img_file:
        response = client.images.edit(
            model=IMAGE_MODEL,
            image=img_file,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

    _save_image_response(response.data[0], output_path)
    print(f"Generated pixel art from reference saved to {output_path}")
    return output_path


def refine_pixel_art(
    image_path: str,
    output_path: str,
    character_description: str = "a game character",
) -> str:
    """
    Call the GPT Image 1.5 API to refine a pixelated image into clean pixel art.
    """
    client = OpenAI()
    prompt = REFINE_PROMPT_TEMPLATE.format(description=character_description)

    print(f"Refining pixel art: {image_path}")
    with open(image_path, "rb") as img_file:
        response = client.images.edit(
            model=IMAGE_MODEL,
            image=img_file,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

    _save_image_response(response.data[0], output_path)
    print(f"Refined image saved to {output_path}")
    return output_path


def generate_pixel_art(
    output_path: str,
    character_description: str = "a game character",
    frame_size: int = 32,
    transparent: bool = False,
) -> str:
    """
    Generate pixel art from scratch using GPT Image 1.5.
    """
    client = OpenAI()

    if transparent:
        bg_instruction = "TRANSPARENT BACKGROUND ONLY — no scenery, no floor, no labels, no UI, no text, no glow."
    else:
        bg_instruction = "SOLID SINGLE-COLOR BACKGROUND (e.g. solid black, solid dark blue, or solid grey) — no scenery, no floor, no labels, no UI, no text, no glow. The background must be one uniform flat color."

    prompt = f"""Create a pixel art character sprite that FILLS THE ENTIRE IMAGE.
The character should be large and take up most of the canvas.
The image must look like {frame_size}x{frame_size} pixel art upscaled with nearest-neighbor scaling — each logical pixel is a uniform square block of solid color with perfectly hard edges. NO anti-aliasing, NO gradients, NO blending between blocks.
The entire image should be a clean grid of exactly {frame_size} columns and {frame_size} rows of crisp, solid-color square blocks.
Restrained color palette (16-32 colors max). Stepped shading only.
{bg_instruction}
IMPORTANT: The character MUST face RIGHT (toward the right side of the image).
Character: {character_description}"""

    print(f"Generating {frame_size}x{frame_size} pixel art: {character_description}")
    response = client.images.generate(
        model=IMAGE_MODEL,
        prompt=prompt,
        n=1,
        size="1024x1024",
        background="transparent" if transparent else "auto",
        quality="medium",
    )

    _save_image_response(response.data[0], output_path)
    print(f"Generated pixel art saved to {output_path}")
    return output_path


def generate_animation_strip(
    canvas_path: str,
    output_path: str,
    animation_type: str = "idle",
    character_description: str = "a game character",
    game_style: str = "platformer",
    num_frames: int = 4,
    canvas_size: int = 1024,
    cols: int | None = None,
    rows: int | None = None,
    slot_size: int | None = None,
    mask_path: str | None = None,
    layout: str = "grid",
) -> str:
    """
    Send the edit canvas to GPT Image 1.5 API and get back a full animation strip.

    Supports grid layouts for many frames (e.g. 4x4 = 16 frames).
    """
    client = OpenAI()

    # Auto-compute grid if not provided
    if cols is None or rows is None or slot_size is None:
        import canvas_builder
        cols, rows, slot_size = canvas_builder.compute_grid_layout(num_frames, canvas_size)

    action_template = ANIMATION_ACTIONS.get(
        animation_type,
        "frames 1-{n} show a smooth " + animation_type + " animation",
    )
    action_description = action_template.format(n=num_frames)

    # Pick prompt template based on layout
    template = ANIMATION_PROMPT_LINEAR if layout == "linear" else ANIMATION_PROMPT_GRID
    prompt = template.format(
        game_style=game_style,
        animation_type=animation_type,
        num_frames=num_frames,
        slot_size=slot_size,
        canvas_size=canvas_size,
        cols=cols,
        rows=rows,
        action_description=action_description,
        character_description=character_description,
    )

    print(f"Generating {animation_type} animation strip from: {canvas_path}")
    print(f"  {num_frames} frames @ {slot_size}x{slot_size} each")

    # Use mask to guide editing: transparent areas = edit, opaque areas = preserve.
    # Per OpenAI docs, the mask is a hint — the model may still modify masked areas.
    with open(canvas_path, "rb") as img_file:
        edit_kwargs = dict(
            model=IMAGE_MODEL,
            image=img_file,
            prompt=prompt,
            n=1,
            size="1024x1024",
        )
        if mask_path and os.path.exists(mask_path):
            edit_kwargs["mask"] = open(mask_path, "rb")
            print(f"  Using mask: {mask_path}")
        response = client.images.edit(**edit_kwargs)
        if "mask" in edit_kwargs and hasattr(edit_kwargs["mask"], "close"):
            edit_kwargs["mask"].close()

    _save_image_response(response.data[0], output_path)
    print(f"Raw animation strip saved to {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="AI pixel art refiner and animator (GPT Image 1.5)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # generate sub-command
    gen_p = subparsers.add_parser("generate", help="Generate pixel art from scratch")
    gen_p.add_argument("-o", "--output", default="generated_sprite.png", help="Output path")
    gen_p.add_argument("-d", "--description", default="a game character", help="Character description")
    gen_p.add_argument("-f", "--frame-size", type=int, default=32, help="Frame size (default: 32)")

    # refine sub-command
    refine_p = subparsers.add_parser("refine", help="Refine pixel art using GPT Image 1.5")
    refine_p.add_argument("input", help="Input pixel art image")
    refine_p.add_argument("-o", "--output", help="Output path")
    refine_p.add_argument("-d", "--description", default="a game character", help="Character description")

    # animate sub-command
    anim_p = subparsers.add_parser("animate", help="Generate animation strip from edit canvas")
    anim_p.add_argument("canvas", help="Edit canvas PNG (built by canvas_builder.py)")
    anim_p.add_argument("-o", "--output", help="Output path for the raw strip")
    anim_p.add_argument(
        "-t", "--type",
        choices=list(ANIMATION_ACTIONS.keys()),
        default="idle",
        help="Animation type",
    )
    anim_p.add_argument("-d", "--description", default="a game character", help="Character description")
    anim_p.add_argument("-g", "--game-style", default="platformer", help="Game style / genre")
    anim_p.add_argument("-n", "--num-frames", type=int, default=4, help="Number of animation frames")

    args = parser.parse_args()

    if args.command == "generate":
        generate_pixel_art(args.output, args.description, args.frame_size)

    elif args.command == "refine":
        base = os.path.splitext(os.path.basename(args.input))[0]
        output = args.output or f"{os.path.dirname(args.input) or '.'}/{base}_refined.png"
        refine_pixel_art(args.input, output, args.description)

    elif args.command == "animate":
        base = os.path.splitext(os.path.basename(args.canvas))[0]
        output = args.output or f"{os.path.dirname(args.canvas) or '.'}/{base}_strip_raw.png"
        generate_animation_strip(
            args.canvas, output,
            animation_type=args.type,
            character_description=args.description,
            game_style=args.game_style,
            num_frames=args.num_frames,
        )


if __name__ == "__main__":
    main()
