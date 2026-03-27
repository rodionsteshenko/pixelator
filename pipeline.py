#!/usr/bin/env python3
"""
Pixel Art → Animated GIF Pipeline

Full interactive workflow:
  0. Setup   — pick frame size preset, select/provide input image, pick reference
  1. Pixelate — extract pixel art from input image (multiple variants)
  2. AI Refine — use GPT Image API to clean up the best variant (optional)
  3. Confirm — user picks the best pixel art frame
  4. Build Canvas — upscale seed frame into 1024x1024 edit canvas
  5. Generate Animation — GPT Image API generates a full animation strip
  6. Normalize — shared-scale normalization into fixed-size frames
  7. Export GIF — compose frames into animated GIF
"""

import os
import sys
import glob
import random
import argparse
import subprocess
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, Confirm, IntPrompt
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import print as rprint
from PIL import Image

import canvas_builder
import ai_refiner
import sprite_normalizer
import gif_exporter
import pixelator_grid

console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANIMATION_TYPES = ["idle", "walk", "run", "hurt", "attack", "jump", "death"]

FRAME_SIZE_PRESETS = {
    "16x16":  16,
    "32x32":  32,
    "48x48":  48,
    "64x64":  64,
    "96x96":  96,
    "128x128": 128,
}

EXAMPLES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def phase_header(n: int, title: str):
    console.print(f"\n[bold cyan]── Phase {n}: {title} ──[/bold cyan]")


def show_image(path: str):
    """Try to open the image with the system viewer for user review."""
    try:
        if sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", path], check=False)
        elif sys.platform == "win32":
            os.startfile(path)
    except Exception:
        pass


def confirm_or_quit(message: str) -> bool:
    return Confirm.ask(f"[yellow]{message}[/yellow]")


def find_example_images() -> list[str]:
    """Scan the examples/ directory for image files."""
    if not os.path.isdir(EXAMPLES_DIR):
        return []
    found = []
    for f in sorted(os.listdir(EXAMPLES_DIR)):
        ext = os.path.splitext(f)[1].lower()
        if ext in IMAGE_EXTENSIONS:
            found.append(os.path.join(EXAMPLES_DIR, f))
    return found


# ---------------------------------------------------------------------------
# Phase 0: Interactive Setup
# ---------------------------------------------------------------------------

def interactive_setup(args) -> argparse.Namespace:
    """
    Prompt user for frame size, input image, reference image, and animation type
    if they weren't provided via CLI flags.
    """
    console.print(Panel.fit(
        "[bold]Pixel Art → Animated GIF Pipeline[/bold]\n"
        "Interactive setup — configure your run",
        title="[bold magenta]Pixelator[/bold magenta]",
    ))

    # --- Frame size preset ---
    if args.frame_size is None:
        console.print("\n[bold]Select frame size:[/bold]")
        presets = list(FRAME_SIZE_PRESETS.items())
        for i, (label, size) in enumerate(presets, 1):
            marker = " [dim](default)[/dim]" if size == 64 else ""
            console.print(f"  [cyan]{i}[/cyan]. {label}{marker}")

        choice = IntPrompt.ask(
            "Frame size",
            default=4,  # 64x64
        )
        idx = max(0, min(choice - 1, len(presets) - 1))
        args.frame_size = presets[idx][1]

    console.print(f"  Frame size: [green]{args.frame_size}x{args.frame_size}[/green]")

    # --- Input image ---
    if args.input is None:
        examples = find_example_images()

        console.print("\n[bold]Select input image:[/bold]")
        console.print(f"  [cyan]1[/cyan]. Enter a file path")
        if examples:
            console.print(f"  [cyan]2[/cyan]. Pick from examples/ folder ({len(examples)} images)")
            console.print(f"  [cyan]3[/cyan]. AI choice (random from examples)")

        choice = IntPrompt.ask("Selection", default=1)

        if choice == 1:
            args.input = Prompt.ask("Image path")
        elif choice == 2 and examples:
            console.print("\n[bold]Example images:[/bold]")
            for i, path in enumerate(examples, 1):
                name = os.path.basename(path)
                try:
                    img = Image.open(path)
                    dims = f"{img.size[0]}x{img.size[1]}"
                except Exception:
                    dims = "?"
                console.print(f"  [cyan]{i}[/cyan]. {name} ({dims})")
            idx = IntPrompt.ask("Pick an example", default=1)
            args.input = examples[max(0, min(idx - 1, len(examples) - 1))]
        elif choice == 3 and examples:
            args.input = random.choice(examples)
            console.print(f"  AI chose: [green]{os.path.basename(args.input)}[/green]")
        else:
            args.input = Prompt.ask("Image path")

    console.print(f"  Input: [green]{args.input}[/green]")

    # --- Reference image (optional — used as style guide for AI) ---
    if args.reference is None:
        examples = find_example_images()
        if examples:
            console.print("\n[bold]Use a reference image for AI style guidance?[/bold]")
            console.print(f"  [cyan]0[/cyan]. No reference")
            for i, path in enumerate(examples, 1):
                console.print(f"  [cyan]{i}[/cyan]. {os.path.basename(path)}")
            console.print(f"  [cyan]{len(examples)+1}[/cyan]. AI choice (random)")
            console.print(f"  [cyan]{len(examples)+2}[/cyan]. Enter a custom path")

            choice = IntPrompt.ask("Reference", default=0)
            if choice == 0:
                args.reference = None
            elif 1 <= choice <= len(examples):
                args.reference = examples[choice - 1]
            elif choice == len(examples) + 1:
                args.reference = random.choice(examples)
                console.print(f"  AI chose: [green]{os.path.basename(args.reference)}[/green]")
            else:
                path = Prompt.ask("Reference image path")
                args.reference = path if path.strip() else None

        if args.reference:
            console.print(f"  Reference: [green]{args.reference}[/green]")

    # --- Animation type ---
    if args.animation is None:
        console.print("\n[bold]Select animation type:[/bold]")
        for i, atype in enumerate(ANIMATION_TYPES, 1):
            console.print(f"  [cyan]{i}[/cyan]. {atype}")
        choice = IntPrompt.ask("Animation", default=1)
        args.animation = ANIMATION_TYPES[max(0, min(choice - 1, len(ANIMATION_TYPES) - 1))]

    console.print(f"  Animation: [green]{args.animation}[/green]")

    # --- Character name ---
    if not hasattr(args, 'character_name') or args.character_name is None:
        if args.input:
            default_name = os.path.splitext(os.path.basename(args.input))[0]
        else:
            default_name = "character"
        args.character_name = Prompt.ask(
            "\n[bold]Character name[/bold] (used for output directory)",
            default=default_name,
        )

    # --- Character description ---
    if args.description == "a game character":
        desc = Prompt.ask(
            "\n[bold]Character description[/bold] (for AI prompts)",
            default="a game character",
        )
        args.description = desc

    return args


# ---------------------------------------------------------------------------
# Phase 1: Pixelate — generate multiple variants
# ---------------------------------------------------------------------------

def phase_pixelate(image_path: str, output_dir: str, frame_size: int) -> list[str]:
    phase_header(1, "Pixelate — generating variants")

    variants_dir = os.path.join(output_dir, "variants")
    os.makedirs(variants_dir, exist_ok=True)

    # Adapt spacing based on target frame size
    # Smaller frame sizes need finer grids, larger ones can be coarser
    if frame_size <= 16:
        spacings = [4, 6, 8]
    elif frame_size <= 32:
        spacings = [6, 8, 12]
    elif frame_size <= 64:
        spacings = [8, 12, 16]
    else:
        spacings = [12, 16, 24]

    color_counts = [16, 32, None]

    # Build parameter grid
    param_sets = []
    for spacing in spacings:
        for colors in color_counts:
            label = f"sp{spacing}_{'full' if colors is None else f'{colors}col'}"
            param_sets.append((label, spacing, colors, "euclidean"))

    outputs = []
    table = Table(title=f"Pixel Art Variants (target: {frame_size}x{frame_size})", show_lines=True)
    table.add_column("#", style="bold", width=4)
    table.add_column("Label")
    table.add_column("Spacing")
    table.add_column("Colors")
    table.add_column("Output")

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console) as progress:
        task = progress.add_task("Generating variants...", total=len(param_sets))
        for i, (label, spacing, colors, algo) in enumerate(param_sets, 1):
            var_dir = os.path.join(variants_dir, label)
            os.makedirs(var_dir, exist_ok=True)

            progress.update(task, description=f"Variant {i}/{len(param_sets)}: {label}")
            try:
                result = pixelator_grid.process_image(
                    image_path,
                    output_dir=var_dir,
                    row_spacing=spacing,
                    col_spacing=spacing,
                    max_colors=colors,
                    algorithm=algo,
                )
                out_path = result["one_to_one"]
                outputs.append(out_path)
                table.add_row(
                    str(i), label, str(spacing),
                    str(colors) if colors else "all", out_path,
                )
            except Exception as e:
                console.print(f"[red]  Variant {label} failed: {e}[/red]")
            progress.advance(task)

    console.print(table)
    console.print(f"\n[dim]Opening variants directory for review: {variants_dir}[/dim]")
    show_image(variants_dir)
    return outputs


# ---------------------------------------------------------------------------
# Phase 2: AI Refine (optional)
# ---------------------------------------------------------------------------

def phase_ai_refine(
    candidate_path: str,
    output_dir: str,
    character_description: str,
    reference_path: str | None = None,
    num_iterations: int = 1,
) -> str:
    phase_header(2, "AI Refine — improving with GPT Image API")

    if reference_path:
        console.print(f"  Using style reference: [cyan]{os.path.basename(reference_path)}[/cyan]")

    current = candidate_path
    for i in range(num_iterations):
        console.print(f"  Refinement pass {i+1}/{num_iterations}...")
        base = os.path.splitext(os.path.basename(current))[0]
        out = os.path.join(output_dir, f"{base}_refined_pass{i+1}.png")

        # Augment description with reference info if available
        desc = character_description
        if reference_path:
            ref_name = os.path.splitext(os.path.basename(reference_path))[0]
            desc += f". Match the pixel art style of the reference image '{ref_name}'."

        with Progress(SpinnerColumn(), TextColumn("Calling GPT Image API..."), console=console) as p:
            p.add_task("")
            current = ai_refiner.refine_pixel_art(current, out, desc)

        console.print(f"  [green]✓[/green] Pass {i+1} saved: {current}")
        show_image(current)

        if i < num_iterations - 1:
            if not confirm_or_quit("Continue refining this result?"):
                break

    return current


# ---------------------------------------------------------------------------
# Phase 3: User confirmation of seed frame
# ---------------------------------------------------------------------------

def phase_confirm_seed(variants: list[str], output_dir: str, skip_refine: bool) -> tuple:
    phase_header(3, "Confirm — pick the best frame")

    console.print("\n[bold]Variant paths:[/bold]")
    for i, path in enumerate(variants, 1):
        console.print(f"  [cyan]{i}[/cyan]. {path}")
    console.print(f"  [cyan]0[/cyan]. Enter a custom path")

    if len(variants) == 1:
        console.print("[dim]Only one variant — using it automatically.[/dim]")
        chosen = variants[0]
    else:
        choice = IntPrompt.ask("Pick the best variant", default=1)
        if choice == 0:
            chosen = Prompt.ask("Enter the path to your chosen frame")
        else:
            chosen = variants[max(0, min(choice - 1, len(variants) - 1))]

    console.print(f"\n[green]Seed frame selected:[/green] {chosen}")
    show_image(chosen)

    if not skip_refine:
        if confirm_or_quit("Would you like to refine this frame with AI before animating?"):
            return None, chosen  # signal to do refinement
    return chosen, None


# ---------------------------------------------------------------------------
# Phase 4: Build edit canvas
# ---------------------------------------------------------------------------

def phase_build_canvas(seed_path: str, output_dir: str, num_frames: int) -> str:
    phase_header(4, "Build Canvas — preparing edit canvas for GPT Image API")

    base = os.path.splitext(os.path.basename(seed_path))[0]
    canvas_path = os.path.join(output_dir, f"{base}_canvas.png")

    canvas_builder.build_edit_canvas(
        seed_path,
        canvas_path,
        num_frames=num_frames,
    )

    console.print(f"[green]✓[/green] Canvas ready: {canvas_path}")
    show_image(canvas_path)
    return canvas_path


# ---------------------------------------------------------------------------
# Phase 5: Generate animation strip
# ---------------------------------------------------------------------------

def phase_generate_animation(
    canvas_path: str,
    output_dir: str,
    animation_type: str,
    character_description: str,
    game_style: str,
    num_frames: int,
    max_attempts: int = 3,
) -> str:
    phase_header(5, f"Generate Animation — '{animation_type}' strip via GPT Image API")

    base = os.path.splitext(os.path.basename(canvas_path))[0]
    strip_path = None

    for attempt in range(1, max_attempts + 1):
        out = os.path.join(output_dir, f"{base}_{animation_type}_strip_raw_v{attempt}.png")
        console.print(f"  Attempt {attempt}/{max_attempts}...")

        with Progress(SpinnerColumn(), TextColumn("Generating animation strip..."), console=console) as p:
            p.add_task("")
            strip_path = ai_refiner.generate_animation_strip(
                canvas_path,
                out,
                animation_type=animation_type,
                character_description=character_description,
                game_style=game_style,
                num_frames=num_frames,
            )

        console.print(f"  [green]✓[/green] Strip saved: {strip_path}")
        show_image(strip_path)

        if attempt < max_attempts:
            if confirm_or_quit("Happy with this animation strip? (No = try again)"):
                break
        else:
            console.print("[dim]Max attempts reached — using last result.[/dim]")

    return strip_path


# ---------------------------------------------------------------------------
# Phase 6: Normalize
# ---------------------------------------------------------------------------

def phase_normalize(
    strip_path: str,
    output_dir: str,
    num_frames: int,
    target_size: int,
    seed_path: str,
) -> dict:
    phase_header(6, "Normalize — shared-scale frame extraction")

    norm_dir = os.path.join(output_dir, "normalized")
    result = sprite_normalizer.normalize_strip(
        strip_path,
        num_frames=num_frames,
        target_size=target_size,
        output_dir=norm_dir,
        seed_frame_path=seed_path,
    )

    console.print(f"[green]✓[/green] Normalized spritesheet: {result['spritesheet']}")
    show_image(result["spritesheet"])
    return result


# ---------------------------------------------------------------------------
# Phase 7: Export GIF
# ---------------------------------------------------------------------------

def phase_export_gif(
    normalized: dict,
    output_dir: str,
    animation_type: str,
    frame_delay: int,
    scale: int,
) -> str:
    phase_header(7, "Export GIF — composing animated GIF")

    frames = [Image.open(p).convert("RGBA") for p in normalized["frames"]]

    gif_path = os.path.join(output_dir, f"{animation_type}_animation.gif")
    gif_exporter.export_gif(
        frames,
        gif_path,
        frame_delay_ms=frame_delay,
        scale=scale,
    )

    console.print(f"\n[bold green]✓ Animated GIF ready: {gif_path}[/bold green]")
    show_image(gif_path)
    return gif_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_pipeline(args):
    """
    Output directory structure:
        output/<character_name>/
            base/
                seed.png              ← the approved base character sprite
                canvas.png            ← reusable edit canvas (rebuilt per frame count)
                variants/             ← pixelation variants (Phase 1)
            animations/
                <animation_type>/
                    strip_raw.png     ← raw AI output
                    normalized/       ← individual frames + spritesheet
                    <type>.gif        ← final animated GIF
    """
    # Interactive setup if not fully configured via CLI
    args = interactive_setup(args)

    if not os.path.exists(args.input):
        console.print(f"[red]Input file not found: {args.input}[/red]")
        sys.exit(1)

    # Character name for directory structure
    char_name = args.character_name or os.path.splitext(os.path.basename(args.input))[0]
    char_name = char_name.replace(" ", "_").lower()

    root_dir = args.output_dir or "output"
    char_dir = os.path.join(root_dir, char_name)
    base_dir = os.path.join(char_dir, "base")
    anim_dir = os.path.join(char_dir, "animations", args.animation)
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(anim_dir, exist_ok=True)

    # Auto-calculate GIF upscale so preview is ~256px
    gif_scale = args.gif_scale or max(1, 256 // args.frame_size)

    console.print(Panel.fit(
        f"[bold]Configuration[/bold]\n"
        f"Character:   [cyan]{char_name}[/cyan]\n"
        f"Input:       [cyan]{args.input}[/cyan]\n"
        f"Reference:   [cyan]{args.reference or 'none'}[/cyan]\n"
        f"Base dir:    [cyan]{base_dir}[/cyan]\n"
        f"Anim dir:    [cyan]{anim_dir}[/cyan]\n"
        f"Frame size:  [yellow]{args.frame_size}x{args.frame_size}[/yellow]\n"
        f"Animation:   [yellow]{args.animation}[/yellow]\n"
        f"Frames:      [yellow]{args.num_frames}[/yellow]\n"
        f"GIF scale:   [yellow]{gif_scale}x[/yellow]\n"
        f"Description: [yellow]{args.description}[/yellow]",
        title="[bold magenta]Pixelator Pipeline[/bold magenta]",
    ))

    if not confirm_or_quit("Start pipeline?"):
        console.print("[dim]Aborted.[/dim]")
        sys.exit(0)

    # Check if a base seed already exists (reuse for new animations)
    seed_path_file = os.path.join(base_dir, "seed.png")
    if os.path.exists(seed_path_file):
        console.print(f"\n[green]Existing base sprite found:[/green] {seed_path_file}")
        show_image(seed_path_file)
        if confirm_or_quit("Use this existing base sprite? (No = regenerate)"):
            seed_path = seed_path_file
        else:
            seed_path = None
    else:
        seed_path = None

    if seed_path is None:
        # Phase 1: Pixelate
        variants = phase_pixelate(args.input, base_dir, args.frame_size)
        if not variants:
            console.print("[red]No variants generated. Exiting.[/red]")
            sys.exit(1)

        # Phase 3 (first pass) — pick seed before optional refine
        seed_path, refine_candidate = phase_confirm_seed(variants, base_dir, args.no_refine)

        # Phase 2: AI Refine (optional)
        if refine_candidate is not None:
            seed_path = phase_ai_refine(
                refine_candidate,
                base_dir,
                args.description,
                reference_path=args.reference,
                num_iterations=args.refine_passes,
            )
            console.print(f"\n[green]Refined frame:[/green] {seed_path}")
            show_image(seed_path)
            if not confirm_or_quit("Use this refined frame as the seed?"):
                seed_path = refine_candidate
                console.print("[dim]Reverting to un-refined variant.[/dim]")

        # Save as the canonical base seed
        import shutil
        shutil.copy2(seed_path, seed_path_file)
        seed_path = seed_path_file
        console.print(f"[green]Base sprite saved:[/green] {seed_path}")

    # Phase 4: Build canvas
    canvas_path = phase_build_canvas(seed_path, base_dir, args.num_frames)

    # Phase 5: Generate animation
    strip_path = phase_generate_animation(
        canvas_path,
        anim_dir,
        animation_type=args.animation,
        character_description=args.description,
        game_style=args.game_style,
        num_frames=args.num_frames,
    )

    # Phase 6: Normalize
    normalized = phase_normalize(
        strip_path,
        anim_dir,
        num_frames=args.num_frames,
        target_size=args.frame_size,
        seed_path=seed_path,
    )

    # Phase 7: Export GIF
    gif_path = phase_export_gif(
        normalized,
        anim_dir,
        animation_type=args.animation,
        frame_delay=args.delay,
        scale=gif_scale,
    )

    console.print(Panel.fit(
        f"[bold green]Pipeline complete![/bold green]\n\n"
        f"Base sprite:   [cyan]{seed_path}[/cyan]\n"
        f"Spritesheet:   [cyan]{normalized['spritesheet']}[/cyan]\n"
        f"Animated GIF:  [bold cyan]{gif_path}[/bold cyan]\n\n"
        f"[dim]Run again with -a <type> to generate more animations from the same base.[/dim]",
        title="[bold]Done[/bold]",
    ))


def main():
    parser = argparse.ArgumentParser(
        description="Full pipeline: image → pixel art → AI refine → animation → GIF",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Fully interactive mode — prompts for everything
  python pipeline.py

  # Provide input image, rest is interactive
  python pipeline.py -i hero.png

  # Fully specified (non-interactive)
  python pipeline.py -i hero.png -d "pirate, red bandana, blue tunic" -a hurt -f 64

  # Use an example image with a reference
  python pipeline.py -i examples/a6c961c8129e88b084f0f30cfe2e661d.jpg \\
    --ref "examples/minotaur idle.gif" -a idle
        """,
    )

    parser.add_argument("-i", "--input", default=None, help="Input image (omit for interactive picker)")
    parser.add_argument("-o", "--output-dir", help="Output root directory (default: ./output)")
    parser.add_argument(
        "--name",
        dest="character_name",
        default=None,
        help="Character name for directory structure (default: derived from input filename)",
    )
    parser.add_argument(
        "-d", "--description",
        default="a game character",
        help="Character description for AI prompts",
    )
    parser.add_argument(
        "-a", "--animation",
        choices=ANIMATION_TYPES,
        default=None,
        help="Animation type (omit for interactive picker)",
    )
    parser.add_argument(
        "-g", "--game-style",
        default="platformer",
        help="Game style / genre for the AI prompt (default: platformer)",
    )
    parser.add_argument(
        "-n", "--num-frames",
        type=int,
        default=4,
        help="Number of animation frames (default: 4)",
    )
    parser.add_argument(
        "-f", "--frame-size",
        type=int,
        default=None,
        help="Output frame size in pixels (omit for interactive picker: 16/32/48/64/96/128)",
    )
    parser.add_argument(
        "--delay",
        type=int,
        default=100,
        help="GIF frame delay in ms (default: 100)",
    )
    parser.add_argument(
        "--gif-scale",
        type=int,
        default=None,
        help="Upscale GIF for visibility (default: auto to ~256px)",
    )
    parser.add_argument(
        "--ref", "--reference",
        dest="reference",
        default=None,
        help="Reference image for AI style guidance",
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Skip the AI refinement phase",
    )
    parser.add_argument(
        "--refine-passes",
        type=int,
        default=1,
        help="Number of AI refinement passes (default: 1)",
    )

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
