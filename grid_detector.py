#!/usr/bin/env python3
"""
Grid Detector: A tool to detect pixel grid in AI-generated pixel art images.

This script uses the edge_detector module to find differences between rows and columns,
then applies a search window algorithm to find regularly spaced edges that form a grid.
It outputs a visualization of the detected grid overlaid on the original image.
"""

import os
import argparse
import numpy as np
from PIL import Image
import edge_detector


def read_difference_values(file_path):
    """
    Read difference values from a file created by edge_detector.py.

    Args:
        file_path: Path to the difference values file

    Returns:
        Array of raw difference values
    """
    differences = []

    with open(file_path, "r") as f:
        # Skip header lines
        next(f)
        next(f)

        # Read difference values
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    # We only need the raw difference value
                    raw_diff = float(parts[1])
                    differences.append(raw_diff)
                except (ValueError, IndexError):
                    pass

    return np.array(differences)


def find_edges_with_window(
    differences, approx_spacing, window_size, start_pos=0, max_edges=1000
):
    """
    Find regularly spaced edges using a search window approach.

    Args:
        differences: Array of difference values
        approx_spacing: Approximate spacing between edges
        window_size: Size of the search window around the approximate position
        start_pos: Position to start the search from

    Returns:
        List of detected edge positions
    """
    edges = []
    pos = start_pos

    # Enforce a minimum spacing of 2 to prevent getting stuck
    approx_spacing = max(2, approx_spacing)

    # Keep track of the last position to detect if we're stuck
    last_pos = -1

    while pos < len(differences) and len(edges) < max_edges:
        # Check if we're stuck at the same position
        if pos == last_pos:
            break
        last_pos = pos

        # Add the first position
        if not edges:
            # Find the maximum difference in the first section
            search_end = min(pos + approx_spacing * 2, len(differences))
            window = differences[pos:search_end]
            max_idx = pos + np.argmax(window)
            edges.append(max_idx)
            pos = max_idx + 1
            continue

        # Calculate the expected position of the next edge
        expected_pos = edges[-1] + approx_spacing

        # Define the search window
        window_start = max(0, expected_pos - window_size)
        window_end = min(len(differences), expected_pos + window_size + 1)

        # If we've reached the end of the array, break
        if window_start >= len(differences) - 1:
            break

        # Find the maximum within the window
        window = differences[window_start:window_end]

        # If the window is empty or has no variation, break
        if len(window) == 0 or np.max(window) == np.min(window):
            break

        # Find the position of the maximum difference in the window
        max_idx = window_start + np.argmax(window)
        edges.append(max_idx)

        # Move position forward
        pos = max_idx + 1

    return edges


def create_grid_overlay(image_array, row_edges, col_edges, alpha=0.5):
    """
    Create a visualization with a semi-transparent grid overlay.

    Args:
        image_array: Original image as numpy array
        row_edges: List of row edge positions
        col_edges: List of column edge positions
        alpha: Transparency of the grid (0-1)

    Returns:
        Image with grid overlay as numpy array
    """
    # Create a copy of the image
    result = np.copy(image_array).astype(np.float32)
    height, width, _ = result.shape

    # Create grid overlay
    overlay = np.zeros_like(result)
    grid_color = np.array([128, 128, 128])  # 50% gray

    # Draw horizontal grid lines
    for edge in row_edges:
        if 0 <= edge < height:
            overlay[edge, :] = grid_color

    # Draw vertical grid lines
    for edge in col_edges:
        if 0 <= edge < width:
            overlay[:, edge] = grid_color

    # Blend the overlay with the original image
    result = result * (1 - alpha) + overlay * alpha

    # Convert back to uint8
    return np.clip(result, 0, 255).astype(np.uint8)


def detect_grid(
    image_path,
    output_dir=None,
    row_approx_spacing=None,
    col_approx_spacing=None,
    window_size=2,
    row_threshold=90,
    col_threshold=90,
    algorithm="euclidean",
):
    """
    Detect grid in an image and create visualizations.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs (default: same as input)
        row_approx_spacing: Approximate spacing between rows (if None, auto-detect)
        col_approx_spacing: Approximate spacing between columns (if None, auto-detect)
        window_size: Size of the search window
        row_threshold: Percentile threshold for row edge detection
        col_threshold: Percentile threshold for column edge detection
        algorithm: Difference algorithm to use

    Returns:
        Dictionary with paths to the output files
    """
    # Default output directory is the same as input directory
    if output_dir is None:
        output_dir = os.path.dirname(image_path)
        if output_dir == "":
            output_dir = "."

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Base filename for outputs
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Process the image with edge_detector for rows
    print("Processing rows...")
    row_outputs = edge_detector.process_image(
        image_path,
        output_dir=output_dir,
        axis=0,  # rows
        algorithm=edge_detector.DiffAlgorithm(algorithm),
        threshold_percentile=row_threshold,
        normalize=True,
    )

    # Process the image with edge_detector for columns
    print("\nProcessing columns...")
    col_outputs = edge_detector.process_image(
        image_path,
        output_dir=output_dir,
        axis=1,  # columns
        algorithm=edge_detector.DiffAlgorithm(algorithm),
        threshold_percentile=col_threshold,
        normalize=True,
    )

    # Read difference values
    row_differences = read_difference_values(row_outputs["difference_values"])
    col_differences = read_difference_values(col_outputs["difference_values"])

    # Auto-detect approximate spacing if not specified
    if row_approx_spacing is None:
        # Analyze the differences to find the most common peak intervals
        # For simplicity, use the first few detected edges to estimate spacing
        edges = edge_detector.find_edges(row_differences, row_threshold)
        spacings = (
            np.diff(edges[:10]) if len(edges) >= 11 else np.array([13])
        )  # Default to 13 if not enough edges
        row_approx_spacing = int(np.median(spacings))
        # Enforce a minimum spacing of 5 to prevent algorithm from getting stuck
        row_approx_spacing = max(5, row_approx_spacing)
        print(f"Auto-detected row spacing: {row_approx_spacing}")

    if col_approx_spacing is None:
        # Same approach for columns
        edges = edge_detector.find_edges(col_differences, col_threshold)
        spacings = np.diff(edges[:10]) if len(edges) >= 11 else np.array([13])
        col_approx_spacing = int(np.median(spacings))
        # Enforce a minimum spacing of 5 to prevent algorithm from getting stuck
        col_approx_spacing = max(5, col_approx_spacing)
        print(f"Auto-detected column spacing: {col_approx_spacing}")

    # Find edges with search window approach
    print(
        f"Finding rows with approx spacing {row_approx_spacing} and window size {window_size}..."
    )
    row_edges = find_edges_with_window(
        row_differences, row_approx_spacing, window_size, max_edges=500
    )

    print(
        f"Finding columns with approx spacing {col_approx_spacing} and window size {window_size}..."
    )
    col_edges = find_edges_with_window(
        col_differences, col_approx_spacing, window_size, max_edges=500
    )

    print(f"Found {len(row_edges)} rows and {len(col_edges)} columns")

    # Load the original image
    image_array = edge_detector.load_image(image_path)

    # Create grid overlay visualization
    grid_overlay = create_grid_overlay(image_array, row_edges, col_edges)
    grid_overlay_path = f"{output_dir}/{base_filename}_grid_overlay.png"
    Image.fromarray(grid_overlay).save(grid_overlay_path)
    print(f"Saved grid overlay to {grid_overlay_path}")

    # Save grid coordinates
    grid_coords_path = f"{output_dir}/{base_filename}_grid_coords.txt"
    with open(grid_coords_path, "w") as f:
        f.write("# Grid Coordinates\n")
        f.write("# Row positions\n")
        for edge in row_edges:
            f.write(f"{edge}\n")
        f.write("\n# Column positions\n")
        for edge in col_edges:
            f.write(f"{edge}\n")

    print(f"Saved grid coordinates to {grid_coords_path}")

    return {
        "grid_overlay": grid_overlay_path,
        "grid_coords": grid_coords_path,
        "row_edges": row_edges,
        "col_edges": col_edges,
    }


def main():
    """Parse command line arguments and run the grid detection."""
    parser = argparse.ArgumentParser(
        description="Detect pixel grid in AI-generated pixel art images"
    )
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("-o", "--output-dir", help="Directory to save outputs")
    parser.add_argument(
        "--row-spacing", type=int, help="Approximate spacing between rows", default=13
    )
    parser.add_argument(
        "--col-spacing",
        type=int,
        help="Approximate spacing between columns",
        default=13,
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=2,
        help="Size of the search window around expected positions",
    )
    parser.add_argument(
        "--row-threshold",
        type=int,
        default=90,
        help="Percentile threshold for row edge detection (default: 90)",
    )
    parser.add_argument(
        "--col-threshold",
        type=int,
        default=90,
        help="Percentile threshold for column edge detection (default: 90)",
    )
    parser.add_argument(
        "-m",
        "--algorithm",
        choices=[a.value for a in edge_detector.DiffAlgorithm],
        default=edge_detector.DiffAlgorithm.EUCLIDEAN.value,
        help="Algorithm for calculating differences (default: euclidean)",
    )

    args = parser.parse_args()

    # Detect grid
    outputs = detect_grid(
        args.input,
        output_dir=args.output_dir,
        row_approx_spacing=args.row_spacing,
        col_approx_spacing=args.col_spacing,
        window_size=args.window,
        row_threshold=args.row_threshold,
        col_threshold=args.col_threshold,
        algorithm=args.algorithm,
    )

    print("\nSummary:")
    print(f"  Input image: {args.input}")
    print(f"  Algorithm: {args.algorithm}")
    print(
        f"  Detected {len(outputs['row_edges'])} rows and {len(outputs['col_edges'])} columns"
    )
    print(f"  Grid overlay: {outputs['grid_overlay']}")
    print(f"  Grid coordinates: {outputs['grid_coords']}")


if __name__ == "__main__":
    main()
