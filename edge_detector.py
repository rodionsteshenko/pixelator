#!/usr/bin/env python3
"""
Edge Detector: A tool to detect and visualize edges in images.

This script analyzes an image row by row or column by column to find edges
based on color differences. It outputs both numerical differences and
visualizations showing where the most significant changes occur.
"""

import argparse
import os
import numpy as np
from PIL import Image

# We'll use PIL for visualization instead of matplotlib
# import matplotlib.pyplot as plt
# from matplotlib.colors import Normalize
from enum import Enum


class DiffAlgorithm(Enum):
    """Algorithms for measuring differences between rows/columns."""

    ABSOLUTE = "absolute"  # Sum of absolute differences
    SQUARED = "squared"  # Sum of squared differences
    EUCLIDEAN = "euclidean"  # Euclidean distance in RGB space
    MAX = "max"  # Maximum difference across any channel


def load_image(image_path):
    """Load an image from the given path and return it as a numpy array."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    try:
        img = Image.open(image_path)
        return np.array(img)
    except Exception as e:
        raise RuntimeError(f"Error loading image: {str(e)}")


def calculate_differences(image_array, axis=0, algorithm=DiffAlgorithm.ABSOLUTE):
    """
    Calculate differences between adjacent rows or columns.

    Args:
        image_array: Input image as numpy array
        axis: 0 for row differences, 1 for column differences
        algorithm: Algorithm to use for calculating differences

    Returns:
        Array of difference values
    """
    # Calculate differences between adjacent rows/columns
    differences = np.diff(image_array, axis=axis)

    # Apply the selected algorithm
    if algorithm == DiffAlgorithm.ABSOLUTE:
        # Sum of absolute differences across all channels
        diff_values = np.sum(np.abs(differences), axis=(1 if axis == 0 else 0, 2))

    elif algorithm == DiffAlgorithm.SQUARED:
        # Sum of squared differences
        diff_values = np.sum(differences**2, axis=(1 if axis == 0 else 0, 2))
        diff_values = np.sqrt(diff_values)  # Take square root to normalize

    elif algorithm == DiffAlgorithm.EUCLIDEAN:
        # Euclidean distance in RGB space
        squared = differences**2
        sum_squared = np.sum(squared, axis=2)
        diff_values = np.sqrt(np.sum(sum_squared, axis=(1 if axis == 0 else 0)))

    elif algorithm == DiffAlgorithm.MAX:
        # Maximum difference across any channel
        diff_values = np.max(np.abs(differences), axis=(1 if axis == 0 else 0, 2))

    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")

    return diff_values


def normalize_differences(differences, scale_to=100):
    """
    Normalize differences to a given scale (0-100 by default).

    Args:
        differences: Array of difference values
        scale_to: Upper limit for normalization (default: 100)

    Returns:
        Normalized differences array
    """
    min_val = np.min(differences)
    max_val = np.max(differences)

    # Avoid division by zero
    if max_val == min_val:
        return np.zeros_like(differences)

    # Normalize to 0-scale_to range
    normalized = (differences - min_val) / (max_val - min_val) * scale_to
    return normalized


def find_edges(differences, threshold_percentile=90):
    """
    Find edges based on difference values.

    Args:
        differences: Array of difference values
        threshold_percentile: Percentile threshold for edge detection (default: 90)

    Returns:
        List of edge positions
    """
    threshold = np.percentile(differences, threshold_percentile)
    edges = np.where(differences >= threshold)[0]
    return edges


def create_edge_visualization(image_array, edges, axis=0, edge_color=(0, 255, 0)):
    """
    Create a visualization with edges highlighted.

    Args:
        image_array: Original image as numpy array
        edges: List of edge positions
        axis: 0 for horizontal edges, 1 for vertical edges
        edge_color: RGB color for edge highlighting (default: green)

    Returns:
        Annotated image as numpy array
    """
    vis_image = np.copy(image_array)

    # Convert edge_color to numpy array if it's a tuple
    if isinstance(edge_color, tuple):
        edge_color = np.array(edge_color, dtype=np.uint8)

    # Draw edges
    line_thickness = 1

    if axis == 0:  # Horizontal edges (between rows)
        for edge in edges:
            # Edges are between rows, so add 1 to get the actual boundary
            row = edge + 1
            if 0 <= row < vis_image.shape[0]:
                vis_image[row : row + line_thickness, :] = edge_color
    else:  # Vertical edges (between columns)
        for edge in edges:
            # Edges are between columns, so add 1 to get the actual boundary
            col = edge + 1
            if 0 <= col < vis_image.shape[1]:
                vis_image[:, col : col + line_thickness] = edge_color

    return vis_image


def create_difference_plot(image_array, differences, axis=0, normalize=True):
    """
    Create a visualization showing the image with a proper line chart.

    Args:
        image_array: Original image as numpy array
        differences: Array of difference values
        axis: 0 for row differences, 1 for column differences
        normalize: Whether to normalize differences (default: True)

    Returns:
        Combined visualization as numpy array
    """
    # Normalize differences if requested
    if normalize:
        plot_data = normalize_differences(differences)
    else:
        plot_data = differences

    height, width, _ = image_array.shape

    # Create a visualization image that includes the original image and a line chart
    if axis == 0:  # Row differences, chart on the right
        # Create space for the line chart on the right
        chart_width = 200  # Width of the chart area
        vis_width = width + chart_width
        vis_image = np.zeros((height, vis_width, 3), dtype=np.uint8)

        # Set background color to dark gray for the chart area
        vis_image[:, width:] = [30, 30, 30]

        # Copy the original image to the left side
        vis_image[:, :width] = image_array

        # Draw chart grid lines (light gray)
        grid_color = [80, 80, 80]
        for i in range(0, 101, 10):  # Draw grid lines at 0, 10, 20, ..., 100
            # Calculate x position for this value
            x_pos = width + int(i * chart_width / 100)
            # Ensure x_pos is within bounds
            if x_pos < vis_width:
                # Draw vertical grid line
                vis_image[:, x_pos] = grid_color

                # Add value label (as a small white line)
                if i % 20 == 0:  # Label at 0, 20, 40, 60, 80, 100
                    for y in range(height - 10, height):
                        # Ensure label markers are within bounds
                        left_marker = max(0, x_pos - 1)
                        right_marker = min(vis_width - 1, x_pos + 2)
                        vis_image[y, left_marker:right_marker] = [200, 200, 200]

        # Find the maximum value for scaling
        max_diff = np.max(plot_data)
        if max_diff == 0:
            max_diff = 1  # Avoid division by zero

        # Draw the line chart
        line_color = [50, 200, 50]  # Green line
        highlight_color = [255, 200, 0]  # Yellow for high values
        threshold = np.percentile(plot_data, 90)

        # Draw horizontal line for each data point
        for i, value in enumerate(plot_data):
            if i >= height:
                break

            # Calculate x position for this value
            x_pos = width + int(value * chart_width / 100)
            # Ensure x_pos is within bounds
            x_pos = min(vis_width - 1, x_pos)

            # Draw horizontal line from left edge of chart to data point
            color = highlight_color if value > threshold else line_color
            vis_image[i, width : x_pos + 1] = color

            # Draw a small circle marker at the data point
            y_pos = i
            # Simple circle approximation
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    if dx * dx + dy * dy <= 4:  # Circle equation
                        if 0 <= y_pos + dy < height and 0 <= x_pos + dx < vis_width:
                            vis_image[y_pos + dy, x_pos + dx] = [255, 255, 255]

        # Add title and labels
        title = "Row Differences"
        draw_text(vis_image, title, width + 10, 20)

    else:  # Column differences, chart on top
        # Create space for the line chart on top
        chart_height = 200  # Height of the chart area
        vis_height = height + chart_height
        vis_image = np.zeros((vis_height, width, 3), dtype=np.uint8)

        # Set background color to dark gray for the chart area
        vis_image[:chart_height, :] = [30, 30, 30]

        # Copy the original image to the bottom part
        vis_image[chart_height:, :] = image_array

        # Draw chart grid lines (light gray)
        grid_color = [80, 80, 80]
        for i in range(0, 101, 10):  # Draw grid lines at 0, 10, 20, ..., 100
            # Calculate y position for this value (invert y axis)
            y_pos = chart_height - int(i * chart_height / 100)
            # Ensure y_pos is within bounds
            if 0 <= y_pos < vis_height:
                # Draw horizontal grid line
                vis_image[y_pos, :] = grid_color

                # Add value label (as a small white line)
                if i % 20 == 0:  # Label at 0, 20, 40, 60, 80, 100
                    for x in range(0, 10):
                        # Ensure label markers are within bounds
                        top_marker = max(0, y_pos - 1)
                        bottom_marker = min(vis_height - 1, y_pos + 2)
                        vis_image[top_marker:bottom_marker, x] = [200, 200, 200]

        # Draw the line chart
        line_color = [50, 200, 50]  # Green line
        highlight_color = [255, 200, 0]  # Yellow for high values
        threshold = np.percentile(plot_data, 90)

        # Connect data points with lines
        for i in range(len(plot_data) - 1):
            if i >= width - 1:
                break

            # Calculate positions for this segment
            x1, x2 = i, i + 1
            y1 = chart_height - int(plot_data[i] * chart_height / 100)
            y2 = chart_height - int(plot_data[i + 1] * chart_height / 100)

            # Ensure points are within bounds
            y1 = max(0, min(chart_height - 1, y1))
            y2 = max(0, min(chart_height - 1, y2))

            # Select color based on value
            color = (
                highlight_color
                if plot_data[i] > threshold or plot_data[i + 1] > threshold
                else line_color
            )

            # Draw line using Bresenham's algorithm
            draw_line(vis_image, x1, y1, x2, y2, color)

            # Draw markers at data points
            if i % 10 == 0:  # Draw markers at every 10th point to avoid clutter
                # Simple circle approximation
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dx * dx + dy * dy <= 4:  # Circle equation
                            if 0 <= y1 + dy < chart_height and 0 <= x1 + dx < width:
                                vis_image[y1 + dy, x1 + dx] = [255, 255, 255]

        # Add title and labels
        title = "Column Differences"
        draw_text(vis_image, title, 10, 20)

    return vis_image


def draw_line(image, x0, y0, x1, y1, color):
    """
    Draw a line using Bresenham's algorithm.
    """
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        # Set pixel if within bounds
        if 0 <= y0 < image.shape[0] and 0 <= x0 < image.shape[1]:
            image[y0, x0] = color

        # Exit condition
        if x0 == x1 and y0 == y1:
            break

        e2 = 2 * err
        if e2 > -dy:
            if x0 == x1:
                break
            err -= dy
            x0 += sx
        if e2 < dx:
            if y0 == y1:
                break
            err += dx
            y0 += sy


def draw_text(image, text, x, y, color=[255, 255, 255]):
    """
    Simple function to draw text on the image.
    Very basic implementation that only works for short strings.
    """
    # This is a very simplified version that just draws dots for each character
    scale = 1
    spacing = 6 * scale

    for i, char in enumerate(text):
        pos_x = x + i * spacing
        pos_y = y

        # Skip if out of bounds
        if pos_x >= image.shape[1] - spacing:
            break

        # Just draw a small rectangle for each character
        for dy in range(scale * 5):
            for dx in range(scale * 5):
                py = pos_y + dy
                px = pos_x + dx
                if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
                    image[py, px] = color


def process_image(
    image_path,
    output_dir=None,
    axis=0,
    algorithm=DiffAlgorithm.ABSOLUTE,
    threshold_percentile=90,
    normalize=True,
):
    """
    Process an image to detect edges and create visualizations.

    Args:
        image_path: Path to the input image
        output_dir: Directory to save outputs (default: same as input)
        axis: 0 for row analysis, 1 for column analysis
        algorithm: Algorithm to use for difference calculation
        threshold_percentile: Percentile threshold for edge detection
        normalize: Whether to normalize differences

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
    direction = "rows" if axis == 0 else "cols"
    algo_name = algorithm.value

    # Load the image
    image_array = load_image(image_path)
    print(f"Loaded image with shape {image_array.shape}")

    # Calculate differences
    differences = calculate_differences(image_array, axis=axis, algorithm=algorithm)
    print(f"Calculated {direction} differences using {algo_name} algorithm")

    # Find edges
    edges = find_edges(differences, threshold_percentile)
    print(
        f"Found {len(edges)} edges using {threshold_percentile}th percentile threshold"
    )

    # Skip edge visualization

    # Create difference plot
    diff_plot = create_difference_plot(
        image_array, differences, axis=axis, normalize=normalize
    )
    diff_plot_path = (
        f"{output_dir}/{base_filename}_diff_plot_{direction}_{algo_name}.png"
    )
    Image.fromarray(diff_plot).save(diff_plot_path)
    print(f"Saved difference plot to {diff_plot_path}")

    # Save raw difference values to a text file
    diff_values_path = (
        f"{output_dir}/{base_filename}_diff_values_{direction}_{algo_name}.txt"
    )
    with open(diff_values_path, "w") as f:
        # Write header
        f.write(f"# Difference values ({direction}, {algo_name} algorithm)\n")
        f.write("# Position,Raw Difference,Normalized (0-100)\n")

        # Write values
        normalized = normalize_differences(differences) if normalize else differences
        for i, (diff, norm) in enumerate(zip(differences, normalized)):
            f.write(f"{i},{diff:.2f},{norm:.2f}\n")

    print(f"Saved difference values to {diff_values_path}")

    return {"difference_plot": diff_plot_path, "difference_values": diff_values_path}


def main():
    """Parse command line arguments and run the edge detection."""
    parser = argparse.ArgumentParser(description="Detect and visualize edges in images")
    parser.add_argument("input", help="Path to the input image")
    parser.add_argument("-o", "--output-dir", help="Directory to save outputs")
    parser.add_argument(
        "-a",
        "--axis",
        type=int,
        choices=[0, 1],
        default=0,
        help="Analysis direction: 0 for rows, 1 for columns (default: 0)",
    )
    parser.add_argument(
        "-m",
        "--algorithm",
        choices=[a.value for a in DiffAlgorithm],
        default=DiffAlgorithm.ABSOLUTE.value,
        help="Algorithm for calculating differences (default: absolute)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=int,
        default=90,
        help="Percentile threshold for edge detection (default: 90)",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Don't normalize difference values (default: normalize to 0-100)",
    )

    args = parser.parse_args()

    # Convert string algorithm choice to enum
    algorithm = DiffAlgorithm(args.algorithm)

    # Process the image
    outputs = process_image(
        args.input,
        output_dir=args.output_dir,
        axis=args.axis,
        algorithm=algorithm,
        threshold_percentile=args.threshold,
        normalize=not args.no_normalize,
    )

    print("\nSummary:")
    print(f"  Input image: {args.input}")
    print(f"  Analysis direction: {'rows' if args.axis == 0 else 'columns'}")
    print(f"  Difference algorithm: {args.algorithm}")
    print(f"  Edge threshold: {args.threshold}th percentile")
    print(f"  Difference plot: {outputs['difference_plot']}")
    print(f"  Difference values: {outputs['difference_values']}")


if __name__ == "__main__":
    main()
