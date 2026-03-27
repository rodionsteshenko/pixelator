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


def _make_checker_bg(width, height, checker_size=16):
    """Create a checkerboard background image (RGBA) efficiently."""
    arr = np.full((height, width, 4), [30, 30, 40, 255], dtype=np.uint8)
    # Vectorized checkerboard pattern
    yy, xx = np.mgrid[:height, :width]
    mask = ((yy // checker_size) + (xx // checker_size)) % 2 == 0
    arr[mask] = [40, 40, 52, 255]
    return Image.fromarray(arr)


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


def _reconstruction_error(image_array, grid_size):
    """
    Measure how well a grid size explains the image.

    Downscale to grid_size using block mean, upscale back with nearest-neighbor,
    compare with original. Lower error = better match = this IS the pixel grid.

    Uses PIL resize for speed instead of Python loops.
    """
    h, w = image_array.shape[:2]

    # Fast downscale with PIL (BOX filter = block average)
    img = Image.fromarray(image_array.astype(np.uint8))
    small = img.resize((grid_size, grid_size), Image.BOX)
    # Upscale back with nearest neighbor
    reconstructed = small.resize((w, h), Image.NEAREST)

    # MSE
    diff = image_array.astype(np.float32) - np.array(reconstructed, dtype=np.float32)
    return float(np.mean(diff * diff))


def _peak_interval_detection(differences, min_period=4, max_period=256):
    """
    Detect the dominant interval between peaks in the difference signal.

    Finds strong edges (peaks in the diff signal), computes intervals
    between consecutive peaks, and finds the most common interval.

    Returns:
        List of (period, count) tuples sorted by count descending.
    """
    # Find peaks: local maxima above the 60th percentile
    threshold = np.percentile(differences, 60)
    above = differences > threshold

    # Find peak positions (local maxima)
    peaks = []
    for i in range(1, len(differences) - 1):
        if (differences[i] > differences[i-1] and
            differences[i] >= differences[i+1] and
            above[i]):
            peaks.append(i)

    if len(peaks) < 3:
        return []

    # Compute all intervals between consecutive peaks
    intervals = np.diff(peaks)

    # Bin intervals and find most common ones
    from collections import Counter
    # Round intervals to nearest integer
    interval_counts = Counter(intervals)

    # Also count intervals that are within ±1 of each other as the same
    merged = {}
    for interval, count in sorted(interval_counts.items()):
        if interval < min_period or interval > max_period:
            continue
        # Check if close to an existing bucket
        matched = False
        for key in merged:
            if abs(interval - key) <= 1:
                merged[key] += count
                matched = True
                break
        if not matched:
            merged[interval] = count

    return sorted(merged.items(), key=lambda x: x[1], reverse=True)


def detect_pixel_grid(image_path, min_grid=8, max_grid=128, top_k=5):
    """
    Auto-detect the logical pixel grid size in an AI-generated pixel art image.

    Uses three complementary approaches:
      1. FFT frequency analysis of row/column difference signals
      2. Peak interval detection (finding common spacing between edges)
      3. Reconstruction error (downscale → upscale → compare with original)

    The reconstruction error is the primary confidence metric: the grid size
    that best reconstructs the original image is the true pixel grid.

    Args:
        image_path: Path to the image (typically 1024x1024 AI output)
        min_grid: Minimum logical grid size to consider
        max_grid: Maximum logical grid size to consider
        top_k: Number of top candidates to return

    Returns:
        List of dicts: [{grid_size, pixel_block, confidence, mse}, ...]
        sorted by confidence descending (lowest reconstruction error).
    """
    image_array = edge_detector.load_image(image_path)
    h, w = image_array.shape[:2]

    # Work with RGB (drop alpha for analysis)
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array]*3, axis=2)
        analysis_arr = image_array
    elif image_array.shape[2] == 4:
        analysis_arr = image_array[:, :, :3]
    else:
        analysis_arr = image_array

    # --- Step 1: Gather candidate grid sizes from multiple methods ---

    # Method A: FFT frequency analysis
    row_diffs = edge_detector.calculate_differences(
        analysis_arr, axis=0, algorithm=edge_detector.DiffAlgorithm.EUCLIDEAN,
    )
    col_diffs = edge_detector.calculate_differences(
        analysis_arr, axis=1, algorithm=edge_detector.DiffAlgorithm.EUCLIDEAN,
    )

    min_period = max(2, w // max_grid)
    max_period = min(w // 2, w // min_grid)

    fft_candidates = set()
    for period, _ in _fft_peak_periods(row_diffs, min_period, max_period, top_k=8):
        gs = round(w / period)
        if min_grid <= gs <= max_grid:
            fft_candidates.add(gs)
    for period, _ in _fft_peak_periods(col_diffs, min_period, max_period, top_k=8):
        gs = round(w / period)
        if min_grid <= gs <= max_grid:
            fft_candidates.add(gs)

    # Method B: Peak interval detection
    peak_candidates = set()
    for period, count in _peak_interval_detection(row_diffs, min_period, max_period)[:5]:
        gs = round(w / period)
        if min_grid <= gs <= max_grid:
            peak_candidates.add(gs)
    for period, count in _peak_interval_detection(col_diffs, min_period, max_period)[:5]:
        gs = round(w / period)
        if min_grid <= gs <= max_grid:
            peak_candidates.add(gs)

    # Method C: Common pixel art sizes as fallbacks
    common_sizes = {16, 24, 32, 48, 64}

    # Combine all candidates
    all_candidates = fft_candidates | peak_candidates | common_sizes
    all_candidates = {gs for gs in all_candidates if min_grid <= gs <= max_grid}

    # Also add neighbors of detected sizes (±1, ±2) for fine-tuning
    neighbors = set()
    for gs in list(fft_candidates | peak_candidates):
        for delta in [-2, -1, 1, 2]:
            n = gs + delta
            if min_grid <= n <= max_grid:
                neighbors.add(n)
    all_candidates |= neighbors

    print(f"Grid candidates to evaluate: {sorted(all_candidates)}")

    # --- Step 2: Score each candidate by reconstruction error ---
    results = []
    for gs in sorted(all_candidates):
        mse = _reconstruction_error(analysis_arr, gs)
        pixel_block = w / gs
        results.append({
            "grid_size": gs,
            "pixel_block": round(pixel_block, 2),
            "mse": round(mse, 2),
        })

    if not results:
        return []

    # --- Step 3: Convert MSE to confidence (lower MSE = higher confidence) ---
    mse_values = [r["mse"] for r in results]
    min_mse = min(mse_values)
    max_mse = max(mse_values)
    mse_range = max_mse - min_mse if max_mse > min_mse else 1.0

    for r in results:
        # Confidence = 1.0 for the best (lowest MSE), 0.0 for the worst
        r["confidence"] = round(1.0 - (r["mse"] - min_mse) / mse_range, 3)

    # Sort by confidence (best reconstruction first)
    results.sort(key=lambda x: x["confidence"], reverse=True)

    return results[:top_k]


def _fft_peak_periods(signal, min_period, max_period, top_k=10):
    """
    Find dominant periods in a 1D signal using FFT.

    Returns:
        List of (period, magnitude) tuples sorted by magnitude descending.
    """
    n = len(signal)
    fft = np.fft.rfft(signal)
    magnitudes = np.abs(fft)

    # Skip DC component (index 0) and very low frequencies
    magnitudes[0] = 0

    # Convert to periods
    freqs = np.fft.rfftfreq(n)
    candidates = []
    for i in range(1, len(magnitudes)):
        if freqs[i] == 0:
            continue
        period = 1.0 / freqs[i]
        if min_period <= period <= max_period:
            candidates.append((round(period), float(magnitudes[i])))

    # Sort by magnitude
    candidates.sort(key=lambda x: x[1], reverse=True)

    # Deduplicate nearby periods (within 1 of each other)
    filtered = []
    for period, mag in candidates:
        if not any(abs(period - p) <= 1 for p, _ in filtered):
            filtered.append((period, mag))
        if len(filtered) >= top_k:
            break

    return filtered


def _validate_period(differences, period, tolerance=2):
    """
    Validate a candidate period by checking spacing consistency of peaks.

    Returns confidence score 0.0-1.0.
    """
    threshold = np.percentile(differences, 70)
    peaks = np.where(differences > threshold)[0]

    if len(peaks) < 3:
        return 0.0

    # Check what fraction of peak-to-peak intervals are close to the period
    intervals = np.diff(peaks)
    # Group consecutive peaks that are within 2 pixels of each other
    # (they're part of the same boundary)
    gap_intervals = intervals[intervals > period * 0.5]

    if len(gap_intervals) == 0:
        return 0.0

    # How many gaps are close to the expected period or a small multiple
    good = 0
    for gap in gap_intervals:
        remainder = gap % period
        if remainder <= tolerance or (period - remainder) <= tolerance:
            good += 1

    return good / len(gap_intervals)


def _find_boundary_peaks(edge_strength, merge_radius=3):
    """
    Find true pixel boundary positions in an edge strength signal.

    Uses the 85th percentile as threshold, finds local maxima,
    then merges peaks within merge_radius of each other (anti-aliasing
    causes double peaks at real boundaries).

    Returns:
        Sorted list of boundary positions.
    """
    threshold = np.percentile(edge_strength, 85)

    # Find local maxima above threshold
    raw_peaks = []
    for i in range(1, len(edge_strength) - 1):
        if (edge_strength[i] > threshold and
            edge_strength[i] >= edge_strength[i-1] and
            edge_strength[i] >= edge_strength[i+1]):
            raw_peaks.append(i)

    if not raw_peaks:
        return []

    # Merge nearby peaks (keep the strongest in each cluster)
    merged = []
    cluster = [raw_peaks[0]]
    for p in raw_peaks[1:]:
        if p - cluster[-1] <= merge_radius:
            cluster.append(p)
        else:
            # Pick the strongest peak in the cluster
            best = max(cluster, key=lambda x: edge_strength[x])
            merged.append(best)
            cluster = [p]
    best = max(cluster, key=lambda x: edge_strength[x])
    merged.append(best)

    return merged


def click_detect_grid(image_path, click_x, click_y):
    """
    Detect pixel grid from a single click point.

    Finds all strong edge peaks in the image, identifies the two nearest
    peaks bracketing the click point (giving us the seed pixel boundaries),
    then expands outward to map the full grid.

    Args:
        image_path: Path to the image
        click_x, click_y: Click coordinates in image space (e.g. 0-1023)

    Returns:
        Same as seed_grid_detect, plus seed_x, seed_y
    """
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float64)
    h, w = img_arr.shape[:2]

    cx = max(1, min(w - 2, click_x))
    cy = max(1, min(h - 2, click_y))

    # Edge strength signals
    col_edge = np.mean(
        np.sqrt(np.sum((img_arr[:, 1:, :] - img_arr[:, :-1, :]) ** 2, axis=2)),
        axis=0,
    )
    row_edge = np.mean(
        np.sqrt(np.sum((img_arr[1:, :, :] - img_arr[:-1, :, :]) ** 2, axis=2)),
        axis=1,
    )

    # Find all real boundary peaks on both axes
    col_peaks = _find_boundary_peaks(col_edge)
    row_peaks = _find_boundary_peaks(row_edge)

    print(f"Raw peaks: {len(col_peaks)} col, {len(row_peaks)} row")

    # Compute median interval for each axis
    col_intervals = np.diff(col_peaks) if len(col_peaks) > 1 else np.array([w])
    row_intervals = np.diff(row_peaks) if len(row_peaks) > 1 else np.array([h])

    # Filter out tiny intervals (anti-aliasing artifacts, < 40% of median)
    col_med = float(np.median(col_intervals))
    row_med = float(np.median(row_intervals))
    col_good = col_intervals[col_intervals > col_med * 0.4]
    row_good = row_intervals[row_intervals > row_med * 0.4]

    col_spacing = float(np.median(col_good)) if len(col_good) else col_med
    row_spacing = float(np.median(row_good)) if len(row_good) else row_med

    # Pixel art has square pixels — use the average of the two axes
    # weighted toward the axis with more consistent spacing (lower std)
    col_std = float(np.std(col_good)) if len(col_good) > 1 else 999
    row_std = float(np.std(row_good)) if len(row_good) > 1 else 999
    print(f"Spacing: col={col_spacing:.1f}±{col_std:.1f}, row={row_spacing:.1f}±{row_std:.1f}")

    # Use the more consistent axis, or average if both are similar
    if col_std < row_std * 0.5:
        pixel_size = round(col_spacing)
    elif row_std < col_std * 0.5:
        pixel_size = round(row_spacing)
    else:
        pixel_size = round((col_spacing + row_spacing) / 2)

    pixel_size = max(6, pixel_size)  # enforce minimum
    print(f"Chosen pixel size: {pixel_size}")

    # Find the two column peaks bracketing the click point
    left_peaks = [p for p in col_peaks if p <= cx]
    right_peaks = [p for p in col_peaks if p > cx]
    left = left_peaks[-1] if left_peaks else 0
    right = right_peaks[0] if right_peaks else w

    # Find the two row peaks bracketing the click point
    top_peaks = [p for p in row_peaks if p <= cy]
    bottom_peaks = [p for p in row_peaks if p > cy]
    top = top_peaks[-1] if top_peaks else 0
    bottom = bottom_peaks[0] if bottom_peaks else h

    # Override with the square pixel size
    seed_w = pixel_size
    seed_h = pixel_size

    print(f"Click ({click_x}, {click_y}) → pixel at ({left}, {top}), size {seed_w}×{seed_h}")

    result = seed_grid_detect(image_path, left, top, seed_w, seed_h)
    result["seed_x"] = left
    result["seed_y"] = top
    return result


def seed_grid_detect(image_path, seed_x, seed_y, seed_w, seed_h):
    """
    Detect pixel grid by expanding outward from a user-identified seed pixel.

    The user draws a box around one logical pixel. From that known pixel size
    and position, the algorithm searches outward, finding boundaries between
    logical pixels by looking for color transitions. Where adjacent pixels
    share similar colors, it falls back to the expected spacing.

    Each step is relative to the last found boundary, so the algorithm
    naturally handles grid drift across the image.

    Args:
        image_path: Path to the image
        seed_x, seed_y: Top-left corner of the seed rectangle (image coords)
        seed_w, seed_h: Width/height of the seed rectangle in image pixels

    Returns:
        Dict with col_boundaries, row_boundaries, grid_w, grid_h
    """
    img = Image.open(image_path).convert("RGB")
    img_arr = np.array(img, dtype=np.float64)
    h, w = img_arr.shape[:2]

    # Compute edge strength signals (averaged over the perpendicular axis)
    # col_edges[x] = mean color diff between column x and x+1 (across all rows)
    col_edge_strength = np.mean(
        np.sqrt(np.sum((img_arr[:, 1:, :] - img_arr[:, :-1, :]) ** 2, axis=2)),
        axis=0,
    )
    # row_edges[y] = mean color diff between row y and y+1 (across all columns)
    row_edge_strength = np.mean(
        np.sqrt(np.sum((img_arr[1:, :, :] - img_arr[:-1, :, :]) ** 2, axis=2)),
        axis=1,
    )

    # Snap seed start to nearest edge, use fixed pixel size for both axes
    pixel_size = max(seed_w, seed_h)  # caller should pass square size
    seed_x = _snap_boundary(col_edge_strength, seed_x, snap_radius=max(2, pixel_size // 4))
    seed_y = _snap_boundary(row_edge_strength, seed_y, snap_radius=max(2, pixel_size // 4))

    pixel_w = pixel_size
    pixel_h = pixel_size

    # Expand boundaries using the same pixel size for both axes
    col_bounds = _expand_boundaries(col_edge_strength, seed_x, seed_x + pixel_size, pixel_size, w)
    row_bounds = _expand_boundaries(row_edge_strength, seed_y, seed_y + pixel_size, pixel_size, h)

    grid_w = len(col_bounds) - 1
    grid_h = len(row_bounds) - 1

    print(f"Seed-based detection: {grid_w}x{grid_h} grid, pixel≈{pixel_w}x{pixel_h}")
    print(f"  Column boundaries ({len(col_bounds)}): {col_bounds[:10]}{'...' if len(col_bounds) > 10 else ''}")
    print(f"  Row boundaries ({len(row_bounds)}): {row_bounds[:10]}{'...' if len(row_bounds) > 10 else ''}")

    return {
        "col_boundaries": col_bounds,
        "row_boundaries": row_bounds,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "pixel_w": pixel_w,
        "pixel_h": pixel_h,
    }


def _snap_boundary(edge_strength, pos, snap_radius=3):
    """Snap a rough boundary position to the nearest strong edge."""
    start = max(0, pos - snap_radius)
    end = min(len(edge_strength), pos + snap_radius + 1)
    if start >= end:
        return pos
    window = edge_strength[start:end]
    return int(start + np.argmax(window))


def _expand_boundaries(edge_strength, seed_start, seed_end, expected_spacing, extent):
    """
    Expand pixel boundaries outward from a seed pixel in both directions.

    At each step: look for the strongest color edge in a window around the
    expected next boundary. If the edge is strong, use it. If weak (similar
    colors on both sides), fall back to the expected spacing.
    """
    tolerance = max(2, round(expected_spacing * 0.3))
    # Use a high percentile — real pixel boundaries are much stronger than noise
    weak_threshold = float(np.percentile(edge_strength, 70))

    boundaries = [seed_start, seed_end]

    # --- Expand forward (right / down) ---
    pos = seed_end
    while pos < extent - 2:
        expected_next = pos + expected_spacing

        search_lo = max(0, int(expected_next - tolerance))
        search_hi = min(len(edge_strength), int(expected_next + tolerance + 1))
        if search_lo >= search_hi:
            break

        window = edge_strength[search_lo:search_hi]
        best_local = int(np.argmax(window))
        best_pos = search_lo + best_local
        strength = float(window[best_local])

        if strength > weak_threshold:
            boundaries.append(best_pos)
            pos = best_pos
        else:
            # No visible edge — place boundary at expected position
            fallback = min(round(expected_next), extent)
            boundaries.append(fallback)
            pos = fallback

        if pos >= extent - 1:
            break

    # --- Expand backward (left / up) ---
    pos = seed_start
    while pos > 1:
        expected_prev = pos - expected_spacing

        search_lo = max(0, int(expected_prev - tolerance))
        search_hi = min(len(edge_strength), int(expected_prev + tolerance + 1))
        if search_lo >= search_hi:
            break

        window = edge_strength[search_lo:search_hi]
        best_local = int(np.argmax(window))
        best_pos = search_lo + best_local
        strength = float(window[best_local])

        if strength > weak_threshold:
            boundaries.insert(0, best_pos)
            pos = best_pos
        else:
            fallback = max(0, round(expected_prev))
            boundaries.insert(0, fallback)
            pos = fallback

        if pos <= 0:
            break

    # Ensure we cover the full extent
    boundaries = sorted(set(boundaries))
    if boundaries[0] > 0:
        boundaries.insert(0, 0)
    if boundaries[-1] < extent:
        boundaries.append(extent)

    return boundaries


def downscale_with_boundaries(image_path, col_bounds, row_bounds, method="mode"):
    """
    Downscale an image using non-uniform pixel boundaries.

    Unlike downscale_to_grid (uniform), this uses the actual detected
    boundaries from seed_grid_detect, handling grid drift.

    Returns:
        PIL Image at grid_w x grid_h
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)

    grid_h = len(row_bounds) - 1
    grid_w = len(col_bounds) - 1
    result = np.zeros((grid_h, grid_w, 4), dtype=np.uint8)

    for r in range(grid_h):
        r0, r1 = row_bounds[r], row_bounds[r+1]
        if r0 >= r1:
            continue
        row_strip = arr[r0:r1]
        for c in range(grid_w):
            c0, c1 = col_bounds[c], col_bounds[c+1]
            if c0 >= c1:
                continue
            block = row_strip[:, c0:c1]
            # Use center pixel — fastest and good enough for pixel art
            cy = (r0 + r1) // 2
            cx = (c0 + c1) // 2
            result[r, c] = arr[min(cy, arr.shape[0]-1), min(cx, arr.shape[1]-1)]

    return Image.fromarray(result)


def score_grid_alignment(image_path_or_array, col_bounds, row_bounds):
    """
    Score how well a grid fits the image by computing per-cell variance.

    For each grid cell, computes the variance of pixel colors within that cell.
    A perfect grid alignment means each cell is a single solid color (variance=0).
    Higher variance = worse fit.

    Fully vectorized with numpy — no Python loops over pixels.

    Args:
        image_path_or_array: file path (str) or pre-loaded numpy array (H,W,3) float32
        col_bounds: list of column boundary positions
        row_bounds: list of row boundary positions

    Returns: (score, 0, 0)  — shift values kept for API compat but always 0
    """
    if isinstance(image_path_or_array, np.ndarray):
        original = image_path_or_array
    else:
        original = np.array(Image.open(image_path_or_array).convert("RGB"), dtype=np.float32)

    h, w = original.shape[:2]
    grid_h = len(row_bounds) - 1
    grid_w = len(col_bounds) - 1

    if grid_h < 1 or grid_w < 1:
        return float('inf'), 0, 0

    # Build the reconstructed image using vectorized operations:
    # 1. Sample center pixel of each cell (downscale)
    # 2. Paint each cell with that color (upscale via index mapping)
    # 3. MSE = mean((original - reconstructed)^2)

    # Build pixel-to-cell mapping arrays for fast reconstruction
    # Row mapping: for each pixel row, which grid row does it belong to?
    row_map = np.zeros(h, dtype=np.int32)
    row_centers = np.zeros(grid_h, dtype=np.int32)
    for r in range(grid_h):
        r0, r1 = int(row_bounds[r]), min(int(row_bounds[r + 1]), h)
        if r0 < r1:
            row_map[r0:r1] = r
            row_centers[r] = min((r0 + r1) // 2, h - 1)

    col_map = np.zeros(w, dtype=np.int32)
    col_centers = np.zeros(grid_w, dtype=np.int32)
    for c in range(grid_w):
        c0, c1 = int(col_bounds[c]), min(int(col_bounds[c + 1]), w)
        if c0 < c1:
            col_map[c0:c1] = c
            col_centers[c] = min((c0 + c1) // 2, w - 1)

    # Downscale: sample center pixels — fully vectorized
    downscaled = original[np.ix_(row_centers, col_centers)]  # (grid_h, grid_w, 3)

    # Upscale: map each pixel to its cell's color — fully vectorized
    reconstructed = downscaled[np.ix_(row_map, col_map)]  # (h, w, 3)

    # MSE
    diff = original - reconstructed
    score = float(np.mean(diff * diff))
    return score, 0, 0


def _find_edges_from_peak(differences, approx_spacing, window_size, extent):
    """
    Find regularly spaced edges starting from the strongest signal region.

    Instead of starting from position 0 (which may be empty background),
    this finds the position with the strongest color difference signal,
    then expands outward in both directions using the known spacing.

    Args:
        differences: Array of difference values
        approx_spacing: Approximate spacing between edges
        window_size: Size of the search window around expected positions
        extent: Total extent (width or height) of the image

    Returns:
        List of detected edge positions (sorted)
    """
    if len(differences) == 0 or approx_spacing < 2:
        return []

    # Find the region with the strongest signal
    # Use a sliding window of approx_spacing to find the area with highest total signal
    kernel_size = max(1, int(approx_spacing))
    if kernel_size >= len(differences):
        return []

    # Convolve with a box filter to find the densest signal region
    cumsum = np.cumsum(differences)
    windowed = cumsum[kernel_size:] - cumsum[:-kernel_size]
    peak_center = int(np.argmax(windowed)) + kernel_size // 2

    # Find the nearest strong edge to use as anchor
    search_lo = max(0, peak_center - int(approx_spacing))
    search_hi = min(len(differences), peak_center + int(approx_spacing))
    window = differences[search_lo:search_hi]
    if len(window) == 0:
        return []
    anchor = search_lo + int(np.argmax(window))

    edges = [anchor]

    # Expand forward from anchor
    pos = anchor
    while True:
        expected = pos + approx_spacing
        lo = max(0, int(expected - window_size))
        hi = min(len(differences), int(expected + window_size + 1))
        if lo >= len(differences) - 1 or lo >= hi:
            break
        w = differences[lo:hi]
        if len(w) == 0 or (np.max(w) == np.min(w)):
            # No variation — place at expected position if within bounds
            fallback = int(round(expected))
            if fallback >= extent:
                break
            edges.append(fallback)
            pos = fallback
        else:
            best = lo + int(np.argmax(w))
            edges.append(best)
            pos = best
        if pos >= extent - 2:
            break

    # Expand backward from anchor
    pos = anchor
    while True:
        expected = pos - approx_spacing
        lo = max(0, int(expected - window_size))
        hi = min(len(differences), int(expected + window_size + 1))
        if hi <= 0 or lo >= hi:
            break
        w = differences[lo:hi]
        if len(w) == 0 or (np.max(w) == np.min(w)):
            fallback = int(round(expected))
            if fallback <= 0:
                break
            edges.append(fallback)
            pos = fallback
        else:
            best = lo + int(np.argmax(w))
            edges.append(best)
            pos = best
        if pos <= 1:
            break

    return sorted(set(edges))


def score_grid_batch(image_path, grid_sizes, window_size=5):
    """
    Score multiple grid sizes at once, sharing image load and diff computation.
    Returns list of (grid_size, score, col_bounds, row_bounds).
    """
    # Load image once
    image_array = edge_detector.load_image(image_path)
    h, w = image_array.shape[:2]
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=2)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Compute diffs once (the expensive part)
    row_diffs = edge_detector.calculate_differences(
        image_array, axis=0, algorithm=edge_detector.DiffAlgorithm.EUCLIDEAN,
    )
    col_diffs = edge_detector.calculate_differences(
        image_array, axis=1, algorithm=edge_detector.DiffAlgorithm.EUCLIDEAN,
    )

    # Pre-load as float32 for scoring
    original = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32)

    results = []
    for grid_size in grid_sizes:
        approx_spacing = w / grid_size
        row_spacing = max(2, round(approx_spacing))
        col_spacing = row_spacing

        # Find edges starting from the strongest signal region
        col_edges = _find_edges_from_peak(col_diffs, col_spacing, window_size, w)
        row_edges = _find_edges_from_peak(row_diffs, row_spacing, window_size, h)

        # Fall back to uniform grid if edge detection finds too few edges
        min_expected = max(2, grid_size * 3 // 4)
        if len(col_edges) < min_expected:
            col_edges = [round(i * w / grid_size) for i in range(1, grid_size)]
        if len(row_edges) < min_expected:
            row_edges = [round(i * h / grid_size) for i in range(1, grid_size)]

        col_bounds = sorted(set([0] + [int(e) for e in col_edges] + [w]))
        row_bounds = sorted(set([0] + [int(e) for e in row_edges] + [h]))

        # Score using pre-loaded image array
        score, _, _ = score_grid_alignment(original, col_bounds, row_bounds)
        results.append((grid_size, score, col_bounds, row_bounds))

    return results


def create_seed_grid_preview(image_path, col_bounds, row_bounds, output_path, preview_size=512):
    """
    Create grid overlay + pixel preview for a seed-detected (non-uniform) grid.

    Returns dict with grid_overlay and pixel_preview paths.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size

    # --- Grid overlay ---
    bg = _make_checker_bg(preview_size, preview_size, max(4, preview_size // 32))
    img_resized = img.resize((preview_size, preview_size), Image.NEAREST)
    bg.paste(img_resized, (0, 0), img_resized)

    overlay = np.array(bg)
    grid_color = np.array([255, 40, 40, 255], dtype=np.uint8)

    for bx in col_bounds:
        px = round(bx * preview_size / w)
        if 0 < px < preview_size:
            overlay[:, px, :] = grid_color
    for by in row_bounds:
        py = round(by * preview_size / h)
        if 0 < py < preview_size:
            overlay[py, :, :] = grid_color

    Image.fromarray(overlay).save(output_path, "PNG")

    # --- Pixel preview ---
    downscaled = downscale_with_boundaries(image_path, col_bounds, row_bounds)
    grid_w = len(col_bounds) - 1
    grid_h = len(row_bounds) - 1

    small_bg = _make_checker_bg(grid_w, grid_h, max(1, min(grid_w, grid_h) // 8))
    small_bg.paste(downscaled, (0, 0), downscaled)
    pixel_preview = small_bg.resize((preview_size, preview_size), Image.NEAREST)

    base, ext = os.path.splitext(output_path)
    pixel_path = f"{base}_pixel{ext}"
    pixel_preview.save(pixel_path, "PNG")

    return {"grid_overlay": output_path, "pixel_preview": pixel_path}


def detect_grid_edges(image_path, approx_spacing=None, window_size=5, offset_x=0, offset_y=0):
    """
    Detect grid edges using color differences and find_edges_with_window.

    This is the original grid detection algorithm: it computes color differences
    between adjacent rows/columns, auto-detects approximate spacing from the
    difference signal peaks, then uses find_edges_with_window() to find edges
    that snap to actual color transitions — allowing variability rather than
    forcing a perfectly uniform grid.

    Args:
        image_path: Path to the image
        approx_spacing: Approximate pixel block size in source image pixels.
                        If None, auto-detects from edge peaks.
        window_size: Search window around expected positions (default: 2)
        offset_x, offset_y: Starting position for edge search

    Returns:
        Dict with col_boundaries, row_boundaries (lists of pixel positions),
        plus detected_row_spacing and detected_col_spacing.
    """
    image_array = edge_detector.load_image(image_path)
    h, w = image_array.shape[:2]

    # Work with RGB
    if len(image_array.shape) == 2:
        image_array = np.stack([image_array] * 3, axis=2)
    elif image_array.shape[2] == 4:
        image_array = image_array[:, :, :3]

    # Compute color differences
    row_diffs = edge_detector.calculate_differences(
        image_array, axis=0, algorithm=edge_detector.DiffAlgorithm.EUCLIDEAN,
    )
    col_diffs = edge_detector.calculate_differences(
        image_array, axis=1, algorithm=edge_detector.DiffAlgorithm.EUCLIDEAN,
    )

    # Auto-detect spacing from edge peaks if not provided
    if approx_spacing is None:
        row_edges_initial = edge_detector.find_edges(row_diffs, 90)
        if len(row_edges_initial) >= 11:
            row_spacing = int(np.median(np.diff(row_edges_initial[:10])))
        else:
            row_spacing = 13
        row_spacing = max(5, row_spacing)

        col_edges_initial = edge_detector.find_edges(col_diffs, 90)
        if len(col_edges_initial) >= 11:
            col_spacing = int(np.median(np.diff(col_edges_initial[:10])))
        else:
            col_spacing = 13
        col_spacing = max(5, col_spacing)

        print(f"Auto-detected spacing: rows={row_spacing}, cols={col_spacing}")
    else:
        row_spacing = max(2, round(approx_spacing))
        col_spacing = row_spacing

    # Find edges using window search — snaps to actual color transitions
    col_edges = find_edges_with_window(col_diffs, col_spacing, window_size, start_pos=offset_x)
    row_edges = find_edges_with_window(row_diffs, row_spacing, window_size, start_pos=offset_y)

    # Fall back to uniform grid if edge detection finds too few edges
    expected_cols = max(2, w // col_spacing // 2)
    expected_rows = max(2, h // row_spacing // 2)
    if len(col_edges) < expected_cols:
        col_edges = [round(i * col_spacing) for i in range(1, w // col_spacing)]
    if len(row_edges) < expected_rows:
        row_edges = [round(i * row_spacing) for i in range(1, h // row_spacing)]

    # Convert edges to boundaries (add 0 at start and extent at end)
    # Convert to plain Python ints for JSON serialization
    col_bounds = sorted(set([0] + [int(e) for e in col_edges] + [w]))
    row_bounds = sorted(set([0] + [int(e) for e in row_edges] + [h]))

    grid_w = len(col_bounds) - 1
    grid_h = len(row_bounds) - 1

    print(f"Edge-based detection: {grid_w}x{grid_h} grid, row_spacing={row_spacing}, col_spacing={col_spacing}")

    return {
        "col_boundaries": col_bounds,
        "row_boundaries": row_bounds,
        "grid_w": grid_w,
        "grid_h": grid_h,
        "detected_row_spacing": int(row_spacing),
        "detected_col_spacing": int(col_spacing),
    }


def downscale_to_grid(image_path, grid_size, method="mode", offset_x=0, offset_y=0):
    """
    Downscale an image to the detected logical pixel grid.

    Args:
        image_path: Path to the image
        grid_size: Target grid size (e.g. 32 for 32x32)
        method: "mode" (most common color), "average", or "center"
        offset_x, offset_y: Grid origin offset in source pixels

    Returns:
        PIL Image (may be slightly larger than grid_size if offset causes extra cells)
    """
    img = Image.open(image_path).convert("RGBA")
    arr = np.array(img)
    h, w = arr.shape[:2]
    pixel_block = w / grid_size

    # Compute block boundaries with offset
    col_bounds = []
    x = offset_x
    while x < w:
        col_bounds.append(round(x))
        x += pixel_block
    col_bounds.append(w)
    if col_bounds[0] > 0:
        col_bounds.insert(0, 0)

    row_bounds = []
    y = offset_y
    while y < h:
        row_bounds.append(round(y))
        y += pixel_block
    row_bounds.append(h)
    if row_bounds[0] > 0:
        row_bounds.insert(0, 0)

    out_w = len(col_bounds) - 1
    out_h = len(row_bounds) - 1

    result = np.zeros((out_h, out_w, 4), dtype=np.uint8)

    for r in range(out_h):
        for c in range(out_w):
            block = arr[row_bounds[r]:row_bounds[r+1], col_bounds[c]:col_bounds[c+1]]

            if block.size == 0:
                continue

            if method == "center":
                cy = (row_bounds[r] + row_bounds[r+1]) // 2
                cx = (col_bounds[c] + col_bounds[c+1]) // 2
                result[r, c] = arr[min(cy, h-1), min(cx, w-1)]

            elif method == "average":
                result[r, c] = block.reshape(-1, 4).mean(axis=0).astype(np.uint8)

            else:  # mode
                pixels = block.reshape(-1, 4)
                quantized = (pixels // 8) * 8
                tuples = [tuple(p) for p in quantized]
                from collections import Counter
                most_common = Counter(tuples).most_common(1)[0][0]
                target = np.array(most_common)
                dists = np.sum(np.abs(quantized.astype(int) - target.astype(int)), axis=1)
                best_idx = np.argmin(dists)
                result[r, c] = pixels[best_idx]

    return Image.fromarray(result)


def create_grid_preview(image_path, grid_size, output_path, preview_size=512, offset_x=0, offset_y=0):
    """
    Create grid overlay and pixel preview images separately.

    offset_x, offset_y: grid origin offset in source image pixels (0 to pixel_block-1).

    Returns:
        Dict with 'grid_overlay' and 'pixel_preview' paths.
    """
    img = Image.open(image_path).convert("RGBA")
    w, h = img.size
    pixel_block = w / grid_size

    # --- Grid overlay ---
    bg = _make_checker_bg(preview_size, preview_size, max(4, preview_size // 32))
    img_resized = img.resize((preview_size, preview_size), Image.NEAREST)
    bg.paste(img_resized, (0, 0), img_resized)

    overlay = np.array(bg)
    grid_color = np.array([255, 40, 40, 255], dtype=np.uint8)

    # Grid lines with offset
    scale = preview_size / w
    x = offset_x
    while x < w:
        px = round(x * scale)
        if 0 < px < preview_size:
            overlay[:, px, :] = grid_color
        x += pixel_block
    y = offset_y
    while y < h:
        py = round(y * scale)
        if 0 < py < preview_size:
            overlay[py, :, :] = grid_color
        y += pixel_block

    Image.fromarray(overlay).save(output_path, "PNG")

    # --- Pixel preview ---
    downscaled = downscale_to_grid(image_path, grid_size, method="mode", offset_x=offset_x, offset_y=offset_y)
    dw, dh = downscaled.size
    small_bg = _make_checker_bg(dw, dh, max(1, min(dw, dh) // 8))
    small_bg.paste(downscaled, (0, 0), downscaled)
    pixel_preview = small_bg.resize((preview_size, preview_size), Image.NEAREST)

    base, ext = os.path.splitext(output_path)
    pixel_path = f"{base}_pixel{ext}"
    pixel_preview.save(pixel_path, "PNG")

    return {"grid_overlay": output_path, "pixel_preview": pixel_path}


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
