"""Membrane thickness analysis of cryo-ET data in three stages:

  1. Surface extraction — generate a marching-cubes mesh from a membrane segmentation, perform normal refinement and surface split.
  2. Geometric matching — GPU/CPU normal-cone matching between the separated surfaces.
  3. Intensity profile analysis — extract intensity profiles from the tomogram along matched pairs,
      identify minima, maxima, and inflection points to calculate thickness.
"""
import os
import time
import json
import logging
from pathlib import Path
import traceback
from typing import Literal, Sequence, Any
import warnings

import numpy as np
import pandas as pd
from cryocat.core import cryomap
from cryocat._types import MapSource, PathOrStr
from cryocat.core.surface import Mesh, DiscreteSurface

from scipy.spatial import KDTree as ScipyKDTree
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d

import numba
from numba import cuda
import math


inflection_point_method = frozenset(
    {"max_max", "max_anchor", "anchor_max"}
)
method_max_max = frozenset({"max_max"})
method_max_anchor = frozenset({"max_anchor", "anchor_max"})
method_minima_only = frozenset({"minima_only"})


def _matched_table_base_name(thickness_csv: str| Path) -> str:
    """Stem of a matched-points / thickness CSV with standard suffixes removed (matches ``save_int_results`` naming)."""
    base_name = Path(thickness_csv).stem
    for suffix in (
        "_matched_points_2to1",
        "_matched_points",
        "_thickness_2to1",
        "_thickness",
        "_membrane_thickness",
    ):
        if base_name.endswith(suffix):
            base_name = base_name[: -len(suffix)]
            break
    return base_name

def _matched_point_geometry_counts(thickness_df: pd.DataFrame) -> tuple[int, int] | None:
    """Return (n_rows_csv, n_valid_coordinate_pairs) if geometry columns exist, else ``None``."""
    cols = ("x1_voxel", "y1_voxel", "z1_voxel", "x2_voxel", "y2_voxel", "z2_voxel")
    if not set(cols).issubset(thickness_df.columns):
        return None
    n_rows = len(thickness_df)
    p1 = thickness_df[list(cols[:3])].to_numpy(dtype=float)
    p2 = thickness_df[list(cols[3:])].to_numpy(dtype=float)
    valid_mask = ~(np.isnan(p1).any(axis=1) | np.isnan(p2).any(axis=1))
    return (n_rows, int(valid_mask.sum()))

def _infer_membrane_suffix_from_csv(
    thickness_csv: PathOrStr,
    segmentation_path: PathOrStr,
) -> str | None:
    """
    If ``thickness_csv`` stem strips to ``{{segStem}}_{label}``, return ``label`` (from pipeline naming).

    Example: ``2738_seg_OMM_matched_points.csv`` → ``OMM`` when segmentation stem is ``2738_seg``.
    """
    seg_stem = Path(segmentation_path).stem
    base = _matched_table_base_name(Path(thickness_csv))
    pref = seg_stem + "_"
    if base.startswith(pref) and len(base) > len(pref):
        return base[len(pref) :].lstrip("_").strip() or None
    return None

def _sanitize_log_fragment(s: str) -> str:
    """Make a substring safe for filenames (Windows-macOS-safe)."""
    if not s or not str(s).strip():
        return "memthick"
    bad = '\\/:*?"<>|\t\r\n '
    out = "".join("_" if c in bad else c for c in str(s))
    while "__" in out:
        out = out.replace("__", "_")
    return out.strip("_") or "memthick"

def pipeline_analysis_log_filename(
    segmentation_path: PathOrStr | None = None,
    tomogram_path: PathOrStr | None = None,
    *,
    membrane_labels: str | Sequence[str] | None = None,
) -> str:
    """
    Chronological pipeline log basename::

        {segmentation_stem}_{membrane_labels}_{tomogram_stem}_analysis.log

    ``membrane_labels`` may be one name or an iterable (multiple names sorted and joined).

    Missing pieces are omitted left-to-right; see fallbacks below.
    """
    seg_stem_p = Path(segmentation_path).expanduser() if segmentation_path else None
    tomo_stem_p = Path(tomogram_path).expanduser() if tomogram_path else None

    label_token: str | None = None
    if membrane_labels is not None:
        if isinstance(membrane_labels, str):
            if membrane_labels.strip():
                label_token = _sanitize_log_fragment(membrane_labels.strip())
        else:
            labs = [_sanitize_log_fragment(str(x).strip()) for x in membrane_labels if str(x).strip()]
            if labs:
                label_token = "_".join(sorted(labs))

    parts: list[str] = []
    if seg_stem_p is not None:
        parts.append(_sanitize_log_fragment(seg_stem_p.stem))
    if label_token:
        parts.append(label_token)
    if tomo_stem_p is not None:
        parts.append(_sanitize_log_fragment(tomo_stem_p.stem))

    if parts:
        return "_".join(parts) + "_analysis.log"
    return "memthick_analysis.log"

def _boundary_stats_file_preamble_lines(
    matched_points_csv: Path,
    *,
    tomogram_path: PathOrStr | None = None,
    segmentation_path: PathOrStr | None = None,
) -> str:
    """Provenance block prepended to ``*_boundary_stats.txt`` (before ``print_summary`` capture)."""
    lines = [
        "=== Boundary statistics ===",
        f"Matched-point pairs table: {matched_points_csv.name}",
        f"  Path: {matched_points_csv.expanduser().resolve()}",
    ]
    if tomogram_path is not None:
        tp = Path(tomogram_path).expanduser().resolve()
        lines.extend((f"Tomogram MRC: {tp.name}", f"  Path: {tp}"))
    if segmentation_path is not None:
        sp = Path(segmentation_path).expanduser().resolve()
        lines.extend((f"Segmentation MRC: {sp.name}", f"  Path: {sp}"))
    lines.extend(("=" * 50, ""))
    return "\n".join(lines)


#############################################
# Logging and Utility Functions For Thickness Measurement Pipeline
#############################################

def setup_logger(
    output_path: PathOrStr,
    name: str = "MembraneThickness",
    *,
    log_filename: str | Path | None = "thickness_analysis.log",
) -> logging.Logger:
    """
    Set up logger for the analysis with both file and console handlers.

    Parameters
    ----------
    output_path : PathOrStr
        Directory where log file will be saved (ignored if ``log_filename`` is absolute)
    name : str, default "MembraneThickness"
        Name of the logger
    log_filename : str | pathlib.Path | None, default "thickness_analysis.log"
        File name relative to ``output_path``, **or** an absolute path. ``None``
        restores the historical default basename ``thickness_analysis.log``.

    Returns
    -------
    logging.Logger
        Configured logger instance with file and console handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    if log_filename is None:
        log_path = output_path / "thickness_analysis.log"
    else:
        lp = Path(log_filename)
        log_path = lp if lp.is_absolute() else (output_path / lp)

    file_handler = logging.FileHandler(log_path)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def validate_seg_tomo_compatibility(
    segmentation_map: MapSource,
    tomogram_map: MapSource,
    tolerance: float = 0.01,
    logger: logging.Logger = None,
) -> tuple[bool, dict]:
    """
    Validate compatibility between segmentation and tomogram files.

    Checks dimensions and pixel size compatibility between a segmentation
    MRC file and its corresponding tomogram before sampling intensity profiles.

    Parameters
    ----------
    segmentation_map : MapSource
        Path to the MRC segmentation file, or an ndarray.
    tomogram_map : MapSource
        Path to the MRC tomogram file, or an ndarray.
    tolerance : float, default 0.01
        Tolerance for pixel size comparison (in nanometers).
    logger : logging.Logger, optional
        Logger instance for status messages.

    Returns
    -------
    compatible : bool
        True if files are compatible for pixel-aligned tomogram sampling.
    details : dict
        Dictionary containing compatibility details:
        - 'segmentation_shape': tuple of segmentation dimensions (ZYX)
        - 'tomogram_shape': tuple of tomogram dimensions (ZYX)
        - 'segmentation_pixel_size': float, pixel size in nm
        - 'tomogram_pixel_size': float, pixel size in nm
        - 'dimensions_match': bool, whether shapes are identical
        - 'pixel_sizes_match': bool, whether pixel sizes are within tolerance
        - 'error': str, error message if validation failed
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    try:
        log_msg("Validating segmentation and tomogram dimensions and pixel size compatibility...")

        seg_shape, seg_pixel_size_a, _ = cryomap.get_metadata(segmentation_map)
        seg_pixel_size = seg_pixel_size_a / 10.0  # Å → nm

        tomo_shape, tomo_pixel_size_a, _ = cryomap.get_metadata(tomogram_map)
        tomo_pixel_size = tomo_pixel_size_a / 10.0  # Å → nm

        dims_match = seg_shape == tomo_shape
        pixel_size_match = abs(seg_pixel_size - tomo_pixel_size) < tolerance
        compatible = dims_match and pixel_size_match

        details = {
            "segmentation_shape": seg_shape,
            "tomogram_shape": tomo_shape,
            "segmentation_pixel_size": seg_pixel_size,
            "tomogram_pixel_size": tomo_pixel_size,
            "dimensions_match": dims_match,
            "pixel_sizes_match": pixel_size_match,
        }

        if compatible:
            log_msg("✓ Segmentation and tomogram are compatible in dimensions and pixel size")
        else:
            log_msg("✗ Segmentation and tomogram are not compatible:")
            if not dims_match:
                log_msg(f"  Dimension mismatch: {seg_shape} vs {tomo_shape}")
            if not pixel_size_match:
                log_msg(f"  Pixel size mismatch: {seg_pixel_size:.4f} vs {tomo_pixel_size:.4f} nm")
                log_msg(f"  Difference: {abs(seg_pixel_size - tomo_pixel_size):.4f} nm (tolerance: {tolerance:.4f} nm)")

        return compatible, details

    except Exception as e:
        error_msg = f"Error during compatibility check: {e}"
        log_msg(error_msg)
        return False, {"error": error_msg}

def verify_and_save_outputs(
    aligned_vertices: np.ndarray,
    aligned_normals: np.ndarray,
    vertex_volume: np.ndarray,
    surface1_mask: np.ndarray,
    surface2_mask: np.ndarray,
    membrane_name: str,
    base_name: str,
    output_path: PathOrStr,
    pixel_size: float,
    origin: tuple,
    save_vertices_mrc: bool = False,
    logger: logging.Logger = None,
) -> bool:
    """
    Save surface analysis outputs in multiple formats.

    Parameters
    ----------
    aligned_vertices : np.ndarray
        2D array (N, 3) of vertex coordinates in voxel units (ZYX order)
    aligned_normals : np.ndarray
        2D array (N, 3) of normal vectors (ZYX order)
    vertex_volume : np.ndarray
        3D binary volume marking vertex positions
    surface1_mask : np.ndarray
        1D boolean array for surface 1 assignment
    surface2_mask : np.ndarray
        1D boolean array for surface 2 assignment
    membrane_name : str
        Name identifier for this membrane
    base_name : str
        Base filename for outputs
    output_path : str
        Output directory path
    pixel_size : float
        Voxel size in nanometers
    origin : tuple
        Origin coordinates (x, y, z) in nanometers
    save_vertices_mrc : bool, default False
        Whether to save vertex volume as MRC file
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    bool
        True if all requested outputs saved successfully
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    log_msg(f"\nSaving outputs for {membrane_name}:")
    log_msg(
        f"Vertices: {len(aligned_vertices)}, Surface 1: {np.sum(surface1_mask)}, Surface 2: {np.sum(surface2_mask)}"
    )

    try:
        # Scale vertices once for all uses. aligned_vertices may be floating-point
        # mesh coordinates in voxel units.
        scaled_vertices = aligned_vertices * pixel_size + np.array(origin)
        rounded_vertices = np.rint(aligned_vertices).astype(np.int32)

        # Save MRC file if requested
        if save_vertices_mrc:
            mrc_output = os.path.join(output_path, f"{base_name}_{membrane_name}_vertices.mrc")
            cryomap.write(vertex_volume.astype(np.int16), mrc_output,
                          transpose=False, pixel_size=pixel_size * 10.0)  # nm → Å
            log_msg(f"Saved MRC to {mrc_output}")

        # Save CSV with coordinates, normals, and surface masks (ZYX → XYZ conversion)
        csv_output = os.path.join(output_path, f"{base_name}_{membrane_name}_vertices_normals.csv")
        df = pd.DataFrame(
            {
                "x_voxel": aligned_vertices[:, 2],
                "y_voxel": aligned_vertices[:, 1],
                "z_voxel": aligned_vertices[:, 0],
                "x_voxel_int": rounded_vertices[:, 2],
                "y_voxel_int": rounded_vertices[:, 1],
                "z_voxel_int": rounded_vertices[:, 0],
                "x_physical": scaled_vertices[:, 2],
                "y_physical": scaled_vertices[:, 1],
                "z_physical": scaled_vertices[:, 0],
                "normal_x": aligned_normals[:, 2],
                "normal_y": aligned_normals[:, 1],
                "normal_z": aligned_normals[:, 0],
                "surface1": surface1_mask,
                "surface2": surface2_mask,
            }
        )
        df.to_csv(csv_output, index=False)
        log_msg(f"Saved CSV with {len(df)} vertices to {csv_output}")

    except Exception as e:
        log_msg(f"Error saving outputs: {e}")
        traceback.print_exc()
        return False

    return True


#############################################
# On stats
#############################################


def generate_matching_statistics(
    thickness_results: np.ndarray,
    valid_mask: np.ndarray,
    points: np.ndarray,
    surface1_mask: np.ndarray,
    surface2_mask: np.ndarray,
    pixel_size: float,
) -> dict:
    """
    Generate comprehensive statistics for thickness measurements.

    Parameters
    ----------
    thickness_results : np.ndarray
        1D array of thickness measurements in physical units (nm)
    valid_mask : np.ndarray
        1D boolean array indicating valid measurements
    point_pairs : np.ndarray
        1D array of indices for paired points
    points : np.ndarray
        2D array (N, 3) of vertex coordinates in voxel units
    surface1_mask : np.ndarray
        1D boolean array for surface 1 points
    surface2_mask : np.ndarray
        1D boolean array for surface 2 points
    pixel_size : float
        Voxel size in nanometers for coordinate scaling

    Returns
    -------
    dict
        Statistics dictionary containing:
        - total_points, surface1_points, surface2_points, valid_measurements
        - coverage_percentage
        - thickness statistics (mean, std, median, min, max, percentiles)
        - thickness_histogram with counts and bin_edges
        - spatial_distribution with mean and std coordinates
    """
    stats = {
        "total_points": len(points),
        "surface1_points": np.sum(surface1_mask),
        "surface2_points": np.sum(surface2_mask),
        "valid_measurements": np.sum(valid_mask),
        "coverage_percentage": np.sum(valid_mask) / np.sum(surface1_mask) * 100,
    }

    valid_thicknesses = thickness_results[valid_mask]
    if len(valid_thicknesses) > 0:
        stats.update(
            {
                "mean_thickness": np.mean(valid_thicknesses),
                "std_thickness": np.std(valid_thicknesses),
                "median_thickness": np.median(valid_thicknesses),
                "min_thickness": np.min(valid_thicknesses),
                "max_thickness": np.max(valid_thicknesses),
                "percentile_25": np.percentile(valid_thicknesses, 25),
                "percentile_75": np.percentile(valid_thicknesses, 75),
            }
        )

        # Calculate histogram data
        hist, bins = np.histogram(valid_thicknesses, bins=50)
        stats["thickness_histogram"] = {"counts": hist.tolist(), "bin_edges": bins.tolist()}

        # Spatial distribution analysis
        valid_points = points[valid_mask]
        stats["spatial_distribution"] = {
            "x_mean": np.mean(valid_points[:, 0]) * pixel_size,
            "y_mean": np.mean(valid_points[:, 1]) * pixel_size,
            "z_mean": np.mean(valid_points[:, 2]) * pixel_size,
            "x_std": np.std(valid_points[:, 0]) * pixel_size,
            "y_std": np.std(valid_points[:, 1]) * pixel_size,
            "z_std": np.std(valid_points[:, 2]) * pixel_size,
        }

    return stats

def save_matching_statistics(
    stats: dict,
    output_path: PathOrStr,
    logger: logging.Logger = None,
    *,
    matching_params: dict[str, Any] | None = None,
) -> None:
    """
    Save geometric matched-point statistics to formatted text file.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from generate_matching_statistics()
    output_path : PathOrStr
        Path where statistics file will be saved (typically ``*_stats.txt``).
    logger : logging.Logger, optional
        Logger instance for status messages
    matching_params : dict, optional
        Matching configuration and provenance (written as the first section when provided).
        Expected keys typically include distances/angles/direction/backend and input paths,
        populated by ``match_points``.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    with open(output_path, "w") as f:
        f.write("=== Matched segmentation points statistics ===\n\n")

        if matching_params:
            f.write("=== Parameters ===\n")
            for key in sorted(matching_params.keys()):
                val = matching_params[key]
                if val is None:
                    line = "(not set)"
                else:
                    line = str(val)
                f.write(f"{key}: {line}\n")
            f.write("\n")

        f.write("General:\n")
        f.write(f"Total points: {stats['total_points']}\n")
        f.write(f"Surface 1 points: {stats['surface1_points']}\n")
        f.write(f"Surface 2 points: {stats['surface2_points']}\n")
        f.write(f"Matched point pairs: {stats['valid_measurements']}\n")
        f.write(f"Coverage percentage: {stats['coverage_percentage']:.2f}%\n\n")

        if "mean_thickness" in stats:
            f.write("Matched-point distances (nm):\n")
            f.write(f"Mean distance: {stats['mean_thickness']:.2f} ± {stats['std_thickness']:.2f}\n")
            f.write(f"Median distance: {stats['median_thickness']:.2f}\n")
            f.write(f"Range: {stats['min_thickness']:.2f} - {stats['max_thickness']:.2f}\n")
            f.write(f"Interquartile range: {stats['percentile_25']:.2f} - {stats['percentile_75']:.2f}\n\n")

            f.write("Matched-point distance histogram (nm):\n")
            hist = stats["thickness_histogram"]
            for count, (left, right) in zip(hist["counts"], zip(hist["bin_edges"][:-1], hist["bin_edges"][1:])):
                f.write(f"{left:.2f}-{right:.2f}: {count}\n")

        log_msg(f"Statistics saved to {output_path}")


#############################################
# Surface Processing Classes and Functions
#############################################


def create_vertex_volume(aligned_vertices: np.ndarray, membrane_mask_shape: tuple) -> np.ndarray:
    """
    Create binary volume marking vertex positions.

    Parameters
    ----------
    aligned_vertices : np.ndarray
        2D array (N, 3) of vertex coordinates in voxel units
    membrane_mask_shape : tuple
        Shape (z, y, x) of the target volume

    Returns
    -------
    np.ndarray
        3D binary volume with 1 at vertex positions, 0 elsewhere
    """
    vertex_volume = np.zeros(membrane_mask_shape, dtype=np.uint8)
    voxel_vertices = np.rint(aligned_vertices).astype(np.int32)
    for x, y, z in voxel_vertices:
        if 0 <= x < vertex_volume.shape[0] and 0 <= y < vertex_volume.shape[1] and 0 <= z < vertex_volume.shape[2]:
            vertex_volume[x, y, z] = 1
    return vertex_volume


def is_surface_point(point: np.ndarray, segmentation: np.ndarray) -> bool:
    """
    Validate if a point is truly on the segmentation surface boundary.

    This function determines whether a given voxel position lies on the
    surface of a binary segmentation by checking if it is inside the
    segmentation and has at least one immediate neighbor outside it.

    The function uses a 6-connected neighborhood check (face neighbors only)
    to determine surface membership. A point is considered on the surface
    if it satisfies both conditions:
    1. The point itself is inside the segmentation (True/1)
    2. At least one of its 6 face neighbors is outside the segmentation (False/0)

    Parameters
    ----------
    point : array-like
        Coordinates [x, y, z] of the point to check
    segmentation : np.ndarray
        3D binary segmentation volume where True/1 indicates inside
        the segmentation and False/0 indicates outside. Shape should
        be (Z, Y, X) in voxel coordinates.

    Returns
    -------
    bool
        True if the point is on the surface boundary, False otherwise.
        A point is on the surface if it's inside the segmentation and
        has at least one face neighbor outside the segmentation.

    Raises
    ------
    IndexError
        If the point coordinates are outside the segmentation array bounds.
    ValueError
        If the segmentation array is not 3D or contains non-boolean values.

    Notes
    -----
    - Uses 6-connected neighborhood (face neighbors only, not edge or corner)
    - Face neighbors are at positions: [x±1, y, z], [x, y±1, z], [x, y, z±1]
    - Edge and corner neighbors are not considered for surface detection
    - This conservative approach ensures only true boundary points are identified
    - Processing time is O(1) per point (constant time)

    """
    x, y, z = point

    # First check if point is in segmentation
    if not segmentation[x, y, z]:
        return False

    # Check if any immediate neighbor is outside segmentation
    neighbors = [[x + 1, y, z], [x - 1, y, z], [x, y + 1, z], [x, y - 1, z], [x, y, z + 1], [x, y, z - 1]]

    for nx, ny, nz in neighbors:
        if (
            0 <= nx < segmentation.shape[0]
            and 0 <= ny < segmentation.shape[1]
            and 0 <= nz < segmentation.shape[2]
            and not segmentation[nx, ny, nz]
        ):
            return True

    return False

def _filter_to_segmentation_boundary(
    vertices: np.ndarray,
    normals: np.ndarray,
    membrane_mask: np.ndarray,
    logger: logging.Logger = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Round mesh vertices to integer voxels and keep only those on the segmentation boundary.

    Used when ``snap_vertices_to_boundary=True``. Marching-cubes vertices are
    floating-point and may land inside, outside, or exactly on the segmentation surface.
    This filter rounds each vertex to its nearest integer voxel and discards any that do
    not satisfy ``is_surface_point`` (inside the segmentation with at least one outside
    face-neighbor). Duplicates at the same integer position are also collapsed.

    Parameters
    ----------
    vertices : np.ndarray
        (N, 3) float array of marching-cubes vertex coordinates in voxel units.
    normals : np.ndarray
        (N, 3) array of corresponding normal vectors.
    membrane_mask : np.ndarray
        3-D binary mask of the membrane label (ZYX order).
    logger : logging.Logger, optional

    Returns
    -------
    boundary_vertices : np.ndarray
        (M, 3) int32 array, M ≤ N, each row a unique boundary voxel.
    boundary_normals : np.ndarray
        (M, 3) float32 array of corresponding normals.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    int_verts = np.rint(vertices).astype(np.int32)
    seen = {}
    for i, v in enumerate(int_verts):
        key = (v[0], v[1], v[2])
        if key not in seen and is_surface_point(v, membrane_mask):
            seen[key] = i

    if not seen:
        log_msg("WARNING: _filter_to_segmentation_boundary: no boundary voxels found")
        return np.empty((0, 3), dtype=np.int32), np.empty((0, 3), dtype=np.float32)

    indices = list(seen.values())
    boundary_vertices = int_verts[indices].astype(np.int32)
    boundary_normals = normals[indices].astype(np.float32)
    log_msg(f"Boundary filter: {len(vertices)} mesh vertices → {len(boundary_vertices)} boundary voxels")
    return boundary_vertices, boundary_normals

#############################################
# Point matching functions
#############################################

def find_all_possible_matches_kernel(
    points,
    normals,
    surface1_mask,
    surface2_mask,
    match_distances,
    match_indices,
    match_counts,
    max_thickness_voxels,
    max_angle_cos,
    max_matches_per_point,
):
    """
    CUDA kernel to find all possible matches for each surface1 point.

    Parameters
    ----------
    points : ndarray
        Vertex coordinates
    normals : ndarray
        Normal vectors
    surface1_mask : ndarray
        Boolean mask for first surface
    surface2_mask : ndarray
        Boolean mask for second surface
    match_distances : ndarray
        Output array for match distances
    match_indices : ndarray
        Output array for match indices
    match_counts : ndarray
        Output array for match counts
    max_thickness_voxels : float
        Maximum thickness in voxel units
    max_angle_cos : float
        Cosine of maximum angle for cone search
    max_matches_per_point : int
        Maximum number of matches per point
    """
    idx = cuda.grid(1)
    if idx < points.shape[0] and surface1_mask[idx]:
        point = points[idx]
        normal = normals[idx]
        match_count = 0

        # Find all possible matches within constraints
        for i in range(points.shape[0]):
            if surface2_mask[i]:
                dx = points[i, 0] - point[0]
                dy = points[i, 1] - point[1]
                dz = points[i, 2] - point[2]

                # Calculate distance in voxel units (unscaled)
                dist = math.sqrt(dx * dx + dy * dy + dz * dz)
                if dist < max_thickness_voxels:
                    proj = dx * normal[0] + dy * normal[1] + dz * normal[2]
                    if proj > 0:
                        lateral_dx = dx - proj * normal[0]
                        lateral_dy = dy - proj * normal[1]
                        lateral_dz = dz - proj * normal[2]
                        lateral_dist_sq = lateral_dx**2 + lateral_dy**2 + lateral_dz**2

                        if lateral_dist_sq < (max_angle_cos * proj * proj):
                            if match_count < max_matches_per_point:
                                # Store this match (in voxel units)
                                base_idx = idx * max_matches_per_point + match_count
                                match_distances[base_idx] = dist
                                match_indices[base_idx] = i
                                match_count += 1

        match_counts[idx] = match_count

def process_matches_gpu2cpu(match_distances, match_indices, match_counts, n_points, max_matches_per_point, pixel_size):
    """
    Process matches on CPU to ensure one-to-one matching and convert to physical units.

    Parameters
    ----------
    match_distances : ndarray
        Match distances in voxel units
    match_indices : ndarray
        Match indices
    match_counts : ndarray
        Match counts
    n_points : int
        Number of points
    max_matches_per_point : int
        Maximum number of matches per point
    pixel_size : float
        Voxel size for scaling

    Returns
    -------
    thickness_results : ndarray
        Thickness measurements in physical units
    valid_mask : ndarray
        Boolean mask for valid measurements
    point_pairs : ndarray
        Indices of paired points
    """
    # Create arrays for final results (still in voxel units)
    thickness_results = np.zeros(n_points, dtype=np.float32)
    valid_mask = np.zeros(n_points, dtype=np.bool_)
    point_pairs = np.zeros(n_points, dtype=np.int32)

    # Create list of all possible matches
    all_matches = []
    for i in range(n_points):
        count = match_counts[i]
        for j in range(count):
            match_idx = i * max_matches_per_point + j
            all_matches.append(
                (
                    match_distances[match_idx],  # distance in voxel units
                    i,  # surface1 point index
                    match_indices[match_idx],  # surface2 point index
                )
            )

    # Sort matches by distance
    all_matches.sort()

    # Track assigned points
    surface1_assigned = set()
    surface2_assigned = set()

    # Assign matches
    for dist, surf1_idx, surf2_idx in all_matches:
        if surf1_idx not in surface1_assigned and surf2_idx not in surface2_assigned:
            # Assign match (still in voxel units)
            thickness_results[surf1_idx] = dist
            valid_mask[surf1_idx] = True
            point_pairs[surf1_idx] = surf2_idx

            surface1_assigned.add(surf1_idx)
            surface2_assigned.add(surf2_idx)

    # Convert thickness results to physical units before returning
    thickness_results = thickness_results * pixel_size

    return thickness_results, valid_mask, point_pairs

def measure_thickness_gpu(
    points,
    normals,
    surface1_mask,
    surface2_mask,
    pixel_size,
    max_distance_nm=8.0,
    max_angle_degrees=1.0,
    direction="1to2",
    logger=None,
):
    """
    GPU-accelerated geometric surface matching with one-to-one assignment.

    Despite the historical ``measure_thickness_*`` name, this routine returns
    **matched-point distances** along the bilayer normal (converted to
    nanometers), not the inflection-based thickness from 1D profiles.

    Workflow:

    1. **GPU preparation**: transfers data to device memory and configures the CUDA grid
    2. **Parallel search**: CUDA kernel enumerates candidate targets per source vertex
       inside the distance + cone constraints
    3. **CPU post-processing**: greedy one-to-one assignment prioritizing shortest distances
    4. **Unit conversion**: scales voxel distances by ``pixel_size`` to nm before return

    Geometric constraints:
    - **Maximum distance** (``max_distance_nm``): hard cap on candidate edge lengths
    - **Cone angle** (``max_angle_degrees``): limits lateral deviation from the normal ray
    - **Leaflet separation**: candidates must belong to the opposite boolean mask

    Parameters
    ----------
    points : ndarray
        Unscaled voxel coordinates
    normals : ndarray
        Normal vectors
    surface1_mask, surface2_mask : ndarray
        Boolean masks for each surface
    pixel_size : float
        Voxel size in **nanometers** (used to convert geometric distances)
    max_distance_nm : float
        Maximum source-target distance in nm (converted to voxels internally)
    max_angle_degrees : float
        Maximum angle for cone search
    direction : str
        Which surface seeds the rays: ``"1to2"`` matches surface1→surface2,
        ``"2to1"`` reverses the roles.
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    thickness_results : np.ndarray
        Length-``N`` array of per-vertex **matched distances in nm** (zeros mark
        unmatched / invalid vertices).
    valid_mask : np.ndarray
        Boolean mask aligned with ``thickness_results`` marking vertices that
        received an accepted one-to-one pairing.
    point_pairs : np.ndarray
        1D array of length N containing indices of matched points for each
        source point. For valid measurements, this gives the index of the
        corresponding point on the opposite surface.

    Raises
    ------
    RuntimeError
        If CUDA is not available or GPU memory allocation fails.
    ValueError
        If input arrays have incompatible shapes or invalid parameters.

    Notes
    -----
    - **GPU Requirements**: Requires CUDA-capable GPU with sufficient memory
    - **Performance**: Significantly faster than CPU implementation for large
      surfaces (typically 10-100x speedup)
    - **Memory**: GPU memory usage scales with N * max_matches_per_point
    - **Fallback**: Automatically falls back to CPU if CUDA unavailable
    - **Constraints**: Maximum 25 potential matches per point to manage memory
    - **Accuracy**: Results are identical to CPU implementation but faster
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    # Switch source and target surfaces if direction is 2to1
    if direction == "2to1":
        log_msg("Matching surface 2 points to surface 1...")
        source_mask, target_mask = surface2_mask, surface1_mask
    else:
        log_msg("Matching surface 1 points to surface 2...")
        source_mask, target_mask = surface1_mask, surface2_mask

    n_points = len(points)
    # Correct cone test: lateral² < tan²(α) * proj²  (was cos(α), giving ~45° for any α ≤ 45°)
    max_angle_cos = math.tan(math.radians(max_angle_degrees)) ** 2

    # Convert max thickness from nm to voxels
    max_thickness_voxels = max_distance_nm / pixel_size

    # Set maximum number of potential matches per point
    max_matches_per_point = 25

    log_msg(f"Starting GPU surface matching with {n_points} points...")
    log_msg(
        f"Source points: {np.sum(source_mask)}, target points: {np.sum(target_mask)}, "
        f"max match distance: {max_distance_nm:.2f} nm ({max_thickness_voxels:.2f} vox)"
    )
    log_msg(f"Max angle: {max_angle_degrees:.1f} degrees")

    # Prepare GPU arrays for all possible matches
    match_distances = cuda.to_device(np.zeros(n_points * max_matches_per_point, dtype=np.float32))
    match_indices = cuda.to_device(np.zeros(n_points * max_matches_per_point, dtype=np.int32))
    match_counts = cuda.to_device(np.zeros(n_points, dtype=np.int32))

    # Configure CUDA grid
    threads_per_block = 256
    blocks_per_grid = (n_points + threads_per_block - 1) // threads_per_block

    log_msg(f"Launching CUDA kernel with {blocks_per_grid} blocks and {threads_per_block} threads per block...")

    # Find all possible matches - works in voxel space!!
    find_all_possible_matches_kernel[blocks_per_grid, threads_per_block](
        cuda.to_device(points.astype(np.float32)),
        cuda.to_device(normals.astype(np.float32)),
        cuda.to_device(source_mask.astype(np.int32)),
        cuda.to_device(target_mask.astype(np.int32)),
        match_distances,
        match_indices,
        match_counts,
        max_thickness_voxels,
        max_angle_cos,
        max_matches_per_point,
    )

    # Get results back from GPU
    log_msg("Retrieving results from GPU...")
    match_distances_cpu = match_distances.copy_to_host()
    match_indices_cpu = match_indices.copy_to_host()
    match_counts_cpu = match_counts.copy_to_host()

    # Process matches and convert to physical units
    log_msg("Resolving one-to-one surface matches...")
    thickness_results, valid_mask, point_pairs = process_matches_gpu2cpu(
        match_distances_cpu, match_indices_cpu, match_counts_cpu, n_points, max_matches_per_point, pixel_size
    )

    log_msg(f"Resolved {np.sum(valid_mask)} one-to-one surface matches")

    return thickness_results, valid_mask, point_pairs

def _batched_query(tree, source_points, radius, batch_size):
    """Yield (i, neighbors) from query_ball_point in memory-bounded batches."""
    n = len(source_points)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        for j, neighbors in enumerate(tree.query_ball_point(source_points[start:end], radius)):
            yield start + j, neighbors

def measure_thickness_cpu(
    points,
    normals,
    surface1_mask,
    surface2_mask,
    pixel_size,
    max_distance_nm=8.0,
    max_angle_degrees=1.0,
    direction="1to2",
    num_threads=None,
    logger=None,
    max_matches_per_point=25,
    query_batch_size=200000,
):
    """
    CPU-based geometric surface matching (Numba + SciPy KDTree).

    Mirrors ``measure_thickness_gpu`` but executes on the host. Returned arrays
    encode **matched-point distances in nm** (historical ``thickness_*`` naming).

    Parameters
    ----------
    points : ndarray, shape (N, 3)
        Unscaled voxel coordinates.
    normals : ndarray, shape (N, 3)
        Normal vectors.
    surface1_mask, surface2_mask : ndarray of bool, shape (N,)
        Boolean masks for each bilayer leaflet.
    pixel_size : float
        Voxel size in **nanometers** (used to convert distances).
    max_distance_nm : float, default 8.0
        Maximum source-target distance in nm (converted to voxels internally).
    max_angle_degrees : float, default 1.0
        Half-angle of the normal-aligned search cone in degrees.
    direction : {"1to2", "2to1"}, default "1to2"
        Which surface seeds the matching rays.
    num_threads : int, optional
        Numba thread count (``None`` = all available threads).
    logger : logging.Logger, optional
    max_matches_per_point : int, default 25
        Maximum candidate targets retained per source point before one-to-one
        assignment.
    query_batch_size : int, optional
        KDTree ball-query batch size (``None`` = unbatched). Reduces peak
        memory usage on large surfaces.

    Returns
    -------
    thickness_results : ndarray, shape (N,)
        Per-vertex matched distances in nm (zeros for unmatched vertices).
    valid_mask : ndarray of bool, shape (N,)
        True for vertices that received a valid one-to-one pairing.
    point_pairs : ndarray, shape (N,)
        Index of the matched partner on the opposite surface; meaningful only
        where ``valid_mask`` is True.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    # Set number of threads if specified
    if num_threads is not None:
        numba.set_num_threads(num_threads)
        log_msg(f"Using {num_threads} CPU threads")
    else:
        log_msg(f"Using all available CPU threads (numba default)")

    # Switch source and target surfaces if direction is 2to1
    if direction == "2to1":
        log_msg("Matching surface 2 points to surface 1...")
        source_mask, target_mask = surface2_mask, surface1_mask
    else:
        log_msg("Matching surface 1 points to surface 2...")
        source_mask, target_mask = surface1_mask, surface2_mask

    n_points = len(points)
    # Correct cone test: lateral² < tan²(α) * proj²  (was cos(α), giving ~45° for any α ≤ 45°)
    max_angle_cos = np.tan(np.radians(max_angle_degrees)) ** 2

    # Convert max thickness from nm to voxels
    max_thickness_voxels = max_distance_nm / pixel_size

    log_msg(f"Starting CPU surface matching with {n_points} points...")
    log_msg(
        f"Source points: {np.sum(source_mask)}, target points: {np.sum(target_mask)}, "
        f"max match distance: {max_distance_nm:.2f} nm ({max_thickness_voxels:.2f} vox)"
    )
    log_msg(f"Max angle: {max_angle_degrees:.1f} degrees")

    target_indices = np.where(target_mask)[0]
    target_points = points[target_indices]
    source_indices = np.where(source_mask)[0]
    source_points = points[source_indices]

    # Build KD-tree
    log_msg("Building KD-tree for target points...")
    target_tree = ScipyKDTree(target_points)

    # Pre-filter matches using ball query
    start_time = time.time()
    flat_matches = []

    # --- batched query path (remove this block + else to revert to unbatched) ---
    if query_batch_size is not None:
        log_msg(f"Querying KD-tree in batches of {query_batch_size} ({len(source_points)} source points)...")
        query_iter = _batched_query(target_tree, source_points, max_thickness_voxels, query_batch_size)
    # --- end batched query path ---
    else:
        log_msg(f"Querying KD-tree for {len(source_points)} source points...")
        query_iter = enumerate(target_tree.query_ball_point(source_points, max_thickness_voxels))

    for i, neighbors in query_iter:
        if not neighbors:
            continue
        source_idx = source_indices[i]
        src_n = normals[source_idx]
        src_p = points[source_idx]

        nb = np.asarray(neighbors, dtype=np.intp)
        v = target_points[nb] - src_p          # (M, 3) vectors to all candidates
        proj = v @ src_n                        # (M,)   projection onto normal

        fwd = proj > 0
        if not fwd.any():
            continue

        proj_f = proj[fwd]
        v_f = v[fwd]
        lat = v_f - proj_f[:, None] * src_n
        lat_sq = (lat * lat).sum(axis=1)

        cone = lat_sq < (max_angle_cos * proj_f * proj_f)
        if not cone.any():
            continue

        nb_valid = nb[fwd][cone][:max_matches_per_point]
        v_valid = v_f[cone][:max_matches_per_point]
        dists = np.sqrt((v_valid * v_valid).sum(axis=1))

        for d, tidx in zip(dists.tolist(), target_indices[nb_valid].tolist()):
            flat_matches.append((d, source_idx, tidx))

    log_msg(f"KD-tree neighborhood search completed in {time.time() - start_time:.2f} seconds")
    log_msg(f"Found {len(flat_matches)} potential matches across all source points")

    # Process matches to ensure one-to-one matching
    log_msg("Resolving one-to-one surface matches...")
    thickness_results, valid_mask, point_pairs = process_matches_cpu2cpu(flat_matches, n_points, pixel_size)

    log_msg(f"Resolved {np.sum(valid_mask)} one-to-one surface matches")

    return thickness_results, valid_mask, point_pairs

def process_matches_cpu2cpu(flat_matches, n_points, pixel_size):
    """
    Process matches on CPU to ensure one-to-one matching and convert to physical units.

    Parameters
    ----------
    flat_matches : list
        List of tuples (distance, source_idx, target_idx)
    n_points : int
        Total number of points
    pixel_size : float
        Voxel size for scaling

    Returns
    -------
    thickness_results : ndarray
        Thickness measurements in physical units
    valid_mask : ndarray
        Boolean mask for valid measurements
    point_pairs : ndarray
        Indices of paired points
    """
    # Create arrays for final results (still in voxel units)
    thickness_results = np.zeros(n_points, dtype=np.float32)
    valid_mask = np.zeros(n_points, dtype=np.bool_)
    point_pairs = np.zeros(n_points, dtype=np.int32)

    # Sort matches by distance
    flat_matches.sort()

    # Track assigned points
    source_assigned = set()
    target_assigned = set()

    # Assign matches
    for dist, source_idx, target_idx in flat_matches:
        if source_idx not in source_assigned and target_idx not in target_assigned:
            # Assign match (still in voxel units)
            thickness_results[source_idx] = dist
            valid_mask[source_idx] = True
            point_pairs[source_idx] = target_idx

            source_assigned.add(source_idx)
            target_assigned.add(target_idx)

    # Convert thickness results to physical units before returning
    thickness_results = thickness_results * pixel_size

    return thickness_results, valid_mask, point_pairs


#############################################
# Intensity Profile Analysis Functions
#############################################

## Utility Functions

def save_int_results(
    results: dict,
    thickness_csv: Path,
    output_path: PathOrStr,
    profiles: list[dict],
    save_cleaned_df: bool = True,
    save_profiles: bool = True,
    save_statistics: bool = True,
    csv_label: str = "thickness",
    stats_infix: str = "",
    tomogram_path: PathOrStr | None = None,
    segmentation_path: PathOrStr | None = None,
    logger: logging.Logger = None,
) -> dict[str, Path]:
    """
    Persist boundary-finding outputs produced by ``resolve_profile_features``.

    Parameters
    ----------
    results : dict
        Must include ``resolved_thickness_df``, ``boundary_results``,
        ``resolved_profile_indices``, ``resolved_profiles``, and ``parameters``.
    thickness_csv : pathlib.Path
        Input table path (typically ``*_matched_points*.csv``). Used only for naming
        and provenance text in the statistics file.
    tomogram_path, segmentation_path : str | pathlib.Path | None, optional
        When provided, echoed in the stats file preamble (basename and resolved paths).
        Segmentation can be omitted (e.g. intensity-only callers that do not pass it).
    output_path : PathOrStr
        Destination directory (created if missing).
    profiles : list of dict
        Full extracted profile list (same length ordering as the input table).
    save_cleaned_df : bool
        Write ``{base}_{csv_label}.csv`` with the **kept-after-filter** rows only (inflection
        cohort plus ``minima_only`` rows within ``max_distance_nm``; see ``_resolve``).
    save_profiles : bool
        Write ``{base}_int_profiles.pkl`` with resolved profiles merged with boundary metadata.
    save_statistics : bool
        Write ``{base}{stats_infix}_boundary_stats.txt`` (header + ``print_summary`` capture + parameters).
    csv_label : str, default "thickness"
        Infix in the output CSV filename: ``{base}_{csv_label}.csv``.
    stats_infix : str, default ""
        Extra token inserted before ``_boundary_stats.txt`` in the statistics filename.
    logger : logging.Logger, optional
        Logger for informational messages.

    Returns
    -------
    dict[str, pathlib.Path]
        Keys among ``thickness_csv``, ``int_profiles``, ``statistics`` for paths actually written.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate base name for output files
    base_name = _matched_table_base_name(thickness_csv)

    saved_files = {}

    resolved_df = results.get("resolved_thickness_df")
    boundary_results = results.get("boundary_results", [])
    resolved_indices = results.get("resolved_profile_indices", [])

    if save_cleaned_df and resolved_df is not None:
        thickness_path = output_path / f"{base_name}_{csv_label}.csv"
        resolved_df.to_csv(thickness_path, index=False)
        saved_files["thickness_csv"] = thickness_path
        log_msg(f"Saved thickness table: {thickness_path}")

    if save_profiles:
        import pickle

        resolved_profiles = results.get("resolved_profiles", [])
        if resolved_profiles:
            resolved_profiles_with_features = _merge_profile_features(
                profiles=resolved_profiles,
                feature_results=boundary_results,
                profile_indices=resolved_indices,
                thickness_df=results.get("membrane_thickness_df"),
            )

            resolved_profiles_path = output_path / f"{base_name}_int_profiles.pkl"
            with open(resolved_profiles_path, "wb") as f:
                pickle.dump(resolved_profiles_with_features, f)
            saved_files["int_profiles"] = resolved_profiles_path
            log_msg(f"Saved intensity profiles: {resolved_profiles_path}")

    if save_statistics:
        stats_path = output_path / f"{base_name}{stats_infix}_boundary_stats.txt"

        with open(stats_path, "w") as f:
            import sys
            from io import StringIO

            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()

            print_summary(results)

            sys.stdout = old_stdout
            stats_content = mystdout.getvalue()

            f.write(
                _boundary_stats_file_preamble_lines(
                    thickness_csv,
                    tomogram_path=tomogram_path,
                    segmentation_path=segmentation_path,
                )
            )
            f.write(stats_content)

            f.write("\n=== Processing Parameters ===\n")
            params = results.get("parameters", {})
            for key in sorted(params):
                f.write(f"{key}: {params[key]}\n")

            f.write(f"\n=== Extraction Information ===\n")
            f.write(f"Total extracted profiles: {len(profiles)}\n")
            f.write(f"Profiles kept after max-distance filter: {len(results.get('resolved_profiles', []))}\n")

            f.write(f"\n=== Output Files ===\n")
            for file_type, file_path in saved_files.items():
                f.write(f"{file_type}: {file_path.name}\n")

        saved_files["statistics"] = stats_path
        log_msg(f"Saved statistics: {stats_path}")

    return saved_files

def _merge_profile_features(
    profiles: list[dict],
    feature_results: list[dict],
    profile_indices: list[int] | None = None,
    thickness_df: pd.DataFrame | None = None,
) -> list[dict]:
    """
    Attach correction metadata to extracted profiles for pickle export.

    When ``thickness_df`` is the per-row ``membrane_thickness_df`` from
    ``resolve_profile_features``, ``membrane_thickness_*`` / ``delta_thickness_nm`` in
    ``features`` are taken from that frame (so pickles match CSV semantics, e.g. NaN
    inflection thickness for exported ``minima_only`` rows).
    """
    enhanced_profiles = []

    for i, profile in enumerate(profiles):
        enhanced_profile = profile.copy()

        if profile_indices is not None and i < len(profile_indices):
            feature_idx = profile_indices[i]
        else:
            feature_idx = i

        features = {
            "data_source": "feature_detection",
            "feature_index": feature_idx,
            "has_features": False,
            "minima1_position": np.nan,
            "minima1_intensity": np.nan,
            "minima2_position": np.nan,
            "minima2_intensity": np.nan,
            "central_max_position": np.nan,
            "central_max_intensity": np.nan,
            "separation_distance": np.nan,
            "resolved": False,
            "failure_reason": None,
            "p1_projection": np.nan,
            "p2_projection": np.nan,
            "left_boundary_position": np.nan,
            "right_boundary_position": np.nan,
            "left_boundary_mode": None,
            "right_boundary_mode": None,
            "detection_mode": None,
            "minima_identified": None,
            "membrane_thickness_vox": np.nan,
            "membrane_thickness_nm": np.nan,
            "matched_points_distance_vox": np.nan,
            "matched_points_distance_nm": np.nan,
            "delta_thickness_nm": np.nan,
            "left_slope_anchor_position": np.nan,
            "right_slope_anchor_position": np.nan,
            "left_slope_anchor_type": None,
            "right_slope_anchor_type": None,
            "left_slope_anchor_subtype": None,
            "right_slope_anchor_subtype": None,
            "boundary_quality": None,
        }

        if feature_idx < len(feature_results):
            feature_result = feature_results[feature_idx]
            features["has_features"] = True
            features["resolved"] = bool(feature_result.get("resolved", False))
            features["failure_reason"] = feature_result.get("failure_reason")
            features["p1_projection"] = feature_result.get("p1_projection", np.nan)
            features["p2_projection"] = feature_result.get("p2_projection", np.nan)
            features["left_boundary_position"] = feature_result.get("left_boundary_position", np.nan)
            features["right_boundary_position"] = feature_result.get("right_boundary_position", np.nan)
            features["left_boundary_mode"] = feature_result.get("left_boundary_mode")
            features["right_boundary_mode"] = feature_result.get("right_boundary_mode")
            features["detection_mode"] = feature_result.get("detection_mode")
            su = feature_result.get("detection_mode")
            if su == "minima_only":
                features["boundary_quality"] = "minima_only"
            elif features["resolved"] and su in inflection_point_method:
                features["boundary_quality"] = "inflection"
            elif not features["resolved"]:
                features["boundary_quality"] = "unresolved"
            else:
                features["boundary_quality"] = "other"
            features["minima_identified"] = feature_result.get("minima_identified")
            features["membrane_thickness_vox"] = feature_result.get("membrane_thickness_vox", np.nan)
            features["membrane_thickness_nm"] = feature_result.get("membrane_thickness_nm", np.nan)
            features["matched_points_distance_vox"] = feature_result.get("matched_points_distance_vox", np.nan)
            features["matched_points_distance_nm"] = feature_result.get("matched_points_distance_nm", np.nan)
            features["delta_thickness_nm"] = feature_result.get("delta_thickness_nm", np.nan)

            def _outward_anchor_pos(out) -> float:
                if not isinstance(out, dict):
                    return float("nan")
                p = out.get("position")
                try:
                    v = float(p)
                except (TypeError, ValueError):
                    return float("nan")
                return v if np.isfinite(v) else float("nan")

            lof = feature_result.get("left_outward_feature")
            rof = feature_result.get("right_outward_feature")
            features["left_slope_anchor_position"] = _outward_anchor_pos(lof)
            features["right_slope_anchor_position"] = _outward_anchor_pos(rof)
            features["left_slope_anchor_type"] = (
                lof.get("anchor_type") if isinstance(lof, dict) else None
            )
            features["right_slope_anchor_type"] = (
                rof.get("anchor_type") if isinstance(rof, dict) else None
            )
            features["left_slope_anchor_subtype"] = (
                lof.get("anchor_subtype") if isinstance(lof, dict) else None
            )
            features["right_slope_anchor_subtype"] = (
                rof.get("anchor_subtype") if isinstance(rof, dict) else None
            )

            left_min = feature_result.get("left_min")
            right_min = feature_result.get("right_min")
            central_max = feature_result.get("central_max")

            if left_min is not None:
                features["minima1_position"] = left_min.get("position", np.nan)
                features["minima1_intensity"] = left_min.get("intensity", np.nan)
            if right_min is not None:
                features["minima2_position"] = right_min.get("position", np.nan)
                features["minima2_intensity"] = right_min.get("intensity", np.nan)
            if central_max is not None:
                features["central_max_position"] = central_max.get("position", np.nan)
                features["central_max_intensity"] = central_max.get("intensity", np.nan)

            if np.isfinite(features["minima1_position"]) and np.isfinite(features["minima2_position"]):
                features["separation_distance"] = abs(features["minima2_position"] - features["minima1_position"])

        if thickness_df is not None and 0 <= feature_idx < len(thickness_df):
            row = thickness_df.iloc[feature_idx]
            for key in ("membrane_thickness_nm", "membrane_thickness_vox", "delta_thickness_nm"):
                if key in row.index and key in features:
                    val = row[key]
                    try:
                        features[key] = float(val) if pd.notna(val) else float("nan")
                    except (TypeError, ValueError):
                        features[key] = float("nan")

        enhanced_profile["features"] = features
        enhanced_profiles.append(enhanced_profile)

    return enhanced_profiles

def print_summary(results: dict, logger: logging.Logger = None) -> None:
    """Print summary of profile-based boundary finding results."""

    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    stats = results["statistics"]

    def _format_dual_value(value_nm: float, value_vox: float) -> str:
        if np.isfinite(value_nm) and np.isfinite(value_vox):
            return f"{value_nm:.3f} nm ({value_vox:.3f} vox)"
        if np.isfinite(value_nm):
            return f"{value_nm:.3f} nm"
        if np.isfinite(value_vox):
            return f"{value_vox:.3f} vox"
        return "nan"

    def _stat_line(name: str, stat_dict: dict, include_range: bool = True) -> None:
        if not stat_dict or not np.isfinite(stat_dict.get("mean_nm", np.nan)):
            return
        log_msg(f"{name} mean: {_format_dual_value(stat_dict['mean_nm'], stat_dict['mean_vox'])}")
        log_msg(f"{name} median: {_format_dual_value(stat_dict['median_nm'], stat_dict['median_vox'])}")
        if include_range:
            lo_nm, hi_nm = stat_dict.get("range_nm", (np.nan, np.nan))
            lo_vox, hi_vox = stat_dict.get("range_vox", (np.nan, np.nan))
            if np.isfinite(lo_nm) and np.isfinite(hi_nm) and np.isfinite(lo_vox) and np.isfinite(hi_vox):
                log_msg(
                    f"{name} range: {lo_nm:.3f}-{hi_nm:.3f} nm "
                    f"({lo_vox:.3f}-{hi_vox:.3f} vox)"
                )

    log_msg("=== Intensity profiles summary ===")
    log_msg(f"Number of intensity profiles resolved: {stats['profiles_resolved']:,}/{stats['total_profiles']:,} ({stats['resolution_rate']:.1%})")
    log_msg(
        f"Number of profiles after max-distance filter ({stats['max_distance_nm']:.3f} nm): "
        f"{stats['profiles_kept_after_distance_filter']:,}/{stats['profiles_resolved']:,} "
        f"({stats['distance_filter_rate_resolved']:.1%} of resolved; {stats['distance_filter_rate_total']:.1%} overall)"
    )
    if "profiles_exported_inflection_nm" in stats:
        log_msg(
            f"Number of profiles with thickness calculated as distances between inflection points: "
            f"{int(stats.get('profiles_exported_inflection_nm', 0)):,}"
        )
        log_msg(
            f"Number of profiles where only minima were detected (no inflection points): "
            f"{int(stats.get('profiles_exported_minima_only', 0)):,}"
        )

    if "pass2_flagged_profiles" in stats:
        log_msg("=== Profiles with missing outward maxima ===")
        n_flag = int(stats.get("pass2_flagged_profiles", 0))
        n_tot = int(stats["total_profiles"])
        denom = f" / {n_tot:,} sampled pairs ({(n_flag / n_tot):.1%})" if n_tot > 0 else ""
        log_msg(
            f"Profiles lacking an outward-maximum on one or both leaflets: "
            f"{n_flag:,}{denom}"
        )
        log_msg(
            "One maximum missing → opposite-side minima-to-maximum distances mirrored: "
            f"{int(stats.get('pass2_mirror_anchor_sides', 0)):,}"
        )
        log_msg(
            "Both maxima missing → no inflection-point distances calculated: "
            f"{int(stats.get('pass2_minima_only_dual', 0)):,}"
        )


    quality_metrics = stats.get("quality_metrics", {})
    minima_sep_stats = quality_metrics.get("separation_stats", {})
    if minima_sep_stats and np.isfinite(minima_sep_stats.get("mean_nm", np.nan)):
        log_msg("=== Distances between minima (all profiles with two detected minima) ===")
        _stat_line("Distances between minima", minima_sep_stats)

    thickness_stats = stats.get("membrane_thickness_stats", {})
    if thickness_stats and np.isfinite(thickness_stats.get("mean_nm", np.nan)):
        log_msg("=== Membrane thickness from inflection points ===")
        _stat_line("Inflection-point distances", thickness_stats)

    delta_stats = stats.get("delta_thickness_stats", {})
    if delta_stats and np.isfinite(delta_stats.get("mean_nm", np.nan)):
        log_msg("=== Distance shift from matched segmentation points to inflection points ===")
        _stat_line("Distance shift", delta_stats, include_range=False)


class IntensityProfileAnalyzer:
    """Two-step membrane boundary detection pipeline.

    Step 1 — ``_detect_single_profile``: per-profile minima/maxima/inflection search.
    Step 2 — ``_mirror_pass``: batch fallback for profiles where one or both strict
    outward maxima are missing (mirror from good side, or fall back to minima-only).

    Parameters
    ----------
    smooth_sigma_intensity_profiles : float
        Gaussian smoothing sigma applied along the profile axis.
    extrema_prominence_threshold : float
        Prominence threshold for minima/maxima detection.
    minima_search_nm : tuple of (float, float)
        ``(primary_half_width_nm, relaxed_half_width_nm)`` — search window radii
        applied symmetrically around each matched point.
    anchor_search_nm : float
        Half-width (nm) of the outward search window for strict find_peaks maxima.
    mirror_anchor_slope_ratio_threshold : float
        Mirrored anchors move inward while local outward slope stays below this
        fraction of the opposite side's min→max reference rise.
    mirror_anchor_max_inward_steps : int
        Maximum inward index steps when refining a mirrored anchor.
    n_jobs : int
        Worker count for parallel profile processing (-1 = all cores, 1 = serial).
    minima_top_k : int
        Top-K candidates kept per side during minima search.
    inflection_slope_fraction : float
        Weighted-median fallback threshold: keep segments with slope >=
        ``inflection_slope_fraction * max_slope``.

    Typical usage
    -------------
    analyzer = IntensityProfileAnalyzer(smooth_sigma_intensity_profiles=1.0)
    results = analyzer.detect(profiles, thickness_df, profile_half_width_nm=6.0, max_distance_nm=10.0)
    """

    def __init__(
        self,
        smooth_sigma_intensity_profiles: float = 0.5,
        extrema_prominence_threshold: float = 0.1,
        minima_search_nm: tuple = (3.0, 4.0),
        anchor_search_nm: float = 4.0,
        mirror_anchor_slope_ratio_threshold: float = 0.5,
        mirror_anchor_max_inward_steps: int = 10,
        n_jobs: int = -1,
        minima_top_k: int = 3,
        inflection_slope_fraction: float = 0.75,
    ):
        self.smooth_sigma_intensity_profiles = smooth_sigma_intensity_profiles
        self.extrema_prominence_threshold = extrema_prominence_threshold
        self.minima_search_nm = minima_search_nm
        self.anchor_search_nm = anchor_search_nm
        self.mirror_anchor_slope_ratio_threshold = mirror_anchor_slope_ratio_threshold
        self.mirror_anchor_max_inward_steps = mirror_anchor_max_inward_steps
        self.n_jobs = n_jobs
        self.minima_top_k = minima_top_k
        self.inflection_slope_fraction = inflection_slope_fraction

    # ── Public ──────────────────────────────────────────────────────────────────

    def detect(
        self,
        profiles: list[dict],
        thickness_df: pd.DataFrame,
        profile_half_width_nm: float = 6.0,
        max_distance_nm: float = 8.0,
        logger: logging.Logger = None,
    ) -> dict:
        """Run the full two-step detection pipeline on a list of profiles."""
        return self._resolve(profiles, thickness_df, profile_half_width_nm, max_distance_nm, logger)

    # ── Orchestration ────────────────────────────────────────────────────────────

    def _resolve(
        self,
        profiles: list[dict],
        thickness_df: pd.DataFrame,
        profile_half_width_nm: float = 6.0,
        max_distance_nm: float = 8.0,
        logger: logging.Logger = None,
    ) -> dict:
        if len(thickness_df) != len(profiles):
            warnings.warn(
                f"Thickness DF length ({len(thickness_df)}) != profiles length ({len(profiles)})"
            )
            min_len = min(len(thickness_df), len(profiles))
            thickness_df = thickness_df.iloc[:min_len].copy()
            profiles = profiles[:min_len]

        if self.n_jobs not in (None, 0, 1):
            from concurrent.futures import ProcessPoolExecutor
            actual_workers = None if self.n_jobs == -1 else self.n_jobs
            chunk = max(1, len(profiles) // ((actual_workers or os.cpu_count() or 4) * 4))
            with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                boundary_results = list(
                    executor.map(self._detect_single_profile, profiles, chunksize=chunk)
                )
            for i, r in enumerate(boundary_results):
                r["profile_index"] = i
        else:
            boundary_results = []
            for i, prof in enumerate(profiles):
                r = self._detect_single_profile(prof)
                r["profile_index"] = i
                boundary_results.append(r)

        pass2_infill_counts: dict[str, int] = {
            "pass2_flagged_profiles": 0,
            "pass2_minima_only_dual": 0,
            "pass2_minima_only_single_mirror_failed": 0,
            "pass2_mirror_anchor_sides": 0,
        }
        pass2_infill_counts = self._mirror_pass(boundary_results, profiles)

        corrected_df, resolved_df = IntensityProfileAnalyzer._build_membrane_thickness_dataframe(
            thickness_df, profiles, boundary_results
        )

        primary_hw, relaxed_hw = self.minima_search_nm
        max_distance_nm = max(0.1, float(max_distance_nm))

        corrected_df["thickness_geom_correction_delta_nm"] = 0.0
        corrected_df["thickness_geom_correction_source"] = "none"

        mask_trusted_export = (
            corrected_df["resolved"]
            & corrected_df["detection_mode"].isin(list(inflection_point_method))
            & np.isfinite(corrected_df["membrane_thickness_nm"])
            & (corrected_df["membrane_thickness_nm"] <= float(max_distance_nm))
        )
        mask_minima_only_export = (
            corrected_df["resolved"]
            & (corrected_df["detection_mode"] == "minima_only")
            & np.isfinite(corrected_df["minima_separation_nm"])
            & (corrected_df["minima_separation_nm"] <= float(max_distance_nm))
        )
        within_max_distance = mask_trusted_export | mask_minima_only_export
        corrected_df["within_max_distance"] = within_max_distance
        _mo = corrected_df["detection_mode"] == "minima_only"
        corrected_df.loc[_mo, "membrane_thickness_nm"] = np.nan
        corrected_df.loc[_mo, "membrane_thickness_vox"] = np.nan
        corrected_df.loc[_mo, "delta_thickness_nm"] = np.nan
        corrected_df["membrane_thickness_nm_corrected"] = pd.to_numeric(
            corrected_df["membrane_thickness_nm"], errors="coerce"
        )
        resolved_df = corrected_df.loc[within_max_distance].copy()
        kept_indices = corrected_df.index[within_max_distance].to_numpy(dtype=int)

        results = {
            "boundary_results": boundary_results,
            "resolved_profile_indices": kept_indices.tolist(),
            "resolved_profiles": [profiles[i] for i in kept_indices],
            "membrane_thickness_df": corrected_df,
            "resolved_thickness_df": resolved_df,
            "statistics": {},
            "parameters": {
                "profile_half_width_nm": float(profile_half_width_nm),
                "max_distance_nm": float(max_distance_nm),
                "smooth_sigma_intensity_profiles": self.smooth_sigma_intensity_profiles,
                "extrema_prominence_threshold": self.extrema_prominence_threshold,
                "minima_search_nm_primary": primary_hw,
                "minima_search_nm_relaxed": relaxed_hw,
                "minima_top_k": self.minima_top_k,
                "anchor_search_nm": self.anchor_search_nm,
                "inflection_slope_fraction": self.inflection_slope_fraction,
                "mirror_anchor_slope_ratio_threshold": float(self.mirror_anchor_slope_ratio_threshold),
                "mirror_anchor_max_inward_steps": int(self.mirror_anchor_max_inward_steps),
            },
        }

        results["statistics"] = IntensityProfileAnalyzer._calculate_statistics(
            results, len(profiles), max_distance_nm=max_distance_nm
        )
        results["statistics"].update(pass2_infill_counts)
        n_export = int(within_max_distance.sum())
        results["statistics"]["profiles_kept_after_distance_filter"] = n_export
        n_tr = int(mask_trusted_export.sum())
        n_mo = int(mask_minima_only_export.sum())
        results["statistics"]["profiles_exported_inflection_nm"] = n_tr
        results["statistics"]["profiles_exported_minima_only"] = n_mo
        n_res = int(results["statistics"]["profiles_resolved"])
        tot = int(results["statistics"]["total_profiles"])
        results["statistics"]["distance_filter_rate_resolved"] = (
            (n_export / n_res) if n_res > 0 else 0.0
        )
        results["statistics"]["distance_filter_rate_total"] = (
            (n_export / tot) if tot > 0 else 0.0
        )
        return results

    def _detect_single_profile(self, prof: dict) -> dict:
        axis_data = IntensityProfileAnalyzer._get_profile_axis_data(prof)
        if axis_data is None:
            return {
                "resolved": False,
                "failure_reason": "invalid_profile_axis",
                "minima_identified": None,
                "detection_mode": None,
                "left_boundary_mode": None,
                "right_boundary_mode": None,
                "left_boundary_position": np.nan,
                "right_boundary_position": np.nan,
                "left_outward_feature": None,
                "right_outward_feature": None,
                "left_outward_anchor": None,
                "right_outward_anchor": None,
                "left_outward_max": None,
                "right_outward_max": None,
                "p1_projection": np.nan,
                "p2_projection": np.nan,
                "matched_points_distance_vox": np.nan,
                "matched_points_distance_nm": np.nan,
                "membrane_thickness_vox": np.nan,
                "membrane_thickness_nm": np.nan,
                "delta_thickness_nm": np.nan,
                "distances": None,
                "intensities": None,
                "smoothed_profile": None,
                "pixel_size": np.nan,
            }

        distances, intensities, _, _, pixel_size = axis_data
        if not np.isfinite(pixel_size) or pixel_size <= 0:
            return {
                "resolved": False,
                "failure_reason": "invalid_pixel_size",
                "minima_identified": None,
                "detection_mode": None,
                "left_boundary_mode": None,
                "right_boundary_mode": None,
                "left_boundary_position": np.nan,
                "right_boundary_position": np.nan,
                "left_outward_feature": None,
                "right_outward_feature": None,
                "left_outward_anchor": None,
                "right_outward_anchor": None,
                "left_outward_max": None,
                "right_outward_max": None,
                "p1_projection": np.nan,
                "p2_projection": np.nan,
                "matched_points_distance_vox": np.nan,
                "matched_points_distance_nm": np.nan,
                "membrane_thickness_vox": np.nan,
                "membrane_thickness_nm": np.nan,
                "delta_thickness_nm": np.nan,
                "distances": distances,
                "intensities": intensities,
                "smoothed_profile": None,
                "pixel_size": pixel_size,
            }

        p1_proj, p2_proj = IntensityProfileAnalyzer._calculate_point_projections(prof)
        matched_points_distance_vox = max(p1_proj, p2_proj) - min(p1_proj, p2_proj)
        matched_points_distance_nm = (
            matched_points_distance_vox * pixel_size if np.isfinite(pixel_size) else np.nan
        )
        smoothed = (
            gaussian_filter1d(intensities, sigma=self.smooth_sigma_intensity_profiles)
            if self.smooth_sigma_intensity_profiles > 0
            else intensities.copy()
        )

        def _failed_result(failure_reason, minima_result=None, left_boundary=None, right_boundary=None):
            result = {
                "resolved": False,
                "failure_reason": failure_reason,
                "minima_identified": minima_result.get("detection_mode") if minima_result else None,
                "detection_mode": None,
                "left_boundary_mode": left_boundary.get("boundary_mode") if left_boundary else None,
                "right_boundary_mode": right_boundary.get("boundary_mode") if right_boundary else None,
                "left_boundary_position": left_boundary.get("boundary_position", np.nan) if left_boundary else np.nan,
                "right_boundary_position": right_boundary.get("boundary_position", np.nan) if right_boundary else np.nan,
                "left_outward_feature": left_boundary.get("outward_feature") if left_boundary else None,
                "right_outward_feature": right_boundary.get("outward_feature") if right_boundary else None,
                "left_outward_anchor": left_boundary.get("outward_anchor") if left_boundary else None,
                "right_outward_anchor": right_boundary.get("outward_anchor") if right_boundary else None,
                "left_outward_max": left_boundary.get("outward_max") if left_boundary else None,
                "right_outward_max": right_boundary.get("outward_max") if right_boundary else None,
                "p1_projection": float(p1_proj),
                "p2_projection": float(p2_proj),
                "matched_points_distance_vox": float(matched_points_distance_vox),
                "matched_points_distance_nm": float(matched_points_distance_nm),
                "membrane_thickness_vox": np.nan,
                "membrane_thickness_nm": np.nan,
                "delta_thickness_nm": np.nan,
                "distances": distances,
                "intensities": intensities,
                "smoothed_profile": smoothed,
                "pixel_size": pixel_size,
            }
            if minima_result is not None:
                result.update(minima_result)
                result["resolved"] = False
                result["failure_reason"] = failure_reason
                result["minima_identified"] = minima_result.get("detection_mode")
            return result

        if np.isnan(p1_proj) or np.isnan(p2_proj):
            return _failed_result("invalid_point_projections")

        minima_result = self._find_anchored_minima_pair(
            distances=distances,
            intensities=intensities,
            p1_proj=p1_proj,
            p2_proj=p2_proj,
            pixel_size=pixel_size,
        )

        if not minima_result["resolved"]:
            return _failed_result(
                minima_result.get("failure_reason", "unresolved_minima"),
                minima_result=minima_result,
            )

        left_min_idx = minima_result["left_min"]["index_full"]
        right_min_idx = minima_result["right_min"]["index_full"]

        left_boundary = self._find_side_boundary(
            distances=distances, intensities=smoothed,
            min_idx=left_min_idx, direction="left", pixel_size=pixel_size,
        )
        right_boundary = self._find_side_boundary(
            distances=distances, intensities=smoothed,
            min_idx=right_min_idx, direction="right", pixel_size=pixel_size,
        )

        merged = IntensityProfileAnalyzer._merge_resolved_profile_detection(
            prof, minima_result, left_boundary, right_boundary, smoothed,
            distances, intensities,
            float(p1_proj), float(p2_proj),
            float(matched_points_distance_vox), float(matched_points_distance_nm),
            float(pixel_size),
        )
        if merged is not None:
            return merged

        missing: list[str] = []
        if (not left_boundary["resolved"]) and left_boundary.get("outward_feature") is None:
            missing.append("left")
        if (not right_boundary["resolved"]) and right_boundary.get("outward_feature") is None:
            missing.append("right")

        if not missing:
            if not left_boundary["resolved"]:
                fr = left_boundary.get("failure_reason", "unresolved_boundary")
            elif not right_boundary["resolved"]:
                fr = right_boundary.get("failure_reason", "unresolved_boundary")
            else:
                fr = "invalid_feature_order_after_boundary_detection"
            return _failed_result(
                fr, minima_result=minima_result,
                left_boundary=left_boundary, right_boundary=right_boundary,
            )

        out = _failed_result(
            "awaiting_predicted_anchor",
            minima_result=minima_result,
            left_boundary=left_boundary, right_boundary=right_boundary,
        )
        out["pass2_missing_sides"] = missing
        return out

    def _mirror_pass(
        self,
        boundary_results: list[dict],
        profiles: list[dict],
    ) -> dict[str, int]:
        counts = {
            "pass2_flagged_profiles": 0,
            "pass2_minima_only_dual": 0,
            "pass2_minima_only_single_mirror_failed": 0,
            "pass2_mirror_anchor_sides": 0,
        }
        n = len(boundary_results)
        if n == 0 or not any(r.get("pass2_missing_sides") for r in boundary_results):
            return counts

        for i in range(n):
            r = boundary_results[i]
            missing = r.get("pass2_missing_sides") or []
            if not missing:
                continue
            counts["pass2_flagged_profiles"] += 1
            axis_data = IntensityProfileAnalyzer._get_profile_axis_data(profiles[i])
            if axis_data is None:
                continue
            distances, intensities_axis, _, _, vs_prof = axis_data
            if not np.isfinite(vs_prof) or vs_prof <= 0:
                continue
            smoothed = r.get("smoothed_profile")
            if smoothed is None:
                smoothed = np.asarray(distances, dtype=float)

            if len(missing) == 2:
                fin = IntensityProfileAnalyzer._finalize_minima_only_row(
                    profiles[i], r, distances, intensities_axis, smoothed, vs_prof
                )
                boundary_results[i] = fin
                counts["pass2_minima_only_dual"] += 1
                if r.get("profile_index") is not None:
                    boundary_results[i]["profile_index"] = r["profile_index"]
                continue

            p1p = float(r.get("p1_projection", np.nan))
            p2p = float(r.get("p2_projection", np.nan))
            if not (np.isfinite(p1p) and np.isfinite(p2p)):
                continue

            left_min_idx = int(r["left_min"]["index_full"])
            right_min_idx = int(r["right_min"]["index_full"])
            left_min_pos = float(r["left_min"]["position"])
            right_min_pos = float(r["right_min"]["position"])
            single_missing = len(missing) == 1
            mirror_filled = False

            for side in missing:
                used_mirror = False
                if single_missing:
                    if side == "left":
                        span_vox = IntensityProfileAnalyzer._outward_max_span_vox_from_result(r, "right")
                        if span_vox is not None:
                            pred_pos = left_min_pos - span_vox
                            anchor_idx = IntensityProfileAnalyzer._snap_outward_anchor_index(
                                distances, pred_pos, left_min_idx, "left"
                            )
                            if anchor_idx is not None:
                                anchor_idx = self._refine_mirrored_anchor_idx_by_gradient(
                                    distances, smoothed, left_min_idx, anchor_idx, "left", "right", r
                                )
                                feat: dict[str, Any] = {
                                    "index_full": anchor_idx,
                                    "position": float(distances[anchor_idx]),
                                    "intensity": float(smoothed[anchor_idx]),
                                    "anchor_type": "anchor",
                                    "anchor_subtype": "mirrored_max_span",
                                    "donor_k": 0,
                                }
                                b = self._finish_side_boundary_with_outward_feature(
                                    distances, smoothed, left_min_idx, "left", feat
                                )
                                r["left_boundary_position"] = b.get("boundary_position", np.nan)
                                r["left_boundary_mode"] = b.get("boundary_mode")
                                r["left_outward_feature"] = b.get("outward_feature")
                                r["left_outward_anchor"] = b.get("outward_anchor")
                                r["left_outward_max"] = b.get("outward_max")
                                used_mirror = True
                    else:
                        span_vox = IntensityProfileAnalyzer._outward_max_span_vox_from_result(r, "left")
                        if span_vox is not None:
                            pred_pos = right_min_pos + span_vox
                            anchor_idx = IntensityProfileAnalyzer._snap_outward_anchor_index(
                                distances, pred_pos, right_min_idx, "right"
                            )
                            if anchor_idx is not None:
                                anchor_idx = self._refine_mirrored_anchor_idx_by_gradient(
                                    distances, smoothed, right_min_idx, anchor_idx, "right", "left", r
                                )
                                feat = {
                                    "index_full": anchor_idx,
                                    "position": float(distances[anchor_idx]),
                                    "intensity": float(smoothed[anchor_idx]),
                                    "anchor_type": "anchor",
                                    "anchor_subtype": "mirrored_max_span",
                                    "donor_k": 0,
                                }
                                b = self._finish_side_boundary_with_outward_feature(
                                    distances, smoothed, right_min_idx, "right", feat
                                )
                                r["right_boundary_position"] = b.get("boundary_position", np.nan)
                                r["right_boundary_mode"] = b.get("boundary_mode")
                                r["right_outward_feature"] = b.get("outward_feature")
                                r["right_outward_anchor"] = b.get("outward_anchor")
                                r["right_outward_max"] = b.get("outward_max")
                                used_mirror = True
                if used_mirror:
                    counts["pass2_mirror_anchor_sides"] += 1
                    mirror_filled = True

            if single_missing and not mirror_filled:
                fin = IntensityProfileAnalyzer._finalize_minima_only_row(
                    profiles[i], r, distances, intensities_axis, smoothed, vs_prof
                )
                boundary_results[i] = fin
                counts["pass2_minima_only_single_mirror_failed"] += 1
                if r.get("profile_index") is not None:
                    boundary_results[i]["profile_index"] = r["profile_index"]
                continue

            left_b = {
                "resolved": bool(np.isfinite(r.get("left_boundary_position", np.nan))),
                "boundary_position": r.get("left_boundary_position", np.nan),
                "boundary_mode": r.get("left_boundary_mode"),
                "outward_feature": r.get("left_outward_feature"),
                "outward_anchor": r.get("left_outward_anchor"),
                "outward_max": r.get("left_outward_max"),
                "failure_reason": None,
            }
            right_b = {
                "resolved": bool(np.isfinite(r.get("right_boundary_position", np.nan))),
                "boundary_position": r.get("right_boundary_position", np.nan),
                "boundary_mode": r.get("right_boundary_mode"),
                "outward_feature": r.get("right_outward_feature"),
                "outward_anchor": r.get("right_outward_anchor"),
                "outward_max": r.get("right_outward_max"),
                "failure_reason": None,
            }
            minima_sub = {
                k: r[k]
                for k in (
                    "left_min", "right_min", "central_max",
                    "left_window", "right_window", "adjacent_repair_applied",
                )
                if k in r
            }
            _inten = r.get("intensities")
            inten = _inten if _inten is not None else intensities_axis
            merged = IntensityProfileAnalyzer._merge_resolved_profile_detection(
                profiles[i], minima_sub, left_b, right_b,
                np.asarray(smoothed, dtype=float), distances, np.asarray(inten, dtype=float),
                p1p, p2p,
                float(r.get("matched_points_distance_vox", np.nan)),
                float(r.get("matched_points_distance_nm", np.nan)),
                float(r.get("pixel_size", vs_prof)),
            )
            if merged is not None:
                pi = r.get("profile_index")
                boundary_results[i] = merged
                if pi is not None:
                    boundary_results[i]["profile_index"] = pi
            else:
                r.pop("pass2_missing_sides", None)
                r["failure_reason"] = "invalid_feature_order_after_predicted_anchor"

        return counts

    # ── Per-profile signal helpers ───────────────────────────────────────────────

    def _find_anchored_minima_pair(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        p1_proj: float,
        p2_proj: float,
        pixel_size: float,
    ) -> dict:
        result = {
            "resolved": False, "detection_mode": None,
            "left_min": None, "right_min": None, "central_max": None,
            "left_window": None, "right_window": None, "failure_reason": None,
        }

        smoothed = (
            gaussian_filter1d(intensities, sigma=self.smooth_sigma_intensity_profiles)
            if self.smooth_sigma_intensity_profiles > 0
            else intensities.copy()
        )

        left_anchor = min(p1_proj, p2_proj)
        right_anchor = max(p1_proj, p2_proj)
        primary_hw, relaxed_hw = self.minima_search_nm
        primary_vox = float(primary_hw) / pixel_size
        relaxed_vox = float(relaxed_hw) / pixel_size

        search_stages = [
            {"name": "primary_search", "inward_range": primary_vox, "outward_slack": primary_vox},
            {"name": "relaxed_search", "inward_range": relaxed_vox, "outward_slack": relaxed_vox},
        ]

        for stage in search_stages:
            inward_range = stage["inward_range"]
            outward_slack = stage["outward_slack"]

            left_window = IntensityProfileAnalyzer._extract_search_window(
                distances, smoothed, left_anchor - outward_slack, left_anchor + inward_range,
            )
            right_window = IntensityProfileAnalyzer._extract_search_window(
                distances, smoothed, right_anchor - inward_range, right_anchor + outward_slack,
            )

            if left_window is None:
                result["failure_reason"] = f"no_left_window_{stage['name']}"; continue
            if right_window is None:
                result["failure_reason"] = f"no_right_window_{stage['name']}"; continue

            left_idx_full, left_dist, left_int = left_window
            right_idx_full, right_dist, right_int = right_window

            left_candidates = IntensityProfileAnalyzer._find_minima_candidates_in_window(
                left_dist, left_int, left_anchor,
                self.extrema_prominence_threshold, top_k=self.minima_top_k,
            )
            right_candidates = IntensityProfileAnalyzer._find_minima_candidates_in_window(
                right_dist, right_int, right_anchor,
                self.extrema_prominence_threshold, top_k=self.minima_top_k,
            )

            if len(left_candidates) == 0:
                result["failure_reason"] = f"no_left_minimum_{stage['name']}"; continue
            if len(right_candidates) == 0:
                result["failure_reason"] = f"no_right_minimum_{stage['name']}"; continue

            maxima_peaks, maxima_props = find_peaks(
                smoothed, prominence=self.extrema_prominence_threshold, distance=3, plateau_size=1
            )
            maxima_prom = maxima_props.get("prominences", np.zeros(len(maxima_peaks)))

            best_pair = None
            best_pair_score = -np.inf

            for left_cand in left_candidates:
                left_idx = left_idx_full[left_cand["peak_idx_window"]]
                for right_cand in right_candidates:
                    right_idx = right_idx_full[right_cand["peak_idx_window"]]
                    if left_idx >= right_idx:
                        continue
                    between_mask = (maxima_peaks > left_idx) & (maxima_peaks < right_idx)
                    central_candidates = maxima_peaks[between_mask]
                    if len(central_candidates) == 0:
                        continue
                    central_prom = maxima_prom[between_mask]
                    best_central_local = np.argmax(central_prom)
                    central_idx = central_candidates[best_central_local]
                    central_prominence = central_prom[best_central_local]
                    pair_score = (
                        left_cand["score"] + right_cand["score"]
                        + 1.0 * float(central_prominence)
                        - 0.25 * abs(
                            distances[central_idx]
                            - 0.5 * (left_cand["position"] + right_cand["position"])
                        )
                    )
                    if pair_score > best_pair_score:
                        best_pair_score = pair_score
                        best_pair = {
                            "left_min": {
                                "index_full": int(left_idx),
                                "position": float(distances[left_idx]),
                                "intensity": float(smoothed[left_idx]),
                                "prominence": float(left_cand["prominence"]),
                            },
                            "right_min": {
                                "index_full": int(right_idx),
                                "position": float(distances[right_idx]),
                                "intensity": float(smoothed[right_idx]),
                                "prominence": float(right_cand["prominence"]),
                            },
                            "central_max": {
                                "index_full": int(central_idx),
                                "position": float(distances[central_idx]),
                                "intensity": float(smoothed[central_idx]),
                                "prominence": float(central_prominence),
                            },
                            "left_window": (
                                float(left_anchor - outward_slack),
                                float(left_anchor + inward_range),
                            ),
                            "right_window": (
                                float(right_anchor - inward_range),
                                float(right_anchor + outward_slack),
                            ),
                        }

            if best_pair is not None:
                repaired = self._enforce_adjacent_flanking_minima(
                    distances=distances, intensities=smoothed,
                    left_anchor=left_anchor, right_anchor=right_anchor,
                    left_search_bound=float(left_anchor - outward_slack),
                    right_search_bound=float(right_anchor + outward_slack),
                    provisional_left_min=best_pair["left_min"],
                    provisional_right_min=best_pair["right_min"],
                    provisional_central_max=best_pair["central_max"],
                )
                if repaired["resolved"]:
                    best_pair["left_min"] = repaired["left_min"]
                    best_pair["right_min"] = repaired["right_min"]
                    best_pair["central_max"] = repaired["central_max"]
                    best_pair["adjacent_repair_applied"] = repaired.get("adjacent_repair_applied", False)
                    result.update(best_pair)
                    result["resolved"] = True
                    result["detection_mode"] = stage["name"]
                    result["failure_reason"] = None
                    return result
                result["failure_reason"] = repaired["failure_reason"]

            if best_pair is None:
                result["failure_reason"] = f"no_valid_pair_{stage['name']}"

        return result

    def _enforce_adjacent_flanking_minima(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        left_anchor: float,
        right_anchor: float,
        left_search_bound: float,
        right_search_bound: float,
        provisional_left_min: dict,
        provisional_right_min: dict,
        provisional_central_max: dict,
        max_iterations: int = 3,
    ) -> dict:
        maxima_peaks, maxima_props = find_peaks(
            intensities, prominence=self.extrema_prominence_threshold, distance=3, plateau_size=1
        )
        maxima_prom = maxima_props.get("prominences", np.zeros(len(maxima_peaks)))

        current_left = provisional_left_min
        current_right = provisional_right_min
        current_central = provisional_central_max
        repair_applied = False

        for _ in range(max_iterations):
            central_idx = int(current_central["index_full"])
            central_pos = float(distances[central_idx])

            left_window = IntensityProfileAnalyzer._extract_search_window(
                distances, intensities, left_search_bound, central_pos,
            )
            right_window = IntensityProfileAnalyzer._extract_search_window(
                distances, intensities, central_pos, right_search_bound,
            )

            if left_window is None:
                return {"resolved": False, "failure_reason": "no_left_window_for_adjacency_repair"}
            if right_window is None:
                return {"resolved": False, "failure_reason": "no_right_window_for_adjacency_repair"}

            left_idx_full, left_dist, left_int = left_window
            right_idx_full, right_dist, right_int = right_window

            left_candidates = IntensityProfileAnalyzer._find_minima_candidates_in_window(
                left_dist, left_int, left_anchor, self.extrema_prominence_threshold, top_k=None,
            )
            right_candidates = IntensityProfileAnalyzer._find_minima_candidates_in_window(
                right_dist, right_int, right_anchor, self.extrema_prominence_threshold, top_k=None,
            )

            if not left_candidates:
                return {"resolved": False, "failure_reason": "no_left_flanking_minimum_adjacent_to_central_max"}
            if not right_candidates:
                return {"resolved": False, "failure_reason": "no_right_flanking_minimum_adjacent_to_central_max"}

            left_flanking = [
                {"index_full": int(left_idx_full[c["peak_idx_window"]]),
                 "position": float(distances[int(left_idx_full[c["peak_idx_window"]])]),
                 "intensity": float(intensities[int(left_idx_full[c["peak_idx_window"]])]),
                 "prominence": float(c["prominence"]), "score": float(c["score"])}
                for c in left_candidates
                if int(left_idx_full[c["peak_idx_window"]]) < central_idx
            ]
            right_flanking = [
                {"index_full": int(right_idx_full[c["peak_idx_window"]]),
                 "position": float(distances[int(right_idx_full[c["peak_idx_window"]])]),
                 "intensity": float(intensities[int(right_idx_full[c["peak_idx_window"]])]),
                 "prominence": float(c["prominence"]), "score": float(c["score"])}
                for c in right_candidates
                if int(right_idx_full[c["peak_idx_window"]]) > central_idx
            ]

            if not left_flanking:
                return {"resolved": False, "failure_reason": "no_left_flanking_minimum_adjacent_to_central_max"}
            if not right_flanking:
                return {"resolved": False, "failure_reason": "no_right_flanking_minimum_adjacent_to_central_max"}

            new_left = max(left_flanking, key=lambda x: x["index_full"])
            new_right = min(right_flanking, key=lambda x: x["index_full"])

            if new_left["index_full"] >= new_right["index_full"]:
                return {"resolved": False, "failure_reason": "invalid_adjacent_minima_order"}

            between_mask = (maxima_peaks > new_left["index_full"]) & (maxima_peaks < new_right["index_full"])
            central_candidates = maxima_peaks[between_mask]
            if len(central_candidates) == 0:
                return {"resolved": False, "failure_reason": "no_central_maximum_between_adjacent_minima"}

            central_prom = maxima_prom[between_mask]
            best_central_local = int(np.argmax(central_prom))
            central_idx_new = int(central_candidates[best_central_local])
            new_central = {
                "index_full": central_idx_new,
                "position": float(distances[central_idx_new]),
                "intensity": float(intensities[central_idx_new]),
                "prominence": float(central_prom[best_central_local]),
            }

            if (
                new_left["index_full"] == current_left["index_full"]
                and new_right["index_full"] == current_right["index_full"]
                and new_central["index_full"] == current_central["index_full"]
            ):
                return {
                    "resolved": True,
                    "left_min": current_left, "right_min": current_right,
                    "central_max": current_central, "adjacent_repair_applied": repair_applied,
                }

            repair_applied = True
            current_left, current_right, current_central = new_left, new_right, new_central

        return {
            "resolved": True,
            "left_min": current_left, "right_min": current_right,
            "central_max": current_central, "adjacent_repair_applied": repair_applied,
        }

    def _find_first_outward_maximum(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        min_idx: int,
        direction: str,
        pixel_size: float,
    ) -> dict | None:
        outward_range_vox = self.anchor_search_nm / pixel_size
        min_pos = distances[min_idx]

        if direction == "left":
            mask = (distances >= (min_pos - outward_range_vox)) & (distances < min_pos)
        elif direction == "right":
            mask = (distances > min_pos) & (distances <= (min_pos + outward_range_vox))
        else:
            raise ValueError("direction must be 'left' or 'right'")

        candidate_indices = np.where(mask)[0]
        if candidate_indices.size < 3:
            return None

        peaks, _ = find_peaks(
            intensities[candidate_indices],
            prominence=self.extrema_prominence_threshold, distance=3, plateau_size=1,
        )
        if len(peaks) == 0:
            return None

        peak_indices_full = candidate_indices[peaks]
        if direction == "left":
            peak_indices_full = peak_indices_full[(min_idx - peak_indices_full) >= 1]
        else:
            peak_indices_full = peak_indices_full[(peak_indices_full - min_idx) >= 1]

        if peak_indices_full.size == 0:
            return None

        chosen_idx = int(peak_indices_full[-1] if direction == "left" else peak_indices_full[0])
        return {
            "index_full": chosen_idx,
            "position": float(distances[chosen_idx]),
            "intensity": float(intensities[chosen_idx]),
            "anchor_type": "max",
            "anchor_subtype": "strict_outward_find_peaks",
        }

    def _find_outward_anchor(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        min_idx: int,
        direction: str,
        pixel_size: float,
    ) -> dict | None:
        return self._find_first_outward_maximum(distances, intensities, min_idx, direction, pixel_size)

    def _find_side_boundary(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        min_idx: int,
        direction: str,
        pixel_size: float,
    ) -> dict:
        outward_feature = self._find_outward_anchor(distances, intensities, min_idx, direction, pixel_size)
        if outward_feature is None:
            return {
                "resolved": False, "boundary_position": np.nan, "boundary_mode": None,
                "outward_feature": None, "outward_anchor": None, "outward_max": None,
                "failure_reason": f"no_{direction}_outward_max",
            }
        return self._finish_side_boundary_with_outward_feature(
            distances, intensities, min_idx, direction, outward_feature,
        )

    def _find_slope_inflection_between_min_and_anchor_weighted(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        min_idx: int,
        anchor_idx: int,
        direction: str,
        ignore_first_segment_only: bool = True,
    ) -> float:
        if direction == "left":
            if anchor_idx >= min_idx:
                return np.nan
            interval_indices = np.arange(min_idx, anchor_idx - 1, -1)
        elif direction == "right":
            if anchor_idx <= min_idx:
                return np.nan
            interval_indices = np.arange(min_idx, anchor_idx + 1)
        else:
            raise ValueError("direction must be 'left' or 'right'")

        if interval_indices.size < 2:
            return np.nan

        x = distances[interval_indices]
        y = intensities[interval_indices]
        x_out = x[0] - x if direction == "left" else x - x[0]

        dx = np.diff(x_out)
        dy = np.diff(y)
        valid = np.abs(dx) > 1e-12
        if not np.any(valid):
            return np.nan

        seg_x_mid = 0.5 * (x[:-1] + x[1:])
        seg_slope = np.full_like(dy, np.nan, dtype=float)
        seg_slope[valid] = dy[valid] / dx[valid]

        finite_mask = np.isfinite(seg_slope)
        seg_x_mid = seg_x_mid[finite_mask]
        seg_slope = seg_slope[finite_mask]
        if seg_slope.size == 0:
            return np.nan

        if ignore_first_segment_only and seg_slope.size >= 3:
            seg_x_mid = seg_x_mid[1:]
            seg_slope = seg_slope[1:]

        positive = seg_slope > 0
        seg_x_mid = seg_x_mid[positive]
        seg_slope = seg_slope[positive]
        if seg_slope.size == 0:
            return np.nan

        smax = np.max(seg_slope)
        if not np.isfinite(smax) or smax <= 0:
            return np.nan

        best_idx = int(np.argmax(seg_slope))
        refined = IntensityProfileAnalyzer._refine_peak_position_quadratic(seg_x_mid, seg_slope, best_idx)
        if np.isfinite(refined):
            return refined

        keep = seg_slope >= self.inflection_slope_fraction * smax
        kept_x = seg_x_mid[keep]
        kept_w = seg_slope[keep]
        if kept_x.size == 0:
            return float(seg_x_mid[int(np.argmax(seg_slope))])
        return float(np.sum(kept_x * kept_w) / np.sum(kept_w))

    def _finish_side_boundary_with_outward_feature(
        self,
        distances: np.ndarray,
        intensities: np.ndarray,
        min_idx: int,
        direction: str,
        outward_feature: dict,
    ) -> dict:
        slope_inflection = self._find_slope_inflection_between_min_and_anchor_weighted(
            distances, intensities, min_idx, outward_feature["index_full"], direction,
        )
        if np.isnan(slope_inflection):
            return {
                "resolved": False, "boundary_position": np.nan,
                "boundary_mode": outward_feature.get("anchor_type"),
                "outward_feature": outward_feature,
                "outward_anchor": outward_feature if outward_feature["anchor_type"] == "anchor" else None,
                "outward_max": outward_feature if outward_feature["anchor_type"] == "max" else None,
                "failure_reason": f"invalid_{direction}_slope_inflection",
            }
        return {
            "resolved": True, "boundary_position": float(slope_inflection),
            "boundary_mode": outward_feature["anchor_type"],
            "outward_feature": outward_feature,
            "outward_anchor": outward_feature if outward_feature["anchor_type"] == "anchor" else None,
            "outward_max": outward_feature if outward_feature["anchor_type"] == "max" else None,
            "failure_reason": None,
        }

    def _refine_mirrored_anchor_idx_by_gradient(
        self,
        distances: np.ndarray,
        smoothed: np.ndarray,
        min_idx: int,
        anchor_idx: int,
        direction: str,
        good_side: str,
        r: dict,
    ) -> int:
        ref = IntensityProfileAnalyzer._reference_rise_slope_opposite_side(distances, smoothed, r, good_side)
        if not np.isfinite(ref) or ref <= 0:
            return int(anchor_idx)
        a = int(anchor_idx)
        thr = float(self.mirror_anchor_slope_ratio_threshold)
        for _ in range(int(self.mirror_anchor_max_inward_steps)):
            local_sl = IntensityProfileAnalyzer._local_outward_slope_window_at_anchor(
                distances, smoothed, a, direction
            )
            if not np.isfinite(local_sl) or local_sl >= thr * ref:
                break
            if direction == "left":
                anew = a + 1
                if anew >= min_idx or min_idx - anew < 1:
                    break
            else:
                anew = a - 1
                if anew <= min_idx or anew - min_idx < 1:
                    break
            a = anew
        return a

    # ── Pure math helpers (@staticmethod) ────────────────────────────────────────

    @staticmethod
    def _get_profile_axis_data(
        prof: dict,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float] | None:
        try:
            p1, p2 = prof["p1"], prof["p2"]
            midpoint, start, end = prof["midpoint"], prof["start"], prof["end"]
            intensities = prof["profile"]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                return None
            unit_dir = direction / length
            line_points = np.linspace(start, end, len(intensities))
            distances = np.dot(line_points - midpoint, unit_dir)
            return distances, intensities, unit_dir, midpoint, prof.get("pixel_size", np.nan)
        except Exception:
            return None

    @staticmethod
    def _extract_search_window(
        distances: np.ndarray,
        intensities: np.ndarray,
        start_pos: float,
        end_pos: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if start_pos > end_pos:
            start_pos, end_pos = end_pos, start_pos
        mask = (distances >= start_pos) & (distances <= end_pos)
        idx = np.where(mask)[0]
        if idx.size < 5:
            return None
        return idx, distances[idx], intensities[idx]

    @staticmethod
    def _find_minima_candidates_in_window(
        window_distances: np.ndarray,
        window_intensities: np.ndarray,
        anchor_pos: float,
        prominence_threshold: float,
        top_k: int | None = 3,
    ) -> list[dict]:
        peaks, props = find_peaks(
            -window_intensities, prominence=prominence_threshold, distance=3, plateau_size=1
        )
        if len(peaks) == 0:
            return []
        prominences = props.get("prominences", np.zeros(len(peaks)))
        candidates = [
            {
                "peak_idx_window": int(pk),
                "position": float(window_distances[pk]),
                "intensity": float(window_intensities[pk]),
                "prominence": float(prominences[i] if i < len(prominences) else 0.0),
                "score": float(
                    1.0 * (prominences[i] if i < len(prominences) else 0.0)
                    + 0.5 * (-window_intensities[pk])
                    - 0.5 * abs(window_distances[pk] - anchor_pos)
                ),
            }
            for i, pk in enumerate(peaks)
        ]
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates if top_k is None else candidates[:top_k]

    @staticmethod
    def _refine_peak_position_quadratic(x_mid: np.ndarray, slope: np.ndarray, peak_idx: int) -> float:
        if x_mid.size == 0 or peak_idx < 0 or peak_idx >= x_mid.size:
            return np.nan
        if peak_idx == 0 or peak_idx == x_mid.size - 1:
            return float(x_mid[peak_idx])
        x_fit = x_mid[peak_idx - 1: peak_idx + 2]
        y_fit = slope[peak_idx - 1: peak_idx + 2]
        if np.any(~np.isfinite(x_fit)) or np.any(~np.isfinite(y_fit)) or np.unique(x_fit).size < 3:
            return float(x_mid[peak_idx])
        try:
            a, b, _ = np.polyfit(x_fit, y_fit, 2)
        except Exception:
            return float(x_mid[peak_idx])
        if not np.isfinite(a) or abs(a) < 1e-12:
            return float(x_mid[peak_idx])
        x_vertex = -b / (2.0 * a)
        x_lo, x_hi = float(np.min(x_fit)), float(np.max(x_fit))
        if not np.isfinite(x_vertex) or x_vertex < x_lo or x_vertex > x_hi:
            return float(x_mid[peak_idx])
        return float(x_vertex)

    @staticmethod
    def _snap_outward_anchor_index(
        distances: np.ndarray, predicted_pos: float, min_idx: int, direction: str,
    ) -> int | None:
        idx_all = np.arange(len(distances), dtype=int)
        ok = (idx_all - min_idx) >= 1 if direction == "right" else (min_idx - idx_all) >= 1
        valid = np.where(ok)[0]
        if valid.size == 0:
            return None
        return int(valid[np.argmin(np.abs(distances[valid] - predicted_pos))])

    @staticmethod
    def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
        v = np.asarray(values, dtype=float).reshape(-1)
        w = np.asarray(weights, dtype=float).reshape(-1)
        m = np.isfinite(v) & np.isfinite(w) & (w > 0)
        if not np.any(m):
            return float(np.nan)
        v, w = v[m], w[m]
        w = w / np.sum(w)
        order = np.argsort(v)
        v, w = v[order], w[order]
        cum = np.cumsum(w)
        idx = min(int(np.searchsorted(cum, 0.5, side="left")), v.size - 1)
        return float(v[idx])

    @staticmethod
    def _max_abs_slope_on_consecutive_indices(
        distances: np.ndarray, smoothed: np.ndarray, indices: np.ndarray,
    ) -> float:
        if indices.size < 2:
            return 0.0
        m = 0.0
        for a, b in zip(indices[:-1], indices[1:]):
            dd = float(distances[int(b)] - distances[int(a)])
            if abs(dd) < 1e-12:
                continue
            m = max(m, abs((float(smoothed[int(b)]) - float(smoothed[int(a)])) / dd))
        return m

    @staticmethod
    def _index_chain_inclusive(i0: int, i1: int) -> np.ndarray:
        if i0 == i1:
            return np.array([i0], dtype=np.intp)
        return np.arange(i0, i1 + 1, dtype=np.intp) if i1 > i0 else np.arange(i0, i1 - 1, -1, dtype=np.intp)

    @staticmethod
    def _outward_max_span_vox_from_result(r: dict, side: str) -> float | None:
        mx = r.get(f"{side}_outward_max")
        mn = r.get(f"{side}_min")
        if mn is None or mx is None or not isinstance(mx, dict) or mx.get("anchor_type") != "max":
            return None
        try:
            span = abs(float(mn["position"]) - float(mx["position"]))
        except (TypeError, ValueError, KeyError):
            return None
        return float(span) if np.isfinite(span) and span > 0 else None

    @staticmethod
    def _reference_rise_slope_opposite_side(
        distances: np.ndarray, smoothed: np.ndarray, r: dict, good_side: str,
    ) -> float:
        mn = r.get(f"{good_side}_min")
        mx = r.get(f"{good_side}_outward_max")
        if mn is None or mx is None or not isinstance(mx, dict) or mx.get("anchor_type") != "max":
            return 0.0
        try:
            i0, i1 = int(mn["index_full"]), int(mx["index_full"])
        except (TypeError, ValueError, KeyError):
            return 0.0
        chain = IntensityProfileAnalyzer._index_chain_inclusive(i0, i1)
        ref = IntensityProfileAnalyzer._max_abs_slope_on_consecutive_indices(distances, smoothed, chain)
        return float(ref) if ref > 0 else 1e-12

    @staticmethod
    def _local_outward_slope_window_at_anchor(
        distances: np.ndarray, smoothed: np.ndarray,
        anchor_idx: int, direction: str, window_segments: int = 3,
    ) -> float:
        order: list[int] = []
        step = -1 if direction == "left" else 1
        for k in range(window_segments + 1):
            j = int(anchor_idx) + step * k
            if 0 <= j < len(distances):
                order.append(j)
        if len(order) < 2:
            return float("inf")
        chain = np.asarray(order, dtype=np.intp)
        return IntensityProfileAnalyzer._max_abs_slope_on_consecutive_indices(distances, smoothed, chain)

    @staticmethod
    def _boundaries_identified_from_modes(left_mode: str | None, right_mode: str | None) -> str | None:
        if left_mode is None or right_mode is None:
            return None
        if left_mode == "max" and right_mode == "max":
            return "max_max"
        if left_mode == "max" and right_mode == "anchor":
            return "max_anchor"
        if left_mode == "anchor" and right_mode == "max":
            return "anchor_max"
        return "slope_to_mixed_boundary"

    @staticmethod
    def _merge_resolved_profile_detection(
        prof: dict,
        minima_result: dict,
        left_boundary: dict,
        right_boundary: dict,
        smoothed: np.ndarray,
        distances: np.ndarray,
        intensities: np.ndarray,
        p1_proj: float,
        p2_proj: float,
        matched_points_distance_vox: float,
        matched_points_distance_nm: float,
        pixel_size: float,
    ) -> dict | None:
        if not left_boundary["resolved"] or not right_boundary["resolved"]:
            return None
        if (
            left_boundary.get("boundary_mode") == "anchor"
            and right_boundary.get("boundary_mode") == "anchor"
        ):
            r0 = {
                **minima_result,
                "p1_projection": float(p1_proj),
                "p2_projection": float(p2_proj),
                "matched_points_distance_vox": float(matched_points_distance_vox),
                "matched_points_distance_nm": float(matched_points_distance_nm),
                "left_outward_max": left_boundary.get("outward_max"),
                "right_outward_max": right_boundary.get("outward_max"),
            }
            return IntensityProfileAnalyzer._finalize_minima_only_row(
                prof, r0, distances, intensities, smoothed,
                float(pixel_size) if np.isfinite(pixel_size) else float("nan"),
            )
        axis_data = IntensityProfileAnalyzer._get_profile_axis_data(prof)
        if axis_data is None:
            return None
        _, _, unit_dir, midpoint, vs = axis_data
        lbp = float(left_boundary["boundary_position"])
        rbp = float(right_boundary["boundary_position"])
        lmp = minima_result["left_min"]["position"]
        ctp = minima_result["central_max"]["position"]
        rmp = minima_result["right_min"]["position"]
        if not (lbp < lmp < ctp < rmp < rbp):
            return None
        left_xyz = midpoint + lbp * unit_dir
        right_xyz = midpoint + rbp * unit_dir
        thick_vox = rbp - lbp
        thick_nm = thick_vox * vs if np.isfinite(vs) else np.nan
        delta_nm = (
            thick_nm - matched_points_distance_nm
            if np.isfinite(thick_nm) and np.isfinite(matched_points_distance_nm)
            else np.nan
        )
        lm = left_boundary["boundary_mode"]
        rm = right_boundary["boundary_mode"]
        boundaries_id = IntensityProfileAnalyzer._boundaries_identified_from_modes(lm, rm)
        lp = float(min(p1_proj, p2_proj))
        rp = float(max(p1_proj, p2_proj))
        return {
            **minima_result,
            "resolved": True, "failure_reason": None,
            "minima_identified": minima_result.get("detection_mode"),
            "detection_mode": boundaries_id,
            "p1_projection": float(p1_proj), "p2_projection": float(p2_proj),
            "left_boundary_position": lbp, "right_boundary_position": rbp,
            "left_boundary_mode": lm, "right_boundary_mode": rm,
            "left_outward_feature": left_boundary["outward_feature"],
            "right_outward_feature": right_boundary["outward_feature"],
            "left_outward_anchor": left_boundary.get("outward_anchor"),
            "right_outward_anchor": right_boundary.get("outward_anchor"),
            "left_outward_max": left_boundary.get("outward_max"),
            "right_outward_max": right_boundary.get("outward_max"),
            "left_zero_position": lbp, "right_zero_position": rbp,
            "left_zero_xyz": left_xyz, "right_zero_xyz": right_xyz,
            "membrane_thickness_vox": float(thick_vox),
            "membrane_thickness_nm": float(thick_nm),
            "matched_points_distance_vox": float(matched_points_distance_vox),
            "matched_points_distance_nm": float(matched_points_distance_nm),
            "delta_thickness_nm": float(delta_nm),
            "distances": distances, "intensities": intensities, "smoothed_profile": smoothed,
            "pixel_size": pixel_size,
            "left_inflection_minus_projection_nm": float((lbp - lp) * vs) if np.isfinite(vs) else np.nan,
            "right_inflection_minus_projection_nm": float((rbp - rp) * vs) if np.isfinite(vs) else np.nan,
        }

    @staticmethod
    def _finalize_minima_only_row(
        prof: dict, r: dict, distances: np.ndarray,
        intensities_axis: np.ndarray, smoothed: np.ndarray, vs_prof: float,
    ) -> dict:
        axis_data = IntensityProfileAnalyzer._get_profile_axis_data(prof)
        if axis_data is None:
            out = dict(r)
            out.pop("pass2_missing_sides", None)
            out.update({"failure_reason": "minima_only_invalid_axis", "resolved": False})
            return out
        _, _, unit_dir, midpoint, vs_axis = axis_data
        vs_use = vs_prof if np.isfinite(vs_prof) and vs_prof > 0 else vs_axis
        if not np.isfinite(vs_use) or vs_use <= 0:
            out = dict(r)
            out.pop("pass2_missing_sides", None)
            out.update({"failure_reason": "minima_only_invalid_pixel_size", "resolved": False})
            return out
        lm, rm = r.get("left_min"), r.get("right_min")
        if lm is None or rm is None:
            out = dict(r)
            out.pop("pass2_missing_sides", None)
            out.update({"failure_reason": "minima_only_missing_minima", "resolved": False})
            return out
        lp, rp = float(lm["position"]), float(rm["position"])
        sep_vox = abs(rp - lp)
        sep_nm = float(sep_vox * vs_use)
        mpd_nm = float(r.get("matched_points_distance_nm", np.nan))
        delta_nm = sep_nm - mpd_nm if np.isfinite(sep_nm) and np.isfinite(mpd_nm) else float("nan")
        out = dict(r)
        out.pop("pass2_missing_sides", None)
        out.update({
            "resolved": True, "failure_reason": None, "detection_mode": "minima_only",
            "left_boundary_position": float("nan"), "right_boundary_position": float("nan"),
            "left_boundary_mode": None, "right_boundary_mode": None,
            "left_outward_feature": None, "right_outward_feature": None,
            "left_outward_anchor": None, "right_outward_anchor": None,
            "left_outward_max": r.get("left_outward_max"),
            "right_outward_max": r.get("right_outward_max"),
            "left_zero_position": lp, "right_zero_position": rp,
            "left_zero_xyz": midpoint + lp * unit_dir,
            "right_zero_xyz": midpoint + rp * unit_dir,
            "membrane_thickness_vox": float(sep_vox), "membrane_thickness_nm": float(sep_nm),
            "delta_thickness_nm": float(delta_nm),
            "left_inflection_minus_projection_nm": float("nan"),
            "right_inflection_minus_projection_nm": float("nan"),
            "distances": distances, "intensities": intensities_axis,
            "smoothed_profile": np.asarray(smoothed, dtype=float),
            "pixel_size": float(vs_use),
        })
        return out

    @staticmethod
    def _extract_xyz_triplet(value: np.ndarray | None) -> tuple[float, float, float]:
        if value is None:
            return np.nan, np.nan, np.nan
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size != 3 or np.any(~np.isfinite(arr)):
            return np.nan, np.nan, np.nan
        return float(arr[0]), float(arr[1]), float(arr[2])

    @staticmethod
    def _profile_pair_midpoints_xyz(df: pd.DataFrame) -> np.ndarray:
        for x1, y1, z1, x2, y2, z2 in (
            ("x1_corr_voxel", "y1_corr_voxel", "z1_corr_voxel",
             "x2_corr_voxel", "y2_corr_voxel", "z2_corr_voxel"),
            ("x1_voxel", "y1_voxel", "z1_voxel", "x2_voxel", "y2_voxel", "z2_voxel"),
        ):
            if all(c in df.columns for c in (x1, y1, z1, x2, y2, z2)):
                xa = pd.to_numeric(df[x1], errors="coerce").to_numpy(dtype=float)
                ya = pd.to_numeric(df[y1], errors="coerce").to_numpy(dtype=float)
                za = pd.to_numeric(df[z1], errors="coerce").to_numpy(dtype=float)
                xb = pd.to_numeric(df[x2], errors="coerce").to_numpy(dtype=float)
                yb = pd.to_numeric(df[y2], errors="coerce").to_numpy(dtype=float)
                zb = pd.to_numeric(df[z2], errors="coerce").to_numpy(dtype=float)
                return np.column_stack((0.5*(xa+xb), 0.5*(ya+yb), 0.5*(za+zb)))
        return np.full((len(df), 3), np.nan, dtype=float)

    @staticmethod
    def _build_membrane_thickness_dataframe(
        thickness_df: pd.DataFrame,
        profiles: list[dict],
        boundary_results: list[dict],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        def _serialize_profile_array(values: np.ndarray | None) -> str:
            if values is None:
                return "[]"
            arr = np.asarray(values, dtype=float)
            return "[]" if arr.size == 0 else json.dumps(np.round(arr, 6).tolist(), separators=(",", ":"))

        corrected_df = thickness_df.reset_index(drop=True).copy()
        if len(corrected_df) == 0:
            corrected_df["resolved"] = pd.Series(dtype=bool)
            return corrected_df, corrected_df.copy()

        resolved = np.array([bool(r.get("resolved", False)) for r in boundary_results], dtype=bool)
        corrected_df["profile_index"] = np.arange(len(corrected_df), dtype=np.int32)
        corrected_df["resolved"] = resolved
        corrected_df["failure_reason"] = [r.get("failure_reason") for r in boundary_results]
        corrected_df["minima_identified"] = [r.get("minima_identified") for r in boundary_results]
        corrected_df["detection_mode"] = [r.get("detection_mode") for r in boundary_results]
        corrected_df["boundary_quality"] = [
            "minima_only" if r.get("detection_mode") == "minima_only"
            else (
                "inflection" if bool(r.get("resolved", False)) and r.get("detection_mode") in inflection_point_method
                else ("unresolved" if not r.get("resolved", False) else "other")
            )
            for r in boundary_results
        ]
        for col, key in (
            ("left_boundary_mode", "left_boundary_mode"),
            ("right_boundary_mode", "right_boundary_mode"),
            ("left_boundary_position", "left_boundary_position"),
            ("right_boundary_position", "right_boundary_position"),
            ("p1_projection", "p1_projection"),
            ("p2_projection", "p2_projection"),
            ("matched_points_distance_vox", "matched_points_distance_vox"),
            ("matched_points_distance_nm", "matched_points_distance_nm"),
            ("membrane_thickness_vox", "membrane_thickness_vox"),
            ("membrane_thickness_nm", "membrane_thickness_nm"),
            ("delta_thickness_nm", "delta_thickness_nm"),
        ):
            corrected_df[col] = [r.get(key, np.nan) for r in boundary_results]
        corrected_df["matched_point_distance_vox"] = corrected_df["matched_points_distance_vox"]
        corrected_df["matched_point_distance_nm"] = corrected_df["matched_points_distance_nm"]
        corrected_df["inflection_thickness_vox"] = corrected_df["membrane_thickness_vox"]
        corrected_df["inflection_thickness_nm"] = corrected_df["membrane_thickness_nm"]
        _no_max = corrected_df["detection_mode"] == "minima_only"
        corrected_df.loc[_no_max, "inflection_thickness_nm"] = np.nan
        corrected_df.loc[_no_max, "inflection_thickness_vox"] = np.nan
        for col, sub, key in (
            ("left_min_position_vox", "left_min", "position"),
            ("right_min_position_vox", "right_min", "position"),
            ("left_min_intensity", "left_min", "intensity"),
            ("right_min_intensity", "right_min", "intensity"),
            ("central_max_position_vox", "central_max", "position"),
            ("central_max_intensity", "central_max", "intensity"),
        ):
            corrected_df[col] = [
                r.get(sub, {}).get(key, np.nan) if r.get(sub) is not None else np.nan
                for r in boundary_results
            ]
        corrected_df["minima_separation_vox"] = [
            abs(r["right_min"]["position"] - r["left_min"]["position"])
            if r.get("left_min") is not None and r.get("right_min") is not None else np.nan
            for r in boundary_results
        ]

        left_xyz = np.array(
            [IntensityProfileAnalyzer._extract_xyz_triplet(r.get("left_zero_xyz")) for r in boundary_results],
            dtype=float,
        )
        right_xyz = np.array(
            [IntensityProfileAnalyzer._extract_xyz_triplet(r.get("right_zero_xyz")) for r in boundary_results],
            dtype=float,
        )
        for i, col in enumerate(("x1_corr_voxel", "y1_corr_voxel", "z1_corr_voxel")):
            corrected_df[col] = left_xyz[:, i]
        for i, col in enumerate(("x2_corr_voxel", "y2_corr_voxel", "z2_corr_voxel")):
            corrected_df[col] = right_xyz[:, i]

        left_xyz_int = np.nan_to_num(np.rint(left_xyz), nan=-1.0).astype(np.int32, copy=False)
        right_xyz_int = np.nan_to_num(np.rint(right_xyz), nan=-1.0).astype(np.int32, copy=False)
        left_xyz_int[~np.isfinite(left_xyz).all(axis=1)] = -1
        right_xyz_int[~np.isfinite(right_xyz).all(axis=1)] = -1
        for i, col in enumerate(("x1_corr_voxel_int", "y1_corr_voxel_int", "z1_corr_voxel_int")):
            corrected_df[col] = left_xyz_int[:, i]
        for i, col in enumerate(("x2_corr_voxel_int", "y2_corr_voxel_int", "z2_corr_voxel_int")):
            corrected_df[col] = right_xyz_int[:, i]

        pixel_sizes = np.array(
            [p.get("pixel_size", np.nan) if isinstance(p, dict) else np.nan for p in profiles],
            dtype=float,
        )
        corrected_df["pixel_size_nm"] = pixel_sizes
        corrected_df["left_min_position_nm"] = corrected_df["left_min_position_vox"] * pixel_sizes
        corrected_df["right_min_position_nm"] = corrected_df["right_min_position_vox"] * pixel_sizes
        corrected_df["central_max_position_nm"] = corrected_df["central_max_position_vox"] * pixel_sizes
        corrected_df["left_inflection_position_vox"] = corrected_df["left_boundary_position"]
        corrected_df["right_inflection_position_vox"] = corrected_df["right_boundary_position"]
        corrected_df["left_inflection_position_nm"] = corrected_df["left_boundary_position"] * pixel_sizes
        corrected_df["right_inflection_position_nm"] = corrected_df["right_boundary_position"] * pixel_sizes
        corrected_df["minima_separation_nm"] = corrected_df["minima_separation_vox"] * pixel_sizes
        corrected_df["left_inflection_minus_projection_nm"] = [
            r.get("left_inflection_minus_projection_nm", np.nan) for r in boundary_results
        ]
        corrected_df["right_inflection_minus_projection_nm"] = [
            r.get("right_inflection_minus_projection_nm", np.nan) for r in boundary_results
        ]

        profile_positions_vox, profile_positions_nm, profile_intensities_list = [], [], []
        for prof in profiles:
            axis_data = IntensityProfileAnalyzer._get_profile_axis_data(prof)
            if axis_data is None:
                profile_positions_vox.append("[]")
                profile_positions_nm.append("[]")
                profile_intensities_list.append("[]")
                continue
            dists, intens, _, _, ps = axis_data
            profile_positions_vox.append(_serialize_profile_array(dists))
            profile_intensities_list.append(_serialize_profile_array(intens))
            profile_positions_nm.append(
                _serialize_profile_array(dists * ps) if np.isfinite(ps) else "[]"
            )
        corrected_df["profile_positions_vox"] = profile_positions_vox
        corrected_df["profile_positions_nm"] = profile_positions_nm
        corrected_df["profile_intensities"] = profile_intensities_list

        corrected_df["within_max_distance"] = False
        resolved_df = corrected_df.loc[resolved].copy()
        return corrected_df, resolved_df

    @staticmethod
    def _calculate_point_projections(prof: dict) -> tuple[float, float]:
        try:
            p1, p2 = prof["p1"], prof["p2"]
            midpoint = prof["midpoint"]
            direction = p2 - p1
            length = np.linalg.norm(direction)
            if length == 0:
                return np.nan, np.nan
            unit_dir = direction / length
            return np.dot(p1 - midpoint, unit_dir), np.dot(p2 - midpoint, unit_dir)
        except Exception:
            return np.nan, np.nan

    @staticmethod
    def _calculate_statistics(results: dict, total_profiles: int, max_distance_nm: float) -> dict:
        boundary_results = results["boundary_results"]
        resolved_results = [r for r in boundary_results if r.get("resolved", False)]
        unresolved_results = [r for r in boundary_results if not r.get("resolved", False)]
        kept_results = [
            r for r in resolved_results
            if r.get("detection_mode") in inflection_point_method
            and np.isfinite(r.get("membrane_thickness_nm", np.nan))
            and r.get("membrane_thickness_nm", np.nan) <= max_distance_nm
        ]
        n_minima_only = sum(1 for r in resolved_results if r.get("detection_mode") == "minima_only")
        n_resolved = len(resolved_results)
        n_kept = len(kept_results)

        stats = {
            "total_profiles": total_profiles,
            "profiles_resolved": n_resolved,
            "profiles_unresolved": total_profiles - n_resolved,
            "resolution_rate": n_resolved / total_profiles if total_profiles > 0 else 0.0,
            "max_distance_nm": float(max_distance_nm),
            "profiles_kept_after_distance_filter": n_kept,
            "distance_filter_rate_total": n_kept / total_profiles if total_profiles > 0 else 0.0,
            "distance_filter_rate_resolved": n_kept / n_resolved if n_resolved > 0 else 0.0,
            "pixel_size_nm": np.nan,
            "failure_analysis": {},
            "boundary_mode_usage": {},
            "quality_metrics": {},
            "membrane_thickness_stats": {},
            "delta_thickness_stats": {},
            "profiles_minima_only": int(n_minima_only),
        }

        failure_counts: dict[str, int] = {}
        for result in unresolved_results:
            if result.get("failure_reason"):
                reason = result["failure_reason"]
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        stats["failure_analysis"] = failure_counts

        boundary_mode_counts: dict[str, int] = {}
        for result in kept_results:
            for mode in (result.get("left_boundary_mode"), result.get("right_boundary_mode")):
                if mode is not None:
                    boundary_mode_counts[mode] = boundary_mode_counts.get(mode, 0) + 1
        stats["boundary_mode_usage"] = boundary_mode_counts

        def _series_stats(values: list[float]) -> dict:
            arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
            if arr.size == 0:
                return {"mean": np.nan, "std": np.nan, "median": np.nan, "range": (np.nan, np.nan)}
            return {
                "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "median": float(np.median(arr)),
                "range": (float(np.min(arr)), float(np.max(arr))),
            }

        def _paired_stats(nm_vals: list[float], vox_vals: list[float]) -> dict:
            ns, vs = _series_stats(nm_vals), _series_stats(vox_vals)
            return {
                "mean_nm": ns["mean"], "std_nm": ns["std"], "median_nm": ns["median"], "range_nm": ns["range"],
                "mean_vox": vs["mean"], "std_vox": vs["std"], "median_vox": vs["median"], "range_vox": vs["range"],
            }

        separations_nm, separations_vox, prominences = [], [], []
        membrane_thickness_nm, membrane_thickness_vox = [], []
        delta_thickness_nm, delta_thickness_vox = [], []

        for result in resolved_results:
            lm, rm = result.get("left_min"), result.get("right_min")
            ps = result.get("pixel_size", np.nan)
            if lm is not None and rm is not None:
                sep_vox = float(rm["position"] - lm["position"])
                separations_vox.append(sep_vox)
                separations_nm.append(sep_vox * ps if np.isfinite(ps) else np.nan)
                prominences.extend([float(lm.get("prominence", np.nan)), float(rm.get("prominence", np.nan))])

        for result in kept_results:
            membrane_thickness_nm.append(result.get("membrane_thickness_nm", np.nan))
            membrane_thickness_vox.append(result.get("membrane_thickness_vox", np.nan))
            delta_thickness_nm.append(result.get("delta_thickness_nm", np.nan))
            if np.isfinite(result.get("membrane_thickness_vox", np.nan)) and np.isfinite(
                result.get("matched_points_distance_vox", np.nan)
            ):
                delta_thickness_vox.append(
                    result.get("membrane_thickness_vox", np.nan)
                    - result.get("matched_points_distance_vox", np.nan)
                )
            else:
                delta_thickness_vox.append(np.nan)

        stats["quality_metrics"] = {
            "separation_stats": _paired_stats(separations_nm, separations_vox),
            "prominence_stats": _series_stats(prominences),
        }
        stats["membrane_thickness_stats"] = _paired_stats(membrane_thickness_nm, membrane_thickness_vox)
        stats["delta_thickness_stats"] = _paired_stats(delta_thickness_nm, delta_thickness_vox)

        pixel_sizes = [
            float(r["pixel_size"]) for r in boundary_results
            if np.isfinite(r.get("pixel_size", np.nan))
        ]
        if pixel_sizes:
            stats["pixel_size_nm"] = float(np.median(pixel_sizes))
        return stats


#############################################
# Main Pipeline Functions
#############################################

def process_membrane_segmentation(
    segmentation_map: MapSource,
    output_path: PathOrStr = None,
    membrane_labels: dict[str, int] | None = None,
    step_size_marching_cubes: int = 1,
    subdivision_iterations: int = 0,
    surface_separation_mode: Literal["planar", "closed"] | dict[str, Literal["planar", "closed"]] = "planar",
    refine_normals: bool = True,
    radius_hit: float = 3.0,
    flip_normals: bool = True,
    batch_size: int = 2000,
    save_vertices_mrc: bool = False,
    save_split_surface_meshes: bool = False,
    smooth_sigma_segmentation: float | None = None,
    snap_vertices_to_boundary: bool = False,
    logger: logging.Logger = None,
) -> dict:
    """
    Extract and refine bilayer surface meshes from a segmentation volume.

    Runs marching cubes on each membrane label, orients and refines normals,
    separates the two bilayer leaflets, and writes a ``*_vertices_normals.csv``
    per membrane.

    Parameters
    ----------
    segmentation_map : MapSource
        Segmentation MRC path or already-loaded ndarray.
    output_path : PathOrStr, optional
        Output directory (defaults to the segmentation file directory).
    membrane_labels : dict of {str: int}, optional
        ``{name: label_id}`` mapping. Defaults to ``{"membrane": 1}``.
    step_size_marching_cubes : int, default 1
        Marching-cubes stride (larger = faster, coarser mesh).
    smooth_sigma_segmentation : float or None, default None
        Gaussian sigma (voxels) applied to the segmentation before marching cubes.
        ``None`` skips smoothing.
    subdivision_iterations : int, default 0
        Loop subdivision passes after mesh extraction (densifies the mesh).
        Normal flipping is suppressed when > 0 (subdivision normals are inward-facing).
    surface_separation_mode : {"planar", "closed"} or dict, default "planar"
        How the two bilayer leaflets are separated. ``"planar"`` uses PCA on vertex
        coordinates (bilayers, flat patches); ``"closed"`` uses connected-component
        labeling (vesicles, organelle membranes). Pass a ``dict`` to use different
        modes per label, e.g. ``{"NE": "planar", "OMM": "closed"}``. Labels not
        present in the dict fall back to ``"planar"``.
    snap_vertices_to_boundary : bool, default False
        When True, rounds marching-cubes vertices to integer voxels and retains only
        those on the segmentation surface boundary (inside the label with at least one
        outside face-neighbor). Use when you trust the segmentation and want matching
        to occur only between confirmed boundary voxels.
    refine_normals : bool, default True
        Run neighbor-based normal refinement within each separated surface.
    radius_hit : float, default 3.0
        Voxel-radius neighborhood for normal refinement.
    flip_normals : bool, default True
        Flip normals to point toward the membrane interior after refinement.
        Has no effect when ``subdivision_iterations > 0``.
    batch_size : int, default 2000
        Vertex batch size during normal refinement.
    save_vertices_mrc : bool, default False
        Save vertex positions as a binary MRC volume for visualization.
    save_split_surface_meshes : bool, default False
        Save each separated bilayer leaflet as a ``.ply`` file before normal flipping.
    logger : logging.Logger, optional
        Logger instance (created in ``output_path`` if ``None``).

    Returns
    -------
    dict[str, str] or None
        ``{membrane_name: path_to_vertices_normals_csv}`` for each successfully
        processed membrane, or ``None`` if the segmentation could not be read.
    """
    _seg_path = segmentation_map if isinstance(segmentation_map, (str, os.PathLike)) else None

    # Initialize logger and setup
    if logger is None:
        if output_path is None:
            output_path = os.path.dirname(_seg_path) if _seg_path is not None else "."
        logger = setup_logger(output_path)

    if membrane_labels is None:
        membrane_labels = {"membrane": 1}

    # Setup output directory
    if output_path is None:
        output_path = os.path.dirname(_seg_path) if _seg_path is not None else "."
    os.makedirs(output_path, exist_ok=True)

    base_name = (
        os.path.splitext(os.path.basename(_seg_path))[0] if _seg_path is not None else "segmentation"
    )
    try:
        segmentation = cryomap.read(segmentation_map)
        _, pixel_size_a, origin_a = cryomap.get_metadata(segmentation_map)
    except Exception as e:
        logger.error(f"Failed to read segmentation: {e}")
        return None
    pixel_size = pixel_size_a / 10.0  # Å → nm (rename to pixel_size with broader refactor)
    origin = tuple(v / 10.0 for v in origin_a)  # Å → nm

    output_files = {}

    # Process each membrane type
    for membrane_name, label_value in membrane_labels.items():
        logger.info(f"\nProcessing {membrane_name} (label {label_value})")

        membrane_mask = segmentation == label_value
        if not np.any(membrane_mask):
            logger.info(f"No voxels found for {membrane_name}")
            continue

        try:
            # STEP 1: Extract surface mesh (marching cubes via Mesh.from_mrc)
            logger.info(f"Extracting surface points (mesh vertices) with step size {step_size_marching_cubes} from segmentation after smoothing with sigma {smooth_sigma_segmentation}...")
            mesh = Mesh.from_mrc(
                membrane_mask.astype(float),
                pixel_size=1.0,       # keep voxel units throughout
                step_size=step_size_marching_cubes,
                smooth_sigma=smooth_sigma_segmentation,
            )
            if mesh.vertices is None or len(mesh.vertices) == 0:
                logger.info("No surface points found")
                continue
            logger.info(f"Extracted {len(mesh.vertices)} mesh vertices")

            # STEP 1b: Optional Loop subdivision
            if subdivision_iterations > 0:
                n_before = len(mesh.vertices)
                mesh.subdivide_mesh(iterations=subdivision_iterations, recompute_normals=True)
                logger.info(f"Loop subdivision ({subdivision_iterations} iter): {n_before} → {len(mesh.vertices)} vertices")

            # STEP 2: Optionally snap marching-cubes vertices to integer segmentation-boundary voxels
            if snap_vertices_to_boundary:
                logger.info(
                    "snap_vertices_to_boundary=True: retaining only mesh vertices that fall "
                    "on the segmentation surface (inside the label with at least one outside face-neighbour)."
                )
                filtered_verts, filtered_normals = _filter_to_segmentation_boundary(
                    mesh.vertices, mesh.normals, membrane_mask, logger
                )
                if len(filtered_verts) == 0:
                    logger.info("No boundary voxels remain after filter — skipping membrane")
                    continue
                mesh.vertices = filtered_verts
                mesh.normals = filtered_normals
                mesh.faces = None  # faces no longer valid after vertex subsetting

            vertex_volume = create_vertex_volume(mesh.vertices, membrane_mask.shape)

            # STEP 3: Orient normals globally (MST), separate, refine within each surface, flip.
            # Order matters: separate before refinement so that each surface's normals are only
            # smoothed against same-surface neighbors (cross-bilayer averaging corrupts the split).
            effective_flip = flip_normals and (subdivision_iterations == 0)
            if subdivision_iterations > 0 and flip_normals:
                logger.info(
                    "Normal flip suppressed: Loop subdivision normals are already inward-facing, so "
                    "flip_normals has no effect with subdivision_iterations > 0"
                )
            mesh.orient_normals_globally()

            # STEP 4: Separate bilayer surfaces (on raw/oriented normals, before refinement)
            _sep_mode = (
                surface_separation_mode.get(membrane_name, "planar")
                if isinstance(surface_separation_mode, dict)
                else surface_separation_mode
            )
            logger.info(f"Separating surfaces (mode='{_sep_mode}')...")
            if _sep_mode == "planar":
                surface1_mask, surface2_mask = mesh.separate_planar_surface()
            else:
                surface1_mask, surface2_mask = mesh.separate_closed_surface()

            if surface1_mask is not None and surface2_mask is not None and (surface1_mask.any() or surface2_mask.any()):
                logger.info(f"Successfully separated membrane surfaces:")
                logger.info(f"Surface 1: {np.sum(surface1_mask)} points")
                logger.info(f"Surface 2: {np.sum(surface2_mask)} points")
            else:
                logger.info("Could not separate membrane surfaces")
                n = len(mesh.vertices)
                surface1_mask = np.zeros(n, dtype=bool)
                surface2_mask = np.zeros(n, dtype=bool)

            if refine_normals:
                logger.info("Refining normals within surface 1...")
                mesh.refine_normals(radius_hit=radius_hit, batch_size=batch_size,
                                    mask=surface1_mask, inplace=True)
                logger.info("Refining normals within surface 2...")
                mesh.refine_normals(radius_hit=radius_hit, batch_size=batch_size,
                                    mask=surface2_mask, inplace=True)

            if save_split_surface_meshes and surface1_mask.any():
                for surf_mask, suffix in [
                    (surface1_mask, "surface1"),
                    (surface2_mask, "surface2"),
                ]:
                    surf = mesh.apply_vertex_mask(surf_mask)
                    path = os.path.join(output_path, f"{base_name}_{membrane_name}_{suffix}.ply")
                    surf.save(path)
                    logger.info(f"Saved split mesh: {path}")

            if effective_flip:
                mesh.normals = -mesh.normals

            # STEP 5: Save outputs
            if len(mesh.vertices) > 0:
                success = verify_and_save_outputs(
                    mesh.vertices,
                    mesh.normals,
                    vertex_volume,
                    surface1_mask,
                    surface2_mask,
                    membrane_name,
                    base_name,
                    output_path,
                    pixel_size,
                    origin,
                    save_vertices_mrc,
                    logger,
                )

                if success:
                    csv_output = os.path.join(output_path, f"{base_name}_{membrane_name}_vertices_normals.csv")
                    output_files[membrane_name] = csv_output

        except Exception as e:
            logger.error(f"Error processing {membrane_name}: {e}")
            traceback.print_exc()

    return output_files

def match_points(
    segmentation_map: MapSource,
    input_csv: PathOrStr,
    output_csv: PathOrStr | None = None,
    output_path: PathOrStr = None,
    max_distance_nm: float = 8.0,
    max_angle: float = 1.0,
    direction: Literal["1to2", "2to1"] = "1to2",
    use_gpu: bool = True,
    num_cpu_threads: int | None = None,
    query_batch_size: int = 200000,
    surface_separation_mode: Literal["planar", "closed"] = "planar",
    snap_vertices_to_boundary: bool = False,
    pixel_size_nm: float | None = None,
    logger: logging.Logger = None,
) -> tuple[str, str]:
    """
    Geometrically match bilayer surfaces and write a paired-point table.

    The output CSV is a **matched-point** artifact (not the inflection-based
    thickness from profile analysis). Each row stores ``match_distance_nm``
    plus the paired voxel and physical coordinates for both leaflet points.

    Parameters
    ----------
    segmentation_map : MapSource
        Segmentation MRC path or ndarray — read only for pixel size and origin.
    input_csv : PathOrStr
        ``*_vertices_normals.csv`` produced by ``process_membrane_segmentation``
        (columns: ``x_voxel``, ``y_voxel``, ``z_voxel``, ``normal_x/y/z``,
        ``surface1``, ``surface2``).
    output_csv : PathOrStr, optional
        Destination path for the matched-point CSV (auto-generated from
        ``input_csv`` stem + ``"_matched_points[_2to1].csv"`` when ``None``).
    output_path : PathOrStr, optional
        Output directory (defaults to the ``input_csv`` directory).
    max_distance_nm : float, default 8.0
        Hard cap on candidate source-target distance (nm). Converted to voxels
        internally; points farther apart are never considered.
    max_angle : float, default 1.0
        Half-angle (degrees) of the normal-aligned cone. Candidate targets whose
        lateral deviation from the source normal exceeds this are rejected.
    direction : {"1to2", "2to1"}, default "1to2"
        Which surface seeds the rays. ``"1to2"`` matches surface-1 points to
        their nearest surface-2 partner; ``"2to1"`` reverses the roles.
    use_gpu : bool, default True
        Prefer the CUDA kernel when a compatible GPU is available. Falls back to
        the CPU KDTree path automatically if CUDA is unavailable.
    num_cpu_threads : int, optional
        Numba thread count for the CPU path (``None`` = use all available).
    query_batch_size : int, optional
        KDTree query batch size for the CPU matcher (``None`` = unbatched).
        Use on machines where holding the full ball-query result exhausts memory.
    surface_separation_mode : {"planar", "closed"}, default "planar"
        Recorded in the stats file for provenance; does not affect matching logic
        here (surface separation was already applied upstream).
    snap_vertices_to_boundary : bool, default False
        Recorded in the stats file for provenance; does not affect matching logic
        here (vertex snapping was already applied upstream).
    pixel_size_nm : float, optional
        Voxel size in nanometres. When provided, overrides the value read from
        the MRC header. Required when the segmentation file was saved without
        voxel-size metadata (header reports 0 Å) — a common occurrence with
        masks exported from third-party tools.
    logger : logging.Logger, optional
        Logger instance (created in ``output_path`` if ``None``).

    Returns
    -------
    output_csv : str or None
        Path to the matched-point CSV, or ``None`` if matching failed.
    stats_file : str or None
        Path to the companion ``*_stats.txt`` with matching parameters and
        distance histogram.
    """
    _seg_path = segmentation_map if isinstance(segmentation_map, (str, os.PathLike)) else None

    # Validate inputs early with sensible defaults
    if _seg_path is not None and not os.path.exists(_seg_path):
        raise FileNotFoundError(f"Segmentation file not found: {_seg_path}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    # Validate and constrain parameters
    max_distance_nm = max(0.1, float(max_distance_nm))
    max_angle = np.clip(float(max_angle), 0.1, 45.0)
    direction = direction if direction in ["1to2", "2to1"] else "1to2"

    # Set default output directory and CSV
    if output_path is None:
        output_path = os.path.dirname(input_csv)
    os.makedirs(output_path, exist_ok=True)

    if output_csv is None:
        input_base = os.path.splitext(os.path.basename(input_csv))[0]
        dir_suffix = "_2to1" if direction == "2to1" else ""
        output_csv = os.path.join(output_path, f"{input_base}_matched_points{dir_suffix}.csv")

    # Initialize logger if not provided
    if logger is None:
        logger = setup_logger(output_path)

    # Get base name for statistics file
    output_base = os.path.splitext(os.path.basename(output_csv))[0]
    stats_file = os.path.join(output_path, f"{output_base}_stats.txt")

    # Read voxel size from MRC file
    try:
        _, pixel_size_a, origin_a = cryomap.get_metadata(segmentation_map)
    except Exception as e:
        logger.error(f"Failed to read segmentation: {e}")
        return None, None

    if pixel_size_nm is not None and pixel_size_nm > 0:
        pixel_size = float(pixel_size_nm)
        logger.info(f"Using user-supplied pixel size: {pixel_size:.4f} nm")
    else:
        pixel_size = pixel_size_a / 10.0  # Å → nm

    if not np.isfinite(pixel_size) or pixel_size < 0:
        raise ValueError(
            f"Invalid pixel size ({pixel_size_a} Å) read from segmentation MRC. "
            "Pass pixel_size_nm=<pixel_size_nm> explicitly to match_points "
            "or run_full_pipeline."
        )

    if pixel_size_nm is None and pixel_size <= 0:
        for line in [
            "=" * 60,
            "WARNING: No voxel-size metadata found in MRC header (0.0 Å).",
            "Falling back to 1.0 Å (0.1 nm).",
            "If this is not your actual voxel size, pass",
            "pixel_size_nm=<correct_value_in_nm> explicitly.",
            "=" * 60,
        ]:
            logger.warning(line)
        pixel_size = 0.1  # 1.0 Å → nm
    elif pixel_size_nm is None and pixel_size_a == 1.0:
        for line in [
            "=" * 60,
            "WARNING: Pixel size is exactly 1.0 Å (= 0.1 nm) — this is a",
            "common placeholder value written by tools that do not preserve",
            "voxel-size metadata. If this is not your actual voxel size,",
            "pass pixel_size_nm=<correct_value_in_nm> explicitly.",
            "=" * 60,
        ]:
            logger.warning(line)

    # Load data using unscaled voxel coordinates
    logger.info(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Use unscaled voxel coordinates for calculations
    points = df[["x_voxel", "y_voxel", "z_voxel"]].values.astype(np.float32)
    normals = df[["normal_x", "normal_y", "normal_z"]].values.astype(np.float32)
    surface1_mask = df["surface1"].values.astype(bool)
    surface2_mask = df["surface2"].values.astype(bool)

    # Check if CUDA is available if GPU requested
    gpu_available = False
    if use_gpu:
        try:
            import numba.cuda

            gpu_available = numba.cuda.is_available()
            if gpu_available:
                logger.info("CUDA GPU is available and will be used")
            else:
                logger.warning("CUDA GPU requested but not available - falling back to CPU")
        except ImportError:
            logger.warning("CUDA modules not available - falling back to CPU")

    # Run geometric surface matching
    logger.info(f"Starting surface matching ({direction})...")
    start_time = time.time()

    if use_gpu and gpu_available:
        # Use GPU implementation
        thickness_results, valid_mask, point_pairs = measure_thickness_gpu(
            points,
            normals,
            surface1_mask,
            surface2_mask,
            pixel_size=pixel_size,
            max_distance_nm=max_distance_nm,
            max_angle_degrees=max_angle,
            direction=direction,
            logger=logger,
        )
    else:
        # Use CPU implementation
        thickness_results, valid_mask, point_pairs = measure_thickness_cpu(
            points,
            normals,
            surface1_mask,
            surface2_mask,
            pixel_size=pixel_size,
            max_distance_nm=max_distance_nm,
            max_angle_degrees=max_angle,
            direction=direction,
            num_threads=num_cpu_threads,
            logger=logger,
            query_batch_size=query_batch_size,
        )

    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")

    seg_path_rs = Path(_seg_path).expanduser().resolve() if _seg_path is not None else None
    vtx_path_rs = Path(input_csv).expanduser().resolve()
    out_csv_rs = Path(output_csv).expanduser().resolve()

    backend_used = "cuda" if use_gpu and gpu_available else "cpu"
    matching_params: dict[str, Any] = {
        "segmentation_mrc_path": str(seg_path_rs),
        "vertices_normals_csv_path": str(vtx_path_rs),
        "matched_points_csv_path": str(out_csv_rs),
        "pixel_size_nm": pixel_size,
        "max_distance_nm": max_distance_nm,
        "max_angle_degrees": max_angle,
        "direction": direction,
        "surface_separation_mode": surface_separation_mode,
        "snap_vertices_to_boundary": snap_vertices_to_boundary,
        "matching_backend_used": backend_used,
        "num_cpu_threads": num_cpu_threads,
        "query_batch_size_cpu": query_batch_size,
        "matching_wall_time_s": round(processing_time, 4),
    }
    if use_gpu:
        matching_params["cuda_available_at_runtime"] = gpu_available

    # Generate and save statistics
    stats = generate_matching_statistics(
        thickness_results, valid_mask, points, surface1_mask, surface2_mask, pixel_size
    )
    save_matching_statistics(stats, stats_file, logger, matching_params=matching_params)

    # Create matched-point results for valid one-to-one assignments
    logger.info("\nCreating matched-points CSV with valid one-to-one assignments...")

    # Get indices of valid measurements
    valid_indices = np.where(valid_mask)[0]
    matched_indices = point_pairs[valid_mask]

    if len(valid_indices) == 0:
        logger.warning("No valid one-to-one surface matches found!")
        # Create empty DataFrame with expected columns
        thickness_df = pd.DataFrame(
            columns=[
                "match_id",
                "match_distance_nm",
                "point1_idx",
                "x1_voxel",
                "y1_voxel",
                "z1_voxel",
                "normal1_x",
                "normal1_y",
                "normal1_z",
                "surface1",
                "point2_idx",
                "x2_voxel",
                "y2_voxel",
                "z2_voxel",
                "normal2_x",
                "normal2_y",
                "normal2_z",
                "surface2",
            ]
        )
    else:
        src_points = df.loc[valid_indices, ["x_voxel", "y_voxel", "z_voxel"]].values.astype(np.float32)
        tgt_points = df.loc[matched_indices, ["x_voxel", "y_voxel", "z_voxel"]].values.astype(np.float32)
        # Create DataFrame with only valid surface matches
        thickness_df = pd.DataFrame(
            {
                "match_id": range(len(valid_indices)),
                "match_distance_nm": thickness_results[valid_mask],
                # Point 1 (source) information
                "point1_idx": valid_indices,
                "x1_voxel": src_points[:, 0],
                "y1_voxel": src_points[:, 1],
                "z1_voxel": src_points[:, 2],
                "normal1_x": df.loc[valid_indices, "normal_x"].values,
                "normal1_y": df.loc[valid_indices, "normal_y"].values,
                "normal1_z": df.loc[valid_indices, "normal_z"].values,
                "surface1": df.loc[valid_indices, "surface1"].values,
                # Point 2 (matched) information
                "point2_idx": matched_indices,
                "x2_voxel": tgt_points[:, 0],
                "y2_voxel": tgt_points[:, 1],
                "z2_voxel": tgt_points[:, 2],
                "normal2_x": df.loc[matched_indices, "normal_x"].values,
                "normal2_y": df.loc[matched_indices, "normal_y"].values,
                "normal2_z": df.loc[matched_indices, "normal_z"].values,
                "surface2": df.loc[matched_indices, "surface2"].values,
            }
        )

    logger.info(f"Saving {len(thickness_df)} matched point pairs to {output_csv}")
    thickness_df.to_csv(output_csv, index=False)

    # Log summary statistics
    if len(thickness_df) > 0:
        mean_dist_nm = thickness_df["match_distance_nm"].mean()
        median_dist_nm = thickness_df["match_distance_nm"].median()
        min_dist_nm = thickness_df["match_distance_nm"].min()
        max_dist_nm = thickness_df["match_distance_nm"].max()
        logger.info("Matched-point distance summary:")
        logger.info(f"  Mean: {mean_dist_nm:.3f} nm ({mean_dist_nm / pixel_size:.3f} vox)")
        logger.info(f"  Median: {median_dist_nm:.3f} nm ({median_dist_nm / pixel_size:.3f} vox)")
        logger.info(
            f"  Range: {min_dist_nm:.3f}-{max_dist_nm:.3f} nm "
            f"({min_dist_nm / pixel_size:.3f}-{max_dist_nm / pixel_size:.3f} vox)"
        )

    logger.info("Surface matching complete")

    return output_csv, stats_file

def analyse_intensity_profiles(
    thickness_csv: PathOrStr,
    tomogram_map: MapSource,
    output_path: PathOrStr,
    profile_half_width_nm: float = 6.0,
    max_distance_nm: float = 8.0,
    analyzer: "IntensityProfileAnalyzer | None" = None,
    save_cleaned_df: bool = True,
    save_profiles: bool = True,
    save_statistics: bool = True,
    segmentation_path: PathOrStr | None = None,
    membrane_label: str | None = None,
    logger: logging.Logger = None,
) -> dict:
    """
    Extract tomogram intensity profiles along matched point pairs and resolve
    membrane boundaries.

    Steps:

    1. Load ``*_matched_points*.csv`` and the tomogram.
    2. Extract z-score normalized intensity profiles
       (``cryomap.sample_line_profiles``; half-width ``profile_half_width_nm``).
    3. Run boundary detection via ``analyzer.detect()`` (``resolve_profile_features``
       under the hood), then persist outputs with ``save_int_results``.

    Parameters
    ----------
    thickness_csv : str or pathlib.Path
        Matched-point table from geometric matching.
    tomogram_map : MapSource
        Tomogram MRC path or ndarray (same grid as the segmentation).
    output_path : PathOrStr
        Directory for saved tables, pickles, and statistics text.
    profile_half_width_nm : float, default 6.0
        Half-width of sampled profile segments on each side of the midpoint (nm).
        Passed directly to ``cryomap.sample_line_profiles``; not part of the analyzer.
    max_distance_nm : float, default 8.0
        Distance cap (nm) applied to exported rows after boundary detection.
        Should match the ``max_distance_nm`` used for geometric matching upstream.
    analyzer : IntensityProfileAnalyzer, optional
        All detection parameters (smoothing, minima search, anchor search, parallelism).
        ``IntensityProfileAnalyzer()`` defaults are used when ``None``.
    save_cleaned_df : bool, default True
        Save ``{base}_thickness.csv`` (kept rows only).
    save_profiles : bool, default True
        Save ``{base}_int_profiles.pkl``.
    save_statistics : bool, default True
        Save ``{base}{stats_infix}_boundary_stats.txt``.
    membrane_label : str | None, optional
        Explicit membrane basename (e.g. ``OMM``) for the standalone log filename. When omitted
        but ``segmentation_path`` is set, inferred from ``thickness_csv`` if it follows
        ``{{segStem}}_{label}_matched_points*.csv`` naming from ``run_full_pipeline``.
    segmentation_path : str | pathlib.Path | None, optional
        Segmentation vol path for provenance and log naming (not voxel-read here).
    logger : logging.Logger, optional
        When ``None``, ``setup_logger`` writes ``pipeline_analysis_log_filename(...)`` inside ``output_path``.

    Returns
    -------
    dict
        The ``resolve_profile_features`` results dict plus ``saved_files`` and
        ``input_files`` metadata added before return.
    """
    if analyzer is None:
        analyzer = IntensityProfileAnalyzer()

    max_distance_nm = max(0.1, float(max_distance_nm))

    if output_path is None:
        output_path = os.path.dirname(thickness_csv)
    od_path = Path(output_path)
    od_path.mkdir(parents=True, exist_ok=True)

    _tomo_path = tomogram_map if isinstance(tomogram_map, (str, os.PathLike)) else None

    thickness_csv_p = Path(thickness_csv)
    tomogram_file = Path(_tomo_path) if _tomo_path is not None else None
    stats_suffix = ""

    if logger is None:
        ml_log: str | None = None
        if membrane_label is not None and isinstance(membrane_label, str) and membrane_label.strip():
            ml_log = membrane_label.strip()
        elif segmentation_path is not None and _tomo_path is not None:
            ml_log = _infer_membrane_suffix_from_csv(thickness_csv_p, segmentation_path)
        _pip_log_fn = pipeline_analysis_log_filename(
            segmentation_path,
            tomogram_file,
            membrane_labels=ml_log,
        )
        work_logger = setup_logger(str(od_path), log_filename=_pip_log_fn)
        work_logger.info(f"Writing pipeline diagnostics to {_pip_log_fn} (under {od_path.resolve()})")
    else:
        work_logger = logger

    thickness_df = pd.read_csv(thickness_csv_p)
    geo_ct = _matched_point_geometry_counts(thickness_df)

    work_logger.info("=== Extracting intensity profiles ===")
    if tomogram_file is not None:
        work_logger.info(f"Tomogram MRC: {tomogram_file.name}")
        work_logger.info(f"          path: {tomogram_file.expanduser().resolve()}")
    else:
        work_logger.info("Tomogram: <ndarray>")
    if segmentation_path is not None:
        seg_pp = Path(segmentation_path).expanduser()
        work_logger.info(f"Segmentation MRC: {seg_pp.name}")
        work_logger.info(f"            path: {seg_pp.resolve()}")
    work_logger.info(f"Matched-point CSV: {thickness_csv_p.name}")
    work_logger.info(f"             path: {thickness_csv_p.expanduser().resolve()}")
    if geo_ct is None:
        work_logger.info(
            f"             rows: {len(thickness_df):,} "
            "(voxel-coordinate columns incomplete — usable pair count not computed)"
        )
    else:
        nrow, nv = geo_ct
        work_logger.info(f"             rows: {nrow:,}; usable coordinate pairs (non-NaN): {nv:,}")

    _, tomo_pixel_size_a, _ = cryomap.get_metadata(tomogram_map)
    tomo_pixel_nm = tomo_pixel_size_a / 10.0  # Å → nm; used by feature-detection helpers

    work_logger.info("Z-score normalizing tomogram before profile sampling...")
    tomo_normalized = cryomap.normalize(tomogram_map)
    p1 = thickness_df[["x1_voxel", "y1_voxel", "z1_voxel"]].values
    p2 = thickness_df[["x2_voxel", "y2_voxel", "z2_voxel"]].values
    profiles = cryomap.sample_line_profiles(
        p1, p2, tomo_normalized,
        pixel_size_a=tomo_pixel_size_a,
        extension_half_width_a=profile_half_width_nm * 10.0,
    )
    for prof in profiles:
        prof["pixel_size"] = tomo_pixel_nm  # nm/vox; used by _build_membrane_thickness_dataframe

    work_logger.info("\nFinding profile minima and inflection points...")
    results = analyzer.detect(
        profiles=profiles,
        thickness_df=thickness_df,
        profile_half_width_nm=profile_half_width_nm,
        max_distance_nm=max_distance_nm,
        logger=work_logger,
    )

    work_logger.info(f"Applying max-distance filter to resolved profiles ({max_distance_nm:.3f} nm)...")

    # Print summary
    print_summary(results, logger=work_logger)

    # Save results
    work_logger.info(f"\nSaving results to: {od_path}")
    saved_files = save_int_results(
        results=results,
        thickness_csv=thickness_csv_p,
        output_path=od_path,
        profiles=profiles,
        save_cleaned_df=save_cleaned_df,
        save_profiles=save_profiles,
        save_statistics=save_statistics,
        csv_label="thickness",
        stats_infix=stats_suffix,
        tomogram_path=tomogram_file,
        segmentation_path=segmentation_path,
        logger=work_logger,
    )

    results["saved_files"] = saved_files
    input_meta = {
        "thickness_csv": thickness_csv_p.expanduser().resolve(),
        "tomogram_file": tomogram_file.expanduser().resolve() if tomogram_file is not None else None,
    }
    if segmentation_path is not None:
        input_meta["segmentation_path"] = Path(segmentation_path).expanduser().resolve()
    results["input_files"] = input_meta

    return results


def run_full_pipeline(
    segmentation_map: MapSource,
    output_path: PathOrStr = None,
    membrane_labels: dict[str, int] | None = None,
    # ── Surface extraction ────────────────────────────────────────────
    step_size_marching_cubes: int = 1,
    smooth_sigma_segmentation: float | None = None,
    subdivision_iterations: int = 0,
    snap_vertices_to_boundary: bool = False,
    surface_separation_mode: Literal["planar", "closed"] | dict[str, Literal["planar", "closed"]] = "planar",
    refine_normals: bool = True,
    radius_hit: float = 3.0,
    flip_normals: bool = True,
    save_vertices_mrc: bool = False,
    save_split_surface_meshes: bool = False,
    # ── Geometric matching ────────────────────────────────────────────
    max_distance_nm: float = 8.0,
    max_angle: float = 1.0,
    direction: Literal["1to2", "2to1"] = "1to2",
    use_gpu: bool = True,
    num_cpu_threads: int | None = None,
    batch_size: int = 2000,
    query_batch_size: int = 200000,
    pixel_size_nm: float | None = None,
    # ── Profile / boundary stage ──────────────────────────────────────
    extract_intensity_profiles: bool = True,
    tomogram_map: MapSource = None,
    profile_half_width_nm: float = 6.0,
    analyzer: "IntensityProfileAnalyzer | None" = None,
    intensity_save_profiles: bool = True,
    intensity_save_statistics: bool = True,
    compatibility_tolerance_nm: float = 0.01,
    save_thickness_mrc: bool = False,
) -> dict:
    """
    Run surface extraction, geometric surface matching, and optional profile analysis.

    Despite the historical ``extract_intensity_profiles`` flag name, stage three is
    **boundary finding**: extract short tomogram profiles along each geometrically
    matched pair, resolve inflection-based boundaries, and apply a final
    ``max_distance_nm`` cap in nanometers to the inflection thickness column.

    Stages
    ------
    1. ``process_membrane_segmentation`` — marching-cubes vertices/normals, optional
       densification, normal refinement, surface split, CSV export per membrane label.
    2. ``match_points`` — GPU/CPU normal-cone matching, writes
       ``{segbase}_{membrane}_matched_points*.csv`` plus companion ``*_stats.txt``.
    3. ``analyse_intensity_profiles`` (optional) — profiles + ``resolve_profile_features``
       + ``save_int_results`` producing ``*_thickness.csv`` (kept rows),
       ``*_int_profiles.pkl``, and ``*{prefix}_boundary_stats.txt`` (``prefix`` = table base + optional ``_min``).

    Parameters
    ----------
    segmentation_map : MapSource
        Input MRC segmentation on the tomogram grid.
    output_path : PathOrStr, optional
        Destination folder (defaults next to the segmentation file).
    membrane_labels : dict, optional
        ``{name: label_id}`` mapping of membrane names to label ids.

    Surface extraction
    ------------------
    step_size_marching_cubes : int, default 1
        Marching-cubes stride (larger = faster, coarser mesh).
    smooth_sigma_segmentation : float or None, default None
        Gaussian sigma (voxels) applied to the segmentation before marching cubes.
        ``None`` skips smoothing.
    subdivision_iterations : int, default 0
        Loop subdivision passes after mesh extraction (densifies the mesh).
        Normal flipping is suppressed when > 0 (subdivision normals are inward-facing).
    snap_vertices_to_boundary : bool, default False
        When True, replaces marching-cubes vertices with integer segmentation-boundary
        voxels before normal refinement. Use when you trust the segmentation and want
        matching to occur only between confirmed boundary voxels.
    surface_separation_mode : {"planar", "closed"}, default "planar"
        How the two bilayer leaflets are separated. ``"planar"`` uses PCA on vertex
        coordinates to split along the membrane plane (bilayers, flat patches);
        ``"closed"`` uses connected-component labeling (vesicles, organelle membranes).
    refine_normals : bool, default True
        Neighbor-based normal refinement after surface separation.
    radius_hit : float, default 3.0
        Voxel-radius neighborhood used during normal refinement.
    flip_normals : bool, default True
        Flip refined normals to point toward the membrane interior when possible.
        Has no effect when ``subdivision_iterations > 0``.
    save_vertices_mrc : bool, default False
        Save vertex positions as a binary MRC volume for visualization.
    save_split_surface_meshes : bool, default False
        Save each separated bilayer leaflet as a ``.ply`` file before normal flipping.

    Geometric matching
    ------------------
    max_distance_nm : float, default 8.0
        Nanometer cap for **both** the geometric matcher and the downstream
        inflection-thickness filter (passed through as ``max_distance_nm``).
    max_angle : float, default 1.0
        Half-angle (degrees) of the normal-aligned search cone.
    direction : {"1to2", "2to1"}, default "1to2"
        Which surface seeds the ray casting / one-to-one pairing.
    use_gpu : bool, default True
        Prefer CUDA inside ``match_points`` when available.
    num_cpu_threads : int, optional
        Thread hint for the CPU fallback path.
    batch_size : int, default 2000
        Vertex batch size for normal-refinement processing.
    query_batch_size : int, optional
        KD-tree query batch size for the CPU matcher (``None`` = unbatched).
        Useful on machines where holding the full query result in memory is problematic.
    pixel_size_nm : float, optional
        Voxel size in nanometres. Overrides the value embedded in the MRC header.
        Required when the segmentation was saved without voxel-size metadata
        (header reports 0 Å) — pass the true physical voxel size here instead.

    Profile / boundary stage
    ------------------------
    extract_intensity_profiles : bool, default True
        When ``True`` **and** ``tomogram_map`` is provided, run ``analyse_intensity_profiles``
        after geometric matching for every successfully matched membrane.
    tomogram_map : MapSource, optional
        Tomogram MRC path or ndarray aligned with ``segmentation_map`` (required when
        ``extract_intensity_profiles=True``).
    profile_half_width_nm : float, default 6.0
        Half-width of each sampled profile segment (nm) on both sides of the midpoint.
        Controls sampling extent; not part of the analyzer.
    analyzer : IntensityProfileAnalyzer, optional
        All profile detection parameters. ``IntensityProfileAnalyzer()`` defaults are
        used when ``None``.
    intensity_save_profiles, intensity_save_statistics : bool, default True
        Forwarded to ``analyse_intensity_profiles`` / ``save_int_results`` to control
        pickle + text exports (the resolved CSV is always written when the stage runs).
    compatibility_tolerance_nm : float, default 0.01
        Tolerance (nm) passed to ``validate_seg_tomo_compatibility`` before profiling.
    save_thickness_mrc : bool, default False
        Calls ``save_thickness_mrc`` to produce per-voxel median thickness
        MRC volumes. When ``extract_intensity_profiles=True``, uses inflection-point
        thickness; otherwise uses ``match_distance_nm`` from the matched-points CSV.

    Returns
    -------
    dict or None
        ``{membrane_name: {...}}`` on success, otherwise ``None`` when no surfaces were
        produced in stage one. Each value contains:

        - ``input_csv``: ``*_vertices_normals.csv`` from stage one.
        - ``thickness_csv``: **matched-point** CSV from stage two (legacy key name).
        - ``stats_file``: geometric matching statistics log.
        - ``intensity_results`` *(optional)*: lightweight status bundle when stage three runs.
        - ``thickness_mrc`` *(optional)*: paths dict from ``save_per_surface_mrc_helper`` when
          ``save_thickness_mrc=True`` and stage three completed successfully.

        When present, ``intensity_results`` includes ``status`` (``"completed"``, ``"skipped"``,
        ``"failed"``), the full ``analysis_results`` dict from ``analyse_intensity_profiles``,
        ``profiles_extracted``, ``profiles_resolved``, ``profiles_kept_after_distance_filter``,
        and ``resolution_rate``.

    Raises
    ------
    FileNotFoundError
        Segmentation or tomogram path does not exist.
    ValueError
        ``extract_intensity_profiles=True`` without ``tomogram_map``.

    Notes
    -----
    - Geometric outputs use ``*_matched_points*.csv``; profile exports reuse the
      segmentation basename with ``*_thickness.csv`` for the **kept** inflection table.
    - GPU acceleration benefits stage two most; stage three is dominated by Python/SciPy work.

    Examples
    --------
    Geometric stages only:

    >>> from memthick_260415 import run_full_pipeline
    >>> results = run_full_pipeline(
    ...     segmentation_map="membrane_seg.mrc",
    ...     output_path="analysis_results",
    ...     membrane_labels={"plasma_membrane": 1},
    ...     extract_intensity_profiles=False,
    ...     max_distance_nm=8.0,
    ... )

    Full stack including tomogram-driven boundary finding:

    >>> results = run_full_pipeline(
    ...     segmentation_map="membrane_seg.mrc",
    ...     output_path="analysis_with_profiles",
    ...     extract_intensity_profiles=True,
    ...     tomogram_map="tomogram.mrc",
    ...     profile_half_width_nm=6.0,
    ...     analyzer=IntensityProfileAnalyzer(),
    ... )
    >>> kept = results["plasma_membrane"]["intensity_results"]["profiles_kept_after_distance_filter"]
    """
    _seg_path = segmentation_map if isinstance(segmentation_map, (str, os.PathLike)) else None
    _tomo_path = tomogram_map if isinstance(tomogram_map, (str, os.PathLike)) else None

    # Validate inputs and parameters early
    if _seg_path is not None and not os.path.exists(_seg_path):
        raise FileNotFoundError(f"Segmentation file not found: {_seg_path}")

    # Validate optional profile / boundary stage requirements
    if extract_intensity_profiles:
        if tomogram_map is None:
            raise ValueError("tomogram_map is required when extract_intensity_profiles=True")
        if _tomo_path is not None and not os.path.exists(_tomo_path):
            raise FileNotFoundError(f"Tomogram file not found: {_tomo_path}")

    # Validate and set sensible defaults
    step_size_marching_cubes = max(1, int(step_size_marching_cubes))
    radius_hit = max(1.0, float(radius_hit))
    batch_size = max(100, int(batch_size))
    max_distance_nm = max(0.1, float(max_distance_nm))
    max_angle = np.clip(float(max_angle), 0.1, 45.0)
    direction = direction if direction in ["1to2", "2to1"] else "1to2"

    # Set output directory
    if output_path is None:
        output_path = os.path.dirname(_seg_path) if _seg_path is not None else "."
    os.makedirs(output_path, exist_ok=True)

    effective_ml = membrane_labels or {"membrane": 1}
    membrane_tokens = tuple(sorted(effective_ml.keys()))
    pip_log_fn = pipeline_analysis_log_filename(
        _seg_path,
        _tomo_path,
        membrane_labels=membrane_tokens,
    )
    logger = setup_logger(output_path, log_filename=pip_log_fn)
    logger.info(f"Starting full membrane thickness analysis pipeline for {_seg_path or '<ndarray>'}")

    if extract_intensity_profiles:
        logger.info(f"Will attempt finding intensity profile features using tomogram: {_tomo_path or '<ndarray>'}")

    # Get base name for output files
    base_name = (
        os.path.splitext(os.path.basename(_seg_path))[0] if _seg_path is not None else "segmentation"
    )

    # Step 1: Process membrane segmentation
    logger.info("Step 1: Processing membrane segmentation")
    output_files = process_membrane_segmentation(
        segmentation_map=segmentation_map,
        output_path=output_path,
        membrane_labels=membrane_labels,
        step_size_marching_cubes=step_size_marching_cubes,
        snap_vertices_to_boundary=snap_vertices_to_boundary,
        refine_normals=refine_normals,
        radius_hit=radius_hit,
        flip_normals=flip_normals,
        batch_size=batch_size,
        save_vertices_mrc=save_vertices_mrc,
        subdivision_iterations=subdivision_iterations,
        surface_separation_mode=surface_separation_mode,
        save_split_surface_meshes=save_split_surface_meshes,
        smooth_sigma_segmentation=smooth_sigma_segmentation,
        logger=logger,
    )

    if output_files is None or len(output_files) == 0:
        logger.error("No membrane surfaces found. Pipeline terminated.")
        return None

    # Step 2: Match points between the two surfaces
    logger.info("\nStep 2: Matching points between surfaces")
    results = {}

    for membrane_name, input_csv in output_files.items():
        logger.info(f"\nProcessing surface matches for {membrane_name}")

        dir_suffix = "_2to1" if direction == "2to1" else ""
        output_csv = os.path.join(output_path, f"{base_name}_{membrane_name}_matched_points{dir_suffix}.csv")

        _sep_mode = (
            surface_separation_mode.get(membrane_name, "planar")
            if isinstance(surface_separation_mode, dict)
            else surface_separation_mode
        )
        try:
            thickness_csv, stats_file = match_points(
                segmentation_map=segmentation_map,
                input_csv=input_csv,
                output_csv=output_csv,
                output_path=output_path,
                max_distance_nm=max_distance_nm,
                max_angle=max_angle,
                direction=direction,
                use_gpu=use_gpu,
                num_cpu_threads=num_cpu_threads,
                query_batch_size=query_batch_size,
                surface_separation_mode=_sep_mode,
                snap_vertices_to_boundary=snap_vertices_to_boundary,
                pixel_size_nm=pixel_size_nm,
                logger=logger,
            )

            if thickness_csv is not None:
                results[membrane_name] = {
                    "input_csv": input_csv,
                    "thickness_csv": thickness_csv,
                    "stats_file": stats_file,
                }

                logger.info(f"Point matching for {membrane_name} completed successfully")

                # Step 3: Optional profile extraction + boundary finding
                if not extract_intensity_profiles:
                    if save_thickness_mrc:
                        logger.info(f"\nStep 3b: Discretizing boundary distances for {membrane_name}")
                        try:
                            surface_paths = save_thickness_mrc(
                                thickness_csv=thickness_csv,
                                segmentation_map=segmentation_map,
                                output_path=output_path,
                                prefix=f"{base_name}_{membrane_name}",
                                thickness_col="match_distance_nm",
                                coord_suffix="voxel",
                                logger=logger,
                            )
                            results[membrane_name]["surface_volumes"] = surface_paths
                        except Exception as e:
                            logger.error(f"Error generating surface volumes for {membrane_name}: {e}")
                    continue

                if extract_intensity_profiles:
                    logger.info(
                        f"\nStep 3: Extracting tomogram profiles and resolving boundaries for {membrane_name}"
                    )

                    try:
                        # Validate segmentation-tomogram compatibility
                        compatible, details = validate_seg_tomo_compatibility(
                            segmentation_map, tomogram_map, tolerance=compatibility_tolerance_nm, logger=logger
                        )

                        if not compatible:
                            logger.warning(
                                f"Skipping profile-based boundary finding for {membrane_name}: {details}"
                            )
                            results[membrane_name]["intensity_results"] = {
                                "status": "skipped",
                                "reason": "incompatible_files",
                                "details": details,
                            }
                            continue

                        # Run profile pipeline (extract → resolve → save)
                        intensity_results = analyse_intensity_profiles(
                            thickness_csv=thickness_csv,
                            tomogram_map=tomogram_map,
                            output_path=output_path,
                            save_cleaned_df=True,
                            save_profiles=intensity_save_profiles,
                            save_statistics=intensity_save_statistics,
                            profile_half_width_nm=profile_half_width_nm,
                            max_distance_nm=max_distance_nm,
                            analyzer=analyzer,
                            segmentation_path=_seg_path,
                            logger=logger,
                        )

                        results[membrane_name]["intensity_results"] = {
                            "status": "completed",
                            "analysis_results": intensity_results,
                            "profiles_extracted": intensity_results.get("statistics", {}).get(
                                "total_profiles", 0
                            ),
                            "profiles_resolved": intensity_results.get("statistics", {}).get(
                                "profiles_resolved", 0
                            ),
                            "profiles_kept_after_distance_filter": intensity_results.get("statistics", {}).get(
                                "profiles_kept_after_distance_filter", 0
                            ),
                            "resolution_rate": intensity_results.get("statistics", {}).get("resolution_rate", 0.0),
                        }

                        logger.info(f"Successfully analysed intensity profiles for {membrane_name}")

                        # Stage 4: optional per-surface voxel aggregation
                        if save_thickness_mrc:
                            logger.info(f"\nStep 4: Discretizing per-voxel median thickness volumes for {membrane_name}")
                            try:
                                thickness_csv_path = intensity_results.get("saved_files", {}).get("thickness_csv")
                                if thickness_csv_path is not None and Path(thickness_csv_path).exists():
                                    surface_paths = save_thickness_mrc(
                                        thickness_csv=thickness_csv_path,
                                        segmentation_map=segmentation_map,
                                        output_path=output_path,
                                        prefix=f"{base_name}_{membrane_name}",
                                        logger=logger,
                                    )
                                    results[membrane_name]["surface_volumes"] = surface_paths
                                    logger.info(f"Surface volumes saved for {membrane_name}")
                                else:
                                    logger.warning(f"Skipping surface volumes for {membrane_name}: thickness CSV not found")
                            except Exception as e:
                                logger.error(f"Error generating surface volumes for {membrane_name}: {e}")

                    except Exception as e:
                        logger.error(f"Error in profile-based boundary finding for {membrane_name}: {e}")
                        results[membrane_name]["intensity_results"] = {"status": "failed", "error": str(e)}

        except Exception as e:
            logger.error(f"Error matching points for {membrane_name}: {e}")
            traceback.print_exc()

    logger.info("\nFull pipeline completed!")

    # Print compact pipeline summary
    if extract_intensity_profiles:
        for membrane_name, result in results.items():
            if "intensity_results" in result:
                status = result["intensity_results"]["status"]
                if status == "completed":
                    logger.info(
                        f"{membrane_name}: ✓ Completed - {result['intensity_results']['profiles_kept_after_distance_filter']} profiles after filtering"
                    )
                elif status == "skipped":
                    logger.info(f"{membrane_name}: ⚠ Skipped - {result['intensity_results']['reason']}")
                elif status == "failed":
                    logger.info(f"{membrane_name}: ✗ Failed - {result['intensity_results']['error']}")

    return results


#############################################
# Per-Leaflet Voxel Discretization
#############################################


def discretize_thickness_per_voxel(
    df: pd.DataFrame,
    thickness_col: str = "membrane_thickness_nm",
    coord_suffix: str = "corr_voxel",
    logger: logging.Logger = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Discretize thickness measurements to integer voxels per surface.

    Groups float boundary positions by their nearest integer voxel and
    computes the median thickness and re-normalized mean normals per voxel.

    Parameters
    ----------
    df : pd.DataFrame
        Thickness/matched-points table. Must contain ``{thickness_col}``,
        ``x1_{coord_suffix}``, ``y1_{coord_suffix}``, ``z1_{coord_suffix}``,
        ``x2_{coord_suffix}``, ``y2_{coord_suffix}``, ``z2_{coord_suffix}``,
        and normal columns.
    thickness_col : str, default "membrane_thickness_nm"
        Column to aggregate as median thickness. Pass ``"match_distance_nm"`` to
        rasterize geometric matched-point distances instead of inflection thickness.
    coord_suffix : str, default "corr_voxel"
        Suffix appended to ``x1_``, ``y1_``, etc. to select the coordinate columns.
        Pass ``"voxel"`` when using a matched-points CSV (columns ``x1_voxel`` etc.).
    logger : logging.Logger, optional

    Returns
    -------
    g1 : pd.DataFrame
        Surface-1 discretized table with columns: ``x``, ``y``, ``z`` (int32 voxel
        indices), ``median_{thickness_col}`` (median), ``n`` (count), ``normal_x``,
        ``normal_y``, ``normal_z`` (re-normalized mean).
    g2 : pd.DataFrame
        Same structure for surface 2.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    if thickness_col not in df.columns:
        raise ValueError(
            f"Thickness column '{thickness_col}' not found in DataFrame. "
            f"Available columns: {list(df.columns)}. "
            'Pass thickness_col="match_distance_nm" for a *_matched_points.csv, '
            'or thickness_col="membrane_thickness_nm" for a *_thickness.csv.'
        )

    required_coord_cols = [
        f"x1_{coord_suffix}", f"y1_{coord_suffix}", f"z1_{coord_suffix}",
        f"x2_{coord_suffix}", f"y2_{coord_suffix}", f"z2_{coord_suffix}",
    ]
    missing = [c for c in required_coord_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Coordinate columns not found in DataFrame: {missing}. "
            f"Available columns: {list(df.columns)}. "
            'Pass coord_suffix="voxel" for a *_matched_points.csv, '
            'or coord_suffix="corr_voxel" for a *_thickness.csv.'
        )

    m = np.isfinite(df[thickness_col].to_numpy())
    df = df.loc[m].copy()
    log_msg(f"Discretizing {m.sum()} rows to integer voxels ({(~m).sum()} non-finite dropped)")

    for side, xs, ys, zs in [
        (1, f"x1_{coord_suffix}", f"y1_{coord_suffix}", f"z1_{coord_suffix}"),
        (2, f"x2_{coord_suffix}", f"y2_{coord_suffix}", f"z2_{coord_suffix}"),
    ]:
        df[f"gx{side}"] = np.rint(df[xs].astype(float)).astype(np.int32)
        df[f"gy{side}"] = np.rint(df[ys].astype(float)).astype(np.int32)
        df[f"gz{side}"] = np.rint(df[zs].astype(float)).astype(np.int32)

    def _normalize_normals(nx, ny, nz):
        v = np.stack([nx, ny, nz], axis=1).astype(np.float64)
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return (v / norms)[:, 0], (v / norms)[:, 1], (v / norms)[:, 2]

    def _discretize_side(df_in, gx_col, gy_col, gz_col, nx_col, ny_col, nz_col):
        g = (
            df_in.assign(_nx=df_in[nx_col].astype(float), _ny=df_in[ny_col].astype(float), _nz=df_in[nz_col].astype(float))
            .groupby([gx_col, gy_col, gz_col], as_index=False)
            .agg(**{
                f"median_{thickness_col}": (thickness_col, "median"),
                "n": (thickness_col, "count"),
                "normal_x": ("_nx", "mean"),
                "normal_y": ("_ny", "mean"),
                "normal_z": ("_nz", "mean"),
            })
        )
        nx2, ny2, nz2 = _normalize_normals(g["normal_x"].to_numpy(), g["normal_y"].to_numpy(), g["normal_z"].to_numpy())
        g["normal_x"], g["normal_y"], g["normal_z"] = nx2, ny2, nz2
        return g

    g1 = _discretize_side(df, "gx1", "gy1", "gz1", "normal1_x", "normal1_y", "normal1_z")
    g2 = _discretize_side(df, "gx2", "gy2", "gz2", "normal2_x", "normal2_y", "normal2_z")

    g1 = g1.rename(columns={"gx1": "x", "gy1": "y", "gz1": "z"})
    g2 = g2.rename(columns={"gx2": "x", "gy2": "y", "gz2": "z"})

    log_msg(f"Leaflet 1: {len(g1)} unique voxels; Leaflet 2: {len(g2)} unique voxels")
    return g1, g2


def save_per_surface_mrc_helper(
    g1: pd.DataFrame,
    g2: pd.DataFrame,
    segmentation_map: MapSource,
    output_path: PathOrStr,
    prefix: str,
    value_col: str,
    logger: logging.Logger = None,
) -> dict[str, Path]:
    """
    Rasterize per-voxel discretized thickness tables to MRC volumes and CSV files.

    Parameters
    ----------
    g1, g2 : pd.DataFrame
        Discretized surface tables from ``discretize_thickness_per_voxel``.
        Must contain ``x``, ``y``, ``z``, ``{value_col}``, ``n``, and normal columns.
    segmentation_map : MapSource
        Segmentation MRC path or ndarray used to read ``shape_zyx``, ``pixel_size``, and ``origin``.
    output_path : PathOrStr
        Destination directory (created if missing).
    prefix : str
        Base name prefix for output files (e.g. the membrane name stem).
    value_col : str
        Column in ``g1``/``g2`` to rasterize (e.g. ``"median_membrane_thickness_nm"``).
        Used as-is for rasterization and written to the CSV under the same name.
        A ``label_token`` is derived from it for filenames by stripping the leading
        ``"median_"`` and trailing ``"_nm"``.
    logger : logging.Logger, optional

    Returns
    -------
    dict[str, Path]
        Keys ``surface1_mrc``, ``surface2_mrc``, ``surface1_csv``, ``surface2_csv``.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    shape_zyx, pixel_size_a, _ = cryomap.get_metadata(segmentation_map)
    pixel_size_nm = pixel_size_a / 10.0  # Å → nm

    def _rasterize(g):
        vol = np.zeros(shape_zyx, dtype=np.float32)
        z = g["z"].to_numpy(np.int64)
        y = g["y"].to_numpy(np.int64)
        x = g["x"].to_numpy(np.int64)
        t = g[value_col].to_numpy(np.float32)
        Z, Y, X = shape_zyx
        inside = (z >= 0) & (z < Z) & (y >= 0) & (y < Y) & (x >= 0) & (x < X)
        vol[z[inside], y[inside], x[inside]] = t[inside]
        return vol

    label_token = value_col.removeprefix("median_").removesuffix("_nm")

    saved = {}
    for tag, g, label in [("surface1", g1, "Surface 1"), ("surface2", g2, "Surface 2")]:
        vol = _rasterize(g)
        mrc_path = output_path / f"{prefix}_{tag}_{label_token}_discretized_nm.mrc"
        cryomap.write(vol.astype(np.float32), mrc_path,
                      transpose=False, pixel_size=pixel_size_nm * 10.0)  # nm → Å
        log_msg(f"Saved {label} volume: {mrc_path}")
        saved[f"{tag}_mrc"] = mrc_path

        csv_path = output_path / f"{prefix}_{tag}_{label_token}_discretized_nm.csv"
        g.to_csv(csv_path, index=False)
        log_msg(f"Saved {label} discretized CSV: {csv_path}")
        saved[f"{tag}_csv"] = csv_path

    return saved

def save_thickness_mrc(
    thickness_csv: PathOrStr,
    segmentation_map: MapSource,
    output_path: PathOrStr,
    prefix: str = None,
    thickness_col: str = "membrane_thickness_nm",
    coord_suffix: str = "corr_voxel",
    logger: logging.Logger = None,
) -> dict[str, Path]:
    """
    Load a thickness/matched-points CSV, discretize per voxel, and save MRC volumes and CSVs.

    Combines ``discretize_thickness_per_voxel`` and ``save_per_surface_mrc_helper``.

    Output filenames are derived automatically from ``thickness_col`` by stripping
    the leading ``"median_"`` and trailing ``"_nm"`` to form a ``label_token``:
    ``{prefix}_surface1_{label_token}_discretized_nm.mrc`` and
    ``{prefix}_surface1_{label_token}_discretized_nm.csv``.

    Parameters
    ----------
    thickness_csv : str or Path
        Resolved thickness CSV from ``analyse_intensity_profiles``, or a
        matched-points CSV when profiles were not extracted.
    segmentation_map : MapSource
        Segmentation MRC path or ndarray used for shape/voxel/origin metadata.
    output_path : str or Path
        Destination directory.
    prefix : str, optional
        Output file prefix. Defaults to the stem of ``thickness_csv``.
    thickness_col : str, default "membrane_thickness_nm"
        Column to aggregate as median per voxel. Pass ``"match_distance_nm"`` to
        rasterize geometric matched-point distances instead of inflection thickness.
    coord_suffix : str, default "corr_voxel"
        Suffix for coordinate columns (``x1_{coord_suffix}`` etc.). Pass ``"voxel"``
        when using a matched-points CSV (columns ``x1_voxel`` etc.).
    logger : logging.Logger, optional

    Returns
    -------
    dict[str, Path]
        Paths to saved files (two MRCs + two CSVs).
    """
    thickness_csv = Path(thickness_csv)
    if prefix is None:
        prefix = thickness_csv.stem
        for suffix in ["_thickness", "_matched_points"]:
            if prefix.endswith(suffix):
                prefix = prefix[: -len(suffix)]
                break

    df = pd.read_csv(thickness_csv)
    g1, g2 = discretize_thickness_per_voxel(df, thickness_col=thickness_col, coord_suffix=coord_suffix, logger=logger)
    value_col = f"median_{thickness_col}"
    saved = save_per_surface_mrc_helper(g1, g2, segmentation_map, output_path, prefix, value_col=value_col, logger=logger)

    return saved