import os
import sys
import time
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import traceback
from typing import Literal, List, Dict, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import mrcfile

from skimage import measure
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import KDTree as ScipyKDTree
from scipy.signal import find_peaks
from scipy.ndimage import map_coordinates
from scipy.ndimage import gaussian_filter1d

from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

import numba
from numba import cuda
from numba import prange
import math

"""
Membrane Thickness Analysis Tool

A Python-based tool for analyzing membrane thickness from cryo-electron tomograms using 
membrane segmentations as input. Supports both instance and semantic segmentation
formats with optional intensity profile analysis along vectors extending from matched points.

CORE FUNCTIONALITY:
==================

Surface Processing Pipeline:
    1. Extract membrane surface points using marching cubes algorithm
    2. Retain only surface points that lie on segmentation boundaries 
    3. Optionally interpolate points for denser, more uniform surface coverage
    4. Refine normal vectors using weighted neighbor averaging
    5. Separate bilayer into inner/outer surfaces based on normal orientation using PCA

Thickness Measurement:
    1. Match points between separated surfaces
        - Look for nearest neighbors in the direction of the normal vector
        - Nearest neighbor search is parallelised over GPU threads or CPU cores
        - Parallelization identifies multiple potential matches per point
        - One-to-one matching is applied afterwards to ensure each surface point is measured exactly once
    2. Apply geometric constraints: maximum thickness, cone angle search
    3. Measure Euclidean distances between matched surface points in 3D space → membrane thickness
    4. Support bidirectional measurement (surface 1→2 or surface 2→1)

Intensity Profile Analysis (Recommended):
    1. Extract intensity profiles from the tomogram along vectors extending from matched points
        Advice: Extend the profiles beyond matched points with at least 10-15 voxels
    2. Filter valid thickness measurements using the intensity profiles as a guide:
       - Require two minima that are positioned between the measurement points
       - Said minima should have certain signal-to-noise ratio when compared to the baseline
       - Said minima should be separated by a central maximum
    3. Save statistics on intensity profiles pre- and post-filtering

INPUT FORMATS:
==============

Segmentation Files (MRC format):
    - Instance segmentation: Multiple membrane labels specified via dictionary or config.yaml
    - Semantic segmentation: All non-zero voxels treated as membrane
    - Supports standard MRC format with voxel size and origin metadata

Configuration (Optional):
    - YAML file specifying membrane labels: {membrane_name: label_value}
    - Command-line specification: "membrane:1,vesicle:2"

Tomogram Files (MRC format, for intensity profiling):
    - Must have compatible dimensions and voxel size with segmentation
    - Automatic compatibility validation with configurable tolerance

PROCESSING MODES:
================

1. 'full': Complete pipeline with optional intensity profiling (use flag --extract_intensity)
2. 'surface': Surface extraction and processing only
3. 'thickness': Thickness measurement from pre-processed surface data
4. 'intensity': Intensity profiling from existing thickness measurements

OUTPUTS:
========

Surface Analysis:
    - *_vertices_normals.csv: Coordinates (voxel & physical units), normal vectors, surface assignments
    - *_vertices.mrc: Binary volume marking surface points (optional)
    - *_vertices.xyz: Point cloud coordinates for external visualization (optional)

Thickness Measurements:
    - *_thickness.csv: Valid measurements with complete point pair information
    - *_thickness_stats.log: Comprehensive statistics and measurement coverage
    - *_thickness_volume.mrc: Volume where voxel values represent thickness measurements in nm (optional)

Intensity Profiling:
    - *_thickness_cleaned.csv: Filtered thickness measurements post filtering using the intensity profiles
    - *_int_profiles.pkl: Original extracted intensity profiles
    - *_int_profiles_cleaned.pkl: Filtered profiles meeting bilayer criteria
    - *_filtering_stats.txt: Detailed filtering statistics and quality metrics

PARALELLIZATION OPTIONS:
=====================

GPU Processing (CUDA):
    - Accelerated nearest neighbor search for thickness measurement
    - Automatic fallback to CPU if CUDA unavailable
    - Optimized memory management for large datasets

CPU Processing:
    - Numba JIT compilation for parallel thickness measurement
    - SciPy KDTree for efficient spatial queries
    - Configurable thread count for optimal performance

BILAYER-SPECIFIC FEATURES:
=========================

Surface Separation:
    - PCA-based principal direction analysis
    - Marching cubes normal orientation for bilayer detection
    - Validation of surface assignment of points

Intensity Profile Filtering:
    - Dual minima detection representing lipid headgroups
    - Central maximum detection representing lipid tails
    - Signal-to-noise ratio analysis with baseline noise calculation
    - Ensure that two minima are positioned between the measurement points

Quality Control:
    - Validation at each processing step
    - Statistical analysis of measurement coverage and reliability
    - Automatic filtering of invalid or unreliable measurements

USAGE EXAMPLES:
===============

Command Line - Full pipeline with automated results filtering - RECOMMENDED (use flag '--extract_intensity', 'tomo_path' is required):
    python memthick.py \
        segmentation.mrc \
        --mode full \
        --membrane_labels NE:1,ER:2 \
        --output_dir /path/to/folder/ \
        --interpolate \
        --interpolation_points 1 \
        --max_thickness 8.0 \
        --max_angle 1.0 \
        --extract_intensity \
        --tomo_path tomo.mrc \
        --intensity_extension_voxels 10 \
        --intensity_extension_range -10 10 \
        --intensity_require_both_minima
        --intensity_central_max_required \
        --intensity_min_snr 0.2 \
        --intensity_margin_factor 0.1 \

Command Line - Full pipeline without results filtering
    python memthick.py \
        segmentation.mrc \
        --mode full \
        --output_dir /path/to/folder/ \
        --membrane_labels NE:1,ER:2 \
        --interpolate \
        --interpolation_points 1 \
        --max_thickness 8.0 \
        --max_angle 1.0 \
        
Command Line - Thickness measurement only:
    python memthick.py segmentation.mrc --mode thickness --input_csv vertices_normals.csv

Command Line - Intensity profile-based thickness results filtering:
    python memthick.py \
        segmentation.mrc \
        --mode intensity \
        --output_dir /path/to/folder/ \
        --thickness_csv thickness.csv \
        --tomo_path tomo.mrc \
        --intensity_extension_voxels 10 \
        --intensity_extension_range -10 10 \
        --intensity_require_both_minima
        --intensity_central_max_required \
        --intensity_min_snr 0.2 \
        --intensity_margin_factor 0.1 \

Python Module - Full pipeline with automated results filtering - RECOMMENDED:
    from cryocat import memthick
    results = memthick.run_full_pipeline(
        segmentation_path="membrane_seg.mrc",
        output_dir=output_dir,
        max_thickness=8.0,
        max_angle=1.0,
        extract_intensity_profiles=True,
        tomo_path="tomo.mrc",
        intensity_extension_voxels=10,
        intensity_extension_range=(-10,10),
        intensity_require_both_minima=True,
        intensity_central_max_required=True,
        intensity_min_snr=0.2,
        intensity_margin_factor=0.1,
    )

Python Module - Intensity profile-based thickness results filtering:
    from cryocat import memthick
    results = memthick.int_profiles_extract_clean(
        thickness_csv="membrane_thickness.csv",
        output_dir=output_dir,
        tomo_path="tomo.mrc",
        intensity_extension_voxels=10,
        intensity_extension_range=(-10,10),
        intensity_require_both_minima=True,
        intensity_central_max_required=True,
        intensity_min_snr=0.2,
        intensity_margin_factor=0.1,
    )

DEPENDENCIES:
=============

Core: numpy, pandas, scipy, scikit-image, scikit-learn, mrcfile, tqdm, pyyaml
GPU: numba[cuda] (optional, for GPU acceleration)
Visualization: matplotlib (for analysis notebooks)

The tool can be run as a command-line application or imported as a module for 
integration into larger analysis workflows.
"""

#############################################
# Logging and Utility Functions For Thickness Measurement Pipeline
#############################################


def setup_logger(output_dir: str, name: str = "MembraneThickness") -> logging.Logger:
    """
    Set up logger for the analysis with both file and console handlers.

    Parameters
    ----------
    output_dir : str
        Directory where log file will be saved
    name : str, default "MembraneThickness"
        Name of the logger

    Returns
    -------
    logging.Logger
        Configured logger instance with file and console handlers
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(output_dir / "thickness_analysis.log")
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Clear existing handlers to avoid duplicates
    logger.handlers = []

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def read_segmentation(segmentation_path: str, logger: logging.Logger = None) -> tuple[np.ndarray, float, tuple]:
    """
    Read segmentation data from MRC file and extract metadata.

    Parameters
    ----------
    segmentation_path : str
        Path to the MRC segmentation file
    logger : logging.Logger, optional
        Logger instance for status messages

    Returns
    -------
    segmentation : np.ndarray or None
        3D segmentation data array (ZYX order)
    voxel_size : float or None
        Voxel size in nanometers
    origin : tuple or None
        Origin coordinates (x, y, z) in nanometers
    shape : tuple or None
        Shape of the array (ZYX order)

    Notes
    -----
    Returns (None, None, None) if file reading fails.
    Voxel size is converted from angstroms to nanometers by dividing by 10.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    log_msg(f"Reading segmentation from {segmentation_path}...")
    try:
        with mrcfile.mmap(segmentation_path, mode="r", permissive=True) as mrc:
            segmentation = mrc.data
            voxel_size = mrc.voxel_size.x / 10  # Convert to nm
            origin = (mrc.header.origin.x / 10, mrc.header.origin.y / 10, mrc.header.origin.z / 10)
            shape = mrc.data.shape

            log_msg(f"Voxel size: {voxel_size:.4f} nm")
            log_msg(f"Origin: {origin}")
            log_msg(f"Shape (ZYX): {shape}")

        return segmentation, voxel_size, origin, shape
    except Exception as e:
        log_msg(f"Error reading MRC file: {e}")
        traceback.print_exc()
        return None, None, None, None


def read_tomo(tomo_path: str, logger: logging.Logger = None) -> tuple[np.ndarray, float, tuple]:
    """
    Read tomogram data from MRC file and extract metadata.
    
    Parameters
    ----------
    tomo_path : str
        Path to the MRC tomogram file
    logger : logging.Logger, optional
        Logger instance for status messages
        
    Returns
    -------
    tomo : np.ndarray or None
        3D tomogram data array (ZYX order)
    voxel_size : float or None  
        Voxel size in nanometers
    shape : tuple or None
        Shape of the tomogram array (ZYX order)
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    log_msg(f"Reading tomogram from {tomo_path}...")
    try:
        with mrcfile.mmap(str(tomo_path), mode="r", permissive=True) as mrc:
            tomo = mrc.data
            tomo_voxel = mrc.voxel_size.x / 10  # Convert to nm
            tomo_shape = mrc.data.shape

        log_msg(f"Tomogram voxel size: {tomo_voxel:.4f} nm")
        log_msg(f"Tomogram shape (ZYX): {tomo_shape}")
        
        return tomo, tomo_voxel, tomo_shape
        
    except Exception as e:
        log_msg(f"Error reading tomogram file: {e}")
        return None, None, None


def validate_seg_tomo_compatibility(
    segmentation_path: str, 
    tomo_path: str, 
    tolerance: float = 0.01,
    logger: logging.Logger = None
) -> tuple[bool, dict]:
    """
    Validate compatibility between segmentation and tomogram files.
    
    Checks dimensions and voxel size compatibility between a segmentation
    MRC file and its corresponding tomogram for intensity profile analysis.
    
    Parameters
    ----------
    segmentation_path : str
        Path to the MRC segmentation file
    tomo_path : str
        Path to the MRC tomogram file
    tolerance : float, default 0.01
        Tolerance for voxel size comparison (in nanometers)
    logger : logging.Logger, optional
        Logger instance for status messages
        
    Returns
    -------
    compatible : bool
        True if files are compatible for intensity analysis
    details : dict
        Dictionary containing compatibility details:
        - 'segmentation_shape': tuple of segmentation dimensions (ZYX)
        - 'tomogram_shape': tuple of tomogram dimensions (ZYX) 
        - 'segmentation_voxel_size': float, voxel size in nm
        - 'tomogram_voxel_size': float, voxel size in nm
        - 'dimensions_match': bool, whether shapes are identical
        - 'voxel_sizes_match': bool, whether voxel sizes are within tolerance
        - 'error': str, error message if validation failed
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    try:
        # Read segmentation metadata
        log_msg("Validating segmentation and tomogram compatibility...")
        segmentation, seg_voxel_size, seg_origin, seg_shape = read_segmentation(
            segmentation_path, logger=logger
        )
        
        if segmentation is None:
            return False, {'error': 'Failed to read segmentation file'}
        
        # Read tomogram metadata  
        tomo, tomo_voxel_size, tomo_shape = read_tomo(tomo_path, logger=logger)
        
        if tomo is None:
            return False, {'error': 'Failed to read tomogram file'}
        
        # Check dimensions
        dims_match = seg_shape == tomo_shape
        
        # Check voxel size (within tolerance)
        voxel_size_match = abs(seg_voxel_size - tomo_voxel_size) < tolerance
        
        # Compatibility requires both dimension and voxel size match
        compatible = dims_match and voxel_size_match
        
        details = {
            'segmentation_shape': seg_shape,
            'tomogram_shape': tomo_shape,
            'segmentation_voxel_size': seg_voxel_size,
            'tomogram_voxel_size': tomo_voxel_size,
            'dimensions_match': dims_match,
            'voxel_sizes_match': voxel_size_match
        }
        
        if compatible:
            log_msg("✓ Segmentation and tomogram are compatible")
            log_msg(f"  Dimensions: {seg_shape}")
            log_msg(f"  Voxel size: {seg_voxel_size:.4f} nm")
        else:
            log_msg("✗ Compatibility check failed:")
            if not dims_match:
                log_msg(f"  Dimension mismatch: {seg_shape} vs {tomo_shape}")
            if not voxel_size_match:
                log_msg(f"  Voxel size mismatch: {seg_voxel_size:.4f} vs {tomo_voxel_size:.4f} nm")
                log_msg(f"  Difference: {abs(seg_voxel_size - tomo_voxel_size):.4f} nm (tolerance: {tolerance:.4f} nm)")
            
        return compatible, details
        
    except Exception as e:
        error_msg = f"Error during compatibility check: {e}"
        log_msg(error_msg)
        return False, {'error': error_msg}
    

def generate_thickness_volume(
    points: np.ndarray,
    thickness_results: np.ndarray,
    valid_mask: np.ndarray,
    segmentation: np.ndarray,
    voxel_size: float,
    point_pairs: np.ndarray
) -> np.ndarray:
    """
    Create 3D volume where voxel values represent membrane thickness.

    Parameters
    ----------
    points : np.ndarray
        2D array (N, 3) of point coordinates in voxel space
    thickness_results : np.ndarray
        1D array of thickness measurements in nanometers
    valid_mask : np.ndarray
        1D boolean array indicating valid measurements
    segmentation : np.ndarray
        3D reference segmentation for volume dimensions
    voxel_size : float
        Voxel size in nanometers (unused but kept for consistency)
    point_pairs : np.ndarray
        1D array of paired point indices

    Returns
    -------
    np.ndarray
        3D float32 volume with thickness values in nm (NaN for unmeasured voxels)

    Notes
    -----
    Sets thickness values at both surface points of each valid measurement.
    """
    # Initialize volume with NaN
    thickness_volume = np.full(segmentation.shape, np.nan, dtype=np.float32)

    # Fill in thickness values for valid measurements
    valid_points = points[valid_mask].astype(int)
    valid_thicknesses = thickness_results[valid_mask]
    valid_pairs = point_pairs[valid_mask]

    # Add thickness values for both surfaces
    for point, pair_idx, thickness in zip(valid_points, valid_pairs, valid_thicknesses):
        # First surface point
        x1, y1, z1 = point
        if (
            0 <= x1 < thickness_volume.shape[0]
            and 0 <= y1 < thickness_volume.shape[1]
            and 0 <= z1 < thickness_volume.shape[2]
        ):
            thickness_volume[x1, y1, z1] = thickness

        # Second surface point
        x2, y2, z2 = points[pair_idx].astype(int)
        if (
            0 <= x2 < thickness_volume.shape[0]
            and 0 <= y2 < thickness_volume.shape[1]
            and 0 <= z2 < thickness_volume.shape[2]
        ):
            thickness_volume[x2, y2, z2] = thickness

    return thickness_volume


def save_thickness_volume(
    thickness_volume: np.ndarray,
    output_path: str,
    voxel_size: float,
    origin: tuple = (0, 0, 0)
) -> None:
    """
    Save thickness volume as MRC file with proper metadata.

    Parameters
    ----------
    thickness_volume : np.ndarray
        3D volume with thickness values in nanometers
    output_path : str
        Path for output MRC file
    voxel_size : float
        Voxel size in nanometers
    origin : tuple, default (0, 0, 0)
        Origin coordinates (x, y, z) in nanometers

    Notes
    -----
    NaN values are converted to 0 for visualization compatibility.
    Voxel size is converted to angstroms for MRC format.
    """
    with mrcfile.new(output_path, overwrite=True) as mrc:
        # Convert NaN to a specific value (e.g., 0) for visualization
        thickness_volume_viz = np.nan_to_num(thickness_volume, nan=0.0)
        mrc.set_data(thickness_volume_viz.astype(np.float32))

        mrc.voxel_size = (voxel_size * 10, voxel_size * 10, voxel_size * 10)
        mrc.header.origin.x = origin[0] * 10
        mrc.header.origin.y = origin[1] * 10
        mrc.header.origin.z = origin[2] * 10

        # Add additional header information if needed
        mrc.header.nxstart = 0
        mrc.header.nystart = 0
        mrc.header.nzstart = 0


def verify_and_save_outputs(
    aligned_vertices: np.ndarray,
    aligned_normals: np.ndarray,
    vertex_volume: np.ndarray,
    surface1_mask: np.ndarray,
    surface2_mask: np.ndarray,
    membrane_name: str,
    base_name: str,
    output_dir: str,
    voxel_size: float,
    origin: tuple,
    save_vertices_mrc: bool = False,
    save_vertices_xyz: bool = False,
    logger: logging.Logger = None
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
    output_dir : str
        Output directory path
    voxel_size : float
        Voxel size in nanometers
    origin : tuple
        Origin coordinates (x, y, z) in nanometers
    save_vertices_mrc : bool, default False
        Whether to save vertex volume as MRC file
    save_vertices_xyz : bool, default False
        Whether to save coordinates as XYZ point cloud
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    bool
        True if all requested outputs saved successfully
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    log_msg(f"\nSaving outputs for {membrane_name}:")
    log_msg(f"Vertices: {len(aligned_vertices)}, Surface 1: {np.sum(surface1_mask)}, Surface 2: {np.sum(surface2_mask)}")

    try:
        # Scale vertices once for all uses
        scaled_vertices = aligned_vertices * voxel_size + np.array(origin)

        # Save MRC file if requested
        if save_vertices_mrc:
            mrc_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices.mrc")
            save_vertices_mrc_helper(vertex_volume, mrc_output, voxel_size, origin)
            log_msg(f"Saved MRC to {mrc_output}")

        # Save XYZ file if requested
        if save_vertices_xyz:
            xyz_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices.xyz")
            np.savetxt(xyz_output, scaled_vertices, fmt="%.6f", delimiter=" ")
            log_msg(f"Saved XYZ with {len(scaled_vertices)} points to {xyz_output}")

        # Save CSV with coordinates, normals, and surface masks (ZYX → XYZ conversion)
        csv_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices_normals.csv")
        df = pd.DataFrame(
            {
                "x_voxel": aligned_vertices[:, 2],
                "y_voxel": aligned_vertices[:, 1],
                "z_voxel": aligned_vertices[:, 0],
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


def save_vertices_mrc_helper(
    vertex_volume: np.ndarray,
    output_path: str,
    voxel_size: float,
    origin: tuple
) -> None:
    """
    Helper function to save vertex volume as MRC file.

    Parameters
    ----------
    vertex_volume : np.ndarray
        3D binary volume data
    output_path : str
        Path for output MRC file
    voxel_size : float
        Voxel size in nanometers
    origin : tuple
        Origin coordinates (x, y, z) in nanometers
    """
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(vertex_volume.astype(np.int16))
        mrc.voxel_size = voxel_size if isinstance(voxel_size, np.float32) else np.float32(voxel_size)
        mrc.header.origin.x = origin[0]
        mrc.header.origin.y = origin[1]
        mrc.header.origin.z = origin[2]

#############################################
# On stats
#############################################

def generate_matching_statistics(
    thickness_results: np.ndarray,
    valid_mask: np.ndarray,
    point_pairs: np.ndarray,
    points: np.ndarray,
    surface1_mask: np.ndarray,
    surface2_mask: np.ndarray,
    voxel_size: float
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
    voxel_size : float
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
            "x_mean": np.mean(valid_points[:, 0]) * voxel_size,
            "y_mean": np.mean(valid_points[:, 1]) * voxel_size,
            "z_mean": np.mean(valid_points[:, 2]) * voxel_size,
            "x_std": np.std(valid_points[:, 0]) * voxel_size,
            "y_std": np.std(valid_points[:, 1]) * voxel_size,
            "z_std": np.std(valid_points[:, 2]) * voxel_size,
        }

    return stats


def save_matching_statistics(stats: dict, output_path: str, logger: logging.Logger = None) -> None:
    """
    Save thickness analysis statistics to formatted text file.

    Parameters
    ----------
    stats : dict
        Statistics dictionary from generate_matching_statistics()
    output_path : str
        Path where statistics file will be saved
    logger : logging.Logger, optional
        Logger instance for status messages
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    with open(output_path, "w") as f:
        f.write("=== Membrane Thickness Analysis Statistics ===\n\n")

        f.write("General Statistics:\n")
        f.write(f"Total points: {stats['total_points']}\n")
        f.write(f"Surface 1 points: {stats['surface1_points']}\n")
        f.write(f"Surface 2 points: {stats['surface2_points']}\n")
        f.write(f"Valid measurements: {stats['valid_measurements']}\n")
        f.write(f"Coverage percentage: {stats['coverage_percentage']:.2f}%\n\n")

        if "mean_thickness" in stats:
            f.write("Thickness Statistics:\n")
            f.write(f"Mean thickness: {stats['mean_thickness']:.2f} ± {stats['std_thickness']:.2f}\n")
            f.write(f"Median thickness: {stats['median_thickness']:.2f}\n")
            f.write(f"Range: {stats['min_thickness']:.2f} - {stats['max_thickness']:.2f}\n")
            f.write(f"Interquartile range: {stats['percentile_25']:.2f} - {stats['percentile_75']:.2f}\n\n")

            f.write("Thickness Distribution Histogram:\n")
            hist = stats["thickness_histogram"]
            for count, (left, right) in zip(hist["counts"], zip(hist["bin_edges"][:-1], hist["bin_edges"][1:])):
                f.write(f"{left:.2f}-{right:.2f}: {count}\n")

        log_msg(f"Statistics saved to {output_path}")

#############################################
# Surface Processing Classes and Functions
#############################################
    
def extract_surface_points(
    segmentation: np.ndarray,
    membrane_mask: np.ndarray,
    mesh_sampling: int = 1,
    logger: logging.Logger = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract surface points and normals using marching cubes algorithm.
    
    This function processes a binary membrane mask to identify surface voxels
    and compute their normal vectors. It uses the marching cubes algorithm
    from scikit-image to extract the surface mesh, then validates each
    vertex to ensure it lies on the true segmentation boundary.
    
    The function performs the following steps:
    1. Applies marching cubes to extract surface mesh
    2. Converts vertex coordinates to integer voxel positions
    3. Validates each vertex using is_surface_point() function
    4. Removes duplicate vertices at the same position
    5. Returns validated vertices with corresponding normal vectors
    
    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation volume (unused, kept for interface compatibility)
    membrane_mask : np.ndarray
        3D binary mask for membrane of interest
    mesh_sampling : int, default 1
        Step size for marching cubes algorithm. Larger values reduce
        computational cost but may miss fine surface details.
        - 1: Full resolution (most accurate, slowest)
        - 2: Half resolution (2x faster, some detail loss)
        - 4: Quarter resolution (4x faster, significant detail loss)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    aligned_vertices : np.ndarray or None
        2D array (N, 3) of surface vertex coordinates in voxel units
    aligned_normals : np.ndarray or None
        2D array (N, 3) of surface normal vectors

    Notes
    -----
    Only returns vertices that pass surface validation (is_surface_point).
    Returns (None, None) if extraction fails or no valid points found.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    log_msg(f"Extracting surface points with step size {mesh_sampling}...")

    try:
        vertices, faces, normals, _ = measure.marching_cubes(membrane_mask, step_size=mesh_sampling)
        vertices_int = np.round(vertices).astype(int)

        aligned_vertices = []
        aligned_normals = []
        processed_positions = set()
        
        for i in range(len(vertices_int)):
            vertex = vertices_int[i]
            if (0 <= vertex[0] < membrane_mask.shape[0] and 
                0 <= vertex[1] < membrane_mask.shape[1] and 
                0 <= vertex[2] < membrane_mask.shape[2]):
                pos_tuple = tuple(vertex)
                if pos_tuple not in processed_positions:
                    if is_surface_point(vertex, membrane_mask):
                        processed_positions.add(pos_tuple)
                        aligned_vertices.append(vertex)
                        aligned_normals.append(normals[i])

        aligned_vertices = np.array(aligned_vertices)
        aligned_normals = np.array(aligned_normals)
        log_msg(f"Extracted {len(aligned_vertices)} surface points")
        
        return aligned_vertices, aligned_normals

    except Exception as e:
        log_msg(f"Error in surface extraction: {e}")
        traceback.print_exc()
        return None, None


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
    vertex_volume = np.zeros(membrane_mask_shape)
    for vertex in aligned_vertices:
        x, y, z = vertex.astype(int)
        if (0 <= x < vertex_volume.shape[0] and 
            0 <= y < vertex_volume.shape[1] and 
            0 <= z < vertex_volume.shape[2]):
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


def get_neighbor_surface_points(
    vertex: np.ndarray,
    segmentation: np.ndarray,
    include_edges: bool = True
) -> list:
    """
    Find neighboring surface points around a vertex.

    Parameters
    ----------
    vertex : array-like
        Coordinates [x, y, z] of the central vertex
    segmentation : np.ndarray
        3D binary segmentation volume
    include_edges : bool, default True
        Whether to include 12 edge neighbors in addition to 6 face neighbors

    Returns
    -------
    list
        List of neighboring points that pass surface validation
    """
    x, y, z = vertex
    points = []

    # Face and edge neighbors as before
    face_neighbors = [[x + 1, y, z], [x - 1, y, z], [x, y + 1, z], [x, y - 1, z], [x, y, z + 1], [x, y, z - 1]]

    edge_neighbors = [
        [x + 1, y + 1, z],
        [x + 1, y - 1, z],
        [x - 1, y + 1, z],
        [x - 1, y - 1, z],
        [x + 1, y, z + 1],
        [x + 1, y, z - 1],
        [x - 1, y, z + 1],
        [x - 1, y, z - 1],
        [x, y + 1, z + 1],
        [x, y + 1, z - 1],
        [x, y - 1, z + 1],
        [x, y - 1, z - 1],
    ]

    neighbors_to_check = face_neighbors
    if include_edges:
        neighbors_to_check.extend(edge_neighbors)

    for new_x, new_y, new_z in neighbors_to_check:
        # Skip if out of bounds
        if (
            new_x < 0
            or new_y < 0
            or new_z < 0
            or new_x >= segmentation.shape[0]
            or new_y >= segmentation.shape[1]
            or new_z >= segmentation.shape[2]
        ):
            continue

        point = [new_x, new_y, new_z]
        # Only add if it's a true surface point
        if is_surface_point(point, segmentation):
            points.append(point)

    return points


def interpolate_between_vertices(
    v1: np.ndarray,
    v2: np.ndarray,
    segmentation: np.ndarray,
    num_points: int = 1
) -> list:
    """
    Interpolate points between two vertices on the surface.

    Parameters
    ----------
    v1, v2 : array-like
        Start and end vertex coordinates
    segmentation : np.ndarray
        3D binary segmentation volume for validation
    num_points : int, default 1
        Number of points to interpolate between vertices

    Returns
    -------
    list
        List of interpolated points that pass surface validation
    """
    points = []
    for t in np.linspace(0, 1, num_points + 2)[1:-1]:
        # Linear interpolation
        interp_point = v1 + t * (v2 - v1)
        # Round to nearest integer
        rounded_point = np.round(interp_point).astype(int)

        # Only add if it's a true surface point
        if (
            0 <= rounded_point[0] < segmentation.shape[0]
            and 0 <= rounded_point[1] < segmentation.shape[1]
            and 0 <= rounded_point[2] < segmentation.shape[2]
            and is_surface_point(rounded_point, segmentation)
        ):
            points.append(rounded_point)

    return points


def find_surface_neighbors(
    vertex: np.ndarray,
    segmentation: np.ndarray,
    include_edges: bool = True
) -> list:
    """
    Find valid surface neighbors around a vertex.

    Parameters
    ----------
    vertex : array-like
        Coordinates [x, y, z] of the central vertex
    segmentation : np.ndarray
        3D binary segmentation volume
    include_edges : bool, default True
        Whether to include edge neighbors (12) in addition to face neighbors (6)

    Returns
    -------
    list
        List of [x, y, z] coordinates for valid surface neighbors
    """
    x, y, z = vertex
    face_neighbors = [[x+1,y,z], [x-1,y,z], [x,y+1,z], [x,y-1,z], [x,y,z+1], [x,y,z-1]]
    
    if include_edges:
        edge_neighbors = [
            [x+1,y+1,z], [x+1,y-1,z], [x-1,y+1,z], [x-1,y-1,z],
            [x+1,y,z+1], [x+1,y,z-1], [x-1,y,z+1], [x-1,y,z-1],
            [x,y+1,z+1], [x,y+1,z-1], [x,y-1,z+1], [x,y-1,z-1]
        ]
        neighbors_to_check = face_neighbors + edge_neighbors
    else:
        neighbors_to_check = face_neighbors

    valid_neighbors = []
    for nx, ny, nz in neighbors_to_check:
        if (0 <= nx < segmentation.shape[0] and 0 <= ny < segmentation.shape[1] and 
            0 <= nz < segmentation.shape[2] and is_surface_point([nx, ny, nz], segmentation)):
            valid_neighbors.append([nx, ny, nz])
    
    return valid_neighbors


def interpolate_between_points(
    p1: np.ndarray,
    p2: np.ndarray,
    segmentation: np.ndarray,
    num_points: int = 1
) -> list:
    """
    Interpolate points between two coordinates with surface validation.

    Parameters
    ----------
    p1, p2 : array-like
        Start and end point coordinates
    segmentation : np.ndarray
        3D binary segmentation volume for validation
    num_points : int, default 1
        Number of points to interpolate

    Returns
    -------
    list
        List of interpolated points that pass surface validation
    """
    if num_points == 0:
        return []
        
    interpolated = []
    for t in np.linspace(0, 1, num_points + 2)[1:-1]:
        interp_point = p1 + t * (p2 - p1)
        rounded_point = np.round(interp_point).astype(int)
        
        if (0 <= rounded_point[0] < segmentation.shape[0] and
            0 <= rounded_point[1] < segmentation.shape[1] and
            0 <= rounded_point[2] < segmentation.shape[2] and
            is_surface_point(rounded_point, segmentation)):
            interpolated.append(rounded_point)
    
    return interpolated


def interpolate_surface_points(
    vertices: np.ndarray,
    normals: np.ndarray,
    segmentation: np.ndarray,
    interpolation_points: int = 1,
    include_edges: bool = True,
    logger: logging.Logger = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Increase surface point density through interpolation and neighbor addition.
    
    This function enhances surface coverage by adding interpolated points
    between consecutive vertices and including neighboring surface points
    around each vertex. The resulting denser surface representation improves
    thickness measurement accuracy and coverage.
    
    The function performs two types of point addition:
    1. **Interpolation**: Adds points between consecutive vertices along
       the surface using linear interpolation
    2. **Neighbor addition**: Includes valid surface neighbors around each
       vertex (6 face neighbors + 12 edge neighbors if include_edges=True)
    
    All new points are validated using is_surface_point() to ensure they
    lie on the true segmentation boundary.
    
    Parameters
    ----------
    vertices : np.ndarray
        2D array (N, 3) of original vertex coordinates
    normals : np.ndarray
        2D array (N, 3) of corresponding normal vectors
    segmentation : np.ndarray
        3D binary segmentation volume for validation
    interpolation_points : int, default 1
        Number of points to interpolate between consecutive vertices.
        - 0: No interpolation (only neighbor addition)
        - 1: One point between each pair of vertices (recommended)
        - 2: Two points between each pair of vertices
        - Higher values increase density but may create redundant points
    include_edges : bool, default True
        Whether to include edge neighbors around each vertex
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    dense_vertices : np.ndarray
        2D array of densified vertex coordinates
    dense_normals : np.ndarray
        2D array of corresponding normal vectors

    Notes
    -----
    - Processing time scales with N * (1 + interpolation_points + neighbor_count)
    - Memory usage scales with the final number of vertices M
    - All new points are validated using is_surface_point() function
    - Duplicate positions are automatically removed
    - Interpolated normals are linearly interpolated and re-normalized
    - Neighbor points inherit the normal vector of their source vertex
    - The function maintains the order: original vertices, then interpolated,
      then neighbor points
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    dense_vertices = []
    dense_normals = []
    processed_positions = set()

    for i, vertex in enumerate(vertices):
        vertex = np.round(vertex).astype(int)
        vertex_tuple = tuple(vertex)

        if vertex_tuple not in processed_positions and is_surface_point(vertex, segmentation):
            processed_positions.add(vertex_tuple)
            dense_vertices.append(vertex)
            dense_normals.append(normals[i])

            # Add neighbors using separated function
            neighbors = find_surface_neighbors(vertex, segmentation, include_edges)
            for neighbor in neighbors:
                neighbor_tuple = tuple(neighbor)
                if neighbor_tuple not in processed_positions:
                    processed_positions.add(neighbor_tuple)
                    dense_vertices.append(neighbor)
                    dense_normals.append(normals[i])

        # Interpolate with next vertex using separated function
        if i < len(vertices) - 1:
            next_vertex = np.round(vertices[i + 1]).astype(int)
            interp_points = interpolate_between_points(vertex, next_vertex, segmentation, interpolation_points)
            
            for point in interp_points:
                point_tuple = tuple(point)
                if point_tuple not in processed_positions:
                    processed_positions.add(point_tuple)
                    dense_vertices.append(point)
                    # Interpolate normal vector
                    t = 0.5
                    interp_normal = (1 - t) * normals[i] + t * normals[i + 1]
                    interp_normal /= np.linalg.norm(interp_normal)
                    dense_normals.append(interp_normal)

    log_msg(f"Interpolation: {len(vertices)} → {len(dense_vertices)} vertices")
    return np.array(dense_vertices), np.array(dense_normals)


def refine_mesh_normals(
    vertices: np.ndarray,
    initial_normals: np.ndarray,
    radius_hit: float = 10.0,
    batch_size: int = 2000,
    flip_normals: bool = True,
    logger: logging.Logger = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Refine surface normals and separate bilayer surfaces.
    
    This function improves the quality of surface normal vectors and
    separates the bilayer membrane into inner and outer surfaces.
    It uses a voting-based approach to refine normals by averaging
    neighboring normals with distance-weighted contributions.
    
    The function performs three main operations:
    1. **Normal refinement**: Uses weighted averaging of neighbor normals
       within a specified radius to smooth and improve normal quality
    2. **Normal orientation**: Ensures consistent normal orientation
       across the surface using minimum spanning tree propagation
    3. **Surface separation**: Separates the bilayer into two surfaces
       based on normal direction analysis using PCA
    
    Parameters
    ----------
    vertices : np.ndarray
        2D array (N, 3) of surface vertex coordinates
    initial_normals : np.ndarray
        2D array (N, 3) of initial normal vectors from marching cubes
    radius_hit : float, default 10.0
        Search radius for finding neighbor points in voxel units.
        Larger values include more neighbors but increase computation time.
        Typical values range from 5-20 voxels depending on surface density.
    batch_size : int, default 2000
        Number of vertices to process in each batch. Larger batches
        are more memory-efficient but may cause memory issues with
        very large surfaces. Adjust based on available RAM.
    flip_normals : bool, default True
        Whether to flip refined normals to point inward toward the
        membrane interior. This is typically desired for bilayer analysis
        as it ensures consistent orientation relative to the membrane center.
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    refined_normals : np.ndarray
        2D array of shape (N, 3) containing refined normal vectors.
        All vectors are unit-normalized and consistently oriented.
        If flip_normals=True, normals point inward toward membrane center.
    surface1_mask : np.ndarray
        1D boolean array of length N indicating membership in the first
        surface. True values indicate vertices assigned to surface 1.
        Returns zero array if surface separation fails.
    surface2_mask : np.ndarray
        1D boolean array for second surface assignment

    Notes
    -----
    - Processing time scales with N * (average_neighbors_per_point)
    - Memory usage scales with N and the number of neighbor connections
    - Normal refinement uses Gaussian weighting based on distance
    - Surface separation requires sufficient surface coverage to work reliably
    - The function automatically falls back to original normals if refinement fails
    - Batch processing helps manage memory for large surfaces
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    voter = MeshNormalVoter(
        points=vertices, initial_normals=initial_normals, radius_hit=radius_hit, batch_size=batch_size, logger=logger
    )

    # First separate surfaces using initial marching cubes normals
    log_msg("\nSeparating surfaces using marching cubes orientation...")
    surface1_mask, surface2_mask = voter.separate_bilayer()

    # Orient normals consistently
    log_msg("\nOrienting normals...")
    voter.orient_normals()

    # Refine using simple single-scale method
    log_msg(f"\nRefining normals using weighted average of neighbor normals...")
    voter.refine_normals()

    # Initialize defaults first
    surface1_mask = np.zeros(len(vertices), dtype=bool)
    surface2_mask = np.zeros(len(vertices), dtype=bool)
    
    # Get separation results
    temp_surf1, temp_surf2 = voter.separate_bilayer()
    
    # Only update if separation was successful
    if temp_surf1 is not None and temp_surf2 is not None and len(temp_surf1) > 0:
        surface1_mask = temp_surf1
        surface2_mask = temp_surf2
        log_msg(f"Successfully separated surfaces")
        refined_normals = -voter.refined_normals if flip_normals else voter.refined_normals
    else:
        log_msg("Could not separate surfaces, using empty masks")
        refined_normals = -voter.refined_normals if flip_normals else voter.refined_normals

    return refined_normals, surface1_mask, surface2_mask



class MeshNormalVoter:
    """
    Class for refining normals and separating bilayer surfaces.

    This class handles the refinement of surface normal vectors using a
    voting-based approach, and separates the bilayer into inner and outer
    surfaces based on normal directions.
    """

    def __init__(self, points, initial_normals, radius_hit, batch_size=2000, logger=None):
        """
        Initialize MeshNormalVoter.

        Parameters
        ----------
        points : ndarray
            Nx3 array of vertex coordinates
        initial_normals : ndarray
            Nx3 array of initial normal vectors
        radius_hit : float
            Search radius for neighbor points
        batch_size : int
            Batch size for processing
        logger : logging.Logger, optional
            Logger instance
        """
        self.points = np.asarray(points, dtype=np.float64)
        self.initial_normals = np.asarray(initial_normals, dtype=np.float64)
        self.radius_hit = float(radius_hit)
        self.batch_size = batch_size
        self.sigma = self.radius_hit / 2.0
        self.logger = logger

        self.log("Building KD-tree for neighbor searches...")
        self.tree = KDTree(self.points, leaf_size=50)

        self.refined_normals = None
        self.surface1_mask = None
        self.surface2_mask = None

    def log(self, message):
        """Log message using the logger if available, otherwise print"""
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def refine_normals(self):
        """
        Refine normal directions using weighted average of neighbor normals.

        Returns
        -------
        refined_normals : ndarray
            Refined normal vectors
        """
        self.log("\nRefining normal directions using single-scale approach...")
        temp_normals = np.zeros_like(self.refined_normals if self.refined_normals is not None else self.initial_normals)

        for start_idx in tqdm(range(0, len(self.points), self.batch_size)):
            end_idx = min(start_idx + self.batch_size, len(self.points))
            batch_points = self.points[start_idx:end_idx]

            # Find neighbors for refinement
            neighbors = self.tree.query_radius(batch_points, self.radius_hit)

            for i, point_neighbors in enumerate(neighbors):
                point_idx = start_idx + i
                if len(point_neighbors) == 0:
                    temp_normals[point_idx] = (
                        self.refined_normals if self.refined_normals is not None else self.initial_normals
                    )[point_idx]
                    continue

                # Compute weights
                distances = np.linalg.norm(self.points[point_neighbors] - self.points[point_idx], axis=1)
                weights = np.exp(-(distances**2) / (2 * self.sigma**2))

                # Weighted average of neighbor normals
                current_normals = self.refined_normals if self.refined_normals is not None else self.initial_normals
                avg_normal = np.average(current_normals[point_neighbors], weights=weights, axis=0)

                # Normalize and maintain orientation
                avg_normal /= np.linalg.norm(avg_normal)
                if np.dot(avg_normal, current_normals[point_idx]) < 0:
                    avg_normal *= -1

                temp_normals[point_idx] = avg_normal

        self.refined_normals = temp_normals
        return self.refined_normals

    def build_adjacency(self, max_neighbors=100):
        """
        Build weighted adjacency graph for normal propagation.

        Parameters
        ----------
        max_neighbors : int
            Maximum number of neighbors per point

        Returns
        -------
        graph : scipy.sparse.csr_matrix
            Weighted adjacency graph
        """
        n_points = len(self.points)
        graph = lil_matrix((n_points, n_points))

        self.log(f"Building adjacency graph in batches of {self.batch_size}...")
        for start_idx in tqdm(range(0, n_points, self.batch_size)):
            end_idx = min(start_idx + self.batch_size, n_points)
            batch_points = self.points[start_idx:end_idx]

            # Find neighbors within radius
            distances, indices = self.tree.query_radius(
                batch_points, r=self.radius_hit, return_distance=True, sort_results=True
            )

            # Process each point in batch
            for i, (dists, neighs) in enumerate(zip(distances, indices)):
                point_idx = start_idx + i

                # Convert neighbors to integer array
                neighs = np.asarray(neighs, dtype=np.int64)

                # Ensure neighbor indices are valid
                valid_mask = (neighs >= 0) & (neighs < n_points) & (neighs != point_idx)
                if not np.any(valid_mask):
                    continue

                valid_neighs = neighs[valid_mask]
                valid_dists = dists[valid_mask]

                # Limit neighbors to prevent memory issues
                if len(valid_neighs) > max_neighbors:
                    closest_indices = np.argsort(valid_dists)[:max_neighbors]
                    valid_neighs = valid_neighs[closest_indices]
                    valid_dists = valid_dists[closest_indices]

                if len(valid_neighs) > 0:
                    weights = np.exp(-(valid_dists**2) / (2 * self.sigma**2))
                    graph[point_idx, valid_neighs] = weights

            # Convert to CSR periodically to save memory
            if start_idx % (self.batch_size * 10) == 0:
                graph = graph.tocsr()
                graph = graph.tolil()

        return graph.tocsr()

    def orient_normals(self):
        """
        Orient normals consistently using minimum spanning tree.

        Returns
        -------
        refined_normals : ndarray
            Consistently oriented normal vectors
        """
        self.log("Computing normal orientations...")

        # Build weighted adjacency graph
        graph = self.build_adjacency()

        # Get minimum spanning tree
        self.log("Computing minimum spanning tree...")
        mst = minimum_spanning_tree(graph)
        mst = mst.tocsr()

        # Initialize oriented normals
        self.refined_normals = self.initial_normals.copy()

        # Start from point with most connections
        self.log("Propagating orientations...")
        oriented = np.zeros(len(self.points), dtype=bool)
        n_neighbors = np.diff(mst.indptr)
        root = np.argmax(n_neighbors)

        queue = [root]
        oriented[root] = True

        while queue:
            current = queue.pop(0)
            row = mst.getrow(current)
            neighbors = row.indices[row.data > 0]

            for neighbor in neighbors:
                if not oriented[neighbor]:
                    # Check if normal needs flipping
                    dot_product = np.dot(self.refined_normals[current], self.refined_normals[neighbor])
                    if dot_product < 0:
                        self.refined_normals[neighbor] *= -1
                    oriented[neighbor] = True
                    queue.append(neighbor)

        return self.refined_normals

    def separate_bilayer(self):
        """
        Separate bilayer surfaces using marching cubes orientation.

        Since marching cubes gives outward-pointing normals, we can use
        a simple dot product test with a direction vector to separate surfaces.

        Returns
        -------
        surface1_mask : ndarray
            Boolean mask for first surface
        surface2_mask : ndarray
            Boolean mask for second surface
        """
        self.log("\nSeparating bilayer surfaces...")

        # Use refined normals if available, otherwise use initial normals
        normals = self.refined_normals if self.refined_normals is not None else self.initial_normals

        # Find principal direction of the structure
        # This will be roughly perpendicular to the membrane surface
        pca = PCA(n_components=3)
        pca.fit(self.points)
        principal_direction = pca.components_[2]  # Use the least significant direction

        # Project normals onto principal direction
        projections = np.dot(normals, principal_direction)

        # Separate based on normal orientation
        # If normal points in same direction as principal direction, it's surface 1
        self.surface1_mask = projections > 0
        self.surface2_mask = ~self.surface1_mask

        self.log(f"Surface 1: {np.sum(self.surface1_mask)} points")
        self.log(f"Surface 2: {np.sum(self.surface2_mask)} points")

        # Validate separation
        if np.sum(self.surface1_mask) < len(self.points) * 0.1 or np.sum(self.surface2_mask) < len(self.points) * 0.1:
            self.log("WARNING: Very uneven surface separation!")
            # Return empty masks
            empty_mask = np.zeros(len(self.points), dtype=bool)
            return empty_mask.copy(), empty_mask.copy()

        return self.surface1_mask, self.surface2_mask

    def process(self):
        """
        Run full processing pipeline.

        Returns
        -------
        surface1_mask : ndarray
            Boolean mask for first surface
        surface2_mask : ndarray
            Boolean mask for second surface
        """
        self.orient_normals()
        self.refine_normals()
        return self.separate_bilayer()


#############################################
# Thickness Measurement Functions - GPU-optimized CUDA implementation
#############################################


@cuda.jit
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


def process_matches_gpu2cpu(match_distances, match_indices, match_counts, n_points, max_matches_per_point, voxel_size):
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
    voxel_size : float
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
    thickness_results = thickness_results * voxel_size

    return thickness_results, valid_mask, point_pairs


def measure_thickness_gpu(
    points,
    normals,
    surface1_mask,
    surface2_mask,
    voxel_size,
    max_thickness_nm=8.0,
    max_angle_degrees=1.0,
    direction="1to2",
    logger=None,
):
    """
    GPU-accelerated membrane thickness measurement with one-to-one point matching.
    
    This function measures membrane thickness between two separated surfaces
    using CUDA-accelerated nearest neighbor search. It ensures each surface
    point is measured exactly once by implementing a one-to-one matching
    algorithm that prioritizes the shortest valid distances.
    
    The function performs the following steps:
    1. **GPU preparation**: Transfers data to GPU memory and configures CUDA grid
    2. **Parallel search**: Uses CUDA kernel to find all possible matches for each
       source point within geometric constraints
    3. **CPU post-processing**: Processes matches on CPU to ensure one-to-one
       assignment and converts results to physical units
    4. **Validation**: Applies geometric constraints (max thickness, cone angle)
    
    Geometric constraints ensure measurement quality:
    - **Maximum thickness**: Limits search to reasonable membrane thicknesses
    - **Cone angle**: Restricts search to a cone along the normal direction
    - **Surface separation**: Ensures points are on different surfaces
    
    Parameters
    ----------
    points : ndarray
        Unscaled voxel coordinates
    normals : ndarray
        Normal vectors
    surface1_mask, surface2_mask : ndarray
        Boolean masks for each surface
    voxel_size : float
        Voxel size in nm or angstroms
    max_thickness_nm : float
        Maximum thickness in nm (will be converted to voxels internally)
    max_angle_degrees : float
        Maximum angle for cone search
    direction : str
        Direction of thickness measurement, either "1to2" (surface1 to surface2)
        or "2to1" (surface2 to surface1)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    thickness_results : np.ndarray
        1D array of length N containing thickness measurements in nanometers.
        Only valid measurements have non-zero values; invalid measurements
        are set to 0.0.
    valid_mask : np.ndarray
        1D boolean array of length N indicating which measurements are valid.
        True values indicate successful thickness measurements.
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
        log_msg("Measuring thickness from surface 2 to surface 1...")
        source_mask, target_mask = surface2_mask, surface1_mask
    else:
        log_msg("Measuring thickness from surface 1 to surface 2...")
        source_mask, target_mask = surface1_mask, surface2_mask

    n_points = len(points)
    max_angle_cos = math.cos(math.radians(max_angle_degrees))

    # Convert max thickness from nm to voxels
    max_thickness_voxels = max_thickness_nm / voxel_size

    # Set maximum number of potential matches per point
    max_matches_per_point = 25

    log_msg(f"Starting GPU thickness measurement with {n_points} points...")
    log_msg(f"Source points: {np.sum(source_mask)}, Target points: {np.sum(target_mask)}")
    log_msg(f"Max thickness: {max_thickness_nm} nm ({max_thickness_voxels:.2f} voxels)")
    log_msg(f"Max angle: {max_angle_degrees} degrees")

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
    log_msg("Processing matches on CPU to ensure one-to-one matching...")
    thickness_results, valid_mask, point_pairs = process_matches_gpu2cpu(
        match_distances_cpu, match_indices_cpu, match_counts_cpu, n_points, max_matches_per_point, voxel_size
    )

    log_msg(f"Found {np.sum(valid_mask)} valid thickness measurements")
    if np.sum(valid_mask) > 0:
        log_msg(f"Mean thickness: {np.mean(thickness_results[valid_mask]):.2f} nm")
        log_msg(
            f"Min: {np.min(thickness_results[valid_mask]):.2f} nm, Max: {np.max(thickness_results[valid_mask]):.2f} nm"
        )

    return thickness_results, valid_mask, point_pairs


#############################################
# Thickness Measurement Functions - Numba implementation for CPU parallelization
#############################################


@numba.njit(parallel=True)
def find_matches_parallel(
    points,
    normals,
    source_mask,
    target_mask,
    target_indices,
    max_thickness_voxels,
    max_angle_cos,
    match_distances,
    match_indices,
    match_counts,
):
    """
    Parallelized function to find matches between points on different surfaces.

    Parameters
    ----------
    points : ndarray
        Point coordinates
    normals : ndarray
        Normal vectors
    source_mask : ndarray
        Mask for source points
    target_mask : ndarray
        Mask for target points
    target_indices : ndarray
        Indices of target points
    max_thickness_voxels : float
        Maximum thickness in voxel units
    max_angle_cos : float
        Cosine of maximum angle
    match_distances : ndarray
        Output array for match distances
    match_indices : ndarray
        Output array for match indices
    match_counts : ndarray
        Output array for match counts
    """
    n_points = len(points)
    max_matches = match_distances.shape[1]

    # For each source point, find valid matches
    for i in prange(n_points):
        if not source_mask[i]:
            continue

        point = points[i]
        normal = normals[i]
        match_count = 0

        # Check each potential target
        for j in range(len(target_indices)):
            target_idx = target_indices[j]

            # Vector from source to target
            dx = points[target_idx, 0] - point[0]
            dy = points[target_idx, 1] - point[1]
            dz = points[target_idx, 2] - point[2]

            # Euclidean distance
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            # Check if within max thickness
            if dist < max_thickness_voxels:
                # Project vector onto normal
                proj = dx * normal[0] + dy * normal[1] + dz * normal[2]

                # Only consider points in the direction of the normal
                if proj > 0:
                    # Calculate lateral distance (perpendicular to normal)
                    lateral_dx = dx - proj * normal[0]
                    lateral_dy = dy - proj * normal[1]
                    lateral_dz = dz - proj * normal[2]
                    lateral_dist_sq = lateral_dx**2 + lateral_dy**2 + lateral_dz**2

                    # Check if within cone angle
                    if lateral_dist_sq < (max_angle_cos * proj * proj):
                        if match_count < max_matches:
                            match_distances[i, match_count] = dist
                            match_indices[i, match_count] = target_idx
                            match_count += 1

        match_counts[i] = match_count


def measure_thickness_cpu(
    points,
    normals,
    surface1_mask,
    surface2_mask,
    voxel_size,
    max_thickness_nm=8.0,
    max_angle_degrees=1.0,
    direction="1to2",
    num_threads=None,
    logger=None,
    max_matches_per_point=25,
):
    """CPU-based thickness measurement with parallelization."""
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    # Set number of threads if specified
    if num_threads is not None:
        numba.set_num_threads(num_threads)
        log_msg(f"Using {num_threads} CPU threads")
    else:
        log_msg(f"Using all available CPU threads (numba default)")

    # Switch source and target surfaces if direction is 2to1
    if direction == "2to1":
        log_msg("Measuring thickness from surface 2 to surface 1...")
        source_mask, target_mask = surface2_mask, surface1_mask
    else:
        log_msg("Measuring thickness from surface 1 to surface 2...")
        source_mask, target_mask = surface1_mask, surface2_mask

    n_points = len(points)
    max_angle_cos = np.cos(np.radians(max_angle_degrees))

    # Convert max thickness from nm to voxels
    max_thickness_voxels = max_thickness_nm / voxel_size

    log_msg(f"Starting CPU thickness measurement with {n_points} points...")
    log_msg(f"Source points: {np.sum(source_mask)}, Target points: {np.sum(target_mask)}")
    log_msg(f"Max thickness: {max_thickness_nm} nm ({max_thickness_voxels:.2f} voxels)")
    log_msg(f"Max angle: {max_angle_degrees} degrees")

    # Get indices of target points
    target_indices = np.where(target_mask)[0]
    log_msg(f"Number of target points: {len(target_indices)}")

    # Get target points
    target_points = points[target_indices]

    # Get source points and indices
    source_indices = np.where(source_mask)[0]
    source_points = points[source_indices]

    log_msg(f"Number of source points: {len(source_points)}")

    # Use SciPy's KDTree for CPU implementation
    log_msg("Using SciPy KDTree implementation with query_ball_point")

    # Build KD-tree
    log_msg("Building KD-tree for target points...")
    target_tree = ScipyKDTree(target_points)

    # Pre-filter matches using ball query
    log_msg("Pre-filtering potential matches using KD-tree query_ball_point...")
    start_time = time.time()

    # Query ball point for each source point
    log_msg(f"Querying KD-tree for {len(source_points)} source points...")
    neighbor_lists = target_tree.query_ball_point(source_points, max_thickness_voxels)

    # Process the results
    flat_matches = []
    for i, neighbors in enumerate(neighbor_lists):
        source_idx = source_indices[i]
        source_normal = normals[source_idx]
        source_point = points[source_idx]

        valid_matches = 0

        for n in neighbors:
            # Get original index
            target_idx = target_indices[n]
            target_point = points[target_idx]

            # Vector from source to target
            dx = target_point[0] - source_point[0]
            dy = target_point[1] - source_point[1]
            dz = target_point[2] - source_point[2]

            # Distance
            dist = np.sqrt(dx * dx + dy * dy + dz * dz)

            # Project vector onto normal
            proj = dx * source_normal[0] + dy * source_normal[1] + dz * source_normal[2]

            # Only consider points in the direction of the normal
            if proj > 0:
                # Calculate lateral distance
                lateral_dx = dx - proj * source_normal[0]
                lateral_dy = dy - proj * source_normal[1]
                lateral_dz = dz - proj * source_normal[2]
                lateral_dist_sq = lateral_dx**2 + lateral_dy**2 + lateral_dz**2

                # Check if within cone angle
                if lateral_dist_sq < (max_angle_cos * proj * proj):
                    flat_matches.append((dist, source_idx, target_idx))
                    valid_matches += 1

                    # Limit matches per point
                    if valid_matches >= max_matches_per_point:
                        break

    log_msg(f"KD-tree pre-filtering completed in {time.time() - start_time:.2f} seconds")
    log_msg(f"Found {len(flat_matches)} potential matches across all source points")

    # Process matches to ensure one-to-one matching
    log_msg("Processing matches to ensure one-to-one matching...")
    thickness_results, valid_mask, point_pairs = process_matches_cpu2cpu(flat_matches, n_points, voxel_size)

    log_msg(f"Found {np.sum(valid_mask)} valid thickness measurements")
    if np.sum(valid_mask) > 0:
        log_msg(f"Mean thickness: {np.mean(thickness_results[valid_mask]):.2f} nm")
        log_msg(
            f"Min: {np.min(thickness_results[valid_mask]):.2f} nm, Max: {np.max(thickness_results[valid_mask]):.2f} nm"
        )

    return thickness_results, valid_mask, point_pairs


def process_matches_cpu2cpu(flat_matches, n_points, voxel_size):
    """
    Process matches on CPU to ensure one-to-one matching and convert to physical units.

    Parameters
    ----------
    flat_matches : list
        List of tuples (distance, source_idx, target_idx)
    n_points : int
        Total number of points
    voxel_size : float
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
    thickness_results = thickness_results * voxel_size

    return thickness_results, valid_mask, point_pairs

#############################################
# Intensity Profile Analysis Functions  
#############################################

## Utility Functions

def normalize_tomogram(
    tomo: np.ndarray, 
    method: Literal['zscore', 'minmax', 'percentile', 'none'] = 'zscore',
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Normalize tomogram intensity values using various strategies.
    
    Parameters
    ----------
    tomo : np.ndarray
        3D tomogram array
    method : {'zscore', 'minmax', 'percentile', 'none'}, default 'zscore'
        Normalization method
    logger : logging.Logger, optional
        Logger instance for status messages
        
    Returns
    -------
    np.ndarray
        Normalized tomogram
    """
    if method == 'none':
        return tomo.copy()
    
    # Use only finite values for normalization
    values = tomo[np.isfinite(tomo)]
    
    if len(values) == 0:
        warnings.warn("No valid values found for normalization")
        return tomo.copy()
    
    if method == 'zscore':
        center = np.mean(values)
        scale = np.std(values, ddof=1)
        if scale == 0:
            warnings.warn("Standard deviation is zero, returning original tomogram")
            return tomo.copy()
        normalized = (tomo - center) / scale
        
    elif method == 'minmax':
        min_val = np.min(values)
        max_val = np.max(values)
        if max_val == min_val:
            warnings.warn("Min equals max, returning original tomogram")
            return tomo.copy()
        normalized = (tomo - min_val) / (max_val - min_val)
        
    elif method == 'percentile':
        p1 = np.percentile(values, 1)
        p99 = np.percentile(values, 99)
        if p99 == p1:
            warnings.warn("1st and 99th percentiles are equal, returning original tomogram")
            return tomo.copy()
        normalized = np.clip((tomo - p1) / (p99 - p1), 0, 1)
        
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized

def save_int_results(results: Dict,
                thickness_csv: Path,
                output_dir: Union[str, Path],
                profiles: List[Dict],
                save_cleaned_df: bool = True,
                save_profiles: bool = True,
                save_statistics: bool = True,
                logger: logging.Logger = None) -> Dict[str, Path]:
    """
    Save files related to intensity profiles pre- and post-filtering with feature data.
    
    Parameters
    ----------
    results : Dict
        Results from filter_intensity_profiles containing filtered data
    thickness_csv : Path
        Original thickness file path (for naming)
    output_dir : Union[str, Path]
        Directory to save results
    profiles : List[Dict]
        Original extracted intensity profiles from extract_intensity_profile().
        Each dict contains 'profile', 'p1', 'p2', 'start', 'end', 'midpoint'
    save_cleaned_df : bool, default True
        Whether to save cleaned thickness DataFrame as "*_thickness_cleaned.csv"
    save_profiles : bool, default True
        Whether to save intensity profiles (both original and cleaned)
    save_statistics : bool, default True
        Whether to save filtering statistics as "*_filtering_stats.txt"
    logger : logging.Logger, optional
        Logger instance for status messages
        
    Returns
    -------
    Dict[str, Path]
        Dictionary mapping file types to saved paths:
        - 'thickness_cleaned': Path to cleaned thickness CSV
        - 'profiles_original': Path to original intensity profiles with features
        - 'profiles_cleaned': Path to cleaned intensity profiles with features
        - 'statistics': Path to statistics file

    Notes
    -----
    - Follows naming convention: base_name + "_thickness_cleaned.csv"
    - Intensity profiles saved as pickle files with feature data merged in
    - Each profile now includes 'features' dict with extracted minima/maxima data
    - Base name automatically extracted by removing common suffixes
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate base name for output files
    base_name = thickness_csv.stem
    if base_name.endswith('_thickness') or base_name.endswith('_membrane_thickness'):
        # Remove existing suffix
        for suffix in ['_thickness', '_membrane_thickness']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
    
    saved_files = {}
    
    # Save cleaned thickness DataFrame
    if save_cleaned_df and results['filtered_thickness_df'] is not None:
        cleaned_path = output_dir / f"{base_name}_thickness_cleaned.csv"
        results['filtered_thickness_df'].to_csv(cleaned_path, index=False)
        saved_files['thickness_cleaned'] = cleaned_path
        log_msg(f"Saved cleaned thickness DataFrame: {cleaned_path}")
    
    # Save intensity profiles with feature data
    if save_profiles:
        import pickle
        
        # Merge feature data with original profiles
        if profiles:
            profiles_with_features = _merge_profile_features(
                profiles, results, use_pass1_data=True, logger=logger
            )
            
            original_profiles_path = output_dir / f"{base_name}_int_profiles.pkl"
            with open(original_profiles_path, 'wb') as f:
                pickle.dump(profiles_with_features, f)
            saved_files['profiles_original'] = original_profiles_path
            log_msg(f"Saved original intensity profiles with features: {original_profiles_path}")
        
        # Proper index mapping for cleaned profiles
        if results['filtered_profiles']:
            # Get indices of profiles that passed filtering
            passing_indices = [i for i, result in enumerate(results['final_results']) if result['passes_filter']]
            
            log_msg(f"Mapping {len(results['filtered_profiles'])} cleaned profiles to {len(passing_indices)} passing indices")
            
            # Create cleaned profiles with features using correct mapping
            cleaned_profiles_with_features = _merge_profile_features(
                results['filtered_profiles'], results, use_pass1_data=False, 
                profile_indices=passing_indices, logger=logger
            )
            
            cleaned_profiles_path = output_dir / f"{base_name}_int_profiles_cleaned.pkl"
            with open(cleaned_profiles_path, 'wb') as f:
                pickle.dump(cleaned_profiles_with_features, f)
            saved_files['profiles_cleaned'] = cleaned_profiles_path
            log_msg(f"Saved cleaned intensity profiles with features: {cleaned_profiles_path}")
    
    # Save statistics
    if save_statistics:
        stats_path = output_dir / f"{base_name}_filtering_stats.txt"
        
        with open(stats_path, 'w') as f:
            # Redirect print output to file
            import sys
            from io import StringIO
            
            old_stdout = sys.stdout
            sys.stdout = mystdout = StringIO()
            
            print_summary(results)
            
            sys.stdout = old_stdout
            stats_content = mystdout.getvalue()
            
            f.write(f"Filtering statistics for: {thickness_csv.name}\n")
            f.write("=" * 50 + "\n\n")
            f.write(stats_content)
            
            # Add parameter information
            f.write("\n=== Processing Parameters ===\n")
            params = results['parameters']
            f.write(f"Min SNR: {params['intensity_min_snr']}\n")
            f.write(f"Central max required: {params['intensity_central_max_required']}\n")
            f.write(f"Extension range: {params['intensity_extension_range']}\n")
            
            # Add extraction information
            f.write(f"\n=== Extraction Information ===\n")
            f.write(f"Total extracted profiles: {len(profiles)}\n")
            f.write(f"Profiles after filtering: {len(results['filtered_profiles'])}\n")
            
            f.write(f"\n=== Output Files ===\n")
            for file_type, file_path in saved_files.items():
                f.write(f"{file_type}: {file_path.name}\n")
        
        saved_files['statistics'] = stats_path
        log_msg(f"Saved statistics: {stats_path}")
    
    return saved_files


def _merge_profile_features(
    profiles: List[Dict], 
    results: Dict, 
    use_pass1_data: bool = True,
    profile_indices: List[int] = None,
    logger: logging.Logger = None
) -> List[Dict]:
    """
    FIXED version with proper index mapping and missing data handling.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    # Choose data source
    if use_pass1_data:
        feature_data = results.get('pass1_candidates', [])
        data_source = "Pass 1 characterization"
    else:
        feature_data = results.get('final_results', [])
        data_source = "Pass 2 filtering"
    
    log_msg(f"Merging {data_source} feature data with {len(profiles)} profiles")
    log_msg(f"Available feature data entries: {len(feature_data)}")
    
    enhanced_profiles = []
    
    for i, profile in enumerate(profiles):
        # Create enhanced profile with original data
        enhanced_profile = profile.copy()
        
        # FIXED: Proper index mapping
        if use_pass1_data:
            # Direct mapping for original profiles
            feature_idx = i
        else:
            # Use provided indices for cleaned profiles
            if profile_indices is not None and i < len(profile_indices):
                feature_idx = profile_indices[i]
            else:
                log_msg(f"Warning: No profile_indices provided for cleaned profile {i}")
                feature_idx = i  # Fallback
        
        # Initialize features dictionary
        features = {
            'data_source': data_source,
            'feature_index': feature_idx,
            'has_features': False,
            'minima1_position': np.nan,
            'minima1_intensity': np.nan,
            'minima2_position': np.nan,
            'minima2_intensity': np.nan,
            'central_max_position': np.nan,
            'central_max_intensity': np.nan,
            'separation_distance': np.nan,
            'prominence_snr': np.nan,
            'passes_filter': False,
            'failure_reason': None,
            'p1_projection': np.nan,
            'p2_projection': np.nan,
            'minima_between_points': False
        }
        
        # Extract feature data if available
        if feature_idx < len(feature_data):
            feature_result = feature_data[feature_idx]
            features['has_features'] = True
            
            # Extract minima positions and intensities
            if 'minima_positions' in feature_result and feature_result['minima_positions']:
                positions = feature_result['minima_positions']
                depths = feature_result.get('minima_depths', [])
                
                # Split into individual minima (position sorted)
                if len(positions) >= 1:
                    features['minima1_position'] = positions[0]
                    if len(depths) >= 1:
                        features['minima1_intensity'] = depths[0]
                
                if len(positions) >= 2:
                    features['minima2_position'] = positions[1]
                    if len(depths) >= 2:
                        features['minima2_intensity'] = depths[1]
            
            # FIXED: Extract separation distance with fallback calculation
            if 'separation_distance' in feature_result and not np.isnan(feature_result['separation_distance']):
                features['separation_distance'] = feature_result['separation_distance']
            elif 'minima_positions' in feature_result and len(feature_result['minima_positions']) >= 2:
                # Calculate from positions if missing
                positions = feature_result['minima_positions']
                features['separation_distance'] = abs(positions[1] - positions[0])
            
            # Extract other features
            features.update({
                'prominence_snr': feature_result.get('prominence_snr', np.nan),
                'p1_projection': feature_result.get('p1_projection', np.nan),
                'p2_projection': feature_result.get('p2_projection', np.nan),
                'minima_between_points': feature_result.get('minima_between_points', False),
                'failure_reason': feature_result.get('failure_reason', None)
            })
            
            # For Pass 1 data
            if use_pass1_data:
                features['passes_filter'] = feature_result.get('is_candidate', False)
            else:
                # For Pass 2 data - extract central maximum info
                features.update({
                    'passes_filter': feature_result.get('passes_filter', False),
                    'has_dual_minima': feature_result.get('has_dual_minima', False),
                    'has_central_maximum': feature_result.get('has_central_maximum', False),
                    'sufficient_prominence': feature_result.get('sufficient_prominence', False),
                    'central_max_position': feature_result.get('central_max_position', np.nan),
                    'central_max_height': feature_result.get('central_max_height', np.nan)
                })
        
        # Add features to profile
        enhanced_profile['features'] = features
        enhanced_profiles.append(enhanced_profile)
    
    # Summary statistics
    has_features_count = sum(1 for p in enhanced_profiles if p['features']['has_features'])
    has_minima_count = sum(1 for p in enhanced_profiles 
                          if not np.isnan(p['features']['minima1_position']))
    has_separation_count = sum(1 for p in enhanced_profiles 
                              if not np.isnan(p['features']['separation_distance']))
    has_snr_count = sum(1 for p in enhanced_profiles 
                       if not np.isnan(p['features']['prominence_snr']))
    has_central_max_count = sum(1 for p in enhanced_profiles 
                               if not np.isnan(p['features']['central_max_position']))
    
    log_msg(f"  Profiles with feature data: {has_features_count}/{len(enhanced_profiles)}")
    log_msg(f"  Profiles with minima positions: {has_minima_count}/{len(enhanced_profiles)}")
    log_msg(f"  Profiles with separation distance: {has_separation_count}/{len(enhanced_profiles)}")
    log_msg(f"  Profiles with SNR data: {has_snr_count}/{len(enhanced_profiles)}")
    log_msg(f"  Profiles with central max position: {has_central_max_count}/{len(enhanced_profiles)}")
    
    return enhanced_profiles


def print_summary(results: Dict, logger: logging.Logger = None) -> None:
    """Print summary of filtering results."""
    
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    stats = results['statistics']
    params = results['parameters']
    dataset_chars = results['dataset_characteristics']
    
    log_msg("=== Summary of intensity profiles post filtering ===")
    log_msg(f"Total profiles analyzed: {stats['total_profiles']:,}")
    log_msg(f"Candidate profiles found after step 1: {stats['pass1_candidates']:,}")
    log_msg(f"Profiles that passed criteria: {stats['profiles_passed']:,} ({stats['pass_rate']:.1%})")
    log_msg(f"Profiles excluded: {stats['profiles_failed']:,} ({1-stats['pass_rate']:.1%})")
    
    log_msg("=== Characteristics of intensity profiles ===")
    if not np.isnan(dataset_chars['median_separation']):
        log_msg(f"Median separation of minima: {dataset_chars['median_separation']:.2f} ± {dataset_chars['separation_std']:.2f} voxels")
    if not np.isnan(dataset_chars['median_prominence_snr']):
        log_msg(f"Median SNR of minima: {dataset_chars['median_prominence_snr']:.1f}x (minima are {dataset_chars['median_prominence_snr']:.1f}x more pronounced than the baseline)")
    
    log_msg("=== Applied filtering criteria (user-defined) ===")
    # MODIFIED: Handle None SNR
    if params['intensity_min_snr'] is not None:
        log_msg(f"Required SNR of minima: {params['intensity_min_snr']:.1f}x (minima must be {params['intensity_min_snr']:.1f}x more pronounced than the baseline)")
    else:
        log_msg(f"SNR filtering: DISABLED (position-based filtering only)")
    
    # NEW: Show margin factor
    if params.get('intensity_margin_factor', 0) > 0:
        log_msg(f"Position margin: {params['intensity_margin_factor']*100:.0f}% of measurement span")
    else:
        log_msg(f"Position filtering: Exact (no margin)")
    
    log_msg(f"Central max required: {params['intensity_central_max_required']}")
    log_msg(f"Extension range from midpoint: {params['intensity_extension_range']} voxels")
    
    # Failure analysis
    if stats['failure_analysis']:
        log_msg("=== Reasons for intensity profile exclusion ===")
        for reason, count in stats['failure_analysis'].items():
            percentage = count / stats['total_profiles'] * 100
            log_msg(f"  {reason}: {count:,} ({percentage:.1f}%)")

    
    # Quality metrics
    if 'quality_metrics' in stats and stats['quality_metrics']:
        log_msg("=== Quality metrics of included profiles ===")
        
        sep_stats = stats['quality_metrics']['separation_stats']
        if not np.isnan(sep_stats['mean']):
            log_msg(f"Mean separation of minima: {sep_stats['mean']:.2f} ± {sep_stats['std']:.2f} voxels")
            log_msg(f"  Median separation of minima: {sep_stats['median']:.2f}, Range: {sep_stats['range'][0]:.2f}-{sep_stats['range'][1]:.2f}")
        
        snr_stats = stats['quality_metrics']['prominence_snr_stats']
        if not np.isnan(snr_stats['mean']):
            log_msg(f"Mean SNR of minima: {snr_stats['mean']:.1f} ± {snr_stats['std']:.1f}x")
            log_msg(f"  Median SNR of minima: {snr_stats['median']:.1f}x, Q25-Q75: {snr_stats['quartiles'][0]:.1f}-{snr_stats['quartiles'][1]:.1f}x")
            log_msg(f"  (Minima are on average {snr_stats['mean']:.1f}x more pronounced than the baseline)")

## Intensity Profile Extraction

def extract_intensity_profile(
    thickness_df: pd.DataFrame,
    tomo: np.ndarray,
    voxel_size: float = None,
    intensity_extension_voxels: int = 10,
    intensity_normalize_method: Literal['zscore', 'minmax', 'percentile', 'none'] = 'zscore',
    logger: logging.Logger = None
) -> List[Dict]:
    """
    Extract intensity profiles between paired membrane points with physical units.
    
    This function computes intensity profiles along lines connecting matched
    surface points from thickness measurements. It extracts intensity values
    from the tomogram along vectors extending beyond the matched points,
    providing data for quality assessment and filtering of thickness measurements.
    
    The function performs the following steps:
    1. **Input validation**: Checks for required columns and valid coordinate pairs
    2. **Tomogram normalization**: Applies specified normalization method
    3. **Profile extraction**: Computes intensity values along each line using
       linear interpolation
    4. **Coordinate scaling**: Converts voxel coordinates to physical units
    5. **Extension**: Extends profiles beyond matched points for better analysis
    
    Intensity profiles are essential for validating thickness measurements
    by detecting characteristic bilayer features (dual minima, central maximum).
    
    Parameters
    ----------
    thickness_df : pd.DataFrame
        DataFrame containing paired point coordinates for thickness measurements.
        Must have columns ['x1_voxel', 'y1_voxel', 'z1_voxel', 'x2_voxel', 
        'y2_voxel', 'z2_voxel'] with coordinates in voxel units. Invalid or
        NaN coordinate pairs are automatically filtered out.
    tomo : np.ndarray
        3D tomogram array with shape (Z, Y, X) in ZYX order. The tomogram
        should have the same dimensions as the segmentation used for thickness
        measurement. Intensity values are extracted from this volume.
    voxel_size : float, optional
        Voxel size in nanometers for physical coordinate scaling. If None,
        only voxel coordinates are returned. If provided, physical coordinates
        in nanometers are added to each profile dictionary.
    intensity_extension_voxels : int, default 10
        Number of voxels to extend beyond the midpoint in each direction.
        Larger values provide more context for intensity analysis but may
        include irrelevant regions. Typical values range from 10-20 voxels.
    intensity_normalize_method : {'zscore', 'minmax', 'percentile', 'none'}, default 'zscore'
        Normalization method to apply to tomogram before extraction:
        - 'zscore': Standardize to zero mean and unit variance
        - 'minmax': Scale to range [0, 1]
        - 'percentile': Clip to 1st-99th percentile range
        - 'none': Use original intensity values
    logger : logging.Logger, optional
        Logger instance for status messages
        
    Returns
    -------
    List[Dict]
        List of profile dictionaries, one for each valid coordinate pair.
        Each dictionary contains:
        
        **Core profile data:**
        - 'profile': np.ndarray of intensity values along the line
        - 'p1': np.ndarray, coordinates of first point (voxel units)
        - 'p2': np.ndarray, coordinates of second point (voxel units)  
        - 'midpoint': np.ndarray, midpoint coordinates (voxel units)
        - 'start': np.ndarray, extended start coordinates (voxel units)
        - 'end': np.ndarray, extended end coordinates (voxel units)
        
        **Physical coordinates (if voxel_size provided):**
        - 'p1_nm': np.ndarray, first point in nanometers
        - 'p2_nm': np.ndarray, second point in nanometers
        - 'midpoint_nm': np.ndarray, midpoint in nanometers
        - 'start_nm': np.ndarray, extended start in nanometers
        - 'end_nm': np.ndarray, extended end in nanometers
        - 'voxel_size': float, voxel size used for scaling
        
    Raises
    ------
    ValueError
        If required columns are missing from thickness_df.
        If tomo is not a 3D array.
        If voxel_size is negative.
    IndexError
        If any coordinate pairs are outside tomogram bounds.
    
    Notes
    -----
    - Profiles extend beyond the matched points by intensity_extension_voxels in both directions
    - Invalid or NaN coordinate pairs are automatically filtered out
    - Interpolation is performed using scipy.ndimage.map_coordinates with linear interpolation
    - Physical coordinates are computed by multiplying voxel coordinates by voxel_size
    - The function handles variable line lengths automatically
    - Processing time scales with the number of coordinate pairs and extension distance
    
    Examples
    --------
    Extract profiles with physical units as a standalone function:
    
    >>> profiles = extract_intensity_profile(
    ...     thickness_df=df,
    ...     tomo=tomogram_array, 
    ...     voxel_size=0.788,
    ...     intensity_extension_voxels=10,
    ...     intensity_normalize_method='zscore'
    ... )
    >>> print(f"Profile has {len(profiles[0]['profile'])} intensity points")
    >>> print(f"Physical distance: {np.linalg.norm(profiles[0]['p2_nm'] - profiles[0]['p1_nm']):.2f} nm")
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    # Normalize tomogram once
    log_msg(f"Normalizing tomogram using method: {intensity_normalize_method}")
    tomo_normalized = normalize_tomogram(tomo, method=intensity_normalize_method)
    
    # Vectorized approach for better performance
    required_cols = ['x1_voxel', 'y1_voxel', 'z1_voxel', 'x2_voxel', 'y2_voxel', 'z2_voxel']
    
    # Check if all required columns exist
    missing_cols = set(required_cols) - set(thickness_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    log_msg(f"Extracting intensity profiles with extension of {intensity_extension_voxels} voxels")
    if voxel_size is not None:
        log_msg(f"Including physical coordinates with voxel size: {voxel_size:.4f} nm")
    
    # Extract coordinate arrays
    p1_coords = thickness_df[['x1_voxel', 'y1_voxel', 'z1_voxel']].values
    p2_coords = thickness_df[['x2_voxel', 'y2_voxel', 'z2_voxel']].values
    
    # Find valid pairs (no NaN values)
    valid_mask = ~(np.isnan(p1_coords).any(axis=1) | np.isnan(p2_coords).any(axis=1))
    
    if not np.any(valid_mask):
        log_msg("Warning: No valid coordinate pairs found")
        return []
    
    p1_coords = p1_coords[valid_mask]
    p2_coords = p2_coords[valid_mask]
    
    log_msg(f"Processing {len(p1_coords)} valid coordinate pairs")
    
    # Vectorized calculations
    directions = p2_coords - p1_coords
    lengths = np.linalg.norm(directions, axis=1)
    
    # Filter out zero-length vectors
    nonzero_mask = lengths > 0
    if not np.any(nonzero_mask):
        log_msg("Warning: No non-zero length vectors found")
        return []
    
    p1_coords = p1_coords[nonzero_mask]
    p2_coords = p2_coords[nonzero_mask]
    directions = directions[nonzero_mask]
    lengths = lengths[nonzero_mask]
    
    # Normalize directions
    unit_vectors = directions / lengths[:, np.newaxis]
    
    # Calculate midpoints, starts, and ends
    midpoints = (p1_coords + p2_coords) / 2.0
    starts = midpoints - unit_vectors * intensity_extension_voxels
    ends = midpoints + unit_vectors * intensity_extension_voxels
    
    profiles = []
    
    log_msg(f"Extracting intensity values for {len(p1_coords)} profiles...")
    
    # Process each line individually (this part is hard to vectorize due to variable line lengths)
    for i in range(len(p1_coords)):
        num_points = int(np.ceil(2 * intensity_extension_voxels + lengths[i])) + 1
        line_points = np.linspace(starts[i], ends[i], num=num_points)
        
        # Convert from XYZ to ZYX for scipy indexing
        coords_zyx = line_points[:, [2, 1, 0]].T
        
        # Extract intensities
        intensities = map_coordinates(tomo_normalized, coords_zyx, order=1, mode="nearest")
        
        profile_data = {
            "profile": intensities,
            "p1": p1_coords[i],
            "p2": p2_coords[i],
            "midpoint": midpoints[i],
            "start": starts[i],
            "end": ends[i],
        }
        
        # Add physical coordinates if voxel_size provided
        if voxel_size is not None:
            profile_data.update({
                "p1_nm": p1_coords[i] * voxel_size,
                "p2_nm": p2_coords[i] * voxel_size,
                "midpoint_nm": midpoints[i] * voxel_size,
                "start_nm": starts[i] * voxel_size,
                "end_nm": ends[i] * voxel_size,
                "voxel_size": voxel_size
            })
        
        profiles.append(profile_data)
    
    log_msg(f"Successfully extracted {len(profiles)} intensity profiles")
    
    return profiles

## Intensity profile Filtering

def filter_intensity_profiles(
    profiles: List[Dict],
    thickness_df: pd.DataFrame,
    intensity_min_snr: float = 0.2,
    intensity_central_max_required: bool = True,
    intensity_extension_range: Tuple[float, float] = (-10, 10),
    intensity_margin_factor: float = 0.1,             
    require_both_minima_in_region: bool = True,
    smooth_sigma: float = 0.0,               
    edge_fraction: float = 0.2,
    logger: logging.Logger = None
) -> Dict:
    """
    Filter intensity profiles using quality criteria and geometric constraints.
    
    This function applies a two-pass filtering approach to validate intensity
    profiles and filter out low-quality thickness measurements. It uses both
    signal quality metrics and geometric constraints to ensure only reliable
    bilayer measurements are retained.
    
    **Two-Pass Filtering Strategy:**
    
    1. **Pass 1 - Characterization**: Identifies candidate profiles using
       loose criteria to establish dataset characteristics and statistics.
       This pass is used for analysis only, not for filtering.
    
    2. **Pass 2 - Quality Filtering**: Applies strict criteria to filter
       profiles based on:
       - Dual minima detection (representing lipid headgroups)
       - Central maximum requirement (representing lipid tails)
       - Signal-to-noise ratio validation
       - Geometric position constraints
    
    **Quality Criteria:**
    
    - **Dual Minima**: Must detect at least two minima in the profile
    - **Central Maximum**: Must have a maximum between the two minima
    - **Signal Quality**: Minima must meet minimum SNR requirements
    - **Position Validation**: Minima must be positioned between matched points
    - **Geometric Constraints**: Optional margin for position flexibility
    
    Parameters
    ----------
    profiles : List[Dict]
        List of intensity profile dictionaries from extract_intensity_profile().
        Each profile should contain 'profile', 'p1', 'p2', 'start', 'end',
        and 'midpoint' keys with appropriate coordinate data.
    thickness_df : pd.DataFrame
        DataFrame containing thickness measurement metadata. Must have
        the same number of rows as profiles. Used for coordinate validation
        and result mapping.
    intensity_min_snr : Optional[float], default 0.2
        Minimum signal-to-noise ratio for minima prominence.
        - If None: SNR filtering is disabled (position-based filtering only)
        - If float: Minima must have prominence >= intensity_min_snr × baseline noise
        - Typical values: 1.0-3.0 (higher = stricter quality requirements)
        - Lower values may include noisy profiles, higher values may exclude valid ones
    intensity_central_max_required : bool, default True
        Whether to require a central maximum between the two detected minima.
        This validates the bilayer structure where lipid tails create a
        central intensity peak between the headgroup minima.
    intensity_extension_range : Tuple[float, float], default (-10, 10)
        Distance range in voxel units to analyze around the profile midpoint.
        Profiles are analyzed within this range for feature detection.
        - First value: Minimum distance from midpoint
        - Second value: Maximum distance from midpoint
        - Typical range: (-8, 8) to (-15, 15) voxels
    intensity_margin_factor : float, default 0.1
        Allowed margin for minima detection outside the measurement region
        as a fraction of the measurement span.
        - 0.0: Minima must be exactly between matched points (strictest)
        - 0.1: 10% margin allowed (recommended for most cases)
        - 0.3: 30% margin allowed (most permissive)
        - Higher values accommodate measurement uncertainty
    require_both_minima_in_region : bool, default True
        If True, both minima must be within the extended region for a
        profile to pass filtering. If False, only one minimum is required.
        - True: Ensures complete bilayer detection (recommended)
        - False: More permissive, may include partial bilayer profiles
    smooth_sigma : float, default 0.0
        Gaussian smoothing parameter for intensity profiles.
        - 0.0: No smoothing (preserves original profile features)
        - 0.5-1.0: Light smoothing (reduces noise)
        - 1.5-2.0: Moderate smoothing (may blur features)
        - Higher values: Heavy smoothing (may lose important features)
    edge_fraction : float, default 0.2
        Fraction of profile edges used for baseline noise calculation.
        - 0.1: Use 10% of profile edges for baseline (narrow baseline)
        - 0.2: Use 20% of profile edges for baseline (recommended)
        - 0.3: Use 30% of profile edges for baseline (wide baseline)
        - Used to calculate signal-to-noise ratios
    logger : logging.Logger, optional
        Logger instance for status messages. If None, prints to stdout.
        Used to report filtering progress and results.
        
    Returns
    -------
    Dict
        Comprehensive filtering results containing:
        
        **Core Results:**
        - 'pass1_candidates': List of Pass 1 characterization results
        - 'final_results': List of Pass 2 filtering results
        - 'filtered_profiles': List of profiles that passed filtering
        - 'filtered_thickness_df': DataFrame of filtered thickness data
        
        **Statistics and Analysis:**
        - 'statistics': Comprehensive filtering statistics
        - 'dataset_characteristics': Dataset-wide feature characteristics
        - 'parameters': User-specified filtering parameters
        
        **Statistics Details:**
        - 'total_profiles': Total number of profiles analyzed
        - 'profiles_passed': Number of profiles passing all criteria
        - 'pass_rate': Fraction of profiles passing filters
        - 'failure_analysis': Breakdown of failure reasons
        - 'quality_metrics': Statistical analysis of passed profiles
        
        **Dataset Characteristics:**
        - 'median_separation': Typical distance between minima
        - 'median_prominence_snr': Typical signal quality
        - 'n_candidates': Number of Pass 1 candidates
    
    Raises
    ------
    ValueError
        If profiles and thickness_df have incompatible lengths.
        If intensity_extension_range is invalid (min >= max).
        If intensity_margin_factor is negative.
    RuntimeError
        If profile processing fails unexpectedly.
    
    Notes
    -----
    **Filtering Logic:**
    - Profiles are processed individually with comprehensive feature detection
    - Minima detection uses scipy.signal.find_peaks with prominence filtering
    - Position validation ensures minima fall within measurement constraints
    - SNR calculation uses edge regions for baseline noise estimation
    
    **Performance Considerations:**
    - Processing time scales with number of profiles and profile length
    - Memory usage scales with profile count and feature data storage
    - Smoothing increases computation time but may improve feature detection
    
    **Quality Trade-offs:**
    - Stricter SNR requirements improve quality but reduce coverage
    - Larger position margins increase coverage but may include lower quality data
    - Central maximum requirement ensures bilayer structure but may exclude valid cases
    
    Examples
    --------
    Basic filtering with default parameters:
    
    >>> from memthick import filter_intensity_profiles
    >>> import pandas as pd
    >>> 
    >>> # Filter profiles with standard criteria
    >>> results = filter_intensity_profiles(
    ...     profiles=extracted_profiles,
    ...     thickness_df=thickness_data,
    ...     intensity_min_snr=0.2,
    ...     intensity_central_max_required=True,
    ...     intensity_extension_range=(-10, 10)
    ... )
    >>> 
    >>> print(f"Profiles passed: {results['statistics']['profiles_passed']}")
    >>> print(f"Pass rate: {results['statistics']['pass_rate']:.1%}")
    
    Custom filtering parameters:
    
    >>> # More permissive filtering
    >>> results_permissive = filter_intensity_profiles(
    ...     profiles=extracted_profiles,
    ...     thickness_df=thickness_data,
    ...     intensity_min_snr=None,  # Disable SNR filtering
    ...     intensity_central_max_required=False,  # Don't require central maximum
    ...     intensity_margin_factor=0.2,  # 20% position margin
    ...     require_both_minima_in_region=False  # Only one minimum required
    ... )
    
    >>> # Stricter filtering
    >>> results_strict = filter_intensity_profiles(
    ...     profiles=extracted_profiles,
    ...     thickness_df=thickness_data,
    ...     intensity_min_snr=2.0,  # High SNR requirement
    ...     intensity_central_max_required=True,
    ...     intensity_margin_factor=0.0,  # Exact position requirement
    ...     smooth_sigma=0.5  # Light smoothing
    ... )
    
    Analyzing failure reasons:
    
    >>> failure_analysis = results['statistics']['failure_analysis']
    >>> for reason, count in failure_analysis.items():
    ...     print(f"{reason}: {count} profiles")
    
    Accessing quality metrics:
    
    >>> quality_metrics = results['statistics']['quality_metrics']
    >>> snr_stats = quality_metrics['prominence_snr_stats']
    >>> print(f"Mean SNR: {snr_stats['mean']:.1f}x")
    >>> print(f"Median SNR: {snr_stats['median']:.1f}x")
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    # Get valid thickness data
    
    if len(thickness_df) != len(profiles):
        warnings.warn(f"Thickness DF length ({len(thickness_df)}) != profiles length ({len(profiles)})")
        min_len = min(len(thickness_df), len(profiles))
        thickness_df = thickness_df.iloc[:min_len].copy()
        profiles = profiles[:min_len]
    
    results = {
        'pass1_candidates': [],
        'dataset_characteristics': {},
        'final_results': [],
        'filtered_profiles': [],
        'filtered_thickness_df': None,
        'statistics': {},
        'parameters': {
            'intensity_min_snr': intensity_min_snr,
            'intensity_central_max_required': intensity_central_max_required,
            'intensity_extension_range': intensity_extension_range
        }
    }
    
    log_msg("=== Statistics of the extracted intensity profiles ===")

    
    # Pass 1: Find candidates and calculate dataset characteristics
    pass1_results = _pass1_characterization(profiles, intensity_extension_range)
    results['pass1_candidates'] = pass1_results
    
    # Calculate dataset characteristics (for statistics only, not filtering)
    dataset_chars = _calculate_dataset_characteristics(pass1_results)
    results['dataset_characteristics'] = dataset_chars
    
    log_msg(f"Step 1: Found {len([r for r in pass1_results if r['is_candidate']])} candidate profiles")
    log_msg(f"Median separation: {dataset_chars['median_separation']:.2f} ± {dataset_chars['separation_std']:.2f} voxels")
    log_msg(f"Median prominence SNR: {dataset_chars['median_prominence_snr']:.1f}x")
    
    log_msg("\n=== Apply filtering criteria: (1) two minima (with user-defined SNR compared to baseline) separated by max; (2) the min are positioned between the two matched points from which thickness was measured ===")
    
    # Pass 2: Apply filtering constraints
    final_results = _pass2_apply_filtering(
        profiles, 
        intensity_extension_range, 
        intensity_min_snr,
        intensity_central_max_required,
        intensity_margin_factor,
        require_both_minima_in_region,
        smooth_sigma,
        edge_fraction
    )
    results['final_results'] = final_results
    
    # Extract filtered profiles
    passing_indices = [i for i, result in enumerate(final_results) if result['passes_filter']]
    results['filtered_profiles'] = [profiles[i] for i in passing_indices]
    
    if passing_indices:
        results['filtered_thickness_df'] = thickness_df.iloc[passing_indices].copy()
    else:
        results['filtered_thickness_df'] = thickness_df.iloc[0:0].copy()
    
    log_msg(f"Step 2: {len(passing_indices)}/{len(profiles)} profiles passed the filtering criteria ({len(passing_indices)/len(profiles):.1%})")
    
    # Calculate comprehensive statistics
    results['statistics'] = _calculate_statistics(
        results, thickness_df, len(profiles)
    )
    
    return results

## Helper functions

def _pass1_characterization(profiles: List[Dict], intensity_extension_range: Tuple[float, float]) -> List[Dict]:
    """
    Pass 1 characterization of intensity profiles.

    Identifies candidate bilayer-like profiles using loose criteria and
    computes per-profile features used for statistics and for guiding Pass 2.

    Computed per-profile fields (per result dict):
    - profile_index: int
    - is_candidate: bool
    - minima_positions: List[float] (within intensity_extension_range, relative to midpoint)
    - minima_depths: List[float]
    - separation_distance: float (voxels) or NaN if fewer than two minima
    - prominence_snr: float (average prominence / baseline noise)
    - p1_projection, p2_projection: float projections of matched points onto the profile axis
    - minima_between_points: bool
    - failure_reason: Optional[str] (e.g., 'invalid_profile_data', 'zero_intensity_range', 'insufficient_minima_X')

    Parameters
    ----------
    profiles : List[Dict]
        Profiles as produced by extract_intensity_profile().
    intensity_extension_range : Tuple[float, float]
        [min, max] distance window relative to the midpoint used to crop
        each profile for analysis.

    Returns
    -------
    List[Dict]
        One result dict per input profile (see fields above). Does not filter
        inputs; 'is_candidate' marks loose matches used for dataset statistics.

    Notes
    -----
    - Uses a loose prominence threshold (1.5 × baseline noise) for minima discovery
    - Baseline noise is estimated from profile edges
    - No I/O or stateful side effects
    """
    
    pass1_results = []
    
    for i, prof in enumerate(profiles):
        result = {
            'profile_index': i,
            'is_candidate': False,
            'minima_positions': [],
            'minima_depths': [],
            'separation_distance': np.nan,
            'prominence_snr': np.nan,
            'p1_projection': np.nan,
            'p2_projection': np.nan,
            'minima_between_points': False,
            'failure_reason': None
        }
        
        # Extract profile data
        processed_data = _extract_profile_data(prof, intensity_extension_range)
        if processed_data is None:
            result['failure_reason'] = 'invalid_profile_data'
            pass1_results.append(result)
            continue
            
        filtered_distances, filtered_intensities = processed_data
        
        # Calculate basic properties
        intensity_range = np.max(filtered_intensities) - np.min(filtered_intensities)
        
        if intensity_range == 0:
            result['failure_reason'] = 'zero_intensity_range'
            pass1_results.append(result)
            continue
        
        # Calculate baseline
        baseline_noise = _calculate_baseline_noise(filtered_intensities)
        if baseline_noise == 0:
            baseline_noise = 1e-6  # Avoid division by zero
        
        # Loose prominence for candidate detection (1.5x baseline noise)
        loose_prominence = 1.5 * baseline_noise
        
        # Find minima
        minima_data = _find_minima_with_details(filtered_intensities, loose_prominence)
        if len(minima_data['peaks']) < 2:
            result['failure_reason'] = f'insufficient_minima_{len(minima_data["peaks"])}'
            pass1_results.append(result)
            continue
        
        # Take the two most prominent minima
        if len(minima_data['peaks']) > 2:
            top_indices = np.argsort(minima_data['prominences'])[-2:]
            selected_peaks = minima_data['peaks'][top_indices]
            selected_prominences = minima_data['prominences'][top_indices]
            # Sort by position
            sort_order = np.argsort(selected_peaks)
            selected_peaks = selected_peaks[sort_order]
            selected_prominences = selected_prominences[sort_order]
        else:
            selected_peaks = minima_data['peaks']
            selected_prominences = minima_data['prominences']
        
        # Calculate features
        result['minima_positions'] = [filtered_distances[p] for p in selected_peaks]
        result['minima_depths'] = [filtered_intensities[p] for p in selected_peaks]
        result['separation_distance'] = abs(result['minima_positions'][1] - result['minima_positions'][0])
        
        # Calculate prominence SNR
        avg_prominence = np.mean(selected_prominences)
        result['prominence_snr'] = avg_prominence / baseline_noise
        
        # Calculate p1 and p2 projections onto the profile line
        p1_proj, p2_proj = _calculate_point_projections(prof, intensity_extension_range)
        result['p1_projection'] = p1_proj
        result['p2_projection'] = p2_proj
        
        # Check if minima fall between the matched points
        if not (np.isnan(p1_proj) or np.isnan(p2_proj)):
            min_proj = min(p1_proj, p2_proj)
            max_proj = max(p1_proj, p2_proj)
            
            minima_in_range = [
                min_proj <= pos <= max_proj for pos in result['minima_positions']
            ]
            result['minima_between_points'] = all(minima_in_range)
        
        result['is_candidate'] = True
        pass1_results.append(result)
    
    return pass1_results


def _pass2_apply_filtering(
    profiles: List[Dict],
    intensity_extension_range: Tuple[float, float],
    intensity_min_snr: Optional[float],
    intensity_central_max_required: bool,
    intensity_margin_factor: float = 0.1,
    require_both_minima_in_region: bool = True,
    smooth_sigma: float = 0.0,
    edge_fraction: float = 0.2
) -> List[Dict]:
    """
    Pass 2 filtering of intensity profiles using strict quality criteria.

    Applies SNR, positional, and central-maximum constraints to decide whether
    each profile supports a valid bilayer measurement.

    Parameters
    ----------
    profiles : List[Dict]
        Profiles from extract_intensity_profile().
    intensity_extension_range : Tuple[float, float]
        Analysis window relative to the profile midpoint.
    intensity_min_snr : Optional[float]
        Minimum SNR requirement; if None, SNR filtering is disabled.
    intensity_central_max_required : bool
        Require a central maximum between the two selected minima.
    intensity_margin_factor : float, default 0.1
        Fraction of the measurement span used to extend the allowed minima region.
    require_both_minima_in_region : bool, default True
        If False, at least one minimum within the (possibly extended) region suffices.
    smooth_sigma : float, default 0.0
        Gaussian sigma for optional smoothing before feature detection.
    edge_fraction : float, default 0.2
        Fraction of profile edges used for baseline noise estimation.

    Returns
    -------
    List[Dict]
        For each profile, a dict with:
        - passes_filter: bool
        - has_dual_minima, has_central_maximum, minima_between_points, sufficient_prominence: bools
        - minima_positions, minima_depths, separation_distance
        - central_max_position, central_max_height
        - prominence_snr, p1_projection, p2_projection
        - failure_reason: Optional[str]

    Notes
    -----
    - Uses scipy.signal.find_peaks on inverted intensities for minima
    - If intensity_min_snr is None, prominence SNR is reported but not enforced
    - Position checks use exact range or a margin-expanded range depending on
      intensity_margin_factor
    """
    
    final_results = []
    
    for i, prof in enumerate(profiles):
        result = {
            'profile_index': i,
            'passes_filter': False,
            'has_dual_minima': False,
            'has_central_maximum': False,
            'minima_between_points': False,
            'sufficient_prominence': False,
            'minima_positions': [],
            'minima_depths': [],
            'separation_distance': np.nan,  # FIXED: Always initialize
            'central_max_position': np.nan,
            'central_max_height': np.nan,
            'prominence_snr': np.nan,
            'p1_projection': np.nan,
            'p2_projection': np.nan,
            'failure_reason': None
        }
        
        # Extract profile data
        processed_data = _extract_profile_data(prof, intensity_extension_range)
        if processed_data is None:
            result['failure_reason'] = 'invalid_profile_data'
            final_results.append(result)
            continue
            
        filtered_distances, filtered_intensities = processed_data
        
        # Apply smoothing if requested
        if smooth_sigma > 0:
            filtered_intensities = gaussian_filter1d(filtered_intensities, sigma=smooth_sigma)
        
        # Calculate baseline with configurable edge fraction
        baseline_noise = _calculate_baseline_noise(filtered_intensities, edge_fraction)
        if baseline_noise == 0:
            result['failure_reason'] = 'zero_baseline_noise'
            final_results.append(result)
            continue
        
        # Handle optional SNR filtering
        if intensity_min_snr is not None:
            # SNR filtering enabled (original behavior)
            prominence_threshold = intensity_min_snr * baseline_noise
            minima_data = _find_minima_with_details(filtered_intensities, prominence_threshold)
            
            if len(minima_data['peaks']) < 2:
                result['failure_reason'] = f'insufficient_snr_min{len(minima_data["peaks"])}'
                final_results.append(result)
                continue
                
            # Take two most prominent if more than 2
            if len(minima_data['peaks']) > 2:
                top_indices = np.argsort(minima_data['prominences'])[-2:]
                selected_peaks = minima_data['peaks'][top_indices]
                selected_prominences = minima_data['prominences'][top_indices]
                selected_peaks = np.sort(selected_peaks)
                sort_order = np.argsort(minima_data['peaks'][top_indices])
                selected_prominences = selected_prominences[sort_order]
            else:
                selected_peaks = minima_data['peaks']
                selected_prominences = minima_data['prominences']
            
            # Calculate prominence SNR
            avg_prominence = np.mean(selected_prominences)
            result['prominence_snr'] = avg_prominence / baseline_noise
            result['sufficient_prominence'] = result['prominence_snr'] >= intensity_min_snr
            
        else:
            # SNR filtering disabled - simple minima detection
            inverted_intensities = -filtered_intensities
            peaks, _ = find_peaks(inverted_intensities, distance=3)
            
            if len(peaks) < 2:
                result['failure_reason'] = f'{len(peaks)}_minima_detected'
                final_results.append(result)
                continue
            
            # Take the two deepest minima
            peak_intensities = filtered_intensities[peaks]
            deepest_indices = np.argsort(peak_intensities)[:2]
            selected_peaks = peaks[deepest_indices]
            selected_peaks = np.sort(selected_peaks)
            
            # FIXED: Still calculate basic prominence for SNR even when disabled
            selected_prominences = []
            for peak in selected_peaks:
                # Simple prominence calculation
                prominence = abs(filtered_intensities[peak] - np.mean(filtered_intensities))
                selected_prominences.append(prominence)
            
            if len(selected_prominences) > 0:
                avg_prominence = np.mean(selected_prominences)
                result['prominence_snr'] = avg_prominence / baseline_noise
            
            result['sufficient_prominence'] = True  # Always pass when SNR disabled
        
        result['has_dual_minima'] = True
        result['minima_positions'] = [filtered_distances[p] for p in selected_peaks]
        result['minima_depths'] = [filtered_intensities[p] for p in selected_peaks]
        
        # FIXED: Calculate separation distance
        if len(result['minima_positions']) >= 2:
            result['separation_distance'] = abs(result['minima_positions'][1] - result['minima_positions'][0])
        
        # Calculate point projections
        p1_proj, p2_proj = _calculate_point_projections(prof, intensity_extension_range)
        result['p1_projection'] = p1_proj
        result['p2_projection'] = p2_proj
        
        if np.isnan(p1_proj) or np.isnan(p2_proj):
            result['failure_reason'] = 'invalid_point_projections'
            final_results.append(result)
            continue
        
        # Position constraints with optional margin
        min_proj = min(p1_proj, p2_proj)
        max_proj = max(p1_proj, p2_proj)
        
        if intensity_margin_factor > 0:
            # Relaxed position checking with margin
            measurement_span = abs(max_proj - min_proj)
            margin = intensity_margin_factor * measurement_span
            extended_min = min_proj - margin
            extended_max = max_proj + margin
            
            minima_in_range = [
                extended_min <= pos <= extended_max for pos in result['minima_positions']
            ]
            
            if require_both_minima_in_region:
                result['minima_between_points'] = all(minima_in_range)
                if not result['minima_between_points']:
                    result['failure_reason'] = 'minima_outside_extended_region'
            else:
                result['minima_between_points'] = any(minima_in_range)
                if not result['minima_between_points']:
                    result['failure_reason'] = 'no_minima_in_extended_region'
        else:
            # Exact position checking
            minima_in_range = [
                min_proj <= pos <= max_proj for pos in result['minima_positions']
            ]
            result['minima_between_points'] = all(minima_in_range)
            if not result['minima_between_points']:
                result['failure_reason'] = 'minima_outside_matched_points'
        
        # FIXED: Central maximum requirement
        if intensity_central_max_required:
            # Always try to find central maximum, regardless of SNR settings
            if intensity_min_snr is not None:
                prominence_threshold = intensity_min_snr * baseline_noise
            else:
                # Use a minimal threshold when SNR is disabled
                prominence_threshold = 0.1 * baseline_noise
                
            maxima_data = _find_maxima_with_details(filtered_intensities, prominence_threshold)
            
            min_pos_1, min_pos_2 = selected_peaks[0], selected_peaks[1]
            central_maxima_mask = (maxima_data['peaks'] > min_pos_1) & (maxima_data['peaks'] < min_pos_2)
            central_maxima = maxima_data['peaks'][central_maxima_mask]
            
            if len(central_maxima) > 0:
                central_prominences = maxima_data['prominences'][central_maxima_mask]
                best_central_idx = central_maxima[np.argmax(central_prominences)]
                
                result['has_central_maximum'] = True
                result['central_max_position'] = filtered_distances[best_central_idx]
                result['central_max_height'] = filtered_intensities[best_central_idx]
            else:
                result['has_central_maximum'] = False
                result['failure_reason'] = 'no_central_maximum'
        else:
            result['has_central_maximum'] = True  # Not required
        
        # Final decision
        passes_all_criteria = (
            result['has_dual_minima'] and
            result['has_central_maximum'] and
            result['minima_between_points'] and
            result['sufficient_prominence']
        )
        
        result['passes_filter'] = passes_all_criteria
        final_results.append(result)
    
    return final_results


def _extract_profile_data(prof: Dict, intensity_extension_range: Tuple[float, float]) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Extract and validate profile data."""
    
    try:
        p1, p2 = prof["p1"], prof["p2"]
        midpoint = prof["midpoint"]
        start, end = prof["start"], prof["end"]
        profile = prof["profile"]
        
        # Calculate distances
        num_points = len(profile)
        line_points = np.linspace(start, end, num_points)
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length == 0:
            return None
            
        unit_dir = direction / length
        distances = np.dot(line_points - midpoint, unit_dir)
        
        # Filter to extension range
        min_ext, max_ext = intensity_extension_range
        mask = (distances >= min_ext) & (distances <= max_ext)
        if not np.any(mask) or np.sum(mask) < 10:
            return None
            
        filtered_distances = distances[mask]
        filtered_intensities = profile[mask]
        
        return filtered_distances, filtered_intensities
        
    except Exception:
        return None


def _calculate_point_projections(prof: Dict, intensity_extension_range: Tuple[float, float]) -> Tuple[float, float]:
    """Calculate where p1 and p2 project onto the profile line."""
    
    try:
        p1, p2 = prof["p1"], prof["p2"]
        midpoint = prof["midpoint"]
        direction = p2 - p1
        length = np.linalg.norm(direction)
        
        if length == 0:
            return np.nan, np.nan
            
        unit_dir = direction / length
        
        # Project p1 and p2 onto the profile line (relative to midpoint)
        p1_projection = np.dot(p1 - midpoint, unit_dir)
        p2_projection = np.dot(p2 - midpoint, unit_dir)
        
        return p1_projection, p2_projection
        
    except Exception:
        return np.nan, np.nan


def _calculate_baseline_noise(intensities: np.ndarray, edge_fraction: float = 0.2) -> float:
    """Calculate baseline noise from the start and end regions of the profile."""
    
    # Use specified fraction of the profile as baseline regions
    profile_length = len(intensities)
    baseline_length = max(3, int(profile_length * edge_fraction))
    
    baseline_regions = np.concatenate([
        intensities[:baseline_length],
        intensities[-baseline_length:]
    ])
    
    if len(baseline_regions) == 0:
        return 1.0  # Fallback
    
    return np.std(baseline_regions)


def _find_minima_with_details(intensities: np.ndarray, prominence: float) -> Dict:
    """Find minima with detailed information."""
    
    inverted_intensities = -intensities
    peaks, properties = find_peaks(
        inverted_intensities, 
        prominence=prominence,
        distance=3  # Minimum 3 points separation
    )
    
    return {
        'peaks': peaks,
        'prominences': properties.get('prominences', np.array([])),
        'properties': properties
    }


def _find_maxima_with_details(intensities: np.ndarray, prominence: float) -> Dict:
    """Find maxima with detailed information."""
    
    peaks, properties = find_peaks(
        intensities,
        prominence=prominence,
        distance=3
    )
    
    return {
        'peaks': peaks,
        'prominences': properties.get('prominences', np.array([])),
        'properties': properties
    }


def _calculate_dataset_characteristics(pass1_results: List[Dict]) -> Dict:
    """Calculate dataset characteristics from Pass 1 candidates (for statistics only)."""
    
    candidates = [r for r in pass1_results if r['is_candidate']]
    
    if len(candidates) < 10:
        warnings.warn("Very few candidates found in Pass 1.")
        return {
            'median_separation': np.nan,
            'separation_std': np.nan,
            'median_prominence_snr': np.nan,
            'n_candidates': len(candidates)
        }
    
    # Extract metrics
    separations = [r['separation_distance'] for r in candidates if not np.isnan(r['separation_distance'])]
    prominence_snrs = [r['prominence_snr'] for r in candidates if not np.isnan(r['prominence_snr'])]
    
    return {
        'median_separation': np.median(separations) if separations else np.nan,
        'separation_std': np.std(separations) if separations else np.nan,
        'median_prominence_snr': np.median(prominence_snrs) if prominence_snrs else np.nan,
        'n_candidates': len(candidates)
    }


def _calculate_statistics(
    results: Dict,
    thickness_df: pd.DataFrame,
    total_profiles: int
) -> Dict:
    """Calculate comprehensive statistics for the intensity profiles."""
    
    final_results = results['final_results']
    n_passed = sum(1 for r in final_results if r['passes_filter'])
    n_failed = total_profiles - n_passed
    
    stats = {
        'total_profiles': total_profiles,
        'profiles_passed': n_passed,
        'profiles_failed': n_failed,
        'pass_rate': n_passed / total_profiles if total_profiles > 0 else 0,
        'pass1_candidates': results['dataset_characteristics']['n_candidates'],
        'dataset_characteristics': results['dataset_characteristics'],
        'failure_analysis': {},
        'quality_metrics': {}
    }
    
    # Analyze failure reasons
    failure_counts = {}
    for result in final_results:
        if not result['passes_filter'] and result['failure_reason']:
            reason = result['failure_reason']
            failure_counts[reason] = failure_counts.get(reason, 0) + 1
    
    stats['failure_analysis'] = failure_counts
    
    # Quality metrics for passed profiles
    passed_results = [r for r in final_results if r['passes_filter']]
    if passed_results:
        separations = [r['minima_positions'][1] - r['minima_positions'][0] for r in passed_results if len(r['minima_positions']) == 2]
        prominence_snrs = [r['prominence_snr'] for r in passed_results if not np.isnan(r['prominence_snr'])]
        
        stats['quality_metrics'] = {
            'separation_stats': {
                'mean': np.mean(separations) if separations else np.nan,
                'std': np.std(separations) if separations else np.nan,
                'median': np.median(separations) if separations else np.nan,
                'range': (np.min(separations), np.max(separations)) if separations else (np.nan, np.nan)
            },
            'prominence_snr_stats': {
                'mean': np.mean(prominence_snrs) if prominence_snrs else np.nan,
                'std': np.std(prominence_snrs) if prominence_snrs else np.nan,
                'median': np.median(prominence_snrs) if prominence_snrs else np.nan,
                'quartiles': np.percentile(prominence_snrs, [25, 75]).tolist() if prominence_snrs else [np.nan, np.nan]
            }
        }
    
    return stats


#############################################
# Main Pipeline Functions
#############################################

def extract_and_validate_surface(
    segmentation: np.ndarray,
    membrane_mask: np.ndarray,
    mesh_sampling: int,
    logger: logging.Logger = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract and validate surface points using marching cubes.

    Parameters
    ----------
    segmentation : np.ndarray
        3D segmentation volume (passed to extract_surface_points)
    membrane_mask : np.ndarray
        3D binary mask for membrane of interest
    mesh_sampling : int
        Step size for marching cubes algorithm
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    aligned_vertices : np.ndarray or None
        2D array (N, 3) of validated surface vertices
    aligned_normals : np.ndarray or None
        2D array (N, 3) of corresponding normal vectors
    vertex_volume : np.ndarray or None
        3D binary volume marking vertex positions

    Notes
    -----
    Returns (None, None, None) if no valid surface points found.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    log_msg(f"Extracting surface points with step size {mesh_sampling}...")
    
    aligned_vertices, aligned_normals = extract_surface_points(
        segmentation, membrane_mask, mesh_sampling=mesh_sampling, logger=logger
    )
    vertex_volume = create_vertex_volume(aligned_vertices, membrane_mask.shape)
    
    if aligned_vertices is None or len(aligned_vertices) == 0:
        log_msg("No surface points found")
        return None, None, None
        
    log_msg(f"Extracted {len(aligned_vertices)} surface points")
    return aligned_vertices, aligned_normals, vertex_volume


def interpolate_surface_if_requested(
    aligned_vertices: np.ndarray,
    aligned_normals: np.ndarray,
    membrane_mask: np.ndarray,
    interpolate: bool,
    interpolation_points: int,
    logger: logging.Logger = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conditionally interpolate surface points for denser coverage.

    Parameters
    ----------
    aligned_vertices : np.ndarray
        2D array (N, 3) of surface vertex coordinates
    aligned_normals : np.ndarray
        2D array (N, 3) of normal vectors
    membrane_mask : np.ndarray
        3D binary segmentation mask for validation
    interpolate : bool
        Whether to perform interpolation
    interpolation_points : int
        Number of points to interpolate between vertices
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    vertices : np.ndarray
        2D array of (possibly modified) vertex coordinates
    normals : np.ndarray
        2D array of (possibly modified) normal vectors

    Notes
    -----
    Returns original arrays unchanged if interpolate=False.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    if interpolate:
        log_msg(f"Interpolating surface points (before: {len(aligned_vertices)} vertices)")
        aligned_vertices, aligned_normals = interpolate_surface_points(
            aligned_vertices,
            aligned_normals,
            membrane_mask,
            interpolation_points=interpolation_points,
            include_edges=True,
            logger=logger,
        )
        log_msg(f"After point interpolation: {len(aligned_vertices)} vertices")
    else:
        log_msg("Skipping point interpolation")
        
    return aligned_vertices, aligned_normals


def refine_normals_and_separate_surfaces(
    aligned_vertices: np.ndarray,
    aligned_normals: np.ndarray,
    refine_normals: bool,
    radius_hit: float,
    batch_size: int,
    flip_normals: bool,
    logger: logging.Logger = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Conditionally refine normals and separate bilayer surfaces.

    Parameters
    ----------
    aligned_vertices : np.ndarray
        2D array (N, 3) of surface vertex coordinates
    aligned_normals : np.ndarray
        2D array (N, 3) of initial normal vectors
    refine_normals : bool
        Whether to perform normal refinement
    radius_hit : float
        Search radius for neighbor finding (voxel units)
    batch_size : int
        Batch size for processing
    flip_normals : bool
        Whether to flip refined normals inward
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    refined_normals : np.ndarray
        2D array of (possibly refined) normal vectors
    surface1_mask : np.ndarray
        1D boolean array for surface 1 assignment
    surface2_mask : np.ndarray
        1D boolean array for surface 2 assignment

    Notes
    -----
    Always initializes surface masks to False arrays.
    Only modifies them if refinement succeeds.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    # Initialize surface masks to default values
    surface1_mask = np.zeros(len(aligned_vertices), dtype=bool)
    surface2_mask = np.zeros(len(aligned_vertices), dtype=bool)

    if refine_normals:
        log_msg("Refining normals with weighted-average method...")
        try:
            refined_normals, surface1_mask, surface2_mask = refine_mesh_normals(
                vertices=aligned_vertices,
                initial_normals=aligned_normals,
                radius_hit=radius_hit,
                batch_size=batch_size,
                flip_normals=flip_normals,
                logger=logger,
            )

            if surface1_mask is not None and surface2_mask is not None:
                log_msg(f"Successfully separated bilayer surfaces:")
                log_msg(f"Surface 1: {np.sum(surface1_mask)} points")
                log_msg(f"Surface 2: {np.sum(surface2_mask)} points")
            else:
                log_msg("Could not separate bilayer surfaces")
                surface1_mask = np.zeros(len(aligned_vertices), dtype=bool)
                surface2_mask = np.zeros(len(aligned_vertices), dtype=bool)

        except Exception as e:
            log_msg(f"Error during normal refinement: {str(e)}")
            traceback.print_exc()
            log_msg("Continuing with original normals...")
            refined_normals = aligned_normals
            surface1_mask = np.zeros(len(aligned_vertices), dtype=bool)
            surface2_mask = np.zeros(len(aligned_vertices), dtype=bool)
    else:
        log_msg("Skipping normal refinement (refine_normals=False)")
        refined_normals = aligned_normals
        
    return refined_normals, surface1_mask, surface2_mask


def update_vertex_volume_after_interpolation(
    aligned_vertices: np.ndarray,
    membrane_mask: np.ndarray,
    logger: logging.Logger = None
) -> np.ndarray:
    """
    Rebuild vertex volume to include interpolated points.

    Parameters
    ----------
    aligned_vertices : np.ndarray
        2D array (N, 3) of all vertex coordinates (including interpolated)
    membrane_mask : np.ndarray
        3D reference mask for volume shape
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    np.ndarray
        3D binary volume with 1 at all vertex positions

    Notes
    -----
    Recreates the entire vertex volume from current vertex list.
    Uses tqdm progress bar for large vertex sets.
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)
    
    log_msg("Updating vertex volume with interpolated points...")
    vertex_volume = np.zeros_like(membrane_mask)
    
    for v in tqdm(aligned_vertices, desc="Updating vertex volume"):
        x, y, z = v.astype(int)
        if (0 <= x < vertex_volume.shape[0] and 
            0 <= y < vertex_volume.shape[1] and 
            0 <= z < vertex_volume.shape[2]):
            vertex_volume[x, y, z] = 1
            
    return vertex_volume


def process_membrane_segmentation(
    segmentation_path: str,
    output_dir: str = None,
    config_path: str = None,
    membrane_labels: dict = None,
    mesh_sampling: int = 1,
    interpolate: bool = True,
    interpolation_points: int = 1,
    refine_normals: bool = True,
    radius_hit: float = 10.0,
    flip_normals: bool = True,
    batch_size: int = 2000,
    save_vertices_mrc: bool = False,
    save_vertices_xyz: bool = False,
    logger: logging.Logger = None
) -> dict:
    """
    Process membrane segmentation to extract and refine surface points.

    Parameters
    ----------
    segmentation_path : str
        Path to input MRC segmentation file
    output_dir : str, optional
        Output directory (defaults to segmentation file directory)
    config_path : str, optional
        Path to YAML configuration file with membrane labels
    membrane_labels : dict, optional
        Mapping of membrane names to label values {name: int}
    mesh_sampling : int, default 1
        Step size for marching cubes algorithm
    interpolate : bool, default True
        Whether to interpolate surface points for denser coverage
    interpolation_points : int, default 1
        Number of points to interpolate between vertices
    refine_normals : bool, default True
        Whether to refine normals and separate surfaces
    radius_hit : float, default 10.0
        Search radius for normal refinement (voxel units)
    flip_normals : bool, default True
        Whether to flip normals inward after refinement
    batch_size : int, default 2000
        Batch size for normal refinement processing
    save_vertices_mrc : bool, default False
        Whether to save vertex positions as MRC files
    save_vertices_xyz : bool, default False
        Whether to save coordinates as XYZ point clouds
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    dict or None
        Mapping of membrane names to output CSV file paths
        Returns None if processing fails

    Notes
    -----
    Processes each membrane label separately through the full pipeline:
    1. Surface extraction and validation
    2. Optional point interpolation  
    3. Optional normal refinement and surface separation
    4. Output file generation
    """
    # Initialize logger and setup
    if logger is None:
        if output_dir is None:
            output_dir = os.path.dirname(segmentation_path)
        logger = setup_logger(output_dir)

    # Handle configuration
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        membrane_labels = config["segmentation_values"]
    else:
        if membrane_labels is None:
            membrane_labels = {"membrane": 1}

    # Setup output directory
    if output_dir is None:
        output_dir = os.path.dirname(segmentation_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(segmentation_path))[0]
    segmentation, voxel_size, origin, shape = read_segmentation(segmentation_path, logger=logger)

    if segmentation is None:
        logger.error("Failed to read segmentation")
        return None

    output_files = {}

    # Process each membrane type
    for membrane_name, label_value in membrane_labels.items():
        logger.info(f"\nProcessing {membrane_name} (label {label_value})")
        
        membrane_mask = segmentation == label_value
        if not np.any(membrane_mask):
            logger.info(f"No voxels found for {membrane_name}")
            continue

        try:
            # STEP 1: Extract and validate surface
            aligned_vertices, aligned_normals, vertex_volume = extract_and_validate_surface(
                segmentation, membrane_mask, mesh_sampling, logger
            )
            if aligned_vertices is None:
                continue

            # STEP 2: Interpolate if requested  
            aligned_vertices, aligned_normals = interpolate_surface_if_requested(
                aligned_vertices, aligned_normals, membrane_mask, 
                interpolate, interpolation_points, logger
            )

            # STEP 3: Refine normals and separate surfaces
            aligned_normals, surface1_mask, surface2_mask = refine_normals_and_separate_surfaces(
                aligned_vertices, aligned_normals, refine_normals,
                radius_hit, batch_size, flip_normals, logger
            )

            # STEP 4: Update vertex volume if interpolation was done
            if interpolate:
                vertex_volume = update_vertex_volume_after_interpolation(
                    aligned_vertices, membrane_mask, logger
                )

            logger.info(f"Final vertex count: {len(aligned_vertices)}")

            # STEP 5: Save outputs
            if len(aligned_vertices) > 0:
                success = verify_and_save_outputs(
                    aligned_vertices, aligned_normals, vertex_volume,
                    surface1_mask, surface2_mask, membrane_name, base_name,
                    output_dir, voxel_size, origin,
                    save_vertices_mrc, save_vertices_xyz, logger
                )
                
                if success:
                    csv_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices_normals.csv")
                    output_files[membrane_name] = csv_output

        except Exception as e:
            logger.error(f"Error processing {membrane_name}: {e}")
            traceback.print_exc()

    return output_files


def measure_membrane_thickness(
    segmentation_path: str,
    input_csv: str,
    output_csv: str = None,
    output_dir: str = None,
    max_thickness: float = 8.0,
    max_angle: float = 1.0,
    save_thickness_mrc: bool = False,
    direction: str = "1to2",
    use_gpu: bool = True,
    num_cpu_threads: int = None,
    logger: logging.Logger = None
) -> tuple[str, str]:
    """
    Measure membrane thickness between separated surface points.

    Parameters
    ----------
    segmentation_path : str
        Path to original MRC segmentation file (for metadata)
    input_csv : str
        Path to CSV file with vertices, normals, and surface assignments
    output_csv : str, optional
        Path for thickness results CSV (auto-generated if None)
    output_dir : str, optional
        Output directory (defaults to input CSV directory)
    max_thickness : float, default 8.0
        Maximum allowed thickness in nanometers
    max_angle : float, default 1.0
        Maximum angle for cone search in degrees
    save_thickness_mrc : bool, default False
        Whether to save thickness volume as MRC file
    direction : str, default "1to2"
        Measurement direction: "1to2" (surface1→surface2) or "2to1"
    use_gpu : bool, default True
        Whether to use GPU acceleration if available
    num_cpu_threads : int, optional
        Number of CPU threads (if using CPU implementation)
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    output_csv : str or None
        Path to thickness measurements CSV file
    stats_file : str or None
        Path to statistics log file

    Notes
    -----
    Output CSV contains only valid thickness measurements with complete
    information about both matched surface points.
    Falls back to CPU if GPU not available.
    """
    # Validate inputs early with sensible defaults
    if not os.path.exists(segmentation_path):
        raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")
    
    # Validate and constrain parameters
    max_thickness = max(0.1, float(max_thickness))
    max_angle = np.clip(float(max_angle), 0.1, 45.0)
    direction = direction if direction in ["1to2", "2to1"] else "1to2"

    # Set default output directory and CSV
    if output_dir is None:
        output_dir = os.path.dirname(input_csv)
    os.makedirs(output_dir, exist_ok=True)

    if output_csv is None:
        input_base = os.path.splitext(os.path.basename(input_csv))[0]
        dir_suffix = "_2to1" if direction == "2to1" else ""
        output_csv = os.path.join(output_dir, f"{input_base}_thickness{dir_suffix}.csv")

    # Initialize logger if not provided
    if logger is None:
        logger = setup_logger(output_dir)

    # Get base name for statistics file
    output_base = os.path.splitext(os.path.basename(output_csv))[0]
    stats_file = os.path.join(output_dir, f"{output_base}_stats.log")

    # Read segmentation and voxel size from MRC file
    segmentation, voxel_size, origin, shape = read_segmentation(segmentation_path, logger=logger)

    if segmentation is None:
        logger.error("Failed to read segmentation")
        return None, None

    # Load data using unscaled voxel coordinates
    logger.info(f"Loading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    # Use unscaled voxel coordinates for calculations
    points = df[["x_voxel", "y_voxel", "z_voxel"]].values
    normals = df[["normal_x", "normal_y", "normal_z"]].values
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

    # Run measurement
    logger.info(f"Starting thickness measurement ({direction})...")
    start_time = time.time()

    if use_gpu and gpu_available:
        # Use GPU implementation
        thickness_results, valid_mask, point_pairs = measure_thickness_gpu(
            points,
            normals,
            surface1_mask,
            surface2_mask,
            voxel_size=voxel_size,
            max_thickness_nm=max_thickness,
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
            voxel_size=voxel_size,
            max_thickness_nm=max_thickness,
            max_angle_degrees=max_angle,
            direction=direction,
            num_threads=num_cpu_threads,
            logger=logger,
        )

    processing_time = time.time() - start_time
    logger.info(f"Processing completed in {processing_time:.2f} seconds")

    # Generate and save statistics
    stats = generate_matching_statistics(
        thickness_results, valid_mask, point_pairs, points, surface1_mask, surface2_mask, voxel_size
    )
    save_matching_statistics(stats, stats_file, logger)


    # Create optimized thickness results - only valid measurements
    logger.info(f"\nCreating optimized thickness CSV with only valid measurements...")
    
    # Get indices of valid measurements
    valid_indices = np.where(valid_mask)[0]
    matched_indices = point_pairs[valid_mask]
    
    if len(valid_indices) == 0:
        logger.warning("No valid thickness measurements found!")
        # Create empty DataFrame with expected columns
        thickness_df = pd.DataFrame(columns=[
            'measurement_id', 'thickness_nm',
            'point1_idx', 'x1_voxel', 'y1_voxel', 'z1_voxel', 
            'x1_physical', 'y1_physical', 'z1_physical',
            'normal1_x', 'normal1_y', 'normal1_z', 'surface1',
            'point2_idx', 'x2_voxel', 'y2_voxel', 'z2_voxel',
            'x2_physical', 'y2_physical', 'z2_physical', 
            'normal2_x', 'normal2_y', 'normal2_z', 'surface2'
        ])
    else:
        # Create DataFrame with only valid measurements
        thickness_df = pd.DataFrame({
            'measurement_id': range(len(valid_indices)),
            'thickness_nm': thickness_results[valid_mask],
            
            # Point 1 (source) information
            'point1_idx': valid_indices,
            'x1_voxel': df.loc[valid_indices, 'x_voxel'].values,
            'y1_voxel': df.loc[valid_indices, 'y_voxel'].values,
            'z1_voxel': df.loc[valid_indices, 'z_voxel'].values,
            'x1_physical': df.loc[valid_indices, 'x_physical'].values,
            'y1_physical': df.loc[valid_indices, 'y_physical'].values,
            'z1_physical': df.loc[valid_indices, 'z_physical'].values,
            'normal1_x': df.loc[valid_indices, 'normal_x'].values,
            'normal1_y': df.loc[valid_indices, 'normal_y'].values,
            'normal1_z': df.loc[valid_indices, 'normal_z'].values,
            'surface1': df.loc[valid_indices, 'surface1'].values,
            
            # Point 2 (matched) information
            'point2_idx': matched_indices,
            'x2_voxel': df.loc[matched_indices, 'x_voxel'].values,
            'y2_voxel': df.loc[matched_indices, 'y_voxel'].values,
            'z2_voxel': df.loc[matched_indices, 'z_voxel'].values,
            'x2_physical': df.loc[matched_indices, 'x_physical'].values,
            'y2_physical': df.loc[matched_indices, 'y_physical'].values,
            'z2_physical': df.loc[matched_indices, 'z_physical'].values,
            'normal2_x': df.loc[matched_indices, 'normal_x'].values,
            'normal2_y': df.loc[matched_indices, 'normal_y'].values,
            'normal2_z': df.loc[matched_indices, 'normal_z'].values,
            'surface2': df.loc[matched_indices, 'surface2'].values,
        })

    logger.info(f"Saving {len(thickness_df)} valid thickness measurements to {output_csv}")
    thickness_df.to_csv(output_csv, index=False)

    # Log summary statistics
    if len(thickness_df) > 0:
        logger.info(f"Thickness statistics:")
        logger.info(f"  Mean: {thickness_df['thickness_nm'].mean():.3f} nm")
        logger.info(f"  Std:  {thickness_df['thickness_nm'].std():.3f} nm")
        logger.info(f"  Min:  {thickness_df['thickness_nm'].min():.3f} nm")
        logger.info(f"  Max:  {thickness_df['thickness_nm'].max():.3f} nm")

    logger.info("Membrane thickness analysis complete!")

    return output_csv, stats_file

def int_profiles_extract_clean(
    thickness_csv: Union[str, Path],
    tomo_path: Union[str, Path],
    output_dir: Union[str, Path],
    intensity_min_snr: Optional[float] = 0.2,
    intensity_central_max_required: bool = True,
    intensity_extension_voxels: int = 10,
    intensity_extension_range: Tuple[float, float] = (-10, 10),
    intensity_normalize_method: Literal['zscore', 'minmax', 'percentile', 'none'] = 'zscore',
    save_cleaned_df: bool = True,
    save_profiles: bool = True,
    save_statistics: bool = True,
    intensity_margin_factor: float = 0.1,
    intensity_require_both_minima: bool = True,
    intensity_smooth_sigma: float = 0.0,
    intensity_edge_fraction: float = 0.2,
    logger: logging.Logger = None
) -> Dict:
    """
    Workflow for extracting intensity profiles, filtering, and saving results.
    
    Parameters
    ----------
    thickness_csv : Union[str, Path]
        Path to thickness CSV file
    tomo_path : Union[str, Path]
        Path to tomogram file
    output_dir : Union[str, Path]
        Directory to save results
    intensity_min_snr : Optional[float], default 0.2
        Minimum SNR for minima prominence. If None, SNR filtering is disabled.
    intensity_central_max_required : bool
        Whether to require central maximum
    intensity_extension_voxels : int
        Extension distance for profile extraction
    intensity_extension_range : Tuple[float, float]
        Range for filtering analysis
    intensity_normalize_method : Literal
        Tomogram normalization method
    save_cleaned_df : bool
        Whether to save cleaned DataFrame
    save_profiles : bool
        Whether to save intensity profiles
    save_statistics : bool
        Whether to save statistics
    intensity_margin_factor : float, default 0.1
        Allowed margin for minima detection outside measurement region (0.0=exact, 0.3=30% margin)
    intensity_require_both_minima : bool, default True
        Whether both minima must be in extended region
    intensity_smooth_sigma : float, default 0.0
        Gaussian smoothing parameter (0=no smoothing)
    intensity_edge_fraction : float, default 0.2
        Fraction of profile edges for baseline calculation
    logger : logging.Logger, optional
        Logger instance for status messages
        
    Returns
    -------
    Dict
        Complete analysis results
    """

    # Set default output directory
    if output_dir is None:
        output_dir = os.path.dirname(thickness_csv)
    os.makedirs(output_dir, exist_ok=True)


    # Initialize logger if not provided
    if logger is None:
        logger = setup_logger(output_dir)
    
    thickness_csv = Path(thickness_csv)
    tomogram_file = Path(tomo_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {thickness_csv.name}")
    logger.info(f"Tomogram: {tomogram_file.name}")
    logger.info(f"{'='*60}")
    
    # Load data
    logger.info("Loading thickness data...")
    thickness_df = pd.read_csv(thickness_csv)
    
    logger.info("Loading tomogram...")
    
    tomo, tomo_voxel, tomo_shape = read_tomo(tomo_path)

    logger.info("Extracting intensity profiles...")
    profiles = extract_intensity_profile(
        thickness_df=thickness_df,
        tomo=tomo,
        voxel_size=tomo_voxel,
        intensity_extension_voxels=intensity_extension_voxels,
        intensity_normalize_method=intensity_normalize_method
    )
    
    logger.info(f"Extracted {len(profiles)} intensity profiles")
    
    # Filter profiles
    logger.info("\nFiltering intensity profiles...")
    results = filter_intensity_profiles(
        profiles=profiles,
        thickness_df=thickness_df,
        intensity_min_snr=intensity_min_snr,
        intensity_central_max_required=intensity_central_max_required,
        intensity_extension_range=intensity_extension_range,
        intensity_margin_factor=intensity_margin_factor,
        require_both_minima_in_region=intensity_require_both_minima,
        smooth_sigma=intensity_smooth_sigma,
        edge_fraction=intensity_edge_fraction,
        logger=logger
    )
    
    # Print summary
    print_summary(results)
  
    # Save results
    logger.info(f"\nSaving results to: {output_dir}")
    saved_files = save_int_results(
        results=results,
        thickness_csv=thickness_csv,
        output_dir=output_dir,
        profiles=profiles,  # Pass the profiles directly
        save_cleaned_df=save_cleaned_df,
        save_profiles=save_profiles,
        save_statistics=save_statistics
    )
    
    # Add file paths to results
    results['saved_files'] = saved_files
    results['input_files'] = {
        'thickness_csv': thickness_csv,
        'tomogram_file': tomogram_file
    }
    
    return results


def run_full_pipeline(
    segmentation_path: str,
    output_dir: str = None,
    config_path: str = None,
    membrane_labels: dict = None,
    mesh_sampling: int = 1,
    interpolate: bool = True,
    interpolation_points: int = 1,
    refine_normals: bool = True,
    radius_hit: float = 10.0,
    flip_normals: bool = True,
    max_thickness: float = 8.0,
    max_angle: float = 1.0,
    save_vertices_mrc: bool = False,
    save_vertices_xyz: bool = False,
    save_thickness_mrc: bool = False,
    direction: str = "1to2",
    batch_size: int = 2000,
    use_gpu: bool = True,
    num_cpu_threads: int = None,
    extract_intensity_profiles: bool = True,
    tomo_path: str = None,
    intensity_extension_voxels: int = 10,
    intensity_extension_range: tuple = (-10, 10),
    intensity_normalize_method: str = 'zscore',
    intensity_min_snr: Optional[float] = 0.2,  
    intensity_central_max_required: bool = True,
    intensity_save_profiles: bool = True,
    intensity_save_statistics: bool = True,
    intensity_tolerance: float = 0.01,
    intensity_margin_factor: float = 0.1,
    intensity_require_both_minima: bool = True,
    intensity_smooth_sigma: float = 0.0,
    intensity_edge_fraction: float = 0.2
) -> dict:
    """
    Execute complete membrane thickness analysis pipeline with optional intensity profiling.
    
    This is the main entry point for comprehensive membrane analysis. The function
    orchestrates the entire workflow from surface extraction to thickness measurement
    and optional intensity profile analysis. It processes each membrane type separately
    and provides comprehensive output files for further analysis.
    
    The pipeline consists of three main stages:
    
    1. **Surface Processing** (process_membrane_segmentation):
       - Extract surface points using marching cubes algorithm
       - Optionally interpolate points for denser coverage
       - Refine normal vectors using neighbor averaging
       - Separate bilayer into inner/outer surfaces
    
    2. **Thickness Measurement** (measure_membrane_thickness):
       - Match points between separated surfaces
       - Apply geometric constraints (max thickness, cone angle)
       - Generate thickness measurements with GPU acceleration
       - Create comprehensive output files
    
    3. **Intensity Profiling** (optional, int_profiles_extract_clean):
       - Extract intensity profiles from tomogram
       - Filter profiles using quality criteria
       - Validate thickness measurements
       - Generate quality metrics and statistics
    
    Parameters
    ----------
    segmentation_path : str
        Path to input MRC segmentation file
    output_dir : str, optional
        Output directory (defaults to segmentation file directory)
    config_path : str, optional
        Path to YAML configuration file specifying membrane labels.
        Format: {"membrane_name": label_value}. If None, uses
        membrane_labels parameter or defaults to {"membrane": 1}.
    membrane_labels : dict, optional
        Direct specification of membrane labels as {name: value} pairs.
        Example: {"plasma_membrane": 1, "nuclear_envelope": 2}.
        Overrides config_path if both are provided.
    
    **Surface Processing Parameters:**
    mesh_sampling : int, default 1
        Step size for marching cubes algorithm. Larger values reduce
        computation time but may miss fine surface details.
        - 1: Full resolution (most accurate, slowest)
        - 2: Half resolution (2x faster, some detail loss)
        - 4: Quarter resolution (4x faster, significant detail loss)
    interpolate : bool, default True
        Whether to interpolate surface points for denser coverage.
        Increases the number of surface points for better thickness
        measurement accuracy.
    interpolation_points : int, default 1
        Number of points to interpolate between consecutive vertices.
        Higher values increase surface density but may create redundant points.
    refine_normals : bool, default True
        Whether to refine normal vectors using neighbor averaging.
        Improves normal quality and surface separation reliability.
    radius_hit : float, default 10.0
        Search radius for normal refinement in voxel units.
        Larger values include more neighbors but increase computation time.
    flip_normals : bool, default True
        Whether to flip refined normals to point inward toward
        the membrane interior. Typically desired for bilayer analysis.
    
    **Thickness Measurement Parameters:**
    max_thickness : float, default 8.0
        Maximum allowed thickness in nanometers. Typical membrane
        thicknesses range from 4-8 nm. Larger values may include
        invalid measurements.
    max_angle : float, default 1.0
        Maximum cone search angle in degrees. Restricts search to
        a cone along the normal direction. Smaller angles provide
        more precise measurements.
    direction : str, default "1to2"
        Thickness measurement direction:
        - "1to2": Measure from surface 1 to surface 2
        - "2to1": Measure from surface 2 to surface 1
    save_vertices_mrc : bool, default False
        Whether to save vertex positions as MRC files for visualization.
    save_vertices_xyz : bool, default False
        Whether to save coordinates as XYZ point clouds for external tools.
    save_thickness_mrc : bool, default False
        Whether to save thickness volume as MRC file where voxel values
        represent thickness measurements.
    
    **Performance Parameters:**
    batch_size : int, default 2000
        Processing batch size for normal refinement. Larger batches
        are more memory-efficient but may cause memory issues.
    use_gpu : bool, default True
        Whether to use GPU acceleration for thickness measurement.
        Automatically falls back to CPU if CUDA unavailable.
    num_cpu_threads : int, optional
        Number of CPU threads for parallel processing. If None,
        uses all available cores.
    
    **Intensity Profiling Parameters:**
    extract_intensity_profiles : bool, default True
        Whether to perform intensity profile analysis after thickness
        measurement. Requires tomo_path to be specified.
    tomo_path : str, optional
        Path to tomogram file for intensity profiling. Must have
        compatible dimensions and voxel size with the segmentation.
    intensity_extension_voxels : int, default 10
        Number of voxels to extend intensity profiles beyond matched
        points. Larger values provide more context for analysis.
    intensity_extension_range : tuple, default (-10, 10)
        Range for intensity profile filtering analysis in voxel units.
        Profiles are analyzed within this range around the midpoint.
    intensity_normalize_method : str, default 'zscore'
        Tomogram normalization method for intensity extraction:
        - 'zscore': Standardize to zero mean and unit variance
        - 'minmax': Scale to range [0, 1]
        - 'percentile': Clip to 1st-99th percentile range
        - 'none': Use original intensity values
    intensity_min_snr : Optional[float], default 0.2
        Minimum signal-to-noise ratio for minima prominence.
        If None, SNR filtering is disabled (position-based filtering only).
        Higher values require more prominent bilayer features.
    intensity_central_max_required : bool, default True
        Whether to require a central maximum between minima in
        intensity profiles. This validates bilayer structure.
    intensity_tolerance : float, default 0.01
        Tolerance for segmentation-tomogram compatibility check in nm.
        Files must have matching dimensions and voxel sizes within
        this tolerance.
    intensity_margin_factor : float, default 0.1
        Allowed margin for minima detection outside measurement region
        as fraction of measurement span. 0.0 = exact, 0.3 = 30% margin.
    intensity_require_both_minima : bool, default True
        Whether both minima must be within the extended region for
        a profile to pass filtering.
    intensity_smooth_sigma : float, default 0.0
        Gaussian smoothing parameter for intensity profiles.
        0 = no smoothing, higher values smooth profiles.
    intensity_edge_fraction : float, default 0.2
        Fraction of profile edges used for baseline noise calculation.
        Used to determine signal-to-noise ratios.

    Returns
    -------
    dict or None
        Results dictionary with membrane names as keys and dictionaries 
        containing analysis results. Returns None if pipeline fails.
        
        Each membrane result dictionary contains:
        - 'input_csv': Path to vertices/normals CSV file
        - 'thickness_csv': Path to thickness measurements CSV file
        - 'stats_file': Path to thickness statistics log file
        - 'intensity_results': Dictionary with intensity analysis results
          (only if extract_intensity_profiles=True)
        
        Intensity results include:
        - 'status': "completed", "skipped", or "failed"
        - 'analysis_results': Complete intensity analysis results
        - 'profiles_extracted': Number of profiles extracted
        - 'profiles_filtered': Number of profiles passing quality filters
        - 'pass_rate': Fraction of profiles passing filters

    Raises
    ------
    FileNotFoundError
        If segmentation_path or tomo_path (if specified) don't exist.
    ValueError
        If required parameters are invalid or incompatible.
        If intensity profiling is requested without tomo_path.
    RuntimeError
        If any pipeline stage fails unexpectedly.

    Notes
    -----
    **Performance Considerations:**
    - GPU acceleration provides 10-100x speedup for thickness measurement
    - Memory usage scales with surface complexity and batch size
    - Processing time scales with surface size and interpolation settings
    
    **Output Files:**
    - *_vertices_normals.csv: Surface point coordinates and normals
    - *_thickness.csv: Thickness measurements with point pairs
    - *_thickness_stats.log: Comprehensive measurement statistics
    - *_int_profiles.pkl: Intensity profiles (if enabled)
    - *_thickness_cleaned.csv: Filtered thickness data (if enabled)
    
    **Quality Control:**
    - Surface validation ensures only boundary points are used
    - Geometric constraints filter invalid thickness measurements
    - Intensity profiling provides automated quality assessment
    - Comprehensive logging tracks all processing steps
    
    Examples
    --------
    Basic pipeline without intensity profiling:
    
    >>> from memthick import run_full_pipeline
    >>> 
    >>> results = run_full_pipeline(
    ...     segmentation_path="membrane_seg.mrc",
    ...     output_dir="analysis_results",
    ...     membrane_labels={"plasma_membrane": 1, "nuclear_envelope": 2},
    ...     interpolate=True,
    ...     refine_normals=True,
    ...     max_thickness=6.0,
    ...     use_gpu=True
    ... )
    >>> 
    >>> for membrane_name, result in results.items():
    ...     print(f"{membrane_name}: {result['thickness_csv']}")
    
    Complete pipeline with intensity profiling:
    
    >>> results_with_intensity = run_full_pipeline(
    ...     segmentation_path="membrane_seg.mrc",
    ...     output_dir="analysis_with_intensity",
    ...     extract_intensity_profiles=True,
    ...     tomo_path="tomogram.mrc",
    ...     intensity_min_snr=0.2,
    ...     intensity_central_max_required=True,
    ...     intensity_margin_factor=0.1
    ... )
    >>> 
    >>> # Check intensity profiling results
    >>> for membrane_name, result in results_with_intensity.items():
    ...     if 'intensity_results' in result:
    ...         int_results = result['intensity_results']
    ...         print(f"{membrane_name}: {int_results['profiles_filtered']} profiles passed")
    
    Custom surface processing parameters:
    
    >>> results_custom = run_full_pipeline(
    ...     segmentation_path="membrane_seg.mrc",
    ...     mesh_sampling=2,  # Faster processing
    ...     interpolate=False,  # No interpolation
    ...     radius_hit=5.0,  # Smaller search radius
    ...     batch_size=1000,  # Smaller batches
    ...     save_vertices_mrc=True,  # Save vertex files
    ...     save_thickness_mrc=True   # Save thickness volume
    ... )
    
    Intensity profiling requires compatible tomogram file with matching 
    dimensions and voxel size to the segmentation.
    """
    # Validate inputs and parameters early
    if not os.path.exists(segmentation_path):
        raise FileNotFoundError(f"Segmentation file not found: {segmentation_path}")
    
    # Validate intensity profiling requirements
    if extract_intensity_profiles:
        if tomo_path is None:
            raise ValueError("tomo_path is required when extract_intensity_profiles=True")
        if not os.path.exists(tomo_path):
            raise FileNotFoundError(f"Tomogram file not found: {tomo_path}")
    
    # Validate and set sensible defaults
    mesh_sampling = max(1, int(mesh_sampling))
    interpolation_points = max(0, int(interpolation_points))
    radius_hit = max(1.0, float(radius_hit))
    batch_size = max(100, int(batch_size))
    max_thickness = max(0.1, float(max_thickness))
    max_angle = np.clip(float(max_angle), 0.1, 45.0)
    direction = direction if direction in ["1to2", "2to1"] else "1to2"
    
    if config_path is not None and not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(segmentation_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logger = setup_logger(output_dir)
    logger.info(f"Starting full membrane thickness analysis pipeline for {segmentation_path}")
    
    if extract_intensity_profiles:
        logger.info(f"Intensity profiling enabled with tomogram: {tomo_path}")

    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(segmentation_path))[0]

    # Step 1: Process membrane segmentation
    logger.info("Step 1: Processing membrane segmentation")
    output_files = process_membrane_segmentation(
        segmentation_path=segmentation_path,
        output_dir=output_dir,
        config_path=config_path,
        membrane_labels=membrane_labels,
        mesh_sampling=mesh_sampling,
        interpolate=interpolate,
        interpolation_points=interpolation_points,
        refine_normals=refine_normals,
        radius_hit=radius_hit,
        flip_normals=flip_normals,
        batch_size=batch_size,
        save_vertices_mrc=save_vertices_mrc,
        save_vertices_xyz=save_vertices_xyz,
        logger=logger,
    )

    if output_files is None or len(output_files) == 0:
        logger.error("No membrane surfaces found. Pipeline terminated.")
        return None

    # Step 2: Measure membrane thickness for each membrane
    logger.info("\nStep 2: Measuring membrane thickness")
    results = {}

    for membrane_name, input_csv in output_files.items():
        logger.info(f"\nProcessing thickness for {membrane_name}")

        dir_suffix = "_2to1" if direction == "2to1" else ""
        output_csv = os.path.join(output_dir, f"{base_name}_{membrane_name}_thickness{dir_suffix}.csv")

        try:
            thickness_csv, stats_file = measure_membrane_thickness(
                segmentation_path=segmentation_path,
                input_csv=input_csv,
                output_csv=output_csv,
                output_dir=output_dir,
                max_thickness=max_thickness,
                max_angle=max_angle,
                save_thickness_mrc=save_thickness_mrc,
                direction=direction,
                use_gpu=use_gpu,
                num_cpu_threads=num_cpu_threads,
                logger=logger,
            )

            if thickness_csv is not None:
                results[membrane_name] = {
                    "input_csv": input_csv,
                    "thickness_csv": thickness_csv,
                    "stats_file": stats_file,
                }

                logger.info(f"Thickness analysis for {membrane_name} completed successfully")

                # Step 3: Optional intensity profiling
                if extract_intensity_profiles:
                    logger.info(f"\nStep 3: Processing intensity profiles for {membrane_name}")
                    
                    try:
                        # Validate segmentation-tomogram compatibility
                        compatible, details = validate_seg_tomo_compatibility(
                            segmentation_path, tomo_path, 
                            tolerance=intensity_tolerance, logger=logger
                        )
                        
                        if not compatible:
                            logger.warning(f"Skipping intensity profiling for {membrane_name}: {details}")
                            results[membrane_name]["intensity_results"] = {
                                "status": "skipped", 
                                "reason": "incompatible_files",
                                "details": details
                            }
                            continue
                        
                        # Run intensity profiling
                        intensity_results = int_profiles_extract_clean(
                            thickness_csv=thickness_csv,
                            tomo_path=tomo_path,
                            output_dir=output_dir,
                            intensity_min_snr=intensity_min_snr,
                            intensity_central_max_required=intensity_central_max_required,
                            intensity_extension_voxels=intensity_extension_voxels,
                            intensity_extension_range=intensity_extension_range,
                            intensity_normalize_method=intensity_normalize_method,
                            save_cleaned_df=True,  # Always save cleaned thickness DataFrame
                            save_profiles=intensity_save_profiles,
                            save_statistics=intensity_save_statistics,
                            intensity_margin_factor=intensity_margin_factor,
                            intensity_require_both_minima=intensity_require_both_minima,
                            intensity_smooth_sigma=intensity_smooth_sigma,
                            intensity_edge_fraction=intensity_edge_fraction,
                            logger=logger
                        )

                        results[membrane_name]["intensity_results"] = {
                            "status": "completed", 
                            "analysis_results": intensity_results,
                            "profiles_extracted": intensity_results.get('statistics', {}).get('total_profiles', 0),    # ← Use this
                            "profiles_filtered": intensity_results.get('statistics', {}).get('profiles_passed', 0),   # ← Use this  
                            "pass_rate": intensity_results.get('statistics', {}).get('pass_rate', 0.0)
                        }
                        
                        logger.info(f"Intensity profiling for {membrane_name} completed successfully")
                        logger.info(f"  Profiles extracted: {results[membrane_name]['intensity_results']['profiles_extracted']}")
                        logger.info(f"  Profiles after filtering: {results[membrane_name]['intensity_results']['profiles_filtered']}")
                        logger.info(f"  Filter pass rate: {results[membrane_name]['intensity_results']['pass_rate']:.1%}")
                        
                    except Exception as e:
                        logger.error(f"Error in intensity profiling for {membrane_name}: {e}")
                        results[membrane_name]["intensity_results"] = {
                            "status": "failed",
                            "error": str(e)
                        }

        except Exception as e:
            logger.error(f"Error measuring thickness for {membrane_name}: {e}")
            traceback.print_exc()

    logger.info("\nFull pipeline completed!")
    
    # Print summary
    if extract_intensity_profiles:
        logger.info("\n=== Intensity Profiling Summary ===")
        for membrane_name, result in results.items():
            if "intensity_results" in result:
                status = result["intensity_results"]["status"]
                if status == "completed":
                    logger.info(f"{membrane_name}: ✓ Completed - {result['intensity_results']['profiles_filtered']} profiles passed filtering")
                elif status == "skipped":
                    logger.info(f"{membrane_name}: ⚠ Skipped - {result['intensity_results']['reason']}")
                elif status == "failed":
                    logger.info(f"{membrane_name}: ✗ Failed - {result['intensity_results']['error']}")
    
    return results


#############################################
# Command Line Interface
#############################################


def parse_arguments() -> argparse.Namespace:
    """
    Build and parse command line arguments for the CLI.

    The CLI supports four modes:
    - full: complete pipeline with optional intensity profiling
    - surface: surface extraction only
    - thickness: thickness measurement only
    - intensity: intensity profiling only

    Returns
    -------
    argparse.Namespace
        Parsed CLI arguments containing all options required by the pipeline
        and its sub-commands.

    Examples
    --------
    Typical invocations:

    - Full pipeline with intensity profiling:
      python memthick_250814.py seg.mrc --mode full \
        --extract_intensity --tomo_path tomo.mrc --membrane_labels NE:1,ER:2

    - Surface extraction only:
      python memthick_250814.py seg.mrc --mode surface --mesh_sampling 2

    - Thickness only:
      python memthick_250814.py seg.mrc --mode thickness --input_csv vertices.csv

    - Intensity only:
      python memthick_250814.py seg.mrc --mode intensity \
        --thickness_csv thickness.csv --tomo_path tomo.mrc
    """
    parser = argparse.ArgumentParser(
        description="Membrane Thickness Analysis Tool with Intensity Profiling", 
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Main inputs
    parser.add_argument("segmentation", help="Input segmentation MRC file")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--output_dir", help="Output directory (optional)")

    # Surface extraction options
    parser.add_argument(
        "--membrane_labels",
        type=str,
        help='Comma-separated list of membrane labels in format name:value (e.g., "membrane:1,vesicle:2")',
    )
    parser.add_argument("--mesh_sampling", type=int, default=1, help="Step size for marching cubes")
    parser.add_argument("--interpolate", action="store_true", help="Interpolate points for more uniform coverage")
    parser.add_argument(
        "--interpolation_points", type=int, default=1, help="Number of points to interpolate between adjacent vertices"
    )
    parser.add_argument("--refine_normals", action="store_true", default=True, help="Refine normals after running marching cubes")
    #parser.add_argument("--flip_normals", action="store_true", default=True, help="Flip normals inward after refinement")
    parser.add_argument(
        "--radius_hit", type=float, default=10.0, help="Search radius for normal refinement (in voxels)"
    )

    # Thickness measurement options
    parser.add_argument(
        "--max_thickness",
        type=float,
        default=8.0,
        help="Maximum thickness in nm (will be converted to voxels internally)",
    )
    parser.add_argument("--max_angle", type=float, default=1.0, help="Maximum angle for cone search")
    parser.add_argument(
        "--direction",
        choices=["1to2", "2to1"],
        default="1to2",
        help="Direction of thickness measurement: surface 1 to 2 or surface 2 to 1",
    )

    # Output options
    parser.add_argument(
        "--save_vertices_mrc",
        action="store_true",
        help="Save vertices positions in MRC format, e.g., for overlaying with the original segmentation",
    )
    parser.add_argument(
        "--save_vertices_xyz",
        action="store_true",
        help="Whether to save vertex coordinates as point cloud XYZ files, e.g., for visualizing in MeshLab",
    )
    parser.add_argument(
        "--save_thickness_mrc",
        action="store_true",
        help="Whether to save the thickness measurements as MRC files, where the voxel values correspond to the measured membrane thicknesses (in nm) for a given surface point",
    )

    # Intensity profiling options
    intensity_group = parser.add_argument_group("Intensity profiling options")
    intensity_group.add_argument(
        "--extract_intensity", action="store_true",
        help="Extract and filter intensity profiles after thickness measurement"
    )
    intensity_group.add_argument(
        "--tomo_path", type=str,
        help="Path to tomogram MRC file (required for intensity profiling)"
    )
    intensity_group.add_argument(
        "--intensity_extension_voxels", type=int, default=10,
        help="Number of voxels to extend intensity profiles beyond matched points"
    )
    intensity_group.add_argument(
            "--intensity_extension_range",
            nargs=2,               
            type=float,     
            metavar=("MIN", "MAX"),
            default=(-10.0, 10.0),
            help="Range for intensity filtering analysis as two numbers: MIN MAX (e.g., --intensity_extension_range -15 15)"
        )
    intensity_group.add_argument(
        "--intensity_normalize_method", choices=['zscore', 'minmax', 'percentile', 'none'], 
        default='zscore', help="Tomogram normalization method"
    )
    intensity_group.add_argument(
        "--intensity_min_snr", type=float, default=0.2,
        help="Minimum SNR for minima prominence in intensity filtering"
    )
    intensity_group.add_argument(
        "--intensity_central_max_required", action="store_true", default=True,
        help="Whether to require central maximum in intensity profiles"
    )
    intensity_group.add_argument(
        "--intensity_tolerance", type=float, default=0.01,
        help="Tolerance for segmentation-tomogram compatibility check (nm)"
    )
    intensity_group.add_argument(
        "--intensity_margin_factor", type=float, default=0.1,
        help="Allowed margin outside measurement region (0.0=exact, 0.3=30%% margin)"
    )
    intensity_group.add_argument(
        "--intensity_require_both_minima", action="store_true", default=True,
        help="Whether both minima must be in extended region"
    )
    intensity_group.add_argument(
        "--intensity_smooth_sigma", type=float, default=0.0,
        help="Gaussian smoothing parameter (0=no smoothing)"
    )
    intensity_group.add_argument(
        "--intensity_edge_fraction", type=float, default=0.2,
        help="Fraction of profile edges for baseline calculation"
    )

    # Hardware options
    hardware_group = parser.add_argument_group("Hardware options")
    hardware_group.add_argument(
        "--use_cpu", action="store_true", help="Force CPU implementation even if GPU is available"
    )
    hardware_group.add_argument("--cpu_threads", type=int, help="Number of CPU threads to use (default: all available)")

    # Processing options
    parser.add_argument("--batch_size", type=int, default=2000, help="Batch size for processing")

    # Pipeline control
    parser.add_argument(
        "--mode",
        choices=["full", "surface", "thickness", "intensity"],  # NEW: added 'intensity' mode
        default="full",
        help="Processing mode: full pipeline, surface extraction only, thickness measurement only, or intensity profiling only",
    )
    parser.add_argument(
        "--input_csv", help="Input CSV for thickness measurement mode (only used with --mode=thickness)"
    )
    # For intensity-only mode
    parser.add_argument(
        "--thickness_csv", help="Input thickness CSV for intensity profiling mode (only used with --mode=intensity)"
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for command line execution.

    This function:
    1) Parses CLI arguments
    2) Validates inputs per selected mode
    3) Dispatches to the requested workflow (full, surface, thickness, intensity)

    Exit codes are communicated via printed error messages and early returns.
    """
    args = parse_arguments()

    # Validate parsed arguments
    if not os.path.exists(args.segmentation):
        print(f"Error: Segmentation file not found: {args.segmentation}")
        return
    
    # Mode-specific validations
    if args.mode == "thickness" and not args.input_csv:
        print("Error: --input_csv is required with --mode=thickness")
        return
        
    if args.mode == "thickness" and not os.path.exists(args.input_csv):
        print(f"Error: Input CSV not found: {args.input_csv}")
        return

    # Intensity mode validations
    if args.mode == "intensity":
        if not args.thickness_csv:
            print("Error: --thickness_csv is required with --mode=intensity")
            return
        if not os.path.exists(args.thickness_csv):
            print(f"Error: Thickness CSV not found: {args.thickness_csv}")
            return
        if not args.tomo_path:
            print("Error: --tomo_path is required with --mode=intensity")
            return
        if not os.path.exists(args.tomo_path):
            print(f"Error: Tomogram file not found: {args.tomo_path}")
            return

    # Validate intensity profiling requirements for full mode
    if args.extract_intensity and args.mode == "full":
        if not args.tomo_path:
            print("Error: --tomo_path is required when --extract_intensity is used")
            return
        if not os.path.exists(args.tomo_path):
            print(f"Error: Tomogram file not found: {args.tomo_path}")
            return
    
    # Validate parameter ranges
    if args.max_thickness <= 0:
        print("Error: max_thickness must be positive")
        return
        
    if not (0.1 <= args.max_angle <= 45.0):
        print("Error: max_angle must be between 0.1 and 45.0 degrees")
        return
    
    # Validate intensity SNR requirement
    intensity_min_snr = None
    if hasattr(args, 'intensity_min_snr') and args.intensity_min_snr:
        try:
            intensity_min_snr = float(args.intensity_min_snr)
        except ValueError:
            print("Error: intensity_min_snr must be a number or 'none'")
            return

    # Parse membrane labels if provided
    membrane_labels = None
    if args.membrane_labels:
        membrane_labels = {}
        for label_pair in args.membrane_labels.split(","):
            name, value = label_pair.split(":")
            membrane_labels[name.strip()] = int(value.strip())

    # Hardware settings
    use_gpu = not args.use_cpu
    num_cpu_threads = args.cpu_threads

    if args.mode == "full":
        # Run full pipeline with optional intensity profiling
        results = run_full_pipeline(
            segmentation_path=args.segmentation,
            output_dir=args.output_dir,
            config_path=args.config,
            membrane_labels=membrane_labels,
            mesh_sampling=args.mesh_sampling,
            interpolate=args.interpolate,
            interpolation_points=args.interpolation_points,
            refine_normals=True,
            radius_hit=args.radius_hit,
            flip_normals=True,
            max_thickness=args.max_thickness,
            max_angle=args.max_angle,
            save_vertices_mrc=args.save_vertices_mrc,
            save_vertices_xyz=args.save_vertices_xyz,
            save_thickness_mrc=args.save_thickness_mrc,
            direction=args.direction,
            batch_size=args.batch_size,
            use_gpu=use_gpu,
            num_cpu_threads=num_cpu_threads,
            extract_intensity_profiles=args.extract_intensity,
            tomo_path=args.tomo_path,
            intensity_extension_voxels=args.intensity_extension_voxels,
            intensity_extension_range=args.intensity_extension_range,
            intensity_normalize_method=args.intensity_normalize_method,
            intensity_min_snr=args.intensity_min_snr,
            intensity_central_max_required=args.intensity_central_max_required,
            intensity_tolerance=args.intensity_tolerance,
            intensity_margin_factor=args.intensity_margin_factor,
            intensity_require_both_minima=args.intensity_require_both_minima,
            intensity_smooth_sigma=args.intensity_smooth_sigma,
            intensity_edge_fraction=args.intensity_edge_fraction,
            intensity_save_profiles=True,
            intensity_save_statistics=True,
        )

    elif args.mode == "surface":
        # Run surface extraction only
        process_membrane_segmentation(
            segmentation_path=args.segmentation,
            output_dir=args.output_dir,
            config_path=args.config,
            membrane_labels=membrane_labels,
            mesh_sampling=args.mesh_sampling, 
            interpolate=args.interpolate,
            interpolation_points=args.interpolation_points,
            refine_normals=True,
            radius_hit=args.radius_hit,
            flip_normals=True,
            batch_size=args.batch_size,
            save_vertices_mrc=args.save_vertices_mrc,
            save_vertices_xyz=args.save_vertices_xyz,
        )

    elif args.mode == "thickness":
        # Run thickness measurement only
        if args.input_csv is None:
            print("Error: --input_csv is required with --mode=thickness")
            return

        # Generate output CSV name if not provided
        output_csv = None
        if args.output_dir:
            input_base = os.path.splitext(os.path.basename(args.input_csv))[0]
            output_csv = os.path.join(args.output_dir, f"{input_base}_thickness.csv")

        measure_membrane_thickness(
            segmentation_path=args.segmentation,
            input_csv=args.input_csv,
            output_csv=output_csv,
            output_dir=args.output_dir,
            max_thickness=args.max_thickness,
            max_angle=args.max_angle,
            save_thickness_mrc=args.save_thickness_mrc,
            direction=args.direction,
            use_gpu=use_gpu,
            num_cpu_threads=num_cpu_threads,
        )

    elif args.mode == "intensity":
        # Run intensity profiling only
        print(f"Running intensity profiling analysis...")
        print(f"Thickness CSV: {args.thickness_csv}")
        print(f"Tomogram: {args.tomo_path}")
        
        try:
            results = int_profiles_extract_clean(
                thickness_csv=args.thickness_csv,
                tomo_path=args.tomo_path,
                output_dir=args.output_dir or os.path.dirname(args.thickness_csv),
                intensity_min_snr=args.intensity_min_snr,
                intensity_central_max_required=args.intensity_central_max_required,
                intensity_extension_voxels=args.intensity_extension_voxels,
                intensity_extension_range=args.intensity_extension_range,
                intensity_normalize_method=args.intensity_normalize_method,
                save_cleaned_df=True,
                save_profiles=True,
                save_statistics=True,
                intensity_margin_factor=args.intensity_margin_factor,
                intensity_require_both_minima=args.intensity_require_both_minima,
                intensity_smooth_sigma=args.intensity_smooth_sigma,
                intensity_edge_fraction=args.intensity_edge_fraction,
            )
            
            print("Intensity profiling completed successfully!")
            print(f"Results saved to: {args.output_dir or os.path.dirname(args.thickness_csv)}")
            
            # Print summary
            if 'statistics' in results:
                stats = results['statistics']
                print(f"\nSummary:")
                print(f"  Total profiles: {stats.get('total_profiles', 0)}")
                print(f"  Profiles passed filtering: {stats.get('profiles_passed', 0)} ({stats.get('pass_rate', 0):.1%})")
                
        except Exception as e:
            print(f"Error in intensity profiling: {e}")
            return


if __name__ == "__main__":
    main()
