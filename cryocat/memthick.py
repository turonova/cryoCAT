import os
import sys
import time
import yaml
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import traceback

import numpy as np
import pandas as pd
import mrcfile

from skimage import measure
from scipy.sparse import lil_matrix
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import KDTree as ScipyKDTree

from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA

import numba
from numba import cuda
from numba import prange
import math

"""
Membrane Thickness Analysis Tool

This tool processes membrane segmentations from cryo-electron tomograms to measure bilayer
thickness.

Input Processing:
    1. Extract and refine the coordinates and orientations of surface points:
        - Instance segmentation inputs: membrane labels of interest can be specified using a dictionary or in a config.yaml file
        - Semantic segmentation inputs: considers all non-zero voxels as membrane

    2. Assign points to the two membrane surfaces based on their orientation

    3. Measure membrane thickness between points on the two surfaces as Euclidean distance in 3D using:
        - GPU-accelerated nearest neighbour search in the direction of the normals
        - (Alternatively) CPU-parallelization with Numba JIT compilation
        - 1-1 point match with a CPU vote

Outputs:
    - Vertices CSV: coordinates (in voxels and in nm) and orientations of the sampled surface points, including to which surface they've been assigned
    - Thickness CSV: membrane thickness measurements (in nm, scaled by input voxel size), including the IDs and positions of paired points from each surface
    - Statistics log: general statistics including mean/median thicknesses, interquartile range, histograms, number of points, and measurement coverage
    - (Optional) MRC files: coordinates of sampled surface points (voxel value = 1) to overlay with original membrane segmentations
    - (Optional) XYZ point clouds: coordinates of sampled surface points for visualization in external software (e.g., MeshLab)
    - (Optional) Thickness volumes: MRC files where voxel values correspond to measured membrane thicknesses (in nm) for each surface point

The tool can be run as a command-line application or imported as a module.
"""

#############################################
# Logging and Utility Functions
#############################################


def setup_logger(output_dir, name="MembraneThickness"):
    """Sets up logger for the analysis."""
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


def read_segmentation(segmentation_path, logger=None):
    """
    Read segmentation data from MRC file and get voxel size and origin.

    Parameters
    ----------
    segmentation_path : str
        Path to the MRC file
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    segmentation : ndarray
        Segmentation data
    voxel_size : float
        Voxel size in nm or angstroms
    origin : tuple
        Origin coordinates
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    log_msg(f"Reading segmentation from {segmentation_path}...")
    try:
        with mrcfile.mmap(segmentation_path, mode="r", permissive=True) as mrc:
            segmentation = mrc.data
            voxel_size = mrc.voxel_size.x / 10  # Convert to nm
            origin = (mrc.header.origin.x / 10, mrc.header.origin.y / 10, mrc.header.origin.z / 10)

            log_msg(f"Voxel size: {voxel_size:.4f} nm")
            log_msg(f"Origin: {origin}")

        return segmentation, voxel_size, origin
    except Exception as e:
        log_msg(f"Error reading MRC file: {e}")
        traceback.print_exc()
        return None, None, None


def save_vertices_mrc(data, output_path, voxel_size, origin=(0, 0, 0)):
    """
    Save vertices position as MRC file with correct voxel size and origin.

    Parameters
    ----------
    data : ndarray
        3D volume data
    output_path : str
        Path to save the MRC file
    voxel_size : float
        Voxel size in nm or angstroms
    origin : tuple
        Origin coordinates
    """
    with mrcfile.new(output_path, overwrite=True) as mrc:
        mrc.set_data(data.astype(np.float32))

        # Convert nm to Å
        mrc.voxel_size = voxel_size * 10 if isinstance(voxel_size, np.float32) else np.float32(voxel_size * 10)
        mrc.header.origin.x = origin[0] * 10
        mrc.header.origin.y = origin[1] * 10
        mrc.header.origin.z = origin[2] * 10

        # Add additional header information
        mrc.header.nxstart = 0
        mrc.header.nystart = 0
        mrc.header.nzstart = 0


#############################################
# Surface Processing Classes and Functions
#############################################


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

    def separate_bilayer_simple(self):
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
            return None, None

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
        return self.separate_bilayer_simple()


def is_surface_point(point, segmentation):
    """
    Check if a point is truly on the surface by verifying it has at least
    one neighbor that is not part of the segmentation.

    Parameters
    ----------
    point : array-like
        [x, y, z] coordinates of the point
    segmentation : ndarray
        Binary segmentation volume

    Returns
    -------
    bool
        True if the point is on the surface
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


def get_neighbor_surface_points(vertex, segmentation, include_edges=True):
    """
    Get neighboring surface points for a vertex.

    Parameters
    ----------
    vertex : array-like
        [x, y, z] coordinates of the vertex
    segmentation : ndarray
        Binary segmentation volume
    include_edges : bool
        Whether to include edge neighbors

    Returns
    -------
    points : list
        List of neighboring surface points
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


def interpolate_between_vertices(v1, v2, segmentation, num_points=1):
    """
    Interpolate between two vertices while ensuring points lie on the surface.

    Parameters
    ----------
    v1, v2 : array-like
        Vertices to interpolate between
    segmentation : ndarray
        Binary segmentation volume
    num_points : int
        Number of points to interpolate

    Returns
    -------
    points : list
        List of interpolated points
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


def interpolate_surface_points(
    vertices, normals, segmentation, interpolation_points=1, include_edges=True, logger=None
):
    """
    Increase density of surface points with strict surface constraints.

    Parameters
    ----------
    vertices : ndarray
        Original vertex coordinates
    normals : ndarray
        Normal vectors for original vertices
    segmentation : ndarray
        Binary segmentation volume
    interpolation_points : int
        Number of points to interpolate between adjacent vertices
    include_edges : bool
        Whether to include edge neighbors in addition to face neighbors
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    dense_vertices : ndarray
        Densified vertex coordinates
    dense_normals : ndarray
        Normal vectors for densified vertices
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    dense_vertices = []
    dense_normals = []
    processed_positions = set()

    # First verify all original vertices are on surface
    for i, vertex in enumerate(vertices):
        vertex = np.round(vertex).astype(int)
        vertex_tuple = tuple(vertex)

        # Only add if it's a valid surface point and not processed
        if vertex_tuple not in processed_positions and is_surface_point(vertex, segmentation):
            processed_positions.add(vertex_tuple)
            dense_vertices.append(vertex)
            dense_normals.append(normals[i])

            # Add surface-constrained neighbors
            neighbor_points = get_neighbor_surface_points(vertex, segmentation, include_edges=include_edges)

            for point in neighbor_points:
                point_tuple = tuple(point)
                if point_tuple not in processed_positions:
                    processed_positions.add(point_tuple)
                    dense_vertices.append(point)
                    dense_normals.append(normals[i])  # Use same normal as original vertex

        # Interpolate with next vertex if available
        if i < len(vertices) - 1:
            next_vertex = np.round(vertices[i + 1]).astype(int)
            interp_points = interpolate_between_vertices(vertex, next_vertex, segmentation, interpolation_points)

            for point in interp_points:
                point_tuple = tuple(point)
                if point_tuple not in processed_positions:
                    processed_positions.add(point_tuple)
                    dense_vertices.append(point)
                    # Interpolate normal vector
                    t = 0.5  # Could be more sophisticated
                    interp_normal = (1 - t) * normals[i] + t * normals[i + 1]
                    interp_normal /= np.linalg.norm(interp_normal)
                    dense_normals.append(interp_normal)

    dense_vertices = np.array(dense_vertices)
    dense_normals = np.array(dense_normals)

    log_msg("\nPoint interpolation details:")
    log_msg(f"Original vertices shape: {vertices.shape}")
    log_msg(f"Original normals shape: {normals.shape}")
    log_msg(f"Shape of vertices post interpolation: {dense_vertices.shape}")
    log_msg(f"Shape of normals post interpolation: {dense_normals.shape}")

    return dense_vertices, dense_normals


def refine_mesh_normals(vertices, initial_normals, radius_hit=10.0, batch_size=2000, flip_normals=True, logger=None):
    """
    Refine normals and separate surfaces.

    Parameters
    ----------
    vertices : ndarray
        Surface vertex coordinates
    initial_normals : ndarray
        Initial normal vectors
    radius_hit : float
        Search radius for neighbor points
    batch_size : int
        Batch size for processing
    flip_normals : bool
        Whether to flip normals inward
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    refined_normals : ndarray
        Refined normal vectors
    surface1_mask : ndarray
        Boolean mask for first surface
    surface2_mask : ndarray
        Boolean mask for second surface
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    voter = MeshNormalVoter(
        points=vertices, initial_normals=initial_normals, radius_hit=radius_hit, batch_size=batch_size, logger=logger
    )

    # First separate surfaces using initial marching cubes normals
    log_msg("\nSeparating surfaces using marching cubes orientation...")
    surface1_mask, surface2_mask = voter.separate_bilayer_simple()

    # Orient normals consistently
    log_msg("\nOrienting normals...")
    voter.orient_normals()

    # Refine using simple single-scale method
    log_msg(f"\nRefining normals using weighted average of neighbor normals...")
    voter.refine_normals()

    if surface1_mask is not None and surface2_mask is not None:
        log_msg(f"Surface 1: {np.sum(surface1_mask)} points")
        log_msg(f"Surface 2: {np.sum(surface2_mask)} points")

        refined_normals = -voter.refined_normals if flip_normals else voter.refined_normals
    else:
        log_msg("Could not separate surfaces!")
        refined_normals = -voter.refined_normals if flip_normals else voter.refined_normals
        surface1_mask = np.zeros(len(vertices), dtype=bool)
        surface2_mask = np.zeros(len(vertices), dtype=bool)

    return refined_normals, surface1_mask, surface2_mask


def verify_and_save_outputs(
    aligned_vertices,
    aligned_normals,
    vertex_volume,
    surface1_mask,
    surface2_mask,
    membrane_name,
    base_name,
    output_dir,
    voxel_size,
    origin,
    save_vertices_mrc=False,
    save_vertices_xyz=False,
    logger=None,
):
    """
    Verify and save all output files with additional checks.

    Parameters
    ----------
    aligned_vertices : ndarray
        Vertex coordinates
    aligned_normals : ndarray
        Normal vectors
    vertex_volume : ndarray
        Binary volume for vertices
    surface1_mask : ndarray
        Boolean mask for first surface
    surface2_mask : ndarray
        Boolean mask for second surface
    membrane_name : str
        Name of the membrane
    base_name : str
        Base name for output files
    output_dir : str
        Output directory
    voxel_size : float
        Voxel size for scaling
    origin : tuple
        Origin coordinates
    save_vertices_mrc : bool
        Whether to save MRC file with vertices position, e.g., for overlaying with the original segmentation
    save_vertices_xyz : bool
        Whether to save vertex coordinates as point cloud XYZ files, e.g., for visualizing in MeshLab
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    bool
        True if successful
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    log_msg(f"\nVerifying and saving outputs for {membrane_name}:")
    log_msg(f"Number of vertices to save: {len(aligned_vertices)}")
    log_msg(f"Shape of aligned_vertices: {aligned_vertices.shape}")
    log_msg(f"Shape of aligned_normals: {aligned_normals.shape}")
    log_msg(f"Surface 1 points: {np.sum(surface1_mask)}")
    log_msg(f"Surface 2 points: {np.sum(surface2_mask)}")

    try:
        # Scale vertices
        scaled_vertices = aligned_vertices * voxel_size + np.array(origin)

        # Save MRC file if requested
        if save_vertices_mrc:
            mrc_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices.mrc")
            with mrcfile.new(mrc_output, overwrite=True) as mrc:
                mrc.set_data(vertex_volume.astype(np.float32))
                mrc.voxel_size = voxel_size if isinstance(voxel_size, np.float32) else np.float32(voxel_size)
                mrc.header.origin.x = origin[0]
                mrc.header.origin.y = origin[1]
                mrc.header.origin.z = origin[2]
            log_msg(f"Saved MRC to {mrc_output}")

        # Save XYZ file if requested
        if save_vertices_xyz:
            xyz_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices.xyz")
            np.savetxt(xyz_output, scaled_vertices, fmt="%.6f", delimiter=" ")
            log_msg(f"Saved XYZ to {xyz_output}")

        # Save CSV with both unscaled and scaled coordinates, normals, and surface masks
        csv_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices_normals.csv")
        df = pd.DataFrame(
            {
                "x_voxel": aligned_vertices[:, 0],
                "y_voxel": aligned_vertices[:, 1],
                "z_voxel": aligned_vertices[:, 2],
                "x_physical": scaled_vertices[:, 0],  # in nm or A
                "y_physical": scaled_vertices[:, 1],
                "z_physical": scaled_vertices[:, 2],
                "normal_x": aligned_normals[:, 0],
                "normal_y": aligned_normals[:, 1],
                "normal_z": aligned_normals[:, 2],
                "surface1": surface1_mask,
                "surface2": surface2_mask,
            }
        )
        df.to_csv(csv_output, index=False)

        # Verify files
        if save_vertices_xyz:
            line_count = sum(1 for _ in open(xyz_output))
            log_msg(f"Saved XYZ file with {line_count} lines to {xyz_output}")
            if line_count != len(aligned_vertices):
                log_msg(
                    f"WARNING: XYZ file line count ({line_count}) doesn't match vertex count ({len(aligned_vertices)})"
                )

        df_verify = pd.read_csv(csv_output)
        log_msg(f"Saved CSV with {len(df_verify)} rows to {csv_output}")
        if len(df_verify) != len(aligned_vertices):
            log_msg(f"WARNING: CSV row count ({len(df_verify)}) doesn't match vertex count ({len(aligned_vertices)})")

    except Exception as e:
        log_msg(f"Error saving outputs: {e}")
        traceback.print_exc()

    return True


def extract_surface_points(segmentation, membrane_mask, mesh_sampling=1, logger=None):
    """
    Extract surface points and normals using marching cubes.

    Parameters
    ----------
    segmentation : ndarray
        Full segmentation volume
    membrane_mask : ndarray
        Binary mask for the membrane of interest
    mesh_sampling : int
        Step size for marching cubes
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    vertices : ndarray
        Vertex coordinates
    faces : ndarray
        Faces defining the mesh
    normals : ndarray
        Normal vectors
    """
    log_msg = lambda msg: logger.info(msg) if logger else print(msg)

    log_msg(f"Extracting surface points with step size {mesh_sampling}...")

    try:
        # Get vertices using marching cubes
        vertices, faces, normals, _ = measure.marching_cubes(membrane_mask, step_size=mesh_sampling)

        vertices_int = np.round(vertices).astype(int)

        # Create binary volume for vertices
        vertex_volume = np.zeros_like(membrane_mask)

        aligned_vertices = []
        aligned_normals = []

        # Initial vertex processing
        processed_positions = set()
        for i in range(len(vertices_int)):
            vertex = vertices_int[i]
            if (
                0 <= vertex[0] < membrane_mask.shape[0]
                and 0 <= vertex[1] < membrane_mask.shape[1]
                and 0 <= vertex[2] < membrane_mask.shape[2]
            ):
                pos_tuple = tuple(vertex)
                if pos_tuple not in processed_positions:
                    if is_surface_point(vertex, membrane_mask):
                        processed_positions.add(pos_tuple)
                        aligned_vertices.append(vertex)
                        aligned_normals.append(normals[i])
                        vertex_volume[vertex[0], vertex[1], vertex[2]] = 1

        aligned_vertices = np.array(aligned_vertices)
        aligned_normals = np.array(aligned_normals)

        log_msg(f"Extracted {len(aligned_vertices)} surface points")

        return aligned_vertices, aligned_normals, vertex_volume

    except Exception as e:
        log_msg(f"Error in surface extraction: {e}")
        traceback.print_exc()
        return None, None, None


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
    max_angle_degrees=3.0,
    direction="1to2",
    logger=None,
):
    """
    GPU thickness measurement ensuring one-to-one matching.
    Works with unscaled (voxel) coordinates and converts to physical units after measurement.

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
    thickness_results : ndarray
        Thickness measurements in physical units
    valid_mask : ndarray
        Boolean mask for valid measurements
    point_pairs : ndarray
        Indices of paired points
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


def generate_matching_statistics(
    thickness_results, valid_mask, point_pairs, points, surface1_mask, surface2_mask, voxel_size
):
    """
    Generate detailed matching statistics.

    Parameters
    ----------
    thickness_results : ndarray
        Thickness measurements in physical units
    valid_mask : ndarray
        Boolean mask for valid measurements
    point_pairs : ndarray
        Indices of paired points
    points : ndarray
        Vertex coordinates
    surface1_mask, surface2_mask : ndarray
        Boolean masks for each surface
    voxel_size : float
        Voxel size for scaling

    Returns
    -------
    stats : dict
        Dictionary of statistics
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


def save_matching_statistics(stats, output_path, logger=None):
    """
    Save matching statistics to a formatted log file.

    Parameters
    ----------
    stats : dict
        Dictionary of statistics
    output_path : str
        Path to save the statistics file
    logger : logging.Logger, optional
        Logger instance
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


def generate_thickness_volume(points, thickness_results, valid_mask, segmentation, voxel_size, point_pairs):
    """
    Generate a volume where voxel values represent membrane thickness at both surfaces.

    Parameters
    ----------
    points : ndarray
        Nx3 array of point coordinates in voxel space
    thickness_results : ndarray
        Array of thickness measurements in nm
    valid_mask : ndarray
        Boolean mask indicating valid measurements
    segmentation : ndarray
        Original segmentation volume to get dimensions
    voxel_size : float
        Voxel size in nm
    point_pairs : ndarray
        Array of indices matching points on surface 1 to points on surface 2

    Returns
    -------
    thickness_volume : ndarray
        3D volume with thickness values in nm
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


def save_thickness_volume(thickness_volume, output_path, voxel_size, origin=(0, 0, 0)):
    """
    Save thickness volume as MRC file with correct voxel size and origin.

    Parameters
    ----------
    thickness_volume : ndarray
        3D volume with thickness values in nm
    output_path : str
        Path to save the MRC file
    voxel_size : float
        Voxel size in nm
    origin : tuple
        Origin coordinates in nm
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
    max_angle_degrees=5.0,
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
# Main Pipeline Functions
#############################################


def process_membrane_segmentation(
    segmentation_path,
    output_dir=None,
    config_path=None,
    membrane_labels=None,
    mesh_sampling=1,
    interpolate=True,
    interpolation_points=1,
    refine_normals=True,
    radius_hit=10.0,
    flip_normals=True,
    batch_size=2000,
    save_vertices_mrc=False,
    save_vertices_xyz=False,
    logger=None,
):
    """
    Process a labeled membrane segmentation using marching cubes, optionally interpolate
    points and refine their normals before saving output files.

    Parameters
    ----------
    segmentation_path : str
        Path to the segmentation MRC file
    output_dir : str, optional
        Output directory
    config_path : str, optional
        Path to YAML configuration file
    membrane_labels : dict, optional
        Dictionary mapping membrane names to label values
    interpolate : bool
        Whether to interpolate surface points
    interpolation_points : int
        Number of points to interpolate between adjacent vertices
    refine_normals : bool
        Whether to refine normals
    radius_hit : float
        Search radius for neighbor points
    flip_normals : bool
        Whether to flip normals inward
    batch_size : int
        Batch size for processing
    save_vertices_mrc : bool
        Whether to save MRC file with vertices position, e.g., for overlaying with the original segmentation
    save_vertices_xyz : bool
        Whether to save vertex coordinates as point cloud XYZ files, e.g., for visualizing in MeshLab
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    output_files : dict
        Dictionary mapping membrane names to output file paths
    """
    # Initialize logger if not provided
    if logger is None:
        if output_dir is None:
            output_dir = os.path.dirname(segmentation_path)
        logger = setup_logger(output_dir)

    # Handle configuration
    if config_path is not None:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        membrane_labels = config["segmentation_values"]
        mesh_sampling = mesh_sampling
    else:
        if membrane_labels is None:
            membrane_labels = {"membrane": 1}
        mesh_sampling = mesh_sampling

    logger.info(f"Using mesh sampling step size: {mesh_sampling}")

    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(segmentation_path)
    os.makedirs(output_dir, exist_ok=True)

    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(segmentation_path))[0]

    # Read segmentation
    segmentation, voxel_size, origin = read_segmentation(segmentation_path, logger=logger)

    if segmentation is None:
        logger.error("Failed to read segmentation")
        return None

    output_files = {}

    # Process each membrane type
    for membrane_name, label_value in membrane_labels.items():
        logger.info(f"\nProcessing {membrane_name} (label {label_value})")

        # Create binary mask for this membrane
        membrane_mask = segmentation == label_value

        if not np.any(membrane_mask):
            logger.info(f"No voxels found for {membrane_name}")
            continue

        try:
            # Extract surface points and normals
            aligned_vertices, aligned_normals, vertex_volume = extract_surface_points(
                segmentation, membrane_mask, mesh_sampling=mesh_sampling, logger=logger
            )

            if aligned_vertices is None or len(aligned_vertices) == 0:
                logger.info(f"No surface points found for {membrane_name}")
                continue

            # Interpolate if requested
            if interpolate:
                logger.info(f"\nInterpolating surface points (before: {len(aligned_vertices)} vertices)")
                aligned_vertices, aligned_normals = interpolate_surface_points(
                    aligned_vertices,
                    aligned_normals,
                    membrane_mask,
                    interpolation_points=interpolation_points,
                    include_edges=True,
                    logger=logger,
                )
                logger.info(f"After point interpolation: {len(aligned_vertices)} vertices")

            # Refine normals if requested
            if refine_normals:
                logger.info(f"\nRefining normals with weighted-average method...")
                try:
                    refined_normals, surface1_mask, surface2_mask = refine_mesh_normals(
                        vertices=aligned_vertices,
                        initial_normals=aligned_normals,
                        radius_hit=radius_hit,
                        batch_size=batch_size,
                        flip_normals=flip_normals,
                        logger=logger,
                    )
                    # Update normals with refined ones
                    aligned_normals = refined_normals

                    if surface1_mask is not None and surface2_mask is not None:
                        logger.info(f"Successfully separated bilayer surfaces:")
                        logger.info(f"Surface 1: {np.sum(surface1_mask)} points")
                        logger.info(f"Surface 2: {np.sum(surface2_mask)} points")
                    else:
                        logger.warning("Could not separate bilayer surfaces")
                        surface1_mask = np.zeros(len(aligned_vertices), dtype=bool)
                        surface2_mask = np.zeros(len(aligned_vertices), dtype=bool)

                except Exception as e:
                    logger.error(f"Error during normal refinement: {str(e)}")
                    traceback.print_exc()
                    logger.info("Continuing with original normals...")
                    surface1_mask = np.zeros(len(aligned_vertices), dtype=bool)
                    surface2_mask = np.zeros(len(aligned_vertices), dtype=bool)

            # Update vertex volume
            vertex_volume = np.zeros_like(membrane_mask)
            for v in tqdm(aligned_vertices, desc="Updating vertex volume"):
                x, y, z = v.astype(int)
                if (
                    0 <= x < vertex_volume.shape[0]
                    and 0 <= y < vertex_volume.shape[1]
                    and 0 <= z < vertex_volume.shape[2]
                ):
                    vertex_volume[x, y, z] = 1

            logger.info(f"Final vertex count: {len(aligned_vertices)}")

            # Save outputs with updated normals
            if len(aligned_vertices) > 0:
                verify_and_save_outputs(
                    aligned_vertices,
                    aligned_normals,
                    vertex_volume,
                    surface1_mask,
                    surface2_mask,
                    membrane_name,
                    base_name,
                    output_dir,
                    voxel_size,
                    origin,
                    save_vertices_mrc=save_vertices_mrc,
                    save_vertices_xyz=save_vertices_xyz,
                    logger=logger,
                )

                csv_output = os.path.join(output_dir, f"{base_name}_{membrane_name}_vertices_normals.csv")
                output_files[membrane_name] = csv_output

        except Exception as e:
            logger.error(f"Error processing {membrane_name}: {e}")
            traceback.print_exc()

    return output_files


def measure_membrane_thickness(
    segmentation_path,
    input_csv,
    output_csv=None,
    output_dir=None,
    max_thickness=8.0,
    max_angle=3.0,
    save_thickness_mrc=False,
    direction="1to2",
    use_gpu=True,
    num_cpu_threads=None,
    logger=None,
):
    """
    Measure membrane thickness from a CSV file with vertices and normals.

    Parameters
    ----------
    segmentation_path : str
        Path to the segmentation MRC file
    input_csv : str
        Path to the input CSV file with vertices and normals
    output_csv : str, optional
        Path to save the output CSV file
    output_dir : str, optional
        Output directory
    max_thickness : float
        Maximum thickness in nm
    max_angle : float
        Maximum angle for cone search
    save_thickness_mrc : bool
        Whether to save thickness volume as MRC file
    direction : str
        Direction of thickness measurement, either "1to2" (surface1 to surface2)
        or "2to1" (surface2 to surface1)
    use_gpu : bool
        Whether to use GPU acceleration if available
    num_cpu_threads : int, optional
        Number of CPU threads to use if using CPU implementation
    logger : logging.Logger, optional
        Logger instance

    Returns
    -------
    output_csv : str
        Path to the output CSV file
    stats_file : str
        Path to the statistics file
    """
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
    segmentation, voxel_size, origin = read_segmentation(segmentation_path, logger=logger)

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

    # Save results to CSV
    df["thickness"] = thickness_results
    df["valid_measurement"] = valid_mask
    df["paired_point_idx"] = point_pairs

    logger.info(f"\nSaving results to {output_csv}")
    df.to_csv(output_csv, index=False)

    # Generate and save thickness volume if requested
    if save_thickness_mrc:
        volume_output = os.path.join(output_dir, f"{output_base}_volume.mrc")
        logger.info(f"Generating thickness volume...")

        thickness_volume = generate_thickness_volume(
            points, thickness_results, valid_mask, segmentation, voxel_size, point_pairs
        )

        logger.info(f"Saving thickness volume to {volume_output}")
        save_thickness_volume(thickness_volume, volume_output, voxel_size, origin)

        # Print volume statistics
        valid_thicknesses = thickness_volume[~np.isnan(thickness_volume)]
        logger.info(f"Thickness volume statistics:")
        logger.info(f"Number of voxels with thickness values: {len(valid_thicknesses)}")
        logger.info(f"Mean thickness: {np.mean(valid_thicknesses):.2f} nm")
        logger.info(f"Std deviation: {np.std(valid_thicknesses):.2f} nm")

    logger.info("Membrane thickness analysis complete!")

    return output_csv, stats_file


def run_full_pipeline(
    segmentation_path,
    output_dir=None,
    config_path=None,
    membrane_labels=None,
    interpolate=True,
    interpolation_points=1,
    refine_normals=True,
    radius_hit=10.0,
    flip_normals=True,
    max_thickness=8.0,
    max_angle=3.0,
    save_vertices_mrc=False,
    save_vertices_xyz=False,
    save_thickness_mrc=False,
    direction="1to2",
    batch_size=2000,
    use_gpu=True,
    num_cpu_threads=None,
):
    """
    Run the full membrane thickness analysis pipeline:
    1. Process membrane segmentation
    2. Measure membrane thickness

    Parameters
    ----------
    segmentation_path : str
        Path to the segmentation MRC file
    output_dir : str, optional
        Output directory
    config_path : str, optional
        Path to YAML configuration file
    membrane_labels : dict, optional
        Dictionary mapping membrane names to label values
    interpolate : bool
        Whether to interpolate surface points
    interpolation_points : int
        Number of points to interpolate between adjacent vertices
    refine_normals : bool
        Whether to refine normals
    radius_hit : float
        Search radius for neighbor points
    flip_normals : bool
        Whether to flip normals inward
    max_thickness : float
        Maximum thickness in nm
    max_angle : float
        Maximum angle for cone search
    save_vertices_mrc : bool
        Whether to save MRC file with vertices position, e.g., for overlaying with the original segmentation
    save_vertices_xyz : bool
        Whether to save vertex coordinates as point cloud XYZ files, e.g., for visualizing in MeshLab
    save_thickness_mrc : bool
        Whether to save the thickness measurements as MRC files, where the voxel values correspond to the measured membrane thicknesses (in nm) for a given surface point
    direction : str
        Direction of thickness measurement, either "1to2" (surface1 to surface2)
        or "2to1" (surface2 to surface1)
    batch_size : int
        Batch size for processing
    use_gpu : bool
        Whether to use GPU acceleration if available
    num_cpu_threads : int, optional
        Number of CPU threads to use if using CPU implementation

    Returns
    -------
    results : dict
        Dictionary with results for each membrane
    """
    # Set output directory
    if output_dir is None:
        output_dir = os.path.dirname(segmentation_path)
    os.makedirs(output_dir, exist_ok=True)

    # Initialize logger
    logger = setup_logger(output_dir)
    logger.info(f"Starting full membrane thickness analysis pipeline for {segmentation_path}")

    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(segmentation_path))[0]

    # Step 1: Process membrane segmentation
    logger.info("Step 1: Processing membrane segmentation")
    output_files = process_membrane_segmentation(
        segmentation_path=segmentation_path,
        output_dir=output_dir,
        config_path=config_path,
        membrane_labels=membrane_labels,
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

        except Exception as e:
            logger.error(f"Error measuring thickness for {membrane_name}: {e}")
            traceback.print_exc()

    logger.info("\nFull pipeline completed!")
    return results


#############################################
# Command Line Interface
#############################################


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Membrane Thickness Analysis Tool", formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
    parser.add_argument("--interpolate", action="store_true", help="Interpolate points for more uniform coverage")
    parser.add_argument(
        "--interpolation_points", type=int, default=1, help="Number of points to interpolate between adjacent vertices"
    )
    parser.add_argument("--refine_normals", action="store_true", help="Refine normals after running marching cubes")
    parser.add_argument("--flip_normals", action="store_true", help="Flip normals inward after refinement")
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
    parser.add_argument("--max_angle", type=float, default=3.0, help="Maximum angle for cone search")
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
        help="Save vertices posiitons in MRC format, e.g., for overlaying with the original segmentation",
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
        choices=["full", "surface", "thickness"],
        default="full",
        help="Processing mode: full pipeline, surface extraction only, or thickness measurement only",
    )
    parser.add_argument(
        "--input_csv", help="Input CSV for thickness measurement mode (only used with --mode=thickness)"
    )

    return parser.parse_args()


def main():
    """Main function for command line execution."""
    args = parse_arguments()

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
        # Run full pipeline
        results = run_full_pipeline(
            segmentation_path=args.segmentation,
            output_dir=args.output_dir,
            config_path=args.config,
            membrane_labels=membrane_labels,
            interpolate=args.interpolate,
            interpolation_points=args.interpolation_points,
            refine_normals=args.refine_normals,
            radius_hit=args.radius_hit,
            flip_normals=args.flip_normals,
            max_thickness=args.max_thickness,
            max_angle=args.max_angle,
            save_vertices_mrc=args.save_vertices_mrc,
            save_vertices_xyz=args.save_vertices_xyz,
            save_thickness_mrc=args.save_thickness_mrc,
            direction=args.direction,
            batch_size=args.batch_size,
            use_gpu=use_gpu,
            num_cpu_threads=num_cpu_threads,
        )

    elif args.mode == "surface":
        # Run surface extraction only
        process_membrane_segmentation(
            segmentation_path=args.segmentation,
            output_dir=args.output_dir,
            config_path=args.config,
            membrane_labels=membrane_labels,
            interpolate=args.interpolate,
            interpolation_points=args.interpolation_points,
            refine_normals=args.refine_normals,
            radius_hit=args.radius_hit,
            flip_normals=args.flip_normals,
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


if __name__ == "__main__":
    main()
