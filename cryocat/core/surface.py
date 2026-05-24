from __future__ import annotations
import numpy as np
import pandas as pd
import open3d as o3d

from abc import ABC, abstractmethod
from pathlib import Path
import copy
import warnings
from tqdm import tqdm
import multiprocessing as mp
from typing import Any, Callable

import skimage.measure
import scipy.ndimage
from scipy.spatial import KDTree
from scipy.spatial import cKDTree
from scipy.optimize import fsolve
from sklearn.decomposition import PCA

from cryocat.core import cryomap
from cryocat.utils import geom
from cryocat.core import cryomotl
from cryocat._types import PathOrStr


def _axis_aligned_bbox_from_input(bbox: o3d.geometry.AxisAlignedBoundingBox | dict[str, Any]) -> o3d.geometry.AxisAlignedBoundingBox:
    """Return an Open3D AxisAlignedBoundingBox from a dict or pass through existing bbox."""
    if isinstance(bbox, dict):
        return o3d.geometry.AxisAlignedBoundingBox(
            min_bound=bbox["min_bound"],
            max_bound=bbox["max_bound"],
        )
    return bbox

class Surface(ABC):
    """Common root for all surface representations (discrete and analytic)."""

    def __init__(self):
        #: Semantic units of vertex coordinates (e.g. ``'angstrom'``, ``'nanometer'``, ``'pixel'``).
        self.units = None
        #: Pixel/voxel spacing along X, Y, Z in the same units as vertices, shape ``(3,)``, when known.
        self.pixel_size = None

    @staticmethod
    def _coerce_pixel_size_array(pixel_size: float | np.ndarray) -> np.ndarray | None:
        """Normalize pixel/voxel spacing to a float ndarray of shape (3,)."""
        if pixel_size is None:
            return None
        arr = np.asarray(pixel_size, dtype=float).reshape(-1)
        if arr.size == 1:
            v = float(arr[0])
            return np.array([v, v, v], dtype=float)
        if arr.size != 3:
            raise ValueError(
                "pixel_size must be a scalar or a sequence of length 3 "
                f"(got {arr.size} values)"
            )
        return arr.copy()

    def inherit_coordinate_metadata(self, source: "Surface"):
        """Copy ``units`` and ``pixel_size`` from another surface."""
        self.units = getattr(source, "units", None)
        vs = getattr(source, "pixel_size", None)
        self.pixel_size = None if vs is None else self._coerce_pixel_size_array(vs)

    @abstractmethod
    def transform(self, transformation_matrix: np.ndarray) -> "DiscreteSurface":
        """Apply transformation matrix to the surface."""
        pass

    @abstractmethod
    def translate(self, translation_vector: np.ndarray) -> "DiscreteSurface":
        """Translate the surface."""
        pass

    @abstractmethod
    def rotate(self, rotation_matrix: np.ndarray) -> "DiscreteSurface":
        """Rotate the surface."""
        pass


class DiscreteSurface(Surface):
    """Abstract base class for discrete surface representations (mesh and point cloud)."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def _annotate_shortest_ray_hit(result: dict[str, Any]) -> dict[str, Any]:
        """Add global shortest-hit metadata to a ray-cast result (legacy ``return_shortest``)."""
        t_hit = np.asarray(result["t_hit"], dtype=np.float64)
        finite_hits = t_hit[np.isfinite(t_hit)]
        out = dict(result)
        if len(finite_hits) > 0:
            shortest_distance = float(np.min(finite_hits))
            shortest_indices = np.where(np.isclose(t_hit, shortest_distance))[0]
            out["shortest_distance"] = shortest_distance
            out["shortest_indices"] = shortest_indices
        else:
            out["shortest_distance"] = np.inf
            out["shortest_indices"] = np.array([], dtype=np.intp)
        return out

    @staticmethod
    def apply_one_hit_per_target_distance(result: dict[str, Any]) -> dict[str, Any]:
        """Keep one closest source per target point (``distance_to_pointcloud`` output)."""
        distances = np.asarray(result["distances"], dtype=np.float64)
        hit_mask = np.asarray(result["hit_mask"], dtype=bool)
        closest_indices = np.asarray(result["closest_indices"])
        n = len(distances)

        valid = hit_mask & (closest_indices >= 0)
        if not np.any(valid):
            return result

        winners: dict[int, int] = {}
        winner_dist: dict[int, float] = {}
        for src in np.where(valid)[0]:
            tgt = int(closest_indices[src])
            dist = float(distances[src])
            if tgt not in winners or dist < winner_dist[tgt]:
                winners[tgt] = int(src)
                winner_dist[tgt] = dist

        winner_sources = np.array(list(winners.values()), dtype=np.intp)
        new_hit_mask = np.zeros(n, dtype=bool)
        new_hit_mask[winner_sources] = True
        new_distances = np.full(n, np.inf, dtype=np.float64)
        new_distances[winner_sources] = distances[winner_sources]
        new_closest = np.full(n, -1, dtype=closest_indices.dtype)
        new_closest[winner_sources] = closest_indices[winner_sources]

        out = dict(result)
        out["distances"] = new_distances
        out["hit_mask"] = new_hit_mask
        out["closest_indices"] = new_closest

        if "closest_points" in result:
            closest_points = np.asarray(result["closest_points"])
            new_points = np.full_like(closest_points, np.nan)
            new_points[winner_sources] = closest_points[winner_sources]
            out["closest_points"] = new_points

        if "closest_normals" in result:
            closest_normals = np.asarray(result["closest_normals"])
            new_normals = np.full_like(closest_normals, np.nan)
            new_normals[winner_sources] = closest_normals[winner_sources]
            out["closest_normals"] = new_normals

        if "used_reverse_normals" in result:
            used_reverse = np.asarray(result["used_reverse_normals"])
            new_reverse = np.full(n, False, dtype=bool)
            new_reverse[winner_sources] = used_reverse[winner_sources]
            out["used_reverse_normals"] = new_reverse

        if "stats" in result:
            valid_d = new_distances[new_hit_mask]
            stats = dict(result["stats"])
            stats["n_hits"] = int(new_hit_mask.sum())
            stats["n_targets_with_hits"] = len(winners)
            stats["one_hit_per_target"] = True
            if len(valid_d) > 0:
                stats["min_distance"] = float(np.min(valid_d))
                stats["max_distance"] = float(np.max(valid_d))
                stats["mean_distance"] = float(np.mean(valid_d))
                stats["median_distance"] = float(np.median(valid_d))
                stats["std_distance"] = float(np.std(valid_d))
            out["stats"] = stats

        out["one_hit_per_target"] = True
        out["n_hits_per_target"] = len(winners)
        return out

    @staticmethod
    def ray_hit_orientations(
        surface,
        intersection_result: dict[str, Any],
        target_orientation: str | Callable,
    ) -> np.ndarray | None:
        """
        Orientation vectors at ray hit sites (subset of rays with finite t_hit).

        Supports Mesh (primitive normals / curvature) and OrientedPointCloud (hit normals).
        """
        hit_mask = np.isfinite(intersection_result["t_hit"])
        n_hits = int(hit_mask.sum())

        if n_hits == 0:
            return None

        if callable(target_orientation):
            try:
                primitive_ids = intersection_result.get("primitive_ids", None)
                hit_points = intersection_result.get("hit_points", None)
                if primitive_ids is None:
                    return None
                orientations = target_orientation(
                    surface, primitive_ids[hit_mask], hit_points[hit_mask] if hit_points is not None else None
                )
                if orientations.shape != (n_hits, 3):
                    raise ValueError(
                        f"Custom target_orientation must return (N, 3), got {orientations.shape}"
                    )
                return orientations
            except Exception as e:
                raise ValueError(f"Error in custom target_orientation: {e}") from e

        target_orientation = str(target_orientation).lower()

        if isinstance(surface, OrientedPointCloud):
            if target_orientation == "normal":
                hit_normals = intersection_result.get("hit_normals", None)
                if hit_normals is not None:
                    return hit_normals[hit_mask]
            return None

        if isinstance(surface, Mesh):
            if target_orientation == "normal":
                primitive_normals = intersection_result.get("primitive_normals", None)
                if primitive_normals is not None:
                    return primitive_normals[hit_mask]
                return None

            if target_orientation in ("principal_1", "principal_2"):
                if surface._principal_directions is None:
                    raise ValueError(
                        "Principal curvature directions not computed. Call mesh.compute_curvatures() first."
                    )
                primitive_ids = intersection_result.get("primitive_ids", None)
                if primitive_ids is None:
                    return None
                hit_primitive_ids = primitive_ids[hit_mask]
                hit_triangles = surface.faces[hit_primitive_ids]
                vertex_ids = hit_triangles[:, 0]
                if target_orientation == "principal_1":
                    return surface._principal_directions[vertex_ids, :, 0]
                return surface._principal_directions[vertex_ids, :, 1]

            raise ValueError(
                f"Unknown target_orientation '{target_orientation}' for Mesh. "
                "Use 'normal', 'principal_1', 'principal_2', or a callable."
            )

        return None

    @abstractmethod
    def get_vertices(self):
        """Return surface vertices."""
        pass

    @abstractmethod
    def get_normals(self):
        """Return surface normals."""
        pass

    @abstractmethod
    def oversample(self, **kwargs):
        """Oversample the surface."""
        pass

    @abstractmethod
    def compute_normals(self, **kwargs):
        """Compute or refine surface normals."""
        pass

    @abstractmethod
    def flip_normals(self, inplace: bool = True, **kwargs):
        """Reverse surface normal directions."""
        pass

    @abstractmethod
    def remove_nonfinite_vertices(self, inplace: bool = True, **kwargs):
        """Remove vertices/samples with NaN or infinite coordinates."""
        pass

    @abstractmethod
    def crop(self, bbox, inplace: bool = False):
        """Crop surface to axis-aligned bounding box."""
        pass

    @staticmethod
    def separate_surfaces(surface, threshold_angle: float = 90.0, reference_point: np.ndarray | None = None) -> np.ndarray:
        """
        Separate inner and outer surfaces based on normal direction relative to reference point.
        
        Works for both Mesh and OrientedPointCloud since both have vertices and normals.
        
        Parameters
        ----------
        surface : Mesh or OrientedPointCloud
            DiscreteSurface object with vertices and normals
        threshold_angle : float, default=90.0
            Angle threshold in degrees. Normals with angle < threshold to reference direction
            are considered inner (pointing toward reference).
        reference_point : np.ndarray (3,), optional
            Reference point for classification. If None, uses centroid of vertices.
            
        Returns
        -------
        vertex_labels : np.ndarray
            Array of shape (N,) where 0 = inner, 1 = outer
        """
        vertices = surface.get_vertices()
        normals = surface.get_normals()
        
        # Use provided reference point or compute centroid
        if reference_point is None:
            reference_point = np.mean(vertices, axis=0)
        else:
            reference_point = np.asarray(reference_point)
        
        # Vector from each vertex to reference point
        to_reference = reference_point - vertices
        to_reference_norm = np.linalg.norm(to_reference, axis=1, keepdims=True)
        to_reference_normalized = to_reference / (to_reference_norm + 1e-12)
        
        # Dot product between normals and direction to reference
        dot_products = np.sum(normals * to_reference_normalized, axis=1)
        
        # Convert to angle
        angles = np.degrees(np.arccos(np.clip(dot_products, -1, 1)))
        
        # Classify: inner if angle < threshold (normal points toward reference)
        vertex_labels = np.where(angles < threshold_angle, 0, 1)  # 0=inner, 1=outer
        
        return vertex_labels

    @staticmethod
    def refine_normals_from_arrays(
        points: np.ndarray,
        normals: np.ndarray,
        radius_hit: float = 3.0,
        batch_size: int = 2000,
        n_iter: int = 1,
        mask: np.ndarray | None = None,
        logger=None,
    ) -> np.ndarray:
        """
        Refine normals by Gaussian-weighted neighbor averaging with local dominant-direction alignment.

        Uses batched ``cKDTree.query_ball_point`` queries. When ``mask`` is provided, only vertices
        with ``mask[i]==True`` are updated; neighborhoods are restricted to vertices with ``mask[j]==True``.
        Rows with ``mask[i]==False`` keep the input normals from this call.

        Parameters
        ----------
        points : (N, 3) ndarray
            Vertex/sample coordinates.
        normals : (N, 3) ndarray
            Normal vectors aligned with rows of ``points``.
        radius_hit : float
            Ball radius for neighbor search (same units as ``points``).
        batch_size : int
            Batch size when querying neighborhoods.
        n_iter : int
            Number of refinement passes.
        mask : (N,) bool ndarray, optional
            If ``None``, all vertices participate as targets and neighbors.
        logger : logging.Logger, optional

        Returns
        -------
        refined_normals : (N, 3) ndarray
        """

        pts = np.asarray(points, dtype=np.float64)
        nrm_init = np.asarray(normals, dtype=np.float64)
        n_pts = pts.shape[0]
        if nrm_init.shape != (n_pts, 3):
            raise ValueError(
                f"normals must have shape (N, 3)==({n_pts}, 3), got {nrm_init.shape}"
            )
        if not np.isfinite(pts).all():
            n_bad = int((~np.isfinite(pts).all(axis=1)).sum())
            raise ValueError(
                f"Cannot refine normals: {n_bad} vertices/samples contain NaN or Inf "
                "coordinates. Call remove_nonfinite_vertices() first."
            )
        if not np.isfinite(nrm_init).all():
            n_bad = int((~np.isfinite(nrm_init).all(axis=1)).sum())
            raise ValueError(
                f"Cannot refine normals: {n_bad} normals contain NaN or Inf values. "
                "Recompute normals before refinement."
            )

        log = lambda m: logger.info(m) if logger else print(m)
        log(
            f"Refining normals (radius={radius_hit}, batch={batch_size}, "
            f"iter={n_iter}, mask={'all' if mask is None else f'{int(mask.sum())}/{len(mask)}'})"
        )

        if mask is None:
            eff_mask = np.ones(n_pts, dtype=bool)
        else:
            eff_mask = np.asarray(mask).astype(bool).reshape(-1)
            if eff_mask.shape[0] != n_pts:
                raise ValueError(
                    f"mask length {eff_mask.shape[0]} must match points {n_pts}"
                )

        normals_init = nrm_init.copy()
        refined_normals = normals_init.astype(np.float64).copy()

        sigma = radius_hit / 2.0
        tree = cKDTree(pts)

        target_indices = np.flatnonzero(eff_mask)
        full_mask = mask is None

        for it in range(n_iter):
            log(f"Iteration {it + 1}/{n_iter}")
            new_normals = refined_normals.copy()

            if not np.any(eff_mask):
                break

            for start_idx in tqdm(range(0, len(target_indices), batch_size)):
                end_idx = min(start_idx + batch_size, len(target_indices))
                batch_vertex_idx = target_indices[start_idx:end_idx]
                batch_points = pts[batch_vertex_idx]

                try:
                    neighbor_lists = tree.query_ball_point(batch_points, r=radius_hit, workers=-1)
                except TypeError:
                    neighbor_lists = tree.query_ball_point(batch_points, r=radius_hit)

                for local_i, neighbor_indices in enumerate(neighbor_lists):
                    point_idx = int(batch_vertex_idx[local_i])
                    nei = np.asarray(neighbor_indices, dtype=np.int64)
                    if nei.size <= 1:
                        new_normals[point_idx] = refined_normals[point_idx]
                        continue

                    if not full_mask:
                        nei = nei[eff_mask[nei]]
                    nei = nei[nei != point_idx]

                    if nei.size <= 1:
                        new_normals[point_idx] = refined_normals[point_idx]
                        continue

                    neighbor_points = pts[nei]
                    neighbor_normals = refined_normals[nei].copy()
                    neighbor_normals /= np.linalg.norm(neighbor_normals, axis=1, keepdims=True) + 1e-12

                    distances = np.linalg.norm(neighbor_points - pts[point_idx], axis=1)
                    weights = np.exp(-(distances ** 2) / (2 * sigma ** 2))

                    avg = np.average(neighbor_normals, weights=weights, axis=0)
                    signs = np.sign(np.dot(neighbor_normals, avg))
                    aligned = neighbor_normals * signs[:, np.newaxis]
                    avg_normal = np.average(aligned, weights=weights, axis=0)

                    nm = np.linalg.norm(avg_normal)
                    if nm < 1e-12:
                        avg_normal = refined_normals[point_idx]
                    else:
                        avg_normal /= nm

                    new_normals[point_idx] = avg_normal

            if full_mask:
                refined_normals = new_normals
            else:
                refined_normals = new_normals
                refined_normals[~eff_mask] = normals_init[~eff_mask]

        if not full_mask:
            refined_normals[~eff_mask] = normals_init[~eff_mask]

        log("Normal refinement complete.")
        return refined_normals

    def refine_normals(
        self,
        radius_hit: float = 3.0,
        batch_size: int = 2000,
        n_iter: int = 1,
        mask: np.ndarray | None = None,
        logger = None,
        inplace: bool = True,
    ):
        """
        Refine normals for this surface's current vertices/samples.

        Uses :meth:`refine_normals_from_arrays`. When ``inplace=False``, returns a deep copy with
        updated normals.
        """

        target = self if inplace else copy.deepcopy(self)
        verts = np.asarray(target.get_vertices(), dtype=np.float64)
        finite_vertices = np.isfinite(verts).all(axis=1)
        if not finite_vertices.all():
            n_bad = int((~finite_vertices).sum())
            print(f"Removing {n_bad} vertices/samples with NaN or Inf coordinates before normal refinement")
            if mask is not None:
                mask = np.asarray(mask, dtype=bool).reshape(-1)
                if len(mask) != len(finite_vertices):
                    raise ValueError(
                        f"mask length {len(mask)} must match vertices {len(finite_vertices)}"
                    )
                mask = mask[finite_vertices]
            target.remove_nonfinite_vertices(inplace=True, recompute_normals=True)

        norms = np.asarray(target.get_normals(), dtype=np.float64)
        if not np.isfinite(norms).all():
            if isinstance(target, Mesh):
                print("Recomputing mesh normals before refinement because existing normals contain NaN or Inf values")
                target.compute_normals()
                norms = np.asarray(target.get_normals(), dtype=np.float64)
            if not np.isfinite(norms).all():
                n_bad = int((~np.isfinite(norms).all(axis=1)).sum())
                raise ValueError(
                    f"Cannot refine normals: {n_bad} normals contain NaN or Inf values. "
                    "Recompute normals or remove affected samples first."
                )

        verts = np.asarray(target.get_vertices(), dtype=np.float64)
        refined = target.refine_normals_from_arrays(
            verts,
            norms,
            radius_hit=radius_hit,
            batch_size=batch_size,
            n_iter=n_iter,
            mask=mask,
            logger=logger,
        )

        target.normals = refined
        if hasattr(target, "_motl"):
            target._motl = None
        invalidate = getattr(target, "_invalidate_cache", None)
        if callable(invalidate):
            invalidate()
        return target

    def filter_by_normals(self, angle_threshold: float = 90.0, reference_normal: np.ndarray | None = None, inplace: bool = False) -> "DiscreteSurface" | None:
        """
        Remove vertices/points whose normal deviates more than ``angle_threshold`` degrees
        from the mean (or a supplied reference) normal direction.

        Parameters
        ----------
        angle_threshold : float, default=90.0
            Maximum allowed angle in degrees between a vertex normal and the reference.
        reference_normal : np.ndarray (3,), optional
            Reference direction. If None, the mean of all normals is used.
        inplace : bool, default=False
            If True, modify in place and return None. If False, return a new instance.

        Returns
        -------
        DiscreteSurface or None
        """
        vertices = self.get_vertices()
        normals = self.get_normals()

        if len(vertices) == 0:
            return None if inplace else copy.deepcopy(self)

        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)

        if reference_normal is None:
            reference_normal = np.mean(normals, axis=0)
        reference_normal = np.asarray(reference_normal, dtype=float)
        reference_normal = reference_normal / (np.linalg.norm(reference_normal) + 1e-12)

        dots = np.clip(np.dot(normals, reference_normal), -1.0, 1.0)
        keep_mask = np.degrees(np.arccos(np.abs(dots))) <= angle_threshold

        if inplace:
            self._apply_filter_mask(keep_mask)
            return None
        new_surface = copy.deepcopy(self)
        new_surface._apply_filter_mask(keep_mask)
        return new_surface
    
    def _apply_filter_mask(self, keep_mask):
        """Apply filtering mask to surface."""
        pass
    
    def filter_by_mask(self, mask: np.ndarray | str, transpose: bool = True, mask_origin: np.ndarray | None = None, mask_pixel_spacing: float | np.ndarray = 1.0, inplace: bool = False) -> "DiscreteSurface" | None:
        """
        Filter vertices/points based on 3D binary mask.

        Parameters
        ----------
        mask : np.ndarray or str
            3D binary mask array, or path to .mrc file
        transpose : bool, default=True
        mask_origin : array-like (3,), optional
            If None, assumes mask origin is at [0, 0, 0]
        mask_pixel_spacing : float or array-like (3,), default=1.0
            If array-like, should be [x, y, z] spacing.
        inplace : bool, default=False
        
        Returns
        -------
        DiscreteSurface instance or None
            New instance if inplace=False, else None
        """
        vertices = self.get_vertices()
        
        if len(vertices) == 0:
            if inplace:
                return None
            else:
                return copy.deepcopy(self)
        
        # Load mask if string path
        if isinstance(mask, str):
            mask = cryomap.read(mask, transpose = transpose)
        
        mask = np.asarray(mask)
        
        # Parse mask origin
        if mask_origin is None:
            mask_origin = np.array([0.0, 0.0, 0.0])
        else:
            mask_origin = np.asarray(mask_origin)
        
        # Parse pixel/voxel spacing
        if np.isscalar(mask_pixel_spacing):
            mask_pixel_spacing = np.array([mask_pixel_spacing, mask_pixel_spacing, mask_pixel_spacing])
        else:
            mask_pixel_spacing = np.asarray(mask_pixel_spacing)
        
        # Convert vertices to mask pixel/voxel coordinates
        vertices_pixel = (vertices - mask_origin) / mask_pixel_spacing
        
        # Determine which vertices are inside mask
        v_idx = np.round(vertices_pixel).astype(int)  # (N, 3)
        in_bounds = (
            (v_idx[:, 0] >= 0) & (v_idx[:, 0] < mask.shape[0]) &
            (v_idx[:, 1] >= 0) & (v_idx[:, 1] < mask.shape[1]) &
            (v_idx[:, 2] >= 0) & (v_idx[:, 2] < mask.shape[2])
        )
        keep_mask = np.zeros(len(vertices), dtype=bool)
        keep_mask[in_bounds] = mask[v_idx[in_bounds, 0], v_idx[in_bounds, 1], v_idx[in_bounds, 2]] > 0
        
        # Apply filter
        if inplace:
            self._apply_filter_mask(keep_mask)
            return None
        else:
            new_surface = copy.deepcopy(self)
            new_surface._apply_filter_mask(keep_mask)
            return new_surface

class Mesh(DiscreteSurface):
    """Base mesh class with triangular connectivity."""
    
    def __init__(self):
        super().__init__()
        self.vertices = None  # N x 3 array
        self.faces = None     # M x 3 array of triangle indices
        self.normals = None   # N x 3 array of vertex normals
        self._curvature_cache = None
        
        
        # Curvature attributes - initialized to None
        self._principal_curvatures = None      # Nx2 array [k1, k2]
        self._principal_directions = None      # Nx3x2 array [dir1, dir2]  
        self._mean_curvature = None           # N array (k1 + k2)/2
        self._gaussian_curvature = None       # N array k1 * k2
        self._curvature_tensors = None        # Nx2x2 array
        
        # Cached geometric properties for optimization
        self._edge_vectors = None
        self._edge_lengths = None
        self._face_areas = None
        self._face_normals_cached = None
        
        # Curvature computation parameters
        self._min_triangle_area = 1e-12
        self._lstsq_rcond = 1e-12
    
    def get_vertices(self):
        return self.vertices
    
    def get_normals(self):
        if self.normals is None:
            self.compute_normals()
        return self.normals

    def transform(self, transformation_matrix: np.ndarray) -> "Mesh":
        """Apply 4x4 transformation matrix to vertices and normals."""
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        
        # Transform vertices (homogeneous coordinates)
        vertices_homo = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
        transformed_vertices = (transformation_matrix @ vertices_homo.T).T
        self.vertices = transformed_vertices[:, :3]
        
        # Transform normals (only rotation part, no translation)
        if self.normals is not None:
            rotation_matrix = transformation_matrix[:3, :3]
            self.normals = (rotation_matrix @ self.normals.T).T
            # Renormalize
            self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)
        
        # Invalidate cached properties
        self._invalidate_cache()

    def translate(self, translation_vector: np.ndarray) -> "Mesh":
        """Translate vertices by given vector."""
        translation_vector = np.asarray(translation_vector)
        if translation_vector.shape != (3,):
            raise ValueError("Translation vector must be 3D")
        self.vertices += translation_vector
        
    def rotate(self, rotation_matrix: np.ndarray) -> "Mesh":
        """
        Apply 3x3 rotation matrix to vertices and normals.
        
        Parameters
        ----------
        rotation_matrix : array_like (3, 3)
            Rotation matrix (should be orthogonal)
        """
        rotation_matrix = np.asarray(rotation_matrix)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        
        return self.transform(T)

    def flip_normals(self, inplace: bool = True, flip_faces: bool = True) -> "Mesh" | None:
        """
        Reverse mesh normal directions.

        Parameters
        ----------
        inplace : bool, default=True
            If True, modify this mesh. If False, return a flipped copy.
        flip_faces : bool, default=True
            If True, also reverse triangle winding so normals recomputed from faces keep
            the flipped orientation.

        Returns
        -------
        Mesh or None
            New mesh if ``inplace=False``, otherwise ``None``.
        """
        target = self if inplace else copy.deepcopy(self)

        if target.normals is None:
            target.compute_normals()
        target.normals = -np.asarray(target.normals, dtype=float)

        if flip_faces:
            if target.faces is None:
                raise ValueError("Cannot flip mesh face winding without faces")
            target.faces = np.asarray(target.faces)[:, [0, 2, 1]]
            target._invalidate_neighbor_cache()

        target._invalidate_cache()
        if inplace:
            return None
        return target

    def remove_nonfinite_vertices(self, inplace: bool = True, recompute_normals: bool = True) -> "Mesh":
        """
        Remove mesh vertices with NaN or infinite coordinates and drop affected faces.

        Parameters
        ----------
        inplace : bool, default=True
            If True, modify this mesh. If False, return a repaired copy.
        recompute_normals : bool, default=True
            If True, recompute normals after removing vertices/faces.

        Returns
        -------
        Mesh
            The repaired mesh.
        """
        target = self if inplace else copy.deepcopy(self)
        if target.vertices is None:
            return target

        vertices = np.asarray(target.vertices)
        finite_mask = np.isfinite(vertices).all(axis=1)
        if finite_mask.all():
            return target

        n_removed = int((~finite_mask).sum())
        keep_indices = np.flatnonzero(finite_mask)

        if target.faces is None:
            target.vertices = vertices[keep_indices].copy()
            if target.normals is not None:
                target.normals = np.asarray(target.normals)[keep_indices].copy()
        else:
            index_map = np.full(len(vertices), -1, dtype=int)
            index_map[keep_indices] = np.arange(len(keep_indices), dtype=int)

            faces = np.asarray(target.faces)
            valid_faces_mask = finite_mask[faces].all(axis=1) if len(faces) else np.array([], dtype=bool)
            n_removed_faces = int((~valid_faces_mask).sum())

            target.vertices = vertices[keep_indices].copy()
            target.faces = index_map[faces[valid_faces_mask]].copy()
            if target.normals is not None:
                target.normals = np.asarray(target.normals)[keep_indices].copy()
            print(
                f"Removed {n_removed} non-finite vertices and {n_removed_faces} affected faces"
            )

        if recompute_normals and target.faces is not None and len(target.faces) > 0:
            target.compute_normals()

        target._invalidate_cache()
        target._invalidate_neighbor_cache()
        return target
    
    def _invalidate_cache(self):
        """Invalidate all cached geometric and curvature properties."""
        self._curvature_cache = None
        self._principal_curvatures = None
        self._principal_directions = None
        self._mean_curvature = None
        self._gaussian_curvature = None
        self._curvature_tensors = None
        self._edge_vectors = None
        self._edge_lengths = None
        self._face_areas = None
        self._face_normals_cached = None
        if hasattr(self, "_raycasting_scene"):
            delattr(self, "_raycasting_scene")

    def _invalidate_neighbor_cache(self):
        """
        Invalidate cached KDTree and centroids for triangle neighbors.
        Call this after modifying the mesh geometry.
        """
        if hasattr(self, '_centroid_kdtree'):
            delattr(self, '_centroid_kdtree')
        if hasattr(self, '_triangle_centroids'):
            delattr(self, '_triangle_centroids')

    def extract_submesh(self, triangle_ids: np.ndarray, preserve_curvatures: bool = True) -> "Mesh":
        """
        Return a new mesh containing only selected triangles.

        Per-vertex arrays (normals and, when requested, curvature fields) are copied for the
        vertices referenced by ``triangle_ids`` and faces are remapped to the new vertex order.
        """
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh must have vertices and faces")

        tri_ids = np.asarray(triangle_ids, dtype=int).reshape(-1)
        if tri_ids.size == 0:
            sub = Mesh()
            sub.vertices = np.empty((0, 3), dtype=float)
            sub.faces = np.empty((0, 3), dtype=np.int32)
            sub.inherit_coordinate_metadata(self)
            return sub

        n_faces = len(self.faces)
        valid = (tri_ids >= 0) & (tri_ids < n_faces)
        if not valid.all():
            warnings.warn(
                f"Ignoring {int((~valid).sum())} triangle ids outside [0, {n_faces})",
                UserWarning,
                stacklevel=2,
            )
        tri_ids = np.unique(tri_ids[valid])
        if tri_ids.size == 0:
            raise ValueError("No valid triangle ids provided")

        selected_faces = np.asarray(self.faces, dtype=np.int64)[tri_ids]
        unique_vertices, inverse = np.unique(selected_faces.ravel(), return_inverse=True)

        sub = Mesh()
        sub.vertices = np.asarray(self.vertices)[unique_vertices].copy()
        sub.faces = inverse.reshape(-1, 3).astype(np.int32)
        sub.inherit_coordinate_metadata(self)

        for attr in ("seg_shape", "seg_transposed"):
            if hasattr(self, attr):
                setattr(sub, attr, copy.deepcopy(getattr(self, attr)))

        if self.normals is not None:
            sub.normals = np.asarray(self.normals)[unique_vertices].copy()

        if preserve_curvatures:
            copied_curvatures = False
            if self._principal_curvatures is not None:
                sub._principal_curvatures = self._principal_curvatures[unique_vertices].copy()
                copied_curvatures = True
            if self._principal_directions is not None:
                sub._principal_directions = self._principal_directions[unique_vertices].copy()
                copied_curvatures = True
            if self._mean_curvature is not None:
                sub._mean_curvature = self._mean_curvature[unique_vertices].copy()
                copied_curvatures = True
            if self._gaussian_curvature is not None:
                sub._gaussian_curvature = self._gaussian_curvature[unique_vertices].copy()
                copied_curvatures = True
            if self._curvature_tensors is not None:
                sub._curvature_tensors = self._curvature_tensors[unique_vertices].copy()
                copied_curvatures = True
            if copied_curvatures:
                sub._curvature_cache = True

        sub._invalidate_neighbor_cache()
        return sub
    
    def compute_normals(self, **kwargs):
        """Compute vertex normals from face connectivity."""

        kwargs.pop('knn', None)
        kwargs.pop('orient_normals', None)
        kwargs.pop('tangent_plane_knn', None)
        kwargs.pop('inplace', None)
        kwargs.pop('refine', None)
        if kwargs:
            raise TypeError(f"Mesh.compute_normals got unexpected keywords: {list(kwargs)!r}")

        if self.faces is None:
            raise ValueError("Cannot compute normals without face connectivity")
        
        # Use Open3D for normal computation
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        o3d_mesh.compute_vertex_normals()
        self.normals = np.asarray(o3d_mesh.vertex_normals)

    def _get_raycasting_scene(self):
        """RaycastingScene (tensor triangles); invalidated with mesh edits via _invalidate_cache."""
        scene = getattr(self, "_raycasting_scene", None)
        if scene is None:
            o3d_mesh = self._to_open3d()
            mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh)
            self._raycasting_scene = o3d.t.geometry.RaycastingScene()
            self._raycasting_scene.add_triangles(mesh_t)
            scene = self._raycasting_scene
        return scene

    def distance_to_points(
        self,
        target: np.ndarray,
        compute_occupancy: bool = True,
        compute_signed: bool = False,
        return_closest_points: bool = False,
    ) -> dict[str, Any]:
        """
        Point-to-mesh distances using Open3D RaycastingScene (unsigned/signed distance, occupancy).
        """
        target = np.atleast_2d(np.asarray(target, dtype=np.float32))
        rs = self._get_raycasting_scene()
        result: dict[str, Any] = {"n_total": len(target)}

        if return_closest_points or compute_signed:
            closest_result = rs.compute_closest_points(target)
            result["closest_points"] = closest_result["points"].numpy()
            result["primitive_ids"] = closest_result["primitive_ids"].numpy()
            distances_from_closest = np.linalg.norm(
                target - result["closest_points"], axis=1
            )

        if compute_signed:
            distances = rs.compute_signed_distance(target)
            result["distance_type"] = "signed"
            result["distances"] = distances.numpy()
            compute_occupancy = True
            if return_closest_points:
                result["closest_distances"] = distances_from_closest
        else:
            distances = rs.compute_distance(target)
            result["distance_type"] = "unsigned"
            result["distances"] = distances.numpy()
            if return_closest_points:
                result["closest_distances"] = result["distances"]

        if compute_occupancy:
            if compute_signed:
                occupancy = (result["distances"] < 0).astype(np.float32)
            else:
                occupancy = rs.compute_occupancy(target).numpy()
            result["occupancy"] = occupancy
            result["inside_mask"] = occupancy > 0.5
            result["outside_mask"] = occupancy < 0.5
            result["n_inside"] = int(np.sum(result["inside_mask"]))
            result["n_outside"] = int(np.sum(result["outside_mask"]))

        return result

    def cast_rays(
        self,
        rays: np.ndarray,
        one_hit_per_target: bool = False,
    ) -> dict[str, Any]:
        """Ray-mesh intersection using cached RaycastingScene."""
        rays = np.atleast_2d(np.asarray(rays, dtype=np.float32))
        scene = self._get_raycasting_scene()
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        cast = scene.cast_rays(rays_tensor)

        origins = rays[:, :3]
        directions = rays[:, 3:]
        t_param = cast["t_hit"].numpy()

        direction_magnitudes = np.linalg.norm(directions, axis=1)
        t_hit = t_param * direction_magnitudes

        hit_points = np.full((len(rays), 3), np.nan)
        hit_mask = np.isfinite(t_param)
        hit_points[hit_mask] = origins[hit_mask] + directions[hit_mask] * t_param[hit_mask, np.newaxis]

        result_dict = {
            "t_hit": t_hit,
            "geometry_ids": cast["geometry_ids"].numpy(),
            "primitive_ids": cast["primitive_ids"].numpy(),
            "primitive_normals": cast["primitive_normals"].numpy(),
            "primitive_uvs": cast["primitive_uvs"].numpy(),
            "hit_points": hit_points,
        }

        if one_hit_per_target and len(rays) > 1:
            result_dict = DiscreteSurface._annotate_shortest_ray_hit(result_dict)

        return result_dict

    def distance_to_pointcloud(
        self,
        target: "OrientedPointCloud",
        ray_length: float | None = None,
        max_distance: float | None = None,
        reverse_normals: bool = False,
        bidirectional: bool = False,
        one_hit_per_target: bool = False,
    ) -> dict[str, Any]:
        """Cast rays along mesh normals and intersect a target point cloud."""
        if not isinstance(target, OrientedPointCloud):
            raise TypeError(
                f"target must be OrientedPointCloud, got {type(target).__name__}"
            )
        if target.vertices is None or len(target.vertices) == 0:
            raise ValueError("target must have vertices")

        if ray_length is None:
            effective_ray_length = max_distance if max_distance is not None else 1e10
        else:
            effective_ray_length = ray_length

        pts = np.asarray(self.vertices, dtype=np.float64)
        normals = self.get_normals()
        if normals.ndim == 1:
            normals = normals.reshape(-1, 3)
        nm = np.linalg.norm(normals, axis=1, keepdims=True)
        normals_normalized = normals / (nm + 1e-10)

        def _cast_with_normals(normals_for_rays):
            ray_directions = normals_for_rays * effective_ray_length
            rays = np.concatenate([pts, ray_directions], axis=1).astype(np.float32)
            result = target.cast_rays(
                rays, knn_radius=effective_ray_length * 0.1
            )

            t_hit = result["t_hit"]
            hit_mask = t_hit < effective_ray_length
            if max_distance is not None and max_distance != effective_ray_length:
                hit_mask = hit_mask & (t_hit <= max_distance)

            distances = np.where(hit_mask, t_hit, np.inf)
            point_indices = np.where(hit_mask, result["primitive_ids"], -1)
            output = {
                "distances": distances,
                "closest_points": result["hit_points"],
                "closest_indices": point_indices,
                "hit_mask": hit_mask,
            }

            hn = result.get("hit_normals")
            if hn is not None:
                output["closest_normals"] = np.where(
                    hit_mask[:, np.newaxis], hn, np.nan
                )
            return output

        primary_normals = -normals_normalized if reverse_normals else normals_normalized
        output = _cast_with_normals(primary_normals)

        if bidirectional:
            opposite = _cast_with_normals(-primary_normals)
            use_opposite = opposite["distances"] < output["distances"]
            output["distances"] = np.where(
                use_opposite, opposite["distances"], output["distances"]
            )
            output["closest_indices"] = np.where(
                use_opposite, opposite["closest_indices"], output["closest_indices"]
            )
            output["hit_mask"] = np.isfinite(output["distances"])
            output["closest_points"] = np.where(
                use_opposite[:, np.newaxis],
                opposite["closest_points"],
                output["closest_points"],
            )
            if "closest_normals" in output and "closest_normals" in opposite:
                output["closest_normals"] = np.where(
                    use_opposite[:, np.newaxis],
                    opposite["closest_normals"],
                    output["closest_normals"],
                )
            output["used_reverse_normals"] = np.where(
                use_opposite, not reverse_normals, reverse_normals
            )

        if one_hit_per_target:
            output = DiscreteSurface.apply_one_hit_per_target_distance(output)

        hit_mask = output["hit_mask"]
        valid_distances = output["distances"][hit_mask]
        output["stats"] = {
            "n_source": len(pts),
            "n_target": len(target.vertices),
            "n_hits": int(hit_mask.sum()),
            "min_distance": valid_distances.min() if len(valid_distances) > 0 else np.inf,
            "max_distance": valid_distances.max() if len(valid_distances) > 0 else np.inf,
            "mean_distance": valid_distances.mean() if len(valid_distances) > 0 else np.nan,
            "median_distance": np.median(valid_distances) if len(valid_distances) > 0 else np.nan,
            "std_distance": valid_distances.std() if len(valid_distances) > 0 else np.nan,
            "bidirectional": bool(bidirectional),
            "one_hit_per_target": bool(one_hit_per_target),
        }

        print(f"Ray casting: {np.sum(hit_mask)}/{len(pts)} mesh vertices intersect with point cloud")
        return output

    @classmethod
    def read(cls, input_path: PathOrStr, units: str | None = None) -> "Mesh":
        """
        Load triangle mesh directly from file (ply, obj, stl, off).
        
        Parameters
        ----------
        input_path : str or Path
            Path to triangle mesh file
        units : str, optional
            Coordinate units for the mesh. Common values:
            - 'angstrom' or 'A': Angstroms
            - 'nanometer' or 'nm': nanometers
            - 'pixel' or 'pix': pixels
            If None, units remain unspecified (you should set them later!)
            
        Returns
        -------
        Mesh
            Loaded triangle mesh instance

        """
        input_path = str(input_path)  # Convert Path objects to string

        print(f"Loading triangle mesh: {input_path}")
        
        # Load with Open3D
        o3d_mesh = o3d.io.read_triangle_mesh(input_path)
        
        if len(o3d_mesh.vertices) == 0:
            raise ValueError(f"Failed to load mesh from {input_path}. "
                        "Check if file exists and contains triangle mesh data.")
        
        if len(o3d_mesh.triangles) == 0:
            raise ValueError(f"File {input_path} contains vertices but no triangles. "
                        "Use PointCloudMesh.from_ply() for point cloud data.")

        print(f"Loaded: {len(o3d_mesh.vertices)} vertices, {len(o3d_mesh.triangles)} faces")
        
        # Create mesh instance
        mesh = cls()
        mesh.vertices = np.asarray(o3d_mesh.vertices)
        mesh.faces = np.asarray(o3d_mesh.triangles)
        mesh.units = units
        
        # Print unit warning
        if units is None:
            print("\nCoordinate units not specified!")
            print("   Set with: mesh.units = 'nm'  (or 'angstrom', 'pixel', etc.)")
            print("   This determines curvature units (e.g., nm⁻¹ for nanometers)")
        else:
            print(f"Coordinate units: {units}")
            unit_info = Mesh._get_curvature_unit_str(units)
            print(f"  → Curvatures will be in: {unit_info}")
        
        return mesh

    @staticmethod
    def _decode_field_scalar(field_array):
        """Return a Python string from pyvista field_data scalar/length-1 array."""
        if field_array is None or len(field_array) == 0:
            return None
        item = np.asarray(field_array).flat[0]
        if isinstance(item, (bytes, np.bytes_)):
            return bytes(item).decode("utf-8", errors="replace")
        return str(item)
    
    def smooth(self, iterations: int = 1, recompute_normals: bool = True, repair_nonfinite: bool = True) -> "Mesh":
        """Smooth the mesh using the Taubin filter to prevent shrinkage of the mesh when using Laplacian filtering."""
        if repair_nonfinite:
            self.remove_nonfinite_vertices(inplace=True, recompute_normals=False)
        elif self.vertices is not None and not np.isfinite(self.vertices).all():
            raise ValueError(
                "Cannot smooth mesh: vertices contain NaN or Inf values. "
                "Call remove_nonfinite_vertices() first or set repair_nonfinite=True."
            )

        o3d_mesh = self._to_open3d()
        o3d_mesh = o3d_mesh.filter_smooth_taubin(iterations)
        self._from_open3d(o3d_mesh)

        if self.vertices is not None and not np.isfinite(self.vertices).all():
            if repair_nonfinite:
                self.remove_nonfinite_vertices(inplace=True, recompute_normals=False)
            else:
                raise ValueError(
                    "Mesh smoothing produced NaN or Inf vertex coordinates. "
                    "Set repair_nonfinite=True to remove affected vertices/faces."
                )

        if recompute_normals:
            self.compute_normals()
        self._invalidate_cache()

    def assess_mesh(self, verbose: bool = True, check_topology: bool = False) -> dict:
        """
        Assess mesh properties.

        Geometry metrics (edge lengths, areas, aspect ratios) are always computed
        and are fast for any mesh size.

        Topology checks (manifoldness, watertightness, Euler characteristic) require
        building a full half-edge table via Open3D and can be very slow on large meshes
        (> ~100k faces). Enable them explicitly with ``check_topology=True``.

        Parameters
        ----------
        verbose : bool, default=True
            Print a human-readable summary.
        check_topology : bool, default=False
            Run Open3D topology checks. Slow on large meshes.
        """
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh must have vertices and faces for assessment")

        # Ensure geometric properties are computed
        self.compute_mesh_properties()

        # Topology checks (opt-in — slow on large meshes)
        if check_topology:
            o3d_mesh = self._to_open3d()
            non_manifold_edges = o3d_mesh.get_non_manifold_edges(allow_boundary_edges=True)
            non_manifold_vertices = o3d_mesh.get_non_manifold_vertices()
            num_non_manifold_edges = len(non_manifold_edges)
            num_non_manifold_vertices = len(non_manifold_vertices)
            is_edge_manifold = (num_non_manifold_edges == 0)
            is_vertex_manifold = (num_non_manifold_vertices == 0)
            is_watertight = o3d_mesh.is_watertight()
            euler_char = o3d_mesh.euler_poincare_characteristic()
        else:
            num_non_manifold_edges = None
            num_non_manifold_vertices = None
            is_edge_manifold = None
            is_vertex_manifold = None
            is_watertight = None
            euler_char = None
        
        # Get edge vectors and compute metrics
        e0, e1, e2 = self._edge_vectors
        de0, de1, de2 = self._edge_lengths
        face_areas = self._face_areas
        
        # Edge length statistics
        all_edge_lengths = np.concatenate([de0, de1, de2])
        min_edge = np.min(all_edge_lengths)
        max_edge = np.max(all_edge_lengths)
        mean_edge = np.mean(all_edge_lengths)
        edge_length_ratio = max_edge / (min_edge + 1e-12)
        
        # Triangle area statistics
        min_area = np.min(face_areas)
        max_area = np.max(face_areas)
        mean_area = np.mean(face_areas)
        area_ratio = max_area / (min_area + 1e-12)
        
        # Triangle aspect ratios
        perimeters = de0 + de1 + de2
        aspect_ratios = perimeters**2 / (12 * np.sqrt(3) * face_areas + 1e-12)
        mean_aspect_ratio = np.mean(aspect_ratios)
        max_aspect_ratio = np.max(aspect_ratios)
        
        # Count problematic triangles
        degenerate_triangles = np.sum(face_areas < self._min_triangle_area)
        high_aspect_triangles = np.sum(aspect_ratios > 10)
        
        quality_metrics = {
          # Topology
            'is_edge_manifold': is_edge_manifold,
            'is_vertex_manifold': is_vertex_manifold,
            'num_non_manifold_edges': num_non_manifold_edges,
            'num_non_manifold_vertices': num_non_manifold_vertices,
            'is_watertight': is_watertight,
            'euler_characteristic': euler_char,
            
            # Geometry
            'vertices': len(self.vertices),
            'faces': len(self.faces),
            'edge_length_ratio': edge_length_ratio,
            'area_ratio': area_ratio,
            'mean_aspect_ratio': mean_aspect_ratio,
            'max_aspect_ratio': max_aspect_ratio,
            'degenerate_triangles': degenerate_triangles,
            'high_aspect_triangles': high_aspect_triangles,
            'min_edge_length': min_edge,
            'max_edge_length': max_edge,
            'mean_edge_length': mean_edge,
            'min_area': min_area,
            'max_area': max_area,
            'mean_area': mean_area
        }
        
        if verbose:
            print("Topology:")
            if check_topology:
                print(f"Non-manifold edges:     {num_non_manifold_edges:,}")
                print(f"Non-manifold vertices:  {num_non_manifold_vertices:,}")
                print(f"Watertight:             {'Yes' if is_watertight else 'No, has boundary edges'}")
                print(f"Euler χ:                {euler_char:,}")
            else:
                print("  (skipped — pass check_topology=True to enable; slow on large meshes)")

            print("Geometry:")
            print(f"Vertices: {quality_metrics['vertices']}, Faces: {quality_metrics['faces']}")
            
            print(f"\nEdges:")
            print(f"  Length range: [{min_edge:.2e}, {max_edge:.2e}]")
            print(f"  Mean length: {mean_edge:.2e}")
            print(f"  Length ratio: {edge_length_ratio:.1f}")
            
            print(f"\nTriangles:")
            print(f"  Area range: [{min_area:.2e}, {max_area:.2e}]")
            print(f"  Mean area: {mean_area:.2e}")
            print(f"  Area ratio: {area_ratio:.1f}")
            print(f"  Mean aspect ratio: {mean_aspect_ratio:.2f}")
            print(f"  Max aspect ratio: {max_aspect_ratio:.1f}")
            print(f"  Degenerate triangles: {degenerate_triangles}")
            print(f"  High aspect ratio triangles: {high_aspect_triangles}")
        
        return quality_metrics

    def cleanup_mesh(
        self,
        simplify_mesh: bool = False,
        target_number_of_triangles: int = 500000,
        recompute_normals: bool = True,
    ) -> "Mesh":
        """
        Repair mesh topology and optionally simplify it.

        Normals are recomputed by default after cleanup because removing vertices,
        triangles, or simplifying connectivity changes the local mesh geometry.
        Normal refinement is intentionally separate; call ``refine_normals`` after
        cleanup/smoothing when neighborhood-smoothed normals are needed.
        """
        self.remove_nonfinite_vertices(inplace=True, recompute_normals=False)
        o3d_mesh = self._to_open3d()
        
        o3d_mesh.remove_duplicated_vertices()
        o3d_mesh.remove_duplicated_triangles()
        o3d_mesh.remove_degenerate_triangles()
        o3d_mesh.remove_unreferenced_vertices()

        if simplify_mesh:
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles)
        
        self._from_open3d(o3d_mesh)
        self.remove_nonfinite_vertices(inplace=True, recompute_normals=False)
        
        if recompute_normals:
            print("Recomputing normals after mesh cleanup...")
            self.compute_normals()
        else:
            self.normals = None  # Force lazy recomputation on next get_normals()
        
        self._invalidate_cache()
        print(f"Mesh after cleaning: {len(self.vertices)} vertices, {len(self.faces)} faces")
        return self

    def remove_disconnected_components(self, min_component_size: int = 100, logger: logging.Logger | None = None) -> "Mesh":
        """Remove small disconnected components that are likely artifacts."""
        log_msg = lambda msg: logger.info(msg) if logger else print(msg)
        
        if self.vertices is None or self.faces is None:
            return self
            
        log_msg(f"Removing disconnected components smaller than {min_component_size} vertices...")
        
        try:
            import networkx as nx
        except ImportError:
            log_msg("NetworkX not available - skipping component removal")
            return self
        
        # Build graph from face connectivity
        G = nx.Graph()
        for face in self.faces:
            for i in range(3):
                for j in range(i+1, 3):
                    G.add_edge(face[i], face[j])
        
        # Find connected components
        components = list(nx.connected_components(G))
        log_msg(f"Found {len(components)} connected components")
        
        # Keep only large components
        vertices_to_keep = set()
        large_components = 0
        for component in components:
            if len(component) >= min_component_size:
                vertices_to_keep.update(component)
                large_components += 1
        
        removed_components = len(components) - large_components
        if removed_components > 0:
            log_msg(f"Removing {removed_components} small components")
            self._filter_by_vertex_set(vertices_to_keep)
        else:
            log_msg("No small components to remove")
        
        return self

    def remove_boundary_vertices(self, erosion_layers: int = 1, min_edge_faces: int = 2) -> "Mesh":
        """
        Remove boundary vertices and optionally erode inward to clean mesh edges.
        
        This method identifies and removes vertices at mesh boundaries where curvature
        artifacts are common. Useful for cleaning segmentation-derived meshes.
        
        Parameters
        ----------
        erosion_layers : int, default=1
            Number of erosion iterations. Each layer removes:
            - Layer 1: Vertices on boundary edges (edges with < min_edge_faces faces)
            - Layer 2+: Neighbors of previously removed vertices
            Higher values remove more material from edges.
        min_edge_faces : int, default=2
            Minimum number of faces an edge should belong to.
            - 2: Remove only true boundary edges (most conservative)
            - 3: Also remove edges with poor connectivity (more aggressive)
        
        Returns
        -------
        Mesh
        """
        if self.vertices is None or self.faces is None:
            print("Warning: No mesh to clean")
            return self
        
        print(f"Removing boundary vertices (erosion_layers={erosion_layers}, min_edge_faces={min_edge_faces})...")
        original_n_vertices = len(self.vertices)
        
        # Step 1: Build edge-to-face map
        # An edge is represented as a sorted tuple of two vertex indices
        edge_to_faces = {}
        
        for face_idx, face in enumerate(self.faces):
            # Each triangle has 3 edges
            edges = [
                tuple(sorted([face[0], face[1]])),  # edge between vertex 0 and 1
                tuple(sorted([face[1], face[2]])),  # edge between vertex 1 and 2
                tuple(sorted([face[2], face[0]])),  # edge between vertex 2 and 0
            ]
            
            for edge in edges:
                if edge not in edge_to_faces:
                    edge_to_faces[edge] = []
                edge_to_faces[edge].append(face_idx)
        
        # Step 2: Identify boundary edges
        # In a closed manifold mesh, every edge belongs to exactly 2 faces
        # Boundary edges belong to only 1 face (or fewer than min_edge_faces)
        boundary_edges = {
            edge for edge, faces in edge_to_faces.items() 
            if len(faces) < min_edge_faces
        }
        
        print(f"  Found {len(boundary_edges)} boundary edges")
        
        if len(boundary_edges) == 0:
            print("  No boundary edges found - mesh may be closed")
            return self
        
        # Step 3: Mark boundary vertices (vertices that belong to boundary edges)
        vertices_to_remove = set()
        for edge in boundary_edges:
            vertices_to_remove.add(edge[0])
            vertices_to_remove.add(edge[1])
        
        print(f"  Layer 1: Marked {len(vertices_to_remove)} boundary vertices")
        
        # Step 4: Additional erosion layers (optional)
        # For each additional layer, mark neighbors of already-marked vertices
        if erosion_layers > 1:
            # Build vertex-to-vertex connectivity map from faces
            vertex_neighbors = {i: set() for i in range(len(self.vertices))}
            for face in self.faces:
                for i in range(3):
                    for j in range(3):
                        if i != j:
                            vertex_neighbors[face[i]].add(face[j])
            
            # Iteratively expand the removal set
            for layer in range(2, erosion_layers + 1):
                new_vertices_to_remove = set()
                for vertex_idx in vertices_to_remove:
                    # Add all neighbors of this vertex
                    new_vertices_to_remove.update(vertex_neighbors[vertex_idx])
                
                # Update the removal set
                before_count = len(vertices_to_remove)
                vertices_to_remove.update(new_vertices_to_remove)
                after_count = len(vertices_to_remove)
                
                print(f"  Layer {layer}: Marked {after_count - before_count} additional vertices")
        
        # Step 5: Filter mesh to keep only non-boundary vertices
        vertices_to_keep = set(range(len(self.vertices))) - vertices_to_remove
        
        if len(vertices_to_keep) == 0:
            print("  Warning: Would remove all vertices! Aborting.")
            return self
        
        self._filter_by_vertex_set(vertices_to_keep)
        
        removed_count = original_n_vertices - len(self.vertices)
        removed_pct = 100 * removed_count / original_n_vertices
        print(f"  Removed {removed_count} vertices ({removed_pct:.1f}%)")
        print(f"  Mesh now has {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        # Invalidate curvatures since geometry changed
        self._invalidate_cache()
        
        return self

    def _filter_by_vertex_set(self, vertices_to_keep: set, recompute_normals: bool = True):
        """Filter mesh to keep only specified vertices."""
        vertices_to_keep = set(vertices_to_keep)  # Ensure it's a set for O(1) lookup
        
        if len(vertices_to_keep) == len(self.vertices):
            return  # Nothing to filter
        
        # Create vertex mapping for face updates
        vertex_mapping = {}
        new_vertices = []
        new_normals = [] if self.normals is not None else None
        
        new_idx = 0
        for old_idx in range(len(self.vertices)):
            if old_idx in vertices_to_keep:
                vertex_mapping[old_idx] = new_idx
                new_vertices.append(self.vertices[old_idx])
                if self.normals is not None:
                    new_normals.append(self.normals[old_idx])
                new_idx += 1
        
        # Update vertices and normals
        self.vertices = np.array(new_vertices)
        if self.normals is not None:
            self.normals = np.array(new_normals)
        
        # Update faces - only keep faces where all vertices are in the kept set
        valid_faces = []
        for face in self.faces:
            if all(v in vertex_mapping for v in face):
                new_face = [vertex_mapping[v] for v in face]
                valid_faces.append(new_face)
        
        self.faces = np.array(valid_faces) if valid_faces else np.array([]).reshape(0, 3)
        if recompute_normals and len(self.faces) > 0:
            self.compute_normals()
        else:
            self.normals = None
        
        # Invalidate cached properties
        self._invalidate_cache()
    
    def _apply_filter_mask(self, keep_mask):
        """Apply filtering mask to mesh vertices and faces."""
        vertices_to_keep = np.where(keep_mask)[0]
        self._filter_by_vertex_set(vertices_to_keep)

    def filter_by_labels(self, vertex_labels: np.ndarray, surface_type: str = 'inner', inplace: bool = False) -> "Mesh" | None:
        """
        Create a new mesh containing only inner or outer vertices based on labels.
        
        Parameters
        ----------
        vertex_labels : np.ndarray
            Labels from separate_surfaces (0=inner, 1=outer)
        surface_type : str, default='inner'
            Which surface to extract: 'inner' or 'outer'
        inplace : bool, default=False
            If True, modify in place. If False, return new instance.
            
        Returns
        -------
        Mesh or None
            New mesh instance if inplace=False, else None
        """
        # Determine which label to keep
        target_label = 0 if surface_type == 'inner' else 1
        mask = (vertex_labels == target_label)
        vertices_to_keep = np.where(mask)[0]
        
        if inplace:
            self._filter_by_vertex_set(vertices_to_keep)
            return None
        else:
            new_mesh = copy.deepcopy(self)
            new_mesh._filter_by_vertex_set(vertices_to_keep)
            return new_mesh

    def get_euler_characteristic(self):
        """
        Compute the Euler-Poincaré characteristic (χ = V - E + F).
        
        Returns
        -------
        int
            Euler characteristic
        """
        mesh_o3d = self._to_open3d()
        return mesh_o3d.euler_poincare_characteristic()

    def get_connected_triangles(self, triangle_id: int, max_hops: int = 1) -> set:
        """
        Find triangles connected by shared edges (topology-based).
        
        Parameters
        ----------
        triangle_id : int
            ID of the seed triangle
        max_hops : int, default=1
            Maximum number of edge connections to traverse
        
        Returns
        -------
        set
            Set of triangle IDs including the seed triangle
        """
        triangles = np.asarray(self.o3d_mesh.triangles)
        
        # Build edge-to-triangle mapping
        edge_to_triangles = {}
        for tri_id, tri in enumerate(triangles):
            # Each triangle has 3 edges (v0-v1, v1-v2, v2-v0)
            edges = [
                tuple(sorted([tri[0], tri[1]])),
                tuple(sorted([tri[1], tri[2]])),
                tuple(sorted([tri[2], tri[0]]))
            ]
            for edge in edges:
                if edge not in edge_to_triangles:
                    edge_to_triangles[edge] = []
                edge_to_triangles[edge].append(tri_id)
        
        # BFS to find neighbors
        visited = {triangle_id}
        current_layer = {triangle_id}
        
        for hop in range(max_hops):
            next_layer = set()
            for tri_id in current_layer:
                tri = triangles[tri_id]
                edges = [
                    tuple(sorted([tri[0], tri[1]])),
                    tuple(sorted([tri[1], tri[2]])),
                    tuple(sorted([tri[2], tri[0]]))
                ]
                for edge in edges:
                    for neighbor_id in edge_to_triangles.get(edge, []):
                        if neighbor_id not in visited:
                            next_layer.add(neighbor_id)
                            visited.add(neighbor_id)
            current_layer = next_layer
        
        return visited

    def get_triangles_within_radius(self, triangle_id: int, radius: float, use_kdtree: bool = True) -> dict:
        """
        Find triangles whose centroids are within radius (distance-based).
        
        Parameters
        ----------
        triangle_id : int
            ID of the seed triangle
        radius : float
            Search radius
        use_kdtree : bool, default=True
            Use KDTree for faster search (recommended for large meshes)
        
        Returns
        -------
        dict
            Dictionary containing:
            - 'neighbor_ids': array of triangle IDs within radius
            - 'distances': distances from seed triangle centroid
            - 'seed_centroid': centroid of seed triangle
            - 'n_neighbors': number of neighbors found
        """
        # Ensure o3d_mesh exists
        if not hasattr(self, 'o3d_mesh') or self.o3d_mesh is None:
            self.o3d_mesh = self._to_open3d()
        
        vertices = np.asarray(self.o3d_mesh.vertices)
        triangles = np.asarray(self.o3d_mesh.triangles)
        
        # CACHE centroids computation (key optimization!)
        # Only recompute if mesh geometry changed or centroids not cached
        if not hasattr(self, '_triangle_centroids') or self._triangle_centroids is None:
            self._triangle_centroids = vertices[triangles].mean(axis=1)
        
        centroids = self._triangle_centroids
        seed_centroid = centroids[triangle_id]
        
        if use_kdtree:
            # Fast search using KDTree (also cached)
            if not hasattr(self, '_centroid_kdtree') or self._centroid_kdtree is None:
                self._centroid_kdtree = KDTree(centroids)
            neighbor_ids = self._centroid_kdtree.query_ball_point(seed_centroid, radius)
            distances = np.linalg.norm(centroids[neighbor_ids] - seed_centroid, axis=1)
        else:
            # Direct distance computation
            distances = np.linalg.norm(centroids - seed_centroid, axis=1)
            neighbor_ids = np.where(distances <= radius)[0]
            distances = distances[neighbor_ids]
        
        return {
            'neighbor_ids': np.array(neighbor_ids),
            'distances': distances,
            'seed_centroid': seed_centroid,
            'n_neighbors': len(neighbor_ids)
        }
    
    def get_surface_area(self):
        """
        Compute total surface area of the mesh.
        
        Returns
        -------
        float
            Total surface area (sum of all face areas)
        """
        self.compute_mesh_properties()  # Ensures _face_areas is computed
        return np.sum(self._face_areas)

    def convex_hull(self):
        """
        Compute convex hull of the mesh.

        Returns
        -------
        hull : open3d.geometry.TriangleMesh
            The convex hull mesh.
        stats : dict
            Hull statistics: volume, surface area, vertex/face counts,
            bounding-box extent, compactness, and ratios to the original mesh.
        """
        mesh_o3d = self._to_open3d()
        hull, _ = mesh_o3d.compute_convex_hull()
        
        # Compute hull statistics
        hull_vertices = np.asarray(hull.vertices)
        hull_faces = np.asarray(hull.triangles)
        
        # Volume and surface area
        volume = hull.get_volume()
        surface_area = hull.get_surface_area()
        
        # Bounding box dimensions
        bbox = hull.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        
        # Compactness measure (how close to a sphere)
        compactness = (36 * np.pi * volume**2) / (surface_area**3) if surface_area > 0 else 0
        
        # Original mesh statistics for comparison
        original_volume = mesh_o3d.get_volume()
        original_surface_area = mesh_o3d.get_surface_area()
        
        stats = {
            'hull_volume': volume,
            'hull_surface_area': surface_area,
            'hull_vertices': len(hull_vertices),
            'hull_faces': len(hull_faces),
            'bounding_box_extent': bbox_extent,
            'compactness': compactness,
            'original_volume': original_volume,
            'original_surface_area': original_surface_area,
            'volume_ratio': volume / original_volume if original_volume > 0 else 0,
        }
        
        return hull, stats

    def crop(self, bbox, inplace: bool = False, recompute_normals: bool = True) -> "Mesh" | None:
        """
        Crop mesh to bounding box.
        
        Parameters
        ----------
        bbox : o3d.geometry.AxisAlignedBoundingBox or dict
            Bounding box for cropping. If dict, should have 'min_bound' and 'max_bound' keys
        inplace : bool, default=False
            If True, modify in place. If False, return new instance
        recompute_normals : bool, default=True
            If True, recompute normals after cropping because face neighborhoods change.
        
        Returns
        -------
        Mesh or None
            New instance if inplace=False, else None
        """
        mesh_o3d = self._to_open3d()
        bbox = _axis_aligned_bbox_from_input(bbox)
        cropped_o3d = mesh_o3d.crop(bbox)
        
        original_vertices = len(self.vertices)
        original_faces = len(self.faces)
        cropped_vertices = len(cropped_o3d.vertices)
        cropped_faces = len(cropped_o3d.triangles)
        
        if inplace:
            self.vertices = np.asarray(cropped_o3d.vertices)
            self.faces = np.asarray(cropped_o3d.triangles)
            if recompute_normals and len(self.faces) > 0:
                self.compute_normals()
            elif cropped_o3d.has_vertex_normals():
                self.normals = np.asarray(cropped_o3d.vertex_normals)
            else:
                self.normals = None
            self._invalidate_cache()
            print(f"Cropped mesh: {original_vertices}→{cropped_vertices} vertices, {original_faces}→{cropped_faces} faces")
            return None
        else:
            new_mesh = copy.deepcopy(self)
            new_mesh.vertices = np.asarray(cropped_o3d.vertices)
            new_mesh.faces = np.asarray(cropped_o3d.triangles)
            if recompute_normals and len(new_mesh.faces) > 0:
                new_mesh.compute_normals()
            elif cropped_o3d.has_vertex_normals():
                new_mesh.normals = np.asarray(cropped_o3d.vertex_normals)
            else:
                new_mesh.normals = None
            new_mesh._invalidate_cache()
            print(f"Created cropped mesh: {original_vertices}→{cropped_vertices} vertices, {original_faces}→{cropped_faces} faces")
            return new_mesh
    
    def subdivide_mesh(self, iterations: int = 1, recompute_normals: bool = True) -> "Mesh":
        """
        Apply Loop subdivision to create a smoother, denser mesh by adding vertices at edge midpoints
        and face centers, then repositioning vertices using weighted averages.
        
        Parameters
        ----------
        iterations : int, default=1
            Number of subdivision iterations. Each iteration roughly quadruples
            the triangle count:
            - 1: ~4x triangles (recommended for most cases)
            - 2: ~16x triangles (very dense)
            - 3: ~64x triangles (extremely dense, use with caution)
        recompute_normals : bool, default=True
            If True, recompute normals after subdivision.
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh must have vertices and faces for subdivision")
        
        print(f"Applying Loop subdivision ({iterations} iterations)")
        print(f"Original: {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        # Convert to Open3D mesh
        o3d_mesh = self._to_open3d()
        
        # Apply Loop subdivision
        for i in range(iterations):
            o3d_mesh = o3d_mesh.subdivide_loop(1)
            if i == 0:  # Report after first iteration
                print(f"After iteration 1: {len(o3d_mesh.vertices)} vertices, {len(o3d_mesh.triangles)} faces")
        
        # Update mesh from subdivided result
        self._from_open3d(o3d_mesh)
        if recompute_normals:
            self.compute_normals()
        else:
            self.normals = None
        self._invalidate_cache()
        
        print(f"Final: {len(self.vertices)} vertices, {len(self.faces)} faces")
        
        return self

    def sample_points_poisson_disk(self, number_of_points: int | None = None, init_factor: int = 1) -> "OrientedPointCloud":
        """
        Sample points from mesh using Poisson disk sampling. Uses triangle centers as initial points by default.
        
        Parameters
        ----------
        number_of_points : int, optional
            Target number of points to sample. If None, uses init_factor * num_vertices
        init_factor : int, default=1
            Multiplier for number_of_points if not specified: init_factor * num_vertices

        Returns
        -------
        OrientedPointCloud
            New OrientedPointCloud instance
        """
        mesh_o3d = self._to_open3d()
        
        # Determine number of points
        if number_of_points is None:
            number_of_points = init_factor * len(self.vertices)
    
        # Use triangle centers as initial points
        triangle_centers = np.mean(self.vertices[self.faces], axis=1)
        pcl_init = o3d.geometry.PointCloud()
        pcl_init.points = o3d.utility.Vector3dVector(triangle_centers)

        
        # Perform Poisson disk sampling
        pcl_sampled = mesh_o3d.sample_points_poisson_disk(
            number_of_points=number_of_points,
            init_factor=init_factor,
            pcl=pcl_init
        )
        
        # Extract vertices and normals
        vertices = np.asarray(pcl_sampled.points)
        normals = np.asarray(pcl_sampled.normals) if pcl_sampled.has_normals() else None
        
        # If no normals, estimate them
        if normals is None:
            print("No normals found, estimating from sampled points...")
            pcl_sampled.compute_normals()
            normals = np.asarray(pcl_sampled.normals)
        
        oriented_pcd = OrientedPointCloud()
        oriented_pcd.vertices = vertices
        oriented_pcd.normals = normals
        oriented_pcd.inherit_coordinate_metadata(self)
 
        print(f"Sampled {len(vertices)} points from mesh using Poisson disk sampling")
        return oriented_pcd

    def oversample(self, oversample_factor: float | None = None, point_spacing: float | None = None, 
                            poisson_init_factor: int = 5) -> "OrientedPointCloud":
        """
        Oversample or undersample mesh using subdivision (if needed) and Poisson disk sampling.
        
        Ttwo modes:
        1. Factor-based: Use oversample_factor to specify desired increase in points
        2. Spacing-based: Use point_spacing to specify desired spacing between points
        
        Parameters
        ----------
        oversample_factor : float, optional
            Desired factor of increase in vertices/points.
            If None and point_spacing is provided, will be computed from spacing.
            If both are None, defaults to 1.0 (no change).
        point_spacing : float, optional
            Desired spacing between sampled points (same units as mesh coordinates).
            If provided, uses two-pass Poisson disk sampling:
            1. Calibration pass: samples mesh to measure actual Poisson disk spacing
            2. Final pass: samples with target number of points computed from calibration
        poisson_init_factor : int, default=5
            Initial candidate factor for Poisson-disk sampling (larger -> more uniform)
        
        Returns
        -------
        OrientedPointCloud
            Sampled point cloud with normals
        """
        if self.vertices is None or self.faces is None:
            raise ValueError("Cannot oversample: mesh must have vertices and faces")

        mesh_o3d = self._to_open3d()
        original_vertices = len(self.vertices)
        
        # Ensure mesh has normals
        if not mesh_o3d.has_vertex_normals():
            mesh_o3d.compute_vertex_normals()
        
        # Determine target number of points
        if point_spacing is not None:
            # Calibration: sample original mesh to measure Poisson disk spacing
            calibration_points = min(original_vertices, 20000)
            
            pcd_cal = mesh_o3d.sample_points_poisson_disk(
                number_of_points=calibration_points,
                init_factor=poisson_init_factor
            )
            
            cal_vertices = np.asarray(pcd_cal.points)
            tree_cal = cKDTree(cal_vertices)
            distances, _ = tree_cal.query(cal_vertices, k=2)
            nn_distances = distances[:, 1]
            current_spacing = np.mean(nn_distances)
            
            # Compute target points using quadratic scaling
            spacing_ratio = current_spacing / point_spacing
            target_points = int(calibration_points * spacing_ratio ** 2)
            
            print(f"Target points: {target_points} ({target_points/original_vertices:.2f}x original)")
            
        elif oversample_factor is not None:
            target_points = int(original_vertices * oversample_factor)
            print(f"Target points: {target_points} ({oversample_factor:.2f}x)")
        else:
            target_points = original_vertices
            print(f"Target points: {target_points}")
        
        # Subdivision (if needed)
        subdivided = mesh_o3d
        if target_points > original_vertices:
            iter_count = 0
            while len(subdivided.vertices) < target_points and iter_count < 10:
                subdivided = subdivided.subdivide_loop(number_of_iterations=1)
                iter_count += 1
            print(f"Subdivided to {len(subdivided.vertices)} vertices")
        
        # Final Poisson disk sampling
        pcd_sampled = subdivided.sample_points_poisson_disk(
            number_of_points=target_points,
            init_factor=poisson_init_factor
        )
        
        # Extract vertices and normals
        vertices = np.asarray(pcd_sampled.points)
        normals = np.asarray(pcd_sampled.normals) if pcd_sampled.has_normals() else None
        
        if normals is None:
            subdivided.compute_vertex_normals()
            tree = o3d.geometry.KDTreeFlann(subdivided)
            normals = []
            for point in vertices:
                [_, idx, _] = tree.search_knn_vector_3d(point, 1)
                normal = np.asarray(subdivided.vertex_normals)[idx[0]]
                normals.append(normal)
            normals = np.array(normals)
        
        oriented_pcd = OrientedPointCloud()
        oriented_pcd.vertices = vertices
        oriented_pcd.normals = normals
        oriented_pcd.inherit_coordinate_metadata(self)
        
        print(f"Final: {len(vertices)} points")
        
        return oriented_pcd

    def _to_open3d(self):
        """Convert to Open3D mesh."""
        o3d_mesh = o3d.geometry.TriangleMesh()
        o3d_mesh.vertices = o3d.utility.Vector3dVector(self.vertices)
        o3d_mesh.triangles = o3d.utility.Vector3iVector(self.faces)
        if self.normals is not None:
            o3d_mesh.vertex_normals = o3d.utility.Vector3dVector(self.normals)
        return o3d_mesh
    
    def _from_open3d(self, o3d_mesh):
        """Update from Open3D mesh."""
        self.vertices = np.asarray(o3d_mesh.vertices)
        self.faces = np.asarray(o3d_mesh.triangles)
        if o3d_mesh.has_vertex_normals():
            self.normals = np.asarray(o3d_mesh.vertex_normals)
        else:
            self.normals = None

    def to_oriented_points(self):
        """Convert mesh to oriented point representation."""
        return OrientedPointCloud.from_mesh(self)

    @classmethod
    def read_curvatures(cls, input_path: PathOrStr, units: str | None = None) -> "Mesh":
        """
        Load triangle mesh from a VTK/PolyData file including curvature point data.

        Expects meshes written by :meth:`save` with ``format='vtp'`` and
        ``include_curvatures=True``. Requires PyVista.

        Plain ``.ply`` triangle meshes **without** per-vertex curvature arrays should be
        loaded with :meth:`read` (Open3D).

        Parameters
        ----------
        input_path : str or Path
            Path to ``.vtp`` or compatible VTK polydata file
        units : str, optional
            Coordinate units for the mesh. If None, ``coordinate_units`` field data
            from the file is used when present.

        Returns
        -------
        Mesh
            Loaded mesh with curvature attributes populated when present in the file.
        """
        try:
            import pyvista as pv
        except ImportError as e:
            raise ImportError(
                "pyvista is required to read curvature meshes (VTP). Install with:\n"
                "  pip install pyvista"
            ) from e

        input_path = Path(input_path)
        suf = input_path.suffix.lower()
        if suf == ".ply":
            raise ValueError(
                "read_curvatures does not load curvature PLY files. "
                "Use Mesh.read() for geometry-only PLY, or save/load curvatures as VTP "
                "(save(..., format='vtp', include_curvatures=True))."
            )

        print(f"Loading triangle mesh with curvatures: {input_path}")

        data = pv.read(str(input_path))
        if isinstance(data, pv.UnstructuredGrid):
            surf = data.extract_surface()
        else:
            surf = data
        surf = surf.triangulate()
        if surf.n_cells == 0:
            raise ValueError(f"No surface cells in {input_path}")

        vertices = np.asarray(surf.points, dtype=np.float64)
        if hasattr(surf, "regular_faces") and surf.regular_faces is not None and len(surf.regular_faces):
            faces = np.asarray(surf.regular_faces, dtype=np.int32)
        else:
            cells = np.asarray(surf.faces)
            if len(cells) == 0 or cells[0] != 3 or len(cells) % 4 != 0:
                raise ValueError(
                    "Expected triangular mesh connectivity; could not derive (n, 3) faces."
                )
            faces = cells.reshape(-1, 4)[:, 1:4].astype(np.int32)

        if len(vertices) == 0:
            raise ValueError(f"No vertices in {input_path}")
        if faces.size and (np.max(faces) >= len(vertices) or np.min(faces) < 0):
            raise ValueError("Invalid face indices relative to vertex count")

        mesh = cls()
        mesh.vertices = vertices
        mesh.faces = faces

        cu = cls._decode_field_scalar(surf.field_data.get("coordinate_units"))
        mesh.units = units if units is not None else cu

        pd = surf.point_data
        if "normals" in pd:
            mesh.normals = np.asarray(pd["normals"], dtype=np.float64)
        elif all(k in pd for k in ("normal_x", "normal_y", "normal_z")):
            mesh.normals = np.column_stack(
                [pd["normal_x"], pd["normal_y"], pd["normal_z"]]
            ).astype(np.float64)

        curvatures_loaded = False
        if "mean_curvature" in pd:
            mesh._mean_curvature = np.asarray(pd["mean_curvature"], dtype=np.float64).ravel()
            curvatures_loaded = True
        if "gaussian_curvature" in pd:
            mesh._gaussian_curvature = np.asarray(
                pd["gaussian_curvature"], dtype=np.float64
            ).ravel()
            curvatures_loaded = True
        if "k1" in pd and "k2" in pd:
            k1 = np.asarray(pd["k1"], dtype=np.float64).ravel()
            k2 = np.asarray(pd["k2"], dtype=np.float64).ravel()
            mesh._principal_curvatures = np.column_stack([k1, k2])
            curvatures_loaded = True

        if "principal_direction_1" in pd and "principal_direction_2" in pd:
            d1 = np.asarray(pd["principal_direction_1"], dtype=np.float64)
            d2 = np.asarray(pd["principal_direction_2"], dtype=np.float64)
            if d1.shape == (len(vertices), 3) and d2.shape == (len(vertices), 3):
                mesh._principal_directions = np.stack([d1, d2], axis=2)

        print(f"Loaded: {len(vertices)} vertices, {len(faces)} faces")

        if mesh.units is None:
            print("\nCoordinate units not specified!")
            print("   Set with: mesh.units = 'nm'  (or 'angstrom', 'pixel', etc.)")
        else:
            print(f"Coordinate units: {mesh.units}")
            unit_info = cls._get_curvature_unit_str(mesh.units)
            print(f"Curvatures will be in: {unit_info}")

        if curvatures_loaded:
            print("Curvature data loaded from file")
            if mesh._principal_curvatures is not None:
                print(
                    "  Principal curvatures: k1=["
                    f"{mesh._principal_curvatures[:, 0].min():.6e}, "
                    f"{mesh._principal_curvatures[:, 0].max():.6e}], "
                    f"k2=[{mesh._principal_curvatures[:, 1].min():.6e}, "
                    f"{mesh._principal_curvatures[:, 1].max():.6e}]"
                )
            if mesh._mean_curvature is not None:
                print(
                    "  Mean curvature: ["
                    f"{mesh._mean_curvature.min():.6e}, {mesh._mean_curvature.max():.6e}]"
                )
            if mesh._gaussian_curvature is not None:
                print(
                    "  Gaussian curvature: ["
                    f"{mesh._gaussian_curvature.min():.6e}, "
                    f"{mesh._gaussian_curvature.max():.6e}]"
                )
        else:
            print(
                "No curvature point data found in file (expected keys like "
                "mean_curvature, k1, k2)"
            )

        return mesh

    @classmethod
    def from_mrc(cls, input_path: PathOrStr, transpose: bool = True, labels_dict: dict | None = None, level: float = 0.5, pixel_size: float = 1.0, 
                smooth_sigma: float | None = None) -> "Mesh":
        """
        Create mesh from MRC segmentation file (marching cubes + optional Gaussian smoothing).

        Vertex normals come from marching cubes. For smoother or more globally consistent normals,
        call :meth:`refine_normals` on the returned mesh (see that method for ``radius_hit``,
        ``batch_size``, and ``n_iter``).

        Coordinate System Conventions:

        1. Physical Units (for software expecting real-world coordinates)
            - Set `pixel_size` to the physical pixel/voxel size (e.g., `pixel_size=0.7884`)
            - Set `sampling_distance` in the same physical units (e.g., `sampling_distance=0.55` for 0.55 nm)
            - Output coordinates are in physical units (nm)

        2. Pixel/voxel Units (for software expecting pixel_size=1)
            - Set `pixel_size=1` (normalized)
            - Set `sampling_distance` in pixel/voxel units (e.g., scaling factor `sampling_distance=0.55/0.7884`)
            - Output coordinates are in pixel/voxel units

        Parameters
        ----------
        input_path : str
            Path to MRC file
        transpose : bool, default=True
            Whether to transpose the segmentation data
        labels_dict : dict, optional
            Dictionary mapping membrane names to label values.
            If None, treats as binary segmentation.
        level : float, default=0.5
            Iso-level for marching cubes
        pixel_size : float, default=1.0
            Pixel/voxel size for scaling the mesh coordinates.
            - If > 1: Physical pixel/voxel size (e.g., 0.7884 for 0.7884 nm/pixel). Coordinates will be in physical units.
            - If 1: Normalized pixel/voxel units.
        smooth_sigma : float, optional
            Gaussian smoothing sigma (in pixel/voxel units) to apply before meshing.
            Typical values: 0.5-2.0. Reduces step-like artifacts from marching cubes.
            If None, no smoothing is applied.

        Notes
        -----
        Segmentation volumes load only via :func:`cryocat.core.cryomap.read`. If loading fails due
        to a malformed or non-standard MRC header, repair the file before calling this method.

        Returns
        -------
        Mesh or dict
            Single mesh if ``labels_dict`` is None, else dict mapping each membrane name to a
            :class:`Mesh`.
        """

        segmentation = cryomap.read(input_path, transpose=transpose)

        if labels_dict is None:
            # Single binary segmentation
            return cls._create_mesh_from_seg(
                segmentation, level, pixel_size, smooth_sigma, transpose)
        else:
            # Multiple labels
            meshes = {}
            for membrane_name, label_value in labels_dict.items():
                membrane_mask = (segmentation == label_value).astype(float)
                if membrane_mask.sum() > 0:  # Only create mesh if label exists
                    mesh = cls._create_mesh_from_seg(
                        membrane_mask, level, pixel_size, smooth_sigma, transpose)
                    meshes[membrane_name] = mesh
            return meshes

    @classmethod
    def _create_mesh_from_seg(cls, segmentation: np.ndarray, level: float, pixel_size: float,
                            smooth_sigma: float | None = None, transpose: bool = True) -> "Mesh":
        """Create mesh from segmentation volume with marching cubes."""

        # Apply Gaussian smoothing to reduce step-like artifacts from marching cubes
        if smooth_sigma is not None and smooth_sigma > 0:
            print(f"Applying Gaussian smoothing (sigma={smooth_sigma})...")
            segmentation = scipy.ndimage.gaussian_filter(segmentation.astype(float), sigma=smooth_sigma)
            print(f"Smoothed field range: [{segmentation.min():.3f}, {segmentation.max():.3f}]")

        vertices_world, faces, normals, vertices_pixel = cls._extract_surface_points(
            segmentation, pixel_size, level)

        if vertices_world is None or len(vertices_world) == 0:
            raise ValueError("DiscreteSurface extraction failed - no valid vertices found")

        mesh = cls()
        mesh.vertices = vertices_world
        mesh.faces = faces
        mesh.normals = normals
        mesh.seg_shape = segmentation.shape  # (X, Y, Z) if transpose=True, (Z, Y, X) if transpose=False
        mesh.pixel_size = DiscreteSurface._coerce_pixel_size_array(pixel_size)
        mesh.seg_transposed = transpose

        return mesh

    @staticmethod
    def _extract_surface_points(segmentation: np.ndarray, pixel_size: float, level: float = 0.5):
        """
        Extract dual surface from segmentation using marching cubes.

        Maintains float precision throughout.

        Parameters
        ----------
        segmentation : ndarray
            3D binary segmentation volume (0=background, 1=membrane)
        pixel_size : float
            Voxel size in physical units, used to scale the marching-cubes
            output to world coordinates.
        level : float, default=0.5
            Iso-level for marching cubes

        Returns
        -------
        vertices_world : ndarray
            Nx3 array in world coordinates (Angstroms), float precision
        faces : ndarray
            Mx3 array of triangle indices
        normals : ndarray
            Nx3 array of vertex normals
        vertices_pixel : ndarray
            Nx3 array in voxel coordinates, float precision
        """

        # Run marching cubes
        print(f"Running marching cubes (level={level})...")
        vertices_world, faces, normals, _ = skimage.measure.marching_cubes(
            segmentation, level=level, spacing=(pixel_size, pixel_size, pixel_size)
        )

        # Convert to voxel coordinates (keep float precision!)
        vertices_pixel = vertices_world / pixel_size

        return vertices_world, faces, normals, vertices_pixel

    def save(self, output_path: PathOrStr, format: str | None = None, include_curvatures: bool = False):
        """
        Save mesh to file with optional curvature data.
        
        Supports multiple output formats with optional per-vertex curvature properties.
        Curvature export is supported only via VTP (requires pyvista). Use ``format='vtp'``.
        
        Parameters
        ----------
        output_path : str or Path
            Output file path
        format : str, optional
            Output format:
            - 'ply': PLY format (geometry only; use ``include_curvatures=False``)
            - 'vtp': VTK PolyData format (requires pyvista); use for curvature export
            If None, inferred from ``output_path`` suffix and defaults to 'ply' when missing.
        include_curvatures : bool, default=False
            If True, include curvature data as vertex properties.
            Requires curvatures to be computed first via compute_curvatures().
            
            Curvature fields saved:
            - Scalars: mean_curvature, gaussian_curvature, k1, k2, curvature_anisotropy
            - Vectors: normals, principal_direction_1, principal_direction_2
        
        Returns
        -------
        dict or None
            If include_curvatures=True, returns metadata dict with saved fields.
            If include_curvatures=False, returns None.
        """
        output_path = Path(output_path)
        
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh must have vertices and faces to save")
        
        format_lower = output_path.suffix.lower().lstrip(".") if format is None else str(format).lower()
        if not format_lower:
            format_lower = "ply"
        
        if not include_curvatures:
            return self._save_mesh(output_path, format_lower)
        else:
            if format_lower == 'ply':
                warnings.warn(
                    "Use VTP for curvature export (format='vtp', pyvista). "
                    "PLY with per-vertex curvature fields is no longer supported.",
                    UserWarning,
                    stacklevel=2,
                )
                raise ValueError(
                    "Cannot save curvatures with format='ply'. "
                    "Use save(..., format='vtp', include_curvatures=True) "
                    "or save without curvatures via include_curvatures=False."
                )
            elif format_lower == 'vtp':
                return self._save_mesh_vtp_with_curvatures(output_path)
            else:
                raise ValueError(
                    f"Unsupported format '{format}' for curvature export. Use 'vtp'."
                )

    def _save_mesh(self, output_path: PathOrStr, format: str = 'ply'):
        """Save mesh without curvature using Open3D or pyvista (for VTP)."""
        output_path = Path(output_path)
        
        # VTP format requires pyvista
        if format == 'vtp':
            try:
                import pyvista as pv
            except ImportError:
                raise ImportError(
                    "pyvista library required for VTP export. Install with:\n"
                    "  pip install pyvista\n"
                    "For PLY export, use format='ply' instead"
                )
            
            # Create PyVista mesh
            faces_vtk = np.hstack([
                np.full((len(self.faces), 1), 3),
                self.faces
            ]).astype(np.int64)
            
            mesh_pv = pv.PolyData(self.vertices, faces_vtk)
            
            # Add normals if available
            if self.normals is not None:
                mesh_pv.point_data['normals'] = self.normals
            else:
                print("Warning: No normals available, computing...")
                self.compute_normals()
                mesh_pv.point_data['normals'] = self.normals
            
            # Save to VTP
            mesh_pv.save(str(output_path))
            
            print(f"Saved mesh to {output_path}")
            print(f"  Format: VTP (VTK PolyData)")
            print(f"  Vertices: {len(self.vertices):,}")
            print(f"  Faces: {len(self.faces):,}")
            
            return None
        
        # For other formats, use Open3D
        import open3d as o3d
        
        o3d_mesh = self._to_open3d()
        success = o3d.io.write_triangle_mesh(str(output_path), o3d_mesh)
        
        if success:
            print(f"Saved mesh to {output_path}")
            print(f"  Vertices: {len(self.vertices):,}")
            print(f"  Faces: {len(self.faces):,}")
        else:
            raise IOError(f"Failed to write mesh to {output_path}")
        
        return None

    def _save_mesh_vtp_with_curvatures(self, output_path: PathOrStr):
        """Save mesh to VTP with curvature data using pyvista."""
        try:
            import pyvista as pv
        except ImportError:
            raise ImportError(
                "pyvista library required for VTP export. Install with:\n"
                "  pip install pyvista\n"
                "For PLY export, use format='ply' instead"
            )
        
        output_path = Path(output_path)
        
        # Check if curvatures are available
        if self._principal_curvatures is None:
            raise ValueError(
                "Curvatures not computed. Call compute_curvatures() first, or "
                "set include_curvatures=False for basic mesh export"
            )
        
        # Prepare curvature data
        k1 = self._principal_curvatures[:, 0]
        k2 = self._principal_curvatures[:, 1]
        H = self._mean_curvature
        K = self._gaussian_curvature
        dir1 = self._principal_directions[:, :, 0]
        dir2 = self._principal_directions[:, :, 1]
        curvature_anisotropy = np.abs(k1 - k2)
        
        # Create PyVista mesh
        faces_vtk = np.hstack([
            np.full((len(self.faces), 1), 3),
            self.faces
        ]).astype(np.int64)
        
        mesh_pv = pv.PolyData(self.vertices, faces_vtk)
        
        # Add scalar fields
        mesh_pv.point_data['mean_curvature'] = H
        mesh_pv.point_data['gaussian_curvature'] = K
        mesh_pv.point_data['k1'] = k1
        mesh_pv.point_data['k2'] = k2
        mesh_pv.point_data['curvature_anisotropy'] = curvature_anisotropy
        
        # Add vector fields
        if self.normals is not None:
            mesh_pv.point_data['normals'] = self.normals
        else:
            print("Warning: No normals available, computing...")
            self.compute_normals()
            mesh_pv.point_data['normals'] = self.normals
        
        mesh_pv.point_data['principal_direction_1'] = dir1
        mesh_pv.point_data['principal_direction_2'] = dir2
        
        if self.units:
            mesh_pv.field_data['coordinate_units'] = [self.units]
            unit_info_short = Mesh._get_curvature_unit_str(self.units).split(' (')[0]
            mesh_pv.field_data['curvature_units_linear'] = [unit_info_short]
        else:
            mesh_pv.field_data['coordinate_units'] = ['unknown']
            mesh_pv.field_data['curvature_units_linear'] = ['unknown']
        
        # Add curvature range metadata
        mesh_pv.field_data['mean_curvature_range'] = np.array([H.min(), H.max()])
        mesh_pv.field_data['gaussian_curvature_range'] = np.array([K.min(), K.max()])
        
        # Save to VTP
        mesh_pv.save(str(output_path))
        
        # Print summary
        print(f"Saved mesh with curvatures to {output_path}")
        print(f"  Format: VTP (VTK PolyData)")
        print(f"  Vertices: {len(self.vertices):,}")
        print(f"  Faces: {len(self.faces):,}")
        print(f"  Scalar fields (5):")
        print(f"    - mean_curvature: [{H.min():.6e}, {H.max():.6e}]")
        print(f"    - gaussian_curvature: [{K.min():.6e}, {K.max():.6e}]")
        print(f"    - k1: [{k1.min():.6e}, {k1.max():.6e}]")
        print(f"    - k2: [{k2.min():.6e}, {k2.max():.6e}]")
        print(f"    - curvature_anisotropy: [{curvature_anisotropy.min():.6e}, {curvature_anisotropy.max():.6e}]")
        print(f"  Vector fields (3): normals, principal_direction_1, principal_direction_2")
        if self.units:
            unit_info = Mesh._get_curvature_unit_str(self.units)
            print(f"  Curvature units: {unit_info}")
        else:
            print(f"  Curvature units: unknown (mesh.units not set)")
        
        # Return metadata
        metadata = {
            'format': 'VTP',
            'path': str(output_path),
            'n_vertices': len(self.vertices),
            'n_faces': len(self.faces),
            'coordinate_units': self.units,  # ← Added
            'scalar_fields': ['mean_curvature', 'gaussian_curvature', 'k1', 'k2', 'curvature_anisotropy'],
            'vector_fields': ['normals', 'principal_direction_1', 'principal_direction_2'],
            'statistics': {
                'mean_curvature': {'min': float(H.min()), 'max': float(H.max()), 'mean': float(H.mean())},
                'gaussian_curvature': {'min': float(K.min()), 'max': float(K.max()), 'mean': float(K.mean())},
                'k1': {'min': float(k1.min()), 'max': float(k1.max()), 'mean': float(k1.mean())},
                'k2': {'min': float(k2.min()), 'max': float(k2.max()), 'mean': float(k2.mean())},
                'curvature_anisotropy': {'min': float(curvature_anisotropy.min()), 'max': float(curvature_anisotropy.max()), 'mean': float(curvature_anisotropy.mean())}
            }
        }
        
        return metadata

    def _check_segmentation_metadata(self):
        """
        Required attributes:
            - seg_shape: tuple (X, Y, Z) or (Z, Y, X)
            - pixel_size: array-like (3,) in same units as vertices
        """
        missing = []
        if not hasattr(self, "seg_shape") or self.seg_shape is None:
            missing.append("seg_shape")
        if getattr(self, "pixel_size", None) is None:
            missing.append("pixel_size")
        if missing:
            raise ValueError(
                f"Segmentation metadata missing on mesh: {', '.join(missing)}. "
                "For meshes built from MRC via Mesh.from_mrc this is "
                "set automatically. For other meshes, call "
                "mesh.assign_segmentation_metadata(...) first."
            )

    def assign_segmentation_metadata(self, seg_shape: tuple[int, int, int], pixel_size: float | np.ndarray = 1.0):
        """
        Manually attach segmentation metadata to a mesh.

        Parameters
        ----------
        seg_shape : tuple
            Shape (X, Y, Z) or (Z, Y, X) of the target segmentation volume.
        pixel_size : float or array-like (3,)
            Pixel/voxel size in same units as mesh coordinates (world units).
        """
        self.seg_shape = tuple(seg_shape)
        self.pixel_size = self._coerce_pixel_size_array(pixel_size)

    def get_pixel_indices(self) -> np.ndarray:
        """
        Map mesh vertices (world coordinates) into pixel/voxel indices of the source segmentation.

        Returns
        -------
        indices_pixel : (N, 3) ndarray
            Pixel/voxel indices in (X, Y, Z) order
        """
        self._check_segmentation_metadata()

        v = np.asarray(self.vertices, dtype=float)
        spacing = np.asarray(self.pixel_size, dtype=float)
        origin = np.array([0.0, 0.0, 0.0], dtype=float)

        # v is in (X, Y, Z), so indices are also (X, Y, Z)
        indices = (v - origin) / spacing
        indices_int = np.rint(indices).astype(int)

        return indices_int

    def _pixel_grid_to_angstrom(self) -> np.ndarray:
        """
        Convert ``pixel_size`` to angstrom spacing for MRC export.
        """
        self._check_segmentation_metadata()

        if self.units is None:
            raise ValueError("Mesh.units is not set; cannot convert pixel/voxel size to A.")

        unit = self.units.lower()

        if unit in ["angstrom", "a", "A"]:
            factor = 1.0
            spacing = np.asarray(self.pixel_size, dtype=float)
            return spacing * factor

        if unit in ["nanometer", "nm"]:
            factor = 10.0
            spacing = np.asarray(self.pixel_size, dtype=float)
            return spacing * factor

        if unit in ["pixel", "pix"]:
            return np.array([1.0, 1.0, 1.0], dtype=float)

        raise ValueError(f"Unrecognized mesh.units='{self.units}'")

    def save_vertices_csv(self, output_path: PathOrStr, include_normals: bool = True):
        """
        Save mesh vertices (and optionally normals) to CSV, including both
        pixel/voxel indices and world coordinates.
        """
        self._check_segmentation_metadata()

        if self.vertices is None or len(self.vertices) == 0:
            raise ValueError("Mesh has no vertices to save")

        vertices_world = np.asarray(self.vertices, dtype=float)          # (N,3) = (X,Y,Z) in world units
        indices_pixel = np.asarray(self.get_pixel_indices(), dtype=int)  # (N,3) = (X,Y,Z) in pixel/voxel indices

        data = {
            "x_pixel": indices_pixel[:, 0],
            "y_pixel": indices_pixel[:, 1],
            "z_pixel": indices_pixel[:, 2],
            "x_world": vertices_world[:, 0],
            "y_world": vertices_world[:, 1],
            "z_world": vertices_world[:, 2],
        }

        if include_normals:
            normals = np.asarray(self.get_normals(), dtype=float)  # (N,3) = (X,Y,Z)
            data["normal_x"] = normals[:, 0]
            data["normal_y"] = normals[:, 1]
            data["normal_z"] = normals[:, 2]

        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} vertices to {output_path}")

    def save_vertices_mrc(self, output_path: PathOrStr):
        """
        Save a binary vertex-occupancy volume as MRC.
        """

        self._check_segmentation_metadata()

        if self.vertices is None or len(self.vertices) == 0:
            raise ValueError("Mesh has no vertices to rasterize")

        # indices are (X, Y, Z)
        idx = np.asarray(self.get_pixel_indices(), dtype=int)

        # seg_shape is stored exactly as the segmentation array shape (after transpose)
        sx, sy, sz = self.seg_shape
        vertex_volume = np.zeros((sx, sy, sz), dtype=np.int16)

        # binary occupancy
        for x, y, z in idx:
            if 0 <= x < sx and 0 <= y < sy and 0 <= z < sz:
                vertex_volume[x, y, z] = 1

        pixel_size_angstrom = self._pixel_grid_to_angstrom()
        # Write out
        cryomap.write(vertex_volume, str(output_path), transpose=True, pixel_size=pixel_size_angstrom)
        print(f"Saved mrc file to {output_path} (seg_shape={self.seg_shape})")

    # === CURVATURE COMPUTATION METHODS ===

    @staticmethod
    def _get_curvature_unit_str(units):
        """Get readable curvature unit string."""
        if units is None:
            return "unknown"
        
        unit_lower = units.lower()
        
        if unit_lower in ['angstrom', 'a', 'A']:
            return "A^(-1) (principal/mean), A^(-2) (Gaussian)"
        elif unit_lower in ['nanometer', 'nm']:
            return "nm^(-1) (principal/mean), nm^(-2) (Gaussian)"
        elif unit_lower in ['pixel', 'pix']:
            return "pix^(-1) (principal/mean), pix^(-2) (Gaussian) - convert to physical units!"
        else:
            return f"{units}^(-1) (principal/mean), {units}^(-2) (Gaussian)"

    def compute_curvatures(self, force_recompute: bool = False, 
                        min_triangle_area: float = 1e-12, lstsq_rcond: float = 1e-12,
                        smoothing_iterations: int = 1, smoothing_kernel_rings: int = 1, n_jobs: int | None = None):
        """
        Compute principal curvatures and directions for the mesh.
        
        Parameters
        ----------
        force_recompute : bool, default=False
            Force recomputation even if cached
        min_triangle_area : float, default=1e-12
            Minimum triangle area threshold for numerical stability
        lstsq_rcond : float, default=1e-12
            Condition number cutoff for least squares
        smoothing_iterations : int, default=0
            Number of smoothing iterations to apply to curvature tensors.
            0 = no smoothing, 1+ = number of smoothing passes.
        smoothing_kernel_rings : int, default=1
            Size of neighborhood for smoothing (1-ring, 2-ring, etc.).
        n_jobs : int, optional
            Number of parallel jobs for tensor smoothing.
            If None, uses all available CPU cores.
            If 1, uses sequential processing (no parallelization).
            Only used when smoothing_iterations > 0.
            
        Returns
        -------
        self
            Returns self for method chaining
        """
        if self._curvature_cache is not None and not force_recompute:
            return self
            
        if self.vertices is None or self.faces is None:
            raise ValueError("Mesh must have vertices and faces to compute curvatures")
        
        # Warn about units
        if self.units is None:
            print("\n Warning: Coordinate units not set!")
            print("   Curvature values will have unknown units.")
            print("   Set units with: mesh.units = 'angstrom'  (or 'nanometer', etc.)")
        else:
            unit_info = Mesh._get_curvature_unit_str(self.units)
            print(f"Computing curvatures (units: {unit_info})")
        
        # Store parameters
        self._min_triangle_area = min_triangle_area
        self._lstsq_rcond = lstsq_rcond
        
        self._compute_rusinkiewicz_curvatures(
            smoothing_iterations=smoothing_iterations,
            smoothing_kernel_rings=smoothing_kernel_rings,
            n_jobs=n_jobs
        )
            
        return self
    
    def get_principal_curvatures(self):
        """Return principal curvatures k1, k2."""
        if self._principal_curvatures is None:
            self.compute_curvatures()
        return self._principal_curvatures
    
    def get_mean_curvature(self):
        """Return mean curvature (k1 + k2)/2."""
        if self._mean_curvature is None:
            self.compute_curvatures()
        return self._mean_curvature
    
    def get_gaussian_curvature(self):
        """Return Gaussian curvature k1 * k2."""
        if self._gaussian_curvature is None:
            self.compute_curvatures()
        return self._gaussian_curvature
    
    def get_curvature_directions(self):
        """Return principal curvature directions."""
        if self._principal_directions is None:
            self.compute_curvatures()
        return self._principal_directions
    
    def get_curvature_tensors(self):
        """Return curvature tensors."""
        if self._curvature_tensors is None:
            self.compute_curvatures()
        return self._curvature_tensors
    
    def compute_mesh_properties(self):
        """Commonly used geometric properties."""
        if self._edge_vectors is not None:
            return  # Already computed
        
        # Compute edge vectors
        e0 = self.vertices[self.faces[:, 2]] - self.vertices[self.faces[:, 1]]  # v2 - v1
        e1 = self.vertices[self.faces[:, 0]] - self.vertices[self.faces[:, 2]]  # v0 - v2
        e2 = self.vertices[self.faces[:, 1]] - self.vertices[self.faces[:, 0]]  # v1 - v0
        self._edge_vectors = (e0, e1, e2)
        
        # Compute edge lengths
        self._edge_lengths = (
            np.linalg.norm(e0, axis=1),
            np.linalg.norm(e1, axis=1), 
            np.linalg.norm(e2, axis=1)
        )
        
        # Compute face areas
        self._face_areas = 0.5 * np.linalg.norm(np.cross(e0, e1), axis=1)
        
        # Compute face normals
        face_normals = np.cross(e0, e1)
        norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
        self._face_normals_cached = face_normals / (norms + 1e-12)

    def _compute_rusinkiewicz_curvatures(self, smoothing_iterations=1, smoothing_kernel_rings=1, n_jobs=None):
        """Compute curvatures using Rusinkiewicz's method."""
        # Validate mesh
        self._validate_mesh_for_curvature()
        
        # Precompute geometric properties
        self.compute_mesh_properties()
        
        # Step 1: Use cached face normals
        face_normals = self._face_normals_cached
        
        # Step 2: Compute vertex normals, areas, and coordinate systems  
        vertex_normals, vertex_areas, corner_areas, up_vectors, vp_vectors = \
            self._compute_vertex_properties_curvature(face_normals)
        
        # Step 3: Compute face curvature tensors via vectorized least squares
        face_tensors = self._compute_face_curvature_tensors(
            vertex_normals, face_normals, up_vectors, vp_vectors
        )
        
        # Step 4: Accumulate tensors at vertices using sparse matrices
        vertex_tensors = self._accumulate_vertex_tensors(
            face_tensors, corner_areas, vertex_areas, 
            up_vectors, vp_vectors, face_normals
        )
        
        # Step 4.5: Smooth curvature tensors (if requested)
        if smoothing_iterations > 0:
            vertex_tensors = self._smooth_curvature_tensors(
                vertex_tensors, up_vectors, vp_vectors, vertex_normals,
                smoothing_iterations, smoothing_kernel_rings, n_jobs
            )
        
        # Step 5: Extract principal curvatures and directions vectorized
        principal_curvatures, principal_directions = self._extract_principal_curvatures(
            vertex_tensors, up_vectors, vp_vectors, vertex_normals
        )
        
        # Compute derived quantities
        mean_curvature = 0.5 * (principal_curvatures[:, 0] + principal_curvatures[:, 1])
        gaussian_curvature = principal_curvatures[:, 0] * principal_curvatures[:, 1]
        
        # Store results in class attributes
        self._principal_curvatures = principal_curvatures
        self._principal_directions = principal_directions
        self._mean_curvature = mean_curvature
        self._gaussian_curvature = gaussian_curvature
        self._curvature_tensors = vertex_tensors
        
        # Mark as cached
        self._curvature_cache = True
    
    def _validate_mesh_for_curvature(self):
        """Validate mesh is suitable for curvature computation."""
        if self.vertices.shape[1] != 3:
            raise ValueError("Vertices must be 3D")
        if self.faces.shape[1] != 3:
            raise ValueError("Faces must be triangular")
        if len(self.vertices) < 3:
            raise ValueError("Mesh must have at least 3 vertices")
        if len(self.faces) < 1:
            raise ValueError("Mesh must have at least 1 face")
            
        # Check for degenerate triangles using cached areas
        if self._face_areas is None:
            self.compute_mesh_properties()
        
        n_degenerate = np.sum(self._face_areas < self._min_triangle_area)
        if n_degenerate > 0:
            warnings.warn(f"Found {n_degenerate} degenerate triangles (area < {self._min_triangle_area})")
    
    def _compute_vertex_properties_curvature(self, face_normals):
        """Computation of vertex normals, Voronoi areas, and coordinate systems."""
        n_vertices = len(self.vertices)
        n_faces = len(self.faces)
        
        # Use cached edge vectors and lengths
        e0, e1, e2 = self._edge_vectors
        de0, de1, de2 = self._edge_lengths
        face_areas = self._face_areas
        
        # Initialize outputs
        vertex_normals = np.zeros((n_vertices, 3))
        vertex_areas = np.zeros(n_vertices)
        corner_areas = np.zeros((n_faces, 3))
        up_vectors = np.zeros((n_vertices, 3))
        vp_vectors = np.zeros((n_vertices, 3))
        
        # Vectorized edge normalization
        e0_norm = e0 / (de0[:, np.newaxis] + 1e-12)
        e1_norm = e1 / (de1[:, np.newaxis] + 1e-12)
        e2_norm = e2 / (de2[:, np.newaxis] + 1e-12)
        
        # Vectorized squared lengths and barycentric coordinates
        l2 = np.column_stack([de0**2, de1**2, de2**2])
        ew = np.column_stack([
            l2[:, 0] * (l2[:, 1] + l2[:, 2] - l2[:, 0]),
            l2[:, 1] * (l2[:, 2] + l2[:, 0] - l2[:, 1]), 
            l2[:, 2] * (l2[:, 0] + l2[:, 1] - l2[:, 2])
        ])
        
        # Vectorized vertex normal weights
        wfv = face_areas[:, np.newaxis] / np.column_stack([
            de1**2 * de2**2, de0**2 * de2**2, de1**2 * de0**2
        ])
        wfv = np.where(wfv > 1e-12, wfv, 0)
        
        # Direct accumulation of vertex normals (faster than sparse for this case)
        for i in range(n_faces):
            face = self.faces[i]
            face_normal = face_normals[i]
            weights = wfv[i]
            
            vertex_normals[face[0]] += weights[0] * face_normal
            vertex_normals[face[1]] += weights[1] * face_normal
            vertex_normals[face[2]] += weights[2] * face_normal
        
        # Vectorized Voronoi area computation
        valid_faces = face_areas >= self._min_triangle_area
        
        # Obtuse triangle detection (vectorized)
        obtuse_0 = ew[:, 0] <= 0
        obtuse_1 = ew[:, 1] <= 0  
        obtuse_2 = ew[:, 2] <= 0
        acute = ~(obtuse_0 | obtuse_1 | obtuse_2)
        
        # Vectorized corner area computation
        # Obtuse at vertex 0
        mask = obtuse_0 & valid_faces
        if np.any(mask):
            corner_areas[mask, 1] = -0.25 * l2[mask, 2] * face_areas[mask] / np.einsum('ij,ij->i', e0[mask], e2[mask])
            corner_areas[mask, 2] = -0.25 * l2[mask, 1] * face_areas[mask] / np.einsum('ij,ij->i', e0[mask], e1[mask])
            corner_areas[mask, 0] = face_areas[mask] - corner_areas[mask, 1] - corner_areas[mask, 2]
        
        # Obtuse at vertex 1
        mask = obtuse_1 & valid_faces
        if np.any(mask):
            corner_areas[mask, 2] = -0.25 * l2[mask, 0] * face_areas[mask] / np.einsum('ij,ij->i', e1[mask], e0[mask])
            corner_areas[mask, 0] = -0.25 * l2[mask, 2] * face_areas[mask] / np.einsum('ij,ij->i', e1[mask], e2[mask])
            corner_areas[mask, 1] = face_areas[mask] - corner_areas[mask, 0] - corner_areas[mask, 2]
        
        # Obtuse at vertex 2
        mask = obtuse_2 & valid_faces
        if np.any(mask):
            corner_areas[mask, 0] = -0.25 * l2[mask, 1] * face_areas[mask] / np.einsum('ij,ij->i', e2[mask], e1[mask])
            corner_areas[mask, 1] = -0.25 * l2[mask, 0] * face_areas[mask] / np.einsum('ij,ij->i', e2[mask], e0[mask])
            corner_areas[mask, 2] = face_areas[mask] - corner_areas[mask, 0] - corner_areas[mask, 1]
        
        # Acute triangles
        mask = acute & valid_faces
        if np.any(mask):
            ewscale = 0.5 * face_areas[mask] / (ew[mask, 0] + ew[mask, 1] + ew[mask, 2])
            corner_areas[mask, 0] = ewscale * (ew[mask, 1] + ew[mask, 2])
            corner_areas[mask, 1] = ewscale * (ew[mask, 0] + ew[mask, 2])
            corner_areas[mask, 2] = ewscale * (ew[mask, 1] + ew[mask, 0])
        
        # Direct accumulation of vertex areas
        for i in range(n_faces):
            face = self.faces[i]
            vertex_areas[face] += corner_areas[i]
        
        # Initial coordinate system setup (direct assignment)
        up_vectors[self.faces[:, 0]] = e2_norm
        up_vectors[self.faces[:, 1]] = e0_norm
        up_vectors[self.faces[:, 2]] = e1_norm
        
        # Normalize vertex normals
        norms = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        vertex_normals = vertex_normals / (norms + 1e-12)
        
        # Vectorized coordinate system correction
        valid_vertices = vertex_areas > 1e-12
        
        # Make up vectors orthogonal to normals
        up_vectors[valid_vertices] = np.cross(up_vectors[valid_vertices], vertex_normals[valid_vertices])
        up_norms = np.linalg.norm(up_vectors[valid_vertices], axis=1, keepdims=True)
        
        # Handle cases where cross product is near zero
        valid_up = (up_norms.ravel() > 1e-12)
        valid_idx = np.where(valid_vertices)[0][valid_up]
        
        up_vectors[valid_idx] /= up_norms[valid_up]
        vp_vectors[valid_idx] = np.cross(vertex_normals[valid_idx], up_vectors[valid_idx])
        
        # Fallback for problematic vertices
        invalid_idx = np.where(valid_vertices)[0][~valid_up]
        for i in invalid_idx:
            up_vectors[i] = self._get_orthogonal_vector(vertex_normals[i])
            vp_vectors[i] = np.cross(vertex_normals[i], up_vectors[i])
        
        return vertex_normals, vertex_areas, corner_areas, up_vectors, vp_vectors
    
    def _get_orthogonal_vector(self, v):
        """Get a vector orthogonal to v."""
        v = v / (np.linalg.norm(v) + 1e-12)
        # Choose axis with smallest component
        abs_v = np.abs(v)
        min_idx = np.argmin(abs_v)
        orthogonal = np.zeros(3)
        orthogonal[min_idx] = 1.0
        result = orthogonal - np.dot(orthogonal, v) * v
        return result / (np.linalg.norm(result) + 1e-12)
    
    def _compute_face_curvature_tensors(self, vertex_normals, face_normals, up_vectors, vp_vectors):
        """
        Vectorized computation of face curvature tensors using batch least squares.
        """
        n_faces = len(self.faces)
        
        # Use cached edge vectors
        e0, e1, e2 = self._edge_vectors
        face_areas = self._face_areas
        
        # Filter out degenerate faces
        valid_faces = face_areas >= self._min_triangle_area
        n_valid = np.sum(valid_faces)
        
        if n_valid == 0:
            return np.zeros((n_faces, 2, 2))
        
        # Work with valid faces only
        valid_indices = np.where(valid_faces)[0]
        e0_valid = e0[valid_faces]
        e1_valid = e1[valid_faces] 
        face_normals_valid = face_normals[valid_faces]
        faces_valid = self.faces[valid_faces]
        
        # Vectorized coordinate frame setup
        e0_norm = e0_valid / (np.linalg.norm(e0_valid, axis=1, keepdims=True) + 1e-12)
        B = np.cross(face_normals_valid, e0_norm)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        
        # Get vertex normals for all valid faces at once
        n0 = vertex_normals[faces_valid[:, 0]]
        n1 = vertex_normals[faces_valid[:, 1]]
        n2 = vertex_normals[faces_valid[:, 2]]
        
        # Vectorized least squares system construction
        # A matrix: [n_valid, 6, 3] for all faces
        A_batch = np.zeros((n_valid, 6, 3))
        
        # Fill A matrices using broadcasting
        e0_t = np.einsum('ij,ij->i', e0_valid, e0_norm)  # dot products
        e0_B = np.einsum('ij,ij->i', e0_valid, B)
        e1_t = np.einsum('ij,ij->i', e1_valid, e0_norm)
        e1_B = np.einsum('ij,ij->i', e1_valid, B)
        e2_valid = -(e0_valid + e1_valid)  # e2 = -(e0 + e1)
        e2_t = np.einsum('ij,ij->i', e2_valid, e0_norm)
        e2_B = np.einsum('ij,ij->i', e2_valid, B)
        
        A_batch[:, 0, 0] = e0_t
        A_batch[:, 0, 1] = e0_B
        A_batch[:, 1, 1] = e0_t
        A_batch[:, 1, 2] = e0_B
        A_batch[:, 2, 0] = e1_t
        A_batch[:, 2, 1] = e1_B
        A_batch[:, 3, 1] = e1_t
        A_batch[:, 3, 2] = e1_B
        A_batch[:, 4, 0] = e2_t
        A_batch[:, 4, 1] = e2_B
        A_batch[:, 5, 1] = e2_t
        A_batch[:, 5, 2] = e2_B
        
        # b vector: [n_valid, 6]
        b_batch = np.zeros((n_valid, 6))
        
        # Normal differences
        dn_01 = n2 - n1
        dn_12 = n0 - n2  
        dn_20 = n1 - n0
        
        b_batch[:, 0] = np.einsum('ij,ij->i', dn_01, e0_norm)
        b_batch[:, 1] = np.einsum('ij,ij->i', dn_01, B)
        b_batch[:, 2] = np.einsum('ij,ij->i', dn_12, e0_norm)
        b_batch[:, 3] = np.einsum('ij,ij->i', dn_12, B)
        b_batch[:, 4] = np.einsum('ij,ij->i', dn_20, e0_norm)
        b_batch[:, 5] = np.einsum('ij,ij->i', dn_20, B)
        
        # Batch least squares solve
        try:
            # Use numpy's vectorized least squares
            x_batch = np.linalg.lstsq(A_batch.transpose(0, 2, 1) @ A_batch, 
                                    A_batch.transpose(0, 2, 1) @ b_batch[:, :, np.newaxis], 
                                    rcond=self._lstsq_rcond)[0].squeeze()
        except np.linalg.LinAlgError:
            # Fallback to individual solving for problematic cases
            x_batch = np.zeros((n_valid, 3))
            for i in range(n_valid):
                try:
                    x_batch[i] = np.linalg.lstsq(A_batch[i], b_batch[i], rcond=self._lstsq_rcond)[0]
                except np.linalg.LinAlgError:
                    x_batch[i] = np.zeros(3)
        
        # Build tensors from solutions
        face_tensors = np.zeros((n_faces, 2, 2))
        face_tensors[valid_faces, 0, 0] = x_batch[:, 0]  # ku
        face_tensors[valid_faces, 0, 1] = x_batch[:, 1]  # kuv
        face_tensors[valid_faces, 1, 0] = x_batch[:, 1]  # kuv (symmetric)
        face_tensors[valid_faces, 1, 1] = x_batch[:, 2]  # kv
        
        return face_tensors
    
    def _accumulate_vertex_tensors(self, face_tensors, corner_areas, vertex_areas, 
                                        up_vectors, vp_vectors, face_normals):
        """Sparse matrix-based accumulation of face tensors to vertices."""
        n_vertices = len(self.vertices)
        n_faces = len(self.faces)
        
        # Use cached edge vectors
        e0, e1, e2 = self._edge_vectors
        
        # Vectorized face coordinate systems
        e0_norm = e0 / (np.linalg.norm(e0, axis=1, keepdims=True) + 1e-12)
        B = np.cross(face_normals, e0_norm)
        B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
        
        # Filter valid faces and vertices
        valid_faces = self._face_areas >= self._min_triangle_area
        # Check for nonzero tensors by looking at the Frobenius norm
        tensor_norms = np.linalg.norm(face_tensors.reshape(n_faces, -1), axis=1)
        nonzero_tensors = tensor_norms > 1e-12
        valid_mask = valid_faces & nonzero_tensors
        
        if not np.any(valid_mask):
            return np.zeros((n_vertices, 2, 2))
        
        # Work with valid data only
        valid_face_indices = np.where(valid_mask)[0]
        n_valid = len(valid_face_indices)
        
        # Prepare data for sparse operations
        # Each face contributes to 3 vertices, so we have 3*n_valid contributions
        vertex_indices = self.faces[valid_mask].ravel()  # Shape: (3*n_valid,)
        face_indices_expanded = np.repeat(valid_face_indices, 3)  # Shape: (3*n_valid,)
        corner_indices = np.tile([0, 1, 2], n_valid)  # Shape: (3*n_valid,)
        
        # Weights for accumulation
        weights = corner_areas[valid_mask].ravel()  # Shape: (3*n_valid,)
        vertex_area_weights = vertex_areas[vertex_indices]
        weights = weights / (vertex_area_weights + 1e-12)
        
        # Filter out zero weights
        nonzero_weights = weights > 1e-12
        if not np.any(nonzero_weights):
            return np.zeros((n_vertices, 2, 2))
        
        vertex_indices = vertex_indices[nonzero_weights]
        face_indices_expanded = face_indices_expanded[nonzero_weights]
        corner_indices = corner_indices[nonzero_weights]
        weights = weights[nonzero_weights]
        
        # Vectorized tensor transformations
        # For each contribution, we need to transform the face tensor to vertex coordinates
        n_contributions = len(vertex_indices)
        transformed_tensors = np.zeros((n_contributions, 2, 2))
        
        for i, (v_idx, f_idx, c_idx) in enumerate(zip(vertex_indices, face_indices_expanded, corner_indices)):
            # Get face tensor
            face_tensor = face_tensors[f_idx]
            
            # Get coordinate systems
            uf = e0_norm[f_idx]
            vf = B[f_idx]
            nf = face_normals[f_idx]
            up = up_vectors[v_idx]
            vp = vp_vectors[v_idx]
            
            # Transform tensor
            transformed_tensors[i] = self._project_curvature_tensor(uf, vf, nf, face_tensor, up, vp)
        
        # Accumulate using sparse matrix
        # Create sparse matrix for tensor accumulation
        vertex_tensors = np.zeros((n_vertices, 2, 2))
        
        # Manual accumulation (could be further optimized with sparse tensors)
        for i, v_idx in enumerate(vertex_indices):
            vertex_tensors[v_idx] += weights[i] * transformed_tensors[i]
        
        return vertex_tensors
    
    def _project_curvature_tensor(self, uf, vf, nf, old_tensor, up, vp):
        """Project curvature tensor from face to vertex coordinate system."""
        # Rotate coordinate system
        r_new_u, r_new_v = self._rotate_coordinate_system(up, vp, nf)
        
        # Transformation coefficients
        u1 = np.dot(r_new_u, uf)
        v1 = np.dot(r_new_u, vf)
        u2 = np.dot(r_new_v, uf)
        v2 = np.dot(r_new_v, vf)
        
        # Apply tensor transformation
        old_ku = old_tensor[0, 0]
        old_kuv = old_tensor[0, 1]
        old_kv = old_tensor[1, 1]
        
        new_ku = old_ku * u1**2 + 2 * old_kuv * u1 * v1 + old_kv * v1**2
        new_kuv = old_ku * u1 * u2 + old_kuv * (u1 * v2 + u2 * v1) + old_kv * v1 * v2
        new_kv = old_ku * u2**2 + 2 * old_kuv * u2 * v2 + old_kv * v2**2
        
        return np.array([[new_ku, new_kuv], [new_kuv, new_kv]])
    
    def _rotate_coordinate_system(self, up, vp, nf):
        """Rotate coordinate system to align with face normal."""
        np_vertex = np.cross(up, vp)
        np_vertex = np_vertex / (np.linalg.norm(np_vertex) + 1e-12)
        
        ndot = np.dot(nf, np_vertex)
        
        if ndot <= -1:
            return -up, -vp
        
        perp = nf - ndot * np_vertex
        dperp = (np_vertex + nf) / (1 + ndot)
        
        r_new_u = up - dperp * np.dot(perp, up)
        r_new_v = vp - dperp * np.dot(perp, vp)
        
        return r_new_u, r_new_v
    
    def _extract_principal_curvatures(self, vertex_tensors, up_vectors, vp_vectors, vertex_normals):
        """Vectorized extraction of principal curvatures and directions."""

        n_vertices = len(self.vertices)

        # Extract tensor components
        ku = vertex_tensors[:, 0, 0]
        kuv = vertex_tensors[:, 0, 1]
        kv = vertex_tensors[:, 1, 1]
        
        # Analytical eigenvalues for 2x2 symmetric matrix
        trace = ku + kv  # k1 + k2
        det = ku * kv - kuv**2  # k1 * k2
        discriminant = trace**2 - 4 * det
        discriminant = np.maximum(discriminant, 0)  # Numerical stability
        
        sqrt_disc = np.sqrt(discriminant)
        k1 = 0.5 * (trace + sqrt_disc)
        k2 = 0.5 * (trace - sqrt_disc)
        
        # Ensure k1 >= k2 (convention)
        swap_mask = k1 < k2
        k1[swap_mask], k2[swap_mask] = k2[swap_mask], k1[swap_mask]
        
        principal_curvatures = np.column_stack([k1, k2])
        
        # Vectorized principal direction computation
        principal_directions = np.zeros((n_vertices, 3, 2))
        
        # Compute eigenvectors analytically where possible
        nonzero_kuv = np.abs(kuv) > 1e-12
        
        # For vertices with non-zero off-diagonal elements
        if np.any(nonzero_kuv):
            # First eigenvector direction in local 2D coordinates
            v1_2d = np.zeros((n_vertices, 2))
            v2_2d = np.zeros((n_vertices, 2))
            
            # Case 1: kuv != 0
            v1_2d[nonzero_kuv, 0] = k1[nonzero_kuv] - kv[nonzero_kuv]
            v1_2d[nonzero_kuv, 1] = kuv[nonzero_kuv]
            
            v2_2d[nonzero_kuv, 0] = k2[nonzero_kuv] - kv[nonzero_kuv]
            v2_2d[nonzero_kuv, 1] = kuv[nonzero_kuv]
            
            # Case 2: kuv == 0 (already diagonal)
            zero_kuv = ~nonzero_kuv
            v1_2d[zero_kuv, 0] = 1.0
            v1_2d[zero_kuv, 1] = 0.0
            v2_2d[zero_kuv, 0] = 0.0
            v2_2d[zero_kuv, 1] = 1.0
            
            # Normalize 2D vectors
            norms_v1 = np.linalg.norm(v1_2d, axis=1, keepdims=True)
            norms_v2 = np.linalg.norm(v2_2d, axis=1, keepdims=True)
            v1_2d = v1_2d / (norms_v1 + 1e-12)
            v2_2d = v2_2d / (norms_v2 + 1e-12)
        else:
            # All matrices are diagonal
            v1_2d = np.zeros((n_vertices, 2))
            v2_2d = np.zeros((n_vertices, 2))
            v1_2d[:, 0] = 1.0
            v2_2d[:, 1] = 1.0
        
        # Transform from 2D local coordinates to 3D world coordinates
        # Need to rotate local coordinate system to align with vertex normal
        for i in range(n_vertices):
            # Get rotated coordinate system
            r_old_u, r_old_v = self._rotate_coordinate_system(
                up_vectors[i], vp_vectors[i], vertex_normals[i]
            )
            
            # Transform eigenvectors to 3D
            dir1 = v1_2d[i, 0] * r_old_u + v1_2d[i, 1] * r_old_v
            dir2 = v2_2d[i, 0] * r_old_u + v2_2d[i, 1] * r_old_v
            
            # Ensure orthogonality and proper orientation
            dir1 = dir1 / (np.linalg.norm(dir1) + 1e-12)
            dir2 = np.cross(vertex_normals[i], dir1)
            dir2 = dir2 / (np.linalg.norm(dir2) + 1e-12)
            
            principal_directions[i, :, 0] = dir1
            principal_directions[i, :, 1] = dir2
        
        return principal_curvatures, principal_directions

    def _build_vertex_neighborhoods(self, kernel_rings=1):
        """
        Build vertex neighborhoods using edge connectivity.
        
        Parameters
        ----------
        kernel_rings : int, default=1
            Number of rings (1-ring = immediate neighbors, 2-ring = neighbors of neighbors, etc.)
        
        Returns
        -------
        neighborhoods : list of np.ndarray
            neighborhoods[i] is a numpy array of vertex indices in the neighborhood of vertex i
        """
        n_vertices = len(self.vertices)
        
        # Build vertex-to-vertex connectivity from faces
        vertex_neighbors = [set() for _ in range(n_vertices)]
        for face in self.faces:
            v0, v1, v2 = face[0], face[1], face[2]
            vertex_neighbors[v0].add(v1)
            vertex_neighbors[v0].add(v2)
            vertex_neighbors[v1].add(v0)
            vertex_neighbors[v1].add(v2)
            vertex_neighbors[v2].add(v0)
            vertex_neighbors[v2].add(v1)
        
        # BFS to build k-ring neighborhoods
        neighborhoods = [set() for _ in range(n_vertices)]
        
        for v_idx in range(n_vertices):
            visited = {v_idx}
            current_ring = {v_idx}
            
            for ring in range(kernel_rings + 1):  # +1 to include the vertex itself
                neighborhoods[v_idx].update(current_ring)
                
                if ring < kernel_rings:
                    next_ring = set()
                    for v in current_ring:
                        next_ring.update(vertex_neighbors[v] - visited)
                    visited.update(next_ring)
                    current_ring = next_ring
        
        # Convert sets to numpy arrays for faster iteration
        neighborhoods_arrays = [np.array(sorted(neighbors), dtype=np.int32) for neighbors in neighborhoods]
        
        return neighborhoods_arrays
    
    @staticmethod
    def _project_curvature_tensor_static(uf, vf, nf, old_tensor, up, vp):
        """
        Static version of _project_curvature_tensor for use in multiprocessing.
        Project curvature tensor from face to vertex coordinate system.
        """
        # Rotate coordinate system (using static version)
        r_new_u, r_new_v = Mesh._rotate_coordinate_system_static(up, vp, nf)
        
        # Transformation coefficients
        u1 = np.dot(r_new_u, uf)
        v1 = np.dot(r_new_u, vf)
        u2 = np.dot(r_new_v, uf)
        v2 = np.dot(r_new_v, vf)
        
        # Apply tensor transformation
        old_ku = old_tensor[0, 0]
        old_kuv = old_tensor[0, 1]
        old_kv = old_tensor[1, 1]
        
        new_ku = old_ku * u1**2 + 2 * old_kuv * u1 * v1 + old_kv * v1**2
        new_kuv = old_ku * u1 * u2 + old_kuv * (u1 * v2 + u2 * v1) + old_kv * v1 * v2
        new_kv = old_ku * u2**2 + 2 * old_kuv * u2 * v2 + old_kv * v2**2
        
        return np.array([[new_ku, new_kuv], [new_kuv, new_kv]])
    
    @staticmethod
    def _rotate_coordinate_system_static(up, vp, nf):
        """Static version of _rotate_coordinate_system for use in multiprocessing."""
        np_vertex = np.cross(up, vp)
        np_vertex = np_vertex / (np.linalg.norm(np_vertex) + 1e-12)
        
        ndot = np.dot(nf, np_vertex)
        
        if ndot <= -1:
            return -up, -vp
        
        perp = nf - ndot * np_vertex
        dperp = (np_vertex + nf) / (1 + ndot)
        
        r_new_u = up - dperp * np.dot(perp, up)
        r_new_v = vp - dperp * np.dot(perp, vp)
        
        return r_new_u, r_new_v
    
    @staticmethod
    def _process_vertex(args):
        """
        Vectorized processing of batch of vertices for tensor smoothing (worker function for multiprocessing).
        
        Parameters
        ----------
        args : tuple
            (vertex_indices, smoothed_tensors, neighborhoods, up_vectors, vp_vectors, vertex_normals)
            where neighborhoods is a list of numpy arrays
        
        Returns
        -------
        tuple
            (vertex_indices, new_tensors for those vertices)
        """
        vertex_indices, smoothed_tensors, neighborhoods, up_vectors, vp_vectors, vertex_normals = args
        
        batch_new_tensors = np.zeros((len(vertex_indices), 2, 2))
        
        for i, v_idx in enumerate(vertex_indices):
            neighbor_array = neighborhoods[v_idx]  # Already a numpy array
            n_neighbors = len(neighbor_array)
            
            if n_neighbors == 0:
                batch_new_tensors[i] = smoothed_tensors[v_idx]
                continue
            
            # Pre-allocate array for transformed tensors (instead of list)
            transformed_tensors = np.zeros((n_neighbors, 2, 2))
            transformed_count = 0
            
            # Get target vertex coordinate system (compute once per vertex)
            up_target = up_vectors[v_idx]
            vp_target = vp_vectors[v_idx]
            nf_target = vertex_normals[v_idx]
            
            # Pre-compute target coordinate system rotation (once per vertex)
            np_vertex = np.cross(up_target, vp_target)
            np_vertex_norm = np.linalg.norm(np_vertex)
            if np_vertex_norm > 1e-12:
                np_vertex /= np_vertex_norm
            else:
                # Fallback
                batch_new_tensors[i] = smoothed_tensors[v_idx]
                continue
                
            ndot = np.dot(nf_target, np_vertex)
            if ndot <= -1:
                r_new_u_target = -up_target
                r_new_v_target = -vp_target
            else:
                perp = nf_target - ndot * np_vertex
                dperp = (np_vertex + nf_target) / (1 + ndot)
                r_new_u_target = up_target - dperp * np.dot(perp, up_target)
                r_new_v_target = vp_target - dperp * np.dot(perp, vp_target)
            
            # Process all neighbors (neighbor_array is already numpy array)
            for n_idx in neighbor_array:
                if n_idx == v_idx:
                    # Include the vertex itself
                    transformed_tensors[transformed_count] = smoothed_tensors[v_idx]
                    transformed_count += 1
                    continue
                
                # Get neighbor tensor and coordinate system
                neighbor_tensor = smoothed_tensors[n_idx]
                up_source = up_vectors[n_idx]
                vp_source = vp_vectors[n_idx]
                
                # Transform neighbor tensor to target vertex's coordinate system
                # Using pre-computed target rotation
                u1 = np.dot(r_new_u_target, up_source)
                v1 = np.dot(r_new_u_target, vp_source)
                u2 = np.dot(r_new_v_target, up_source)
                v2 = np.dot(r_new_v_target, vp_source)
                
                # Apply tensor transformation (vectorized operations)
                old_ku = neighbor_tensor[0, 0]
                old_kuv = neighbor_tensor[0, 1]
                old_kv = neighbor_tensor[1, 1]
                
                new_ku = old_ku * u1**2 + 2 * old_kuv * u1 * v1 + old_kv * v1**2
                new_kuv = old_ku * u1 * u2 + old_kuv * (u1 * v2 + u2 * v1) + old_kv * v1 * v2
                new_kv = old_ku * u2**2 + 2 * old_kuv * u2 * v2 + old_kv * v2**2
                
                transformed_tensors[transformed_count, 0, 0] = new_ku
                transformed_tensors[transformed_count, 0, 1] = new_kuv
                transformed_tensors[transformed_count, 1, 0] = new_kuv
                transformed_tensors[transformed_count, 1, 1] = new_kv
                transformed_count += 1
            
            # Average the transformed tensors (using only the filled portion)
            if transformed_count > 0:
                batch_new_tensors[i] = np.mean(transformed_tensors[:transformed_count], axis=0)
            else:
                batch_new_tensors[i] = smoothed_tensors[v_idx]
        
        return (vertex_indices, batch_new_tensors)

    def _smooth_curvature_tensors(self, vertex_tensors, up_vectors, vp_vectors, 
                                vertex_normals, smoothing_iterations, kernel_rings, n_jobs=None):
        """
        Smooth curvature tensors by averaging over vertex neighborhoods.
        
        Parameters
        ----------
        vertex_tensors : np.ndarray (N, 2, 2)
            Curvature tensors at each vertex
        up_vectors : np.ndarray (N, 3)
            Up vectors for coordinate systems
        vp_vectors : np.ndarray (N, 3)
            Vp vectors for coordinate systems
        vertex_normals : np.ndarray (N, 3)
            Vertex normals
        smoothing_iterations : int
            Number of smoothing passes
        kernel_rings : int
            Neighborhood size (1-ring, 2-ring, etc.)
        
        Returns
        -------
        smoothed_tensors : np.ndarray (N, 2, 2)
            Smoothed curvature tensors
        """
        n_vertices = len(self.vertices)
        smoothed_tensors = vertex_tensors.copy()
        
        # Build neighborhoods once (already returns numpy arrays)
        neighborhoods = self._build_vertex_neighborhoods(kernel_rings)
        
        # Determine if we should use parallel processing
        if n_jobs is None:
            n_jobs = mp.cpu_count()
        use_parallel = (n_jobs > 1 and n_vertices > 1000)
        
        # Use vectorized version for better performance
        process_func = Mesh._process_vertex if use_parallel else None
        
        # Create process pool once and reuse across iterations
        pool = None
        if use_parallel:
            pool = mp.Pool(processes=n_jobs)
        
        try:
            for iteration in range(smoothing_iterations):
                new_tensors = np.zeros_like(smoothed_tensors)
                
                if use_parallel:
                    # Parallel processing with vectorized function
                    chunk_size = max(1, n_vertices // (n_jobs * 4))
                    vertex_chunks = [list(range(i, min(i + chunk_size, n_vertices))) 
                                for i in range(0, n_vertices, chunk_size)]
                    
                    worker_args = [
                        (chunk, smoothed_tensors, neighborhoods, up_vectors, vp_vectors, vertex_normals)
                        for chunk in vertex_chunks
                    ]
                    
                    # Reuse the same pool across iterations
                    results = pool.map(process_func, worker_args)
                    
                    # Collect results
                    for vertex_indices, batch_tensors in results:
                        new_tensors[vertex_indices] = batch_tensors
                else:
                    # Sequential processing with vectorized approach
                    for v_idx in range(n_vertices):
                        neighbor_array = neighborhoods[v_idx]  # Already a numpy array
                        n_neighbors = len(neighbor_array)
                        
                        if n_neighbors == 0:
                            new_tensors[v_idx] = smoothed_tensors[v_idx]
                            continue
                        
                        # Pre-allocate array
                        transformed_tensors = np.zeros((n_neighbors, 2, 2))
                        transformed_count = 0
                        
                        # Get target coordinate system (once per vertex)
                        up_target = up_vectors[v_idx]
                        vp_target = vp_vectors[v_idx]
                        nf_target = vertex_normals[v_idx]
                        
                        # Pre-compute target rotation
                        np_vertex = np.cross(up_target, vp_target)
                        np_vertex_norm = np.linalg.norm(np_vertex)
                        if np_vertex_norm > 1e-12:
                            np_vertex /= np_vertex_norm
                        else:
                            new_tensors[v_idx] = smoothed_tensors[v_idx]
                            continue
                        
                        ndot = np.dot(nf_target, np_vertex)
                        if ndot <= -1:
                            r_new_u_target = -up_target
                            r_new_v_target = -vp_target
                        else:
                            perp = nf_target - ndot * np_vertex
                            dperp = (np_vertex + nf_target) / (1 + ndot)
                            r_new_u_target = up_target - dperp * np.dot(perp, up_target)
                            r_new_v_target = vp_target - dperp * np.dot(perp, vp_target)
                        
                        # Process neighbors (neighbor_array is already numpy array)
                        for n_idx in neighbor_array:
                            if n_idx == v_idx:
                                transformed_tensors[transformed_count] = smoothed_tensors[v_idx]
                                transformed_count += 1
                                continue
                            
                            neighbor_tensor = smoothed_tensors[n_idx]
                            up_source = up_vectors[n_idx]
                            vp_source = vp_vectors[n_idx]
                            
                            # Inline tensor transformation
                            u1 = np.dot(r_new_u_target, up_source)
                            v1 = np.dot(r_new_u_target, vp_source)
                            u2 = np.dot(r_new_v_target, up_source)
                            v2 = np.dot(r_new_v_target, vp_source)
                            
                            old_ku = neighbor_tensor[0, 0]
                            old_kuv = neighbor_tensor[0, 1]
                            old_kv = neighbor_tensor[1, 1]
                            
                            new_ku = old_ku * u1**2 + 2 * old_kuv * u1 * v1 + old_kv * v1**2
                            new_kuv = old_ku * u1 * u2 + old_kuv * (u1 * v2 + u2 * v1) + old_kv * v1 * v2
                            new_kv = old_ku * u2**2 + 2 * old_kuv * u2 * v2 + old_kv * v2**2
                            
                            transformed_tensors[transformed_count, 0, 0] = new_ku
                            transformed_tensors[transformed_count, 0, 1] = new_kuv
                            transformed_tensors[transformed_count, 1, 0] = new_kuv
                            transformed_tensors[transformed_count, 1, 1] = new_kv
                            transformed_count += 1
                        
                        # Average
                        if transformed_count > 0:
                            new_tensors[v_idx] = np.mean(transformed_tensors[:transformed_count], axis=0)
                        else:
                            new_tensors[v_idx] = smoothed_tensors[v_idx]
                
                smoothed_tensors = new_tensors
        
        finally:
            # Clean up process pool
            if pool is not None:
                pool.close()
                pool.join()
        
        return smoothed_tensors

class OrientedPointCloud(DiscreteSurface):
    """Oriented point clouds with vertices and normals."""
    
    def __init__(self):
        super().__init__()
        self.vertices = None  # N x 3 array
        self.normals = None   # N x 3 array
        self._motl = None
    
    def get_vertices(self):
        return self.vertices
    
    def get_normals(self):
        return self.normals

    def transform(self, transformation_matrix: np.ndarray):
        """Apply 4x4 transformation matrix to vertices and normals."""
        if transformation_matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must be 4x4")
        
        # Transform vertices (homogeneous coordinates)
        vertices_homo = np.hstack([self.vertices, np.ones((len(self.vertices), 1))])
        transformed_vertices = (transformation_matrix @ vertices_homo.T).T
        self.vertices = transformed_vertices[:, :3]
        
        # Transform normals (only rotation part, no translation)
        if self.normals is not None:
            rotation_matrix = transformation_matrix[:3, :3]
            self.normals = (rotation_matrix @ self.normals.T).T
            # Renormalize
            self.normals = self.normals / np.linalg.norm(self.normals, axis=1, keepdims=True)

    def translate(self, translation_vector: np.ndarray):
        """Translate all points."""
        translation_vector = np.asarray(translation_vector)
        self.vertices += translation_vector

    def translate_along_normals(self, distance: float, inplace: bool = False):
        """Translate points along their normal vectors. Default is to return a new object."""
        if inplace:
            self.vertices += distance * self.normals
            return None
        else:
            new_obj = copy.deepcopy(self)
            new_obj.vertices += distance * new_obj.normals
            return new_obj

    def rotate(self, rotation_matrix: np.ndarray):
        """
        Apply 3x3 rotation matrix to vertices and normals.
        
        Parameters
        ----------
        rotation_matrix : array_like (3, 3)
            Rotation matrix (should be orthogonal)
        """
        rotation_matrix = np.asarray(rotation_matrix)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrix must be 3x3")
        
        # Build 4x4 transformation matrix
        T = np.eye(4)
        T[:3, :3] = rotation_matrix
        
        return self.transform(T)

    def flip_normals(self, inplace: bool = True, **kwargs):
        """
        Reverse point-cloud normal directions.

        Parameters
        ----------
        inplace : bool, default=True
            If True, modify this point cloud. If False, return a flipped copy.

        Returns
        -------
        OrientedPointCloud or None
            New point cloud if ``inplace=False``, otherwise ``None``.
        """
        if kwargs:
            raise TypeError(
                f"OrientedPointCloud.flip_normals got unexpected keywords: {list(kwargs)!r}"
            )

        target = self if inplace else copy.deepcopy(self)
        if target.normals is None:
            raise ValueError("Cannot flip normals: point cloud has no normals")

        target.normals = -np.asarray(target.normals, dtype=float)
        target._motl = None
        if inplace:
            return None
        return target

    def remove_nonfinite_vertices(self, inplace: bool = True, recompute_normals: bool = False) -> "OrientedPointCloud":
        """
        Remove point samples with NaN or infinite coordinates.

        Parameters
        ----------
        inplace : bool, default=True
            If True, modify this point cloud. If False, return a repaired copy.
        recompute_normals : bool, default=False
            If True, recompute normals after filtering.

        Returns
        -------
        OrientedPointCloud
            The repaired point cloud.
        """
        target = self if inplace else copy.deepcopy(self)
        if target.vertices is None:
            return target

        vertices = np.asarray(target.vertices)
        finite_mask = np.isfinite(vertices).all(axis=1)
        if finite_mask.all():
            return target

        n_removed = int((~finite_mask).sum())
        target.vertices = vertices[finite_mask].copy()
        if target.normals is not None:
            target.normals = np.asarray(target.normals)[finite_mask].copy()
        target._motl = None
        print(f"Removed {n_removed} non-finite point-cloud samples")

        if recompute_normals:
            target.compute_normals(inplace=True)

        return target

    def extract_points(self, point_ids: np.ndarray | None = None, mask: np.ndarray | None = None) -> "OrientedPointCloud":
        """
        Return a new point cloud containing selected points.

        Provide either integer ``point_ids`` or a boolean ``mask`` with one value per point.
        Normals are copied when present.
        """
        if self.vertices is None:
            raise ValueError("Point cloud must have vertices")
        if (point_ids is None) == (mask is None):
            raise ValueError("Provide exactly one of point_ids or mask")

        n_points = len(self.vertices)
        if mask is not None:
            keep = np.asarray(mask, dtype=bool).reshape(-1)
            if keep.shape[0] != n_points:
                raise ValueError(f"mask length {keep.shape[0]} must match points {n_points}")
            indices = np.flatnonzero(keep)
        else:
            indices = np.asarray(point_ids, dtype=int).reshape(-1)
            if indices.size == 0:
                indices = np.array([], dtype=int)
            valid = (indices >= 0) & (indices < n_points)
            if not valid.all():
                warnings.warn(
                    f"Ignoring {int((~valid).sum())} point ids outside [0, {n_points})",
                    UserWarning,
                    stacklevel=2,
                )
            indices = np.unique(indices[valid])

        new_pcd = OrientedPointCloud()
        new_pcd.vertices = np.asarray(self.vertices)[indices].copy()
        if self.normals is not None:
            new_pcd.normals = np.asarray(self.normals)[indices].copy()
        new_pcd.inherit_coordinate_metadata(self)
        new_pcd._motl = None
        return new_pcd

    @staticmethod
    def _compute_and_orient_normals(pcd_o3d, knn: int = 30, orient_normals: bool = True, tangent_plane_knn: int = 50) -> np.ndarray:
        """
        Estimate and orient normals for an Open3D point cloud.
        
        Parameters
        ----------
        pcd_o3d : o3d.geometry.PointCloud
            Point cloud to process (modified in-place)
        knn : int, default=30
            Number of nearest neighbors for normal estimation
        orient_normals : bool, default=True
            Whether to orient normals
        tangent_plane_knn : int, default=50
            Number of neighbors for normal orientation
        
        Returns
        -------
        np.ndarray
            Nx3 array of normals
        """
        print(f"Estimating normals (knn={knn})...")
        pcd_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )
        
        if orient_normals:
            print(f"Orienting normals (knn={tangent_plane_knn})...")
            pcd_o3d.orient_normals_consistent_tangent_plane(k=tangent_plane_knn)
        
        return np.asarray(pcd_o3d.normals)

    def compute_normals(self, **kwargs):
        """
        Compute or recompute normals from point cloud geometry. Usecases:
        - Add normals to point clouds
        - Recompute normals with different parameters
        - Refine existing normals
        
        Parameters
        ----------
        knn : int, default=30
            Number of nearest neighbors for normal estimation
        orient_normals : bool, default=True
            Whether to orient normals consistently using tangent plane method
        tangent_plane_knn : int, default=50
            Number of neighbors for normal orientation
        inplace : bool, default=True
            If True, update normals in place. If False, return new instance.
        
        Returns
        -------
        OrientedPointCloud or None
            New instance if inplace=False, else None
        """
        knn = int(kwargs.pop("knn", 30))
        orient_normals = bool(kwargs.pop("orient_normals", True))
        tangent_plane_knn = int(kwargs.pop("tangent_plane_knn", 50))
        inplace = bool(kwargs.pop("inplace", True))
        kwargs.pop("refine", None)

        if kwargs:
            raise TypeError(
                f"OrientedPointCloud.compute_normals got unexpected keywords: {list(kwargs)!r}"
            )
        if self.vertices is None:
            raise ValueError("Cannot compute normals: no vertices present")
        
        # Create Open3D point cloud
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.vertices)
        
        # Estimate and orient normals
        normals = self._compute_and_orient_normals(
            pcd_o3d, knn, orient_normals, tangent_plane_knn
        )
        
        if inplace:
            self.normals = normals
            print(f"Updated normals for {len(self.vertices)} points")
            return None
        else:
            new_pcd = copy.deepcopy(self)
            new_pcd.normals = normals
            print(f"Created new point cloud with updated normals")
            return new_pcd

    @classmethod
    def read(cls, input_path: PathOrStr, recompute_normals: bool = False, knn: int = 30, 
                orient_normals: bool = True, tangent_plane_knn: int = 50) -> "OrientedPointCloud":
        """
        Load oriented point cloud from file (.ply, .xyz, .pcd, .pts).
        
        Automatically detects if normals are present in the file. If normals exist,
        they are used by default unless recompute_normals=True. If no normals are
        present, they are estimated from geometry.
        
        Parameters
        ----------
        input_path : str
            Path to point cloud file. Supported formats: .ply, .xyz, .pcd, .pts
        recompute_normals : bool, default=False
            If True, recompute normals from geometry even if file contains normals.
            If False and file has normals, use them; if no normals, compute them.
        knn : int, default=30
            Number of nearest neighbors for normal estimation
        orient_normals : bool, default=True
            Whether to orient normals consistently using tangent plane method
        tangent_plane_knn : int, default=50
            Number of neighbors for consistent normal orientation
            
        Returns
        -------
        OrientedPointCloud
            Point cloud with vertices and normals
            
        Examples
        --------
        # Load PLY with existing normals
        >>> pcd = OrientedPointCloud.read('data.ply')
        
        # Load XYZ without normals (will auto-compute)
        >>> pcd = OrientedPointCloud.read('points.xyz', knn=50)
        
        # Force recompute normals even if they exist
        >>> pcd = OrientedPointCloud.read('data.ply', recompute_normals=True)
        """
        input_path = Path(input_path)
        
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_path}")
        
        # Supported formats
        supported_formats = {'.ply', '.xyz', '.pcd', '.pts'}
        if input_path.suffix.lower() not in supported_formats:
            raise ValueError(
                f"Unsupported format: {input_path.suffix}. "
                f"Supported: {', '.join(supported_formats)}"
            )
        
        print(f"Loading point cloud from {input_path.name}...")
        
        # Load using Open3D
        pcd_o3d = o3d.io.read_point_cloud(str(input_path))
        
        # Check if point cloud is empty
        if not pcd_o3d.has_points():
            raise ValueError(f"No points found in {input_path}")
        
        vertices = np.asarray(pcd_o3d.points)
        print(f"Loaded {len(vertices)} points")
        
        # Check for existing normals
        has_normals = pcd_o3d.has_normals()
        
        if has_normals and not recompute_normals:
            # Use existing normals from file
            normals = np.asarray(pcd_o3d.normals)
            print(f"Using existing normals from file")

        else:
            # Compute normals from geometry
            if has_normals and recompute_normals:
                print(f"File has normals, but recomputing from geometry...")
            else:
                print(f"No normals in file, computing from geometry...")
            
            normals = cls._compute_and_orient_normals(
                pcd_o3d, knn, orient_normals, tangent_plane_knn
            )
        
        # Create OrientedPointCloud instance
        oriented = cls()
        oriented.vertices = vertices
        oriented.normals = normals

        print(f"Successfully created OrientedPointCloud from {input_path.name}")
        
        return oriented
    
    @classmethod
    def from_mesh(cls, mesh: Mesh, recompute_normals: bool = False, knn: int = 30, 
                orient_normals: bool = True, tangent_plane_knn: int = 50) -> "OrientedPointCloud":
        """
        Create oriented point cloud from mesh.
        
        Parameters
        ----------
        mesh : Mesh
            Source mesh
        recompute_normals : bool, default=False
            If True, recompute normals from point geometry instead of copying from mesh
        knn, orient_normals, tangent_plane_knn : 
            Normal estimation parameters (if recompute_normals=True)
        
        Returns
        -------
        OrientedPointCloud
        """
        oriented = cls()
        oriented.vertices = mesh.get_vertices().copy()
        
        if recompute_normals:
            pcd_o3d = o3d.geometry.PointCloud()
            pcd_o3d.points = o3d.utility.Vector3dVector(oriented.vertices)
            
            oriented.normals = cls._compute_and_orient_normals(
                pcd_o3d, knn, orient_normals, tangent_plane_knn
            )
        else:
            oriented.normals = mesh.get_normals().copy()
        
        oriented.inherit_coordinate_metadata(mesh)
        return oriented

    @classmethod
    def from_motl(cls, input_path: PathOrStr, group_by_subtomo_id: bool = False, recompute_normals: bool = False, knn: int = 30, 
                orient_normals: bool = True, tangent_plane_knn: int = 50) -> "OrientedPointCloud" | dict[str, "OrientedPointCloud"]:
        """
        Load oriented point cloud directly from motl file.
        
        Parameters
        ----------
        input_path : str
            Path to motl file
        group_by_subtomo_id : bool, default=False
            If True and multiple subtomo_ids exist, returns dict of OrientedPointCloud 
            instances keyed by subtomo_id. If False, returns single OrientedPointCloud 
            with all points.
        recompute_normals : bool, default=False
            If True, recompute normals from geometry instead of using Euler angles.
        knn : int, default=30
            Number of neighbors for normal estimation (if recompute_normals=True)
        orient_normals : bool, default=True
            Whether to orient normals consistently (if recompute_normals=True)
        tangent_plane_knn : int, default=50
            Neighbors for orientation (if recompute_normals=True)
        
        Returns
        -------
        OrientedPointCloud or dict
            Single point cloud if group_by_subtomo_id=False, else dict keyed by subtomo_id
        """
        motl = cryomotl.Motl.load(input_path)
        
        # Get subtomo_ids
        subtomo_ids = motl.df['subtomo_id'].values
        unique_ids = np.unique(subtomo_ids)
        
        # Check if grouping is needed
        if group_by_subtomo_id and len(unique_ids) > 1:
            # Return dictionary of point clouds
            pcds = {}
            for obj_id in unique_ids:
                mask = (subtomo_ids == obj_id)
                
                oriented = cls()
                oriented.vertices = motl.get_coordinates()[mask]
                
                if recompute_normals:
                    # Compute normals from geometry
                    pcd_o3d = o3d.geometry.PointCloud()
                    pcd_o3d.points = o3d.utility.Vector3dVector(oriented.vertices)
                    
                    oriented.normals = cls._compute_and_orient_normals(
                        pcd_o3d, knn, orient_normals, tangent_plane_knn
                    )
                else:
                    # Convert angles to normals
                    angles = motl.get_angles()[mask]
                    oriented.normals = geom.euler_angles_to_normals(angles)
                    
                    # Normalize normals
                    magnitudes = np.linalg.norm(oriented.normals, axis=1, keepdims=True)
                    oriented.normals = oriented.normals / (magnitudes + 1e-8)
                
                pcds[obj_id] = oriented
            
            return pcds
        else:
            # Return single point cloud (original behavior)
            oriented = cls()
            oriented.vertices = motl.get_coordinates()
            
            if recompute_normals:
                # Compute normals from geometry
                pcd_o3d = o3d.geometry.PointCloud()
                pcd_o3d.points = o3d.utility.Vector3dVector(oriented.vertices)
                
                oriented.normals = cls._compute_and_orient_normals(
                    pcd_o3d, knn, orient_normals, tangent_plane_knn
                )
            else:
                # Convert angles to normals
                angles = motl.get_angles()
                oriented.normals = geom.euler_angles_to_normals(angles)
                
                # Normalize normals
                magnitudes = np.linalg.norm(oriented.normals, axis=1, keepdims=True)
                oriented.normals = oriented.normals / (magnitudes + 1e-8)
            
            return oriented

    @classmethod
    def from_mrc(cls, input_path: PathOrStr, labels_dict: dict[str, int] | None = None, pixel_size: float | np.ndarray | None = None, 
                compute_normals: bool = True, knn: int = 30, orient_normals: bool = True, 
                tangent_plane_knn: int = 50, transpose: bool = True, smooth_sigma: float | None = None) -> "OrientedPointCloud" | dict[str, "OrientedPointCloud"]:
        """
        Create oriented point cloud from MRC segmentation file.
        
        Extracts coordinates from segmented pixel/voxels and estimates normals using
        local geometry. Supports both single binary segmentations and multi-label
        segmentations.

       Coordinate System Conventions:
        
        1. Physical Units (for software expecting real-world coordinates)
            - Set `pixel_size` to the physical pixel/voxel size (e.g., `pixel_size=0.7884` for 0.7884 nm/pixel)
            - Set `sampling_distance` in the same physical units (e.g., `sampling_distance=0.55` for 0.55 nm)
            - Output coordinates are in physical units (nm)

        2. Pixel/voxel Units (for software expecting pixel_size=1)
            - Set `pixel_size=1` (normalized)
            - Set `sampling_distance` in pixel/voxel units (e.g., scaling factor `sampling_distance=55/7.84`)
            - Output coordinates are in pixel/voxel units
        
        Parameters
        ----------
        input_path : str
            Path to MRC segmentation file
        labels_dict : dict, optional
            Dictionary mapping membrane names to label values.
            If None, treats as binary segmentation (any non-zero pixel/voxel).
            Example: {'IMM': 1, 'OMM': 2, 'cristae': 3}
        pixel_size : float or array-like, optional
            Pixel/voxel size for coordinate scaling.
            - If float: Physical pixel/voxel size in nm (e.g., 0.7884). Coordinates will be in physical units.
            - If 1: Normalized pixel/voxel units. Coordinates will be in pixel/voxel units.
            - If array-like: [x, y, z] spacing
            - If None: Defaults to 1.0 (pixel/voxel units)
        compute_normals : bool, default=True
            Whether to estimate normals from local geometry
        knn : int, default=30
            Number of nearest neighbors for normal estimation
        orient_normals : bool, default=True
            Whether to orient normals consistently using tangent plane method
        tangent_plane_knn : int, default=50
            Number of neighbors for consistent normal orientation
        smooth_sigma : float, optional
            Gaussian smoothing sigma (in pixel/voxel units) to apply before extracting points.
        transpose : bool, default=True
            Whether to transpose MRC data when loading (depends on coordinate convention)

        Notes
        -----
        Loads only through :func:`cryocat.core.cryomap.read`. Repair invalid MRC headers on disk
        if reading fails.

        Returns
        -------
        OrientedPointCloud or dict
            Single point cloud if labels_dict is None, else dict of point clouds by label name
        """
        
        segmentation = cryomap.read(input_path, transpose=transpose)

        # Apply Gaussian smoothing 
        if smooth_sigma is not None:
            print(f"Applying Gaussian smoothing (sigma={smooth_sigma})...")
            
            # Check if this is a multi-label segmentation
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels != 0]
            
            if len(unique_labels) > 1:
                # Multi-label: smooth each label separately to preserve labels
                smoothed_seg = np.zeros_like(segmentation, dtype=segmentation.dtype)
                for label in unique_labels:
                    label_mask = (segmentation == label).astype(float)
                    label_smoothed = scipy.ndimage.gaussian_filter(label_mask, sigma=smooth_sigma)
                    # Assign label to pixel/voxels where smoothed value > 0.5
                    smoothed_seg[label_smoothed > 0.5] = label
                segmentation = smoothed_seg
                print(f"Smoothed multi-label segmentation ({len(unique_labels)} labels)")
            else:
                # Binary segmentation: use existing behavior
                segmentation = scipy.ndimage.gaussian_filter(segmentation.astype(float), sigma=smooth_sigma)
                segmentation = (segmentation > 0.5).astype(float)
                print(f"Smoothed and re-thresholded binary segmentation")
        
        # Parse pixel/voxel size
        if pixel_size is not None:
            if np.isscalar(pixel_size):
                pixel_size = np.array([pixel_size, pixel_size, pixel_size])
            else:
                pixel_size = np.asarray(pixel_size)
        else:
            pixel_size = np.array([1.0, 1.0, 1.0])

        if labels_dict is None:
            # Check if segmentation has multiple labels
            unique_labels = np.unique(segmentation)
            unique_labels = unique_labels[unique_labels != 0]  # Remove background
            
            if len(unique_labels) > 1:
                # Auto-detect: treat as multi-label
                pcds = {}
                for label in unique_labels:
                    label_mask = (segmentation == label)
                    if label_mask.sum() > 0:
                        pcd = cls._create_pcd_from_seg(
                            label_mask, pixel_size,
                            knn, orient_normals, tangent_plane_knn
                        )
                        # Store label in metadata
                        if not hasattr(pcd, 'metadata'):
                            pcd.metadata = {}
                        pcd.metadata['label'] = int(label)
                        pcds[label] = pcd
                return pcds
            else:
                return cls._create_pcd_from_seg(
                    segmentation, pixel_size, 
                    knn, orient_normals, tangent_plane_knn
                )
        else:
            # Multiple labels
            pcds = {}
            for membrane_name, label_value in labels_dict.items():
                membrane_mask = (segmentation == label_value)
                if membrane_mask.sum() > 0:  # Only create point cloud if label exists
                    print(f"\nProcessing label '{membrane_name}' (value={label_value})")
                    pcd = cls._create_pcd_from_seg(
                        membrane_mask, pixel_size,
                        knn, orient_normals, tangent_plane_knn
                    )
                    pcds[membrane_name] = pcd
                else:
                    print(f"Warning: No pixels/voxels found for label '{membrane_name}' (value={label_value})")
            return pcds

    @classmethod
    def _create_pcd_from_seg(cls, segmentation: np.ndarray, pixel_size: float | np.ndarray,
                            knn: int = 30, orient_normals: bool = True, tangent_plane_knn: int = 50) -> "OrientedPointCloud":
        """Create oriented point cloud from binary segmentation volume."""
        # Extract point coordinates from segmentation
        print(f"Extracting points from segmentation (shape={segmentation.shape})...")
        points = np.argwhere(segmentation > 0)
        
        if len(points) == 0:
            raise ValueError("No non-zero pixels/voxels found in segmentation")
        
        print(f"Found {len(points)} points")
        
        # Scale to world coordinates
        points_world = points * pixel_size
        
        # Create OrientedPointCloud instance
        oriented_pcd = cls()
        oriented_pcd.vertices = points_world
        
        # Estimate normals
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(points_world)
            
        oriented_pcd.normals = cls._compute_and_orient_normals(
            pcd_o3d, knn, orient_normals, tangent_plane_knn
        )
        oriented_pcd.pixel_size = cls._coerce_pixel_size_array(pixel_size)
        
        print(f"Created oriented point cloud: {len(oriented_pcd.vertices)} points")
        
        return oriented_pcd
    
    def as_motl(self):
        """Get motl representation for motl-specific operations."""
        if self._motl is None:
            self._motl = self._create_motl()
        return self._motl
    
    def _to_open3d(self):
        """Convert to Open3D PointCloud."""
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.vertices)
        if self.normals is not None:
            pcd_o3d.normals = o3d.utility.Vector3dVector(self.normals)
        return pcd_o3d

    def save(self, output_path: PathOrStr, format: str | None = None, **kwargs):
        """
        Save oriented point cloud to a supported output format.

        Parameters
        ----------
        output_path : str or Path
            Output path.
        format : str, optional
            Output format. If None, inferred from the file suffix. Supported values:
            - 'ply': point cloud geometry (with normals when present)
            - 'motl' or 'em': motive list using normals as orientations
        **kwargs
            Forwarded to the selected format helper.
        """
        output_path = Path(output_path)
        fmt = output_path.suffix.lower().lstrip(".") if format is None else str(format).lower()

        if fmt == "ply":
            return self._save_as_ply(output_path, **kwargs)
        if fmt in ("motl", "em"):
            return self._save_as_motl(output_path, **kwargs)
        raise ValueError(
            f"Unsupported point-cloud save format '{fmt}'. Use 'ply', 'motl', or 'em'."
        )

    def _save_as_motl(self, output_path: PathOrStr, input_dict: dict[str, Any] | None = None, subtomo_ids: np.ndarray | None = None, tomo_id: int | float | np.ndarray | None = None):
        """
        Save oriented point cloud as motl file.
        
        Parameters
        ----------
        output_path : str
            Output path for motl file
        input_dict : dict, optional
            Additional motl columns to fill
        subtomo_ids : array-like, optional
            Subtomo IDs for each point.
            If None, sequential IDs are assigned (1, 2, 3, ...).
            Shape should be (N,) where N is the number of vertices.
        tomo_id : int, float, or array-like, optional
            Tomogram ID for all points or per point.
            - If scalar (int or float): Same tomo_id for all points
            - If array-like: Different tomo_id for each point (shape should be (N,))
            - If None: tomo_id defaults to 0.0 (standard motl default)
        """
        # Convert normals to Euler angles (zxz convention)
        angles = geom.normals_to_euler_angles(self.normals)
        
        # Create motl
        motl = cryomotl.Motl()
        
        # Fill coordinates and angles
        motl.fill({"coord": self.vertices})
        motl.fill({"angles": angles})  # phi, theta, psi
        
        # Set default values
        motl.df["class"] = 1.0
        
        # Assign subtomo_id based on object grouping or sequential
        if subtomo_ids is not None:
            subtomo_ids = np.asarray(subtomo_ids)
            if len(subtomo_ids) != len(self.vertices):
                raise ValueError(
                    f"subtomo_ids length ({len(subtomo_ids)}) must match number of vertices ({len(self.vertices)})"
                )
            
            # Map unique object IDs to sequential subtomo_ids (1, 2, 3, ...)
            unique_objects, inverse_indices = np.unique(subtomo_ids, return_inverse=True)
            # Assign sequential subtomo_id starting from 1 for each unique object
            motl.df["subtomo_id"] = (inverse_indices + 1).astype(float)
        else:
            # Default: sequential IDs for each point
            motl.df["subtomo_id"] = np.arange(1, len(self.vertices) + 1, dtype=float)
        
        # Assign tomo_id
        if tomo_id is not None:
            tomo_id_array = np.asarray(tomo_id)
            if tomo_id_array.ndim == 0:
                # Scalar: same tomo_id for all points
                motl.df["tomo_id"] = float(tomo_id_array)
            else:
                # Array: different tomo_id per point
                if len(tomo_id_array) != len(self.vertices):
                    raise ValueError(
                        f"tomo_id length ({len(tomo_id_array)}) must match number of vertices ({len(self.vertices)})"
                    )
                motl.df["tomo_id"] = tomo_id_array.astype(float)
        
        # Fill additional columns
        if input_dict is not None:
            # Prevent overwriting coordinates, angles, and now tomo_id/subtomo_id if set
            forbidden_keys = {"x", "y", "z", "coord", "phi", "theta", "psi", "angles"}
            if tomo_id is not None:
                forbidden_keys.add("tomo_id")
            if subtomo_ids is not None:
                forbidden_keys.add("subtomo_id")
            
            if forbidden_keys.intersection(input_dict.keys()):
                raise ValueError(
                    "Coordinates, angles, tomo_id, and subtomo_id are set by dedicated parameters. "
                    "Cannot override these fields via input_dict."
                )
            motl.fill(input_dict)
        
        # Write out
        motl.write_out(output_path)
        print(f"Saved {len(self.vertices)} points with normals to {output_path}")
    

    def _save_as_ply(self, output_path: PathOrStr, write_ascii: bool = False):
        """
        Save oriented point cloud as PLY file.
        
        Parameters
        ----------
        output_path : str
            Output path for PLY file
        write_ascii : bool, default=False
            Whether to write in ASCII format (True) or binary (False)
        """
        pcd_o3d = o3d.geometry.PointCloud()
        pcd_o3d.points = o3d.utility.Vector3dVector(self.vertices)
        if self.normals is not None:
            pcd_o3d.normals = o3d.utility.Vector3dVector(self.normals)
        
        o3d.io.write_point_cloud(str(output_path), pcd_o3d, write_ascii=write_ascii)
        print(f"Saved point cloud with normals to {output_path}")

    def convex_hull(self):
        """
        Compute convex hull of the point cloud.

        Returns
        -------
        hull : open3d.geometry.TriangleMesh
            The convex hull mesh.
        stats : dict
            Hull statistics: volume, surface area, vertex/face counts, and
            bounding-box extent.
        """
        pcd_o3d = self._to_open3d()
        hull, _ = pcd_o3d.compute_convex_hull()
        
        # Compute hull statistics
        hull_vertices = np.asarray(hull.vertices)
        hull_faces = np.asarray(hull.triangles)
        
        # Volume and surface area
        volume = hull.get_volume()
        surface_area = hull.get_surface_area()
        
        # Bounding box dimensions
        bbox = hull.get_axis_aligned_bounding_box()
        bbox_extent = bbox.get_extent()
        
        stats = {
            'volume': volume,
            'surface_area': surface_area,
            'hull_vertices': len(hull_vertices),
            'hull_faces': len(hull_faces),
            'bounding_box_extent': bbox_extent,
        }
        
        return hull, stats

    def crop(self, bbox, inplace: bool = False) -> "OrientedPointCloud" | None:
        """
        Crop point cloud to bounding box.
        
        Parameters
        ----------
        bbox : o3d.geometry.AxisAlignedBoundingBox or dict
            Bounding box for cropping. If dict, should have 'min_bound' and 'max_bound' keys
        inplace : bool, default=False
            If True, modify in place. If False, return new instance
        
        Returns
        -------
        OrientedPointCloud or None
            New instance if inplace=False, else None
        """
        pcd_o3d = self._to_open3d()
        bbox = _axis_aligned_bbox_from_input(bbox)
        cropped_o3d = pcd_o3d.crop(bbox)
        
        if inplace:
            self.vertices = np.asarray(cropped_o3d.points)
            if cropped_o3d.has_normals():
                self.normals = np.asarray(cropped_o3d.normals)
            print(f"Cropped to {len(self.vertices)} points")
            return None
        else:
            new_pcd = copy.deepcopy(self)
            new_pcd.vertices = np.asarray(cropped_o3d.points)
            if cropped_o3d.has_normals():
                new_pcd.normals = np.asarray(cropped_o3d.normals)
            print(f"Created cropped point cloud with {len(new_pcd.vertices)} points")
            return new_pcd

    def distance_to_points(
        self,
        target: np.ndarray | list[float],
        return_closest_points: bool = False,
    ) -> dict[str, Any]:
        """
        Nearest-neighbor distance from query points to this point cloud (unsigned only).

        Occupancy and signed-distance queries are not supported; use :class:`Mesh` for those.
        """
        target = np.atleast_2d(np.asarray(target, dtype=np.float32))
        result: dict[str, Any] = {"n_total": len(target)}
        tree = KDTree(self.vertices)
        distances, nearest_indices = tree.query(target)
        result["distances"] = distances
        result["distance_type"] = "unsigned"
        if return_closest_points:
            result["closest_points"] = self.vertices[nearest_indices]
            result["primitive_ids"] = nearest_indices
            result["closest_distances"] = distances
        return result

    def cast_rays(
        self,
        rays: np.ndarray,
        knn_radius: float = 10.0,
        one_hit_per_target: bool = False,
    ) -> dict[str, Any]:
        """
        Approximate ray intersection: KDTree-guided search along rays (not triangle ray casting).
        """
        rays = np.atleast_2d(np.asarray(rays, dtype=np.float32))
        n_rays = len(rays)
        origins = rays[:, :3]
        directions = rays[:, 3:]
        dir_magnitudes = np.linalg.norm(directions, axis=1, keepdims=True)
        dir_normalized = directions / (dir_magnitudes + 1e-10)
        tree = KDTree(self.vertices)
        t_hit = np.full(n_rays, np.inf)
        hit_points = np.full((n_rays, 3), np.nan)
        hit_point_ids = np.full(n_rays, -1, dtype=np.int32)
        hit_normals = np.full((n_rays, 3), np.nan)
        max_ray_length = float(dir_magnitudes.max())

        for i in range(n_rays):
            origin = origins[i]
            direction = dir_normalized[i]
            ray_length = dir_magnitudes[i, 0]
            search_radius = ray_length + knn_radius
            candidate_indices = tree.query_ball_point(origin, r=search_radius)

            if len(candidate_indices) == 0:
                continue

            candidate_points = self.vertices[np.array(candidate_indices, dtype=np.intp)]
            vecs_to_points = candidate_points - origin
            t_params = np.dot(vecs_to_points, direction)

            valid_mask = (t_params > 0) & (t_params <= ray_length)

            if not np.any(valid_mask):
                continue

            valid_t = t_params[valid_mask]
            valid_indices = np.array(candidate_indices)[valid_mask]
            valid_points = candidate_points[valid_mask]
            projections = origin + valid_t[:, np.newaxis] * direction
            perpendicular_vectors = valid_points - projections
            perpendicular_distances = np.linalg.norm(perpendicular_vectors, axis=1)

            within_radius_mask = perpendicular_distances <= knn_radius

            if not np.any(within_radius_mask):
                continue

            valid_perp_distances = perpendicular_distances[within_radius_mask]
            best_idx_in_valid = int(np.argmin(valid_perp_distances))
            valid_within_radius = np.where(within_radius_mask)[0]
            best_idx = int(valid_within_radius[best_idx_in_valid])

            best_t = valid_t[best_idx]
            best_point_id = int(valid_indices[best_idx])

            t_hit[i] = best_t
            hit_points[i] = self.vertices[best_point_id]
            hit_point_ids[i] = best_point_id
            if self.normals is not None:
                hit_normals[i] = self.normals[best_point_id]

        result_dict: dict[str, Any] = {
            "t_hit": t_hit,
            "hit_points": hit_points,
            "primitive_ids": hit_point_ids,
            "hit_normals": hit_normals,
        }

        if one_hit_per_target and len(rays) > 1:
            result_dict = DiscreteSurface._annotate_shortest_ray_hit(result_dict)

        return result_dict

    def distance_to_pointcloud(
        self,
        target: "OrientedPointCloud",
        method: str = "nn_unoriented",
        max_distance: float | None = None,
        ray_length: float | None = None,
        reverse_normals: bool = False,
        knn_radius: float = 10.0,
        one_hit_per_target: bool = False,
    ) -> dict[str, Any]:
        """
        Distance from this point cloud to a target point cloud.

        ``method='nn_unoriented'`` uses nearest-neighbor distances and ignores normals.
        ``method='nn_oriented'`` casts rays along this point cloud's normals and finds
        target points near each ray.
        """
        if not isinstance(target, OrientedPointCloud):
            raise TypeError(
                f"target must be OrientedPointCloud, got {type(target).__name__}"
            )
        if self.vertices is None or len(self.vertices) == 0:
            raise ValueError("source point cloud must have vertices")
        if target.vertices is None or len(target.vertices) == 0:
            raise ValueError("target must have vertices")

        method = str(method).lower()
        aliases = {
            "nn": "nn_unoriented",
            "nearest": "nn_unoriented",
            "nearest_neighbor": "nn_unoriented",
            "raycast": "nn_oriented",
            "oriented": "nn_oriented",
        }
        method = aliases.get(method, method)

        if method == "nn_unoriented":
            out = self._distance_to_pointcloud_nn(
                target=target, max_distance=max_distance
            )
        elif method == "nn_oriented":
            if self.normals is None:
                raise ValueError("method='nn_oriented' requires source normals")
            out = self._distance_to_pointcloud_raycast(
                target=target,
                ray_length=ray_length,
                reverse_normals=reverse_normals,
                knn_radius=knn_radius,
                max_distance=max_distance,
            )
        else:
            raise ValueError(
                f"method must be 'nn_unoriented' or 'nn_oriented', got '{method}'"
            )

        if one_hit_per_target:
            out = DiscreteSurface.apply_one_hit_per_target_distance(out)
        return out

    def _distance_to_pointcloud_nn(
        self, target: "OrientedPointCloud", max_distance: float | None = None
    ) -> dict[str, Any]:
        """Per-source-point nearest neighbor distance to target vertices."""
        source_points = np.asarray(self.vertices)
        target_points = np.asarray(target.vertices)
        tree = KDTree(target_points)
        distances, indices = tree.query(source_points)

        if max_distance is not None:
            hit_mask = distances <= max_distance
        else:
            hit_mask = np.ones(len(distances), dtype=bool)

        closest_points = target_points[indices]
        closest_normals = (
            target.normals[indices]
            if target.normals is not None
            else None
        )

        valid_distances = distances[hit_mask]
        stats = {
            "n_source": len(source_points),
            "n_target": len(target_points),
            "n_hits": int(hit_mask.sum()),
            "min_distance": valid_distances.min() if len(valid_distances) > 0 else np.inf,
            "max_distance": valid_distances.max() if len(valid_distances) > 0 else np.inf,
            "mean_distance": valid_distances.mean() if len(valid_distances) > 0 else np.nan,
            "median_distance": np.median(valid_distances)
            if len(valid_distances) > 0
            else np.nan,
            "std_distance": valid_distances.std()
            if len(valid_distances) > 0
            else np.nan,
        }

        out: dict[str, Any] = {
            "distances": distances,
            "closest_points": closest_points,
            "closest_indices": indices,
            "hit_mask": hit_mask,
            "stats": stats,
        }

        if closest_normals is not None:
            out["closest_normals"] = closest_normals

        return out

    def _distance_to_pointcloud_raycast(
        self,
        target: "OrientedPointCloud",
        ray_length: float | None = None,
        reverse_normals: bool = False,
        knn_radius: float = 10.0,
        max_distance: float | None = None,
    ) -> dict[str, Any]:
        """Rays from this cloud along normals; hits resolved on target via target.cast_rays."""
        rays = geom.construct_rays(
            points=self.vertices,
            normals=self.normals,
            ray_length=ray_length,
            reverse_direction=reverse_normals,
        )   
        result = target.cast_rays(rays, knn_radius=knn_radius)

        distances = result["t_hit"]
        if max_distance is not None:
            hit_mask = (np.isfinite(distances)) & (distances <= max_distance)
        else:
            hit_mask = np.isfinite(distances)

        valid_distances = distances[hit_mask]
        stats = {
            "n_source": len(self.vertices),
            "n_target": len(target.vertices),
            "n_hits": int(hit_mask.sum()),
            "min_distance": valid_distances.min() if len(valid_distances) > 0 else np.inf,
            "max_distance": valid_distances.max() if len(valid_distances) > 0 else np.inf,
            "mean_distance": valid_distances.mean() if len(valid_distances) > 0 else np.nan,
            "median_distance": np.median(valid_distances)
            if len(valid_distances) > 0
            else np.nan,
            "std_distance": valid_distances.std()
            if len(valid_distances) > 0
            else np.nan,
        }

        out = {
            "distances": distances,
            "closest_points": result["hit_points"],
            "closest_indices": result["primitive_ids"],
            "hit_mask": hit_mask,
            "stats": stats,
        }

        if "hit_normals" in result:
            out["closest_normals"] = result["hit_normals"]

        return out

    def get_point_neighborhoods(
        self,
        seed_point_ids: np.ndarray,
        radii: list[float],
    ) -> dict[str, np.ndarray]:
        """
        Build point-index regions around seed points at increasing 3D radii.

        Always includes ``"hit points"`` (unique seeds). For each radius ``r``,
        adds ``"r <= {r} nm"`` and annulus bands between consecutive radii.
        """
        seeds = np.unique(np.asarray(seed_point_ids, dtype=np.intp))
        vertices = np.asarray(self.vertices, dtype=np.float64)
        tree = KDTree(vertices)

        regions: dict[str, np.ndarray] = {"hit points": np.sort(seeds)}
        radii = [float(r) for r in radii]
        if len(radii) == 0:
            return regions

        cumulative: list[set] = []
        for radius in radii:
            expanded: set = set()
            for point_id in seeds:
                neighbors = tree.query_ball_point(vertices[point_id], r=radius)
                expanded.update(int(i) for i in neighbors)
            cumulative.append(expanded)
            regions[f"r <= {radius:g} nm"] = np.array(sorted(expanded), dtype=int)

        for idx_inner in range(len(radii) - 1):
            r_inner = radii[idx_inner]
            r_outer = radii[idx_inner + 1]
            ring_set = cumulative[idx_inner + 1] - cumulative[idx_inner]
            regions[f"{r_inner:g} < r <= {r_outer:g} nm"] = np.array(
                sorted(ring_set), dtype=int
            )

        return regions

    def downsample_pixel(self, pixel_size: float, recompute_normals: bool = True, knn: int = 30, 
                        orient_normals: bool = True, tangent_plane_knn: int = 50, inplace: bool = False) -> "OrientedPointCloud" | None:
        """
        Downsample point cloud using pixel/voxel grid.
        
        Parameters
        ----------
        pixel_size : float
            Size of pixel/voxels for downsampling
        recompute_normals : bool, default=True
            Whether to recompute normals after downsampling. Recommended because
            downsampling changes the local neighborhood structure.
        knn : int, default=30
            Number of nearest neighbors for normal estimation (if recompute_normals=True)
        orient_normals : bool, default=True
            Whether to orient normals consistently (if recompute_normals=True)
        tangent_plane_knn : int, default=50
            Number of neighbors for consistent normal orientation (if recompute_normals=True)
        inplace : bool, default=False
            If True, modify in place. If False, return new instance
        
        Returns
        -------
        OrientedPointCloud or None
            New instance if inplace=False, else None
        """
        pcd_o3d = self._to_open3d()
        
        # Downsample
        downsampled_o3d = pcd_o3d.pixel_down_sample(pixel_size)
        
        original_count = len(self.vertices)
        downsampled_count = len(downsampled_o3d.points)
        reduction_factor = original_count / downsampled_count if downsampled_count > 0 else 0
        
        # Recompute normals
        if recompute_normals:
            print(f"  Recomputing normals for downsampled points...")
            normals = self._compute_and_orient_normals(
                downsampled_o3d, knn, orient_normals, tangent_plane_knn
            )
        else:
            # Use existing normals
            if downsampled_o3d.has_normals():
                normals = np.asarray(downsampled_o3d.normals)
            else:
                normals = None
        
        if inplace:
            self.vertices = np.asarray(downsampled_o3d.points)
            self.normals = normals
            print(f"Downsampled from {original_count} to {downsampled_count} points (factor: {reduction_factor:.1f}x)")
            if recompute_normals:
                print(f"  Normals recomputed with knn={knn}")
            return None
        else:
            new_pcd = copy.deepcopy(self)
            new_pcd.vertices = np.asarray(downsampled_o3d.points)
            new_pcd.normals = normals

            print(f"Created downsampled point cloud: {original_count} → {downsampled_count} points (factor: {reduction_factor:.1f}x)")
            if recompute_normals:
                print(f"  Normals recomputed with knn={knn}")
            return new_pcd

    def filter_by_number_nn(self, nb_points: int = 20, radius: float = 1.0, inplace: bool = False) -> "OrientedPointCloud" | None:
        """
        Remove points using number of nearest neighbors.
        
        Parameters
        ----------
        nb_points : int, default=20
            Minimum number of neighbors within radius
        radius : float, default=1.0
            Search radius
        inplace : bool, default=False
            If True, modify in place. If False, return new instance
        
        Returns
        -------
        OrientedPointCloud or None
            New instance if inplace=False, else None
        """
        pcd_o3d = self._to_open3d()
        
        # Remove outliers
        filtered_o3d, outlier_indices = pcd_o3d.remove_radius_outlier(
            nb_points=nb_points, radius=radius
        )
        
        num_outliers = len(outlier_indices)
        num_remaining = len(filtered_o3d.points)
        
        if inplace:
            self.vertices = np.asarray(filtered_o3d.points)
            if filtered_o3d.has_normals():
                self.normals = np.asarray(filtered_o3d.normals)
            print(f"Removed {num_outliers} points, {num_remaining} points remaining")
            return None
        else:
            new_pcd = copy.deepcopy(self)
            new_pcd.vertices = np.asarray(filtered_o3d.points)
            if filtered_o3d.has_normals():
                new_pcd.normals = np.asarray(filtered_o3d.normals)

            print(f"Created filtered point cloud: {num_outliers} points removed, {num_remaining} points remaining")
            return new_pcd
    
    def _apply_filter_mask(self, keep_mask: np.ndarray):
        """Apply filtering mask to point cloud."""
        self.vertices = self.vertices[keep_mask]
        if self.normals is not None:
            self.normals = self.normals[keep_mask]

    def filter_by_labels(self, vertex_labels: np.ndarray, surface_type: str = 'inner', inplace: bool = False) -> "OrientedPointCloud" | None:
        """
        Create a new point cloud containing only inner or outer points based on labels.
        
        Parameters
        ----------
        vertex_labels : np.ndarray
            Labels from separate_surfaces (0=inner, 1=outer)
        surface_type : str, default='inner'
            Which surface to extract: 'inner' or 'outer'
        inplace : bool, default=False
            If True, modify in place. If False, return new instance.
            
        Returns
        -------
        OrientedPointCloud or None
            New point cloud instance if inplace=False, else None
        """
        # Determine which label to keep
        target_label = 0 if surface_type == 'inner' else 1
        mask = (vertex_labels == target_label)
        
        if inplace:
            self.vertices = self.vertices[mask]
            if self.normals is not None:
                self.normals = self.normals[mask]
            return None
        else:
            new_pcd = copy.deepcopy(self)
            new_pcd.vertices = self.vertices[mask].copy()
            if self.normals is not None:
                new_pcd.normals = self.normals[mask].copy()
            return new_pcd

    def compute_nearest_neighbor_distances(self, k: int = 1, return_indices: bool = False) -> dict[str, Any]:
        """
        Compute nearest neighbor distances for each point.
        
        Parameters
        ----------
        k : int, default=1
            Number of nearest neighbors to consider
        return_indices : bool, default=False
            If True, also return neighbor indices
        
        Returns
        -------
        dict
            Statistics including distances, density metrics, and outlier flags
        """
        pcd_o3d = self._to_open3d()
        
        # Build KDTree
        tree = o3d.geometry.KDTreeFlann(pcd_o3d)
        
        distances = []
        neighbor_indices = []
        
        for i in range(len(self.vertices)):
            [_, idx, dist] = tree.search_knn_vector_3d(pcd_o3d.points[i], k + 1)  # +1 to exclude self
            distances.append(dist[1:])  # Exclude self (distance=0)
            neighbor_indices.append(idx[1:])
        
        distances = np.array(distances)
        neighbor_indices = np.array(neighbor_indices)
        
        # Compute statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        
        # Density estimation (inverse of mean distance)
        density = 1.0 / (mean_dist + 1e-12)
        
        # Outlier detection (points with distances > mean + 2*std)
        outlier_threshold = mean_dist + 2 * std_dist
        outlier_mask = np.any(distances > outlier_threshold, axis=1)
        num_outliers = np.sum(outlier_mask)
        
        stats = {
            'distances': distances,
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'min_distance': min_dist,
            'max_distance': max_dist,
            'density_estimate': density,
            'outlier_threshold': outlier_threshold,
            'num_outliers': num_outliers,
            'outlier_mask': outlier_mask,
            'outlier_fraction': num_outliers / len(self.vertices)
        }
        
        if return_indices:
            stats['neighbor_indices'] = neighbor_indices
        
        return stats

    def distance_to_pointcloud(self, other_pcd: "OrientedPointCloud", return_distances: bool = False) -> dict[str, Any]:
        """
        Compute distances from this point cloud to another point cloud.
        
        Parameters
        ----------
        other_pcd : OrientedPointCloud
            Target point cloud
        return_distances : bool, default=False
            If True, return individual point distances
        
        Returns
        -------
        dict
            Distance statistics and analysis
        """
        pcd1_o3d = self._to_open3d()
        pcd2_o3d = other_pcd._to_open3d()
        
        # Compute distances from pcd1 to pcd2
        distances = pcd1_o3d.compute_point_cloud_distance(pcd2_o3d)
        distances = np.asarray(distances)
        
        # Statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        median_dist = np.median(distances)
        
        # Percentiles for detailed analysis
        percentiles = np.percentile(distances, [25, 75, 90, 95, 99])
        
        # Registration quality metrics
        rmse = np.sqrt(np.mean(distances**2))
        
        stats = {
            'mean_distance': mean_dist,
            'std_distance': std_dist,
            'min_distance': min_dist,
            'max_distance': max_dist,
            'median_distance': median_dist,
            'rmse': rmse,
            'percentiles': {
                'p25': percentiles[0],
                'p75': percentiles[1],
                'p90': percentiles[2],
                'p95': percentiles[3],
                'p99': percentiles[4]
            },
            'num_points': len(distances)
        }
        
        if return_distances:
            stats['distances'] = distances
        
        return stats

    def oversample(self, oversample_factor: float | None = None, point_spacing: float | None = None, random_seed: int | None = None) -> "OrientedPointCloud":
        """
        Oversample or undersample point cloud using Poisson disk sampling.
        
        Two modes:
        1. Factor-based: Use oversample_factor to specify desired increase in points
        2. Spacing-based: Use point_spacing to specify desired spacing between points
        
        Parameters
        ----------
        oversample_factor : float, optional
            Desired factor of increase in points.
            If None and point_spacing is provided, will be computed from spacing.
            If both are None, defaults to 1.0 (no change).
        point_spacing : float, optional
            Desired spacing between sampled points (same units as point cloud coordinates).
            If provided, uses greedy Poisson disk sampling to enforce spacing directly.
        random_seed : int, optional
            Random seed for reproducible results. If None, uses current random state.
        
        Returns
        -------
        OrientedPointCloud
            New point cloud with sampled points and normals
        """
        if self.vertices is None:
            raise ValueError("Point cloud must have vertices")
        
        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)
        
        original_points = len(self.vertices)
        
        # Build KDTree
        tree = cKDTree(self.vertices)
        
        # Determine target spacing
        if point_spacing is not None:
            # Compute current mean nearest-neighbor distance for info
            distances, _ = tree.query(self.vertices, k=2)
            nn_distances = distances[:, 1]  # Exclude self (distance=0)
            current_spacing = np.mean(nn_distances)
            
            target_spacing = point_spacing
            
            print(f"Point cloud sampling: {original_points} points")
            print(f"  Current spacing: {current_spacing:.2f}, target spacing: {point_spacing:.2f}")
            
        elif oversample_factor is not None:
            # Compute current mean nearest-neighbor distance
            distances, _ = tree.query(self.vertices, k=2)
            nn_distances = distances[:, 1]  # Exclude self (distance=0)
            current_spacing = np.mean(nn_distances)
            
            # Quadratic scaling: if we want N_new = N_old * factor,
            # then spacing_new = spacing_old / sqrt(factor)
            target_spacing = current_spacing / np.sqrt(oversample_factor)
            
            print(f"Point cloud sampling: {original_points} points")
            print(f"  Current spacing: {current_spacing:.2f}, target spacing: {target_spacing:.2f} ({oversample_factor:.2f}x)")
        else:
            # Default: no change, return copy
            print(f"Point cloud sampling: {original_points} → {original_points} points (no change)")
            result = OrientedPointCloud()
            result.vertices = self.vertices.copy()
            result.normals = self.normals.copy() if self.normals is not None else None
            result.inherit_coordinate_metadata(self)
            return result
        
        # Greedy Poisson disk sampling
        sampled_indices = []
        remaining_indices = set(range(len(self.vertices)))
        
        # Start with a random point
        if len(remaining_indices) > 0:
            current_idx = np.random.choice(list(remaining_indices))
            sampled_indices.append(current_idx)
            remaining_indices.remove(current_idx)
            
            # Remove neighbors within target_spacing
            neighbors = tree.query_ball_point(
                self.vertices[current_idx], 
                r=target_spacing
            )
            remaining_indices -= set(neighbors)
        
        # Iteratively add points that are at least target_spacing away
        while remaining_indices:
            if not remaining_indices:
                break
            
            # Try to find a valid point
            candidate_idx = np.random.choice(list(remaining_indices))
            candidate_point = self.vertices[candidate_idx]
            
            # Check distance to all sampled points
            if len(sampled_indices) > 0:
                sampled_points = self.vertices[sampled_indices]
                distances = np.linalg.norm(sampled_points - candidate_point, axis=1)
                min_distance = np.min(distances)
                
                if min_distance >= target_spacing:
                    sampled_indices.append(candidate_idx)
                    # Remove neighbors within target_spacing
                    neighbors = tree.query_ball_point(candidate_point, r=target_spacing)
                    remaining_indices -= set(neighbors)
                else:
                    remaining_indices.remove(candidate_idx)
            else:
                sampled_indices.append(candidate_idx)
                neighbors = tree.query_ball_point(candidate_point, r=target_spacing)
                remaining_indices -= set(neighbors)
        
        sampled_indices = np.array(sampled_indices)
        
        # Create new OrientedPointCloud
        result = OrientedPointCloud()
        result.vertices = self.vertices[sampled_indices]
        
        # Transfer normals
        if self.normals is not None:
            result.normals = self.normals[sampled_indices]
        result.inherit_coordinate_metadata(self)
        
        print(f"Final: {len(result.vertices)} points")
        
        return result

    @staticmethod
    def _surface_poisson_oversample_pca(points: np.ndarray, normals: np.ndarray, r: float = 1.0, n_iter: int = 5,
                                    k_neighbors: int = 10, candidate_multiplier: int = 2,
                                    jitter_scale: float = 0.05, smooth_iter: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """
        Core Poisson disk oversampling implementation.
        """
        points = np.asarray(points)
        normals = np.asarray(normals)
        
        all_points = points.copy()
        all_normals = normals.copy()
        
        for iter_idx in range(n_iter):
            print(f"  Iteration {iter_idx+1}/{n_iter} -- points so far: {len(all_points)}")
            kdtree_all = cKDTree(all_points)
            new_samples = []
            
            for i, p in enumerate(all_points):
                # KD-tree for neighbors
                dists, idxs = kdtree_all.query(p, k=k_neighbors)
                neighbors = all_points[idxs]
                
                # Weighted PCA tangent plane
                origin, tangent, bitangent, normal = OrientedPointCloud._weighted_pca_tangent_plane(
                    p, neighbors, sigma=r
                )
                
                # Adaptive local radius
                local_r = max(0.5 * dists.mean(), 0.1*r)
                
                # Generate multiple candidates per point
                n_candidates = k_neighbors * candidate_multiplier
                theta = np.random.rand(n_candidates) * 2 * np.pi
                radius_offsets = local_r * (np.sqrt(np.random.rand(n_candidates)) * 0.8 + 0.2)
                offsets = np.outer(np.cos(theta), tangent) + np.outer(np.sin(theta), bitangent)
                offsets *= radius_offsets[:, None]
                candidates = origin + offsets
                
                # Add slight jitter along normal
                n_offsets = np.random.randn(len(candidates))[:, None] * (r * jitter_scale)
                candidates += n_offsets * normal[None, :]
                
                new_samples.append(candidates)
            
            new_samples = np.vstack(new_samples)
            
            # Poisson-disk rejection
            mask = np.array([len(kdtree_all.query_ball_point(pt, r*0.2)) == 0 for pt in new_samples])
            valid_samples = new_samples[mask]
            if len(valid_samples) == 0:
                print("  No new points added this iteration, stopping early.")
                break
            
            # Assign normals from nearest neighbor
            _, nn_idx = kdtree_all.query(valid_samples)
            valid_normals = all_normals[nn_idx]
            
            # Append new points
            all_points = np.vstack((all_points, valid_samples))
            all_normals = np.vstack((all_normals, valid_normals))
            
            # Optional smoothing
            for _ in range(smooth_iter):
                kdtree_all = cKDTree(all_points)
                smoothed_points = all_points.copy()
                for i, p in enumerate(all_points):
                    dists, idxs = kdtree_all.query(p, k=min(5, len(all_points)))
                    neighbors = all_points[idxs]
                    smoothed_points[i] += 0.2 * (neighbors.mean(axis=0) - p)
                all_points = smoothed_points
        
        return all_points, all_normals

    @staticmethod
    def _weighted_pca_tangent_plane(point: np.ndarray, neighbors: np.ndarray, sigma: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute weighted PCA tangent plane.
        """
        dists = np.linalg.norm(neighbors - point, axis=1)
        weights = np.exp(-dists**2 / (2 * sigma**2))
        weights /= weights.sum()
        
        centroid = (neighbors * weights[:, None]).sum(axis=0)
        centered = neighbors - centroid
        cov = (centered * weights[:, None]).T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Smallest eigenvector is normal
        normal = eigvecs[:, 0]
        tangent = eigvecs[:, 1]
        bitangent = eigvecs[:, 2]
        return centroid, tangent, bitangent, normal

    def extract_midplane(self, pixel_size: float | np.ndarray | None = None, max_thickness_pixels: int = 3) -> "OrientedPointCloud" | dict[str, "OrientedPointCloud"]:
        """
        Extract mid-plane from point cloud using median depth approach.
        
        Uses PCA to find the thickness direction, then extracts points at median depth
        along that direction. Works on both single point clouds and dictionaries.
        
        Parameters
        ----------
        pixel_size : float or array-like, optional
            Pixel/voxel size for thickness calculation. If None, estimates from point spacing.
        max_thickness_pixels : int, default=3
            Maximum thickness of resulting mid-plane slice
            
        Returns
        -------
        OrientedPointCloud or dict
            Mid-plane point cloud(s). Returns dict if self is a dict, else single instance.
            Normals are preserved from original points (nearest neighbor transfer).
        """
        # Handle dict input (multiple labels)
        if isinstance(self, dict):
            result = {}
            for label, pcd in self.items():
                result[label] = pcd.extract_midplane(pixel_size, max_thickness_pixels)
            return result
        
        if self.vertices is None:
            raise ValueError("Point cloud must have vertices")
        
        vertices = self.vertices
        
        # Use PCA of coordinates to find thickness direction
        pca = PCA(n_components=3)
        pca.fit(vertices)
        
        # Find axis with smallest variance = thickness direction
        variances = pca.explained_variance_
        thickness_axis_idx = np.argmin(variances)
        plane_normal = pca.components_[thickness_axis_idx]
        
        # Compute center of mass
        center_of_mass = np.mean(vertices, axis=0)
        
        # Project points onto thickness direction
        points_centered = vertices - center_of_mass
        signed_distances = np.dot(points_centered, plane_normal)
        
        # Find median depth
        median_depth = np.median(signed_distances)
        
        # Extract points within max_thickness around median depth
        if pixel_size is None:
            tree = cKDTree(vertices)
            distances, _ = tree.query(vertices, k=2)
            avg_spacing = np.mean(distances[:, 1])
            pixel_size_est = avg_spacing
        else:
            if np.isscalar(pixel_size):
                pixel_size_est = pixel_size
            else:
                pixel_size_est = np.mean(pixel_size)
        
        thickness_threshold = max_thickness_pixels * pixel_size_est
        distances_from_median = np.abs(signed_distances - median_depth)
        mask = distances_from_median <= thickness_threshold
        
        skeleton_vertices = vertices[mask]
        
        # Create new OrientedPointCloud with mid-plane points
        midplane_pcd = OrientedPointCloud()
        midplane_pcd.vertices = skeleton_vertices
        
        # Transfer normals from original points (nearest neighbor)
        if self.normals is not None:
            tree = cKDTree(vertices)
            _, nearest_indices = tree.query(skeleton_vertices, k=1)
            midplane_pcd.normals = self.normals[nearest_indices]
        
        midplane_pcd.inherit_coordinate_metadata(self)
        return midplane_pcd

class AnalyticSurface(Surface):
    """Abstract base for analytic/parametric surface representations (ellipsoids, cylinders, etc.)."""

    def __init__(self):
        super().__init__()
        self.center = None

    def translate(self, v):
        """Shift the surface center by vector v."""
        self.center = np.asarray(self.center, dtype=float) + np.asarray(v, dtype=float)

    def distance_point_center(self, point):
        """Euclidean distance from point to surface center."""
        return float(np.linalg.norm(np.asarray(point, dtype=float) - self.center))

    @classmethod
    @abstractmethod
    def fit_to_points(cls, points: np.ndarray) -> "AnalyticSurface":
        """Fit the surface to a set of 3D points, returning a new instance."""

    @abstractmethod
    def distance_point_surface(self, point: np.ndarray) -> float:
        """Shortest Euclidean distance from point to the surface."""

    @abstractmethod
    def transform(self, transformation_matrix: np.ndarray) -> None:
        """Apply a 4x4 homogeneous transformation matrix in-place."""

    @abstractmethod
    def rotate(self, rotation_matrix: np.ndarray) -> None:
        """Rotate the surface in-place."""


class Cylinder(AnalyticSurface):
    """
    Cylinder representation for filament fitting.
    
    Stores cylinder parameters (center point on axis, axis direction, radius)
    and provides methods for fitting and sampling.
    """
    
    def __init__(self, center=None, axis_direction=None, radius=None, 
                 centerline_length=None):
        """
        Initialize cylinder.
        
        Parameters
        ----------
        center : np.ndarray (3,), optional
            Point on the cylinder axis (typically centroid of fitted points)
        axis_direction : np.ndarray (3,), optional
            Normalized direction vector along cylinder axis
        radius : float, optional
            Cylinder radius
        centerline_length : float, optional
            Length of centerline extent
        """
        super().__init__()
        self.center = center
        self.axis_direction = axis_direction
        self.radius = radius
        self.centerline_length = centerline_length
    
    @classmethod
    def fit_to_points(cls, points: np.ndarray) -> "Cylinder":
        """
        Fit cylinder to a set of points using PCA.
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
            Point coordinates
            
        Returns
        -------
        Cylinder
            Fitted cylinder instance
        """        
        if len(points) < 3:
            raise ValueError(f"Need at least 3 points to fit cylinder, got {len(points)}")
        
        # Center the points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # PCA: first principal component is the cylinder axis
        pca = PCA(n_components=3)
        pca.fit(points_centered)
        
        # Cylinder axis is the first principal component (largest variance)
        axis_direction = pca.components_[0]
        
        # Ensure consistent orientation
        projections = np.dot(points_centered, axis_direction)
        if np.mean(projections) < 0:
            axis_direction = -axis_direction
            projections = -projections
        
        # Compute radius as mean distance from points to axis
        axis_projections = np.outer(projections, axis_direction)
        radial_vectors = points_centered - axis_projections
        distances_to_axis = np.linalg.norm(radial_vectors, axis=1)
        radius = np.mean(distances_to_axis)
        
        # Get centerline extent
        centerline_positions = np.dot(points_centered, axis_direction)
        centerline_length = np.max(centerline_positions) - np.min(centerline_positions)
        
        return cls(
            center=centroid,
            axis_direction=axis_direction,
            radius=radius,
            centerline_length=centerline_length
        )
    
    def sample_along_axis(self, spacing: float) -> np.ndarray:
        """
        Sample points along cylinder centerline at regular intervals.
        
        Parameters
        ----------
        spacing : float
            Spacing between sampled points
            
        Returns
        -------
        np.ndarray (M, 3)
            Sampled points along centerline
        """
        if self.center is None or self.axis_direction is None:
            raise ValueError("Cylinder must be fitted before sampling")
        
        half_length = self.centerline_length / 2.0
        n_samples = max(2, int(np.ceil(self.centerline_length / spacing)) + 1)
        sampled_positions = np.linspace(-half_length, half_length, n_samples)
        
        sampled_points = self.center + np.outer(sampled_positions, self.axis_direction)
        return sampled_points
    
    def distance_point_surface(self, point: np.ndarray) -> float:
        """
        Compute distance from point to cylinder surface.
        
        Parameters
        ----------
        point : np.ndarray (3,)
            Query point
            
        Returns
        -------
        float
            Distance to cylinder surface
        """
        if self.center is None or self.axis_direction is None or self.radius is None:
            raise ValueError("Cylinder must be fitted before computing distances")
        
        # Vector from center to point
        vec_to_point = point - self.center
        
        # Project onto axis
        proj_length = np.dot(vec_to_point, self.axis_direction)
        proj_point = self.center + proj_length * self.axis_direction
        
        # Radial vector (perpendicular to axis)
        radial_vec = point - proj_point
        radial_distance = np.linalg.norm(radial_vec)
        
        # Distance to surface = |radial_distance - radius|
        return np.abs(radial_distance - self.radius)
    
    def distance_point_center(self, point: np.ndarray) -> float:
        """Compute distance from point to cylinder center."""
        if self.center is None:
            raise ValueError("Cylinder must have center defined")
        return np.linalg.norm(point - self.center)

    @staticmethod
    def _extract_centerline(points: np.ndarray, n_bins: int = 50) -> np.ndarray:
        """
        Extract centerline by binning points along PCA axis and computing centroids.
        
        Simple approach that doesn't require skeletonization.
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
            Point coordinates
        n_bins : int, default=50
            Number of bins along the axis
            
        Returns
        -------
        centerline_points : np.ndarray (M, 3)
            Ordered centerline points (M <= n_bins, depending on bins with points)
        """        
        if len(points) < 3:
            raise ValueError(f"Need at least 3 points, got {len(points)}")
        
        # Quick PCA to get axis
        pca = PCA(n_components=1)
        pca.fit(points)
        axis_direction = pca.components_[0]
        
        # Center points
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # Project onto axis
        projections = np.dot(points_centered, axis_direction)
        
        # Bin along axis
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        bin_edges = np.linspace(min_proj, max_proj, n_bins + 1)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        
        # Compute centroids for each bin
        centerline_points = []
        for i in range(n_bins):
            mask = (projections >= bin_edges[i]) & (projections < bin_edges[i+1])
            if i == n_bins - 1:  # Include upper boundary for last bin
                mask = (projections >= bin_edges[i]) & (projections <= bin_edges[i+1])
            
            if np.any(mask):
                bin_points = points[mask]
                bin_centroid = np.mean(bin_points, axis=0)
                centerline_points.append(bin_centroid)
        
        return np.array(centerline_points) if len(centerline_points) > 0 else points
    
    @classmethod
    def fit_to_points_spline(cls, points: np.ndarray, n_bins: int = 50, smoothing: float = 0) -> "Cylinder":
        """
        Fit cylinder using B-spline centerline (handles curves well).
        
        Parameters
        ----------
        points : np.ndarray (N, 3)
            Point coordinates
        n_bins : int, default=50
            Number of bins for centerline extraction
        smoothing : float, default=0
            Smoothing factor for spline (0 = no smoothing, exact fit)
            
        Returns
        -------
        Cylinder
            Fitted cylinder instance with spline-based centerline
        """
        from scipy.interpolate import splprep, splev
        
        if len(points) < 3:
            raise ValueError(f"Need at least 3 points to fit cylinder, got {len(points)}")
        
        # Extract centerline
        centerline = cls._extract_centerline(points, n_bins=n_bins)
        
        if len(centerline) < 4:
            # Too few centerline points, fall back to PCA
            return cls.fit_to_points(points)
        
        # Fit B-spline to centerline
        try:
            tck, u = splprep(centerline.T, s=smoothing, k=min(3, len(centerline)-1))
        except:
            # If spline fitting fails, fall back to PCA
            return cls.fit_to_points(points)
        
        # Compute radius (mean distance from all points to centerline)
        # Sample spline densely and find distances
        u_dense = np.linspace(0, 1, 100)
        centerline_dense = np.array(splev(u_dense, tck)).T
        
        # For each point, find distance to nearest centerline point
        from scipy.spatial.distance import cdist
        distances = cdist(points, centerline_dense)
        min_distances = np.min(distances, axis=1)
        radius = np.mean(min_distances)
        
        # Compute total centerline length (arc length)
        centerline_diff = np.diff(centerline_dense, axis=0)
        centerline_length = np.sum(np.linalg.norm(centerline_diff, axis=1))
        
        # Create cylinder instance with spline info
        cylinder = cls(
            center=centerline[0],  # Use first centerline point
            axis_direction=None,  # Will be computed on-the-fly from spline
            radius=radius,
            centerline_length=centerline_length
        )
        cylinder._spline_tck = tck  # Store spline parameters
        cylinder._spline_type = 'bspline'
        
        return cylinder
    
    
    def sample_along_axis_spline(self, spacing: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample points along spline centerline at regular arc-length intervals.
        
        Parameters
        ----------
        spacing : float
            Spacing between sampled points (arc-length distance)
            
        Returns
        -------
        np.ndarray (M, 3)
            Sampled points along centerline
        np.ndarray (M, 3)
            Tangent vectors (orientations) at sampled points
        """
        if not hasattr(self, '_spline_type') or self._spline_type != 'bspline':
            raise ValueError("Spline sampling only available for B-spline fitted cylinders")
        
        from scipy.interpolate import splev
        
        # Sample at regular arc-length intervals
        n_samples = max(2, int(np.ceil(self.centerline_length / spacing)) + 1)
        u_new = np.linspace(0, 1, n_samples)
        
        # Evaluate spline
        sampled_points = np.array(splev(u_new, self._spline_tck)).T
        
        # Get tangent vectors (first derivative)
        tangents = np.array(splev(u_new, self._spline_tck, der=1)).T
        # Normalize
        tangent_norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        tangents = tangents / (tangent_norms + 1e-12)
        
        return sampled_points, tangents

    def oversample(self, sampling_distance: float | None = None, point_spacing: float | None = None) -> "OrientedPointCloud":
        """
        Sample the cylinder centerline as an :class:`OrientedPointCloud`.

        Parameters
        ----------
        sampling_distance : float, optional
            Spacing between sampled points along the centerline.
        point_spacing : float, optional
            Alias for ``sampling_distance`` for consistency with other surface classes.

        Returns
        -------
        OrientedPointCloud
            Points sampled along the cylinder axis or spline centerline, with tangent
            vectors stored as normals.
        """
        spacing = sampling_distance if sampling_distance is not None else point_spacing
        if spacing is None:
            raise ValueError("Provide sampling_distance or point_spacing")

        if hasattr(self, "_spline_type") and self._spline_type == "bspline":
            sampled_points, tangents = self.sample_along_axis_spline(spacing)
        else:
            sampled_points = self.sample_along_axis(spacing)
            if self.axis_direction is None:
                raise ValueError("Cylinder must have axis_direction to oversample")
            tangent = np.asarray(self.axis_direction, dtype=float)
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            tangents = np.tile(tangent, (len(sampled_points), 1))

        oriented_pcd = OrientedPointCloud()
        oriented_pcd.vertices = sampled_points
        oriented_pcd.normals = tangents
        oriented_pcd.inherit_coordinate_metadata(self)
        return oriented_pcd

    def transform(self, transformation_matrix: np.ndarray):
        """Apply a 4x4 homogeneous transformation matrix in-place."""
        M = np.asarray(transformation_matrix, dtype=float)
        self.center = M[:3, :3].dot(self.center) + M[:3, 3]
        if self.axis_direction is not None:
            R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
            self.axis_direction = R.dot(self.axis_direction)
            self.axis_direction /= np.linalg.norm(self.axis_direction)

    def rotate(self, rotation_matrix: np.ndarray):
        """Rotate the cylinder in-place."""
        R = np.asarray(rotation_matrix, dtype=float)
        self.center = R.dot(self.center)
        if self.axis_direction is not None:
            self.axis_direction = R.dot(self.axis_direction)
            self.axis_direction /= np.linalg.norm(self.axis_direction)


class Ellipsoid(AnalyticSurface):
    """
    Ellipsoid representation.

    Fits ellipsoids to point clouds and provides geometric operations,
    serialization, and ray intersection.
    """

    columns = [
        "cx", "cy", "cz",
        "rx", "ry", "rz",
        "ev1x", "ev1y", "ev1z",
        "ev2x", "ev2y", "ev2z",
        "ev3x", "ev3y", "ev3z",
        "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10",
    ]

    def __init__(self, center: np.ndarray | None = None, radii: np.ndarray | None = None, e_vec1: np.ndarray | None = None, e_vec2: np.ndarray | None = None, e_vec3: np.ndarray | None = None, params: np.ndarray | None = None):
        super().__init__()
        self.center = center if center is not None else np.zeros(3)
        self.radii = radii if radii is not None else np.zeros(3)
        self.e_vec1 = e_vec1 if e_vec1 is not None else np.zeros(3)
        self.e_vec2 = e_vec2 if e_vec2 is not None else np.zeros(3)
        self.e_vec3 = e_vec3 if e_vec3 is not None else np.zeros(3)
        self.params = params if params is not None else np.zeros(10)
        self.singular = False
    
    @classmethod
    def fit_to_points(cls, points: np.ndarray) -> "Ellipsoid":
        """
        Fit ellipsoid to a set of 3D points using algebraic least squares.

        Based on http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

        Parameters
        ----------
        points : np.ndarray (N, 3)

        Returns
        -------
        Ellipsoid
            Fitted instance, or an instance with ``singular=True`` if too few points.
        """
        if len(points) < 3:
            el = cls()
            el.singular = True
            return el

        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        D = np.array([
            x * x + y * y - 2 * z * z,
            x * x + z * z - 2 * y * y,
            2 * x * y, 2 * x * z, 2 * y * z,
            2 * x, 2 * y, 2 * z,
            1 - 0 * x,
        ])
        d2 = (x * x + y * y + z * z).T
        u = np.linalg.solve(D.dot(D.T), D.dot(d2))
        params = np.array([u[0] + u[1] - 1, u[0] - 2 * u[1] - 1, u[1] - 2 * u[0] - 1,
                           u[2], u[3], u[4], u[5], u[6], u[7], u[8]])
        el = cls(params=params)
        el.compute_props()
        return el

    def compute_props(self):
        """Derive center, radii, and eigenvectors from algebraic parameters."""
        p = self.params
        A = np.array([
            [p[0], p[3], p[4], p[6]],
            [p[3], p[1], p[5], p[7]],
            [p[4], p[5], p[2], p[8]],
            [p[6], p[7], p[8], p[9]],
        ])
        self.center = np.linalg.solve(-A[:3, :3], p[6:9])
        T = np.eye(4)
        T[3, :3] = self.center.T
        R = T.dot(A).dot(T.T)
        evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
        evecs = evecs.T
        self.e_vec1, self.e_vec2, self.e_vec3 = evecs[0], evecs[1], evecs[2]
        self.radii = np.sqrt(1.0 / np.abs(evals)) * np.sign(evals)

    @classmethod
    def from_dict(cls, input_dict):
        """Construct from a dict with keys: center, radii, e_vec1-3, params (optional)."""
        el = cls()
        for key in ("center", "radii", "e_vec1", "e_vec2", "e_vec3", "params"):
            if key in input_dict:
                setattr(el, key, np.asarray(input_dict[key], dtype=float))
        if not hasattr(el, "params") or el.params is None:
            raise ValueError("dict must contain 'params' for algebraic reconstruction.")
        if not all(k in input_dict for k in ("center", "radii", "e_vec1", "e_vec2", "e_vec3")):
            el.compute_props()
        return el

    @classmethod
    def from_df(cls, df):
        """Construct from a DataFrame or Series with the standard column set."""
        el = cls()
        el.center = np.asarray(df[["cx", "cy", "cz"]].values, dtype=float).flatten()
        el.radii = np.asarray(df[["rx", "ry", "rz"]].values, dtype=float).flatten()
        el.e_vec1 = np.asarray(df[["ev1x", "ev1y", "ev1z"]].values, dtype=float).flatten()
        el.e_vec2 = np.asarray(df[["ev2x", "ev2y", "ev2z"]].values, dtype=float).flatten()
        el.e_vec3 = np.asarray(df[["ev3x", "ev3y", "ev3z"]].values, dtype=float).flatten()
        el.params = np.asarray(
            df[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]].values, dtype=float
        ).flatten()
        return el

    @classmethod
    def from_array_like(cls, array_like):
        """
        Construct from a flat array or list.

        - Length 10: algebraic params; geometric properties computed.
        - Length 25: full vector [center(3), radii(3), ev1-3(9), params(10)].
        - Shape (N, 3): point coordinates; fits ellipsoid.
        """
        al = np.atleast_1d(np.asarray(array_like, dtype=float))
        if al.size == 10:
            el = cls(params=al.flatten())
            el.compute_props()
            return el
        elif al.size == 25:
            al = al.flatten()
            return cls(
                center=al[:3], radii=al[3:6],
                e_vec1=al[6:9], e_vec2=al[9:12], e_vec3=al[12:15],
                params=al[15:],
            )
        elif al.ndim == 2 and al.shape[1] == 3:
            return cls.fit_to_points(al)
        else:
            raise ValueError(f"Cannot construct Ellipsoid from array of shape {al.shape}.")

    def get_props_as_ndarray(self):
        """Return all properties as a flat (25,) array."""
        out = np.zeros(25)
        out[:3] = self.center
        out[3:6] = self.radii
        out[6:9] = self.e_vec1
        out[9:12] = self.e_vec2
        out[12:15] = self.e_vec3
        out[15:] = self.params
        return out

    def get_props_as_dict(self):
        """Return all properties as a dict."""
        return {
            "center": self.center, "radii": self.radii,
            "e_vec1": self.e_vec1, "e_vec2": self.e_vec2, "e_vec3": self.e_vec3,
            "params": self.params,
        }

    def get_props_as_list(self):
        """Return all properties as a list of arrays."""
        return [self.center, self.radii, self.e_vec1, self.e_vec2, self.e_vec3, self.params]

    def get_props_as_df(self):
        """Return all properties as a single-row DataFrame."""
        df = pd.DataFrame(columns=self.columns)
        df[["cx", "cy", "cz"]] = [self.center]
        df[["rx", "ry", "rz"]] = [self.radii]
        df[["ev1x", "ev1y", "ev1z"]] = [self.e_vec1]
        df[["ev2x", "ev2y", "ev2z"]] = [self.e_vec2]
        df[["ev3x", "ev3y", "ev3z"]] = [self.e_vec3]
        df[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]] = [self.params]
        return df
    
    def distance_point_surface(self, point):
        """Shortest Euclidean distance from point to ellipsoid surface."""
        e_vecs = np.column_stack([self.e_vec1, self.e_vec2, self.e_vec3])
        p_local = e_vecs.T.dot(np.asarray(point) - self.center)

        def scale_equation(lmbda):
            return np.sum((p_local / ((1 + lmbda) * self.radii)) ** 2) - 1

        lmbda = fsolve(scale_equation, 0)[0]
        closest_global = e_vecs.dot(p_local / (1 + lmbda)) + self.center
        return float(np.linalg.norm(np.asarray(point) - closest_global))

    def transform(self, transformation_matrix: np.ndarray):
        """Apply a 4x4 homogeneous transformation matrix in-place."""
        M = np.asarray(transformation_matrix, dtype=float)
        self.center = M[:3, :3].dot(self.center) + M[:3, 3]
        R = M[:3, :3] / np.linalg.norm(M[:3, :3], axis=0)
        self.e_vec1 = R.dot(self.e_vec1)
        self.e_vec2 = R.dot(self.e_vec2)
        self.e_vec3 = R.dot(self.e_vec3)

    def rotate(self, rotation_matrix: np.ndarray):
        """Rotate the ellipsoid in-place."""
        R = np.asarray(rotation_matrix, dtype=float)
        self.center = R.dot(self.center)
        self.e_vec1 = R.dot(self.e_vec1)
        self.e_vec2 = R.dot(self.e_vec2)
        self.e_vec3 = R.dot(self.e_vec3)

    def intersection_line(self, line):
        """
        Compute intersections between a line and the ellipsoid.

        Parameters
        ----------
        line : object with attributes ``p`` (3,) and ``dir`` (3,)

        Returns
        -------
        p1, p2 : np.ndarray or nan
        d1, d2 : float or nan — signed distances from line.p
        is_inside : bool
        """
        x, y, z = line.p[0], line.p[1], line.p[2]
        n1, n2, n3 = line.dir[0], line.dir[1], line.dir[2]
        a, b, c, d, e, f, g, h, i, j = self.params

        A = a*n1**2 + b*n2**2 + c*n3**2 + 2*d*n1*n2 + 2*e*n1*n3 + 2*f*n3*n2
        B = 2*(a*x*n1 + b*y*n2 + c*z*n3
               + d*x*n2 + d*y*n1 + e*x*n3 + e*z*n1 + f*z*n2 + f*y*n3
               + g*n1 + h*n2 + i*n3)
        C = j + a*x**2 + b*y**2 + c*z**2 + 2*(i*z + h*y + g*x + d*x*y + e*x*z + f*z*y)

        disc = B**2 - 4*A*C
        if disc < 0:
            return np.nan, np.nan, np.nan, np.nan, False
        elif disc == 0:
            t1 = -B / (2*A)
            p1 = line.p + t1 * line.dir
            return p1, np.nan, float(np.sign(t1) * np.linalg.norm(p1 - line.p)), np.nan, False
        else:
            t1 = (-B + np.sqrt(disc)) / (2*A)
            t2 = (-B - np.sqrt(disc)) / (2*A)
            p1 = line.p + t1 * line.dir
            p2 = line.p + t2 * line.dir
            d1 = float(np.sign(t1) * np.linalg.norm(p1 - line.p))
            d2 = float(np.sign(t2) * np.linalg.norm(p2 - line.p))
            is_inside = (d1 < 0) != (d2 < 0)
            ps = np.column_stack((p1, p2))
            dists = [d1, d2]
            idx = int(np.argmin(np.abs(dists)))
            return ps[:, idx], ps[:, 1-idx], dists[idx], dists[1-idx], is_inside

    def intersection_ray(self, ray):
        """Intersections with a ray (positive direction only). Same return signature as intersection_line."""
        p1, p2, d1, d2, is_inside = self.intersection_line(ray)
        if not np.isnan(d1) and not np.isnan(d2):
            if d1 < 0 < d2:
                return p2, np.nan, d2, np.nan, True
            elif d2 < 0 < d1:
                return p1, np.nan, d1, np.nan, True
            elif d1 < 0 and d2 < 0:
                return np.nan, np.nan, np.nan, np.nan, False
            return p1, p2, d1, d2, False
        elif not np.isnan(d1):
            if d1 < 0:
                return np.nan, np.nan, np.nan, np.nan, False
            return p1, np.nan, d1, np.nan, False
        return p1, p2, d1, d2, is_inside

    def intersection_line_segment(self, line_segment):
        """Intersections clipped to segment length. line_segment must have a ``length`` attribute."""
        p1, p2, d1, d2, is_inside = self.intersection_ray(line_segment)
        if not np.isnan(d1):
            if d1 > line_segment.length:
                return np.nan, np.nan, np.nan, np.nan, is_inside
            if not np.isnan(d2) and d2 > line_segment.length:
                return p1, np.nan, d1, np.nan, is_inside
        return p1, p2, d1, d2, is_inside


class QuadricsM:
    """Container managing a collection of analytic surfaces keyed by (tomo_id, feature_id)."""

    def __init__(self, input_data, quadric="ellipsoid", feature_id="object_id"):
        self.dict = {}
        self.f_id = feature_id

        if isinstance(input_data, cryomotl.Motl):
            input_motl = cryomotl.Motl.load(input_data)
            for f in input_motl.get_unique_values(column_name=feature_id):
                fm = input_motl.get_motl_subset(column_values=f, column_name=feature_id, reset_index=True)
                tomo_id = fm.df["tomo_id"].values[0]
                coords = fm.get_coordinates()
                if quadric == "ellipsoid":
                    self.dict[(tomo_id, f)] = Ellipsoid.fit_to_points(coords)
                else:
                    raise ValueError(f"Unsupported quadric type: {quadric!r}")
        elif isinstance(input_data, str):
            df = pd.read_csv(input_data)
            for _, row in df.iterrows():
                f = row[feature_id]
                tomo_id = row["tomo_id"]
                if quadric == "ellipsoid":
                    self.dict[(tomo_id, f)] = Ellipsoid.from_df(row)
                else:
                    raise ValueError(f"Unsupported quadric type: {quadric!r}")
        else:
            raise ValueError(f"Unsupported input type: {type(input_data)}")

        self.dict = {k: v for k, v in self.dict.items() if not v.singular}

    def write_out(self, output_name):
        """Write all surfaces to a CSV file."""
        frames = [
            surface.get_props_as_df().assign(tomo_id=tomo_id, **{self.f_id: f_id})
            for (tomo_id, f_id), surface in self.dict.items()
        ]
        pd.concat(frames, ignore_index=True).to_csv(output_name, index=False)

    def distance_point_surface(self, tomo_id, feature_id, points):
        """Return list of surface distances for each point."""
        surf = self.dict[(tomo_id, feature_id)]
        return [surf.distance_point_surface(p) for p in points]

    def distance_point_center(self, tomo_id, feature_id, points):
        """Return list of center distances for each point."""
        surf = self.dict[(tomo_id, feature_id)]
        return [surf.distance_point_center(p) for p in points]

    def find_closest_quadric(self, tomo_id, points):
        """Return feature_id of the closest surface (by center distance) for each point."""
        f_ids = [k[1] for k in self.dict if k[0] == tomo_id]
        n = len(points)
        closest_ids = np.full(n, -1)
        closest_dists = np.full(n, np.inf)
        for i, p in enumerate(points):
            for f in f_ids:
                d = self.dict[(tomo_id, f)].distance_point_center(p)
                if d < closest_dists[i]:
                    closest_dists[i] = d
                    closest_ids[i] = f
        return closest_ids
