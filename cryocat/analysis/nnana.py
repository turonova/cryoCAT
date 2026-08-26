"""Nearest-neighbor analysis for particle motls.

This module is organised in three layers:

1. **Stateless array helpers** (``find_nn_indices``, ``find_nn_within_radius``,
   ``centered_nn_coords``, ``rotated_nn_coords``, ``angular_distances``,
   ``relative_rotations``, ``rotations_to_unit_vectors``).  These take plain
   numpy arrays of coordinates and Euler angles and have no Motl dependency.
   Use them when you already have coords/angles in hand and only need
   per-particle NN geometry.

2. **The** ``NearestNeighbors`` **class** — motl-aware container that splits
   particles by ``feature_id`` (typically ``tomo_id``), runs the layer-1
   primitives per feature, and stores everything in ``self.df``.  Use it when
   you want a single object you can repeatedly query.

3. **Module-level wrappers** (``get_nn_stats``, ``get_nn_distances``,
   ``get_nn_rotations``, ``get_feature_nn_indices``, …).  Backward-compatible
   functions that accept motls and return numpy arrays / DataFrames.  Each is
   a thin shim around layer 1 / layer 2.

A standalone ``trace_chains`` function (plus its private helpers) lives at
the bottom of this file.  Tracing is NN-based but conceptually distinct from
the ``NearestNeighbors`` class; downstream chain analysis (occupancies,
chain stats, etc.) lives in ``cryocat.analysis.structure.Chain``.
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
import pandas as pd
import sklearn.neighbors as sn
from scipy.spatial.transform import Rotation as srot

from cryocat._types import (
    ArrayLike,
    ListLike,
    MapSource,
    MotlColumn,
    NNType,
    PathOrStr,
    RotationDistanceType,
)
from cryocat.core import cryomap, cryomotl
from cryocat.utils import geom, ioutils

if TYPE_CHECKING:
    # Import the alias lazily to avoid the cryomotl ↔ nnana circular import
    # (cryomotl loads nnana before its own module body finishes executing).
    from cryocat.core.cryomotl import MotlSource


# =============================================================================
# Layer 1 — stateless array-level helpers
# =============================================================================


def find_nn_indices(
    coords_qp: ArrayLike,
    coords_nn: ArrayLike,
    k: int = 1,
    remove_qp: bool = False,
    qp_labels: np.ndarray | None = None,
    nn_labels: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """k-nearest-neighbor search on raw coordinate arrays.

    Parameters
    ----------
    coords_qp : numpy.ndarray
        Query coordinates, shape ``(N, 3)``.
    coords_nn : numpy.ndarray
        Candidate-neighbor coordinates, shape ``(M, 3)``.
    k : int, default=1
        Number of neighbors to retrieve per query point.
    remove_qp : bool, default=False
        Set to True when ``coords_qp is coords_nn`` (or otherwise share
        particles) and the trivial zero-distance self-match should be dropped.
    qp_labels : numpy.ndarray or None, optional
        Shape ``(N,)``.  Label for each query particle.
    nn_labels : numpy.ndarray or None, optional
        Shape ``(M,)``.  Label for each candidate.  When both *qp_labels* and
        *nn_labels* are given, candidate ``j`` is excluded from query ``i``'s
        result when ``nn_labels[j] == qp_labels[i]``.

    Returns
    -------
    qp_idx : numpy.ndarray
        ``np.arange(N)``.
    nn_idx : numpy.ndarray
        Shape ``(N, k_eff)``.
    nn_dist : numpy.ndarray
        Shape ``(N, k_eff)``.
    k_eff : int
        ``min(k, M - int(remove_qp))`` — the number of neighbors actually
        retrieved per query point.
    """
    coords_qp = np.atleast_2d(coords_qp)
    coords_nn = np.atleast_2d(coords_nn)

    if qp_labels is not None and nn_labels is not None:
        # Determine how many extra candidates we need to absorb same-label exclusions.
        max_group = int(np.unique(nn_labels, return_counts=True)[1].max()) if len(nn_labels) else 0
        over_k = min(k + int(remove_qp) + max_group, coords_nn.shape[0])
        kdt = sn.KDTree(coords_nn)
        nn_dist_over, nn_idx_over = kdt.query(coords_qp, k=over_k)
        nn_dist_over = np.atleast_2d(nn_dist_over)
        nn_idx_over  = np.atleast_2d(nn_idx_over)
        # Build a keep-mask: True where candidate passes label filter (and self filter).
        mask = nn_labels[nn_idx_over] != qp_labels[:, None]
        if remove_qp:
            mask &= nn_idx_over != np.arange(nn_idx_over.shape[0])[:, None]
        # Stable argsort puts kept candidates (mask=True → ~mask=0) before excluded ones
        # while preserving distance order within the kept group.
        order = np.argsort(~mask, axis=1, kind="stable")
        sel  = np.take_along_axis(nn_idx_over,  order, axis=1)[:, :k]
        seld = np.take_along_axis(nn_dist_over, order, axis=1)[:, :k]
        keep = np.take_along_axis(mask,          order, axis=1)[:, :k]
        nn_idx  = np.where(keep, sel,  0).astype(int)
        nn_dist = np.where(keep, seld, 0.0)
        return np.arange(nn_idx.shape[0]), nn_idx, nn_dist, k

    query_k = k + 1 if remove_qp else k
    query_k = min(query_k, coords_nn.shape[0])
    kdt = sn.KDTree(coords_nn)
    nn_dist, nn_idx = kdt.query(coords_qp, k=query_k)
    nn_dist = np.atleast_2d(nn_dist)
    nn_idx  = np.atleast_2d(nn_idx)
    qp_idx  = np.arange(nn_idx.shape[0])
    if remove_qp:
        nn_dist = nn_dist[:, 1 : k + 1]
        nn_idx  = nn_idx[:, 1 : k + 1]
    return qp_idx, nn_idx, nn_dist, nn_dist.shape[1]


def find_nn_within_radius(
    coords_qp: ArrayLike,
    coords_nn: ArrayLike,
    radius: float,
    remove_qp: bool = False,
    qp_labels: np.ndarray | None = None,
    nn_labels: np.ndarray | None = None,
) -> tuple[list[int], list[np.ndarray]]:
    """Radius search on raw coordinate arrays.

    Parameters
    ----------
    coords_qp, coords_nn : numpy.ndarray
        Shapes ``(N, 3)`` and ``(M, 3)``.
    radius : float
        Search radius (same units as the coordinates).
    remove_qp : bool, default=False
        Drop self-matches (only meaningful when query and reference are the
        same set).
    qp_labels : numpy.ndarray or None, optional
        Shape ``(N,)``.  Label for each query particle.
    nn_labels : numpy.ndarray or None, optional
        Shape ``(M,)``.  Label for each candidate.  When both *qp_labels* and
        *nn_labels* are given, candidate ``j`` is excluded from query ``i``'s
        result when ``nn_labels[j] == qp_labels[i]``.

    Returns
    -------
    qp_idx : list of int
        Query indices that have at least one neighbor.
    nn_idx : list of numpy.ndarray
        For each kept query index, the sorted neighbor indices.
    """
    coords_qp = np.atleast_2d(coords_qp)
    coords_nn = np.atleast_2d(coords_nn)
    kdt = sn.KDTree(coords_nn)
    raw = kdt.query_radius(coords_qp, radius)

    qp_idx = []
    nn_idx = []
    for i, neighbors in enumerate(raw):
        if remove_qp:
            neighbors = neighbors[neighbors != i]
        if qp_labels is not None and nn_labels is not None:
            neighbors = neighbors[nn_labels[neighbors] != qp_labels[i]]
        if len(neighbors) > 0:
            qp_idx.append(i)
            nn_idx.append(np.sort(neighbors))
    return qp_idx, nn_idx


def find_nn_within_self(
    coords: ArrayLike,
    radius: float,
    unique_only: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Radius self-NN: each particle's neighbors within `radius`.

    Parameters
    ----------
    coords : numpy.ndarray
        Shape ``(N, 3)``.
    radius : float
        Search radius.
    unique_only : bool, default=True
        Deduplicate symmetric pairs (so pair ``(i, j)`` is reported once).

    Returns
    -------
    center_idx : numpy.ndarray or list of int
    nn_idx : list of numpy.ndarray
        Per-center neighbor indices (self excluded).
    """

    def _unique_arrays(list_of_arrays):
        seen, out = set(), []
        for la in list_of_arrays:
            t = tuple(la)
            if t not in seen:
                out.append(la)
                seen.add(t)
        return out

    coords = np.atleast_2d(coords)
    kdt = sn.KDTree(coords)
    nn_idx = kdt.query_radius(coords, radius)

    ordered_idx = np.arange(nn_idx.shape[0])
    keep = [i for i, row in zip(ordered_idx, nn_idx) if len(row) > 1]
    nn_indices = nn_idx[keep]

    if unique_only:
        sorted_rows = [np.sort(row) for row in nn_indices]
        sorted_rows = _unique_arrays(sorted_rows)
        center_idx = np.array([row[0] for row in sorted_rows])
        nn_idx_out = [row[1:] for row in sorted_rows]
    else:
        center_idx = np.array(keep)
        nn_idx_out = [
            np.array([e for e in row if e != center_idx[i]])
            for i, row in enumerate(nn_indices)
        ]

    return center_idx, nn_idx_out


def nms_by_distance(
    coords: ArrayLike,
    scores: ArrayLike,
    distance: float,
    keep_greater: bool = True,
) -> np.ndarray:
    """Greedy non-maximum suppression by Euclidean distance.

    Walks through the points in score order and keeps the current one, then
    suppresses all not-yet-suppressed points within *distance* of it. Identical
    in spirit to bounding-box NMS in object detection, only the suppression
    criterion is pairwise Euclidean distance rather than IoU.

    Parameters
    ----------
    coords : array-like
        Shape ``(N, 3)``. Point positions.
    scores : array-like
        Shape ``(N,)``. Per-point score used to break ties.
    distance : float
        Suppression radius. Points within this distance of a kept point
        are removed.
    keep_greater : bool, default=True
        When ``True``, points are visited in descending score order, so the
        highest-scoring point in each cluster survives. When ``False``,
        ascending order; the lowest-scoring point survives.

    Returns
    -------
    numpy.ndarray
        Boolean keep-mask of shape ``(N,)``.
    """
    coords = np.asarray(coords)
    scores = np.asarray(scores)
    n = coords.shape[0]

    sort_idx = np.argsort(scores)
    if keep_greater:
        sort_idx = sort_idx[::-1]

    keep = np.ones(n, dtype=bool)
    for j in sort_idx:
        if not keep[j]:
            continue
        dist = geom.point_pairwise_dist(coords[j, :], coords)
        within = dist < distance
        within[j] = False  # never suppress the current point
        keep[within] = False

    return keep


def centered_nn_coords(
    coords_qp: ArrayLike,
    qp_idx: ArrayLike,
    coords_nn: ArrayLike,
    nn_idx: ArrayLike,
    pixel_size: float = 1.0,
) -> np.ndarray:
    """Per-pair centered coordinates: ``coords_nn[nn_idx] - coords_qp[qp_idx]``.

    Parameters
    ----------
    coords_qp, coords_nn : numpy.ndarray
        Shapes ``(N, 3)`` and ``(M, 3)``.
    qp_idx : numpy.ndarray
        Shape ``(N,)``.
    nn_idx : numpy.ndarray
        Shape ``(N, k)``.
    pixel_size : float, default=1.0

    Returns
    -------
    numpy.ndarray
        Shape ``(N * k, 3)``.
    """
    nn_idx = np.atleast_2d(nn_idx)
    k = nn_idx.shape[1]
    qp_expanded = np.repeat(qp_idx, k)
    nn_flat = nn_idx.reshape(-1)
    return (coords_nn[nn_flat] - coords_qp[qp_expanded]) * pixel_size


def rotated_nn_coords(
    centered_coords: ArrayLike,
    qp_angles_per_pair: ArrayLike,
) -> np.ndarray:
    """Express centered NN coordinates in the local frame of each query point.

    Parameters
    ----------
    centered_coords : numpy.ndarray
        Shape ``(M, 3)``.
    qp_angles_per_pair : numpy.ndarray
        Shape ``(M, 3)``, zxz Euler degrees.

    Returns
    -------
    numpy.ndarray
        Shape ``(M, 3)``.
    """
    inv_angles = -qp_angles_per_pair[:, [2, 1, 0]]
    rot = srot.from_euler("zxz", angles=inv_angles, degrees=True)
    return rot.apply(centered_coords)


def angular_distances(
    qp_angles_per_pair: ArrayLike,
    nn_angles_per_pair: ArrayLike,
    rotation_type: RotationDistanceType = "angular_distance",
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-pair angular distances between qp and nn rotations.

    Parameters
    ----------
    qp_angles_per_pair, nn_angles_per_pair : numpy.ndarray
        Shape ``(M, 3)``, zxz Euler in degrees.
    rotation_type : str, default='angular_distance'
        One of ``{"all", "angular_distance", "cone_distance", "in_plane_distance"}``.

    Returns
    -------
    numpy.ndarray (or tuple of three for ``rotation_type='all'``)
    """
    return geom.compare_rotations(
        qp_angles_per_pair, nn_angles_per_pair, rotation_type=rotation_type
    )


def relative_rotations(
    qp_angles_per_pair: ArrayLike,
    nn_angles_per_pair: ArrayLike,
) -> srot:
    """Return the qp→nn relative rotation as a scipy ``Rotation`` object.

    Computes ``R_qp⁻¹ · R_nn`` for each pair, i.e. the rotation that
    transforms the qp orientation into the nn orientation.

    Parameters
    ----------
    qp_angles_per_pair : numpy.ndarray
        Shape ``(M, 3)``, zxz Euler angles in degrees.
    nn_angles_per_pair : numpy.ndarray
        Shape ``(M, 3)``, zxz Euler angles in degrees.

    Returns
    -------
    scipy.spatial.transform.Rotation
        Length-``M`` Rotation object.
    """
    inv_qp = -qp_angles_per_pair[:, [2, 1, 0]]
    rot_qp_to_zero = srot.from_euler("zxz", angles=inv_qp, degrees=True)
    rot_nn = srot.from_euler("zxz", angles=nn_angles_per_pair, degrees=True)
    return rot_qp_to_zero * rot_nn


def rotations_to_unit_vectors(rotations: srot) -> tuple[np.ndarray, np.ndarray]:
    """Convert rotations to their representative unit vectors and Euler angles.

    Parameters
    ----------
    rotations : scipy.spatial.transform.Rotation
        Length-``M`` Rotation object.

    Returns
    -------
    points_on_sphere : numpy.ndarray
        Shape ``(M, 3)``.  Unit vectors on the sphere obtained by applying
        each rotation to a reference direction.
    euler_angles : numpy.ndarray
        Shape ``(M, 3)``.  zxz Euler angles in degrees.
    """
    points = geom.rotations_to_z_normals(rotations)
    angles = rotations.as_euler("zxz", degrees=True)
    return points, angles


# =============================================================================
# NearestNeighbors class
# =============================================================================


class NearestNeighbors:
    """Container holding a per-pair NN DataFrame for one or two motls.

    Parameters
    ----------
    input_data : str, Motl or list of (str / Motl), optional
        A single motl (or path) or a list of motls. If a single motl is
        given, NN search is run on that motl against itself (with
        ``remove_qp`` forced to True). If a list is given, the first
        element is the query and each subsequent element is searched
        against the query.
    column_name : str, default='tomo_id'
        Column name used to partition particles before NN search; only
        particles sharing the same value (e.g. same tomogram) are paired.
    nn_type : NNType, default='closest_dist'
        Search mode -- one of {"closest_dist", "radius"}.
    type_param : int or float, optional
        Parameter for the search mode. For ``"closest_dist"`` it is the
        number K of nearest neighbours to keep per query (cast to int).
        For ``"radius"`` it is the search radius in voxels. Defaults to 1
        (or 1 voxel) when omitted.
    remove_qp : bool, optional
        Drop the query particle from its own neighbour set. Forced True
        when ``input_data`` is a single motl. Defaults to None (False
        unless self-pairing).
    remove_duplicates : bool, default=False
        Drop duplicate ``(qp_id, nn_id)`` pairs from the resulting table.
    paired : bool, default=False
        If True, angles are taken from ``motl_a`` only (entry/exit pairs).
    exclude_column_name : str or None, default=None
        When set, NN candidates sharing the query particle's value in this
        column are excluded from the result.  See also :meth:`add_motl_columns`.

    Notes
    -----
    ``self.motls`` holds live references to the loaded Motl objects
    (``motls[0]`` = query, ``motls[1:]`` = neighbour motls).  Mutating the
    originals is visible here; call ``motl.copy()`` explicitly if you need a
    snapshot.
    """

    _QP_COORD_COLS = ["qp_coord_x", "qp_coord_y", "qp_coord_z"]
    _NN_COORD_COLS = ["nn_coord_x", "nn_coord_y", "nn_coord_z"]
    _QP_ANGLE_COLS = ["qp_angles_phi", "qp_angles_theta", "qp_angles_psi"]
    _NN_ANGLE_COLS = ["nn_angles_phi", "nn_angles_theta", "nn_angles_psi"]
    _NORM_COORD_COLS = ["norm_nn_x", "norm_nn_y", "norm_nn_z"]
    _ROT_COORD_COLS = ["rot_nn_x", "rot_nn_y", "rot_nn_z"]

    def __init__(
        self,
        input_data: MotlSource | list[MotlSource] | None = None,
        column_name: MotlColumn = "tomo_id",
        nn_type: NNType = "closest_dist",
        type_param: float | None = None,
        remove_qp: bool | None = None,
        remove_duplicates: bool = False,
        paired: bool = False,
        exclude_column_name: MotlColumn | None = None,
    ) -> None:
        if input_data is None:
            self.features = None
            self.df = None
            self.motls = None
            self.column_name = column_name
            self.paired = paired
            self.exclude_column_name = exclude_column_name
            return

        self.column_name = column_name
        self.paired = paired
        self.exclude_column_name = exclude_column_name

        if exclude_column_name is not None and exclude_column_name == column_name:
            import warnings
            warnings.warn(
                f"exclude_column_name={exclude_column_name!r} is the same as column_name; "
                "this excludes the entire partition (all candidates share the feature label).",
                UserWarning, stacklevel=2,
            )

        if not isinstance(input_data, list):
            motl_list = [cryomotl.Motl.load(input_data), cryomotl.Motl.load(input_data)]
            single_motl = True
        else:
            motl_list = [cryomotl.Motl.load(m) for m in input_data]
            single_motl = False

        single_motl = bool(remove_qp) or single_motl

        features = motl_list[0].get_unique_values(column_name)
        for m in motl_list[1:]:
            features = np.intersect1d(features, m.get_unique_values(column_name), assume_unique=True)

        columns = [
            "motl_id", column_name,
            "qp_id", "qp_subtomo_id",
            "nn_id", "nn_subtomo_id",
            *self._QP_ANGLE_COLS, *self._QP_COORD_COLS,
            *self._NN_ANGLE_COLS, *self._NN_COORD_COLS,
        ]
        if nn_type == "closest_dist":
            columns.append("nn_dist")

        # Pre-extract per-motl arrays once; slice per feature via groupby indices
        # to avoid repeated get_motl_subset / get_coordinates / get_angles calls.
        _m_coords   = [m.get_coordinates()               for m in motl_list]
        _m_angles   = [m.get_angles()                    for m in motl_list]
        _m_subtomos = [m.df["subtomo_id"].values         for m in motl_list]
        _m_labels   = [
            m.df[exclude_column_name].values if exclude_column_name else None
            for m in motl_list
        ]
        _m_groups   = [dict(m.df.groupby(column_name).indices) for m in motl_list]

        results = []
        for f in features:
            qp_rows     = _m_groups[0][f]
            qp_subtomos = _m_subtomos[0][qp_rows]
            qp_coord    = _m_coords[0][qp_rows]
            qp_angles   = _m_angles[0][qp_rows]
            qp_labels   = _m_labels[0][qp_rows] if _m_labels[0] is not None else None

            for motl_idx in range(1, len(motl_list)):
                nn_rows     = _m_groups[motl_idx][f]
                nn_subtomos = _m_subtomos[motl_idx][nn_rows]
                nn_coord    = _m_coords[motl_idx][nn_rows]
                nn_angles   = qp_angles if paired else _m_angles[motl_idx][nn_rows]
                nn_labels   = _m_labels[motl_idx][nn_rows] if _m_labels[motl_idx] is not None else None

                if nn_type == "closest_dist":
                    # type_param is the K of K-nearest neighbours; sklearn's
                    # kneighbors needs an int, but the GUI form yields a float.
                    nn_count = int(type_param) if type_param else 1
                    qp_idx, nn_idx, nn_dist, _ = find_nn_indices(
                        qp_coord, nn_coord, k=nn_count,
                        remove_qp=single_motl or paired,
                        qp_labels=qp_labels, nn_labels=nn_labels,
                    )
                    stacked = self._stack_nn_results(
                        motl_idx, column_name, f, qp_idx, nn_idx,
                        qp_subtomos, nn_subtomos,
                        qp_angles, nn_angles,
                        qp_coord, nn_coord,
                        nn_dist=nn_dist,
                    )
                elif nn_type == "radius":
                    radius = type_param or 1
                    qp_idx, nn_idx_list = find_nn_within_radius(
                        qp_coord, nn_coord, radius=radius,
                        remove_qp=single_motl or paired,
                        qp_labels=qp_labels, nn_labels=nn_labels,
                    )
                    stacked = self._stack_nn_results_radius(
                        motl_idx, column_name, f, qp_idx, nn_idx_list,
                        qp_subtomos, nn_subtomos,
                        qp_angles, nn_angles,
                        qp_coord, nn_coord,
                    )
                else:
                    raise ValueError(
                        f"The type {nn_type} is not supported, choose between 'closest_dist' and 'radius'."
                    )

                if stacked is not None:
                    results.append(stacked)

        if not results:
            self.df = pd.DataFrame(columns=columns)
        else:
            merged = {col: np.concatenate([r[col] for r in results]) for col in results[0]}
            self.df = pd.DataFrame(merged, columns=columns)

        int_cols = ["motl_id", column_name, "qp_id", "qp_subtomo_id", "nn_id", "nn_subtomo_id"]
        float_cols = [
            *self._QP_ANGLE_COLS, *self._QP_COORD_COLS,
            *self._NN_ANGLE_COLS, *self._NN_COORD_COLS,
        ]
        if nn_type == "closest_dist":
            float_cols.append("nn_dist")
        if not self.df.empty:
            self.df = self.df.astype(
                {**{c: np.int32 for c in int_cols}, **{c: np.float32 for c in float_cols}}
            )

        if remove_duplicates:
            self.df = self.drop_symmetric_duplicates()
        self.features = features
        self.motls = motl_list

    @staticmethod
    def _stack_nn_results(
        motl_idx: int,
        column_name: str,
        column_value: Any,
        qp_idx: np.ndarray,
        nn_idx: np.ndarray,
        qp_subtomos: np.ndarray,
        nn_subtomos: np.ndarray,
        qp_angles: np.ndarray,
        nn_angles: np.ndarray,
        qp_coord: np.ndarray,
        nn_coord: np.ndarray,
        nn_dist: np.ndarray,
    ) -> dict | None:
        """Stack k-NN pair data into a column-keyed dict of 1-D arrays.

        Returns ``None`` when *k* is 0.  Each value is a 1-D array of length
        ``N * k``; integer columns are returned as integer arrays (no float64
        detour).
        """
        nn_idx  = np.atleast_2d(nn_idx)
        nn_dist = np.atleast_2d(nn_dist)
        k = nn_idx.shape[1]
        if k == 0:
            return None
        qp_expanded = np.tile(qp_idx, k)
        nn_flat     = nn_idx.T.reshape(-1)
        n_pairs     = len(nn_flat)
        qa = qp_angles[qp_expanded]
        qc = qp_coord[qp_expanded]
        na = nn_angles[nn_flat]
        nc = nn_coord[nn_flat]
        return {
            "motl_id":           np.repeat(motl_idx,     n_pairs),
            column_name:         np.repeat(column_value, n_pairs),
            "qp_id":             qp_expanded,
            "qp_subtomo_id":     qp_subtomos[qp_expanded],
            "nn_id":             nn_flat,
            "nn_subtomo_id":     nn_subtomos[nn_flat],
            "qp_angles_phi":     qa[:, 0],
            "qp_angles_theta":   qa[:, 1],
            "qp_angles_psi":     qa[:, 2],
            "qp_coord_x":        qc[:, 0],
            "qp_coord_y":        qc[:, 1],
            "qp_coord_z":        qc[:, 2],
            "nn_angles_phi":     na[:, 0],
            "nn_angles_theta":   na[:, 1],
            "nn_angles_psi":     na[:, 2],
            "nn_coord_x":        nc[:, 0],
            "nn_coord_y":        nc[:, 1],
            "nn_coord_z":        nc[:, 2],
            "nn_dist":           nn_dist.T.reshape(-1),
        }

    @staticmethod
    def _stack_nn_results_radius(
        motl_idx: int,
        column_name: str,
        column_value: Any,
        qp_idx: list[int] | np.ndarray,
        nn_idx_list: list[np.ndarray],
        qp_subtomos: np.ndarray,
        nn_subtomos: np.ndarray,
        qp_angles: np.ndarray,
        nn_angles: np.ndarray,
        qp_coord: np.ndarray,
        nn_coord: np.ndarray,
    ) -> dict | None:
        """Stack radius-NN pair data into a column-keyed dict of 1-D arrays.

        Returns ``None`` when ``nn_idx_list`` is empty or the total neighbor
        count is zero.
        """
        if not nn_idx_list:
            return None
        counts = np.array([len(n) for n in nn_idx_list])
        if counts.sum() == 0:
            return None
        qp_idx_arr  = np.asarray(qp_idx)
        nn_flat     = np.concatenate(nn_idx_list).astype(int)
        qp_expanded = np.repeat(qp_idx_arr, counts)
        n_pairs     = counts.sum()
        qa = qp_angles[qp_expanded]
        qc = qp_coord[qp_expanded]
        na = nn_angles[nn_flat]
        nc = nn_coord[nn_flat]
        return {
            "motl_id":           np.repeat(motl_idx,     n_pairs),
            column_name:         np.repeat(column_value, n_pairs),
            "qp_id":             qp_expanded,
            "qp_subtomo_id":     qp_subtomos[qp_expanded],
            "nn_id":             nn_flat,
            "nn_subtomo_id":     nn_subtomos[nn_flat],
            "qp_angles_phi":     qa[:, 0],
            "qp_angles_theta":   qa[:, 1],
            "qp_angles_psi":     qa[:, 2],
            "qp_coord_x":        qc[:, 0],
            "qp_coord_y":        qc[:, 1],
            "qp_coord_z":        qc[:, 2],
            "nn_angles_phi":     na[:, 0],
            "nn_angles_theta":   na[:, 1],
            "nn_angles_psi":     na[:, 2],
            "nn_coord_x":        nc[:, 0],
            "nn_coord_y":        nc[:, 1],
            "nn_coord_z":        nc[:, 2],
        }

    def drop_symmetric_duplicates(self) -> pd.DataFrame:
        """Return a copy of ``self.df`` with symmetric (a, b)/(b, a) pairs deduped."""
        a  = self.df["qp_subtomo_id"].values
        b  = self.df["nn_subtomo_id"].values
        lo = np.minimum(a, b)
        hi = np.maximum(a, b)
        pairs = np.empty(len(lo), dtype=[("lo", lo.dtype), ("hi", hi.dtype)])
        pairs["lo"] = lo
        pairs["hi"] = hi
        _, keep_idx = np.unique(pairs, return_index=True)
        return self.df.iloc[np.sort(keep_idx)]

    def get_unique_values(self) -> np.ndarray:
        """Return the feature values present in ``self.df``.

        Returns
        -------
        numpy.ndarray
            Unique feature values (e.g. tomogram IDs).
        """
        return self.features

    def get_nn_subset(
        self,
        motl_id_values: "ListLike[int]",
        column_values: Any,
    ) -> "NearestNeighbors":
        """Return a new :class:`NearestNeighbors` restricted to the given subset.

        Parameters
        ----------
        motl_id_values : int or list of int
            ``motl_id`` values to keep.
        column_values : scalar or list
            Feature values (e.g. tomo IDs) to keep.

        Returns
        -------
        NearestNeighbors
            New instance with a filtered ``df`` and matching ``features``.
        """
        from cryocat.utils.classutils import as_list

        sub = NearestNeighbors()
        sub.column_name = self.column_name
        sub.paired = self.paired
        sub.motls = self.motls
        motl_id_values = as_list(motl_id_values)
        column_values = as_list(column_values)
        sub.df = self.df[
            (self.df["motl_id"].isin(motl_id_values))
            & (self.df[self.column_name].isin(column_values))
        ].copy()
        sub.features = column_values
        return sub

    def write_out(self, file_path: PathOrStr) -> None:
        """Write the per-pair NN table to CSV.

        The ``motl_id`` and feature (``column_name``) columns are written first
        and second so :meth:`load` can recover the feature column by position
        regardless of any extra columns present.  Motls and metadata are not
        stored; reattach motls via :meth:`load`.

        Parameters
        ----------
        file_path : PathOrStr
            Destination CSV path.
        """
        lead = ["motl_id", self.column_name]
        cols = lead + [c for c in self.df.columns if c not in lead]
        self.df[cols].to_csv(file_path, index=False)

    @classmethod
    def get_required_columns(cls, column_name: str = "tomo_id") -> list[str]:
        """Return every column name that a saved NearestNeighbors CSV must contain."""
        return [
            "motl_id", column_name,
            "qp_id", "qp_subtomo_id",
            "nn_id", "nn_subtomo_id",
            *cls._QP_ANGLE_COLS,
            *cls._QP_COORD_COLS,
            *cls._NN_ANGLE_COLS,
            *cls._NN_COORD_COLS,
        ]

    @staticmethod
    def check_nn_columns(df: "pd.DataFrame", column_name: str = "tomo_id") -> list[str]:
        """Return column names required by NearestNeighbors that are missing from df."""
        required = NearestNeighbors.get_required_columns(column_name=column_name)
        return [c for c in required if c not in df.columns]

    @classmethod
    def load(
        cls,
        file_path: PathOrStr,
        motls: "MotlSource | list[MotlSource] | None" = None,
        column_name: "MotlColumn | None" = None,
        paired: bool = False,
        exclude_column_name: "MotlColumn | None" = None,
    ) -> "NearestNeighbors":
        """Load a ``NearestNeighbors`` instance from a CSV written by :meth:`write_out`.

        Parameters
        ----------
        file_path : PathOrStr
            Path to the CSV file.
        motls : MotlSource or list of MotlSource or None, optional
            Source motls to reattach, in the same order as the original
            ``input_data`` (``[qp, nn1, nn2, …]``).  Pass ``None`` to get a
            df-only instance (``motls`` will be ``None``).
        column_name : MotlColumn or None, optional
            Feature column used to partition the NN search.  Inferred from the
            second column of the CSV when not given (``write_out`` guarantees
            ``motl_id`` first and the feature column second).
        paired : bool, default=False
            Stored as provenance only; does not affect behaviour of the loaded
            instance.
        exclude_column_name : MotlColumn or None, optional
            Stored as provenance only.

        Returns
        -------
        NearestNeighbors
            Loaded instance with ``df``, ``column_name``, ``features``, and
            optionally ``motls`` populated.

        Raises
        ------
        ValueError
            If the first column of the CSV is not ``"motl_id"`` and
            ``column_name`` was not supplied, or if the supplied motl list does
            not cover all ``motl_id`` values in the table, or if any
            ``qp_subtomo_id`` / ``nn_subtomo_id`` value is absent from the
            corresponding source motl.
        """
        from cryocat.utils.classutils import as_list

        df = ioutils.df_load(str(file_path))

        # ── column_name recovery ─────────────────────────────────────────────
        if column_name is None:
            if df.columns[0] != "motl_id":
                raise ValueError(
                    f"First column of {file_path!r} is {df.columns[0]!r}, not 'motl_id'. "
                    "The file does not follow the NearestNeighbors CSV convention; "
                    "pass column_name explicitly."
                )
            column_name = str(df.columns[1])

        # ── re-apply dtypes ──────────────────────────────────────────────────
        int_base = ["motl_id", column_name, "qp_id", "qp_subtomo_id", "nn_id", "nn_subtomo_id"]
        float_patterns = ("_angle", "_coord", "nn_dist")
        int_cols = {c for c in int_base if c in df.columns}
        float_cols = {
            c for c in df.columns
            if any(p in c for p in float_patterns) and c not in int_cols
        }
        cast = {**{c: np.int32 for c in int_cols}, **{c: np.float32 for c in float_cols}}
        if cast:
            df = df.astype(cast)

        # ── build instance ───────────────────────────────────────────────────
        obj = cls()
        obj.df = df
        obj.column_name = column_name
        obj.paired = paired
        obj.exclude_column_name = exclude_column_name
        obj.features = np.unique(df[column_name].values) if not df.empty else df[column_name].values

        # ── reattach motls ───────────────────────────────────────────────────
        if motls is None:
            obj.motls = None
        else:
            motl_list = [cryomotl.Motl.load(m) for m in as_list(motls)]
            obj.motls = motl_list

            # validate coverage
            max_mid = int(df["motl_id"].max())
            if max_mid >= len(motl_list):
                raise ValueError(
                    f"motl_id {max_mid} in the table requires motl_list[{max_mid}], "
                    f"but only {len(motl_list)} motl(s) were supplied. "
                    "Check the motl list order: [qp_motl, nn_motl1, nn_motl2, …]."
                )
            qp_ids = set(motl_list[0].df["subtomo_id"].values)
            missing_qp = set(df["qp_subtomo_id"].values) - qp_ids
            if missing_qp:
                raise ValueError(
                    f"{len(missing_qp)} qp_subtomo_id value(s) not found in motls[0] "
                    f"(first missing: {next(iter(missing_qp))})."
                )
            for mid in df["motl_id"].unique():
                nn_ids = set(motl_list[int(mid)].df["subtomo_id"].values)
                mask = df["motl_id"] == mid
                missing_nn = set(df.loc[mask, "nn_subtomo_id"].values) - nn_ids
                if missing_nn:
                    raise ValueError(
                        f"{len(missing_nn)} nn_subtomo_id value(s) for motl_id={mid} "
                        f"not found in motls[{int(mid)}] "
                        f"(first missing: {next(iter(missing_nn))})."
                    )

        return obj

    def get_normalized_coord(self, add_to_df: bool = True) -> np.ndarray:
        """Return centered NN coordinates ``nn_coord - qp_coord``.

        Parameters
        ----------
        add_to_df : bool, default=True
            Store the result in ``self.df`` under columns
            ``norm_nn_x/y/z`` for reuse.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, 3)``.  Coordinates are in the same units as the
            motl (voxels unless a pixel size was applied).
        """
        if all(c in self.df.columns for c in self._NORM_COORD_COLS):
            return self.df[self._NORM_COORD_COLS].to_numpy()
        norm = (
            self.df[self._NN_COORD_COLS].to_numpy()
            - self.df[self._QP_COORD_COLS].to_numpy()
        )
        if add_to_df:
            self.df[self._NORM_COORD_COLS] = norm
        return norm

    def get_rotated_coord(self, add_to_df: bool = True) -> np.ndarray:
        """Return centered NN coordinates rotated into the qp local frame.

        Parameters
        ----------
        add_to_df : bool, default=True
            Store the result in ``self.df`` under columns
            ``rot_nn_x/y/z`` for reuse.

        Returns
        -------
        numpy.ndarray
            Shape ``(N, 3)``.
        """
        if all(c in self.df.columns for c in self._ROT_COORD_COLS):
            return self.df[self._ROT_COORD_COLS].to_numpy()
        centered = self.get_normalized_coord(add_to_df=add_to_df)
        rot = rotated_nn_coords(centered, self.df[self._QP_ANGLE_COLS].to_numpy())
        if add_to_df:
            self.df[self._ROT_COORD_COLS] = rot
        return rot

    def add_motl_columns(
        self,
        column_names: "ListLike[MotlColumn]",
        *,
        sides: "ListLike[str]" = ("qp", "nn"),
        add_to_df: bool = True,
    ) -> "NearestNeighbors":
        """Enrich ``self.df`` with extra columns pulled from the source motls.

        Parameters
        ----------
        column_names : str or list of str
            One or more Motl column names to pull (e.g. ``"object_id"``).
        sides : str or list of str, default=("qp", "nn")
            Which side(s) to populate.  Each entry must be ``"qp"`` or ``"nn"``.
        add_to_df : bool, default=True
            Write the new columns into ``self.df`` and return ``self``; when
            ``False``, return without modifying ``self.df`` (dry-run / validate).

        Returns
        -------
        NearestNeighbors
            ``self`` (for chaining).

        Raises
        ------
        RuntimeError
            When ``self.motls is None`` (instance was created with
            ``input_data=None``).
        KeyError
            When a requested column is absent from a source motl.
        """
        from cryocat.utils.classutils import as_list

        if self.motls is None:
            raise RuntimeError(
                "No source motls stored on this instance. "
                "Construct NearestNeighbors with input_data != None to use add_motl_columns."
            )
        column_names = as_list(column_names)
        sides = as_list(sides)

        for col in column_names:
            if "qp" in sides:
                src = self.motls[0].df.set_index("subtomo_id")
                if col not in src.columns:
                    raise KeyError(f"Column {col!r} not found in query motl.")
                self.df[f"qp_{col}"] = self.df["qp_subtomo_id"].map(src[col])
            if "nn" in sides:
                nn_col_data = pd.Series(index=self.df.index, dtype=object)
                for mid in self.df["motl_id"].unique():
                    mask = self.df["motl_id"] == mid
                    src = self.motls[int(mid)].df.set_index("subtomo_id")
                    if col not in src.columns:
                        raise KeyError(f"Column {col!r} not found in motl[{int(mid)}].")
                    nn_col_data[mask] = self.df.loc[mask, "nn_subtomo_id"].map(src[col])
                self.df[f"nn_{col}"] = nn_col_data
        return self

    def get_qp_rotations(self) -> srot:
        """Return the query-particle rotations as a scipy ``Rotation`` object.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Length-``N`` Rotation parsed from the zxz Euler angles stored in
            ``self.df``.
        """
        return srot.from_euler("zxz", degrees=True, angles=self.df[self._QP_ANGLE_COLS].to_numpy())

    def get_nn_rotations(self) -> srot:
        """Return the nearest-neighbor rotations as a scipy ``Rotation`` object.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Length-``N`` Rotation parsed from the zxz Euler angles stored in
            ``self.df``.
        """
        return srot.from_euler("zxz", degrees=True, angles=self.df[self._NN_ANGLE_COLS].to_numpy())

    def get_relative_rotations(self) -> srot:
        """Return per-pair qp→nn relative rotations.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Length-``N`` Rotation representing ``R_qp⁻¹ · R_nn`` for each
            pair in ``self.df``.
        """
        return relative_rotations(
            self.df[self._QP_ANGLE_COLS].to_numpy(),
            self.df[self._NN_ANGLE_COLS].to_numpy(),
        )

    def get_angular_distances(
        self,
        rotation_type: RotationDistanceType = "angular_distance",
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return per-pair angular distances between qp and nn orientations.

        Parameters
        ----------
        rotation_type : RotationDistanceType, default='angular_distance'
            One of ``{"all", "angular_distance", "cone_distance",
            "in_plane_distance"}``.  ``"all"`` returns a tuple of three
            arrays.

        Returns
        -------
        numpy.ndarray or tuple of numpy.ndarray
            Angular distances in degrees.  Shape ``(N,)`` for a single type;
            tuple of three ``(N,)`` arrays for ``"all"``.
        """
        return angular_distances(
            self.df[self._QP_ANGLE_COLS].to_numpy(),
            self.df[self._NN_ANGLE_COLS].to_numpy(),
            rotation_type=rotation_type,
        )

    def to_stats_dataframe(
        self,
        pixel_size: float = 1.0,
        rotation_type: RotationDistanceType = "angular_distance",
    ) -> pd.DataFrame:
        """Build the canonical per-pair statistics DataFrame.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Pixel size in Å.  Distances and coordinates are multiplied by
            this value before being stored.
        rotation_type : str, default='angular_distance'
            One of ``{"angular_distance", "cone_distance", "in_plane_distance"}``.
            Determines which angular metric is stored in the ``angular_distance``
            column.

        Returns
        -------
        pandas.DataFrame
            One row per nearest-neighbor pair.  Columns:

            ``distance``
                Euclidean distance between qp and nn in physical units.
            ``coord_x/y/z``
                Centered ``nn - qp`` displacement in physical units.
            ``coord_rx/ry/rz``
                Displacement rotated into the qp local frame.
            ``angular_distance``
                Rotation distance (type controlled by *rotation_type*).
            ``rot_x/y/z``
                Unit-vector representation of the qp→nn relative rotation.
            ``phi/theta/psi``
                zxz Euler angles of the relative rotation (degrees).
            ``subtomo_idx`` / ``subtomo_nn_idx``
                Subtomogram IDs of qp and nn.
            ``type``
                Always ``"nn"``.

        Raises
        ------
        ValueError
            When the instance was constructed with ``nn_type='radius'`` (no
            ``nn_dist`` column is available).
        """
        if "nn_dist" not in self.df.columns:
            raise ValueError(
                "to_stats_dataframe requires a 'closest_dist' run "
                "(no nn_dist column in self.df)."
            )
        centered = self.get_normalized_coord(add_to_df=False) * pixel_size
        rotated = rotated_nn_coords(centered, self.df[self._QP_ANGLE_COLS].to_numpy())
        ang = angular_distances(
            self.df[self._QP_ANGLE_COLS].to_numpy(),
            self.df[self._NN_ANGLE_COLS].to_numpy(),
            rotation_type=rotation_type,
        )
        rel = self.get_relative_rotations()
        points, angles = rotations_to_unit_vectors(rel)
        distance = self.df["nn_dist"].to_numpy() * pixel_size

        out = pd.DataFrame({
            "distance":         distance,
            "coord_x":          centered[:, 0],
            "coord_y":          centered[:, 1],
            "coord_z":          centered[:, 2],
            "coord_rx":         rotated[:, 0],
            "coord_ry":         rotated[:, 1],
            "coord_rz":         rotated[:, 2],
            "angular_distance": ang,
            "rot_x":            points[:, 0],
            "rot_y":            points[:, 1],
            "rot_z":            points[:, 2],
            "phi":              angles[:, 0],
            "theta":            angles[:, 1],
            "psi":              angles[:, 2],
            "subtomo_idx":      self.df["qp_subtomo_id"].to_numpy(),
            "subtomo_nn_idx":   self.df["nn_subtomo_id"].to_numpy(),
        })
        out["type"] = "nn"
        return out


# =============================================================================
# Layer 2 — motl-accepting wrappers (backward compatible)
# =============================================================================


def get_feature_nn_indices(
    motl_a: MotlSource,
    motl_nn: MotlSource,
    nn_number: int = 1,
    remove_qp: bool = False,
    column_name: MotlColumn = "tomo_id",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """k-nearest-neighbor indices and distances for two motls.

    Thin wrapper around :func:`find_nn_indices` that accepts motl paths or
    objects instead of raw coordinate arrays.

    Parameters
    ----------
    motl_a : str or Motl
        Query motl.
    motl_nn : str or Motl
        Neighbor motl.
    nn_number : int, default=1
        Number of neighbors to retrieve per query point.
    remove_qp : bool, default=False
        Drop self-matches (use when query and neighbor are the same motl).
    column_name : str, default='tomo_id'
        Not used by this function; kept for API symmetry.

    Returns
    -------
    qp_idx : numpy.ndarray
    nn_idx : numpy.ndarray, shape ``(N, nn_number)``
    nn_dist : numpy.ndarray, shape ``(N, nn_number)``
    k_eff : int
    """
    motl_a = cryomotl.Motl.load(motl_a)
    motl_nn = cryomotl.Motl.load(motl_nn)
    return find_nn_indices(motl_a.get_coordinates(), motl_nn.get_coordinates(),
                           k=nn_number, remove_qp=remove_qp)


def get_feature_nn_within_radius(
    motl_a: MotlSource,
    motl_nn: MotlSource,
    radius: float,
    remove_qp: bool = False,
) -> tuple[list[int], list[np.ndarray]]:
    """Radius search for two motls.

    Thin wrapper around :func:`find_nn_within_radius` that accepts motl paths
    or objects instead of raw coordinate arrays.

    Parameters
    ----------
    motl_a : str or Motl
        Query motl.
    motl_nn : str or Motl
        Neighbor motl.
    radius : float
        Search radius in voxels.
    remove_qp : bool, default=False
        Drop self-matches.

    Returns
    -------
    qp_idx : list of int
    nn_idx : list of numpy.ndarray
    """
    motl_a = cryomotl.Motl.load(motl_a)
    motl_nn = cryomotl.Motl.load(motl_nn)
    return find_nn_within_radius(motl_a.get_coordinates(), motl_nn.get_coordinates(),
                                 radius=radius, remove_qp=remove_qp)


def get_nn_within_distance(
    feature_motl: MotlSource,
    radius: float,
    unique_only: bool = True,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Self-NN within a radius for a single motl.

    Thin wrapper around :func:`find_nn_within_self`.

    Parameters
    ----------
    feature_motl : str or Motl
        Particle list.
    radius : float
        Search radius in voxels.
    unique_only : bool, default=True
        Deduplicate symmetric pairs so ``(i, j)`` is reported once.

    Returns
    -------
    center_idx : numpy.ndarray
    nn_idx : list of numpy.ndarray
    """
    feature_motl = cryomotl.Motl.load(feature_motl)
    return find_nn_within_self(feature_motl.get_coordinates(), radius, unique_only=unique_only)


def get_nn_within_radius(
    motl_a: MotlSource,
    motl_nn: MotlSource,
    nn_radius: float,
    pixel_size: float = 1.0,
    column_name: MotlColumn = "tomo_id",
) -> np.ndarray:
    """Per-particle count of neighbors within *nn_radius*, grouped by column_name.

    Parameters
    ----------
    motl_a : str or Motl
        Query motl.
    motl_nn : str or Motl
        Neighbor motl.
    nn_radius : float
        Search radius in physical units (voxels × *pixel_size*).
    pixel_size : float, default=1.0
        Scale factor applied to coordinates before the radius search.
    column_name : str, default='tomo_id'
        Column used to split particles into groups before searching.

    Returns
    -------
    numpy.ndarray
        Shape ``(N,)`` — number of neighbors within *nn_radius* for each
        query particle, ordered by column_name then by row position in the motl.
    """
    motl_a = cryomotl.Motl.load(motl_a)
    motl_nn = cryomotl.Motl.load(motl_nn)

    features_a = np.unique(motl_a.df.loc[:, column_name].values)
    features_nn = np.unique(motl_nn.df.loc[:, column_name].values)
    features = np.intersect1d(features_a, features_nn, assume_unique=True)

    counts = []
    for f in features:
        fm_a = motl_a.get_motl_subset(f, column_name=column_name)
        fm_nn = motl_nn.get_motl_subset(f, column_name=column_name)
        coord_a = fm_a.get_coordinates() * pixel_size
        coord_nn = fm_nn.get_coordinates() * pixel_size
        kdt = sn.KDTree(coord_nn)
        counts.append(kdt.query_radius(coord_a, r=nn_radius, count_only=True))

    return np.concatenate(counts, axis=0) if counts else np.array([])


def get_nn_stats(
    motl_a: MotlSource,
    motl_nn: MotlSource,
    pixel_size: float = 1.0,
    column_name: MotlColumn = "tomo_id",
    nn_number: int = 1,
    rotation_type: RotationDistanceType = "angular_distance",
    paired: bool = False,
    remove_duplicates: bool = False,
) -> pd.DataFrame:
    """Return a per-pair statistics DataFrame for two motls.

    Parameters
    ----------
    motl_a : str or Motl
        Query motl.
    motl_nn : str or Motl
        Neighbor motl.
    pixel_size : float, default=1.0
        Pixel size in Å; applied to distances and coordinates.
    column_name : str, default='tomo_id'
        Column used to group particles before searching.
    nn_number : int, default=1
        Number of nearest neighbors per query particle.
    rotation_type : str, default='angular_distance'
        Angular metric: ``"angular_distance"``, ``"cone_distance"``, or
        ``"in_plane_distance"``.
    paired : bool, default=False
        When ``True``, angles are taken from *motl_a* for both sides
        (entry/exit pair convention).
    remove_duplicates : bool, default=False
        Drop symmetric ``(a, b)`` / ``(b, a)`` pairs.

    Returns
    -------
    pandas.DataFrame
        See :meth:`NearestNeighbors.to_stats_dataframe` for column details.
    """
    nn = NearestNeighbors(
        input_data=[motl_a, motl_nn],
        column_name=column_name,
        nn_type="closest_dist",
        type_param=nn_number,
        paired=paired,
        remove_duplicates=remove_duplicates,
    )
    return nn.to_stats_dataframe(pixel_size=pixel_size, rotation_type=rotation_type)


def get_nn_distances(
    motl_a: MotlSource,
    motl_nn: MotlSource,
    pixel_size: float = 1.0,
    nn_number: int = 1,
    column_name: MotlColumn = "tomo_id",
    rotation_type: RotationDistanceType = "angular_distance",
    paired: bool = False,
    remove_duplicates: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-pair geometry as a flat tuple (backward-compatible).

    Parameters
    ----------
    motl_a : str or Motl
        Query motl.
    motl_nn : str or Motl
        Neighbor motl.
    pixel_size : float, default=1.0
        Pixel size in Å.
    nn_number : int, default=1
        Number of nearest neighbors per query particle.
    column_name : str, default='tomo_id'
        Column used to group particles before searching.
    rotation_type : str, default='angular_distance'
        Angular metric for the returned angular-distance array.
    paired : bool, default=False
        When ``True``, angles are taken from *motl_a* for both sides.
    remove_duplicates : bool, default=False
        Drop symmetric pairs.

    Returns
    -------
    centered : numpy.ndarray, shape ``(N, 3)``
        ``nn - qp`` displacement in physical units.
    rotated : numpy.ndarray, shape ``(N, 3)``
        Displacement rotated into the qp local frame.
    nn_dist : numpy.ndarray, shape ``(N,)``
        Euclidean distances in physical units.
    ang : numpy.ndarray, shape ``(N,)``
        Angular distances in degrees.
    qp_subtomo_id : numpy.ndarray, shape ``(N,)``
    nn_subtomo_id : numpy.ndarray, shape ``(N,)``
    """
    nn = NearestNeighbors(
        input_data=[motl_a, motl_nn],
        column_name=column_name,
        nn_type="closest_dist",
        type_param=nn_number,
        paired=paired,
        remove_duplicates=remove_duplicates,
    )
    centered = nn.get_normalized_coord(add_to_df=False) * pixel_size
    rotated = rotated_nn_coords(centered, nn.df[nn._QP_ANGLE_COLS].to_numpy())
    ang = angular_distances(
        nn.df[nn._QP_ANGLE_COLS].to_numpy(),
        nn.df[nn._NN_ANGLE_COLS].to_numpy(),
        rotation_type=rotation_type,
    )
    nn_dist = nn.df["nn_dist"].to_numpy() * pixel_size
    return (centered, rotated, nn_dist, ang,
            nn.df["qp_subtomo_id"].to_numpy(), nn.df["nn_subtomo_id"].to_numpy())


def get_nn_rotations(
    motl_a: MotlSource,
    motl_nn: MotlSource,
    nn_number: int = 1,
    column_name: MotlColumn = "tomo_id",
    paired: bool = False,
    remove_duplicates: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the qp→nn relative rotations as unit vectors and Euler angles.

    Parameters
    ----------
    motl_a : str or Motl
        Query motl.
    motl_nn : str or Motl
        Neighbor motl.
    nn_number : int, default=1
        Number of nearest neighbors per query particle.
    column_name : str, default='tomo_id'
        Column used to group particles before searching.
    paired : bool, default=False
        When ``True``, angles are taken from *motl_a* for both sides.
    remove_duplicates : bool, default=False
        Drop symmetric pairs.

    Returns
    -------
    points_on_sphere : numpy.ndarray, shape ``(N, 3)``
    euler_angles : numpy.ndarray, shape ``(N, 3)``
        zxz Euler angles in degrees.
    """
    nn = NearestNeighbors(
        input_data=[motl_a, motl_nn],
        column_name=column_name,
        nn_type="closest_dist",
        type_param=nn_number,
        paired=paired,
        remove_duplicates=remove_duplicates,
    )
    return rotations_to_unit_vectors(nn.get_relative_rotations())


def get_nn_stats_within_radius(
    input_motl: MotlSource,
    nn_radius: float,
    column_name: MotlColumn = "tomo_id",
    index_by_feature: bool = True,
) -> pd.DataFrame:
    """Build a per-pair stats DataFrame for all self-NN pairs within a radius.

    Unlike :func:`get_nn_stats`, this function uses the same motl for both
    query particles and neighbors (self-NN) and collects *all* neighbors
    within ``nn_radius`` rather than a fixed number.

    Parameters
    ----------
    input_motl : str or Motl
        Input motl; loaded with :meth:`~cryocat.core.cryomotl.Motl.load` if a
        path string is given.
    nn_radius : float
        Search radius in voxels.
    column_name : str, default='tomo_id'
        Column used to partition particles before searching.
    index_by_feature : bool, default=True
        When ``True``, row indices in the returned DataFrame refer to the
        per-column_name subset; when ``False``, they refer to the global motl index.

    Returns
    -------
    pandas.DataFrame
        One row per (query-particle, neighbor) pair with columns:

        ``qp_subtomo_id``, ``nn_subtomo_id``
            ``subtomo_id`` values of the query particle and its neighbor.
        ``coord_x``, ``coord_y``, ``coord_z``
            Centered displacement vector (neighbor − query) in voxels.
        ``coord_rx``, ``coord_ry``, ``coord_rz``
            Displacement rotated into the query-particle reference frame.
        ``angular_distance``, ``cone_distance``, ``inplane_distance``
            Angular distances in degrees.
        ``rot_x``, ``rot_y``, ``rot_z``
            Unit-vector representation of the relative rotation.
        ``phi``, ``theta``, ``psi``
            zxz Euler angles of the relative rotation in degrees.
        ``qp_motl_id``, ``nn_motl_idx``
            Row indices (column_name-local or global) of the two particles.
    """
    input_motl = cryomotl.Motl.load(input_motl)
    features = np.unique(input_motl.df.loc[:, column_name].values)

    rows = []
    for f in features:
        fm = input_motl.get_motl_subset(f, column_name=column_name)
        coord = fm.get_coordinates()
        center_idx, nn_idx_list = find_nn_within_self(coord, nn_radius, unique_only=False)

        if len(center_idx) == 0:
            continue

        angles_all = fm.get_angles()
        subtomos = fm.df["subtomo_id"].to_numpy()
        motl_idx = (fm.df.index.to_numpy() if index_by_feature
                    else input_motl.df.index[input_motl.df[column_name] == f].to_numpy())

        for i, c in enumerate(center_idx):
            for n in nn_idx_list[i]:
                qp_ang = angles_all[c:c + 1]
                nn_ang = angles_all[n:n + 1]
                centered = (coord[n] - coord[c]).reshape(1, 3)
                rotated = rotated_nn_coords(centered, qp_ang)
                ang_dist, cone_dist, inplane_dist = angular_distances(qp_ang, nn_ang, rotation_type="all")
                rel = relative_rotations(qp_ang, nn_ang)
                pts, eul = rotations_to_unit_vectors(rel)

                rows.append({
                    "qp_subtomo_id":    subtomos[c],
                    "nn_subtomo_id":    subtomos[n],
                    "coord_x":          centered[0, 0],
                    "coord_y":          centered[0, 1],
                    "coord_z":          centered[0, 2],
                    "coord_rx":         rotated[0, 0],
                    "coord_ry":         rotated[0, 1],
                    "coord_rz":         rotated[0, 2],
                    "angular_distance": np.atleast_1d(ang_dist)[0],
                    "cone_distance":    np.atleast_1d(cone_dist)[0],
                    "inplane_distance": np.atleast_1d(inplane_dist)[0],
                    "rot_x":            pts[0, 0],
                    "rot_y":            pts[0, 1],
                    "rot_z":            pts[0, 2],
                    "phi":              eul[0, 0],
                    "theta":            eul[0, 1],
                    "psi":              eul[0, 2],
                    "qp_motl_id":       motl_idx[c],
                    "nn_motl_idx":      motl_idx[n],
                })

    return pd.DataFrame(rows)


def filter_nn_radial_stats(
    input_stats: pd.DataFrame,
    binary_mask: MapSource,
) -> pd.DataFrame:
    """Keep only rows whose rotated coordinate falls inside a binary mask.

    Pairs whose ``(coord_rx, coord_ry, coord_rz)`` maps to a voxel outside
    the mask or outside the mask array bounds are dropped.

    Parameters
    ----------
    input_stats : pandas.DataFrame
        Output of :func:`get_nn_stats_within_radius`; must contain columns
        ``coord_rx``, ``coord_ry``, ``coord_rz``.
    binary_mask : str or numpy.ndarray
        3-D binary volume.  Values ≥ 0.5 are treated as *inside*.  If a path
        string is given the file is loaded with :func:`~cryocat.core.cryomap.read`.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``input_stats`` (index reset), with the temporary
        integer-coordinate columns removed.
    """
    if isinstance(binary_mask, np.ndarray):
        boolean_mask = binary_mask
    else:
        boolean_mask = cryomap.read(binary_mask)
    boolean_mask = np.where(boolean_mask < 0.5, False, True)
    dx, dy, dz = np.asarray(boolean_mask.shape) // 2

    nn_stats = input_stats.copy()
    nn_stats["x_int"] = (nn_stats["coord_rx"] + dx).astype(int)
    nn_stats["y_int"] = (nn_stats["coord_ry"] + dy).astype(int)
    nn_stats["z_int"] = (nn_stats["coord_rz"] + dz).astype(int)

    in_bounds = (
        (nn_stats["x_int"] >= 0) & (nn_stats["x_int"] < 2 * dx)
        & (nn_stats["y_int"] >= 0) & (nn_stats["y_int"] < 2 * dy)
        & (nn_stats["z_int"] >= 0) & (nn_stats["z_int"] < 2 * dz)
    )
    nn_stats = nn_stats[in_bounds]
    mask_values = boolean_mask[nn_stats["x_int"], nn_stats["y_int"], nn_stats["z_int"]]
    return nn_stats[mask_values].drop(columns=["x_int", "y_int", "z_int"]).reset_index(drop=True)


# =============================================================================
# Convenience wrapper for class-assignment by NN
# =============================================================================


def assign_class_by_nn(
    motl_unassigned: MotlSource,
    motl_list: list[MotlSource],
    starting_class: int = 1,
    dist_threshold: float = 20,
    output_motl: PathOrStr | None = None,
    unassigned_class: int = 0,
    update_coord: bool = False,
) -> "cryomotl.Motl":
    """Assign each particle in ``motl_unassigned`` the class of its nearest neighbor.

    For every motl in ``motl_list`` the nearest particle in ``motl_unassigned``
    (within ``dist_threshold`` voxels) is found and labeled with the
    corresponding class index.  Particles with no neighbor within the threshold
    remain labeled ``unassigned_class``.

    Parameters
    ----------
    motl_unassigned : str or Motl
        Motl whose particles are to be classified.
    motl_list : list of str or Motl
        Ordered list of motls, one per class.  The first motl gets class
        ``starting_class``, the second gets ``starting_class + 1``, and so on.
    starting_class : int, default=1
        Class label assigned to particles matched by ``motl_list[0]``.
    dist_threshold : float, default=20
        Maximum distance (voxels) for a match to be accepted.
    output_motl : str, optional
        If given, the result is saved to this path.
    unassigned_class : int, default=0
        Class label for particles that are not matched by any motl.
    update_coord : bool, default=False
        When ``True``, overwrite coordinates and orientations of matched
        particles with the values from the matching classified motl.

    Returns
    -------
    Motl
        Copy of ``motl_unassigned`` with updated ``class`` column (and
        optionally updated coordinates).  Overlap counts are printed to
        stdout.
    """
    motl = cryomotl.Motl.load(motl_unassigned)
    motl.df["class"] = unassigned_class
    classified, overlaps, cl = 0, 0, starting_class
    _tomo_pos = dict(motl.df.groupby("tomo_id").indices)

    for m in motl_list:
        cm = cryomotl.Motl.load(m)
        classified += cm.df.shape[0]
        for t in np.unique(cm.df.loc[:, "tomo_id"].values):
            tomo_rows = _tomo_pos[t]
            tm_coord = cm.get_coordinates(t)
            all_coord = motl.get_coordinates(t)
            tm = cm.get_motl_subset(t, return_df=True, reset_index=False)
            tm_all = motl.df.iloc[tomo_rows].copy()
            tm_idx = np.arange(tm.shape[0])

            kdt = sn.KDTree(all_coord)
            dist, idx = kdt.query(tm_coord, k=1)
            dist = np.atleast_1d(dist).ravel()
            idx = np.atleast_1d(idx).ravel()

            keep = dist < dist_threshold
            idx = idx[keep]
            tm_idx = tm_idx[keep]

            unique_idx, counts = np.unique(idx, return_counts=True)
            duplicates = unique_idx[counts > 1]
            if duplicates.size > 0:
                identical = np.concatenate([np.where(idx == d) for d in duplicates]).flatten()
                subtomo_idx = tm.loc[tm.index[identical], ["subtomo_id"]].values.flatten()
                print(f"Following particles in motl {m} are identical: {subtomo_idx}")
                overlaps += np.sum(counts) - counts.size

            tm_all.loc[tm_all.index[idx], ["geom1"]] += 1
            tm_all.loc[tm_all["geom1"] > 1, ["geom2"]] = tm_all.loc[tm_all["geom1"] > 1, ["class"]]
            tm_all.loc[tm_all.index[idx], ["class"]] = cl

            if update_coord:
                tm_all.loc[tm_all.index[idx], ["phi", "psi", "theta"]] = (
                    tm.loc[tm.index[tm_idx], ["phi", "psi", "theta"]].values
                )
                tm_all.loc[tm_all.index[idx], ["geom3", "geom4", "geom5"]] = (
                    tm.loc[tm.index[tm_idx], ["x", "y", "z"]].values
                    + tm.loc[tm.index[tm_idx], ["shift_x", "shift_y", "shift_z"]].values
                )
            motl.df.iloc[tomo_rows] = tm_all.values
        cl += 1

    assigned = motl.df.loc[motl.df["geom1"] > 0].shape[0]
    print(f"Particles in classified motls: {classified}, "
          f"number of assigned particles: {assigned}, number of overlaps: {overlaps}")

    if update_coord:
        motl.df.loc[motl.df["class"] != unassigned_class, ["x", "y", "z"]] = (
            motl.df.loc[motl.df["class"] != unassigned_class, ["geom3", "geom4", "geom5"]].values
        )
        motl.df.loc[motl.df["class"] != unassigned_class, ["shift_x", "shift_y", "shift_z"]] = 0.0
        motl.df["geom3"] = 0.0

    motl.df["geom4"] = motl.df["geom2"].values
    motl.df["geom5"] = motl.df["geom1"].values
    motl.df[["geom1", "geom2"]] = 0.0
    motl.update_coordinates()

    if output_motl is not None:
        motl.write_to_emfile(output_motl)
    return motl


# =============================================================================
# Standalone chain tracing
# =============================================================================


def _get_nn_dist(
    kdt: sn.KDTree,
    query_point: np.ndarray,
    dist_max: float,
    dist_min: float,
    active_points: np.ndarray,
    test_value: bool,
) -> tuple[int, float | list]:
    """Return the nearest active point within ``[dist_min, dist_max]``.

    Queries *kdt* within *dist_max*, then filters by *active_points* and
    *dist_min*, returning the closest point that passes both criteria.

    Parameters
    ----------
    kdt : sklearn.neighbors.KDTree
        KD-tree built on the candidate coordinate set.
    query_point : numpy.ndarray, shape ``(1, 3)``
        The coordinate to search from.
    dist_max : float
        Maximum search radius.
    dist_min : float
        Minimum distance threshold; results closer than this are excluded.
    active_points : numpy.ndarray of bool, shape ``(N,)``
        Mask of which points are still available (not yet used in a chain).
    test_value : bool
        Expected value of *active_points* at candidate indices; only points
        where ``active_points[idx] == test_value`` are considered.

    Returns
    -------
    idx : int
        Index of the nearest valid neighbor, or ``-1`` if none was found.
    dist : float or list
        Distance to the neighbor, or ``[]`` if none was found.
    """
    id_max, dist = kdt.query_radius(query_point, dist_max,
                                    return_distance=True, sort_results=True)
    id_max = id_max[0]
    dist = dist[0]
    if id_max.size == 0:
        return -1, []

    rp_idx = id_max[active_points[id_max] == test_value]
    rp_dist = dist[active_points[id_max] == test_value]

    if rp_idx.size == 0:
        return -1, []
    if dist_min > 0:
        rp_idx = rp_idx[rp_dist > dist_min]
        rp_dist = rp_dist[rp_dist > dist_min]

    if rp_idx.size == 0:
        return -1, []
    return rp_idx[0], rp_dist[0]


def _add_chain_suffix(
    ch_cls: int,
    ch_orders: list,
    nfm_idx1: np.ndarray,
    nfm_idx2: np.ndarray,
    nfm_dist: np.ndarray,
    pos_of_sub: dict,
    cls_to_rows: dict,
    exit_subtomos: np.ndarray,
    subtomo_id: int,
    current_dist: float,
) -> int | None:
    """Append the current chain after an existing one.

    Looks up the particle at *subtomo_id* in the nfm arrays, finds the chain
    it belongs to, and — if the link is the last in that chain and the new
    distance is shorter — re-assigns the tail of the old chain to the current
    chain or vice versa.  All lookups are O(1) via *pos_of_sub* / *cls_to_rows*.

    Parameters
    ----------
    ch_cls : int
        Current chain-class id of the chain being built.
    ch_orders : list of int
        Within-chain position values for each row in the chain (modified in place).
    nfm_idx1 : numpy.ndarray
        store_idx1 (class) column for committed nfm rows (modified in place).
    nfm_idx2 : numpy.ndarray
        store_idx2 (order) column for committed nfm rows (modified in place).
    nfm_dist : numpy.ndarray
        store_dist column for committed nfm rows (modified in place).
    pos_of_sub : dict
        subtomo_id → row position in the nfm arrays (O(1) lookup).
    cls_to_rows : dict
        class_id → list of row positions (modified in place).
    exit_subtomos : numpy.ndarray
        ``subtomo_id`` values for the exit motl, indexed by row position.
    subtomo_id : int
        Row index (within exit motl) of the particle whose chain we append to.
    current_dist : float
        Distance of the new link being considered.

    Returns
    -------
    int or None
        New ``ch_cls`` on success; ``None`` if the existing link was shorter
        and the append was aborted.
    """
    particle_id = int(exit_subtomos[subtomo_id])
    row_pos = pos_of_sub[particle_id]
    temp_cl_id = int(nfm_idx1[row_pos])
    order_id = int(nfm_idx2[row_pos])
    previous_dist = float(nfm_dist[row_pos])
    chain_rows = cls_to_rows[temp_cl_id]
    chain_max_order = int(nfm_idx2[chain_rows].max())

    if chain_max_order != order_id:
        if previous_dist <= current_dist:
            return None
        head = [r for r in chain_rows if nfm_idx2[r] <= order_id]
        tail = [r for r in chain_rows if nfm_idx2[r] > order_id]
        for r in tail:
            nfm_idx1[r] = ch_cls
        cls_to_rows[temp_cl_id] = head
        cls_to_rows.setdefault(ch_cls, []).extend(tail)
        for new_pos, r in enumerate(cls_to_rows[ch_cls]):
            nfm_idx2[r] = new_pos + 1
        chain_max_order = int(nfm_idx2[head].max()) if head else 0

    nfm_dist[row_pos] = current_dist
    for k_i in range(len(ch_orders)):
        ch_orders[k_i] += chain_max_order
    return temp_cl_id


def _add_chain_prefix(
    ch_cls: int,
    ch_orders: list,
    ch_dists: list,
    nfm_idx1: np.ndarray,
    nfm_idx2: np.ndarray,
    nfm_dist: np.ndarray,
    pos_of_sub: dict,
    cls_to_rows: dict,
    entry_subtomos: np.ndarray,
    subtomo_id: int,
    current_dist: float,
    class_max: tuple[int, int] | None = None,
) -> int | None:
    """Prepend the current chain before an existing one.

    Looks up the particle at *subtomo_id* in the nfm arrays and inserts the
    current chain fragment at the beginning of that particle's chain, shifting
    existing within-chain positions accordingly.  All lookups are O(1).

    Parameters
    ----------
    ch_cls : int
        Current chain-class id of the chain being built.
    ch_orders : list of int
        Within-chain position values for each row in the chain.
    ch_dists : list of float
        Distance values for each row in the chain (last element modified in place).
    nfm_idx1 : numpy.ndarray
        store_idx1 (class) column for committed nfm rows (modified in place).
    nfm_idx2 : numpy.ndarray
        store_idx2 (order) column for committed nfm rows (modified in place).
    nfm_dist : numpy.ndarray
        store_dist column for committed nfm rows (modified in place).
    pos_of_sub : dict
        subtomo_id → row position in the nfm arrays.
    cls_to_rows : dict
        class_id → list of row positions (modified in place).
    entry_subtomos : numpy.ndarray
        ``subtomo_id`` values for the entry motl, indexed by row position.
    subtomo_id : int
        Row index (within entry motl) of the particle at the start of the
        existing chain.
    current_dist : float
        Distance of the new link being considered.
    class_max : tuple of int or None, default=None
        When not ``None``, a ``(max_order, fallback_class)`` pair used when
        both a suffix and prefix merge happen simultaneously.

    Returns
    -------
    int or None
        New ``ch_cls`` on success; ``None`` if the existing link was shorter
        and the operation was aborted.
    """
    particle_id = int(entry_subtomos[subtomo_id])
    row_pos_nm = pos_of_sub[particle_id]
    class_to_change = int(nfm_idx1[row_pos_nm])
    order_id = int(nfm_idx2[row_pos_nm])
    cut_off_size = 0

    if order_id != 1:
        ctc_rows = cls_to_rows[class_to_change]
        prev_row = next(r for r in ctc_rows if nfm_idx2[r] == order_id - 1)
        if nfm_dist[prev_row] <= current_dist:
            return None
        cut_rows = [r for r in ctc_rows if nfm_idx2[r] < order_id]
        cut_off_size = len(cut_rows)
        sentinel = ch_cls if class_max is None else -1
        for r in cut_rows:
            nfm_idx1[r] = sentinel
        cls_to_rows.setdefault(sentinel, []).extend(cut_rows)
        cls_to_rows[class_to_change] = [r for r in ctc_rows if nfm_idx2[r] >= order_id]

    ctc_remain = cls_to_rows[class_to_change]

    if class_max is None:
        new_ch_cls = class_to_change
        class_max_val = max(ch_orders) if ch_orders else 0
        offset = class_max_val - cut_off_size
        for r in ctc_remain:
            nfm_idx2[r] += offset
    else:
        temp_cl_id = ch_cls
        offset = class_max[0] - cut_off_size
        for r in ctc_remain:
            nfm_idx2[r] += offset
            nfm_idx1[r] = temp_cl_id
        if temp_cl_id != class_to_change:
            cls_to_rows.setdefault(temp_cl_id, []).extend(ctc_remain)
            del cls_to_rows[class_to_change]
        if order_id != 1:
            neg_rows = cls_to_rows.pop(-1, [])
            for r in neg_rows:
                nfm_idx1[r] = class_max[1]
            cls_to_rows.setdefault(class_max[1], []).extend(neg_rows)
        new_ch_cls = ch_cls

    ch_dists[-1] = current_dist
    return new_ch_cls


def trace_chains(
    motl_entry: MotlSource,
    motl_exit: MotlSource | None = None,
    max_distance: float | None = None,
    min_distance: float = 0,
    column_name: MotlColumn = "tomo_id",
    output_motl: PathOrStr | None = None,
    store_idx1: str = "object_id",
    store_idx2: str = "geom2",
    store_dist: str = "geom4",
) -> "cryomotl.Motl":
    """Build chains by linking the exit of particle A to the entry of particle B.

    Iterates over particles sorted by their exit positions and greedily links
    each to the closest unvisited entry particle within ``[min_distance,
    max_distance]``.  Chain stitching (suffix- and prefix-merging) handles
    cases where a new chain can extend or prepend an existing one.

    Parameters
    ----------
    motl_entry : str or Motl
        Motl representing particle *entry* points.
    motl_exit : str or Motl, optional
        Motl representing particle *exit* points.  When ``None``, ``motl_entry``
        is used for both sides (single-motl / symmetric mode).
    max_distance : float
        Maximum allowed link distance in voxels.  **Required.**
    min_distance : float, default=0
        Minimum allowed link distance in voxels.
    column_name : str, default='tomo_id'
        Column used to partition the motl before tracing (usually ``'tomo_id'``).
    output_motl : str, optional
        If given, the resulting motl is written to this path.
    store_idx1 : str, default='object_id'
        Column in which the chain identifier is stored.
    store_idx2 : str, default='geom2'
        Column in which the within-chain position (1-based) is stored.
    store_dist : str, default='geom4'
        Column in which the distance to the next particle is stored.

    Returns
    -------
    Motl
        A copy of ``motl_entry`` with ``store_idx1``, ``store_idx2``, and
        ``store_dist`` populated according to the traced chains.

    Raises
    ------
    ValueError
        If ``max_distance`` is ``None`` or if the two motls have different
        column_name sets.
    """
    if max_distance is None:
        raise ValueError("max_distance must be specified")

    motl_entry = cryomotl.Motl.load(motl_entry)
    motl_exit = motl_entry if motl_exit is None else cryomotl.Motl.load(motl_exit)

    features1 = np.unique(motl_entry.df.loc[:, column_name])
    features2 = np.unique(motl_exit.df.loc[:, column_name])
    if not np.array_equal(features1, features2):
        raise ValueError("Provided motls have different features sets!")

    traced_motl = cryomotl.Motl.create_empty_motl_df()

    for f in features1:
        fm_entry = motl_entry.get_motl_subset(f, column_name, reset_index=False)
        fm_exit  = motl_exit.get_motl_subset(f, column_name, reset_index=False)

        fm_size = fm_entry.df.shape[0]
        remain_entry = np.full(fm_size, True)
        remain_exit  = np.full(fm_size, True)
        class_c = 1

        coord_entry = fm_entry.get_coordinates()
        coord_exit  = fm_exit.get_coordinates()
        kdt_entry = sn.KDTree(coord_entry)
        kdt_exit  = sn.KDTree(coord_exit)

        entry_subtomos = fm_entry.df["subtomo_id"].values
        exit_subtomos  = fm_exit.df["subtomo_id"].values

        # Pre-allocate nfm working arrays (filled row-by-row as chains complete).
        _nfm_rows = np.empty(fm_size, dtype=int)   # positional index into fm_entry.df
        _nfm_idx1 = np.zeros(fm_size, dtype=int)   # store_idx1 (chain class)
        _nfm_idx2 = np.zeros(fm_size, dtype=int)   # store_idx2 (within-chain order)
        _nfm_dist = np.zeros(fm_size, dtype=float)  # store_dist
        _nfm_filled = 0
        _pos_of_sub  = {}   # subtomo_id → nfm row position (O(1) lookup)
        _cls_to_rows = {}   # class_id   → list of nfm row positions

        for i, current_point in enumerate(coord_exit):
            if not remain_exit[i]:
                continue

            ch_row_idxs = []   # positional indices into fm_entry.df
            ch_orders   = []   # store_idx2 values (1-based within-chain positions)
            ch_dists    = []   # store_dist values (0.0 until a next link is found)
            chain_id    = 1
            trace_chain = True
            p_idx       = i
            used_idx    = []

            while trace_chain:
                ch_row_idxs.append(p_idx)
                ch_orders.append(chain_id)
                ch_dists.append(0.0)
                chain_id += 1

                remain_entry[p_idx] = False
                remain_exit[p_idx]  = False
                used_idx.append(p_idx)

                p_coord = coord_exit[p_idx, None, :]

                if np.all(remain_entry == False):
                    np_idx = -1
                else:
                    np_idx, np_dist = _get_nn_dist(
                        kdt_entry, p_coord, max_distance, min_distance, remain_entry, True
                    )

                if np_idx != -1:
                    p_idx = np_idx
                    ch_dists[-1] = np_dist
                else:
                    ch_cls  = class_c
                    class_c += 1

                    if _nfm_filled > 0:
                        first_pos = ch_row_idxs[0]
                        first_row = fm_entry.df.iloc[first_pos]
                        first_coord = (
                            first_row[["x", "y", "z"]].values
                            + first_row[["shift_x", "shift_y", "shift_z"]].values
                        ).reshape(1, 3)
                        remain_entry[used_idx] = True
                        remain_exit[used_idx]  = True
                        nm_idx, nm_dist = _get_nn_dist(
                            kdt_entry, p_coord, max_distance, min_distance, remain_entry, False
                        )
                        first_idx, first_dist = _get_nn_dist(
                            kdt_exit, first_coord, max_distance, min_distance, remain_exit, False
                        )
                        remain_entry[used_idx] = False
                        remain_exit[used_idx]  = False

                        if first_idx == nm_idx and first_idx != -1 and len(ch_row_idxs) == 1:
                            if first_dist <= nm_dist:
                                nm_idx = -1
                            else:
                                first_idx = -1
                        elif first_idx != -1 and nm_idx != -1:
                            part1 = int(exit_subtomos[first_idx])
                            part2 = int(entry_subtomos[nm_idx])
                            cl1 = int(_nfm_idx1[_pos_of_sub[part1]])
                            cl2 = int(_nfm_idx1[_pos_of_sub[part2]])
                            if cl1 == cl2:
                                if first_dist <= nm_dist:
                                    nm_idx = -1
                                else:
                                    first_idx = -1

                        ch_changed = False
                        if first_idx != -1:
                            result = _add_chain_suffix(
                                ch_cls, ch_orders,
                                _nfm_idx1, _nfm_idx2, _nfm_dist,
                                _pos_of_sub, _cls_to_rows,
                                exit_subtomos, first_idx, first_dist,
                            )
                            if result is not None:
                                ch_cls     = result
                                ch_changed = True
                        if nm_idx != -1:
                            class_max = None
                            if ch_changed:
                                current_class = class_c - 1
                                cl_max = max(ch_orders) if ch_orders else 0
                                if cl_max > 1:
                                    class_max = (cl_max, current_class)
                            result = _add_chain_prefix(
                                ch_cls, ch_orders, ch_dists,
                                _nfm_idx1, _nfm_idx2, _nfm_dist,
                                _pos_of_sub, _cls_to_rows,
                                entry_subtomos, nm_idx, nm_dist,
                                class_max=class_max,
                            )
                            if result is not None:
                                ch_cls = result

                    # Commit this chain's rows to the nfm working arrays.
                    start = _nfm_filled
                    for k_i, p in enumerate(ch_row_idxs):
                        sub_id = int(entry_subtomos[p])
                        _nfm_rows[_nfm_filled] = p
                        _nfm_idx1[_nfm_filled] = ch_cls
                        _nfm_idx2[_nfm_filled] = ch_orders[k_i]
                        _nfm_dist[_nfm_filled] = ch_dists[k_i]
                        _pos_of_sub[sub_id]    = _nfm_filled
                        _nfm_filled += 1
                    _cls_to_rows.setdefault(ch_cls, []).extend(range(start, start + len(ch_row_idxs)))

                    trace_chain = False

        # Reconstruct the feature's nfm DataFrame from the pre-allocated arrays.
        if _nfm_filled > 0:
            nfm_df = fm_entry.df.iloc[_nfm_rows[:_nfm_filled]].copy().reset_index(drop=True)
            nfm_df[store_idx1] = _nfm_idx1[:_nfm_filled]
            nfm_df[store_idx2] = _nfm_idx2[:_nfm_filled]
            nfm_df[store_dist] = _nfm_dist[:_nfm_filled]
            traced_motl = pd.concat([traced_motl, nfm_df])

    traced_motl = cryomotl.Motl(motl_df=traced_motl)
    if output_motl is not None:
        traced_motl.write_out(output_motl)
    return traced_motl


