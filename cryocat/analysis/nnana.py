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

import numpy as np
import pandas as pd
import sklearn.neighbors as sn
from scipy.spatial.transform import Rotation as srot

from cryocat.core import cryomotl
from cryocat.core import cryomap
from cryocat.utils import geom
from cryocat._types import ArrayLike


# =============================================================================
# Layer 1 — stateless array-level helpers
# =============================================================================


def find_nn_indices(coords_qp, coords_nn, k=1, remove_qp=False):
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

    query_k = k + 1 if remove_qp else k
    query_k = min(query_k, coords_nn.shape[0])

    kdt = sn.KDTree(coords_nn)
    nn_dist, nn_idx = kdt.query(coords_qp, k=query_k)

    nn_dist = np.atleast_2d(nn_dist)
    nn_idx = np.atleast_2d(nn_idx)
    qp_idx = np.arange(nn_idx.shape[0])

    if remove_qp:
        nn_dist = nn_dist[:, 1 : k + 1]
        nn_idx = nn_idx[:, 1 : k + 1]

    return qp_idx, nn_idx, nn_dist, nn_dist.shape[1]


def find_nn_within_radius(coords_qp, coords_nn, radius, remove_qp=False):
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
        if len(neighbors) > 0:
            qp_idx.append(i)
            nn_idx.append(np.sort(neighbors))
    return qp_idx, nn_idx


def find_nn_within_self(coords, radius, unique_only=True):
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


def centered_nn_coords(coords_qp, qp_idx, coords_nn, nn_idx, pixel_size=1.0):
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


def rotated_nn_coords(centered_coords, qp_angles_per_pair):
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


def angular_distances(qp_angles_per_pair, nn_angles_per_pair, rotation_type="angular_distance"):
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


def relative_rotations(qp_angles_per_pair, nn_angles_per_pair):
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


def rotations_to_unit_vectors(rotations):
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
        A single motl (or path) or a list of motls.  If a single motl is
        given, NN search is run on that motl against itself (with
        ``remove_qp`` forced to True).  If a list is given, the first
        element is the query and each subsequent element is searched
        against the query.
    feature_id : str, default='tomo_id'
    nn_type : str, {'closest_dist', 'radius'}, default='closest_dist'
    type_param : int or float, optional
    remove_qp : bool, optional
    remove_duplicates : bool, default=False
    paired : bool, default=False
        If True, angles are taken from ``motl_a`` only (entry/exit pairs).
    """

    _QP_COORD_COLS = ["qp_coord_x", "qp_coord_y", "qp_coord_z"]
    _NN_COORD_COLS = ["nn_coord_x", "nn_coord_y", "nn_coord_z"]
    _QP_ANGLE_COLS = ["qp_angles_phi", "qp_angles_theta", "qp_angles_psi"]
    _NN_ANGLE_COLS = ["nn_angles_phi", "nn_angles_theta", "nn_angles_psi"]
    _NORM_COORD_COLS = ["norm_nn_x", "norm_nn_y", "norm_nn_z"]
    _ROT_COORD_COLS = ["rot_nn_x", "rot_nn_y", "rot_nn_z"]

    def __init__(
        self,
        input_data=None,
        column_name="tomo_id",
        nn_type="closest_dist",
        type_param=None,
        remove_qp=None,
        remove_duplicates=False,
        paired=False,
    ):
        if input_data is None:
            self.features = None
            self.df = None
            self.column_name = column_name
            self.paired = paired
            return

        self.column_name = column_name
        self.paired = paired

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

        results = []
        for f in features:
            fm_qp = motl_list[0].get_motl_subset(column_values=f, column_name=column_name)
            qp_subtomos = fm_qp.df["subtomo_id"].values
            qp_coord = fm_qp.get_coordinates()
            qp_angles = fm_qp.get_angles()

            for motl_idx, m in enumerate(motl_list[1:], start=1):
                fm_nn = m.get_motl_subset(column_values=f, column_name=column_name)
                nn_subtomos = fm_nn.df["subtomo_id"].values
                nn_coord = fm_nn.get_coordinates()
                nn_angles = qp_angles if paired else fm_nn.get_angles()

                if nn_type == "closest_dist":
                    nn_count = type_param or 1
                    qp_idx, nn_idx, nn_dist, _ = find_nn_indices(
                        qp_coord, nn_coord, k=nn_count, remove_qp=single_motl or paired
                    )
                    stacked = self._stack_nn_results(
                        motl_idx, f, qp_idx, nn_idx,
                        qp_subtomos, nn_subtomos,
                        qp_angles, nn_angles,
                        qp_coord, nn_coord,
                        nn_dist=nn_dist,
                    )
                elif nn_type == "radius":
                    radius = type_param or 1
                    qp_idx, nn_idx_list = find_nn_within_radius(
                        qp_coord, nn_coord, radius=radius, remove_qp=single_motl or paired
                    )
                    stacked = self._stack_nn_results_radius(
                        motl_idx, f, qp_idx, nn_idx_list,
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

        self.df = (
            pd.DataFrame(columns=columns) if not results
            else pd.DataFrame(np.vstack(results), columns=columns)
        )
        if remove_duplicates:
            self.df = self.drop_symmetric_duplicates()
        self.features = features

    @staticmethod
    def _stack_nn_results(motl_idx, column_value, qp_idx, nn_idx,
                          qp_subtomos, nn_subtomos, qp_angles, nn_angles,
                          qp_coord, nn_coord, nn_dist):
        nn_idx = np.atleast_2d(nn_idx)
        nn_dist = np.atleast_2d(nn_dist)
        k = nn_idx.shape[1]
        if k == 0:
            return None
        # tile so all nn1 rows come before all nn2 rows (matches legacy ordering)
        qp_expanded = np.tile(qp_idx, k)
        nn_flat = nn_idx.T.reshape(-1)
        n_pairs = len(nn_flat)
        return np.column_stack([
            np.repeat(motl_idx, n_pairs),
            np.repeat(column_value, n_pairs),
            qp_expanded, qp_subtomos[qp_expanded],
            nn_flat, nn_subtomos[nn_flat],
            qp_angles[qp_expanded], qp_coord[qp_expanded],
            nn_angles[nn_flat], nn_coord[nn_flat],
            nn_dist.T.reshape(-1),
        ])

    @staticmethod
    def _stack_nn_results_radius(motl_idx, column_value, qp_idx, nn_idx_list,
                                 qp_subtomos, nn_subtomos, qp_angles, nn_angles,
                                 qp_coord, nn_coord):
        if not nn_idx_list:
            return None
        counts = np.array([len(n) for n in nn_idx_list])
        if counts.sum() == 0:
            return None
        qp_idx_arr = np.asarray(qp_idx)
        nn_flat = np.concatenate(nn_idx_list).astype(int)
        qp_expanded = np.repeat(qp_idx_arr, counts)
        n_pairs = counts.sum()
        return np.column_stack([
            np.repeat(motl_idx, n_pairs),
            np.repeat(column_value, n_pairs),
            qp_expanded, qp_subtomos[qp_expanded],
            nn_flat, nn_subtomos[nn_flat],
            qp_angles[qp_expanded], qp_coord[qp_expanded],
            nn_angles[nn_flat], nn_coord[nn_flat],
        ])

    def drop_symmetric_duplicates(self):
        """Return a copy of ``self.df`` with symmetric (a, b)/(b, a) pairs deduped."""
        pairs = self.df[["qp_subtomo_id", "nn_subtomo_id"]].apply(
            lambda row: tuple(sorted(row)), axis=1
        )
        df = self.df.copy()
        df["_pair_key"] = pairs
        return df.drop_duplicates(subset="_pair_key").drop(columns="_pair_key")

    def get_unique_values(self):
        """Return the feature values present in ``self.df``.

        Returns
        -------
        numpy.ndarray
            Unique feature values (e.g. tomogram IDs).
        """
        return self.features

    def get_nn_subset(self, motl_id_values, column_values):
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
        sub = NearestNeighbors()
        sub.feature_id = self.feature_id
        sub.paired = self.paired
        if not isinstance(motl_id_values, list):
            motl_id_values = [motl_id_values]
        if not isinstance(column_values, list):
            column_values = [column_values]
        sub.df = self.df[
            (self.df["motl_id"].isin(motl_id_values))
            & (self.df[self.feature_id].isin(column_values))
        ].copy()
        sub.features = column_values
        return sub

    def get_normalized_coord(self, add_to_df=True):
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

    def get_rotated_coord(self, add_to_df=True):
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

    def get_qp_rotations(self):
        """Return the query-particle rotations as a scipy ``Rotation`` object.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Length-``N`` Rotation parsed from the zxz Euler angles stored in
            ``self.df``.
        """
        return srot.from_euler("zxz", degrees=True, angles=self.df[self._QP_ANGLE_COLS].to_numpy())

    def get_nn_rotations(self):
        """Return the nearest-neighbor rotations as a scipy ``Rotation`` object.

        Returns
        -------
        scipy.spatial.transform.Rotation
            Length-``N`` Rotation parsed from the zxz Euler angles stored in
            ``self.df``.
        """
        return srot.from_euler("zxz", degrees=True, angles=self.df[self._NN_ANGLE_COLS].to_numpy())

    def get_relative_rotations(self):
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

    def get_angular_distances(self, rotation_type="all"):
        """Return per-pair angular distances between qp and nn orientations.

        Parameters
        ----------
        rotation_type : str, default='all'
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

    def to_stats_dataframe(self, pixel_size=1.0, rotation_type="angular_distance"):
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


def get_feature_nn_indices(motl_a, motl_nn, nn_number=1, remove_qp=False, column_name="tomo_id"):
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


def get_feature_nn_within_radius(motl_a, motl_nn, radius, remove_qp=False):
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


def get_nn_within_distance(feature_motl, radius, unique_only=True):
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


def get_nn_within_radius(motl_a, motl_nn, nn_radius, pixel_size=1.0, column_name="tomo_id"):
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


def get_nn_stats(motl_a, motl_nn, pixel_size=1.0, column_name="tomo_id",
                 nn_number=1, rotation_type="angular_distance",
                 paired=False, remove_duplicates=False):
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


def get_nn_distances(motl_a, motl_nn, pixel_size=1.0, nn_number=1,
                     column_name="tomo_id", rotation_type="angular_distance",
                     paired=False, remove_duplicates=False):
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


def get_nn_rotations(motl_a, motl_nn, nn_number=1, column_name="tomo_id",
                     paired=False, remove_duplicates=False):
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


def get_nn_stats_within_radius(input_motl, nn_radius, column_name="tomo_id",
                               index_by_feature=True):
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


def filter_nn_radial_stats(input_stats, binary_mask):
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


def assign_class_by_nn(motl_unassigned, motl_list, starting_class=1,
                       dist_threshold=20, output_motl=None,
                       unassigned_class=0, update_coord=False):
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

    for m in motl_list:
        cm = cryomotl.Motl.load(m)
        classified += cm.df.shape[0]
        for t in np.unique(cm.df.loc[:, "tomo_id"].values):
            tm_coord = cm.get_coordinates(t)
            all_coord = motl.get_coordinates(t)
            tm = cm.get_motl_subset(t, return_df=True, reset_index=False)
            tm_all = motl.get_motl_subset(t, return_df=True, reset_index=False)
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
            motl.df.loc[motl.df["tomo_id"] == t] = tm_all
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


def _get_nn_dist(kdt, query_point, dist_max, dist_min, active_points, test_value):
    """First active point within ``[dist_min, dist_max]``, sorted by distance."""
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


def _add_chain_suffix(chain_df, motl, traced_df, subtomo_id, current_dist,
                      store_idx1="object_id", store_idx2="geom2", store_dist="geom4"):
    """Append the current chain after an existing one."""
    particle_id = motl.df.loc[motl.df.index[subtomo_id], "subtomo_id"]
    temp_cl_id, order_id, previous_dist = traced_df.loc[
        traced_df["subtomo_id"] == particle_id,
        [store_idx1, store_idx2, store_dist],
    ].values[0]
    chain_max_order = np.max(traced_df.loc[traced_df[store_idx1] == temp_cl_id, [store_idx2]].values)

    if chain_max_order != order_id:
        if previous_dist <= current_dist:
            return False
        current_class = chain_df[store_idx1].values[0]
        traced_df.loc[
            (traced_df[store_idx1] == temp_cl_id) & (traced_df[store_idx2] > order_id),
            store_idx1,
        ] = current_class
        new_size = traced_df.loc[traced_df[store_idx1] == current_class, store_idx2].shape[0]
        traced_df.loc[traced_df[store_idx1] == current_class, store_idx2] = np.arange(1, new_size + 1)
        chain_max_order = np.max(traced_df.loc[traced_df[store_idx1] == temp_cl_id, [store_idx2]].values)

    traced_df.loc[traced_df["subtomo_id"] == particle_id, store_dist] = current_dist
    chain_df[store_idx1] = temp_cl_id
    chain_df[store_idx2] += chain_max_order
    return True


def _add_chain_prefix(chain_df, motl, traced_df, subtomo_id, current_dist,
                      store_idx1="object_id", store_idx2="geom2", store_dist="geom4",
                      class_max=None):
    """Prepend the current chain before an existing one."""
    particle_id = motl.df.loc[motl.df.index[subtomo_id], "subtomo_id"]
    class_to_change = traced_df.loc[traced_df["subtomo_id"] == particle_id, store_idx1].values[0]
    order_id = traced_df.loc[traced_df["subtomo_id"] == particle_id, store_idx2].values[0]
    current_class = chain_df[store_idx1].values[0]
    cut_off_size = 0

    if order_id != 1:
        previous_dist = traced_df.loc[
            (traced_df[store_idx1] == class_to_change)
            & (traced_df[store_idx2] == order_id - 1),
            store_dist,
        ].values[0]
        if previous_dist <= current_dist:
            return -1
        cut_off_size = traced_df.loc[
            (traced_df[store_idx1] == class_to_change) & (traced_df[store_idx2] < order_id)
        ].shape[0]
        traced_df.loc[
            (traced_df[store_idx1] == class_to_change) & (traced_df[store_idx2] < order_id),
            store_idx1,
        ] = (current_class if class_max is None else -1)

    if class_max is None:
        chain_df[store_idx1] = class_to_change
        class_max_val = np.max(chain_df[store_idx2].values)
        traced_df.loc[traced_df[store_idx1] == class_to_change, [store_idx2]] += class_max_val - cut_off_size
    else:
        temp_cl_id = chain_df[store_idx1][0]
        traced_df.loc[traced_df[store_idx1] == class_to_change, [store_idx2]] += class_max[0] - cut_off_size
        traced_df.loc[traced_df[store_idx1] == class_to_change, [store_idx1]] = temp_cl_id
        if order_id != 1:
            traced_df.loc[traced_df[store_idx1] == -1, [store_idx1]] = class_max[1]

    chain_df.loc[chain_df.index[-1], store_dist] = current_dist


def trace_chains(motl_entry, motl_exit=None, max_distance=None, min_distance=0,
                 column_name="tomo_id", output_motl=None,
                 store_idx1="object_id", store_idx2="geom2", store_dist="geom4"):
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
        fm_exit = motl_exit.get_motl_subset(f, column_name, reset_index=False)
        nfm_df = cryomotl.Motl.create_empty_motl_df()

        fm_size = fm_entry.df.shape[0]
        remain_entry = np.full((fm_size,), True)
        remain_exit = np.full((fm_size,), True)
        class_c = 1

        coord_entry = fm_entry.get_coordinates()
        coord_exit = fm_exit.get_coordinates()
        kdt_entry = sn.KDTree(coord_entry)
        kdt_exit = sn.KDTree(coord_exit)

        for i, current_point in enumerate(coord_exit):
            if not remain_exit[i]:
                continue

            ch_m = cryomotl.Motl.create_empty_motl_df()
            chain_id = 1
            trace_chain = True
            p_idx = i
            used_idx = []

            while trace_chain:
                ch_m = pd.concat([ch_m, fm_entry.df.iloc[[p_idx]]], ignore_index=True)
                ch_m.loc[ch_m.index[-1], [store_idx2]] = chain_id
                chain_id += 1

                remain_entry[p_idx] = False
                remain_exit[p_idx] = False
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
                    ch_m.loc[ch_m.index[-1], [store_dist]] = np_dist
                else:
                    ch_m.loc[:, store_idx1] = class_c
                    class_c += 1

                    if nfm_df.size != 0:
                        first_coord = (
                            ch_m.loc[ch_m.index[0], ["x", "y", "z"]].values
                            + ch_m.loc[ch_m.index[0], ["shift_x", "shift_y", "shift_z"]].values
                        ).reshape(1, 3)
                        remain_entry[used_idx] = True
                        remain_exit[used_idx] = True
                        nm_idx, nm_dist = _get_nn_dist(
                            kdt_entry, p_coord, max_distance, min_distance, remain_entry, False
                        )
                        first_idx, first_dist = _get_nn_dist(
                            kdt_exit, first_coord, max_distance, min_distance, remain_exit, False
                        )
                        remain_entry[used_idx] = False
                        remain_exit[used_idx] = False

                        if first_idx == nm_idx and first_idx != -1 and ch_m.shape[0] == 1:
                            if first_dist <= nm_dist:
                                nm_idx = -1
                            else:
                                first_idx = -1
                        elif first_idx != -1 and nm_idx != -1:
                            part1 = fm_exit.df.loc[fm_exit.df.index[first_idx], "subtomo_id"]
                            part2 = fm_entry.df.loc[fm_entry.df.index[nm_idx], "subtomo_id"]
                            cl1 = nfm_df.loc[nfm_df["subtomo_id"] == part1, store_idx1].values[0]
                            cl2 = nfm_df.loc[nfm_df["subtomo_id"] == part2, store_idx1].values[0]
                            if cl1 == cl2:
                                if first_dist <= nm_dist:
                                    nm_idx = -1
                                else:
                                    first_idx = -1

                        ch_changed = False
                        if first_idx != -1:
                            ch_changed = _add_chain_suffix(
                                ch_m, fm_exit, nfm_df, first_idx, first_dist,
                                store_idx1, store_idx2,
                            )
                        if nm_idx != -1:
                            class_max = None
                            if ch_changed:
                                current_class = class_c - 1
                                cl_max = np.max(ch_m[store_idx2].values)
                                if cl_max > 1:
                                    class_max = (cl_max, current_class)
                            _add_chain_prefix(
                                ch_m, fm_entry, nfm_df, nm_idx, nm_dist,
                                store_idx1, store_idx2,
                                class_max=class_max,
                            )

                    nfm_df = pd.concat([nfm_df, ch_m])
                    trace_chain = False

        traced_motl = pd.concat([traced_motl, nfm_df])

    traced_motl = cryomotl.Motl(motl_df=traced_motl)
    if output_motl is not None:
        traced_motl.write_out(output_motl)
    return traced_motl


