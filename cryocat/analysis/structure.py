from __future__ import annotations
import copy
import logging
import numpy as np
import pandas as pd
import warnings

# pandas ≥ 2.2 excludes grouping columns from apply() frames by default.
# Pass include_groups=False explicitly so the behaviour is stable across versions.
_PD_HAS_INCLUDE_GROUPS: bool = tuple(int(x) for x in pd.__version__.split(".")[:2]) >= (2, 2)
import decimal
import os
from scipy.spatial.transform import Rotation as srot
from scipy.optimize import linear_sum_assignment
from cryocat.core import cryomotl
from cryocat.core import cryomap
from cryocat.core import cryomask
from cryocat.utils import geom
from cryocat.utils import mathutils
from cryocat.analysis import nnana
from cryocat.analysis import clustering as _clustering
from cryocat.utils import ioutils
from cryocat._types import (
    MapSource,
    MotlColumn,
    PathOrStr,
    TomoDimensions,
    TripletLike,
    RotationLike,
    MotlType,
    ArrayLike,
    Symmetry,
    ListLike,
)
from cryocat.core.cryomotl import MotlSource
from cryocat.core.surface import (
    Surface,
    DiscreteSurface,
    Mesh,
    OrientedPointCloud,
    AnalyticSurface,
    Cylinder,
    Ellipsoid,
    QuadricsM,
)
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
import math as _math
from typing import Any, Literal
from cryocat.utils.classutils import gui_exposed

# =============================================================================
# Chain — generic linear-chain analysis on traced particles
# =============================================================================


class Chain:
    """Generic linear-chain analysis on a traced motl.

    A *traced motl* is a motl whose ``object_id`` (or other ``store_idx1``)
    identifies the chain that each particle belongs to, and whose
    ``geom2`` (or other ``store_idx2``) gives the position within the
    chain.  ``geom4`` (or ``store_dist``) holds the distance to the next
    particle in the chain.  These columns are produced by
    :func:`nnana.trace_chains`.

    Use the class constructor when you already have a traced motl;
    use :py:meth:`from_motls` / :py:meth:`from_motl` when you have raw
    entry/exit motls and want tracing performed in one step.

    Parameters
    ----------
    traced_motl : str or Motl
    pixel_size : float, default=1.0
    column_name : str, default='tomo_id'
    chain_id_col : str, default='object_id'
    order_id_col : str, default='geom2'
    step_dist_col : str, default='geom4'
    """

    def __init__(
        self,
        traced_motl: MotlSource,
        pixel_size: float = 1.0,
        column_name: MotlColumn = "tomo_id",
        chain_id_col: MotlColumn = "object_id",
        order_id_col: MotlColumn = "geom2",
        step_dist_col: MotlColumn = "geom4",
    ) -> None:
        self.traced_motl = cryomotl.Motl.load(traced_motl)
        self.pixel_size = pixel_size
        self.column_name = column_name
        self.chain_id_col = chain_id_col
        self.order_id_col = order_id_col
        self.step_dist_col = step_dist_col

    @classmethod
    def from_motls(
        cls,
        motl_entry: MotlSource,
        motl_exit: MotlSource,
        max_distance: float,
        min_distance: float = 0,
        column_name: MotlColumn = "tomo_id",
        pixel_size: float = 1.0,
        output_motl: PathOrStr | None = None,
        chain_id_col: MotlColumn = "object_id",
        order_id_col: MotlColumn = "geom2",
        step_dist_col: MotlColumn = "geom4",
    ) -> "Chain":
        """Build a :class:`Chain` by tracing an entry/exit motl pair.

        Calls :func:`nnana.trace_chains` on *motl_entry* and *motl_exit* and
        wraps the resulting traced motl in a :class:`Chain` instance.

        Parameters
        ----------
        motl_entry : MotlSource
            Entry-site particle list.
        motl_exit : MotlSource
            Exit-site particle list.
        max_distance : float
            Maximum allowed step distance (in voxels) between successive
            entry/exit pairs.
        min_distance : float, default=0
            Minimum allowed step distance.
        column_name : MotlColumn, default='tomo_id'
            Column used to group particles before tracing.
        pixel_size : float, default=1.0
            Pixel size in Å; stored on the instance for later distance scaling.
        output_motl : PathOrStr, optional
            Path to save the traced motl.
        chain_id_col : MotlColumn, default='object_id'
            Column that receives the chain identifier.
        order_id_col : MotlColumn, default='geom2'
            Column that receives the within-chain position index.
        step_dist_col : MotlColumn, default='geom4'
            Column that receives the step distance.

        Returns
        -------
        Chain
        """
        traced = nnana.trace_chains(
            motl_entry,
            motl_exit,
            max_distance=max_distance,
            min_distance=min_distance,
            column_name=column_name,
            output_motl=output_motl,
            store_idx1=chain_id_col,
            store_idx2=order_id_col,
            store_dist=step_dist_col,
        )
        return cls(
            traced,
            pixel_size=pixel_size,
            column_name=column_name,
            chain_id_col=chain_id_col,
            order_id_col=order_id_col,
            step_dist_col=step_dist_col,
        )

    @classmethod
    def from_motl(
        cls,
        motl: MotlSource,
        max_distance: float,
        min_distance: float = 0,
        column_name: MotlColumn = "tomo_id",
        pixel_size: float = 1.0,
        output_motl: PathOrStr | None = None,
        chain_id_col: MotlColumn = "object_id",
        order_id_col: MotlColumn = "geom2",
        step_dist_col: MotlColumn = "geom4",
    ) -> "Chain":
        """Build a :class:`Chain` by tracing a single motl (single-site mode).

        Useful for structures where each particle has only one binding site,
        such as nucleosomes in a chromatin chain.  Passes the same motl as
        both entry and exit to :func:`nnana.trace_chains`.

        Parameters
        ----------
        motl : MotlSource
            Particle list to trace.
        max_distance : float
            Maximum allowed step distance (in voxels).
        min_distance : float, default=0
            Minimum allowed step distance.
        column_name : MotlColumn, default='tomo_id'
            Column used to group particles before tracing.
        pixel_size : float, default=1.0
            Pixel size in Å.
        output_motl : PathOrStr, optional
            Path to save the traced motl.
        chain_id_col : MotlColumn, default='object_id'
            Column that receives the chain identifier.
        order_id_col : MotlColumn, default='geom2'
            Column that receives the within-chain position index.
        step_dist_col : MotlColumn, default='geom4'
            Column that receives the step distance.

        Returns
        -------
        Chain
        """
        traced = nnana.trace_chains(
            motl,
            motl_exit=None,
            max_distance=max_distance,
            min_distance=min_distance,
            column_name=column_name,
            output_motl=output_motl,
            store_idx1=chain_id_col,
            store_idx2=order_id_col,
            store_dist=step_dist_col,
        )
        return cls(
            traced,
            pixel_size=pixel_size,
            column_name=column_name,
            chain_id_col=chain_id_col,
            order_id_col=order_id_col,
            step_dist_col=step_dist_col,
        )

    def _step_distances_and_rotated_coords(self, df: pd.DataFrame) -> np.ndarray:
        entry_coord = (df[["x", "y", "z"]].values + df[["shift_x", "shift_y", "shift_z"]].values) * self.pixel_size
        if {"exit_x", "exit_y", "exit_z"}.issubset(df.columns):
            exit_coord = df[["exit_x", "exit_y", "exit_z"]].values * self.pixel_size
            entry_coord = entry_coord[1:, :]
            exit_coord = exit_coord[0:-1, :]
        else:
            exit_coord = entry_coord[0:-1, :]
            entry_coord = entry_coord[1:, :]

        chain_dist = np.linalg.norm(entry_coord - exit_coord, axis=1).reshape(-1, 1)
        centered = entry_coord - exit_coord
        qp_angles = df[["phi", "theta", "psi"]].values[0:-1, :]
        rotated = nnana.rotated_nn_coords(centered, qp_angles)

        n_steps = entry_coord.shape[0]
        chain_size = np.full((n_steps, 1), df.shape[0])  # true chain length, repeated per step

        return np.hstack(
            [
                chain_size,
                chain_dist,
                centered,
                rotated,
            ]
        )

    def _step_rotations(self, df: pd.DataFrame) -> np.ndarray:
        qp_angles = df[["phi", "theta", "psi"]].values[0:-1, :]
        nn_angles = df[["phi", "theta", "psi"]].values[1:, :]
        rel = nnana.relative_rotations(qp_angles, nn_angles)
        points, eul = nnana.rotations_to_unit_vectors(rel)
        zero_rot = srot.from_euler("zxz", angles=np.zeros_like(qp_angles), degrees=True)
        ang_dist = geom.angular_distance(rel, zero_rot)[0].reshape(-1, 1)
        return np.hstack([ang_dist, points, eul])

    def get_chain_stats(self, min_chain_size: int = 2) -> pd.DataFrame:
        """Per-step statistics across all chains.

        Parameters
        ----------
        min_chain_size : int, default=2
            Skip chains shorter than this.

        Returns
        -------
        pandas.DataFrame
            Columns: ``chain_size``, ``distance``, ``coord_x/y/z``,
            ``coord_rx/ry/rz``, ``angular_distance``, ``rot_x/y/z``,
            ``phi/theta/psi``, ``type``.
        """
        df = self.traced_motl.df.copy()
        df.sort_values([self.column_name, self.chain_id_col, self.order_id_col], inplace=True)
        chain_sizes = df.groupby([self.column_name, self.chain_id_col])[self.order_id_col].transform("max")
        df = df[chain_sizes >= min_chain_size]
        if df.empty:
            return pd.DataFrame()

        _kw = {"include_groups": False} if _PD_HAS_INCLUDE_GROUPS else {}
        dist_stats = df.groupby([self.column_name, self.chain_id_col]).apply(
            self._step_distances_and_rotated_coords, **_kw
        )
        rot_stats = df.groupby([self.column_name, self.chain_id_col]).apply(self._step_rotations, **_kw)

        dist_stats = np.vstack(dist_stats.values)
        rot_stats = np.vstack(rot_stats.values)

        out = pd.DataFrame(
            np.hstack([dist_stats, rot_stats]),
            columns=[
                "chain_size",
                "distance",
                "coord_x",
                "coord_y",
                "coord_z",
                "coord_rx",
                "coord_ry",
                "coord_rz",
                "angular_distance",
                "rot_x",
                "rot_y",
                "rot_z",
                "phi",
                "theta",
                "psi",
            ],
        )
        out["type"] = "chain"
        return out

    def get_occupancy(
        self,
        occupancy_id: MotlColumn = "geom1",
        output_motl: PathOrStr | None = None,
    ) -> "cryomotl.Motl":
        """Write the chain length (occupancy) per particle into ``occupancy_id``.

        Each particle receives the length of its chain, i.e. the maximum
        within-chain position index in ``order_id_col``.  The result is stored
        in ``self.traced_motl`` in place.

        Parameters
        ----------
        occupancy_id : MotlColumn, default='geom1'
            Column name that receives the chain-length value.
        output_motl : PathOrStr, optional
            Path to save the updated motl.

        Returns
        -------
        Motl
            The updated ``self.traced_motl``.
        """
        self.traced_motl.df[occupancy_id] = self.traced_motl.df.groupby([self.column_name, self.chain_id_col])[
            self.order_id_col
        ].transform("max")
        if output_motl is not None:
            self.traced_motl.write_out(output_motl)
        return self.traced_motl

    def add_traced_info(
        self,
        input_motl: MotlSource,
        output_motl_path: PathOrStr | None = None,
        sort_by_subtomo: bool = True,
        occupancy_id: MotlColumn = "geom1",
    ) -> "cryomotl.Motl":
        """Copy chain columns from the traced motl onto *input_motl*.

        The columns ``occupancy_id``, ``order_id_col``, ``step_dist_col``, and
        ``chain_id_col`` are transferred by matching ``subtomo_id`` values.
        If occupancy has not yet been computed it is computed first.

        Parameters
        ----------
        input_motl : MotlSource
            Target motl that will receive the chain annotations.
        output_motl_path : PathOrStr, optional
            Path to save the annotated motl.
        sort_by_subtomo : bool, default=True
            Sort both motls by ``subtomo_id`` before copying to ensure correct
            row alignment.
        occupancy_id : MotlColumn, default='geom1'
            Column that holds (or will hold) the chain-length value.

        Returns
        -------
        Motl
            A new Motl with chain columns populated.

        Raises
        ------
        ValueError
            When *input_motl* contains different subtomogram IDs than the
            traced motl.
        """
        if occupancy_id not in self.traced_motl.df.columns or self.traced_motl.df[occupancy_id].isna().all():
            self.get_occupancy(occupancy_id=occupancy_id)

        traced_motl = self.traced_motl
        input_motl = cryomotl.Motl.load(input_motl)

        if sort_by_subtomo:
            traced_motl.df.sort_values(["subtomo_id"], inplace=True)
            input_motl.df.sort_values(["subtomo_id"], inplace=True)

        if not np.array_equal(traced_motl.df["subtomo_id"].values, input_motl.df["subtomo_id"].values):
            raise ValueError("The input motl has different subtomograms than the traced motl.")

        cols = [occupancy_id, self.order_id_col, self.step_dist_col, self.chain_id_col]
        input_motl.df[cols] = traced_motl.df[cols].values
        input_motl.df.sort_values([self.column_name, self.chain_id_col, self.order_id_col], inplace=True)

        if output_motl_path is not None:
            input_motl.write_out(output_motl_path)
        return input_motl

    def get_class_chain_occupancies(
        self,
        mode: Literal["mp", "mdp"] = "mp",
        occupancy_id: MotlColumn = "geom1",
        class_col: MotlColumn = "class",
    ) -> pd.DataFrame:
        """Return per-class chain-occupancy counts broken down by chain type.

        Parameters
        ----------
        mode : {'mp', 'mdp'}, default='mp'
            Breakdown resolution:

            ``'mp'``
                Two categories — monomers (chain length 1) vs. polysomes
                (chain length > 1).
            ``'mdp'``
                Three categories — monomers, disomes (length 2), and
                polysomes (length > 2).
        occupancy_id : MotlColumn, default='geom1'
            Column that holds chain-length values.  Computed automatically
            if not yet present.
        class_col : MotlColumn, default='class'
            Column used to group particles by class.

        Returns
        -------
        pandas.DataFrame
            For ``mode='mp'``: columns ``class``, ``particle_number``,
            ``chain_type``, ``percentage``.
            For ``mode='mdp'``: columns ``class``, ``particle_number``,
            ``chain_type``.

        Raises
        ------
        ValueError
            When *mode* is not ``'mp'`` or ``'mdp'``.
        """
        df = self.traced_motl.df
        if occupancy_id not in df.columns or df[occupancy_id].isna().all():
            self.get_occupancy(occupancy_id=occupancy_id)
            df = self.traced_motl.df

        u_classes = np.unique(df.loc[:, class_col].values)
        rows = []

        if mode == "mp":
            n_total = df.shape[0]
            for c in u_classes:
                mono = df[(df[class_col] == c) & (df[occupancy_id] == 1)].shape[0]
                poly = df[(df[class_col] == c) & (df[occupancy_id] > 1)].shape[0]
                rows.append([c, mono, "monosomes", mono / n_total * 100])
                rows.append([c, poly, "polysomes", poly / n_total * 100])
            return pd.DataFrame(rows, columns=["class", "particle_number", "chain_type", "percentage"])
        elif mode == "mdp":
            for c in u_classes:
                mono = df[(df[class_col] == c) & (df[occupancy_id] == 1)].shape[0]
                di = df[(df[class_col] == c) & (df[occupancy_id] == 2)].shape[0]
                poly = df[(df[class_col] == c) & (df[occupancy_id] > 2)].shape[0]
                rows.append([c, mono, "monosomes"])
                rows.append([c, di, "disomes"])
                rows.append([c, poly, "polysomes"])
            return pd.DataFrame(rows, columns=["class", "particle_number", "chain_type"])
        else:
            raise ValueError(f"mode must be 'mp' or 'mdp', got {mode!r}.")

    @gui_exposed(label="Step statistics", group="Statistics", order=10, returns="dataframe")
    def get_step_stats(self) -> pd.DataFrame:
        """Per-step angular and normal-vector distances along each chain.

        Within each chain particles are sorted by ``order_id_col`` and
        orientation distances are computed for every consecutive pair
        (particle *i* → particle *i* + 1).  Calls
        :func:`nnana.angular_distances` with ``rotation_type="all"`` on the
        two paired ZXZ-angle arrays.

        Grouping and sorting use ``self.column_name``, ``self.chain_id_col``,
        and ``self.order_id_col``; these come from the :class:`Chain` instance
        and are never hardcoded.  A chain of one particle contributes no rows.

        Returns
        -------
        pandas.DataFrame
            One row per step; *n* particles → *n* − 1 rows.

            ========================  ========  ==========================================
            ``<column_name>``         —         value of ``self.column_name`` (e.g. tomo_id)
            ``chain_id``              —         value of ``self.chain_id_col``
            ``step``                  —         ``order_id_col`` value of the upstream
                                                particle
            ``step_dist``             motl's own units    ``step_dist_col`` of the upstream particle;
                                                NaN when the column is absent
            ``angular_distance``      degrees   SO(3) geodesic distance between orientations
            ``cone_distance``         degrees   angle between z-axes (normal-vector distance)
            ``in_plane_distance``     degrees   in-plane rotation component
            ========================  ========  ==========================================

        Notes
        -----
        Sorting is always by ``order_id_col`` within each group — never by
        DataFrame row order.  Duplicate ``order_id_col`` values within one
        chain leave the order among tied particles undefined (stable sort,
        arbitrary tie-break).  Gaps in ``order_id_col`` are harmless.
        """
        df = self.traced_motl.df
        parts: list[pd.DataFrame] = []
        for (tomo_val, chain_val), group in df.groupby([self.column_name, self.chain_id_col], sort=True):
            group_sorted = group.sort_values(self.order_id_col)
            if len(group_sorted) < 2:
                continue
            qp_angles = group_sorted[["phi", "theta", "psi"]].values[:-1]
            nn_angles = group_sorted[["phi", "theta", "psi"]].values[1:]
            ang_dist, cone_dist, inplane_dist = nnana.angular_distances(qp_angles, nn_angles, rotation_type="all")
            n_steps = len(qp_angles)
            step_dists = (
                group_sorted[self.step_dist_col].values[:-1]
                if self.step_dist_col in group_sorted.columns
                else np.full(n_steps, np.nan)
            )
            parts.append(
                pd.DataFrame(
                    {
                        self.column_name: tomo_val,
                        "chain_id": chain_val,
                        "step": group_sorted[self.order_id_col].values[:-1],
                        "step_dist": step_dists,
                        "angular_distance": ang_dist,
                        "cone_distance": cone_dist,
                        "in_plane_distance": inplane_dist,
                    }
                )
            )
        if not parts:
            return pd.DataFrame(
                columns=[
                    self.column_name,
                    "chain_id",
                    "step",
                    "step_dist",
                    "angular_distance",
                    "cone_distance",
                    "in_plane_distance",
                ]
            )
        return pd.concat(parts, ignore_index=True)


# =============================================================================
# Utilities for symmetric complexes
# =============================================================================

_GROUP_ORDER: dict[str, Callable[[int], int]] = {
    "C": lambda n: n,
    "D": lambda n: 2 * n,
    "T": lambda n: n,
    "O": lambda n: n,
    "I": lambda n: n,
}


def complex_centers(
    motl: MotlSource,
    *,
    affiliation_column: MotlColumn = "object_id",
    tomo_id_column: MotlColumn = "tomo_id",
    weights: ArrayLike | None = None,
) -> "cryomotl.Motl":
    """Return one barycentric centre particle per (tomogram, object) group.

    Parameters
    ----------
    motl : MotlSource
        Particle list.
    affiliation_column : MotlColumn, default='object_id'
        Column that identifies which object each particle belongs to.
    tomo_id_column : MotlColumn, default='tomo_id'
        Column that identifies the tomogram.
    weights : ArrayLike, optional
        Per-particle weights forwarded to :func:`geom.barycenter`.

    Returns
    -------
    cryomotl.Motl
        One row per ``(tomo_id, affiliation)`` pair.  ``tomo_id`` and
        ``object_id`` carry the group identifiers; all other columns are
        zero-filled.
    """
    m = cryomotl.Motl.load(motl)
    central_points: list[np.ndarray] = []
    tomo_ids: list[float] = []
    object_ids: list[float] = []

    for t in m.get_unique_values(tomo_id_column):
        tm = m.get_motl_subset(column_values=[t], column_name=tomo_id_column, reset_index=True)
        for o in tm.get_unique_values(affiliation_column):
            om = tm.get_motl_subset(column_values=[o], column_name=affiliation_column, reset_index=True)
            coords = om.get_coordinates()
            center = geom.barycenter(coords, weights) if coords.shape[0] > 0 else np.zeros(3)
            central_points.append(center)
            tomo_ids.append(float(t))
            object_ids.append(float(o))

    out = cryomotl.Motl()
    if central_points:
        pts = np.vstack(central_points)
        out.fill(
            {
                "x": pts[:, 0],
                "y": pts[:, 1],
                "z": pts[:, 2],
                "tomo_id": np.array(tomo_ids),
                "object_id": np.array(object_ids),
            }
        )
        out.renumber_particles()
    out.df.fillna(0.0, inplace=True)
    return out


def expand_motl(
    motl: MotlSource,
    shift_vecs: np.ndarray,
    *,
    original_id_col: MotlColumn = "object_id",
    order_id_col: MotlColumn = "geom1",
    sort_vectors: bool = True,
    orientation: Literal["radial", "keep"] = "radial",
    start_index: int = 0,
) -> "cryomotl.Motl":
    """Expand a particle list by placing one copy at each shift vector.

    For every shift vector ``v_k`` and every source particle with rotation
    ``R`` and centre ``c``, a new particle is placed at ``c + R.apply(v_k)``.
    When *orientation* is ``"radial"``, the new particle's z-axis is aligned
    with ``v_k`` and a random in-plane rotation (phi) is drawn.  When
    *orientation* is ``"keep"``, the original orientation is left unchanged.

    Parameters
    ----------
    motl : MotlSource
        Input particle list.
    shift_vecs : numpy.ndarray, shape (M, 3)
        Site displacement vectors in the particle frame (voxels).
    original_id_col : MotlColumn, default="object_id"
        Column in which to store the source particle's ``subtomo_id``.
    order_id_col : MotlColumn, default="geom1"
        Column in which to store the site index (``start_index + position``).
    sort_vectors : bool, default=True
        When ``True``, vectors are sorted by (x, y, z) before expansion so
        that ``order_id_col`` values follow a deterministic order.
    orientation : {"radial", "keep"}, default="radial"
        ``"radial"`` rotates each copy so that its z-axis points along the
        shift vector and assigns a random phi.  ``"keep"`` copies the source
        orientation without modification.
    start_index : int, default=0
        Value assigned to ``order_id_col`` for the first shift vector.
        Subsequent vectors receive ``start_index + 1``, ``start_index + 2``, …

    Returns
    -------
    cryomotl.Motl
        Expanded particle list with ``M × N`` rows (M vectors × N source
        particles), sorted by (``original_id_col``, ``order_id_col``) and
        renumbered.

    Raises
    ------
    ValueError
        If *orientation* is not a recognised string.

    Notes
    -----
    When ``orientation="radial"`` and the shift vector is collinear with the
    z-axis, the cross-product used to build the rotation is zero.  The guard
    is: ``‖v × ẑ‖ < 1e-12`` → identity rotation for +z, 180° around x for
    −z.  This avoids ``NaN`` Euler angles for shifts along ±z.
    """
    if orientation not in ("radial", "keep"):
        raise ValueError(f"orientation must be 'radial' or 'keep', got {orientation!r}")

    motl = cryomotl.Motl.load(motl)
    shift_vecs = np.asarray(shift_vecs, dtype=float)

    if sort_vectors:
        idx = np.lexsort((shift_vecs[:, 2], shift_vecs[:, 1], shift_vecs[:, 0]))
        shift_vecs = shift_vecs[idx]

    motl_subparticles = []
    for position in range(len(shift_vecs)):
        sv = shift_vecs[position]
        df_copy = motl.df.copy()
        df_copy["score"] = 0
        df_copy["subtomo_mean"] = 0
        df_copy[original_id_col] = motl.df["subtomo_id"]
        df_copy[order_id_col] = start_index + position

        motl_sub = cryomotl.Motl(df_copy)
        motl_sub.shift_positions(sv)
        motl_sub.update_coordinates()

        if orientation == "radial":
            target_normal = geom.normalize_vector(sv)
            reference_normal = np.array([0.0, 0.0, 1.0])
            dot = float(np.clip(np.dot(reference_normal, target_normal), -1.0, 1.0))
            axis_raw = np.cross(reference_normal, target_normal)
            axis_norm = np.linalg.norm(axis_raw)
            if axis_norm < 1e-12:
                rotation = srot.identity() if dot > 0 else srot.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
            else:
                rotation = srot.from_rotvec(np.arccos(dot) * axis_raw / axis_norm)
            motl_sub.apply_rotation(rotation)
            motl_sub.fill({"phi": np.random.rand(len(motl_sub.df)) * 360})

        motl_subparticles.append(motl_sub)

    output_motl = motl_subparticles[0]
    for mp in motl_subparticles[1:]:
        output_motl = output_motl + mp

    output_motl.df = output_motl.df.sort_values(by=[original_id_col, order_id_col], ascending=[True, True]).reset_index(
        drop=True
    )
    output_motl.renumber_particles()
    return output_motl


def _check_column_collisions(method_name: str, col_map: dict[str, str]) -> None:
    seen: dict[str, str] = {}
    for field, col in col_map.items():
        if col in seen:
            raise ValueError(
                f"{method_name}: {field} and {seen[col]} "
                f"both map to column '{col}'; set one to a different column."
            )
        seen[col] = field


def trace_faces(
    partner: np.ndarray,
    block: np.ndarray,
    site: np.ndarray,
    n_sites: np.ndarray,
) -> np.ndarray:
    """Assign face IDs to half-edges by walking the contact graph.

    A face is a closed cycle obtained by repeatedly crossing a contact and
    advancing to the next site counter-clockwise on the partner block.  Open
    paths (hitting an unmatched site or a previously visited half-edge) are
    labelled −1 (boundary).

    Parameters
    ----------
    partner : numpy.ndarray, shape (H,)
        Index of the matching half-edge for each half-edge, or −1 if
        unmatched.  Indices are 0-based positions in the calling array.
    block : numpy.ndarray, shape (H,)
        Block ID for each half-edge.
    site : numpy.ndarray, shape (H,)
        1-based site index for each half-edge.
    n_sites : numpy.ndarray, shape (H,)
        Number of sites on the block of each half-edge.

    Returns
    -------
    numpy.ndarray, shape (H,), dtype int
        ``face[h]`` is −1 for boundary half-edges, or a positive integer
        identifying the closed face that contains ``h``.  Face IDs are
        assigned in traversal order (first closed face found gets id 1,
        second gets 2, …).

    Notes
    -----
    The traversal rule is: after crossing to half-edge ``p = partner[h]``,
    advance to the next CCW site on ``p``'s block:
    ``lookup[(block[p], site[p] % n_sites[p] + 1)]``.

    The "next" map is injective on matched half-edges, so every half-edge
    lies on exactly one cycle (closed face) or one path (boundary).
    """
    H = len(partner)
    lookup: dict[tuple[int, int], int] = {(int(block[h]), int(site[h])): h for h in range(H)}
    face = np.full(H, -2, dtype=np.intp)
    next_id = 1
    for start in range(H):
        if face[start] != -2:
            continue
        walk: list[int] = []
        h = start
        closed = False
        while True:
            walk.append(h)
            p = int(partner[h])
            if p < 0:
                break
            ns_p = int(n_sites[p])
            next_site = int(site[p]) % ns_p + 1
            h_next = lookup.get((int(block[p]), next_site))
            if h_next is None:
                break
            h = h_next
            if h == start:
                closed = True
                break
            if face[h] != -2:
                break
        arr = np.array(walk, dtype=np.intp)
        face[arr] = next_id if closed else -1
        if closed:
            next_id += 1
    return face


# =============================================================================
# SymmetricComplex — generic point-group symmetric multi-subunit complex
# =============================================================================


class SymmetricComplex:
    """Base class for point-group symmetric multi-subunit complexes.

    Encapsulates a motl of subunit particles that form a symmetric complex
    (cyclic, dihedral, or Platonic) and provides per-object centre
    computation, orientation unification, and geometric statistics that are
    independent of the specific symmetry type.

    Parameters
    ----------
    motl : MotlSource
        Subunit particle list.
    symmetry : Symmetry
        Symmetry specifier, e.g. ``"C8"``, ``"D6"``, ``"T"``, ``"O"``,
        ``"I"``, or a bare integer (interpreted as cyclic).
    affiliation_column : MotlColumn, default='object_id'
        Column that identifies which object each particle belongs to.
    order_column : MotlColumn, default='geom1'
        Column that subunit-ordering methods write indices into.
    tomo_id_column : MotlColumn, default='tomo_id'
        Column that identifies the tomogram.
    """

    def __init__(
        self,
        motl: MotlSource,
        symmetry: Symmetry,
        *,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
    ) -> None:
        self._setup(
            motl,
            symmetry,
            affiliation_column=affiliation_column,
            order_column=order_column,
            tomo_id_column=tomo_id_column,
        )

    def _setup(
        self,
        motl: MotlSource,
        symmetry: Symmetry,
        *,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
    ) -> None:
        """Shared constructor body; called by :meth:`__init__` and subclass constructors."""
        self.motl = cryomotl.Motl.load(motl)
        self.group, self.fold = geom.as_symmetry(symmetry)
        self.n_subunits: int = _GROUP_ORDER[self.group](self.fold)
        self.affiliation_column: MotlColumn = affiliation_column
        self.order_column: MotlColumn = order_column
        self.tomo_id_column: MotlColumn = tomo_id_column

    # ------------------------------------------------------------------
    # Centre computation
    # ------------------------------------------------------------------

    @gui_exposed(label="Get centers as motl", group="Statistics", order=10, returns="motl")
    def get_centers_as_motl(self) -> "cryomotl.Motl":
        """Return a Motl with one barycentric centre particle per object per tomogram.

        Iterates over all tomograms in ``self.motl``, groups by
        ``affiliation_column`` within each tomogram, and returns the
        barycentric centre of each group.

        Returns
        -------
        Motl
            One row per (tomogram, object) pair.  ``tomo_id`` holds the
            tomogram identifier and ``object_id`` holds the affiliation value.
            All other columns are zero-filled.
        """
        central_points: list[np.ndarray] = []
        tomo_ids: list[float] = []
        object_ids: list[float] = []

        for t in self.motl.get_unique_values(self.tomo_id_column):
            tm = self.motl.get_motl_subset(column_values=[t], column_name=self.tomo_id_column, reset_index=True)
            for o in tm.get_unique_values(self.affiliation_column):
                om = tm.get_motl_subset(column_values=[o], column_name=self.affiliation_column, reset_index=True)
                coords = om.get_coordinates()
                center = geom.barycenter(coords) if coords.shape[0] > 0 else np.zeros(3)
                central_points.append(center)
                tomo_ids.append(float(t))
                object_ids.append(float(o))

        out = cryomotl.Motl()
        if central_points:
            pts = np.vstack(central_points)
            out.fill(
                {
                    "x": pts[:, 0],
                    "y": pts[:, 1],
                    "z": pts[:, 2],
                    "tomo_id": np.array(tomo_ids),
                    "object_id": np.array(object_ids),
                }
            )
            out.renumber_particles()
        out.df.fillna(0.0, inplace=True)
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_affiliation(self) -> None:
        """Raise if ``affiliation_column`` is absent from ``self.motl.df``."""
        if self.affiliation_column not in self.motl.df.columns:
            raise ValueError(
                f"{type(self).__name__}: affiliation column {self.affiliation_column!r} is not present "
                "in self.motl.df.  Run affiliation first, or set affiliation_column correctly."
            )

    # ------------------------------------------------------------------
    # Subunit ordering (dispatch hook — subclasses override)
    # ------------------------------------------------------------------

    def assign_subunit_order(self) -> None:
        """Assign subunit indices into ``self.order_column``.

        Subclasses must override this method with a symmetry-specific
        implementation.

        Raises
        ------
        NotImplementedError
            Always; subclasses define subunit ordering.
        """
        raise NotImplementedError("subclasses define subunit ordering")

    # ------------------------------------------------------------------
    # Per-object evaluations
    # ------------------------------------------------------------------

    @gui_exposed(label="Occupancy", group="Statistics", order=20, returns="dataframe")
    def occupancy(self) -> pd.DataFrame:
        """Per-object subunit occupancy.

        For every ``(tomo_id, object_id)`` group, counts present subunits
        and computes the fraction of the expected ``n_subunits``.  When
        ``order_column`` is populated, the *missing* indices
        (1 … n_subunits) are also reported.

        Returns
        -------
        pandas.DataFrame
            Columns:

            ``tomo_id``, ``object_id``
                Group identifiers.
            ``n_present``
                Number of particles in the group.
            ``occupancy``
                ``n_present / self.n_subunits``.
            ``missing``
                Sorted list of 1-based subunit indices absent from
                ``order_column`` (empty list when fully occupied);
                ``None`` when ``order_column`` is not in the motl.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        """
        self._require_affiliation()

        has_order = self.order_column in self.motl.df.columns
        rows: list[dict] = []
        all_expected = set(range(1, self.n_subunits + 1))

        for (tomo_id, object_id), group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            n_present = len(group)
            if has_order:
                present = set(int(v) for v in group[self.order_column].dropna())
                missing: list[int] | None = sorted(all_expected - present)
            else:
                missing = None
            rows.append(
                {
                    "tomo_id": float(tomo_id),
                    "object_id": float(object_id),
                    "n_present": n_present,
                    "occupancy": n_present / self.n_subunits,
                    "missing": missing,
                }
            )

        return pd.DataFrame(rows)

    @gui_exposed(label="Clean per object", group="Statistics", order=30, returns="motl")
    def clean_per_object(
        self,
        column: MotlColumn,
        keep: Literal["high", "low"] = "high",
        *,
        n: int | None = None,
    ) -> "cryomotl.Motl":
        """Keep the *n* best rows per object and drop the rest.

        For each ``(tomo_id_column, affiliation_column)`` group: sort by
        *column*, keep the top-*n* rows according to *keep*, and discard the
        remainder.  Objects that already have at most *n* rows are returned
        unchanged.

        Parameters
        ----------
        column : MotlColumn
            Column to sort and filter by.
        keep : {'high', 'low'}, default='high'
            ``'high'`` retains the *n* rows with the largest values (e.g.
            scores); ``'low'`` retains the *n* rows with the smallest values
            (e.g. cone distances).
        n : int, optional
            Number of rows to keep per object.  Defaults to ``self.n_subunits``.

        Returns
        -------
        cryomotl.Motl
            A copy of ``self.motl`` with over-occupied objects trimmed.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        """
        self._require_affiliation()

        n_keep = self.n_subunits if n is None else n
        ascending = keep == "low"

        rows: list[pd.DataFrame] = []
        for (_t, _o), grp in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            if len(grp) <= n_keep:
                rows.append(grp)
            else:
                rows.append(grp.sort_values(column, ascending=ascending).head(n_keep))

        df_out = pd.concat(rows).reset_index(drop=True)
        return cryomotl.Motl(df_out)

    # ------------------------------------------------------------------
    # Object deduplication
    # ------------------------------------------------------------------

    @gui_exposed(label="Merge subunits", group="Affiliation", order=20, returns="none")
    def merge_subunits(
        self,
        radius: float = 55,
    ) -> None:
        """Merge near-duplicate objects whose centres are within *radius*.

        For each tomogram:

        1. Compute per-object barycentric centres.
        2. Find object-centre pairs within *radius* using
           :func:`nnana.get_nn_within_distance`.
        3. Re-assign ``affiliation_column`` of near objects to the first
           encountered partner.
        4. Recount occupancy into ``geom1`` and recompute subunit order for
           all objects via :meth:`assign_subunit_order`.

        Parameters
        ----------
        radius : float, default=55
            Distance threshold in voxels.

        Notes
        -----
        Modifies ``self.motl.df`` in place.
        """
        motl = self.motl
        aff_col = self.affiliation_column
        tomo_col = self.tomo_id_column

        for t in motl.get_unique_values(tomo_col):
            tm = motl.get_motl_subset(column_values=[t], column_name=tomo_col, reset_index=True)

            pts: list[np.ndarray] = []
            obj_ids: list[float] = []
            for o in tm.get_unique_values(aff_col):
                om = tm.get_motl_subset(column_values=[o], column_name=aff_col, reset_index=True)
                coords = om.get_coordinates()
                center = geom.barycenter(coords) if coords.shape[0] > 0 else np.zeros(3)
                pts.append(center)
                obj_ids.append(float(o))

            centers_motl = cryomotl.Motl()
            if pts:
                pts_arr = np.vstack(pts)
                centers_motl.fill(
                    {
                        "x": pts_arr[:, 0],
                        "y": pts_arr[:, 1],
                        "z": pts_arr[:, 2],
                        "tomo_id": t,
                        "object_id": obj_ids,
                    }
                )
                centers_motl.renumber_particles()
            centers_motl.df.fillna(0.0, inplace=True)

            if centers_motl.df.shape[0] > 1:
                center_stats = nnana.get_nn_stats(centers_motl, centers_motl)
                if any(center_stats["distance"] <= radius):
                    center_idx, nn_idx = nnana.get_nn_within_distance(centers_motl, radius)
                    for i, o in enumerate(center_idx):
                        o_id1 = centers_motl.df.loc[centers_motl.df.index[o], "object_id"]
                        for j in nn_idx[i]:
                            o_id2 = centers_motl.df.loc[centers_motl.df.index[j], "object_id"]
                            tm.df.loc[tm.df[aff_col] == o_id2, aff_col] = o_id1

            tm.df["geom1"] = tm.df.groupby([aff_col])[aff_col].transform("count")
            tm.df[aff_col] = tm.df[aff_col].rank(method="dense").astype(int)

            update_cols = list({aff_col, "geom1"})
            motl.df.loc[motl.df[tomo_col] == t, update_cols] = tm.df[update_cols].values

        motl.df.reset_index(inplace=True, drop=True)
        motl.df["geom1"] = motl.df.groupby([tomo_col, aff_col])[aff_col].transform("count")
        motl.df[aff_col] = motl.df[aff_col].rank(method="dense").astype(int)
        self.assign_subunit_order()

    # ------------------------------------------------------------------
    # Affiliation creation
    # ------------------------------------------------------------------

    @gui_exposed(label="Create affiliation", group="Affiliation", order=10, returns="motl")
    def create_affiliation(
        self,
        method: Literal["tracing", "radius"] = "radius",
        *,
        shift: float | None = None,
        radius: float | None = None,
        normals_threshold: float | None = None,
        occupancy_column: MotlColumn | None = None,
        cone_distance_column: MotlColumn | None = None,
        min_occupancy: int | None = None,
    ) -> "cryomotl.Motl":
        """Cluster subunit particles into objects and write affiliation labels.

        Operates on a copy of ``self.motl`` and returns it with
        ``affiliation_column`` populated.  For every object that ends up with
        more than ``self.n_subunits`` particles, outlier-normal particles are
        removed (see *normals_threshold*).  If an object remains over-populated
        after cleaning its ``geom5`` column is set to ``1.0`` and a one-line
        notice is printed.

        Subunit ordering is a separate stage-2 call to
        :meth:`assign_subunit_order`.  Occupancy and normal deviations are
        computed on demand via :meth:`occupancy` and :meth:`normal_deviations`.

        Parameters
        ----------
        method : {'radius', 'tracing'}, default='radius'
            Clustering strategy.

            ``'radius'``
                Optionally shift particles along their local −x axis by
                *shift*, then run a self nearest-neighbour search within
                *radius*.  Connected components of the NN graph become
                objects.  Isolated particles (no NN within *radius*) are
                kept as singleton objects with unique ``affiliation_column``
                values.

            ``'tracing'``
                Optionally shift particles along their local −x axis by
                *shift*, then trace chains via :func:`nnana.trace_chains`
                with *radius* as the maximum link distance.  Each chain
                becomes one object.

        shift : float, optional
            Magnitude of the local-frame shift along −x applied before
            clustering (voxels).  When ``None`` no recentring is performed.
            Typical value: approximate ring radius.
        radius : float
            For ``method='radius'``: NN search radius (voxels).
            For ``method='tracing'``: maximum chain-link distance (voxels).
            **Required.**
        normals_threshold : float or None, default=None
            Cone-distance cutoff (degrees) applied **only** to over-populated
            objects (size > ``n_subunits``).  ``None`` uses the adaptive rule
            ``median(cone) + 1 × std(cone)`` per object.
        occupancy_column : MotlColumn or None, default=None
            Accepted but ignored.  Use :meth:`occupancy` on demand.
        cone_distance_column : MotlColumn or None, default=None
            Accepted but ignored.  Use :meth:`normal_deviations` on demand.
        min_occupancy : int or None, default=None
            When set, objects whose size (after over-population cleaning) is
            below this value are removed.  ``None`` keeps all objects.

        Returns
        -------
        cryomotl.Motl
            A new motl with ``affiliation_column`` and ``geom5`` populated.
            ``geom5`` is ``0.0`` for all particles; ``1.0`` marks objects that
            are still over-populated after cleaning.

        Raises
        ------
        ValueError
            If *radius* is ``None`` or *method* is unrecognised.
        """
        if radius is None:
            raise ValueError(
                f"{type(self).__name__}.create_affiliation: 'radius' is required "
                "(NN search radius for 'radius'; max link distance for 'tracing')."
            )

        motl_out = cryomotl.Motl(self.motl.df.copy())
        motl_out.df.reset_index(drop=True, inplace=True)
        motl_out.renumber_particles()

        if method == "radius":
            self._affiliating_by_radius(motl_out, shift=shift, radius=radius)
        elif method == "tracing":
            self._affiliating_by_tracing(motl_out, shift=shift, radius=radius)
        else:
            raise ValueError(
                f"{type(self).__name__}.create_affiliation: unknown method {method!r}. " "Choose 'radius' or 'tracing'."
            )

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        # Initialize inspection flag column
        motl_out.df["geom5"] = 0.0

        # Clean over-populated objects (size > n_subunits) only
        sizes = motl_out.df.groupby([tomo_col, aff_col]).size()
        over = sizes[sizes > self.n_subunits]

        if not over.empty:
            all_cone = self._cone_angles(motl_out.df, tomo_col, aff_col)
            tomo_vals = motl_out.df[tomo_col].values
            aff_vals = motl_out.df[aff_col].values
            drop_mask = np.zeros(len(motl_out.df), dtype=bool)

            for (t, o) in over.index:
                pos = np.where((tomo_vals == t) & (aff_vals == o))[0]
                obj_angles = all_cone[pos]
                if normals_threshold is not None:
                    cutoff = float(normals_threshold)
                else:
                    cutoff = float(np.median(obj_angles) + np.std(obj_angles))
                drop_mask[pos] |= obj_angles > cutoff

            motl_out = cryomotl.Motl(motl_out.df[~drop_mask].reset_index(drop=True))

            # Flag objects still over-populated after cleaning
            new_sizes = motl_out.df.groupby([tomo_col, aff_col]).size()
            still_over = new_sizes[new_sizes > self.n_subunits]
            if not still_over.empty:
                flag_loc = motl_out.df.columns.get_loc("geom5")
                new_tomo = motl_out.df[tomo_col].values
                new_aff = motl_out.df[aff_col].values
                for (t, o), sz in still_over.items():
                    pos = np.where((new_tomo == t) & (new_aff == o))[0]
                    motl_out.df.iloc[pos, flag_loc] = 1.0
                    print(f"object ({t}, {o}): {sz} > n={self.n_subunits} subunits after cleaning")

        # Drop under-populated objects when min_occupancy is set
        if min_occupancy is not None:
            occ = motl_out.df.groupby([tomo_col, aff_col])[aff_col].transform("size")
            motl_out = cryomotl.Motl(motl_out.df[occ >= min_occupancy].reset_index(drop=True))

        return motl_out

    def _affiliating_by_radius(
        self,
        motl_out: "cryomotl.Motl",
        *,
        shift: float | None,
        radius: float,
    ) -> None:
        """Label ``affiliation_column`` via radius-NN connected components.

        Modifies *motl_out*.df in place.  Isolated particles (no NN within
        *radius*) receive unique sequential labels per tomogram.

        Parameters
        ----------
        motl_out : cryomotl.Motl
            Working copy (must have a 0-based integer index and unique
            ``subtomo_id`` values — guaranteed by the caller).
        shift : float or None
            If given, particles are shifted along local −x by *shift* before
            the NN search.  The NN search uses shifted coordinates; the
            positions stored in *motl_out* are unchanged.
        radius : float
            NN search radius in voxels.
        """
        if shift is not None:
            motl_search = motl_out.shift_positions([-shift, 0.0, 0.0], inplace=False)
        else:
            motl_search = motl_out

        all_coords = motl_search.get_coordinates()  # (N, 3)
        motl_out.df[self.affiliation_column] = np.nan

        for tomo_val, group_df in motl_out.df.groupby(self.tomo_id_column):
            row_pos = group_df.index.to_numpy()
            coords = all_coords[row_pos]
            subtomo_ids = group_df["subtomo_id"].to_numpy()

            qp_idx, nn_idx_list = nnana.find_nn_within_radius(coords, coords, radius, remove_qp=True)

            qp_ids: list = []
            nn_ids: list = []
            for qi, nns in zip(qp_idx, nn_idx_list):
                for ni in nns:
                    qp_ids.append(int(subtomo_ids[qi]))
                    nn_ids.append(int(subtomo_ids[ni]))

            next_id = 1
            in_component: set = set()

            if qp_ids:
                components = _clustering.connected_component_clusters(qp_ids, nn_ids, min_size=1)
                for comp in components:
                    comp_ids = set(comp.nodes())
                    mask = group_df["subtomo_id"].isin(comp_ids)
                    motl_out.df.loc[group_df[mask].index, self.affiliation_column] = float(next_id)
                    in_component.update(comp_ids)
                    next_id += 1

            # Isolated particles — not in any NN edge
            isolated = group_df[~group_df["subtomo_id"].isin(in_component)]
            for idx in isolated.index:
                motl_out.df.loc[idx, self.affiliation_column] = float(next_id)
                next_id += 1

    def _affiliating_by_tracing(
        self,
        motl_out: "cryomotl.Motl",
        *,
        shift: float | None,
        radius: float,
    ) -> None:
        """Label ``affiliation_column`` by chain-tracing.

        Calls :func:`nnana.trace_chains` and copies the resulting chain IDs
        into *motl_out*.df in place.

        Parameters
        ----------
        motl_out : cryomotl.Motl
            Working copy (must have unique ``subtomo_id`` values).
        shift : float or None
            If given, shift motl along local −x before tracing so that the
            trace links shifted (recentred) positions.
        radius : float
            Maximum chain-link distance (voxels), passed as ``max_distance``
            to :func:`nnana.trace_chains`.
        """
        if shift is not None:
            motl_entry = motl_out.shift_positions([-shift, 0.0, 0.0], inplace=False)
        else:
            motl_entry = cryomotl.Motl(motl_out.df.copy())

        traced = nnana.trace_chains(
            motl_entry,
            motl_exit=None,
            max_distance=radius,
            column_name=self.tomo_id_column,
            store_idx1=self.affiliation_column,
            store_idx2="_cns_trace_order_tmp_",
        )

        # Copy affiliation to motl_out by subtomo_id alignment
        traced.df.sort_values("subtomo_id", inplace=True)
        traced.df.reset_index(drop=True, inplace=True)
        motl_out.df.sort_values("subtomo_id", inplace=True)
        motl_out.df.reset_index(drop=True, inplace=True)
        motl_out.df[self.affiliation_column] = traced.df[self.affiliation_column].values

    @staticmethod
    def _cone_angles(
        df: pd.DataFrame,
        tomo_col: str,
        aff_col: str,
    ) -> np.ndarray:
        """Per-particle angle (degrees) to per-object mean z-normal.

        Parameters
        ----------
        df : pd.DataFrame
            Must have a 0-based integer RangeIndex and columns
            ``phi``, ``theta``, ``psi`` (zxz Euler, degrees).
        tomo_col, aff_col : str
            Tomogram and affiliation column names used for grouping.

        Returns
        -------
        np.ndarray of shape ``(len(df),)``
            Angle in degrees for each row, aligned to *df*'s positional index.
        """
        euler = df[["phi", "theta", "psi"]].to_numpy(dtype=float)
        z_axes = geom.euler_angles_to_normals(euler)
        out = np.zeros(len(df), dtype=float)
        for _, grp in df.groupby([tomo_col, aff_col]):
            pos = grp.index.to_numpy()
            zv = z_axes[pos]
            mean_z = zv.mean(axis=0)
            norm = float(np.linalg.norm(mean_z))
            mean_z = mean_z / norm if norm > 1e-10 else np.array([0.0, 0.0, 1.0])
            out[pos] = np.degrees(np.arccos(np.clip(zv @ mean_z, -1.0, 1.0)))
        return out

    def normal_deviations(self) -> pd.DataFrame:
        """Per-particle angle to per-object consensus z-axis with object stats.

        For each particle, reports the angle (degrees) between its z-normal
        (from ``phi``/``theta``/``psi`` via :func:`geom.euler_angles_to_normals`)
        and the mean z-normal of its object.  Also reports per-object
        ``median`` and ``std`` of those angles.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``object_id``, ``subtomo_id``,
            ``deviation``, ``object_median``, ``object_std``.
        """
        self._require_affiliation()
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        df = self.motl.df.reset_index(drop=True)
        angles = self._cone_angles(df, tomo_col, aff_col)

        result = df[[tomo_col, aff_col, "subtomo_id"]].copy()
        result = result.rename(columns={tomo_col: "tomo_id", aff_col: "object_id"})
        result["deviation"] = angles

        stats = (
            result.groupby(["tomo_id", "object_id"])["deviation"]
            .agg(object_median="median", object_std="std")
            .reset_index()
        )
        return result.merge(stats, on=["tomo_id", "object_id"])


# =============================================================================
# CnComplex — cyclic Cn ring structure
# =============================================================================


class CnComplex(SymmetricComplex):
    """Cyclic Cn-symmetric ring structure.

    Extends :class:`SymmetricComplex` with methods specific to cyclic
    symmetry: subunit ordering, affiliation clustering, occupancy analysis,
    and diameter computation.

    Parameters
    ----------
    motl : MotlSource
        Subunit particle list.
    symmetry : Symmetry
        Cyclic fold, e.g. ``"C8"`` or ``8``.  Dihedral or Platonic
        symmetries raise :class:`ValueError`.
    affiliation_column : MotlColumn, default='object_id'
        Column that identifies which object each particle belongs to.
    order_column : MotlColumn, default='geom1'
        Column that :meth:`assign_subunit_order` writes cyclic indices into.
    tomo_id_column : MotlColumn, default='tomo_id'
        Column that identifies the tomogram.
    center_method : {'circle_fit', 'barycentric'}, default='circle_fit'
        Algorithm used by :meth:`get_centers_as_motl` and related helpers.
        ``'circle_fit'`` falls back to barycentric when the fit fails.

    Raises
    ------
    ValueError
        When *symmetry* is not a cyclic Cn group.
    """

    def __init__(
        self,
        motl: MotlSource,
        symmetry: Symmetry,
        *,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
        center_method: Literal["circle_fit", "barycentric"] = "circle_fit",
    ) -> None:
        super().__init__(
            motl,
            symmetry,
            affiliation_column=affiliation_column,
            order_column=order_column,
            tomo_id_column=tomo_id_column,
        )
        if self.group != "C":
            raise ValueError(f"CnComplex requires cyclic Cn symmetry, got {symmetry!r}.")
        self.center_method: Literal["circle_fit", "barycentric"] = center_method
        self._cyclic_setup()

    def _cyclic_setup(self) -> None:
        """Initialise cyclic-ring attributes shared with :class:`DnComplex`.

        Sets ``self.n`` to the half-ring fold, ``self._ring_group_columns``
        to ``[tomo_id_column, affiliation_column]``, and ``self._components``
        to an empty dict.  Called from :meth:`CnComplex.__init__` and
        :meth:`DnComplex.__init__`.
        """
        self.n: int = self.fold
        self._ring_group_columns: list[MotlColumn] = [
            self.tomo_id_column,
            self.affiliation_column,
        ]
        self._components: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Centre computation (circle-fit with barycentric fallback)
    # ------------------------------------------------------------------

    def _compute_object_center(
        self,
        object_motl: "cryomotl.Motl",
    ) -> tuple[np.ndarray, float]:
        """Compute the centre of one cyclic-ring object, respecting ``center_method``.

        Parameters
        ----------
        object_motl : Motl
            Particles belonging to one affiliation group.

        Returns
        -------
        center : numpy.ndarray, shape (3,)
        radius : float
            Fitted circle radius; zero for barycentric or degenerate inputs.

        Notes
        -----
        Falls back to :func:`geom.barycenter` when the circle fit fails,
        emitting a :class:`UserWarning` with the object identifier and reason.
        """
        coords = object_motl.get_coordinates()

        if coords.shape[0] == 0:
            return np.zeros(3), 0.0

        if self.center_method == "barycentric":
            return geom.barycenter(coords), 0.0

        if coords.shape[0] == 1:
            return geom.barycenter(coords), 0.0

        if coords.shape[0] <= 3:
            vector_x = np.asarray([-1.0, 0.0, 0.0])
            try:
                rot = object_motl.get_rotations()
                rot_vec = rot.apply(vector_x)
                end_coord = coords + rot_vec
                center, _ = geom.ray_ray_intersection_3d(starting_points=coords, ending_points=end_coord)
                return center, 0.0
            except Exception as exc:
                obj_id = (
                    object_motl.df[self.affiliation_column].iloc[0]
                    if self.affiliation_column in object_motl.df.columns
                    else "?"
                )
                warnings.warn(
                    f"{type(self).__name__}: ray-ray intersection failed for object {obj_id!r} "
                    f"({exc}); falling back to barycentric centre.  "
                    "Consider center_method='barycentric'.",
                    stacklevel=3,
                )
                return geom.barycenter(coords), 0.0

        caught: list = []
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                center, radius, _ = geom.fit_circle_3d_pratt(coords)
        except Exception as exc:
            obj_id = (
                object_motl.df[self.affiliation_column].iloc[0]
                if self.affiliation_column in object_motl.df.columns
                else "?"
            )
            warnings.warn(
                f"{type(self).__name__}: circle fit failed for object {obj_id!r} "
                f"({exc}); falling back to barycentric centre.  "
                "Consider center_method='barycentric'.",
                stacklevel=3,
            )
            return geom.barycenter(coords), 0.0

        if caught:
            obj_id = (
                object_motl.df[self.affiliation_column].iloc[0]
                if self.affiliation_column in object_motl.df.columns
                else "?"
            )
            msg = "; ".join(str(w.message) for w in caught)
            warnings.warn(
                f"{type(self).__name__}: circle fit warning for object {obj_id!r} "
                f"({msg}); falling back to barycentric centre.  "
                "Consider center_method='barycentric'.",
                stacklevel=3,
            )
            return geom.barycenter(coords), 0.0

        return center, radius

    @gui_exposed(label="Get centers as motl", group="Statistics", order=10, returns="motl")
    def get_centers_as_motl(self) -> "cryomotl.Motl":
        """Return a Motl with one centre particle per object per tomogram.

        Overrides the barycentric base implementation: uses the Pratt
        circle fit (for ``center_method='circle_fit'``) with automatic
        fallback to barycentric when the fit fails.

        Returns
        -------
        Motl
            One row per (tomogram, object) pair.  ``tomo_id`` holds the
            tomogram identifier and ``object_id`` holds the affiliation value.
            All other columns are zero-filled.
        """
        central_points: list[np.ndarray] = []
        tomo_ids: list[float] = []
        object_ids: list[float] = []

        for t in self.motl.get_unique_values(self.tomo_id_column):
            tm = self.motl.get_motl_subset(column_values=[t], column_name=self.tomo_id_column, reset_index=True)
            for o in tm.get_unique_values(self.affiliation_column):
                om = tm.get_motl_subset(column_values=[o], column_name=self.affiliation_column, reset_index=True)
                center, _ = self._compute_object_center(om)
                central_points.append(center)
                tomo_ids.append(float(t))
                object_ids.append(float(o))

        out = cryomotl.Motl()
        if central_points:
            pts = np.vstack(central_points)
            out.fill(
                {
                    "x": pts[:, 0],
                    "y": pts[:, 1],
                    "z": pts[:, 2],
                    "tomo_id": np.array(tomo_ids),
                    "object_id": np.array(object_ids),
                }
            )
            out.renumber_particles()
        out.df.fillna(0.0, inplace=True)
        return out

    def _circumradius_for_group(self, coords: np.ndarray) -> float:
        """Return the circumradius for a group of coordinates.

        Tries :func:`geom.fit_circle_3d_pratt` (≥ 4 points); if the fit
        fails or returns zero, falls back to the mean distance from the
        barycenter.

        Parameters
        ----------
        coords : numpy.ndarray, shape (N, 3)
            Particle coordinates for one object.

        Returns
        -------
        float
            Circumradius in voxels; zero if *coords* is empty.
        """
        if coords.shape[0] == 0:
            return 0.0
        if coords.shape[0] >= 4:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _, radius, _ = geom.fit_circle_3d_pratt(coords)
                if radius > 0:
                    return float(radius)
            except Exception:
                pass
        center = geom.barycenter(coords)
        return float(np.mean(np.linalg.norm(coords - center, axis=1)))

    @gui_exposed(label="Circumference", group="Statistics", order=40, returns="dataframe")
    def circumference(self, *, pixel_size: float | None = None) -> pd.DataFrame:
        """Per-object circumference derived from the circumradius.

        Computes ``2 π × circumradius × pixel_size`` for each object.
        The circumradius is estimated via the Pratt circle fit (≥ 4 particles)
        or falls back to the mean particle–to–barycenter distance.

        Parameters
        ----------
        pixel_size : float or None, default None
            Ångström-per-voxel scale factor.  When ``None``, uses
            ``self.pixel_size`` (defaulting to ``1.0`` when unset).

        Returns
        -------
        pandas.DataFrame
            Columns ``tomo_id``, ``object_id``, ``circumference``.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        """
        self._require_affiliation()
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)

        coord = self.motl.get_coordinates()
        rows: list[dict] = []

        for keys, group in self.motl.df.groupby(self._ring_group_columns):
            tomo_id, object_id = keys[0], keys[1]
            coords_grp = coord[group.index.to_numpy(), :]
            r = self._circumradius_for_group(coords_grp)
            row: dict = {
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "circumference": 2.0 * np.pi * r * ps,
            }
            for extra_col, extra_val in zip(self._ring_group_columns[2:], keys[2:]):
                row[extra_col] = extra_val
            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Symmetry-derived properties
    # ------------------------------------------------------------------

    @property
    def central_angle(self) -> float:
        """Angle between adjacent subunits (360 / n degrees)."""
        return 360.0 / self.n

    @property
    def interior_angle(self) -> float:
        """Interior angle of the regular Cn polygon ((n-2)*180 / n degrees)."""
        return (self.n - 2) * 180.0 / self.n

    # ------------------------------------------------------------------
    # Cyclic subunit ordering
    # ------------------------------------------------------------------

    def _cyclic_indices_for_object(
        self,
        object_motl: "cryomotl.Motl",
        ref_direction: np.ndarray | None = None,
    ) -> tuple[list[int], float]:
        """Compute 1-based cyclic subunit indices for one object using grid-fit.

        Parameters
        ----------
        object_motl : Motl
            Particles belonging to one ring object.
        ref_direction : ndarray, optional
            Global reference direction (3-vector) projected onto the ring plane
            to define azimuth zero.  When *None* (default) the first particle
            is used as the reference, consistent with the legacy behaviour.

        Returns
        -------
        indices : list of int
            Indices, same length as ``object_motl.df``.
        max_residual : float
            Maximum deviation from an ideal grid position, in degrees.
        """
        center, _ = self._compute_object_center(object_motl)
        su_coord = object_motl.get_coordinates()
        vectors = su_coord - np.tile(center, (su_coord.shape[0], 1))
        n_su = len(vectors)

        if n_su == 1:
            return [1], 0.0

        obj_id = (
            object_motl.df[self.affiliation_column].iloc[0]
            if self.affiliation_column in object_motl.df.columns
            else "?"
        )

        if n_su < 3:
            warnings.warn(
                f"{type(self).__name__}: fewer than 3 subunits for object"
                f" {obj_id!r}; ring normal is underdetermined.",
                UserWarning,
                stacklevel=3,
            )

        _, _, Vt = np.linalg.svd(vectors, full_matrices=True)
        ring_normal = Vt[-1]

        if ref_direction is not None:
            rd = np.asarray(ref_direction, dtype=float)
            proj = rd - np.dot(rd, ring_normal) * ring_normal
            norm_proj = np.linalg.norm(proj)
            ref_vec = proj / norm_proj if norm_proj > 1e-10 else vectors[0]
        else:
            ref_vec = vectors[0]

        delta = self.central_angle

        def _fit(normal: np.ndarray) -> tuple[list[int], float, int]:
            azs = np.array(
                [np.degrees(geom.vector_angular_distance_signed(ref_vec, v, normal)) % 360.0 for v in vectors]
            )
            phase = 2.0 * np.pi * (azs % delta) / delta
            theta0 = (
                delta * (np.arctan2(np.mean(np.sin(phase)), np.mean(np.cos(phase))) % (2.0 * np.pi)) / (2.0 * np.pi)
            )
            aligned = (azs - theta0) % 360.0
            ks = np.array(
                [
                    int(decimal.Decimal(str(a / delta)).to_integral_value(rounding=decimal.ROUND_HALF_UP))
                    for a in aligned
                ]
            )
            indices = [(int(k) % self.n) + 1 for k in ks]
            max_res = float(np.max(np.abs(aligned - ks * delta)))
            n_dupes = sum(v - 1 for v in Counter(indices).values())
            return indices, max_res, n_dupes

        idx_pos, res_pos, dup_pos = _fit(ring_normal)
        idx_neg, res_neg, dup_neg = _fit(-ring_normal)

        if dup_pos < dup_neg or (dup_pos == dup_neg and res_pos <= res_neg):
            s_idx, max_residual = idx_pos, res_pos
        else:
            s_idx, max_residual = idx_neg, res_neg

        return s_idx, max_residual

    @gui_exposed(label="Assign subunit order", group="Affiliation", order=30, returns="none")
    def assign_subunit_order(self, ref_direction: np.ndarray | None = None) -> dict:
        """Assign 1-based cyclic subunit indices into ``self.order_column``.

        For every ring group (as defined by ``_ring_group_columns``): fits the
        angular grid to all particles and assigns each one the nearest grid
        position (1 … n), choosing the chirality that minimises duplicates then
        residual.  Writes results into ``self.motl.df[self.order_column]``.

        Parameters
        ----------
        ref_direction : ndarray, optional
            Global 3-vector that defines azimuth zero after projection onto each
            ring plane.  When supplied, all rings use the same external reference
            so cross-ring index 1 points in a consistent direction.  When *None*
            (default) the first particle in each ring group sets the reference.

        Notes
        -----
        Object centres are determined by :meth:`_compute_object_center`, which
        respects ``self.center_method``.  Modifies ``self.motl.df`` in place.
        The ``@gui_exposed(returns="none")`` decorator means the GUI ignores the
        return value; Python callers can use it to filter distorted rings.

        Returns
        -------
        dict
            Mapping of group key tuples ``(tomo_id, affiliation_id)`` to the
            per-object maximum grid residual in degrees.
        """
        residuals: dict = {}
        for keys, group in self.motl.df.groupby(self._ring_group_columns):
            om = cryomotl.Motl(group.reset_index(drop=True))
            s_idx, max_res = self._cyclic_indices_for_object(om, ref_direction=ref_direction)
            self.motl.df.loc[group.index, self.order_column] = s_idx
            residuals[keys] = max_res
        return residuals

    def check_for_duplicates(self) -> pd.DataFrame:
        """Return ring groups whose ``order_column`` has a within-group duplicate.

        Groups by the ring-group columns (``tomo_id``, ``affiliation_column``,
        and ``_ring_column`` when present) and collects every group where any
        ``order_column`` value appears more than once.

        Returns
        -------
        pandas.DataFrame
            One row per offending group.  Columns: ``tomo_id``, ``object_id``,
            and ``ring`` (only when a ring column is part of the grouping).
            Empty when no duplicates are found.
        """
        group_cols = self._ring_group_columns
        has_ring = len(group_cols) > 2
        order_col = self.order_column
        rows: list[dict] = []
        for keys, grp in self.motl.df.groupby(group_cols):
            if grp[order_col].duplicated().any():
                if has_ring:
                    tomo_id, object_id, ring_id = keys
                    rows.append({"tomo_id": tomo_id, "object_id": object_id, "ring": ring_id})
                else:
                    tomo_id, object_id = keys
                    rows.append({"tomo_id": tomo_id, "object_id": object_id})
        return pd.DataFrame(rows)

    def renumber_subunits(
        self,
        object_id: float | int,
        mapping: dict | tuple,
        *,
        tomo_id: float | int | None = None,
        flag_column: str = "geom5",
    ) -> None:
        """Manually correct the subunit order for one object.

        Applies *mapping* to ``order_column`` values across all rings of the
        object, validates the result, then clears the ambiguity flag so the
        object re-enters the per-subunit analyses.

        Parameters
        ----------
        object_id : int or float
            Affiliation ID of the object to correct.
        mapping : dict or tuple of (old_values, new_values)
            Maps old ``order_column`` value → new value.  Accept a ``dict``
            ``{old: new, …}`` or a pair of equal-length sequences
            ``([old, …], [new, …])``.  Values not in the mapping are unchanged.
        tomo_id : int, float, or None, optional
            Required when *object_id* appears in more than one tomogram.
        flag_column : str, default ``'geom5'``
            Column cleared (set to ``0.0``) on success.

        Raises
        ------
        ValueError
            If *mapping* is not injective (duplicate target values).
        ValueError
            If *object_id* is ambiguous and *tomo_id* is not given.
        ValueError
            If the remapped values would create within-ring duplicates.
        """
        if isinstance(mapping, dict):
            map_dict = {float(k): float(v) for k, v in mapping.items()}
        else:
            old_seq, new_seq = mapping
            if len(old_seq) != len(new_seq):
                raise ValueError("renumber_subunits: old and new sequences must have equal length.")
            map_dict = {float(k): float(v) for k, v in zip(old_seq, new_seq)}

        if len(set(map_dict.values())) != len(map_dict):
            raise ValueError("renumber_subunits: mapping is not injective (duplicate target values).")

        aff_col = self.affiliation_column
        tomo_col = self.tomo_id_column
        order_col = self.order_column
        df = self.motl.df
        obj_id_f = float(object_id)

        if tomo_id is None:
            unique_tomos = df.loc[df[aff_col] == obj_id_f, tomo_col].unique()
            if len(unique_tomos) > 1:
                raise ValueError(
                    f"renumber_subunits: object_id {object_id!r} appears in "
                    f"tomograms {list(unique_tomos)} — provide tomo_id."
                )
            tomo_id_f = float(unique_tomos[0]) if len(unique_tomos) == 1 else None
        else:
            tomo_id_f = float(tomo_id)

        mask = df[aff_col] == obj_id_f
        if tomo_id_f is not None:
            mask &= df[tomo_col] == tomo_id_f
        rows_pos = np.where(mask)[0]
        if len(rows_pos) == 0:
            return

        # Apply mapping to a working copy and validate.
        cur_order = df.iloc[rows_pos][order_col].values.astype(float)
        new_order = cur_order.copy()
        for i, cv in enumerate(cur_order):
            if cv in map_dict:
                new_order[i] = map_dict[cv]

        tmp = df.iloc[rows_pos].copy().reset_index(drop=True)
        tmp[order_col] = new_order
        for _, grp in tmp.groupby(self._ring_group_columns):
            if grp[order_col].duplicated().any():
                dups = sorted(grp[order_col][grp[order_col].duplicated(keep=False)].unique().tolist())
                raise ValueError(
                    f"renumber_subunits: mapping creates duplicate order values "
                    f"{dups!r} in one ring — choose a different mapping."
                )

        # Write positionally.
        order_col_loc = df.columns.get_loc(order_col)
        df.iloc[rows_pos, order_col_loc] = new_order
        if flag_column in df.columns:
            flag_col_loc = df.columns.get_loc(flag_column)
            df.iloc[rows_pos, flag_col_loc] = 0.0

    # ------------------------------------------------------------------
    # Central-angle analysis
    # ------------------------------------------------------------------

    @gui_exposed(label="Central angles", group="Geometry", order=35, returns="dataframe")
    def central_angles(
        self,
        gaps: Literal["holey", "full"] = "holey",
        inward_axis: np.ndarray | None = None,
        include_flagged: bool = False,
        flag_column: str = "geom5",
    ) -> pd.DataFrame:
        """Compute central angles between neighbouring subunit pairs.

        For each ring group two methods are evaluated for every consecutive
        pair (query particle -> nearest in ring order):

        * **Positional**: ``|atan2(n * (u x v), u * v)|`` where
          ``u = S1 - C`` and ``v = S2 - C``, with *C* the fitted ring centre
          and *n* the ring normal from SVD of displacement vectors.
        * **Orientational**: same formula applied to the inward-axis
          vectors obtained by rotating *inward_axis* by each subunit's stored
          orientation.  Does not depend on the accuracy of the circle fit for
          the centre position.

        All reported angle columns are **non-negative** (absolute values).
        The underlying signed values are retained in ``angle_pos_signed`` and
        ``angle_ori_signed`` for callers that need directionality (e.g. to
        detect CW/CCW consistency across a ring).

        Call :meth:`assign_subunit_order` before this method to populate
        ``self.order_column``.

        Parameters
        ----------
        gaps : {'holey', 'full'}, default='holey'
            Pairing mode.  ``'holey'`` pairs every present subunit with the
            next present one (by sorted position); ``'full'`` skips pairs
            whose target index (order + 1 mod n) is absent from the ring.
        inward_axis : array-like of shape (3,) or None, default=None
            Local axis pointing toward the ring centre in the particle frame.
            ``None`` uses ``[-1, 0, 0]``, the convention in
            :meth:`_compute_object_center` and :meth:`assign_subunit_order`.
        include_flagged : bool, default=False
            When ``False``, pores with *flag_column* == 1.0 are excluded from
            the output.
        flag_column : str, default ``'geom5'``
            Column checked for the per-pore ambiguity flag.

        Returns
        -------
        pandas.DataFrame
            One row per neighbouring pair with columns:

            ``tomo_id``, ``object_id``, ``qp_subtomo_id``,
            ``nn_subtomo_id``, ``qp_idx``, ``nn_idx``,
            ``idx_diff``,
            ``angle_pos``, ``angle_ori``, ``angle_diff``,
            ``angle_pos_per_pos``, ``angle_ori_per_pos``,
            ``dev_pos``, ``dev_ori``,
            ``angle_pos_signed``, ``angle_ori_signed``

            All *angle_** and *dev_** columns are in **degrees**.
            ``angle_pos`` and ``angle_ori`` are non-negative (magnitudes).
            ``angle_diff = |angle_pos - angle_ori|`` (unsigned disagreement).
            ``dev_pos`` and ``dev_ori`` are signed deviations from the ideal
            ``idx_diff x central_angle`` (positive = wider than ideal,
            negative = narrower).
            ``angle_pos_signed`` / ``angle_ori_signed`` retain the sign from
            ``atan2``; their sign depends on the SVD normal orientation and is
            arbitrary between rings -- use only for within-ring consistency
            checks, never for cross-ring comparison.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        ValueError
            If ``order_column`` is absent from ``self.motl.df``.
        ValueError
            If a ring group has duplicate ``order_column`` values.
        """
        self._require_affiliation()
        if self.order_column not in self.motl.df.columns:
            raise ValueError(f"order_column {self.order_column!r} not in motl; " "call assign_subunit_order() first.")

        ia = np.array([-1.0, 0.0, 0.0]) if inward_axis is None else np.asarray(inward_axis, dtype=float)
        norm = np.linalg.norm(ia)
        if norm > 1e-10:
            ia = ia / norm

        rows: list[dict] = []
        df = self.motl.df
        if not include_flagged and flag_column in df.columns:
            _flagged = set(
                map(tuple, df.loc[df[flag_column] == 1.0, [self.tomo_id_column, self.affiliation_column]].values.tolist())
            )
            if _flagged:
                _keep = ~df[[self.tomo_id_column, self.affiliation_column]].apply(tuple, axis=1).isin(_flagged)
                df = df[_keep]

        for keys, group in df.groupby(self._ring_group_columns):
            tomo_id = keys[0]
            object_id = keys[1]

            if len(group) < 2:
                continue

            grp = group.sort_values(self.order_column)
            order_vals = grp[self.order_column].to_numpy(dtype=float)

            u_vals, u_counts = np.unique(order_vals, return_counts=True)
            if np.any(u_counts > 1):
                dupes = sorted(u_vals[u_counts > 1].tolist())
                raise ValueError(
                    f"Tomo {tomo_id!r}, object {object_id!r}: duplicate "
                    f"'{self.order_column}' values {dupes}. "
                    "Ordered pairing is ambiguous with duplicate order values."
                )

            coords = grp[["x", "y", "z"]].to_numpy(dtype=float) + grp[["shift_x", "shift_y", "shift_z"]].to_numpy(
                dtype=float
            )
            angles_euler = grp[["phi", "theta", "psi"]].to_numpy(dtype=float)
            subtomo_ids = grp["subtomo_id"].to_numpy(dtype=float)
            n_present = len(grp)

            om = cryomotl.Motl(grp.reset_index(drop=True))
            center, _ = self._compute_object_center(om)
            vectors = coords - center[np.newaxis, :]
            _, _, Vt = np.linalg.svd(vectors, full_matrices=True)
            ring_normal = Vt[-1]

            if gaps == "holey":
                qi = np.arange(n_present, dtype=np.intp)
                ni = (qi + 1) % n_present
            else:
                min_o = order_vals.min()
                o_norm = order_vals - min_o
                targets = ((o_norm + 1) % self.n) + min_o
                order_to_pos = {float(o): int(i) for i, o in enumerate(order_vals)}
                valid = np.array([float(t) in order_to_pos for t in targets])
                qi = np.where(valid)[0].astype(np.intp)
                if len(qi) == 0:
                    continue
                ni = np.array([order_to_pos[float(targets[j])] for j in qi], dtype=np.intp)

            u_vecs = coords[qi] - center
            v_vecs = coords[ni] - center
            cross_pos = np.cross(u_vecs, v_vecs)
            signed_pos = np.degrees(
                np.arctan2(
                    np.einsum("ij,j->i", cross_pos, ring_normal),
                    np.einsum("ij,ij->i", u_vecs, v_vecs),
                )
            )
            angle_pos = np.abs(signed_pos)

            rots_qi = srot.from_euler("zxz", angles_euler[qi], degrees=True)
            rots_ni = srot.from_euler("zxz", angles_euler[ni], degrees=True)
            qp_inward = rots_qi.apply(ia)
            nn_inward = rots_ni.apply(ia)
            cross_ori = np.cross(qp_inward, nn_inward)
            signed_ori = np.degrees(
                np.arctan2(
                    np.einsum("ij,j->i", cross_ori, ring_normal),
                    np.einsum("ij,ij->i", qp_inward, nn_inward),
                )
            )
            angle_ori = np.abs(signed_ori)

            raw_diff = order_vals[ni] - order_vals[qi]
            idx_diff = np.where(raw_diff > 0, raw_diff, raw_diff + self.n)
            ideal = idx_diff * self.central_angle
            angle_pos_per_pos = angle_pos / idx_diff
            angle_ori_per_pos = angle_ori / idx_diff
            dev_pos = angle_pos - ideal
            dev_ori = angle_ori - ideal

            for k in range(len(qi)):
                rows.append(
                    {
                        "tomo_id": float(tomo_id),
                        "object_id": float(object_id),
                        "qp_subtomo_id": subtomo_ids[qi[k]],
                        "nn_subtomo_id": subtomo_ids[ni[k]],
                        "qp_idx": order_vals[qi[k]],
                        "nn_idx": order_vals[ni[k]],
                        "idx_diff": float(idx_diff[k]),
                        "angle_pos": float(angle_pos[k]),
                        "angle_ori": float(angle_ori[k]),
                        "angle_diff": float(abs(angle_pos[k] - angle_ori[k])),
                        "angle_pos_per_pos": float(angle_pos_per_pos[k]),
                        "angle_ori_per_pos": float(angle_ori_per_pos[k]),
                        "dev_pos": float(dev_pos[k]),
                        "dev_ori": float(dev_ori[k]),
                        "angle_pos_signed": float(signed_pos[k]),
                        "angle_ori_signed": float(signed_ori[k]),
                    }
                )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Per-object evaluations
    # ------------------------------------------------------------------

    def diameter(
        self,
        *,
        pixel_size: float = 1.0,
        store_column: MotlColumn = "geom4",
    ) -> tuple[pd.DataFrame, "cryomotl.Motl"]:
        """Compute the mean diameter for each object.

        For even *n* with ``order_column`` present, opposite-subunit pairs
        ``(i, i + n//2)`` are matched (1-based, matching
        :meth:`assign_subunit_order`'s convention).  For odd *n* or when
        ``order_column`` is absent, the diameter is derived from the
        circumradius (``2 × circumradius × pixel_size``).

        Parameters
        ----------
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor applied to all distances.
        store_column : MotlColumn, default='geom4'
            Column in the returned *motl_out* that carries each object's
            mean diameter on every row; ``NaN`` for objects with no result.

        Returns
        -------
        summary_df : pandas.DataFrame
            One row per ``(tomo_id, object_id)`` with columns
            ``tomo_id``, ``object_id``, ``mean_diameter``, ``n_pairs``.
            ``n_pairs`` is 0 for the circumradius fallback.
        motl_out : Motl
            Copy of ``self.motl`` with *store_column* populated.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.

        Warns
        -----
        UserWarning
            When the circumradius fallback is used (odd *n*, missing
            ``order_column``, or even *n* but no pairs could be matched).
        """
        self._require_affiliation()

        motl_out = cryomotl.Motl(self.motl.df.copy())
        motl_out.df.reset_index(drop=True, inplace=True)

        has_order = self.order_column in motl_out.df.columns
        even_n = self.n % 2 == 0
        use_pairs = even_n and has_order

        if not has_order and even_n:
            warnings.warn(
                f"{type(self).__name__}.diameter: order column {self.order_column!r} not in motl; "
                "diameter derived from 2 × circumradius for all objects.",
                stacklevel=2,
            )
        elif not even_n:
            warnings.warn(
                f"{type(self).__name__}.diameter: n={self.n} is odd — no exact opposite subunit; "
                "diameter derived from 2 × circumradius for all objects.",
                stacklevel=2,
            )

        coord = motl_out.get_coordinates()
        diameters_col = np.full(len(motl_out.df), np.nan)
        rows: list[dict] = []

        for keys, group in motl_out.df.groupby(self._ring_group_columns):
            tomo_id, object_id = keys[0], keys[1]
            grp_idx = group.index.to_numpy()
            coords_grp = coord[grp_idx, :]
            mean_d: float
            n_pairs: int

            if use_pairs:
                half = self.n // 2
                pair_rows: list[list[int]] = []
                for i in range(1, half + 1):
                    j = i + half
                    mask_i = group[self.order_column] == i
                    mask_j = group[self.order_column] == j
                    if mask_i.any() and mask_j.any():
                        pair_rows.append(
                            [
                                group.index[mask_i][0],
                                group.index[mask_j][0],
                            ]
                        )

                if pair_rows:
                    idx = np.asarray(pair_rows)
                    dists = geom.point_pairwise_dist(coord[idx[:, 0], :], coord[idx[:, 1], :]) * pixel_size
                    mean_d = float(np.mean(dists))
                    n_pairs = int(len(dists))
                else:
                    warnings.warn(
                        f"{type(self).__name__}.diameter: no opposite-pair matches for object "
                        f"{object_id!r} in tomo {tomo_id!r} (even n={self.n} but no "
                        "paired subunit indices); diameter derived from 2 × circumradius.",
                        stacklevel=2,
                    )
                    mean_d = 2.0 * self._circumradius_for_group(coords_grp) * pixel_size
                    n_pairs = 0
            else:
                mean_d = 2.0 * self._circumradius_for_group(coords_grp) * pixel_size
                n_pairs = 0

            diameters_col[grp_idx] = mean_d
            row: dict = {
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "mean_diameter": mean_d,
                "n_pairs": n_pairs,
            }
            for extra_col, extra_val in zip(self._ring_group_columns[2:], keys[2:]):
                row[extra_col] = extra_val
            rows.append(row)

        motl_out.df[store_column] = diameters_col
        base_cols = ["tomo_id", "object_id", "mean_diameter", "n_pairs"]
        extra_cols = list(self._ring_group_columns[2:])
        summary_df = pd.DataFrame(
            rows if rows else [],
            columns=base_cols + extra_cols,
        )
        return summary_df, motl_out

    @gui_exposed(label="Get object stats", group="Statistics", order=50, returns="dataframe")
    def get_object_stats(self, *, pixel_size: float | None = None) -> pd.DataFrame:
        """Comprehensive per-object statistics table.

        Composes :meth:`occupancy`, :meth:`circumference`,
        :meth:`diameter`, and centre/radius computation into one row per
        ``(tomo_id, object_id)`` group.

        Parameters
        ----------
        pixel_size : float or None, default None
            Ångström-per-voxel scale factor for distance columns.  When
            ``None``, uses ``self.pixel_size`` (defaulting to ``1.0`` when
            unset).

        Returns
        -------
        pandas.DataFrame
            One row per ``(tomo_id, object_id)``.  Columns:

            ``tomo_id``, ``object_id``
                Group identifiers.
            ``n_present``, ``occupancy``, ``missing``
                From :meth:`occupancy`.
            ``x``, ``y``, ``z``
                Object centre coordinates (voxels).
            ``radius``
                Circumradius (voxels).
            ``circumference``
                From :meth:`circumference`.
            ``mean_diameter``, ``n_pairs``
                From :meth:`diameter`.
            ``mean_angle_pos``, ``std_angle_pos``, ``var_angle_pos``
                Mean, standard deviation, and variance of the per-position
                positional central angle (degrees).  Present and non-``NaN``
                only when :meth:`assign_subunit_order` has been called and
                the ring has ≥ 2 subunits.
            ``mean_angle_ori``, ``std_angle_ori``, ``var_angle_ori``
                Same statistics for the orientational central angle.
            ``n_angle_pairs``
                Number of pairs contributing to the angle statistics.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        """
        self._require_affiliation()
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)

        occ_df = self.occupancy()
        circ_df = self.circumference(pixel_size=ps)
        diam_df, _ = self.diameter(pixel_size=ps)

        coord = self.motl.get_coordinates()
        center_rows: list[dict] = []
        for (tomo_id, object_id), group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            coords_grp = coord[group.index.to_numpy(), :]
            r = self._circumradius_for_group(coords_grp)
            center = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            center_rows.append(
                {
                    "tomo_id": float(tomo_id),
                    "object_id": float(object_id),
                    "x": float(center[0]),
                    "y": float(center[1]),
                    "z": float(center[2]),
                    "radius": r,
                }
            )
        geo_df = pd.DataFrame(center_rows)

        result = occ_df.merge(geo_df, on=["tomo_id", "object_id"], how="outer")
        result = result.merge(circ_df, on=["tomo_id", "object_id"], how="left")
        result = result.merge(
            diam_df[["tomo_id", "object_id", "mean_diameter", "n_pairs"]],
            on=["tomo_id", "object_id"],
            how="left",
        )

        # Central-angle statistics — only when subunit order has been assigned
        _angle_stat_cols = [
            "mean_angle_pos",
            "std_angle_pos",
            "var_angle_pos",
            "mean_angle_ori",
            "std_angle_ori",
            "var_angle_ori",
            "n_angle_pairs",
        ]
        order_col = self.order_column
        if order_col in self.motl.df.columns and self.motl.df[order_col].gt(0).any():
            ang_df = self.central_angles()
            if len(ang_df) > 0:
                ang_agg = (
                    ang_df.groupby(["tomo_id", "object_id"])
                    .agg(
                        mean_angle_pos=("angle_pos_per_pos", "mean"),
                        std_angle_pos=("angle_pos_per_pos", "std"),
                        var_angle_pos=("angle_pos_per_pos", "var"),
                        mean_angle_ori=("angle_ori_per_pos", "mean"),
                        std_angle_ori=("angle_ori_per_pos", "std"),
                        var_angle_ori=("angle_ori_per_pos", "var"),
                        n_angle_pairs=("angle_pos_per_pos", "count"),
                    )
                    .reset_index()
                )
                result = result.merge(ang_agg, on=["tomo_id", "object_id"], how="left")
            else:
                for col in _angle_stat_cols:
                    result[col] = float("nan")

        return result

    # ------------------------------------------------------------------
    # Extra components (non-ring particles, e.g. NR basket)
    # ------------------------------------------------------------------

    def add_component(
        self,
        motl=None,
        name: str = "basket",
        *,
        assign_affiliation: bool = False,
        assign_su_order: bool = False,
        copy_su_order: bool = False,
        reference_ring: int = 0,
        affiliation_max_distance: float | None = None,
        order_max_distance: float | None = None,
    ) -> None:
        """Register a non-ring component and optionally affiliate and order it.

        The component is stored in ``self._components[name]`` and never added
        to ``self._ring_motls``, so it is invisible to all ring-fitting, merge,
        and statistics methods.

        Behaviour is driven only by explicit flags — no inference from column
        presence or absence.

        Parameters
        ----------
        motl : MotlSource
            Component particle list.  Loaded with :meth:`cryomotl.Motl.load`.
            ``None`` raises :class:`ValueError`.
        name : str, default ``'basket'``
            Key under which the component is stored in ``self._components``.
        assign_affiliation : bool, default False
            When ``True``, run NN-based affiliation against the reference ring
            within each ``tomo_id``, overwriting any existing values.
            When ``False``, the affiliation column is left untouched.
        assign_su_order : bool, default False
            When ``True``, run chain ordering, overwriting any existing values.
            Mutually exclusive with ``copy_su_order``.
        copy_su_order : bool, default False
            When ``True``, copy ``order_column`` from the nearest reference-ring
            particle within the same ``(tomo_id, affiliation_column)`` pore.
            Requires affiliation to be set (either pre-existing or via
            ``assign_affiliation=True``).  Mutually exclusive with
            ``assign_su_order``.
        reference_ring : int, default 0
            0-based ring index used as the affiliation/copy reference.  For
            NPC: 0=CR, 1=IR, 2=NR, 3=gp210.  A warning is printed (once per
            call) when the default (0) is used and affiliation NN or copy is
            performed.
        affiliation_max_distance : float or None
            Maximum voxel distance for affiliation NN assignment.  Required
            when ``assign_affiliation=True``.
        order_max_distance : float or None
            Absolute chain-link cutoff (voxels) for ordering or copying.
            Required when ``assign_su_order=True`` or ``copy_su_order=True``.
        """
        if assign_su_order and copy_su_order:
            raise ValueError(
                "add_component: assign_su_order and copy_su_order are mutually exclusive."
            )
        if motl is None:
            raise ValueError("add_component: motl must not be None — nothing to add.")
        if isinstance(motl, cryomotl.Motl):
            comp = copy.deepcopy(motl)
        else:
            comp = cryomotl.Motl.load(motl)
        comp.df.reset_index(drop=True, inplace=True)

        aff_col = self.affiliation_column
        order_col = self.order_column
        tomo_col = self.tomo_id_column

        _warned_ref_ring = False

        def _ref_df():
            rings_merged = getattr(self, "_rings_merged", False)
            ring_col = getattr(self, "_ring_column", None)
            if rings_merged and ring_col is not None:
                return self.motl.df[self.motl.df[ring_col] == float(reference_ring + 1)]
            return self.motl.df

        # ---- Affiliation -----------------------------------------------
        if assign_affiliation:
            if affiliation_max_distance is None:
                raise ValueError(
                    "add_component: assign_affiliation=True requires affiliation_max_distance."
                )
            ref_df = _ref_df()
            if ref_df.empty:
                raise ValueError(
                    f"add_component: reference ring {reference_ring} has no particles. "
                    "Call merge() first or choose a different reference_ring."
                )
            if reference_ring == 0:
                print(
                    "add_component: no reference_ring specified; using ring 0 (the first ring). "
                    "Set reference_ring= to suppress this warning."
                )
                _warned_ref_ring = True
            if aff_col not in comp.df.columns:
                comp.df[aff_col] = np.nan
            all_comp_coords = comp.get_coordinates()
            all_ref_coords = self.motl.get_coordinates()
            for tomo_id, comp_tomo_df in comp.df.groupby(tomo_col):
                ref_tomo_df = ref_df[ref_df[tomo_col] == tomo_id]
                if ref_tomo_df.empty:
                    continue
                comp_idx = comp_tomo_df.index.to_numpy()
                ref_idx = ref_tomo_df.index.to_numpy()
                comp_coords = all_comp_coords[comp_idx, :]
                ref_coords = all_ref_coords[ref_idx, :]
                ref_aff = ref_tomo_df[aff_col].to_numpy()
                _, nn_idx, nn_dist, _ = nnana.find_nn_indices(comp_coords, ref_coords, k=1)
                nn_idx = nn_idx.reshape(-1)
                nn_dist = nn_dist.reshape(-1)
                within = nn_dist <= affiliation_max_distance
                comp.df.loc[comp_idx[within], aff_col] = ref_aff[nn_idx[within]]
                n_unaffiliated = int((~within).sum())
                if n_unaffiliated:
                    print(
                        f"add_component: {n_unaffiliated} particle(s) in tomo {tomo_id} beyond "
                        f"affiliation_max_distance={affiliation_max_distance} — left unaffiliated (NaN)."
                    )

        # ---- Order -----------------------------------------------------
        if assign_su_order:
            if order_max_distance is None:
                raise ValueError(
                    "add_component: assign_su_order=True requires order_max_distance."
                )
            if order_col not in comp.df.columns:
                comp.df[order_col] = 0.0
            n = self.n_subunits
            _chain_fn = getattr(self, "_chain_order_for_ring_object", None)
            for (_t, _o), grp_df in comp.df.groupby([tomo_col, aff_col]):
                pos = grp_df.index.to_numpy()
                if len(pos) < 2:
                    continue
                obj_motl = cryomotl.Motl(grp_df.reset_index(drop=True))
                coords = obj_motl.get_coordinates()
                if _chain_fn is not None:
                    center = geom.barycenter(coords)
                    r_comp = float(np.mean(np.linalg.norm(coords[:, :2] - center[:2], axis=1)))
                    shift_size = r_comp * np.sin(np.pi / n)
                    ev = [0.0, -shift_size, 0.0]
                    xv = [0.0, +shift_size, 0.0]
                    d_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                    np.fill_diagonal(d_mat, np.inf)
                    s_ring = float(np.median(d_mat.min(axis=1)))
                    factor = order_max_distance / max(s_ring, 1e-6)
                    order_arr, _ = _chain_fn(
                        obj_motl,
                        entry_shift_vec=ev,
                        exit_shift_vec=xv,
                        max_distance_factor=factor,
                        ambiguity_tol=self.central_angle / 4.0,
                        ref_direction=None,
                    )
                else:
                    s_idx, _ = self._cyclic_indices_for_object(obj_motl, ref_direction=None)
                    order_arr = np.array(s_idx, dtype=float)
                comp.df.iloc[pos, comp.df.columns.get_loc(order_col)] = order_arr

        elif copy_su_order:
            if aff_col not in comp.df.columns or comp.df[aff_col].isna().all():
                raise ValueError(
                    f"add_component: copy_su_order=True requires '{aff_col}' to be set. "
                    "Run with assign_affiliation=True first, or ensure the column is populated."
                )
            ref_df = _ref_df()
            if reference_ring == 0 and not _warned_ref_ring:
                print(
                    "add_component: no reference_ring specified; using ring 0 (the first ring). "
                    "Set reference_ring= to suppress this warning."
                )
            if order_col not in comp.df.columns:
                comp.df[order_col] = 0.0
            all_comp_coords = comp.get_coordinates()
            all_ref_coords = self.motl.get_coordinates()
            for (tomo_id, pore_id), comp_pore_df in comp.df.groupby([tomo_col, aff_col]):
                ref_pore_df = ref_df[
                    (ref_df[tomo_col] == tomo_id) & (ref_df[aff_col] == pore_id)
                ]
                if ref_pore_df.empty:
                    continue
                comp_idx = comp_pore_df.index.to_numpy()
                ref_idx = ref_pore_df.index.to_numpy()
                comp_coords = all_comp_coords[comp_idx, :]
                ref_coords = all_ref_coords[ref_idx, :]
                ref_order = ref_pore_df[order_col].to_numpy()
                _, nn_idx, _, _ = nnana.find_nn_indices(comp_coords, ref_coords, k=1)
                nn_idx = nn_idx.reshape(-1)
                comp.df.loc[comp_idx, order_col] = ref_order[nn_idx]

        # ---- Store -------------------------------------------------------
        self._components[name] = {"motl": comp, "reference_ring": reference_ring}

        # ---- Diagnostic -------------------------------------------------
        n_total = len(comp.df)
        if aff_col in comp.df.columns:
            n_affiliated = int(comp.df[aff_col].notna().sum())
            n_unaffiliated = n_total - n_affiliated
            ring_col = getattr(self, "_ring_column", None)
            rings_merged = getattr(self, "_rings_merged", False)
            if rings_merged and ring_col is not None:
                ref_aff_vals = set(
                    self.motl.df.loc[
                        self.motl.df[ring_col] == float(reference_ring + 1), aff_col
                    ].dropna()
                )
            else:
                ref_aff_vals = set(self.motl.df[aff_col].dropna())
            comp_aff_vals = set(comp.df[aff_col].dropna())
            subset_flag = comp_aff_vals <= ref_aff_vals
            print(
                f"add_component '{name}': {n_total} particles, "
                f"affiliated={n_affiliated}, unaffiliated={n_unaffiliated}, "
                f"{aff_col} subset of ref-ring {reference_ring}: {subset_flag}"
            )
        else:
            print(
                f"add_component '{name}': {n_total} particles, "
                f"no '{aff_col}' column present."
            )

    def get_component_stats(
        self,
        name: str = "basket",
        *,
        pixel_size: float | None = None,
        degrees: bool = True,
        gaps: str = "full",
    ) -> "pd.DataFrame":
        """Per-particle geometry of a registered component relative to its reference ring.

        All geometry is measured against the reference ring's centre and pore
        axis — never a component-fitted centre.  Positions are scaled by
        ``self.pixel_size``.

        Parameters
        ----------
        name : str, default ``'basket'``
            Key used in :meth:`add_component`.
        degrees : bool, default True
            Return angles in degrees when ``True``, radians otherwise.
        gaps : ``'full'`` or ``'holey'``, default ``'full'``
            Passed to :meth:`nnana.NearestNeighbors.ordered_pairs`.  Ignored
            when the component has no ``order_column``.

        Returns
        -------
        pandas.DataFrame
            One row per component particle.  Columns: ``tomo_id``,
            ``affiliation_column``, ``subtomo_id``, ``radius``,
            ``axial_offset``, ``distance_to_centre``, ``tilt_angle``.
            ``order_column``, ``central_angle_prev``, ``central_angle_next``,
            and ``interior_angle`` are added when the component carries an
            ``order_column``.
        """
        if name not in self._components:
            raise ValueError(
                f"get_component_stats: no component named '{name}'. "
                "Call add_component first."
            )
        comp_entry = self._components[name]
        comp = comp_entry["motl"]
        reference_ring = comp_entry["reference_ring"]

        aff_col = self.affiliation_column
        order_col = self.order_column
        tomo_col = self.tomo_id_column
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)

        rings_merged = getattr(self, "_rings_merged", False)
        ring_col = getattr(self, "_ring_column", None)
        if rings_merged and ring_col is not None:
            ref_df = self.motl.df[self.motl.df[ring_col] == float(reference_ring + 1)]
        else:
            ref_df = self.motl.df

        all_comp_coords = comp.get_coordinates()
        has_order = order_col in comp.df.columns

        rows: list[dict] = []

        for (tomo_id, pore_id), comp_pore_df in comp.df.groupby([tomo_col, aff_col]):
            ref_pore_df = ref_df[
                (ref_df[tomo_col] == tomo_id) & (ref_df[aff_col] == pore_id)
            ]

            if ref_pore_df.empty:
                for _, row_s in comp_pore_df.iterrows():
                    row: dict = {
                        tomo_col: float(tomo_id),
                        aff_col: float(pore_id),
                        "subtomo_id": float(row_s["subtomo_id"]),
                        "radius": float("nan"),
                        "axial_offset": float("nan"),
                        "distance_to_centre": float("nan"),
                        "tilt_angle": float("nan"),
                    }
                    if has_order:
                        row[order_col] = float(row_s[order_col])
                        row["central_angle_prev"] = float("nan")
                        row["central_angle_next"] = float("nan")
                        row["interior_angle"] = float("nan")
                        row["cone_to_ref"] = float("nan")
                        row["inplane_to_ref"] = float("nan")
                    rows.append(row)
                continue

            ref_pore_motl = cryomotl.Motl(ref_pore_df.reset_index(drop=True))
            centre, _ = self._compute_object_center(ref_pore_motl)
            angles_ref = ref_pore_motl.get_angles()
            z_vecs_ref = geom.euler_angles_to_normals(angles_ref)
            mean_normal = z_vecs_ref.mean(axis=0)
            n_norm = float(np.linalg.norm(mean_normal))
            pore_axis = mean_normal / n_norm if n_norm > 1e-10 else mean_normal

            ref_order_to_angles: dict[float, np.ndarray] = {}
            if has_order and order_col in ref_pore_df.columns:
                for loc_i, (_, ref_row_s) in enumerate(ref_pore_df.iterrows()):
                    ref_order_to_angles[float(ref_row_s[order_col])] = angles_ref[loc_i]

            comp_idx = comp_pore_df.index.to_numpy()
            pore_coords = all_comp_coords[comp_idx, :]

            pore_motl_tmp = cryomotl.Motl(comp_pore_df.reset_index(drop=True))
            angles_comp = pore_motl_tmp.get_angles()
            z_vecs_comp = geom.euler_angles_to_normals(angles_comp)

            n_pts = len(comp_idx)
            next_local_map: dict[int, int] = {}
            prev_local_map: dict[int, int] = {}
            if has_order and n_pts >= 2:
                op = nnana.NearestNeighbors.ordered_pairs(
                    pore_motl_tmp,
                    tomo_col,
                    order_col,
                    topology="circular",
                    ring_size=self.n_subunits,
                    gaps=gaps,
                )
                sids = comp_pore_df["subtomo_id"].to_numpy()
                sid_to_local = {float(s): i for i, s in enumerate(sids)}
                for _, pr in op.df.iterrows():
                    qi = sid_to_local.get(float(pr["qp_subtomo_id"]))
                    ni = sid_to_local.get(float(pr["nn_subtomo_id"]))
                    if qi is not None and ni is not None:
                        next_local_map[qi] = ni
                        prev_local_map[ni] = qi

            for local_i, (global_i, row_s) in enumerate(comp_pore_df.iterrows()):
                v = pore_coords[local_i] - centre
                axial_comp = float(np.dot(v, pore_axis))
                radial = v - axial_comp * pore_axis
                radius = float(np.linalg.norm(radial)) * ps
                axial_offset = axial_comp * ps
                dist3d = float(np.linalg.norm(v)) * ps

                tilt = float(geom.vector_angular_distance(z_vecs_comp[local_i], pore_axis))
                if not degrees:
                    tilt = np.radians(tilt)

                row = {
                    tomo_col: float(tomo_id),
                    aff_col: float(pore_id),
                    "subtomo_id": float(row_s["subtomo_id"]),
                    "radius": radius,
                    "axial_offset": axial_offset,
                    "distance_to_centre": dist3d,
                    "tilt_angle": tilt,
                }

                if has_order:
                    row[order_col] = float(row_s[order_col])
                    ca_prev = float("nan")
                    ca_next = float("nan")
                    interior = float("nan")

                    if n_pts >= 2:
                        pi_ = prev_local_map.get(local_i)
                        ni_ = next_local_map.get(local_i)

                        v_self = pore_coords[local_i] - centre
                        vn_s = float(np.linalg.norm(v_self))

                        if pi_ is not None:
                            v_prev = pore_coords[pi_] - centre
                            vn_p = float(np.linalg.norm(v_prev))
                            if vn_s > 1e-10 and vn_p > 1e-10:
                                ca_prev = float(
                                    geom.vector_angular_distance(v_self / vn_s, v_prev / vn_p)
                                )
                                if not degrees:
                                    ca_prev = np.radians(ca_prev)

                        if ni_ is not None:
                            v_next = pore_coords[ni_] - centre
                            vn_n = float(np.linalg.norm(v_next))
                            if vn_s > 1e-10 and vn_n > 1e-10:
                                ca_next = float(
                                    geom.vector_angular_distance(v_self / vn_s, v_next / vn_n)
                                )
                                if not degrees:
                                    ca_next = np.radians(ca_next)

                        if pi_ is not None and ni_ is not None:
                            a_vec = pore_coords[pi_] - pore_coords[local_i]
                            b_vec = pore_coords[ni_] - pore_coords[local_i]
                            na = float(np.linalg.norm(a_vec))
                            nb = float(np.linalg.norm(b_vec))
                            if na > 1e-10 and nb > 1e-10:
                                interior = float(
                                    geom.vector_angular_distance(a_vec / na, b_vec / nb)
                                )
                                if not degrees:
                                    interior = np.radians(interior)

                    row["central_angle_prev"] = ca_prev
                    row["central_angle_next"] = ca_next
                    row["interior_angle"] = interior

                    ord_val = float(row_s[order_col])
                    ref_a = ref_order_to_angles.get(ord_val)
                    if ref_a is not None:
                        cone_val, inplane_val = geom.cone_inplane_distance(
                            angles_comp[local_i], ref_a, degrees=degrees, cyclic_symmetry=1
                        )
                        row["cone_to_ref"] = float(np.asarray(cone_val).flat[0])
                        row["inplane_to_ref"] = float(np.asarray(inplane_val).flat[0])
                    else:
                        row["cone_to_ref"] = float("nan")
                        row["inplane_to_ref"] = float("nan")

                rows.append(row)

        return pd.DataFrame(rows)


# =============================================================================
# Free geometry functions over ring centres
# =============================================================================


def _ring_centre_spacing(
    c0: np.ndarray,
    c1: np.ndarray,
    *,
    axis: np.ndarray | None = None,
) -> float:
    """Spacing between two ring centres.

    Parameters
    ----------
    c0, c1 : np.ndarray, shape (3,)
        Mean positions of the two ring centres.
    axis : np.ndarray, shape (3,), optional
        Unit vector.  When given, returns ``|dot(c1 − c0, axis)|`` (axial
        projection, used by DnComplex).  When absent, returns the Euclidean
        distance (used by NPC).
    """
    diff = c1 - c0
    if axis is not None:
        return float(abs(np.dot(diff, axis)))
    return float(np.linalg.norm(diff))


def _ring_centre_twist(
    positions0: np.ndarray,
    positions1: np.ndarray,
    n: int,
    axis: np.ndarray,
    *,
    degrees: bool = True,
) -> float:
    """Signed n-fold rotational twist between two rings about *axis*.

    Projects each ring's subunit positions (relative to a shared barycentre)
    onto the plane perpendicular to *axis*, computes the n-fold circular mean
    phase for each ring, and returns the signed phase difference folded into
    ``[−π/n, π/n)`` (i.e. ``[−180/n, 180/n)`` in degrees).  A value of 0
    means the rings are rotationally aligned; positive values mean ring 1 is
    rotated in the positive (right-hand) sense relative to ring 0 about
    *axis*.

    Parameters
    ----------
    positions0, positions1 : np.ndarray, shape (k, 3)
        Subunit positions relative to the shared barycentre.
    n : int
        Fold symmetry for the circular mean.
    axis : np.ndarray, shape (3,)
        Unit rotation axis.
    degrees : bool, default True
        Return angle in degrees when ``True``, radians when ``False``.
    """
    e1 = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(e1, axis)) > 0.9:
        e1 = np.array([0.0, 1.0, 0.0])
    e1 = e1 - np.dot(e1, axis) * axis
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)

    def _phase(pts: np.ndarray) -> float:
        angles = np.arctan2(pts @ e2, pts @ e1)
        z = np.sum(np.exp(1j * n * angles))
        return float(np.angle(z) / n)

    phase0 = _phase(positions0)
    phase1 = _phase(positions1)
    ca = 2.0 * np.pi / n
    twist_rad = (phase1 - phase0 + ca / 2.0) % ca - ca / 2.0
    return float(np.degrees(twist_rad)) if degrees else float(twist_rad)


# =============================================================================
# DnComplex — dihedral Dn-symmetric structures (two stacked Cn rings)
# =============================================================================


class DnComplex(CnComplex):
    """Dihedral Dn-symmetric structure modelled as two stacked Cn rings.

    Extends :class:`CnComplex` with ring-splitting, ring-aware subunit
    ordering (1 … n for the top ring, n+1 … 2n for the bottom ring), and
    inter-ring metrics (axial spacing, rotational twist).

    Parameters
    ----------
    motl : MotlSource
        Subunit particle list.
    symmetry : Symmetry
        Dihedral fold, e.g. ``"D6"`` or ``6`` (integer folds are accepted
        and treated as Dn).  Non-dihedral symmetries raise :class:`ValueError`.
    affiliation_column : MotlColumn, default='object_id'
        Column that identifies which object each particle belongs to.
    order_column : MotlColumn, default='geom1'
        Column that :meth:`assign_subunit_order` writes subunit indices into.
    tomo_id_column : MotlColumn, default='tomo_id'
        Column that identifies the tomogram.
    center_method : {'circle_fit', 'barycentric'}, default='circle_fit'
        Algorithm used by centre-computation helpers.

    Raises
    ------
    ValueError
        When *symmetry* is not a dihedral Dn group.

    Notes
    -----
    ``n_subunits`` equals ``2 * fold`` (full dihedral group size).  ``n``
    (inherited from :class:`CnComplex` via :meth:`_cyclic_setup`) equals
    ``fold`` — the per-ring subunit count.

    Ring 0 is the ring whose subunits have a *higher* mean axial coordinate
    along ``_split_axis`` (the "top" ring).  Ring 1 is the "bottom" ring.
    After :meth:`assign_subunit_order`, ring 0 subunits receive indices
    1 … n and ring 1 subunits receive indices n+1 … 2n.
    """

    def __init__(
        self,
        motl: MotlSource,
        symmetry: Symmetry,
        *,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
        center_method: Literal["circle_fit", "barycentric"] = "circle_fit",
    ) -> None:
        self._setup(
            motl,
            symmetry,
            affiliation_column=affiliation_column,
            order_column=order_column,
            tomo_id_column=tomo_id_column,
        )
        if self.group != "D":
            raise ValueError(f"DnComplex requires dihedral Dn symmetry, got {symmetry!r}.")
        self.center_method: Literal["circle_fit", "barycentric"] = center_method
        self._cyclic_setup()
        self._ring_column: MotlColumn = "geom5"
        self._split_axis: np.ndarray = np.array([0.0, 0.0, 1.0])
        self._rings_split: bool = False

    # ------------------------------------------------------------------
    # Ring splitting
    # ------------------------------------------------------------------

    @gui_exposed(label="Split rings", group="Affiliation", order=40, returns="motl")
    def split_rings(
        self,
        *,
        ring_column: MotlColumn = "geom5",
        axis: ArrayLike = (0.0, 0.0, 1.0),
    ) -> "cryomotl.Motl":
        """Partition subunits into two axial rings and label them 0 / 1.

        For each object, projects every subunit's position relative to the
        object barycentre along *axis*.  Subunits with a non-negative
        projection (higher axial coordinate) are labelled ring 0; those
        with a negative projection are labelled ring 1.

        The result is written into ``motl.df[ring_column]`` and
        ``self._ring_group_columns`` is updated to
        ``[tomo_id_column, affiliation_column, ring_column]``.

        Parameters
        ----------
        ring_column : MotlColumn, default='geom5'
            Column to write ring labels (0 or 1) into.
        axis : array-like of shape (3,), default=(0, 0, 1)
            Splitting axis.  Need not be normalised.

        Returns
        -------
        cryomotl.Motl
            ``self.motl`` with *ring_column* populated in place.
        """
        axis_arr = np.asarray(axis, dtype=float)
        axis_arr = axis_arr / np.linalg.norm(axis_arr)
        self._split_axis = axis_arr
        self._ring_column = ring_column

        coord = self.motl.get_coordinates()
        ring_labels = np.zeros(len(self.motl.df), dtype=float)

        for keys, group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            grp_idx = group.index.to_numpy()
            coords_grp = coord[grp_idx, :]
            bary = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            projections = (coords_grp - bary) @ axis_arr
            ring_labels[grp_idx] = np.where(projections >= 0, 0.0, 1.0)

        self.motl.df[ring_column] = ring_labels

        self._ring_group_columns = [
            self.tomo_id_column,
            self.affiliation_column,
            ring_column,
        ]
        self._rings_split = True
        return self.motl

    # ------------------------------------------------------------------
    # Subunit ordering (ring-aware)
    # ------------------------------------------------------------------

    @gui_exposed(label="Assign subunit order", group="Affiliation", order=30, returns="none")
    def assign_subunit_order(self) -> None:
        """Assign 1-based subunit indices across both rings.

        Calls :meth:`split_rings` when the ring column is absent from
        ``self.motl.df``.  Then delegates per-ring cyclic ordering to
        :meth:`CnComplex.assign_subunit_order` (indices 1 … n within
        each ring).  Finally offsets ring 1 indices by ``self.n`` so that
        the full range is 1 … 2n (ring 0 first, ring 1 second).

        Modifies ``self.motl.df`` in place.
        """
        if not self._rings_split:
            self.split_rings(ring_column=self._ring_column, axis=self._split_axis)

        super().assign_subunit_order()

        ring1_mask = self.motl.df[self._ring_column] == 1.0
        self.motl.df.loc[ring1_mask, self.order_column] = self.motl.df.loc[ring1_mask, self.order_column] + self.n

    # ------------------------------------------------------------------
    # Inter-ring metrics
    # ------------------------------------------------------------------

    @gui_exposed(label="Ring spacing", group="Statistics", order=40, returns="dataframe")
    def ring_spacing(self, *, pixel_size: float | None = None) -> pd.DataFrame:
        """Axial distance between the two rings for each object.

        Thin wrapper over :func:`_ring_centre_spacing` that passes the dihedral
        axis, preserving the axial-projection behaviour.  Auto-splits the rings
        on first call if :meth:`split_rings` has not been called explicitly.

        Parameters
        ----------
        pixel_size : float or None, default None
            Ångström-per-voxel scale factor.  When ``None``, uses
            ``self.pixel_size`` (defaulting to ``1.0`` when unset).

        Returns
        -------
        pandas.DataFrame
            Columns ``tomo_id``, ``object_id``, ``ring_spacing``.
        """
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)
        if not self._rings_split:
            self.split_rings(ring_column=self._ring_column, axis=self._split_axis)

        coord = self.motl.get_coordinates()
        rows: list[dict] = []

        for (tomo_id, object_id), group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            mask0 = (group[self._ring_column] == 0.0).to_numpy()
            mask1 = (group[self._ring_column] == 1.0).to_numpy()
            grp_idx = group.index.to_numpy()
            coords_grp = coord[grp_idx, :]

            if not mask0.any() or not mask1.any():
                spacing = float("nan")
            else:
                c0 = np.mean(coords_grp[mask0, :], axis=0)
                c1 = np.mean(coords_grp[mask1, :], axis=0)
                spacing = _ring_centre_spacing(c0, c1, axis=self._split_axis) * ps

            rows.append(
                {
                    "tomo_id": float(tomo_id),
                    "object_id": float(object_id),
                    "ring_spacing": spacing,
                }
            )

        return pd.DataFrame(rows)

    @gui_exposed(label="Inter-ring twist", group="Statistics", order=45, returns="dataframe")
    def inter_ring_twist(self, *, degrees: bool = True) -> pd.DataFrame:
        """Rotational twist between the two rings for each object.

        Thin wrapper over :func:`_ring_centre_twist` that passes the dihedral
        axis, preserving the n-fold circular-mean phase computation.  Auto-splits
        the rings on first call if :meth:`split_rings` has not been called explicitly.

        Returns the signed n-fold phase difference in ``[−180/n, 180/n)``
        degrees (0 = rotational alignment).  An eclipsed arrangement gives
        ``0``; a perfectly staggered arrangement (offset by exactly 180°/n)
        lies at the fold boundary and returns ``−180/n``.

        Parameters
        ----------
        degrees : bool, default=True
            Return twist in degrees when ``True``, radians when ``False``.

        Returns
        -------
        pandas.DataFrame
            Columns ``tomo_id``, ``object_id``, ``inter_ring_twist``.
        """
        if not self._rings_split:
            self.split_rings(ring_column=self._ring_column, axis=self._split_axis)

        coord = self.motl.get_coordinates()
        rows: list[dict] = []

        for (tomo_id, object_id), group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            mask0 = (group[self._ring_column] == 0.0).to_numpy()
            mask1 = (group[self._ring_column] == 1.0).to_numpy()
            grp_idx = group.index.to_numpy()
            coords_grp = coord[grp_idx, :]
            bary = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            rel = coords_grp - bary

            if not mask0.any() or not mask1.any():
                twist = float("nan")
            else:
                twist = _ring_centre_twist(rel[mask0, :], rel[mask1, :], self.n, self._split_axis, degrees=degrees)

            rows.append(
                {
                    "tomo_id": float(tomo_id),
                    "object_id": float(object_id),
                    "inter_ring_twist": twist,
                }
            )

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # Per-object statistics
    # ------------------------------------------------------------------

    @gui_exposed(label="Get object stats", group="Statistics", order=50, returns="dataframe")
    def get_object_stats(self, *, pixel_size: float | None = None) -> pd.DataFrame:
        """Comprehensive per-object statistics for dihedral structures.

        Composes per-object occupancy, ring spacing, inter-ring twist, and
        per-ring diameter/circumference (averaged to one row per object).

        Parameters
        ----------
        pixel_size : float or None, default None
            Ångström-per-voxel scale factor for distance columns.  When
            ``None``, uses ``self.pixel_size`` (defaulting to ``1.0`` when
            unset).

        Returns
        -------
        pandas.DataFrame
            One row per ``(tomo_id, object_id)``.  Columns:

            ``tomo_id``, ``object_id``
                Group identifiers.
            ``n_present``, ``occupancy``, ``missing``
                From :meth:`occupancy`.
            ``x``, ``y``, ``z``
                Object barycentre (voxels).
            ``radius``
                Mean circumradius across both rings (voxels).
            ``ring_spacing``
                Axial distance between rings (scaled by *pixel_size*).
            ``inter_ring_twist``
                Rotational twist between rings (degrees).
            ``circumference``
                Mean per-ring circumference (scaled by *pixel_size*).
            ``mean_diameter``, ``n_pairs``
                Mean per-ring diameter and total pair count.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        """
        self._require_affiliation()
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)

        occ_df = self.occupancy()
        spacing_df = self.ring_spacing(pixel_size=ps)
        twist_df = self.inter_ring_twist(degrees=True)

        circ_df_ring = self.circumference(pixel_size=ps)
        diam_df_ring, _ = self.diameter(pixel_size=ps)

        merge_cols = ["tomo_id", "object_id"]
        circ_agg = circ_df_ring.groupby(merge_cols)["circumference"].mean().reset_index()
        diam_agg = (
            diam_df_ring.groupby(merge_cols)
            .agg(mean_diameter=("mean_diameter", "mean"), n_pairs=("n_pairs", "sum"))
            .reset_index()
        )

        coord = self.motl.get_coordinates()
        center_rows: list[dict] = []
        for (tomo_id, object_id), group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            coords_grp = coord[group.index.to_numpy(), :]
            bary = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            r = self._circumradius_for_group(coords_grp)
            center_rows.append(
                {
                    "tomo_id": float(tomo_id),
                    "object_id": float(object_id),
                    "x": float(bary[0]),
                    "y": float(bary[1]),
                    "z": float(bary[2]),
                    "radius": r,
                }
            )
        geo_df = pd.DataFrame(center_rows)

        result = occ_df.merge(geo_df, on=merge_cols, how="outer")
        result = result.merge(spacing_df, on=merge_cols, how="left")
        result = result.merge(twist_df, on=merge_cols, how="left")
        result = result.merge(circ_agg, on=merge_cols, how="left")
        result = result.merge(diam_agg, on=merge_cols, how="left")
        return result


# =============================================================================
# NPC
# =============================================================================


class NPC(CnComplex):
    """NPC-specific extensions of :class:`CnComplex`.

    Inherits all single-ring methods (centre computation, subunit ordering,
    and object merging) from :class:`CnComplex`.
    The methods below are NPC-specific: orientation unification,
    multi-ring assembly, and opposite-subunit diameter analysis.

    Typical workflow for a multi-ring NPC:

    1. :meth:`cluster_subunits_to_rings` — trace subunits into rings and
       merge nearby rings (call once per ring motl).
    2. :meth:`unify_nn_orientations` — flip ambiguous orientations (call
       with ``ring_index`` to target one ring in the multi-motl list).
    3. :meth:`merge` — unify ``object_id`` across rings, stamp ring column,
       and reconcile IR orientations.  One-way irreversible.
    """

    _ring_column: MotlColumn = "geom3"

    _DISPATCH_PER_RING: frozenset[str] = frozenset(
        {
            # Inherited from SymmetricComplex
            "step_statistics",
            "occupancy",
            "clean_per_object",
            "merge_subunits",
            "create_affiliation",
            # Inherited from CnComplex
            "circumference",
            "assign_subunit_order",
            "central_angles",
            # NPC override
            "get_object_stats",
        }
    )

    def __init__(
        self,
        motl: "ListLike[MotlSource]",
        symmetry=None,
        *,
        radius: float | None = None,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
        center_method: Literal["circle_fit", "barycentric"] = "circle_fit",
        ring_column: MotlColumn = "geom3",
        pixel_size: float = 1.0,
    ) -> None:
        # NPC always uses C8 symmetry; symmetry param accepted for compat only.
        raw = motl if isinstance(motl, (list, tuple)) else [motl]
        loaded = [cryomotl.Motl.load(m) for m in raw]
        super().__init__(
            loaded[0],
            "C8",
            affiliation_column=affiliation_column,
            order_column=order_column,
            tomo_id_column=tomo_id_column,
            center_method=center_method,
        )
        self.pixel_size: float = float(pixel_size)
        self.radius: float | None = radius
        self._ring_column: MotlColumn = ring_column
        self._ring_motls: list[cryomotl.Motl] = loaded
        self._rings_merged: bool = len(loaded) == 1
        self._per_ring_active: bool = False
        self._last_alignment: dict = {}
        self._store_sid_column: str | None = None
        # _setup (via super().__init__) copied loaded[0]; sync back so in-place
        # methods that write to self.motl stay visible through _ring_motls[0].
        self.motl = self._ring_motls[0]

    # ------------------------------------------------------------------
    # Multi-ring dispatch
    # ------------------------------------------------------------------

    def per_ring(self, method_name: str, *args, **kwargs):
        """Call *method_name* on each held ring motl and return aggregated results.

        When rings are already merged, delegates directly to the instance method.
        Otherwise, walks the MRO (skipping NPC itself) to find the base-class
        implementation, temporarily swaps ``self.motl`` to each ring, calls
        the base implementation, and tags the result with a ring index in
        ``_ring_column`` (``geom3``).  ``_per_ring_active`` is set to ``True``
        during the loop so that routing guards on NPC overrides (e.g.
        :meth:`occupancy`) skip re-dispatch when called from inside
        the base method.  DataFrame results are concatenated; Motl results are
        returned as a list.
        """
        if self._rings_merged:
            return getattr(self, method_name)(*args, **kwargs)
        base_fn = None
        for _cls in type(self).__mro__[1:]:
            if method_name in _cls.__dict__:
                _fn = _cls.__dict__[method_name]
                if callable(_fn):
                    base_fn = _fn.__get__(self, type(self))
                break
        if base_fn is None:
            raise AttributeError(f"{type(self).__name__} has no base-class implementation of '{method_name}'")
        orig_motl = self.motl
        parts = []
        self._per_ring_active = True
        try:
            for ring_idx, ring_motl in enumerate(self._ring_motls):
                self.motl = ring_motl
                result = base_fn(*args, **kwargs)
                if isinstance(result, pd.DataFrame):
                    result = result.copy()
                    result[self._ring_column] = float(ring_idx + 1)
                elif isinstance(result, cryomotl.Motl):
                    tagged = cryomotl.Motl(result.df.copy())
                    tagged.df[self._ring_column] = float(ring_idx + 1)
                    result = tagged
                parts.append(result)
        finally:
            self.motl = orig_motl
            self._per_ring_active = False
        if parts and isinstance(parts[0], pd.DataFrame):
            return pd.concat(parts, ignore_index=True)
        return parts

    # ------------------------------------------------------------------
    # Overrides that respect ring column after merge
    # ------------------------------------------------------------------

    def occupancy(self) -> pd.DataFrame:
        """Per-object (per-ring when merged) subunit occupancy.

        Overrides :meth:`SymmetricComplex.occupancy` to group by
        ``_ring_group_columns`` so that, after :meth:`merge`, each
        ``(tomo_id, object_id, ring)`` group is reported separately.

        Pre-merge direct calls dispatch via :meth:`per_ring`; calls from inside
        :meth:`per_ring` (``_per_ring_active=True``) run this body directly on
        the current ``self.motl``.
        """
        if not self._rings_merged and not self._per_ring_active:
            return self.per_ring("occupancy")
        self._require_affiliation()
        has_order = self.order_column in self.motl.df.columns
        rows: list[dict] = []
        all_expected = set(range(1, self.n_subunits + 1))
        for keys, group in self.motl.df.groupby(self._ring_group_columns):
            tomo_id = keys[0]
            object_id = keys[1]
            n_present = len(group)
            if has_order:
                present = set(int(v) for v in group[self.order_column].dropna())
                missing: list[int] | None = sorted(all_expected - present)
            else:
                missing = None
            row: dict = {
                self.tomo_id_column: float(tomo_id),
                self.affiliation_column: float(object_id),
                "n_present": n_present,
                "occupancy": n_present / self.n_subunits,
                "missing": missing,
            }
            for extra_col, extra_val in zip(self._ring_group_columns[2:], keys[2:]):
                row[extra_col] = extra_val
            rows.append(row)
        return pd.DataFrame(rows)

    def get_object_stats(self, *, pixel_size: float | None = None) -> pd.DataFrame:
        """Comprehensive per-object statistics for NPC structures.

        Pre-merge direct calls dispatch per-ring via :meth:`per_ring`;
        post-merge calls run on the full merged motl, grouping by
        ``(tomo_id, object_id, ring_column)`` so each ring gets its own
        centre, radius and circumference row.

        Uses ``geom.barycenter`` and ``_circumradius_for_group``.
        Circumference = 2π × circumradius × pixel_size.
        Spacing and twist (per adjacent ring pair, not per ring) are merged
        on ``(tomo_id, object_id)``.
        """
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)
        if not self._rings_merged and not self._per_ring_active:
            return self.per_ring("get_object_stats", pixel_size=ps)
        self._require_affiliation()

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column

        # Use _ring_group_columns so both single-ring (never-merged) NPCs and
        # genuinely merged multi-ring NPCs are handled correctly.
        # _ring_group_columns = [tomo, object] before merge(); it gains ring_col
        # only after merge() finishes (line 3310).  _rings_merged=True on a
        # single-ring NPC from __init__ but _ring_group_columns stays two-column.
        group_cols = self._ring_group_columns
        has_ring_group = len(group_cols) > 2

        occ_df = self.occupancy()

        coord = self.motl.get_coordinates()
        center_rows: list[dict] = []
        for keys, group in self.motl.df.groupby(group_cols):
            if has_ring_group:
                tomo_id, object_id, ring_id = keys
            else:
                tomo_id, object_id = keys
                ring_id = None
            coords_grp = coord[group.index.to_numpy(), :]
            bary = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            r = self._circumradius_for_group(coords_grp)
            row: dict = {
                tomo_col: float(tomo_id),
                aff_col: float(object_id),
                "x": float(bary[0]),
                "y": float(bary[1]),
                "z": float(bary[2]),
                "radius": r,
                "circumference": 2.0 * np.pi * r * ps,
                "mean_diameter": 2.0 * r * ps,
            }
            if has_ring_group:
                row[ring_col] = float(ring_id)
            center_rows.append(row)
        geo_df = pd.DataFrame(center_rows)

        result = occ_df.merge(geo_df, on=group_cols, how="outer")
        if len(self._ring_motls) > 1:
            spacing_df = self.ring_spacing(pixel_size=ps)
            result = result.merge(spacing_df, on=[tomo_col, aff_col], how="left")
            if hasattr(self, "inter_ring_twist"):
                twist_df = self.inter_ring_twist(degrees=True)
                result = result.merge(twist_df, on=[tomo_col, aff_col], how="left")
        return result

    @gui_exposed(label="Assign subunit order", group="Affiliation", order=30, returns="motl")
    def assign_subunit_order(
        self,
        ref_direction: np.ndarray | None = None,
        flag_column: str = "geom5",
        ambiguity_tol: float | None = None,
        *,
        radius: float | None = None,
        entry_shift: tuple[float, float, float] | None = None,
        exit_shift: tuple[float, float, float] | None = None,
        max_distance_factor: float = 1.65,
    ) -> "cryomotl.Motl":
        """Assign 1-based subunit indices per ring-object via chain tracing.

        For each ring in ``self._ring_motls``, groups particles by
        ``(tomo_id_column, affiliation_column)`` and calls
        :meth:`_chain_order_for_ring_object`.  The cross-ring alignment step
        is deferred to :meth:`_finalize_subunit_order` (called by
        :meth:`merge`).

        Parameters
        ----------
        ref_direction : numpy.ndarray or None
            Forwarded to the pooled-azimuth fallback inside
            :meth:`_chain_order_for_ring_object`.
        flag_column : str, default ``'geom5'``
            Column written with ``1.0`` for objects whose ordering used the
            fallback or involved multiple chains.  Created in each ring motl
            if absent.
        ambiguity_tol : float or None
            Forwarded to the fallback.  Defaults to ``central_angle / 4``.
        radius : float or None
            Circumradius of the NPC ring in voxels, used only to compute the
            default tangential entry/exit shift ``R × sin(π/n)``.  Needed
            only when *entry_shift*/*exit_shift* are not both given; falls
            back to ``self.radius``, and a ``ValueError`` is raised if both
            are ``None``.
        entry_shift : 3-tuple of float or None
            Local-frame shift vector for the entry throwaway motl.  When
            given together with *exit_shift*, overrides the auto-computed
            value for every object and no *radius* is required.
        exit_shift : 3-tuple of float or None
            Local-frame shift vector for the exit throwaway motl.  When
            given together with *entry_shift*, overrides the auto-computed
            value for every object and no *radius* is required.
        max_distance_factor : float, default 1.65
            Per-ring chain-link cutoff = ``max_distance_factor × median
            nearest-neighbour spacing`` of the ring's subunits (voxels).
            Default 1.65 spans dilated-adjacent pairs up to ~1.65× the
            ring's own spacing while staying below the distance to a
            non-adjacent subunit.

        Returns
        -------
        cryomotl.Motl
            Combined motl of all rings with ``order_column``,
            ``_ring_column``, and *flag_column* populated.
        """
        if not self._ring_motls:
            raise ValueError("no ring motls — split rings first")

        n = self.n_subunits
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        order_col = self.order_column

        if ambiguity_tol is None:
            ambiguity_tol = self.central_angle / 4.0

        global_entry = list(entry_shift) if entry_shift is not None else None
        global_exit = list(exit_shift) if exit_shift is not None else None

        for rm in self._ring_motls:
            rm.df.reset_index(drop=True, inplace=True)
            if flag_column not in rm.df.columns:
                rm.df[flag_column] = 0.0
            if order_col not in rm.df.columns:
                rm.df[order_col] = 0.0

            order_col_loc = rm.df.columns.get_loc(order_col)
            flag_col_loc = rm.df.columns.get_loc(flag_column)

            for (_t, _o), grp_df in rm.df.groupby([tomo_col, aff_col]):
                pos = grp_df.index.to_numpy()
                obj_motl = cryomotl.Motl(grp_df.reset_index(drop=True))

                if global_entry is not None and global_exit is not None:
                    ev = global_entry
                    xv = global_exit
                else:
                    R = radius if radius is not None else self.radius
                    if R is None:
                        raise ValueError(
                            "NPC.assign_subunit_order: ring radius unknown — "
                            "pass radius= or set NPC(..., radius=...)"
                        )
                    delta_vox = float(R) * np.sin(np.pi / n)
                    ev = [0.0, -delta_vox, 0.0]
                    xv = [0.0, +delta_vox, 0.0]

                order_arr, flagged = self._chain_order_for_ring_object(
                    obj_motl,
                    entry_shift_vec=ev,
                    exit_shift_vec=xv,
                    max_distance_factor=max_distance_factor,
                    ambiguity_tol=ambiguity_tol,
                    ref_direction=ref_direction,
                )

                order_arr_raw = order_arr
                order_arr, deduped = self._dedupe_ring_order(order_arr_raw, n)
                if deduped:
                    flagged = True

                rm.df.iloc[pos, order_col_loc] = order_arr
                if flagged:
                    rm.df.iloc[pos, flag_col_loc] = 1.0

        frames = []
        for ring_idx, rm in enumerate(self._ring_motls):
            df = rm.df.copy()
            df[self._ring_column] = float(ring_idx + 1)
            frames.append(df)
        return cryomotl.Motl(pd.concat(frames, ignore_index=True))

    @staticmethod
    def _dedupe_ring_order(order_arr: np.ndarray, n: int) -> tuple[np.ndarray, bool]:
        """Replace later duplicates in *order_arr* with unique sentinel values ≥ n+1.

        Walk positionally: keep the first occurrence of each value unchanged;
        reassign each subsequent duplicate to the smallest integer ≥ n+1 not
        already present in the array.  The input array is not modified.

        Parameters
        ----------
        order_arr : np.ndarray
            1-D float array of subunit order values for one ring.
        n : int
            Ring size (valid slots are 1..n).  Sentinels start at n+1.

        Returns
        -------
        new_arr : np.ndarray
            Deduplicated array; identical to *order_arr* when *changed* is False.
        changed : bool
            True if any sentinel was assigned.

        """
        new_arr = order_arr.astype(float).copy()
        used: set[int] = {int(v) for v in order_arr}
        seen: set[int] = set()
        changed = False
        next_sentinel = n + 1
        for i, v in enumerate(order_arr):
            iv = int(v)
            if iv not in seen:
                seen.add(iv)
            else:
                while next_sentinel in used:
                    next_sentinel += 1
                new_arr[i] = float(next_sentinel)
                used.add(next_sentinel)
                seen.add(next_sentinel)
                next_sentinel += 1
                changed = True
        return new_arr, changed

    def _chain_order_for_ring_object(
        self,
        object_motl: "cryomotl.Motl",
        *,
        entry_shift_vec: list[float],
        exit_shift_vec: list[float],
        max_distance_factor: float,
        ambiguity_tol: float,
        ref_direction: np.ndarray | None,
    ) -> tuple[np.ndarray, bool]:
        """Assign 1-based order for one ring-object via chain tracing.

        Chain scratch columns live only on throwaway motls and are never
        written to the real motl.

        Parameters
        ----------
        object_motl : cryomotl.Motl
            Particles for one (tomo_id, object_id) group, with 0-based
            integer RangeIndex and populated ``subtomo_id``.
        entry_shift_vec, exit_shift_vec : list[float]
            Local-frame shifts (applied via :meth:`Motl.shift_positions`) to
            build the entry and exit throwaway motls.
        max_distance_factor : float
            Per-ring cutoff = ``max_distance_factor × median nearest-neighbour
            spacing`` of the ring's subunit positions (voxels).
        ambiguity_tol : float
            Forwarded to :meth:`_cyclic_indices_for_object` in the fallback.
        ref_direction : ndarray or None
            Forwarded to the pooled-azimuth fallback.

        Returns
        -------
        order_array : np.ndarray of shape (k,)
            1-based subunit slots, positionally aligned to *object_motl*.
        flagged : bool
            ``True`` when the fallback was used or multiple chains were found.
        """
        k = len(object_motl.df)
        n = self.n_subunits
        tomo_col = self.tomo_id_column

        # --- Per-ring max_distance from NN spacing ---
        coords = object_motl.get_coordinates()  # (k, 3); reused in winding block
        if k >= 2:
            d_mat = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
            np.fill_diagonal(d_mat, np.inf)
            s_ring = float(np.median(d_mat.min(axis=1)))
            max_distance = max_distance_factor * s_ring
        else:
            max_distance = 0.0

        # --- Build throwaway entry/exit motls ---
        motl_entry = object_motl.shift_positions(entry_shift_vec, inplace=False)
        motl_exit = object_motl.shift_positions(exit_shift_vec, inplace=False)

        # --- Trace chains on throwaway ---
        try:
            traced = nnana.trace_chains(
                motl_entry,
                motl_exit,
                max_distance=max_distance,
                column_name=tomo_col,
                store_idx1="__cid__",
                store_idx2="__cpos__",
            )
        except Exception:
            traced = None

        def _fallback() -> tuple[np.ndarray, bool]:
            s_idx, _ = self._cyclic_indices_for_object(object_motl, ref_direction=ref_direction)
            return np.array(s_idx, dtype=float), True

        if traced is None or "__cid__" not in traced.df.columns:
            return _fallback()

        # --- Align traced result back to object_motl via subtomo_id ---
        traced.df.sort_values("subtomo_id", inplace=True)
        traced.df.reset_index(drop=True, inplace=True)
        sid_to_orig = {float(s): i for i, s in enumerate(object_motl.df["subtomo_id"].values)}

        chain_ids = traced.df["__cid__"].values
        chain_pos_vals = traced.df["__cpos__"].values
        traced_sids = traced.df["subtomo_id"].values

        # Check for usable chains (size > 1)
        chain_sizes: dict[int, int] = {}
        for cid in chain_ids:
            chain_sizes[int(cid)] = chain_sizes.get(int(cid), 0) + 1
        usable = {cid for cid, sz in chain_sizes.items() if sz > 1}

        if not usable:
            return _fallback()

        order_final = np.zeros(k, dtype=float)

        # --- Shared axis computation (needed for winding in all paths) ---
        all_coords = coords  # already fetched above for NN spacing
        all_angles = object_motl.get_angles()
        center, _ = self._compute_object_center(object_motl)
        rel = all_coords - center
        _, _, Vt = np.linalg.svd(rel, full_matrices=True)
        axis = Vt[-1]
        normals_v = geom.euler_angles_to_normals(all_angles)
        mean_n = normals_v.mean(axis=0)
        norm_mn = float(np.linalg.norm(mean_n))
        if norm_mn > 1e-10 and np.dot(axis, mean_n / norm_mn) < 0:
            axis = -axis

        # --- Build per-chain sorted sequences: (chain_pos, orig_idx) ---
        chain_seqs: dict[int, list[tuple[int, int]]] = {}
        for sid, cid_f, cp in zip(traced_sids, chain_ids, chain_pos_vals):
            chain_seqs.setdefault(int(cid_f), []).append((int(cp), sid_to_orig[float(sid)]))
        for cid in chain_seqs:
            chain_seqs[cid].sort(key=lambda x: x[0])

        # --- Winding orientation: reverse chain if it winds negatively ---
        def _orient_chain(seq: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], bool]:
            """Return (oriented seq, winding_flagged).
            flagged when < 3 members or |winding| < 150° (sense untrustworthy)."""
            if len(seq) < 3:
                return seq, True
            vecs = []
            for _, oi in seq:
                v = rel[oi] - np.dot(rel[oi], axis) * axis
                vn = float(np.linalg.norm(v))
                vecs.append(v / vn if vn > 1e-10 else np.zeros(3))
            winding = float(sum(
                np.degrees(geom.vector_angular_distance_signed(vecs[i], vecs[i + 1], axis))
                for i in range(len(vecs) - 1)
            ))
            if abs(winding) < 150.0:
                return seq, True
            return (seq[::-1] if winding < 0 else seq), False

        if len(chain_sizes) == 1:
            # --- Single chain: orient and label 1-based ---
            cid = next(iter(chain_seqs))
            seq_o, winding_flagged = _orient_chain(chain_seqs[cid])
            for slot_0, (_, oi) in enumerate(seq_o):
                order_final[oi] = float(slot_0 + 1)
            return order_final, winding_flagged

        # --- Multiple chains ---
        ref_v0 = rel[0] - np.dot(rel[0], axis) * axis
        n0 = float(np.linalg.norm(ref_v0))
        ref_vec = ref_v0 / n0 if n0 > 1e-10 else rel[0] / max(float(np.linalg.norm(rel[0])), 1e-12)
        ref_vec = ref_vec / float(np.linalg.norm(ref_vec))

        azs = np.empty(k, dtype=float)
        for i in range(k):
            vp = rel[i] - np.dot(rel[i], axis) * axis
            vp_n = float(np.linalg.norm(vp))
            azs[i] = (
                0.0 if vp_n < 1e-10
                else np.degrees(geom.vector_angular_distance_signed(ref_vec, vp / vp_n, axis)) % 360.0
            )

        # Orient each chain; singletons kept as-is.
        chain_oriented: dict[int, list[tuple[int, int]]] = {}
        for cid, seq in chain_seqs.items():
            if chain_sizes[cid] > 1:
                seq_o, _ = _orient_chain(seq)
            else:
                seq_o = seq
            chain_oriented[cid] = seq_o

        # First/last azimuths from oriented sequences.
        chain_first_az: dict[int, float] = {
            cid: azs[seq_o[0][1]] for cid, seq_o in chain_oriented.items()
        }
        chain_last_az: dict[int, float] = {
            cid: azs[seq_o[-1][1]] for cid, seq_o in chain_oriented.items()
        }

        sorted_chain_ids = sorted(chain_sizes.keys(), key=lambda c: chain_first_az[c])
        n_chains = len(sorted_chain_ids)
        delta = self.central_angle

        chain_start_slot: dict[int, int] = {}
        slot = 1
        for ci, cid in enumerate(sorted_chain_ids):
            chain_start_slot[cid] = slot
            sz = chain_sizes[cid]
            next_cid = sorted_chain_ids[(ci + 1) % n_chains]
            arc = (chain_first_az[next_cid] - chain_last_az[cid]) % 360.0
            gap = max(1, int(round(arc / delta)))
            slot += sz + gap

        for cid in sorted_chain_ids:
            start = chain_start_slot[cid]
            for pos_0, (_, oi) in enumerate(chain_oriented[cid]):
                order_final[oi] = float((start + pos_0 - 1) % n + 1)

        return order_final, True  # multi-chain always flagged

    def _finalize_subunit_order(
        self,
        flag_column: str = "geom5",
        ambiguity_tol: float | None = None,
    ) -> None:
        """Align subunit order across rings by 3D nearest-neighbour matching.

        Called at the end of :meth:`merge`.  For each merged pore, selects the
        most-populated ring whose existing subunit numbers form a contiguous
        n-fold cycle as the reference, then assigns every other ring's subunit
        numbers by matching each of its particles to the nearest reference
        particle via :func:`scipy.optimize.linear_sum_assignment`.

        The within-ring ordering assigned by :meth:`assign_subunit_order` is
        preserved; only the global label offset (and chirality) is adjusted.

        Parameters
        ----------
        flag_column : str, default ``'geom5'``
            Column written with ``1.0`` for pores where no contiguous reference
            ring could be found (ambiguous alignment) and for pores that still
            carry duplicate slots after relabelling.  Set to ``0.0`` for all
            other rows.
        ambiguity_tol : float or None
            Accepted for interface compatibility; not used in this path.
        """
        if not self._rings_merged:
            return

        n = self.n_subunits
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column
        order_col = self.order_column

        self.motl.df[flag_column] = 0.0

        all_coords = self.motl.get_coordinates()
        tomo_vals = self.motl.df[tomo_col].values
        aff_vals = self.motl.df[aff_col].values
        ring_vals = self.motl.df[ring_col].values

        pore_keys: set[tuple[float, float]] = set()
        for tv, av in zip(tomo_vals.tolist(), aff_vals.tolist()):
            pore_keys.add((float(tv), float(av)))

        order_col_loc = self.motl.df.columns.get_loc(order_col)
        flag_col_loc = self.motl.df.columns.get_loc(flag_column)

        def _is_contiguous(nums: np.ndarray) -> bool:
            """True if nums forms a run of k consecutive values mod n (1-based)."""
            k = len(nums)
            if k == 0 or k > n:
                return False
            s = {int(v) - 1 for v in nums}  # convert to 0-based
            if len(s) != k:
                return False  # duplicates present
            for start in s:
                if {(start + i) % n for i in range(k)} == s:
                    return True
            return False

        for tomo_id, object_id in sorted(pore_keys):
            pore_rows = np.where((tomo_vals == tomo_id) & (aff_vals == object_id))[0]
            if len(pore_rows) == 0:
                continue

            pore_coords = all_coords[pore_rows]
            pore_ring_vals = ring_vals[pore_rows]
            pore_order = self.motl.df.iloc[pore_rows][order_col].to_numpy(dtype=float)

            ring_keys_present = sorted(set(pore_ring_vals.tolist()))
            if len(ring_keys_present) < 2:
                continue

            # Per-ring data: local indices into pore_rows, 3D coords, order numbers.
            ring_data: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
            for rk in ring_keys_present:
                local_j = np.where(pore_ring_vals == rk)[0]
                ring_data[rk] = (local_j, pore_coords[local_j], pore_order[local_j])

            # Reference ring: most-populated contiguous ring; tie-break by smallest key.
            contiguous_keys = [rk for rk in ring_keys_present if _is_contiguous(ring_data[rk][2])]
            pore_flagged = len(contiguous_keys) == 0
            candidate_keys = contiguous_keys if contiguous_keys else ring_keys_present
            ref_rk = max(candidate_keys, key=lambda rk: (len(ring_data[rk][0]), -rk))

            _, ref_coords, ref_numbers = ring_data[ref_rk]

            if pore_flagged:
                self.motl.df.iloc[pore_rows, flag_col_loc] = 1.0

            # Assign each non-reference ring's numbers by 3D NN to the reference.
            for rk in ring_keys_present:
                if rk == ref_rk:
                    continue
                local_j, ring_coords_rk, _ = ring_data[rk]
                k_ring = len(local_j)

                C = np.linalg.norm(
                    ring_coords_rk[:, None, :] - ref_coords[None, :, :], axis=-1
                )  # (k_ring, k_ref)

                row_ind, col_ind = linear_sum_assignment(C)
                new_numbers = np.empty(k_ring, dtype=float)
                assigned = np.zeros(k_ring, dtype=bool)
                for r, c in zip(row_ind.tolist(), col_ind.tolist()):
                    new_numbers[r] = ref_numbers[c]
                    assigned[r] = True
                # Unmatched particles (k_ring > k_ref): assign nearest reference.
                for i in np.where(~assigned)[0].tolist():
                    new_numbers[i] = ref_numbers[int(C[i].argmin())]

                ring_rows = pore_rows[local_j]
                self.motl.df.iloc[ring_rows, order_col_loc] = new_numbers

                final_order, deduped = self._dedupe_ring_order(new_numbers, n)
                if deduped:
                    self.motl.df.iloc[ring_rows, order_col_loc] = final_order
                    self.motl.df.iloc[ring_rows, flag_col_loc] = 1.0

        dup_df = self.check_for_duplicates()
        if not dup_df.empty:
            pore_pairs = dup_df[["tomo_id", "object_id"]].drop_duplicates()
            for _, row in pore_pairs.iterrows():
                tid, oid = float(row["tomo_id"]), float(row["object_id"])
                dup_rows = np.where((tomo_vals == tid) & (aff_vals == oid))[0]
                if len(dup_rows) > 0:
                    self.motl.df.iloc[dup_rows, flag_col_loc] = 1.0

        # Summarise pores that still carry sentinel values (> n) after NN alignment.
        sentinel_mask = self.motl.df[order_col] > n
        if sentinel_mask.any():
            sent_df = self.motl.df.loc[sentinel_mask, [tomo_col, aff_col]].drop_duplicates()
            by_tomo: dict[int, list[int]] = {}
            for _, r in sent_df.iterrows():
                t_key = int(r[tomo_col])
                by_tomo.setdefault(t_key, []).append(int(r[aff_col]))
            parts = "; ".join(
                f"tomo {t}: objects {sorted(oids)}" for t, oids in sorted(by_tomo.items())
            )
            print(
                f"NPC: {sent_df.shape[0]} pore(s) have sentinelled subunits after merge — "
                f"{parts}. Use renumber_subunits to correct."
            )

    @gui_exposed(label="Cluster subunits to rings", group="NPC workflow", order=10, returns="motl")
    @staticmethod
    def cluster_subunits_to_rings(
        input_motl: MotlSource,
        npc_radius: float | None = None,
        max_trace_distance: float | None = None,
        min_trace_distance: float = 0,
        *,
        mask_size: TripletLike | None = None,
        entry_mask_coord: TripletLike | None = None,
        exit_mask_coord: TripletLike | None = None,
        entry_mask: MapSource | None = None,
        exit_mask: MapSource | None = None,
        tomo_id_column: "MotlColumn" = "tomo_id",
        affiliation_column: "MotlColumn" = "object_id",
    ) -> "cryomotl.Motl":
        """Cluster NPC subunit particles into rings.

        Workflow:

        1. Build (or accept) spherical entry/exit masks.
        2. Re-centre the input motl to the entry and exit sub-particle
           positions.
        3. Trace entry/exit pairs into chains with
           :meth:`Chain.from_motls`.
        4. Copy chain annotations onto the original motl and merge nearby
           subunits with :meth:`merge_subunits`.

        Mask handling is fully in-memory: when a coord + ``mask_size`` is
        given, :func:`cryocat.core.cryomask.spherical_mask` is called
        without ``output_path`` and the returned ndarray is forwarded to
        :meth:`cryocat.core.cryomotl.Motl.recenter_to_subparticle` (whose
        ``input_map`` parameter accepts ndarrays via
        :func:`cryocat.core.cryomap.read`).  No temporary files are written.

        For each of the entry and exit sides, either the mask itself or the
        ``(coord + mask_size)`` pair must be supplied.

        Parameters
        ----------
        input_motl : MotlSource
            Subunit particle list.  A :class:`Motl`, a DataFrame, or a path
            to a motl file -- :meth:`cryocat.core.cryomotl.Motl.load`
            normalises all three.
        npc_radius : float
            Approximate NPC ring radius (voxels) used by
            :meth:`merge_subunits`.
        max_trace_distance : float
            Maximum allowed step distance during chain tracing (voxels).
        min_trace_distance : float, default=0
            Minimum allowed step distance during chain tracing.
        mask_size : TripletLike, optional
            Box size for the in-memory entry / exit masks.  Required when
            ``entry_mask`` / ``exit_mask`` are not supplied; ignored
            otherwise.
        entry_mask_coord : TripletLike, optional
            Centre of the entry spherical mask (voxels).  Required when
            ``entry_mask`` is not supplied.
        exit_mask_coord : TripletLike, optional
            Centre of the exit spherical mask (voxels).  Required when
            ``exit_mask`` is not supplied.
        entry_mask : MapSource, optional
            User-provided entry mask.  Path or ndarray.  When supplied,
            ``entry_mask_coord`` / ``mask_size`` are ignored on the entry
            side.
        exit_mask : MapSource, optional
            User-provided exit mask.  Path or ndarray.  When supplied,
            ``exit_mask_coord`` / ``mask_size`` are ignored on the exit
            side.

        tomo_id_column : str, default='tomo_id'
            Column holding the tomogram identifier.
        affiliation_column : str, default='object_id'
            Column holding the ring/object identifier used during sorting and
            merging.

        Returns
        -------
        Motl
            Motl with *affiliation_column* identifying each ring,
            ``geom1`` holding ring occupancy, and ``geom2`` the within-ring
            subunit index.

        Raises
        ------
        ValueError
            If, for either side, neither the mask nor the
            ``(coord + mask_size)`` pair was supplied.
        """
        if npc_radius is None:
            raise ValueError(
                "NPC.cluster_subunits_to_rings: npc_radius is required — "
                "pass npc_radius= or set NPC(..., radius=...) and call via an instance"
            )
        if max_trace_distance is None:
            raise ValueError(
                "NPC.cluster_subunits_to_rings: max_trace_distance is required"
            )
        if entry_mask is None:
            if entry_mask_coord is None or mask_size is None:
                raise ValueError(
                    "cluster_subunits_to_rings: supply either `entry_mask` or "
                    "both `entry_mask_coord` and `mask_size`."
                )
            entry_mask = cryomask.spherical_mask(mask_size, 3, center=entry_mask_coord)
        if exit_mask is None:
            if exit_mask_coord is None or mask_size is None:
                raise ValueError(
                    "cluster_subunits_to_rings: supply either `exit_mask` or " "both `exit_mask_coord` and `mask_size`."
                )
            exit_mask = cryomask.spherical_mask(mask_size, 3, center=exit_mask_coord)

        motl = cryomotl.Motl.load(input_motl)
        motl.renumber_particles()

        motl_entry = cryomotl.Motl.recenter_to_subparticle(motl, entry_mask)
        motl_exit = cryomotl.Motl.recenter_to_subparticle(motl, exit_mask)

        chain = Chain.from_motls(
            motl_entry,
            motl_exit,
            max_distance=max_trace_distance,
            min_distance=min_trace_distance,
        )
        chain.traced_motl.df.sort_values([tomo_id_column, affiliation_column, "geom2"], inplace=True)
        chain.get_occupancy()
        motl = chain.add_traced_info(motl)

        motl = NPC._merge_by_radius(
            motl,
            npc_radius,
            tomo_id_column=tomo_id_column,
            affiliation_column=affiliation_column,
        )
        return motl

    # ------------------------------------------------------------------
    # Orientation unification
    # ------------------------------------------------------------------

    @gui_exposed(label="Unify NN orientations", group="Affiliation", order=50, returns="none")
    def unify_nn_orientations(
        self,
        dist_threshold: float = 10000,
        ring_index: int = 0,
        *,
        reference: "cryomotl.Motl | None" = None,
    ) -> None:
        """Flip orientations so that neighbouring subunits point consistently.

        When *reference* is supplied, delegates to :meth:`_align_to_reference`:
        any particle in ``self.motl`` whose z-normal is more than 90° away
        from the per-object mean z-normal of *reference* is flipped 180° about
        z.  *dist_threshold* and *ring_index* are ignored in this branch.

        Without *reference*, traces particles into chains via
        :func:`nnana.trace_chains`, then walks each chain and applies a 180°
        rotation (``srot.from_euler("zxz", [0,180,0])``) whenever the cone
        angle (``geom.cone_distance``) between successive subunits exceeds 90°.
        Updates ``self.motl`` in place, and when multiple ring motls are held,
        also updates ``self._ring_motls[ring_index]``.

        Parameters
        ----------
        dist_threshold : float, default=10000
            Maximum nearest-neighbour distance for tracing (voxels).
            Ignored when *reference* is given.
        ring_index : int, default=0
            Which ring motl to operate on when the NPC holds multiple unmerged
            rings.  Must be 0 after :meth:`merge` is called (raises otherwise).
            Ignored when *reference* is given.
        reference : cryomotl.Motl or None, default=None
            Reference motl that defines the expected orientation direction.
            When provided, each particle is compared against the per-object
            mean z-normal of *reference* rather than against its chain
            neighbours.

        Raises
        ------
        ValueError
            If called post-merge with ``ring_index != 0`` (chain-tracing path
            only; the *reference* path never raises this).  After :meth:`merge`
            there is a single combined motl; a non-zero ``ring_index`` has no
            target and would silently operate on the wrong data.
        """
        if reference is not None:
            self._align_to_reference(self.motl, reference)
            return

        if self._rings_merged and ring_index != 0:
            raise ValueError(
                f"unify_nn_orientations: ring_index={ring_index} is not valid after merge() "
                "— the rings have been combined into a single motl.  "
                "Call with ring_index=0, or call before merge()."
            )
        if not self._rings_merged:
            orig_self_motl = self.motl
            self.motl = self._ring_motls[ring_index]

        traced_motl = nnana.trace_chains(
            self.motl,
            motl_exit=None,
            max_distance=dist_threshold,
            min_distance=0,
            column_name=self.tomo_id_column,
            output_motl=None,
            store_idx1=self.affiliation_column,
            store_idx2="geom2",
            store_dist="geom4",
        )

        rot_180 = srot.from_euler("zxz", angles=[0, 180, 0], degrees=True)

        for t in traced_motl.get_unique_values(self.tomo_id_column):
            tm = traced_motl.get_motl_subset(column_values=[t], column_name=self.tomo_id_column, reset_index=True)
            rotations = tm.get_rotations()
            for i in np.arange(1, tm.df["geom2"].max(), dtype=int):
                cone_angle = geom.cone_distance(rotations[i - 1], rotations[i])
                if cone_angle > 90.0:
                    rotations[i] = rotations[i] * rot_180

            angles = rotations.as_euler("zxz", degrees=True)
            tm.fill({"angles": angles})
            traced_motl.df.loc[traced_motl.df[self.tomo_id_column] == t, :] = tm.df.values

        self.motl = cryomotl.Motl(traced_motl.df.sort_values(by="subtomo_id"))

        if not self._rings_merged:
            self._ring_motls[ring_index] = self.motl
            if ring_index != 0:
                self.motl = orig_self_motl

    def _align_to_reference(self, target: "cryomotl.Motl", reference: "cryomotl.Motl") -> None:
        """Flip *target* orientations toward *reference* per (tomo, object).

        For each (tomo_id_column, affiliation_column) group in *target*, finds
        the matching group in *reference*, computes the mean z-normal of that
        group via :func:`geom.euler_angles_to_normals`, then flips every
        *target* particle whose z-normal deviates by more than 90° by applying
        ``srot.from_euler("zxz", [0, 180, 0])``.

        Operates in place on *target*.  *reference* is read-only.

        Parameters
        ----------
        target : cryomotl.Motl
            Particles whose orientations may need flipping.
        reference : cryomotl.Motl
            Particles that define the expected orientation direction.
        """
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        rot_180 = srot.from_euler("zxz", angles=[0, 180, 0], degrees=True)

        for t in target.get_unique_values(tomo_col):
            ref_t_df = reference.df[reference.df[tomo_col] == t]
            if ref_t_df.shape[0] == 0:
                continue
            tgt_t_df = target.df[target.df[tomo_col] == t]
            for o in tgt_t_df[aff_col].unique():
                ref_o_df = ref_t_df[ref_t_df[aff_col] == o]
                if ref_o_df.shape[0] == 0:
                    continue
                ref_z_vecs = geom.euler_angles_to_normals(
                    cryomotl.Motl(ref_o_df).get_angles()
                )
                ref_mean_z = ref_z_vecs.mean(axis=0)
                ref_norm = float(np.linalg.norm(ref_mean_z))
                if ref_norm < 1e-10:
                    continue
                ref_mean_z /= ref_norm

                tgt_mask = (target.df[tomo_col] == t) & (target.df[aff_col] == o)
                tgt_o_df = target.df[tgt_mask]
                tgt_angles = cryomotl.Motl(tgt_o_df).get_angles()
                tgt_z_vecs = geom.euler_angles_to_normals(tgt_angles)
                for global_idx, tgt_z, angles_row in zip(tgt_o_df.index, tgt_z_vecs, tgt_angles):
                    if geom.vector_angular_distance(tgt_z, ref_mean_z) > 90.0:
                        rot = srot.from_euler("zxz", angles_row, degrees=True)
                        flipped = rot_180 * rot
                        target.df.loc[global_idx, ["phi", "theta", "psi"]] = (
                            flipped.as_euler("zxz", degrees=True)
                        )

    # ------------------------------------------------------------------
    # Multi-ring merge
    # ------------------------------------------------------------------

    @gui_exposed(label="Merge", group="NPC workflow", order=40, returns="motl")
    def merge(
        self,
        *,
        npc_radius: float | None = None,
        ring_order: list[int] | None = None,
        distance_threshold: float = 40,
        store_sid_column: str | None = None,
    ) -> "cryomotl.Motl":
        """Unify affiliation across rings, stamp ring column, and reconcile IR orientations.

        This is a one-way irreversible operation.  After the call,
        ``self.motl`` holds all rings as a single merged motl,
        ``self._rings_merged`` is ``True``, and
        ``self._ring_group_columns`` is extended to include ``_ring_column``
        (``geom3``).

        Steps:

        1. Re-order rings according to *ring_order* (default: input order).
        2. Stamp ring index into ``_ring_column`` for every particle.
        3. Renumber ``affiliation_column`` sequentially so every ring starts
           fresh, then use nearest-neighbour pore matching
           (``nnana.find_nn_indices``) to unify ``affiliation_column`` across
           rings that belong to the same pore.
        4. Reconcile IR orientations via :meth:`_align_to_reference`: flip
           any IR particle (ring index 1) whose z-normal is more than 90°
           away from the per-pore mean CR z-normal.
        5. Concatenate and store.
        6. If *store_sid_column* is given, save the original ``subtomo_id``
           values into that column and call
           :meth:`~cryocat.core.cryomotl.Motl.renumber_particles` so the
           merged motl has globally unique particle IDs.

        Parameters
        ----------
        npc_radius : float
            Ring radius in voxels.  Forwarded to :meth:`get_centers_as_motl`.
        ring_order : list of int or None
            Permutation of ``range(len(rings))``.  Index 0 = CR, index 1 = IR,
            index 2 = NR.  Defaults to input order.
        distance_threshold : float, default=40
            Maximum pore-centre distance (voxels) for two rings to be
            considered the same pore.
        store_sid_column : str or None, default=None
            When set, the original ``subtomo_id`` values (per-ring particle
            IDs) are stored in this column before renumbering.  Useful for
            tracing each merged particle back to its source.
        """
        _R = npc_radius if npc_radius is not None else self.radius
        if _R is None:
            raise ValueError(
                "NPC.merge: ring radius unknown — "
                "pass npc_radius= or set NPC(..., radius=...)"
            )
        _R = float(_R)

        if self._rings_merged:
            return self.motl
        for _ring_idx, _rm in enumerate(self._ring_motls):
            if (_rm.df[self.order_column] == 0).all():
                raise ValueError(
                    f"Ring {_ring_idx} has no subunit ordering "
                    f"('{self.order_column}' is all zeros) — "
                    f"call assign_subunit_order on ring {_ring_idx} before merging."
                )
        if ring_order is None:
            ring_order = list(range(len(self._ring_motls)))
        ordered = [cryomotl.Motl(self._ring_motls[i].df.copy()) for i in ring_order]

        # Step 1: stamp ring column (1-based: ring 1 = CR, ring 2 = IR, …)
        for ring_idx, rm in enumerate(ordered):
            rm.df[self._ring_column] = float(ring_idx + 1)

        # Step 2: unify affiliation — sequential renumber then NN pore matching
        aff_col = self.affiliation_column
        tomo_col = self.tomo_id_column
        starting_number = 1
        for rm in ordered:
            rm.renumber_objects_sequentially(starting_number=starting_number)
            starting_number = int(rm.df[aff_col].max()) + 1

        ring_pairs = mathutils.get_all_pairs(list(range(len(ordered))))
        for i_pair in ring_pairs:
            i, j = i_pair
            for t in ordered[i].get_unique_values(tomo_col):
                tm1 = ordered[i].get_motl_subset(column_values=[t], column_name=tomo_col, reset_index=True)
                tm2 = ordered[j].get_motl_subset(column_values=[t], column_name=tomo_col, reset_index=True)
                if tm2.df.shape[0] == 0:
                    continue
                centers1 = NPC.get_centers_as_motl(tm1, radius=_R, tomo_id_column=tomo_col, affiliation_column=aff_col)
                centers2 = NPC.get_centers_as_motl(tm2, radius=_R, tomo_id_column=tomo_col, affiliation_column=aff_col)
                _, obj1_idx, distances, _ = nnana.find_nn_indices(
                    centers2.get_coordinates(),
                    centers1.get_coordinates(),
                    k=1,
                )
                distances = distances.reshape(-1)
                obj1_idx = obj1_idx.reshape(-1)
                close_idx = distances <= distance_threshold
                if np.all(~close_idx):
                    continue
                for o1, o2 in zip(obj1_idx[close_idx], np.arange(centers2.df.shape[0])[close_idx]):
                    obj1_id = centers1.df.loc[centers1.df.index[o1], aff_col]
                    obj2_id = centers2.df.loc[centers2.df.index[o2], aff_col]
                    ordered[j].df.loc[
                        (ordered[j].df[tomo_col] == t) & (ordered[j].df[aff_col] == obj2_id),
                        aff_col,
                    ] = obj1_id

        # Step 3: reconcile IR orientations (ring index 1 = IR)
        if len(ordered) >= 2:
            self._align_to_reference(ordered[1], ordered[0])

        # Step 4: merge into single motl, update state
        merged_df = pd.concat([rm.df for rm in ordered], ignore_index=True)
        if store_sid_column is not None:
            merged_df[store_sid_column] = merged_df["subtomo_id"].copy()
        self.motl = cryomotl.Motl(merged_df)
        self._rings_merged = True
        self._store_sid_column = store_sid_column
        self._ring_group_columns = [self.tomo_id_column, self.affiliation_column, self._ring_column]
        if store_sid_column is not None:
            self.motl.renumber_particles()
        self._finalize_subunit_order()
        return self.motl

    def update_ring_motls(self, flag_column: str = "geom5") -> None:
        """Propagate corrected subunit order from the merged motl back to _ring_motls.

        After manual corrections to the merged motl (e.g. fixing flagged pores
        with :meth:`renumber_subunits`), call this to sync the updated
        ``order_column`` values back into each source ring motl so they can be
        written out.  Uses the original ``subtomo_id`` values stored in
        ``_store_sid_column`` during :meth:`merge` to match each merged row to
        its source ring motl row.

        Parameters
        ----------
        flag_column : str, default ``'geom5'``
            When present in the merged motl, its values are also propagated back
            so the ring motls record which particles were flagged.

        Raises
        ------
        ValueError
            If :meth:`merge` has not been called, or if ``store_sid_column``
            was ``None`` during :meth:`merge` (making back-tracking impossible).
        """
        if not self._rings_merged:
            raise ValueError("update_ring_motls requires a merged NPC — call merge() first.")
        if self._store_sid_column is None:
            raise ValueError(
                "update_ring_motls requires store_sid_column to have been set during merge(). "
                "Re-run merge(store_sid_column=...) to enable back-tracking."
            )

        sid_col = self._store_sid_column
        order_col = self.order_column
        ring_col = self._ring_column
        merged = self.motl.df

        for i, rm in enumerate(self._ring_motls):
            ring_df = merged[merged[ring_col] == float(i + 1)]
            order_map = ring_df.set_index(sid_col)[order_col].to_dict()
            rm.df[order_col] = rm.df["subtomo_id"].map(order_map).fillna(rm.df[order_col])

            if flag_column in merged.columns:
                if flag_column not in rm.df.columns:
                    rm.df[flag_column] = 0.0
                flag_map = ring_df.set_index(sid_col)[flag_column].to_dict()
                rm.df[flag_column] = rm.df["subtomo_id"].map(flag_map).fillna(rm.df[flag_column])

    def write_ring_motls(self, path_prefix: str, suffix: str = ".em") -> None:
        """Write each ring motl to ``{path_prefix}{ring_index}{suffix}``.

        Parameters
        ----------
        path_prefix : str
            Directory + condition string, e.g. ``"/data/control_"``.
            Produces ``/data/control_0.em``, ``/data/control_1.em``, …
        suffix : str, default ``'.em'``
            File extension.
        """
        for i, rm in enumerate(self._ring_motls):
            rm.write_out(f"{path_prefix}{i}{suffix}")

    def split_by_ring(self, result: pd.DataFrame) -> list[pd.DataFrame]:
        """Split *result* into per-ring sub-DataFrames using :data:`_ring_column`.

        Returns a list of length ``len(_ring_motls)``, one element per ring.
        Each element contains only rows whose ``_ring_column`` value equals
        that ring's index.  If *result* has no ring column (e.g. single-ring
        NPC or a pre-dispatch call), the whole DataFrame is returned as a
        one-element list.

        This is the library-side of the send-to-motl splitting contract: the
        caller maps each returned DataFrame to the corresponding source motl
        pool entry.  No new matching logic is needed because the ring column
        records the origin of every particle.
        """
        if self._ring_column not in result.columns:
            return [result]
        parts = []
        for ring_idx in range(len(self._ring_motls)):
            mask = result[self._ring_column] == float(ring_idx + 1)
            parts.append(result[mask].copy())
        return parts

    @gui_exposed(label="Ring spacing", group="Statistics", order=40, returns="dataframe")
    def ring_spacing(self, *, pixel_size: float | None = None) -> pd.DataFrame:
        """Inter-ring spacing per NPC pore, computed from the merged motl.

        For every unordered pair of rings, computes the 3-D Euclidean distance
        between the mean coordinates of the two ring centres for each pore and
        scales by *pixel_size*.  Produces ``n*(n-1)/2`` columns for *n* rings.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor applied to the raw voxel distances.

        Returns
        -------
        pandas.DataFrame
            One row per ``(tomo_id, object_id)``.  Columns:

            ``tomo_id``, ``object_id``
                Pore identifiers.
            ``spacing_{r0}_{r1}``
                Distance (Å when *pixel_size* given) between the centre of ring
                *r0* and ring *r1*; one column per ring pair, including
                non-adjacent pairs.  ``NaN`` when one or both ring centres are
                absent for a pore.

        Raises
        ------
        ValueError
            If called before :meth:`merge`.
        """
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)
        if not self._rings_merged:
            raise ValueError("ring_spacing requires a merged NPC — call merge() first.")
        df = self.motl.df
        coord = self.motl.get_coordinates()
        ring_col = self._ring_column
        n_rings = len(self._ring_motls)

        centres: dict[tuple[float, float, float], np.ndarray] = {}
        for (tomo_id, object_id, ring_idx), group in df.groupby(
            [self.tomo_id_column, self.affiliation_column, ring_col]
        ):
            idx = group.index.to_numpy()
            coords_grp = coord[idx, :]
            centres[(float(tomo_id), float(object_id), float(ring_idx))] = (
                coords_grp.mean(axis=0) if coords_grp.shape[0] > 0 else np.zeros(3)
            )

        pore_keys: list[tuple[float, float]] = sorted({(k[0], k[1]) for k in centres})
        pairs = [tuple(sorted((a + 1, b + 1))) for a, b in mathutils.get_all_pairs(list(range(n_rings)))]
        rows: list[dict] = []
        for tomo_id, object_id in pore_keys:
            row: dict = {self.tomo_id_column: tomo_id, self.affiliation_column: object_id}
            for r0, r1 in pairs:
                key0 = (tomo_id, object_id, float(r0))
                key1 = (tomo_id, object_id, float(r1))
                if key0 in centres and key1 in centres:
                    dist = _ring_centre_spacing(centres[key0], centres[key1]) * ps
                else:
                    dist = float("nan")
                row[f"spacing_{r0}_{r1}"] = dist
            rows.append(row)
        return pd.DataFrame(rows)

    @gui_exposed(label="Inter-ring twist", group="Statistics", order=45, returns="dataframe")
    def inter_ring_twist(self, *, degrees: bool = True) -> pd.DataFrame:
        """Rotational twist between adjacent rings about the pore axis.

        For each pore and every unordered ring pair, the pore axis is estimated
        as the unit vector from ring *r0*'s centre to ring *r1*'s centre.
        Produces ``n*(n-1)/2`` columns for *n* rings.  The rotational twist is the
        signed n-fold circular-mean phase difference folded into ``[−π/n, π/n)``
        (``[−180/n, 180/n)`` degrees); 0 means the rings are rotationally aligned.

        Parameters
        ----------
        degrees : bool, default=True
            Return twist in degrees when ``True``, radians when ``False``.

        Returns
        -------
        pandas.DataFrame
            One row per ``(tomo_id, object_id)``.  Columns:

            ``tomo_id``, ``object_id``
                Pore identifiers.
            ``twist_{r0}_{r1}``
                Rotational twist between ring *r0* and ring *r1*; one column per
                ring pair, including non-adjacent pairs.  ``NaN`` when one or
                both rings are absent or when ring centres coincide.

        Raises
        ------
        ValueError
            If called before :meth:`merge`.
        """
        if not self._rings_merged:
            raise ValueError("inter_ring_twist requires a merged NPC — call merge() first.")
        df = self.motl.df
        coord = self.motl.get_coordinates()
        ring_col = self._ring_column
        n_rings = len(self._ring_motls)

        ring_positions: dict[tuple[float, float, float], np.ndarray] = {}
        for (tomo_id, object_id, ring_idx), group in df.groupby(
            [self.tomo_id_column, self.affiliation_column, ring_col]
        ):
            idx = group.index.to_numpy()
            ring_positions[(float(tomo_id), float(object_id), float(ring_idx))] = coord[idx, :]

        pore_keys = sorted({(k[0], k[1]) for k in ring_positions})
        pairs = [tuple(sorted((a + 1, b + 1))) for a, b in mathutils.get_all_pairs(list(range(n_rings)))]
        rows: list[dict] = []
        for tomo_id, object_id in pore_keys:
            row: dict = {self.tomo_id_column: tomo_id, self.affiliation_column: object_id}
            for r0, r1 in pairs:
                key0 = (tomo_id, object_id, float(r0))
                key1 = (tomo_id, object_id, float(r1))
                if key0 in ring_positions and key1 in ring_positions:
                    c0 = ring_positions[key0].mean(axis=0)
                    c1 = ring_positions[key1].mean(axis=0)
                    diff = c1 - c0
                    norm_val = np.linalg.norm(diff)
                    if norm_val < 1e-12:
                        twist = float("nan")
                    else:
                        axis = diff / norm_val
                        all_pts = np.concatenate([ring_positions[key0], ring_positions[key1]], axis=0)
                        bary = all_pts.mean(axis=0)
                        rel0 = ring_positions[key0] - bary
                        rel1 = ring_positions[key1] - bary
                        twist = _ring_centre_twist(rel0, rel1, self.n, axis, degrees=degrees)
                else:
                    twist = float("nan")
                row[f"twist_{r0}_{r1}"] = twist
            rows.append(row)
        return pd.DataFrame(rows)

    def flagged_pores(self, flag_column: str = "geom5") -> pd.DataFrame:
        """Return a summary of pores whose *flag_column* is set to 1.0.

        Each row of the returned :class:`~pandas.DataFrame` corresponds to
        one (tomo_id, object_id, ring) group.  The ``order_sequence`` column
        contains the sorted list of :attr:`order_column` values assigned to
        subunits in that ring, making it easy to spot missing or duplicated
        slots.

        Parameters
        ----------
        flag_column : str, default ``'geom5'``
            Column written by :meth:`assign_subunit_order` /
            :meth:`_finalize_subunit_order` to mark ambiguous pores.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``object_id``, ``ring``, ``order_sequence``.
            Empty when no pores are flagged.
        """
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column
        order_col = self.order_column
        df = self.motl.df

        if flag_column not in df.columns:
            return pd.DataFrame(columns=["tomo_id", "object_id", "ring", "order_sequence"])

        flagged_mask = df[flag_column] == 1.0
        if not flagged_mask.any():
            return pd.DataFrame(columns=["tomo_id", "object_id", "ring", "order_sequence"])

        rows: list[dict] = []
        for (tomo_id, object_id, ring), grp in df[flagged_mask].groupby(
            [tomo_col, aff_col, ring_col]
        ):
            rows.append(
                {
                    "tomo_id": tomo_id,
                    "object_id": object_id,
                    "ring": ring,
                    "order_sequence": sorted(grp[order_col].tolist()),
                }
            )
        return pd.DataFrame(rows)

    @gui_exposed(label="Subunit spacing", group="Statistics", order=50, returns="dataframe")
    def get_subunit_spacing(
        self,
        *,
        pixel_size: float = 1.0,
        degrees: bool = True,
        reference_ring: int | None = None,
        include_flagged: bool = False,
        flag_column: str = "geom5",
    ) -> pd.DataFrame:
        """Per-subunit spacing and twist between adjacent rings.

        For each subunit order *k*, reports the Euclidean distance and
        azimuthal twist between the corresponding subunit in each pair of
        adjacent rings.  The per-pore summary numbers from
        :meth:`ring_spacing` and :meth:`inter_ring_twist` remain available
        as aggregate columns.

        Ring centres are computed with :meth:`_compute_object_center`
        (respects *center_method*).

        Requires :meth:`merge` to have been called.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Voxel size in physical units; distances are multiplied by this.
        degrees : bool, default=True
            Return twist angles in degrees when ``True``, radians otherwise.
        reference_ring : int or None, default=None
            Ring whose subunit order defines the iteration.  ``None`` uses
            the first ring present in each pore.
        include_flagged : bool, default=False
            When ``False``, pores with *flag_column* == 1.0 are excluded from
            the output.
        flag_column : str, default ``'geom5'``
            Column checked for the per-pore ambiguity flag.

        Returns
        -------
        pandas.DataFrame
            One row per (pore, subunit).  Columns:

            - ``tomo_id``, ``object_id``, ``order_column`` (subunit index)
            - ``spacing_{r0}_{r1}`` — Euclidean distance (voxels × pixel_size)
              between subunit *k* in ring *r0* and ring *r1* for each ring pair,
              including non-adjacent pairs
            - ``twist_{r0}_{r1}`` — signed azimuthal angle (degrees) of
              subunit *k* in ring *r1* relative to its position in ring *r0*,
              measured from the ring-pair axis; one column per ring pair
        """
        if not self._rings_merged:
            raise ValueError("get_subunit_spacing requires a merged NPC — call merge() first.")

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column
        order_col = self.order_column
        df = self.motl.df
        if not include_flagged and flag_column in df.columns:
            _flagged = set(
                map(tuple, df.loc[df[flag_column] == 1.0, [tomo_col, aff_col]].values.tolist())
            )
            if _flagged:
                df = df[~df[[tomo_col, aff_col]].apply(tuple, axis=1).isin(_flagged)]
        coord = self.motl.get_coordinates()
        n_rings = len(self._ring_motls)
        pairs = [tuple(sorted((a + 1, b + 1))) for a, b in mathutils.get_all_pairs(list(range(n_rings)))]

        rows: list[dict] = []
        for (tomo_id, object_id), pore_grp in df.groupby([tomo_col, aff_col]):
            # Ring centres (respects center_method) and per-(ring, order) position map
            ring_centres: dict[float, np.ndarray] = {}
            for ring_id, rg in pore_grp.groupby(ring_col):
                ring_motl_tmp = cryomotl.Motl(rg.copy())
                centre_tmp, _ = self._compute_object_center(ring_motl_tmp)
                ring_centres[float(ring_id)] = centre_tmp

            pos_map: dict[tuple[float, float], np.ndarray] = {}
            for global_i, row_s in pore_grp.iterrows():
                pos_map[(float(row_s[ring_col]), float(row_s[order_col]))] = coord[global_i]

            # Subunit orders defined by reference_ring (default: first ring present)
            _ref = (
                float(reference_ring)
                if reference_ring is not None
                else float(sorted(pore_grp[ring_col].unique())[0])
            )
            all_k = sorted(pore_grp.loc[pore_grp[ring_col] == _ref, order_col].unique())

            for k in all_k:
                row: dict = {
                    tomo_col: float(tomo_id),
                    aff_col: float(object_id),
                    order_col: float(k),
                }
                for r0, r1 in pairs:
                    key0 = (float(r0), float(k))
                    key1 = (float(r1), float(k))
                    if key0 in pos_map and key1 in pos_map:
                        p0, p1 = pos_map[key0], pos_map[key1]
                        row[f"spacing_{r0}_{r1}"] = float(np.linalg.norm(p1 - p0)) * pixel_size
                        # Azimuthal twist: angle of p1 relative to p0 around the pore axis
                        c0 = ring_centres.get(float(r0), np.zeros(3))
                        c1 = ring_centres.get(float(r1), np.zeros(3))
                        axis = c1 - c0
                        axis_norm = float(np.linalg.norm(axis))
                        if axis_norm > 1e-10:
                            axis = axis / axis_norm
                            rel0 = p0 - c0
                            rel1 = p1 - c1
                            proj0 = rel0 - np.dot(rel0, axis) * axis
                            proj1 = rel1 - np.dot(rel1, axis) * axis
                            n0 = float(np.linalg.norm(proj0))
                            n1 = float(np.linalg.norm(proj1))
                            if n0 > 1e-10 and n1 > 1e-10:
                                twist_rad = geom.vector_angular_distance_signed(proj0 / n0, proj1 / n1, axis)
                                row[f"twist_{r0}_{r1}"] = float(np.degrees(twist_rad)) if degrees else float(twist_rad)
                            else:
                                row[f"twist_{r0}_{r1}"] = float("nan")
                        else:
                            row[f"twist_{r0}_{r1}"] = float("nan")
                    else:
                        row[f"spacing_{r0}_{r1}"] = float("nan")
                        row[f"twist_{r0}_{r1}"] = float("nan")
                rows.append(row)

        return pd.DataFrame(rows)

    @gui_exposed(label="Subunit orientation", group="Statistics", order=51, returns="dataframe")
    def get_subunit_orientation(
        self,
        *,
        degrees: bool = True,
        reference_ring: int | None = None,
        include_flagged: bool = False,
        flag_column: str = "geom5",
    ) -> pd.DataFrame:
        """Per-subunit orientation difference between ring pairs.

        For each subunit order *k* matched across rings, reports the cone
        (out-of-plane normal deviation) and in-plane (roll) differences between
        each ring pair, via :func:`geom.cone_inplane_distance`.

        Requires :meth:`merge` to have been called.

        Parameters
        ----------
        degrees : bool, default=True
            Return angles in degrees when ``True``, radians otherwise.
        reference_ring : int or None, default=None
            Ring whose subunit order defines the iteration.  ``None`` uses
            the first ring present in each pore.
        include_flagged : bool, default=False
            When ``False``, pores with *flag_column* == 1.0 are excluded.
        flag_column : str, default ``'geom5'``
            Column checked for the per-pore ambiguity flag.

        Returns
        -------
        pandas.DataFrame
            One row per (pore, subunit).  Columns:

            - ``tomo_id``, ``affiliation_column``, ``order_column``
            - ``cone_{r0}_{r1}`` — cone (normal) angular difference for ring
              pair; ``NaN`` when the subunit is absent in either ring
            - ``inplane_{r0}_{r1}`` — in-plane (roll) angular difference;
              ``NaN`` when the subunit is absent in either ring
        """
        if not self._rings_merged:
            raise ValueError(
                "get_subunit_orientation requires a merged NPC — call merge() first."
            )

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column
        order_col = self.order_column
        df = self.motl.df
        if not include_flagged and flag_column in df.columns:
            _flagged = set(
                map(tuple, df.loc[df[flag_column] == 1.0, [tomo_col, aff_col]].values.tolist())
            )
            if _flagged:
                df = df[~df[[tomo_col, aff_col]].apply(tuple, axis=1).isin(_flagged)]
        all_angles = self.motl.get_angles()
        n_rings = len(self._ring_motls)
        pairs = [
            tuple(sorted((a + 1, b + 1)))
            for a, b in mathutils.get_all_pairs(list(range(n_rings)))
        ]

        rows: list[dict] = []
        for (tomo_id, object_id), pore_grp in df.groupby([tomo_col, aff_col]):
            angles_map: dict[tuple[float, float], np.ndarray] = {}
            for global_i, row_s in pore_grp.iterrows():
                angles_map[(float(row_s[ring_col]), float(row_s[order_col]))] = all_angles[global_i]

            _ref = (
                float(reference_ring)
                if reference_ring is not None
                else float(sorted(pore_grp[ring_col].unique())[0])
            )
            all_k = sorted(pore_grp.loc[pore_grp[ring_col] == _ref, order_col].unique())

            for k in all_k:
                row: dict = {
                    tomo_col: float(tomo_id),
                    aff_col: float(object_id),
                    order_col: float(k),
                }
                for r0, r1 in pairs:
                    key0 = (float(r0), float(k))
                    key1 = (float(r1), float(k))
                    if key0 in angles_map and key1 in angles_map:
                        cone_val, inplane_val = geom.cone_inplane_distance(
                            angles_map[key0], angles_map[key1],
                            degrees=degrees, cyclic_symmetry=1,
                        )
                        row[f"cone_{r0}_{r1}"] = float(np.asarray(cone_val).flat[0])
                        row[f"inplane_{r0}_{r1}"] = float(np.asarray(inplane_val).flat[0])
                    else:
                        row[f"cone_{r0}_{r1}"] = float("nan")
                        row[f"inplane_{r0}_{r1}"] = float("nan")
                rows.append(row)

        return pd.DataFrame(rows)

    @gui_exposed(label="Subunit stats", group="Statistics", order=55, returns="dataframe")
    def get_subunit_stats(
        self,
        *,
        pixel_size: float | None = None,
        degrees: bool = True,
        include_flagged: bool = False,
        flag_column: str = "geom5",
        gaps: str = "full",
    ) -> pd.DataFrame:
        """Per-subunit geometry statistics.

        Computes local geometry for every particle in the NPC motl.
        Prev/next neighbours within each ring are found with
        :meth:`nnana.NearestNeighbors.ordered_pairs` (circular topology),
        which respects the assigned subunit order rather than distance.
        Ring centres use :meth:`_compute_object_center` (respects
        *center_method*).  Z-normals use :func:`geom.euler_angles_to_normals`.

        Returns
        -------
        pandas.DataFrame
            One row per particle.  Columns:

            - ``tomo_id``, ``object_id``, ring_column (post-merge),
              ``order_column``, ``subtomo_id``
            - ``distance_to_centre`` — distance from particle to ring
              centre (voxels × *pixel_size*)
            - ``tilt_angle`` — angle (°) between particle z-normal and
              ring mean normal
            - ``central_angle_prev``, ``central_angle_next`` — unsigned
              angle (°) at ring centre between this particle and each
              order-adjacent neighbour; ``NaN`` when the required
              order-adjacent slot is absent and *gaps* is ``'full'``
            - ``interior_angle`` — unsigned vertex angle (°) at this particle:
              the angle between the two edge vectors meeting at this vertex,
              in [0°, 180°], direction-independent.  ``NaN`` when either
              neighbour is absent.  Cannot distinguish a genuine reflex
              concave vertex (it folds to ≤180°); real pores are convex so
              this is intentional.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Voxel size; distances are multiplied by this.
        degrees : bool, default=True
            Return angles in degrees when ``True``, radians otherwise.
        include_flagged : bool, default=False
            When ``False``, pores with *flag_column* == 1.0 are excluded from
            the output.
        flag_column : str, default ``'geom5'``
            Column checked for the per-pore ambiguity flag.
        gaps : ``'full'`` or ``'holey'``, default ``'full'``
            Passed directly to :meth:`nnana.NearestNeighbors.ordered_pairs`.

            ``'full'`` — a subunit pairs only with the subunit whose order
            value is exactly order ± 1 (circular).  If that slot is absent
            (milling-truncated edge, internal gap), there is no neighbour on
            that side and the corresponding ``central_angle_*`` and
            ``interior_angle`` are ``NaN``.

            ``'holey'`` — a subunit pairs with the previous/next *present*
            subunit by order, skipping over absent slots.  Every subunit in
            a ring with ≥ 2 present members gets both neighbours, so no
            gap-``NaN`` arises; the angle across a skipped gap reflects the
            actual spacing, not the ideal 45°.
        """
        if gaps not in ("full", "holey"):
            raise ValueError(f"gaps must be 'full' or 'holey', got {gaps!r}.")
        gaps_arg = gaps
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column
        order_col = self.order_column
        df = self.motl.df
        if not include_flagged and flag_column in df.columns:
            _flagged = set(
                map(tuple, df.loc[df[flag_column] == 1.0, [tomo_col, aff_col]].values.tolist())
            )
            if _flagged:
                df = df[~df[[tomo_col, aff_col]].apply(tuple, axis=1).isin(_flagged)]
        coord = self.motl.get_coordinates()

        group_cols = self._ring_group_columns
        has_ring_group = len(group_cols) > 2

        rows: list[dict] = []
        for keys, ring_grp in df.groupby(group_cols):
            if has_ring_group:
                tomo_id, object_id, ring_id = keys
            else:
                tomo_id, object_id = keys
                ring_id = None

            idx = ring_grp.index.to_numpy()
            coords_ring = coord[idx]

            ring_motl_tmp = cryomotl.Motl(ring_grp.copy())
            centre, _ = self._compute_object_center(ring_motl_tmp)

            # Ring mean normal via geom (no manual rotation maths)
            angles_ring = ring_motl_tmp.get_angles()
            z_vecs = geom.euler_angles_to_normals(angles_ring)
            mean_normal = z_vecs.mean(axis=0)
            n_norm = float(np.linalg.norm(mean_normal))
            mean_normal = mean_normal / n_norm if n_norm > 1e-10 else mean_normal

            # Prev/next by order within ring (topology-aware, respects missing subunits)
            n_pts = len(idx)
            next_local_map: dict[int, int] = {}
            prev_local_map: dict[int, int] = {}
            if n_pts >= 2:
                op = nnana.NearestNeighbors.ordered_pairs(
                    ring_motl_tmp,
                    tomo_col,
                    order_col,
                    topology="circular",
                    ring_size=self.n,
                    gaps=gaps_arg,
                )
                sids = ring_grp["subtomo_id"].to_numpy()
                sid_to_local = {float(s): i for i, s in enumerate(sids)}
                for _, pr in op.df.iterrows():
                    qi = sid_to_local.get(float(pr["qp_subtomo_id"]))
                    ni = sid_to_local.get(float(pr["nn_subtomo_id"]))
                    if qi is not None and ni is not None:
                        next_local_map[qi] = ni
                        prev_local_map[ni] = qi

            for local_i, global_i in enumerate(idx):
                row_s = ring_grp.loc[global_i]
                pos = coords_ring[local_i]
                dist = float(np.linalg.norm(pos - centre)) * ps

                tilt = float(geom.vector_angular_distance(z_vecs[local_i], mean_normal))
                if not degrees:
                    tilt = np.radians(tilt)

                ca_prev = float("nan")
                ca_next = float("nan")
                interior = float("nan")

                if n_pts >= 2:
                    pi_ = prev_local_map.get(local_i)
                    ni_ = next_local_map.get(local_i)

                    v_self = pos - centre
                    vn_s = float(np.linalg.norm(v_self))

                    if pi_ is not None:
                        pos_prev = coords_ring[pi_]
                        v_prev = pos_prev - centre
                        vn_p = float(np.linalg.norm(v_prev))
                        if vn_s > 1e-10 and vn_p > 1e-10:
                            ca_prev = float(geom.vector_angular_distance(v_self / vn_s, v_prev / vn_p))
                            if not degrees:
                                ca_prev = np.radians(ca_prev)

                    if ni_ is not None:
                        pos_next = coords_ring[ni_]
                        v_next = pos_next - centre
                        vn_n = float(np.linalg.norm(v_next))
                        if vn_s > 1e-10 and vn_n > 1e-10:
                            ca_next = float(geom.vector_angular_distance(v_self / vn_s, v_next / vn_n))
                            if not degrees:
                                ca_next = np.radians(ca_next)

                    if pi_ is not None and ni_ is not None:
                        a_vec = pos_prev - pos
                        b_vec = pos_next - pos
                        na = float(np.linalg.norm(a_vec))
                        nb = float(np.linalg.norm(b_vec))
                        if na > 1e-10 and nb > 1e-10:
                            interior = float(geom.vector_angular_distance(a_vec / na, b_vec / nb))
                            if not degrees:
                                interior = np.radians(interior)

                row: dict = {
                    tomo_col: float(tomo_id),
                    aff_col: float(object_id),
                    order_col: float(row_s[order_col]),
                    "subtomo_id": float(row_s["subtomo_id"]),
                    "distance_to_centre": dist,
                    "tilt_angle": tilt,
                    "central_angle_prev": ca_prev,
                    "central_angle_next": ca_next,
                    "interior_angle": interior,
                }
                if has_ring_group:
                    row[ring_col] = float(ring_id)
                rows.append(row)

        return pd.DataFrame(rows)

    def ring_ellipticity(
        self,
        *,
        pixel_size: float | None = None,
        flag_column: str = "geom5",
        include_flagged: bool = False,
    ) -> pd.DataFrame:
        """Ellipticity of each ring from a least-squares 2-fold radial fit.

        For each ring, particles are projected onto the ring plane (defined by
        :meth:`_compute_object_center` and the mean z-normal) and the in-plane
        radii fitted to ``r = c0 + c2·cos(2θ) + s2·sin(2θ)`` by least squares.
        The 2-fold amplitude ``A = hypot(c2, s2)`` characterises the ellipse.

        Rings with fewer than 5 particles or azimuthal coverage below 180° are
        flagged (``partial_arc`` == 1) and metric columns set to NaN, as partial
        arcs produce biased 2-fold estimates.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Voxel size; ``major`` and ``minor`` are multiplied by this.
        flag_column : str, default ``'geom5'``
            Column checked for the per-pore ambiguity flag.
        include_flagged : bool, default=False
            When ``False``, pores with *flag_column* == 1.0 are excluded.

        Returns
        -------
        pandas.DataFrame
            One row per ring group (``tomo_id``, ``object_id``, ring_column
            when post-merge).  Columns:

            - ``n_used`` — number of particles in the ring
            - ``ellipticity`` — fractional 2-fold amplitude ``A / c0``
              (0 = circle)
            - ``axis_ratio`` — minor/major ratio ``(c0 − A) / (c0 + A)``
            - ``major``, ``minor`` — semi-axes (voxels × *pixel_size*)
            - ``orientation`` — major-axis angle (°) in the ring plane,
              ``0.5 · atan2(s2, c2)``
            - ``partial_arc`` — 1 if coverage < 180° or n_used < 5, else 0
        """
        ps: float = getattr(self, "pixel_size", 1.0) if pixel_size is None else float(pixel_size)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        ring_col = self._ring_column
        df = self.motl.df
        if not include_flagged and flag_column in df.columns:
            _flagged = set(
                map(tuple, df.loc[df[flag_column] == 1.0, [tomo_col, aff_col]].values.tolist())
            )
            if _flagged:
                df = df[~df[[tomo_col, aff_col]].apply(tuple, axis=1).isin(_flagged)]
        coord = self.motl.get_coordinates()

        group_cols = self._ring_group_columns
        has_ring_group = len(group_cols) > 2

        rows: list[dict] = []
        for keys, ring_grp in df.groupby(group_cols):
            if has_ring_group:
                tomo_id, object_id, ring_id = keys
            else:
                tomo_id, object_id = keys
                ring_id = None

            idx = ring_grp.index.to_numpy()
            coords_ring = coord[idx]
            n_used = len(idx)

            ring_motl_tmp = cryomotl.Motl(ring_grp.copy())
            centre, _ = self._compute_object_center(ring_motl_tmp)

            angles_ring = ring_motl_tmp.get_angles()
            z_vecs = geom.euler_angles_to_normals(angles_ring)
            mean_normal = z_vecs.mean(axis=0)
            n_norm = float(np.linalg.norm(mean_normal))
            mean_normal = mean_normal / n_norm if n_norm > 1e-10 else mean_normal

            row: dict = {
                tomo_col: float(tomo_id),
                aff_col: float(object_id),
                "n_used": n_used,
                "ellipticity": float("nan"),
                "axis_ratio": float("nan"),
                "major": float("nan"),
                "minor": float("nan"),
                "orientation": float("nan"),
                "partial_arc": 1,
            }
            if has_ring_group:
                row[ring_col] = float(ring_id)

            if n_used < 5:
                rows.append(row)
                continue

            # Project onto ring plane (subtract axial component along mean_normal)
            rel = coords_ring - centre
            axial = np.dot(rel, mean_normal)[:, None] * mean_normal
            inplane = rel - axial  # (n, 3)

            inplane_norms = np.linalg.norm(inplane, axis=1)
            if inplane_norms[0] < 1e-10:
                rows.append(row)
                continue
            ref_dir = inplane[0] / inplane_norms[0]

            # Signed azimuths relative to first particle (radians)
            theta = np.zeros(n_used)
            for i in range(1, n_used):
                if inplane_norms[i] > 1e-10:
                    theta[i] = geom.vector_angular_distance_signed(
                        ref_dir, inplane[i] / inplane_norms[i], mean_normal
                    )

            # Azimuthal coverage: 360° minus the largest gap between sorted azimuths
            theta_sorted = np.sort(theta % (2.0 * np.pi))
            gaps = np.diff(np.append(theta_sorted, theta_sorted[0] + 2.0 * np.pi))
            coverage_deg = 360.0 - float(np.degrees(np.max(gaps)))
            if coverage_deg < 180.0:
                rows.append(row)
                continue

            # 2-fold least-squares fit: r = c0 + c2*cos(2θ) + s2*sin(2θ)
            r_i = inplane_norms
            two_theta = 2.0 * theta
            A_mat = np.column_stack([np.ones(n_used), np.cos(two_theta), np.sin(two_theta)])
            coeffs, *_ = np.linalg.lstsq(A_mat, r_i, rcond=None)
            c0, c2, s2 = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

            if c0 < 1e-10:
                rows.append(row)
                continue

            amp = float(np.hypot(c2, s2))
            row["ellipticity"] = amp / c0
            row["axis_ratio"] = (c0 - amp) / (c0 + amp) if (c0 + amp) > 1e-10 else float("nan")
            row["major"] = (c0 + amp) * ps
            row["minor"] = (c0 - amp) * ps
            row["orientation"] = float(np.degrees(0.5 * np.arctan2(s2, c2)))
            row["partial_arc"] = 0

            rows.append(row)

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------
    # NPC-specific private helpers (radius-shift centre estimation)
    # ------------------------------------------------------------------

    @staticmethod
    def _center_by_radius_shift(
        object_motl: "cryomotl.Motl",
        npc_radius: float,
    ) -> np.ndarray:
        """Estimate ring centre by shifting each subunit inward by *npc_radius*.

        Shifts every particle by ``(-npc_radius, 0, 0)`` along its local X
        axis (i.e. toward the pore centre) and returns the mean of the
        resulting positions.  Works for any number of subunits including 1.

        Parameters
        ----------
        object_motl : cryomotl.Motl
            Subunit particles belonging to one ring.
        npc_radius : float
            Approximate ring radius in voxels.

        Returns
        -------
        numpy.ndarray
            Estimated centre, shape ``(3,)``.
        """
        shifted = cryomotl.Motl(object_motl.df.copy())
        shifted.shift_positions(np.asarray([-npc_radius, 0.0, 0.0]))
        return np.mean(shifted.get_coordinates(), axis=0)

    @staticmethod
    def _assign_subunit_index(
        object_motl: "cryomotl.Motl",
        npc_radius: float,
        symmetry: int = 8,
    ) -> list[int]:
        """Assign 1-based angular subunit indices for a merged NPC ring.

        Computes each subunit's angle relative to the first one using the
        radius-shift centre estimate, divides by ``360 / symmetry``, and
        rounds to the nearest integer.

        Parameters
        ----------
        object_motl : cryomotl.Motl
            Subunit particles of the merged ring.
        npc_radius : float
            Approximate ring radius in voxels.
        symmetry : int, default=8
            Rotational symmetry order.

        Returns
        -------
        list of int
            1-based subunit indices, same length as ``object_motl.df``.
        """
        center = NPC._center_by_radius_shift(object_motl, npc_radius)
        coords = object_motl.get_coordinates()
        vectors = coords - center
        div_angle = 360.0 / symmetry
        s_idx = [1]
        for vec in vectors[1:]:
            angle = geom.vector_angular_distance(vectors[0], vec) / div_angle
            s_idx.append(int(decimal.Decimal(angle).to_integral_value(rounding=decimal.ROUND_HALF_UP)) + 1)
        return s_idx

    @staticmethod
    def _merge_by_radius(
        motl: "cryomotl.Motl",
        npc_radius: float,
        *,
        tomo_id_column: "MotlColumn" = "tomo_id",
        affiliation_column: "MotlColumn" = "object_id",
    ) -> "cryomotl.Motl":
        """Merge NPC chains whose radius-shift centres are within *npc_radius*.

        Restores the original ``NPC.merge_subunits`` behaviour: centre
        estimation uses :meth:`_center_by_radius_shift` so that even single-
        particle chains correctly converge to the ring centre, enabling robust
        merging.  Sets ``geom1`` to the merged group count and recomputes
        ``geom2`` (subunit index) for any rings that were actually merged.

        Parameters
        ----------
        motl : cryomotl.Motl
            Chain-traced motl with *affiliation_column* and ``geom2`` populated.
        npc_radius : float
            Distance threshold for merging (voxels).
        tomo_id_column : str, default='tomo_id'
            Column holding the tomogram identifier.
        affiliation_column : str, default='object_id'
            Column holding the ring/object identifier.

        Returns
        -------
        cryomotl.Motl
            Updated motl with consolidated ring labels.
        """
        for t in motl.get_unique_values(tomo_id_column):
            tm = motl.get_motl_subset(column_values=[t], column_name=tomo_id_column, reset_index=True)

            # Build centres motl using radius-shift approach
            central_points: list[np.ndarray] = []
            obj_ids: list[float] = []
            for o in tm.get_unique_values(affiliation_column):
                om = tm.get_motl_subset(column_values=[o], column_name=affiliation_column, reset_index=True)
                central_points.append(NPC._center_by_radius_shift(om, npc_radius))
                obj_ids.append(o)

            centers_motl = cryomotl.Motl()
            if central_points:
                ca = np.vstack(central_points)
                centers_motl.fill({
                    "x": ca[:, 0], "y": ca[:, 1], "z": ca[:, 2],
                    tomo_id_column: t, affiliation_column: obj_ids,
                })
                centers_motl.renumber_particles()
            centers_motl.df.fillna(0.0, inplace=True)

            changed_objects: list[float] = []
            if centers_motl.df.shape[0] > 1:
                center_stats = nnana.get_nn_stats(centers_motl, centers_motl)
                if any(center_stats["distance"] <= npc_radius):
                    center_idx, nn_idx_list = nnana.get_nn_within_distance(centers_motl, npc_radius)
                    for i, pos in enumerate(center_idx):
                        o_id1 = centers_motl.df.loc[centers_motl.df.index[pos], affiliation_column]
                        changed_objects.append(o_id1)
                        for j in nn_idx_list[i]:
                            o_id2 = centers_motl.df.loc[centers_motl.df.index[j], affiliation_column]
                            tm.df.loc[tm.df[affiliation_column] == o_id2, affiliation_column] = o_id1

            tm.df["geom1"] = tm.df.groupby(affiliation_column)[affiliation_column].transform("count")
            for o in changed_objects:
                om = tm.get_motl_subset(column_values=o, column_name=affiliation_column, reset_index=True)
                s_idx = NPC._assign_subunit_index(om, npc_radius)
                tm.df.loc[tm.df[affiliation_column] == o, "geom2"] = s_idx

            tm.df[affiliation_column] = tm.df[affiliation_column].rank(method="dense").astype(int)
            motl.df.loc[motl.df[tomo_id_column] == t, [affiliation_column, "geom1", "geom2"]] = tm.df[
                [affiliation_column, "geom1", "geom2"]
            ].values

        motl.df.reset_index(inplace=True, drop=True)
        motl.df["geom1"] = motl.df.groupby([tomo_id_column, affiliation_column])[affiliation_column].transform("count")
        motl.df[affiliation_column] = motl.df[affiliation_column].rank(method="dense").astype(int)
        return motl

    @staticmethod
    def compute_diameter(
        input_motl: MotlSource,
        *,
        pixel_size: float = 1.0,
        store_column: MotlColumn = "geom4",
        symmetry: int = 8,
        tomo_id_column: "MotlColumn" = "tomo_id",
        affiliation_column: "MotlColumn" = "object_id",
    ) -> tuple[pd.DataFrame, "cryomotl.Motl"]:
        """Compute the mean NPC diameter per ring using opposite-subunit pairs.

        Matches subunit pairs ``(i, i + symmetry//2)`` using the 1-based
        index stored in ``geom2``.  Objects with no matching opposite pair
        are omitted from the summary and receive ``NaN`` in *store_column*.
        Unlike :meth:`CnComplex.diameter`, no circumradius fallback is
        applied.

        Parameters
        ----------
        input_motl : MotlSource
            Particle list with NPC subunits.  Requires *affiliation_column*
            for ring affiliation and ``geom2`` for the 1-based subunit order
            within each ring.
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor applied to all distances.
        store_column : MotlColumn, default='geom4'
            Column in the returned motl that carries each ring's mean
            diameter.  ``NaN`` for rings with no opposite-pair matches.
        symmetry : int, default=8
            Rotational symmetry order.  Determines the pair offset
            ``symmetry // 2``.
        tomo_id_column : str, default='tomo_id'
            Column holding the tomogram identifier.
        affiliation_column : str, default='object_id'
            Column holding the ring/object identifier.

        Returns
        -------
        summary_df : pandas.DataFrame
            One row per ``(tomo_id_column, affiliation_column)`` that produced
            at least one opposite-subunit pair.  Columns:
            *tomo_id_column*, *affiliation_column*, ``mean_diameter``,
            ``n_pairs``.  Empty when no ring has matching pairs.
        motl_out : Motl
            Copy of *input_motl* with *store_column* populated; ``NaN``
            for rings without pairs.
        """
        motl = cryomotl.Motl.load(input_motl)
        motl_out = cryomotl.Motl(motl.df.copy())
        motl_out.df.reset_index(drop=True, inplace=True)

        coord = motl_out.get_coordinates()
        diameters_col = np.full(len(motl_out.df), np.nan)
        half = symmetry // 2
        rows: list[dict] = []

        for (tomo_id, object_id), group in motl_out.df.groupby([tomo_id_column, affiliation_column]):
            grp_idx = group.index.to_numpy()
            pair_rows: list[list[int]] = []
            for i in range(1, half + 1):
                j = i + half
                mask_i = group["geom2"] == i
                mask_j = group["geom2"] == j
                if mask_i.any() and mask_j.any():
                    pair_rows.append(
                        [
                            group.index[mask_i][0],
                            group.index[mask_j][0],
                        ]
                    )
            if not pair_rows:
                continue
            idx = np.asarray(pair_rows)
            dists = geom.point_pairwise_dist(coord[idx[:, 0], :], coord[idx[:, 1], :]) * pixel_size
            mean_d = float(np.mean(dists))
            diameters_col[grp_idx] = mean_d
            rows.append(
                {
                    tomo_id_column: float(tomo_id),
                    affiliation_column: float(object_id),
                    "mean_diameter": mean_d,
                    "n_pairs": int(len(dists)),
                }
            )

        motl_out.df[store_column] = diameters_col
        summary_df = pd.DataFrame(
            rows if rows else [],
            columns=[tomo_id_column, affiliation_column, "mean_diameter", "n_pairs"],
        )
        return summary_df, motl_out

    @staticmethod
    def get_centers_as_motl(
        tomo_motl: MotlSource,
        *,
        tomo_id: float | None = None,
        radius: float | None = None,
        tomo_id_column: "MotlColumn" = "tomo_id",
        affiliation_column: "MotlColumn" = "object_id",
    ) -> "cryomotl.Motl":
        """Return one centre particle per ring using the radius-shift estimator.

        Shifts each subunit by ``(-radius, 0, 0)`` along its local X axis
        and averages the resulting positions to estimate the NPC ring centre.
        Unlike the inherited :meth:`CnComplex.get_centers_as_motl`, this
        method works correctly for any ring occupancy including a single
        subunit.

        Parameters
        ----------
        tomo_motl : MotlSource
            Particle list for one tomogram (or all tomograms).
        tomo_id : float, optional
            Tomogram identifier stored in the output motl.  Defaults to the
            *tomo_id_column* value found on each ring's particles.
        radius : float, default=55.0
            Approximate NPC ring radius in voxels.
        tomo_id_column : str, default='tomo_id'
            Column holding the tomogram identifier.
        affiliation_column : str, default='object_id'
            Column holding the ring/object identifier.

        Returns
        -------
        Motl
            One row per unique *affiliation_column* with the estimated ring
            centre in ``x``, ``y``, ``z``.
        """
        if radius is None:
            raise ValueError(
                "NPC.get_centers_as_motl: radius is required — pass radius="
            )
        motl = cryomotl.Motl.load(tomo_motl)
        centers: list[np.ndarray] = []
        tomo_ids: list[float] = []
        object_ids: list[float] = []

        for o in motl.get_unique_values(affiliation_column):
            om = motl.get_motl_subset(column_values=[o], column_name=affiliation_column, reset_index=True)
            center = NPC._center_by_radius_shift(om, npc_radius=radius)
            centers.append(center)
            t = float(tomo_id) if tomo_id is not None else float(om.df[tomo_id_column].iloc[0])
            tomo_ids.append(t)
            object_ids.append(float(o))

        result = cryomotl.Motl()
        if centers:
            pts_arr = np.vstack(centers)
            result.fill(
                {
                    "x": pts_arr[:, 0],
                    "y": pts_arr[:, 1],
                    "z": pts_arr[:, 2],
                    tomo_id_column: tomo_ids,
                    affiliation_column: object_ids,
                }
            )
            result.renumber_particles()
        result.df.fillna(0.0, inplace=True)
        return result

    @staticmethod
    def merge_rings(
        input_motls: list[MotlSource],
        npc_radius: float | None = None,
        distance_threshold: float = 40,
        *,
        tomo_id_column: "MotlColumn" = "tomo_id",
        affiliation_column: "MotlColumn" = "object_id",
    ) -> list["cryomotl.Motl"]:
        """Merge corresponding rings across multiple ring-motls.

        Assigns sequential *affiliation_column* values across all motls, then
        for every pair of motls finds rings (by their estimated centres) that
        are closer than *distance_threshold* and unifies their
        *affiliation_column* entries so matched rings share the same identifier.

        Parameters
        ----------
        input_motls : list of MotlSource
            At least two ring-motls to merge.
        npc_radius : float
            Ring radius in voxels, forwarded to :meth:`get_centers_as_motl`.
        distance_threshold : float, default=40
            Maximum centre-to-centre distance (voxels) for two rings from
            different motls to be considered the same NPC.
        tomo_id_column : MotlColumn, default="tomo_id"
            Column used to group particles by tomogram.
        affiliation_column : MotlColumn, default="object_id"
            Column used to identify which NPC pore a particle belongs to.

        Returns
        -------
        list of Motl
            The input motls with updated *affiliation_column* values so that
            matched rings share the same identifier.

        Raises
        ------
        UserWarning
            When *input_motls* is not a list or contains fewer than two items.
        """
        if npc_radius is None:
            raise ValueError(
                "NPC.merge_rings: npc_radius is required — pass npc_radius="
            )
        if not isinstance(input_motls, list) or len(input_motls) <= 1:
            raise UserWarning(
                "The input has to be list of valid motl specifications and has to contain more than one element!"
            )

        ring_motls = []
        for m in input_motls:
            if isinstance(m, (str, pd.DataFrame)):
                ring_motls.append(cryomotl.Motl.load(m))
            else:
                ring_motls.append(m)

        starting_number = 1
        for r in ring_motls:
            r.renumber_objects_sequentially(starting_number=starting_number)
            starting_number = int(r.df[affiliation_column].max()) + 1

        ring_pairs = mathutils.get_all_pairs(list(range(len(ring_motls))))

        for i in ring_pairs:
            for t in ring_motls[i[0]].get_unique_values(tomo_id_column):
                tm1 = ring_motls[i[0]].get_motl_subset(column_values=[t], column_name=tomo_id_column, reset_index=True)
                tm2 = ring_motls[i[1]].get_motl_subset(column_values=[t], column_name=tomo_id_column, reset_index=True)
                if tm2.df.shape[0] > 0:
                    centers1 = NPC.get_centers_as_motl(tm1, radius=npc_radius)
                    centers2 = NPC.get_centers_as_motl(tm2, radius=npc_radius)

                    _, obj1_idx, distances, _ = nnana.find_nn_indices(
                        centers2.get_coordinates(),
                        centers1.get_coordinates(),
                        k=1,
                    )
                    distances = distances.reshape(-1)
                    obj1_idx = obj1_idx.reshape(-1)

                    close_idx = distances <= distance_threshold
                    if np.all(~close_idx):
                        continue
                    obj1_idx = obj1_idx[close_idx]
                    obj2_idx = np.arange(centers2.df.shape[0])[close_idx]
                    for o1, o2 in zip(obj1_idx, obj2_idx):
                        obj1_id = centers1.df.loc[centers1.df.index[o1], affiliation_column]
                        obj2_id = centers2.df.loc[centers2.df.index[o2], affiliation_column]
                        ring_motls[i[1]].df.loc[
                            (ring_motls[i[1]].df[tomo_id_column] == t)
                            & (ring_motls[i[1]].df[affiliation_column] == obj2_id),
                            affiliation_column,
                        ] = obj1_id

        return ring_motls


def _npc_per_ring_wrapper(method_name: str):
    """Build a routing wrapper for one entry in ``NPC._DISPATCH_PER_RING``.

    Pre-merge direct calls (``_rings_merged=False`` and ``_per_ring_active=False``)
    are routed through :meth:`NPC.per_ring`.  All other calls (post-merge, or
    calls from inside an active per_ring dispatch) fall through to the base-class
    implementation via ``super(NPC, self)``.
    """

    def _method(self, *args, **kwargs):
        if not self._rings_merged and not self._per_ring_active:
            return self.per_ring(method_name, *args, **kwargs)
        return getattr(super(NPC, self), method_name)(*args, **kwargs)

    _method.__name__ = method_name
    _method.__qualname__ = f"NPC.{method_name}"
    return _method


for _name in NPC._DISPATCH_PER_RING - {"occupancy", "get_object_stats", "assign_subunit_order"}:
    # Skip static/class methods — they don't dispatch on self.motl
    if not isinstance(NPC.__dict__.get(_name), (staticmethod, classmethod)):
        setattr(NPC, _name, _npc_per_ring_wrapper(_name))
del _npc_per_ring_wrapper, _name


# =============================================================================
# Block-assembly geometry: ContactSite and BlockDefinition
# =============================================================================


@dataclass(frozen=True)
class ContactSite:
    """A single contact site in the block frame.

    Parameters
    ----------
    vector : tuple[float, float, float]
        Displacement from the block origin to the site, in voxels, expressed
        in the block's local coordinate frame.
    site_type : str, default="site"
        Logical type label used when defining which site pairs are allowed to
        contact each other in a :class:`BlockDefinition`.

    Raises
    ------
    ValueError
        If *vector* does not have length 3 or if its in-plane part (x, y) is
        zero (which would leave the site without a defined azimuth).
    """

    vector: tuple[float, float, float]
    site_type: str = "site"

    def __post_init__(self) -> None:
        if len(self.vector) != 3:
            raise ValueError(f"ContactSite vector must have exactly 3 components, got {len(self.vector)}.")
        x, y, _ = self.vector
        if x * x + y * y == 0.0:
            raise ValueError(
                "ContactSite has zero in-plane part (x = y = 0). " "Every site must have a defined azimuth."
            )


@dataclass(frozen=True)
class BlockDefinition:
    """Geometry and contact rules for one class of building block.

    Parameters
    ----------
    sites : tuple[ContactSite, ...]
        Ordered, counter-clockwise sequence of contact sites.  At least one
        site is required.
    pairing : tuple[tuple[str, str], ...], default=(("site", "site"),)
        Allowed contact pairs as ``(type_a, type_b)`` tuples.  Both type
        strings must appear in the site types of *sites*.
    flip_site : int, default=1
        1-based index of the site used to define the polarity-flip axis.
        The flip rotation is a 180° rotation around the in-plane part of this
        site's vector.
    fold : int or None, default=None
        Cyclic symmetry order.  When set, must equal :attr:`n_sites`.  Used
        by :meth:`PleomorphicSurface.get_sites_as_motl` to delegate to
        :meth:`~cryocat.core.cryomotl.Motl.split_in_asymmetric_subunits` so
        that the returned positions also carry per-symmetry-copy orientations.
        Set automatically by :meth:`cyclic`; ``None`` for non-cyclic
        definitions such as those created with :meth:`microtubule`.

    Raises
    ------
    ValueError
        If *sites* is empty, any site vector has the wrong length or zero
        in-plane part, *flip_site* is out of range, any pairing type is
        missing from the site types, the sites are not listed in
        counter-clockwise azimuth order, or *fold* is set but does not equal
        :attr:`n_sites`.
    """

    sites: tuple[ContactSite, ...]
    pairing: tuple[tuple[str, str], ...] = (("site", "site"),)
    flip_site: int = 1
    symmetry: str | None = None

    def __post_init__(self) -> None:
        if not self.sites:
            raise ValueError("BlockDefinition requires at least one ContactSite.")
        if not (1 <= self.flip_site <= len(self.sites)):
            raise ValueError(f"flip_site={self.flip_site} is out of range 1..{len(self.sites)}.")
        type_set = set(self.site_types)
        for a, b in self.pairing:
            for t in (a, b):
                if t not in type_set:
                    raise ValueError(f"Pairing type '{t}' not found in site types {type_set}.")
        if len(self.sites) > 1:
            alpha = [_math.atan2(s.vector[1], s.vector[0]) for s in self.sites]
            alpha0 = alpha[0]
            betas = [(_math.degrees(a - alpha0)) % 360.0 for a in alpha]
            for i in range(1, len(betas)):
                if betas[i] <= betas[i - 1]:
                    raise ValueError(
                        "Sites must be listed in strict counter-clockwise azimuth order. "
                        f"Site {i + 1} (β={betas[i]:.2f}°) is not strictly after "
                        f"site {i} (β={betas[i - 1]:.2f}°)."
                    )

    @gui_exposed(category="builder", label="Block definition - cyclic", returns="block_def")
    @classmethod
    def cyclic(
        cls,
        symmetry: "Symmetry",
        site_shift: TripletLike,
        site_type: str = "site",
    ) -> "BlockDefinition":
        """Create a C_n-symmetric block with equally-spaced sites.

        Site k (1-based) is *site_shift* rotated about +z by
        ``360·(k-1)/n`` degrees — the same convention as
        :meth:`~cryocat.core.cryomotl.Motl.split_in_asymmetric_subunits`.

        Parameters
        ----------
        symmetry : Symmetry
            Cyclic symmetry specifier: an integer *n*, or a string ``"Cn"``
            (e.g. ``"C3"``).  Non-cyclic groups (D, T, O, I) raise
            :exc:`ValueError`.
        site_shift : TripletLike
            Shift from the block origin to the first contact site, in
            voxels, expressed in the block's local frame.  Typically about
            half the centre-to-centre distance, pointing along one leg.
        site_type : str, default="site"
            Type label for all sites.

        Returns
        -------
        BlockDefinition
            A frozen :class:`BlockDefinition` with a single site (the shift)
            and :attr:`symmetry` set to ``"C{n}"``.

        Examples
        --------
        >>> BlockDefinition.cyclic(3, [-5, 0, 0])      # C3, arm along −x
        >>> BlockDefinition.cyclic("C6", [-3, 0, -1])  # C6, tilted arm
        """
        group, n = geom.as_symmetry(symmetry)
        if group != "C":
            raise ValueError(f"cyclic() requires a Cn symmetry specifier; got '{group}{n}'.")
        shift = geom.as_triplet(site_shift)
        site = ContactSite(vector=tuple(float(c) for c in shift), site_type=site_type)
        return cls(sites=(site,), pairing=((site_type, site_type),), symmetry=f"C{n}")

    @gui_exposed(category="builder", label="Block definition – microtubule", returns="block_def")
    @classmethod
    def microtubule(
        cls,
        lateral_length: float,
        axial_length: float,
        lateral_elevation: float = 0.0,
    ) -> "BlockDefinition":
        """Create a 4-site block definition for a microtubule protofilament.

        The block frame convention is: **x** points toward the right
        neighbouring protofilament, **y** points toward the plus end of the
        protofilament, and **z** points radially outward from the tube axis.
        The four sites are placed at:

        * site 1 — ``lateral_right``: ``(lc, 0, −ls)``
        * site 2 — ``plus``: ``(0, axial_length, 0)``
        * site 3 — ``lateral_left``: ``(−lc, 0, −ls)``
        * site 4 — ``minus``: ``(0, −axial_length, 0)``

        where ``lc = lateral_length × cos(lateral_elevation)`` and
        ``ls = lateral_length × sin(lateral_elevation)``.

        Parameters
        ----------
        lateral_length : float
            Distance from the block origin to each lateral site in voxels.
        axial_length : float
            Distance from the block origin to each axial (plus/minus) site.
        lateral_elevation : float, default=0.0
            Elevation (degrees) of the lateral sites below the xy-plane in
            the block frame.  Positive values tilt the sites toward the tube
            axis (negative z).

        Returns
        -------
        BlockDefinition
            Four-site definition with ``pairing=(("plus", "minus"),
            ("lateral_right", "lateral_left"))`` and ``flip_site=2`` (plus end).
        """
        e = _math.radians(float(lateral_elevation))
        ell = float(lateral_length)
        a = float(axial_length)
        lc = ell * _math.cos(e)
        ls = ell * _math.sin(e)
        return cls(
            sites=(
                ContactSite(vector=(lc, 0.0, -ls), site_type="lateral_right"),
                ContactSite(vector=(0.0, a, 0.0), site_type="plus"),
                ContactSite(vector=(-lc, 0.0, -ls), site_type="lateral_left"),
                ContactSite(vector=(0.0, -a, 0.0), site_type="minus"),
            ),
            pairing=(("plus", "minus"), ("lateral_right", "lateral_left")),
            flip_site=2,
        )

    @property
    def n_sites(self) -> int:
        """Number of contact sites stored in this definition."""
        return len(self.sites)

    @property
    def effective_n_sites(self) -> int:
        """Number of sites after symmetry expansion.

        For cyclic definitions (created via :meth:`cyclic`), returns the cyclic
        order *n*.  For non-cyclic definitions, returns :attr:`n_sites`.
        """
        if self.symmetry is not None:
            _, n = geom.as_symmetry(self.symmetry)
            return n
        return len(self.sites)

    @property
    def site_types(self) -> tuple[str, ...]:
        """Ordered tuple of site-type labels."""
        return tuple(s.site_type for s in self.sites)

    def site_vectors(self) -> np.ndarray:
        """Return site vectors as an ``(n_sites, 3)`` float array."""
        return np.array([s.vector for s in self.sites], dtype=float)


# =============================================================================
# PleomorphicSurface for discrete surfaces (Mesh and OrientedPointCloud)
# =============================================================================


class BlockLayer:
    """Block layer bundling blocks motl, block definition, column names, and scale.

    Validates inputs once; pass the result to :class:`PleomorphicSurface` as
    ``block_layer=``.  Use :meth:`PleomorphicSurface.from_blocks` instead of
    constructing :class:`BlockLayer` directly unless you already have a
    :class:`BlockDefinition`.

    Parameters
    ----------
    blocks : MotlSource
        Block particle list.  Loaded with ``Motl.load``; the input is never
        modified.
    block_definition : BlockDefinition or dict[float, BlockDefinition]
        Geometry for one block type, or a mapping from block-type float values
        to per-type definitions.
    block_type_column : MotlColumn, default="class"
        Column in *blocks* that identifies the block class when
        *block_definition* is a dict.
    block_id_column : MotlColumn, default="geom3"
        Column used to identify blocks in site-motl output.
    affiliation_column : MotlColumn, default="object_id"
        Column whose value identifies which assembly each block belongs to.
        Blocks with different values in this column are never connected, even
        when their sites are within *max_distance*.
    site_index_column : MotlColumn, default="geom1"
        Column written with the 1-based site index.
    site_type_column : MotlColumn, default="geom2"
        Column written with the integer site-type code.
    assembly_id_column : MotlColumn, default="geom4"
        Column written with the assembly (connected-component) id in the faces
        motl produced by :meth:`~PleomorphicSurface.get_faces_as_motl`.
    face_id_column : MotlColumn, default="geom5"
        Column written with the face id in both the faces motl
        (:meth:`~PleomorphicSurface.get_faces_as_motl`) and the missing-block
        motl (:meth:`~PleomorphicSurface.get_missing_block_motl`).
    face_size_column : MotlColumn, default="score"
        Column written with the face size (vertex count) in the faces motl.
    source_block_count_column : MotlColumn, default="subtomo_mean"
        Column written with the number of distinct source blocks in the gaps
        motl (:meth:`~PleomorphicSurface.get_gaps_as_motl`).
    pixel_size : float, default=1.0
        Ångström-per-voxel scale factor.
    tomo_id_column : MotlColumn, default="tomo_id"
        Column identifying the tomogram each block belongs to.
    bonding_tolerance : float or None, default=None
        Half-width (voxels) of the acceptance window around the nominal
        centre-to-centre bonding distance ``2 × |site_shift|``.  ``None``
        derives to ``0.4 × |site_shift|`` at call time.
    tip_tolerance : float or None, default=None
        Radius tolerance (voxels) for accepting a leg tip as near the
        candidate centre.  ``None`` derives to ``0.3 × |site_shift|`` at
        call time.
    angle_tolerance : float or None, default=None
        Maximum allowed angular residual (degrees) for the Cn pairwise
        spacing check.  ``None`` derives to ``0.125 × (360 / n)`` at call
        time (giving 15° for C3, 7.5° for C6, 1.15° for C39).  Raises
        ``ValueError`` if the resolved value is ``>= 0.5 × (360 / n)``.
    merge_tolerance : float or None, default=None
        Half-width (voxels) used as the default merge distance in
        :meth:`~PleomorphicSurface.infer_missing_blocks` when
        *merge_distance* is ``None``.  ``None`` derives to
        ``0.5 × |site_shift|``.
    clean_distance : float or None, default=None
        Default clean distance (voxels) for
        :meth:`~PleomorphicSurface.infer_missing_blocks`.  ``None``
        derives to ``0.4 × |site_shift|`` at call time.

    Raises
    ------
    ValueError
        If *block_definition* is ``None``, if *blocks* has non-unique
        ``subtomo_id`` values, or if *block_definition* is a dict and the
        block-type column contains values without a matching definition.
    """

    def __init__(
        self,
        blocks: MotlSource,
        block_definition: "BlockDefinition | dict[float, BlockDefinition]",
        *,
        block_type_column: MotlColumn = "geom1",
        block_id_column: MotlColumn = "geom3",
        affiliation_column: MotlColumn = "object_id",
        site_index_column: MotlColumn = "geom1",
        site_type_column: MotlColumn = "geom2",
        assembly_id_column: MotlColumn = "geom4",
        face_id_column: MotlColumn = "geom4",
        face_size_column: MotlColumn = "geom5",
        source_block_count_column: MotlColumn = "geom2",
        pixel_size: float = 1.0,
        tomo_id_column: MotlColumn = "tomo_id",
        max_cycle_length: int = 25,
        bonding_tolerance: float | None = None,
        tip_tolerance: float | None = None,
        angle_tolerance: float | None = None,
        merge_tolerance: float | None = None,
        clean_distance: float | None = None,
    ) -> None:
        if block_definition is None:
            raise ValueError("block_definition is required.")
        loaded = cryomotl.Motl.load(blocks)
        if loaded.df["subtomo_id"].duplicated().any():
            raise ValueError("subtomo_id must be unique in the blocks motl.")
        if isinstance(block_definition, dict):
            cls_vals = set(loaded.df[block_type_column].unique())
            missing = cls_vals - set(block_definition.keys())
            if missing:
                raise ValueError(
                    f"Block type column '{block_type_column}' contains values "
                    f"without a BlockDefinition: {sorted(missing)}."
                )
        self.blocks: cryomotl.Motl = loaded
        self.block_definition: "BlockDefinition | dict[float, BlockDefinition]" = block_definition
        self.block_type_column: MotlColumn = block_type_column
        self.block_id_column: MotlColumn = block_id_column
        self.affiliation_column: MotlColumn = affiliation_column
        self.site_index_column: MotlColumn = site_index_column
        self.site_type_column: MotlColumn = site_type_column
        self.assembly_id_column: MotlColumn = assembly_id_column
        self.face_id_column: MotlColumn = face_id_column
        self.face_size_column: MotlColumn = face_size_column
        self.source_block_count_column: MotlColumn = source_block_count_column
        self.pixel_size: float = float(pixel_size)
        self.tomo_id_column: MotlColumn = tomo_id_column
        self.max_cycle_length: int = int(max_cycle_length)
        self.bonding_tolerance: float | None = float(bonding_tolerance) if bonding_tolerance is not None else None
        self.tip_tolerance: float | None = float(tip_tolerance) if tip_tolerance is not None else None
        self.angle_tolerance: float | None = float(angle_tolerance) if angle_tolerance is not None else None
        self.merge_tolerance: float | None = float(merge_tolerance) if merge_tolerance is not None else None
        self.clean_distance: float | None = float(clean_distance) if clean_distance is not None else None

        defs_list = list(block_definition.values()) if isinstance(block_definition, dict) else [block_definition]
        all_types = sorted({st for d in defs_list for st in d.site_types})
        self.site_type_codes: dict[str, int] = {t: i + 1 for i, t in enumerate(all_types)}
        self._allowed_pairs: set[frozenset] = set()
        for d in defs_list:
            for a, b in d.pairing:
                self._allowed_pairs.add(frozenset({a, b}))


class PleomorphicSurface:
    """Pleomorphic lattice assembly: an envelope layer and/or a block layer.

    The two layers are independent.

    * **Envelope layer** (``_surface``): a :class:`Mesh` or
      :class:`OrientedPointCloud` representing the outer surface.  Access it
      via :attr:`surface`; check its presence with :attr:`has_envelope`.
    * **Block layer** (``blocks``): a :class:`~cryocat.core.cryomotl.Motl`
      together with a :class:`BlockDefinition` (or dict thereof) describing the
      building-block geometry.  Access it via :attr:`blocks`; check its
      presence with :attr:`has_blocks`.

    Either layer can be absent, but at least one is required.

    **Blocks-only example** (no envelope, using the plain-parameter builder)::

        ps = PleomorphicSurface.from_blocks(
            motl, symmetry="C3", site_length=9.0, site_elevation=-10.0
        )
        ps.connect(max_distance=4.0)
        mesh = ps.envelope_from_faces()
        ps.surface = mesh          # attach the envelope for polarity queries

    For the GUI entry point see :meth:`from_blocks`.  For direct construction
    see :meth:`__init__`.
    """

    def __init__(
        self,
        surface: Mesh | OrientedPointCloud | "PleomorphicSurface" | None = None,
        *,
        block_layer: "BlockLayer | None" = None,
    ) -> None:
        """Create a :class:`PleomorphicSurface`.

        Parameters
        ----------
        surface : Mesh, OrientedPointCloud, PleomorphicSurface, or None
            Envelope surface.  When a :class:`PleomorphicSurface` is passed,
            its ``_surface`` is extracted; if *block_layer* is also ``None``,
            the block layer is copied from it as well.  ``None`` means no
            envelope.
        block_layer : BlockLayer or None, default=None
            Pre-built block layer carrying the blocks motl, block definition,
            column names, and scale factor.  Use :meth:`from_blocks` to
            construct one without building a :class:`BlockLayer` directly.

        Raises
        ------
        TypeError
            If neither *surface* nor *block_layer* is provided, or if
            *surface* has an unsupported type.

        Notes
        -----
        Envelope operations that return a new object (e.g. ``crop``,
        ``extract_region``, ``convex_hull``, ``oversample``, and
        ``apply_vertex_mask`` / ``flip_normals`` / ``refine_normals`` /
        ``remove_nonfinite_vertices`` with ``inplace=False``) return an
        envelope-only :class:`PleomorphicSurface`; the block layer is not
        carried.  :meth:`envelope_from_faces` returns a bare
        :class:`~cryocat.core.surface.Mesh` built from the block layer;
        attach it with ``ps.surface = mesh``.
        """
        self._surface: Mesh | OrientedPointCloud | None = None
        # --- Block layer defaults ---
        self.blocks: cryomotl.Motl | None = None
        self.block_definition: BlockDefinition | dict[float, BlockDefinition] | None = None
        self.block_type_column: MotlColumn = "geom1"
        self.block_id_column: MotlColumn = "geom3"
        self.site_index_column: MotlColumn = "geom1"
        self.site_type_column: MotlColumn = "geom2"
        self.assembly_id_column: MotlColumn = "geom4"
        self.face_id_column: MotlColumn = "geom4"
        self.face_size_column: MotlColumn = "geom5"
        self.source_block_count_column: MotlColumn = "geom2"
        self._block_pixel_size: float = 1.0
        self.tomo_id_column: MotlColumn = "tomo_id"
        self.affiliation_column: MotlColumn = "object_id"
        self._site_table: pd.DataFrame | None = None
        self._faces: list[dict] | None = None
        self.site_type_codes: dict[str, int] = {}
        self._allowed_pairs: set[frozenset] = set()
        self._symmetry: str | None = None
        self._symmetry_column: MotlColumn | None = None
        self._site_shift: np.ndarray | None = None
        self._max_distance: float | None = None
        self._origin_column: MotlColumn = "geom5"
        self._source_id_column: MotlColumn = "geom4"
        self._added: cryomotl.Motl | None = None
        self._all_state: dict | None = None
        self.max_cycle_length: int = 25
        self.bonding_tolerance: float | None = None
        self.tip_tolerance: float | None = None
        self.angle_tolerance: float | None = None
        self.merge_tolerance: float | None = None
        self.clean_distance: float | None = None
        self._envelopes: dict[tuple[float, float], "Mesh"] | None = None
        self._envelope_scope: str | None = None

        # --- Envelope ---
        if isinstance(surface, PleomorphicSurface):
            self._surface = surface._surface
            if block_layer is None:
                self.blocks = copy.deepcopy(surface.blocks)
                self.block_definition = copy.deepcopy(surface.block_definition)
                self.block_type_column = surface.block_type_column
                self.block_id_column = surface.block_id_column
                self.site_index_column = surface.site_index_column
                self.site_type_column = surface.site_type_column
                self.assembly_id_column = surface.assembly_id_column
                self.face_id_column = surface.face_id_column
                self.face_size_column = surface.face_size_column
                self.source_block_count_column = surface.source_block_count_column
                self._block_pixel_size = surface._block_pixel_size
                self.tomo_id_column = surface.tomo_id_column
                self.affiliation_column = surface.affiliation_column
                self._site_table = copy.deepcopy(surface._site_table)
                self._faces = copy.deepcopy(surface._faces)
                self.site_type_codes = copy.deepcopy(surface.site_type_codes)
                self._allowed_pairs = copy.deepcopy(surface._allowed_pairs)
                self._symmetry = surface._symmetry
                self._symmetry_column = surface._symmetry_column
                self._site_shift = copy.deepcopy(surface._site_shift)
                self._max_distance = surface._max_distance
                self._origin_column = surface._origin_column
                self._source_id_column = surface._source_id_column
                self._added = copy.deepcopy(surface._added)
                self._all_state = copy.deepcopy(surface._all_state)
                self.max_cycle_length = surface.max_cycle_length
                self.bonding_tolerance = surface.bonding_tolerance
                self.tip_tolerance = surface.tip_tolerance
                self.angle_tolerance = surface.angle_tolerance
                self.merge_tolerance = surface.merge_tolerance
                self.clean_distance = surface.clean_distance
                self._envelopes = copy.deepcopy(surface._envelopes)
                self._envelope_scope = surface._envelope_scope
        elif surface is not None:
            if not isinstance(surface, (Mesh, OrientedPointCloud)):
                raise TypeError(
                    f"Unsupported surface type: {type(surface)}. "
                    "Must be Mesh, OrientedPointCloud, or PleomorphicSurface."
                )
            self._surface = surface

        if self._surface is None and self.blocks is None and block_layer is None:
            raise TypeError(
                "PleomorphicSurface requires at least a surface " "(Mesh or OrientedPointCloud) or a block_layer."
            )

        # --- Block layer ---
        if block_layer is not None:
            self.blocks = block_layer.blocks
            self.block_definition = block_layer.block_definition
            self.block_type_column = block_layer.block_type_column
            self.block_id_column = block_layer.block_id_column
            self.site_index_column = block_layer.site_index_column
            self.site_type_column = block_layer.site_type_column
            self.assembly_id_column = block_layer.assembly_id_column
            self.face_id_column = block_layer.face_id_column
            self.face_size_column = block_layer.face_size_column
            self.source_block_count_column = block_layer.source_block_count_column
            self._block_pixel_size = block_layer.pixel_size
            self.tomo_id_column = block_layer.tomo_id_column
            self.affiliation_column = block_layer.affiliation_column
            self.site_type_codes = block_layer.site_type_codes
            self._allowed_pairs = block_layer._allowed_pairs
            self.max_cycle_length = block_layer.max_cycle_length
            self.bonding_tolerance = block_layer.bonding_tolerance
            self.tip_tolerance = block_layer.tip_tolerance
            self.angle_tolerance = block_layer.angle_tolerance
            self.merge_tolerance = block_layer.merge_tolerance
            self.clean_distance = block_layer.clean_distance
            surf_ps = getattr(self._surface, "pixel_size", None)
            if surf_ps is not None:
                surf_scalar = float(np.mean(np.asarray(surf_ps, dtype=float)))
                if abs(self._block_pixel_size - surf_scalar) > 1e-9:
                    import warnings as _warn

                    _warn.warn(
                        f"block_layer.pixel_size ({self._block_pixel_size}) differs from "
                        f"surface pixel_size ({surf_scalar}). "
                        f"Block-layer computations use {self._block_pixel_size}; "
                        f"surface measurements use the mesh's own scale.",
                        UserWarning,
                        stacklevel=2,
                    )

    @property
    def block_pixel_size(self) -> float:
        """Ångström-per-voxel scale factor for block-layer computations.

        Block-layer methods (contact distances, face centroids, site
        annotations) multiply voxel-space coordinates by this value.  It never
        overwrites the mesh's own scale; surface measurements use
        ``self.surface.pixel_size`` directly.
        """
        return self._block_pixel_size

    @property
    def pixel_size(self) -> float:
        """Ångström-per-voxel scale for block-layer computations.

        Alias for :attr:`block_pixel_size`.  The mesh's own scale is
        stored on the :class:`~cryocat.core.surface.Mesh` object and is
        never overwritten by this value.
        """
        return self._block_pixel_size

    @pixel_size.setter
    def pixel_size(self, value: float) -> None:
        self._block_pixel_size = float(value)

    @property
    def surface(self) -> Mesh | OrientedPointCloud:
        """The wrapped :class:`Mesh` or :class:`OrientedPointCloud`.

        Raises
        ------
        ValueError
            If this instance has no envelope (was created with blocks only).
        """
        if self._surface is None:
            raise ValueError(
                "PleomorphicSurface has no envelope (Mesh/OrientedPointCloud). "
                "Check has_envelope before accessing .surface."
            )
        return self._surface

    @surface.setter
    def surface(self, value: Mesh | OrientedPointCloud | None) -> None:
        self._surface = value

    @property
    def has_envelope(self) -> bool:
        """True when an envelope surface (Mesh or OrientedPointCloud) is present."""
        return self._surface is not None

    @property
    def has_blocks(self) -> bool:
        """True when a block layer is present."""
        return self.blocks is not None

    @property
    def has_envelopes(self) -> bool:
        """True when per-object envelopes have been built."""
        return self._envelopes is not None and len(self._envelopes) > 0

    @property
    def envelopes(self) -> "dict[tuple[float, float], Mesh] | None":
        """Per-object mesh envelopes keyed by ``(tomo_id, assembly_id)``, or ``None``."""
        return self._envelopes

    @classmethod
    def from_blocks(
        cls,
        blocks: MotlSource,
        symmetry: Symmetry = "C3",
        *,
        site_shift: TripletLike,
        symmetry_column: MotlColumn | None = None,
        affiliation_column: MotlColumn = "object_id",
        pixel_size: float = 1.0,
        tomo_id_column: MotlColumn = "tomo_id",
    ) -> "PleomorphicSurface":
        """Build a :class:`PleomorphicSurface` from a block motl without
        constructing a :class:`BlockDefinition` directly.

        This is the GUI entry point and the preferred programmatic path.
        Builds a :class:`BlockLayer` internally and passes it to
        :meth:`__init__`.  Example::

            PleomorphicSurface.from_blocks("trimers.em", symmetry="C3", site_shift=[-5, 0, 0])

        For a single symmetry (``symmetry_column=None``), one
        :meth:`BlockDefinition.cyclic` definition is created with *symmetry*
        and *site_shift* and applied to all blocks.

        For mixed folds (``symmetry_column`` set), the fold order for each
        block is read from that column of *blocks*.  Every unique value must be
        a positive integer (floats like ``3.0`` are accepted; non-integer floats
        like ``2.5`` raise :exc:`ValueError`).  One
        :meth:`BlockDefinition.cyclic` definition is created per unique fold,
        using the shared *site_shift*.

        The result has no envelope layer; use :meth:`envelope_from_faces` after
        :meth:`connect` if one is needed.

        Parameters
        ----------
        blocks : MotlSource
            Block particle list (path or :class:`~cryocat.core.cryomotl.Motl`).
        symmetry : Symmetry, default="C3"
            Cyclic symmetry specifier for the single-symmetry path: an integer
            *n* or a string like ``"C3"``.  Ignored when *symmetry_column* is
            set.
        site_shift : TripletLike
            Shift from each block origin to its first contact site, in
            voxels, expressed in the block's local frame.  Typically about
            half the centre-to-centre distance, pointing along one leg.
        symmetry_column : MotlColumn or None, default=None
            If set, read the fold order from this column of *blocks* instead
            of using *symmetry*.  All unique values must be positive integers.
        affiliation_column : MotlColumn, default="object_id"
            Column whose value identifies which assembly each block belongs to.
            Blocks with different values are never connected by :meth:`connect`.
        pixel_size : float, default=1.0
            Pixel size passed to :class:`PleomorphicSurface`.
        tomo_id_column : MotlColumn, default="tomo_id"
            Column identifying the tomogram each block belongs to.

        Returns
        -------
        PleomorphicSurface
            A blocks-only surface (no envelope).

        Raises
        ------
        ValueError
            If *symmetry_column* contains non-integer or less-than-1 values.

        Notes
        -----
        *blocks* is loaded with :func:`~cryocat.core.cryomotl.Motl.load`,
        which copies a :class:`~cryocat.core.cryomotl.Motl` instance; the
        input is never modified.  :meth:`unify_polarity`,
        :meth:`store_block_stats` and :meth:`store_block_stat` change only
        this object's copy (angles, or the requested columns; row order and
        index are kept).  Use :meth:`get_blocks_as_motl` to obtain the
        current state, e.g. unified orientations for averaging or for
        building a new assembly.
        """
        shift = geom.as_triplet(site_shift)
        if symmetry_column is None:
            block_def: BlockDefinition | dict[float, BlockDefinition] = BlockDefinition.cyclic(symmetry, shift)
            ps = cls(
                block_layer=BlockLayer(
                    blocks,
                    block_def,
                    affiliation_column=affiliation_column,
                    pixel_size=pixel_size,
                    tomo_id_column=tomo_id_column,
                )
            )
            ps._symmetry = symmetry
            ps._site_shift = shift
            return ps
        else:
            loaded = cryomotl.Motl.load(blocks)
            raw_vals = loaded.df[symmetry_column].unique()
            block_def_dict: dict[float, BlockDefinition] = {}
            for val in raw_vals:
                fval = float(val)
                n = int(round(fval))
                if abs(fval - n) > 1e-9 or n < 1:
                    raise ValueError(
                        f"from_blocks: symmetry_column '{symmetry_column}' contains "
                        f"non-integer or <1 value {val!r}. All values must be positive integers."
                    )
                block_def_dict[fval] = BlockDefinition.cyclic(n, shift)
            ps = cls(
                block_layer=BlockLayer(
                    loaded,
                    block_def_dict,
                    block_type_column=symmetry_column,
                    affiliation_column=affiliation_column,
                    pixel_size=pixel_size,
                    tomo_id_column=tomo_id_column,
                )
            )
            ps._symmetry_column = symmetry_column
            ps._site_shift = shift
            return ps

    @staticmethod
    def _unwrap_surface(surface: Mesh | OrientedPointCloud | "PleomorphicSurface") -> Mesh | OrientedPointCloud:
        """Return the concrete Mesh / OrientedPointCloud behind an optional wrapper."""
        if isinstance(surface, PleomorphicSurface):
            return surface.surface
        if isinstance(surface, (Mesh, OrientedPointCloud)):
            return surface
        raise TypeError(
            f"Unsupported surface type: {type(surface)}. " "Must be Mesh, OrientedPointCloud, or PleomorphicSurface."
        )

    # ------------------------------------------------------------------
    # Block-assembly methods
    # ------------------------------------------------------------------

    @gui_exposed(label="Sites as motl", group="Lattice setup", order=40, returns="motl", category="pleomorphic-op")
    def get_sites_as_motl(self, scope: Literal["original", "inferred", "all"] = "original") -> "cryomotl.Motl":
        """Return a particle list with one row per contact site of every block.

        For each block in the pool selected by *scope*, each contact site
        defined in :attr:`block_definition` is placed at ``c + R.apply(v_k)``,
        where ``c`` is the block centre, ``R`` its rotation, and ``v_k`` the
        k-th site vector.

        When a :class:`BlockDefinition` has :attr:`~BlockDefinition.symmetry` set
        (created via :meth:`~BlockDefinition.cyclic`), this method delegates to
        :meth:`~cryomotl.Motl.split_in_asymmetric_subunits`, which also
        rotates the output angles by the corresponding symmetry operation so
        that each site row carries the block rotation composed with the
        k-th cyclic rotation ``R_k = R_z(360*(k-1)/n)``.

        For definitions without :attr:`~BlockDefinition.symmetry` (non-symmetric /
        typed sites), :func:`expand_motl` is used and angles stay as the block
        rotation (``orientation="keep"``).

        The returned motl is independent of the internal block layer: later
        calls to :meth:`connect` and all statistics
        methods read rotations from :attr:`blocks` directly, not from the
        returned motl.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses :attr:`blocks` only.  ``'all'`` uses lattice +
            inferred blocks combined.  ``'inferred'`` uses only inferred blocks.

        Returns
        -------
        cryomotl.Motl
            One row per (block, site) combination.  The following columns
            carry values written by this method:

            * ``block_id_column`` (default ``geom3``) — source block's ``subtomo_id``
            * ``site_index_column`` (default ``geom1``) — 1-based site index (CCW order)
            * ``site_type_column`` (default ``geom2``) — integer site-type code from :attr:`site_type_codes`
            * ``affiliation_column`` (default ``object_id``) — assembly affiliation copied from source block (propagated when ``affiliation_column != block_id_column``)
            * ``x``, ``y``, ``z`` — site position in voxels

            All other columns are inherited from the parent block rows unchanged.

        Raises
        ------
        ValueError
            If no block layer is present, or *scope* is invalid.
        """
        _check_column_collisions("get_sites_as_motl", {
            "site_index_column": self.site_index_column,
            "site_type_column": self.site_type_column,
            "block_id_column": self.block_id_column,
        })
        _blks = self._blocks_for_scope(scope)

        def _expand_one(group: "cryomotl.Motl", definition: "BlockDefinition") -> "cryomotl.Motl":
            if definition.symmetry is not None:
                _, n = geom.as_symmetry(definition.symmetry)
                site_motl = group.split_in_asymmetric_subunits(n, definition.site_vectors()[0])
                # geom5 = original subtomo_id → block_id_column
                # geom2 = 1-based CCW subunit index → site_index_column
                # site_type_column ← uniform type code for cyclic definitions
                site_motl.df[self.block_id_column] = site_motl.df["geom5"]
                site_motl.df[self.site_index_column] = site_motl.df["geom2"]
                code = float(self.site_type_codes[definition.sites[0].site_type])
                site_motl.df[self.site_type_column] = code
            else:
                site_motl = expand_motl(
                    group,
                    definition.site_vectors(),
                    original_id_col=self.block_id_column,
                    order_id_col=self.site_index_column,
                    sort_vectors=False,
                    orientation="keep",
                    start_index=1,
                )
                for i, site in enumerate(definition.sites):
                    code = self.site_type_codes[site.site_type]
                    site_motl.df.loc[site_motl.df[self.site_index_column] == (i + 1), self.site_type_column] = float(
                        code
                    )
            return site_motl

        if isinstance(self.block_definition, dict):
            group_motls: list[cryomotl.Motl] = []
            for cls_val, definition in self.block_definition.items():
                mask = _blks.df[self.block_type_column] == cls_val
                group_df = _blks.df[mask].copy().reset_index(drop=True)
                if len(group_df) == 0:
                    continue
                group = cryomotl.Motl(group_df)
                group_motls.append(_expand_one(group, definition))
        else:
            group_motls = [_expand_one(_blks, self.block_definition)]

        if not group_motls:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns))

        combined_df = pd.concat([m.df for m in group_motls], ignore_index=True)
        combined_df = combined_df.sort_values(
            by=[self.tomo_id_column, self.block_id_column, self.site_index_column]
        ).reset_index(drop=True)

        aff_col = self.affiliation_column
        bid_col = self.block_id_column
        if aff_col != bid_col:
            _aff_map: dict[float, float] = dict(
                zip(
                    _blks.df["subtomo_id"].astype(float).values,
                    _blks.df[aff_col].astype(float).values,
                )
            )
            combined_df[aff_col] = combined_df[bid_col].astype(float).map(_aff_map).fillna(0.0)

        result = cryomotl.Motl(combined_df)
        result.renumber_particles()
        return result

    def _get_flip_rot(self, global_idx: int) -> srot:
        """Return the 180° flip rotation for block at *global_idx* in blocks.df."""
        if isinstance(self.block_definition, dict):
            cls_val = self.blocks.df.iloc[global_idx][self.block_type_column]
            defn = self.block_definition[cls_val]
        else:
            defn = self.block_definition
        v = np.array(defn.sites[defn.flip_site - 1].vector)
        a_unnorm = np.array([v[0], v[1], 0.0])
        a = a_unnorm / np.linalg.norm(a_unnorm)
        return srot.from_rotvec(np.pi * a)

    @gui_exposed(label="Unify polarity", group="Lattice setup", order=20, returns="none", category="pleomorphic-op")
    def unify_polarity(
        self,
        max_block_distance: float | None = None,
        reference: Literal["neighbours", "centroid", "envelope"] = "neighbours",
        *,
        _annotate_fn: Callable | None = None,
    ) -> int:
        """Align block polarities so neighbours point the same way.

        Works per tomogram.  For ``reference='neighbours'`` or ``'centroid'``,
        uses a BFS walk over the block-proximity graph (edges between blocks
        whose centres are within *max_block_distance*): when a newly visited
        block has ``z_n · z_m < 0`` relative to its BFS parent, it is flipped
        by the 180° rotation around the in-plane part of its flip_site vector.

        For ``reference='centroid'``, each connected component is additionally
        aligned so that its blocks point outward from the component centroid
        (``Σ z · (c - g) > 0``).

        For ``reference='envelope'``, an envelope surface must be attached
        (via ``ps.surface = ...``).  *max_block_distance* may be ``None``.

        Parameters
        ----------
        max_block_distance : float or None, default=None
            Maximum centre-to-centre distance (voxels) for two blocks to be
            considered neighbours.  Required for ``'neighbours'`` and
            ``'centroid'``; ignored for ``'envelope'``.
        reference : {"neighbours", "centroid", "envelope"}, default="neighbours"
            Alignment reference.
        _annotate_fn : Callable or None, default=None
            Private override for the envelope annotation callable.  When
            ``None`` and ``reference='envelope'``, falls back to
            :meth:`annotate_with_envelope`.

        Returns
        -------
        int
            Number of blocks whose final orientation differs from the initial one.

        Raises
        ------
        ValueError
            If no block layer is present, *reference* is invalid,
            *max_block_distance* is None for a BFS-based reference, or
            ``reference='envelope'`` is requested but no envelope is attached.
        """
        if self.blocks is None:
            raise ValueError("unify_polarity requires a block layer (blocks=...).")
        if reference not in ("neighbours", "centroid", "envelope"):
            raise ValueError(f"reference must be 'neighbours', 'centroid', or 'envelope'; got {reference!r}.")
        if reference != "envelope" and max_block_distance is None:
            raise ValueError(f"max_block_distance is required for reference='{reference}'.")
        if reference == "envelope" and _annotate_fn is None:
            if not self.has_envelope:
                raise ValueError(
                    "unify_polarity(reference='envelope') requires an envelope surface "
                    "(attach one via ps.surface = ... or pass a mesh to PleomorphicSurface)."
                )
            _annotate_fn = self.annotate_with_envelope

        N = len(self.blocks.df)

        angles_arr = self.blocks.get_angles().astype(float)
        all_R_init = srot.from_euler("zxz", angles_arr, degrees=True)
        R_list: list[srot] = [all_R_init[i] for i in range(N)]
        z_unit = np.array([0.0, 0.0, 1.0])
        z_arr = np.array([R_list[i].apply(z_unit) for i in range(N)])
        all_c = self.blocks.get_coordinates()

        for tomo_val in sorted(self.blocks.df[self.tomo_id_column].unique()):
            tomo_mask = (self.blocks.df[self.tomo_id_column] == tomo_val).values
            tomo_pos = np.where(tomo_mask)[0]
            n_tomo = len(tomo_pos)
            if n_tomo == 0:
                continue

            if reference == "envelope":
                annot = _annotate_fn(tomo_id=float(tomo_val))
                bid_to_global: dict[float, int] = {float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in tomo_pos}
                for _, arow in annot.iterrows():
                    if float(arow["normal_angle"]) > 90.0:
                        g_idx = bid_to_global[float(arow["block_id"])]
                        flip_rot = self._get_flip_rot(g_idx)
                        R_list[g_idx] = R_list[g_idx] * flip_rot
                        z_arr[g_idx] = -z_arr[g_idx]
                continue

            c_tomo = all_c[tomo_pos]
            qp_idx_list, nn_idx_list = nnana.find_nn_within_radius(c_tomo, c_tomo, max_block_distance, remove_qp=True)

            adj: list[list[int]] = [[] for _ in range(n_tomo)]
            for qi, nns in zip(qp_idx_list, nn_idx_list):
                for ni in nns:
                    if ni not in adj[qi]:
                        adj[qi].append(ni)
                    if qi not in adj[ni]:
                        adj[ni].append(qi)

            subtomo_ids = self.blocks.df["subtomo_id"].values[tomo_pos]
            order = np.argsort(subtomo_ids)

            visited = np.zeros(n_tomo, dtype=bool)
            components: list[list[int]] = []

            for start_local in order:
                if visited[start_local]:
                    continue
                component: list[int] = []
                queue = [int(start_local)]
                visited[start_local] = True
                while queue:
                    m_local = queue.pop(0)
                    component.append(m_local)
                    m_global = int(tomo_pos[m_local])
                    for n_local in adj[m_local]:
                        if not visited[n_local]:
                            visited[n_local] = True
                            n_global = int(tomo_pos[n_local])
                            if np.dot(z_arr[n_global], z_arr[m_global]) < 0:
                                flip_rot = self._get_flip_rot(n_global)
                                R_list[n_global] = R_list[n_global] * flip_rot
                                z_arr[n_global] = -z_arr[n_global]
                            queue.append(n_local)
                components.append(component)

            if reference == "centroid":
                for comp in components:
                    g = c_tomo[comp].mean(axis=0)
                    total = sum(np.dot(z_arr[tomo_pos[i]], c_tomo[i] - g) for i in comp)
                    if total < 0:
                        for i in comp:
                            g_i = int(tomo_pos[i])
                            flip_rot = self._get_flip_rot(g_i)
                            R_list[g_i] = R_list[g_i] * flip_rot
                            z_arr[g_i] = -z_arr[g_i]

        final_angles = np.array([R_list[i].as_euler("zxz", degrees=True) for i in range(N)])
        self.blocks.df["phi"] = final_angles[:, 0]
        self.blocks.df["theta"] = final_angles[:, 1]
        self.blocks.df["psi"] = final_angles[:, 2]

        changed = 0
        for i in range(N):
            rel_angle = np.linalg.norm((R_list[i].inv() * all_R_init[i]).as_rotvec())
            if rel_angle > 1e-9:
                changed += 1

        self._site_table = None
        self._faces = None
        self._all_state = None
        return changed

    def get_inverted_blocks(self, max_block_distance: float, *, scope: Literal["original", "inferred", "all"] = "original") -> pd.DataFrame:
        """Return blocks whose orientation disagrees with a majority of their neighbours.

        Uses the same proximity graph and z-normal comparison as
        :meth:`unify_polarity` (``reference='neighbours'``).  Does **not**
        modify the motl.

        A block is considered inverted when ``n_disagree > n_agree``, i.e.
        more than half of its proximity neighbours have a negative dot product
        with its z-normal (``z_block · z_neighbour < 0``).

        Parameters
        ----------
        max_block_distance : float
            Maximum centre-to-centre distance (voxels) for two blocks to be
            considered neighbours.  Identical definition to
            :meth:`unify_polarity`.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses :attr:`blocks` only.  ``'all'`` uses lattice +
            inferred blocks combined.  ``'inferred'`` uses only inferred blocks.

        Returns
        -------
        pandas.DataFrame
            One row per inverted block, empty when none are found.

            ============================  =============================================
            ``tomo_id``                   tomogram identifier
            *affiliation_column*          assembly affiliation (from
                                          :attr:`affiliation_column`)
            ``subtomo_id``                particle identifier
            ``n_neighbours``              total proximity neighbours
            ``n_agree``                   neighbours with ``z·z > 0``
            ``n_disagree``                neighbours with ``z·z < 0``
            ``angle_to_mean_normal_deg``  angle (°) between this block's z-normal
                                          and the mean z-normal of its neighbours
            ============================  =============================================

        Raises
        ------
        ValueError
            If no block layer is present, or *scope* is invalid.
        """
        _blks = self._blocks_for_scope(scope)

        N = len(_blks.df)
        angles_arr = _blks.get_angles().astype(float)
        all_R = srot.from_euler("zxz", angles_arr, degrees=True)
        z_unit = np.array([0.0, 0.0, 1.0])
        z_arr = np.array([all_R[i].apply(z_unit) for i in range(N)])
        all_c = _blks.get_coordinates()

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        rows: list[dict] = []

        for tomo_val in sorted(_blks.df[tomo_col].unique()):
            tomo_mask = (_blks.df[tomo_col] == tomo_val).values
            tomo_pos = np.where(tomo_mask)[0]
            n_tomo = len(tomo_pos)
            if n_tomo == 0:
                continue

            c_tomo = all_c[tomo_pos]
            qp_idx_list, nn_idx_list = nnana.find_nn_within_radius(c_tomo, c_tomo, max_block_distance, remove_qp=True)

            adj: list[list[int]] = [[] for _ in range(n_tomo)]
            for qi, nns in zip(qp_idx_list, nn_idx_list):
                for ni in nns:
                    if ni not in adj[qi]:
                        adj[qi].append(ni)
                    if qi not in adj[ni]:
                        adj[ni].append(qi)

            for local_i in range(n_tomo):
                neighbours = adj[local_i]
                if not neighbours:
                    continue
                g_i = int(tomo_pos[local_i])
                z_i = z_arr[g_i]

                n_agree = sum(1 for nl in neighbours if np.dot(z_i, z_arr[int(tomo_pos[nl])]) > 0)
                n_disagree = sum(1 for nl in neighbours if np.dot(z_i, z_arr[int(tomo_pos[nl])]) < 0)
                if n_disagree <= n_agree:
                    continue

                mean_z = np.mean([z_arr[int(tomo_pos[nl])] for nl in neighbours], axis=0)
                norm_mz = np.linalg.norm(mean_z)
                if norm_mz < 1e-9:
                    angle_deg = 0.0
                else:
                    angle_deg = float(np.degrees(np.arccos(np.clip(np.dot(z_i, mean_z / norm_mz), -1.0, 1.0))))

                block_row = _blks.df.iloc[g_i]
                rows.append(
                    {
                        tomo_col: tomo_val,
                        aff_col: float(block_row[aff_col]),
                        "subtomo_id": float(block_row["subtomo_id"]),
                        "n_neighbours": len(neighbours),
                        "n_agree": n_agree,
                        "n_disagree": n_disagree,
                        "angle_to_mean_normal_deg": angle_deg,
                    }
                )

        return pd.DataFrame(rows)

    def get_inverted_blocks_as_motl(self, max_block_distance: float, *, scope: Literal["original", "inferred", "all"] = "original") -> "cryomotl.Motl":
        """Return inverted blocks as a :class:`~cryocat.core.cryomotl.Motl`.

        Calls :meth:`get_inverted_blocks` and filters the selected block pool
        to those rows.  Returns an empty :class:`~cryocat.core.cryomotl.Motl`
        when no inverted blocks are found.

        Parameters
        ----------
        max_block_distance : float
            Passed directly to :meth:`get_inverted_blocks`.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            Passed directly to :meth:`get_inverted_blocks`.

        Returns
        -------
        cryomotl.Motl
            Subset of the selected block pool containing only the inverted
            blocks, reset to a clean 1-based ``subtomo_id`` index.

        Raises
        ------
        ValueError
            If no block layer is present, or *scope* is invalid.
        """
        _blks = self._blocks_for_scope(scope)
        inv_df = self.get_inverted_blocks(max_block_distance, scope=scope)
        if inv_df.empty:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns))
        inv_ids = set(inv_df["subtomo_id"].values)
        mask = _blks.df["subtomo_id"].isin(inv_ids)
        return cryomotl.Motl(_blks.df[mask].copy().reset_index(drop=True))

    # ------------------------------------------------------------------
    # Contact graph
    # ------------------------------------------------------------------

    @gui_exposed(label="Connect", group="Lattice setup", order=30, returns="none", category="pleomorphic-op")
    def connect(
        self,
        max_distance: float,
        *,
        scope: str = "original",
    ) -> None:
        """Build and cache the contact graph from the block layer.

        Matches sites across blocks within *max_distance* voxels, assigns
        each block to a connected-component assembly, and computes faces as
        the minimum cycle basis of the block contact graph (weighted by
        centre-to-centre distance).

        After calling this method the following getters become available:
        :meth:`get_contact_stats`, :meth:`get_face_stats`,
        :meth:`get_block_stats`, :meth:`get_assembly_stats`.

        Parameters
        ----------
        max_distance : float
            Upper bound on site tip-to-tip distance (voxels).  Two contact
            points bond when they are within this distance of each other.
        scope : str, default=``'original'``
            ``'original'`` connects using :attr:`blocks` only and stores
            the result in the instance's primary contact-graph slots
            (``_site_table``, ``_faces``).  ``'all'`` is accepted for
            backward compatibility but has no lasting effect; the combined
            real + added graph is recomputed on demand whenever a
            ``scope='all'`` getter is called.

        Notes
        -----
        Matching algorithm:

        1. Two contact points from different blocks are candidates when
           their tip-to-tip distance is ≤ *max_distance*.
        2. Candidates are sorted by increasing tip distance and committed
           greedily: a pair is accepted only when both legs are still
           free.  Each leg therefore pairs with at most one partner.

        Raises
        ------
        ValueError
            If no block layer is present, or *scope* is not ``'original'``
            or ``'all'``.
        """
        if scope not in ("original", "all"):
            raise ValueError(f"connect() scope must be 'original' or 'all', got {scope!r}.")
        if self.blocks is None:
            raise ValueError("connect() requires a block layer (blocks=...).")
        _blks = self._blocks_for_scope(scope)
        self._connect_impl(max_distance, scope=scope, blocks=_blks)
        if scope == "original":
            self._all_state = None

    def _connect_impl(
        self,
        max_distance: float,
        *,
        scope: str = "original",
        blocks: "cryomotl.Motl | None" = None,
    ) -> "tuple[pd.DataFrame, list[dict], cryomotl.Motl]":
        """Internal connect logic; *blocks* is the motl to connect (defaults to ``self.blocks``)."""
        _blks = blocks if blocks is not None else self.blocks
        if _blks is None:
            raise ValueError("connect() requires a block layer (blocks=...).")

        # ----------------------------------------------------------------
        # 1. Build site table
        # ----------------------------------------------------------------
        site_motl = self.get_sites_as_motl(scope=scope)
        site_df = site_motl.df.copy()

        # Site positions (including shifts)
        site_coords = site_motl.get_coordinates()  # (H, 3)

        # Inverse site_type_codes lookup
        inv_site_type = {v: k for k, v in self.site_type_codes.items()}

        # Lookup effective_n_sites per block_id (cyclic: cyclic order; non-cyclic: n_sites)
        def _n_sites_for_block(bid: float) -> int:
            if isinstance(self.block_definition, dict):
                block_row = _blks.df[_blks.df["subtomo_id"] == bid]
                cls_val = float(block_row.iloc[0][self.block_type_column])
                return self.block_definition[cls_val].effective_n_sites
            return self.block_definition.effective_n_sites

        # Block centres
        block_coords_by_id: dict[float, np.ndarray] = {}
        for _, brow in _blks.df.iterrows():
            bid = float(brow["subtomo_id"])
            sx = float(brow.get("shift_x", 0.0))
            sy = float(brow.get("shift_y", 0.0))
            sz = float(brow.get("shift_z", 0.0))
            block_coords_by_id[bid] = np.array(
                [
                    float(brow["x"]) + sx,
                    float(brow["y"]) + sy,
                    float(brow["z"]) + sz,
                ]
            )

        H = len(site_df)
        block_ids = site_df[self.block_id_column].values.astype(float)
        site_indices = site_df[self.site_index_column].values.astype(float)
        site_type_codes_col = site_df[self.site_type_column].values.astype(float)
        tomo_ids = site_df[self.tomo_id_column].values

        site_type_str = np.array([inv_site_type.get(int(c), "") for c in site_type_codes_col])
        n_sites_arr = np.array([_n_sites_for_block(bid) for bid in block_ids], dtype=np.intp)
        cx_arr = np.array([block_coords_by_id[bid][0] for bid in block_ids])
        cy_arr = np.array([block_coords_by_id[bid][1] for bid in block_ids])
        cz_arr = np.array([block_coords_by_id[bid][2] for bid in block_ids])

        site_table = pd.DataFrame(
            {
                self.tomo_id_column: tomo_ids,
                "block_id": block_ids,
                "site": site_indices.astype(np.intp),
                "site_type": site_type_str,
                "n_sites": n_sites_arr,
                "x": site_coords[:, 0],
                "y": site_coords[:, 1],
                "z": site_coords[:, 2],
                "cx": cx_arr,
                "cy": cy_arr,
                "cz": cz_arr,
            }
        )
        # Alias for stable reference
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        # Map subtomo_id → affiliation value; -1 for any block not in blocks.df
        _aff_map: dict[float, float] = dict(
            zip(
                _blks.df["subtomo_id"].astype(float).values,
                _blks.df[aff_col].astype(float).values,
            )
        )
        site_table[aff_col] = [_aff_map.get(float(bid), -1.0) for bid in block_ids]
        site_table = site_table.sort_values(by=[tomo_col, aff_col, "block_id", "site"]).reset_index(drop=True)

        # ----------------------------------------------------------------
        # 2. Per-tomogram matching and assembly
        # ----------------------------------------------------------------
        partner_arr = np.full(H, -1, dtype=np.intp)
        n_candidates_arr = np.zeros(H, dtype=np.intp)
        assembly_id_arr = np.full(H, -1, dtype=float)
        face_id_arr = np.full(H, -1, dtype=np.intp)

        face_records: list[dict] = []

        # Precompute unit site-direction vectors: u[h] = (P_h - c_h) / |...|
        p_arr = site_table[["x", "y", "z"]].values
        c_arr = np.column_stack([cx_arr, cy_arr, cz_arr])[site_table.index]  # after sort

        # Recompute from sorted table
        p_arr = site_table[["x", "y", "z"]].values
        c_arr_sorted = site_table[["cx", "cy", "cz"]].values

        import networkx as _nx

        for (tomo_val, aff_val), grp_df in site_table.groupby([tomo_col, aff_col], sort=True):
            t_idx = grp_df.index.to_numpy()  # global row positions in site_table
            n_t = len(t_idx)

            P_t = p_arr[t_idx]
            block_t = site_table["block_id"].values[t_idx]
            site_t = site_table["site"].values[t_idx].astype(np.intp)
            stype_t = site_table["site_type"].values[t_idx]
            n_sites_t = site_table["n_sites"].values[t_idx].astype(np.intp)
            cc_arr = c_arr_sorted[t_idx]  # (n_t, 3) block centres for this group

            qp_idx, nn_idx_list = nnana.find_nn_within_radius(P_t, P_t, max_distance, remove_qp=True)

            nn_map: dict[int, np.ndarray] = {}
            for qi, nns in zip(qp_idx, nn_idx_list):
                nn_map[qi] = nns

            # Collect valid candidates (qi → set of ni) applying all filters.
            cand_for: dict[int, set[int]] = {}
            for qi in range(n_t):
                nns = nn_map.get(qi, np.array([], dtype=np.intp))
                for ni in nns:
                    # Filter: different block
                    if block_t[ni] == block_t[qi]:
                        continue
                    # Filter: allowed site-type pairing
                    if frozenset({stype_t[qi], stype_t[ni]}) not in self._allowed_pairs:
                        continue
                    cand_for.setdefault(qi, set()).add(ni)

            n_cand_local = np.array([len(cand_for.get(qi, set())) for qi in range(n_t)], dtype=np.intp)

            # Greedy matching: sort all valid undirected pairs by tip distance,
            # then commit each pair in order as long as both legs are still free.
            all_pairs: list[tuple[float, int, int]] = []
            for qi, cands in cand_for.items():
                for ni in cands:
                    if qi < ni:
                        d = float(np.linalg.norm(P_t[ni] - P_t[qi]))
                        all_pairs.append((d, qi, ni))
            all_pairs.sort()

            partner_local = np.full(n_t, -1, dtype=np.intp)
            matched = np.zeros(n_t, dtype=bool)
            for _d, qi, ni in all_pairs:
                if not matched[qi] and not matched[ni]:
                    partner_local[qi] = ni
                    partner_local[ni] = qi
                    matched[qi] = True
                    matched[ni] = True

            # Copy to global arrays
            for local_i, global_i in enumerate(t_idx):
                partner_arr[global_i] = t_idx[partner_local[local_i]] if partner_local[local_i] >= 0 else -1
                n_candidates_arr[global_i] = n_cand_local[local_i]

            # ---- Assemblies ----
            qp_ids_asm: list[float] = []
            nn_ids_asm: list[float] = []
            for local_i in range(n_t):
                lp = partner_local[local_i]
                if lp >= 0:
                    qp_ids_asm.append(float(block_t[local_i]))
                    nn_ids_asm.append(float(block_t[lp]))

            block_ids_tomo = np.unique(block_t)
            block_assembly: dict[float, int] = {}

            if qp_ids_asm:
                components = _clustering.connected_component_clusters(qp_ids_asm, nn_ids_asm, min_size=1)
                next_asm = 1
                in_component: set = set()
                for comp in components:
                    comp_ids = set(comp.nodes())
                    for bid in comp_ids:
                        block_assembly[float(bid)] = next_asm
                    in_component.update(comp_ids)
                    next_asm += 1
                for bid in block_ids_tomo:
                    if bid not in in_component:
                        block_assembly[bid] = next_asm
                        next_asm += 1
            else:
                for k, bid in enumerate(sorted(block_ids_tomo), start=1):
                    block_assembly[bid] = k

            for local_i, global_i in enumerate(t_idx):
                assembly_id_arr[global_i] = float(block_assembly.get(block_t[local_i], -1))

            # Node ids are plain Python ints — numpy integers can break networkx.
            from collections import deque as _deque

            G_faces = _nx.Graph()
            he_lookup: dict[tuple[int, int], int] = {}  # (block_a, block_b) → local_i
            for local_i in range(n_t):
                p = int(partner_local[local_i])
                if p >= 0:
                    ba = int(block_t[local_i])
                    bb = int(block_t[p])
                    he_lookup[(ba, bb)] = local_i
                    if not G_faces.has_edge(ba, bb):
                        dist = float(np.linalg.norm(cc_arr[p] - cc_arr[local_i]))
                        G_faces.add_edge(ba, bb, weight=dist)

            cycles_to_assign = [
                _c for _c in _nx.minimum_cycle_basis(G_faces, weight="weight")
                if len(_c) <= self.max_cycle_length
            ]
            n_mcb = len(cycles_to_assign)

            _block_center: dict[int, np.ndarray] = {}
            for _li in range(n_t):
                _bid = int(block_t[_li])
                if _bid not in _block_center:
                    _block_center[_bid] = cc_arr[_li]

            def _cycle_area_3d(_nodes: list) -> float:
                _pts = np.array([_block_center.get(int(_n), np.zeros(3)) for _n in _nodes])
                _c = _pts.mean(0)
                _n = len(_pts)
                _av = np.zeros(3)
                for _ii in range(_n):
                    _av += np.cross(_pts[_ii] - _c, _pts[(_ii + 1) % _n] - _c)
                return float(np.linalg.norm(_av)) / 2.0

            _cycle_areas = [_cycle_area_3d(_cn) for _cn in cycles_to_assign]

            fwd_edges: list[set[tuple[int, int]]] = []
            for _cn in cycles_to_assign:
                _nc = len(_cn)
                fwd_edges.append({(int(_cn[k]), int(_cn[(k + 1) % _nc])) for k in range(_nc)})
            edge_to_fwd: dict[tuple[int, int], list[int]] = {}
            for _ci, _fe in enumerate(fwd_edges):
                for _e in _fe:
                    edge_to_fwd.setdefault(_e, []).append(_ci)

            def _run_bfs(_mask: list[bool]) -> list[bool | None]:
                _ori: list[bool | None] = [None] * n_mcb
                _q: _deque[int] = _deque()
                for _s in range(n_mcb):
                    if not _mask[_s] or _ori[_s] is not None:
                        continue
                    _ori[_s] = True
                    _q.append(_s)
                    while _q:
                        _c = _q.popleft()
                        _mdir: set[tuple[int, int]] = (
                            fwd_edges[_c] if _ori[_c] else {(_b, _a) for (_a, _b) in fwd_edges[_c]}
                        )
                        for _a, _b in _mdir:
                            for _adj in edge_to_fwd.get((_b, _a), []):
                                if _adj != _c and _mask[_adj] and _ori[_adj] is None:
                                    _ori[_adj] = True
                                    _q.append(_adj)
                            for _adj in edge_to_fwd.get((_a, _b), []):
                                if _adj != _c and _mask[_adj] and _ori[_adj] is None:
                                    _ori[_adj] = False
                                    _q.append(_adj)
                return _ori

            _all_face: list[bool] = [True] * n_mcb
            _ori_p1 = _run_bfs(_all_face)

            _he_claimed_p1: dict[int, int] = {}
            _has_conflict_p1 = False
            _conflict_pair: tuple[int, int] = (-1, -1)
            for _ci, _cn in enumerate(cycles_to_assign):
                _nc = len(_cn)
                _d = _cn if _ori_p1[_ci] else list(reversed(_cn))
                for _i in range(_nc):
                    _ba = int(_d[_i])
                    _bb = int(_d[(_i + 1) % _nc])
                    _he = he_lookup.get((_ba, _bb))
                    if _he is not None:
                        if _he in _he_claimed_p1:
                            _has_conflict_p1 = True
                            _conflict_pair = (_he_claimed_p1[_he], _ci)
                            break
                        _he_claimed_p1[_he] = _ci
                if _has_conflict_p1:
                    break

            def _check_conflicts(_fm: list[bool], _ori: list[bool | None]) -> bool:
                _hc: dict[int, int] = {}
                for _ci2, _cn2 in enumerate(cycles_to_assign):
                    if not _fm[_ci2]:
                        continue
                    _nc2 = len(_cn2)
                    _d2 = _cn2 if _ori[_ci2] else list(reversed(_cn2))
                    for _i2 in range(_nc2):
                        _ba2 = int(_d2[_i2])
                        _bb2 = int(_d2[(_i2 + 1) % _nc2])
                        _he2 = he_lookup.get((_ba2, _bb2))
                        if _he2 is not None:
                            if _he2 in _hc:
                                return True
                            _hc[_he2] = _ci2
                return False

            if not _has_conflict_p1:
                _face_mask: list[bool] = _all_face
                orientation: list[bool | None] = _ori_p1
            else:
                _cf, _cs = _conflict_pair
                _pair_order = sorted([_cf, _cs], key=lambda _i: _cycle_areas[_i], reverse=True)
                _topo_loop_set: set[int] = set()
                orientation = _ori_p1
                _face_mask = _all_face
                _resolved = False
                for _cand in _pair_order:
                    _fm_try: list[bool] = [_ci != _cand for _ci in range(n_mcb)]
                    _ori_try = _run_bfs(_fm_try)
                    if not _check_conflicts(_fm_try, _ori_try):
                        _face_mask = _fm_try
                        orientation = _ori_try
                        _topo_loop_set = {_cand}
                        _resolved = True
                        break
                if not _resolved:
                    _area_order = sorted(range(n_mcb), key=lambda _i: _cycle_areas[_i], reverse=True)
                    _topo_loop_set = set()
                    for _candidate in _area_order:
                        _topo_loop_set.add(_candidate)
                        _fm_try = [_ci not in _topo_loop_set for _ci in range(n_mcb)]
                        _ori_try = _run_bfs(_fm_try)
                        if not _check_conflicts(_fm_try, _ori_try):
                            _face_mask = _fm_try
                            orientation = _ori_try
                            break

            local_fid = 0
            assigned_he: set[int] = set()
            for _ci, cycle_nodes in enumerate(cycles_to_assign):
                if not _face_mask[_ci]:
                    continue
                n_cycle = len(cycle_nodes)
                _dir = cycle_nodes if orientation[_ci] else list(reversed(cycle_nodes))
                local_hedges: list[int] = []
                _ok = True
                for _i in range(n_cycle):
                    _ba = int(_dir[_i])
                    _bb = int(_dir[(_i + 1) % n_cycle])
                    _he = he_lookup.get((_ba, _bb))
                    if _he is None or _he in assigned_he:
                        _ok = False
                        break
                    local_hedges.append(_he)
                if not _ok:
                    continue
                local_fid += 1
                assigned_he.update(local_hedges)
                for li in local_hedges:
                    face_id_arr[t_idx[li]] = local_fid
                global_hedges = [int(t_idx[li]) for li in local_hedges]
                asm_id = float(block_assembly.get(float(block_t[local_hedges[0]]), -1)) if local_hedges else -1.0
                face_records.append(
                    {
                        tomo_col: tomo_val,
                        aff_col: aff_val,
                        "face_id": local_fid,
                        "assembly_id": asm_id,
                        "half_edges": global_hedges,
                    }
                )

            _boundary_block_ids: set[int] = {int(block_t[_li]) for _li in range(n_t) if partner_local[_li] < 0}

            _outgoing: dict[int, tuple[int, int]] = {}
            for (_ba, _bb), _li in he_lookup.items():
                if _li not in assigned_he:
                    _outgoing[_ba] = (_bb, _li)

            _visited_rec: set[int] = set()
            while _outgoing:
                _starts = [_k for _k in _outgoing if _k not in _visited_rec]
                if not _starts:
                    break
                _start = _starts[0]
                _cycle_he: list[int] = []
                _ba = _start
                while True:
                    if _ba not in _outgoing:
                        _cycle_he = []
                        break
                    _bb, _li = _outgoing[_ba]
                    _cycle_he.append(_li)
                    _visited_rec.add(_ba)
                    del _outgoing[_ba]
                    _ba = _bb
                    if _ba == _start:
                        break
                    if _ba in _visited_rec:
                        _cycle_he = []
                        break
                if len(_cycle_he) < 3:
                    continue
                if not _cycle_he:
                    continue
                if _boundary_block_ids:
                    continue
                local_fid += 1
                assigned_he.update(_cycle_he)
                for _li in _cycle_he:
                    face_id_arr[t_idx[_li]] = local_fid
                global_hedges = [int(t_idx[_li]) for _li in _cycle_he]
                asm_id = float(block_assembly.get(float(block_t[_cycle_he[0]]), -1)) if _cycle_he else -1.0
                face_records.append(
                    {
                        tomo_col: tomo_val,
                        aff_col: aff_val,
                        "face_id": local_fid,
                        "assembly_id": asm_id,
                        "half_edges": global_hedges,
                    }
                )

        # ----------------------------------------------------------------
        # 3. Store
        # ----------------------------------------------------------------
        site_table["partner"] = partner_arr
        site_table["n_candidates"] = n_candidates_arr
        site_table["assembly_id"] = assembly_id_arr
        site_table["face_id"] = face_id_arr

        if scope != "all":
            self._site_table = site_table
            self._faces = face_records
            self._max_distance = float(max_distance)
        return site_table, face_records, _blks

    def _require_connect(self, scope: str = "original") -> None:
        """Raise ValueError if connect() has not been called."""
        if self._faces is None:
            raise ValueError("Call connect() first to build the contact graph.")
        if scope not in ("original", "all", "inferred"):
            raise ValueError(f"Unknown scope {scope!r}. Use 'original', 'inferred', or 'all'.")

    def _blocks_for_scope(self, scope: str) -> "cryomotl.Motl":
        """Return the block motl for *scope*.

        ``'original'`` returns :attr:`blocks` unchanged.  ``'all'`` returns a
        merged motl of real blocks (origin 1 in ``_origin_column``) and any
        inferred blocks stored in :attr:`_added` (origin 2), renumbered from 1.
        ``'inferred'`` returns only the blocks in :attr:`_added`.
        """
        if scope not in ("original", "all", "inferred"):
            raise ValueError(f"Unknown scope {scope!r}. Use 'original', 'inferred', or 'all'.")
        if self.blocks is None:
            raise ValueError("_blocks_for_scope requires a block layer (blocks=...).")
        if scope == "inferred":
            if self._added is None or len(self._added.df) == 0:
                raise ValueError(
                    "No inferred blocks: call infer_missing_blocks() first."
                )
            return self._added
        if scope == "original" or self._added is None or len(self._added.df) == 0:
            return self.blocks
        real_df = self.blocks.df.copy()
        _oc = self._origin_column
        _unset = real_df[_oc] == 0.0
        if _unset.any():
            real_df.loc[_unset, _oc] = 1.0
        added_df = self._added.df.copy()
        merged_df = pd.concat([real_df, added_df], ignore_index=True)
        merged_df["subtomo_id"] = np.arange(1, len(merged_df) + 1, dtype=float)
        type_col = self.block_type_column
        zero_mask = merged_df[type_col] == 0.0
        if zero_mask.any():
            mode_series = self.blocks.df[type_col].mode()
            mode_val = float(mode_series.iloc[0]) if len(mode_series) > 0 else 1.0
            if mode_val == 0.0:
                mode_val = 1.0
            merged_df.loc[zero_mask, type_col] = mode_val
        return cryomotl.Motl(merged_df)

    def _state_for_scope(
        self, scope: str
    ) -> "tuple[pd.DataFrame, list[dict], cryomotl.Motl]":
        """Return ``(site_table, faces, blocks_motl)`` for *scope*.

        Raises ``ValueError`` if the required :meth:`connect` call has not
        been made.
        """
        if scope == "original":
            self._require_connect(scope="original")
            return self._site_table, self._faces, self.blocks
        elif scope == "all":
            self._require_connect(scope="original")
            if self._all_state is None:
                raise ValueError(
                    "No inferred blocks: call infer_missing_blocks() first."
                )
            return (
                self._all_state["site_table"],
                self._all_state["faces"],
                self._all_state["blocks"],
            )
        elif scope == "inferred":
            self._require_connect(scope="original")
            if self._all_state is None:
                raise ValueError(
                    "No inferred blocks: call infer_missing_blocks() first."
                )
            return (
                self._all_state["site_table"],
                self._all_state["added_faces"],
                self._all_state["blocks"],
            )
        else:
            raise ValueError(f"Unknown scope {scope!r}. Use 'original', 'inferred', or 'all'.")

    def _update_added(self, new_df: pd.DataFrame, origin: int) -> None:
        """Replace the *origin*-tagged portion of :attr:`_added` with *new_df*.

        Rows in ``_added`` whose ``_origin_column`` value equals *origin* are
        dropped and replaced by *new_df*.  The combined result is renumbered.
        The ``'all'`` scope graph is recomputed on demand from the current
        :attr:`_added`, so no cache invalidation is needed.
        """
        new_df = new_df.copy()
        new_df[self._origin_column] = float(origin)
        if self._added is None or len(self._added.df) == 0:
            combined = new_df
        else:
            keep_df = self._added.df[
                self._added.df[self._origin_column] != float(origin)
            ].copy()
            combined = pd.concat([keep_df, new_df], ignore_index=True)
        if len(combined) > 0:
            self._added = cryomotl.Motl(combined)
            _next_id = int(self.blocks.df["subtomo_id"].max()) + 1
            self._added.df["subtomo_id"] = [float(_next_id + i) for i in range(len(self._added.df))]
        else:
            self._added = None

        # Rebuild the "all" scope state so statistics calls read current data.
        if self._added is not None and self._faces is not None:
            _blks_a = self._blocks_for_scope("all")
            _st_a, _faces_a, _blks_a_out = self._connect_impl(
                self._max_distance,
                scope="all",
                blocks=_blks_a,
            )

            # he_lookup for all-scope site_table: (new_block_a, new_block_b) -> row index
            _a_bid_vals = _st_a["block_id"].values
            _a_part_vals = _st_a["partner"].values
            _all_he: dict[tuple[int, int], int] = {}
            for _hi in range(len(_a_bid_vals)):
                _pid = int(_a_part_vals[_hi])
                if _pid >= 0:
                    _all_he[(int(_a_bid_vals[_hi]), int(_a_bid_vals[_pid]))] = _hi

            # _blocks_for_scope("all") places original blocks first, renumbered 1..N_orig.
            _orig_blk_df = self.blocks.df
            _orig_subid_to_new: dict[float, int] = {
                float(_orig_blk_df.iloc[_i]["subtomo_id"]): _i + 1
                for _i in range(len(_orig_blk_df))
            }

            tomo_col = self.tomo_id_column
            aff_col = self.affiliation_column
            _union_vsets: set[frozenset] = set()
            _union_faces: list[dict] = []
            _added_faces: list[dict] = []
            _fid_ctr: dict[tuple, int] = {}

            # Remap original faces to all-scope site_table indices.
            for _orig_ff in self._faces:
                _orig_bids = [float(self._site_table.at[_h, "block_id"]) for _h in _orig_ff["half_edges"]]
                _new_bids = [_orig_subid_to_new.get(_b) for _b in _orig_bids]
                if any(_b is None for _b in _new_bids):
                    continue
                _vset = frozenset(int(_b) for _b in _new_bids)
                if _vset in _union_vsets:
                    continue
                _new_hes: list[int] = []
                _map_ok = True
                for _i in range(len(_new_bids)):
                    _ba = int(_new_bids[_i])
                    _bb = int(_new_bids[(_i + 1) % len(_new_bids)])
                    _he = _all_he.get((_ba, _bb))
                    if _he is None:
                        _he = _all_he.get((_bb, _ba))
                    if _he is None:
                        _map_ok = False
                        break
                    _new_hes.append(_he)
                if not _map_ok or not _new_hes:
                    continue
                _tv = _orig_ff[tomo_col]
                _av = _orig_ff[aff_col]
                _k = (_tv, _av)
                _fid_ctr[_k] = _fid_ctr.get(_k, 0) + 1
                _union_vsets.add(_vset)
                _union_faces.append({
                    tomo_col: _tv,
                    aff_col: _av,
                    "face_id": _fid_ctr[_k],
                    "assembly_id": _orig_ff["assembly_id"],
                    "half_edges": _new_hes,
                })

            # Add all-scope MCB faces not already covered by the original.
            for _all_ff in _faces_a:
                _vset = frozenset(int(_a_bid_vals[_h]) for _h in _all_ff["half_edges"])
                if _vset in _union_vsets:
                    continue
                _tv = _all_ff[tomo_col]
                _av = _all_ff[aff_col]
                _k = (_tv, _av)
                _fid_ctr[_k] = _fid_ctr.get(_k, 0) + 1
                _union_vsets.add(_vset)
                _rec = {
                    tomo_col: _tv,
                    aff_col: _av,
                    "face_id": _fid_ctr[_k],
                    "assembly_id": _all_ff["assembly_id"],
                    "half_edges": _all_ff["half_edges"],
                }
                _union_faces.append(_rec)
                _added_faces.append(_rec)

            # Update face_id in the all-scope site_table to reflect the union.
            _st_union = _st_a.copy()
            _st_union["face_id"] = 0.0
            for _uf in _union_faces:
                for _h in _uf["half_edges"]:
                    _st_union.at[_h, "face_id"] = float(_uf["face_id"])

            self._all_state = {
                "site_table": _st_union,
                "faces": _union_faces,
                "added_faces": _added_faces,
                "blocks": _blks_a_out,
            }
        else:
            self._all_state = None

    # ------------------------------------------------------------------
    # Statistics getters
    # ------------------------------------------------------------------

    @gui_exposed(
        label="Contact stats", group="Lattice statistics", order=10, returns="dataframe", category="pleomorphic-op"
    )
    def get_contact_stats(self, scope: Literal["original", "inferred", "all"] = "original") -> pd.DataFrame:
        """Return one row per matched half-edge (each contact appears twice).

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'`` uses
            the combined lattice + inferred block graph.  ``'inferred'`` returns
            only contacts that participate in inferred faces.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``block_id``, ``site``, ``site_type``,
            ``partner_block_id``, ``partner_site``, ``assembly_id``,
            ``face_id``, ``n_candidates``, ``site_distance``,
            ``block_distance``, ``bend_angle``, ``label_offset``,
            ``mismatch_x``, ``mismatch_y``, ``mismatch_z``.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called for *scope*, or *scope* is
            ``'inferred'`` and :meth:`infer_missing_blocks` has not been called.
        """
        st, _, _blocks = self._state_for_scope(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        matched_mask = st["partner"].values >= 0
        h_idx = np.where(matched_mask)[0]

        if len(h_idx) == 0:
            return pd.DataFrame(
                columns=[
                    tomo_col,
                    aff_col,
                    "block_id",
                    "site",
                    "site_type",
                    "partner_block_id",
                    "partner_site",
                    "assembly_id",
                    "face_id",
                    "n_candidates",
                    "site_distance",
                    "block_distance",
                    "bend_angle",
                    "label_offset",
                    "mismatch_x",
                    "mismatch_y",
                    "mismatch_z",
                ]
            )

        # Block rotations
        angles_all = _blocks.get_angles().astype(float)
        all_R = srot.from_euler("zxz", angles_all, degrees=True)
        block_to_row: dict[float, int] = {
            float(_blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(_blocks.df))
        }
        z_unit = np.array([0.0, 0.0, 1.0])

        rows = []
        for h in h_idx:
            p = int(st.at[h, "partner"])
            block_h = float(st.at[h, "block_id"])
            block_p = float(st.at[p, "block_id"])
            P_h = np.array([st.at[h, "x"], st.at[h, "y"], st.at[h, "z"]])
            P_p = np.array([st.at[p, "x"], st.at[p, "y"], st.at[p, "z"]])
            c_h = np.array([st.at[h, "cx"], st.at[h, "cy"], st.at[h, "cz"]])
            c_p = np.array([st.at[p, "cx"], st.at[p, "cy"], st.at[p, "cz"]])

            site_dist = float(np.linalg.norm(P_p - P_h)) * self._block_pixel_size
            block_dist = float(np.linalg.norm(c_p - c_h)) * self._block_pixel_size

            R_h = all_R[block_to_row[block_h]]
            R_p = all_R[block_to_row[block_p]]
            z_h = R_h.apply(z_unit)
            z_p = R_p.apply(z_unit)
            bend = float(geom.angle_between_vectors(z_h.reshape(1, 3), z_p.reshape(1, 3))[0])

            ns_h = int(st.at[h, "n_sites"])
            ns_p = int(st.at[p, "n_sites"])
            if ns_h == ns_p:
                label_offset = float((int(st.at[p, "site"]) - int(st.at[h, "site"])) % ns_h)
            else:
                label_offset = float("nan")

            mismatch_world = P_p - P_h
            mismatch_local = R_h.inv().apply(mismatch_world)

            rows.append(
                {
                    tomo_col: st.at[h, tomo_col],
                    aff_col: float(st.at[h, aff_col]),
                    "block_id": block_h,
                    "site": int(st.at[h, "site"]),
                    "site_type": st.at[h, "site_type"],
                    "partner_block_id": block_p,
                    "partner_site": int(st.at[p, "site"]),
                    "assembly_id": float(st.at[h, "assembly_id"]),
                    "face_id": int(st.at[h, "face_id"]),
                    "n_candidates": int(st.at[h, "n_candidates"]),
                    "site_distance": site_dist,
                    "block_distance": block_dist,
                    "bend_angle": bend,
                    "label_offset": label_offset,
                    "mismatch_x": float(mismatch_local[0]),
                    "mismatch_y": float(mismatch_local[1]),
                    "mismatch_z": float(mismatch_local[2]),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _face_hole_analysis(centres: np.ndarray, normal: np.ndarray) -> dict:
        """Detect missing blocks in a face via two complementary angle tests.

        **Primary — interior angles**: in a hexagonal lattice every vertex
        corner is ≈ 120° regardless of the face it bounds.  A genuine n-gon
        would have corners of (n-2)·180/n degrees; if the median corner is more
        than 20° below that expectation the face is classified as *merged* (an
        oversized face created by removing a block).

        **Secondary — central angle spacing**: azimuths from the face centroid
        to each vertex should be spaced 360/n apart.  This test is reported for
        diagnostics but is **not** used for the merged/not-merged decision:
        in a trivalent network a merged 12-vertex face still has all 12 vertices
        evenly spaced (≈ 30° each) around the centroid, indistinguishable from a
        genuine 12-gon by this measure alone.

        Both angle computations reuse :func:`geom.vector_angular_distance_signed`
        (``atan2(n·(u×v), u·v)``).

        **Missing block count** (for merged faces): each missing interior block
        leaves exactly three *reflex* vertices in the boundary — vertices where
        the boundary turns opposite to the dominant winding direction by more
        than π/6 (30°).  ``n_missing = n_reflex // 3``.  Non-multiples of 3 are
        returned as 0 (ambiguous).

        **Position**: centroid of each group of three reflex neighbour positions
        — exact for a regular lattice, approximate otherwise.

        Parameters
        ----------
        centres : ndarray, shape (n, 3)
            Block centre positions **in boundary-walk order**.
        normal : ndarray, shape (3,)
            Unit vector normal to the face plane.

        Returns
        -------
        dict
            ``n_missing`` : int
            ``missing_positions`` : list[ndarray shape (3,)]
            ``neighbour_groups`` : list[list[int]]
            ``is_merged`` : bool — True when median interior angle deviates from
                the ideal n-gon value by more than 20°.
            ``interior_angles_deg`` : ndarray shape (n,)
            ``median_interior_deg`` : float
            ``expected_interior_deg`` : float — (n-2)·180/n
            ``median_spacing_deg`` : float — central-angle spacing (secondary)
            ``expected_spacing_deg`` : float — 360/n
            ``n_reflex`` : int — reflex-vertex count (turn angle < −30° in the
                minority winding direction); non-zero only when ``is_merged`` is
                True and the count is a multiple of 3.
        """
        n = len(centres)
        _empty: dict = {
            "n_missing": 0,
            "missing_positions": [],
            "neighbour_groups": [],
            "is_merged": False,
            "interior_angles_deg": np.zeros(max(n, 1)),
            "median_interior_deg": 0.0,
            "expected_interior_deg": 0.0,
            "median_spacing_deg": 0.0,
            "expected_spacing_deg": 0.0,
            "n_reflex": 0,
        }
        if n < 4:
            return _empty

        n_unit = normal / (np.linalg.norm(normal) + 1e-30)
        centroid = centres.mean(axis=0)
        centred = centres - centroid
        in_plane = centred - (centred @ n_unit)[:, None] * n_unit

        # ── turn angles (exterior angles at each boundary vertex) ─────────────
        turn_angles = np.empty(n)
        for i in range(n):
            u = in_plane[i] - in_plane[i - 1]
            v = in_plane[(i + 1) % n] - in_plane[i]
            u_len = float(np.linalg.norm(u))
            v_len = float(np.linalg.norm(v))
            if u_len < 1e-12 or v_len < 1e-12:
                turn_angles[i] = 0.0
                continue
            turn_angles[i] = geom.vector_angular_distance_signed(u / u_len, v / v_len, n_unit)

        # ── interior angles = π − turn_angle ─────────────────────────────────
        interior_angles_deg = np.degrees(np.pi - turn_angles)
        median_interior_deg = float(np.median(interior_angles_deg))
        expected_interior_deg = float((n - 2) * 180.0 / n)
        is_merged = median_interior_deg < expected_interior_deg - 20.0

        # ── central angle spacing (secondary, diagnostic only) ────────────────
        radii = np.array([float(np.linalg.norm(ip)) for ip in in_plane])
        if np.any(radii > 1e-12):
            spacings_deg: list[float] = []
            for i in range(n):
                r0 = in_plane[i]
                r1 = in_plane[(i + 1) % n]
                r0n = float(np.linalg.norm(r0))
                r1n = float(np.linalg.norm(r1))
                if r0n < 1e-12 or r1n < 1e-12:
                    continue
                ang = geom.vector_angular_distance_signed(r0 / r0n, r1 / r1n, n_unit)
                spacings_deg.append(abs(float(np.degrees(ang))))
            median_spacing_deg = float(np.median(spacings_deg)) if spacings_deg else 0.0
        else:
            median_spacing_deg = 0.0
        expected_spacing_deg = 360.0 / n

        # ── reflex vertex detection (requires is_merged) ──────────────────────
        thr_dom = np.pi / 12  # 15°
        n_pos = int(np.sum(turn_angles > thr_dom))
        n_neg = int(np.sum(turn_angles < -thr_dom))
        if n_pos == 0 and n_neg == 0 or not is_merged:
            return {
                "n_missing": 0,
                "missing_positions": [],
                "neighbour_groups": [],
                "is_merged": is_merged,
                "interior_angles_deg": interior_angles_deg,
                "median_interior_deg": median_interior_deg,
                "expected_interior_deg": expected_interior_deg,
                "median_spacing_deg": median_spacing_deg,
                "expected_spacing_deg": expected_spacing_deg,
                "n_reflex": 0,
            }
        dominant_sign = 1 if n_pos >= n_neg else -1

        thr_reflex = np.pi / 6  # 30°
        reflex_idx: list[int] = [i for i in range(n) if turn_angles[i] * dominant_sign < -thr_reflex]

        n_reflex = len(reflex_idx)
        if n_reflex == 0 or n_reflex % 3 != 0:
            return {
                "n_missing": 0,
                "missing_positions": [],
                "neighbour_groups": [],
                "is_merged": is_merged,
                "interior_angles_deg": interior_angles_deg,
                "median_interior_deg": median_interior_deg,
                "expected_interior_deg": expected_interior_deg,
                "median_spacing_deg": median_spacing_deg,
                "expected_spacing_deg": expected_spacing_deg,
                "n_reflex": n_reflex,
            }

        n_missing = n_reflex // 3
        missing_positions: list[np.ndarray] = []
        neighbour_groups: list[list[int]] = []
        for k in range(n_missing):
            group = reflex_idx[3 * k : 3 * k + 3]
            missing_positions.append(centres[group].mean(axis=0))
            neighbour_groups.append(group)

        return {
            "n_missing": n_missing,
            "missing_positions": missing_positions,
            "neighbour_groups": neighbour_groups,
            "is_merged": is_merged,
            "interior_angles_deg": interior_angles_deg,
            "median_interior_deg": median_interior_deg,
            "expected_interior_deg": expected_interior_deg,
            "median_spacing_deg": median_spacing_deg,
            "expected_spacing_deg": expected_spacing_deg,
            "n_reflex": n_reflex,
        }

    @gui_exposed(
        label="Face stats", group="Lattice statistics", order=20, returns="dataframe", category="pleomorphic-op"
    )
    def get_face_stats(self, scope: Literal["original", "inferred", "all"] = "original") -> pd.DataFrame:
        """Return one row per closed face.

        Faces reflect **connectivity only** — the undirected block contact graph.
        A block whose orientation is inverted relative to its neighbours keeps
        its bonds, so it does not change the face count or topology (χ is
        unaffected).  Use :meth:`get_inverted_blocks` to identify such blocks
        before or after connecting.

        All columns are derived directly from the contact graph; no inference is
        performed here.  For merged-face detection and missing block counts use
        :meth:`get_face_inference`.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.
            ``'all'`` uses the combined lattice + inferred block graph and adds
            the per-face column ``n_added`` (count of inferred blocks per face).
            ``'inferred'`` returns only faces that exist in ``'all'`` but not in
            ``'original'``, also with ``n_added``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``face_id``, ``assembly_id``, ``size``,
            ``n_unique_blocks``, ``block_ids``, ``centroid_x``, ``centroid_y``,
            ``centroid_z``, ``normal_x``, ``normal_y``, ``normal_z``,
            ``n_vertices`` (= ``size``),
            ``n_boundary_blocks`` (face blocks with at least one unmatched site),
            ``n_inferred_blocks`` (blocks whose origin column equals 2; 0 when no
            inferred blocks are present),
            ``planarity`` (RMS out-of-plane deviation of cycle blocks divided by
            the cycle radius; 0 for a perfectly flat face, larger for a cycle
            that wanders off the shell),
            ``centroid_depth`` (distance from the face centroid to the nearest
            block in the assembly, divided by the cycle radius; small for a face
            lying on the shell, large for a shortcut cycle crossing the interior).
            When *scope* is ``'all'`` or ``'inferred'``: additionally ``n_added``
            (count of origin-2 blocks in the face).

            Radius is the mean Euclidean distance of the cycle's blocks from
            their centroid.  The face normal is the normalised mean of the
            block z-vectors (each block's orientation applied to [0, 0, 1]).

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called, or *scope* is ``'inferred'``
            and :meth:`infer_missing_blocks` has not been called.
        """
        st, _faces, _blocks = self._state_for_scope(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        # Blocks with at least one unmatched site, keyed by (tomo_id, block_id).
        boundary_block_set: set[tuple] = set(
            (row[tomo_col], row["block_id"])
            for _, row in st[st["partner"] < 0][[tomo_col, "block_id"]].drop_duplicates().iterrows()
        )

        # Per-assembly unique block centres for centroid_depth.
        asm_block_centres: dict[tuple, np.ndarray] = {}
        for (_tv, _av), _grp in st.groupby([tomo_col, aff_col]):
            _pts = _grp.drop_duplicates(subset=["block_id"])[["cx", "cy", "cz"]].values.astype(float)
            asm_block_centres[(_tv, _av)] = _pts

        _origin_col = self._origin_column
        if _blocks is not None and _origin_col in _blocks.df.columns:
            _origin_lookup: dict[float, float] = dict(
                zip(_blocks.df["subtomo_id"].astype(float), _blocks.df[_origin_col].astype(float))
            )
        else:
            _origin_lookup = None

        # Particle rotations for face-normal computation.
        if _blocks is not None and len(_blocks.df) > 0:
            _ang_fs = _blocks.get_angles().astype(float)
            _all_R_fs = srot.from_euler("zxz", _ang_fs, degrees=True)
            _b2r_fs: dict[float, int] = {
                float(_blocks.df.iloc[i]["subtomo_id"]): i
                for i in range(len(_blocks.df))
            }
        else:
            _all_R_fs = None
            _b2r_fs = None
        _zu_fs = np.array([0.0, 0.0, 1.0])

        rows = []
        for rec in _faces:
            hedges = rec["half_edges"]
            size = len(hedges)
            block_ids_walk = [float(st.at[h, "block_id"]) for h in hedges]
            cx_vals = np.array([st.at[h, "cx"] for h in hedges])
            cy_vals = np.array([st.at[h, "cy"] for h in hedges])
            cz_vals = np.array([st.at[h, "cz"] for h in hedges])
            centres = np.column_stack([cx_vals, cy_vals, cz_vals])
            centroid = centres.mean(axis=0)

            # Face normal: mean z-vector of the face's block orientations.
            if _all_R_fs is not None:
                _zvecs_fs = [
                    _all_R_fs[_b2r_fs[bid]].apply(_zu_fs)
                    for bid in block_ids_walk
                    if bid in _b2r_fs
                ]
            else:
                _zvecs_fs = []
            if _zvecs_fs:
                normal_raw = np.array(_zvecs_fs).mean(axis=0)
            else:
                normal_raw = np.array([0.0, 0.0, 1.0])
            n_norm = float(np.linalg.norm(normal_raw))
            normal_unit = normal_raw / n_norm if n_norm > 1e-15 else np.array([0.0, 0.0, 1.0])

            tomo_val = rec[tomo_col]
            aff_val = rec[aff_col]

            # Radius: mean distance of cycle blocks from centroid.
            radius = float(np.linalg.norm(centres - centroid, axis=1).mean())

            # planarity: RMS out-of-plane deviation normalised by radius.
            deviations = (centres - centroid) @ normal_unit
            rms_dev = float(np.sqrt(np.mean(deviations**2)))
            planarity = rms_dev / radius if radius > 1e-15 else 0.0

            # centroid_depth: nearest block in the assembly / radius.
            _block_pts = asm_block_centres.get((tomo_val, aff_val))
            if _block_pts is not None and len(_block_pts) > 0:
                _d_min = float(np.linalg.norm(_block_pts - centroid, axis=1).min())
                centroid_depth = _d_min / radius if radius > 1e-15 else 0.0
            else:
                centroid_depth = 0.0

            n_boundary_blocks = sum(1 for bid in block_ids_walk if (tomo_val, bid) in boundary_block_set)
            n_inferred_blocks = (
                sum(1 for bid in block_ids_walk if _origin_lookup.get(bid, 1.0) == 2.0)
                if _origin_lookup is not None
                else 0
            )

            row_dict: dict = {
                tomo_col: tomo_val,
                aff_col: aff_val,
                "face_id": rec["face_id"],
                "assembly_id": rec["assembly_id"],
                "size": size,
                "n_unique_blocks": len(set(block_ids_walk)),
                "block_ids": block_ids_walk,
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
                "centroid_z": float(centroid[2]),
                "normal_x": float(normal_unit[0]),
                "normal_y": float(normal_unit[1]),
                "normal_z": float(normal_unit[2]),
                "n_vertices": size,
                "n_boundary_blocks": n_boundary_blocks,
                "n_inferred_blocks": n_inferred_blocks,
                "planarity": planarity,
                "centroid_depth": centroid_depth,
            }
            if scope in ("all", "inferred") and _origin_lookup is not None:
                row_dict["n_added"] = sum(
                    1 for bid in block_ids_walk if _origin_lookup.get(bid, 1.0) == 2.0
                )
            rows.append(row_dict)
        return pd.DataFrame(rows)

    def get_face_filling_summary(self, other: "PleomorphicSurface") -> pd.DataFrame:
        """Return per-size face counts before and after block filling.

        Parameters
        ----------
        other : PleomorphicSurface
            A surface produced by :meth:`with_inferred_blocks` from this surface.

        Returns
        -------
        pandas.DataFrame
            Columns: ``size``, ``n_before``, ``n_after``, ``n_with_inferred``.
            One row per face size that appears in either surface.
        """
        self._require_connect()
        other._require_connect()

        before = self.get_face_stats().groupby("size").size().rename("n_before")
        after_stats = other.get_face_stats()
        after = after_stats.groupby("size").size().rename("n_after")

        if "n_inferred_blocks" in after_stats.columns:
            with_inferred = (
                after_stats[after_stats["n_inferred_blocks"] > 0]
                .groupby("size")
                .size()
                .rename("n_with_inferred")
            )
        else:
            with_inferred = pd.Series(dtype=int, name="n_with_inferred")

        return (
            pd.concat([before, after, with_inferred], axis=1)
            .fillna(0)
            .astype(int)
            .reset_index()
        )

    @gui_exposed(
        label="Block stats", group="Lattice statistics", order=30, returns="dataframe", category="pleomorphic-op"
    )
    def get_block_stats(self, scope: Literal["original", "inferred", "all"] = "original") -> pd.DataFrame:
        """Return one row per block.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.
            ``'all'`` uses the combined lattice + inferred block graph and adds
            an ``origin`` column (1 = lattice, 2 = inferred).
            ``'inferred'`` returns only blocks that participate in inferred faces,
            also with ``origin``.

        Returns
        -------
        pandas.DataFrame
            Fixed columns: ``tomo_id``, ``block_id``, ``block_type``,
            ``n_sites``, ``degree``, ``complete``, ``assembly_id``,
            ``face_signature``, ``angle_sum``, ``angle_deficit``,
            ``face_sizes``, ``n_faces``.
            Dynamic columns (one per distinct face size present in the
            lattice): ``n_faces_<m>`` — how many faces of size *m* border
            this block.  The set of *m* values is observed from the data;
            no size is hardcoded.
            When *scope* is ``'all'`` or ``'inferred'``: additionally
            ``origin``.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called for *scope*.
        """
        from collections import Counter as _Counter

        st, _faces, _blocks = self._state_for_scope(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        # Face sizes keyed by (tomo, affiliation, face_id) — face_id restarts per (tomo, aff) group
        face_size: dict[tuple, int] = {}
        for rec in _faces:
            key = (rec[tomo_col], rec[aff_col], rec["face_id"])
            face_size[key] = len(rec["half_edges"])

        # All distinct face sizes present in this lattice (for dynamic n_faces_<m> columns)
        all_face_sizes: list[int] = sorted({sz for sz in face_size.values() if sz > 0})

        # Block type lookup
        block_type_map: dict[float, float] = {}
        if isinstance(self.block_definition, dict):
            for _, row in _blocks.df.iterrows():
                block_type_map[float(row["subtomo_id"])] = float(row[self.block_type_column])

        # Origin lookup for scope="all"
        _origin_col = self._origin_column
        _origin_lookup_bs: dict[float, float] | None = None
        if scope in ("all", "inferred") and _blocks is not None and _origin_col in _blocks.df.columns:
            _origin_lookup_bs = dict(
                zip(_blocks.df["subtomo_id"].astype(float), _blocks.df[_origin_col].astype(float))
            )

        rows = []
        for (tomo_val, block_val), grp in st.groupby([tomo_col, "block_id"]):
            aff_val = float(grp[aff_col].iloc[0])
            n_sites_val = int(grp["n_sites"].iloc[0])
            degree = int((grp["partner"] >= 0).sum())

            # face_signature: sorted face sizes per site (0 for boundary), preserving original semantics
            face_sizes_for_sig = []
            # unique face membership (deduplicated by face_id) for vertex-class columns
            seen_fids: set[int] = set()
            unique_face_sizes: list[int] = []
            for _, hrow in grp.iterrows():
                fid = int(hrow["face_id"])
                if fid > 0:
                    key = (tomo_val, aff_val, fid)
                    sz = face_size.get(key, 0)
                    face_sizes_for_sig.append(sz)
                    if fid not in seen_fids and sz > 0:
                        seen_fids.add(fid)
                        unique_face_sizes.append(sz)
                else:
                    face_sizes_for_sig.append(0)
            face_sig = "-".join(str(s) for s in sorted(face_sizes_for_sig))
            face_sizes_tup = tuple(sorted(unique_face_sizes))
            n_faces_val = len(face_sizes_tup)
            face_cnt = _Counter(face_sizes_tup)

            # angle_sum / angle_deficit (only if complete)
            if degree == n_sites_val:
                bond_vecs = []
                for site_idx in sorted(grp["site"].unique()):
                    site_rows = grp[grp["site"] == site_idx]
                    h_row = site_rows.iloc[0]
                    p = int(h_row["partner"])
                    c_p = np.array([st.at[p, "cx"], st.at[p, "cy"], st.at[p, "cz"]])
                    c_h = np.array([h_row["cx"], h_row["cy"], h_row["cz"]])
                    bond_vecs.append(c_p - c_h)
                angle_sum_val = 0.0
                n_b = len(bond_vecs)
                for idx_b in range(n_b):
                    v1 = bond_vecs[idx_b]
                    v2 = bond_vecs[(idx_b + 1) % n_b]
                    angle_sum_val += float(geom.angle_between_vectors(v1.reshape(1, 3), v2.reshape(1, 3))[0])
                angle_deficit_val = 360.0 - angle_sum_val
            else:
                angle_sum_val = float("nan")
                angle_deficit_val = float("nan")

            btype = (
                block_type_map.get(float(block_val), float("nan"))
                if isinstance(self.block_definition, dict)
                else float("nan")
            )
            asm_id = float(grp["assembly_id"].iloc[0])

            bs_row: dict = {
                tomo_col: tomo_val,
                aff_col: aff_val,
                "block_id": float(block_val),
                "block_type": btype,
                "n_sites": n_sites_val,
                "degree": degree,
                "complete": degree == n_sites_val,
                "assembly_id": asm_id,
                "face_signature": face_sig,
                "angle_sum": angle_sum_val,
                "angle_deficit": angle_deficit_val,
                "face_sizes": face_sizes_tup,
                "n_faces": n_faces_val,
            }
            for m in all_face_sizes:
                bs_row[f"n_faces_{m}"] = face_cnt.get(m, 0)
            if _origin_lookup_bs is not None:
                bs_row["origin"] = _origin_lookup_bs.get(float(block_val), 1.0)
            rows.append(bs_row)
        return pd.DataFrame(rows)

    @gui_exposed(
        label="Assembly stats", group="Lattice statistics", order=40, returns="dataframe", category="pleomorphic-op"
    )
    def get_assembly_stats(self, scope: Literal["original", "inferred", "all"] = "original") -> pd.DataFrame:
        """Return one row per (tomo_id, assembly_id).

        All measurements are derived from the contact graph that
        :meth:`connect` builds; no assumed ideal lattice is required.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'`` uses
            the combined lattice + inferred block graph.  ``'inferred'`` counts
            only assemblies that have at least one inferred face.

        Returns
        -------
        pandas.DataFrame
            Fixed columns: ``tomo_id``, ``assembly_id``, ``n_blocks``,
            ``n_contacts``, ``n_faces``, ``n_boundary_half_edges``,
            ``n_boundary_blocks`` (blocks with at least one unmatched site),
            ``closed``, ``euler_characteristic``, ``angle_deficit_sum``,
            ``majority_face_size`` (mode of observed face sizes, NaN on tie or
            no faces), ``majority_face_defect`` (sum (mode − size) / mode over
            all faces; **positive** when the assembly contains more faces smaller
            than the background — e.g. pentagons in a hexagonal lattice close a
            sphere and give +2; **negative** when it contains faces larger than
            the background).
            Dynamic columns: one ``n_faces_<m>`` per distinct observed face size
            (``size``) and one ``n_degree_<k>`` per distinct vertex degree found
            in the data.

        Notes
        -----
        ``euler_characteristic`` (V − E + F) is the closure measurement.  A
        connected closed shell gives 2; an open sheet gives a different value.
        ``majority_face_defect`` measures face-topology deviation from the
        observed background face size.  For a sphere built from hexagons with
        twelve pentagons the background is six-sided, the deviation is +2.
        A geodesic dome of all triangles has no face-size defect (0.0); its
        closure is captured by ``euler_characteristic`` = 2 and twelve
        degree-5 vertices in the ``n_degree_5`` column.
        When the face-size distribution has no unique mode (equal split between
        two or more sizes), both ``majority_face_size`` and
        ``majority_face_defect`` are NaN.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called for *scope*.
        """
        st, _, _ = self._state_for_scope(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        block_stats = self.get_block_stats(scope=scope)
        face_stats = self.get_face_stats(scope=scope)

        # Column headers spanning all assemblies
        all_face_sizes = sorted(face_stats["size"].unique().tolist()) if len(face_stats) > 0 else []
        all_degrees = sorted(block_stats["degree"].unique().tolist()) if len(block_stats) > 0 else []

        # Boundary blocks per (tomo, block_id): has at least one unmatched site
        boundary_block_set: set[tuple] = set(
            (row[tomo_col], row["block_id"])
            for _, row in st[st["partner"] < 0][[tomo_col, "block_id"]].drop_duplicates().iterrows()
        )

        rows = []
        for (tomo_val, aff_val), block_grp in block_stats.groupby([tomo_col, aff_col]):
            n_blocks = len(block_grp)
            block_id_set = set(block_grp["block_id"].tolist())

            # Contacts (matched half-edges / 2)
            st_asm = st[(st[tomo_col] == tomo_val) & (st["block_id"].isin(block_id_set))]
            n_matched = int((st_asm["partner"] >= 0).sum())
            n_contacts = n_matched // 2

            # Faces for this affiliation group
            if len(face_stats) > 0:
                face_asm = face_stats[(face_stats[tomo_col] == tomo_val) & (face_stats[aff_col] == aff_val)]
            else:
                face_asm = face_stats
            n_faces = len(face_asm)

            # Boundary half-edges and boundary blocks
            n_boundary = int((st_asm["face_id"] <= 0).sum())
            n_boundary_blocks = int(sum(1 for bid in block_id_set if (tomo_val, bid) in boundary_block_set))

            closed = n_boundary == 0
            euler = n_blocks - n_contacts + n_faces

            # angle_deficit_sum
            deficit_vals = block_grp["angle_deficit"].dropna()
            angle_deficit_sum = float(deficit_vals.sum())

            # Majority face statistics — derived from inferred sizes so that
            # damaged larger faces are counted as their intact size, not as
            # the smaller observed size.
            if len(face_asm) > 0:
                face_counts = face_asm["size"].value_counts()
                max_cnt = face_counts.max()
                majority_candidates = face_counts[face_counts == max_cnt]
                if len(majority_candidates) == 1:
                    majority_fs: int | float = int(majority_candidates.index[0])
                    majority_fd: float = float(((majority_fs - face_asm["size"]) / majority_fs).sum())
                else:
                    majority_fs = float("nan")
                    majority_fd = float("nan")
            else:
                majority_fs = float("nan")
                majority_fd = float("nan")

            # Vertex-degree distribution
            degree_counts: dict[str, int] = {f"n_degree_{k}": 0 for k in all_degrees}
            for k, cnt in block_grp["degree"].value_counts().items():
                key = f"n_degree_{k}"
                if key in degree_counts:
                    degree_counts[key] = int(cnt)

            # Face-size distribution (n_faces_<m>)
            face_size_counts: dict[str, int] = {f"n_faces_{m}": 0 for m in all_face_sizes}
            if len(face_asm) > 0:
                for m, cnt in face_asm["size"].value_counts().items():
                    key = f"n_faces_{m}"
                    if key in face_size_counts:
                        face_size_counts[key] = int(cnt)

            row: dict = {
                tomo_col: tomo_val,
                aff_col: float(aff_val),
                "n_blocks": n_blocks,
                "n_contacts": n_contacts,
                "n_faces": n_faces,
                "n_boundary_half_edges": n_boundary,
                "n_boundary_blocks": n_boundary_blocks,
                "closed": closed,
                "euler_characteristic": euler,
                "angle_deficit_sum": angle_deficit_sum,
                "majority_face_size": majority_fs,
                "majority_face_defect": majority_fd,
            }
            row.update(face_size_counts)
            row.update(degree_counts)
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        for m in all_face_sizes:
            col = f"n_faces_{m}"
            if col not in result.columns:
                result[col] = 0
        for k in all_degrees:
            col = f"n_degree_{k}"
            if col not in result.columns:
                result[col] = 0
        return result

    @gui_exposed(
        label="Face inference", group="Lattice statistics", order=45, returns="dataframe", category="pleomorphic-op"
    )
    def get_face_inference(self) -> pd.DataFrame:
        """Per-face inference table: merged-face detection and missing block count.

        Runs :meth:`_face_hole_analysis` on every closed face and returns the
        diagnostic result.  All decisions are derived from interior-angle
        geometry; no assumed ideal lattice is required.  The face normal used
        for interior-angle projection is the normalised mean of the block
        z-vectors (each block's orientation applied to [0, 0, 1]), matching
        :meth:`get_face_stats`.

        :meth:`get_face_stats` reports the **measured** face topology; this
        method reports the **inferred** interpretation.
        :meth:`get_missing_block_motl` uses the same detector to produce
        positions and orientations for the inferred missing blocks.

        Returns
        -------
        pandas.DataFrame
            Columns per face:

            ``tomo_id``, ``object_id`` — identity, from the face record.
            ``face_id`` — face identifier.
            ``size`` — observed boundary length (number of half-edges).
            ``is_merged`` — True when the median interior angle deviates from
                the ideal n-gon value by more than 20°.
            ``n_reflex`` — count of reflex vertices (turn angle < −30° in the
                minority winding direction); the raw signal driving ``n_missing``.
            ``n_missing`` — inferred missing trivalent blocks
                (``n_reflex // 3``); 0 when ``is_merged`` is False or the reflex
                count is not a multiple of 3.
            ``n_faces_recovered`` — faces that would be restored: 0 when
                ``n_missing == 0``; ``1 + 2 * n_missing`` otherwise (Euler:
                each re-inserted unit adds ΔV=+1, ΔE=+3 → ΔF=+2, Δχ=0).
            ``recovered_face_size`` — majority observed face size for this
                assembly (the expected size of each recovered face); NaN when
                ``n_missing == 0`` or the assembly size distribution has no
                unique mode.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called.
        """
        self._require_connect()
        st = self._site_table
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        # Per-assembly majority face size, computed directly from _faces.
        asm_sizes: dict[tuple, Counter] = {}
        for _rec in self._faces:
            _key = (_rec[tomo_col], _rec[aff_col])
            asm_sizes.setdefault(_key, Counter())[len(_rec["half_edges"])] += 1
        asm_majority: dict[tuple, int | float] = {}
        for _key, _ctr in asm_sizes.items():
            _max = max(_ctr.values())
            _cands = [s for s, c in _ctr.items() if c == _max]
            asm_majority[_key] = int(_cands[0]) if len(_cands) == 1 else float("nan")

        # Rotation lookup for mean-z-vector normal (orientation carries correct sign).
        if self.blocks is not None and len(self.blocks.df) > 0:
            _ang_fi = self.blocks.get_angles().astype(float)
            _all_R_fi = srot.from_euler("zxz", _ang_fi, degrees=True)
            _b2r_fi: dict[float, int] = {
                float(self.blocks.df.iloc[i]["subtomo_id"]): i
                for i in range(len(self.blocks.df))
            }
        else:
            _all_R_fi = None
            _b2r_fi = None
        _zu_fi = np.array([0.0, 0.0, 1.0])

        rows: list[dict] = []
        for rec in self._faces:
            tomo_val = rec[tomo_col]
            aff_val = rec[aff_col]
            hedges = rec["half_edges"]
            size = len(hedges)

            cx_vals = np.array([st.at[h, "cx"] for h in hedges])
            cy_vals = np.array([st.at[h, "cy"] for h in hedges])
            cz_vals = np.array([st.at[h, "cz"] for h in hedges])
            centres = np.column_stack([cx_vals, cy_vals, cz_vals])

            # Face normal: mean z-vector of the face's block orientations.
            block_ids_fi = [float(st.at[h, "block_id"]) for h in hedges]
            if _all_R_fi is not None:
                _zvecs_fi = [
                    _all_R_fi[_b2r_fi[bid]].apply(_zu_fi)
                    for bid in block_ids_fi
                    if bid in _b2r_fi
                ]
            else:
                _zvecs_fi = []
            if _zvecs_fi:
                normal_raw = np.array(_zvecs_fi).mean(axis=0)
            else:
                normal_raw = np.array([0.0, 0.0, 1.0])
            n_norm = float(np.linalg.norm(normal_raw))
            normal_unit = normal_raw / n_norm if n_norm > 1e-15 else np.array([0.0, 0.0, 1.0])

            h = self._face_hole_analysis(centres, normal_unit)
            n_missing = int(h["n_missing"])
            n_reflex = int(h["n_reflex"])
            is_merged = bool(h["is_merged"])

            if n_missing > 0:
                n_faces_recovered = 1 + 2 * n_missing
                recovered_face_size = float(asm_majority.get((tomo_val, aff_val), float("nan")))
            else:
                n_faces_recovered = 0
                recovered_face_size = float("nan")

            rows.append(
                {
                    tomo_col: tomo_val,
                    aff_col: aff_val,
                    "face_id": rec["face_id"],
                    "size": size,
                    "is_merged": is_merged,
                    "n_reflex": n_reflex,
                    "n_missing": n_missing,
                    "n_faces_recovered": n_faces_recovered,
                    "recovered_face_size": recovered_face_size,
                }
            )

        return pd.DataFrame(rows)

    @gui_exposed(
        label="Missing block motl",
        group="Lattice statistics",
        order=50,
        returns="motl",
        category="pleomorphic-op",
    )
    def get_missing_block_motl(self, *, scope: Literal["original", "inferred", "all"] = "original") -> "cryomotl.Motl":
        """Infer positions of missing blocks from oversized face cycles.

        A face cycle of length L in an assembly whose majority face size is m
        encloses ``k = L/m − 1`` missing blocks when L is an exact multiple of
        m and L > m.  Faces whose length is not a multiple of m are skipped
        (rim cycles, irregular holes).

        **Position**: centroid of the three cycle vertices that bond to the
        missing block.  For the j-th missing block (j = 0 … k−1) the bonding
        triplet is taken at equally-spaced positions around the cycle:
        ``round(j·L/k)``, ``round(j·L/k + L/3)``, ``round(j·L/k + 2·L/3)``
        (all modulo L).

        **Orientation**: mean SO(3) rotation of the three bonding blocks,
        computed with ``scipy.spatial.transform.Rotation.mean()``.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'`` uses
            the combined lattice + inferred block graph.  ``'inferred'`` uses
            only inferred faces.

        Returns
        -------
        cryomotl.Motl
            One row per inferred missing block.  Column assignments follow
            :class:`BlockLayer` constructor parameters (defaults in
            parentheses):

            * ``tomo_id_column`` (``tomo_id``) — from the owning face
            * ``affiliation_column`` (``object_id``) — assembly id
            * ``face_id_column`` (``geom2``) — face_id of the owning face
            * ``x``, ``y``, ``z`` — inferred 3-D position
            * ``phi``, ``theta``, ``psi`` — ZXZ Euler of averaged neighbours

            All other Motl columns are 0.  An empty Motl is returned when
            no eligible faces are found.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called for the requested scope.
        """
        self._require_connect(scope)
        st, faces, _blocks = self._state_for_scope(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        angles_all = _blocks.get_angles().astype(float)
        all_R = srot.from_euler("zxz", angles_all, degrees=True)
        block_to_row: dict[float, int] = {
            float(_blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(_blocks.df))
        }

        # Per-assembly majority face size (mode of observed cycle lengths).
        _asm_sizes: dict[tuple, Counter] = {}
        for _rec in faces:
            _key = (_rec[tomo_col], _rec[aff_col])
            _asm_sizes.setdefault(_key, Counter())[len(_rec["half_edges"])] += 1
        asm_majority: dict[tuple, int] = {}
        for _key, _ctr in _asm_sizes.items():
            _max_c = max(_ctr.values())
            _cands = [s for s, c in _ctr.items() if c == _max_c]
            asm_majority[_key] = int(_cands[0]) if len(_cands) == 1 else 0

        rows: list[dict] = []
        subtomo_counter = 1

        for rec in faces:
            tomo_val = rec[tomo_col]
            aff_val = rec[aff_col]
            hedges = rec["half_edges"]
            L = len(hedges)

            maj_size = asm_majority.get((tomo_val, aff_val), 0)
            if maj_size == 0 or L % maj_size != 0 or L <= maj_size:
                continue

            k = L // maj_size - 1
            block_ids_walk = [float(st.at[h, "block_id"]) for h in hedges]
            cx_vals = np.array([st.at[h, "cx"] for h in hedges])
            cy_vals = np.array([st.at[h, "cy"] for h in hedges])
            cz_vals = np.array([st.at[h, "cz"] for h in hedges])
            centres = np.column_stack([cx_vals, cy_vals, cz_vals])

            for j in range(k):
                i0 = int(round(j * L / k)) % L
                i1 = int(round(j * L / k + L / 3.0)) % L
                i2 = int(round(j * L / k + 2.0 * L / 3.0)) % L
                pos = centres[[i0, i1, i2]].mean(axis=0)
                neighbour_bids = [block_ids_walk[i0], block_ids_walk[i1], block_ids_walk[i2]]
                valid_R = [all_R[block_to_row[bid]] for bid in neighbour_bids if bid in block_to_row]
                if not valid_R:
                    continue
                euler = srot.concatenate(valid_R).mean().as_euler("zxz", degrees=True)

                row = {col: 0.0 for col in cryomotl.Motl.motl_columns}
                row.update(
                    {
                        tomo_col: float(tomo_val),
                        aff_col: float(aff_val),
                        "subtomo_id": float(subtomo_counter),
                        "x": float(pos[0]),
                        "y": float(pos[1]),
                        "z": float(pos[2]),
                        "phi": float(euler[0]),
                        "theta": float(euler[1]),
                        "psi": float(euler[2]),
                        self.face_id_column: float(rec["face_id"]),
                    }
                )
                rows.append(row)
                subtomo_counter += 1

        if not rows:
            motl = cryomotl.Motl()
            motl.df = cryomotl.Motl.create_empty_motl_df()
            return motl
        motl = cryomotl.Motl()
        motl.df = pd.DataFrame(rows)[cryomotl.Motl.motl_columns]
        return motl

    def check_object_grouping(self, scope: Literal["original", "inferred", "all"] = "original") -> pd.DataFrame:
        """Diagnose whether each affiliation holds one physical assembly.

        An affiliation whose contact graph is disconnected contains multiple
        independent pieces (``n_components > 1``).  An affiliation whose blocks
        have contacts reaching into another affiliation is physically joined to
        it (``n_cross_affiliation_contacts > 0``).  Use :meth:`regroup_objects`
        to act on these findings.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'`` uses
            the combined lattice + inferred block graph.  ``'inferred'`` checks
            grouping for inferred faces only.

        Returns
        -------
        pandas.DataFrame
            One row per ``(tomo_id, affiliation)``.  Columns:

            * ``tomo_id`` — tomogram identifier
            * ``object_id`` (or the configured affiliation column) — affiliation value
            * ``n_blocks`` — total blocks in this affiliation
            * ``n_contacts`` — undirected contacts (matched half-edges / 2) within
              the affiliation
            * ``n_components`` — connected components of the within-affiliation
              contact graph (> 1 means the affiliation contains multiple pieces)
            * ``euler_characteristic`` — V − E + F for this affiliation
            * ``component_sizes`` — blocks per component, sorted descending
            * ``n_cross_affiliation_contacts`` — half-edges leaving this affiliation
              to a block in a different affiliation of the same tomogram; each
              undirected cross contact contributes 1 to both affiliations involved

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called for *scope*.
        """
        import networkx as _nx

        st, _, _ = self._state_for_scope(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        face_stats = self.get_face_stats(scope=scope)
        n_faces_map: dict[tuple, int] = {}
        if len(face_stats) > 0:
            for (tv, av), grp in face_stats.groupby([tomo_col, aff_col]):
                n_faces_map[(tv, av)] = len(grp)

        rows: list[dict] = []
        for (tomo_val, aff_val), aff_grp in st.groupby([tomo_col, aff_col]):
            block_ids = aff_grp["block_id"].unique()
            n_blocks = len(block_ids)

            matched = aff_grp[aff_grp["partner"] >= 0]
            n_matched_within = 0
            cross_contacts = 0

            G = _nx.Graph()
            G.add_nodes_from(block_ids.tolist())

            for _, row in matched.iterrows():
                p = int(row["partner"])
                if p not in st.index:
                    continue
                p_tomo = st.at[p, tomo_col]
                p_aff = st.at[p, aff_col]
                if p_tomo != tomo_val:
                    continue
                if p_aff == aff_val:
                    G.add_edge(row["block_id"], st.at[p, "block_id"])
                    n_matched_within += 1
                else:
                    cross_contacts += 1

            n_contacts = n_matched_within // 2
            n_components = _nx.number_connected_components(G)
            component_sizes = sorted([len(c) for c in _nx.connected_components(G)], reverse=True)
            n_faces = n_faces_map.get((tomo_val, aff_val), 0)
            euler = n_blocks - n_contacts + n_faces

            rows.append(
                {
                    tomo_col: tomo_val,
                    aff_col: float(aff_val),
                    "n_blocks": n_blocks,
                    "n_contacts": n_contacts,
                    "n_components": n_components,
                    "euler_characteristic": euler,
                    "component_sizes": component_sizes,
                    "n_cross_affiliation_contacts": cross_contacts,
                }
            )

        return pd.DataFrame(rows)

    def regroup_objects(
        self,
        split: bool = True,
        merge: bool = True,
        original_column: str = "original_object_id",
    ) -> "cryomotl.Motl":
        """Return a new motl with affiliations reassigned so each holds one connected piece.

        Acts on :attr:`blocks` without modifying it.  The original affiliation
        value is preserved in *original_column* so the change is traceable.

        Parameters
        ----------
        split : bool, default=True
            When True, each connected component of the within-affiliation
            contact graph is promoted to its own affiliation.  Multi-piece
            affiliations are broken apart.
        merge : bool, default=True
            When True, affiliations joined by cross-affiliation contacts are
            folded into one.  The block-level contact graph (including
            cross-affiliation edges) determines which affiliations are
            physically connected.
        original_column : str, default="original_object_id"
            Name of the column added to the returned motl to record the
            original affiliation value.

        Returns
        -------
        cryomotl.Motl
            Deep copy of :attr:`blocks` with ``affiliation_column`` rewritten.
            Affiliation values are assigned sequentially from 1 within each
            tomogram, ordered by the smallest block ``subtomo_id`` in the group.
            The returned motl is independent of ``self``; :attr:`blocks` is
            unchanged.

        Notes
        -----
        ``split=False, merge=False`` returns an unchanged copy (plus the
        *original_column*).

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called.
        """
        import networkx as _nx

        self._require_connect()
        st = self._site_table
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        new_motl = copy.deepcopy(self.blocks)
        df = new_motl.df
        df[original_column] = df[aff_col].copy()

        for tomo_val, tomo_block_grp in df.groupby(tomo_col):
            st_tomo = st[st[tomo_col] == tomo_val]
            all_block_ids = tomo_block_grp["subtomo_id"].unique().tolist()

            # Map subtomo_id → affiliation for blocks in this tomo.
            bid_to_aff: dict[float, float] = dict(zip(tomo_block_grp["subtomo_id"], tomo_block_grp[aff_col]))

            # Build block-level graph.  Edges depend on the flags.
            G = _nx.Graph()
            G.add_nodes_from(all_block_ids)

            for _, row in st_tomo[st_tomo["partner"] >= 0].iterrows():
                p = int(row["partner"])
                if p not in st.index:
                    continue
                if st.at[p, tomo_col] != tomo_val:
                    continue
                src_bid = row["block_id"]
                dst_bid = st.at[p, "block_id"]
                src_aff = row[aff_col]
                dst_aff = st.at[p, aff_col]
                is_cross = src_aff != dst_aff
                if is_cross and not merge:
                    continue
                G.add_edge(src_bid, dst_bid)

            if not split:
                # Glue disconnected pieces within each original affiliation
                # so they stay together (unless they were already merged by
                # a cross-affiliation edge when merge=True).
                aff_groups: dict[float, list[float]] = {}
                for bid, aff in bid_to_aff.items():
                    aff_groups.setdefault(aff, []).append(bid)
                for _, members in aff_groups.items():
                    for i in range(1, len(members)):
                        G.add_edge(members[0], members[i])

            # Find connected components and assign new sequential labels.
            components = sorted(
                _nx.connected_components(G),
                key=lambda c: min(c),
            )
            bid_to_new_aff: dict[float, float] = {}
            for new_idx, comp in enumerate(components, start=1):
                for bid in comp:
                    bid_to_new_aff[bid] = float(new_idx)

            # Write new affiliation back into the motl copy.
            tomo_mask = df[tomo_col] == tomo_val
            df.loc[tomo_mask, aff_col] = df.loc[tomo_mask, "subtomo_id"].map(bid_to_new_aff)

        return new_motl

    def store_block_stats(self, columns: dict[str, MotlColumn]) -> None:
        """Write selected block-stats columns into the blocks motl.

        Parameters
        ----------
        columns : dict[str, MotlColumn]
            Mapping of :meth:`get_block_stats` column name → motl column to
            write into.  Only numeric columns are accepted.

        Raises
        ------
        ValueError
            If a requested column is non-numeric (e.g. ``face_signature``)
            or not a recognised :meth:`get_block_stats` column.
        """
        self._require_connect()
        _NON_NUMERIC = {"face_signature", "block_type"}
        block_stats = self.get_block_stats()
        for src, dst in columns.items():
            if src in _NON_NUMERIC:
                raise ValueError(f"Column '{src}' is non-numeric and cannot be written to the blocks motl.")
            if src not in block_stats.columns:
                raise ValueError(f"Unknown get_block_stats() column: '{src}'.")
            tomo_col = self.tomo_id_column
            for _, row in block_stats.iterrows():
                mask = self.blocks.df["subtomo_id"] == row["block_id"]
                self.blocks.df.loc[mask, dst] = float(row[src])

    _BLOCK_STAT_CHOICES = ("n_sites", "degree", "complete", "assembly_id", "angle_sum", "angle_deficit")

    @gui_exposed(
        label="Store block stat", group="Lattice statistics", order=50, returns="motl", category="pleomorphic-op"
    )
    def store_block_stat(
        self,
        stat: Literal["n_sites", "degree", "complete", "assembly_id", "angle_sum", "angle_deficit"],
        column: MotlColumn,
    ) -> "cryomotl.Motl":
        """Write one block-stats column into the blocks motl and return the updated blocks.

        Parameters
        ----------
        stat : {"n_sites", "degree", "complete", "assembly_id", "angle_sum", "angle_deficit"}
            Name of the :meth:`get_block_stats` column to store.
        column : MotlColumn
            Motl column to write the values into.

        Returns
        -------
        cryomotl.Motl
            Deep copy of :attr:`blocks` after writing.

        Raises
        ------
        ValueError
            If *stat* is not one of the accepted column names.
        """
        if stat not in self._BLOCK_STAT_CHOICES:
            raise ValueError(f"Unknown stat {stat!r}. Must be one of {self._BLOCK_STAT_CHOICES}.")
        self.store_block_stats({stat: column})
        return copy.deepcopy(self.blocks)

    @gui_exposed(label="Blocks as motl", group="Lattice setup", order=50, returns="motl", category="pleomorphic-op")
    def get_blocks_as_motl(self, scope: Literal["original", "inferred", "all"] = "original") -> "cryomotl.Motl":
        """Return the block layer as an independent :class:`~cryocat.core.cryomotl.Motl`.

        Parameters
        ----------
        scope : {"original", "inferred", "all"}, default="original"
            ``"original"`` returns :attr:`blocks` only.  ``"inferred"``
            returns the inferred blocks from :attr:`_added`.  ``"all"``
            returns both concatenated: original IDs are preserved;
            inferred IDs continue from ``max(subtomo_id) + 1``.

        Returns
        -------
        cryomotl.Motl
            Independent copy of the requested block set.

        Raises
        ------
        ValueError
            If no block layer is present, or ``scope="inferred"``/``"all"``
            is requested but no inferred blocks exist.
        """
        if self.blocks is None:
            raise ValueError("get_blocks_as_motl requires a block layer (blocks=...).")

        if scope == "original":
            return copy.deepcopy(self.blocks)

        if self._added is None or len(self._added.df) == 0:
            raise ValueError("No inferred blocks; call infer_missing_blocks() first.")

        if scope == "inferred":
            return copy.deepcopy(self._added)

        # scope == "all": IDs are already disjoint (inferred starts from max(original)+1)
        combined_df = pd.concat([self.blocks.df, self._added.df], ignore_index=True)
        return cryomotl.Motl(combined_df[cryomotl.Motl.motl_columns])

    @gui_exposed(label="Faces as motl", group="Lattice motls", order=10, returns="motl", category="pleomorphic-op")
    def get_faces_as_motl(
        self,
        size: int | None = None,
        *,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> "cryomotl.Motl":
        """Return a :class:`~cryocat.core.cryomotl.Motl` with one row per closed face.

        Column assignments are controlled by :class:`BlockLayer` constructor
        parameters (defaults in parentheses):

        * ``tomo_id_column`` (``tomo_id``) — tomogram identifier
        * ``affiliation_column`` (``object_id``) — assembly affiliation
        * ``face_id_column`` (``geom4``) — 1-based face id within the assembly
        * ``face_size_column`` (``geom5``) — face size (number of vertices)
        * ``x``, ``y``, ``z`` — face centroid
        * ``phi``, ``theta``, ``psi`` — Euler angles from face normal

        Parameters
        ----------
        size : int or None, default=None
            When set, return only faces with exactly *size* vertices (e.g.
            ``size=5`` for pentagons).  ``None`` returns all faces.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'`` uses
            the combined lattice + inferred block graph.  ``'inferred'`` returns
            only faces that exist in ``'all'`` but not in ``'original'``.  Use
            the ``n_added`` column returned by :meth:`get_face_stats` to count
            inferred blocks per face.
        """
        _check_column_collisions("get_faces_as_motl", {
            "face_id_column": self.face_id_column,
            "face_size_column": self.face_size_column,
        })
        _, _, _blocks = self._state_for_scope(scope)
        face_stats = self.get_face_stats(scope=scope)
        if size is not None:
            face_stats = face_stats[face_stats["size"] == size].reset_index(drop=True)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        n = len(face_stats)

        normals = face_stats[["normal_x", "normal_y", "normal_z"]].values.astype(float)
        euler_angles = geom.normals_to_euler_angles(normals)

        data: dict[str, Any] = {c: np.zeros(n) for c in cryomotl.Motl.motl_columns}
        data["x"] = face_stats["centroid_x"].values
        data["y"] = face_stats["centroid_y"].values
        data["z"] = face_stats["centroid_z"].values
        data["phi"] = euler_angles[:, 0]
        data["theta"] = euler_angles[:, 1]
        data["psi"] = euler_angles[:, 2]
        data[tomo_col] = face_stats[tomo_col].values
        data[aff_col] = face_stats[aff_col].values
        data[self.face_size_column] = face_stats["size"].values.astype(float)
        data[self.face_id_column] = face_stats["face_id"].values

        motl = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])
        motl.renumber_particles()
        motl.update_coordinates()
        return motl

    def get_bond_elevations(
        self,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> pd.DataFrame:
        """Return one directed row per bonded site pair with the elevation angle.

        For each bond A → B, the elevation is the signed angle between the
        bond vector (from A's block centre to B's block centre) and the plane
        perpendicular to A's z-axis (A's normal direction in world coordinates).
        Positive means B is on the same side as A's normal; negative means the
        opposite side.  The angle is not folded to the positive half-space.

        Because elevation is a directed quantity (A looking at B differs from
        B looking at A when their orientations differ), each undirected bond
        produces **two** rows — one for each ordered pair.  Rows where
        ``unit_id == partner_id`` are never produced.

        :meth:`connect` must be called before this method.

        Parameters
        ----------
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``

        Returns
        -------
        pandas.DataFrame
            Columns:

            * ``tomo_id`` — tomogram identifier
            * ``assembly_id``
            * ``unit_id`` — ``subtomo_id`` of the source block
            * ``partner_id`` — ``subtomo_id`` of the target block
            * ``elevation_deg`` — signed elevation angle in degrees
            * ``face_sizes`` — tuple of unique face sizes bordering the unit
            * ``n_faces`` — number of unique faces bordering the unit
            * ``n_faces_<m>`` (one per observed face size) — face count by size

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called.
        """
        self._require_connect(scope)
        st, _faces, _blocks = self._state_for_scope(scope)

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        # Build block → rotation mapping
        block_sids = _blocks.df["subtomo_id"].astype(float).values
        block_angles = _blocks.get_angles().astype(float)
        _sid_to_rot: dict[float, Any] = {
            float(s): srot.from_euler("zxz", block_angles[i], degrees=True)
            for i, s in enumerate(block_sids)
        }

        # Build block → face-class stats from get_block_stats
        block_stats = self.get_block_stats(scope=scope)
        _bs_idx: dict[float, int] = {
            float(r["block_id"]): idx for idx, r in block_stats.iterrows()
        }
        # Collect all dynamic face-size columns present
        _dyn_cols: list[str] = [c for c in block_stats.columns if c.startswith("n_faces_")]

        _z_hat = np.array([0.0, 0.0, 1.0])
        rows: list[dict] = []

        for _, srow in st.iterrows():
            p_idx = int(srow["partner"])
            if p_idx < 0:
                continue
            unit_bid = float(srow["block_id"])
            partner_bid = float(st.at[p_idx, "block_id"])
            if unit_bid == partner_bid:
                continue

            # Block centres
            unit_ctr = np.array([float(srow["cx"]), float(srow["cy"]), float(srow["cz"])])
            partner_ctr = np.array([
                float(st.at[p_idx, "cx"]),
                float(st.at[p_idx, "cy"]),
                float(st.at[p_idx, "cz"]),
            ])
            d = partner_ctr - unit_ctr
            d_len = float(np.linalg.norm(d))
            if d_len < 1e-15:
                continue
            d_norm = d / d_len

            R_unit = _sid_to_rot.get(unit_bid)
            if R_unit is None:
                continue
            z_world = R_unit.apply(_z_hat)
            elevation = float(np.degrees(np.arcsin(float(np.clip(np.dot(d_norm, z_world), -1.0, 1.0)))))

            asm_id = float(srow["assembly_id"])
            tomo_val = srow[tomo_col]

            brow: dict = {
                tomo_col: tomo_val,
                "assembly_id": asm_id,
                "unit_id": unit_bid,
                "partner_id": partner_bid,
                "elevation_deg": elevation,
            }
            bs_i = _bs_idx.get(unit_bid)
            if bs_i is not None:
                bstat = block_stats.iloc[bs_i]
                brow["face_sizes"] = bstat["face_sizes"]
                brow["n_faces"] = int(bstat["n_faces"])
                for dc in _dyn_cols:
                    brow[dc] = int(bstat[dc])
            else:
                brow["face_sizes"] = ()
                brow["n_faces"] = 0
                for dc in _dyn_cols:
                    brow[dc] = 0
            rows.append(brow)

        return pd.DataFrame(rows)

    def get_face_vertices_as_motl(
        self,
        size: int | None = None,
        *,
        vertex_class: dict[int, int] | None = None,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> "cryomotl.Motl":
        """Return a :class:`~cryocat.core.cryomotl.Motl` with one row per (block, face) pair.

        Intended for face-opening subtomogram averaging: each output particle
        sits at a block position with its normal unchanged, but the in-plane
        angle rotated so the direction from the block to its face's centroid
        falls on the particle **X-axis**.  Averaging all output particles then
        superposes the openings at a consistent orientation.

        A block bounding two qualifying faces appears twice with different
        in-plane angles — that is intentional, not a duplicate.

        Column assignments use the :class:`BlockLayer` constructor parameters
        (defaults in parentheses):

        * ``tomo_id_column`` (``tomo_id``) — tomogram identifier
        * ``affiliation_column`` (``object_id``) — assembly affiliation
        * ``block_id_column`` (``geom3``) — ``subtomo_id`` of the source block
          in :attr:`blocks`
        * ``face_id_column`` (``geom4``) — 1-based face id within the assembly
        * ``face_size_column`` (``geom5``) — face vertex count

        Parameters
        ----------
        size : int or None, default=None
            Restrict to faces with exactly *size* vertices (e.g. ``size=6``
            for hexagons).  ``None`` includes every face.
        vertex_class : dict of {int: int} or None, default=None
            Exact vertex-class filter: ``{face_size: count, ...}``.  A block
            is included only when its set of face memberships matches *every*
            entry exactly — ``{5: 1, 6: 2}`` keeps only blocks that border
            exactly one pentagon and exactly two hexagons, with no other faces.
            A partial specification such as ``{5: 1}`` means exactly one
            pentagon **and nothing else** (not "one pentagon and any
            remainder").  ``None`` keeps all blocks.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'``
            includes inferred blocks.  ``'inferred'`` returns only blocks
            from inferred faces.

        Returns
        -------
        cryomotl.Motl
            One row per (block, face) pair, renumbered from 1.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called, or if any two of
            ``block_id_column``, ``face_id_column``, ``face_size_column``
            share the same column name.
        """
        from collections import Counter as _Counter

        _check_column_collisions("get_face_vertices_as_motl", {
            "face_id_column": self.face_id_column,
            "face_size_column": self.face_size_column,
            "block_id_column": self.block_id_column,
        })
        self._require_connect(scope)

        _, _, _blocks = self._state_for_scope(scope)
        if _blocks is None or len(_blocks.df) == 0:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns))

        face_stats = self.get_face_stats(scope=scope)
        if size is not None:
            face_stats = face_stats[face_stats["size"] == size].reset_index(drop=True)
        if len(face_stats) == 0:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns))

        # Build block → face_size Counter for vertex_class filtering
        _block_face_cnt: dict[float, _Counter] | None = None
        if vertex_class is not None:
            _block_face_cnt = {}
            for _, frow in self.get_face_stats(scope=scope).iterrows():
                sz = int(frow["size"])
                for bid in frow["block_ids"]:
                    bid_f = float(bid)
                    if bid_f not in _block_face_cnt:
                        _block_face_cnt[bid_f] = _Counter()
                    _block_face_cnt[bid_f][sz] += 1

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        block_sids = _blocks.df["subtomo_id"].astype(float).values
        block_coords = _blocks.get_coordinates()
        block_angles = _blocks.get_angles().astype(float)

        _sid_to_idx: dict[float, int] = {float(s): i for i, s in enumerate(block_sids)}

        _z_hat = np.array([0.0, 0.0, 1.0])
        out_rows: list[dict] = []

        for _, face_row in face_stats.iterrows():
            fc = np.array([
                float(face_row["centroid_x"]),
                float(face_row["centroid_y"]),
                float(face_row["centroid_z"]),
            ])
            face_id_val = float(face_row["face_id"])
            face_size_val = float(face_row["size"])
            face_tomo = float(face_row[tomo_col])
            face_aff = float(face_row[aff_col])

            for bid in face_row["block_ids"]:
                idx = _sid_to_idx.get(float(bid))
                if idx is None:
                    continue
                if _block_face_cnt is not None:
                    cnt = _block_face_cnt.get(float(bid), _Counter())
                    if dict(cnt) != vertex_class:
                        continue

                xyz = block_coords[idx]
                R = srot.from_euler("zxz", block_angles[idx], degrees=True)

                d = fc - xyz
                n = R.apply(_z_hat)
                d_perp = d - np.dot(d, n) * n
                d_perp_len = float(np.linalg.norm(d_perp))

                if d_perp_len > 1e-15:
                    d_in_particle = R.inv().apply(d_perp / d_perp_len)
                    delta = float(np.degrees(np.arctan2(d_in_particle[1], d_in_particle[0])))
                    R_new = R * srot.from_euler("z", delta, degrees=True)
                else:
                    R_new = R

                phi_n, theta_n, psi_n = R_new.as_euler("zxz", degrees=True)

                row: dict = {c: 0.0 for c in cryomotl.Motl.motl_columns}
                row[tomo_col] = face_tomo
                row[aff_col] = face_aff
                row["x"] = float(xyz[0])
                row["y"] = float(xyz[1])
                row["z"] = float(xyz[2])
                row["phi"] = float(phi_n)
                row["theta"] = float(theta_n)
                row["psi"] = float(psi_n)
                row[self.block_id_column] = float(bid)
                row[self.face_id_column] = face_id_val
                row[self.face_size_column] = face_size_val
                out_rows.append(row)

        if not out_rows:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns))

        motl = cryomotl.Motl(pd.DataFrame(out_rows)[cryomotl.Motl.motl_columns])
        motl.renumber_particles()
        motl.update_coordinates()
        return motl

    def plot_object(
        self,
        object_id: float,
        *,
        plot_faces: bool = True,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> Any:
        """Return a Plotly figure showing all blocks of one object, coloured by provenance.

        One trace is drawn per origin class present in the data:
        ``"Real"`` (origin 1) and ``"Inferred"`` (origin 2).
        When *plot_faces* is ``True``, face centroids are overlaid as a second set of
        traces — one per distinct face edge-count — using diamond markers larger than
        the block markers.

        The figure is returned without being displayed.  Pass it to ``fig.show()``
        or use :meth:`browse_objects` for an interactive notebook browser.

        Parameters
        ----------
        object_id : float
            Value of :attr:`affiliation_column` that identifies the object.
        plot_faces : bool, default=True
            When ``True``, overlay face centroids coloured by edge count.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` draws lattice blocks only.  ``'all'`` includes
            inferred blocks as well.  ``'inferred'`` draws only inferred faces.

        Returns
        -------
        plotly.graph_objects.Figure
        """
        import plotly.graph_objects as _go
        from cryocat.analysis import visplot as _vp

        self._require_connect(scope)

        _, _, _blocks = self._state_for_scope(scope)
        if _blocks is None or len(_blocks.df) == 0:
            return _go.Figure()

        aff_col = self.affiliation_column
        oc = self._origin_column
        pal = _vp.resolve_palette(None)

        block_df = _blocks.df[_blocks.df[aff_col] == float(object_id)].copy()
        if len(block_df) == 0:
            return _go.Figure()

        # Treat unset origin (0) as real (1).
        block_df.loc[block_df[oc] == 0.0, oc] = 1.0

        _ORIGIN_LABELS = {1.0: "Real", 2.0: "Inferred"}
        _ORIGIN_ORDER = [1.0, 2.0]

        fig = _go.Figure()

        for k, orig in enumerate(_ORIGIN_ORDER):
            sub = block_df[block_df[oc] == orig]
            if sub.empty:
                continue
            xyz = sub[["x", "y", "z"]].values + sub[["shift_x", "shift_y", "shift_z"]].values
            fig.add_trace(_go.Scatter3d(
                x=xyz[:, 0],
                y=xyz[:, 1],
                z=xyz[:, 2],
                mode="markers",
                name=_ORIGIN_LABELS[orig],
                marker=dict(size=4, color=pal[k % len(pal)], opacity=0.85),
            ))

        if plot_faces:
            try:
                face_stats = self.get_face_stats(scope=scope)
            except ValueError:
                face_stats = pd.DataFrame()

            if len(face_stats) > 0:
                if aff_col in face_stats.columns:
                    face_stats = face_stats[face_stats[aff_col] == float(object_id)]

                for j, size_val in enumerate(sorted(face_stats["size"].unique())):
                    sub_f = face_stats[face_stats["size"] == size_val]
                    fig.add_trace(_go.Scatter3d(
                        x=sub_f["centroid_x"].values,
                        y=sub_f["centroid_y"].values,
                        z=sub_f["centroid_z"].values,
                        mode="markers",
                        name=str(int(size_val)),
                        marker=dict(
                            size=8,
                            symbol="diamond",
                            color=pal[(len(_ORIGIN_ORDER) + j) % len(pal)],
                            opacity=0.7,
                        ),
                    ))

        _vp.apply_defaults(fig)
        fig.update_layout(
            title=f"Object {object_id}",
            scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
        )
        return fig

    def browse_objects(self, *, plot_faces: bool = True, scope: Literal["original", "inferred", "all"] = "original") -> None:
        """Browse objects interactively in a Jupyter notebook.

        Shows one object at a time using :meth:`plot_object`, with Previous / Next
        buttons and a dropdown to jump directly to an object.  Requires
        ``ipywidgets``; if it is not installed a clear :exc:`ImportError` is raised
        with an install hint.

        Parameters
        ----------
        plot_faces : bool, default=True
            Passed to :meth:`plot_object`.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            Passed to :meth:`plot_object`.
        """
        from cryocat.analysis import visplot as _vp

        self._require_connect(scope)
        _, _, _blocks = self._state_for_scope(scope)
        if _blocks is None or len(_blocks.df) == 0:
            raise ValueError("No blocks available for the requested scope.")

        aff_col = self.affiliation_column
        object_ids = sorted(float(v) for v in _blocks.df[aff_col].unique())

        _vp._browse_widget(
            object_ids,
            lambda oid: self.plot_object(oid, plot_faces=plot_faces, scope=scope),
        )

    @gui_exposed(label="Gaps as motl", group="Lattice motls", order=20, returns="motl", category="pleomorphic-op")
    def get_gaps_as_motl(
        self,
        cluster_radius: float,
        min_blocks: int | None = None,
        *,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> "tuple[cryomotl.Motl, pd.DataFrame]":
        """Predict missing-block positions from unmatched contact sites.

        Parameters
        ----------
        cluster_radius : float
            Maximum distance (voxels) between two gap points to be in one cluster.
        min_blocks : int or None, default=None
            Minimum number of distinct source blocks in a cluster to keep it.
            Defaults to the highest derivable ideal degree from the block
            definitions (3, 4, or 6 based on symmetry), else 3.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            ``'original'`` uses the real-block contact graph.  ``'all'`` uses
            the combined lattice + inferred block graph.  ``'inferred'`` uses
            only contacts in inferred faces.

        Returns
        -------
        motl : cryomotl.Motl
            One row per kept cluster.  Column assignments are controlled by
            :class:`BlockLayer` constructor parameters (defaults in
            parentheses):

            * ``tomo_id_column`` (``tomo_id``) — tomogram identifier
            * ``affiliation_column`` (``object_id``) — assembly affiliation
            * ``x``, ``y``, ``z`` — cluster centroid
            * ``phi``, ``theta``, ``psi`` — Euler from mean z of source blocks
        diagnostics : pd.DataFrame
            One row per kept cluster, indexed by ``gap_id`` (matching
            ``subtomo_id`` in *motl*).  Columns:

            * ``gap_id`` — 1-based cluster id
            * ``n_source_blocks`` — distinct source blocks contributing to
              this cluster
        """
        st, _, _blocks = self._state_for_scope(scope)
        import networkx as _nx

        if min_blocks is None:
            if self.block_definition is not None:
                _defs = (
                    list(self.block_definition.values())
                    if isinstance(self.block_definition, dict)
                    else [self.block_definition]
                )
                _ns = [d.effective_n_sites for d in _defs if d.effective_n_sites in (3, 4, 6)]
                min_blocks = max(_ns) if _ns else 3
            else:
                min_blocks = 3

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        block_to_row: dict[float, int] = {
            float(_blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(_blocks.df))
        }
        angles_all = _blocks.get_angles().astype(float)
        all_R = srot.from_euler("zxz", angles_all, degrees=True)
        z_unit = np.array([0.0, 0.0, 1.0])

        unmatched = st[st["partner"] < 0]
        gap_pts_by_key: dict[tuple, list] = {}
        gap_bids_by_key: dict[tuple, list] = {}
        gap_zax_by_key: dict[tuple, list] = {}

        for _, row in unmatched.iterrows():
            tomo_val = row[tomo_col]
            aff_val = float(row[aff_col])
            key = (tomo_val, aff_val)
            b_id = float(row["block_id"])
            c_b = np.array([row["cx"], row["cy"], row["cz"]])
            P_h = np.array([row["x"], row["y"], row["z"]])
            g_h = c_b + 2.0 * (P_h - c_b)
            z_ax = all_R[block_to_row[b_id]].apply(z_unit)
            gap_pts_by_key.setdefault(key, []).append(g_h)
            gap_bids_by_key.setdefault(key, []).append(b_id)
            gap_zax_by_key.setdefault(key, []).append(z_ax)

        out_rows: list[dict] = []
        diag_rows: list[dict] = []
        for (tomo_val, aff_val), pts_list in gap_pts_by_key.items():
            pts = np.array(pts_list)
            bids = np.array(gap_bids_by_key[(tomo_val, aff_val)])
            z_axes = np.array(gap_zax_by_key[(tomo_val, aff_val)])
            n_pts = len(pts)

            G = _nx.Graph()
            G.add_nodes_from(range(n_pts))
            if n_pts >= 2:
                qp_idx, nn_idx = nnana.find_nn_within_radius(pts, pts, cluster_radius, remove_qp=True)
                for qi, nns in zip(qp_idx, nn_idx):
                    for ni in nns:
                        G.add_edge(qi, int(ni))

            for comp_nodes in _nx.connected_components(G):
                nodes = sorted(comp_nodes)
                comp_bids = bids[nodes]
                n_distinct = len(set(comp_bids.tolist()))
                if n_distinct < min_blocks:
                    continue
                mean_pos = pts[nodes].mean(axis=0)
                mean_z = z_axes[nodes].mean(axis=0)
                nrm = float(np.linalg.norm(mean_z))
                mean_z = mean_z / nrm if nrm > 1e-15 else mean_z
                euler = geom.normals_to_euler_angles(mean_z.reshape(1, 3))[0]
                out_rows.append(
                    {
                        tomo_col: tomo_val,
                        aff_col: aff_val,
                        "x": mean_pos[0],
                        "y": mean_pos[1],
                        "z": mean_pos[2],
                        "phi": float(euler[0]),
                        "theta": float(euler[1]),
                        "psi": float(euler[2]),
                    }
                )
                diag_rows.append({"n_source_blocks": float(n_distinct)})

        data: dict[str, Any] = {c: np.zeros(len(out_rows)) for c in cryomotl.Motl.motl_columns}
        for k, row in enumerate(out_rows):
            for col, val in row.items():
                data[col][k] = val
        motl = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])
        if out_rows:
            motl.renumber_particles()
        diag_df = pd.DataFrame(diag_rows, columns=["n_source_blocks"])
        diag_df.insert(0, "gap_id", range(1, len(diag_df) + 1))
        return motl, diag_df

    def _fit_orientation_from_tips(
        self,
        V: np.ndarray,
        near_bids: np.ndarray,
        block_to_row: "dict[float, int]",
        all_R: "srot",
        site_shift_use: np.ndarray,
        n: int | None,
        ang_step: float,
        tol_angle: float,
        min_support: int,
        _diagnostics: "dict | None" = None,
    ) -> "tuple[float, float, float, float, float] | None":
        """Fit block orientation from tip vectors relative to the candidate centre.

        Returns ``(phi, theta, psi, angular_rms, tip_plane_angle)`` in degrees,
        or ``None`` when the Cn check fails or no real blocks supply an axis.
        Non-Cn blocks always return a result (``angular_rms=0``, ``tip_plane_angle=0``).

        Parameters
        ----------
        V : ndarray (K, 3)
            Tip vectors relative to the candidate centre (NOT normalised).
        near_bids : ndarray (K,)
            Block id for each tip; virtual legs use ids absent from *block_to_row*.
        block_to_row : dict
            Maps real block_id → row index into *all_R*.
        all_R : scipy.spatial.transform.Rotation
            Rotations for each real block, same order as *block_to_row* values.
        site_shift_use : ndarray (3,)
            Site-shift vector used for in-plane alignment.
        n : int or None
            Cn symmetry order; ``None`` or ≤ 1 → non-Cn path.
        ang_step : float
            Ideal angular step ``360/n`` (degrees); 0 for non-Cn.
        tol_angle : float
            Maximum allowed pairwise residual for the Cn check (degrees).
        min_support : int
            Minimum projectable tips required for the Cn check to proceed.
        """
        _z_unit = np.array([0.0, 0.0, 1.0])
        site_length = float(np.linalg.norm(site_shift_use))

        phi_c = theta_c = psi_c = 0.0
        angular_rms = 0.0
        tip_plane_angle = 0.0

        if n is not None and len(V) >= 2 and ang_step > 0.0:
            # ── Cn symmetry check ──────────────────────────────────────────
            V_n = V / np.where(
                np.linalg.norm(V, axis=1, keepdims=True) < 1e-15,
                1.0,
                np.linalg.norm(V, axis=1, keepdims=True),
            )

            # Symmetry axis from mean z-vector of real supporting blocks.
            _z_axes_cn: list[np.ndarray] = []
            for _bid_cn in near_bids:
                _ri_cn = block_to_row.get(float(_bid_cn))
                if _ri_cn is not None:
                    _z_axes_cn.append(all_R[_ri_cn].apply(_z_unit))
            if not _z_axes_cn:
                if _diagnostics is not None:
                    _diagnostics["reason"] = "no_axis"
                return None  # no real blocks — orientation undefined
            _mean_z_cn = np.array(_z_axes_cn).mean(axis=0)
            _mz_norm = float(np.linalg.norm(_mean_z_cn))
            z_c = _mean_z_cn / _mz_norm if _mz_norm > 1e-15 else _z_unit

            # Tip-plane quality: angle between SVD axis and mean-particle normal.
            if len(V_n) >= 3:
                _, _, _Vt_q = np.linalg.svd(V_n, full_matrices=False)
                _tip_ax = _Vt_q[-1]
            else:
                _tip_ax = np.cross(V_n[0], V_n[1])
                _tp_m = float(np.linalg.norm(_tip_ax))
                _tip_ax = _tip_ax / _tp_m if _tp_m > 1e-12 else z_c
            tip_plane_angle = float(np.degrees(np.arccos(
                np.clip(abs(float(np.dot(_tip_ax, z_c))), 0.0, 1.0)
            )))

            # Project tip vectors onto the equatorial plane.
            V_proj = V_n - np.outer(V_n @ z_c, z_c)
            V_proj_mag = np.linalg.norm(V_proj, axis=1)
            good = V_proj_mag > 1e-12
            if int(good.sum()) < min_support:
                if _diagnostics is not None:
                    _diagnostics["reason"] = "too_few_projectable"
                return None
            V_proj_n = V_proj[good] / V_proj_mag[good, None]

            # Check pairwise angular spacings are multiples of 360/n.
            residuals: list[float] = []
            failed = False
            for _ii in range(len(V_proj_n)):
                for _jj in range(_ii + 1, len(V_proj_n)):
                    cos_a = float(np.clip(np.dot(V_proj_n[_ii], V_proj_n[_jj]), -1.0, 1.0))
                    ang = float(np.degrees(np.arccos(cos_a)))
                    k_near = max(1, round(ang / ang_step))
                    res = abs(ang - k_near * ang_step)
                    if res > tol_angle:
                        failed = True
                        if _diagnostics is not None:
                            _diagnostics["reason"] = "cn_check"
                            _diagnostics["ang_residual"] = res
                            _diagnostics["ang_tolerance"] = tol_angle
                        break
                    residuals.append(res)
                if failed:
                    break
            if failed:
                return None

            angular_rms = float(np.sqrt(np.mean(np.array(residuals) ** 2))) if residuals else 0.0

            # Orientation: Rodrigues rotation maps [0,0,1] → z_c,
            # then in-plane alignment to first observed tip direction.
            _z0 = np.array([0.0, 0.0, 1.0])
            _cross = np.cross(_z0, z_c)
            _cross_mag = float(np.linalg.norm(_cross))
            if _cross_mag < 1e-12:
                R_ax = (
                    srot.identity() if z_c[2] > 0.0 else srot.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
                )
            else:
                _angle_z = float(np.arccos(np.clip(float(np.dot(_z0, z_c)), -1.0, 1.0)))
                R_ax = srot.from_rotvec((_cross / _cross_mag) * _angle_z)

            # Local first-site direction mapped to global frame by R_ax.
            v_site_w = R_ax.apply(site_shift_use / site_length)
            # Project onto equatorial plane.
            v_eq = v_site_w - float(np.dot(v_site_w, z_c)) * z_c
            v_eq_mag = float(np.linalg.norm(v_eq))
            if v_eq_mag > 1e-12:
                v_eq /= v_eq_mag
                ref_dir = V_proj_n[0]
                cos_d = float(np.clip(np.dot(v_eq, ref_dir), -1.0, 1.0))
                sin_d = float(np.dot(z_c, np.cross(v_eq, ref_dir)))
                delta = float(np.arctan2(sin_d, cos_d))
                R_psi = srot.from_rotvec(z_c * delta)
                R_final = R_psi * R_ax
            else:
                R_final = R_ax

            phi_c, theta_c, psi_c = (float(a) for a in R_final.as_euler("zxz", degrees=True))

        else:
            # Non-Cn or single tip: derive orientation from mean z-axis of supporting blocks.
            _z_axes: list[np.ndarray] = []
            for _bid in near_bids:
                _ri = block_to_row.get(float(_bid))
                if _ri is not None:
                    _z_axes.append(all_R[_ri].apply(_z_unit))
            if _z_axes:
                mean_z = np.mean(_z_axes, axis=0)
                _nrm = float(np.linalg.norm(mean_z))
                mean_z = mean_z / _nrm if _nrm > 1e-15 else mean_z
                _euler = geom.normals_to_euler_angles(mean_z.reshape(1, 3))[0]
                phi_c = float(_euler[0])
                theta_c = float(_euler[1])
                psi_c = float(_euler[2])

        return (phi_c, theta_c, psi_c, angular_rms, tip_plane_angle)

    @gui_exposed(
        label="Infer missing blocks",
        group="Lattice motls",
        order=25,
        returns="motl",
        category="pleomorphic-op",
    )
    def infer_missing_blocks(
        self,
        min_support: int = 2,
        tip_tolerance: float | None = None,
        angle_tolerance: float | None = None,
        max_rounds: int = 5,
        site_shift: "TripletLike | None" = None,
        clean_distance: float | None = None,
        min_real_support: int = 0,
    ) -> "tuple[cryomotl.Motl, pd.DataFrame]":
        """Infer missing-block positions from unpaired legs via Cn symmetry fitting.

        A missing block leaves its neighbours' legs unpaired.  For each
        unpaired leg the candidate centre is placed two site-shift lengths
        along the leg from the owning block's centre.  A candidate is kept
        only when at least *min_support* unpaired legs agree and their tips
        form a Cn-consistent arrangement.

        The symmetry axis (z_c) is taken from the mean z-vector of the
        supporting real blocks, which carries the correct sign from the
        particles' own orientations.  The in-plane angle is then fitted
        in that frame.  The tip-plane angle (angle between the SVD-derived
        tip-plane normal and the supporting blocks' mean z-vector) is
        returned in the diagnostics DataFrame; large values flag
        geometrically suspect candidates.

        Duplicate candidates (closer than *clean_distance*) are suppressed;
        the one with lower angular RMS wins; support breaks ties.  Iteration
        continues until no new candidates are found or *max_rounds* is
        reached; later rounds carry higher iteration numbers in the output.

        When ``block_definition`` is a dict each candidate position is tried
        against every definition; the type with the lowest angular RMS wins.
        The winning type is recorded in ``block_type_column`` and the
        runner-up RMS is returned in the diagnostics DataFrame.

        After all rounds a final cleaning pass is run over the complete
        accepted set (all groups combined).  Real blocks always beat any
        candidate; among candidates the priority is: earlier round, then
        higher support, then lower angular RMS.  The number removed by this
        final pass is printed to stdout.

        When the returned motl is non-empty the accepted candidates are also
        stored in :attr:`_added` (tagged with origin 2 in
        :attr:`_origin_column`).  A subsequent call replaces the inferred
        portion of :attr:`_added`.  Call ``connect(scope='all')`` to build a
        contact graph that includes both real and inferred blocks.

        Parameters
        ----------
        min_support : int, default=2
            Minimum number of unpaired legs that must support a candidate.
            1 is ambiguous (may be a genuine lattice edge); 2 is the
            practical minimum.
        tip_tolerance : float or None, default=None
            Tolerance (voxels) for accepting a leg tip as near the candidate
            centre.  ``None`` falls back to ``self.tip_tolerance``; if that
            is also ``None``, derives to ``0.3 × |site_shift|``.
        angle_tolerance : float or None, default=None
            Maximum angular residual (degrees) for the Cn pairwise spacing
            check.  Only applied to Cn-symmetric block definitions.  ``None``
            falls back to ``self.angle_tolerance``; if that is also ``None``,
            derives to ``0.125 × (360 / n)``.  Raises ``ValueError`` when
            the resolved value is ``>= 0.5 × (360 / n)``.
        max_rounds : int, default=5
            Cap on iteration rounds.
        site_shift : TripletLike or None, default=None
            Override the shift vector used for candidate placement and
            support collection.  ``None`` uses the block definition's own
            first-site vector.  Supply the corrected or mirrored vector
            when the contact is asymmetric — e.g. if the definition uses
            ``[45, -3, 0]`` but the partner's tip sits at ``[45, +3, 0]``.
            The magnitude determines the site length; the direction
            governs the in-plane orientation fit and virtual-leg placement.
        clean_distance : float or None, default=None
            Deduplication distance (voxels) applied at two moments: (1) after
            each round, candidates closer than this to a better-ranked
            candidate or real block are discarded; (2) after all rounds, a
            final global pass removes any remaining duplicates.  Higher
            support wins; lower angular RMS breaks ties.  ``None`` falls back
            to ``self.clean_distance``; if that is also ``None``, derives to
            ``0.4 × |site_shift|``.
        min_real_support : int, default=0
            Minimum number of real-block legs a candidate must have to
            appear in the returned motl.  0 returns everything;
            1 excludes pure-virtual candidates; 2 requires at least two
            confirmations from actual data.

        Returns
        -------
        motl : cryomotl.Motl
            One row per inferred block.  Carries position, orientation,
            affiliation, block type, and origin.  Columns:
            ``tomo_id_column`` (``tomo_id``), ``affiliation_column``
            (``object_id``), ``x``, ``y``, ``z``, ``phi``, ``theta``,
            ``psi``, ``block_type_column`` (default ``class``),
            ``_origin_column`` (default ``geom5``, value 2).  All other
            columns are zero.
        diagnostics : pd.DataFrame
            One row per inferred block, keyed by ``subtomo_id`` (matches
            the returned motl).  Columns:

            * ``subtomo_id``
            * ``n_support`` — total unpaired legs supporting the candidate
              (real + virtual); this is the value used by ``min_support``
            * ``n_real_support`` — legs from real blocks; filtered by
              ``min_real_support``
            * ``angular_rms`` — angular RMS of the winning type's
              orientation fit (degrees)
            * ``iteration`` — round that accepted this candidate (1 = first)
            * ``block_type`` — type code of the winning definition
            * ``runner_up_rms`` — angular RMS of the runner-up definition
              (degrees); ``inf`` when only one definition was tried; a small
              margin (winner ≈ runner-up) means the type assignment is
              ambiguous
            * ``tip_plane_angle`` — angle (degrees) between the SVD-derived
              tip-plane normal and the supporting blocks' mean z-vector; 0
              for non-Cn blocks; large values flag geometrically suspect
              candidates

        Raises
        ------
        ValueError
            If ``connect()`` has not been called or no block definition is
            set.
        """
        self._require_connect()
        if self.block_definition is None or self.blocks is None:
            raise ValueError("No block definition: call from_blocks() or set block_definition.")
        _check_column_collisions("infer_missing_blocks", {
            "block_type_column": self.block_type_column,
            "_origin_column": self._origin_column,
        })

        # ── Site geometry — per type ──────────────────────────────────────
        import re as _re_s

        _bd_items_inf: list[tuple[float, "BlockDefinition"]] = (
            [(float(k), v) for k, v in self.block_definition.items()]
            if isinstance(self.block_definition, dict)
            else [(1.0, self.block_definition)]
        )
        # {type_code: (sv, site_length, tol_r, clean, n, ang_step, angle_tolerance)}
        _type_geom_inf: dict[float, tuple] = {}
        for _tc_inf, _bd_inf in _bd_items_inf:
            _sv_local: np.ndarray = _bd_inf.site_vectors()[0].astype(float)
            _sv_use_t: np.ndarray = geom.as_triplet(site_shift) if site_shift is not None else _sv_local
            _sl_t: float = float(np.linalg.norm(_sv_use_t))
            if _sl_t < 1e-15:
                raise ValueError(f"site_shift vector has zero length for block type {_tc_inf}.")
            _tol_r_t: float = (
                tip_tolerance if tip_tolerance is not None
                else self.tip_tolerance if self.tip_tolerance is not None
                else 0.3 * _sl_t
            )
            _clean_t: float = (
                clean_distance if clean_distance is not None
                else self.clean_distance if self.clean_distance is not None
                else 0.4 * _sl_t
            )
            _sm_t = _re_s.match(r"C(\d+)", _bd_inf.symmetry or "")
            _n_t: int | None = int(_sm_t.group(1)) if _sm_t else None
            _ang_step_t: float = 360.0 / _n_t if (_n_t is not None and _n_t > 1) else 0.0
            _tol_ang_raw_t = angle_tolerance if angle_tolerance is not None else self.angle_tolerance
            if _n_t is not None and _ang_step_t > 0.0:
                _tol_ang_t: float = _tol_ang_raw_t if _tol_ang_raw_t is not None else 0.125 * _ang_step_t
                if _tol_ang_t >= 0.5 * _ang_step_t:
                    raise ValueError(
                        f"angle_tolerance {_tol_ang_t:.3f}° >= 0.5 × angular step "
                        f"{_ang_step_t:.3f}° (C{_n_t}, block type {_tc_inf}); assignment is ambiguous."
                    )
            else:
                _tol_ang_t = _tol_ang_raw_t if _tol_ang_raw_t is not None else 15.0
            _type_geom_inf[_tc_inf] = (_sv_use_t, _sl_t, _tol_r_t, _clean_t, _n_t, _ang_step_t, _tol_ang_t)

        _clean_global: float = min(g[3] for g in _type_geom_inf.values())

        # ── Block orientations (needed for non-Cn fallback) ────────────────
        _block_to_row: dict[float, int] = {
            float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(self.blocks.df))
        }
        _angles_blocks = self.blocks.get_angles().astype(float)
        _all_R = srot.from_euler("zxz", _angles_blocks, degrees=True)
        _z_unit = np.array([0.0, 0.0, 1.0])

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        st = self._site_table

        out_rows: list[dict] = []

        for (tomo_val, aff_val), grp in st.groupby([tomo_col, aff_col], sort=True):
            unpaired = grp[grp["partner"] < 0].reset_index(drop=True)
            M = len(unpaired)
            if M < min_support:
                continue

            # Mutable lists — virtual legs are appended after each round.
            tips_list: list[np.ndarray] = list(unpaired[["x", "y", "z"]].values.astype(float))
            cents_list: list[np.ndarray] = list(unpaired[["cx", "cy", "cz"]].values.astype(float))
            bids_list: list[float] = list(unpaired["block_id"].values.astype(float))

            unexplained: set[int] = set(range(M))

            # Per-group tracking for per-round cleaning.
            group_rows: list[dict] = []
            group_active: list[bool] = []
            _vleg_for_cand: dict[int, list[int]] = {}
            _rb_mask = self.blocks.df[tomo_col] == tomo_val
            if aff_col in self.blocks.df.columns:
                _rb_mask = _rb_mask & (self.blocks.df[aff_col] == float(aff_val))
            _real_xyz = self.blocks.get_coordinates()[_rb_mask.values]

            for round_idx in range(max_rounds):
                if len(unexplained) < min_support:
                    break

                unexpl_idx: list[int] = sorted(unexplained)
                tips_arr = np.array(tips_list)
                cents_arr = np.array(cents_list)
                tips_r = tips_arr[unexpl_idx]  # (K, 3)
                cents_r = cents_arr[unexpl_idx]  # (K, 3)

                round_cands: list[dict] = []

                for i, h in enumerate(unexpl_idx):
                    P_h = tips_r[i]
                    c_h = cents_r[i]
                    d = P_h - c_h
                    d_mag = float(np.linalg.norm(d))
                    if d_mag < 1e-15:
                        continue
                    d_unit = d / d_mag

                    # Try each definition; keep the best (lowest angular RMS) fit.
                    _best_cand: dict | None = None
                    _best_rms: float = float("inf")
                    _runner_up_rms: float = float("inf")

                    for _tc_c, (_sv_c, _sl_c, _tol_r_c, _, _n_c, _ang_step_c, _tol_ang_c) in _type_geom_inf.items():
                        C = c_h + 2.0 * d_unit * _sl_c

                        dists = np.linalg.norm(tips_r - C, axis=1)
                        near_local = np.where(dists <= _sl_c + _tol_r_c)[0]
                        if len(near_local) < min_support:
                            continue

                        V = tips_r[near_local] - C
                        radii = np.linalg.norm(V, axis=1)
                        ok_r = np.abs(radii - _sl_c) <= _tol_r_c
                        near_local = near_local[ok_r]
                        V = V[ok_r]
                        if len(near_local) < min_support:
                            continue

                        near_global = np.array(unexpl_idx)[near_local]
                        _near_bids_fit = np.array(bids_list)[near_global]
                        _fit = self._fit_orientation_from_tips(
                            V, _near_bids_fit, _block_to_row, _all_R, _sv_c,
                            _n_c, _ang_step_c, _tol_ang_c, min_support,
                        )
                        if _fit is None:
                            continue

                        phi_c, theta_c, psi_c, angular_rms, tip_plane_angle = _fit
                        _bids_near = np.array(bids_list)[near_global]
                        _n_real = int(np.sum(_bids_near >= 0.0))

                        if angular_rms < _best_rms:
                            _runner_up_rms = _best_rms
                            _best_rms = angular_rms
                            _best_cand = {
                                tomo_col: tomo_val,
                                aff_col: float(aff_val),
                                "x": float(C[0]),
                                "y": float(C[1]),
                                "z": float(C[2]),
                                "phi": phi_c,
                                "theta": theta_c,
                                "psi": psi_c,
                                "n_support": int(len(near_global)),
                                "n_real_support": _n_real,
                                "angular_rms": angular_rms,
                                "tip_angle": tip_plane_angle,
                                "iteration": round_idx + 1,
                                "_near_global": near_global,
                                "block_type": _tc_c,
                                "_sv_c": _sv_c,
                                "_n_c": _n_c,
                                "_ang_step_c": _ang_step_c,
                            }
                        else:
                            _runner_up_rms = min(_runner_up_rms, angular_rms)

                    if _best_cand is None:
                        continue

                    _best_cand["runner_up_rms"] = _runner_up_rms
                    round_cands.append(_best_cand)

                if not round_cands:
                    break

                # Deduplicate: keep best angular RMS (lowest) candidate when two
                # centres are within merge_distance of each other.
                round_cands.sort(key=lambda c: (c["angular_rms"], -c["n_support"]))
                kept: list[dict] = []
                for cand in round_cands:
                    C_arr = np.array([cand["x"], cand["y"], cand["z"]])
                    if all(float(np.linalg.norm(C_arr - np.array([k["x"], k["y"], k["z"]]))) > _clean_global for k in kept):
                        kept.append(cand)

                if not kept:
                    break

                for cand in kept:
                    unexplained -= set(cand["_near_global"].tolist())
                    _gr_idx = len(group_rows)
                    group_rows.append(cand)
                    group_active.append(True)
                    # Generate virtual legs from the accepted candidate so that
                    # round n+1 can use them to support further candidates.
                    _cand_n_c = cand.get("_n_c")
                    _cand_ang_step_c = cand.get("_ang_step_c", 0.0)
                    _cand_sv_c: np.ndarray = cand.get("_sv_c", np.zeros(3))
                    if _cand_n_c is not None and _cand_ang_step_c > 0.0:
                        _vlegs: list[int] = []
                        R_cand = srot.from_euler(
                            "zxz",
                            [cand["phi"], cand["theta"], cand["psi"]],
                            degrees=True,
                        )
                        C_cand = np.array([cand["x"], cand["y"], cand["z"]])
                        for _k in range(_cand_n_c):
                            _ang_k = float(_k) * _cand_ang_step_c
                            _R_k = srot.from_rotvec(_z_unit * np.radians(_ang_k))
                            _tip_k = C_cand + R_cand.apply(_R_k.apply(_cand_sv_c))
                            _new_idx = len(tips_list)
                            tips_list.append(_tip_k)
                            cents_list.append(C_cand)
                            bids_list.append(-1.0)  # virtual block
                            unexplained.add(_new_idx)
                            _vlegs.append(_new_idx)
                        _vleg_for_cand[_gr_idx] = _vlegs

                # ── Clean duplicates after this round ──────────────────────
                if _clean_global > 0.0 and any(group_active):
                    _active_idx = [i for i, a in enumerate(group_active) if a]
                    _N_real = len(_real_xyz)
                    _tmp_data: dict[str, list] = {c: [] for c in cryomotl.Motl.motl_columns}
                    # Real blocks → always survive (high score).
                    for _rx, _ry, _rz in _real_xyz:
                        for _c in cryomotl.Motl.motl_columns:
                            _tmp_data[_c].append(0.0)
                        _tmp_data[tomo_col][-1] = float(tomo_val)
                        _tmp_data["x"][-1] = float(_rx)
                        _tmp_data["y"][-1] = float(_ry)
                        _tmp_data["z"][-1] = float(_rz)
                        _tmp_data["score"][-1] = 1e9
                    # Active candidates — ranked by support then angular fit.
                    for _ai in _active_idx:
                        _c2 = group_rows[_ai]
                        for _c in cryomotl.Motl.motl_columns:
                            _tmp_data[_c].append(0.0)
                        _tmp_data[tomo_col][-1] = float(tomo_val)
                        _tmp_data["x"][-1] = _c2["x"]
                        _tmp_data["y"][-1] = _c2["y"]
                        _tmp_data["z"][-1] = _c2["z"]
                        _tmp_data["score"][-1] = float(_c2["n_support"]) * 1000.0 - _c2["angular_rms"]
                    _tmp_df = pd.DataFrame({c: np.array(v, dtype=float) for c, v in _tmp_data.items()})[
                        cryomotl.Motl.motl_columns
                    ]
                    # Encode tracking in subtomo_id.
                    _tmp_sids = list(range(1, _N_real + 1)) + [_N_real + j + 1 for j in range(len(_active_idx))]
                    _tmp_df["subtomo_id"] = _tmp_sids
                    _keep_pr = nnana.nms_by_distance(
                        _tmp_df[["x", "y", "z"]].values,
                        _tmp_df["score"].values,
                        _clean_global,
                        keep_greater=True,
                    )
                    _surv_sids: set[float] = set(_tmp_df["subtomo_id"].values[_keep_pr].tolist())
                    for _j, _ai in enumerate(_active_idx):
                        if float(_N_real + _j + 1) not in _surv_sids:
                            group_active[_ai] = False
                            for _vli in _vleg_for_cand.get(_ai, []):
                                unexplained.discard(_vli)

            out_rows.extend(r for i, r in enumerate(group_rows) if group_active[i])

        # ── Build output motl ──────────────────────────────────────────────
        if min_real_support > 0:
            out_rows = [r for r in out_rows if r.get("n_real_support", 0) >= min_real_support]

        # ── Final global cleaning: real blocks always beat any candidate ───
        # One pass over the complete accepted set (all groups, all rounds).
        # Priority: real block > earlier round > higher support > lower RMS.
        if _clean_global > 0.0 and out_rows:
            _max_iter_f = max(r["iteration"] for r in out_rows)
            _cands_by_key_f: dict[tuple, list[tuple[int, dict]]] = {}
            for _jf, _rf in enumerate(out_rows):
                _cands_by_key_f.setdefault((_rf[tomo_col], _rf[aff_col]), []).append((_jf, _rf))

            _survive_f: set[int] = set()
            for (_tv_f, _av_f), _cands_f in _cands_by_key_f.items():
                _rb_mask_f = self.blocks.df[tomo_col] == _tv_f
                if aff_col in self.blocks.df.columns:
                    _rb_mask_f = _rb_mask_f & (self.blocks.df[aff_col] == float(_av_f))
                _real_xyz_f = self.blocks.get_coordinates()[_rb_mask_f.values]
                _N_real_f = len(_real_xyz_f)

                # Build combined data dict with the same row-by-row pattern as
                # the per-round cleaning so that coordinate encoding is identical.
                _tmp_data_f: dict[str, list] = {c: [] for c in cryomotl.Motl.motl_columns}
                for _rx_f, _ry_f, _rz_f in _real_xyz_f:
                    for _c in cryomotl.Motl.motl_columns:
                        _tmp_data_f[_c].append(0.0)
                    _tmp_data_f[tomo_col][-1] = float(_tv_f)
                    _tmp_data_f["x"][-1] = float(_rx_f)
                    _tmp_data_f["y"][-1] = float(_ry_f)
                    _tmp_data_f["z"][-1] = float(_rz_f)
                    _tmp_data_f["score"][-1] = 1e12
                for _, _rc_f in _cands_f:
                    for _c in cryomotl.Motl.motl_columns:
                        _tmp_data_f[_c].append(0.0)
                    _tmp_data_f[tomo_col][-1] = float(_tv_f)
                    _tmp_data_f["x"][-1] = _rc_f["x"]
                    _tmp_data_f["y"][-1] = _rc_f["y"]
                    _tmp_data_f["z"][-1] = _rc_f["z"]
                    _tmp_data_f["score"][-1] = (
                        float(_max_iter_f - _rc_f["iteration"] + 1) * 1e7
                        + float(_rc_f["n_support"]) * 1e3
                        - _rc_f["angular_rms"]
                    )
                _tmp_sids_f = (
                    list(range(1, _N_real_f + 1))
                    + [_N_real_f + _jj_f + 1 for _jj_f in range(len(_cands_f))]
                )
                _tmp_df_f = pd.DataFrame(
                    {c: np.array(v, dtype=float) for c, v in _tmp_data_f.items()}
                )[cryomotl.Motl.motl_columns]
                _tmp_df_f["subtomo_id"] = _tmp_sids_f
                _keep_f = nnana.nms_by_distance(
                    _tmp_df_f[["x", "y", "z"]].values,
                    _tmp_df_f["score"].values,
                    _clean_global,
                    keep_greater=True,
                )
                _surv_sids_f: set[float] = set(_tmp_df_f["subtomo_id"].values[_keep_f].tolist())
                for _jj_f, (_jf, _) in enumerate(_cands_f):
                    if float(_N_real_f + _jj_f + 1) in _surv_sids_f:
                        _survive_f.add(_jf)

            _n_removed_f = len(out_rows) - len(_survive_f)
            out_rows = [r for j, r in enumerate(out_rows) if j in _survive_f]
            print(f"Final pass removed {_n_removed_f} candidates duplicating real blocks or lower-priority candidates.")

        data: dict[str, Any] = {c: np.zeros(len(out_rows)) for c in cryomotl.Motl.motl_columns}
        _diag_rows: list[dict] = []
        for k, row in enumerate(out_rows):
            data[tomo_col][k] = row[tomo_col]
            data[aff_col][k] = row[aff_col]
            data["x"][k] = row["x"]
            data["y"][k] = row["y"]
            data["z"][k] = row["z"]
            data["phi"][k] = row["phi"]
            data["theta"][k] = row["theta"]
            data["psi"][k] = row["psi"]
            data[self.block_type_column][k] = float(row.get("block_type", 1.0))
            data[self._origin_column][k] = 2.0
            _diag_rows.append({
                "subtomo_id": 0.0,
                "n_support": int(row["n_support"]),
                "n_real_support": int(row.get("n_real_support", 0)),
                "angular_rms": float(row["angular_rms"]),
                "iteration": int(row["iteration"]),
                "block_type": float(row.get("block_type", 1.0)),
                "runner_up_rms": float(row.get("runner_up_rms", float("inf"))),
                "tip_plane_angle": float(row.get("tip_angle", 0.0)),
            })

        motl = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])
        if out_rows:
            self._update_added(motl.df.copy(), origin=2)
            _origin_2_mask = self._added.df[self._origin_column] == 2.0
            _assigned_sids = self._added.df.loc[_origin_2_mask, "subtomo_id"].values
            motl.df["subtomo_id"] = _assigned_sids
            for _k, _sid in enumerate(_assigned_sids):
                _diag_rows[_k]["subtomo_id"] = float(_sid)

        _diag_cols = ["subtomo_id", "n_support", "n_real_support", "angular_rms", "iteration", "block_type", "runner_up_rms", "tip_plane_angle"]
        _diagnostics = (
            pd.DataFrame(_diag_rows, columns=_diag_cols)
            if _diag_rows
            else pd.DataFrame(columns=_diag_cols)
        )
        return motl, _diagnostics

    def recover_orientations(
        self,
        motl: "cryomotl.Motl",
        *,
        min_neighbours: int = 2,
        bonding_tolerance: float | None = None,
        angle_tolerance: float | None = None,
    ) -> "tuple[cryomotl.Motl, pd.DataFrame]":
        """Recover orientations for particles removed from :attr:`blocks`.

        Fits each candidate's orientation from the surrounding reference blocks
        in :attr:`blocks` (the cleaned population).  Candidates never support
        each other — only real reference blocks are used.  Results are stored
        in :attr:`_added` tagged with origin 3 in :attr:`_origin_column`.
        Running again replaces the origin-3 portion.

        The input *motl* may be either the discarded particles (all rows are
        treated as candidates) or the full population (rows whose
        ``subtomo_id`` is **not** in :attr:`blocks` become candidates).

        When ``block_definition`` is a dict, each candidate's definition is
        resolved from its ``block_type_column`` value.  A candidate whose type
        has no definition is skipped and included in *unfitted*; a warning
        naming the missing types is printed.

        Parameters
        ----------
        motl : cryomotl.Motl
            Particles to recover.
        min_neighbours : int, default=2
            Minimum number of distinct reference-block neighbours a candidate
            needs for a valid fit.
        bonding_tolerance : float or None, default=None
            Half-width (voxels) of the acceptance window around the nominal
            centre-to-centre bonding distance ``2 × |site_shift|``.  Overrides
            :attr:`bonding_tolerance` for this call only.  ``None`` falls back
            to ``self.bonding_tolerance``; if that is also ``None``, derives
            to ``0.4 × |site_shift|``.
        angle_tolerance : float or None, default=None
            Maximum allowed angular residual (degrees) for the Cn pairwise
            spacing check.  Overrides :attr:`angle_tolerance` for this call
            only.  ``None`` falls back to ``self.angle_tolerance``; if that
            is also ``None``, derives to ``0.125 × (360 / n)``.  Raises
            ``ValueError`` when the resolved value is ``>= 0.5 × (360 / n)``.

        Returns
        -------
        motl : cryomotl.Motl
            One row per successfully fitted candidate.  Carries position,
            orientation, affiliation, block type, and origin.  Columns:
            ``tomo_id_column`` (``tomo_id``), ``affiliation_column``
            (``object_id``), ``x``, ``y``, ``z``, ``phi``, ``theta``,
            ``psi``, ``block_type_column`` (default ``class``),
            ``_origin_column`` (default ``geom5``, value 3).  All other
            columns are zero.
        diagnostics : pd.DataFrame
            One row per candidate (fitted and unfitted), keyed by
            ``subtomo_id``.  Columns:

            * ``subtomo_id``
            * ``status`` — ``"fitted"`` or ``"unfitted"``
            * ``n_neighbours`` — number of accepted reference neighbours
            * ``angular_rms`` — angular RMS of the Cn check (degrees);
              NaN for candidates that failed before the Cn check
            * ``normal_spread`` — RMS spread of the accepted neighbours'
              normals around their mean (degrees); NaN for unfitted
            * ``tip_plane_angle`` — angle (degrees) between the SVD-derived
              tip-plane normal and the neighbours' mean z-vector; NaN for
              unfitted candidates
            * ``reason`` — empty string for fitted candidates; for unfitted:
              ``"no_reference_blocks"``, ``"undefined_type"``,
              ``"insufficient_neighbours"``, or ``"cn_check_failed"``

        Raises
        ------
        ValueError
            If no block definition or block layer is present.
        """
        self._require_connect()
        if self.block_definition is None or self.blocks is None:
            raise ValueError("No block definition: call from_blocks() or set block_definition.")
        _check_column_collisions("recover_orientations", {
            "block_type_column": self.block_type_column,
            "source_block_count_column": self.source_block_count_column,
            "_origin_column": self._origin_column,
        })

        # ── Site geometry — per type ─────────────────────────────────────
        import re as _re_r

        _bd_items_ro: list[tuple[float, "BlockDefinition"]] = (
            [(float(k), v) for k, v in self.block_definition.items()]
            if isinstance(self.block_definition, dict)
            else [(1.0, self.block_definition)]
        )
        # {type_code: (sv, cc_lo, cc_hi, n, ang_step, ang_tol)}
        _type_geom_ro: dict[float, tuple] = {}
        for _tc_ro, _bd_ro in _bd_items_ro:
            _sv_ro: np.ndarray = _bd_ro.site_vectors()[0].astype(float)
            _sl_ro: float = float(np.linalg.norm(_sv_ro))
            if _sl_ro < 1e-15:
                raise ValueError(f"site_shift vector has zero length for block type {_tc_ro}.")
            _nom_ro: float = 2.0 * _sl_ro
            _bt_raw_ro = bonding_tolerance if bonding_tolerance is not None else self.bonding_tolerance
            _bt_ro: float = _bt_raw_ro if _bt_raw_ro is not None else 0.4 * _sl_ro
            _cc_lo_ro: float = _nom_ro - _bt_ro
            _cc_hi_ro: float = _nom_ro + _bt_ro
            _sm_ro = _re_r.match(r"C(\d+)", _bd_ro.symmetry or "")
            _n_ro: int | None = int(_sm_ro.group(1)) if _sm_ro else None
            _ang_step_ro: float = 360.0 / _n_ro if (_n_ro is not None and _n_ro > 1) else 0.0
            _ang_tol_raw_ro = angle_tolerance if angle_tolerance is not None else self.angle_tolerance
            if _n_ro is not None and _ang_step_ro > 0.0:
                _ang_tol_ro: float = _ang_tol_raw_ro if _ang_tol_raw_ro is not None else 0.125 * _ang_step_ro
                if _ang_tol_ro >= 0.5 * _ang_step_ro:
                    raise ValueError(
                        f"angle_tolerance {_ang_tol_ro:.3f}° >= 0.5 × angular step "
                        f"{_ang_step_ro:.3f}° (C{_n_ro}, block type {_tc_ro}); assignment is ambiguous."
                    )
            else:
                _ang_tol_ro = _ang_tol_raw_ro if _ang_tol_raw_ro is not None else 15.0
            _type_geom_ro[_tc_ro] = (_sv_ro, _cc_lo_ro, _cc_hi_ro, _n_ro, _ang_step_ro, _ang_tol_ro)

        _cc_hi_max: float = max(g[2] for g in _type_geom_ro.values())
        _is_dict_bd: bool = isinstance(self.block_definition, dict)

        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        # ── Reference block data ─────────────────────────────────────────
        _block_to_row: dict[float, int] = {
            float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(self.blocks.df))
        }
        _ref_pairs: set[tuple[float, float]] = set(
            zip(
                self.blocks.df[tomo_col].astype(float),
                self.blocks.df["subtomo_id"].astype(float),
            )
        )
        _angles_blocks = self.blocks.get_angles().astype(float)
        _all_R = srot.from_euler("zxz", _angles_blocks, degrees=True)
        _z_unit = np.array([0.0, 0.0, 1.0])

        # ── Resolve candidates ───────────────────────────────────────────
        _motl_pairs = list(
            zip(motl.df[tomo_col].astype(float), motl.df["subtomo_id"].astype(float))
        )
        _not_in_ref = pd.Series(
            [p not in _ref_pairs for p in _motl_pairs], index=motl.df.index
        )
        cand_df = motl.df[_not_in_ref].copy().reset_index(drop=True)
        if len(cand_df) == 0:
            # All (tomo_id, subtomo_id) pairs in motl are also in blocks → treat entire motl as candidates
            cand_df = motl.df.copy().reset_index(drop=True)

        # ── Reference block positions ────────────────────────────────────
        ref_coords = self.blocks.get_coordinates()  # (N_ref, 3)

        from collections import Counter as _Counter

        out_rows: list[dict] = []
        _fitted_diag: list[dict] = []
        unfitted_rows: list[dict] = []
        _undefined_types: set[float] = set()

        for cand_tomo in sorted(cand_df[tomo_col].unique()):
            cand_t = cand_df[cand_df[tomo_col] == cand_tomo].reset_index(drop=True)
            cand_coords_t = (cand_t[["x", "y", "z"]].values + cand_t[["shift_x", "shift_y", "shift_z"]].values).astype(float)

            # Reference blocks restricted to this tomogram — tomo separation is structural
            ref_tomo_mask = (self.blocks.df[tomo_col] == cand_tomo).values
            ref_tomo_idx = np.where(ref_tomo_mask)[0]

            if len(ref_tomo_idx) == 0:
                for _, crow in cand_t.iterrows():
                    unfitted_rows.append({"subtomo_id": float(crow["subtomo_id"]), "status": "unfitted", "n_neighbours": 0.0, "angular_rms": float("nan"), "normal_spread": float("nan"), "tip_plane_angle": float("nan"), "reason": "no_reference_blocks"})
                continue

            ref_tomo_coords = ref_coords[ref_tomo_idx]

            # Batch search: candidates against reference block centres within bonding range.
            b_qp, b_nn = nnana.find_nn_within_radius(cand_coords_t, ref_tomo_coords, _cc_hi_max)
            block_nn_map: dict[int, np.ndarray] = {int(qi): nns for qi, nns in zip(b_qp, b_nn)}

            for local_i, cand_row in cand_t.iterrows():
                C = cand_coords_t[local_i]
                _cand_sid = float(cand_row["subtomo_id"])

                # Resolve definition for this candidate's block type.
                if _is_dict_bd:
                    _cand_type = float(cand_row[self.block_type_column])
                    if _cand_type not in _type_geom_ro:
                        unfitted_rows.append({"subtomo_id": _cand_sid, "status": "unfitted", "n_neighbours": 0.0, "angular_rms": float("nan"), "normal_spread": float("nan"), "tip_plane_angle": float("nan"), "reason": "undefined_type"})
                        _undefined_types.add(_cand_type)
                        continue
                    _sv_cand, _cc_lo_cand, _cc_hi_cand, _n_cand, _ang_step_cand, _ang_tol_cand = _type_geom_ro[_cand_type]
                else:
                    _cand_type = 1.0
                    _sv_cand, _cc_lo_cand, _cc_hi_cand, _n_cand, _ang_step_cand, _ang_tol_cand = _type_geom_ro[1.0]

                nn_local = block_nn_map.get(local_i, np.array([], dtype=np.intp))

                # Plausibility filter: keep only bonding-distance neighbours for this type.
                if len(nn_local) > 0:
                    dists_cc = np.linalg.norm(ref_tomo_coords[nn_local] - C, axis=1)
                    ok_cc = (dists_cc >= _cc_lo_cand) & (dists_cc <= _cc_hi_cand)
                    accepted_local = nn_local[ok_cc]
                else:
                    accepted_local = np.array([], dtype=np.intp)

                accepted_global_idx = ref_tomo_idx[accepted_local]
                n_accepted = int(len(accepted_global_idx))

                if n_accepted < min_neighbours:
                    unfitted_rows.append({"subtomo_id": _cand_sid, "status": "unfitted", "n_neighbours": float(n_accepted), "angular_rms": float("nan"), "normal_spread": float("nan"), "tip_plane_angle": float("nan"), "reason": "insufficient_neighbours"})
                    continue

                aff_majority = float(
                    _Counter(
                        self.blocks.df.iloc[accepted_global_idx][aff_col].astype(float).values.tolist()
                    ).most_common(1)[0][0]
                )

                row_base: dict = {c: 0.0 for c in cryomotl.Motl.motl_columns}
                row_base["subtomo_id"] = _cand_sid
                row_base[self.block_type_column] = float(cand_row[self.block_type_column])
                row_base[tomo_col] = float(cand_tomo)
                row_base[aff_col] = aff_majority
                row_base["x"] = float(C[0])
                row_base["y"] = float(C[1])
                row_base["z"] = float(C[2])
                row_base[self.source_block_count_column] = float(n_accepted)

                # Vectors from candidate to accepted neighbour centres: the bond directions.
                V = ref_tomo_coords[accepted_local] - C
                accepted_bids = np.array([
                    float(self.blocks.df.iloc[k]["subtomo_id"]) for k in accepted_global_idx
                ])
                _diag: dict = {}
                _fit = self._fit_orientation_from_tips(
                    V, accepted_bids, _block_to_row, _all_R, _sv_cand,
                    _n_cand, _ang_step_cand, _ang_tol_cand, min_neighbours, _diag,
                )
                if _fit is None:
                    unfitted_rows.append({
                        "subtomo_id": _cand_sid,
                        "status": "unfitted",
                        "n_neighbours": float(n_accepted),
                        "angular_rms": float(_diag.get("ang_residual", float("nan"))),
                        "normal_spread": float("nan"),
                        "tip_plane_angle": float("nan"),
                        "reason": "cn_check_failed",
                    })
                    continue

                phi_c, theta_c, psi_c, angular_rms, tip_plane_angle = _fit

                nb_angles = self.blocks.get_angles()[accepted_global_idx].astype(float)
                nb_normals = geom.euler_angles_to_normals(nb_angles)
                mean_normal = nb_normals.mean(axis=0)
                _mn = float(np.linalg.norm(mean_normal))
                if _mn > 1e-15:
                    mean_normal /= _mn
                _dots = np.clip(np.abs(nb_normals @ mean_normal), 0.0, 1.0)
                normal_spread = float(np.sqrt(np.mean(np.degrees(np.arccos(_dots)) ** 2)))

                row_base["phi"] = phi_c
                row_base["theta"] = theta_c
                row_base["psi"] = psi_c
                row_base[self._origin_column] = 3.0
                out_rows.append(row_base)
                _fitted_diag.append({
                    "subtomo_id": _cand_sid,
                    "status": "fitted",
                    "n_neighbours": float(n_accepted),
                    "angular_rms": float(angular_rms),
                    "normal_spread": float(normal_spread),
                    "tip_plane_angle": float(tip_plane_angle),
                    "reason": "",
                })

        if _undefined_types:
            _type_list = ", ".join(str(int(t) if t == int(t) else t) for t in sorted(_undefined_types))
            print(f"recover_orientations: no definition for block type(s) {_type_list}; those candidates were not fitted.")

        _diag_cols = ["subtomo_id", "status", "n_neighbours", "angular_rms", "normal_spread", "tip_plane_angle", "reason"]
        _all_diag_rows = _fitted_diag + unfitted_rows
        diagnostics = (
            pd.DataFrame(_all_diag_rows, columns=_diag_cols)
            if _all_diag_rows
            else pd.DataFrame(columns=_diag_cols)
        )

        if not out_rows:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns)), diagnostics

        motl_out = cryomotl.Motl(pd.DataFrame(out_rows)[cryomotl.Motl.motl_columns])
        return motl_out, diagnostics

    def adopt_recovered(
        self,
        motl: "cryomotl.Motl",
        diagnostics: "pd.DataFrame | None" = None,
        *,
        min_neighbours: int | None = None,
        max_angular_rms: float | None = None,
        max_normal_dev: float | None = None,
    ) -> "cryomotl.Motl":
        """Adopt recovered particles into the original lattice.

        Merges the accepted subset of *motl* (from :meth:`recover_orientations`)
        into :attr:`blocks`, tags them with origin 1 in :attr:`_origin_column`,
        and rebuilds the contact graph so that ``scope='original'`` reflects
        the corrected baseline.  Existing blocks retain origin 1.

        Any inferred blocks (:attr:`_added`, origin 2) are cleared because
        they were computed from the pre-adoption baseline.
        ``"Inference state cleared."`` is printed if they existed.

        State after this call
        --------------------
        ``_site_table``, ``_faces``
            Rebuilt by the internal :meth:`connect` call on the corrected
            :attr:`blocks`.
        ``_added``
            ``None``.
        ``_all_state``
            ``None``; rebuilt by the next :meth:`infer_missing_blocks` call.

        Parameters
        ----------
        motl : cryomotl.Motl
            Output of :meth:`recover_orientations`.
        diagnostics : pd.DataFrame or None, default None
            Diagnostics DataFrame returned by :meth:`recover_orientations`.
            When provided, ``min_neighbours``, ``max_angular_rms``, and
            ``max_normal_dev`` are applied against its ``n_neighbours``,
            ``angular_rms``, and ``normal_spread`` columns respectively.
            When ``None``, those filters are not applied.
        min_neighbours : int or None, default None
            Keep rows whose ``n_neighbours`` (from *diagnostics*) is at
            least this value.  Requires *diagnostics*.
        max_angular_rms : float or None, default None
            Keep rows whose ``angular_rms`` (from *diagnostics*) is at
            most this value.  Requires *diagnostics*.
        max_normal_dev : float or None, default None
            Keep rows whose ``normal_spread`` (from *diagnostics*) is at
            most this value.  Requires *diagnostics*.

        Returns
        -------
        cryomotl.Motl
            The adopted subset, tagged with origin 1 in
            :attr:`_origin_column`.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called, or if any filter
            parameter is given without *diagnostics*.
        """
        self._require_connect()

        rec_df = motl.df.copy()

        if diagnostics is None:
            _missing = [
                name for name, val in (
                    ("min_neighbours", min_neighbours),
                    ("max_angular_rms", max_angular_rms),
                    ("max_normal_dev", max_normal_dev),
                ) if val is not None
            ]
            if _missing:
                raise ValueError(
                    f"adopt_recovered: filter(s) {_missing} require the diagnostics DataFrame "
                    f"from recover_orientations, but diagnostics=None."
                )

        if diagnostics is not None and len(diagnostics) > 0 and (
            min_neighbours is not None or max_angular_rms is not None or max_normal_dev is not None
        ):
            _fd = diagnostics[diagnostics["status"] == "fitted"][["subtomo_id", "n_neighbours", "angular_rms", "normal_spread"]]
            rec_df = rec_df.merge(_fd, on="subtomo_id", how="left")
            if min_neighbours is not None:
                rec_df = rec_df[rec_df["n_neighbours"].fillna(0.0) >= float(min_neighbours)]
            if max_angular_rms is not None:
                rec_df = rec_df[rec_df["angular_rms"].fillna(float("inf")) <= float(max_angular_rms)]
            if max_normal_dev is not None:
                rec_df = rec_df[rec_df["normal_spread"].fillna(float("inf")) <= float(max_normal_dev)]
            rec_df = rec_df[cryomotl.Motl.motl_columns]
        rec_df = rec_df.copy().reset_index(drop=True)

        real_df = self.blocks.df.copy()
        _oz = real_df[self._origin_column] == 0.0
        if _oz.any():
            real_df.loc[_oz, self._origin_column] = 1.0

        # Assign class to recovered particles.  Keep any non-zero class already
        # in the motl; for rows whose class is 0, use the nearest real block's class.
        _zero_mask = rec_df["class"] == 0.0
        if _zero_mask.any():
            from scipy.spatial import cKDTree as _cKDTree
            _real_xyz = (
                real_df[["x", "y", "z"]].values.astype(float)
                + real_df[["shift_x", "shift_y", "shift_z"]].values.astype(float)
            )
            _rec_xyz = (
                rec_df.loc[_zero_mask, ["x", "y", "z"]].values.astype(float)
                + rec_df.loc[_zero_mask, ["shift_x", "shift_y", "shift_z"]].values.astype(float)
            )
            _, _nn_idx = _cKDTree(_real_xyz).query(_rec_xyz, k=1)
            _nn_classes = real_df["class"].values.astype(float)[_nn_idx]
            _nn_classes[_nn_classes == 0.0] = 1.0
            rec_df.loc[_zero_mask, "class"] = _nn_classes

        rec_df[self._source_id_column] = rec_df["subtomo_id"].copy()
        rec_df[self._origin_column] = 1.0

        merged_df = pd.concat([real_df, rec_df], ignore_index=True)
        merged_df["subtomo_id"] = np.arange(1, len(merged_df) + 1, dtype=float)
        self.blocks = cryomotl.Motl(merged_df[cryomotl.Motl.motl_columns])

        self.connect(self._max_distance)

        if self._added is not None and len(self._added.df) > 0:
            print("Inference state cleared.")
        self._added = None
        # _all_state was already cleared by connect() above.

        adopted = cryomotl.Motl(rec_df[cryomotl.Motl.motl_columns])
        return adopted

    @gui_exposed(
        label="Envelope from faces", group="Lattice envelope", order=10, returns="surface", category="pleomorphic-op"
    )
    def envelope_from_faces(
        self,
        assembly_id: int | None = None,
        tomo_id: float | None = None,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> "dict[tuple[float, float], Mesh]":
        """Build one triangulated mesh envelope per (tomo_id, assembly_id) object.

        Each object's mesh is built from its face polygon network: block
        positions become vertices, face centroids become hub vertices, and
        each face is fan-triangulated from its centroid.  Normals are computed
        on the returned meshes.

        The result is also stored as :attr:`_envelopes` so that
        :meth:`get_shape_stats` can use it without rebuilding.

        Parameters
        ----------
        assembly_id : int or None, default=None
            Restrict to this assembly.  ``None`` processes every assembly.
        tomo_id : float or None, default=None
            Restrict to this tomogram.  ``None`` processes every tomogram.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            Block pool to use (see :meth:`_blocks_for_scope`).

        Returns
        -------
        dict of {(tomo_id, assembly_id): Mesh}
            One mesh per object.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called, or no faces remain after
            filtering.
        """
        self._require_connect(scope)
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column
        _blks = self._blocks_for_scope(scope)
        face_stats = self.get_face_stats(scope=scope)

        if assembly_id is not None:
            face_stats = face_stats[face_stats["assembly_id"] == float(assembly_id)]
        if tomo_id is not None:
            face_stats = face_stats[face_stats[tomo_col] == tomo_id]
        face_stats = face_stats.reset_index(drop=True)

        if len(face_stats) == 0:
            raise ValueError("No faces to build envelope from (after filtering).")

        block_to_row: dict[float, int] = {
            float(_blks.df.iloc[i]["subtomo_id"]): i for i in range(len(_blks.df))
        }
        _blks_coords = _blks.get_coordinates()  # (N, 3) shifted, voxels

        result: dict[tuple[float, float], Mesh] = {}

        for (tval, aval), grp in face_stats.groupby([tomo_col, aff_col]):
            grp = grp.reset_index(drop=True)

            all_block_ids_ordered: list[float] = []
            seen: set[float] = set()
            for block_list in grp["block_ids"]:
                for bid in block_list:
                    if bid not in seen:
                        seen.add(bid)
                        all_block_ids_ordered.append(bid)

            block_id_to_vx: dict[float, int] = {bid: i for i, bid in enumerate(all_block_ids_ordered)}
            n_blocks = len(all_block_ids_ordered)

            block_verts = np.array(
                [
                    _blks_coords[block_to_row[bid]] * self._block_pixel_size
                    for bid in all_block_ids_ordered
                ]
            )

            face_centroid_verts = (
                grp[["centroid_x", "centroid_y", "centroid_z"]].values * self._block_pixel_size
            )
            vertices = np.vstack([block_verts, face_centroid_verts])

            triangles: list[list[int]] = []
            for fi, frow in grp.iterrows():
                block_ids_walk: list[float] = frow["block_ids"]
                m_idx = n_blocks + int(fi)
                n_walk = len(block_ids_walk)
                face_normal = np.array([frow["normal_x"], frow["normal_y"], frow["normal_z"]])
                fn_norm = float(np.linalg.norm(face_normal))
                if fn_norm > 1e-15:
                    face_normal = face_normal / fn_norm

                b0 = block_id_to_vx[block_ids_walk[0]]
                b1 = block_id_to_vx[block_ids_walk[1]]
                v_m = vertices[m_idx]
                candidate_n = np.cross(vertices[b0] - v_m, vertices[b1] - v_m)
                reverse = np.dot(candidate_n, face_normal) < 0

                for i in range(n_walk):
                    bi = block_id_to_vx[block_ids_walk[i]]
                    bj = block_id_to_vx[block_ids_walk[(i + 1) % n_walk]]
                    if reverse:
                        triangles.append([m_idx, bj, bi])
                    else:
                        triangles.append([m_idx, bi, bj])

            faces = np.array(triangles, dtype=np.int32)
            mesh = Mesh()
            mesh.vertices = vertices
            mesh.faces = faces
            mesh.compute_normals()
            mesh.pixel_size = self._block_pixel_size
            result[(float(tval), float(aval))] = mesh

        self._envelopes = result
        self._envelope_scope = scope
        return result

    def envelope_from_blocks(
        self,
        radius: float | None = None,
        assembly_id: int | None = None,
        tomo_id: float | None = None,
        min_blocks: int = 4,
        scope: Literal["original", "inferred", "all"] = "original",
    ) -> "dict[tuple[float, float], Mesh]":
        """Build one triangulated mesh envelope per object using block positions.

        A ball-pivoting reconstruction is applied to the block-centre point
        cloud of each object.  Block orientations are used directly as normals;
        no normal re-estimation is performed.  When *radius* is ``None``, the
        pivot radius is estimated as the mean nearest-neighbour distance among
        the blocks of each object (via
        :meth:`~cryocat.core.surface.Mesh.from_ball_pivoting`).

        Objects with fewer than *min_blocks* blocks, objects whose meshing
        raises, and meshes with fewer than 4 faces are skipped with a warning.
        A :exc:`ValueError` is raised only when no object could be meshed.

        The result is also stored as :attr:`_envelopes`.

        Parameters
        ----------
        radius : float or None, default=None
            Ball-pivoting radius.  ``None`` auto-estimates per object.
        assembly_id : int or None, default=None
            Restrict to this assembly.  ``None`` processes every assembly.
        tomo_id : float or None, default=None
            Restrict to this tomogram.  ``None`` processes every tomogram.
        min_blocks : int, default=4
            Minimum number of blocks to attempt meshing an object.
        scope : {``'original'``, ``'inferred'``, ``'all'``}, default=``'original'``
            Block pool to use (see :meth:`_blocks_for_scope`).

        Returns
        -------
        dict of {(tomo_id, assembly_id): Mesh}
            One mesh per object.

        Raises
        ------
        ImportError
            If ``open3d`` is not available (required by
            :meth:`~cryocat.core.surface.Mesh.from_ball_pivoting`).
        ValueError
            If no blocks remain after filtering, or no object could be meshed.
        """
        tomo_col = self.tomo_id_column
        aff_col = self.affiliation_column

        _blks = self._blocks_for_scope(scope)
        mask = np.ones(len(_blks.df), dtype=bool)
        if tomo_id is not None:
            mask &= (_blks.df[tomo_col] == tomo_id).values
        if assembly_id is not None:
            mask &= (_blks.df[aff_col] == float(assembly_id)).values

        if not mask.any():
            raise ValueError("No blocks remain after filtering.")

        filtered_motl = cryomotl.Motl(_blks.df[mask].copy().reset_index(drop=True))

        result: dict[tuple[float, float], Mesh] = {}
        skipped: list[tuple[float, float]] = []
        n_tried = 0

        for tval_f in sorted(filtered_motl.df[tomo_col].unique()):
            tomo_df = filtered_motl.df[filtered_motl.df[tomo_col] == tval_f]
            tomo_motl = cryomotl.Motl(tomo_df.copy().reset_index(drop=True))

            # from_motl returns a dict keyed by aff_col when >1 unique value, bare OPC otherwise
            opc_raw = OrientedPointCloud.from_motl(tomo_motl, group_by=aff_col)
            if isinstance(opc_raw, OrientedPointCloud):
                unique_aff = tomo_motl.df[aff_col].unique()
                opc_by_aff: dict[float, OrientedPointCloud] = {float(unique_aff[0]): opc_raw}
            else:
                opc_by_aff = {float(k): v for k, v in opc_raw.items()}

            for aff_val_f, opc in opc_by_aff.items():
                n_tried += 1
                # Key from loop variables, not from a row lookup
                if len(opc.vertices) < min_blocks:
                    skipped.append((tval_f, aff_val_f))
                    continue

                import copy as _copy
                s = self._block_pixel_size
                opc_scaled = _copy.deepcopy(opc)
                opc_scaled.transform(np.diag([s, s, s, 1.0]))
                opc_scaled.pixel_size = s

                radii_arg = [float(radius), float(radius) * 2.0] if radius is not None else None
                try:
                    mesh = Mesh.from_ball_pivoting(opc_scaled, radii=radii_arg)
                except Exception:
                    skipped.append((tval_f, aff_val_f))
                    continue

                if mesh.faces is None or len(mesh.faces) < 4:
                    skipped.append((tval_f, aff_val_f))
                    continue

                result[(tval_f, aff_val_f)] = mesh

        if skipped:
            import warnings as _warnings
            _warnings.warn(
                f"envelope_from_blocks: {len(skipped)} of {n_tried} object(s) skipped "
                f"(< {min_blocks} blocks, meshing failed, or < 4 faces): {skipped}",
                UserWarning,
                stacklevel=2,
            )

        if not result:
            raise ValueError(
                f"envelope_from_blocks: no object could be meshed "
                f"({n_tried} tried, all skipped)."
            )

        self._envelopes = result
        self._envelope_scope = scope
        return result

    def get_shape_stats(self) -> pd.DataFrame:
        """Return shape statistics for each object envelope in :attr:`_envelopes`.

        Call :meth:`envelope_from_faces` or :meth:`envelope_from_blocks` first
        to populate :attr:`_envelopes`.  Curvatures are computed internally on
        each mesh.

        Returns
        -------
        pandas.DataFrame
            One row per object.  Columns:

            * ``tomo_id``
            * ``assembly_id``
            * ``scope`` — block pool used when the envelopes were built
            * ``radius`` — ``1 / median(|H|)`` where *H* is the mean curvature
              at each vertex; equivalent to the characteristic curvature radius
            * ``sphere_fraction`` — ``Σ(K_i · area_i) / (4π)`` where *K_i* is
              the Gaussian curvature and *area_i* the Voronoi area at vertex *i*;
              equals 1 for a topological sphere, < 1 for partial coverage
            * ``area`` — total mesh surface area
            * ``b_a`` — ratio of the second-largest to largest semi-axis of the
              fitted ellipsoid (range 0–1; 1 = oblate/prolate degenerate, near 1
              = near-sphere)
            * ``c_a`` — ratio of the smallest to largest semi-axis (range 0–1)
            * ``long_axis_x/y/z`` — unit vector along the longest ellipsoid axis
            * ``short_axis_x/y/z`` — unit vector along the shortest ellipsoid axis
            * ``fit_residual`` — RMS spread of per-vertex curvature radii around
              the median radius

        Raises
        ------
        ValueError
            If :attr:`_envelopes` is ``None`` (no envelope has been built).
        """
        if self._envelopes is None:
            raise ValueError(
                "No envelopes found.  Call envelope_from_faces() or "
                "envelope_from_blocks() first."
            )

        rows: list[dict] = []
        for (tval, aval), mesh in self._envelopes.items():
            # Curvatures (also populates _face_areas and _vertex_areas)
            mesh.compute_curvatures()

            # Radius: 1 / median(|H|) where H is mean curvature
            abs_H = np.abs(mesh._mean_curvature)
            median_abs_H = float(np.median(abs_H))
            radius = 1.0 / median_abs_H if median_abs_H > 1e-15 else float("nan")

            # Sphere fraction: area-weighted integral of Gaussian curvature / 4π
            sphere_fraction = float(
                np.sum(mesh._gaussian_curvature * mesh._vertex_areas) / (4.0 * np.pi)
            )

            # Total mesh surface area from face areas
            area = float(np.sum(mesh._face_areas))

            # Axis ratios and directions from geometric ellipsoid fit
            _nan3 = [float("nan"), float("nan"), float("nan")]
            try:
                _, radii_fit, evecs_fit, _ = geom.fit_ellipsoid(mesh.vertices)
                idx_sorted = np.argsort(np.abs(radii_fit))[::-1]  # descending a >= b >= c
                sorted_r = np.abs(radii_fit)[idx_sorted]
                a_ax, b_ax, c_ax = float(sorted_r[0]), float(sorted_r[1]), float(sorted_r[2])
                b_a = b_ax / a_ax if a_ax > 1e-15 else float("nan")
                c_a = c_ax / a_ax if a_ax > 1e-15 else float("nan")
                long_ax = evecs_fit[idx_sorted[0]].tolist()   # eigenvector of largest radius
                short_ax = evecs_fit[idx_sorted[2]].tolist()  # eigenvector of smallest radius
            except Exception:
                b_a = float("nan")
                c_a = float("nan")
                long_ax = _nan3
                short_ax = _nan3

            # fit_residual: RMS spread of per-vertex curvature radii around median
            safe_H = np.where(abs_H > 1e-15, abs_H, 1e-15)
            local_radii = 1.0 / safe_H
            fit_residual = float(np.sqrt(np.mean((local_radii - radius) ** 2))) if not np.isnan(radius) else float("nan")

            rows.append({
                self.tomo_id_column: tval,
                "assembly_id": aval,
                "scope": self._envelope_scope,
                "radius": radius,
                "sphere_fraction": sphere_fraction,
                "area": area,
                "b_a": b_a,
                "c_a": c_a,
                "long_axis_x": long_ax[0],
                "long_axis_y": long_ax[1],
                "long_axis_z": long_ax[2],
                "short_axis_x": short_ax[0],
                "short_axis_y": short_ax[1],
                "short_axis_z": short_ax[2],
                "fit_residual": fit_residual,
            })

        return pd.DataFrame(rows)

    def annotate_with_envelope(
        self,
        tomo_id: float | None = None,
    ) -> pd.DataFrame:
        """Annotate blocks with their distance to the envelope surface.

        Requires both an envelope surface (``self.surface``) and a block layer
        (``self.blocks``).  The envelope is never built internally; the caller
        is responsible for attaching one, e.g.::

            ps_clean = PleomorphicSurface(blocks=motl, block_definition=bd, ...)
            ps_clean.connect(max_distance=4.0)
            mesh = ps_clean.envelope_from_faces()
            ps = PleomorphicSurface(mesh, blocks=motl, block_definition=bd, ...)
            annot = ps.annotate_with_envelope()

        Do not pass the envelope back to the same instance that produced it;
        ``envelope_from_faces`` uses block centres as vertices, so the
        resulting mesh is self-referentially tied to those same blocks and
        cannot give a meaningful distance query.

        Parameters
        ----------
        tomo_id : float or None, default=None
            Restrict blocks to this tomogram.  The surface itself is never
            filtered; the full envelope is queried for the selected blocks.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``block_id``, ``envelope_distance``,
            ``closest_x``, ``closest_y``, ``closest_z``, ``primitive_id``,
            ``normal_angle``, ``mean_curvature``, ``gaussian_curvature``.

        Raises
        ------
        ValueError
            If no block layer or no envelope surface is present.
        """
        if self.blocks is None:
            raise ValueError("annotate_with_envelope requires a block layer (blocks=...).")
        tomo_col = self.tomo_id_column
        envelope = self.surface  # raises ValueError if _surface is None

        if tomo_id is not None:
            mask = self.blocks.df[tomo_col] == tomo_id
            blocks_df = self.blocks.df[mask].reset_index(drop=True)
            angles = self.blocks.get_angles()[mask.values].astype(float)
            c = self.blocks.get_coordinates()[mask.values]
        else:
            blocks_df = self.blocks.df.reset_index(drop=True)
            angles = self.blocks.get_angles().astype(float)
            c = self.blocks.get_coordinates()
        q = c * self._block_pixel_size
        all_R = srot.from_euler("zxz", angles, degrees=True)
        z_unit = np.array([0.0, 0.0, 1.0])
        block_normals = np.array([R.apply(z_unit) for R in all_R])

        result = envelope.distance_to_points(
            q,
            compute_occupancy=False,
            compute_signed=False,
            return_closest_points=True,
        )

        distances = np.asarray(result["distances"])
        closest_pts = np.asarray(result["closest_points"])
        prim_ids = np.asarray(result["primitive_ids"], dtype=int)

        tri_verts = envelope.faces[prim_ids]
        vert_normals = envelope.normals
        face_normals_at_blocks = vert_normals[tri_verts].mean(axis=1)
        fn_norm = np.linalg.norm(face_normals_at_blocks, axis=1, keepdims=True)
        face_normals_at_blocks = face_normals_at_blocks / np.where(fn_norm > 1e-15, fn_norm, 1.0)

        dots = np.clip(np.sum(block_normals * face_normals_at_blocks, axis=1), -1.0, 1.0)
        normal_angles = np.degrees(np.arccos(dots))

        try:
            env_ps = PleomorphicSurface(envelope)
            curv_table = env_ps._mesh_triangle_curvature_table()
            mean_curv = curv_table["mean_curvature"][prim_ids]
            gauss_curv = curv_table["gaussian_curvature"][prim_ids]
        except Exception:
            mean_curv = np.full(len(q), float("nan"))
            gauss_curv = np.full(len(q), float("nan"))

        return pd.DataFrame(
            {
                tomo_col: blocks_df[tomo_col].values,
                "block_id": blocks_df["subtomo_id"].values,
                "envelope_distance": distances,
                "closest_x": closest_pts[:, 0],
                "closest_y": closest_pts[:, 1],
                "closest_z": closest_pts[:, 2],
                "primitive_id": prim_ids,
                "normal_angle": normal_angles,
                "mean_curvature": mean_curv,
                "gaussian_curvature": gauss_curv,
            }
        )

    @classmethod
    def read(
        cls,
        input_path: PathOrStr,
        method: Literal[
            "mesh",
            "mesh_curvatures",
            "mesh_from_mrc",
            "point_cloud",
            "point_cloud_from_mrc",
            "point_cloud_from_motl",
        ] = "mesh",
        **kwargs: Any,
    ) -> "PleomorphicSurface":
        """
        Create a wrapped surface from common on-disk inputs.

        Parameters
        ----------
        input_path : str or Path
            Input file path.
        method : {'mesh', 'mesh_curvatures', 'mesh_from_mrc', 'point_cloud', \
'point_cloud_from_mrc', 'point_cloud_from_motl'}, default='mesh'
            Loader to use:
            - "mesh": geometry-only triangle mesh via :meth:`Mesh.read`
            - "mesh_curvatures": VTP triangle mesh with curvature fields via
              :meth:`Mesh.read_curvatures`
            - "mesh_from_mrc": segmentation-to-mesh via :meth:`Mesh.from_mrc`
            - "point_cloud": oriented point cloud via :meth:`OrientedPointCloud.read`
            - "point_cloud_from_mrc": segmentation-to-point-cloud via
              :meth:`OrientedPointCloud.from_mrc`
            - "point_cloud_from_motl": oriented point cloud from a motl file via
              :meth:`OrientedPointCloud.from_motl`. Pass ``group_by`` (e.g.
              ``'object_id'`` or ``'subtomo_id'``) to split the motl into one wrapped
              surface per unique value; returns a ``dict`` of ``PleomorphicSurface``
              in that case, otherwise a single ``PleomorphicSurface``.
        **kwargs
            Forwarded to the selected loader. Accepted keywords depend on ``method``:

            *"mesh"* and *"mesh_curvatures"*:

            - ``units`` : str, optional — coordinate units (``'nm'``, ``'angstrom'``, ``'pixel'``, …).

            *"mesh_from_mrc"*:

            - ``transpose`` : bool, default=True — transpose the segmentation array on load.
            - ``labels_dict`` : dict, optional — map label names to integer values; binary if None.
            - ``level`` : float, default=0.5 — marching-cubes iso-level.
            - ``pixel_size`` : float, default=1.0 — voxel size for coordinate scaling.
            - ``smooth_sigma`` : float, optional — Gaussian pre-smooth sigma.
            - ``step_size`` : int, default=1 — marching-cubes step size.

            *"point_cloud"*:

            - ``recompute_normals`` : bool, default=False — recompute even if file has normals.
            - ``knn`` : int, default=30 — neighbors for normal estimation.
            - ``orient_normals`` : bool, default=True — orient normals consistently.
            - ``tangent_plane_knn`` : int, default=50 — neighbors for normal orientation.

            *"point_cloud_from_mrc"*:

            - ``labels_dict`` : dict, optional — map label names to integer values; binary if None.
            - ``pixel_size`` : float or array-like, optional — voxel size.
            - ``compute_normals`` : bool, default=True — estimate normals after extraction.
            - ``knn`` : int, default=30 — neighbors for normal estimation.
            - ``orient_normals`` : bool, default=True — orient normals consistently.
            - ``tangent_plane_knn`` : int, default=50 — neighbors for normal orientation.
            - ``transpose`` : bool, default=True — transpose the segmentation array on load.
            - ``smooth_sigma`` : float, optional — Gaussian pre-smooth sigma.

            *"point_cloud_from_motl"*:

            - ``group_by`` : MotlColumn, optional — motl column to split on (e.g.
              ``'object_id'``, ``'subtomo_id'``). If given and >1 unique value exists,
              returns a dict of wrapped surfaces keyed by that column's value.
            - ``recompute_normals`` : bool, default=False — recompute normals from geometry.
            - ``knn`` : int, default=30 — neighbors for normal estimation.
            - ``orient_normals`` : bool, default=True — orient normals consistently.
            - ``tangent_plane_knn`` : int, default=50 — neighbors for normal orientation.

        Returns
        -------
        PleomorphicSurface or dict[Any, PleomorphicSurface]
            Wrapped surface loaded from ``input_path``. For
            ``method="point_cloud_from_motl"`` with ``group_by`` splitting into
            multiple groups, a dict of wrapped surfaces keyed by group value.
        """
        method = str(method).lower()
        aliases = {
            "curvatures": "mesh_curvatures",
            "mesh_with_curvatures": "mesh_curvatures",
            "mrc_mesh": "mesh_from_mrc",
            "pcd": "point_cloud",
            "pointcloud": "point_cloud",
            "mrc_point_cloud": "point_cloud_from_mrc",
            "mrc_pointcloud": "point_cloud_from_mrc",
            "motl": "point_cloud_from_motl",
            "motl_point_cloud": "point_cloud_from_motl",
            "point_cloud_motl": "point_cloud_from_motl",
        }
        method = aliases.get(method, method)

        if method == "mesh":
            surface = Mesh.read(input_path, **kwargs)
        elif method == "mesh_curvatures":
            surface = Mesh.read_curvatures(input_path, **kwargs)
        elif method == "mesh_from_mrc":
            surface = Mesh.from_mrc(input_path, **kwargs)
        elif method == "point_cloud":
            surface = OrientedPointCloud.read(input_path, **kwargs)
        elif method == "point_cloud_from_mrc":
            surface = OrientedPointCloud.from_mrc(input_path, **kwargs)
        elif method == "point_cloud_from_motl":
            surface = OrientedPointCloud.from_motl(input_path, **kwargs)
            # from_motl returns a dict when splitting by group_by into >1 group.
            if isinstance(surface, dict):
                return {gid: cls(pcd) for gid, pcd in surface.items()}
        else:
            from cryocat.utils.exceptions import UserInputError

            raise UserInputError(
                f"Unknown read method '{method}'. Valid values: 'mesh', 'mesh_curvatures', "
                "'mesh_from_mrc', 'point_cloud', 'point_cloud_from_mrc', "
                "'point_cloud_from_motl'."
            )

        return cls(surface)

    @property
    def is_mesh(self) -> bool:
        """True when the backing geometry has triangle connectivity (:class:`Mesh`)."""
        return isinstance(self._surface, Mesh)

    @property
    def is_point_cloud(self) -> bool:
        """True when the backing geometry is discrete samples (:class:`OrientedPointCloud`)."""
        return isinstance(self._surface, OrientedPointCloud)

    @property
    def vertices(self):
        """DiscreteSurface vertices / points."""
        return self.surface.get_vertices()

    @property
    def normals(self):
        """DiscreteSurface normals."""
        return self.surface.get_normals()

    @property
    def faces(self):
        """Triangle connectivity for mesh-backed surfaces."""
        if not isinstance(self.surface, Mesh):
            raise TypeError("faces are only available for Mesh-backed PleomorphicSurface")
        return self.surface.faces

    @property
    def units(self):
        """Coordinate units stored on the wrapped surface."""
        return self.surface.units

    @units.setter
    def units(self, value):
        """Set coordinate units on the wrapped mesh or oriented point cloud."""
        self.surface.units = value

    def get_principal_curvatures(self) -> np.ndarray:
        """Return per-vertex principal curvatures for a mesh-backed surface.

        Returns
        -------
        np.ndarray, shape (N, 2)
            Columns are the two principal curvature values k1 and k2 at each vertex.
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_principal_curvatures()

    def get_mean_curvature(self) -> np.ndarray:
        """Return per-vertex mean curvature for a mesh-backed surface.

        Returns
        -------
        np.ndarray, shape (N,)
            Mean curvature H = (k1 + k2) / 2 at each vertex.
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_mean_curvature()

    def get_gaussian_curvature(self) -> np.ndarray:
        """Return per-vertex Gaussian curvature for a mesh-backed surface.

        Returns
        -------
        np.ndarray, shape (N,)
            Gaussian curvature K = k1 * k2 at each vertex.
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_gaussian_curvature()

    def get_curvature_directions(self) -> np.ndarray:
        """Return per-vertex principal curvature direction vectors for a mesh-backed surface.

        Returns
        -------
        np.ndarray, shape (N, 3, 2)
            Direction vectors at each vertex: ``[:, :, 0]`` is the first principal direction
            (k1), ``[:, :, 1]`` is the second (k2).
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_curvature_directions()

    def get_shape_index(self) -> np.ndarray:
        """Return per-vertex shape index for a mesh-backed surface.

        Returns
        -------
        np.ndarray, shape (N,)
            Shape index S = (2/pi) * arctan2(k1 + k2, k1 - k2) at each vertex, in [-1, 1].
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_shape_index()

    def get_curvedness(self) -> np.ndarray:
        """Return per-vertex curvedness for a mesh-backed surface.

        Returns
        -------
        np.ndarray, shape (N,)
            Curvedness C = sqrt((k1^2 + k2^2) / 2) at each vertex, in [0, inf).
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_curvedness()

    def get_surface_type(self, as_labels: bool = False) -> np.ndarray:
        """Return per-vertex categorical surface type for a mesh-backed surface.

        Parameters
        ----------
        as_labels : bool, default=False
            If True, return string labels (e.g. ``"cap"``); otherwise integer
            category codes (-1 flat, 0 cup .. 8 cap).

        Returns
        -------
        np.ndarray, shape (N,)
            Surface type per vertex.
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Curvatures are only available for Mesh-backed PleomorphicSurface")
        return self.surface.get_surface_type(as_labels=as_labels)

    def get_surface_area(self) -> float:
        """Return total surface area of a mesh-backed surface."""
        if not isinstance(self.surface, Mesh):
            raise TypeError(
                "get_surface_area is only available for Mesh-backed PleomorphicSurface. "
                "An OrientedPointCloud has no face connectivity from which to compute area."
            )
        return self.surface.get_surface_area()

    def save(
        self, output_path: PathOrStr, format: Literal["ply", "vtp", "motl", "em"] | None = None, **kwargs: Any
    ) -> None:
        """
        Save the wrapped surface.

        If ``format`` is None, the wrapped surface may infer it from ``output_path``.
        Additional keyword arguments are forwarded to the concrete surface save method.

        Parameters
        ----------
        output_path : PathOrStr
            Output file path.
        format : {'ply', 'vtp', 'motl', 'em'}, optional
            File format.  ``'ply'`` and ``'vtp'`` are Mesh formats; ``'motl'``/``'em'``
            are OrientedPointCloud formats.  If None, inferred from the file suffix.
        **kwargs
            Forwarded to the concrete save method. Accepted keywords depend on ``format``:

            *Mesh* (``format='ply'`` or ``'vtp'``):

            - ``include_curvatures`` : bool, default=False — embed per-vertex curvature scalars
              and principal-direction vectors; requires curvatures to have been computed and
              ``format='vtp'``.

            *OrientedPointCloud — PLY* (``format='ply'``):

            - ``write_ascii`` : bool, default=False — write ASCII rather than binary PLY.

            *OrientedPointCloud — MOTL / EM* (``format='motl'`` or ``'em'``):

            - ``input_dict`` : dict, optional — extra motive-list columns to fill.
            - ``subtomo_ids`` : array-like, shape (N,), optional — per-point subtomogram IDs;
              sequential IDs are assigned when None.
            - ``tomo_id`` : int, float, or array-like, optional — tomogram ID; scalar applies to
              all points, array assigns per-point IDs.

        Returns
        -------
        None
        """
        return self.surface.save(output_path, format=format, **kwargs)

    def compute_normals(self, **kwargs: Any) -> "PleomorphicSurface":
        """
        Delegates to :meth:`Mesh.compute_normals` or :meth:`OrientedPointCloud.compute_normals`.

        Parameters
        ----------
        **kwargs
            *Mesh*: no keywords are used; unknown keys are silently consumed.

            *OrientedPointCloud*:

            - ``knn`` : int, default=30 — neighbors for normal estimation.
            - ``orient_normals`` : bool, default=True — orient normals consistently.
            - ``tangent_plane_knn`` : int, default=50 — neighbors for normal orientation.
            - ``inplace`` : bool, default=True — update in place; if False, wrap and return a
              new :class:`PleomorphicSurface`.

        Returns
        -------
        PleomorphicSurface
            ``self`` when the delegate updates in place; a new wrapper when the delegate returns
            a copy (point cloud with ``inplace=False``).
        """
        out = self.surface.compute_normals(**kwargs)
        if out is None:
            return self
        return PleomorphicSurface(out)

    def flip_normals(self, inplace: bool = True, **kwargs: Any) -> "PleomorphicSurface" | None:
        """
        Delegate normal-direction flipping to the wrapped surface.

        Parameters
        ----------
        inplace : bool, default=True
            If True, modify the wrapped surface in place and return ``self``.
            If False, return a new :class:`PleomorphicSurface` wrapping a flipped copy.
        **kwargs
            *Mesh* only:

            - ``flip_faces`` : bool, default=True — also reverse triangle winding so that
              normals recomputed from faces keep the flipped orientation.

            *OrientedPointCloud*: no extra keywords are accepted; passing any raises
            :exc:`TypeError`.

        Returns
        -------
        PleomorphicSurface or None
            ``self`` when ``inplace=True``; a new wrapper when ``inplace=False``.
        """
        out = self.surface.flip_normals(inplace=inplace, **kwargs)
        if inplace:
            return self
        return PleomorphicSurface(out)

    def refine_normals(
        self,
        radius_hit: float = 3.0,
        batch_size: int = 2000,
        n_iter: int = 1,
        mask: np.ndarray | None = None,
        logger: logging.Logger | None = None,
        inplace: bool = True,
        **kwargs: Any,
    ) -> "PleomorphicSurface":
        """
        Refine normals on the wrapped surface by neighborhood averaging.

        Delegates to :meth:`Mesh.refine_normals` or :meth:`OrientedPointCloud.refine_normals`
        (both inherit :meth:`DiscreteSurface.refine_normals`).

        Parameters
        ----------
        radius_hit : float, default=3.0
            Neighborhood radius for normal averaging, in mesh/point-cloud units.
        batch_size : int, default=2000
            Batch size for spatial neighbor queries.
        n_iter : int, default=1
            Number of refinement passes.
        mask : np.ndarray, optional
            Boolean mask of vertices/samples to update. If None, all are refined.
        logger : logging.Logger, optional
            Logger passed through to the delegate.
        inplace : bool, default=True
            If True, update the wrapped surface in place. If False, return a new wrapper.
        **kwargs
            Additional keyword arguments forwarded to the delegate.

        Returns
        -------
        PleomorphicSurface
            ``self`` when ``inplace=True``; a new wrapper when ``inplace=False``.
        """
        if not (self.is_mesh or self.is_point_cloud):
            raise TypeError(
                f"Unsupported surface type: {type(self.surface)}. "
                "refine_normals requires a Mesh or OrientedPointCloud backing."
            )
        out = self.surface.refine_normals(
            radius_hit=radius_hit,
            batch_size=batch_size,
            n_iter=n_iter,
            mask=mask,
            logger=logger,
            inplace=inplace,
            **kwargs,
        )
        if inplace:
            return self
        return PleomorphicSurface(out)

    def remove_nonfinite_vertices(self, inplace: bool = True, **kwargs: Any) -> "PleomorphicSurface":
        """
        Remove NaN/Inf vertices or point samples from the wrapped surface.

        For meshes, affected faces are also dropped and vertex connectivity is remapped.

        Parameters
        ----------
        inplace : bool, default=True
            If True, modify the wrapped surface in place and return ``self``.
            If False, return a new :class:`PleomorphicSurface` wrapping a repaired copy.
        **kwargs
            - ``recompute_normals`` : bool — recompute normals after filtering.
              Default is ``True`` for :class:`Mesh`, ``False`` for
              :class:`OrientedPointCloud`.

        Returns
        -------
        PleomorphicSurface
            ``self`` when ``inplace=True``; a new wrapper around the repaired surface when
            ``inplace=False``.
        """
        out = self.surface.remove_nonfinite_vertices(inplace=inplace, **kwargs)
        if inplace:
            return self
        return PleomorphicSurface(out)

    def oversample(self, **kwargs: Any) -> "PleomorphicSurface":
        """
        Delegate to ``oversample`` on :attr:`surface`; mesh and point-cloud semantics differ.

        Parameters
        ----------
        **kwargs
            *Mesh* (:meth:`Mesh.oversample`):

            - ``oversample_factor`` : float, optional — desired factor increase in vertices.
              Defaults to 1.0 (no change) when both this and ``point_spacing`` are None.
            - ``point_spacing`` : float, optional — desired spacing between sampled points
              (same units as mesh coordinates). Uses two-pass Poisson-disk calibration.
            - ``poisson_init_factor`` : int, default=5 — initial candidate factor for
              Poisson-disk sampling (larger → more uniform distribution).

            *OrientedPointCloud* (:meth:`OrientedPointCloud.oversample`):

            - ``oversample_factor`` : float, optional — desired factor increase in points.
              Defaults to 1.0 when both this and ``point_spacing`` are None.
            - ``point_spacing`` : float, optional — desired spacing; uses greedy Poisson-disk
              sampling to enforce spacing directly.
            - ``random_seed`` : int, optional — seed for reproducible sampling.

        Returns
        -------
        PleomorphicSurface
            New wrapper around the resampled surface.
        """
        return PleomorphicSurface(self.surface.oversample(**kwargs))

    def crop(self, bbox: Any, inplace: bool = False) -> "PleomorphicSurface | None":
        """
        Delegate to :meth:`Mesh.crop` / :meth:`OrientedPointCloud.crop`.

        Parameters
        ----------
        bbox : open3d.geometry.AxisAlignedBoundingBox or dict
            Bounding box for cropping. When a dict, must have ``'min_bound'`` and
            ``'max_bound'`` keys.
        inplace : bool, default=False
            If True, modify in place and return None. If False, return a new wrapper.

        Returns
        -------
        PleomorphicSurface or None
            Wrapped surface when ``inplace=False``; ``None`` when ``inplace=True``.
        """
        out = self.surface.crop(bbox, inplace=inplace)
        if inplace:
            return None
        return PleomorphicSurface(out)

    def extract_region(
        self,
        indices: np.ndarray,
        element: Literal["triangles", "points", "mask"] = "triangles",
        preserve_curvatures: bool = True,
    ) -> "PleomorphicSurface":
        """
        Extract an indexed subregion from the wrapped surface.

        For meshes, ``element='triangles'`` extracts a triangle submesh and preserves
        per-vertex curvature fields by default. For point clouds, ``element='points'``
        extracts selected points; ``element='mask'`` treats ``indices`` as a boolean mask.

        Parameters
        ----------
        indices : np.ndarray
            Integer indices of the elements to keep, or a boolean mask when
            ``element='mask'``.
        element : {'triangles', 'points', 'mask'}, default='triangles'
            Which surface primitive ``indices`` refers to:

            - ``'triangles'`` (Mesh only): select by triangle index.
            - ``'points'`` (OrientedPointCloud only): select by point index.
            - ``'mask'`` (OrientedPointCloud only): boolean selection mask.
        preserve_curvatures : bool, default=True
            Mesh only — when True, per-vertex curvature fields are copied to
            the extracted submesh. Ignored for point clouds.

        Returns
        -------
        PleomorphicSurface
            New wrapper containing only the extracted elements.

        Raises
        ------
        ValueError
            If ``element`` is not valid for the wrapped surface's representation
            (e.g. ``'triangles'`` on an OrientedPointCloud).
        TypeError
            If the wrapped surface is neither :class:`Mesh` nor
            :class:`OrientedPointCloud`.
        """
        element = str(element).lower()
        if isinstance(self.surface, Mesh):
            if element not in ("triangle", "triangles"):
                raise ValueError("Mesh subregions currently support element='triangles'")
            out = self.surface.extract_submesh(indices, preserve_curvatures=preserve_curvatures)
            return PleomorphicSurface(out)

        if isinstance(self.surface, OrientedPointCloud):
            if element in ("point", "points"):
                out = self.surface.extract_points(point_ids=indices)
            elif element == "mask":
                out = self.surface.extract_points(mask=indices)
            else:
                raise ValueError("Point-cloud subregions support element='points' or element='mask'")
            return PleomorphicSurface(out)

        raise TypeError(f"Unsupported surface type: {type(self.surface)}")

    @gui_exposed(category="surface-op", label="[Mesh/OPC] Convex hull", group="Geometry", order=20, returns="surface")
    def convex_hull(self) -> "PleomorphicSurface":
        """Return the convex hull as a :class:`PleomorphicSurface` wrapping a :class:`Mesh`.

        Works for both Mesh and OrientedPointCloud backing surfaces. The per-hull
        statistics (volume, surface area, etc.) returned by Open3D are discarded; call
        :meth:`Mesh.convex_hull` directly if you need them.

        Returns
        -------
        PleomorphicSurface
            New wrapper whose backing surface is a :class:`Mesh` of the convex hull.
        """
        if isinstance(self.surface, Mesh):
            hull_o3d, _ = self.surface._to_open3d().compute_convex_hull()
        elif isinstance(self.surface, OrientedPointCloud):
            hull_o3d, _ = self.surface._to_open3d().compute_convex_hull()
        else:
            raise TypeError(f"Unsupported surface type: {type(self.surface)}")
        hull_mesh = Mesh.from_open3d(hull_o3d)
        print(f"Computed convex hull with {len(hull_mesh.vertices)} vertices")
        return PleomorphicSurface(hull_mesh)

    @gui_exposed(
        category="surface-op", label="[Mesh/OPC] Clean by normals angle", group="Mask", order=30, returns="none"
    )
    def clean_by_normals(self, max_angle_deg: float = 90.0) -> "PleomorphicSurface":
        """Remove points whose normal deviates more than ``max_angle_deg`` from the mean direction.

        Cleans against the *mean* normal of the surface. To clean against a fixed
        axis/direction instead, use :meth:`clean_by_angle`. To split by orientation
        relative to a reference *point* (e.g. centroid), use
        :meth:`separate_surfaces` with ``surface_type='closed'``.

        Parameters
        ----------
        max_angle_deg : float, default=90.0
            Maximum allowed angle (degrees) between a point's normal and the mean
            normal of the surface. Points exceeding this threshold are removed.

        Returns
        -------
        PleomorphicSurface
            ``self`` (modified in-place).
        """
        self.surface.apply_normals_mask(
            angle_threshold=max_angle_deg,
            reference_normal=None,
            inplace=True,
        )
        print("Cleaned by normals (angle vs mean)")
        return self

    @gui_exposed(category="surface-op", label="[Mesh/OPC] Clean by angle", group="Mask", order=40, returns="none")
    def clean_by_angle(
        self, max_angle_deg: float, reference_normal: np.ndarray, signed: bool = False
    ) -> "PleomorphicSurface":
        """Remove points whose normal deviates more than ``max_angle_deg`` from a given axis.

        Companion to :meth:`clean_by_normals`; both delegate to the same engine
        (:meth:`~DiscreteSurface.apply_normals_mask`) but this variant cleans against a
        caller-supplied direction rather than the mean normal.

        Parameters
        ----------
        max_angle_deg : float
            Maximum allowed angle (degrees) between a point's normal and
            ``reference_normal``. Points exceeding this threshold are removed.
        reference_normal : np.ndarray (3,)
            Reference direction/axis to measure each point's normal against.
        signed : bool, default=False
            If False (default), antiparallel normals count as aligned. If True, a
            normal pointing opposite ``reference_normal`` is treated as a 180°
            deviation (directional cleaning).

        Returns
        -------
        PleomorphicSurface
            ``self`` (modified in-place).
        """
        self.surface.apply_normals_mask(
            angle_threshold=max_angle_deg,
            reference_normal=reference_normal,
            inplace=True,
            signed=signed,
        )
        print("Cleaned by angle (angle vs given axis)")
        return self

    @gui_exposed(
        category="surface-op", label="[Mesh/OPC] Separate surfaces", group="Geometry", order=30, returns="surface_pair"
    )
    def separate_surfaces(
        self,
        surface_type: Literal["closed", "planar"] = "closed",
        threshold_angle: float = 90.0,
        reference_point: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate the two halves of a Mesh or OrientedPointCloud surface.

        Parameters
        ----------
        surface_type : {'closed', 'planar'}, default='closed'
            Strategy for separation:

            ``'closed'``
                For enclosed volumes (vesicles, organelles). Classifies each
                vertex by whether its normal points toward or away from a
                reference point (default: centroid).
                Returns ``(inner_mask, outer_mask)``.
                Calls :meth:`~DiscreteSurface.separate_closed_surface`.

            ``'planar'``
                For surfaces with lower curvature or flatter geometry. Uses
                PCA on normals to find the axis of greatest normal spread and
                splits by projection sign.
                Returns ``(surface1_mask, surface2_mask)`` — no inherent
                inner/outer meaning; inspect spatially to assign labels.
                Calls :meth:`~DiscreteSurface.separate_planar_surface`.

        threshold_angle : float, default=90.0
            Angle threshold in degrees. Only used when ``surface_type='closed'``.
        reference_point : np.ndarray (3,), optional
            Reference point for ``'closed'`` separation. Defaults to centroid.
            Ignored for ``'planar'``.

        Returns
        -------
        mask1, mask2 : (N,) bool ndarray each
            For ``'closed'``: ``(inner_mask, outer_mask)``.
            For ``'planar'``: ``(surface1_mask, surface2_mask)``.
            Pass either mask directly to :meth:`apply_vertex_mask`.
        """
        if surface_type == "closed":
            return self.surface.separate_closed_surface(threshold_angle, reference_point)
        elif surface_type == "planar":
            return self.surface.separate_planar_surface()
        else:
            raise ValueError(
                f"Unknown surface_type {surface_type!r}. "
                "Use 'closed' (enclosed volumes) or 'planar' (flatter surfaces)."
            )

    def apply_vertex_mask(self, mask: np.ndarray, inplace: bool = False) -> "PleomorphicSurface | None":
        """
        Return a surface containing only vertices where ``mask`` is True.

        Pass one of the boolean masks returned by :meth:`separate_surfaces`:

        - For ``surface_type='closed'``: pass ``inner_mask`` or ``outer_mask``.
        - For ``surface_type='planar'``: pass ``surface1_mask`` or ``surface2_mask``.

        Example::

            inner_mask, outer_mask = ps.separate_surfaces(surface_type='closed')
            inner = ps.apply_vertex_mask(inner_mask)

            s1_mask, s2_mask = ps.separate_surfaces(surface_type='planar')
            half1 = ps.apply_vertex_mask(s1_mask)

        Parameters
        ----------
        mask : np.ndarray
            Boolean array of shape (N,) aligned with ``self.surface.vertices``.
        inplace : bool, default=False
            If True, modify this instance in place and return None.
            If False, return a new PleomorphicSurface.

        Returns
        -------
        PleomorphicSurface or None
            New instance if ``inplace=False``, else None.
        """
        if not isinstance(self.surface, (Mesh, OrientedPointCloud)):
            raise TypeError(f"Unsupported surface type: {type(self.surface)}")
        filtered_surface = self.surface.apply_vertex_mask(mask, inplace=inplace)
        if inplace:
            return None
        return PleomorphicSurface(filtered_surface)

    @gui_exposed(category="surface-op", label="[Mesh/OPC] Distance to points", group="Analysis", order=50)
    def distance_to_points(
        self,
        target: np.ndarray,
        compute_occupancy: bool = True,
        compute_signed: bool = False,
        return_closest_points: bool = False,
    ) -> dict:
        """
        Compute distance from a point to a Mesh or an OrientedPointCloud surface.

        Parameters
        ----------
        target : np.ndarray
            Query points as (N, 3) array
        compute_occupancy : bool, default=True
            Compute occupancy (inside/outside). Only for Mesh.
        compute_signed : bool, default=False
            Compute signed distance instead of unsigned. Only for Mesh.
            If True, compute_occupancy is automatically enabled.
        return_closest_points : bool, default=False
            Return closest surface points and triangle/point IDs

        Returns
        -------
        dict
            Dictionary containing:
            - 'distances': unsigned or signed distances for each point
            - 'distance_type': 'signed' or 'unsigned'
            - 'n_total': total number of query points

            If compute_occupancy=True and Mesh:
            - 'occupancy': binary array (1=inside, 0=outside)
            - 'inside_mask': boolean mask for inside points
            - 'outside_mask': boolean mask for outside points
            - 'n_inside': number of points inside
            - 'n_outside': number of points outside

            If return_closest_points=True:
            - 'closest_points': closest points on surface (N, 3)
            - 'primitive_ids': triangle IDs (Mesh) or point IDs (PointCloud) (N,)
            - 'closest_distances': distances to closest points (same as 'distances' for unsigned)

        Raises
        ------
        TypeError
            If trying to compute occupancy/signed distance for non-Mesh surface
        """
        target = np.atleast_2d(target).astype(np.float32)
        if isinstance(self.surface, Mesh):
            return self.surface.distance_to_points(
                target=target,
                compute_occupancy=compute_occupancy,
                compute_signed=compute_signed,
                return_closest_points=return_closest_points,
            )
        elif isinstance(self.surface, OrientedPointCloud):
            if compute_occupancy or compute_signed:
                raise TypeError("OrientedPointCloud does not support occupancy or signed distance queries; use Mesh.")
            return self.surface.distance_to_points(
                target=target,
                return_closest_points=return_closest_points,
            )
        raise TypeError(f"Unsupported surface type: {type(self.surface)}")

    def get_points_within_distance(self, target: np.ndarray, threshold: float) -> dict:
        """
        Find points within a distance threshold from a Mesh or an OrientedPointCloud surface.

        Parameters
        ----------
        target : np.ndarray
            Query points as (N, 3) array
        threshold : float
            Distance threshold

        Returns
        -------
        dict
            Dictionary containing:
            - 'mask': boolean array indicating points within threshold
            - 'distances': unsigned distances for all points
            - 'indices': indices of points within threshold
            - 'within_points': coordinates of points within threshold
            - 'n_within': number of points within threshold
            - 'n_total': total number of query points
        """
        target = np.atleast_2d(target).astype(np.float32)
        dist_result = self.distance_to_points(
            target=target,
            compute_occupancy=False,
            compute_signed=False,
            return_closest_points=False,
        )

        distances = dist_result["distances"]

        # Find points within threshold
        mask = distances <= threshold
        indices = np.where(mask)[0]

        result = {
            "mask": mask,
            "distances": distances,
            "indices": indices,
            "within_points": target[mask],
            "n_within": np.sum(mask),
            "n_total": len(target),
        }

        return result

    def get_neighboring_triangles(
        self, triangle_id: int, method: Literal["edge-connected", "radius"] = "edge-connected", **kwargs: Any
    ) -> set | dict:
        """
        Get neighboring triangles (Mesh only).

        Parameters
        ----------
        triangle_id : int
            ID of the seed triangle.
        method : {'edge-connected', 'radius'}, default='edge-connected'
            Traversal strategy.  ``'edge-connected'`` — topological edge walk.
            ``'radius'`` — distance-based search (requires ``radius`` kwarg).
        **kwargs
            Additional parameters:
            - For 'q': max_hops (int, default=1)
            - For 'radius': radius (float, required), use_kdtree (bool, default=True)

        Returns
        -------
        set or dict
            For 'edge-connected': set of triangle IDs
            For 'radius': dict with 'neighbor_ids', 'distances', 'seed_centroid', 'n_neighbors'

        Raises
        ------
        TypeError
            If surface is not a Mesh
        ValueError
            If invalid method or missing required parameters
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError(f"Triangle neighbors only available for Mesh, not {type(self.surface).__name__}")

        if method == "edge-connected":
            max_hops = kwargs.get("max_hops", 1)
            return self.surface.get_connected_triangles(triangle_id, max_hops=max_hops)

        elif method == "radius":
            if "radius" not in kwargs:
                raise ValueError("'radius' parameter required for method='radius'")
            radius = kwargs["radius"]
            use_kdtree = kwargs.get("use_kdtree", True)
            return self.surface.get_triangles_within_radius(triangle_id, radius, use_kdtree=use_kdtree)

        else:
            raise ValueError(f"Unknown method: {method}. Use 'topology' or 'radius'")

    def get_connected_triangles(self, triangle_id: int, max_hops: int = 1) -> set:
        """Return edge-connected neighboring triangle IDs for a mesh-backed surface.

        Parameters
        ----------
        triangle_id : int
            Seed triangle index.
        max_hops : int, default=1
            Number of edge-traversal steps from the seed. ``max_hops=1`` returns only
            immediate face-neighbors; higher values expand the region.
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError(f"Triangle neighbors only available for Mesh, not {type(self.surface).__name__}")
        return self.surface.get_connected_triangles(triangle_id, max_hops=max_hops)

    def get_triangles_within_radius(self, triangle_id: int, radius: float, use_kdtree: bool = True) -> dict:
        """Return triangle-neighborhood query result for a mesh-backed surface.

        Parameters
        ----------
        triangle_id : int
            Seed triangle index.
        radius : float
            Maximum centroid-to-centroid distance for a triangle to be included.
        use_kdtree : bool, default=True
            Use a KDTree for fast radius queries (recommended for large meshes).
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError(f"Triangle neighbors only available for Mesh, not {type(self.surface).__name__}")
        return self.surface.get_triangles_within_radius(triangle_id, radius, use_kdtree=use_kdtree)

    def ray_intersections(
        self,
        rays: np.ndarray,
        one_hit_per_target: bool = False,
        knn_radius: float = 10.0,
        return_orientations: bool = False,
        target_orientation: Literal["normal", "principal_1", "principal_2"] | Callable = "normal",
    ) -> dict:
        """
        Compute ray intersections with this surface.

        For meshes, uses Open3D's exact raycasting. For oriented point clouds,
        uses KDTree-based nearest neighbor search along ray trajectories.

        Parameters
        ----------
        rays : np.ndarray, shape (N, 6)
            Ray array where each row is [origin_x, origin_y, origin_z, dir_x, dir_y, dir_z]
        one_hit_per_target : bool, default=False
            If True and multiple rays are supplied, add ``shortest_distance`` and
            ``shortest_indices`` for the global shortest hit across all rays.
        knn_radius : float, default=10.0
            For OrientedPointCloud only: maximum search radius for finding
            nearest points along ray trajectory
        return_orientations : bool, default=False
            If True, compute relative orientations between ray directions and
            surface orientations at hit points. Returns additional fields:
            - 'ray_directions': normalized ray direction vectors
            - 'surface_orientations': orientation vectors at hit points
            - 'angles_deg': angles in degrees between ray and surface orientation
            - 'dot_products': dot products (cosine of angle)
        target_orientation : {'normal', 'principal_1', 'principal_2'} or callable, default='normal'
            Which orientation to use for comparison. Options:

            For OrientedPointCloud:
                - 'normal': Use the normals stored in the point cloud
                  (for filaments, these are the axis/tangent directions)

            For Mesh:
                - 'normal': Use surface normals at hit points (default)
                - 'principal_1': Use first principal curvature direction
                - 'principal_2': Use second principal curvature direction

            Custom function:
                - A callable that takes (surface, primitive_ids, hit_points) and returns
                  an (N, 3) array of orientation vectors, where N is the number of hits.
                  Example: lambda surf, ids, pts: surf.get_curvature_directions()[ids, :, 0]

        Returns
        -------
        dict
            Always contains:

            - ``t_hit`` (R,): ray travel distance to the intersection; ``inf`` for misses.
            - ``primitive_ids`` (R,): triangle index (Mesh) or point index (OrientedPointCloud)
              of the hit; ``-1`` for misses.
            - ``hit_points`` (R, 3): 3-D coordinates of hit points; NaN for misses.
            - ``primitive_normals`` (R, 3): surface normals at hit points (Mesh only); NaN for
              misses or when the backing surface is an OrientedPointCloud.
            - ``geometry_ids`` (R,): geometry identifier (Mesh only).

            When ``return_orientations=True``, also adds:

            - ``ray_directions`` (R, 3): normalized ray direction vectors.
            - ``surface_orientations`` (R, 3): surface orientation vectors at hit points;
              NaN for misses.
            - ``angles_deg`` (R,): angle in degrees between the ray and surface orientation;
              NaN for misses.
            - ``dot_products`` (R,): cosine of that angle; NaN for misses.
        """
        surface = self.surface
        rays = np.atleast_2d(rays).astype(np.float32)

        if isinstance(surface, Mesh):
            result = surface.cast_rays(
                rays,
                one_hit_per_target=one_hit_per_target,
            )
        elif isinstance(surface, OrientedPointCloud):
            result = surface.cast_rays(
                rays,
                knn_radius=knn_radius,
                one_hit_per_target=one_hit_per_target,
            )
        else:
            raise TypeError(f"Unsupported surface type: {type(surface)}. Must be Mesh or OrientedPointCloud")

        # Compute orientation metrics if requested
        if return_orientations:
            origins = rays[:, :3]
            directions = rays[:, 3:]

            # Normalize ray directions
            dir_magnitudes = np.linalg.norm(directions, axis=1, keepdims=True)
            ray_directions = directions / (dir_magnitudes + 1e-10)

            # Get surface orientations at hit points based on target_orientation
            surface_orientations_hits = DiscreteSurface.ray_hit_orientations(surface, result, target_orientation)

            if surface_orientations_hits is not None:
                # Create full-size array for all rays (fill with NaN for non-hits)
                hit_mask = np.isfinite(result["t_hit"])
                n_rays = len(rays)
                n_hits = hit_mask.sum()

                surface_orientations = np.full((n_rays, 3), np.nan)
                if n_hits > 0:
                    surface_orientations[hit_mask] = surface_orientations_hits

                # Normalize surface orientations (only for valid hits)
                normalized_orientations = np.full((n_rays, 3), np.nan)
                if n_hits > 0:
                    orient_magnitudes = np.linalg.norm(surface_orientations_hits, axis=1, keepdims=True)
                    normalized_orientations[hit_mask] = surface_orientations_hits / (orient_magnitudes + 1e-10)

                # Compute dot products (cosine of angle)
                # For rays that didn't hit, use NaN
                dot_products = np.full(n_rays, np.nan)

                if n_hits > 0:
                    # Compute dot product for valid hits only
                    dots = np.sum(ray_directions[hit_mask] * normalized_orientations[hit_mask], axis=1)
                    # Clamp to [-1, 1] for numerical stability
                    dots = np.clip(dots, -1.0, 1.0)
                    dot_products[hit_mask] = dots

                # Compute angles in degrees
                angles_deg = np.full(n_rays, np.nan)
                if n_hits > 0:
                    valid_dots = dot_products[hit_mask]
                    angles_deg[hit_mask] = np.arccos(valid_dots) * 180.0 / np.pi

                result["ray_directions"] = ray_directions
                result["surface_orientations"] = surface_orientations
                result["angles_deg"] = angles_deg
                result["dot_products"] = dot_products
            else:
                # No orientations available
                result["ray_directions"] = ray_directions
                result["surface_orientations"] = np.full((len(rays), 3), np.nan)
                result["angles_deg"] = np.full(len(rays), np.nan)
                result["dot_products"] = np.full(len(rays), np.nan)

        return result

    def invalidate_caches(self) -> None:
        """Invalidate cached geometry on the wrapped surface (mesh ray scene, neighbor trees, etc.)."""
        if isinstance(self.surface, Mesh):
            self.surface._invalidate_cache()
            self.surface._invalidate_neighbor_cache()

    def distance_to_pointcloud(
        self,
        target: "PleomorphicSurface" | OrientedPointCloud,
        method: Literal["nn_unoriented", "nn", "nearest", "nn_oriented", "ray"] = "nn_unoriented",
        max_distance: float | None = None,
        ray_length: float | None = None,
        reverse_normals: bool = False,
        bidirectional: bool = False,
        one_hit_per_target: bool = False,
        knn_radius: float = 10.0,
        return_stats: bool = True,
    ) -> dict:
        """
        Compute distance from this surface to another point cloud surface.
        - If source is Mesh: always uses raycasting
        - If source is OrientedPointCloud search nearest neighbours (unoriented or along normals)

        Parameters
        ----------
        target : PleomorphicSurface or OrientedPointCloud
            Target surface. Wrapped targets are unwrapped internally; the concrete
            target must be an OrientedPointCloud.
        method : {'nn_unoriented', 'nn', 'nearest', 'nn_oriented', 'ray'}, default='nn_unoriented'
            Distance computation method (only used if source is OrientedPointCloud).
            ``'nn_unoriented'`` (aliases ``'nn'``, ``'nearest'``) — KDTree nearest
            neighbour.  ``'nn_oriented'`` (alias ``'ray'``) — ray cast along normals.
        max_distance : float, optional
            Maximum distance threshold. Points beyond this distance are excluded.
        ray_length : float, optional
            For raycasting: maximum ray length. If None, uses infinite rays.
            If max_distance is set and ray_length is None, ray_length = max_distance.
        reverse_normals : bool, default=False
            For normal: if True, cast rays opposite to normal direction
        bidirectional : bool, default=False
            For Mesh sources, cast along both normal directions and keep the closer hit.
        one_hit_per_target : bool, default=False
            For mesh sources: keep only the closest mesh vertex per target particle
            (deduplicate ``distance_to_pointcloud`` hits).
        knn_radius : float, default=10.0
            For point cloud search along normals: search radius for finding points along ray trajectory
        return_stats : bool, default=True
            If True, return stats dictionary. If False, return only distances array.

        Returns
        -------
        dict or np.ndarray
            If return_details=True, returns dictionary with:
                'distances': np.ndarray (N,) - distance for each source point
                'closest_points': np.ndarray (N, 3) - coordinates of closest/hit points
                'closest_indices': np.ndarray (N,) - indices in target point cloud (-1 if no hit)
                'hit_mask': np.ndarray (N,) - boolean mask of successful matches
                'closest_normals': np.ndarray (N, 3) - normals at closest points (if available)
                'stats': dict with min, max, mean, median, std distance statistics

            If return_stats=False, returns only distances array (N,)
        """
        target_surface = self._unwrap_surface(target)
        if not isinstance(target_surface, OrientedPointCloud):
            raise TypeError(f"Target surface must be OrientedPointCloud, got {type(target_surface).__name__}. ")

        if isinstance(self.surface, Mesh):
            result = self.surface.distance_to_pointcloud(
                target=target_surface,
                ray_length=ray_length,
                max_distance=max_distance,
                reverse_normals=reverse_normals,
                bidirectional=bidirectional,
                one_hit_per_target=one_hit_per_target,
            )
        elif isinstance(self.surface, OrientedPointCloud):
            if bidirectional:
                raise ValueError("bidirectional=True is only supported for Mesh sources")
            result = self.surface.distance_to_pointcloud(
                target=target_surface,
                method=method,
                max_distance=max_distance,
                ray_length=ray_length,
                reverse_normals=reverse_normals,
                knn_radius=knn_radius,
                one_hit_per_target=one_hit_per_target,
            )
        else:
            raise TypeError(f"Unsupported source surface type: {type(self.surface)}")

        if return_stats:
            return result
        return result["distances"]

    @staticmethod
    def _infer_query_type(result: dict[str, Any]) -> str:
        """Infer result format from keys produced by ray or distance queries."""
        if "t_hit" in result:
            return "ray"
        if "hit_mask" in result and "closest_indices" in result:
            return "distance_to_pointcloud"
        raise ValueError(
            "Could not infer query_type from result. " "Pass query_type='ray' or query_type='distance_to_pointcloud'."
        )

    @staticmethod
    def _filter_hits_by_distance(
        distances: np.ndarray,
        min_distance_source_target: float | None = None,
        max_distance_source_target: float | None = None,
    ) -> np.ndarray:
        """Boolean mask for hits within an optional source-target distance interval."""
        keep = np.ones(len(distances), dtype=bool)
        if min_distance_source_target is not None:
            keep &= distances >= min_distance_source_target
        if max_distance_source_target is not None:
            keep &= distances <= max_distance_source_target
        return keep

    @staticmethod
    def _parse_ray_hits(
        result: dict[str, Any],
        min_distance_source_target: float | None = None,
        max_distance_source_target: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract per-ray hit rows from :meth:`ray_intersections` output."""
        t_hit = np.asarray(result["t_hit"])
        hit_mask = np.isfinite(t_hit)
        source_ids = np.where(hit_mask)[0]
        distances = t_hit[hit_mask]

        if "primitive_ids" not in result:
            raise KeyError(
                "Ray result must contain 'primitive_ids'. "
                "Both Mesh and OrientedPointCloud cast_rays return this key."
            )
        target_ids = np.asarray(result["primitive_ids"])[hit_mask]

        keep = PleomorphicSurface._filter_hits_by_distance(
            distances, min_distance_source_target, max_distance_source_target
        )
        out: dict[str, np.ndarray] = {
            "source_ids": source_ids[keep],
            "target_ids": target_ids[keep],
            "distances": distances[keep],
        }
        if "hit_points" in result:
            hit_points = np.asarray(result["hit_points"])[hit_mask][keep]
            out["hit_points"] = hit_points
        return out

    @staticmethod
    def _parse_distance_hits(
        result: dict[str, Any],
        min_distance_source_target: float | None = None,
        max_distance_source_target: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Extract per-source hit rows from :meth:`distance_to_pointcloud` output."""
        hit_mask = np.asarray(result["hit_mask"], dtype=bool)
        source_ids = np.where(hit_mask)[0]
        distances = np.asarray(result["distances"])[hit_mask]
        target_ids = np.asarray(result["closest_indices"])[hit_mask]

        keep = PleomorphicSurface._filter_hits_by_distance(
            distances, min_distance_source_target, max_distance_source_target
        )
        out: dict[str, np.ndarray] = {
            "source_ids": source_ids[keep],
            "target_ids": target_ids[keep],
            "distances": distances[keep],
        }
        if "closest_points" in result:
            out["hit_points"] = np.asarray(result["closest_points"])[hit_mask][keep]
        if "used_reverse_normals" in result:
            out["used_reverse_normals"] = np.asarray(result["used_reverse_normals"])[hit_mask][keep]
        return out

    def _mesh_triangle_curvature_table(self) -> dict[str, np.ndarray]:
        """Per-triangle mean and Gaussian curvature (vertex average over face corners)."""
        if not isinstance(self.surface, Mesh):
            raise TypeError("Triangle curvature table requires a Mesh-backed PleomorphicSurface")
        faces = self.surface.faces
        mean_vertex = self.get_mean_curvature()
        gaussian_vertex = self.get_gaussian_curvature()
        return {
            "mean_curvature": mean_vertex[faces].mean(axis=1),
            "gaussian_curvature": gaussian_vertex[faces].mean(axis=1),
        }

    def _triangles_from_vertices(self, vertex_ids: np.ndarray) -> np.ndarray:
        """Triangle IDs incident on any of the given mesh vertex indices."""
        if not isinstance(self.surface, Mesh):
            raise TypeError("Vertex-to-triangle lookup requires a Mesh-backed PleomorphicSurface")
        vertex_ids = np.unique(np.asarray(vertex_ids, dtype=np.intp))
        faces = self.surface.faces
        return np.flatnonzero(np.isin(faces, vertex_ids).any(axis=1))

    def get_triangle_neighborhoods(
        self,
        seed_triangle_ids: np.ndarray,
        radii: ArrayLike,
        use_kdtree: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Expand seed triangles on the mesh using centroid-distance radii.

        Always includes the ``"hit triangles"`` key. For each radius ``r`` in ``radii``,
        adds a cumulative ``"r <= {r} nm"`` key and an annulus band
        ``"{r_inner} < r <= {r_outer} nm"`` between consecutive radii.

        Parameters
        ----------
        seed_triangle_ids : np.ndarray
            Integer indices of the seed triangles to expand from.
        radii : ArrayLike
            Expansion radii in the same units as the mesh coordinates. Each entry
            produces a cumulative shell and (for consecutive pairs) an annulus band.
        use_kdtree : bool, default=True
            If True, use a KD-tree for centroid lookups; otherwise use brute-force search.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"hit triangles"``, ``"r <= {r} nm"`` for each radius, and
            ``"{r_inner} < r <= {r_outer} nm"`` for each consecutive pair.
            Values are sorted integer arrays of triangle indices.
        """
        if not isinstance(self.surface, Mesh):
            raise TypeError("Triangle neighborhoods require a Mesh-backed PleomorphicSurface")

        seeds = np.unique(np.asarray(seed_triangle_ids, dtype=np.intp))
        regions: dict[str, np.ndarray] = {
            "hit triangles": np.sort(seeds),
        }

        radii = [float(r) for r in radii]
        if len(radii) == 0:
            return regions

        cumulative: list[set] = []
        for radius in radii:
            expanded: set = set()
            for triangle_id in seeds:
                neighbors = self.get_triangles_within_radius(int(triangle_id), radius, use_kdtree=use_kdtree)[
                    "neighbor_ids"
                ]
                expanded.update(np.asarray(neighbors, dtype=np.intp).tolist())
            cumulative.append(expanded)
            regions[f"r <= {radius:g} nm"] = np.array(sorted(expanded), dtype=int)

        for idx_inner, idx_outer in enumerate(range(len(radii) - 1)):
            r_inner = radii[idx_inner]
            r_outer = radii[idx_inner + 1]
            ring_set = cumulative[idx_inner + 1] - cumulative[idx_inner]
            regions[f"{r_inner:g} < r <= {r_outer:g} nm"] = np.array(sorted(ring_set), dtype=int)

        return regions

    @staticmethod
    def _summarize_triangle_regions(
        regions: dict[str, np.ndarray],
        mean_tri: np.ndarray,
        gaussian_tri: np.ndarray,
    ) -> pd.DataFrame:
        """Summarize per-triangle curvature statistics for named mesh regions."""
        rows = []
        for name, tri_ids in regions.items():
            tri_ids = np.asarray(tri_ids, dtype=int)
            if len(tri_ids) == 0:
                rows.append(
                    {
                        "region": name,
                        "n_triangles": 0,
                        "mean_curvature_mean": np.nan,
                        "mean_curvature_median": np.nan,
                        "gaussian_curvature_mean": np.nan,
                        "gaussian_curvature_median": np.nan,
                    }
                )
                continue
            mean_vals = mean_tri[tri_ids]
            gauss_vals = gaussian_tri[tri_ids]
            rows.append(
                {
                    "region": name,
                    "n_triangles": len(tri_ids),
                    "mean_curvature_mean": float(np.mean(mean_vals)),
                    "mean_curvature_median": float(np.median(mean_vals)),
                    "gaussian_curvature_mean": float(np.mean(gauss_vals)),
                    "gaussian_curvature_median": float(np.median(gauss_vals)),
                }
            )
        return pd.DataFrame(rows)

    def get_point_neighborhoods(
        self,
        seed_point_ids: np.ndarray,
        radii: ArrayLike,
    ) -> dict[str, np.ndarray]:
        """
        Expand seed points on an oriented point cloud using surface radii.

        Delegates to :meth:`OrientedPointCloud.get_point_neighborhoods`. The returned
        dict always includes ``"hit points"``; for each radius ``r``, adds ``"r <= {r} nm"``
        and annulus bands ``"{r_inner} < r <= {r_outer} nm"`` between consecutive radii.

        Parameters
        ----------
        seed_point_ids : np.ndarray
            Integer indices of the seed points to expand from.
        radii : ArrayLike
            Expansion radii in the same units as the point-cloud coordinates.

        Returns
        -------
        dict[str, np.ndarray]
            Keys: ``"hit points"``, ``"r <= {r} nm"`` for each radius, and
            ``"{r_inner} < r <= {r_outer} nm"`` for each consecutive pair.
            Values are sorted integer arrays of point indices.
        """
        if not isinstance(self.surface, OrientedPointCloud):
            raise TypeError("Point neighborhoods require an OrientedPointCloud-backed PleomorphicSurface")
        return self.surface.get_point_neighborhoods(seed_point_ids, radii)

    @staticmethod
    def _summarize_point_regions(
        regions: dict[str, np.ndarray],
        normals: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Summarize point regions (counts; optional mean normal components)."""
        rows = []
        for name, point_ids in regions.items():
            point_ids = np.asarray(point_ids, dtype=int)
            row: dict[str, Any] = {
                "region": name,
                "n_points": len(point_ids),
            }
            if normals is not None and len(point_ids) > 0:
                n = normals[point_ids]
                row["normal_x_mean"] = float(np.mean(n[:, 0]))
                row["normal_y_mean"] = float(np.mean(n[:, 1]))
                row["normal_z_mean"] = float(np.mean(n[:, 2]))
            rows.append(row)
        return pd.DataFrame(rows)

    def _default_surface_element(self, query_type: str) -> str:
        """Default mesh/point element for region expansion on ``self``."""
        if query_type == "ray":
            return "points" if isinstance(self.surface, OrientedPointCloud) else "triangles"
        return "points" if isinstance(self.surface, OrientedPointCloud) else "vertices"

    def _resolve_region_seed_ids(
        self,
        parsed: dict[str, np.ndarray],
        query_type: str,
        surface_element: str,
        surface_seeds: str,
    ) -> np.ndarray:
        """Map hit rows to seed indices used for ``surface_radii`` expansion."""
        surface_seeds = str(surface_seeds).lower()
        aliases = {
            "auto": "auto",
            "default": "auto",
            "sources": "hit_sources",
            "source": "hit_sources",
            "targets": "hit_targets",
            "target": "hit_targets",
        }
        surface_seeds = aliases.get(surface_seeds, surface_seeds)

        if surface_seeds == "auto":
            if query_type == "ray":
                return np.asarray(parsed["target_ids"], dtype=np.intp)
            if surface_element in ("point", "points"):
                return np.asarray(parsed["source_ids"], dtype=np.intp)
            if surface_element in ("vertex", "vertices"):
                return np.asarray(parsed["source_ids"], dtype=np.intp)
            return np.asarray(parsed["target_ids"], dtype=np.intp)
        if surface_seeds == "hit_sources":
            return np.asarray(parsed["source_ids"], dtype=np.intp)
        if surface_seeds == "hit_targets":
            return np.asarray(parsed["target_ids"], dtype=np.intp)
        raise ValueError("surface_seeds must be 'auto', 'hit_sources', or 'hit_targets'")

    def _build_surface_regions(
        self,
        seed_ids: np.ndarray,
        surface_element: str,
        surface_radii: ArrayLike,
        use_kdtree: bool = True,
    ) -> dict[str, np.ndarray]:
        """
        Expand hit seeds on ``self`` using ``surface_radii``.

        For meshes, ``vertices`` seeds are mapped to incident triangles before
        triangle-centroid expansion. For point clouds, ``points`` use 3D ball queries.
        """
        element = str(surface_element).lower()
        element = {
            "triangle": "triangles",
            "vertex": "vertices",
            "point": "points",
        }.get(element, element)

        if element in ("triangles", "vertices"):
            if not isinstance(self.surface, Mesh):
                raise TypeError(f"surface_element='{surface_element}' requires a Mesh-backed surface")
            seed_triangles = (
                self._triangles_from_vertices(seed_ids)
                if element == "vertices"
                else np.asarray(seed_ids, dtype=np.intp)
            )
            return self.get_triangle_neighborhoods(seed_triangles, radii=surface_radii, use_kdtree=use_kdtree)

        if element == "points":
            if not isinstance(self.surface, OrientedPointCloud):
                raise TypeError("surface_element='points' requires an OrientedPointCloud-backed surface")
            return self.get_point_neighborhoods(seed_ids, radii=surface_radii)

        raise ValueError("surface_element must be 'triangles', 'vertices', or 'points'")

    def intersection_data(
        self,
        result: dict[str, Any],
        query_type: Literal["ray", "distance_to_pointcloud"] | None = None,
        min_distance_source_target: float | None = None,
        max_distance_source_target: float | None = None,
        source_id_name: str = "source_id",
        target_id_name: str = "target_id",
        include_curvatures: bool = True,
        surface_radii: list[float] | None = None,
        surface_element: Literal["triangles", "vertices", "points"] | None = None,
        surface_seeds: Literal["auto", "hit_sources", "hit_targets"] = "auto",
        use_kdtree: bool = True,
    ) -> dict[str, Any]:
        """
        Turn raw intersection or distance-query output into analysis-ready tables.

        Works with results from :meth:`ray_intersections` (mesh or point cloud target)
        and :meth:`distance_to_pointcloud`. Optional ``surface_radii`` expansion grows
        regions around hit sites on ``self`` (triangles/vertices on meshes, points on
        oriented point clouds).

        Parameters
        ----------
        result : dict
            Output of :meth:`ray_intersections` or :meth:`distance_to_pointcloud`.
        query_type : {'ray', 'distance_to_pointcloud'}, optional
            Inferred from ``result`` when omitted. The body also accepts a small
            set of aliases (``'rays'``, ``'distance'``, ``'pointcloud'``,
            ``'distance_to_point_cloud'``) for backward compatibility.
        min_distance_source_target, max_distance_source_target : float, optional
            Keep hits whose source-target distance lies in this interval.
        source_id_name, target_id_name : str
            Column names for source and target indices in the hit table.
        include_curvatures : bool, default=True
            Attach curvature columns for mesh-backed ``self``.
        surface_radii : sequence of float, optional
            Radii for cumulative regions around hit seeds on ``self``.
        surface_element : {'triangles', 'vertices', 'points'}, optional
            Defaults: ray+mesh → triangles; distance+mesh → vertices; point
            cloud → points.
        surface_seeds : {'auto', 'hit_sources', 'hit_targets'}, default='auto'
            Which hit IDs seed expansion.
        use_kdtree : bool, default=True
            Passed to mesh triangle expansion.

        Returns
        -------
        dict
            - ``hits``: hit table
            - ``regions``: region name → index arrays (if ``surface_radii`` set)
            - ``region_summary``: per-region summary table
            - ``triangle_curvatures``: per-triangle arrays (mesh-backed ``self``)
        """
        if query_type is None:
            query_type = self._infer_query_type(result)
        query_type = str(query_type).lower()
        aliases = {
            "rays": "ray",
            "distance": "distance_to_pointcloud",
            "distance_to_point_cloud": "distance_to_pointcloud",
            "pointcloud": "distance_to_pointcloud",
        }
        query_type = aliases.get(query_type, query_type)

        if query_type == "ray":
            parsed = self._parse_ray_hits(result, min_distance_source_target, max_distance_source_target)
        elif query_type == "distance_to_pointcloud":
            parsed = self._parse_distance_hits(result, min_distance_source_target, max_distance_source_target)
        else:
            raise ValueError(f"query_type must be 'ray' or 'distance_to_pointcloud', got '{query_type}'")

        if surface_element is None:
            surface_element = self._default_surface_element(query_type)

        hits_dict: dict[str, Any] = {
            source_id_name: parsed["source_ids"],
            target_id_name: parsed["target_ids"],
            "distance_nm": parsed["distances"],
        }
        if "hit_points" in parsed:
            hits_dict["hit_point_x"] = parsed["hit_points"][:, 0]
            hits_dict["hit_point_y"] = parsed["hit_points"][:, 1]
            hits_dict["hit_point_z"] = parsed["hit_points"][:, 2]
        if "used_reverse_normals" in parsed:
            hits_dict["used_reverse_normals"] = parsed["used_reverse_normals"]

        triangle_curvatures = None
        if include_curvatures and isinstance(self.surface, Mesh):
            triangle_curvatures = self._mesh_triangle_curvature_table()
            mean_tri = triangle_curvatures["mean_curvature"]
            gaussian_tri = triangle_curvatures["gaussian_curvature"]
            if query_type == "ray":
                tri_ids = parsed["target_ids"]
                hits_dict["mean_curvature"] = mean_tri[tri_ids]
                hits_dict["gaussian_curvature"] = gaussian_tri[tri_ids]
            else:
                vert_ids = parsed["source_ids"]
                hits_dict["mean_curvature"] = self.get_mean_curvature()[vert_ids]
                hits_dict["gaussian_curvature"] = self.get_gaussian_curvature()[vert_ids]

        hits = pd.DataFrame(hits_dict)

        out: dict[str, Any] = {"hits": hits}
        if triangle_curvatures is not None:
            out["triangle_curvatures"] = triangle_curvatures

        if surface_radii is not None and len(surface_radii) > 0:
            seed_ids = self._resolve_region_seed_ids(parsed, query_type, surface_element, surface_seeds)
            regions = self._build_surface_regions(
                seed_ids,
                surface_element=surface_element,
                surface_radii=surface_radii,
                use_kdtree=use_kdtree,
            )
            out["regions"] = regions

            element = str(surface_element).lower()
            element = {
                "triangle": "triangles",
                "vertex": "vertices",
                "point": "points",
            }.get(element, element)

            if element in ("triangles", "vertices") and triangle_curvatures is not None:
                out["region_summary"] = self._summarize_triangle_regions(
                    regions,
                    triangle_curvatures["mean_curvature"],
                    triangle_curvatures["gaussian_curvature"],
                )
            elif element == "points" and isinstance(self.surface, OrientedPointCloud):
                normals = self.surface.normals if self.surface.normals is not None else None
                out["region_summary"] = self._summarize_point_regions(regions, normals)

        return out


# =============================================================================
# ParametricSurface — wrapper for analytic (ellipsoid) surface workflows
# =============================================================================


class ParametricSurface:
    """Wrapper around :class:`QuadricsM` for the ellipsoid particle-assignment workflow.

    Mirrors the static-method interface of the old ``PleomorphicSurface`` ellipsoid
    methods but as a proper class with instance state.

    Parameters
    ----------
    quadrics : QuadricsM
        Already-constructed container of analytic surfaces.
    column_name : MotlColumn, default='object_id'
        Column name used as the surface-object identifier (one fitted surface
        per unique ``(tomo_id, column_name)`` group). Per the column-naming
        convention this is :data:`cryocat._types.MotlColumn`.
    """

    def __init__(self, quadrics: QuadricsM, column_name: MotlColumn = "object_id") -> None:
        self.quadrics = quadrics
        self.column_name = column_name

    @classmethod
    def from_motl(
        cls,
        input_motl: MotlSource,
        surface_type: Literal["ellipsoid"] = "ellipsoid",
        column_name: MotlColumn = "object_id",
    ) -> "ParametricSurface":
        """Fit analytic surfaces to particle groups and return a ParametricSurface.

        Parameters
        ----------
        input_motl : MotlSource
            Input particle list. One surface is fitted per unique
            ``(tomo_id, column_name)`` group.
        surface_type : {'ellipsoid'}, default='ellipsoid'
            Quadric type; currently only ellipsoid is supported.
        column_name : MotlColumn, default='object_id'
            Column used to group particles.

        Returns
        -------
        ParametricSurface
        """
        quadrics = QuadricsM(input_motl, quadric=surface_type, feature_id=column_name)
        return cls(quadrics, column_name=column_name)

    @classmethod
    def from_csv(
        cls,
        path: PathOrStr,
        surface_type: Literal["ellipsoid"] = "ellipsoid",
        column_name: MotlColumn = "object_id",
    ) -> "ParametricSurface":
        """Load analytic surface parameters from a CSV file.

        Parameters
        ----------
        path : PathOrStr
            Path to a CSV produced by :meth:`write_out`.
        surface_type : {'ellipsoid'}, default='ellipsoid'
            Quadric type to materialise.
        column_name : MotlColumn, default='object_id'
            Surface-object identifier column the file is keyed on.

        Returns
        -------
        ParametricSurface
        """
        quadrics = QuadricsM(path, quadric=surface_type, feature_id=column_name)
        return cls(quadrics, column_name=column_name)

    def write_out(self, output_path: PathOrStr) -> None:
        """Write the surface parameter table to *output_path* as CSV.

        Parameters
        ----------
        output_path : PathOrStr
            Destination CSV file path.
        """
        self.quadrics.write_out(output_path)

    def compute_point_surface_distance(
        self,
        input_motl: MotlSource,
        output_path: PathOrStr | None = None,
        store_column_name: MotlColumn = "geom4",
    ) -> "cryomotl.Motl":
        """Compute the shortest distance from each particle to its assigned surface.

        Parameters
        ----------
        input_motl : MotlSource
            Particles with ``self.column_name`` already assigned.
        output_path : PathOrStr, optional
            Path to save the result.
        store_column_name : MotlColumn, default='geom4'
            Column that receives the distance values.

        Returns
        -------
        Motl
            Input motl with ``store_column_name`` populated.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.column_name)
        assigned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.column_name, reset_index=True)
            coord = fm.get_coordinates()
            tomo_id = fm.df["tomo_id"].values[0]
            fm.df[store_column_name] = self.quadrics.distance_point_surface(tomo_id, f, coord)
            assigned_motl_df = pd.concat([assigned_motl_df, fm.df])

        assigned_motl = cryomotl.Motl(assigned_motl_df)
        assigned_motl.df.reset_index(drop=True, inplace=True)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    def assign_affiliation_distance_based(
        self,
        input_motl: MotlSource,
        output_path: PathOrStr | None = None,
        unassigned_value: float | None = None,
    ) -> "cryomotl.Motl":
        """Assign each particle to the nearest surface centre.

        Parameters
        ----------
        input_motl : MotlSource
            Particles to assign.
        output_path : PathOrStr, optional
            Path to save the result.
        unassigned_value : float, optional
            When provided, only particles whose current ``self.column_name``
            value equals this value are re-assigned; the rest are kept
            unchanged.

        Returns
        -------
        Motl
            Motl with ``self.column_name`` updated.
        """
        in_motl = cryomotl.Motl.load(input_motl)

        if unassigned_value is not None:
            assigned_motl = cryomotl.Motl(in_motl.df)
            in_motl.df = in_motl.df[in_motl.df[self.column_name] == unassigned_value]
            in_motl.df.reset_index(drop=True, inplace=True)

        tomos = in_motl.get_unique_values(column_name="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(column_values=t, column_name="tomo_id", reset_index=True)
            coord = tm.get_coordinates()
            closest_ids = self.quadrics.find_closest_quadric(t, coord)
            tm.df[self.column_name] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        if unassigned_value is not None:
            assigned_motl.df.loc[assigned_motl.df[self.column_name] == unassigned_value, :] = assigned_motl_df.values
        else:
            assigned_motl = cryomotl.Motl(assigned_motl_df)

        assigned_motl.df.reset_index(drop=True, inplace=True)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    # TODO: only Ellipsoid currently supported for intersection methods

    def assign_affiliation_intersection_based(
        self,
        input_motl: MotlSource,
        output_path: PathOrStr | None = None,
        keep_unassigned: bool = True,
    ) -> "cryomotl.Motl":
        """Assign each particle to the surface it points toward (ray casting).

        A ray is cast along the negated particle normal. The particle is
        labelled with the identifier of the surface whose intersection is
        closest along that ray. Particles that lie inside a surface or have
        no valid intersection receive ``-1``.

        Parameters
        ----------
        input_motl : MotlSource
            Particles to assign. Euler angles are used to derive normals.
        output_path : PathOrStr, optional
            Path to save the result.
        keep_unassigned : bool, default=True
            When ``False``, particles whose ``self.column_name`` is ``-1``
            are removed.

        Returns
        -------
        Motl
            Motl with ``self.column_name`` updated.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        tomos = in_motl.get_unique_values(column_name="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(column_values=t, column_name="tomo_id")
            coord = tm.get_coordinates()
            normal_vectors = -geom.euler_angles_to_normals(tm.get_angles())

            tomo_keys = [(tid, fid) for (tid, fid) in self.quadrics.dict if tid == t]
            num_points = coord.shape[0]
            closest_ids = np.full(num_points, -1)
            closest_distances = np.full(num_points, np.inf)

            for i in range(num_points):
                for tid, fid in tomo_keys:
                    params_array = self.quadrics.dict[(tid, fid)].params
                    _, _, d1, d2, is_inside = geom.ray_ellipsoid_intersection_3d(
                        coord[i, :], normal_vectors[i, :], params_array
                    )
                    if is_inside:
                        closest_distances[i] = np.inf
                        closest_ids[i] = -1
                        continue
                    distances_pos = [p for p in [d1, d2] if not np.isnan(p) and p > 0]
                    for d in distances_pos:
                        if abs(d) < abs(closest_distances[i]):
                            closest_distances[i] = d
                            closest_ids[i] = fid

            tm.df[self.column_name] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        unassigned = assigned_motl_df[assigned_motl_df[self.column_name] == -1].shape[0]

        if not keep_unassigned:
            assigned_motl_df = assigned_motl_df[assigned_motl_df[self.column_name] != -1]

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        print(f"{unassigned} particles did not have any intersection or were inside.")
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    def compute_intersection(self, input_motl: MotlSource) -> pd.DataFrame:
        """Compute ray-ellipsoid intersection distances for each particle.

        For every particle a ray is cast along ``-euler_angles_to_normals``
        and the two intersection distances with the assigned ellipsoid are
        returned.

        Parameters
        ----------
        input_motl : MotlSource
            Particles grouped by ``self.column_name``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``subtomo_id``, ``<self.column_name>``, ``d1``, ``d2``.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.column_name)
        intersection_points = pd.DataFrame(columns=["subtomo_id", self.column_name, "d1", "d2"])

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.column_name, reset_index=True)
            coord = fm.get_coordinates()
            normal_vectors = -geom.euler_angles_to_normals(fm.get_angles())
            tomo_id = fm.df["tomo_id"].values[0]
            key = (tomo_id, f)
            if key not in self.quadrics.dict:
                continue
            params_array = self.quadrics.dict[key].params
            for i in range(coord.shape[0]):
                _, _, d1, d2, _ = geom.ray_ellipsoid_intersection_3d(coord[i, :], normal_vectors[i, :], params_array)
                new_row = pd.Series(
                    {"subtomo_id": fm.df.iloc[i]["subtomo_id"], self.column_name: f, "d1": d1, "d2": d2}
                )
                intersection_points = pd.concat([intersection_points, new_row.to_frame().T], ignore_index=True)

        return intersection_points

    def compute_normals_angle(
        self,
        input_motl: MotlSource,
        store_column_name: MotlColumn = "geom4",
        output_path: PathOrStr | None = None,
    ) -> "cryomotl.Motl":
        """Compute the angle between each particle's orientation and the ellipsoid radial normal.

        The radial normal is the vector from the fitted ellipsoid centre to
        the particle. The stored value is the angle (degrees) between that
        vector and the particle's orientation normal.

        Parameters
        ----------
        input_motl : MotlSource
            Particles grouped by ``self.column_name``.
        store_column_name : MotlColumn, default='geom4'
            Column that receives the angle values.
        output_path : PathOrStr, optional
            Path to save the result.

        Returns
        -------
        Motl
            Input motl with ``store_column_name`` populated.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.column_name)
        assigned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.column_name, reset_index=True)
            coord = fm.get_coordinates()
            normals = geom.euler_angles_to_normals(fm.get_angles())
            tomo_id = fm.df["tomo_id"].values[0]
            key = (tomo_id, f)
            if key not in self.quadrics.dict:
                assigned_motl_df = pd.concat([assigned_motl_df, fm.df])
                continue
            center = self.quadrics.dict[key].center
            normals_t = coord - np.tile(center, (coord.shape[0], 1))
            fm.df[store_column_name] = geom.angle_between_n_vectors(normals, normals_t)
            assigned_motl_df = pd.concat([assigned_motl_df, fm.df])

        assigned_motl_df.index = in_motl.df.index
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    def clean_by_normals(
        self,
        input_motl: MotlSource,
        compute_normals: bool = True,
        normals_id: MotlColumn = "geom4",
        threshold: float | None = None,
        output_path: PathOrStr | None = None,
    ) -> "cryomotl.Motl":
        """Remove particles whose orientation deviates too far from the surface normal.

        Parameters
        ----------
        input_motl : MotlSource
            Source particles.
        compute_normals : bool, default=True
            Recompute the angle-to-normal column before filtering.
        normals_id : MotlColumn, default='geom4'
            Column holding the angle-to-normal values.
        threshold : float, optional
            Maximum allowed angle (degrees). Defaults to one standard deviation.
        output_path : PathOrStr, optional
            Path to save the result.

        Returns
        -------
        Motl
        """
        in_motl = cryomotl.Motl.load(input_motl)
        orig_number = in_motl.df.shape[0]

        if compute_normals:
            in_motl = self.compute_normals_angle(in_motl, store_column_name=normals_id)

        diff_angles = in_motl.df[normals_id].values
        to_remove = (
            np.where(np.abs(diff_angles) > np.std(diff_angles))
            if threshold is None
            else np.where(np.abs(diff_angles) > threshold)
        )

        mask = np.ones(len(in_motl.df), dtype=bool)
        mask[to_remove[0]] = False
        in_motl.df = in_motl.df.iloc[mask]
        in_motl.df.reset_index(drop=True, inplace=True)

        print(
            f"{orig_number - in_motl.df.shape[0]} particles "
            f"({((orig_number - in_motl.df.shape[0]) / orig_number * 100):.2f}%) were removed from the list."
        )
        if output_path is not None:
            in_motl.write_out(output_path)
        return in_motl

    def clean_by_radius(
        self,
        input_motl: MotlSource,
        threshold: float | None = None,
        output_path: PathOrStr | None = None,
    ) -> "cryomotl.Motl":
        """Remove particles that lie too far from the mean ellipsoid radius.

        Parameters
        ----------
        input_motl : MotlSource
            Source particles.
        threshold : float, optional
            Half-width of the allowed distance band. Defaults to one standard
            deviation.
        output_path : PathOrStr, optional
            Path to save the result.

        Returns
        -------
        Motl
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.column_name)
        cleaned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.column_name, reset_index=True)
            coord = fm.get_coordinates()
            tomo_id = fm.df["tomo_id"].values[0]
            key = (tomo_id, f)
            if key not in self.quadrics.dict:
                cleaned_motl_df = pd.concat([cleaned_motl_df, fm.df])
                continue
            el = self.quadrics.dict[key]
            center = el.center
            radius = float(np.mean(el.radii))
            distances = np.linalg.norm(coord - center, axis=1)
            thr = np.std(distances) if threshold is None else threshold
            mask = (distances >= radius - thr) & (distances <= radius + thr)
            fm.df = fm.df.iloc[mask]
            cleaned_motl_df = pd.concat([cleaned_motl_df, fm.df])

        cleaned_motl_df.reset_index(drop=True, inplace=True)
        cleaned_motl = cryomotl.Motl(cleaned_motl_df)
        print(
            f"{in_motl.df.shape[0] - cleaned_motl.df.shape[0]} particles "
            f"({((in_motl.df.shape[0] - cleaned_motl.df.shape[0]) / in_motl.df.shape[0] * 100):.2f}%) were removed."
        )
        if output_path is not None:
            cleaned_motl.write_out(output_path)
        return cleaned_motl

    @staticmethod
    def assign_affiliation_mask_based(
        input_motl: MotlSource,
        object_motl: MotlSource,
        tomo_dim: TomoDimensions,
        shell_size: int,
        column_name: MotlColumn = "object_id",
        output_path: PathOrStr | None = None,
        radius_offset: float = 0.0,
        motl_radius_id: MotlColumn = "geom5",
    ) -> "cryomotl.Motl":
        """Assign each particle to a surface object using a spherical-shell mask.

        Parameters
        ----------
        input_motl : MotlSource
            Particles to assign.
        object_motl : MotlSource
            Surface-object positions (one row per object).
        tomo_dim : TomoDimensions
            Tomogram dimensions; normalized via :func:`ioutils.dimensions_load`.
        shell_size : int
            Thickness of the spherical shell mask in voxels.
        column_name : MotlColumn, default='object_id'
            Identifier column on ``input_motl`` that receives the assigned
            object id.
        output_path : PathOrStr, optional
            Path to save the result.
        radius_offset : float, default=0.0
            Constant added to ``object_motl[motl_radius_id]`` to pad the shell.
        motl_radius_id : MotlColumn, default='geom5'
            Column in ``object_motl`` holding each object's radius.

        Returns
        -------
        Motl
        """
        in_motl = cryomotl.Motl.load(input_motl)
        object_motl = cryomotl.Motl.load(object_motl)
        tomo_dim = ioutils.dimensions_load(tomo_dim)
        tomos = in_motl.get_unique_values(column_name="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(column_values=t, column_name="tomo_id", reset_index=True)
            tm_dim = tomo_dim.loc[tomo_dim["tomo_id"] == t, ["x", "y", "z"]].values[0]
            coords = tm.get_coordinates().astype(int)
            tom = object_motl.get_motl_subset(column_values=t, column_name="tomo_id")
            for o in tom.get_unique_values(column_name=column_name):
                om = tom.get_motl_subset(column_values=o, column_name=column_name)
                om.df["class"] = 1
                to_radius = tom.df.iloc[0][motl_radius_id] + radius_offset
                object_mask = cryomask.generate_mask("s_shell_r" + str(int(to_radius)) + "_s" + str(int(shell_size)))
                tomo_mask = cryomap.place_object(object_mask, om, volume_shape=tm_dim, feature_to_color="class")
                mask_values = tomo_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
                idx_to_keep = np.where(mask_values == 1)[0]
                tm.df[column_name] = o
                assigned_motl_df = pd.concat([assigned_motl_df, tm.df.iloc[idx_to_keep]])

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    @staticmethod
    def create_spherical_oversampling(
        input_motl: MotlSource,
        motl_radius_id: MotlColumn,
        sampling_distance: float,
        sampling_angle: float = 360.0,
        output_path: PathOrStr | None = None,
    ) -> "cryomotl.Motl":
        """Generate oversampled particles on a sphere around each input particle.

        Parameters
        ----------
        input_motl : MotlSource
            Source particles (one sphere per row).
        motl_radius_id : MotlColumn
            Column holding the sphere radius for each particle.
        sampling_distance : float
            Angular sampling step forwarded to :func:`geom.sample_cone`.
        sampling_angle : float, default=360.0
            Half-opening angle of the sampling cone. ``360`` samples the full
            sphere.
        output_path : PathOrStr, optional
            Path to save the result.

        Returns
        -------
        Motl
        """
        motl = cryomotl.Motl.load(input_motl)
        new_motl_df = pd.DataFrame()
        for tomo in motl.get_unique_values("tomo_id"):
            tm = motl.get_motl_subset(tomo)
            coord = tm.get_coordinates()
            radii = tm.df[motl_radius_id].values
            objects = tm.df["object_id"].values
            for i, r in enumerate(radii):
                points = geom.sample_cone(sampling_angle, sampling_distance, center=coord[i, :], radius=r)
                normals = points - np.tile(coord[i, :], (points.shape[0], 1))
                angles = geom.normals_to_euler_angles(normals, output_order="zxz")
                em = cryomotl.Motl.create_empty_motl_df()
                em[["x", "y", "z"]] = points
                em[["phi", "theta", "psi"]] = angles
                em["object_id"] = objects[i]
                em["tomo_id"] = tomo
                em["class"] = 1
                new_motl_df = pd.concat((new_motl_df, em))

        new_motl_df.fillna(0, inplace=True)
        motl = cryomotl.Motl(new_motl_df)
        motl.update_coordinates()
        motl.renumber_particles()
        if output_path is not None:
            motl.write_out(output_path)
        return motl


# =============================================================================
# PolyhedralComplex — abstract base for T/O/I Platonic-solid motl analysis
# =============================================================================


class PolyhedralComplex(SymmetricComplex):
    """Abstract base for Platonic-solid (T/O/I) motl-analysis complexes.

    Do not instantiate this class directly.  Use one of the concrete
    subclasses:

    * :class:`TetrahedralComplex` — tetrahedral symmetry (T, order 12)
    * :class:`OctahedralComplex` — octahedral symmetry (O, order 24)
    * :class:`IcosahedralComplex` — icosahedral symmetry (I, order 60)

    Subclasses declare two class attributes:

    ``_solid``
        The :mod:`cryocat.utils.geom` solid class
        (e.g. ``geom.Icosahedron``).
    ``_symmetry``
        The symmetry letter ``"T"``, ``"O"``, or ``"I"``.

    All logic lives here; subclasses only override those two attributes.
    """

    _solid: type | None = None
    _symmetry: str | None = None

    # Instance-level geometry slot; set by fit_geometry.
    solid: "geom.Polyhedron | None" = None
    center: "np.ndarray | None" = None

    def __init__(
        self,
        motl: MotlSource,
        *,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
    ) -> None:
        if self._symmetry is None or self._solid is None:
            raise TypeError(
                "PolyhedralComplex is abstract; use TetrahedralComplex, " "OctahedralComplex, or IcosahedralComplex."
            )
        self.solid = None
        self.center = None
        self._pixel_size: float | None = None
        self._setup(
            motl,
            self._symmetry,
            affiliation_column=affiliation_column,
            order_column=order_column,
            tomo_id_column=tomo_id_column,
        )

    # ------------------------------------------------------------------
    # Geometry fitting
    # ------------------------------------------------------------------

    @gui_exposed(label="Fit geometry from markers", group="Geometry", order=10, returns="none")
    def fit_geometry(
        self,
        markers: PathOrStr,
        reference_map: PathOrStr,
        center: TripletLike | None = None,
    ) -> None:
        """Fit self.solid and self.center from two non-collinear markers.

        Reads the first two marker positions from *markers*, resolves the box
        centre from *reference_map* when *center* is None, and sets
        ``self.solid = self._solid.from_vectors(v1 - c, v2 - c)``.
        Pixel size is read from the map and used to convert marker coordinates
        (Å) to voxels before building the solid.
        """
        input_map_metadata = cryomap.get_metadata(reference_map)
        map_size = geom.as_triplet(input_map_metadata[0], dtype=int)
        pixel_size = input_map_metadata[1]
        center_vox = geom.as_triplet(center, reference_size=map_size)

        input_vert_marks = ioutils.marker_coords_load(markers)
        v1 = input_vert_marks.iloc[0].to_numpy() / pixel_size
        v2 = input_vert_marks.iloc[1].to_numpy() / pixel_size

        self.solid = self._solid.from_vectors(v1 - center_vox, v2 - center_vox)
        self.center = center_vox
        self._pixel_size = float(pixel_size)

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @gui_exposed(label="Assign subunit order", group="Affiliation", order=50, returns="none")
    def assign_subunit_order(self) -> None:
        """Assign 1-based subunit indices ordered by x→y→z (ascending)."""
        for (_tomo_id, _aff_id), group in self.motl.df.groupby([self.tomo_id_column, self.affiliation_column]):
            orig_idx = group.index.to_numpy()
            coords = self.motl.get_coordinates()[orig_idx]
            sorted_positions = np.lexsort((coords[:, 2], coords[:, 1], coords[:, 0]))
            ranks = np.empty(len(sorted_positions), dtype=int)
            ranks[sorted_positions] = np.arange(1, len(sorted_positions) + 1)
            self.motl.df.loc[orig_idx, self.order_column] = ranks

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    @gui_exposed(label="Feature vectors", group="Geometry", order=20, returns="features")
    def feature_vectors(
        self,
        mode: Literal["vertices", "edges", "faces"] = "vertices",
        project_to_sphere: bool = False,
        radius: float = 1.0,
    ) -> np.ndarray:
        """Return feature centres for the fitted solid (or a canonical one at *radius*).

        Uses ``self.solid`` when geometry has been fitted via :meth:`fit_geometry`;
        otherwise falls back to the canonical solid at *radius*.

        Parameters
        ----------
        mode : {"vertices", "edges", "faces"}, default="vertices"
            Which feature centres to return.
        project_to_sphere : bool, default=False
            When True, rescale each vector to the circumscribed sphere radius.
        radius : float, default=1.0
            Fallback radius when no geometry is fitted.

        Returns
        -------
        np.ndarray
            Array of shape (N, 3).
        """
        solid = self.solid if self.solid is not None else self._solid(radius)
        vecs = getattr(solid, mode)
        if project_to_sphere:
            norms = np.linalg.norm(vecs, axis=1)
            vecs = (vecs / norms[:, None]) * solid.radius
        return vecs

    @gui_exposed(label="Write features to CMM", group="Geometry", order=25, returns="none")
    def write_features_cmm(
        self,
        output_path: PathOrStr,
        mode: Literal["vertices", "edges", "faces"] = "vertices",
        project_to_sphere: bool = False,
    ) -> None:
        """Write feature vectors as ChimeraX marker coordinates to a .cmm file.

        Requires that :meth:`fit_geometry` has been called first.

        Parameters
        ----------
        output_path : PathOrStr
            Destination ``.cmm`` file path.
        mode : {"vertices", "edges", "faces"}, default="vertices"
            Feature type to write.
        project_to_sphere : bool, default=False
            Project features to the circumscribed sphere before writing.

        Raises
        ------
        ValueError
            When no geometry has been fitted yet.
        """
        if self.solid is None or self.center is None or self._pixel_size is None:
            raise ValueError("No geometry fitted. Call fit_geometry(markers, reference_map) first.")
        vecs = self.feature_vectors(mode=mode, project_to_sphere=project_to_sphere)
        features_coords = (vecs + self.center) * self._pixel_size
        ioutils.write_coords_to_cmm_file(features_coords, output_path)

    # ------------------------------------------------------------------
    # Symmetry expansion
    # ------------------------------------------------------------------

    @gui_exposed(
        label="Expand to subparticles",
        group="Expansion",
        order=30,
        returns="motl",
        hide=(),
    )
    def expand(
        self,
        *,
        mode: Literal["vertices", "edges", "faces"] = "vertices",
        project_to_sphere: bool = False,
        radius: float = 1.0,
        shift_vecs: np.ndarray | None = None,
        original_id_col: MotlColumn = "object_id",
        order_id_col: MotlColumn = "geom1",
        output_motl_type: MotlType = "emmotl",
        output_path: PathOrStr | None = None,
        **output_kwargs,
    ) -> MotlSource:
        """Expand each particle into subparticles at polyhedral feature positions.

        Parameters
        ----------
        mode : {"vertices", "edges", "faces"}, default="vertices"
            Feature type used when *shift_vecs* is None.
        project_to_sphere : bool, default=False
            Project features to the circumscribed sphere (passed to
            :meth:`feature_vectors`).
        radius : float, default=1.0
            Fallback solid radius when no geometry is fitted and *shift_vecs*
            is None.
        shift_vecs : np.ndarray of shape (N, 3), optional
            Explicit shift vectors.  When None, vectors are derived from
            :meth:`feature_vectors`.
        original_id_col : MotlColumn, default="object_id"
            Column that stores the ``subtomo_id`` of the source particle.
        order_id_col : MotlColumn, default="geom1"
            Column that stores the 0-based subunit extraction index.
        output_motl_type : MotlType, default="emmotl"
            Format of the returned/written motive list.
        output_path : PathOrStr, optional
            Write path.  No file is written when None.
        **output_kwargs
            Forwarded to :func:`cryocat.core.cryomotl.motl_converter_kwargs`.

        Returns
        -------
        MotlSource
            Expanded motive list in the requested format.

        Raises
        ------
        ValueError
            If *shift_vecs* has wrong shape or a required column is missing.
        """
        if shift_vecs is None:
            shift_vecs = self.feature_vectors(mode=mode, project_to_sphere=project_to_sphere, radius=radius)
        if not isinstance(shift_vecs, np.ndarray) or shift_vecs.ndim != 2 or shift_vecs.shape[1] != 3:
            raise ValueError("shift_vecs should be a numpy array of shape (N, 3)")
        for col in [original_id_col, order_id_col]:
            if col not in self.motl.df.columns:
                raise ValueError(f"original_id_col {col} not found in the columns of the input motive list")

        output_motl = expand_motl(
            self.motl,
            shift_vecs,
            original_id_col=original_id_col,
            order_id_col=order_id_col,
            sort_vectors=True,
            orientation="radial",
            start_index=0,
        )
        return cryomotl.motl_converter_kwargs(output_motl, output_motl_type, output_path=output_path, **output_kwargs)

    # ------------------------------------------------------------------
    # Feature recovery
    # ------------------------------------------------------------------

    @gui_exposed(
        label="Recover features",
        group="Geometry",
        order=90,
        returns="features",
        hide=("output_cmm_file",),
    )
    @classmethod
    def recover_features(
        cls,
        input_cmm_file: PathOrStr,
        input_map: PathOrStr,
        *,
        center: TripletLike | None = None,
        mode: Literal["vertices", "edges", "faces"] = "vertices",
        project_to_sphere: bool = False,
        output_cmm_file: PathOrStr | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover polyhedral feature coordinates from two marker positions.

        Must be called on a concrete subclass
        (``TetrahedralComplex.recover_features(…)``, etc.).

        Parameters
        ----------
        input_cmm_file : PathOrStr
            Path to a ``.cmm`` file with markers for two non-collinear vertices.
        input_map : PathOrStr
            Map used to prepare the marker file (supplies pixel size and box
            dimensions).
        center : TripletLike, optional
            Centre of the solid in map voxels.  Defaults to the box centre.
        mode : {"vertices", "edges", "faces"}, default="vertices"
            Feature type to recover.
        project_to_sphere : bool, default=False
            Project features to the circumscribed sphere.
        output_cmm_file : PathOrStr, optional
            If given, write recovered features to this ``.cmm`` file.

        Returns
        -------
        feature_vec : np.ndarray of shape (N, 3)
            Vectors from the solid centre to each feature.
        features_coords : np.ndarray of shape (N, 3)
            Feature positions in the map box (in Å).

        Raises
        ------
        TypeError
            When called on :class:`PolyhedralComplex` directly rather than on
            a concrete subclass.
        ValueError
            If *mode* is invalid.
        """
        if cls._solid is None:
            raise TypeError(
                "recover_features must be called on a concrete subclass "
                "(TetrahedralComplex, OctahedralComplex, or IcosahedralComplex)."
            )
        if mode not in ("vertices", "edges", "faces"):
            raise ValueError(f"Invalid mode: {mode}. Mode should be one of 'vertices', 'edges', or 'faces'.")

        input_map_metadata = cryomap.get_metadata(input_map)
        map_size = geom.as_triplet(input_map_metadata[0], dtype=int)
        center = geom.as_triplet(center, reference_size=map_size)

        input_vert_marks = ioutils.marker_coords_load(input_cmm_file)
        v1 = input_vert_marks.iloc[0].to_numpy() / input_map_metadata[1]
        v2 = input_vert_marks.iloc[1].to_numpy() / input_map_metadata[1]

        solid = cls._solid.from_vectors(v1 - center, v2 - center)
        feature_vec = getattr(solid, mode)

        if project_to_sphere:
            norms = np.linalg.norm(feature_vec, axis=1)
            feature_vec = (feature_vec / norms[:, None]) * solid.radius

        features_coords = np.add(feature_vec, center) * input_map_metadata[1]

        if output_cmm_file is not None:
            ioutils.write_coords_to_cmm_file(features_coords, output_cmm_file)

        return feature_vec, features_coords


# =============================================================================
# Concrete T/O/I subclasses
# =============================================================================


class TetrahedralComplex(PolyhedralComplex):
    """Tetrahedral (T, order 12) motl-analysis complex."""

    _solid = geom.Tetrahedron
    _symmetry = "T"


class OctahedralComplex(PolyhedralComplex):
    """Octahedral (O, order 24) motl-analysis complex."""

    _solid = geom.Octahedron
    _symmetry = "O"


class IcosahedralComplex(PolyhedralComplex):
    """Icosahedral (I, order 60) motl-analysis complex."""

    _solid = geom.Icosahedron
    _symmetry = "I"


# =============================================================================
# Module-level helpers for the ray-intersection workflow
# =============================================================================


def rays_from_motl(
    motl: "cryomotl.Motl",
    pixel_size: float,
    reverse_direction: bool = False,
    ray_length: float | None = None,
) -> np.ndarray:
    """Build a ray array from a Motl's particle positions and orientations.

    Coordinates are scaled by *pixel_size* to convert from voxels to physical
    units (e.g. nm) **without modifying the input Motl**. Normal vectors are
    derived from the particles' Euler angles via
    :func:`~cryocat.utils.geom.euler_angles_to_normals` (zxz convention,
    z-axis as the particle's forward axis).

    Parameters
    ----------
    motl : cryomotl.Motl
        Particle list. Coordinates must be in voxels.
    pixel_size : float
        Physical units per voxel (e.g. nm/voxel).
    reverse_direction : bool, default=False
        When ``True``, negate the normals before building ray directions so
        rays fly away from the surface rather than toward it.
    ray_length : float, optional
        Forwarded to :func:`~cryocat.utils.geom.construct_rays`. ``None``
        gives effectively infinite rays (magnitude 1e10).

    Returns
    -------
    numpy.ndarray
        Shape ``(N, 6)``; columns ``[ox, oy, oz, dx, dy, dz]``.
    """
    coords = motl.get_coordinates() * float(pixel_size)
    normals = geom.euler_angles_to_normals(motl.get_angles())
    return geom.construct_rays(
        points=coords,
        normals=normals,
        reverse_direction=bool(reverse_direction),
        ray_length=ray_length,
    )
