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
    def merge_subunits(self, radius: float = 55) -> None:
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
        occupancy_column: MotlColumn = "geom2",
        cone_distance_column: MotlColumn = "geom3",
        min_occupancy: int = 1,
        drop_below_min_occupancy: bool = False,
    ) -> "cryomotl.Motl":
        """Cluster subunit particles into objects and write affiliation labels.

        Operates on a copy of ``self.motl`` and returns it with
        ``affiliation_column`` populated.  After assigning affiliation the
        method also:

        * writes subunit indices into ``order_column`` via
          :meth:`assign_subunit_order`,
        * writes the per-object particle count into ``occupancy_column``,
        * computes each particle's cone-distance to its object's consensus
          z-axis and stores it in ``cone_distance_column``,
        * emits a :class:`UserWarning` for any object that exceeds
          ``self.n_subunits`` subunits, suggesting :meth:`clean_per_object`
          as a remedy,
        * optionally drops outlier-normal particles (``normals_threshold``)
          and/or objects below a minimum size (``drop_below_min_occupancy``).

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
        normals_threshold : float, optional
            Per-object cone-distance cutoff (degrees).  Particles whose
            cone-distance to the object's consensus z-axis exceeds this
            value are dropped.  When ``None`` the cone distances are stored
            for inspection but no particles are removed.
        occupancy_column : MotlColumn, default='geom2'
            Column that receives the per-object particle count.
        cone_distance_column : MotlColumn, default='geom3'
            Column that receives each particle's cone distance (degrees) to
            its object's consensus z-axis.
        min_occupancy : int, default=1
            Minimum object size used by ``drop_below_min_occupancy``.
        drop_below_min_occupancy : bool, default=False
            When ``True``, remove objects whose size after all filtering is
            below *min_occupancy*.  When ``False`` all objects (including
            singletons) are kept.

        Returns
        -------
        cryomotl.Motl
            A new motl with ``affiliation_column``, ``order_column``,
            ``occupancy_column``, and ``cone_distance_column`` populated.

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

        # Assign subunit order via motl-swap
        orig_motl = self.motl
        self.motl = motl_out
        self.assign_subunit_order()
        motl_out = self.motl
        self.motl = orig_motl

        # Per-object occupancy count
        motl_out.df[occupancy_column] = motl_out.df.groupby([self.tomo_id_column, self.affiliation_column])[
            self.affiliation_column
        ].transform("size")

        # Cone distance to per-object consensus z-axis (mirrors geom.cone_distance)
        motl_out.df[cone_distance_column] = 0.0
        for (_t, _o), grp in motl_out.df.groupby([self.tomo_id_column, self.affiliation_column]):
            euler = grp[["phi", "theta", "psi"]].to_numpy()
            z_axes = srot.from_euler("zxz", euler, degrees=True).apply([0.0, 0.0, 1.0])
            mean_z = z_axes.mean(axis=0)
            norm = np.linalg.norm(mean_z)
            mean_z = mean_z / norm if norm > 0 else np.array([0.0, 0.0, 1.0])
            dots = np.clip(z_axes @ mean_z, -1.0, 1.0)
            motl_out.df.loc[grp.index, cone_distance_column] = np.degrees(np.arccos(dots))

        # Normals threshold: drop per-object outliers
        if normals_threshold is not None:
            keep = motl_out.df[cone_distance_column] <= normals_threshold
            motl_out = cryomotl.Motl(motl_out.df[keep].reset_index(drop=True))

        # Over-occupancy warning
        sizes = motl_out.df.groupby([self.tomo_id_column, self.affiliation_column]).size()
        over = sizes[sizes > self.n_subunits]
        if not over.empty:
            obj_list = ", ".join(f"tomo={t} obj={o}" for t, o in over.index)
            warnings.warn(
                f"{type(self).__name__}.create_affiliation: {len(over)} object(s) exceed "
                f"n={self.n_subunits} subunits ({obj_list}). Use clean_per_object() to reduce.",
                UserWarning,
                stacklevel=2,
            )

        # Optionally prune small objects
        if drop_below_min_occupancy:
            keep = motl_out.df[occupancy_column] >= min_occupancy
            motl_out = cryomotl.Motl(motl_out.df[keep].reset_index(drop=True))

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

        Sets ``self.n`` to the half-ring fold and ``self._ring_group_columns``
        to ``[tomo_id_column, affiliation_column]``.  Called from
        :meth:`CnComplex.__init__` and :meth:`DnComplex.__init__`.
        """
        self.n: int = self.fold
        self._ring_group_columns: list[MotlColumn] = [
            self.tomo_id_column,
            self.affiliation_column,
        ]

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
    def circumference(self, *, pixel_size: float = 1.0) -> pd.DataFrame:
        """Per-object circumference derived from the circumradius.

        Computes ``2 π × circumradius × pixel_size`` for each object.
        The circumradius is estimated via the Pratt circle fit (≥ 4 particles)
        or falls back to the mean particle–to–barycenter distance.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor.

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

        coord = self.motl.get_coordinates()
        rows: list[dict] = []

        for keys, group in self.motl.df.groupby(self._ring_group_columns):
            tomo_id, object_id = keys[0], keys[1]
            coords_grp = coord[group.index.to_numpy(), :]
            r = self._circumradius_for_group(coords_grp)
            row: dict = {
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "circumference": 2.0 * np.pi * r * pixel_size,
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

        if max_residual > delta / 2.0:
            warnings.warn(
                f"{type(self).__name__}: max grid residual {max_residual:.1f}° exceeds"
                f" {delta / 2.0:.1f}° for object {obj_id!r}. Ring may be distorted.",
                UserWarning,
                stacklevel=3,
            )

        dupes = {k: v for k, v in Counter(s_idx).items() if v > 1}
        if dupes:
            warnings.warn(
                f"{type(self).__name__}: duplicate subunit indices for object"
                f" {obj_id!r}: {dupes!r}. Ring may be distorted.",
                UserWarning,
                stacklevel=3,
            )

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

    # ------------------------------------------------------------------
    # Central-angle analysis
    # ------------------------------------------------------------------

    @gui_exposed(label="Central angles", group="Geometry", order=35, returns="dataframe")
    def central_angles(
        self,
        gaps: str = "holey",
        inward_axis: np.ndarray | None = None,
    ) -> pd.DataFrame:
        """Compute central angles between neighbouring subunit pairs.

        For each ring group two methods are evaluated for every consecutive
        pair (query particle → nearest in ring order):

        * **Positional** (HC1): ``|atan2(n · (u × v), u · v)|`` where
          ``u = S1 − C`` and ``v = S2 − C``, with *C* the fitted ring centre
          and *n* the ring normal from SVD of displacement vectors.
        * **Orientational** (HC1b): same formula applied to the inward-axis
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
            Pairing mode forwarded to
            :meth:`~nnana.NearestNeighbors.ordered_pairs`.
            ``'holey'`` pairs every present subunit with the next present one
            (skipping absent indices); ``'full'`` skips pairs whose target
            index is absent.
        inward_axis : array-like of shape (3,) or None, default=None
            Local axis pointing toward the ring centre in the particle frame.
            ``None`` uses ``[-1, 0, 0]``, the convention in
            :meth:`_compute_object_center` and :meth:`assign_subunit_order`.

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
            ``angle_diff = |angle_pos − angle_ori|`` (unsigned disagreement).
            ``dev_pos`` and ``dev_ori`` are signed deviations from the ideal
            ``idx_diff × central_angle`` (positive = wider than ideal,
            negative = narrower).
            ``angle_pos_signed`` / ``angle_ori_signed`` retain the sign from
            ``atan2``; their sign depends on the SVD normal orientation and is
            arbitrary between rings — use only for within-ring consistency
            checks, never for cross-ring comparison.

        Raises
        ------
        ValueError
            If ``affiliation_column`` is absent from ``self.motl.df``.
        ValueError
            If ``order_column`` is absent from ``self.motl.df``.
        """
        self._require_affiliation()
        if self.order_column not in self.motl.df.columns:
            raise ValueError(f"order_column {self.order_column!r} not in motl; " "call assign_subunit_order() first.")

        ia = np.array([-1.0, 0.0, 0.0]) if inward_axis is None else np.asarray(inward_axis, dtype=float)
        norm = np.linalg.norm(ia)
        if norm > 1e-10:
            ia = ia / norm

        rows: list[dict] = []

        for keys, group in self.motl.df.groupby(self._ring_group_columns):
            tomo_id = keys[0]
            object_id = keys[1]
            om = cryomotl.Motl(group.reset_index(drop=True))

            if len(om.df) < 2:
                continue

            # Centre and ring normal
            center, _ = self._compute_object_center(om)
            su_coord = om.get_coordinates()
            vectors = su_coord - np.tile(center, (su_coord.shape[0], 1))
            _, _, Vt = np.linalg.svd(vectors, full_matrices=True)
            ring_normal = Vt[-1]

            # Ordered pairs for this group
            nn_result = nnana.NearestNeighbors.ordered_pairs(
                om,
                group_column=self.affiliation_column,
                order_column=self.order_column,
                step=1,
                topology="circular",
                ring_size=self.n,
                gaps=gaps,
            )

            if nn_result.df.empty:
                continue

            # subtomo_id → order index
            sid_to_idx: dict[float, int] = {
                float(r["subtomo_id"]): int(r[self.order_column]) for _, r in group.iterrows()
            }

            for _, pair in nn_result.df.iterrows():
                qp_sub = float(pair["qp_subtomo_id"])
                nn_sub = float(pair["nn_subtomo_id"])
                qp_coord = np.array(
                    [
                        float(pair["qp_coord_x"]),
                        float(pair["qp_coord_y"]),
                        float(pair["qp_coord_z"]),
                    ],
                    dtype=float,
                )
                nn_coord = np.array(
                    [
                        float(pair["nn_coord_x"]),
                        float(pair["nn_coord_y"]),
                        float(pair["nn_coord_z"]),
                    ],
                    dtype=float,
                )

                # Positional angle — signed for diagnostics, absolute for output (HC1)
                u = qp_coord - center
                v = nn_coord - center
                signed_pos = np.degrees(geom.vector_angular_distance_signed(u, v, ring_normal))
                angle_pos = abs(signed_pos)

                # Orientational angle — same treatment (HC1b)
                qp_rot = srot.from_euler(
                    "zxz",
                    [
                        float(pair["qp_angles_phi"]),
                        float(pair["qp_angles_theta"]),
                        float(pair["qp_angles_psi"]),
                    ],
                    degrees=True,
                )
                nn_rot = srot.from_euler(
                    "zxz",
                    [
                        float(pair["nn_angles_phi"]),
                        float(pair["nn_angles_theta"]),
                        float(pair["nn_angles_psi"]),
                    ],
                    degrees=True,
                )
                qp_inward = qp_rot.apply(ia)
                nn_inward = nn_rot.apply(ia)
                signed_ori = np.degrees(geom.vector_angular_distance_signed(qp_inward, nn_inward, ring_normal))
                angle_ori = abs(signed_ori)

                # Index difference (circular, 1 ≤ idx_diff ≤ n-1 normally)
                qp_idx: int | None = sid_to_idx.get(qp_sub)
                nn_idx: int | None = sid_to_idx.get(nn_sub)

                if qp_idx is not None and nn_idx is not None:
                    raw_diff = nn_idx - qp_idx
                    idx_diff_val: int = raw_diff if raw_diff > 0 else raw_diff + self.n
                    ideal = idx_diff_val * self.central_angle
                    angle_pos_per_pos = angle_pos / idx_diff_val
                    angle_ori_per_pos = angle_ori / idx_diff_val
                    dev_pos = angle_pos - ideal  # signed: positive = wider, negative = narrower
                    dev_ori = angle_ori - ideal
                else:
                    idx_diff_val = None
                    ideal = float("nan")
                    angle_pos_per_pos = float("nan")
                    angle_ori_per_pos = float("nan")
                    dev_pos = float("nan")
                    dev_ori = float("nan")

                rows.append(
                    {
                        "tomo_id": float(tomo_id),
                        "object_id": float(object_id),
                        "qp_subtomo_id": qp_sub,
                        "nn_subtomo_id": nn_sub,
                        "qp_idx": float(qp_idx) if qp_idx is not None else float("nan"),
                        "nn_idx": float(nn_idx) if nn_idx is not None else float("nan"),
                        "idx_diff": float(idx_diff_val) if idx_diff_val is not None else float("nan"),
                        "angle_pos": angle_pos,
                        "angle_ori": angle_ori,
                        "angle_diff": abs(angle_pos - angle_ori),
                        "angle_pos_per_pos": angle_pos_per_pos,
                        "angle_ori_per_pos": angle_ori_per_pos,
                        "dev_pos": dev_pos,
                        "dev_ori": dev_ori,
                        "angle_pos_signed": signed_pos,
                        "angle_ori_signed": signed_ori,
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
    def get_object_stats(self, *, pixel_size: float = 1.0) -> pd.DataFrame:
        """Comprehensive per-object statistics table.

        Composes :meth:`occupancy`, :meth:`circumference`,
        :meth:`diameter`, and centre/radius computation into one row per
        ``(tomo_id, object_id)`` group.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor for distance columns.

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

        occ_df = self.occupancy()
        circ_df = self.circumference(pixel_size=pixel_size)
        diam_df, _ = self.diameter(pixel_size=pixel_size)

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
    def ring_spacing(self, *, pixel_size: float = 1.0) -> pd.DataFrame:
        """Axial distance between the two rings for each object.

        Computes the mean position of ring 0 and ring 1 subunits separately
        and returns the absolute axial distance between them projected onto
        ``self._split_axis``.

        Parameters
        ----------
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor.

        Returns
        -------
        pandas.DataFrame
            Columns ``tomo_id``, ``object_id``, ``ring_spacing``.
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

            if not mask0.any() or not mask1.any():
                spacing = np.nan
            else:
                c0 = np.mean(coords_grp[mask0, :], axis=0)
                c1 = np.mean(coords_grp[mask1, :], axis=0)
                spacing = float(abs(np.dot(c0 - c1, self._split_axis)) * pixel_size)

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

        For each ring, projects subunit positions onto the plane perpendicular
        to ``self._split_axis`` and computes the n-fold circular mean phase:
        ``angle(Σ exp(i·n·θ_k)) / n``.  The twist is the phase difference
        ring 1 − ring 0, wrapped into ``[0, 2π/n)``.

        A perfectly staggered arrangement gives ``180 / n`` degrees; an
        eclipsed arrangement gives ``0`` degrees.

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

        axis = self._split_axis
        e1 = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(e1, axis)) > 0.9:
            e1 = np.array([0.0, 1.0, 0.0])
        e1 = e1 - np.dot(e1, axis) * axis
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(axis, e1)

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
                twist = np.nan
            else:

                def _ring_phase(rel_grp: np.ndarray) -> float:
                    angles = np.arctan2(rel_grp @ e2, rel_grp @ e1)
                    z = np.sum(np.exp(1j * self.n * angles))
                    return float(np.angle(z) / self.n)

                phase0 = _ring_phase(rel[mask0, :])
                phase1 = _ring_phase(rel[mask1, :])
                central_angle_rad = 2.0 * np.pi / self.n
                twist_rad = (phase1 - phase0) % central_angle_rad
                twist = float(np.degrees(twist_rad)) if degrees else float(twist_rad)

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
    def get_object_stats(self, *, pixel_size: float = 1.0) -> pd.DataFrame:
        """Comprehensive per-object statistics for dihedral structures.

        Composes per-object occupancy, ring spacing, inter-ring twist, and
        per-ring diameter/circumference (averaged to one row per object).

        Parameters
        ----------
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor for distance columns.

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

        occ_df = self.occupancy()
        spacing_df = self.ring_spacing(pixel_size=pixel_size)
        twist_df = self.inter_ring_twist(degrees=True)

        circ_df_ring = self.circumference(pixel_size=pixel_size)
        diam_df_ring, _ = self.diameter(pixel_size=pixel_size)

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

    Typical workflow:

    1. :meth:`cluster_subunits_to_rings` — trace subunits into rings and
       merge nearby rings.
    2. :meth:`unify_nn_orientations` — flip ambiguous orientations.
    3. :meth:`merge_rings` — merge rings from multiple ring-motls.
    """

    def __init__(
        self,
        motl: MotlSource,
        symmetry=None,
        *,
        affiliation_column: MotlColumn = "object_id",
        order_column: MotlColumn = "geom1",
        tomo_id_column: MotlColumn = "tomo_id",
        center_method: Literal["circle_fit", "barycentric"] = "circle_fit",
    ) -> None:
        # NPC always uses C8 symmetry; symmetry param accepted for compat only.
        super().__init__(
            motl,
            "C8",
            affiliation_column=affiliation_column,
            order_column=order_column,
            tomo_id_column=tomo_id_column,
            center_method=center_method,
        )

    @staticmethod
    def cluster_subunits_to_rings(
        input_motl: MotlSource,
        npc_radius: float,
        max_trace_distance: float,
        min_trace_distance: float = 0,
        *,
        mask_size: TripletLike | None = None,
        entry_mask_coord: TripletLike | None = None,
        exit_mask_coord: TripletLike | None = None,
        entry_mask: MapSource | None = None,
        exit_mask: MapSource | None = None,
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

        Returns
        -------
        Motl
            Motl with ``object_id`` identifying each ring, ``geom1`` holding
            ring occupancy, and ``geom2`` the within-ring subunit index.

        Raises
        ------
        ValueError
            If, for either side, neither the mask nor the
            ``(coord + mask_size)`` pair was supplied.
        """
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
        chain.traced_motl.df.sort_values(["tomo_id", "object_id", "geom2"], inplace=True)
        chain.get_occupancy()
        motl = chain.add_traced_info(motl)

        return NPC._merge_by_radius(motl, npc_radius)

    # ------------------------------------------------------------------
    # Orientation unification
    # ------------------------------------------------------------------

    @gui_exposed(label="Unify NN orientations", group="Affiliation", order=50, returns="none")
    def unify_nn_orientations(self, dist_threshold: float = 10000) -> None:
        """Flip orientations so that neighbouring subunits point consistently.

        Traces particles into chains via :func:`nnana.trace_chains`, then walks
        each chain and applies a 180° rotation around X whenever the cone angle
        between successive subunits exceeds 90°.  Updates ``self.motl``
        in place.

        Parameters
        ----------
        dist_threshold : float, default=10000
            Maximum nearest-neighbour distance for tracing (voxels).
        """
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
            Chain-traced motl with ``object_id`` and ``geom2`` populated.
        npc_radius : float
            Distance threshold for merging (voxels).

        Returns
        -------
        cryomotl.Motl
            Updated motl with consolidated ring labels.
        """
        for t in motl.get_unique_values("tomo_id"):
            tm = motl.get_motl_subset(column_values=[t], column_name="tomo_id", reset_index=True)

            # Build centres motl using radius-shift approach
            central_points: list[np.ndarray] = []
            obj_ids: list[float] = []
            for o in tm.get_unique_values("object_id"):
                om = tm.get_motl_subset(column_values=[o], column_name="object_id", reset_index=True)
                central_points.append(NPC._center_by_radius_shift(om, npc_radius))
                obj_ids.append(o)

            centers_motl = cryomotl.Motl()
            if central_points:
                ca = np.vstack(central_points)
                centers_motl.fill({"x": ca[:, 0], "y": ca[:, 1], "z": ca[:, 2], "tomo_id": t, "object_id": obj_ids})
                centers_motl.renumber_particles()
            centers_motl.df.fillna(0.0, inplace=True)

            changed_objects: list[float] = []
            if centers_motl.df.shape[0] > 1:
                center_stats = nnana.get_nn_stats(centers_motl, centers_motl)
                if any(center_stats["distance"] <= npc_radius):
                    center_idx, nn_idx_list = nnana.get_nn_within_distance(centers_motl, npc_radius)
                    for i, pos in enumerate(center_idx):
                        o_id1 = centers_motl.df.loc[centers_motl.df.index[pos], "object_id"]
                        changed_objects.append(o_id1)
                        for j in nn_idx_list[i]:
                            o_id2 = centers_motl.df.loc[centers_motl.df.index[j], "object_id"]
                            tm.df.loc[tm.df["object_id"] == o_id2, "object_id"] = o_id1

            tm.df["geom1"] = tm.df.groupby("object_id")["object_id"].transform("count")
            for o in changed_objects:
                om = tm.get_motl_subset(column_values=o, column_name="object_id", reset_index=True)
                s_idx = NPC._assign_subunit_index(om, npc_radius)
                tm.df.loc[tm.df["object_id"] == o, "geom2"] = s_idx

            tm.df["object_id"] = tm.df["object_id"].rank(method="dense").astype(int)
            motl.df.loc[motl.df["tomo_id"] == t, ["object_id", "geom1", "geom2"]] = tm.df[
                ["object_id", "geom1", "geom2"]
            ].values

        motl.df.reset_index(inplace=True, drop=True)
        motl.df["geom1"] = motl.df.groupby(["tomo_id", "object_id"])["object_id"].transform("count")
        motl.df["object_id"] = motl.df["object_id"].rank(method="dense").astype(int)
        return motl

    @staticmethod
    def compute_diameter(
        input_motl: MotlSource,
        *,
        pixel_size: float = 1.0,
        store_column: MotlColumn = "geom4",
        symmetry: int = 8,
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
            Particle list with NPC subunits.  Requires ``object_id`` for ring
            affiliation and ``geom2`` for the 1-based subunit order within
            each ring.
        pixel_size : float, default=1.0
            Ångström-per-voxel scale factor applied to all distances.
        store_column : MotlColumn, default='geom4'
            Column in the returned motl that carries each ring's mean
            diameter.  ``NaN`` for rings with no opposite-pair matches.
        symmetry : int, default=8
            Rotational symmetry order.  Determines the pair offset
            ``symmetry // 2``.

        Returns
        -------
        summary_df : pandas.DataFrame
            One row per ``(tomo_id, object_id)`` that produced at least one
            opposite-subunit pair.  Columns:
            ``tomo_id``, ``object_id``, ``mean_diameter``, ``n_pairs``.
            Empty when no ring has matching pairs.
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

        for (tomo_id, object_id), group in motl_out.df.groupby(["tomo_id", "object_id"]):
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
                    "tomo_id": float(tomo_id),
                    "object_id": float(object_id),
                    "mean_diameter": mean_d,
                    "n_pairs": int(len(dists)),
                }
            )

        motl_out.df[store_column] = diameters_col
        summary_df = pd.DataFrame(
            rows if rows else [],
            columns=["tomo_id", "object_id", "mean_diameter", "n_pairs"],
        )
        return summary_df, motl_out

    @staticmethod
    def get_centers_as_motl(
        tomo_motl: MotlSource,
        *,
        tomo_id: float | None = None,
        radius: float = 55.0,
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
            ``tomo_id`` value found on each ring's particles.
        radius : float, default=55.0
            Approximate NPC ring radius in voxels.

        Returns
        -------
        Motl
            One row per unique ``object_id`` with the estimated ring centre
            in ``x``, ``y``, ``z``.
        """
        motl = cryomotl.Motl.load(tomo_motl)
        centers: list[np.ndarray] = []
        tomo_ids: list[float] = []
        object_ids: list[float] = []

        for o in motl.get_unique_values("object_id"):
            om = motl.get_motl_subset(column_values=[o], column_name="object_id", reset_index=True)
            center = NPC._center_by_radius_shift(om, npc_radius=radius)
            centers.append(center)
            t = float(tomo_id) if tomo_id is not None else float(om.df["tomo_id"].iloc[0])
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
                    "tomo_id": tomo_ids,
                    "object_id": object_ids,
                }
            )
            result.renumber_particles()
        result.df.fillna(0.0, inplace=True)
        return result

    @staticmethod
    def merge_rings(
        input_motls: list[MotlSource],
        npc_radius: float,
        distance_threshold: float = 40,
    ) -> list["cryomotl.Motl"]:
        """Merge corresponding rings across multiple ring-motls.

        Assigns sequential ``object_id`` values across all motls, then for
        every pair of motls finds rings (by their estimated centres) that are
        closer than *distance_threshold* and merges their ``object_id``
        entries.

        Parameters
        ----------
        input_motls : list of MotlSource
            At least two ring-motls to merge.
        npc_radius : float
            Ring radius in voxels, forwarded to :meth:`get_centers_as_motl`.
        distance_threshold : float, default=40
            Maximum centre-to-centre distance (voxels) for two rings from
            different motls to be considered the same NPC.

        Returns
        -------
        list of Motl
            The input motls with updated ``object_id`` values so that matched
            rings share the same identifier.

        Raises
        ------
        UserWarning
            When *input_motls* is not a list or contains fewer than two items.
        """
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
            starting_number = r.df["object_id"].max() + 1

        ring_pairs = mathutils.get_all_pairs(np.arange(len(ring_motls)))

        for i in ring_pairs:
            for t in ring_motls[i[0]].get_unique_values("tomo_id"):
                tm1 = ring_motls[i[0]].get_motl_subset(column_values=[t], column_name="tomo_id", reset_index=True)
                tm2 = ring_motls[i[1]].get_motl_subset(column_values=[t], column_name="tomo_id", reset_index=True)
                if tm2.df.shape[0] > 0:
                    centers1 = CnComplex(tm1, symmetry=8).get_centers_as_motl()
                    centers2 = CnComplex(tm2, symmetry=8).get_centers_as_motl()

                    _, obj1_idx, distances, _ = nnana.find_nn_indices(
                        centers2.get_coordinates(),
                        centers1.get_coordinates(),
                        k=1,
                    )
                    distances = distances.reshape(-1)
                    obj1_idx = obj1_idx.reshape(-1)

                    close_idx = distances < distance_threshold
                    if np.all(~close_idx):
                        continue
                    obj1_idx = obj1_idx[close_idx]
                    obj2_idx = np.arange(centers2.df.shape[0])[close_idx]
                    for o1, o2 in zip(obj1_idx, obj2_idx):
                        obj1_id = centers1.df.loc[centers1.df.index[o1], "object_id"]
                        obj2_id = centers2.df.loc[centers2.df.index[o2], "object_id"]
                        ring_motls[i[1]].df.loc[
                            (ring_motls[i[1]].df["tomo_id"] == t) & (ring_motls[i[1]].df["object_id"] == obj2_id),
                            "object_id",
                        ] = obj1_id

        return ring_motls


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
    fold: int | None = None

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
        if self.fold is not None and self.fold != self.n_sites:
            raise ValueError(f"fold={self.fold} must equal n_sites={self.n_sites} when set.")

    @classmethod
    def cyclic(
        cls,
        symmetry: "Symmetry",
        site_shift: TripletLike,
        site_type: str = "site",
    ) -> "BlockDefinition":
        """Create a C_n-symmetric block with equally-spaced sites.

        Site k (1-based) is *site_shift* rotated about +z by
        ``360·(k−1)/n`` degrees — the same convention as
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
            A frozen :class:`BlockDefinition` with *n* equally-spaced sites
            and :attr:`fold` equal to *n*.

        Examples
        --------
        >>> BlockDefinition.cyclic(3, [-5, 0, 0])      # C3, arm along −x
        >>> BlockDefinition.cyclic("C6", [-3, 0, -1])  # C6, tilted arm
        """
        group, n = geom.as_symmetry(symmetry)
        if group != "C":
            raise ValueError(f"cyclic() requires a Cn symmetry specifier; got '{group}{n}'.")
        shift = geom.as_triplet(site_shift)
        sites = tuple(
            ContactSite(
                vector=tuple(float(c) for c in srot.from_euler("z", 360.0 * k / n, degrees=True).apply(shift)),
                site_type=site_type,
            )
            for k in range(n)
        )
        return cls(sites=sites, pairing=((site_type, site_type),), fold=n)

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
        """Number of contact sites."""
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
        blocks: MotlSource | None = None,
        block_definition: "BlockDefinition | dict[float, BlockDefinition] | None" = None,
        block_type_column: MotlColumn = "class",
        pixel_size: float = 1.0,
        ideal_degree: int | None = None,
        ideal_face_size: int | None = None,
        tomo_id_column: MotlColumn = "tomo_id",
    ) -> None:
        """Create a :class:`PleomorphicSurface`.

        Parameters
        ----------
        surface : Mesh, OrientedPointCloud, PleomorphicSurface, or None
            Envelope surface.  When a :class:`PleomorphicSurface` is passed,
            its ``_surface`` is extracted; if *blocks* is also ``None``, the
            block layer is copied from it as well.  ``None`` means no envelope.
        blocks : MotlSource or None, default=None
            Block particle list.  Requires *block_definition*.
        block_definition : BlockDefinition or dict[float, BlockDefinition] or None
            Geometry for one class of block (or a mapping from block-type float
            values to per-class definitions for mixed-fold lattices).
        block_type_column : MotlColumn, default="class"
            Column in *blocks* that identifies the block class when
            *block_definition* is a dict.
        pixel_size : float, default=1.0
            Pixel size in Å (or any consistent unit); used when converting block
            coordinates to physical units for the envelope mesh.
        ideal_degree : int or None, default=None
            Ideal vertex degree *d* of the lattice.  Must be paired with
            *ideal_face_size*.  When both are ``None`` and *block_definition*
            has *n_sites* ∈ {3, 4, 6}, the pair is derived as
            *(d, 2d/(d−2))* — for d=3 → (3, 6), d=4 → (4, 4), d=6 → (6, 3).
        ideal_face_size : int or None, default=None
            Ideal face size *f* of the lattice.  Must satisfy
            ``1/d + 1/f = 1/2``; the constructor enforces this.
        tomo_id_column : MotlColumn, default="tomo_id"
            Column identifying the tomogram each block belongs to.

        Raises
        ------
        TypeError
            If neither *surface* nor *blocks* is provided, or if *surface* has
            an unsupported type.
        ValueError
            If *blocks* is provided without *block_definition*, if
            *ideal_degree* and *ideal_face_size* are inconsistent, if the
            block-type column contains values without a matching definition,
            or if ``subtomo_id`` is not unique in *blocks*.

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

        *blocks* is loaded with :func:`~cryocat.core.cryomotl.Motl.load`,
        which copies a :class:`~cryocat.core.cryomotl.Motl` instance; the
        input is never modified.  :meth:`unify_polarity`,
        :meth:`store_block_stats` and :meth:`store_block_stat` change only
        this object's copy (angles, or the requested columns; row order and
        index are kept).  Use :meth:`get_blocks_as_motl` to obtain the
        current state, e.g. unified orientations for averaging or for
        building a new assembly.
        """
        self._surface: Mesh | OrientedPointCloud | None = None
        self.blocks: cryomotl.Motl | None = None
        self.block_definition: "BlockDefinition | dict[float, BlockDefinition] | None" = None
        self.block_type_column: MotlColumn = block_type_column
        self.pixel_size: float = pixel_size
        self.ideal_degree: int | None = ideal_degree
        self.ideal_face_size: int | None = ideal_face_size
        self.tomo_id_column: MotlColumn = tomo_id_column
        self._site_table: pd.DataFrame | None = None
        self._faces: np.ndarray | None = None

        # --- Envelope ---
        if isinstance(surface, PleomorphicSurface):
            self._surface = surface._surface
            if blocks is None:
                self.blocks = copy.deepcopy(surface.blocks)
                self.block_definition = surface.block_definition
                self.block_type_column = surface.block_type_column
                self.pixel_size = surface.pixel_size
                self.ideal_degree = surface.ideal_degree
                self.ideal_face_size = surface.ideal_face_size
                self.tomo_id_column = surface.tomo_id_column
        elif surface is not None:
            if not isinstance(surface, (Mesh, OrientedPointCloud)):
                raise TypeError(
                    f"Unsupported surface type: {type(surface)}. "
                    "Must be Mesh, OrientedPointCloud, or PleomorphicSurface."
                )
            self._surface = surface

        if self._surface is None and blocks is None and self.blocks is None:
            raise TypeError(
                "PleomorphicSurface requires at least a surface (Mesh or " "OrientedPointCloud) or a blocks motl."
            )

        # --- Ideal lattice check ---
        if (self.ideal_degree is None) != (self.ideal_face_size is None):
            raise ValueError("ideal_degree and ideal_face_size must both be None or both set.")
        if self.ideal_degree is not None:
            check = 1.0 / self.ideal_degree + 1.0 / self.ideal_face_size
            if abs(check - 0.5) > 1e-9:
                raise ValueError(f"1/ideal_degree + 1/ideal_face_size must equal 1/2, " f"got {check:.10f}.")

        # --- Blocks ---
        if blocks is not None:
            if block_definition is None:
                raise ValueError("block_definition is required when blocks is provided.")
            loaded = cryomotl.Motl.load(blocks)
            if loaded.df["subtomo_id"].duplicated().any():
                raise ValueError("subtomo_id must be unique in the blocks motl.")
            self.blocks = loaded
            self.block_definition = block_definition
            self.block_type_column = block_type_column
            self.pixel_size = pixel_size
            self.ideal_degree = ideal_degree
            self.ideal_face_size = ideal_face_size
            self.tomo_id_column = tomo_id_column
            if isinstance(block_definition, dict):
                cls_vals = set(self.blocks.df[block_type_column].unique())
                missing = cls_vals - set(block_definition.keys())
                if missing:
                    raise ValueError(
                        f"Block type column '{block_type_column}' contains values "
                        f"without a BlockDefinition: {sorted(missing)}."
                    )

        # --- Default ideal lattice derivation (both None + block_definition present) ---
        if self.block_definition is not None and self.ideal_degree is None and self.ideal_face_size is None:
            _defs = (
                list(self.block_definition.values())
                if isinstance(self.block_definition, dict)
                else [self.block_definition]
            )
            _d = max(defn.n_sites for defn in _defs)
            if _d in (3, 4, 6):
                self.ideal_degree = _d
                self.ideal_face_size = 2 * _d // (_d - 2)

        # --- Site type codes and allowed pairs ---
        if self.block_definition is not None:
            defs_list = (
                list(self.block_definition.values())
                if isinstance(self.block_definition, dict)
                else [self.block_definition]
            )
            all_types = sorted({st for d in defs_list for st in d.site_types})
            self.site_type_codes: dict[str, int] = {t: i + 1 for i, t in enumerate(all_types)}
            self._allowed_pairs: set[frozenset] = set()
            for d in defs_list:
                for a, b in d.pairing:
                    self._allowed_pairs.add(frozenset({a, b}))
        else:
            self.site_type_codes: dict[str, int] = {}
            self._allowed_pairs: set[frozenset] = set()

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
        """True when a block motl layer is present."""
        return self.blocks is not None

    @classmethod
    def from_blocks(
        cls,
        blocks: MotlSource,
        symmetry: Symmetry = "C3",
        *,
        site_shift: TripletLike,
        symmetry_column: MotlColumn | None = None,
        pixel_size: float = 1.0,
        tomo_id_column: MotlColumn = "tomo_id",
    ) -> "PleomorphicSurface":
        """Build a :class:`PleomorphicSurface` from a block motl without
        constructing a :class:`BlockDefinition` directly.

        This is the preferred entry point for the GUI, which cannot render
        :class:`BlockDefinition` objects.  Example::

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
            block_def: BlockDefinition | dict[float, BlockDefinition] = BlockDefinition.cyclic(
                symmetry, shift
            )
            return cls(
                blocks=blocks,
                block_definition=block_def,
                pixel_size=pixel_size,
                tomo_id_column=tomo_id_column,
            )
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
            return cls(
                blocks=loaded,
                block_definition=block_def_dict,
                block_type_column=symmetry_column,
                pixel_size=pixel_size,
                tomo_id_column=tomo_id_column,
            )

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

    @gui_exposed(label="Sites as motl", group="Lattice setup", order=40, returns="motl")
    def get_sites_as_motl(self) -> "cryomotl.Motl":
        """Return a particle list with one row per contact site of every block.

        For each block in :attr:`blocks`, each contact site defined in
        :attr:`block_definition` is placed at ``c + R.apply(v_k)``, where
        ``c`` is the block centre, ``R`` its rotation, and ``v_k`` the k-th
        site vector.

        When a :class:`BlockDefinition` has :attr:`~BlockDefinition.fold` set
        (created via :meth:`~BlockDefinition.cyclic`), this method delegates to
        :meth:`~cryomotl.Motl.split_in_asymmetric_subunits`, which also
        rotates the output angles by the corresponding symmetry operation so
        that each site row carries the block rotation composed with the
        k-th cyclic rotation ``R_k = R_z(360*(k-1)/n)``.

        For definitions without :attr:`~BlockDefinition.fold` (non-symmetric /
        typed sites), :func:`expand_motl` is used and angles stay as the block
        rotation (``orientation="keep"``).

        The returned motl is independent of the internal block layer: later
        calls to :meth:`connect` and all statistics
        methods read rotations from :attr:`blocks` directly, not from the
        returned motl.

        Returns
        -------
        cryomotl.Motl
            One row per (block, site) combination.  Columns:

            * ``object_id`` — source block's ``subtomo_id``
            * ``geom1`` — 1-based site index within its :class:`BlockDefinition`
              (CCW order)
            * ``geom2`` — integer site-type code from :attr:`site_type_codes`
            * ``x``, ``y``, ``z`` — site position in voxels

        Raises
        ------
        ValueError
            If no block layer is present.
        """
        if self.blocks is None:
            raise ValueError("get_sites_as_motl requires a block layer (blocks=...).")

        def _expand_one(group: "cryomotl.Motl", definition: "BlockDefinition") -> "cryomotl.Motl":
            if definition.fold is not None:
                site_motl = group.split_in_asymmetric_subunits(definition.fold, definition.site_vectors()[0])
                # Map split_in_asymmetric_subunits layout to our layout:
                # geom5 = original subtomo_id → object_id
                # geom2 = 1-based CCW subunit index → geom1
                # geom2 → site_type_code (uniform for cyclic definitions)
                site_motl.df["object_id"] = site_motl.df["geom5"]
                site_motl.df["geom1"] = site_motl.df["geom2"]
                code = float(self.site_type_codes[definition.sites[0].site_type])
                site_motl.df["geom2"] = code
            else:
                site_motl = expand_motl(
                    group,
                    definition.site_vectors(),
                    original_id_col="object_id",
                    order_id_col="geom1",
                    sort_vectors=False,
                    orientation="keep",
                    start_index=1,
                )
                for i, site in enumerate(definition.sites):
                    code = self.site_type_codes[site.site_type]
                    site_motl.df.loc[site_motl.df["geom1"] == (i + 1), "geom2"] = float(code)
            return site_motl

        if isinstance(self.block_definition, dict):
            group_motls: list[cryomotl.Motl] = []
            for cls_val, definition in self.block_definition.items():
                mask = self.blocks.df[self.block_type_column] == cls_val
                group_df = self.blocks.df[mask].copy().reset_index(drop=True)
                if len(group_df) == 0:
                    continue
                group = cryomotl.Motl(group_df)
                group_motls.append(_expand_one(group, definition))
        else:
            group_motls = [_expand_one(self.blocks, self.block_definition)]

        if not group_motls:
            return cryomotl.Motl(pd.DataFrame(columns=cryomotl.Motl.motl_columns))

        combined_df = pd.concat([m.df for m in group_motls], ignore_index=True)
        combined_df = combined_df.sort_values(by=[self.tomo_id_column, "object_id", "geom1"]).reset_index(drop=True)
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

    @gui_exposed(label="Unify polarity", group="Lattice setup", order=20, returns="none")
    def unify_polarity(
        self,
        max_block_distance: float | None = None,
        reference: Literal["neighbours", "centroid", "envelope"] = "neighbours",
    ) -> int:
        """Align block polarities so neighbours point the same way.

        Works per tomogram.  For ``reference='neighbours'`` or ``'centroid'``,
        uses a BFS walk over the block-proximity graph (edges between blocks
        whose centres are within *max_block_distance*): when a newly visited
        block has ``z_n · z_m < 0`` relative to its BFS parent, it is flipped
        by the 180° rotation around the in-plane part of its flip_site vector.

        For ``reference='centroid'``, each connected component is additionally
        aligned so that its blocks point outward from the component centroid
        (``Σ z · (c − g) > 0``).

        For ``reference='envelope'``, the attached envelope surface
        (``self.surface``) must be present before calling; :meth:`has_envelope`
        must be ``True``.  Every block whose ``normal_angle > 90°``
        (block z-axis opposes the envelope face normal at the closest triangle)
        is flipped.  No BFS walk is needed; *max_block_distance* may be
        ``None``.  The envelope surface is never created or replaced by this
        call.

        The envelope must be **independent** of the block orientations — do not
        pass back the mesh produced by :meth:`envelope_from_faces` on the same
        instance without first fixing the block orientations.  The recommended
        pattern is to build a clean (unflipped) instance, call
        :meth:`envelope_from_faces` on it to get a reference mesh, then
        construct a separate :class:`PleomorphicSurface` with that mesh and
        the potentially flipped blocks, and finally call
        ``unify_polarity(reference='envelope')`` on the new instance.

        Parameters
        ----------
        max_block_distance : float or None, default=None
            Maximum centre-to-centre distance (voxels) for two blocks to be
            considered neighbours.  Required for ``'neighbours'`` and
            ``'centroid'``; ignored for ``'envelope'``.
        reference : {"neighbours", "centroid", "envelope"}, default="neighbours"
            Alignment reference.

        Returns
        -------
        int
            Number of blocks whose final orientation differs from the initial one.

        Raises
        ------
        ValueError
            If no block layer is present, *reference* is invalid,
            *max_block_distance* is None for a BFS-based reference, or
            ``reference='envelope'`` is requested but no envelope surface
            is attached.

        Notes
        -----
        The input motl is never modified (it was copied on construction).
        :meth:`unify_polarity` writes the updated angles into
        :attr:`blocks`, changing only the ``phi``, ``theta`` and ``psi``
        columns of the flipped rows; row order and index are kept.  Use
        :meth:`get_blocks_as_motl` to obtain the unified angles as a new
        :class:`~cryocat.core.cryomotl.Motl` ready for averaging or for
        initialising a new assembly.
        """
        if self.blocks is None:
            raise ValueError("unify_polarity requires a block layer (blocks=...).")
        if reference not in ("neighbours", "centroid", "envelope"):
            raise ValueError(f"reference must be 'neighbours', 'centroid', or 'envelope'; got {reference!r}.")
        if reference != "envelope" and max_block_distance is None:
            raise ValueError(f"max_block_distance is required for reference='{reference}'.")
        if reference == "envelope" and not self.has_envelope:
            raise ValueError(
                "unify_polarity(reference='envelope') requires an envelope surface "
                "(attach one via ps.surface = ... or pass a mesh to PleomorphicSurface)."
            )

        N = len(self.blocks.df)

        angles_arr = self.blocks.df[["phi", "theta", "psi"]].values.astype(float)
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
                annot = self.annotate_with_envelope(tomo_id=float(tomo_val))
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
        return changed

    # ------------------------------------------------------------------
    # Contact graph
    # ------------------------------------------------------------------

    @gui_exposed(label="Connect", group="Lattice setup", order=30, returns="none")
    def connect(
        self,
        max_distance: float,
        *,
        max_site_angle: float | None = None,
    ) -> None:
        """Build and cache the contact graph from the block layer.

        Matches sites across blocks within *max_distance* voxels, assigns
        each block to a connected-component assembly, and traces closed
        polygonal faces.

        After calling this method the following getters become available:
        :meth:`get_contact_stats`, :meth:`get_face_stats`,
        :meth:`get_block_stats`, :meth:`get_assembly_stats`.

        Parameters
        ----------
        max_distance : float
            Maximum site-to-site distance (voxels) to consider two sites
            as a candidate contact.
        max_site_angle : float or None, default=None
            When set, only site pairs whose opposing unit vectors
            (``u_h`` and ``−u_p``) enclose an angle ≤ *max_site_angle*
            degrees are considered.

        Raises
        ------
        ValueError
            If no block layer is present.
        """
        if self.blocks is None:
            raise ValueError("connect() requires a block layer (blocks=...).")

        # ----------------------------------------------------------------
        # 1. Build site table
        # ----------------------------------------------------------------
        site_motl = self.get_sites_as_motl()
        site_df = site_motl.df.copy()

        # Site positions (including shifts)
        site_coords = site_motl.get_coordinates()  # (H, 3)

        # Inverse site_type_codes lookup
        inv_site_type = {v: k for k, v in self.site_type_codes.items()}

        # Lookup n_sites per block_id
        def _n_sites_for_block(bid: float) -> int:
            if isinstance(self.block_definition, dict):
                block_row = self.blocks.df[self.blocks.df["subtomo_id"] == bid]
                cls_val = float(block_row.iloc[0][self.block_type_column])
                return self.block_definition[cls_val].n_sites
            return self.block_definition.n_sites

        # Block centres
        block_coords_by_id: dict[float, np.ndarray] = {}
        for _, brow in self.blocks.df.iterrows():
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
        block_ids = site_df["object_id"].values.astype(float)
        site_indices = site_df["geom1"].values.astype(float)
        site_type_codes_col = site_df["geom2"].values.astype(float)
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
        site_table = site_table.sort_values(by=[tomo_col, "block_id", "site"]).reset_index(drop=True)

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
        diff = p_arr - c_arr_sorted
        norms = np.linalg.norm(diff, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        u_arr = diff / norms  # (H, 3) unit site direction

        for tomo_val in sorted(site_table[tomo_col].unique()):
            tomo_mask = (site_table[tomo_col] == tomo_val).values
            t_idx = np.where(tomo_mask)[0]  # global row positions in site_table
            n_t = len(t_idx)
            if n_t == 0:
                continue

            P_t = p_arr[t_idx]
            u_t = u_arr[t_idx]
            block_t = site_table["block_id"].values[t_idx]
            site_t = site_table["site"].values[t_idx].astype(np.intp)
            stype_t = site_table["site_type"].values[t_idx]
            n_sites_t = site_table["n_sites"].values[t_idx].astype(np.intp)

            qp_idx, nn_idx_list = nnana.find_nn_within_radius(P_t, P_t, max_distance, remove_qp=True)

            # Build candidate / best structures
            best_local = np.full(n_t, -1, dtype=np.intp)
            n_cand_local = np.zeros(n_t, dtype=np.intp)
            best_dist = np.full(n_t, np.inf)

            nn_map: dict[int, np.ndarray] = {}
            for qi, nns in zip(qp_idx, nn_idx_list):
                nn_map[qi] = nns

            for qi in range(n_t):
                nns = nn_map.get(qi, np.array([], dtype=np.intp))
                for ni in nns:
                    # Filter: different block
                    if block_t[ni] == block_t[qi]:
                        continue
                    # Filter: allowed pairing
                    pair = frozenset({stype_t[qi], stype_t[ni]})
                    if pair not in self._allowed_pairs:
                        continue
                    # Filter: angle
                    if max_site_angle is not None:
                        cos_a = float(np.dot(u_t[qi], -u_t[ni]))
                        angle_deg = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
                        if angle_deg > max_site_angle:
                            continue
                    n_cand_local[qi] += 1
                    d = float(np.linalg.norm(P_t[ni] - P_t[qi]))
                    if d < best_dist[qi] or (d == best_dist[qi] and ni < best_local[qi]):
                        best_dist[qi] = d
                        best_local[qi] = ni

            # Mutual best match
            partner_local = np.full(n_t, -1, dtype=np.intp)
            for qi in range(n_t):
                bi = best_local[qi]
                if bi >= 0 and best_local[bi] == qi:
                    partner_local[qi] = bi

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

            # ---- Convert global partner to local partner for trace_faces ----
            global_to_local = {int(g): l for l, g in enumerate(t_idx)}
            partner_local_for_trace = np.array(
                [global_to_local.get(int(partner_arr[int(g)]), -1) if partner_arr[int(g)] >= 0 else -1 for g in t_idx],
                dtype=np.intp,
            )

            # ---- Faces ----
            local_face = trace_faces(partner_local_for_trace, block_t, site_t, n_sites_t)

            # Build local lookup so we can re-walk each face in the original
            # trace_faces order.  np.where returns indices sorted by value, not
            # by walk order, so we must reconstruct the walk from the starting
            # half-edge (minimum index in each face, since trace_faces iterates
            # range(H) and the first unvisited index per face is its minimum).
            local_lookup: dict[tuple[int, int], int] = {(int(block_t[h]), int(site_t[h])): h for h in range(n_t)}

            # Collect face records in walk order
            n_local_faces = int(local_face.max()) if local_face.max() > 0 else 0
            for fid in range(1, n_local_faces + 1):
                fmask = local_face == fid
                local_hedges_sorted = np.where(fmask)[0]  # sorted by index
                face_size = len(local_hedges_sorted)
                # Starting half-edge is the minimum-indexed in this face.
                start_local = int(local_hedges_sorted[0])
                # Re-walk from start to recover the exact traversal order.
                ordered_local: list[int] = []
                h = start_local
                while len(ordered_local) < face_size:
                    ordered_local.append(h)
                    p = int(partner_local_for_trace[h])
                    if p < 0:
                        break
                    ns_p = int(n_sites_t[p])
                    next_site = int(site_t[p]) % ns_p + 1
                    h_next = local_lookup.get((int(block_t[p]), next_site))
                    if h_next is None or h_next == start_local:
                        break
                    h = h_next
                global_hedges = [int(t_idx[li]) for li in ordered_local]
                asm_id = float(block_assembly.get(block_t[start_local], -1))
                face_records.append(
                    {
                        tomo_col: tomo_val,
                        "face_id": fid,
                        "assembly_id": asm_id,
                        "half_edges": global_hedges,
                    }
                )

            for local_i, global_i in enumerate(t_idx):
                raw_fid = int(local_face[local_i])
                face_id_arr[global_i] = raw_fid  # -1 for boundary, positive for closed

        # ----------------------------------------------------------------
        # 3. Store
        # ----------------------------------------------------------------
        site_table["partner"] = partner_arr
        site_table["n_candidates"] = n_candidates_arr
        site_table["assembly_id"] = assembly_id_arr
        site_table["face_id"] = face_id_arr

        self._site_table = site_table
        self._faces = face_records

    def _require_connect(self) -> None:
        """Raise ValueError if connect() has not been called."""
        if self._faces is None:
            raise ValueError("Call connect() first to build the contact graph.")

    # ------------------------------------------------------------------
    # Statistics getters
    # ------------------------------------------------------------------

    @gui_exposed(label="Contact stats", group="Lattice statistics", order=10, returns="dataframe")
    def get_contact_stats(self) -> pd.DataFrame:
        """Return one row per matched half-edge (each contact appears twice).

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
            If :meth:`connect` has not been called.
        """
        self._require_connect()
        st = self._site_table
        tomo_col = self.tomo_id_column
        matched_mask = st["partner"].values >= 0
        h_idx = np.where(matched_mask)[0]

        if len(h_idx) == 0:
            return pd.DataFrame(
                columns=[
                    tomo_col,
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
        angles_all = self.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = srot.from_euler("zxz", angles_all, degrees=True)
        block_to_row: dict[float, int] = {
            float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(self.blocks.df))
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

            site_dist = float(np.linalg.norm(P_p - P_h)) * self.pixel_size
            block_dist = float(np.linalg.norm(c_p - c_h)) * self.pixel_size

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

    @gui_exposed(label="Face stats", group="Lattice statistics", order=20, returns="dataframe")
    def get_face_stats(self) -> pd.DataFrame:
        """Return one row per closed face.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``face_id``, ``assembly_id``, ``size``,
            ``n_unique_blocks``, ``block_ids``, ``centroid_x``, ``centroid_y``,
            ``centroid_z``, ``normal_x``, ``normal_y``, ``normal_z``,
            ``planarity_rms``.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called.
        """
        self._require_connect()
        st = self._site_table
        tomo_col = self.tomo_id_column

        # Block rotations
        angles_all = self.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = srot.from_euler("zxz", angles_all, degrees=True)
        block_to_row: dict[float, int] = {
            float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(self.blocks.df))
        }
        z_unit = np.array([0.0, 0.0, 1.0])

        rows = []
        for rec in self._faces:
            hedges = rec["half_edges"]
            size = len(hedges)
            block_ids_walk = [float(st.at[h, "block_id"]) for h in hedges]
            cx_vals = np.array([st.at[h, "cx"] for h in hedges])
            cy_vals = np.array([st.at[h, "cy"] for h in hedges])
            cz_vals = np.array([st.at[h, "cz"] for h in hedges])
            centres = np.column_stack([cx_vals, cy_vals, cz_vals])
            centroid = centres.mean(axis=0)

            z_vecs = np.array([all_R[block_to_row[bid]].apply(z_unit) for bid in block_ids_walk])
            normal_mean = z_vecs.mean(axis=0)
            n_norm = float(np.linalg.norm(normal_mean))
            normal_unit = normal_mean / n_norm if n_norm > 1e-15 else normal_mean

            if size <= 3:
                planarity_rms = 0.0
            else:
                centred = centres - centroid
                _, _, Vt = np.linalg.svd(centred, full_matrices=False)
                plane_normal = Vt[-1]
                dists = centred @ plane_normal
                planarity_rms = float(np.sqrt(np.mean(dists**2)))

            rows.append(
                {
                    tomo_col: rec[tomo_col],
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
                    "planarity_rms": planarity_rms,
                }
            )
        return pd.DataFrame(rows)

    @gui_exposed(label="Block stats", group="Lattice statistics", order=30, returns="dataframe")
    def get_block_stats(self) -> pd.DataFrame:
        """Return one row per block.

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``block_id``, ``block_type``, ``n_sites``,
            ``degree``, ``complete``, ``assembly_id``, ``face_signature``,
            ``angle_sum``, ``angle_deficit``.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called.
        """
        self._require_connect()
        st = self._site_table
        tomo_col = self.tomo_id_column

        # Face sizes by face_id (within tomo)
        face_size: dict[tuple, int] = {}
        for rec in self._faces:
            key = (rec[tomo_col], rec["face_id"])
            face_size[key] = len(rec["half_edges"])

        # Block type lookup
        block_type_map: dict[float, float] = {}
        if isinstance(self.block_definition, dict):
            for _, row in self.blocks.df.iterrows():
                block_type_map[float(row["subtomo_id"])] = float(row[self.block_type_column])

        rows = []
        for (tomo_val, block_val), grp in st.groupby([tomo_col, "block_id"]):
            n_sites_val = int(grp["n_sites"].iloc[0])
            degree = int((grp["partner"] >= 0).sum())
            partner_rows = grp[grp["partner"] >= 0]

            # face_signature: sorted face sizes, 0 for boundary
            face_sizes_for_block = []
            for _, hrow in grp.iterrows():
                fid = int(hrow["face_id"])
                if fid > 0:
                    key = (tomo_val, fid)
                    face_sizes_for_block.append(face_size.get(key, 0))
                else:
                    face_sizes_for_block.append(0)
            face_sig = "-".join(str(s) for s in sorted(face_sizes_for_block))

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

            rows.append(
                {
                    tomo_col: tomo_val,
                    "block_id": float(block_val),
                    "block_type": btype,
                    "n_sites": n_sites_val,
                    "degree": degree,
                    "complete": degree == n_sites_val,
                    "assembly_id": asm_id,
                    "face_signature": face_sig,
                    "angle_sum": angle_sum_val,
                    "angle_deficit": angle_deficit_val,
                }
            )
        return pd.DataFrame(rows)

    @gui_exposed(label="Assembly stats", group="Lattice statistics", order=40, returns="dataframe")
    def get_assembly_stats(self) -> pd.DataFrame:
        """Return one row per (tomo_id, assembly_id).

        Returns
        -------
        pandas.DataFrame
            Columns: ``tomo_id``, ``assembly_id``, ``n_blocks``, ``n_contacts``,
            ``n_faces``, ``n_boundary_half_edges``, ``closed``,
            ``euler_characteristic``, ``angle_deficit_sum``, ``defect_charge``,
            plus one ``n_faces_<m>`` column per distinct face size in the data.

        Raises
        ------
        ValueError
            If :meth:`connect` has not been called.
        """
        self._require_connect()
        st = self._site_table
        tomo_col = self.tomo_id_column
        block_stats = self.get_block_stats()
        face_stats = self.get_face_stats()

        # Collect all face sizes for column headers
        all_face_sizes = sorted(face_stats["size"].unique().tolist()) if len(face_stats) > 0 else []

        rows = []
        for (tomo_val, asm_val), block_grp in block_stats.groupby([tomo_col, "assembly_id"]):
            n_blocks = len(block_grp)
            block_id_set = set(block_grp["block_id"].tolist())

            # Contacts (matched half-edges / 2)
            st_asm = st[(st[tomo_col] == tomo_val) & (st["block_id"].isin(block_id_set))]
            n_matched = int((st_asm["partner"] >= 0).sum())
            n_contacts = n_matched // 2

            # Faces for this assembly
            face_asm = face_stats[(face_stats[tomo_col] == tomo_val) & (face_stats["assembly_id"] == asm_val)]
            n_faces = len(face_asm)

            # Boundary half-edges
            n_boundary = int((st_asm["face_id"] <= 0).sum())

            closed = n_boundary == 0
            euler = n_blocks - n_contacts + n_faces

            # angle_deficit_sum
            deficit_vals = block_grp["angle_deficit"].dropna()
            angle_deficit_sum = float(deficit_vals.sum())

            # defect_charge
            if self.ideal_degree is not None and self.ideal_face_size is not None:
                d0 = self.ideal_degree
                m0 = self.ideal_face_size
                complete_blocks = block_grp[block_grp["complete"]]
                defect_blocks = float(((d0 - complete_blocks["degree"]) / d0).sum())
                defect_faces = float(((m0 - face_asm["size"]) / m0).sum())
                defect_charge_val = defect_blocks + defect_faces
            else:
                defect_charge_val = float("nan")

            face_size_counts: dict[str, int] = {f"n_faces_{m}": 0 for m in all_face_sizes}
            if len(face_asm) > 0:
                for m, cnt in face_asm["size"].value_counts().items():
                    key = f"n_faces_{m}"
                    if key in face_size_counts:
                        face_size_counts[key] = int(cnt)

            row = {
                tomo_col: tomo_val,
                "assembly_id": float(asm_val),
                "n_blocks": n_blocks,
                "n_contacts": n_contacts,
                "n_faces": n_faces,
                "n_boundary_half_edges": n_boundary,
                "closed": closed,
                "euler_characteristic": euler,
                "angle_deficit_sum": angle_deficit_sum,
                "defect_charge": defect_charge_val,
            }
            row.update(face_size_counts)
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        result = pd.DataFrame(rows)
        # Fill any missing n_faces_<m> columns with 0
        for m in all_face_sizes:
            col = f"n_faces_{m}"
            if col not in result.columns:
                result[col] = 0
        return result

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

        Notes
        -----
        The input motl is never modified (it was copied on construction).
        Only the requested destination columns of :attr:`blocks` are
        written; no other column, no row, and the DataFrame index are
        changed.  Use :meth:`get_blocks_as_motl` to retrieve the updated
        state.
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

    @gui_exposed(label="Store block stat", group="Lattice statistics", order=50, returns="motl")
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
            ``complete`` is written as ``0`` / ``1``.
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

        Notes
        -----
        The input motl is never modified (it was copied on construction).
        The requested *column* of :attr:`blocks` is written in place; the
        returned motl is an independent deep copy via
        :meth:`get_blocks_as_motl`.
        """
        if stat not in self._BLOCK_STAT_CHOICES:
            raise ValueError(f"Unknown stat {stat!r}. Must be one of {self._BLOCK_STAT_CHOICES}.")
        self.store_block_stats({stat: column})
        return copy.deepcopy(self.blocks)

    @gui_exposed(label="Blocks as motl", group="Lattice setup", order=50, returns="motl")
    def get_blocks_as_motl(self) -> "cryomotl.Motl":
        """Return the current block layer as an independent :class:`~cryocat.core.cryomotl.Motl`.

        Returns a deep copy of :attr:`blocks`, capturing whatever state it
        is in at call time — original angles, post-:meth:`unify_polarity`
        angles, and any columns written by :meth:`store_block_stats` or
        :meth:`store_block_stat`.

        Does not require :meth:`connect` to have been called first.

        Returns
        -------
        cryomotl.Motl
            Independent deep copy of :attr:`blocks`.

        Raises
        ------
        ValueError
            If no block layer is present.

        Notes
        -----
        *blocks* is loaded with :func:`~cryocat.core.cryomotl.Motl.load`,
        which copies a :class:`~cryocat.core.cryomotl.Motl` instance; the
        input is never modified.  :meth:`unify_polarity`,
        :meth:`store_block_stats` and :meth:`store_block_stat` change only
        this object's copy (angles, or the requested columns; row order and
        index are kept).  Use this method to obtain the current state,
        e.g. unified orientations for averaging or for building a new
        assembly.
        """
        if self.blocks is None:
            raise ValueError("get_blocks_as_motl requires a block layer (blocks=...).")
        return copy.deepcopy(self.blocks)

    @gui_exposed(label="Faces as motl", group="Lattice motls", order=10, returns="motl")
    def get_faces_as_motl(self) -> "cryomotl.Motl":
        """Return a :class:`~cryocat.core.cryomotl.Motl` with one row per closed face.

        Columns set on the motl:
        - position (x/y/z): face centroid
        - orientation (phi/theta/psi): Euler angles from face normal
        - tomo_id: from the assembly
        - object_id: face_id
        - class: face size (polygon vertex count)
        - geom1: assembly_id

        Returns
        -------
        cryomotl.Motl
        """
        self._require_connect()
        face_stats = self.get_face_stats()
        tomo_col = self.tomo_id_column
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
        data["object_id"] = face_stats["face_id"].values
        data["class"] = face_stats["size"].values.astype(float)
        data["geom1"] = face_stats["assembly_id"].values

        motl = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])
        motl.renumber_particles()
        return motl

    @gui_exposed(label="Gaps as motl", group="Lattice motls", order=20, returns="motl")
    def get_gaps_as_motl(
        self,
        cluster_radius: float,
        min_blocks: int | None = None,
    ) -> "cryomotl.Motl":
        """Predict missing-block positions from unmatched contact sites.

        For each unmatched half-edge h on block b with site position P_h and
        block centre c_b, the predicted gap position is
        ``g_h = c_b + 2 * (P_h - c_b)``.

        Gap points are clustered (by centre-to-centre distance ≤ *cluster_radius*)
        and only clusters with at least *min_blocks* distinct source blocks are
        kept.

        Parameters
        ----------
        cluster_radius : float
            Maximum distance (voxels) between two gap points to be in one cluster.
        min_blocks : int or None, default=None
            Minimum number of distinct source blocks in a cluster to keep it.
            Defaults to ``ideal_degree`` when set, else 3.

        Returns
        -------
        cryomotl.Motl
            One row per kept cluster: position = mean of gap points;
            orientation from mean block z-axis; ``geom1`` = n_distinct_blocks.
        """
        self._require_connect()
        import networkx as _nx

        if min_blocks is None:
            min_blocks = self.ideal_degree if self.ideal_degree is not None else 3

        tomo_col = self.tomo_id_column
        st = self._site_table
        block_to_row: dict[float, int] = {
            float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(self.blocks.df))
        }
        angles_all = self.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = srot.from_euler("zxz", angles_all, degrees=True)
        z_unit = np.array([0.0, 0.0, 1.0])

        unmatched = st[st["partner"] < 0]
        gap_pts_by_tomo: dict[Any, list] = {}
        gap_bids_by_tomo: dict[Any, list] = {}
        gap_zax_by_tomo: dict[Any, list] = {}

        for _, row in unmatched.iterrows():
            tomo_val = row[tomo_col]
            b_id = float(row["block_id"])
            c_b = np.array([row["cx"], row["cy"], row["cz"]])
            P_h = np.array([row["x"], row["y"], row["z"]])
            g_h = c_b + 2.0 * (P_h - c_b)
            z_ax = all_R[block_to_row[b_id]].apply(z_unit)
            gap_pts_by_tomo.setdefault(tomo_val, []).append(g_h)
            gap_bids_by_tomo.setdefault(tomo_val, []).append(b_id)
            gap_zax_by_tomo.setdefault(tomo_val, []).append(z_ax)

        out_rows = []
        for tomo_val, pts_list in gap_pts_by_tomo.items():
            pts = np.array(pts_list)
            bids = np.array(gap_bids_by_tomo[tomo_val])
            z_axes = np.array(gap_zax_by_tomo[tomo_val])
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
                        "x": mean_pos[0],
                        "y": mean_pos[1],
                        "z": mean_pos[2],
                        "phi": float(euler[0]),
                        "theta": float(euler[1]),
                        "psi": float(euler[2]),
                        "geom1": float(n_distinct),
                    }
                )

        data: dict[str, Any] = {c: np.zeros(len(out_rows)) for c in cryomotl.Motl.motl_columns}
        for k, row in enumerate(out_rows):
            for col, val in row.items():
                data[col][k] = val
        motl = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])
        if out_rows:
            motl.renumber_particles()
        return motl

    @gui_exposed(label="Envelope from faces", group="Lattice envelope", order=10, returns="surface")
    def envelope_from_faces(
        self,
        assembly_id: int | None = None,
        tomo_id: float | None = None,
    ) -> Mesh:
        """Build a triangulated mesh envelope from the face polygon network.

        Vertices are the block centres (× pixel_size) and face centroids
        (× pixel_size).  Each face polygon of size n contributes n triangles
        of the form (centroid, b_i, b_{i+1}).  Winding is chosen so that each
        triangle normal points in the same direction as the face normal.

        Parameters
        ----------
        assembly_id : int or None, default=None
            Restrict to this assembly.
        tomo_id : float or None, default=None
            Restrict to this tomogram.

        Returns
        -------
        Mesh
            Mesh with vertex normals computed.
        """
        self._require_connect()
        tomo_col = self.tomo_id_column
        face_stats = self.get_face_stats()

        if assembly_id is not None:
            face_stats = face_stats[face_stats["assembly_id"] == float(assembly_id)]
        if tomo_id is not None:
            face_stats = face_stats[face_stats[tomo_col] == tomo_id]
        face_stats = face_stats.reset_index(drop=True)

        if len(face_stats) == 0:
            raise ValueError("No faces to build envelope from (after filtering).")

        all_block_ids_ordered: list[float] = []
        seen: set[float] = set()
        for block_list in face_stats["block_ids"]:
            for bid in block_list:
                if bid not in seen:
                    seen.add(bid)
                    all_block_ids_ordered.append(bid)

        block_id_to_vx: dict[float, int] = {bid: i for i, bid in enumerate(all_block_ids_ordered)}
        n_blocks = len(all_block_ids_ordered)

        block_to_row: dict[float, int] = {
            float(self.blocks.df.iloc[i]["subtomo_id"]): i for i in range(len(self.blocks.df))
        }
        block_verts = np.array(
            [
                self.blocks.df.iloc[block_to_row[bid]][["x", "y", "z"]].values.astype(float) * self.pixel_size
                for bid in all_block_ids_ordered
            ]
        )

        face_centroid_verts = face_stats[["centroid_x", "centroid_y", "centroid_z"]].values * self.pixel_size

        vertices = np.vstack([block_verts, face_centroid_verts])

        triangles: list[list[int]] = []
        for fi, frow in face_stats.iterrows():
            block_ids_walk: list[float] = frow["block_ids"]
            m_idx = n_blocks + int(fi)
            n_walk = len(block_ids_walk)
            face_normal = np.array([frow["normal_x"], frow["normal_y"], frow["normal_z"]])
            fn_norm = float(np.linalg.norm(face_normal))
            if fn_norm > 1e-15:
                face_normal = face_normal / fn_norm

            # Winding check: reverse all triangles in this face if the first
            # triangle's geometric normal opposes the face's mean block z-axis.
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
        return mesh

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
        else:
            blocks_df = self.blocks.df.reset_index(drop=True)

        c = blocks_df[["x", "y", "z"]].values.astype(float)
        q = c * self.pixel_size

        angles = blocks_df[["phi", "theta", "psi"]].values.astype(float)
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
        map_size = geom.as_triplet(input_map_metadata[0])
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
            coords = self.motl.df.loc[orig_idx, ["x", "y", "z"]].to_numpy()
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
        map_size = geom.as_triplet(input_map_metadata[0])
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
