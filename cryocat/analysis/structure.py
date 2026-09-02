from __future__ import annotations
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
from cryocat._types import MapSource, MotlColumn, PathOrStr, TomoDimensions, TripletLike, RotationLike, MotlType, ArrayLike, Symmetry
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
from collections.abc import Callable
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
        rot_stats = df.groupby([self.column_name, self.chain_id_col]).apply(
            self._step_rotations, **_kw
        )

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
        tm = m.get_motl_subset(
            column_values=[t], column_name=tomo_id_column, reset_index=True
        )
        for o in tm.get_unique_values(affiliation_column):
            om = tm.get_motl_subset(
                column_values=[o], column_name=affiliation_column, reset_index=True
            )
            coords = om.get_coordinates()
            center = geom.barycenter(coords, weights) if coords.shape[0] > 0 else np.zeros(3)
            central_points.append(center)
            tomo_ids.append(float(t))
            object_ids.append(float(o))

    out = cryomotl.Motl()
    if central_points:
        pts = np.vstack(central_points)
        out.fill({
            "x": pts[:, 0],
            "y": pts[:, 1],
            "z": pts[:, 2],
            "tomo_id": np.array(tomo_ids),
            "object_id": np.array(object_ids),
        })
        out.renumber_particles()
    out.df.fillna(0.0, inplace=True)
    return out


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
            tm = self.motl.get_motl_subset(
                column_values=[t], column_name=self.tomo_id_column, reset_index=True
            )
            for o in tm.get_unique_values(self.affiliation_column):
                om = tm.get_motl_subset(
                    column_values=[o], column_name=self.affiliation_column, reset_index=True
                )
                coords = om.get_coordinates()
                center = geom.barycenter(coords) if coords.shape[0] > 0 else np.zeros(3)
                central_points.append(center)
                tomo_ids.append(float(t))
                object_ids.append(float(o))

        out = cryomotl.Motl()
        if central_points:
            pts = np.vstack(central_points)
            out.fill({
                "x": pts[:, 0],
                "y": pts[:, 1],
                "z": pts[:, 2],
                "tomo_id": np.array(tomo_ids),
                "object_id": np.array(object_ids),
            })
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

        for (tomo_id, object_id), group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
            n_present = len(group)
            if has_order:
                present = set(int(v) for v in group[self.order_column].dropna())
                missing: list[int] | None = sorted(all_expected - present)
            else:
                missing = None
            rows.append({
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "n_present": n_present,
                "occupancy": n_present / self.n_subunits,
                "missing": missing,
            })

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
        for (_t, _o), grp in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
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
            tm = motl.get_motl_subset(
                column_values=[t], column_name=tomo_col, reset_index=True
            )

            pts: list[np.ndarray] = []
            obj_ids: list[float] = []
            for o in tm.get_unique_values(aff_col):
                om = tm.get_motl_subset(
                    column_values=[o], column_name=aff_col, reset_index=True
                )
                coords = om.get_coordinates()
                center = geom.barycenter(coords) if coords.shape[0] > 0 else np.zeros(3)
                pts.append(center)
                obj_ids.append(float(o))

            centers_motl = cryomotl.Motl()
            if pts:
                pts_arr = np.vstack(pts)
                centers_motl.fill({
                    "x": pts_arr[:, 0],
                    "y": pts_arr[:, 1],
                    "z": pts_arr[:, 2],
                    "tomo_id": t,
                    "object_id": obj_ids,
                })
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
                f"{type(self).__name__}.create_affiliation: unknown method {method!r}. "
                "Choose 'radius' or 'tracing'."
            )

        # Assign subunit order via motl-swap
        orig_motl = self.motl
        self.motl = motl_out
        self.assign_subunit_order()
        motl_out = self.motl
        self.motl = orig_motl

        # Per-object occupancy count
        motl_out.df[occupancy_column] = motl_out.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        )[self.affiliation_column].transform("size")

        # Cone distance to per-object consensus z-axis (mirrors geom.cone_distance)
        motl_out.df[cone_distance_column] = 0.0
        for (_t, _o), grp in motl_out.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
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
        sizes = motl_out.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ).size()
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

            qp_idx, nn_idx_list = nnana.find_nn_within_radius(
                coords, coords, radius, remove_qp=True
            )

            qp_ids: list = []
            nn_ids: list = []
            for qi, nns in zip(qp_idx, nn_idx_list):
                for ni in nns:
                    qp_ids.append(int(subtomo_ids[qi]))
                    nn_ids.append(int(subtomo_ids[ni]))

            next_id = 1
            in_component: set = set()

            if qp_ids:
                components = _clustering.connected_component_clusters(
                    qp_ids, nn_ids, min_size=1
                )
                for comp in components:
                    comp_ids = set(comp.nodes())
                    mask = group_df["subtomo_id"].isin(comp_ids)
                    motl_out.df.loc[group_df[mask].index, self.affiliation_column] = float(
                        next_id
                    )
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
            raise ValueError(
                f"CnComplex requires cyclic Cn symmetry, got {symmetry!r}."
            )
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
                center, _ = geom.ray_ray_intersection_3d(
                    starting_points=coords, ending_points=end_coord
                )
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
            tm = self.motl.get_motl_subset(
                column_values=[t], column_name=self.tomo_id_column, reset_index=True
            )
            for o in tm.get_unique_values(self.affiliation_column):
                om = tm.get_motl_subset(
                    column_values=[o], column_name=self.affiliation_column, reset_index=True
                )
                center, _ = self._compute_object_center(om)
                central_points.append(center)
                tomo_ids.append(float(t))
                object_ids.append(float(o))

        out = cryomotl.Motl()
        if central_points:
            pts = np.vstack(central_points)
            out.fill({
                "x": pts[:, 0],
                "y": pts[:, 1],
                "z": pts[:, 2],
                "tomo_id": np.array(tomo_ids),
                "object_id": np.array(object_ids),
            })
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
    ) -> list[int]:
        """Compute 1-based cyclic subunit indices for one object.

        Parameters
        ----------
        object_motl : Motl
            Particles belonging to one ring object.

        Returns
        -------
        list of int
            Indices, same length as ``object_motl.df``.  The first particle
            always receives index 1.
        """
        center, _ = self._compute_object_center(object_motl)
        su_coord = object_motl.get_coordinates()
        vectors = su_coord - np.tile(center, (su_coord.shape[0], 1))
        div_angle = self.central_angle
        s_idx: list[int] = [1]
        for vec in vectors[1:]:
            angle = geom.vector_angular_distance(vectors[0], vec) / div_angle
            s_idx.append(
                int(decimal.Decimal(angle).to_integral_value(rounding=decimal.ROUND_HALF_UP)) + 1
            )
        return s_idx

    @gui_exposed(label="Assign subunit order", group="Affiliation", order=30, returns="none")
    def assign_subunit_order(self) -> None:
        """Assign 1-based cyclic subunit indices into ``self.order_column``.

        For every ring group (as defined by ``_ring_group_columns``): computes
        the angular position of each particle relative to the first particle
        (stepping by ``central_angle`` = 360 / n), and writes the resulting
        1-based index into ``self.motl.df[self.order_column]``.

        Notes
        -----
        Object centres are determined by :meth:`_compute_object_center`, which
        respects ``self.center_method``.  Modifies ``self.motl.df`` in place.
        """
        for keys, group in self.motl.df.groupby(self._ring_group_columns):
            om = cryomotl.Motl(group.reset_index(drop=True))
            s_idx = self._cyclic_indices_for_object(om)
            self.motl.df.loc[group.index, self.order_column] = s_idx

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
                        pair_rows.append([
                            group.index[mask_i][0],
                            group.index[mask_j][0],
                        ])

                if pair_rows:
                    idx = np.asarray(pair_rows)
                    dists = geom.point_pairwise_dist(
                        coord[idx[:, 0], :], coord[idx[:, 1], :]
                    ) * pixel_size
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
        for (tomo_id, object_id), group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
            coords_grp = coord[group.index.to_numpy(), :]
            r = self._circumradius_for_group(coords_grp)
            center = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            center_rows.append({
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "x": float(center[0]),
                "y": float(center[1]),
                "z": float(center[2]),
                "radius": r,
            })
        geo_df = pd.DataFrame(center_rows)

        result = occ_df.merge(geo_df, on=["tomo_id", "object_id"], how="outer")
        result = result.merge(circ_df, on=["tomo_id", "object_id"], how="left")
        result = result.merge(
            diam_df[["tomo_id", "object_id", "mean_diameter", "n_pairs"]],
            on=["tomo_id", "object_id"],
            how="left",
        )
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
            raise ValueError(
                f"DnComplex requires dihedral Dn symmetry, got {symmetry!r}."
            )
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

        for keys, group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
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
        self.motl.df.loc[ring1_mask, self.order_column] = (
            self.motl.df.loc[ring1_mask, self.order_column] + self.n
        )

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

        for (tomo_id, object_id), group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
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

            rows.append({
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "ring_spacing": spacing,
            })

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

        for (tomo_id, object_id), group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
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

            rows.append({
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "inter_ring_twist": twist,
            })

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
        circ_agg = (
            circ_df_ring.groupby(merge_cols)["circumference"]
            .mean()
            .reset_index()
        )
        diam_agg = (
            diam_df_ring.groupby(merge_cols)
            .agg(mean_diameter=("mean_diameter", "mean"), n_pairs=("n_pairs", "sum"))
            .reset_index()
        )

        coord = self.motl.get_coordinates()
        center_rows: list[dict] = []
        for (tomo_id, object_id), group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
            coords_grp = coord[group.index.to_numpy(), :]
            bary = geom.barycenter(coords_grp) if coords_grp.shape[0] > 0 else np.zeros(3)
            r = self._circumradius_for_group(coords_grp)
            center_rows.append({
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "x": float(bary[0]),
                "y": float(bary[1]),
                "z": float(bary[2]),
                "radius": r,
            })
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
                    "cluster_subunits_to_rings: supply either `exit_mask` or "
                    "both `exit_mask_coord` and `mask_size`."
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
            tm = traced_motl.get_motl_subset(
                column_values=[t], column_name=self.tomo_id_column, reset_index=True
            )
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
            s_idx.append(
                int(decimal.Decimal(angle).to_integral_value(rounding=decimal.ROUND_HALF_UP)) + 1
            )
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
            tm = motl.get_motl_subset(
                column_values=[t], column_name="tomo_id", reset_index=True
            )

            # Build centres motl using radius-shift approach
            central_points: list[np.ndarray] = []
            obj_ids: list[float] = []
            for o in tm.get_unique_values("object_id"):
                om = tm.get_motl_subset(
                    column_values=[o], column_name="object_id", reset_index=True
                )
                central_points.append(NPC._center_by_radius_shift(om, npc_radius))
                obj_ids.append(o)

            centers_motl = cryomotl.Motl()
            if central_points:
                ca = np.vstack(central_points)
                centers_motl.fill(
                    {"x": ca[:, 0], "y": ca[:, 1], "z": ca[:, 2],
                     "tomo_id": t, "object_id": obj_ids}
                )
                centers_motl.renumber_particles()
            centers_motl.df.fillna(0.0, inplace=True)

            changed_objects: list[float] = []
            if centers_motl.df.shape[0] > 1:
                center_stats = nnana.get_nn_stats(centers_motl, centers_motl)
                if any(center_stats["distance"] <= npc_radius):
                    center_idx, nn_idx_list = nnana.get_nn_within_distance(
                        centers_motl, npc_radius
                    )
                    for i, pos in enumerate(center_idx):
                        o_id1 = centers_motl.df.loc[
                            centers_motl.df.index[pos], "object_id"
                        ]
                        changed_objects.append(o_id1)
                        for j in nn_idx_list[i]:
                            o_id2 = centers_motl.df.loc[
                                centers_motl.df.index[j], "object_id"
                            ]
                            tm.df.loc[tm.df["object_id"] == o_id2, "object_id"] = o_id1

            tm.df["geom1"] = (
                tm.df.groupby("object_id")["object_id"].transform("count")
            )
            for o in changed_objects:
                om = tm.get_motl_subset(
                    column_values=o, column_name="object_id", reset_index=True
                )
                s_idx = NPC._assign_subunit_index(om, npc_radius)
                tm.df.loc[tm.df["object_id"] == o, "geom2"] = s_idx

            tm.df["object_id"] = tm.df["object_id"].rank(method="dense").astype(int)
            motl.df.loc[
                motl.df["tomo_id"] == t, ["object_id", "geom1", "geom2"]
            ] = tm.df[["object_id", "geom1", "geom2"]].values

        motl.df.reset_index(inplace=True, drop=True)
        motl.df["geom1"] = motl.df.groupby(["tomo_id", "object_id"])[
            "object_id"
        ].transform("count")
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
                    pair_rows.append([
                        group.index[mask_i][0],
                        group.index[mask_j][0],
                    ])
            if not pair_rows:
                continue
            idx = np.asarray(pair_rows)
            dists = (
                geom.point_pairwise_dist(coord[idx[:, 0], :], coord[idx[:, 1], :])
                * pixel_size
            )
            mean_d = float(np.mean(dists))
            diameters_col[grp_idx] = mean_d
            rows.append({
                "tomo_id": float(tomo_id),
                "object_id": float(object_id),
                "mean_diameter": mean_d,
                "n_pairs": int(len(dists)),
            })

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
            om = motl.get_motl_subset(
                column_values=[o], column_name="object_id", reset_index=True
            )
            center = NPC._center_by_radius_shift(om, npc_radius=radius)
            centers.append(center)
            t = float(tomo_id) if tomo_id is not None else float(om.df["tomo_id"].iloc[0])
            tomo_ids.append(t)
            object_ids.append(float(o))

        result = cryomotl.Motl()
        if centers:
            pts_arr = np.vstack(centers)
            result.fill({
                "x": pts_arr[:, 0],
                "y": pts_arr[:, 1],
                "z": pts_arr[:, 2],
                "tomo_id": tomo_ids,
                "object_id": object_ids,
            })
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
# PleomorphicSurface for discrete surfaces (Mesh and OrientedPointCloud)
# =============================================================================

class PleomorphicSurface:
    """Wrapper around :class:`Mesh` or :class:`OrientedPointCloud`."""

    def __init__(self, surface: Mesh | OrientedPointCloud | "PleomorphicSurface") -> None:
        if isinstance(surface, PleomorphicSurface):
            surface = surface.surface
        if not isinstance(surface, (Mesh, OrientedPointCloud)):
            raise TypeError(
                f"Unsupported surface type: {type(surface)}. "
                "Must be Mesh, OrientedPointCloud, or PleomorphicSurface."
            )
        self.surface = surface

    @staticmethod
    def _unwrap_surface(surface: Mesh | OrientedPointCloud | "PleomorphicSurface") -> Mesh | OrientedPointCloud:
        """Return the concrete Mesh / OrientedPointCloud behind an optional wrapper."""
        if isinstance(surface, PleomorphicSurface):
            return surface.surface
        if isinstance(surface, (Mesh, OrientedPointCloud)):
            return surface
        raise TypeError(
            f"Unsupported surface type: {type(surface)}. "
            "Must be Mesh, OrientedPointCloud, or PleomorphicSurface."
        )

    @classmethod
    def read(cls, input_path: PathOrStr, method: str = "mesh", **kwargs: Any) -> "PleomorphicSurface":
        """
        Create a wrapped surface from common on-disk inputs.

        Parameters
        ----------
        input_path : str or Path
            Input file path.
        method : str, default="mesh"
            Loader to use:
            - "mesh": geometry-only triangle mesh via :meth:`Mesh.read`
            - "mesh_curvatures": VTP triangle mesh with curvature fields via
              :meth:`Mesh.read_curvatures`
            - "mesh_from_mrc": segmentation-to-mesh via :meth:`Mesh.from_mrc`
            - "point_cloud": oriented point cloud via :meth:`OrientedPointCloud.read`
            - "point_cloud_from_mrc": segmentation-to-point-cloud via
              :meth:`OrientedPointCloud.from_mrc`
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

        Returns
        -------
        PleomorphicSurface
            Wrapped surface loaded from ``input_path``.
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
        else:
            raise ValueError(
                f"Unknown read method '{method}'. Use 'mesh', 'mesh_curvatures', "
                "'mesh_from_mrc', 'point_cloud', or 'point_cloud_from_mrc'."
            )

        return cls(surface)

    @property
    def is_mesh(self) -> bool:
        """True when the backing geometry has triangle connectivity (:class:`Mesh`)."""
        return isinstance(self.surface, Mesh)

    @property
    def is_point_cloud(self) -> bool:
        """True when the backing geometry is discrete samples (:class:`OrientedPointCloud`)."""
        return isinstance(self.surface, OrientedPointCloud)

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

    def save(self, output_path: PathOrStr, format: str | None = None, **kwargs: Any) -> None:
        """
        Save the wrapped surface.

        If ``format`` is None, the wrapped surface may infer it from ``output_path``.
        Additional keyword arguments are forwarded to the concrete surface save method.

        Parameters
        ----------
        output_path : PathOrStr
            Destination file path.
        format : str, optional
            Output format (e.g. ``'ply'``, ``'vtp'``, ``'motl'``, ``'em'``).
            If None, inferred from the file suffix.
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

    def clean_by_normals(self, max_angle_deg: float = 90.0) -> "PleomorphicSurface":
        """Remove points whose normal deviates more than ``max_angle_deg`` from the mean direction.

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
        self.surface.filter_by_normal_orientation(
            angle_threshold=max_angle_deg,
            inplace=True,
        )
        print("Cleaned by normals (angle vs mean)")
        return self

    def separate_surfaces(
        self,
        surface_type: str = 'closed',
        threshold_angle: float = 90.0,
        reference_point: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate the two halves of a Mesh or OrientedPointCloud surface.

        Parameters
        ----------
        surface_type : str, default='closed'
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
        if surface_type == 'closed':
            return self.surface.separate_closed_surface(threshold_angle, reference_point)
        elif surface_type == 'planar':
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

    def distance_to_points(self, target: np.ndarray, compute_occupancy: bool = True, compute_signed: bool = False, 
                                    return_closest_points: bool = False) -> dict:
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
                raise TypeError(
                    "OrientedPointCloud does not support occupancy or signed distance queries; use Mesh."
                )
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
        
        distances = dist_result['distances']
        
        # Find points within threshold
        mask = distances <= threshold
        indices = np.where(mask)[0]
        
        result = {
            'mask': mask,
            'distances': distances,
            'indices': indices,
            'within_points': target[mask],
            'n_within': np.sum(mask),
            'n_total': len(target)
        }
        
        return result

    def get_neighboring_triangles(self, triangle_id: int, method: str = 'edge-connected', **kwargs: Any) -> set | dict:
        """
        Get neighboring triangles (Mesh only).
        
        Parameters
        ----------
        triangle_id : int
            ID of the seed triangle
        method : str, default='edge-connected'
            Method to use:
            - 'edge-connected': edge-connected triangles
            - 'radius': distance-based 
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
        
        if method == 'edge-connected':
            max_hops = kwargs.get('max_hops', 1)
            return self.surface.get_connected_triangles(triangle_id, max_hops=max_hops)
        
        elif method == 'radius':
            if 'radius' not in kwargs:
                raise ValueError("'radius' parameter required for method='radius'")
            radius = kwargs['radius']
            use_kdtree = kwargs.get('use_kdtree', True)
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
        return self.surface.get_triangles_within_radius(
            triangle_id, radius, use_kdtree=use_kdtree
        )

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
            raise TypeError(
                f"Unsupported surface type: {type(surface)}. Must be Mesh or OrientedPointCloud"
            )

        # Compute orientation metrics if requested
        if return_orientations:
            origins = rays[:, :3]
            directions = rays[:, 3:]

            # Normalize ray directions
            dir_magnitudes = np.linalg.norm(directions, axis=1, keepdims=True)
            ray_directions = directions / (dir_magnitudes + 1e-10)

            # Get surface orientations at hit points based on target_orientation
            surface_orientations_hits = DiscreteSurface.ray_hit_orientations(
                surface, result, target_orientation
            )
            
            if surface_orientations_hits is not None:
                # Create full-size array for all rays (fill with NaN for non-hits)
                hit_mask = np.isfinite(result['t_hit'])
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
                
                result['ray_directions'] = ray_directions
                result['surface_orientations'] = surface_orientations
                result['angles_deg'] = angles_deg
                result['dot_products'] = dot_products
            else:
                # No orientations available
                result['ray_directions'] = ray_directions
                result['surface_orientations'] = np.full((len(rays), 3), np.nan)
                result['angles_deg'] = np.full(len(rays), np.nan)
                result['dot_products'] = np.full(len(rays), np.nan)
        
        return result

    def invalidate_caches(self) -> None:
        """Invalidate cached geometry on the wrapped surface (mesh ray scene, neighbor trees, etc.)."""
        if isinstance(self.surface, Mesh):
            self.surface._invalidate_cache()
            self.surface._invalidate_neighbor_cache()

    def distance_to_pointcloud(self, 
                                    target: 'PleomorphicSurface' | OrientedPointCloud,
                                    method: str = 'nn_unoriented',
                                    max_distance: float | None = None,
                                    ray_length: float | None = None,
                                    reverse_normals: bool = False,
                                    bidirectional: bool = False,
                                    one_hit_per_target: bool = False,
                                    knn_radius: float = 10.0,
                                    return_stats: bool = True) -> dict:
        """
        Compute distance from this surface to another point cloud surface.
        - If source is Mesh: always uses raycasting
        - If source is OrientedPointCloud search nearest neighbours (unoriented or along normals)
        
        Parameters
        ----------
        target : PleomorphicSurface or OrientedPointCloud
            Target surface. Wrapped targets are unwrapped internally; the concrete
            target must be an OrientedPointCloud.
        method : str, default='nn_unoriented'
            Distance computation method (only used if source is OrientedPointCloud):
            - 'nn_unoriented': Nearest neighbor KDTree search
            - 'nn_oriented': Cast rays along normals from source point cloud
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
            raise TypeError(
                f"Target surface must be OrientedPointCloud, got {type(target_surface).__name__}. "
            )

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
            "Could not infer query_type from result. "
            "Pass query_type='ray' or query_type='distance_to_pointcloud'."
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
            out["used_reverse_normals"] = np.asarray(
                result["used_reverse_normals"]
            )[hit_mask][keep]
        return out

    def _mesh_triangle_curvature_table(self) -> dict[str, np.ndarray]:
        """Per-triangle mean and Gaussian curvature (vertex average over face corners)."""
        if not isinstance(self.surface, Mesh):
            raise TypeError(
                "Triangle curvature table requires a Mesh-backed PleomorphicSurface"
            )
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
            raise TypeError(
                "Vertex-to-triangle lookup requires a Mesh-backed PleomorphicSurface"
            )
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
            raise TypeError(
                "Triangle neighborhoods require a Mesh-backed PleomorphicSurface"
            )

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
                neighbors = self.get_triangles_within_radius(
                    int(triangle_id), radius, use_kdtree=use_kdtree
                )["neighbor_ids"]
                expanded.update(np.asarray(neighbors, dtype=np.intp).tolist())
            cumulative.append(expanded)
            regions[f"r <= {radius:g} nm"] = np.array(sorted(expanded), dtype=int)

        for idx_inner, idx_outer in enumerate(range(len(radii) - 1)):
            r_inner = radii[idx_inner]
            r_outer = radii[idx_inner + 1]
            ring_set = cumulative[idx_inner + 1] - cumulative[idx_inner]
            regions[f"{r_inner:g} < r <= {r_outer:g} nm"] = np.array(
                sorted(ring_set), dtype=int
            )

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
            raise TypeError(
                "Point neighborhoods require an OrientedPointCloud-backed PleomorphicSurface"
            )
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
        raise ValueError(
            "surface_seeds must be 'auto', 'hit_sources', or 'hit_targets'"
        )

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
                raise TypeError(
                    f"surface_element='{surface_element}' requires a Mesh-backed surface"
                )
            seed_triangles = (
                self._triangles_from_vertices(seed_ids)
                if element == "vertices"
                else np.asarray(seed_ids, dtype=np.intp)
            )
            return self.get_triangle_neighborhoods(
                seed_triangles, radii=surface_radii, use_kdtree=use_kdtree
            )

        if element == "points":
            if not isinstance(self.surface, OrientedPointCloud):
                raise TypeError(
                    "surface_element='points' requires an OrientedPointCloud-backed surface"
                )
            return self.get_point_neighborhoods(seed_ids, radii=surface_radii)

        raise ValueError(
            "surface_element must be 'triangles', 'vertices', or 'points'"
        )

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
            parsed = self._parse_ray_hits(
                result, min_distance_source_target, max_distance_source_target
            )
        elif query_type == "distance_to_pointcloud":
            parsed = self._parse_distance_hits(
                result, min_distance_source_target, max_distance_source_target
            )
        else:
            raise ValueError(
                f"query_type must be 'ray' or 'distance_to_pointcloud', got '{query_type}'"
            )

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
                hits_dict["gaussian_curvature"] = self.get_gaussian_curvature()[
                    vert_ids
                ]

        hits = pd.DataFrame(hits_dict)

        out: dict[str, Any] = {"hits": hits}
        if triangle_curvatures is not None:
            out["triangle_curvatures"] = triangle_curvatures

        if surface_radii is not None and len(surface_radii) > 0:
            seed_ids = self._resolve_region_seed_ids(
                parsed, query_type, surface_element, surface_seeds
            )
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
                for (tid, fid) in tomo_keys:
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
                _, _, d1, d2, _ = geom.ray_ellipsoid_intersection_3d(
                    coord[i, :], normal_vectors[i, :], params_array
                )
                new_row = pd.Series({"subtomo_id": fm.df.iloc[i]["subtomo_id"], self.column_name: f, "d1": d1, "d2": d2})
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
                "PolyhedralComplex is abstract; use TetrahedralComplex, "
                "OctahedralComplex, or IcosahedralComplex."
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
        for (_tomo_id, _aff_id), group in self.motl.df.groupby(
            [self.tomo_id_column, self.affiliation_column]
        ):
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
            raise ValueError(
                "No geometry fitted. Call fit_geometry(markers, reference_map) first."
            )
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
        hide=("shift_vecs",),
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
            shift_vecs = self.feature_vectors(
                mode=mode, project_to_sphere=project_to_sphere, radius=radius
            )
        if (
            not isinstance(shift_vecs, np.ndarray)
            or shift_vecs.ndim != 2
            or shift_vecs.shape[1] != 3
        ):
            raise ValueError("shift_vecs should be a numpy array of shape (N, 3)")
        for col in [original_id_col, order_id_col]:
            if col not in self.motl.df.columns:
                raise ValueError(
                    f"original_id_col {col} not found in the columns of the input motive list"
                )

        idx = np.lexsort((shift_vecs[:, 2], shift_vecs[:, 1], shift_vecs[:, 0]))
        shift_vecs = shift_vecs[idx]

        motl_subparticles = []
        for shift in range(len(shift_vecs)):
            df_copy = self.motl.df.copy()
            df_copy["score"] = 0
            df_copy["subtomo_mean"] = 0
            df_copy[original_id_col] = self.motl.df["subtomo_id"]
            df_copy[order_id_col] = shift

            motl_subparticle = cryomotl.Motl(df_copy)
            motl_subparticle.shift_positions(shift_vecs[shift])
            motl_subparticle.update_coordinates()

            target_normal = geom.normalize_vector(shift_vecs[shift])
            reference_normal = np.array([0, 0, 1])
            rotation_axis = geom.normalize_vector(
                np.cross(reference_normal, target_normal)
            )
            rotation_angle = np.arccos(
                np.clip(np.dot(reference_normal, target_normal), -1.0, 1.0)
            )
            rotation = srot.from_rotvec(rotation_angle * rotation_axis)
            motl_subparticle.apply_rotation(rotation)
            motl_subparticle.fill(
                {"phi": np.random.rand(len(motl_subparticle.df)) * 360}
            )
            motl_subparticles.append(motl_subparticle)

        output_motl = motl_subparticles[0]
        for mp in motl_subparticles[1:]:
            output_motl = output_motl + mp

        output_motl.df = output_motl.df.sort_values(
            by=[original_id_col, order_id_col], ascending=[True, True]
        ).reset_index(drop=True)
        output_motl.renumber_particles()
        return cryomotl.motl_converter_kwargs(
            output_motl, output_motl_type, output_path=output_path, **output_kwargs
        )

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
            raise ValueError(
                f"Invalid mode: {mode}. Mode should be one of 'vertices', 'edges', or 'faces'."
            )

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

