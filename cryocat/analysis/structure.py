from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import warnings
import decimal
import os
from scipy.spatial.transform import Rotation as srot
from cryocat.core import cryomotl
from cryocat.core import cryomap
from cryocat.core import cryomask
from cryocat.utils import geom
from cryocat.utils import mathutils
from cryocat.analysis import nnana
from cryocat.utils import ioutils
from cryocat._types import PathOrStr
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
from typing import Any, Callable, Sequence

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
        traced_motl,
        pixel_size=1.0,
        column_name="tomo_id",
        chain_id_col="object_id",
        order_id_col="geom2",
        step_dist_col="geom4",
    ):
        self.traced_motl = cryomotl.Motl.load(traced_motl)
        self.pixel_size = pixel_size
        self.column_name = column_name
        self.chain_id_col = chain_id_col
        self.order_id_col = order_id_col
        self.step_dist_col = step_dist_col

    @classmethod
    def from_motls(
        cls,
        motl_entry,
        motl_exit,
        max_distance,
        min_distance=0,
        column_name="tomo_id",
        pixel_size=1.0,
        output_motl=None,
        chain_id_col="object_id",
        order_id_col="geom2",
        step_dist_col="geom4",
    ):
        """Build a :class:`Chain` by tracing an entry/exit motl pair.

        Calls :func:`nnana.trace_chains` on *motl_entry* and *motl_exit* and
        wraps the resulting traced motl in a :class:`Chain` instance.

        Parameters
        ----------
        motl_entry : str or Motl
            Entry-site particle list.
        motl_exit : str or Motl
            Exit-site particle list.
        max_distance : float
            Maximum allowed step distance (in voxels) between successive
            entry/exit pairs.
        min_distance : float, default=0
            Minimum allowed step distance.
        column_name : str, default='tomo_id'
            Column used to group particles before tracing.
        pixel_size : float, default=1.0
            Pixel size in Å; stored on the instance for later distance scaling.
        output_motl : str, optional
            Path to save the traced motl.
        chain_id_col : str, default='object_id'
            Column that receives the chain identifier.
        order_id_col : str, default='geom2'
            Column that receives the within-chain position index.
        step_dist_col : str, default='geom4'
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
        motl,
        max_distance,
        min_distance=0,
        column_name="tomo_id",
        pixel_size=1.0,
        output_motl=None,
        chain_id_col="object_id",
        order_id_col="geom2",
        step_dist_col="geom4",
    ):
        """Build a :class:`Chain` by tracing a single motl (single-site mode).

        Useful for structures where each particle has only one binding site,
        such as nucleosomes in a chromatin chain.  Passes the same motl as
        both entry and exit to :func:`nnana.trace_chains`.

        Parameters
        ----------
        motl : str or Motl
            Particle list to trace.
        max_distance : float
            Maximum allowed step distance (in voxels).
        min_distance : float, default=0
            Minimum allowed step distance.
        column_name : str, default='tomo_id'
            Column used to group particles before tracing.
        pixel_size : float, default=1.0
            Pixel size in Å.
        output_motl : str, optional
            Path to save the traced motl.
        chain_id_col : str, default='object_id'
            Column that receives the chain identifier.
        order_id_col : str, default='geom2'
            Column that receives the within-chain position index.
        step_dist_col : str, default='geom4'
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

    def _step_distances_and_rotated_coords(self, df):
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

        return np.hstack(
            [
                df[self.column_name].values[:-1].reshape(-1, 1),
                chain_dist,
                centered,
                rotated,
            ]
        )

    def _step_rotations(self, df):
        qp_angles = df[["phi", "theta", "psi"]].values[0:-1, :]
        nn_angles = df[["phi", "theta", "psi"]].values[1:, :]
        rel = nnana.relative_rotations(qp_angles, nn_angles)
        points, eul = nnana.rotations_to_unit_vectors(rel)
        zero_rot = srot.from_euler("zxz", angles=np.zeros_like(qp_angles), degrees=True)
        ang_dist = geom.angular_distance(rel, zero_rot)[0].reshape(-1, 1)
        return np.hstack([ang_dist, points, eul])

    def get_chain_stats(self, min_chain_size=2):
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

        dist_stats = df.groupby([self.column_name, self.chain_id_col]).apply(self._step_distances_and_rotated_coords)
        rot_stats = df.groupby([self.column_name, self.chain_id_col]).apply(self._step_rotations)

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

    def get_occupancy(self, occupancy_id="geom1", output_motl=None):
        """Write the chain length (occupancy) per particle into ``occupancy_id``.

        Each particle receives the length of its chain, i.e. the maximum
        within-chain position index in ``order_id_col``.  The result is stored
        in ``self.traced_motl`` in place.

        Parameters
        ----------
        occupancy_id : str, default='geom1'
            Column name that receives the chain-length value.
        output_motl : str, optional
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

    def add_traced_info(self, input_motl, output_motl_path=None, sort_by_subtomo=True, occupancy_id="geom1"):
        """Copy chain columns from the traced motl onto *input_motl*.

        The columns ``occupancy_id``, ``order_id_col``, ``step_dist_col``, and
        ``chain_id_col`` are transferred by matching ``subtomo_id`` values.
        If occupancy has not yet been computed it is computed first.

        Parameters
        ----------
        input_motl : str or Motl
            Target motl that will receive the chain annotations.
        output_motl_path : str, optional
            Path to save the annotated motl.
        sort_by_subtomo : bool, default=True
            Sort both motls by ``subtomo_id`` before copying to ensure correct
            row alignment.
        occupancy_id : str, default='geom1'
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

    def get_class_chain_occupancies(self, mode="mp", occupancy_id="geom1", class_col="class"):
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
        occupancy_id : str, default='geom1'
            Column that holds chain-length values.  Computed automatically
            if not yet present.
        class_col : str, default='class'
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
# NPC
# =============================================================================


class NPC:
    """Static helpers for Nuclear Pore Complex (NPC) particle-list analysis.

    All methods are ``@staticmethod``; the class is used purely as a
    namespace.  Typical workflow:

    1. :meth:`cluster_subunits_to_rings` — trace subunits into rings and
       merge nearby rings.
    2. :meth:`unify_nn_orientations` — flip ambiguous orientations so that
       all subunits in a ring point consistently.
    3. :meth:`merge_rings` — merge rings from multiple ring-motls.
    """

    @staticmethod
    def compute_diameter(input_motl, pixel_size=1.0, su_id="geom2"):
        """Compute the average diameter of opposite-subunit pairs within NPCs.

        Pairs are defined by the symmetric layout of an 8-fold NPC:
        ``(1,5)``, ``(2,6)``, ``(3,7)``, ``(4,8)``.  For each NPC
        (``tomo_id`` × ``object_id`` group) the mean pairwise distance of
        available opposite pairs is returned.

        Parameters
        ----------
        input_motl : str or Motl
            Subunit particle list.  The ``su_id`` column must contain
            subunit indices 1–8.
        pixel_size : float, default=1.0
            Pixel size in Å; distances are multiplied by this value.
        su_id : str, default='geom2'
            Column that identifies the subunit index within each NPC.

        Returns
        -------
        numpy.ndarray
            Mean diameter per NPC, in physical units (pixel_size × voxels).
        """
        motl_ri = cryomotl.Motl.load(input_motl)
        motl_ri.df.reset_index(inplace=True, drop=True)

        pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]

        def get_pairs_indices(group):
            indices = []
            for pair in pairs:
                if pair[0] in group[su_id].values and pair[1] in group[su_id].values:
                    indices.append(
                        [
                            group.loc[group[su_id] == pair[0]].index.tolist()[0],
                            group.loc[group[su_id] == pair[1]].index.tolist()[0],
                        ]
                    )
            return indices

        result = (
            motl_ri.df.groupby(["tomo_id", "object_id"]).apply(get_pairs_indices).reset_index(name="su_pairs_indices")
        )
        pairs_only = result.loc[result["su_pairs_indices"].apply(bool), "su_pairs_indices"].to_list()
        motl_idx = np.array([item for sublist in pairs_only for item in sublist])

        coord = motl_ri.get_coordinates()
        distances = geom.point_pairwise_dist(coord[motl_idx[:, 0], :], coord[motl_idx[:, 1], :])

        df_values = np.full(len(motl_ri.df), np.nan)
        df_values[motl_idx[:, 0]] = distances * pixel_size
        motl_ri.df["su_distance"] = df_values

        npc_diameters_df = motl_ri.df.groupby(["tomo_id", "object_id"])["su_distance"].mean().dropna()
        return npc_diameters_df.values

    @staticmethod
    def unify_nn_orientations(input_motl, dist_threshold=10000):
        """Flip subunit orientations so that all neighbours point consistently.

        Traces particles into a chain and then walks each chain, applying a
        180° rotation around the X axis whenever the cone angle between
        successive subunits exceeds 90°.

        Parameters
        ----------
        input_motl : str or Motl
            Subunit particle list.
        dist_threshold : float, default=10000
            Maximum nearest-neighbour distance used during tracing (voxels).

        Returns
        -------
        Motl
            A new motl with updated Euler angles, sorted by ``subtomo_id``.
        """
        traced_motl = nnana.trace_chains(
            input_motl,
            motl_exit=None,
            max_distance=dist_threshold,
            min_distance=0,
            column_name="tomo_id",
            output_motl=None,
            store_idx1="object_id",
            store_idx2="geom2",
            store_dist="geom4",
        )

        rot_180 = srot.from_euler("zxz", angles=[0, 180, 0], degrees=True)

        for t in traced_motl.get_unique_values("tomo_id"):
            tm = traced_motl.get_motl_subset(column_values=[t], column_name="tomo_id", reset_index=True)
            rotations = tm.get_rotations()
            for i in np.arange(1, tm.df["geom2"].max(), dtype=int):
                cone_angle = geom.cone_distance(rotations[i - 1], rotations[i])
                if cone_angle > 90.0:
                    rotations[i] = rotations[i] * rot_180

            angles = rotations.as_euler("zxz", degrees=True)
            tm.fill({"angles": angles})
            traced_motl.df.loc[traced_motl.df["tomo_id"] == t, :] = tm.df.values

        return cryomotl.Motl(traced_motl.df.sort_values(by="subtomo_id"))

    @staticmethod
    def cluster_subunits_to_rings(
        input_motl_path,
        mask_size,
        entry_mask_coord,
        exit_mask_coord,
        npc_radius,
        max_trace_distance,
        min_trace_distance=0,
    ):
        """Cluster NPC subunit particles into rings.

        Workflow:

        1. Create temporary spherical entry/exit masks at *entry_mask_coord*
           and *exit_mask_coord*.
        2. Re-centre the input motl to the entry and exit sub-particle
           positions.
        3. Trace entry/exit pairs into chains with
           :meth:`Chain.from_motls`.
        4. Copy chain annotations onto the original motl and merge nearby
           subunits with :meth:`merge_subunits`.

        Parameters
        ----------
        input_motl_path : str
            Path to the subunit motl.  Temporary mask files are written to
            the same directory.
        mask_size : int or array-like
            Box size for the temporary spherical masks.
        entry_mask_coord : array-like
            Centre of the entry spherical mask (voxels).
        exit_mask_coord : array-like
            Centre of the exit spherical mask (voxels).
        npc_radius : float
            Approximate NPC ring radius (voxels) used by
            :meth:`merge_subunits`.
        max_trace_distance : float
            Maximum allowed step distance during chain tracing (voxels).
        min_trace_distance : float, default=0
            Minimum allowed step distance during chain tracing.

        Returns
        -------
        Motl
            Motl with ``object_id`` identifying each ring, ``geom1`` holding
            ring occupancy, and ``geom2`` the within-ring subunit index.
        """
        working_dir, _ = os.path.split(input_motl_path)
        entry_mask = working_dir + "entry_mask.em"
        exit_mask = working_dir + "exit_mask.em"
        _ = cryomask.spherical_mask(mask_size, 3, center=entry_mask_coord, output_path=entry_mask)
        _ = cryomask.spherical_mask(mask_size, 3, center=exit_mask_coord, output_path=exit_mask)

        motl = cryomotl.Motl.load(input_motl_path)
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

        new_traced_motl = NPC.merge_subunits(motl, npc_radius=npc_radius)

        os.remove(entry_mask)
        os.remove(exit_mask)

        return new_traced_motl

    @staticmethod
    def get_center_with_radius(object_motl, radius):
        """Estimate the NPC centre by shifting each subunit inward by *radius*.

        Shifts every particle by ``(-radius, 0, 0)`` along the local X axis
        (i.e. toward the pore centre) and returns the mean of the shifted
        positions.

        Parameters
        ----------
        object_motl : Motl
            Subunit particles belonging to a single NPC ring.
        radius : float
            Approximate ring radius in voxels.

        Returns
        -------
        numpy.ndarray
            Estimated centre coordinates, shape ``(3,)``.
        """
        vector_x = np.asarray([-radius, 0, 0])
        shifted_motl = cryomotl.Motl(object_motl.df.copy())
        shifted_motl.shift_positions(vector_x)
        center_coordinates = shifted_motl.get_coordinates()
        return np.mean(center_coordinates, axis=0)

    @staticmethod
    def get_center_and_radius(object_motl, include_singles=False):
        """Fit a circle to the subunit positions and return its centre and radius.

        For fewer than 4 particles the centre is estimated from ray-ray
        intersections; for 4 or more the Pratt algebraic fit is used.

        Parameters
        ----------
        object_motl : Motl
            Subunit particles belonging to a single NPC ring.
        include_singles : bool, default=False
            When ``True``, a single-particle motl returns its own coordinates
            with radius 0 instead of a zero vector.

        Returns
        -------
        circle_center : numpy.ndarray
            Centre of the fitted circle, shape ``(3,)``.
        circle_radius : float
            Radius of the fitted circle (0.0 for ≤ 3 particles).
        """
        vector_x = np.asarray([-1, 0, 0])

        if object_motl.df.shape[0] <= 1:
            if include_singles:
                return object_motl.get_coordinates(), 0
            else:
                return np.zeros((3,)), 0

        start_coord = object_motl.get_coordinates()
        if object_motl.df.shape[0] <= 3:
            rot = object_motl.get_rotations()
            rot_vec = rot.apply(vector_x)
            end_coord = start_coord + rot_vec
            circle_center, _ = geom.ray_ray_intersection_3d(starting_points=start_coord, ending_points=end_coord)
            circle_radius = 0.0
        else:
            circle_center, circle_radius, confidence = geom.fit_circle_3d_pratt(start_coord)

        return circle_center, circle_radius

    @staticmethod
    def get_centers_as_motl(tomo_motl, tomo_id, radius):
        """Build a motl of estimated NPC centres for all rings in one tomogram.

        For each unique ``object_id`` in *tomo_motl* the ring centre is
        estimated via :meth:`get_center_with_radius` and collected into a
        new motl.

        Parameters
        ----------
        tomo_motl : Motl
            Subunit motl for a single tomogram.
        tomo_id : int or float
            Tomogram identifier written into the output motl.
        radius : float
            Approximate ring radius in voxels, forwarded to
            :meth:`get_center_with_radius`.

        Returns
        -------
        Motl
            One particle per ring, with ``x/y/z`` at the estimated centre and
            ``object_id`` matching the source ring identifier.
        """
        central_points = []
        object_idx = []
        for o in tomo_motl.get_unique_values("object_id"):
            om = tomo_motl.get_motl_subset(column_values=[o], column_name="object_id", reset_index=True)
            cetroid = NPC.get_center_with_radius(om, radius)
            central_points.append(cetroid)
            object_idx.append(o)

        new_object_motl = cryomotl.Motl()
        if len(central_points) > 0:
            central_points = np.vstack(central_points)
            new_object_motl.fill(
                {
                    "x": central_points[:, 0],
                    "y": central_points[:, 1],
                    "z": central_points[:, 2],
                    "tomo_id": tomo_id,
                    "object_id": object_idx,
                }
            )
            new_object_motl.renumber_particles()

        new_object_motl.df.fillna(0.0, inplace=True)
        return new_object_motl

    @staticmethod
    def get_new_subunit_idx(object_motl, npc_radius, symmetry=8):
        """Assign angular subunit indices after ring merging.

        Computes the angle of each subunit relative to the first one, divides
        by the expected angular step (``360 / symmetry``), and rounds to the
        nearest integer to obtain a 1-based subunit index.

        Parameters
        ----------
        object_motl : Motl
            Subunit particles belonging to a single merged NPC ring.
        npc_radius : float
            Approximate ring radius in voxels, used to locate the ring centre.
        symmetry : int, default=8
            Rotational symmetry order of the NPC (determines the angular step).

        Returns
        -------
        list of int
            Subunit indices, same length as ``object_motl.df``.  The first
            particle always receives index 1.
        """
        npc_center = NPC.get_center_with_radius(object_motl, npc_radius)
        su_coord = object_motl.get_coordinates()
        vectors = su_coord - np.tile(npc_center, (su_coord.shape[0], 1))

        div_angle = 360.0 / symmetry
        s_idx = [1]
        for vec in vectors[1:]:
            angle = geom.vector_angular_distance(vectors[0], vec) / div_angle
            s_idx.append(int(decimal.Decimal(angle).to_integral_value(rounding=decimal.ROUND_HALF_UP)) + 1)

        return s_idx

    @staticmethod
    def merge_subunits(input_motl, npc_radius=55):
        """Merge NPC rings whose centres are closer than *npc_radius*.

        For each tomogram:

        1. Estimate ring centres with :meth:`get_centers_as_motl`.
        2. Find rings whose centres are within *npc_radius* of each other
           using :func:`nnana.get_nn_within_distance`.
        3. Re-assign the ``object_id`` of close rings to the nearest partner.
        4. Re-compute ``geom1`` (ring size) and ``geom2`` (subunit index) for
           merged rings.

        Parameters
        ----------
        input_motl : str or pandas.DataFrame or Motl
            Subunit motl, already annotated with ``object_id`` ring labels.
        npc_radius : float, default=55
            Distance threshold in voxels.  Ring centres closer than this are
            merged into one ring.

        Returns
        -------
        Motl
            Updated motl with consolidated ring labels.
        """
        if isinstance(input_motl, (str, pd.DataFrame)):
            input_motl = cryomotl.Motl.load(input_motl)

        for t in input_motl.get_unique_values("tomo_id"):
            tm = input_motl.get_motl_subset(column_values=[t], column_name="tomo_id", reset_index=True)
            new_object_motl = NPC.get_centers_as_motl(tm, t, radius=npc_radius)

            changed_objects = []
            if new_object_motl.df.shape[0] > 1:
                center_stats = nnana.get_nn_stats(new_object_motl, new_object_motl)

                if any(center_stats["distance"] <= npc_radius):
                    center_idx, nn_idx = nnana.get_nn_within_distance(new_object_motl, npc_radius)
                    for i, o in enumerate(center_idx):
                        o_id1 = new_object_motl.df.loc[new_object_motl.df.index[o], "object_id"]
                        changed_objects.append(o_id1)
                        for j in nn_idx[i]:
                            o_id2 = new_object_motl.df.loc[new_object_motl.df.index[j], "object_id"]
                            tm.df.loc[tm.df["object_id"] == o_id2, "object_id"] = o_id1

            tm.df["geom1"] = tm.df.groupby(["object_id"])["object_id"].transform("count")

            for o in changed_objects:
                om = tm.get_motl_subset(column_values=o, column_name="object_id", reset_index=True)
                s_idx = NPC.get_new_subunit_idx(om, npc_radius)
                tm.df.loc[tm.df["object_id"] == o, "geom2"] = s_idx

            tm.df["object_id"] = tm.df["object_id"].rank(method="dense").astype(int)

            input_motl.df.loc[input_motl.df["tomo_id"] == t, ["object_id", "geom1", "geom2"]] = tm.df[
                ["object_id", "geom1", "geom2"]
            ].values

        input_motl.df.reset_index(inplace=True, drop=True)
        input_motl.df["geom1"] = input_motl.df.groupby(["tomo_id", "object_id"])["object_id"].transform("count")
        input_motl.df["object_id"] = input_motl.df["object_id"].rank(method="dense").astype(int)

        return input_motl

    @staticmethod
    def merge_rings(input_motls, npc_radius, distance_threshold=40):
        """Merge corresponding rings across multiple ring-motls.

        Assigns sequential ``object_id`` values across all motls, then for
        every pair of motls finds rings (by their estimated centres) that are
        closer than *distance_threshold* and merges their ``object_id``
        entries.

        Parameters
        ----------
        input_motls : list of str or Motl
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
                    centers1 = NPC.get_centers_as_motl(tm1, t, radius=npc_radius)
                    centers2 = NPC.get_centers_as_motl(tm2, t, radius=npc_radius)

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

    def __init__(self, surface: Mesh | OrientedPointCloud | "PleomorphicSurface"):
        if isinstance(surface, PleomorphicSurface):
            surface = surface.surface
        if not isinstance(surface, (Mesh, OrientedPointCloud)):
            raise TypeError(
                f"Unsupported surface type: {type(surface)}. "
                "Must be Mesh, OrientedPointCloud, or PleomorphicSurface."
            )
        self.surface = surface

    @staticmethod
    def _unwrap_surface(surface):
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
    def read(cls, input_path: PathOrStr, method: str = "mesh", **kwargs) -> "PleomorphicSurface":
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

    def get_surface_area(self) -> float:
        """Return total surface area of a mesh-backed surface."""
        if not isinstance(self.surface, Mesh):
            raise TypeError(
                "get_surface_area is only available for Mesh-backed PleomorphicSurface. "
                "An OrientedPointCloud has no face connectivity from which to compute area."
            )
        return self.surface.get_surface_area()

    def save(self, output_path: PathOrStr, format: str | None = None, **kwargs) -> None:
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

    def compute_normals(self, **kwargs) -> "PleomorphicSurface":
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

    def flip_normals(self, inplace: bool = True, **kwargs) -> "PleomorphicSurface" | None:
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
        **kwargs,
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

    def remove_nonfinite_vertices(self, inplace: bool = True, **kwargs) -> "PleomorphicSurface":
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

    def oversample(self, **kwargs) -> "PleomorphicSurface":
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

    def extract_region(self, indices: np.ndarray, element: str = "triangles", **kwargs) -> "PleomorphicSurface":
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
        element : str, default='triangles'
            Which surface primitive ``indices`` refers to:

            - ``'triangles'`` (Mesh only): select by triangle index.
            - ``'points'`` (OrientedPointCloud only): select by point index.
            - ``'mask'`` (OrientedPointCloud only): boolean selection mask.

        **kwargs
            Forwarded to :meth:`Mesh.extract_submesh` or
            :meth:`OrientedPointCloud.extract_points`.

        Returns
        -------
        PleomorphicSurface
            New wrapper containing only the extracted elements.
        """
        element = str(element).lower()
        if isinstance(self.surface, Mesh):
            if element not in ("triangle", "triangles"):
                raise ValueError("Mesh subregions currently support element='triangles'")
            out = self.surface.extract_submesh(indices, **kwargs)
            return PleomorphicSurface(out)

        if isinstance(self.surface, OrientedPointCloud):
            if element in ("point", "points"):
                out = self.surface.extract_points(point_ids=indices, **kwargs)
            elif element == "mask":
                out = self.surface.extract_points(mask=indices, **kwargs)
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

    def get_neighboring_triangles(self, triangle_id: int, method: str = 'edge-connected', **kwargs) -> set | dict:
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

    def ray_intersections(self,
                                rays: np.ndarray,
                                one_hit_per_target: bool = False,
                                knn_radius: float = 10.0,
                                return_orientations: bool = False,
                                target_orientation: str | Callable = 'normal') -> dict:
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
        target_orientation : str or callable, default='normal'
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
        radii: Sequence[float],
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
        radii : Sequence[float]
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
        radii: Sequence[float],
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
        radii : Sequence[float]
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
        surface_radii: Sequence[float],
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
        query_type: str | None = None,
        min_distance_source_target: float | None = None,
        max_distance_source_target: float | None = None,
        source_id_name: str = "source_id",
        target_id_name: str = "target_id",
        include_curvatures: bool = True,
        surface_radii: list[float] | None = None,
        surface_element: str | None = None,
        surface_seeds: str = "auto",
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
        query_type : str, optional
            ``'ray'`` or ``'distance_to_pointcloud'``. Inferred from ``result`` when
            omitted.
        min_distance_source_target, max_distance_source_target : float, optional
            Keep hits whose source-target distance lies in this interval.
        source_id_name, target_id_name : str
            Column names for source and target indices in the hit table.
        include_curvatures : bool, default=True
            Attach curvature columns for mesh-backed ``self``.
        surface_radii : sequence of float, optional
            Radii for cumulative regions around hit seeds on ``self``.
        surface_element : str, optional
            ``'triangles'``, ``'vertices'`` (mesh only), or ``'points'`` (point cloud
            only). Defaults: ray+mesh → triangles; distance+mesh → vertices;
            point cloud → points.
        surface_seeds : str, default='auto'
            Which hit IDs seed expansion: ``'auto'``, ``'hit_sources'``, or
            ``'hit_targets'``.
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
    feature_id : str
        Column name used as the surface-object identifier.
    """

    def __init__(self, quadrics: QuadricsM, feature_id: str = "object_id"):
        self.quadrics = quadrics
        self.feature_id = feature_id

    @classmethod
    def from_motl(cls, input_motl, surface_type: str = "ellipsoid", feature_id: str = "object_id") -> "ParametricSurface":
        """Fit analytic surfaces to particle groups and return a ParametricSurface.

        Parameters
        ----------
        input_motl : str or Motl
            Input particle list.  One surface is fitted per unique
            ``(tomo_id, feature_id)`` group.
        surface_type : str, default='ellipsoid'
            Quadric type; currently only ``'ellipsoid'`` is supported.
        feature_id : str, default='object_id'
            Column used to group particles.

        Returns
        -------
        ParametricSurface
        """
        quadrics = QuadricsM(input_motl, quadric=surface_type, feature_id=feature_id)
        return cls(quadrics, feature_id=feature_id)

    @classmethod
    def from_csv(cls, path: str, surface_type: str = "ellipsoid", feature_id: str = "object_id") -> "ParametricSurface":
        """Load analytic surface parameters from a CSV file.

        Parameters
        ----------
        path : str
            Path to a CSV produced by :meth:`write_out`.
        surface_type : str, default='ellipsoid'
        feature_id : str, default='object_id'

        Returns
        -------
        ParametricSurface
        """
        quadrics = QuadricsM(path, quadric=surface_type, feature_id=feature_id)
        return cls(quadrics, feature_id=feature_id)

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
        input_motl,
        output_path=None,
        store_id: str = "geom4",
    ):
        """Compute the shortest distance from each particle to its assigned surface.

        Parameters
        ----------
        input_motl : str or Motl
            Particles with ``feature_id`` already assigned.
        output_path : str, optional
            Path to save the result.
        store_id : str, default='geom4'
            Column that receives the distance values.

        Returns
        -------
        Motl
            Input motl with ``store_id`` populated.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.feature_id)
        assigned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.feature_id, reset_index=True)
            coord = fm.get_coordinates()
            tomo_id = fm.df["tomo_id"].values[0]
            fm.df[store_id] = self.quadrics.distance_point_surface(tomo_id, f, coord)
            assigned_motl_df = pd.concat([assigned_motl_df, fm.df])

        assigned_motl = cryomotl.Motl(assigned_motl_df)
        assigned_motl.df.reset_index(drop=True, inplace=True)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    def assign_affiliation_distance_based(
        self,
        input_motl,
        output_path=None,
        unassigned_value=None,
    ):
        """Assign each particle to the nearest surface centre.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to assign.
        output_path : str, optional
            Path to save the result.
        unassigned_value : scalar, optional
            When provided, only particles whose current ``feature_id`` equals
            this value are re-assigned; the rest are kept unchanged.

        Returns
        -------
        Motl
            Motl with ``feature_id`` updated.
        """
        in_motl = cryomotl.Motl.load(input_motl)

        if unassigned_value is not None:
            assigned_motl = cryomotl.Motl(in_motl.df)
            in_motl.df = in_motl.df[in_motl.df[self.feature_id] == unassigned_value]
            in_motl.df.reset_index(drop=True, inplace=True)

        tomos = in_motl.get_unique_values(column_name="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(column_values=t, column_name="tomo_id", reset_index=True)
            coord = tm.get_coordinates()
            closest_ids = self.quadrics.find_closest_quadric(t, coord)
            tm.df[self.feature_id] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        if unassigned_value is not None:
            assigned_motl.df.loc[assigned_motl.df[self.feature_id] == unassigned_value, :] = assigned_motl_df.values
        else:
            assigned_motl = cryomotl.Motl(assigned_motl_df)

        assigned_motl.df.reset_index(drop=True, inplace=True)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    # TODO: only Ellipsoid currently supported for intersection methods

    def assign_affiliation_intersection_based(
        self,
        input_motl,
        output_path=None,
        keep_unassigned: bool = True,
    ):
        """Assign each particle to the surface it points toward (ray casting).

        A ray is cast along the negated particle normal.  The particle is
        labelled with the identifier of the surface whose intersection is
        closest along that ray.  Particles that lie inside a surface or have
        no valid intersection receive ``-1``.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to assign.  Euler angles are used to derive normals.
        output_path : str, optional
            Path to save the result.
        keep_unassigned : bool, default=True
            When ``False``, particles with ``feature_id == -1`` are removed.

        Returns
        -------
        Motl
            Motl with ``feature_id`` updated.
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

            tm.df[self.feature_id] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        unassigned = assigned_motl_df[assigned_motl_df[self.feature_id] == -1].shape[0]

        if not keep_unassigned:
            assigned_motl_df = assigned_motl_df[assigned_motl_df[self.feature_id] != -1]

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        print(f"{unassigned} particles did not have any intersection or were inside.")
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    def compute_intersection(self, input_motl):
        """Compute ray–ellipsoid intersection distances for each particle.

        For every particle a ray is cast along ``-euler_angles_to_normals``
        and the two intersection distances with the assigned ellipsoid are
        returned.

        Parameters
        ----------
        input_motl : str or Motl
            Particles grouped by ``feature_id``.

        Returns
        -------
        pandas.DataFrame
            Columns: ``subtomo_id``, ``feature_id``, ``d1``, ``d2``.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.feature_id)
        intersection_points = pd.DataFrame(columns=["subtomo_id", self.feature_id, "d1", "d2"])

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.feature_id, reset_index=True)
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
                new_row = pd.Series({"subtomo_id": fm.df.iloc[i]["subtomo_id"], self.feature_id: f, "d1": d1, "d2": d2})
                intersection_points = pd.concat([intersection_points, new_row.to_frame().T], ignore_index=True)

        return intersection_points

    def compute_normals_angle(
        self,
        input_motl,
        store_id: str = "geom4",
        output_path=None,
    ):
        """Compute the angle between each particle's orientation and the ellipsoid radial normal.

        The radial normal is the vector from the fitted ellipsoid centre to the
        particle.  The stored value is the angle (degrees) between that vector
        and the particle's orientation normal.

        Parameters
        ----------
        input_motl : str or Motl
            Particles grouped by ``feature_id``.
        store_id : str, default='geom4'
            Column that receives the angle values.
        output_path : str, optional
            Path to save the result.

        Returns
        -------
        Motl
            Input motl with ``store_id`` populated.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.feature_id)
        assigned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.feature_id, reset_index=True)
            coord = fm.get_coordinates()
            normals = geom.euler_angles_to_normals(fm.get_angles())
            tomo_id = fm.df["tomo_id"].values[0]
            key = (tomo_id, f)
            if key not in self.quadrics.dict:
                assigned_motl_df = pd.concat([assigned_motl_df, fm.df])
                continue
            center = self.quadrics.dict[key].center
            normals_t = coord - np.tile(center, (coord.shape[0], 1))
            fm.df[store_id] = geom.angle_between_n_vectors(normals, normals_t)
            assigned_motl_df = pd.concat([assigned_motl_df, fm.df])

        assigned_motl_df.index = in_motl.df.index
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    def clean_by_normals(
        self,
        input_motl,
        compute_normals: bool = True,
        normals_id: str = "geom4",
        threshold=None,
        output_path=None,
    ):
        """Remove particles whose orientation deviates too far from the surface normal.

        Parameters
        ----------
        input_motl : str or Motl
        compute_normals : bool, default=True
            Recompute normal angles before filtering.
        normals_id : str, default='geom4'
            Column holding the angle-to-normal values.
        threshold : float, optional
            Maximum allowed angle (degrees).  Defaults to one standard deviation.
        output_path : str, optional

        Returns
        -------
        Motl
        """
        in_motl = cryomotl.Motl.load(input_motl)
        orig_number = in_motl.df.shape[0]

        if compute_normals:
            in_motl = self.compute_normals_angle(in_motl, store_id=normals_id)

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
        input_motl,
        threshold=None,
        output_path=None,
    ):
        """Remove particles that lie too far from the mean ellipsoid radius.

        Parameters
        ----------
        input_motl : str or Motl
        threshold : float, optional
            Half-width of the allowed distance band.  Defaults to one standard deviation.
        output_path : str, optional

        Returns
        -------
        Motl
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(column_name=self.feature_id)
        cleaned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(column_values=f, column_name=self.feature_id, reset_index=True)
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
        input_motl,
        object_motl,
        tomo_dim,
        shell_size,
        column_name: str = "object_id",
        output_path=None,
        radius_offset: float = 0.0,
        motl_radius_id: str = "geom5",
    ):
        """Assign each particle to a surface object using a spherical-shell mask.

        Parameters
        ----------
        input_motl : str or Motl
        object_motl : str or Motl
            Surface-object positions (one row per object).
        tomo_dim : str or array-like
            Tomogram dimensions table; see :func:`ioutils.dimensions_load`.
        shell_size : int
            Thickness of the spherical shell mask.
        column_name : str, default='object_id'
        output_path : str, optional
        radius_offset : float, default=0.0
        motl_radius_id : str, default='geom5'
            Column in *object_motl* holding each object's radius.

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
        input_motl,
        motl_radius_id: str,
        sampling_distance: float,
        sampling_angle: float = 360,
        output_path=None,
    ):
        """Generate oversampled particles on a sphere around each input particle.

        Parameters
        ----------
        input_motl : str or Motl
        motl_radius_id : str
            Column holding the sphere radius for each particle.
        sampling_distance : float
            Angular sampling step forwarded to :func:`geom.sample_cone`.
        sampling_angle : float, default=360
            Half-opening angle of the sampling cone.  ``360`` samples the full sphere.
        output_path : str, optional

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