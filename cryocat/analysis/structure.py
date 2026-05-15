import numpy as np
import pandas as pd
import warnings
import decimal
import os
from cryocat.core import cryomotl
from cryocat.core import cryomap
from cryocat.core import cryomask
from cryocat.utils import geom
from cryocat.utils import mathutils
from cryocat.analysis import nnana
from cryocat.utils import ioutils
from cryocat.core import quadric
from scipy.spatial.transform import Rotation as srot

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
    feature : str, default='tomo_id'
    chain_id_col : str, default='object_id'
    order_id_col : str, default='geom2'
    step_dist_col : str, default='geom4'
    """

    def __init__(
        self,
        traced_motl,
        pixel_size=1.0,
        feature="tomo_id",
        chain_id_col="object_id",
        order_id_col="geom2",
        step_dist_col="geom4",
    ):
        self.traced_motl = cryomotl.Motl.load(traced_motl)
        self.pixel_size = pixel_size
        self.feature = feature
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
        feature="tomo_id",
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
        feature : str, default='tomo_id'
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
            feature=feature,
            output_motl=output_motl,
            store_idx1=chain_id_col,
            store_idx2=order_id_col,
            store_dist=step_dist_col,
        )
        return cls(
            traced,
            pixel_size=pixel_size,
            feature=feature,
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
        feature="tomo_id",
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
        feature : str, default='tomo_id'
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
            feature=feature,
            output_motl=output_motl,
            store_idx1=chain_id_col,
            store_idx2=order_id_col,
            store_dist=step_dist_col,
        )
        return cls(
            traced,
            pixel_size=pixel_size,
            feature=feature,
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
                df[self.feature].values[:-1].reshape(-1, 1),
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
        df.sort_values([self.feature, self.chain_id_col, self.order_id_col], inplace=True)
        chain_sizes = df.groupby([self.feature, self.chain_id_col])[self.order_id_col].transform("max")
        df = df[chain_sizes >= min_chain_size]
        if df.empty:
            return pd.DataFrame()

        dist_stats = df.groupby([self.feature, self.chain_id_col]).apply(self._step_distances_and_rotated_coords)
        rot_stats = df.groupby([self.feature, self.chain_id_col]).apply(self._step_rotations)

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
        self.traced_motl.df[occupancy_id] = self.traced_motl.df.groupby([self.feature, self.chain_id_col])[
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
        input_motl.df.sort_values([self.feature, self.chain_id_col, self.order_id_col], inplace=True)

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
# NPC, MAK, PleomorphicSurface
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
            feature="tomo_id",
            output_motl=None,
            store_idx1="object_id",
            store_idx2="geom2",
            store_dist="geom4",
        )

        rot_180 = srot.from_euler("zxz", angles=[0, 180, 0], degrees=True)

        for t in traced_motl.get_unique_values("tomo_id"):
            tm = traced_motl.get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
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
        shifted_motl = cryomotl.Motl(object_motl.df)
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
            om = tomo_motl.get_motl_subset(feature_values=[o], feature_id="object_id", reset_index=True)
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
            tm = input_motl.get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
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
                om = tm.get_motl_subset(feature_values=o, feature_id="object_id", reset_index=True)
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
                tm1 = ring_motls[i[0]].get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
                tm2 = ring_motls[i[1]].get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
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


class PleomorphicSurface:
    """Static helpers for assigning particles to pleomorphic surfaces.

    All methods are ``@staticmethod``; the class is used as a namespace.
    Surfaces are described by fitted ellipsoid parameters (centre, semi-axes,
    eigenvectors, and the 10 coefficients of the implicit quadric).
    """

    @staticmethod
    def get_parametric_description(input_motl, feature_id="object_id", output_path=None):
        """Fit an ellipsoid to each group of particles and return the parameters.

        Parameters
        ----------
        input_motl : str or Motl
            Input particle list.
        feature_id : str, default='object_id'
            Column used to group particles (one ellipsoid per unique value).
        output_path : str, optional
            Path for saving the parameter table as a CSV.

        Returns
        -------
        pandas.DataFrame
            One row per feature value with columns ``tomo_id``,
            ``{feature_id}``, ``cx/cy/cz`` (centre), ``rx/ry/rz``
            (semi-axes), ``ev1–ev3 x/y/z`` (eigenvectors), and
            ``p1–p10`` (implicit quadric coefficients).
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(feature_id=feature_id)
        el_params_all = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id)
            coord = fm.get_coordinates()
            el_params = PleomorphicSurface.load_parametric_surface(feature_id=feature_id)
            center, radii, evecs, v = geom.fit_ellipsoid(coord)
            el_params["tomo_id"] = [fm.df.iloc[0]["tomo_id"]]
            el_params[feature_id] = [f]
            el_params[["cx", "cy", "cz"]] = [center]
            el_params[["rx", "ry", "rz"]] = [radii]
            el_params[["ev1x", "ev1y", "ev1z"]] = [evecs[0, :]]
            el_params[["ev2x", "ev2y", "ev2z"]] = [evecs[1, :]]
            el_params[["ev3x", "ev3y", "ev3z"]] = [evecs[2, :]]
            el_params[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]] = [v]
            el_params_all = pd.concat([el_params_all, el_params])

        el_params_all.reset_index(drop=True, inplace=True)

        if output_path is not None:
            el_params_all.to_csv(output_path, index=False)

        return el_params_all

    @staticmethod
    def load_parametric_surface(parametric_surface=None, feature_id="object_id"):
        """Load or initialise the ellipsoid parameter table.

        Parameters
        ----------
        parametric_surface : Motl or str or numpy.ndarray or None, optional
            Source of surface parameters:

            :class:`Motl`
                Fit ellipsoids via :meth:`get_parametric_description`.
            ``str``
                Read a CSV file.
            :class:`numpy.ndarray`
                Wrap in a DataFrame using the expected column names.
            ``None``
                Return an empty DataFrame with the correct columns.
        feature_id : str, default='object_id'
            Column used as the group identifier.

        Returns
        -------
        pandas.DataFrame
            Surface parameter table with columns matching the schema
            produced by :meth:`get_parametric_description`.

        Raises
        ------
        ValueError
            When *parametric_surface* has an unsupported type.
        """
        columns = [
            "tomo_id",
            feature_id,
            "cx",
            "cy",
            "cz",
            "rx",
            "ry",
            "rz",
            "ev1x",
            "ev1y",
            "ev1z",
            "ev2x",
            "ev2y",
            "ev2z",
            "ev3x",
            "ev3y",
            "ev3z",
            "p1",
            "p2",
            "p3",
            "p4",
            "p5",
            "p6",
            "p7",
            "p8",
            "p9",
            "p10",
        ]

        if isinstance(parametric_surface, cryomotl.Motl):
            el_params = PleomorphicSurface.get_parametric_description(parametric_surface, feature_id=feature_id)
        elif isinstance(parametric_surface, str):
            el_params = pd.read_csv("parametric_surface")
        elif isinstance(parametric_surface, np.ndarray):
            el_params = pd.DataFrame(data=parametric_surface, columns=columns)
        elif not parametric_surface:
            el_params = pd.DataFrame(columns=columns)
        else:
            raise ValueError("Invalid type of parametric surface")

        return el_params

    @staticmethod
    def assign_affiliation_mask_based(
        input_motl,
        object_motl,
        tomo_dim,
        shell_size,
        feature_id="object_id",
        output_path=None,
        radius_offset=0.0,
        motl_radius_id="geom5",
    ):
        """Assign each particle to a surface object using a shell mask.

        For every surface object a spherical-shell mask is placed at its
        position.  Particles whose tomogram coordinates fall within the mask
        are labelled with that object's ``feature_id``.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to be assigned.
        object_motl : str or Motl
            Surface-object positions (one row per object).
        tomo_dim : str or array-like
            Tomogram dimensions table; see :func:`ioutils.dimensions_load`.
        shell_size : int
            Thickness of the spherical shell used for the mask.
        feature_id : str, default='object_id'
            Column to write the assigned object identifier into.
        output_path : str, optional
            Path to save the assigned motl.
        radius_offset : float, default=0.0
            Additional offset added to the object radius when building the
            shell mask.
        motl_radius_id : str, default='geom5'
            Column in *object_motl* that holds each object's radius.

        Returns
        -------
        Motl
            Particles with ``feature_id`` set to their assigned object.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        object_motl = cryomotl.Motl.load(object_motl)
        tomo_dim = ioutils.dimensions_load(tomo_dim)
        tomos = in_motl.get_unique_values(feature_id="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(feature_values=t, feature_id="tomo_id", reset_index=True)
            tm_dim = tomo_dim.loc[tomo_dim["tomo_id"] == t, ["x", "y", "z"]].values[0]
            coords = tm.get_coordinates().astype(int)
            print(t)
            tom = object_motl.get_motl_subset(feature_values=t, feature_id="tomo_id")
            for o in tom.get_unique_values(feature_id=feature_id):
                om = tom.get_motl_subset(feature_values=o, feature_id=feature_id)
                om.df["class"] = 1
                to_radius = tom.df.iloc[0][motl_radius_id] + radius_offset
                object_mask = cryomask.generate_mask("s_shell_r" + str(int(to_radius)) + "_s" + str(int(shell_size)))
                tomo_mask = cryomap.place_object(object_mask, om, volume_shape=tm_dim, feature_to_color="class")
                mask_values = tomo_mask[coords[:, 0], coords[:, 1], coords[:, 2]]
                idx_to_keep = np.where(mask_values == 1)[0]
                tm.df[feature_id] = o
                assigned_motl_df = pd.concat([assigned_motl_df, tm.df.iloc[idx_to_keep]])

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    @staticmethod
    def compute_point_surface_distance(
        input_motl,
        parametric_surface,
        surface_type="ellipsoid",
        feature_id="object_id",
        output_path=None,
        store_id="geom4",
    ):
        """Compute the signed distance from each particle to its assigned surface.

        Parameters
        ----------
        input_motl : str or Motl
            Particles with ``feature_id`` already assigned.
        parametric_surface : str or numpy.ndarray or Motl
            Surface parameter table; see :meth:`load_parametric_surface`.
        surface_type : str, default='ellipsoid'
            Quadric surface type forwarded to :class:`quadric.QuadricsM`.
        feature_id : str, default='object_id'
            Column used to match particles to surface objects.
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
        features = in_motl.get_unique_values(feature_id=feature_id)
        assigned_motl_df = pd.DataFrame()
        m_qs = quadric.QuadricsM(parametric_surface, quadric=surface_type, feature_id=feature_id)

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id, reset_index=True)
            coord = fm.get_coordinates()
            fm.df[store_id] = m_qs.distance_point_surface(fm.df["tomo_id"].values[0], f, coord)
            assigned_motl_df = pd.concat([assigned_motl_df, fm.df])

        assigned_motl = cryomotl.Motl(assigned_motl_df)
        assigned_motl.df.reset_index(drop=True, inplace=True)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    @staticmethod
    def assign_affiliation_distance_based(
        input_motl,
        parametric_surface,
        surface_type="ellipsoid",
        feature_id="object_id",
        output_path=None,
        unassigned_value=None,
    ):
        """Assign each particle to the nearest surface centre.

        For each particle the Euclidean distance to every surface centre
        (``cx/cy/cz``) in the same tomogram is computed; the particle is
        labelled with the identifier of the closest centre.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to assign.
        parametric_surface : str or numpy.ndarray or Motl
            Surface parameter table; see :meth:`load_parametric_surface`.
        surface_type : str, default='ellipsoid'
            Quadric surface type (not used in the distance calculation itself,
            stored for downstream compatibility).
        feature_id : str, default='object_id'
            Column that receives the assigned surface identifier.
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

        if unassigned_value:
            assigned_motl = cryomotl.Motl(in_motl.df)
            in_motl.df = in_motl.df[in_motl.df[feature_id] == unassigned_value]
            in_motl.df.reset_index(drop=True, inplace=True)

        el_params = PleomorphicSurface.load_parametric_surface(
            parametric_surface=parametric_surface, feature_id=feature_id
        )
        m_qs = quadric.QuadricsM(parametric_surface, quadric=surface_type, feature_id=feature_id)
        tomos = in_motl.get_unique_values(feature_id="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(feature_values=t, feature_id="tomo_id", reset_index=True)
            coord = tm.get_coordinates()
            fm_ep = el_params.loc[el_params["tomo_id"] == t, [feature_id, "cx", "cy", "cz"]].values

            num_points = coord.shape[0]
            closest_ids = np.full(num_points, -1)
            closest_distances = np.full(num_points, np.inf)

            for i in range(num_points):
                for e in range(fm_ep.shape[0]):
                    distance = np.linalg.norm(coord[i, :] - fm_ep[e, 1:])
                    if distance < closest_distances[i]:
                        closest_distances[i] = distance
                        closest_ids[i] = fm_ep[e, 0]

            tm.df[feature_id] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        if unassigned_value:
            assigned_motl.df.loc[assigned_motl.df[feature_id] == unassigned_value, :] = assigned_motl_df.values
        else:
            assigned_motl = cryomotl.Motl(assigned_motl_df)

        assigned_motl.df.reset_index(drop=True, inplace=True)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    @staticmethod
    def assign_affiliation_intersection_based(
        input_motl,
        parametric_surface,
        feature_id="object_id",
        output_path=None,
        keep_unassigned=True,
    ):
        """Assign each particle to the surface it points toward (ray casting).

        For each particle a ray is cast along the negated particle normal
        (``-euler_angles_to_normals``).  The particle is labelled with the
        identifier of the surface whose intersection is closest along that ray.
        Particles that lie inside a surface or have no valid intersection
        receive ``-1``.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to assign.  Euler angles are used to derive normals.
        parametric_surface : str or numpy.ndarray or Motl
            Surface parameter table; see :meth:`load_parametric_surface`.
        feature_id : str, default='object_id'
            Column that receives the assigned surface identifier.
        output_path : str, optional
            Path to save the result.
        keep_unassigned : bool, default=True
            When ``False``, particles with ``feature_id == -1`` are removed.

        Returns
        -------
        Motl
            Motl with ``feature_id`` updated.  Prints the count of
            unassigned/inside particles.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        el_params = PleomorphicSurface.load_parametric_surface(
            parametric_surface=parametric_surface, feature_id=feature_id
        )
        tomos = in_motl.get_unique_values(feature_id="tomo_id")
        assigned_motl_df = pd.DataFrame()

        for t in tomos:
            tm = in_motl.get_motl_subset(feature_values=t, feature_id="tomo_id")
            coord = tm.get_coordinates()
            normal_vectors = -geom.euler_angles_to_normals(tm.get_angles())
            fm_ep = el_params.loc[
                el_params["tomo_id"] == t, [feature_id, "p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
            ].values

            num_points = coord.shape[0]
            closest_ids = np.full(num_points, -1)
            closest_distances = np.full(num_points, np.inf)

            for i in range(num_points):
                for e in range(fm_ep.shape[0]):
                    _, _, d1, d2, is_inside = geom.ray_ellipsoid_intersection_3d(
                        coord[i, :], normal_vectors[i, :], fm_ep[e, 1:]
                    )
                    if is_inside:
                        closest_distances[i] = np.inf
                        closest_ids[i] = -1
                        continue
                    distances_pos = [p for p in [d1, d2] if not np.isnan(p) and p > 0]
                    for d in distances_pos:
                        if abs(d) < abs(closest_distances[i]):
                            closest_distances[i] = d
                            closest_ids[i] = fm_ep[e, 0]

            tm.df[feature_id] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        unassigned = assigned_motl_df[assigned_motl_df[feature_id] == -1].shape[0]

        if not keep_unassigned:
            assigned_motl_df = assigned_motl_df[assigned_motl_df[feature_id] != -1]

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        print(f"{unassigned} particles did not have any intersection or were inside.")
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    @staticmethod
    def compute_intersection(input_motl, parametric_surface, feature_id="object_id"):
        """Compute ray–ellipsoid intersection distances for each particle.

        For every particle a ray is cast along ``-euler_angles_to_normals``
        and the two intersection distances with the assigned ellipsoid are
        returned.

        Parameters
        ----------
        input_motl : str or Motl
            Particles grouped by ``feature_id``.
        parametric_surface : str or numpy.ndarray or Motl
            Surface parameter table; see :meth:`load_parametric_surface`.
        feature_id : str, default='object_id'
            Column used to match particles to surfaces.

        Returns
        -------
        pandas.DataFrame
            Columns: ``subtomo_id``, ``feature_id``, ``d1``, ``d2``
            (the two signed intersection distances along the ray).
        """
        in_motl = cryomotl.Motl.load(input_motl)
        el_params = PleomorphicSurface.load_parametric_surface(
            parametric_surface=parametric_surface, feature_id=feature_id
        )
        features = in_motl.get_unique_values(feature_id=feature_id)
        intersection_points = pd.DataFrame(columns=["subtomo_id", "feature_id", "d1", "d2"])

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id, reset_index=True)
            coord = fm.get_coordinates()
            normal_vectors = -geom.euler_angles_to_normals(fm.get_angles())
            fm_ep = el_params.loc[
                el_params[feature_id] == f, ["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]
            ].values[0]
            for i in np.arange(0, coord.shape[0]):
                _, _, d1, d2, _ = geom.ray_ellipsoid_intersection_3d(coord[i, :], normal_vectors[i, :], fm_ep)
                new_row = pd.Series({"subtomo_id": fm.df.iloc[i]["subtomo_id"], "feature_id": f, "d1": d1, "d2": d2})
                intersection_points = pd.concat([intersection_points, new_row.to_frame().T], ignore_index=True)

        return intersection_points

    @staticmethod
    def compute_normals(input_motl, surface_params, feature_id="object_id", store_id="geom4", output_path=None):
        """Compute the angle between each particle's orientation and the surface normal.

        The surface normal at a particle's position is approximated as the
        vector from the fitted ellipsoid centre to the particle.  The stored
        value is the angle (in degrees) between that radial vector and the
        particle's orientation normal.

        Parameters
        ----------
        input_motl : str or Motl
            Particles grouped by ``feature_id``.
        surface_params : str or numpy.ndarray or Motl
            Surface parameter table; see :meth:`get_parametric_description`.
        feature_id : str, default='object_id'
            Column used to match particles to surfaces.
        store_id : str, default='geom4'
            Column that receives the angle values (degrees).
        output_path : str, optional
            Path to save the result.

        Returns
        -------
        Motl
            Input motl with ``store_id`` populated.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(feature_id=feature_id)
        el_params = PleomorphicSurface.get_parametric_description(surface_params, feature_id=feature_id)
        assigned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id, reset_index=True)
            coord = fm.get_coordinates()
            normals = geom.euler_angles_to_normals(fm.get_angles())
            center = el_params.loc[el_params[feature_id] == f, ["cx", "cy", "cz"]].values
            normals_t = coord - np.tile(center, (coord.shape[0], 1))
            fm.df[store_id] = geom.angle_between_n_vectors(normals, normals_t)
            assigned_motl_df = pd.concat([assigned_motl_df, fm.df])

        assigned_motl_df.index = in_motl.df.index
        assigned_motl = cryomotl.Motl(assigned_motl_df)
        if output_path is not None:
            assigned_motl.write_out(output_path)
        return assigned_motl

    @staticmethod
    def clean_by_normals(
        input_motl,
        feature_id="object_id",
        compute_normals=True,
        surface_params=None,
        normals_id="geom4",
        threshold=None,
        output_path=None,
    ):
        """Remove particles whose orientation deviates too far from the surface normal.

        Particles are removed when their normal-angle value (column
        *normals_id*) exceeds *threshold*.  When *threshold* is ``None``,
        one standard deviation of the angle distribution is used.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to clean.
        feature_id : str, default='object_id'
            Column used to group particles (forwarded to
            :meth:`compute_normals`).
        compute_normals : bool, default=True
            Recompute normal angles before filtering.  Set to ``False`` when
            *normals_id* is already populated.
        surface_params : str or numpy.ndarray or Motl, optional
            Surface parameter table, required when *compute_normals* is
            ``True``.
        normals_id : str, default='geom4'
            Column holding the angle-to-normal values.
        threshold : float, optional
            Maximum allowed angle (degrees).  Defaults to one standard
            deviation of the distribution.
        output_path : str, optional
            Path to save the cleaned motl.

        Returns
        -------
        Motl
            Cleaned motl.  Prints the number and percentage of removed
            particles.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        orig_number = in_motl.df.shape[0]

        if compute_normals:
            in_motl = PleomorphicSurface.compute_normals(
                in_motl, surface_params, feature_id=feature_id, store_id=normals_id, output_path=None
            )

        diff_angles = in_motl.df[normals_id].values
        to_remove = (
            np.where(np.abs(diff_angles) > np.std(diff_angles))
            if not threshold
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

    @staticmethod
    def clean_by_radius(input_motl, feature_id="object_id", threshold=None, output_path=None):
        """Remove particles that lie too far from the mean ellipsoid radius.

        For each surface group the mean radius ``(rx+ry+rz)/3`` and the
        standard deviation of distances from the centre are computed.
        Particles with distances outside ``[radius - thr, radius + thr]`` are
        removed.  When *threshold* is ``None``, one standard deviation of the
        distance distribution is used.

        Parameters
        ----------
        input_motl : str or Motl
            Particles to clean, grouped by ``feature_id``.
        feature_id : str, default='object_id'
            Column used to group particles.
        threshold : float, optional
            Half-width of the allowed distance band.  Defaults to one
            standard deviation.
        output_path : str, optional
            Path to save the cleaned motl.

        Returns
        -------
        Motl
            Cleaned motl.  Prints the number and percentage of removed
            particles.
        """
        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(feature_id=feature_id)
        el_params = PleomorphicSurface.get_parametric_description(in_motl, feature_id=feature_id)
        cleaned_motl_df = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id, reset_index=True)
            coord = fm.get_coordinates()
            center = el_params.loc[el_params[feature_id] == f, ["cx", "cy", "cz"]].values
            radius = np.mean(el_params.loc[el_params[feature_id] == f, ["rx", "ry", "rz"]].values)
            distances = np.linalg.norm(coord - center, axis=1)
            to_remove = (
                np.where((distances < radius - np.std(distances)) | (distances > radius + np.std(distances)))
                if not threshold
                else np.where((distances < radius - threshold) | (distances > radius + threshold))
            )
            fm.df = fm.df.drop(index=to_remove[0])
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
    def create_spherical_oversampling(
        input_motl,
        motl_radius_id,
        sampling_distance,
        sampling_angle=360,
        output_path=None,
    ):
        """Generate oversampled particles on a sphere around each input particle.

        For every input particle, sample points are placed on a cone of
        half-opening angle *sampling_angle* at the given *sampling_distance*
        from the particle centre.  Each sample point receives Euler angles
        computed from its radial normal.

        Parameters
        ----------
        input_motl : str or Motl
            Template particle list.  Each particle's position and the radius
            stored in *motl_radius_id* define the sampling sphere.
        motl_radius_id : str
            Column that holds the sphere radius for each particle.
        sampling_distance : float
            Angular sampling step on the sphere (degrees or fraction,
            forwarded to :func:`geom.sample_cone`).
        sampling_angle : float, default=360
            Half-opening angle of the sampling cone (degrees).  ``360``
            samples the full sphere.
        output_path : str, optional
            Path to save the new oversampled motl.

        Returns
        -------
        Motl
            New motl with oversampled positions.  ``object_id`` is inherited
            from the source particle; ``class`` is set to 1.
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


class MAK:
    """Static helpers for MAK (multi-subunit complex) particle-list analysis."""

    def get_centre_from_mean_subunit_location(subunit_motl, output_path):
        """Compute the complex centre as the mean of its subunit positions.

        For each unique ``object_id`` the mean x, y, z position (coordinates
        + shifts) is written into the particle with ``geom2 == 1``.

        Parameters
        ----------
        subunit_motl : Motl
            Subunit particle list.  ``object_id`` identifies the complex;
            ``geom2`` identifies the subunit type.
        output_path : str or None
            Path to save the result.  No file is written when ``None``.

        Returns
        -------
        Motl
            Motl containing only the ``geom2 == 1`` particles, with
            coordinates replaced by the per-complex mean position.
        """
        subunit_motl.update_coordinates()
        unique_obj_id = cryomotl.Motl.get_unique_values(subunit_motl, "object_id")
        id_list = [i for i in unique_obj_id]
        motl_object_id = cryomotl.Motl.get_motl_subset(subunit_motl, 1, "geom2", False, False)
        for i in id_list:
            motl_object_i = cryomotl.Motl.get_motl_subset(subunit_motl, i, "object_id", False, False)
            coordx = motl_object_i.df.loc[:, ["x"]].values + motl_object_i.df.loc[:, ["shift_x"]].values
            ctr_coordx = np.mean(coordx, 0)
            coordy = motl_object_i.df.loc[:, ["y"]].values + motl_object_i.df.loc[:, ["shift_y"]].values
            ctr_coordy = np.mean(coordy, 0)
            coordz = motl_object_i.df.loc[:, ["z"]].values + motl_object_i.df.loc[:, ["shift_z"]].values
            ctr_coordz = np.mean(coordz, 0)

            motl_object_id.df.loc[lambda df: df["object_id"] == i, ["x"]] = ctr_coordx
            motl_object_id.df.loc[lambda df: df["object_id"] == i, ["y"]] = ctr_coordy
            motl_object_id.df.loc[lambda df: df["object_id"] == i, ["z"]] = ctr_coordz
        motl_object_i.update_coordinates()
        if output_path is not None:
            cryomotl.Motl.write_out(motl_object_id, output_path)
        return motl_object_id
