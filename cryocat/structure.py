import numpy as np
import pandas as pd
import warnings
import decimal
import os
from cryocat import cryomotl
from cryocat import cryomap
from cryocat import cryomask
from cryocat import geom
from cryocat import mathutils
from cryocat import ribana
from cryocat import nnana
from cryocat import ioutils
from cryocat import quadric
from scipy.spatial.transform import Rotation as srot


class NPC:

    @staticmethod
    def compute_diameter(input_motl, pixel_size=1.0, su_id="geom2"):
        """Compute the average diameter of specific subunit pairs within NPCs.

        Parameters
        ----------
        input_motl : str
            Path to the input motl file containing molecular data.
        pixel_size : float, optional
            The size of each pixel in the data, used to scale the computed distances. Default is 1.0.
        su_id : str, optional
            The identifier used to select subunits within the data. Default is 'geom2'.

        Returns
        -------
        numpy.ndarray
            An array of average diameters for each unique (tomo_id, object_id) group in the data.

        Notes
        -----
        The function loads molecular data from a specified file, identifies specific subunit pairs,
        computes the pairwise distances between these pairs, and then calculates the average diameter
        for each group identified by 'tomo_id' and 'object_id'. Distances are scaled by the pixel size.
        """

        motl_ri = cryomotl.Motl.load(input_motl)
        motl_ri.df.reset_index(inplace=True, drop=True)

        # Define the pairs
        pairs = [(1, 5), (2, 6), (3, 7), (4, 8)]

        # Function to check if both elements of a pair are present in a group and return the indices
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

        # Group by t_id and o_id, and apply the function to get pairs indices
        result = (
            motl_ri.df.groupby(["tomo_id", "object_id"]).apply(get_pairs_indices).reset_index(name="su_pairs_indices")
        )
        # Filter out empty lists
        pairs_only = result.loc[result["su_pairs_indices"].apply(bool), "su_pairs_indices"].to_list()
        motl_idx = np.array([item for sublist in pairs_only for item in sublist])

        coord = motl_ri.get_coordinates()

        distances = geom.point_pairwise_dist(coord[motl_idx[:, 0], :], coord[motl_idx[:, 1], :])

        # Create an array of NaNs with the same length as the dataframe
        df_values = np.full(len(motl_ri.df), np.nan)
        df_values[motl_idx[:, 0]] = distances * pixel_size
        motl_ri.df["su_distance"] = df_values

        npc_diameters_df = motl_ri.df.groupby(["tomo_id", "object_id"])["su_distance"].mean().dropna()

        return npc_diameters_df.values

    @staticmethod
    def unify_nn_orientations(input_motl, dist_threshold=10000):
        traced_motl = ribana.trace_chains(
            input_motl.df,
            input_motl.df,
            dist_threshold,
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

    def cluster_subunits_to_rings(
        input_motl_path,
        mask_size,
        entry_mask_coord,
        exit_mask_coord,
        npc_radius,
        max_trace_distance,
        min_trace_distance=0,
    ):
        working_dir, _ = os.path.split(input_motl_path)
        entry_mask = working_dir + "entry_mask.em"
        exit_mask = working_dir + "exit_mask.em"
        _ = cryomask.spherical_mask(mask_size, 3, center=entry_mask_coord, output_name=entry_mask)
        _ = cryomask.spherical_mask(mask_size, 3, center=exit_mask_coord, output_name=exit_mask)

        motl = cryomotl.Motl.load(input_motl_path)
        motl.renumber_particles()

        motl_entry = cryomotl.Motl.recenter_to_subparticle(motl, entry_mask)
        motl_exit = cryomotl.Motl.recenter_to_subparticle(motl, exit_mask)

        # tracing
        traced_motl = ribana.trace_chains(
            motl_entry.df,
            motl_exit.df,
            max_distance=max_trace_distance,
            min_distance=min_trace_distance,
        )
        traced_motl.df.sort_values(["tomo_id", "object_id", "geom2"], inplace=True)
        ribana.add_occupancy(traced_motl)
        ribana.add_traced_info(traced_motl, motl)

        new_traced_motl = NPC.merge_subunits(motl, npc_radius=npc_radius)

        os.remove(entry_mask)
        os.remove(exit_mask)

        return new_traced_motl

    @staticmethod
    def get_center_with_radius(object_motl, radius):
        vector_x = np.asarray([-radius, 0, 0])
        shifted_motl = cryomotl.Motl(object_motl.df)
        shifted_motl.shift_positions(vector_x)

        center_coordinates = shifted_motl.get_coordinates()
        centroid = np.mean(center_coordinates, axis=0)

        return centroid

    @staticmethod
    def get_center_and_radius(object_motl, include_singles=False):
        vector_x = np.asarray([-1, 0, 0])

        if object_motl.df.shape[0] <= 1:
            if include_singles:
                return object_motl.get_coordinates(), 0
            else:
                # warnings.warn(
                #    f"The object with number {object_motl.df['object_id'].values[0]} from tomogram {object_motl.df['tomo_id'].values[0]} has only one subunit. Center could not be determined!"
                # )
                return np.zeros((3,)), 0

        start_coord = object_motl.get_coordinates()
        if object_motl.df.shape[0] <= 3:
            rot = object_motl.get_rotations()
            rot_vec = rot.apply(vector_x)
            end_coord = start_coord + rot_vec

            circle_center, _ = geom.ray_ray_intersection_3d(starting_points=start_coord, ending_points=end_coord)
            circle_radius = 0.0  # TODO compute properly
        else:
            circle_center, circle_radius, confidence = geom.fit_circle_3d_pratt(start_coord)
            # circle_center, circle_radius, confidence = geom.fit_circle_3d_taubin(start_coord)

        return circle_center, circle_radius

    @staticmethod
    def get_centers_as_motl(tomo_motl, tomo_id, radius):
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
        """Merge subunits within a given radius in input_motl.

        Parameters
        ----------
        input_motl : str or pd.DataFrame
            If a string is provided, it is assumed to be a path to the motl file.
            If a DataFrame is provided, it is used directly.
        npc_radius : int, optional
            The radius within which to search for subunits to merge. Default is 55 pixels.

        Returns
        -------
        input_motl : pd.DataFrame
            The input motl data with merged subunits. The dataframe is modified in-place.

        Notes
        -----
        The function calculates the centers of each objects in the motl and merges objects with a center that's closer than the npc_radius.
        All subunits were renumbered and assign back to the input_motl. The number of subunits per object was kept under geom1 in updated motl.
        """

        if isinstance(input_motl, (str, pd.DataFrame)):
            input_motl = cryomotl.Motl.load(input_motl)

        for t in input_motl.get_unique_values("tomo_id"):
            tm = input_motl.get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)

            # get centers of motls - the motl has to have object_id assigned
            new_object_motl = NPC.get_centers_as_motl(tm, t, radius=npc_radius)

            changed_objects = []
            if new_object_motl.df.shape[0] > 1:
                # get NN stats for centers
                center_stats = ribana.get_nn_stats(new_object_motl, new_object_motl)

                if any(center_stats["distance"] <= npc_radius):
                    # get all centers within npc_radius distance for merging
                    center_idx, nn_idx = nnana.get_nn_within_distance(new_object_motl, npc_radius)
                    for i, o in enumerate(center_idx):
                        # get object_id of the first object
                        o_id1 = new_object_motl.df.loc[new_object_motl.df.index[o], "object_id"]
                        # add it to the list of changed objects
                        changed_objects.append(o_id1)
                        for j in nn_idx[i]:
                            # change object_id of the other object to the one of the first object
                            o_id2 = new_object_motl.df.loc[new_object_motl.df.index[j], "object_id"]
                            tm.df.loc[tm.df["object_id"] == o_id2, "object_id"] = o_id1

            tm.df["geom1"] = tm.df.groupby(["object_id"])["object_id"].transform("count")

            # for all objects that changed renumber subunits
            for o in changed_objects:
                om = tm.get_motl_subset(feature_values=o, feature_id="object_id", reset_index=True)
                s_idx = NPC.get_new_subunit_idx(om, npc_radius)
                tm.df.loc[tm.df["object_id"] == o, "geom2"] = s_idx

            # squeeze the object_idx to be in sequence
            tm.df["object_id"] = tm.df["object_id"].rank(method="dense").astype(int)

            # assign the results back to the original motl
            input_motl.df.loc[input_motl.df["tomo_id"] == t, ["object_id", "geom1", "geom2"]] = tm.df[
                ["object_id", "geom1", "geom2"]
            ].values

        input_motl.df.reset_index(inplace=True, drop=True)
        input_motl.df["geom1"] = input_motl.df.groupby(["tomo_id", "object_id"])["object_id"].transform("count")
        input_motl.df["object_id"] = input_motl.df["object_id"].rank(method="dense").astype(int)

        return input_motl

    @staticmethod
    def merge_rings(input_motls, npc_radius, distance_threshold=40):
        # def find_closest_ring():

        if not isinstance(input_motls, list) or len(input_motls) <= 1:
            raise UserWarning(
                f"The input has to be list of valid motl specifications and has to contain more than one element!"
            )

        ring_motls = []
        for m in input_motls:
            if isinstance(m, (str, pd.DataFrame)):
                ring_motls.append(cryomotl.Motl.load(m))
            else:
                ring_motls.append(m)

        # renumber the objects sequentially, each next motl starting from the last max object_id + 1
        starting_number = 1
        for r in ring_motls:
            r.renumber_objects_sequentially(starting_number=starting_number)
            starting_number = r.df["object_id"].max() + 1

        ring_pairs = mathutils.get_all_pairs(np.arange(len(ring_motls)))

        for i in ring_pairs:
            for t in ring_motls[i[0]].get_unique_values("tomo_id"):
                tm1 = ring_motls[i[0]].get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
                tm2 = ring_motls[i[1]].get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
                # print(t)
                if tm2.df.shape[0] > 0:
                    centers1 = NPC.get_centers_as_motl(tm1, t, radius=npc_radius)
                    centers2 = NPC.get_centers_as_motl(tm2, t, radius=npc_radius)
                    distances, obj1_idx = ribana.get_feature_nn(centers1, centers2)
                    obj1_idx = obj1_idx.reshape((obj1_idx.shape[0],))
                    # centers1_nn = centers1.df.iloc[obj1_idx.reshape((obj1_idx.shape[0],))]

                    close_idx = distances < distance_threshold
                    close_idx = close_idx.reshape((close_idx.shape[0],))
                    if np.all(~close_idx):
                        continue
                    obj1_idx = obj1_idx[close_idx]
                    obj2_idx = np.arange(centers2.df.shape[0])
                    obj2_idx = obj2_idx[close_idx]
                    for o1, o2 in zip(obj1_idx, obj2_idx):
                        obj1_id = centers1.df.loc[centers1.df.index[o1], "object_id"]
                        obj2_id = centers2.df.loc[centers2.df.index[o2], "object_id"]
                        ring_motls[i[1]].df.loc[
                            (ring_motls[i[1]].df["tomo_id"] == t) & (ring_motls[i[1]].df["object_id"] == obj2_id),
                            "object_id",
                        ] = obj1_id

        return ring_motls


class PleomorphicSurface:

    @staticmethod
    def get_parametric_description(input_motl, feature_id="object_id", output_file=None):

        in_motl = cryomotl.Motl.load(input_motl)
        features = in_motl.get_unique_values(feature_id=feature_id)
        el_params_all = pd.DataFrame()

        for f in features:
            fm = in_motl.get_motl_subset(feature_values=f, feature_id=feature_id)
            coord = fm.get_coordinates()
            el_params = PleomorphicSurface.load_parametric_surface(feature_id=feature_id)
            center, radii, evecs, v = geom.fit_ellipsoid(coord)
            el_params["tomo_id"] = [fm.df.iloc[0]["tomo_id"]]  # assuming that each object has unified tomo_id
            el_params[feature_id] = [f]
            el_params[["cx", "cy", "cz"]] = [center]
            el_params[["rx", "ry", "rz"]] = [radii]
            el_params[["ev1x", "ev1y", "ev1z"]] = [evecs[0, :]]
            el_params[["ev2x", "ev2y", "ev2z"]] = [evecs[1, :]]
            el_params[["ev3x", "ev3y", "ev3z"]] = [evecs[2, :]]
            el_params[["p1", "p2", "p3", "p4", "p5", "p6", "p7", "p8", "p9", "p10"]] = [v]
            el_params_all = pd.concat([el_params_all, el_params])

        el_params_all.reset_index(drop=True, inplace=True)

        if output_file is not None:
            el_params_all.to_csv(output_file, index=False)  # `index=False` excludes the row index

        return el_params_all

    @staticmethod
    def load_parametric_surface(parametric_surface=None, feature_id="object_id"):

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
        output_file=None,
        radius_offset=0.0,
        motl_radius_id="geom5",
    ):

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

                # Get the indices of the filtered coordinates
                idx_to_keep = np.where(mask_values == 1)[0]
                tm.df[feature_id] = o
                assigned_motl_df = pd.concat([assigned_motl_df, tm.df.iloc[idx_to_keep]])

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)

        if output_file is not None:
            assigned_motl.write_out(output_file)

        return assigned_motl

    @staticmethod
    def compute_point_surface_distance(
        input_motl,
        parametric_surface,
        surface_type="ellipsoid",
        feature_id="object_id",
        output_file=None,
        store_id="geom4",
    ):

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

        if output_file is not None:
            assigned_motl.write_out(output_file)

        return assigned_motl

    @staticmethod
    def assign_affiliation_distance_based(
        input_motl,
        parametric_surface,
        surface_type="ellipsoid",
        feature_id="object_id",
        output_file=None,
        unassigned_value=None,
    ):

        in_motl = cryomotl.Motl.load(input_motl)

        # if assignment was already done and should be considered only the unassigned values are checked
        if unassigned_value:
            assigned_motl = cryomotl.Motl(in_motl.df)  # copy the motl
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
            closest_ids = np.full(num_points, -1)  # -1 indicates no intersection found
            closest_distances = np.full(num_points, np.inf)  # Start with infinity for closest distance

            for i in range(num_points):
                for e in range(fm_ep.shape[0]):
                    distance = np.linalg.norm(coord[i, :] - fm_ep[e, 1:])
                    if distance < closest_distances[i]:  # Closest absolute distance
                        closest_distances[i] = distance
                        closest_ids[i] = fm_ep[e, 0]  # Store ellipsoid ID

            tm.df[feature_id] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        if unassigned_value:
            assigned_motl.df.loc[assigned_motl.df[feature_id] == unassigned_value, :] = assigned_motl_df.values
        else:
            assigned_motl = cryomotl.Motl(assigned_motl_df)

        assigned_motl.df.reset_index(drop=True, inplace=True)

        if output_file is not None:
            assigned_motl.write_out(output_file)

        return assigned_motl

    @staticmethod
    def assign_affiliation_intersection_based(
        input_motl, parametric_surface, feature_id="object_id", output_file=None, keep_unassigned=True
    ):

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
            closest_ids = np.full(num_points, -1)  # -1 indicates no intersection found
            closest_distances = np.full(num_points, np.inf)  # Start with infinity for closest distance

            for i in range(num_points):
                for e in range(fm_ep.shape[0]):
                    _, _, d1, d2, is_inside = geom.ray_ellipsoid_intersection_3d(
                        coord[i, :], normal_vectors[i, :], fm_ep[e, 1:]
                    )

                    if is_inside:
                        closest_distances[i] = np.inf
                        closest_ids[i] = -1
                        continue

                    # Consider both intersection points (p1, p2) if they are not NaN
                    distances = [p for p in [d1, d2] if not np.isnan(p) and p > 0]

                    for d in distances:
                        # Check if this distance is closer than the current closest
                        if abs(d) < abs(closest_distances[i]):  # Closest absolute distance
                            closest_distances[i] = d
                            closest_ids[i] = fm_ep[e, 0]  # Store ellipsoid ID

            tm.df[feature_id] = closest_ids
            assigned_motl_df = pd.concat([assigned_motl_df, tm.df])

        unassigned = assigned_motl_df[assigned_motl_df[feature_id] == -1].shape[0]

        if not keep_unassigned:
            assigned_motl_df = assigned_motl_df[assigned_motl_df[feature_id] != -1]

        assigned_motl_df.reset_index(drop=True, inplace=True)
        assigned_motl = cryomotl.Motl(assigned_motl_df)

        print(f"{unassigned} particles did not have any intersection or were inside.")

        if output_file is not None:
            assigned_motl.write_out(output_file)

        return assigned_motl

    @staticmethod
    def compute_intersection(input_motl, parametric_surface, feature_id="object_id"):

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
    def compute_normals(input_motl, surface_params, feature_id="object_id", store_id="geom4", output_file=None):
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

        assigned_motl = cryomotl.Motl(assigned_motl_df)

        if output_file is not None:
            assigned_motl.write_out(output_file)

        return assigned_motl

    @staticmethod
    def clean_by_normals(
        input_motl,
        feature_id="object_id",
        compute_normals=True,
        surface_params=None,
        normals_id="geom4",
        threshold=None,
        output_file=None,
    ):

        in_motl = cryomotl.Motl.load(input_motl)
        orig_number = in_motl.df.shape[0]

        if compute_normals:
            in_motl = PleomorphicSurface.compute_normals(
                in_motl, surface_params, feature_id=feature_id, store_id=normals_id, output_file=None
            )

        diff_angles = in_motl.df[normals_id].values

        if not threshold:
            to_remove = np.where(np.abs(diff_angles) > np.std(diff_angles))
        else:
            to_remove = np.where(np.abs(diff_angles) > threshold)

        in_motl.df = in_motl.df.drop(index=to_remove[0])

        in_motl.df.reset_index(drop=True, inplace=True)

        print(
            f"{orig_number-in_motl.df.shape[0]} particles "
            f"({((orig_number-in_motl.df.shape[0])/orig_number*100):.2f}%) were removed from the list."
        )

        if output_file is not None:
            in_motl.write_out(output_file)

        return in_motl

    @staticmethod
    def clean_by_radius(input_motl, feature_id="object_id", threshold=None, output_file=None):

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
            if not threshold:
                to_remove = np.where(
                    (distances < radius - np.std(distances)) | (distances > radius + np.std(distances))
                )  # maybe diff from radius?
            else:
                to_remove = np.where((distances < radius - threshold) | (distances > radius + threshold))

            fm.df = fm.df.drop(index=to_remove[0])
            cleaned_motl_df = pd.concat([cleaned_motl_df, fm.df])

        cleaned_motl_df.reset_index(drop=True, inplace=True)

        cleaned_motl = cryomotl.Motl(cleaned_motl_df)

        print(
            f"{in_motl.df.shape[0]-cleaned_motl.df.shape[0]} particles "
            f"({((in_motl.df.shape[0]-cleaned_motl.df.shape[0])/in_motl.df.shape[0]*100):.2f}%) were removed from the list."
        )

        if output_file is not None:
            cleaned_motl.write_out(output_file)

        return cleaned_motl

    @staticmethod
    def create_spherical_oversampling(
        input_motl,
        motl_radius_id,
        sampling_distance,
        sampling_angle=360,
        output_path=None,
    ):
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
    def get_centre_from_mean_subunit_location(subunit_motl, output_path):
        ## does not distinguish between full and partial complexes, just takes geometric centre
        subunit_motl.update_coordinates()
        unique_obj_id = cryomotl.Motl.get_unique_values(subunit_motl, "object_id")
        id_list = [i for i in unique_obj_id]
        motl_object_id = cryomotl.Motl.get_motl_subset(subunit_motl, 1, "geom2", False, False)
        # display(motl_object_id.df)
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
