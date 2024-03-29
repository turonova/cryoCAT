import numpy as np
import pandas as pd
import warnings
import decimal
import os
from cryocat import cryomotl
from cryocat import cryomask
from cryocat import geom
from cryocat import mathutils
from cryocat import ribana
from cryocat import nnana
from scipy.spatial.transform import Rotation as srot


class NPC:

    @staticmethod
    def compute_diameter(input_motl, pixel_size=1.0, su_id="geom2"):
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
            motl_entry.df, motl_exit.df, max_distance=max_trace_distance, min_distance=min_trace_distance
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
