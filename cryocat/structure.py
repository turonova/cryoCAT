import numpy as np
import pandas as pd
import warnings
import decimal
from cryocat import cryomotl
from cryocat import geom
from cryocat import mathutils
from cryocat import ribana
from cryocat import nnana
from scipy.spatial.transform import Rotation as srot


class NPC:
    @staticmethod
    def unify_nn_orientations(input_motl, recenter_mask, dist_threshold=10):
        if isinstance(input_motl, (str, pd.DataFrame)):
            original_motl = cryomotl.Motl.load(input_motl)
        else:
            original_motl = input_motl

        for t in original_motl.get_unique_values("tomo_id"):
            tm = original_motl.get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)

            number_of_rotations = None
            rot = srot.from_euler("zxz", angles=[0, 180, 0], degrees=True)

            while number_of_rotations is None or number_of_rotations > 0:

                def get_updated_stats(motl_name):
                    new_m = cryomotl.Motl.recenter_to_subparticle(motl_name, recenter_mask)
                    stats = ribana.get_nn_stats(
                        new_m, new_m, pixel_size=1.0, feature_id="tomo_id", angular_dist_type="cone"
                    )

                    if stats["distance"].min() > dist_threshold:
                        return 0, np.array([]), np.zeros((3,))

                    bin_threshold = mathutils.otsu_threshold(stats["distance"].values)
                    flip_candidates = stats.loc[
                        stats["distance"] <= bin_threshold, ["subtomo_idx", "subtomo_nn_idx"]
                    ].values

                    angle_stats = np.asarray(
                        [
                            stats["angular_distance"].mean(),
                            stats["angular_distance"].var(),
                            stats["angular_distance"].std(),
                        ]
                    )
                    return flip_candidates.shape[0], flip_candidates, angle_stats

                flip_n, flip_id, _ = get_updated_stats(tm.df)

                for i in np.arange(flip_id.shape[0]):

                    def rotate_single_subtomo(id_j):
                        m = cryomotl.Motl.load(tm.df)
                        m_sub = m.get_motl_subset(flip_id[i, id_j], feature_id="subtomo_id")
                        m_sub.apply_rotation(rot)
                        m.df.loc[m.df["subtomo_id"] == flip_id[i, id_j], ["phi", "theta", "psi"]] = m_sub.df[
                            ["phi", "theta", "psi"]
                        ].values
                        return m

                    m1 = rotate_single_subtomo(0)
                    m2 = rotate_single_subtomo(1)
                    _, _, new_stats1 = get_updated_stats(m1.df)
                    _, _, new_stats2 = get_updated_stats(m2.df)

                    eval_stats = new_stats1 < new_stats2
                    if sum(eval_stats) >= 2:
                        tm.df = m1.df.copy()
                    else:
                        tm.df = m2.df.copy()

                number_of_rotations = flip_n

            original_motl.df.loc[original_motl.df["tomo_id"] == t, :] = tm.df

        return original_motl

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
    def get_centers_as_motl(tomo_motl, tomo_id, include_singles=False):
        central_points = []
        object_idx = []
        for o in tomo_motl.get_unique_values("object_id"):
            om = tomo_motl.get_motl_subset(feature_values=[o], feature_id="object_id", reset_index=True)

            circle_center, _ = NPC.get_center_and_radius(om, include_singles)
            central_points.append(circle_center)

            object_idx.append(o)

        central_points = np.vstack(central_points)
        new_object_motl = cryomotl.Motl()
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
    def get_new_subunit_idx(object_motl, symmetry=8):
        npc_center, _ = NPC.get_center_and_radius(object_motl)
        su_coord = object_motl.get_coordinates()
        vectors = su_coord - np.tile(npc_center, (su_coord.shape[0], 1))

        div_angle = 360.0 / symmetry
        s_idx = [1]
        for vec in vectors[1:]:
            angle = geom.vector_angular_distance(vectors[0], vec) / div_angle
            s_idx.append(int(decimal.Decimal(angle).to_integral_value(rounding=decimal.ROUND_HALF_UP)) + 1)

        return s_idx

    @staticmethod
    def merge_npc(input_motl, npc_radius=60):
        if isinstance(input_motl, (str, pd.DataFrame)):
            input_motl = cryomotl.Motl.load(input_motl)

        for t in input_motl.get_unique_values("tomo_id"):
            tm = input_motl.get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)

            new_object_motl = NPC.get_centers_as_motl(tm, t)

            if new_object_motl.df.shape[0] > 1:
                center_stats = ribana.get_nn_stats(new_object_motl, new_object_motl)
                dist_threshold = mathutils.otsu_threshold(center_stats["distance"].values)

                changed_objects = []
                if dist_threshold <= npc_radius:
                    center_idx, nn_idx = nnana.get_nn_within_distance(new_object_motl, dist_threshold)
                    for i, o in enumerate(center_idx):
                        o_id1 = new_object_motl.df.loc[new_object_motl.df.index[o], "object_id"]
                        changed_objects.append(o_id1)
                        for j in nn_idx[i, :]:
                            o_id2 = new_object_motl.df.loc[new_object_motl.df.index[j], "object_id"]
                            tm.df.loc[tm.df["object_id"] == o_id2, "object_id"] = o_id1

            tm.df["geom1"] = tm.df.groupby(["object_id"])["object_id"].transform("count")
            single_subunits = tm.df.loc[tm.df["geom1"] == 1, "object_id"].values

            if single_subunits.shape[0] != 0:
                new_object_motl = NPC.get_centers_as_motl(tm, t, include_singles=True)
                if new_object_motl.df.shape[0] > 1:
                    center_stats = ribana.get_nn_stats(new_object_motl, new_object_motl, remove_duplicates=False)
                    for s in single_subunits:
                        s_id = new_object_motl.df.loc[new_object_motl.df["object_id"] == s, "subtomo_id"].values[0]
                        npc_id = center_stats.loc[center_stats["subtomo_idx"] == s_id, "subtomo_nn_idx"].values[0]
                        new_object_id = new_object_motl.df.loc[
                            new_object_motl.df["subtomo_id"] == npc_id, "object_id"
                        ].values[0]
                        tm.df.loc[tm.df["object_id"] == s, "object_id"] = new_object_id
                        changed_objects.append(new_object_id)

            for o in changed_objects:
                om = tm.get_motl_subset(feature_values=o, feature_id="object_id", reset_index=True)
                s_idx = NPC.get_new_subunit_idx(om)
                tm.df.loc[tm.df["object_id"] == o, "geom2"] = s_idx

            tm.df["object_id"] = tm.df["object_id"].rank(method="dense").astype(int)

            input_motl.df.loc[input_motl.df["tomo_id"] == t, ["object_id", "geom1", "geom2"]] = tm.df[
                ["object_id", "geom1", "geom2"]
            ].values

        input_motl.df.reset_index(inplace=True, drop=True)
        input_motl.df["geom1"] = input_motl.df.groupby(["tomo_id", "object_id"])["object_id"].transform("count")
        input_motl.df["object_id"] = input_motl.df["object_id"].rank(method="dense").astype(int)

        return input_motl
