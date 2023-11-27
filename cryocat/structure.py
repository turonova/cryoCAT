import numpy as np
import pandas as pd
import warnings
from cryocat import cryomotl
from cryocat import geom
from cryocat import mathutils
from cryocat import ribana
from scipy.spatial.transform import Rotation as srot


class NPC:
    @staticmethod
    def unify_nn_orientations(input_motl, recenter_mask, dist_threshold=10):
        number_of_rotations = None
        rot = srot.from_euler("zxz", angles=[0, 180, 0], degrees=True)

        original_motl = input_motl

        while number_of_rotations is None or number_of_rotations > 0:

            def get_updated_stats(motl_name):
                new_m = cryomotl.Motl.recenter_to_subparticle(motl_name, recenter_mask)
                stats = ribana.get_nn_stats(
                    new_m, new_m, pixel_size=1.0, feature_id="tomo_id", angular_dist_type="cone"
                )

                if stats["distance"].min() > dist_threshold:
                    return 0, np.array([]), np.zeros((3,))

                stats_bins = np.histogram(stats["distance"], bins=stats.shape[0])
                bin_id = mathutils.otsu_threshold(stats_bins[0])
                flip_candidates = stats.loc[
                    stats["distance"] <= stats_bins[1][bin_id], ["subtomo_idx", "subtomo_nn_idx"]
                ].values

                angle_stats = np.asarray(
                    [stats["angular_distance"].mean(), stats["angular_distance"].var(), stats["angular_distance"].std()]
                )
                return flip_candidates.shape[0], flip_candidates, angle_stats

            flip_n, flip_id, _ = get_updated_stats(original_motl.df)

            for i in np.arange(flip_id.shape[0]):

                def rotate_single_subtomo(id_j):
                    m = cryomotl.Motl.load(original_motl.df)
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
                    original_motl.df = m1.df.copy()
                else:
                    original_motl.df = m2.df.copy()

            number_of_rotations = flip_n

        return original_motl

    @staticmethod
    def get_center_and_radius(object_motl):
        vector_x = np.asarray([-1, 0, 0])

        if object_motl.df.shape[0] <= 1:
            warnings.warn(
                f"The object with number {object_motl.df['object_id'].values[0]} from tomogram {object_motl.df['tomo_id'].values[0]} has only one subunit. Center could not be determined!"
            )
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
    def merge_npc(input_motl, npc_radius=60):
        if isinstance(input_motl, (str, pd.DataFrame)):
            input_motl = cryomotl.Motl.load(input_motl)

        tomo_idx = []
        for t in input_motl.get_unique_values("tomo_id"):
            tm = input_motl.get_motl_subset(feature_values=[t], feature_id="tomo_id", reset_index=True)
            current_tomo_id = tm.df["tomo_id"].values[0]
            tomo_idx.append(current_tomo_id)

            central_points = []
            object_idx = []
            for o in input_motl.get_unique_values("object_id"):
                om = input_motl.get_motl_subset(feature_values=[o], feature_id="object_id", reset_index=True)

                circle_center, _ = NPC.get_center_and_radius(om)
                central_points.append(circle_center)

                object_idx.append(o)

            central_points = np.vstack(central_points)
            new_object_motl = cryomotl.Motl()
            new_object_motl.fill(
                {
                    "x": central_points[:, 0],
                    "y": central_points[:, 1],
                    "z": central_points[:, 2],
                    "tomo_id": current_tomo_id,
                    "object_id": object_idx,
                }
            )
            new_object_motl.renumber_particles()
            new_object_motl.df.fillna(0.0, inplace=True)

            center_stats = ribana.get_nn_stats(new_object_motl, new_object_motl)
            bins = np.histogram(center_stats["distance"].values, bins=center_stats.shape[0])
            bin_id = mathutils.otsu_threshold(bins[0])
            dist_threshold = bins[1][bin_id]

            if dist_threshold <= npc_radius:
                merge_candidates = center_stats.loc[
                    center_stats["distance"] <= dist_threshold, ["subtomo_idx", "subtomo_nn_idx"]
                ].values

                for o in np.arange(merge_candidates.shape[0]):
                    o_id1 = new_object_motl.df.loc[
                        new_object_motl.df["subtomo_id"] == merge_candidates[o, 0], "object_id"
                    ].values[0]
                    o_id2 = new_object_motl.df.loc[
                        new_object_motl.df["subtomo_id"] == merge_candidates[o, 1], "object_id"
                    ].values[0]
                    input_motl.df.loc[
                        (input_motl.df["tomo_id"] == current_tomo_id) & (input_motl.df["object_id"] == o_id2),
                        "object_id",
                    ] = o_id1

        return input_motl
