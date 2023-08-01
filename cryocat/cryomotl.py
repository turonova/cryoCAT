import decimal
import emfile
import numpy as np
import os
import pandas as pd
import starfile
import subprocess
import re

from cryocat.exceptions import UserInputError
from cryocat import cryomaps
from cryocat import geom

from math import ceil
from matplotlib import pyplot as plt
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as rot


class Motl:
    # Motl module example usage
    #
    # Initialize a Motl instance from an emfile
    #   `motl = Motl.load(’path_to_em_file’)`
    # Run clean_by_otsu and write the result to a new file
    #   `motl.clean_by_otsu(4, histogram_bin=20).write_to_emfile('path_to_output_em_file')`
    # Run class_consistency on multiple Motl instances
    #   `motl_intersect, motl_bad, cl_overlap = Motl.class_consistency(Motl.load('emfile1', 'emfile2', 'emfile3'))`

    def __init__(self, motl_df=None, motl_path=None, header=None):
        if motl_df is not None:
            self.df = motl_df
        elif os.path.isfile(motl_path):
            self.df, self.header = Motl.read_in(motl_path)
        else:
            self.df = Motl.create_empty_motl()

        self.header = header if header else {}

    @staticmethod
    def create_empty_motl():
        empty_motl_df = pd.DataFrame(
            columns=[
                "score",
                "geom1",
                "geom2",
                "subtomo_id",
                "tomo_id",
                "object_id",
                "subtomo_mean",
                "x",
                "y",
                "z",
                "shift_x",
                "shift_y",
                "shift_z",
                "geom4",
                "geom5",
                "geom6",
                "phi",
                "psi",
                "theta",
                "class",
            ],
            dtype=float,
        )
        return empty_motl_df

    @staticmethod
    def create_motl_from_data(
        tomo_number,
        object_number,
        coordinates,
        angles,
        max_intensity,
        mean_angular_distance,
    ):
        particle_number = coordinates.shape[0]

        motl_df = pd.DataFrame(data=np.zeros((particle_number, 20)))
        motl_df.columns = [
            "score",
            "geom1",
            "geom2",
            "subtomo_id",
            "tomo_id",
            "object_id",
            "subtomo_mean",
            "x",
            "y",
            "z",
            "shift_x",
            "shift_y",
            "shift_z",
            "geom4",
            "geom5",
            "geom6",
            "phi",
            "psi",
            "theta",
            "class",
        ]
        motl_df["subtomo_id"] = np.arange(1, particle_number + 1, 1)
        motl_df["tomo_id"] = tomo_number
        motl_df["object_id"] = object_number
        motl_df["class"] = 1
        motl_df["geom5"] = max_intensity
        motl_df["geom6"] = mean_angular_distance

        # round coord
        motl_df[["x", "y", "z"]] = np.round(coordinates.values)

        # get shifts
        motl_df[["shift_x", "shift_y", "shift_z"]] = coordinates.values - np.round(
            coordinates.values
        )

        # assign angles
        motl_df[["phi", "psi", "theta"]] = angles.values

        return motl_df

    @staticmethod  # TODO move to different module
    def pad_with_zeros(number, total_digits):
        # Creates a string of the length specified by total_digits, containing a given number and fills the rest
        # (at the beginning) with zeros, i.e. from the input parameters 3 and 6 it creates 000003)
        #
        # Input:  number - number to be padded with zero (converts the number to integer)
        #         total_digits - final length of the output string
        # Output: string of length specified by total_digits and containing the input number at the very end

        zeros = "0" * (total_digits - len(str(int(number))))
        padded_str = zeros + str(int(number))

        return padded_str

    @staticmethod
    def get_feature(cols, feature_id):
        if isinstance(feature_id, int):
            if feature_id < len(cols):
                feature = cols[feature_id]
            else:
                raise UserInputError(
                    f"Given feature index is out of bounds. The index must be within the range 0-{len(cols) - 1}."
                )
        else:
            if feature_id in cols:
                feature = feature_id
            else:
                raise UserInputError(
                    "Given feature name does not correspond to any motl column."
                )

        return feature

    @staticmethod  # TODO move to different (sgmotl) module?
    def batch_stopgap2em(motl_base_name, iter_no):
        em_list = []
        for i in range(iter_no):
            motl_path = f"{motl_base_name}_{str(i+1)}"
            star_motl = starfile.read(f"{motl_path}.star")
            motl = Motl.stopgap_to_av3(star_motl)
            em_path = f"{motl_path}.em"
            motl.write_to_emfile(em_path)
            em_list.append(em_path)

        return em_list

    @staticmethod  # TODO move to different module?
    def otsu_threshold(bin_counts):
        # Taken from: https://www.kdnuggets.com/2018/10/basic-image-analysis-python-p4.html
        s_max = (0, 0)

        for threshold in range(len(bin_counts)):
            # update
            w_0 = sum(bin_counts[:threshold])
            w_1 = sum(bin_counts[threshold:])

            mu_0 = (
                sum([i * bin_counts[i] for i in range(0, threshold)]) / w_0
                if w_0 > 0
                else 0
            )
            mu_1 = (
                sum([i * bin_counts[i] for i in range(threshold, len(bin_counts))])
                / w_1
                if w_1 > 0
                else 0
            )

            # calculate - inter class variance
            s = w_0 * w_1 * (mu_0 - mu_1) ** 2

            if s > s_max[1]:
                s_max = (threshold, s)

        return s_max[0]

    @staticmethod
    def load_dimensions(dims):
        if os.path.isfile(dims):
            dimensions = pd.read_csv(dims, sep="\s+", header=None, dtype=float)
        elif isinstance(dims, pd.DataFrame):
            dimensions = dims
        else:
            dimensions = pd.DataFrame(dims)

        # Test the size - acceptable is 1,3 for the same coordinates for all tomograms, or Nx4 where N is number of tomograms
        if dimensions.shape == (1, 3):
            dimensions.columns = ["x", "y", "z"]
        elif dimensions.shape[1] == 4:
            dimensions.columns = ["tomo_id", "x", "y", "z"]
        else:
            raise UserInputError(
                f"The dimensions should have shape of 1x3 or Nx4, where N is number of tomograms."
                f" Instead following shape was specified: {dimensions.shape}."
            )

        return dimensions

    @staticmethod
    def stopgap2emmotl(stopgap_star_file, output_emmotl=None, update_coord=False):
        if isinstance(stopgap_star_file, str):
            stopgap_df = pd.DataFrame(starfile.read(stopgap_star_file))
        else:
            stopgap_df = stopgap_star_file

        motl_df = Motl.create_empty_motl()

        pairs = {
            "subtomo_id": "subtomo_num",
            "tomo_id": "tomo_num",
            "object_id": "object",
            "x": "orig_x",
            "y": "orig_y",
            "z": "orig_z",
            "score": "score",
            "shift_x": "x_shift",
            "shift_y": "y_shift",
            "shift_z": "z_shift",
            "phi": "phi",
            "psi": "psi",
            "theta": "the",
            "class": "class",
        }

        for em_key, star_key in pairs.items():
            motl_df[em_key] = stopgap_df[star_key]

        motl_df["geom4"] = [
            0.0 if hs.lower() == "a" else 1.0 for hs in stopgap_df["halfset"]
        ]

        new_motl = Motl(motl_df)

        if update_coord:
            new_motl.update_coordinates()

        if output_emmotl is not None:
            new_motl.write_to_emfile(output_emmotl)

        return new_motl

    def get_barycentric_motl(self, idx, nn_idx):
        coord1 = self.get_coordinates()[idx, :]
        new_coord = coord1.copy()

        n_vertices = nn_idx.shape[1]

        coord2 = np.zeros((nn_idx.shape[0], 3, n_vertices))

        for i in range(n_vertices):
            coord2[:, :, i] = self.get_coordinates()[nn_idx[:, i], :]
            new_coord += coord2[:, :, i]

        new_coord = new_coord / (n_vertices + 1)
        print(new_coord[0, :])
        c_coord = new_coord - coord1

        angles = self.df[["phi", "theta", "psi"]].values
        angles = angles[idx, :]

        """
        norm_start_vec = np.linalg.norm([1, 0, 0])
        norm_diff_vecs = np.linalg.norm(rotated_coord, axis=1)
        cos_angles = np.sum([1, 0, 0] * rotated_coord, axis=1) / (
            norm_start_vec * norm_diff_vecs
        )
        phi_rotation = np.rad2deg(np.arccos(cos_angles))
        phi_rotation = np.rad2deg(np.arctan(c_coord[:, 1] / c_coord[:, 0]))
        """
        # starting frame created from the first orientation
        w1 = geom.euler_angles_to_normals(angles[0, :])
        w2 = c_coord[0, :] / np.linalg.norm(c_coord[0, :])
        w3 = np.cross(w1, w2)
        w3 = (w3 / np.linalg.norm(w3)).reshape(
            3,
        )
        w_base_mat = np.asarray([w1.reshape((3,)), w2, w3]).T
        print(w1, w2, w3)
        print(w_base_mat)

        v1 = geom.euler_angles_to_normals(angles)

        rot_angles = np.zeros(angles.shape)

        for i in range(1, angles.shape[0]):
            v2 = c_coord[i, :] / np.linalg.norm(c_coord[i, :])
            v3 = np.cross(v1[i, :], v2)
            v3 = (v3 / np.linalg.norm(v3)).reshape(
                3,
            )
            v_base_mat = np.asarray([v1[i, :].reshape((3,)), v2, v3])
            final_mat = np.matmul(w_base_mat, v_base_mat)
            final_rot = rot.from_matrix(final_mat)
            rot_angles[i, :] = final_rot.as_euler("zxz", degrees=True)

        new_motl_df = Motl.create_empty_motl()

        new_motl_df[["x", "y", "z"]] = new_coord
        new_motl_df[["shift_x", "shift_y", "shift_z"]] = 0.0
        new_motl_df["tomo_id"] = self.df["tomo_id"].values[idx]
        new_motl_df["object_id"] = self.df["object_id"].values[idx]
        new_motl_df["class"] = 1
        new_motl_df["subtomo_id"] = range(1, new_motl_df.shape[0] + 1)
        # new_motl_df[["phi", "psi", "theta"]] = self.df[["phi", "psi", "theta"]].values[
        #    idx
        # ]
        new_motl_df[["phi", "theta", "psi"]] = rot_angles

        new_motl = Motl(new_motl_df)
        new_motl.update_coordinates()

        return new_motl

    def get_relative_position(self, idx, nn_idx):
        coord1 = self.get_coordinates()[idx, :]
        coord2 = self.get_coordinates()[nn_idx, :]
        print(coord1.shape, coord2.shape)
        c_coord = coord2 - coord1

        angles = self.df[["phi", "theta", "psi"]].values
        angles = angles[idx, :]
        rot_coord = rot.from_euler("zxz", angles=angles, degrees=True)
        rotated_coord = rot_coord.apply(c_coord)

        """
        norm_start_vec = np.linalg.norm([1, 0, 0])
        norm_diff_vecs = np.linalg.norm(rotated_coord, axis=1)
        cos_angles = np.sum([1, 0, 0] * rotated_coord, axis=1) / (
            norm_start_vec * norm_diff_vecs
        )
        phi_rotation = np.rad2deg(np.arccos(cos_angles))
        phi_rotation = np.rad2deg(np.arctan(c_coord[:, 1] / c_coord[:, 0]))
        """
        # starting frame created from the first orientation
        w1 = geom.euler_angles_to_normals(angles[0, :])
        w2 = c_coord[0, :] / np.linalg.norm(c_coord[0, :])
        w3 = np.cross(w1, w2)
        w3 = (w3 / np.linalg.norm(w3)).reshape(
            3,
        )
        w_base_mat = np.asarray([w1.reshape((3,)), w2, w3]).T
        print(w1, w2, w3)
        print(w_base_mat)

        v1 = geom.euler_angles_to_normals(angles)

        rot_angles = np.zeros(angles.shape)

        for i in range(1, angles.shape[0]):
            v2 = c_coord[i, :] / np.linalg.norm(c_coord[i, :])
            v3 = np.cross(v1[i, :], v2)
            v3 = (v3 / np.linalg.norm(v3)).reshape(
                3,
            )
            v_base_mat = np.asarray([v1[i, :].reshape((3,)), v2, v3])
            final_mat = np.matmul(w_base_mat, v_base_mat)
            final_rot = rot.from_matrix(final_mat)
            rot_angles[i, :] = final_rot.as_euler("zxz", degrees=True)

        new_coord = (coord1 + coord2) / 2.0
        print(new_coord[0, :])
        new_motl_df = Motl.create_empty_motl()

        new_motl_df[["x", "y", "z"]] = new_coord
        new_motl_df[["shift_x", "shift_y", "shift_z"]] = 0.0
        new_motl_df["tomo_id"] = self.df["tomo_id"].values[idx]
        new_motl_df["object_id"] = self.df["object_id"].values[idx]
        new_motl_df["class"] = 1
        new_motl_df["subtomo_id"] = range(1, new_motl_df.shape[0] + 1)
        # new_motl_df[["phi", "psi", "theta"]] = self.df[["phi", "psi", "theta"]].values[
        #    idx
        # ]
        new_motl_df[["phi", "theta", "psi"]] = rot_angles

        new_motl = Motl(new_motl_df)
        new_motl.update_coordinates()

        return new_motl, rotated_coord

    @staticmethod
    def recenter_particles(df):
        # Python 0.5 rounding: round(1.5) = 2, BUT round(2.5) = 2, while in Matlab round(2.5) = 3
        def round_and_recenter(row):
            new_row = row.copy()
            shifted_x = row["x"] + row["shift_x"]
            shifted_y = row["y"] + row["shift_y"]
            shifted_z = row["z"] + row["shift_z"]
            new_row["x"] = float(
                decimal.Decimal(shifted_x).to_integral_value(
                    rounding=decimal.ROUND_HALF_UP
                )
            )
            new_row["y"] = float(
                decimal.Decimal(shifted_y).to_integral_value(
                    rounding=decimal.ROUND_HALF_UP
                )
            )
            new_row["z"] = float(
                decimal.Decimal(shifted_z).to_integral_value(
                    rounding=decimal.ROUND_HALF_UP
                )
            )
            new_row["shift_x"] = shifted_x - new_row["x"]
            new_row["shift_y"] = shifted_y - new_row["y"]
            new_row["shift_z"] = shifted_z - new_row["z"]
            return new_row

        new_df = df.apply(round_and_recenter, axis=1)

        return new_df

    @staticmethod
    def point2point_distance(point1, point2):
        # Input: point1, point2 - points in form of list or numpy array, e.g. np.array([1, 1, 1]) or [1, 1, 1]
        # Output: distance between the two points, float
        p1 = np.array(point1)
        p2 = np.array(point2)

        dist = (
            np.dot(p1.transpose(), p1)
            + np.dot(p2.transpose(), p2)
            - 2 * (np.dot(p1.transpose(), p2))
        )

        # Set negative values to zeroes
        dist = 0 if dist < 0 else dist
        dist = np.sqrt(dist)

        return dist

    @staticmethod
    def read_in(emfile_path):
        header, parsed_emfile = emfile.read(emfile_path)
        if not len(parsed_emfile[0][0]) == 20:
            raise UserInputError(
                f"Provided file contains {len(parsed_emfile[0][0])} columns, while 20 columns are expected."
            )

        motl_df = pd.DataFrame(
            data=parsed_emfile[0],
            dtype=float,
            columns=[
                "score",
                "geom1",
                "geom2",
                "subtomo_id",
                "tomo_id",
                "object_id",
                "subtomo_mean",
                "x",
                "y",
                "z",
                "shift_x",
                "shift_y",
                "shift_z",
                "geom4",
                "geom5",
                "geom6",
                "phi",
                "psi",
                "theta",
                "class",
            ],
        )

        return motl_df, header

    @staticmethod
    def convert_csv2em(input_csv, output_file_name, max_intensity=1):
        info_table = pd.read_csv(input_csv)

        # Extract position of centroid and convert to int
        if max_intensity == 1:
            pd_points = pd.DataFrame(
                {
                    "x": info_table["max_intensity_0"].round(0).astype("int"),
                    "y": info_table["max_intensity_1"].round(0).astype("int"),
                    "z": info_table["max_intensity_2"].round(0).astype("int"),
                }
            )
            pd_angles = pd.DataFrame(
                {
                    "phi": info_table["max_intensity_phi"],
                    "psi": info_table["max_intensity_psi"],
                    "theta": info_table["max_intensity_theta"],
                }
            )
        else:
            pd_points = pd.DataFrame(
                {
                    "x": info_table["centroid-0"].round(0).astype("int"),
                    "y": info_table["centroid-1"].round(0).astype("int"),
                    "z": info_table["centroid-2"].round(0).astype("int"),
                }
            )
            pd_angles = pd.DataFrame(
                {
                    "phi": info_table["centroid_phi"],
                    "psi": info_table["centroid_psi"],
                    "theta": info_table["centroid_theta"],
                }
            )

        pd_max_intensity = pd.DataFrame({"geom5": info_table["max_intensity"]})
        pd_tomo_id = pd.DataFrame({"tomo_id": info_table["tomo_id"]})

        if "mean_angular_distance" in info_table.columns:
            pd_mean_angular_distance = pd.DataFrame(
                {"geom6": info_table["mean_angular_distance"]}
            )
        else:
            pd_mean_angular_distance = 0

        pd_motl = Motl.create_motl_from_data(
            pd_tomo_id,
            2,
            pd_points,
            pd_angles.astype(float),
            pd_max_intensity,
            pd_mean_angular_distance,
        )
        motl_array = pd_motl.values
        motl_array = motl_array.reshape((1, motl_array.shape[0], motl_array.shape[1]))
        emfile.write(output_file_name, motl_array, overwrite=True)

    @classmethod
    def read_from_emfile(cls, emfile_path):
        header, parsed_emfile = emfile.read(emfile_path)
        if not len(parsed_emfile[0][0]) == 20:
            raise UserInputError(
                f"Provided file contains {len(parsed_emfile[0][0])} columns, while 20 columns are expected."
            )

        motl = pd.DataFrame(
            data=parsed_emfile[0],
            dtype=float,
            columns=[
                "score",
                "geom1",
                "geom2",
                "subtomo_id",
                "tomo_id",
                "object_id",
                "subtomo_mean",
                "x",
                "y",
                "z",
                "shift_x",
                "shift_y",
                "shift_z",
                "geom4",
                "geom5",
                "geom6",
                "phi",
                "psi",
                "theta",
                "class",
            ],
        )
        motl["class"] = motl["class"].fillna(1.0)

        return cls(motl, header)

    def flip_handedness(self, tomo_dimensions=None):
        # Input: dimensions of tomograms in the motl; if not provided only the orientation is changed
        # Output: changed motl

        # Orientation flipped - for ZXZ convention need only change of theta
        # 180-psi,theta,180-phi
        # self.df.loc[:, 'psi'] = self.df.loc[:, 'psi']-180.0
        self.df.loc[:, "theta"] = -self.df.loc[:, "theta"]

        # Position flip
        if tomo_dimensions is not None:
            dims = self.load_dimensions(tomo_dimensions)
            if dims.shape == (1, 3):
                z_dim = float(dims["z"]) + 1
                self.df.loc[:, "z"] = z_dim - self.df.loc[:, "z"]
            else:
                tomos = dims["tomo_id"].unique()
                for t in tomos:
                    z_dim = float(dims[dims["tomo_id"] == t, "z"]) + 1
                    self.df.loc[self.df["tomo_id"] == t, "z"] = (
                        z_dim - self.df.loc[self.df["tomo_id"] == t, "z"]
                    )

    @classmethod
    def load(cls, input_motl):
        # Input: Load one or more emfiles (as a list), or already initialized instances of the Motl class
        #        E.g. `Motl.load([cryo1.em, cryo2.em, motl_instance1])`
        # Output: Returns one instance, or a list of instances if multiple inputs are provided

        loaded = list()
        motls = [input_motl] if not isinstance(input_motl, list) else input_motl
        if len(motls) == 0:
            raise UserInputError(
                "At least one em file or a Motl instance must be provided."
            )
        else:
            for motl in motls:
                if isinstance(motl, cls):
                    new_motl = motl
                elif isinstance(motl, str):
                    if os.path.isfile(motl):
                        if os.path.splitext(motl)[-1] == ".em":
                            new_motl = cls.read_from_emfile(motl)
                        else:
                            raise UserInputError(
                                f"Unknown file type: {motl}. Input needs to be either an em file "
                                f"(.em), or an instance of the Motl class."
                            )
                    else:
                        raise UserInputError(f"Provided file {motl} does not exist.")
                else:
                    raise UserInputError(f"Unkown input type ({motl}).")
                # TODO or will it still be possible to receive the motl in form of a pure matrix?

                if not np.array_equal(
                    new_motl.df.columns, cls.create_empty_motl().columns
                ):
                    raise UserInputError(
                        f"Provided Motl object {motl} seems to be corrupted and can not be loaded."
                    )
                else:
                    loaded.append(new_motl)

            if len(loaded) == 1:
                loaded = loaded[0]

        return loaded

    @classmethod  # TODO move to different (sgmotl) module?
    def stopgap_to_av3(cls, star_motl):
        # Accepts input read from the star file (using the starfile.read), and outputs instance of Motl class
        # To write the resulting em motl to the wile, run write_to_emfile.
        # Example: Motl.stopgap_to_av3(starfile.read('path_to_star_file')).write_to_emfile('path_to_output_emfile')

        motl = cls.create_empty_motl()
        # TODO do we want to use 'motl_idx' as index of the dataframe or drop it?
        pairs = {
            "subtomo_id": "subtomo_num",
            "tomo_id": "tomo_num",
            "object_id": "object",
            "x": "orig_x",
            "y": "orig_y",
            "z": "orig_z",
            "score": "score",
            "shift_x": "x_shift",
            "shift_y": "y_shift",
            "shift_z": "z_shift",
            "phi": "phi",
            "psi": "psi",
            "theta": "the",
            "class": "class",
        }
        for em_key, star_key in pairs.items():
            motl[em_key] = star_motl[star_key]
        motl["geom4"] = [
            0.0 if hs.lower() == "a" else 1.0 for hs in star_motl["halfset"]
        ]

        return cls(motl)

    @classmethod
    def merge_and_renumber(cls, motl_list):
        merged_df = cls.create_empty_motl()
        feature_add = 0

        if not isinstance(motl_list, list) or len(motl_list) == 0:
            raise UserInputError(
                f"You must provide a list of em file paths, or Motl instances. "
                f"Instead, an instance of {type(motl_list).__name__} was given."
            )

        for m in motl_list:
            motl = cls.load(m)
            feature_min = min(motl.df.loc[:, "object_id"])

            if feature_min <= feature_add:
                motl.df.loc[:, "object_id"] = motl.df.loc[:, "object_id"] + (
                    feature_add - feature_min + 1
                )

            merged_df = pd.concat([merged_df, motl.df])
            feature_add = max(motl.df.loc[:, "object_id"])

        merged_motl = cls(merged_df)
        merged_motl.renumber_particles()
        return merged_motl

    @classmethod
    def get_particle_intersection(cls, motl1, motl2):
        m1, m2 = cls.load([motl1, motl2])
        m2_values = m2.df.loc[:, "subtomo_id"].unique()
        intersected = cls.create_empty_motl()

        for value in m2_values:
            submotl = m1.df.loc[m1.df["subtomo_id"] == value]
            intersected = pd.concat([intersected, submotl])

        return cls(intersected.reset_index(drop=True))

    @classmethod
    def class_consistency(cls, *args):
        # Input: list of motls
        # Output: intersection (Motl instance), bad (Motl instance), clo (np array),

        if len(args) < 2:
            raise UserInputError("At least 2 motls are needed for this analysis")

        no_cls, all_classes = 1, []
        loaded = cls.load(list(args))
        min_particles = len(loaded[0].df)

        # get number of classes
        for motl in loaded:
            min_particles = min(min_particles, len(motl.df))
            clss = np.sort(motl.df.loc[:, "class"].unique())
            no_cls = max(len(clss), no_cls)
            if no_cls == len(clss):
                all_classes = clss

        cls_overlap = np.zeros((no_cls, len(loaded) - 1))
        # mid_overlap = np.zeros(min_particles, all_classes)
        # mid_overlap = np.zeros(min_particles, all_classes)

        motl_intersect = cls.create_empty_motl()
        motl_bad = cls.create_empty_motl()

        for i, cl in enumerate(all_classes):
            i_motl = loaded[0]
            i_motl_df = i_motl.df.loc[i_motl.df["class"] == cl]

            for j, motl in enumerate(loaded):
                if j == 0:
                    continue
                j_motl_df = motl.df.loc[motl.df["class"] == cl]

                cl_o = len(pd.merge(i_motl_df, j_motl_df, how="inner", on="subtomo_id"))
                cls_overlap[i, j - 1] = cl_o

                i_subtomos = i_motl_df.loc[:, "subtomo_id"].unique()
                j_subtomos = j_motl_df.loc[:, "subtomo_id"].unique()

                j_bad = j_motl_df.loc[~j_motl_df.subtomo_id.isin(i_subtomos)]
                i_bad = i_motl_df.loc[~i_motl_df.subtomo_id.isin(j_subtomos)]
                motl_bad = pd.concat([motl_bad, j_bad, i_bad]).reset_index(drop=True)

                if cl_o != 0:
                    print(
                        f"The overlap for class {cl} of motl #{j} and #{j+1} is {cl_o} ({cl_o / len(i_motl_df) * 100}% of motl "
                        f"#{j} and {cl_o / len(j_motl_df) * 100}% of motl #{j+1}.)"
                    )
                    i_motl_df = i_motl_df.loc[
                        i_motl_df.subtomo_id.isin(j_motl_df.loc[:, "subtomo_id"].values)
                    ]
                else:
                    print(f"Warning: motl # {j+1} does not contain class #{cl}")

            motl_intersect = pd.concat([motl_intersect, i_motl_df])

        header = {}
        return [
            cls(motl_intersect.reset_index(drop=True), header),
            cls(motl_bad.reset_index(drop=True), header),
            np.array([cls_overlap]),
        ]

    def get_coordinates(self, tomo_number=None):
        if tomo_number is None:
            coord = (
                self.df.loc[:, ["x", "y", "z"]].values
                + self.df.loc[:, ["shift_x", "shift_y", "shift_z"]].values
            )
        else:
            coord = (
                self.df.loc[
                    self.df.loc[:, "tomo_id"] == tomo_number, ["x", "y", "z"]
                ].values
                + self.df.loc[
                    self.df.loc[:, "tomo_id"] == tomo_number,
                    ["shift_x", "shift_y", "shift_z"],
                ].values
            )

        return coord

    def get_rotations(self, tomo_number=None):
        if tomo_number is None:
            angles = self.df.loc[:, ["phi", "theta", "psi"]].values
        else:
            angles = self.df.loc[
                self.df.loc[:, "tomo_id"] == tomo_number, ["phi", "theta", "psi"]
            ].values

        rotations = rot.from_euler("zxz", angles, degrees=True)

        return rotations

    def get_angles(self, tomo_number=None):
        if tomo_number is None:
            angles = self.df.loc[:, ["phi", "theta", "psi"]].values
        else:
            angles = self.df.loc[
                self.df.loc[:, "tomo_id"] == tomo_number, ["phi", "theta", "psi"]
            ].values

        return angles

    def get_features(self, feature_id, tomo_number=None):
        if tomo_number is None:
            features = self.df.loc[:, feature_id].values
        else:
            features = self.df.loc[
                self.df.loc[:, "feature_id"] == tomo_number, feature_id
            ].values

        return features

    def clean_by_distance(
        self, distnace_in_voxels, feature_id, metric_id="score", score_cut=0
    ):
        # Distance cutoff (pixels)
        d_cut = distnace_in_voxels

        # Parse tomograms
        features = np.unique(self.get_features(feature_id))

        # Initialize clean motl
        cleaned_df = pd.DataFrame()

        # Loop through and clean
        for f in features:
            # Parse tomogram
            feature_m = self.get_feature_subset(f, feature_id=feature_id)
            n_temp_motl = feature_m.df.shape[0]

            # Parse positions
            pos = feature_m.get_coordinates()

            # Parse scores
            temp_scores = feature_m.df[metric_id].values

            # Sort scores
            sort_idx = np.argsort(temp_scores)[::-1]

            # Temporary keep index
            temp_keep = np.ones((n_temp_motl,), dtype=bool)
            temp_keep[temp_scores < score_cut] = False

            # Loop through in order of score
            for j in sort_idx:
                if temp_keep[j]:
                    # Calculate distances
                    dist = geom.point_pairwise_dist(pos[j, :], pos)

                    # Find cutoff
                    d_cut_idx = dist < d_cut

                    # Keep current entry
                    d_cut_idx[j] = False

                    # Remove other entries
                    temp_keep[d_cut_idx] = False

            # Add entries to main list
            cleaned_df = pd.concat(
                (cleaned_df, feature_m.df.iloc[temp_keep, :]), ignore_index=True
            )

        self.df = cleaned_df

    def write_to_emfile(self, outfile_path):
        # TODO currently replaces all missing values in the whole df, maybe should be more specific to some columns
        filled_df = self.df.fillna(0.0)
        motl_array = filled_df.to_numpy()
        motl_array = motl_array.reshape(
            (1, motl_array.shape[0], motl_array.shape[1])
        ).astype(np.single)
        self.header = {}  # FIXME fails on writing back the header
        emfile.write(outfile_path, motl_array, self.header, overwrite=True)

    # FIXME apply correct coordinate conversion
    def write_to_model_file(
        self, feature_id, output_base, point_size, binning=1.0, zero_padding=3
    ):
        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        outpath = f"{output_base}_{feature}_"

        for value in uniq_values:
            fm = self.df.loc[self.df[feature] == value].reset_index(drop=True)
            feature_str = self.pad_with_zeros(value, zero_padding)
            output_txt = f"{outpath}{feature_str}_model.txt"
            output_mod = f"{outpath}{feature_str}.mod"

            pos_x = (fm.loc[:, "x"] + fm.loc[:, "shift_x"]) * binning
            pos_y = (fm.loc[:, "y"] + fm.loc[:, "shift_y"]) * binning
            pos_z = (fm.loc[:, "z"] + fm.loc[:, "shift_z"]) * binning
            class_v = fm.loc[:, "class"].astype(int)
            dummy = pd.Series(np.repeat(1, len(fm)))

            pos_df = pd.concat([class_v, dummy, pos_x, pos_y, pos_z], axis=1)
            # pos_df = pos_df.astype(float)
            pos_df.to_csv(output_txt, sep="\t", header=False, index=False)

            # Create model files from the coordinates
            # system(['point2model -sc -sphere ' num2str(point_size) ' ' output_txt ' ' output_mod]);
            subprocess.run(
                [
                    "point2model",
                    "-sc",
                    "-sphere",
                    str(point_size),
                    output_txt,
                    output_mod,
                ]
            )

    def remove_feature(self, feature_id, feature_values):
        # Removes particles based on their feature (i.e. tomo number)
        # Inputs: feature_id - col name or index based on which the particles will be removed (i.e. 4 for tomogram id)
        #         feature_values - list of values to be removed
        #         output_motl_name - name of the new motl; if empty the motl will not be written out
        # Usage: motl.remove_feature(4, [3, 7, 8]) - removes all particles from tomograms number 3, 7, and 8

        feature = self.get_feature(self.df.columns, feature_id)

        if not feature_values:
            raise UserInputError(
                "You must specify at least one feature value, based on witch the particles will be removed."
            )
        else:
            if not isinstance(feature_values, list):
                feature_values = [feature_values]
            for value in feature_values:
                self.df = self.df.loc[self.df[feature] != value]

        return self

    def update_coordinates(self):
        self.df = self.recenter_particles(self.df)
        return self

    def scale_coordinates(self, scaling_factor):
        for coord in ("x", "y", "z"):
            self.df[coord] = self.df[coord] * scaling_factor
            shift_column = "shift_" + coord
            self.df[shift_column] = self.df[shift_column] * scaling_factor

        return self

    def tomo_subset(self, tomo_numbers):  # TODO add tests
        # Updates motl to contain only particles from tomograms specified by tomo numbers
        # Input: tomo_numbers - list of selected tomogram numbers to be included
        #        renumber_particles - renumber from 1 to the size of the new motl if True

        new_motl = self.__class__.create_empty_motl()
        for i in tomo_numbers:
            df_i = self.df.loc[self.df["tomo_id"] == i]
            new_motl = pd.concat([new_motl, df_i])
        self.df = new_motl.reset_index()
        return self

    def get_feature_subset(
        self, feature_values, feature_id="tomo_id", return_df=False, reset_index=False
    ):  # TODO add tests
        if isinstance(feature_values, list):
            feature_values = np.array(feature_values)
        else:
            feature_values = np.array([feature_values])

        new_df = self.__class__.create_empty_motl()
        for i in feature_values:
            df_i = self.df.loc[self.df[feature_id] == i].copy()
            new_df = pd.concat([new_df, df_i])

        if reset_index:
            new_df = new_df.reset_index()

        if return_df:
            return new_df
        else:
            return Motl(motl_df=new_df)

    def renumber_particles(self):  # TODO add tests
        # new_motl(4,:)=1: size(new_motl, 2);
        self.df.loc[:, "subtomo_id"] = list(range(1, len(self.df) + 1))
        return self

    def split_by_feature(
        self, feature_id, write_out=False, output_prefix=None, feature_desc_id=None
    ):
        # Split motl by uniq values of a selected feature
        # Inputs:   feature_id - column name or index of the feature based on witch the motl will be split
        #           write: save all the resulting Motl instances into separate files if True
        #           output_prefix:
        #           feature_desc_id:  # TODO how should that var look like?
        # Output: list of Motl instances, each containing only rows with one unique value of the given feature

        feature = self.get_feature(self.df.columns, feature_id)
        uniq_values = self.df.loc[:, feature].unique()
        motls = list()

        for value in uniq_values:
            submotl = self.__class__(self.df.loc[self.df[feature] == value])
            motls.append(submotl)

            if write_out:
                if feature_desc_id:
                    for d in feature_desc_id:  # FIXME should not iterate here probably
                        # out_name=[out_name '_' num2str(nm(d,1))];
                        out_name = f"{output_prefix}_{str(d)}"
                    out_name = f"{out_name}_.em"
                else:
                    out_name = f"{output_prefix}_{str(int(value))}.em"
                submotl.write_to_emfile(out_name)

        return motls

    def clean_by_otsu(self, feature_id, histogram_bin=None):
        # Cleans motl by Otsu threshold (based on CC values)
        # feature_id: a feature by which the subtomograms will be grouped together for cleaning;
        #             4 or 'tomo_id' to group by tomogram, 5 to clean by a particle (e.g. VLP, virion)
        # histogram_bin: how fine to split the histogram. Default is 30 for feature 5 and 40 for feature 4;
        #             for smaller number of subtomograms per feature the number should be lower

        feature = self.get_feature(self.df.columns, feature_id)
        tomos = self.df.loc[:, "tomo_id"].unique()
        cleaned_motl = self.__class__.create_empty_motl()

        if histogram_bin:
            hbin = histogram_bin
        else:
            if feature == "tomo_id":
                hbin = 40
            elif feature == "object_id":
                hbin = 30
            else:
                raise UserInputError(
                    f"The selected feature ({feature}) does not correspond either to tomo_id, nor to"
                    f"object_id. You need to specify the histogram_bin."
                )

        for t in tomos:  # if feature == object_id, tomo_id needs to be used too
            tm = self.df.loc[self.df["tomo_id"] == t]
            features = tm.loc[:, feature].unique()

            for f in features:
                fm = tm.loc[tm[feature] == f]
                bin_counts, bin_centers, _ = plt.hist(fm.loc[:, "score"])
                bn = self.otsu_threshold(bin_counts)
                cc_t = bin_centers[bn]
                fm = fm.loc[fm["score"] >= cc_t]

                cleaned_motl = pd.concat([cleaned_motl, fm])

        self.df = cleaned_motl.reset_index(drop=True)
        return self

    def adapt_to_trimming(self, trim_coord_start, trim_coord_end):
        trimvol_coord = np.asarray(trim_coord_start) - 1
        tdim = np.asarray(trim_coord_end) - trimvol_coord
        self.df.loc[:, ["x", "y", "z"]] = self.df.loc[:, ["x", "y", "z"]] - np.tile(
            trimvol_coord, (self.df.shape[0], 1)
        )
        self.df = self.df.loc[
            ~((self.df["x"] < 1.0) | (self.df["y"] < 1.0) | (self.df["z"] < 1.0)), :
        ]
        self.df = self.df.loc[
            ~(
                (self.df["x"] > tdim[0])
                | (self.df["y"] > tdim[1])
                | (self.df["z"] > tdim[2])
            ),
            :,
        ]

        return self

    def clean_by_minimal_param_from_other_motl(self, motl2, param, thresh):
        m2 = self.load(motl2)

        # Groups items in second motl based on 'param' and filter out those that are larger then 'thresh'
        m2.df = m2.df.groupby(param).filter(lambda x: len(x) > thresh)

        self.df = self.df[
            self.df[param].isin(m2.df[param])
        ]  # apply the filter to the main motl

        return self.df.reset_index(drop=True)

    def remove_out_of_bounds_particles(
        self, dimensions, boundary_type, box_size=None, recenter_particles=True
    ):
        dim = self.load_dimensions(dimensions)
        original_size = len(self.df)

        # Get type of bounds
        if boundary_type == "whole":
            if box_size:
                boundary = ceil(box_size / 2)
            else:
                raise UserInputError(
                    "You need to specify box_size when boundary_type is set to 'whole'."
                )
        elif boundary_type == "center":
            boundary = 0
        else:
            raise UserInputError(f"Unknown type of boundaries: {boundary_type}")

        recentered = self.recenter_particles(self.df)
        idx_list = []
        for i, row in recentered.iterrows():
            tn = row["tomo_id"]
            tomo_dim = dim.loc[dim["tomo_id"] == tn, "x":"z"].reset_index(drop=True)
            c_min = [c - boundary for c in row["x":"z"]]
            c_max = [c + boundary for c in row["x":"z"]]
            if (
                (all(c_min) >= 0)
                and (c_max[0] < tomo_dim["x"][0])
                and (c_max[1] < tomo_dim["y"][0])
                and (c_max[2] < tomo_dim["z"][0])
            ):
                idx_list.append(i)

        final_motl = recentered if recenter_particles else self.df
        self.df = final_motl.iloc[idx_list].reset_index(drop=True)

        print(f"Removed {original_size - len(self.df)} particles.")
        print(f"Original size {original_size}, new_size {len(self.df)}")

        return self

    def keep_multiple_positions(self, feature_id, min_no_positions, distance_threshold):
        feature = self.get_feature(self.df.columns, feature_id)
        tomos = self.df.loc[:, "tomo_id"].unique()
        new_motl = self.create_empty_motl()

        for t in tomos:  # if feature == object_id, tomo_id needs to be used too
            tm = self.df.loc[self.df["tomo_id"] == t]
            features = tm.loc[:, feature].unique()

            for f in features:
                fm = tm.loc[tm[feature] == f].reset_index(drop=True)

                for i, row in fm.iterrows():
                    p1 = [
                        row["x"] + row["shift_x"],
                        row["y"] + row["shift_y"],
                        row["z"] + row["shift_z"],
                    ]
                    temp_dist = []
                    for j, row in fm.iterrows():
                        if i == j:
                            continue
                        p2 = [
                            row["x"] + row["shift_x"],
                            row["y"] + row["shift_y"],
                            row["z"] + row["shift_z"],
                        ]
                        dist = self.point2point_distance(p1, p2)
                        temp_dist.append(dist)

                    sp = [x for x in temp_dist if x < distance_threshold]
                    if len(sp) > 0:
                        fm.iloc[i, 15] = len(sp)

                new_motl = pd.concat([new_motl, fm])

        new_motl = new_motl.loc[new_motl["geom6"] >= min_no_positions]
        self.df = new_motl.reset_index(drop=True)
        return self

    ############################
    # PARTIALLY FINISHED METHODS

    @staticmethod
    def spline_sampling(coords, sampling_distance):
        # Samples a spline specified by coordinates with a given sampling distance
        # Input:  coords - coordinates of the spline
        #         sampling_distance: sampling frequency in pixels
        # Output: coordinates of points on the spline

        # spline = UnivariateSpline(np.arange(0, len(coords), 1), coords.to_numpy())
        spline = InterpolatedUnivariateSpline(
            np.arange(0, len(coords), 1), coords.to_numpy()
        )

        # Keep track of steps across whole tube
        totalsteps = 0

        for i, row in coords.iterrows():
            if i == 0:
                continue
            # Calculate projected distance between each point
            dist = Motl.point2point_distance(row, coords.iloc[i - 1])

            # Number of steps between two points; steps are roughly in increments of 1 pixel
            stepnumber = round(dist / sampling_distance)
            # Length of each step
            step = 1 / stepnumber
            # Array to hold fraction of each step between points
            t = np.arrange(i - 1, i, step)  # inclusive end in matlab

            # Evaluate piecewise-polynomial, i.e. spline, at steps 't'.
            # This array contains the Cartesian coordinates of each step

            # Ft(:,totalsteps+1:totalsteps+size(t,2))=ppval(F, t) # TODO
            # scipy.interpolate.NdPPoly
            spline_t = spline(t)

            # Increment the step counter
            totalsteps += len(t)

            return spline_t

    def clean_particles_on_carbon(
        self, model_path, model_suffix, distance_threshold, dimensions
    ):
        tomos_dim = self.load_dimensions(dimensions)
        tomos = self.df.loc[:, "tomo_id"].unique()
        # tomos = [142]
        cleaned_motl = self.__class__.create_empty_motl()

        for t in tomos:
            tomo_str = self.pad_with_zeros(t, 3)
            tm = self.df.loc[self.df["tomo_id"] == t].reset_index(drop=True)

            tdim = tomos_dim.loc[tomos_dim["tomo_id"] == t, "x":"z"]
            pos = pd.concat(
                [
                    (tm.loc[:, "x"] + tm.loc[:, "shift_x"]),
                    (tm.loc[:, "y"] + tm.loc[:, "shift_y"]),
                    (tm.loc[:, "z"] + tm.loc[:, "shift_z"]),
                ],
                axis=1,
            )

            mod_file_name = os.path.join(model_path, f"{tomo_str}{model_suffix}")
            if os.path.isfile(f"{mod_file_name}.mod"):
                raise UserInputError(
                    f"File to be generated ({mod_file_name}.mod) already exists in the destination. "
                    f"Aborting the process to avoid overriding the existing file."
                )
            else:
                cleaned_motl = pd.concat([cleaned_motl, tm])

            subprocess.run(
                [
                    "model2point",
                    "-object",
                    f"{mod_file_name}.mod",
                    f"{mod_file_name}.txt",
                ]
            )
            coord = pd.read_csv(f"{mod_file_name}.txt", sep="\t", header=None)
            carbon_edge = self.spline_sampling(coord.iloc[:, 2:5], 2)
            # TODO what is the first column?
            # carbon_edge = pd.read_csv('./example_files/test/spline/carbon_edge.txt', header=None)

            all_points = []
            for z in np.arrange(0, tdim[2], 2):
                z_points = carbon_edge
                z_points[:, 3] = z
                all_points.append(z_points)

            rm_idx = []
            for p, row in pos.iterrows():
                # TODO just a best guess what method to use, it will depend on the format of results from spline_sampling
                kdt = KDTree(all_points)
                npd = kdt.query(row)
                if npd < distance_threshold:
                    # TODO is this reliable? do the idx from pos correspond to idx in tm?
                    rm_idx.append(p)
            tm.drop(rm_idx, inplace=True)
            cleaned_motl = pd.concat([cleaned_motl, tm])

        self.df = cleaned_motl.reset_index(drop=True)
        return self

    @classmethod
    def recenter_subparticle(cls, motl_list, mask_list, size_list, rotations=None):
        # motl_list = ['SU_motl_gp210_bin4_1.em','SU_motl_gp210_bin4_1.em']
        # mask_list = ['temp_mask.em','temp2_mask.em']
        # size_list = [36 36]
        # rotations = [[], [-90 0 0]]
        # Output: Motl instance. To write the result to a file, you can run:
        #           Motl.recenter_subparticle(motl_list, mask_list, size_list).write_to_emfile(outfile_path)
        # New masks are exported at the same location as original masks, marked as '_centered.em'

        # Generete zero rotations in case they were not specified
        if not rotations:
            rotations = np.zeros((len(mask_list), 3))

        if not len(motl_list) == len(mask_list) == len(size_list) == len(rotations):
            raise UserInputError("The number of elements per argument must be equal.")
        else:
            elements = len(motl_list)

        # Error tolerance - should be done differently and not hard-coded!!! TODO note from original code
        epsilon = 0.00001

        new_motl_df = cls.create_empty_motl()

        for el in range(elements):
            mask = cryomaps.read(mask_list[el])
            motl = cls.load(motl_list[el])
            # new_mask_path = mask_list[el].replace('.em', '_centered.em')

            # set sizes of new and old masks
            mask_size = np.array(mask.shape)
            # new_size = np.repeat(size_list[el], 3)
            old_center = mask_size / 2
            # new_center = new_size / 2

            # find center of mask
            i, j, k = np.asarray(mask > epsilon).nonzero()
            s = np.array([min(i), min(j), min(k)])
            e = np.array([max(i), max(j), max(k)])
            mask_center = (s + e) / 2
            mask_center += 1  # account for python 0 based indices

            # get shifts
            shifts = mask_center - old_center
            # shifts2 = mask_center - new_center
            # shifts2 = [float(decimal.Decimal(x).to_integral_value(rounding=decimal.ROUND_HALF_UP)) for x in shifts2]

            # write out transformed mask to check it's as expeceted
            # new_mask = tom_red(mask, shifts2, new_size)  # TODO
            # if not all(rotations[el] == 0):
            #    new_mask = tom_rotate(new_mask, rotations[:, el])  # TODO
            # emfile.write(new_mask_path, new_mask, {}, overwrite=True)

            # change shifts in the motl accordingly
            motl.shift_positions(shifts).update_coordinates()

            # create quatertions for rotation
            if not all(rotations[el] == 0):
                q1 = rot.from_euler(
                    seq="zxz",
                    angles=motl.df.loc[:, ["phi", "theta", "psi"]],
                    degrees=True,
                ).as_quat()
                q2 = rot.from_euler(
                    seq="zxz", angles=rotations[el], degrees=True
                ).as_quat()
                mult = q1.__mul__(q2)
                new_angles = mult.as_euler()

                motl.df.loc[:, "phi"] = new_angles[0]
                motl.df.loc[:, "psi"] = new_angles[2]
                motl.df.loc[:, "theta"] = new_angles[1]

            # add identifier in case of motls' merge
            # motl.df['geom_6'] = el

            new_motl_df = pd.concat([new_motl_df, motl.df])

        new_motl = cls(new_motl_df.reset_index(drop=True))
        new_motl.renumber_particles()

        return new_motl

    def apply_rotation(self, rotation):
        angles = self.df.loc[:, ["phi", "theta", "psi"]].to_numpy()

        angles_rot = rot.from_euler("zxz", angles, degrees=True)
        final_rotation = angles_rot * rotation
        angles = final_rotation.as_euler("zxz", degrees=True)
        self.df.loc[:, ["phi", "theta", "psi"]] = angles

        return self

    def shift_positions(self, shift):
        # Shifts positions of all subtomgorams in the motl in the direction given by subtomos' rotations
        # Input: shift - 3D vector - e.g. [1, 1, 1]. A vector in 3D is then rotated around the origin = [0 0 0].
        #               Note that the coordinates are with respect to the origin!

        def shift_coords(row):
            v = np.array(shift)
            euler_angles = np.array([[row["phi"], row["theta"], row["psi"]]])
            orientations = rot.from_euler(seq="zxz", angles=euler_angles, degrees=True)
            rshifts = orientations.apply(v)

            row["shift_x"] = row["shift_x"] + rshifts[0][0]
            row["shift_y"] = row["shift_y"] + rshifts[0][1]
            row["shift_z"] = row["shift_z"] + rshifts[0][2]
            return row

        self.df = self.df.apply(shift_coords, axis=1).reset_index(drop=True)
        return self

    @classmethod
    def movement_convergence(cls, motl_base_name, iteration_range, f_std=None):
        """outcome: [ad_all, dist_rmse_all, particle_conv]"""
        c = 1
        i = 1
        m = cls.load(f"{motl_base_name}_{i}.em")
        pc = len(m.df)

        particle_stability = np.zeros((pc, 2))
        particle_stability[:, 0] = m.df.loc[:, "subtomo_id"]

        ad_all = []
        dist_rmse_all = []

        particle_conv = np.zeros((pc, iteration_range))

        if f_std:
            fstd = f_std
        else:
            fstd = 1

        for i in range(2, iteration_range + 1):
            m1 = cls.load(f"{motl_base_name}_{i-1}.em")
            m2 = cls.load(f"{motl_base_name}_{i}.em")

            ad = angular_distance(
                m1.df.loc[:, ["phi", "psi", "theta"]],
                m2.df.loc[:, ["phi", "psi", "theta"]],
            )
            dist_rmse = geometry_point_distance(
                m1.df.loc[:, ["shift_x", "shift_y", "shift_z"]],
                m2.df.loc[:, ["shift_x", "shift_y", "shift_z"]],
            )

            ad_m = np.mean(ad)
            ad_std = np.std(ad)

            dr_m = np.mean(dist_rmse)
            dr_std = np.std(dist_rmse)

            # TODO: REWORK
            # ad_pidx=m2(4,ad>ad_m+fstd*ad_std);
            ad_pidx_arr = m2.df["subtomo_id"].values
            ad_pidx_bool = ad > ad_m + f_std * ad_std
            ad_pidx = np.column_stack((ad_pidx_arr, ad_pidx_bool))
            # assuming that each part is an array
            # dr_pidx=m2(4,dist_rmse>dr_m+fstd*dr_std);
            dr_pidx_arr = m2.df["subtomo_id"].values
            dr_pidx_bool = dist_rmse > dr_m + fstd * dr_std
            dr_pidx = np.column_stack((dr_pidx_arr, dr_pidx_bool))

            # f_pidx=ad_pidx(ismember(ad_pidx,dr_pidx));
            f_pidx = ad_pidx[np.isin(ad_pidx, dr_pidx)]

            # particle_conv(i-1,ismember(m2(4,:),f_pidx))=1;
            particle_conv[np.isin(m2.df[:, "subtomo_id"], f_pidx), i - 1] = 1

            ad_all.append(ad)
            dist_rmse_all.append(dist_rmse)

        sp = np.sum(particle_conv, 1)
        p = (sp > np.mean(sp) + np.std(sp)).nonzero()

        good_motl = m2
        good_motl.df[p, :] = []
        bad_motl = m2.df[p, :]

        # TODO: check it's correct this way
        emfile.write(good_motl, [f"{motl_base_name}_good_{iteration_range}.em"])
        emfile.write(bad_motl, [f"{motl_base_name}_bad_{iteration_range}.em"])

        # TODO: REWORK the plots
        plt.hist(sp)
        plt.show()
        plt.plot(np.mean(ad_all, axis=1))
        plt.show()

        return ad_all, dist_rmse_all, particle_conv

    @classmethod
    def class_convergence(cls, motl_base_name, iteration_range):
        """outcome: [ overall_changes, particle_stability, m_idx, particle_changes]"""

        c = 1
        i = 1

        m = cls.load(f"{motl_base_name}_{i}.em")
        pc = m.shape[0]

        particle_stability = np.zeros((pc, 2))
        particle_stability[:, 0] = m.loc[:, "subtomo_id"]

        particle_changes = []
        overall_changes = []

        for i in range(2, iteration_range + 1):
            m1 = cls.load(f"{motl_base_name}_{iteration_range[i-1]}.em")
            m2 = cls.load(f"{motl_base_name}_{iteration_range[i]}.em")

            ch_idx = m1.score.ne(m2.score)
            overall_changes.append(sum(ch_idx))

            for i in range(particle_stability.shape[0]):
                particle_stability[i, 1] += ch_idx[i]

            if c != 1:
                ch_particles = sum(np.isin(m2[ch_idx, "subtomo_id"], m_idx))
                particle_changes.append(ch_particles / (pc * 100))

            # TODO: edit this part and check the fuctionality
            m_idx = m2.loc[ch_idx, "subtomo_id"]
            c += 1

        # TODO: REWORK the plots
        plt.plot(overall_changes)
        plt.show()
        plt.plot(particle_changes)
        plt.show()
        plt.hist(particle_stability[:, 1])
        plt.show()

        return overall_changes, particle_stability, m_idx, particle_changes

    def split_particles_in_assymetric_units(
        self, symmetry, particle_x_shift, start_offset=0.0, output_name=None
    ):
        phi_angles = np.array([])
        inplane_step = 360 / symmetry

        for a in range(0, 360, int(inplane_step)):
            phi_angles = np.append(phi_angles, a + start_offset)

        phi_angles = phi_angles.reshape(
            symmetry,
        )

        # make up vectors
        starting_vector = np.array([particle_x_shift, 0, 0])
        rho = np.sqrt(starting_vector[0] ** 2 + starting_vector[1] ** 2)
        the = np.arctan2(starting_vector[1], starting_vector[0])
        z = starting_vector[2]

        rot_rho = np.full((symmetry,), rho)
        rep_the = np.full((symmetry,), the) + np.deg2rad(phi_angles)
        rep_z = np.full((symmetry,), z)

        # https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
        # [center_shift(:, 1), center_shift(:, 2), center_shift(:, 3)] = pol2cart([0;0.785398163397448;1.570796326794897;2.356194490192345;3.141592653589793;3.926990816987241;4.712388980384690;5.497787143782138], repmat(10,8,1), repmat(0,8,1));
        center_shift = np.zeros([rot_rho.shape[0], 3])
        center_shift[:, 0] = rot_rho * np.cos(rep_the)
        center_shift[:, 1] = rot_rho * np.sin(rep_the)
        center_shift[:, 2] = rep_z

        new_motl_df = pd.concat([self.df] * symmetry)

        new_motl_df["geom6"] = new_motl_df["subtomo_id"]
        new_motl_df = new_motl_df.sort_values(by="subtomo_id")
        new_motl_df["geom2"] = np.tile(
            np.arange(1, symmetry + 1).reshape(symmetry, 1), (len(self.df), 1)
        )  # new_motl.groupby(['subtomo_id']).transform('max')

        euler_angles = new_motl_df[["phi", "theta", "psi"]]
        rotations = rot.from_euler(seq="zxz", angles=euler_angles, degrees=True)
        center_shift = np.tile(center_shift, (len(self.df), 1))
        phi_angles = np.tile(phi_angles.reshape(symmetry, 1), (len(self.df), 1))
        new_motl_df.loc[:, ["shift_x", "shift_y", "shift_z"]] = new_motl_df.loc[
            :, ["shift_x", "shift_y", "shift_z"]
        ] + rotations.apply(center_shift)
        new_motl_df.loc[:, ["phi"]] = new_motl_df.loc[:, ["phi"]] + phi_angles
        new_motl_df["subtomo_id"] = np.arange(1, len(new_motl_df) + 1)

        new_motl = Motl(new_motl_df)
        new_motl = new_motl.update_coordinates()
        return new_motl


    def subunit_expansion(
        self, symmetry, xyz_shift, output_name=None
    ):
        
        if isinstance(symmetry, str):
            nfold = int(re.findall(r'\d+', symmetry)[-1])
            if symmetry.lower().startswith('c'):
                s_type = 1  # c symmetry
            elif symmetry.lower().startswith('d'):
                s_type = 2  # d symmetry
            else:
                ValueError("Unknown symmetry - currently only c and are supported!")
        elif isinstance(symmetry,(int,float)):
            s_type = 1 # c symmetry
            nfold = symmetry
        else:
            ValueError("The symmetry has to be specified as a string (starting with c or d) or as a number (float, int)!")

        inplane_step = 360 / nfold

        if s_type == 1:
            n_subunits = nfold
            phi_angles = np.arange(0, 360, int(inplane_step))
            new_angles = np.zeros((n_subunits,3))
            new_angles[:,0] = phi_angles
        elif s_type == 2:
            n_subunits = nfold * 2
            in_plane_offset = int(inplane_step/2)
            new_angles = np.zeros((n_subunits,3))
            new_angles[0::2,0] = np.arange(0, 360, int(inplane_step))
            new_angles[1::2,0] = np.arange(0+in_plane_offset, 360+in_plane_offset, int(inplane_step))
            new_angles[1::2,1] = 180

            phi_angles = new_angles[:,0].copy()

        phi_angles = phi_angles.reshape(
            n_subunits,
        )

        # make up vectors
        starting_vector = np.array(xyz_shift)
        rho = np.sqrt(starting_vector[0] ** 2 + starting_vector[1] ** 2)
        the = np.arctan2(starting_vector[1], starting_vector[0])

        rot_rho = np.full((n_subunits,), rho)
        rep_the = np.full((n_subunits,), the) + np.deg2rad(phi_angles)
        rep_z = np.full((n_subunits,), starting_vector[2])
        
        if s_type == 2:
            rep_z[1::2] *= -1

        # https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
        # [center_shift(:, 1), center_shift(:, 2), center_shift(:, 3)] = pol2cart([0;0.785398163397448;1.570796326794897;2.356194490192345;3.141592653589793;3.926990816987241;4.712388980384690;5.497787143782138], repmat(10,8,1), repmat(0,8,1));
        center_shift = np.zeros([rot_rho.shape[0], 3])
        center_shift[:, 0] = rot_rho * np.cos(rep_the)
        center_shift[:, 1] = rot_rho * np.sin(rep_the)
        center_shift[:, 2] = rep_z
        
        new_motl_df = pd.concat([self.df] * n_subunits)

        new_motl_df["geom6"] = new_motl_df["subtomo_id"]
        new_motl_df = new_motl_df.sort_values(by="subtomo_id")
        new_motl_df["geom2"] = np.tile(
            np.arange(1, n_subunits + 1).reshape(n_subunits, 1), (len(self.df), 1)
        )  

        euler_angles = new_motl_df[["phi", "theta", "psi"]]
        rotations = rot.from_euler(seq="zxz", angles=euler_angles, degrees=True)
        center_shift = np.tile(center_shift, (len(self.df), 1))
        new_angles = np.tile(new_angles, (len(self.df), 1))
        new_motl_df.loc[:, ["shift_x", "shift_y", "shift_z"]] = new_motl_df.loc[
            :, ["shift_x", "shift_y", "shift_z"]
        ] + rotations.apply(center_shift)

        new_rotations = rotations * rot.from_euler(seq="zxz", angles=new_angles, degrees=True)
        new_motl_df.loc[:, ["phi","theta","psi"]] = new_rotations.as_euler(seq="zxz", degrees=True)
        
        new_motl_df["subtomo_id"] = np.arange(1, len(new_motl_df) + 1)
        new_motl = Motl(new_motl_df)
        new_motl = new_motl.update_coordinates()
        new_motl.df.reset_index(inplace = True, drop = True)
        return new_motl