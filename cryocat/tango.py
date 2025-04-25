import numpy as np
import math
import re

import copy
import networkx as nx
import itertools
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial

from scipy.spatial.transform import Rotation as R
from scipy.linalg import logm
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA

import sklearn.neighbors as sn

from gudhi import AlphaComplex

from cryocat import geom
from cryocat import cryomotl
from cryocat.geom import Matrix
from cryocat.classutils import get_classes_from_names
import pandas as pd


class Particle:

    # TODO: include check for unit quaternions and range of Euler angles

    def __init__(self, rotation, position, tomo_id=None, motl_fid=None, degrees=True, particle_id=0):
        """Initialize a new instance with the specified rotation, position, and optional identifiers.

        Parameters
        ----------
        rotation : R, list, tuple, np.ndarray
            The rotation can be provided as a scipy.Rotation object, Euler angles (3 values, convention zxz),
            a unit quaternion (4 values), or a rotation matrix (3x3 array).
        position : np.ndarray
            A 1D numpy array representing the position, which must have a size of 3.
        tomo_id : int, optional
            An optional identifier for the tomography, must be an integer or float. Default is None.
        motl_fid : str, optional
            An optional motl feature_id for the particle that will be used as additional label. Default is None.
        degrees : bool, default=True
            A flag indicating whether the rotation is provided in degrees. Ddefault is True.
        particle_id : int, default=0
            An identifier for the particle, must be an integer. Default is 0.

        Raises
        ------
        ValueError
            If the rotation input is invalid.
        TypeError
            If the position is not a numpy.ndarray or does not have a size of 3,
            or if tomo_id, motl_fid, or particle_id are not of the expected types.
        """

        if isinstance(rotation, R):

            self.rotation = rotation.as_matrix()

        elif isinstance(rotation, (list, tuple, np.ndarray)):

            rotation = np.asarray(rotation)

            if rotation.ndim == 1:

                if len(rotation) == 3:
                    self.rotation = R.from_euler("zxz", rotation, degrees=degrees).as_matrix()
                elif len(rotation) == 4:
                    self.rotation = R.from_quat(rotation).as_matrix()

            elif np.array(rotation).shape == (3, 3) and Matrix(rotation).is_SO3():

                self.rotation = rotation
        else:
            raise ValueError(
                "Invalid rotation input. Must be a Rotation object, Euler angles (3 values), "
                "unit quaternion (4 values), or a rotation matrix (3x3 array)."
            )

        if not isinstance(position, np.ndarray):
            raise TypeError(f"Expected 'position' to be of type numpy.ndarray, got {type(position).__name__}")
        elif not position.size == 3:
            raise TypeError(f"Expected size of 'position' is 3, got {position.size}")

        self.position = position.reshape(3)

        def check_type(input_param, param_desc, return_int=True):

            if input_param is not None:
                if isinstance(input_param, (int, float)):
                    if return_int:
                        return int(input_param)
                    else:
                        return input_param
                else:
                    raise TypeError(f"A param {param_desc} has to be a float or an int.")
            else:
                return None

        self.tomo_id = check_type(tomo_id, "tomo_id")
        self.motl_fid = motl_fid
        self.id = check_type(particle_id, "particle_id")

    def __str__(self):

        rototranslation = np.hstack((self.rotation, self.position.reshape(-1, 1)))

        out = []
        for row in rototranslation:
            out.append(" ".join([str(u) for u in row]))

        out.append(f"Tomogram:  {self.tomo_id}")
        out.append(f"motl_fid:     {self.motl_fid}")

        return "\n".join(out)

        # return f"Particle at position \n{self.position} \nwith orientation\n{self.rotation}"

    def __mul__(self, other):
        if isinstance(other, Particle):
            rot_product = self.rotation @ other.rotation
            pos_product = self.rotation @ other.position + self.position
            return Particle(rotation=rot_product, position=pos_product)
        else:
            raise ValueError("Invalid input. Both factors need to be Particle objects.")

    def __eq__(self, other):
        if isinstance(other, Particle):
            rot_1 = R.from_matrix(self.rotation)
            rot_2 = R.from_matrix(other.rotation)

            return rot_1.approx_equal(rot_2) and np.allclose(self.position, other.position)

        else:
            raise ValueError("Invalid input. Both objects need to be Particle objects.")

    def __hash__(self):

        rot_part = tuple(self.rotation.flatten())
        pos_part = tuple(self.position)

        return hash((rot_part, pos_part, self.tomo_id))

    def inv(self):
        """
        Compute inverse of input particle.
        """
        rot_new = self.rotation.T
        pos_new = -rot_new @ self.position
        return Particle(rotation=rot_new, position=pos_new)

    def scale(self, scaling_factor, overwrite=True):

        if isinstance(scaling_factor, (int, float)):

            scaled_position = scaling_factor * self.position

            if overwrite:

                self.position = scaled_position

            else:

                return Particle(
                    rotation=self.rotation, position=scaled_position, tomo_id=self.tomo_id, motl_fid=self.motl_fid
                )

        else:
            raise TypeError("'scaling_factor' needs to be of type int or float.")

    def tangent_at_identity(self):
        """
        Compute tangent from identity in SE(3) pointing in direction of
        self.
        """
        log = logm(self.rotation)
        log = log.real
        return Matrix(log).twist_from_skew_translation(self.position)

    def twist_vector(self, other):
        """
        Compute twist vector describing relative pose of input particles.
        """
        if isinstance(other, Particle):
            shift = self.inv()
            shifted_particle = shift * other

            return shifted_particle.tangent_at_identity()

        else:
            raise ValueError("Invalid input. Both inputs need to be Particle objects.")

    def tangent_subspace_projection(self, other, mode="orientation"):
        """
        Project tangent vector at identity pointing in direction self --> other
        onto subspace corresponding to mode.

        Possible modes: 'orientation', 'position', 'mixed'
        """
        if isinstance(other, Particle):

            twist = self.twist_vector(other)

            if mode == "orientation":

                proj_twist = twist[:3]
            elif mode == "position":

                proj_twist = twist[3:6]
            elif mode == "mixed":

                proj_twist = twist.copy()
            else:

                raise ValueError(
                    "'mode' has to be one of the following options: \n" "'orientation', 'position', 'mixed'"
                )

            return proj_twist
        else:

            raise ValueError("Invalid input. Both objects need to be Particle objects.")

    def distance(self, other, mode="orientation", degrees=False):
        """
        Compute distance between particles based on mode.

        Possible modes:
        - 'orientation' for geodesic (angular) distance between particle orientations
        - 'position' for Euclidean distance between physical particle positions
        - 'mixed' for product metric
        """
        if isinstance(other, Particle):

            if mode == "position":
                return np.linalg.norm(self.tangent_subspace_projection(other, mode))

            elif mode == "orientation":

                gd = np.linalg.norm(self.tangent_subspace_projection(other, mode))  # / np.sqrt(2)

                if degrees:

                    gd = np.degrees(gd)

                return gd

            elif mode == "mixed":

                euclidean = np.linalg.norm(self.tangent_subspace_projection(other, "position"))
                geodesic = np.linalg.norm(self.tangent_subspace_projection(other, "orientation"))  # / np.sqrt(2)

                if degrees:
                    geodesic = np.degrees(geodesic)

                return np.sqrt(euclidean**2 + geodesic**2)

            else:
                raise ValueError(
                    "'mode' has to be one of the following options: \n" "'orientation', 'position', 'mixed'"
                )

        else:
            raise ValueError("Invalid input. Both objects need to be Particle objects.")

    def add_noise(self, noise_level=0.05, mode="orientation", degrees=False):
        """
        Add noise to particle based on mode.

        Possible modes:
        - 'orientation' for geodesic (angular) distance between particle orientations
        - 'position' for Euclidean distance between physical particle positions
        - 'mixed' for product metric
        """

        if mode in ["orientation", "position", "mixed"]:

            if mode == "orientation":

                noisy_rot = Matrix(self.rotation).add_noise_and_project_to_so3(noise_level, degrees)
                return Particle(noisy_rot, self.position, self.tomo_id, self.motl_fid)

            if mode == "position":

                noise = np.random.normal(loc=0, scale=noise_level, size=3)
                noisy_pos = self.position + noise
                return Particle(self.rotation, noisy_pos, self.tomo_id, self.motl_fid)

            if mode == "mixed":

                noisy_rot = Matrix(self.rotation).add_noise_and_project_to_so3(noise_level, degrees)
                noise = np.random.normal(loc=0, scale=noise_level, size=3)
                noisy_pos = self.position + noise
                return Particle(noisy_rot, noisy_pos, self.tomo_id, self.motl_fid)

        else:

            raise ValueError("'mode' has to be one of the following options: \n" "'orientation', 'position', 'mixed'")

    @classmethod
    def identity(cls):
        canonical_orientation = np.identity(3)
        origin = np.zeros(3)
        return Particle(canonical_orientation, origin)

    @classmethod
    def random(cls, lower_bound=0.0, upper_bound=100.0):
        """
        Args:
            upper_bound (float, optional): Upper bound on norm of position of randomly generated
            particle. Defaults to 100.0.
        """
        if upper_bound > 0:

            random_orientation = R.random().as_matrix()
            rand_signs = np.random.choice([-1, 1], 3)
            rand_pos = np.random.rand(3) * rand_signs
            rand_pos = rand_pos / np.linalg.norm(rand_pos)
            rand_scale = np.random.uniform(low=lower_bound, high=upper_bound)
            rand_pos = rand_scale * rand_pos

            return Particle(random_orientation, rand_pos)

        else:
            raise ValueError("Upper bound needs to be positive.")

    def in_plane_angle(self, degrees=True):

        eulers = R.from_matrix(self.rotation).as_euler("zxz", degrees=degrees)

        return eulers[0]


##### Subclass for symmetric particles #####


class SymmParticle(Particle):

    def __init__(self, rotation, position, tomo_id=None, motl_fid=None, particle_id=None, symm=None, custom_rot=None):
        """
        Initialize Particle with a rotation and a position.
        The input rotation can be:
        - A scipy Rotation object
        - Euler angles (sequence of three floats in 'zxz' convention)
        - A unit quaternion (4-element array-like)
        - A rotation matrix (3x3 array-like)
        The input position can be:
        - A numpy.ndarray of size 3
        The input tomo_id can be:
        - An integer
        The input motl_fid can be:
        - A float/ an int
        The input symm can be:
        - A string containing one of ['tetra', 'octa', 'cube', 'ico', 'dodeca']
        - An integer n > 1 referring to C_n symmetry
        The input custom_rot can be a rotation matrix for the case that
        the given particle symmetry does not align with the canonical options
        for platonic solids presented here. This is not needed in the case where sym == n >1.
        """
        super().__init__(rotation, position, tomo_id, motl_fid, particle_id=particle_id)

        ## take care of symmetric part:
        platonic = False
        self.category = None

        if isinstance(symm, str):
            if symm.__contains__("tetra"):
                self.category = "tetrahedron"  # 1
                vertices = geom.tetrahedron()
                platonic = True
            elif symm.__contains__("octa"):
                self.category = "octahedron"  # 2
                vertices = geom.octahedron()
                platonic = True
            elif symm.__contains__("cube"):
                self.category = "cube"  # 3
                vertices = geom.cube()
                platonic = True
            elif symm.__contains__("ico"):
                self.category = "icosahedron"  # 4
                vertices = geom.icosahedron()
                platonic = True
            elif symm.__contains__("dodeca"):
                self.category = "dodecahedron"  # 5
                vertices = geom.dodecahedron()
                platonic = True
            elif symm.startswith("c"):
                self.category = int(re.findall(r"\d+", symm)[-1])
                vertices = self.category

        elif isinstance(symm, (int, float)):
            self.category = int(symm)
            vertices = geom.n_gon_points(symm)  # vertices lie in plane

        if self.category is None or (not platonic and self.category == 1):
            raise ValueError(
                f"{symm} is an invalid symmetry type. It must refer to a platonic solid or be an integer > 1."
            )

        if platonic and custom_rot is None:
            self.solid = vertices @ self.rotation.T
        elif custom_rot is not None:

            if isinstance(custom_rot, R):

                custom_rot = custom_rot.as_matrix()
                vertices = vertices @ custom_rot.T
                self.solid = vertices @ self.rotation.T

            if Matrix(custom_rot).is_SO3():

                vertices = vertices @ custom_rot.T
                self.solid = vertices @ self.rotation.T

            else:
                raise ValueError("custom_rot needs to be a rotation object or a rotation matrix.")
        else:
            plane_angle = self.in_plane_angle(degrees=False)

            plane_rot = np.array(
                [[np.cos(plane_angle), -np.sin(plane_angle)], [np.sin(plane_angle), np.cos(plane_angle)]]
            )
            self.solid = vertices @ plane_rot.T

    def max_dissimilarity(self):
        if self.category in ["cube", "icosahedron"]:

            p1, p2 = self.solid[0], self.solid[2]
            return np.pi - geom.great_circle_distance(p1, p2)

        elif self.category in ["tetrahedron", "octahedron", "dodecahedron"]:

            p1, p2 = self.solid[0], self.solid[1]
            return np.pi - geom.great_circle_distance(p1, p2)

        else:  # cyclic symmetry

            n = self.category
            return np.pi / n

    def similarity_symm(self, other, degrees=False, max=None):
        if self.category != other.category:

            raise ValueError("The symmetry tpyes of the input particles don't match!")
        else:

            sim_measure = geom.hausdorff_distance_sphere(
                self.solid, other.solid
            )  # min_great_circle_distance(self.solid, other.solid)

            if max is not None:

                max_val = max

            else:
                max_val = self.max_dissimilarity()

            if degrees:
                sim_measure = np.degrees(sim_measure)
                max_val = np.degrees(max_val)

            return 1 - sim_measure / max_val

    @classmethod
    def equip_symmetry(input_particle: Particle, symm, custom_rot=None):
        """
        Equip an existing particle object with symmetry information.
        """
        result = SymmParticle(
            input_particle.rotation,
            input_particle.position,
            input_particle.tomo_id,
            input_particle.motl_fid,
            input_particle.id,
            symm,
            custom_rot,
        )
        return result


# TODO - move as a function to cryomotl
def convert_to_particle_list(input_motl, motl_fid=None, subset_tomo_id=None, symm=None, custom_rot=None):
    """Convert a Motl object to a list of Particle objects.

    Parameters
    ----------
    input_motl : str or Motl
        The path to the input Motl file or Motl object to be loaded.
    motl_fid : str, optional
        The column name in the Motl dataframe that should be used as motl_fid.
        If None, motl_fids will be set to None for all particles. Default is None.
    subset_tomo_id : int or list, optional
        A tomo_id(s) that should be used. If None, the entire Motl will be used. Default is None.
    symm: int or str, optional
        Specifies symmetry of a particle. If int is passed, cyclic (C) symmetry is assumed. See SymParticle for more
        details on str specifications. If specified, list of SymmParticle objects is created, instead of list of
        Particle objects. Default is None.
    custom_rot: np.array, optional
        The input custom_rot can be a rotation matrix for the case that the given particle symmetry does not align with
        the canonical options for platonic solids as defined in geom. This is not needed in the case where sym == n >1.
        It is used only if symm is specified. Default to None.

    Returns
    -------
    list of Particle or SymmParticle
        A list of Particle or SymmParticle objects, each containing the rotation angles,
        position coordinates, tomo id, motl feature id, particle id, and symmetry (for SymmParticle).

    """

    tm = cryomotl.Motl.load(input_motl)

    if subset_tomo_id is not None:
        tm = tm.get_motl_subset(subset_tomo_id, feature_id="tomo_id")

    coord = tm.get_coordinates()
    angles = tm.get_angles()

    if motl_fid is None:
        features = [None] * coord.shape[0]
    else:
        features = tm.df[motl_fid].values

    particle_list = []
    for i in range(len(input_motl)):
        if symm is None:
            particle_list.append(
                Particle(
                    rotation=angles[i],
                    position=coord[i],
                    tomo_id=tm.df["tomo_id"].values[i],
                    motl_fid=features[i],
                    particle_id=tm.df.iloc[i]["subtomo_id"],
                    degrees=True,
                )
            )
        else:
            particle_list.append(
                SymmParticle(
                    rotation=angles[i],
                    position=coord[i],
                    tomo_id=tm.df["tomo_id"].values[i],
                    motl_fid=features[i],
                    particle_id=tm.df.iloc[i]["subtomo_id"],
                    degrees=True,
                    symm=symm,
                    custom_rot=custom_rot,
                )
            )

    return particle_list


###### TwistFeature Class ######


class TwistFeature:

    def __init__(
        self,
        input_twist=None,
        input_motl=None,
        nn_radius=None,
        tomo_id_selection=None,
        degrees=False,
        symm=False,
        feature=None,
    ):

        if input_twist is not None:
            if isinstance(input_twist, pd.DataFrame):
                self.df = input_twist
            elif isinstance(input_twist, str):
                self.df = self.read_in(input_twist)
        elif input_motl is not None and isinstance(nn_radius, (float, int)):
            motl = cryomotl.Motl.load(input_motl)
            self.df = TwistFeature.get_nn_twist_stats_within_radius(
                motl, nn_radius, tomo_id_selection, degrees, symm, motl_fid=feature
            )
        else:
            raise ValueError(
                "One has to specify (at least) either input_twist or input_motl in combination with nn_radius."
            )

    def __getitem__(self, item):

        return self.df[item]

    def write_out(self, output_file):
        """
        Save self.df to CSV or Pickle depending on file extension.

        Parameters
        ----------
        output_file : str
            File path. Must end with `.csv` or `.pkl`.
        """
        if output_file.endswith(".csv"):
            self.df.to_csv(output_file, index=False)
        elif output_file.endswith(".pkl"):
            self.df.to_pickle(output_file)
        else:
            raise ValueError("Unsupported file type. Use .csv or .pkl")

    @staticmethod
    def read_in(input_file):
        """
        Reads in a pandas DataFrame from a CSV or Pickle file depending on file extension.

        Parameters
        ----------
        input_file : str
            File path. Must end with `.csv` or `.pkl`.
        """
        if input_file.endswith(".csv"):
            return pd.read_csv(input_file)
        elif input_file.endswith(".pkl"):
            return pd.read_pickle(input_file)
        else:
            raise ValueError("Unsupported file type. Use .csv or .pkl")

    @classmethod
    def load(cls, input_data):
        """
        Create a TwistFeature object from input data

        Parameters
        ----------
        input_data : str or TwistFeature
            File path or TwistFeature object.
        """

        if isinstance(input_data, TwistFeature):
            return copy.deepcopy(input_data)
        elif isinstance(input_data, str):
            df = TwistFeature.read_in(input_data)
            return TwistFeature(df)
        else:
            raise ValueError("The input has to be either a TwistFeature object or a file path (in str format).")

    @staticmethod
    def get_pos_feature_ids():

        return ["twist_x", "twist_y", "twist_z"]

    @staticmethod
    def get_rot_feature_ids():

        return ["twist_so_x", "twist_so_y", "twist_so_z"]

    @staticmethod
    def get_mixed_feature_ids():

        return ["twist_so_x", "twist_so_y", "twist_so_z", "twist_x", "twist_y", "twist_z"]

    @staticmethod
    def get_all_feature_ids(symm=False):

        columns = [
            "qp_id",
            "nn_id",
            "tomo_id",
            "twist_so_x",
            "twist_so_y",
            "twist_so_z",
            "twist_x",
            "twist_y",
            "twist_z",
            "nn_inplane",
            "geodesic_distance_rad",
            "euclidean_distance",
            "product_distance",
            "rot_angle_x",
            "rot_angle_y",
            "rot_angle_z",
        ]
        if symm:
            columns.append("angular_score")

        return columns

    @staticmethod
    def process_tomo_twist(tomo_id, input_motl, nn_radius, degrees=False, symm=False, motl_fid=None):

        coord = input_motl.get_motl_subset(tomo_id).get_coordinates()
        rotations = input_motl.get_motl_subset(tomo_id).get_rotations()
        angles = input_motl.get_motl_subset(tomo_id).get_angles()
        subtomo_ids = input_motl.get_motl_subset(tomo_id).df["subtomo_id"].values

        sub_list = convert_to_particle_list(input_motl, motl_fid=motl_fid, subset_tomo_id=tomo_id)

        kdt_nn = sn.KDTree(coord)
        nn_idx = kdt_nn.query_radius(coord, nn_radius)  # Query all neighbors within radius

        nn_indices = [neighbors[neighbors != i] for i, neighbors in enumerate(nn_idx)]  # Remove self-matches

        # Filter out points with no neighbors left and sort neighbors
        center_idx = [i for i, neighbors in enumerate(nn_indices) if len(neighbors) > 0]
        nn_indices = [np.sort(nn_indices[i]) for i in center_idx]

        flat_center_idx = np.repeat(center_idx, [len(nns) for nns in nn_indices])
        flat_nn_idx = np.concatenate(nn_indices)

        if symm:
            # make sure that this is computed only once
            max_val = sub_list[0].max_dissimilarity()  # was p_list before will this be the same?
            symm_cat = sub_list[0].category

            inplane_1 = np.deg2rad(angles[flat_center_idx, 0])
            inplane_2 = np.deg2rad(angles[flat_nn_idx, 0])

            ang_scores = geom.angular_score_for_c_symmetry(inplane_1, inplane_2, symm_cat, max_val)

        # Compute relative quantities
        rel_pos = rotations[flat_center_idx].inv().apply(coord[flat_nn_idx] - coord[flat_center_idx])
        rel_quat = rotations[flat_center_idx].inv() * rotations[flat_nn_idx]

        axis_angle = rel_quat.as_rotvec()
        twists = np.hstack((axis_angle, rel_pos))

        tomo_idx = tomo_id * np.ones(len(twists), dtype=int)

        if not symm:
            twist_df = pd.DataFrame(
                np.column_stack(
                    (subtomo_ids[flat_center_idx], subtomo_ids[flat_nn_idx], tomo_idx, twists, angles[flat_nn_idx, 0])
                ),
                columns=["qp_id", "nn_id", "tomo_id"] + TwistFeature.get_mixed_feature_ids() + ["nn_inplane"],
            )

        else:
            twist_df = pd.DataFrame(
                np.column_stack(
                    (
                        subtomo_ids[flat_center_idx],
                        subtomo_ids[flat_nn_idx],
                        tomo_idx,
                        twists,
                        angles[flat_nn_idx, 0],
                        ang_scores,
                    )
                ),
                columns=["qp_id", "nn_id", "tomo_id"]
                + TwistFeature.get_mixed_feature_ids()
                + ["nn_inplane", "angular_score"],
            )

        return twist_df

    @staticmethod
    def get_nn_twist_stats_within_radius(
        input_motl, nn_radius, tomo_id_selection=None, degrees=True, symm=False, motl_fid=None
    ):

        tomo_idx = input_motl.get_unique_values("tomo_id")
        if tomo_id_selection is not None:
            tomo_idx = [i for i in tomo_idx if i in tomo_id_selection]

        results = []

        for tomo_id in tomo_idx:
            results.append(TwistFeature.process_tomo_twist(tomo_id, input_motl, nn_radius, degrees, symm, motl_fid))

        # if len(tomo_idx) == 1:
        #     work_opt = 1
        # else:
        #     work_opt = len(tomo_idx) // 2
        # num_workers = min(8, work_opt)  # Adjust workers based on the load
        # with ProcessPoolExecutor(max_workers=num_workers) as executor:
        #     futures = [
        #         executor.submit(process_tomo_twist, tomo_id, input_motl, nn_radius, degrees, symm, feature)
        #         for tomo_id in tomo_idx
        #     ]
        #     for future in futures:
        #         results.extend(future.result())

        twist_df = pd.concat(results, ignore_index=True)

        twist_df["geodesic_distance_rad"] = np.sqrt(
            twist_df["twist_so_x"] ** 2 + twist_df["twist_so_y"] ** 2 + twist_df["twist_so_z"] ** 2
        )
        twist_df["euclidean_distance"] = np.sqrt(
            twist_df["twist_x"] ** 2 + twist_df["twist_y"] ** 2 + twist_df["twist_z"] ** 2
        )
        twist_df["product_distance"] = np.sqrt(
            twist_df["geodesic_distance_rad"] ** 2 + twist_df["euclidean_distance"] ** 2
        )

        twist_df["rot_angle_x"] = np.degrees(np.abs(twist_df["geodesic_distance_rad"] - np.abs(twist_df["twist_so_x"])))
        twist_df["rot_angle_y"] = np.degrees(np.abs(twist_df["geodesic_distance_rad"] - np.abs(twist_df["twist_so_y"])))
        twist_df["rot_angle_z"] = np.degrees(np.abs(twist_df["geodesic_distance_rad"] - np.abs(twist_df["twist_so_z"])))

        return twist_df

    @staticmethod
    def get_axis_feature_id(feature_id, axis="z"):
        if isinstance(axis, str):
            if axis.endswith(("x", "y", "z")):
                column = f"{feature_id}_{axis}"
            else:
                raise ValueError("Axis should refer to x-, y-, or z-axis.")
        else:
            print("Invalid choise; default axis is used.")
            column = "{feature_id}_so_z"

        return column

    def get_twist_mixed_np(self):

        return self.df[TwistFeature.get_mixed_feature_ids()].to_numpy()

    def twist_template_comparison(self, referenc_particle: Particle):
        """
        Shift twist vectors by tangent vector corresponding to input particle.
        Ideally, the input particle corresponds to the relative particle pose of interest.
        """
        template_twist = referenc_particle.tangent_at_identity()
        twist_vectors = self.get_twist_mixed_np()

        return twist_vectors - template_twist

    def sort_by_distance(self, twist_feature_id="geodesic_distance_rad"):

        if not (twist_feature_id not in self.df.columns):
            raise ValueError(f"Feature needs to be one of the following: {self.df.columns}.")

        all_distances = self.df["euclidean_distance"]

        dist_tuples = []

        for i in self.df["qp_id"].unique():

            df_qp = self.df[self.df["qp_id"] == i]
            df_qp = df_qp.sort_values(by="euclidean_distance")

            e_distances = df_qp["euclidean_distance"]
            feat_values = df_qp[twist_feature_id]

            dist_tuples.append((e_distances, feat_values))

        # Create a unified distance grid
        all_distances = np.unique(all_distances)
        unified_distances = np.sort(all_distances)

        # Interpolate feature values for each tuple at the unified time grid
        interpolated_values = []
        for time_series, values in dist_tuples:

            interp_func = interp1d(time_series, values, kind="nearest-up", bounds_error=False, fill_value="extrapolate")
            interpolated_values.append(interp_func(unified_distances))

        # Compute the average values at each time step
        average_values = np.mean(interpolated_values, axis=0)

        return unified_distances, average_values

    #### Get weighted product distance for given TwistFeature
    def weighted_stats(self, position_weight: float, orientation_weight: float):
        weighted_data = TwistFeature()
        weighted_df = self.df.copy()

        euc_dist = self.df["euclidean_distance"].to_numpy()
        geo_dist = self.df["geodesic_distance_rad"].to_numpy()

        prod_dist = np.sqrt(position_weight * euc_dist**2 + orientation_weight * geo_dist**2)

        weighted_df["product_distance"] = prod_dist

        weighted_data.df = weighted_df

        return weighted_data

    def proximity_clustering(self, num_connected_components=1):
        """Cluster particles based on spatial proximity. The num_connected_components largest clusters are returned.

        Args:
            twist_stats (TwistFeature or pandas dataframe): Yields particle indices with respect to which to cluster
            num_connected_components (int, optional): Desired number of clusters. Defaults to 1.

        Raises:
            TypeError: if input data is neither a pandas data frame nor a TwistFeature.

        Returns:
            list: List of networkx graphs representing the num_connected_components largest components.
        """

        qp_idx = self.df["qp_id"]
        nn_idx = self.df["nn_id"]

        edges = list(zip(qp_idx, nn_idx))

        G = nx.Graph()

        G.add_edges_from(edges)

        n = nx.number_connected_components(G)

        if num_connected_components >= n:
            num_connected_components = n

        S = [G.subgraph(c).copy() for c in sorted(nx.connected_components(G), key=len, reverse=True)]

        return S[:num_connected_components]

    def get_qp_twist_feat(self, query_particle, tomo_id=None):
        """Get twist features for a specific query particle from a twist dataframe.

        Parameters
        ----------
        query_particle : int, float or Particle
            The index of the particle or a Particle instance for which statistics are to be retrieved.
        tomo_id : int, optional
            The tomogram id to filter the data. If None, data will be retrieved for all tomograms associated with the
            query particle.

        Returns
        -------
        filtered_twist_feat : TwistFeature
            A TwistFeature containing the filtered statistics for the specified query particle and tomo id.

        Raises
        ------
        ValueError
            If `query_particle` is neither an instance of Particle nor an integer index.

        Notes
        -----
        This function assumes that `twist_df` contains a column named "qp_id" for particle IDs and "tomo_id" for tomography IDs.
        """

        if isinstance(query_particle, (int, float)) or isinstance(query_particle, Particle):

            if isinstance(query_particle, Particle):
                ind = query_particle.id
            else:
                ind = int(query_particle)

            if tomo_id is not None and isinstance(tomo_id, int):
                filtered_data = self.df[(self.df["qp_id"] == ind) & (self.df["tomo_id"] == tomo_id)]
            else:
                filtered_data = self.df[(self.df["qp_id"] == ind)]

            return TwistFeature(input_twist=filtered_data)

        else:
            raise ValueError("query_particle has to be an instance of Particle or an index, both associated to p_list.")

    @classmethod
    def get_data_range(cls, twist_feat, twist_feature_id=None, min_value=None, max_value=None):
        """
        Filter a TwistFeature instance by a specific column and return a new instance.

        Parameters
        ----------
        twist_feat : TwistFeature
            The original TwistFeature instance.
        twist_feature_id : str, optional
            The name of the column to filter on. If None, the original twist_feat will be returned.
        min_value : float, optional
            Minimum value for filtering. Default is None.
        max_value : float, optional
            Maximum value for filtering. Default is None.

        Returns
        -------
        TwistFeature
            A new instance of TwistFeature with filtered data.
        """
        # Return unmodified if no filtering is requested
        if twist_feature_id is None or (min_value is None and max_value is None):
            return twist_feat

        df = twist_feat.df.copy()

        if min_value is not None:
            df = df[df[twist_feature_id] >= min_value]
        if max_value is not None:
            df = df[df[twist_feature_id] <= max_value]

        return TwistFeature(df)

    def get_twist_pos_df(self):
        return self.df[TwistFeature.get_pos_feature_ids()]

    def get_twist_rot_df(self):
        return self.df[TwistFeature.get_rot_feature_ids()]

    def get_twist_mixed_df(self):
        return self.df[TwistFeature.get_mixed_feature_ids()]

    def get_twist_pos_np(self):
        return self.df[TwistFeature.get_pos_feature_ids()].to_numpy()

    def get_twist_rot_np(self):
        return self.df[TwistFeature.get_rot_feature_ids()].to_numpy()


class Filter:

    def __init__(self):
        self.filter = None


class AxisRot(Filter):

    def __init__(self, twist_feat, max_angle, axis="z", min_angle=0.0):

        feature_id = TwistFeature.get_axis_feature_id("rot_angle", axis=axis)
        self.filter = TwistFeature.get_data_range(
            twist_feat=twist_feat, twist_feature_id=feature_id, min_value=min_angle, max_value=max_angle
        )


class EuclideanDistNN(Filter):

    def __init__(self, twist_feat, num_neighbors=1):
        twist_df = twist_feat.df.sort_values(by=["qp_id", "euclidean_distance"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistFeature(twist_df)


class GeodesicDistNN(Filter):

    def __init__(self, twist_feat, num_neighbors=1):
        twist_df = twist_feat.df.sort_values(by=["qp_id", "geodesic_distance_rad"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistFeature(twist_df)


class MixedDistNN(Filter):

    def __init__(self, twist_feat, num_neighbors=1):
        twist_df = twist_feat.df.sort_values(by=["qp_id", "product_distance"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistFeature(twist_df)


class AngularScoreNN(Filter):

    def __init__(self, twist_feat, num_neighbors=1):
        twist_df = twist_feat.df.sort_values(by=["qp_id", "angular_score"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistFeature(twist_df)


class Support:

    def __init__(self):
        self.support = None

    @staticmethod
    def set_axis_and_columns(mode, axis=None):

        if mode in ["orientation", "position", "mixed"]:

            if mode != "mixed":

                if axis is None:
                    axis = np.array([0, 0, 1])
                elif not isinstance(axis, np.ndarray) or axis.size != 3:
                    raise ValueError("'axis' needs to be numpy.ndarray of size 3.")

                if mode == "position":
                    columns = TwistFeature.get_pos_feature_ids()
                else:
                    columns = TwistFeature.get_rot_feature_ids()

            else:

                columns = TwistFeature.get_mixed_feature_ids()

                if axis is None:
                    axis = np.array([0, 0, 0, 0, 0, 1])
                elif not isinstance(axis, np.ndarray) or axis.size != 6:
                    raise ValueError("'axis' needs to be numpy.ndarray of size 6.")

        else:
            raise ValueError(
                f"The mode type {mode} is not supported. Allowed values are orientation, position, or mixed."
            )

        axis = axis / np.linalg.norm(axis)
        return axis, columns


class Sphere(Support):

    def __init__(self, twist_feat, radius=None):
        if radius is None:
            self.support = twist_feat
        else:
            self.support = TwistFeature.get_data_range(
                twist_feat=twist_feat, twist_feature_id="euclidean_distance", min_value=0.0, max_value=radius
            )


class Shell(Support):

    def __init__(self, twist_feat, radius_min, radius_max):
        self.support = TwistFeature.get_data_range(
            twist_feat=twist_feat, twist_feature_id="euclidean_distance", min_value=radius_min, max_value=radius_max
        )


class Cone(Support):

    def __init__(self, twist_feat: TwistFeature, cone_height: float, cone_radius: float, axis=None, mode="position"):

        axis, columns = Support.set_axis_and_columns(mode=mode, axis=axis)

        tangent = twist_feat.df[columns].to_numpy()

        if tangent.ndim != 2 or tangent.shape[1] != 3:
            raise ValueError(f"Expected tangent to have shape (n,3), but got {tangent.shape}")

        cone_slope_angle = np.arctan2(cone_radius, cone_height)

        input_axis_angle = np.apply_along_axis(
            geom.angle_between_n_vectors, 1, tangent, axis
        )  # was vector_angle before - in case of error

        # Projection of input point onto axis of revolution
        axis_projection = np.dot(tangent, axis)[:, None] * axis

        # Projection of input point onto orthogonal complement of axis
        perp_projection = tangent - axis_projection

        # Boolean conditions
        height_bool = np.linalg.norm(axis_projection, axis=1) <= cone_height
        radial_bool = np.linalg.norm(perp_projection, axis=1) <= cone_radius
        angular_bool = np.abs(input_axis_angle) <= cone_slope_angle

        # Mask: rows where all conditions hold
        mask = height_bool & radial_bool & angular_bool

        self.support = TwistFeature(twist_feat.df[mask])


class Torus(Support):

    def __init__(
        self,
        twist_feat: TwistFeature,
        inner_radius: float,
        outer_radius: float,
        axis=None,
        mode="position",
    ):

        axis, columns = Support.set_axis_and_columns(mode=mode, axis=axis)

        tangent = twist_feat[columns].to_numpy()

        # a solid torus can be described as cartesian product
        # S^1 \times D^2
        # S^1 can be thought of as the central latitude
        rad_center = (outer_radius + inner_radius) / 2

        # radius of D^2 (outer_rad > inner_rad, but let's distrust user anyway):
        disc_rad = np.abs(outer_radius - inner_radius) / 2

        # projection onto orthogonal complement of span of axis
        axis_projection = np.dot(tangent, axis)[:, None] * axis
        perp_projection = tangent - axis_projection

        # length of vector connecting input point to projection
        proj_len = np.linalg.norm(axis_projection, axis=1)

        # distance between projected point and central S^1:
        circle_in_plane_dist = np.abs(np.linalg.norm(perp_projection, axis=1) - rad_center)

        # distance between input point and central central S^1:
        circle_dist = np.sqrt(proj_len**2 + circle_in_plane_dist**2)

        # Mask: rows where all conditions hold
        mask = circle_dist <= disc_rad

        self.support = TwistFeature(twist_feat[mask])


class Cylinder(Support):
    def __init__(
        self, twist_feat: TwistFeature, radius: float, height: float, axis=None, mode="position", symmetric=True
    ):

        def compute_one_direction(axis):
            tangent = twist_feat.df[columns].to_numpy()

            axis_projection = np.dot(tangent, axis)[:, None] * axis
            proj_dist = np.linalg.norm(axis_projection - tangent, axis=1)

            axis_portion = np.dot(axis_projection, axis)
            height_test = (axis_portion <= height) & (0 <= axis_portion)
            rad_test = proj_dist <= radius

            # Mask: rows where all conditions hold
            mask = height_test & rad_test

            return twist_feat.df[mask].copy()

        set_axis, columns = Support.set_axis_and_columns(mode=mode, axis=axis)
        truncated_stats_df = compute_one_direction(set_axis)

        if symmetric:
            set_axis, columns = Support.set_axis_and_columns(mode=mode, axis=-set_axis)
            truncated_stats_df = pd.concat([truncated_stats_df, compute_one_direction(set_axis)])

        self.support = TwistFeature(truncated_stats_df)


class Descriptor:

    def __init__(self, planar=False, alpha_complex_needed=False, inner_angles_needed=False):
        self.planar = planar
        self.ac_link_needed = alpha_complex_needed
        self.ac_inner_angles_needed = inner_angles_needed

    def compute(self, **kwargs):
        pass


class NNCount(Descriptor):

    def __init__(self, **kwargs):
        super().__init__(False, False, False)

    def compute(self, **kwargs):
        qp_support = kwargs["qp_support"]
        return qp_support.df.shape[0]


class EulerChar(Descriptor):
    """
    Computes the Euler characteristic of a 1-dimensional simplical complex, e.g. for the
    link of a vertex in a 2-dimensional triangulated surface.
    """

    def __init__(self, **kwargs):
        super().__init__(True, True, False)
        self.alpha_param = kwargs.get("alpha_param", 200)

    def compute(self, **kwargs):
        alpha_link = kwargs["alpha_link"]

        vertices = set(itertools.chain.from_iterable(alpha_link))
        return len(vertices) - len(alpha_link)


class LinkChar(Descriptor):
    """
    Computes a simplicial isomorphism invariant for a 1-dimensional simplicial complex,
    e.g. for tha link of a vertex in a 2-dimensional triangulated surface.
    """

    def __init__(self, **kwargs):
        super().__init__(True, True, False)
        self.alpha_param = kwargs.get("alpha_param", 200)

    def compute(self, **kwargs):

        alpha_link = kwargs["alpha_link"]

        vertices = set(itertools.chain.from_iterable(alpha_link))
        return 1 - 0.5 * len(vertices) + len(alpha_link) / 3


class IncidentAngleMedian(Descriptor):
    """
    Compute median of angles incident to a given vertex, as indicated by
    edges making up the given vertex's link.
    """

    def __init__(self, **kwargs):
        super().__init__(True, True, True)
        self.alpha_param = kwargs.get("alpha_param", 200)

    def compute(self, **kwargs):

        angles = kwargs["inner_angles"]

        if angles != np.NAN:
            return np.median(angles)
        else:
            return angles


class IncidentAngleSTD(Descriptor):
    """
    Compute median of angles incident to a given vertex, as indicated by
    edges making up the given vertex's link.
    """

    def __init__(self, **kwargs):
        # TODO - add support for alpha_link  and gnles computation directly to this class
        super().__init__(True, True, True)
        self.alpha_param = kwargs.get("alpha_param", 200)

    def compute(self, **kwargs):

        angles = kwargs["inner_angles"]

        if angles != np.NAN:
            return np.std(angles)
        else:
            return angles


class PlanarAlphaComplex:

    def __init__(self, alpha_param=200):
        self.alpha = alpha_param
        self.link_edges = None
        self.stree = None
        self.filtration = None

    def compute_stree_and_filtration(self, coordinates):

        alpha_complex = AlphaComplex(points=list(coordinates))
        self.stree = alpha_complex.create_simplex_tree()
        self.filtration = self.stree.get_filtration()

    def compute_link(self, vertex_index):
        """
        From an alpha complex tree (gudhi.AlphaComplex.create_simplex_tree()), compute
        a vertex's link.
        """
        star = self.stree.get_star([vertex_index])
        triangles = [x[0] for x in star if len(x[0]) == 3 and x[1] < self.alpha]

        self.link_edges = []
        for x in triangles:
            y = x.copy()
            y.remove(vertex_index)
            self.link_edges.append(y)

        self.link_edges

    def compute_inner_angles(self, coord, vertex_index=0, add_center=True):
        # TODO - would it make sense to do this computation also for 3D?

        vertex_index = vertex_index

        if add_center:
            vertices = np.vstack((np.zeros(3), coord))
        else:
            vertices = coord

        center = vertices[vertex_index]
        angles = []
        for [i, j] in self.link_edges:
            point_i = vertices[i]
            point_j = vertices[j]
            vec1 = point_i - center
            vec2 = point_j - center
            angles.append(geom.angle_between_n_vectors(vec1, vec2))

        if len(angles) == 0:
            return np.NAN
        else:
            angles = [x for x in angles if not (isinstance(x, float) and math.isnan(x))]
            return angles

    @classmethod
    def compute(cls, coord, alpha_param=200, make_planar=True, add_center=True):
        if make_planar:
            planar_points = coord[:, :2]  # takes only x,y
        else:
            planar_points = coord

        if add_center:
            planar_points = np.vstack((np.zeros(2), planar_points))  # add 0,0 for a query point

        ac = PlanarAlphaComplex(alpha_param=alpha_param)
        ac.compute_stree_and_filtration(planar_points)
        ac.compute_link(0)
        return ac


class DescriptorCatalogue:

    def __init__(self, planar=False, alpha_complex_needed=False, inner_angles_needed=False):
        self.planar = planar
        self.ac_needed = alpha_complex_needed
        self.ac_angles_needed = inner_angles_needed
        self.alpha_param = None
        self.descriptors = None

    def set_support_parameters(self, current_desc):

        if current_desc.ac_link_needed:
            self.ac_needed = True
            if self.alpha_param is None:  # save only for the first occurence
                self.alpha_param = current_desc.alpha_param
            elif self.alpha_param != current_desc.alpha_param:
                raise ValueError("The alpha parameter has to be same for all features.")
            if current_desc.ac_inner_angles_needed:
                self.ac_angles_needed = True

    def set_descriptor_parameters(self, qp_support):

        add_kwargs = {"qp_support": qp_support}
        if self.ac_needed:
            coord = qp_support.get_twist_pos_np()
            ac = PlanarAlphaComplex.compute(coord, alpha_param=self.alpha_param, make_planar=True, add_center=True)
            add_kwargs["alpha_link"] = ac.link_edges
            if self.ac_angles_needed:
                add_kwargs["inner_angles"] = ac.compute_inner_angles(coord, vertex_index=0, add_center=True)

        return add_kwargs

    def compute_descriptors(self, twist_feat, descriptor_list, descriptor_kwargs, support_class, support_kwargs):

        query_points = twist_feat.df["qp_id"].unique()
        qp_desc_values = []

        descriptor_list = get_classes_from_names(descriptor_list, "cryocat.tango")
        support_class = get_classes_from_names(support_class, "cryocat.tango")

        if not descriptor_kwargs:
            descriptor_kwargs = [{} for _ in range(len(descriptor_list))]

        for d, d_kwargs in zip(descriptor_list, descriptor_kwargs):
            self.set_support_parameters(d(**d_kwargs))

        for qp in query_points:
            filtered_twist_feat = twist_feat.get_qp_twist_feat(qp)  # filter twist_feat based on the query point
            qp_support = support_class(filtered_twist_feat, **support_kwargs).support  # get support for the query point
            qp_single_desc_row = {}

            add_kwargs = self.set_descriptor_parameters(qp_support)

            for d, d_kwargs in zip(descriptor_list, descriptor_kwargs):
                current_desc = d(**d_kwargs)
                current_desc_kwargs = {**d_kwargs, **add_kwargs}
                qp_single_desc_row[d.__name__] = current_desc.compute(**current_desc_kwargs)

            qp_desc_values.append(qp_single_desc_row)

        self.descriptors = pd.DataFrame(qp_desc_values)

    def pca_analysis(self, variance_threshold=0.95):

        desc_df = self.descriptors.dropna()
        pca = PCA()  #
        _ = pca.fit_transform(desc_df)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        # Choose number of components to explain at least 95% variance
        n_components = np.argmax(cumulative >= variance_threshold) + 1
        # print(explained)
        print(f"Use {n_components} components to explain ≥95% variance")

        # Square of loadings → proportion of variance that each feature contributes to each PC
        components_matrix = pca.components_  # shape: (n_components, n_features)
        loadings = components_matrix[:n_components, :] ** 2

        # Sum contributions across selected components
        feature_scores = loadings.sum(axis=0)

        # Create a series with feature names
        feature_importance = pd.Series(feature_scores, index=desc_df.columns)

        # Sort if you want to see most important features
        important_features = feature_importance.sort_values(ascending=False)
        print(important_features)

        return n_components, important_features


def order_points_on_circle(points):
    """
    Orders points around a query point based on their arrangement on a circle.

    Parameters:
        points (np.ndarray): An array of shape (n, 3) representing the n points in 3D.
            Points are assumed to be shifted towards origin

    Returns:
        ordered_points (np.ndarray): Points ordered by their angular position on the circle.
    """

    # Project points onto a plane (optional if already planar)
    # Assuming points lie approximately in the xy-plane, we discard the z-coordinate.
    planar_points = points[:, :2]

    # Normalize points to unit circle
    magnitudes = np.linalg.norm(planar_points, axis=1, keepdims=True)
    normalized_points = planar_points / magnitudes

    # Compute angles with respect to the x-axis
    angles = np.arctan2(normalized_points[:, 1], normalized_points[:, 0])

    # Sort points by angle
    sorted_indices = np.argsort(angles)
    ordered_points = points[sorted_indices]

    return ordered_points
