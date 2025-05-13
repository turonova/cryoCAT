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
from sklearn.preprocessing import StandardScaler
from scipy.linalg import logm
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.spatial import Delaunay

import sklearn.neighbors as sn

from cryocat import geom
from cryocat import nnana
from cryocat import cryomotl
from cryocat.geom import Matrix
from cryocat.classutils import get_classes_from_names, get_class_names_by_parent
from cryocat import visplot
import pandas as pd
import plotly.graph_objects as go


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
                    symm=symm,
                    custom_rot=custom_rot,
                )
            )

    return particle_list


###### Parent descriptor Class ####


class Descriptor:

    def __init__(self):
        self.desc = None
        self.df = None
        self.pca_components = 1

    @staticmethod
    def remove_nans(df, axis_type="row"):

        if axis_type == "row":
            return df.dropna(axis=0)
        elif axis_type == "column":
            return df.dropna(axis=1)
        else:
            raise ValueError("axis_type must be either 'row' or 'column'")

    def get_important_features(self, pca, input_df, n_components):

        # Square of loadings → proportion of variance that each feature contributes to each PC
        components_matrix = pca.components_  # shape: (n_components, n_features)
        loadings = components_matrix[:n_components, :] ** 2

        # Sum contributions across selected components
        feature_scores = loadings.sum(axis=0)

        # Create a series with feature names
        feature_importance = pd.Series(feature_scores, index=input_df.columns)

        # Sort if you want to see most important features
        important_features = feature_importance.sort_values(ascending=False)

        return important_features

    def pca_analysis(
        self, variance_threshold=0.95, show_fig=True, nan_drop="row", scatter_kwargs=None, bar_kwargs=None
    ):

        desc_df = self.remove_nans(self.desc, axis_type=nan_drop)
        desc_df = desc_df.drop(columns=["qp_id"])
        if "nn_id" in desc_df.columns:
            desc_df = desc_df.drop(columns=["nn_id"])

        pca = PCA()
        _ = pca.fit_transform(desc_df)

        explained = pca.explained_variance_ratio_
        cumulative = np.cumsum(explained)

        # Choose number of components to explain at least 95% variance
        n_components = np.argmax(cumulative >= variance_threshold) + 1
        # print(explained)
        print(f"Use {n_components} components to explain ≥95% variance")

        important_features = self.get_important_features(pca, desc_df, n_components=n_components)
        print(important_features)

        fig = visplot.plot_pca_summary(cumulative, important_features, scatter_kwargs, bar_kwargs)

        if show_fig:
            fig.show()

        self.pca_components = n_components

        return n_components, important_features, fig

    def filter_features(self, input_df, feature_ids="all"):

        if isinstance(feature_ids, str):
            if feature_ids == "all":
                filtered_df = input_df
            elif feature_ids in input_df.columns:
                filtered_df = input_df[[feature_ids, "qp_id"]]
            else:
                raise ValueError(f"{feature_ids} is not a valid option for feature_ids parameter.")
        elif isinstance(feature_ids, list):
            features = [feature for feature in feature_ids if feature in input_df.columns]
            features.append("qp_id")
            if len(features) <= 1:
                raise ValueError("None of the provided features (columns) is in this descriptor catalogue.")
            else:
                filtered_df = input_df[features]
        else:
            raise ValueError("The feature_ids has to be a name of a computed feature, list of features or 'all'.")

        return filtered_df

    def compute_pca(self, pca_components=None, feature_ids="all", nan_drop="row"):

        pca_df = self.remove_nans(self.desc, axis_type=nan_drop)
        pca_df = self.filter_features(pca_df, feature_ids=feature_ids)

        pca_components = pca_components or self.pca_components
        pca_components = min(pca_components, len(pca_df.columns) - 1)

        qp_ids = pca_df["qp_id"].to_numpy()
        pca_df = pca_df.drop(columns=["qp_id"])

        if "nn_id" in pca_df.columns:
            pca_df = pca_df.drop(columns=["nn_id"])

        pca = PCA(n_components=pca_components)
        X_pca = pca.fit_transform(pca_df)

        important_features = self.get_important_features(pca, pca_df, n_components=pca_components)
        columns = important_features.head(pca_components).index.tolist()

        return pd.DataFrame(X_pca, columns=columns), qp_ids

    def k_means_clustering(self, n_clusters, nan_drop="row", pca_dict=None, feature_ids="all", scale_data=True):

        if pca_dict is None:
            km_data = self.remove_nans(self.desc, axis_type=nan_drop)
            km_data = self.filter_features(km_data, feature_ids=feature_ids)
            qp_ids = km_data["qp_id"].to_numpy()
            km_data = km_data.drop(columns=["qp_id"])
            if "nn_id" in km_data.columns:
                km_data = km_data.drop(columns=["nn_id"])
        else:
            km_data, qp_ids = self.compute_pca(**pca_dict, feature_ids=feature_ids, nan_drop=nan_drop)

        # Run k-means
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")

        # Scale data
        if scale_data:
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(km_data)
            clusters = kmeans.fit_predict(scaled_data)
        else:
            clusters = kmeans.fit_predict(km_data)

        # Make DataFrame
        result_df = pd.DataFrame(km_data, columns=km_data.columns)
        result_df["cluster"] = clusters
        result_df["qp_id"] = qp_ids

        return result_df

    def plot_k_means(self, color_column):

        if "nn_id" in self.df.columns and "nn_id" in self.desc.columns:
            merged_df = pd.merge(self.desc, self.df, on=["qp_id", "nn_id"], how="left")
        else:
            merged_df = pd.merge(self.desc, self.df, on="qp_id", how="left")

        fig = go.Figure(
            data=[
                go.Scatter3d(
                    x=merged_df["twist_x"],
                    y=merged_df["twist_y"],
                    z=merged_df["twist_z"],
                    mode="markers",
                    marker=dict(size=3, opacity=1, color=merged_df[color_column], colorbar=dict(title="Group")),
                )
            ]
        )

        fig.update_layout(
            title="K-means analysis",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"),
            margin=dict(l=0, r=0, b=0, t=30),
        )

        fig.show()


###### TwistDescriptor Class ######


class TwistDescriptor(Descriptor):

    def __init__(
        self, input_twist=None, input_motl=None, nn_radius=None, symm=False, remove_qp=False, remove_duplicates=False
    ):

        if input_twist is not None:
            if isinstance(input_twist, pd.DataFrame):
                self.df = input_twist
            elif isinstance(input_twist, str):
                self.df = self.read_in(input_twist)
        elif input_motl is not None and isinstance(nn_radius, (float, int)):
            self.df = TwistDescriptor.get_nn_twist_stats_within_radius(
                input_motl, nn_radius, symm, remove_qp=remove_qp, remove_duplicates=remove_duplicates
            )
        else:
            raise ValueError(
                "One has to specify (at least) either input_twist or input_motl in combination with nn_radius."
            )

        self.desc = self.df

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
        Create a TwistDescriptor object from input data

        Parameters
        ----------
        input_data : str or TwistDescriptor
            File path or TwistDescriptor object.
        """

        if isinstance(input_data, TwistDescriptor):
            return copy.deepcopy(input_data)
        elif isinstance(input_data, str):
            df = TwistDescriptor.read_in(input_data)
            return TwistDescriptor(df)
        else:
            raise ValueError("The input has to be either a TwistDescriptor object or a file path (in str format).")

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
    def process_tomo_twist(t_nn, symm=False, symm_max_value=None, symm_category=None):

        norm_coord = t_nn.get_normalized_coord()
        rotations_qp = t_nn.get_qp_rotations()
        rotations_nn = t_nn.get_nn_rotations()
        phi_qp = t_nn.df["qp_angles_phi"].to_numpy()
        phi_nn = t_nn.df["nn_angles_phi"].to_numpy()
        subtomo_qp = t_nn.df["qp_subtomo_id"].to_numpy()
        subtomo_nn = t_nn.df["nn_subtomo_id"].to_numpy()
        tomo_idx = t_nn.df["tomo_id"].to_numpy()

        if symm:
            ang_scores = geom.angular_score_for_c_symmetry(
                np.deg2rad(phi_qp), np.deg2rad(phi_nn), symm_category, symm_max_value
            )

        # Compute relative quantities
        rel_pos = rotations_qp.inv().apply(norm_coord)
        rel_quat = rotations_qp.inv() * rotations_nn

        axis_angle = rel_quat.as_rotvec()
        twists = np.hstack((axis_angle, rel_pos))

        if not symm:
            twist_df = pd.DataFrame(
                data=np.column_stack((subtomo_qp, subtomo_nn, tomo_idx, twists, phi_nn)),
                columns=["qp_id", "nn_id", "tomo_id"] + TwistDescriptor.get_mixed_feature_ids() + ["nn_inplane"],
            )

        else:
            twist_df = pd.DataFrame(
                data=np.column_stack((subtomo_qp, subtomo_nn, tomo_idx, twists, phi_nn, ang_scores)),
                columns=["qp_id", "nn_id", "tomo_id"]
                + TwistDescriptor.get_mixed_feature_ids()
                + ["nn_inplane", "angular_score"],
            )

        return twist_df

    @staticmethod
    def get_symm_parameters(input_motl, symm):
        if not symm:
            return None, None

        if isinstance(input_motl, list):
            motl = cryomotl.Motl.load(input_motl[0])
        else:
            motl = cryomotl.Motl.load(input_motl)

        pm = motl.get_motl_subset(feature_values=motl.df["subtomo_id"].iloc[0], feature_id="subtomo_id")
        part = convert_to_particle_list(pm, symm= symm)
        return part[0].max_dissimilarity(), part[0].category

    @staticmethod
    def get_nn_twist_stats_within_radius(input_motl, nn_radius, symm=False, remove_qp=None, remove_duplicates=False):

        nn = nnana.NearestNeighbors(
            input_motl,
            feature_id="tomo_id",
            nn_type="radius",
            type_param=nn_radius,
            remove_qp=remove_qp,
            remove_duplicates=remove_duplicates,
        )
        symm_max_value, symm_category = TwistDescriptor.get_symm_parameters(input_motl=input_motl, symm=symm)
        tomo_idx = nn.get_unique_values()

        results = []

        for tomo_id in tomo_idx:
            t_nn = nn.get_nn_subset(motl_values=1, feature_values=tomo_id)
            if t_nn.df.shape[0] > 0:
                results.append(TwistDescriptor.process_tomo_twist(t_nn, symm, symm_max_value, symm_category))

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

        return self.df[TwistDescriptor.get_mixed_feature_ids()].to_numpy()

    def twist_template_comparison(self, referenc_particle: Particle):
        """
        Shift twist vectors by tangent vector corresponding to input particle.
        Ideally, the input particle corresponds to the relative particle pose of interest.
        """
        template_twist = referenc_particle.tangent_at_identity()
        twist_vectors = self.get_twist_mixed_np()

        return twist_vectors - template_twist

    def sort_by_distance(self, twist_descriptor_id="geodesic_distance_rad"):

        if not (twist_descriptor_id not in self.df.columns):
            raise ValueError(f"Feature needs to be one of the following: {self.df.columns}.")

        all_distances = self.df["euclidean_distance"]

        dist_tuples = []

        for i in self.df["qp_id"].unique():

            df_qp = self.df[self.df["qp_id"] == i]
            df_qp = df_qp.sort_values(by="euclidean_distance")

            e_distances = df_qp["euclidean_distance"]
            feat_values = df_qp[twist_descriptor_id]

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

    #### Get weighted product distance for given TwistDescriptor
    def weighted_stats(self, position_weight: float, orientation_weight: float):
        weighted_data = TwistDescriptor()
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
            twist_stats (TwistDescriptor or pandas dataframe): Yields particle indices with respect to which to cluster
            num_connected_components (int, optional): Desired number of clusters. Defaults to 1.

        Raises:
            TypeError: if input data is neither a pandas data frame nor a TwistDescriptor.

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

    def get_qp_twist_desc(self, query_particle, tomo_id=None):
        """Get twist descriptor for a specific query particle from a twist dataframe.

        Parameters
        ----------
        query_particle : int, float or Particle
            The index of the particle or a Particle instance for which statistics are to be retrieved.
        tomo_id : int, optional
            The tomogram id to filter the data. If None, data will be retrieved for all tomograms associated with the
            query particle.

        Returns
        -------
        filtered_twist_desc : TwistDescriptor
            A TwistDescriptor containing the filtered statistics for the specified query particle and tomo id.

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

            return TwistDescriptor(input_twist=filtered_data)

        else:
            raise ValueError("query_particle has to be an instance of Particle or an index, both associated to p_list.")

    @classmethod
    def get_data_range(cls, twist_desc, twist_descriptor_id=None, min_value=None, max_value=None):
        """
        Filter a TwistDescriptor instance by a specific column and return a new instance.

        Parameters
        ----------
        twist_desc : TwistDescriptor
            The original TwistDescriptor instance.
        twist_descriptor_id : str, optional
            The name of the column to filter on. If None, the original twist_desc will be returned.
        min_value : float, optional
            Minimum value for filtering. Default is None.
        max_value : float, optional
            Maximum value for filtering. Default is None.

        Returns
        -------
        TwistDescriptor
            A new instance of TwistDescriptor with filtered data.
        """
        # Return unmodified if no filtering is requested
        if twist_descriptor_id is None or (min_value is None and max_value is None):
            return twist_desc

        df = twist_desc.df.copy()

        if min_value is not None:
            df = df[df[twist_descriptor_id] >= min_value]
        if max_value is not None:
            df = df[df[twist_descriptor_id] <= max_value]

        return TwistDescriptor(df)

    def get_twist_pos_df(self):
        return self.df[TwistDescriptor.get_pos_feature_ids()]

    def get_twist_rot_df(self):
        return self.df[TwistDescriptor.get_rot_feature_ids()]

    def get_twist_mixed_df(self):
        return self.df[TwistDescriptor.get_mixed_feature_ids()]

    def get_twist_pos_np(self):
        return self.df[TwistDescriptor.get_pos_feature_ids()].to_numpy()

    def get_twist_rot_np(self):
        return self.df[TwistDescriptor.get_rot_feature_ids()].to_numpy()


class Filter:

    def __init__(self):
        self.filter = None


class AxisRot(Filter):

    def __init__(self, twist_desc, max_angle, axis="z", min_angle=0.0):

        feature_id = TwistDescriptor.get_axis_feature_id("rot_angle", axis=axis)
        self.filter = TwistDescriptor.get_data_range(
            twist_desc=twist_desc, twist_descriptor_id=feature_id, min_value=min_angle, max_value=max_angle
        )


class EuclideanDistNN(Filter):

    def __init__(self, twist_desc, num_neighbors=1):
        twist_df = twist_desc.df.sort_values(by=["qp_id", "euclidean_distance"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistDescriptor(twist_df)


class GeodesicDistNN(Filter):

    def __init__(self, twist_desc, num_neighbors=1):
        twist_df = twist_desc.df.sort_values(by=["qp_id", "geodesic_distance_rad"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistDescriptor(twist_df)


class MixedDistNN(Filter):

    def __init__(self, twist_desc, num_neighbors=1):
        twist_df = twist_desc.df.sort_values(by=["qp_id", "product_distance"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistDescriptor(twist_df)


class AngularScoreNN(Filter):

    def __init__(self, twist_desc, num_neighbors=1):
        twist_df = twist_desc.df.sort_values(by=["qp_id", "angular_score"]).groupby("qp_id").head(num_neighbors)
        self.filter = TwistDescriptor(twist_df)


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
                    columns = TwistDescriptor.get_pos_feature_ids()
                else:
                    columns = TwistDescriptor.get_rot_feature_ids()

            else:

                columns = TwistDescriptor.get_mixed_feature_ids()

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

    def __init__(self, twist_desc, radius=None):
        if radius is None:
            self.support = twist_desc
        else:
            self.support = TwistDescriptor.get_data_range(
                twist_desc=twist_desc, twist_descriptor_id="euclidean_distance", min_value=0.0, max_value=radius
            )


class Shell(Support):

    def __init__(self, twist_desc, radius_min, radius_max):
        self.support = TwistDescriptor.get_data_range(
            twist_desc=twist_desc, twist_descriptor_id="euclidean_distance", min_value=radius_min, max_value=radius_max
        )


class Cone(Support):

    def __init__(self, twist_desc: TwistDescriptor, cone_height: float, cone_radius: float, axis=None, mode="position"):

        axis, columns = Support.set_axis_and_columns(mode=mode, axis=axis)

        tangent = twist_desc.df[columns].to_numpy()

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

        self.support = TwistDescriptor(twist_desc.df[mask])


class Torus(Support):

    def __init__(
        self,
        twist_desc: TwistDescriptor,
        inner_radius: float,
        outer_radius: float,
        axis=None,
        mode="position",
    ):

        axis, columns = Support.set_axis_and_columns(mode=mode, axis=axis)

        tangent = twist_desc[columns].to_numpy()

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

        self.support = TwistDescriptor(twist_desc[mask])


class Cylinder(Support):
    def __init__(
        self, twist_desc: TwistDescriptor, radius: float, height: float, axis=None, mode="position", symmetric=True
    ):

        def compute_one_direction(axis):
            tangent = twist_desc.df[columns].to_numpy()

            axis_projection = np.dot(tangent, axis)[:, None] * axis
            proj_dist = np.linalg.norm(axis_projection - tangent, axis=1)

            axis_portion = np.dot(axis_projection, axis)
            height_test = (axis_portion <= height) & (0 <= axis_portion)
            rad_test = proj_dist <= radius

            # Mask: rows where all conditions hold
            mask = height_test & rad_test

            return twist_desc.df[mask].copy()

        set_axis, columns = Support.set_axis_and_columns(mode=mode, axis=axis)
        truncated_stats_df = compute_one_direction(set_axis)

        if symmetric:
            set_axis, columns = Support.set_axis_and_columns(mode=mode, axis=-set_axis)
            truncated_stats_df = pd.concat([truncated_stats_df, compute_one_direction(set_axis)])

        self.support = TwistDescriptor(truncated_stats_df)


class Feature:

    def __init__(self, desc_df):
        self.df = desc_df

    def compute(self, **kwargs):
        pass


class NNCount(Feature):

    def __init__(self, twist_df, **kwargs):
        super().__init__(twist_df)

    def compute(self, **kwargs):

        results = []
        for qp_id, group_df in self.df.groupby("qp_id"):
            results.append([qp_id, group_df.shape[0]])

        columns = ["qp_id", self.__class__.__name__]
        return pd.DataFrame(data=np.array(results), columns=columns)


class CountSHOT(Feature):

    def __init__(self, shot_desc, **kwargs):
        super().__init__(shot_desc.shot_df)
        self.cone_number = shot_desc.cone_number
        self.shell_number = shot_desc.shell_number

    def compute(self):

        # Count combinations of (qp_id, cone_id, shell_id)
        counts = self.df.groupby(["qp_id", "cone_id", "shell_id"]).size().reset_index(name="count")

        # Create unique column label for each (cone_id, shell_id)
        counts["cone_shell"] = counts.apply(
            lambda row: f"{self.__class__.__name__}_c{int(row['cone_id'])}_s{int(row['shell_id'])}", axis=1
        )

        # Pivot table: one row per qp_id, columns are cone-shell combos
        pivot = counts.pivot_table(index="qp_id", columns="cone_shell", values="count", fill_value=0)

        # Define all expected column names
        all_labels = [
            f"{self.__class__.__name__}_c{c}_s{s}" for c in range(self.cone_number) for s in range(self.shell_number)
        ]

        # Reindex to include all cone-shell combinations
        pivot = pivot.reindex(columns=all_labels, fill_value=0)

        return pivot.reset_index()


class EulerCharAlpha(Feature):
    """
    Computes the Euler characteristic of a 1-dimensional simplical complex, e.g. for the
    link of a vertex in a 2-dimensional triangulated surface.
    """

    def __init__(self, alpha_desc, **kwargs):
        super().__init__(alpha_desc.alpha_df)
        self.link_edges = alpha_desc.link_edges

    def compute(self, **kwargs):

        results = []
        for qp_id, _ in self.df.groupby("qp_id"):
            vertices = set(itertools.chain.from_iterable(self.link_edges[qp_id]))
            results.append([qp_id, len(vertices) - len(self.link_edges[qp_id])])

        columns = ["qp_id", self.__class__.__name__]
        return pd.DataFrame(data=np.array(results), columns=columns)


class LinkTypeAlpha(Feature):
    """
    Computes a simplicial isomorphism invariant for a 1-dimensional simplicial complex,
    e.g. for tha link of a vertex in a 2-dimensional triangulated surface.
    """

    def __init__(self, alpha_desc, **kwargs):
        super().__init__(alpha_desc.alpha_df)
        self.link_edges = alpha_desc.link_edges

    def compute(self, **kwargs):

        results = []
        for qp_id, _ in self.df.groupby("qp_id"):
            vertices = set(itertools.chain.from_iterable(self.link_edges[qp_id]))
            results.append([qp_id, 1 - 0.5 * len(vertices) + len(self.link_edges[qp_id]) / 3])

        columns = ["qp_id", self.__class__.__name__]
        return pd.DataFrame(data=np.array(results), columns=columns)


class CentralAngleStatsAlpha(Feature):
    """
    Compute median of angles incident to a given vertex, as indicated by
    edges making up the given vertex's link.
    """

    def __init__(self, alpha_desc, **kwargs):
        super().__init__(alpha_desc.alpha_df)

    def compute(self, **kwargs):

        all_stats = []
        for qp_id, group_df in self.df.groupby("qp_id"):
            angles = group_df["central_angle"].values

            if not np.isnan(angles).any():
                all_stats.append([qp_id, np.mean(angles), np.median(np.sort(angles)), np.std(angles), np.var(angles)])
            else:
                all_stats.append([qp_id, np.NAN, np.NAN, np.NAN, np.NAN])

        columns = ["Mean", "Median", "Std", "Var"]
        columns = [f"{self.__class__.__name__}{col}" for col in columns]
        columns.insert(0, "qp_id")
        return pd.DataFrame(data=np.array(all_stats), columns=columns)


class PLComplexDescriptor(Descriptor):

    def __init__(self, twist_df):
        self.ordered_vertices = {}
        self.ordered_nn = {}
        self.triangles = {}
        self.pl_df = pd.DataFrame()

        # Group input_twist by 'qp_id'
        for qp_id, group in twist_df.groupby("qp_id"):

            self.ordered_vertices[qp_id], sorted_indices = geom.order_points_on_circle(
                group[["twist_x", "twist_y", "twist_z"]].to_numpy()
            )
            self.ordered_nn[qp_id] = group["nn_id"].to_numpy()[sorted_indices]
            self.triangles[qp_id] = self.compute_triangles(qp_id)
            if self.triangles[qp_id] is not None:
                self.pl_df = pd.concat([self.pl_df, self.compute_features(qp_id)])

        self.desc = self.pl_df
        self.df = twist_df

    def compute_triangles(self, qp_id):

        if self.ordered_vertices[qp_id].shape[0] < 2:
            return None

        center = np.array([0, 0, 0])
        triangles = []
        for i in range(self.ordered_vertices[qp_id].shape[0] - 1):
            triangles.append(
                geom.Triangle(center, self.ordered_vertices[qp_id][i, :], self.ordered_vertices[qp_id][i + 1, :])
            )

        triangles.append(geom.Triangle(center, self.ordered_vertices[qp_id][-1, :], self.ordered_vertices[qp_id][0]))

        return triangles

    def compute_features(self, qp_id):

        central_angles = []
        internal_angles = []
        in_radii = []
        circle_radii = []
        areas = []
        _, _, internal_part1 = self.triangles[qp_id][-1].inner_angles()
        for t in self.triangles[qp_id]:
            central_angle, internal_part2, next_part1 = t.inner_angles()
            internal_angles.append(internal_part1 + internal_part2)
            central_angles.append(central_angle)
            internal_part1 = next_part1
            _, in_radius = t.inscribed_circle()
            _, circle_radius = t.circumcircle()
            in_radii.append(in_radius)
            circle_radii.append(circle_radius)
            areas.append(t.area())

        f_df = pd.DataFrame()
        f_df["central_angle"] = central_angles
        # f_df["internal_angles"] = internal_angles # complicated for sparse data, not really defined
        f_df["area"] = areas
        f_df["in_radius"] = in_radii
        f_df["circle_radius"] = circle_radii
        f_df["radii_ratio"] = f_df["circle_radius"].values / f_df["in_radius"].values
        f_df["qp_id"] = qp_id
        f_df["nn_id"] = self.ordered_nn[qp_id]

        return f_df

    def compute_outer_edges(self):
        edges = [
            (self.ordered_vertices[i], self.ordered_vertices[i + 1]) for i in range(len(self.ordered_vertices) - 1)
        ]
        edges.append((self.ordered_vertices[-1], self.ordered_vertices[0]))

        return edges


class SHOTDescriptor(Descriptor):

    def __init__(self, twist_df, cone_number=6, shell_number=1, north_pole_axis=None):

        max_radius = twist_df["euclidean_distance"].max()
        self.cone_number = cone_number
        self.shell_number = shell_number

        self.shot_df = pd.DataFrame()
        # Group input_twist by 'qp_id'
        for qp_id, group_df in twist_df.groupby("qp_id"):
            coord = group_df[["twist_x", "twist_y", "twist_z"]].to_numpy()
            assigned_df = self.assign_shell_and_cone_ids(
                qp_id, coord, shell_number, cone_number, radius=max_radius, north_pole_axis=north_pole_axis
            )
            assigned_df["qp_id"] = qp_id
            assigned_df["nn_id"] = group_df["nn_id"].values
            self.shot_df = pd.concat([self.shot_df, assigned_df])

        self.desc = self.shot_df
        self.df = twist_df

    def generate_rotated_axes(self, num_cones, north_pole_axis=None):

        base_vectors = self.fixed_cone_directions(num_cones)
        if north_pole_axis is None:
            return base_vectors
        else:
            new_north = np.asarray(north_pole_axis) / np.linalg.norm(north_pole_axis)
            old_north = np.array([0, 0, 1])
            axis = np.cross(old_north, new_north)
            angle = np.arccos(np.clip(np.dot(old_north, new_north), -1.0, 1.0))
            if np.linalg.norm(axis) < 1e-8:
                rot = R.identity() if angle < 1e-8 else R.from_rotvec([0, 0, np.pi])
            else:
                axis = axis / np.linalg.norm(axis)
                rot = R.from_rotvec(axis * angle)

            rotated_vectors = rot.apply(base_vectors)
            return rotated_vectors

    def fixed_cone_directions(self, num_cones):
        """
        Return evenly distributed unit vectors for small num_cones (hand-picked for symmetry).
        These directions include one at [0, 0, 1] and others at standard axes for low counts.
        """
        if num_cones == 2:
            return np.array([[0, 0, 1], [0, 0, -1]])
        elif num_cones == 4:
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1]])
        elif num_cones == 6:
            return np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 0, -1], [0, -1, 0], [-1, 0, 0]])
        elif num_cones == 18:
            return np.array(
                [
                    [0.0, 0.0, 1.0],
                    [-0.707107, -0.707107, 0.0],
                    [-0.707107, 0.0, -0.707107],
                    [-0.707107, 0.0, 0.707107],
                    [-0.707107, 0.707107, 0.0],
                    [0.0, -1.0, 0.0],
                    [0.0, -0.707107, -0.707107],
                    [0.0, -0.707107, 0.707107],
                    [0.0, 0.0, -1.0],
                    [-1.0, 0.0, 0.0],
                    [0.0, 0.707107, -0.707107],
                    [0.0, 0.707107, 0.707107],
                    [0.0, 1.0, 0.0],
                    [0.707107, -0.707107, 0.0],
                    [0.707107, 0.0, -0.707107],
                    [0.707107, 0.0, 0.707107],
                    [0.707107, 0.707107, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            )
        else:
            raise ValueError(f"No fixed directions for {num_cones} cones. Please define them.")

    def assign_cone_ids_by_dot(points, cone_dirs):
        """
        Assign each point to the cone direction with which it has the largest cosine similarity.
        All inputs must be normalized.
        """
        # Normalize points
        points = points / np.linalg.norm(points, axis=1, keepdims=True)

        # Ensure cone_dirs are normalized too
        cone_dirs = cone_dirs / np.linalg.norm(cone_dirs, axis=1, keepdims=True)

        # Compute cosine similarity: (N_points x N_cones)
        cos_sim = points @ cone_dirs.T  # dot product

        # For each point, pick the cone with highest dot product
        cone_id = np.argmax(cos_sim, axis=1)

        return cone_id

    def assign_shell_and_cone_ids(self, qp_id, points, num_shells, num_cones, radius=1.0, north_pole_axis=None):
        points = np.atleast_2d(np.asarray(points))
        r_norms = np.linalg.norm(points, axis=1)

        # Assign shell_id by radial distance
        shell_edges = np.linspace(0, radius, num_shells + 1)
        shell_id = np.digitize(r_norms, shell_edges) - 1

        # Get predefined cone directions
        cone_dirs = self.generate_rotated_axes(num_cones, north_pole_axis=north_pole_axis)

        # Normalize points and cone directions
        normed_points = points / np.linalg.norm(points, axis=1, keepdims=True)
        normed_cones = cone_dirs / np.linalg.norm(cone_dirs, axis=1, keepdims=True)

        # Compute cosine similarity (dot product)
        cos_sim = normed_points @ normed_cones.T  # shape: (N_points, N_cones)

        # Choose cone with maximum cosine similarity (i.e., smallest angle)
        cone_id = np.argmax(cos_sim, axis=1)

        # Assemble dataframe
        df = pd.DataFrame()
        df["shell_id"] = shell_id
        df["cone_id"] = cone_id
        df["qp_id"] = qp_id

        return df


class AlphaComplexDescriptor(Descriptor):

    def __init__(self, twist_df, alpha_param=200):
        self.alpha_param = alpha_param
        self.alpha_complexes = {}
        self.stars = {}
        self.triangles = {}
        self.link_edges = {}
        self.alpha_df = pd.DataFrame()

        # Group input_twist by 'qp_id'
        for qp_id, group in twist_df.groupby("qp_id"):
            coord = group[["twist_x", "twist_y", "twist_z"]].to_numpy()
            if coord.shape[0] < 2:
                continue

            self.triangles[qp_id], self.link_edges[qp_id] = self.compute_alpha_complex(coord)

            if self.triangles[qp_id] is not None:
                self.alpha_df = pd.concat([self.alpha_df, self.compute_features(qp_id)])

        self.desc = self.alpha_df
        self.df = twist_df

    def compute_alpha_complex(self, coord):

        vertices = coord[:, :2]
        vertices = np.vstack((np.zeros(2), vertices))

        tri = Delaunay(vertices)

        star_triangles = tri.simplices[np.any(tri.simplices == 0, axis=1)]  # remove those without the qp

        if len(star_triangles) == 0:
            return None, None

        star_triangles = np.sort(star_triangles, axis=1)  # qp with idx 0 is always first

        triangles = []
        link_edges = []

        for a, b, c in star_triangles:
            t = geom.Triangle(np.zeros(3), coord[b - 1, :], coord[c - 1, :])
            circum_radius = t.circumcircle_radius()
            if (circum_radius**2) < self.alpha_param:
                triangles.append(t)
                link_edges.append([b - 1, c - 1])

        if len(star_triangles) == 0:
            return None, None
        else:
            return triangles, link_edges

    def compute_features(self, qp_id):

        central_angles = []
        in_radii = []
        circle_radii = []
        areas = []
        for t in self.triangles[qp_id]:
            central_angle, _, _ = t.inner_angles()
            _, in_radius = t.inscribed_circle()

            central_angles.append(central_angle)
            in_radii.append(in_radius)
            circle_radii.append(t.circumcircle_radius())
            areas.append(t.area())

        f_df = pd.DataFrame()
        f_df["central_angle"] = central_angles
        f_df["area"] = areas
        f_df["in_radius"] = in_radii
        f_df["circle_radius"] = circle_radii
        f_df["radii_ratio"] = f_df["circle_radius"].values / f_df["in_radius"].values
        f_df["qp_id"] = qp_id

        return f_df


class Catalog:

    def __init__(self):
        self.module_name = None
        self.parent_class_name = None

    def get_all_supports(self, filter_contains=None, filter_exclude=None):
        return get_class_names_by_parent(self.parent_class_name, self.module_name, filter_contains, filter_exclude)


class SupportCatalog(Catalog):

    def __init__(self):
        self.module_name = "cryocat.tango"
        self.parent_class_name = "Support"


class FeatureCatalog(Catalog):

    def __init__(self):
        self.module_name = "cryocat.tango"
        self.parent_class_name = "Feature"


class FilterCatalog(Catalog):

    def __init__(self):
        self.module_name = "cryocat.tango"
        self.parent_class_name = "Filter"


class CustomDescriptor(Descriptor):

    def __init__(self, input_twist):
        self.desc = None
        self.df = input_twist

    @classmethod
    def load(cls, desc_df):
        dc = CustomDescriptor()
        dc.desc = desc_df
        return dc

    def create_additional_descriptors(self, support, feature_list, feature_kwargs):

        add_kwargs = {"twist_df": support}
        alpha_computed = False
        pl_computed = False
        shot_computed = False
        for f, f_kwargs in zip(feature_list, feature_kwargs):
            if f.__name__.endswith("Alpha") and not alpha_computed:
                add_kwargs["alpha_desc"] = AlphaComplexDescriptor(support, **f_kwargs)
                alpha_computed = True
            elif f.__name__.endswith("PL") and not pl_computed:
                add_kwargs["pl_desc"] = PLComplexDescriptor(support, **f_kwargs)
                pl_computed = True
            elif f.__name__.endswith("SHOT") and not shot_computed:
                add_kwargs["shot_desc"] = SHOTDescriptor(support, **f_kwargs)
                shot_computed = True
            if alpha_computed and pl_computed and shot_computed:
                break

        return add_kwargs

    def build_descriptor(self, feature_list, feature_kwargs, support_class, support_kwargs):

        feature_list = get_classes_from_names(feature_list, "cryocat.tango")
        support_class = get_classes_from_names(support_class, "cryocat.tango")

        support = support_class(self.df, **support_kwargs).support

        if not feature_kwargs:
            feature_kwargs = [{} for _ in range(len(feature_list))]

        add_kwargs = self.create_additional_descriptors(support, feature_list, feature_kwargs)

        self.desc = pd.DataFrame(self.df["qp_id"].unique(), columns=["qp_id"])

        for f, f_kwargs in zip(feature_list, feature_kwargs):
            current_feat_kwargs = {**f_kwargs, **add_kwargs}
            current_feature = f(**current_feat_kwargs)
            self.desc = pd.merge(self.desc, current_feature.compute(**current_feat_kwargs), on="qp_id", how="left")
            # self.desc = pd.concat([, ], axis=1)
