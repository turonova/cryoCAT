import numpy as np
import pandas as pd
from cryocat import cryomotl
from cryocat import cryomap
from scipy.spatial.transform import Rotation as srot
from cryocat import geom
import seaborn as sns
import sklearn.neighbors as sn
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots
import plotly.graph_objects as go


class NearestNeighbors:

    def __init__(
        self,
        input_data=None,
        feature_id="tomo_id",
        nn_type="closest_dist",
        type_param=None,
        remove_qp=None,
        remove_duplicates=False,
    ):

        if input_data is None:
            self.features = None
            self.df = None
            return

        motl_list = []
        single_motl = False
        if not isinstance(input_data, list):
            motl_list.append(cryomotl.Motl.load(input_data))
            motl_list.append(cryomotl.Motl.load(input_data))
            single_motl = True
        else:
            for m in input_data:
                motl_list.append(cryomotl.Motl.load(m))

        single_motl = remove_qp or single_motl  # remove_qp can override the default

        features = motl_list[0].get_unique_values(feature_id)
        for m in motl_list[1:]:
            features = np.intersect1d(features, m.get_unique_values(feature_id), assume_unique=True)

        results = []
        columns = [
            "motl_id",
            feature_id,
            "qp_id",
            "qp_subtomo_id",
            "nn_id",
            "nn_subtomo_id",
            "qp_angles_phi",
            "qp_angles_theta",
            "qp_angles_psi",
            "qp_coord_x",
            "qp_coord_y",
            "qp_coord_z",
            "nn_angles_phi",
            "nn_angles_theta",
            "nn_angles_psi",
            "nn_coord_x",
            "nn_coord_y",
            "nn_coord_z",
        ]

        if nn_type == "closest_dist":
            columns.append("nn_dist")

        for f in features:
            fm_qp = motl_list[0].get_motl_subset(feature_values=f, feature_id=feature_id)
            qp_subtomos = fm_qp.df["subtomo_id"].values
            qp_angles = fm_qp.get_angles()
            qp_coord = fm_qp.get_coordinates()

            for i, m in enumerate(motl_list[1:], start=1):
                fm_nn = m.get_motl_subset(feature_values=f, feature_id=feature_id)
                nn_subtomos = fm_nn.df["subtomo_id"].values
                nn_angles = fm_nn.get_angles()
                nn_coord = fm_nn.get_coordinates()

                def stack_nn_results(qp_idx, nn_idx):
                    counts = np.array([len(n) for n in nn_idx])
                    if counts.sum() == 0:
                        return None

                    nn_idx_flat = np.concatenate(nn_idx)

                    qp_idx_expanded = np.repeat(qp_idx, counts)

                    qp_angles_expanded = qp_angles[qp_idx_expanded]
                    qp_coord_expanded = qp_coord[qp_idx_expanded]
                    nn_angles_expanded = nn_angles[nn_idx_flat]
                    nn_coord_expanded = nn_coord[nn_idx_flat]

                    stacked = np.column_stack(
                        [
                            np.repeat(i, counts.sum()),
                            np.repeat(f, counts.sum()),
                            qp_idx_expanded,
                            qp_subtomos[qp_idx_expanded],
                            nn_idx_flat,
                            nn_subtomos[nn_idx_flat],
                            qp_angles_expanded,
                            qp_coord_expanded,
                            nn_angles_expanded,
                            nn_coord_expanded,
                        ]
                    )
                    return stacked

                if nn_type == "closest_dist":
                    nn_count = type_param or 1
                    qp_idx, nn_idx, nn_dist, _ = get_feature_nn_indices(
                        fm_qp, fm_nn, nn_number=nn_count, remove_qp=single_motl
                    )

                    stacked = stack_nn_results(qp_idx, nn_idx)

                    if stacked is None:
                        continue

                    stacked = np.column_stack((stacked, np.concatenate(nn_dist)))
                    results.append(stacked)

                else:
                    radius = type_param or 1
                    qp_idx, nn_idx = get_feature_nn_within_radius(fm_qp, fm_nn, radius=radius, remove_qp=single_motl)
                    stacked = stack_nn_results(qp_idx, nn_idx)
                    if stacked is None:
                        continue
                    results.append(stacked)

        final_array = np.vstack(results)

        self.df = pd.DataFrame(data=final_array, columns=columns)

        if remove_duplicates:
            self.df = self.drop_symmetric_duplicates()

        # self.motls = motl_list
        self.features = features
        self.feature_id = feature_id
        # self.feature_id = feature_id

    def drop_symmetric_duplicates(self):

        qp_col = "qp_subtomo_id"
        nn_col = "nn_subtomo_id"
        # Create a tuple (min_id, max_id) so (1,2) and (2,1) become the same
        pairs = self.df[[qp_col, nn_col]].apply(lambda row: tuple(sorted(row)), axis=1)

        # Add to a temp column
        self.df = self.df.copy()
        self.df["_pair_key"] = pairs

        # Drop duplicates based on the sorted pair
        df_unique = self.df.drop_duplicates(subset="_pair_key")

        # Remove temp column
        df_unique = df_unique.drop(columns="_pair_key")

        return df_unique

    def get_unique_values(self):
        return self.features

    def get_nn_subset(self, motl_values, feature_values):
        nn_subset = NearestNeighbors()
        if not isinstance(motl_values, list):
            motl_values = [motl_values]
        if not isinstance(feature_values, list):
            feature_values = [feature_values]

        nn_subset.df = self.df[
            (self.df["motl_id"].isin(motl_values)) & (self.df[self.feature_id].isin(feature_values))
        ].copy()
        nn_subset.features = feature_values

        return nn_subset

    def get_normalized_coord(self):

        norm_coord = (
            self.df[["nn_coord_x", "nn_coord_y", "nn_coord_z"]].to_numpy()
            - self.df[["qp_coord_x", "qp_coord_y", "qp_coord_z"]].to_numpy()
        )
        return norm_coord

    def get_qp_rotations(self):
        angles = self.df[["qp_angles_phi", "qp_angles_theta", "qp_angles_psi"]].to_numpy()
        return srot.from_euler("zxz", degrees=True, angles=angles)

    def get_nn_rotations(self):
        angles = self.df[["nn_angles_phi", "nn_angles_theta", "nn_angles_psi"]].to_numpy()
        return srot.from_euler("zxz", degrees=True, angles=angles)


def get_feature_nn_within_radius(fm_a, fm_nn, radius, remove_qp=False):
    """Get nearest neighbors within a specified radius from a set of coordinates which are feature specific.

    Parameters
    ----------
    fm_a : cryomotl.Motl
        A motl for which nearest neighbors are to be found.
    fm_nn : cryomotl.Motl
        A motl in which the nearest neighbors will be searched for.
    radius : float
        The distance within which to search for nearest neighbors.
    remove_qp : bool, default=False
        If True, the query point will be removed from nn_indices. Default is False.

    Returns
    -------
    qp_indices : numpy.ndarray
        An array of indices representing the query particles for which neighbors are found.
    nn_indices : list of list of int
        A list where each element is a list of indices of the nearest neighbors corresponding to each center index.

    Notes
    -----
    This function uses a KDTree for efficient spatial querying of nearest neighbors. If `unique_only` is set to True,
    it ensures that each neighbor is unique by removing duplicates.
    """

    coord_qp = fm_a.get_coordinates()
    coord_nn = fm_nn.get_coordinates()
    kdt_nn = sn.KDTree(coord_nn)
    nn_idx = kdt_nn.query_radius(coord_qp, radius)

    qp_indices = []
    nn_indices = []

    for i, neighbors in enumerate(nn_idx):
        # Optionally remove self from neighbors
        if remove_qp:
            neighbors = neighbors[neighbors != i]

        if len(neighbors) > 0:
            qp_indices.append(i)
            nn_indices.append(np.sort(neighbors))

    return qp_indices, nn_indices


def get_nn_within_distance(feature_motl, radius, unique_only=True):
    """Get nearest neighbors within a specified distance from a set of coordinates.

    Parameters
    ----------
    feature_motl : cryomotl.Motl
        A motl for which each particle should be analyzed.
    radius : float
        The distance within which to search for nearest neighbors.
    unique_only : bool, default=True
        If True, only unique nearest neighbors are returned. Default is True.

    Returns
    -------
    center_idx : numpy.ndarray
        An array of indices representing the center particles for which neighbors are found.
    nn_idx : list of list of int
        A list where each element is a list of indices of the nearest neighbors corresponding to each center index.

    Notes
    -----
    This function uses a KDTree for efficient spatial querying of nearest neighbors. If `unique_only` is set to True,
    it ensures that each neighbor is unique by removing duplicates.
    """

    def remove_duplicates(list_of_arrays):
        unique_arrays = []
        seen_tuples = set()

        for la in list_of_arrays:
            array_tuple = tuple(la)
            if array_tuple not in seen_tuples:
                unique_arrays.append(la)
                seen_tuples.add(array_tuple)

        return unique_arrays

    coord = feature_motl.get_coordinates()
    kdt_nn = sn.KDTree(coord)
    nn_idx = kdt_nn.query_radius(coord, radius)

    # create id array to check where the NN was the same particle
    ordered_idx = np.arange(0, nn_idx.shape[0], 1)

    # each particle is also part of the results -> keep only those that have more than 1 particle
    ordered_idx = [i for i, row in zip(ordered_idx, nn_idx) if len(row) > 1]
    nn_indices = nn_idx[ordered_idx]

    if unique_only:
        # remove duplicates: first sort in each row, then find the unique idx
        sorted_idx = [np.sort(row) for row in nn_indices]
        nn_indices = remove_duplicates(sorted_idx)
        center_idx = np.array([row[0] for row in nn_indices])
        nn_idx = [row[1:] for row in nn_indices]
    else:
        center_idx = ordered_idx
        nn_idx = [[elem for elem in row if elem != center_idx[i]] for i, row in enumerate(nn_indices)]

    return center_idx, nn_idx


def get_feature_nn_indices(fm_a, fm_nn, nn_number=1, remove_qp=False):
    """Get the indices and distances of nearest neighbors for given feature coordinates.

    Parameters
    ----------
    fm_a : cryomotl.Motl
        A motl for which nearest neighbors are to be found.
    fm_nn : cryomotl.Motl
        A motl in which the nearest neighbors will be searched for.
    nn_number : int, default=1
        The number of nearest neighbors to retrieve for each feature. Default is 1.

    Returns
    -------
    ordered_idx : ndarray
        An array of indices corresponding to the ordered features in `fm_a`.
    nn_idx : ndarray
        A 2D array of shape (n_features, nn_count) containing the indices of the nearest neighbors for each feature
        in `fm_a`.
    nn_dist : ndarray
        A 2D array of shape (n_features, nn_count) containing the distances to the nearest neighbors for each feature
        in `fm_a`.
    nn_count : int
        The actual number of nearest neighbors retrieved, which is the minimum of `nn_number` and the number of
        available neighbors.
    remove_qp: bool, default=False
        Whether to remove nn correspodning to the query point. Should be set to True, if the function is called on the same
        motl.

    Notes
    -----
    This function uses a KDTree for efficient nearest neighbor search.
    """

    coord_a = fm_a.get_coordinates()
    coord_nn = fm_nn.get_coordinates()

    if remove_qp:
        nn_number = nn_number + 1  # increase number of nns, so one can delete the query point.

    nn_count = min(nn_number, coord_nn.shape[0])
    kdt_nn = sn.KDTree(coord_nn)
    nn_dist, nn_idx = kdt_nn.query(coord_a, k=nn_count)
    ordered_idx = np.arange(0, nn_idx.shape[0], 1)

    return (
        ordered_idx,
        nn_idx.reshape((nn_idx.shape[0], nn_count)),
        nn_dist.reshape((nn_idx.shape[0], nn_count)),
        nn_count,
    )


def get_nn_stats(motl_a, motl_nn, pixel_size=1.0, feature_id="tomo_id", nn_number=1, rotation_type="angular_distance"):
    """For each particle in motl_a, this function computes nn_number nearest neighbors in motl_nn and returns the
    associated data: distance of neighbor to query point, coordinates of nearest neighbors, coordinates of nearest neighbors
    after being rotated with respect to the coordinate frame of the query point, angular distance between query point and
    nearest neighbor, representations of associated rotation via rotated unit vector + Euler angles, subtomogram-id of query point
    of its associated nearest neighbors.

    Parameters
    ----------
    motl_a : cryocat.cryomotl.Motl or str
        Input particle list of query points.
    motl_nn : cryocat.cryomotl.Motl or str
        Input particle list of with nearest neighbors of interest.
    pixel_size : float, default=1.0
        Pixel size. Defaults to 1.0.
    feature_id : str, default='tomo_id'
        Particle list feature to distinguish between subsets of input motls. Defaults to "tomo_id".
    nn_number : int, default=1
        Number of requested nearest neighbors in motl_nn for each particle in motl_a. Defaults to 1.
    rotation_type : str, default='angular_distance'
        For comparison of rotations. Choice between "all", "angular_distance",
        "cone_distance", and "in_plane_distance". Defaults to "angular_distance".

    Returns
    -------
    pandas dataframe
        Contains statistics of nearest neighbors analysis between input particle lists.
    """
    (
        centered_coord,
        rotated_coord,
        nn_dist,
        ang_dst,
        subtomo_idx,
        subtomo_idx_nn,
    ) = get_nn_distances(
        motl_a, motl_nn, nn_number=nn_number, pixel_size=pixel_size, feature=feature_id, rotation_type=rotation_type
    )

    coord_rot, angles = get_nn_rotations(motl_a, motl_nn, feature=feature_id, nn_number=nn_number)

    nn_stats = pd.DataFrame(
        np.hstack(
            (
                nn_dist.reshape((nn_dist.shape[0], 1)),
                centered_coord,
                rotated_coord,
                ang_dst.reshape((nn_dist.shape[0], 1)),
                coord_rot,
                angles,
                subtomo_idx.reshape((nn_dist.shape[0], 1)),
                subtomo_idx_nn.reshape((nn_dist.shape[0], 1)),
            )
        ),
        columns=[
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
            "subtomo_idx",
            "subtomo_nn_idx",
        ],
    )

    nn_stats["type"] = "nn"

    return nn_stats


def get_nn_distances(motl_a, motl_nn, pixel_size=1.0, nn_number=1, feature="tomo_id", rotation_type="angular_distance"):
    """Get nearest neighbor distances and related information between two sets of particles.

    Parameters
    ----------
    motl_a : str or Motl
        Path to the first motl file or a Motl object containing the first set of particles.
    motl_nn : str or Motl
        Path to the second motl file or a Motl object containing the second set of particles.
    pixel_size : float, default=1.0
        The size of a pixel in the same units as the coordinates. Default is 1.0.
    nn_number : int, default=1
        The number of nearest neighbors to consider. Default is 1.
    feature : str, default='tomo_id'
        The feature to use for splitting the particles. Default is 'tomo_id'.
    rotation_type : str, default='angular_distance'
        The type of rotation distance to compute. Default is 'angular_distance'.

    Returns
    -------
    centered_coord : np.ndarray
        The coordinates of the nearest neighbors centered around the reference particles.
    rotated_coord : np.ndarray
        The coordinates of the nearest neighbors after applying the rotation.
    nn_dist : np.ndarray
        The distances to the nearest neighbors.
    angular_distances : np.ndarray
        The angular distances between the reference particles and their nearest neighbors.
    subtomo_idx : np.ndarray
        The subtomo IDs of the reference motifs.
    subtomo_idx_nn : np.ndarray
        The subtomo IDs of the nearest neighbors.

    Notes
    -----
    This function assumes that the input motifs have angle information and that the
    motl files are compatible with the Motl class. The function will only work with
    the intersection of features present in both motls.
    """

    if isinstance(motl_a, str):
        motl_a = cryomotl.Motl(motl_path=motl_a)

    if isinstance(motl_nn, str):
        motl_nn = cryomotl.Motl(motl_path=motl_nn)

    # Get unique feature idx
    features_a = np.unique(motl_a.df.loc[:, feature].values)
    features_nn = np.unique(motl_nn.df.loc[:, feature].values)

    # Work only with intersection
    features = np.intersect1d(features_a, features_nn, assume_unique=True)

    centered_coord = []
    nn_dist = []
    angular_distances = []
    rotated_coord = []
    subtomo_idx = []
    subtomo_idx_nn = []

    for f in features:
        fm_a = motl_a.get_motl_subset(f, feature_id=feature)
        fm_nn = motl_nn.get_motl_subset(f, feature_id=feature)

        idx, nn_idx, dist, nn_count = get_feature_nn_indices(fm_a, fm_nn, nn_number)

        if len(idx) == 0:
            continue

        coord_nn = fm_nn.get_coordinates() * pixel_size
        coord_a = fm_a.get_coordinates() * pixel_size

        # get angles
        angles_a = fm_a.get_angles()
        angles_a = angles_a[idx, :]
        angles_nn = fm_nn.get_angles()
        rotations = srot.from_euler("zxz", angles=angles_a, degrees=True)

        angles = -fm_a.df[["psi", "theta", "phi"]].values
        angles = angles[idx, :]
        rot = srot.from_euler("zxz", angles=angles, degrees=True)

        subtomos_nn = fm_nn.df["subtomo_id"].to_numpy()
        subtomos_a = fm_a.df["subtomo_id"].to_numpy()

        for i in range(nn_count):
            c_coord = coord_nn[nn_idx[:, i], :] - coord_a[idx, :]
            centered_coord.append(c_coord)
            nn_dist.append(dist[:, i] * pixel_size)

            angles_nn_sel = angles_nn[nn_idx[:, i], :]

            rotations_nn = srot.from_euler("zxz", angles=angles_nn_sel, degrees=True)
            angular_distances.append(geom.compare_rotations(rotations, rotations_nn, rotation_type=rotation_type))

            rotated_coord.append(rot.apply(c_coord))

            subtomo_idx_nn.append(subtomos_nn[nn_idx[:, i]])
            subtomo_idx.append(subtomos_a[idx])

    return (
        np.vstack(centered_coord),
        np.vstack(rotated_coord),
        np.concatenate(nn_dist),
        np.concatenate(angular_distances),
        np.concatenate(subtomo_idx),
        np.concatenate(subtomo_idx_nn),
    )


def get_nn_rotations(motl_a, motl_nn, nn_number=1, feature="tomo_id", type_id="geom1"):
    """Get nearest neighbor rotations based on specified features from two motl objects.

    Parameters
    ----------
    motl_a : str or Motl
        The path to the first motl file or a Motl object containing the first set of data.
    motl_nn : str or Motl
        The path to the second motl file or a Motl object containing the nearest neighbor data.
    nn_number : int, default=1
        The number of nearest neighbors to consider for each feature. Dfault is 1.
    feature : str, default='tomo_id'
        The feature used to identify unique elements in the motl data. Default is 'tomo_id'.
    type_id : str, default='geom1'
        The type identifier for the geometry. Default is 'geom1'.

    Returns
    -------
    points_on_sphere : ndarray
        An array of points on the sphere representing the rotations.
    angles : ndarray
        An array of Euler angles corresponding to the computed rotations in degrees.

    Notes
    -----
    This function assumes that the input motl objects or paths contain the necessary data
    and that the `get_motl_subset` and `get_angles` methods are available for the Motl class.
    """

    if isinstance(motl_a, str):
        motl_a = cryomotl.Motl(motl_path=motl_a)

    if isinstance(motl_nn, str):
        motl_nn = cryomotl.Motl(motl_path=motl_nn)

    # Get unique feature idx
    features_a = np.unique(motl_a.df.loc[:, feature].values)
    features_nn = np.unique(motl_nn.df.loc[:, feature].values)

    # Work only with intersection
    features = np.intersect1d(features_a, features_nn, assume_unique=True)

    nn_rotations = []

    for f in features:
        fm_a = motl_a.get_motl_subset(f, feature_id=feature)
        fm_nn = motl_nn.get_motl_subset(f, feature_id=feature)

        idx, idx_nn, _, nn_count = get_feature_nn_indices(fm_a, fm_nn, nn_number)

        angles_nn = fm_nn.get_angles()
        angles_ref_to_zero = -fm_a.get_feature(["psi", "theta", "phi"])
        rot_to_zero = srot.from_euler("zxz", angles=angles_ref_to_zero[idx, :], degrees=True)

        for i in range(nn_count):
            rot_nn = srot.from_euler("zxz", angles=angles_nn[idx_nn[:, i], :], degrees=True)
            nn_rotations.append(rot_to_zero * rot_nn)

    nn_rotations = srot.concatenate(nn_rotations)
    points_on_sphere = geom.visualize_rotations(nn_rotations, plot_rotations=False)
    angles = nn_rotations.as_euler("zxz", degrees=True)

    return points_on_sphere, angles


def filter_nn_radial_stats(input_stats, binary_mask):
    """Filter nearest neighbor radial statistics based on a binary mask.

    Parameters
    ----------
    input_stats : pandas.DataFrame
        A DataFrame containing the nearest neighbor statistics, which must include
        columns for coordinates and a grouping identifier.

    binary_mask : str
        Path to a binary mask file that will be read to create a boolean mask for filtering.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the filtered nearest neighbor statistics, with rows
        removed based on the provided binary mask.

    Notes
    -----
    The function adjusts the coordinates of the input statistics, ensuring they
    fall within the bounds of the binary mask. It then applies the mask to filter
    the statistics for each group defined by the 'qp_subtomo_id' column.
    """

    def filter_group(group, dx, dy, dz, boolean_mask):
        # Cast x, y, z columns to integers and adjust coordinates
        group["x_int"] = (group["coord_rx"] + dx).astype(int)
        group["y_int"] = (group["coord_ry"] + dy).astype(int)
        group["z_int"] = (group["coord_rz"] + dz).astype(int)

        # Ensure the coordinates are within the range of the boolean mask
        group = group[
            (group["x_int"] >= 0)
            & (group["x_int"] < 2 * dx)
            & (group["y_int"] >= 0)
            & (group["y_int"] < 2 * dy)
            & (group["z_int"] >= 0)
            & (group["z_int"] < 2 * dz)
        ]

        # Apply the boolean mask to filter rows
        mask_values = boolean_mask[group["x_int"], group["y_int"], group["z_int"]]
        final_group = group[mask_values]

        # Drop the temporary integer columns
        final_group = final_group.drop(columns=["x_int", "y_int", "z_int"])

        return final_group

    # Prepare the mask
    boolean_mask = cryomap.read(binary_mask)
    boolean_mask = np.where(boolean_mask < 0.5, False, True)
    dx, dy, dz = np.asarray(boolean_mask.shape) // 2

    # Get copy of the stats
    nn_stats = input_stats.copy()

    # Apply the filter_group function to each group
    result_df = (
        nn_stats.groupby("qp_subtomo_id")
        .apply(lambda group: filter_group(group, dx, dy, dz, boolean_mask))
        .reset_index(drop=True)
    )

    return result_df


def get_nn_stats_within_radius(input_motl, nn_radius, feature="tomo_id", index_by_feature=True):
    """Get nearest neighbor statistics within a specified radius from a given input motl.

    Parameters
    ----------
    input_motl : str or Motl object
        Path to the input motl file or Motl object to be loaded.
    nn_radius : float
        The radius within which to search for nearest neighbors.
    feature : str, default='tomo_id'
        The feature to index by. Default is 'tomo_id'.
    index_by_feature : bool, default=True
        If True, the output will be indexed by the specified feature; otherwise, it will be indexed by the original
        df of the motl. Default is True.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the nearest neighbor statistics, including:
        - qp_subtomo_id: Subtomo ID of the query point.
        - nn_subtomo_id: Subtomo ID of the nearest neighbor.
        - coord_x, coord_y, coord_z: Coordinates of the centered neighbor.
        - coord_rx, coord_ry, coord_rz: Rotated coordinates of the neighbor.
        - angular_distance: Angular distance between the query and nearest neighbor.
        - cone_distance: Cone distance between the query and nearest neighbor.
        - inplane_distance: In-plane distance between the query and nearest neighbor.
        - rot_x, rot_y, rot_z: Rotation angles of the nearest neighbor.
        - phi, theta, psi: Euler angles of the nearest neighbor.
        - qp_motl_id: Motl index of the query point.
        - nn_motl_idx: Motl index of the nearest neighbor.
    """

    input_motl = cryomotl.Motl.load(input_motl)

    # Get unique feature idx
    features = np.unique(input_motl.df.loc[:, feature].values)

    query_points = []
    query_motl_idx = []
    nn_motl_idx = []
    nn_rotations = []
    centered_coord = []
    rotated_coord = []
    angular_distances = []
    cone_distances = []
    inplane_distances = []
    nn_points = []

    for f in features:
        fm = input_motl.get_motl_subset(f, feature_id=feature)

        coord = fm.get_coordinates()
        center_idx, nn_idx = get_nn_within_distance(fm, nn_radius, unique_only=False)

        angles_nn = fm.get_angles()
        angles_ref_to_zero = -fm.get_feature(["psi", "theta", "phi"])

        rotations = fm.get_rotations()

        subtomos_idx = fm.df["subtomo_id"].to_numpy()

        if index_by_feature:
            motl_idx = fm.df.index.to_numpy()
        else:
            motl_idx = input_motl.df.index[input_motl.df[feature] == f].to_numpy()

        for i, c in enumerate(center_idx):
            rot_to_zero = srot.from_euler("zxz", angles=angles_ref_to_zero[c, :], degrees=True)

            for n in nn_idx[i]:
                rot_nn = srot.from_euler("zxz", angles=angles_nn[n, :], degrees=True)
                nn_rotations.append(rot_to_zero * rot_nn)

                c_coord = coord[n, :] - coord[c, :]
                centered_coord.append(c_coord)

                rotated_coord.append(rot_to_zero.apply(c_coord))

                rotations_nn = srot.from_euler("zxz", angles=angles_nn[n, :], degrees=True)
                ang_dist, cone_dist, inplane_dist = geom.compare_rotations(
                    rotations[c], rotations_nn, rotation_type="all"
                )
                angular_distances.append(ang_dist)
                cone_distances.append(cone_dist)
                inplane_distances.append(inplane_dist)
                query_points.append(subtomos_idx[c])
                query_motl_idx.append(motl_idx[c])
                nn_points.append(subtomos_idx[n])
                nn_motl_idx.append(motl_idx[n])

    nn_rotations = srot.concatenate(nn_rotations)
    points_on_sphere = geom.visualize_rotations(nn_rotations, plot_rotations=False)
    angles = nn_rotations.as_euler("zxz", degrees=True)

    centered_coord = np.vstack(centered_coord)
    rotated_coord = np.vstack(rotated_coord)
    angular_distances = np.atleast_2d(np.concatenate(angular_distances)).T
    cone_distances = np.atleast_2d(np.concatenate(cone_distances)).T
    inplane_distances = np.atleast_2d(np.concatenate(inplane_distances)).T
    query_points = [np.atleast_1d(arr) for arr in query_points]
    query_points = np.atleast_2d(np.concatenate(query_points)).T
    query_motl_idx = [np.atleast_1d(arr) for arr in query_motl_idx]
    query_motl_idx = np.atleast_2d(np.concatenate(query_motl_idx)).T
    nn_points = [np.atleast_1d(arr) for arr in nn_points]
    nn_points = np.atleast_2d(np.concatenate(nn_points)).T
    nn_motl_idx = [np.atleast_1d(arr) for arr in nn_motl_idx]
    nn_motl_idx = np.atleast_2d(np.concatenate(nn_motl_idx)).T

    nn_stats = pd.DataFrame(
        np.hstack(
            (
                query_points,
                nn_points,
                centered_coord,
                rotated_coord,
                angular_distances,
                cone_distances,
                inplane_distances,
                points_on_sphere,
                angles,
                query_motl_idx,
                nn_motl_idx,
            )
        ),
        columns=[
            "qp_subtomo_id",
            "nn_subtomo_id",
            "coord_x",
            "coord_y",
            "coord_z",
            "coord_rx",
            "coord_ry",
            "coord_rz",
            "angular_distance",
            "cone_distance",
            "inplane_distance",
            "rot_x",
            "rot_y",
            "rot_z",
            "phi",
            "theta",
            "psi",
            "qp_motl_id",
            "nn_motl_idx",
        ],
    )

    return nn_stats


def get_nn_within_radius(
    motl_a,
    motl_nn,
    nn_radius,
    pixel_size=1.0,
    feature="tomo_id",
):
    """Get the number of nearest neighbors within a specified radius for features in two motls.

    Parameters
    ----------
    motl_a : str or Motl
        Path to the first Motl file or a Motl object containing the particles to analyze.
    motl_nn : str or Motl
        Path to the second Motl file or a Motl object containing the particles for NN search.
    nn_radius : float
        The radius within which to count the nearest neighbors.
    pixel_size : float, default=1.0
        The size of the pixel/voxel that corresponds to the motls. Default is 1.0.
    feature : str, default='tomo_id'
        The feature identifier to use for matching between the two motls. Default is 'tomo_id'.

    Returns
    -------
    numpy.ndarray
        An array containing the count of nearest neighbors for each particle in the first motl within the specified radius.
    """

    if isinstance(motl_a, str):
        motl_a = cryomotl.Motl(motl_path=motl_a)

    if isinstance(motl_nn, str):
        motl_nn = cryomotl.Motl(motl_path=motl_nn)

    # Get unique feature idx
    features_a = np.unique(motl_a.df.loc[:, feature].values)
    features_nn = np.unique(motl_nn.df.loc[:, feature].values)

    # Work only with intersection
    features = np.intersect1d(features_a, features_nn, assume_unique=True)

    nn_count = []

    for f in features:
        fm_a = motl_a.get_motl_subset(f, feature_id=feature)
        fm_nn = motl_nn.get_motl_subset(f, feature_id=feature)

        coord_a = fm_a.get_coordinates() * pixel_size
        coord_nn = fm_nn.get_coordinates() * pixel_size

        kdt_nn = sn.KDTree(coord_nn)
        fm_nn_count = kdt_nn.query_radius(coord_a, r=nn_radius, count_only=True)

        nn_count.append(fm_nn_count)

    return np.concatenate(nn_count, axis=0)


def plot_nn_coord_df(
    df,
    circle_radius,
    output_name=None,
    displ_threshold=None,
    title=None,
    marker_size=20,
):
    """Plot 2D scatter plots of nearest neighbor coordinates.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the coordinates with columns 'coord_x', 'coord_y', and 'coord_z'.
    circle_radius : float
        Radius of the circles to be drawn on the plots.
    output_name : str, optional
        Name of the file to save the figure. If None, the figure will not be saved. Default is None.
    displ_threshold : float, optional
        Threshold for displaying limits on the axes. If None, limits will not be set. Default is None.
    title : str, optional
        Title for the entire figure. If None, no title will be set. Default is None.
    marker_size : int, default=20
        Size of the markers in the scatter plots. Default is 20.

    Returns
    -------
    None
        The function displays the plots and optionally saves the figure to a file.
    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title("XY distribution")
    sns.scatterplot(ax=axs[0], data=df, x="coord_x", y="coord_y", s=marker_size)
    axs[1].set_title("XZ distribution")
    sns.scatterplot(ax=axs[1], data=df, x="coord_x", y="coord_z", s=marker_size)
    axs[2].set_title("YZ distribution")
    sns.scatterplot(ax=axs[2], data=df, x="coord_y", y="coord_z", s=marker_size)

    # sns.move_legend(axs[0], "upper right")
    # sns.move_legend(axs[1], "upper right")
    # sns.move_legend(axs[2], "upper right")

    circle1 = plt.Circle((0, 0), circle_radius, color="gold", alpha=0.4, fill=True)
    circle2 = plt.Circle((0, 0), circle_radius, color="gold", alpha=0.4, fill=True)
    circle3 = plt.Circle((0, 0), circle_radius, color="gold", alpha=0.4, fill=True)

    axs[0].set_aspect("equal", "box")
    axs[1].set_aspect("equal", "box")
    axs[2].set_aspect("equal", "box")

    if displ_threshold is not None:
        limits = [-displ_threshold, displ_threshold]
        axs[0].set(xlim=limits, ylim=limits)
        axs[1].set(xlim=limits, ylim=limits)
        axs[2].set(xlim=limits, ylim=limits)

    if title is not None:
        fig.suptitle(title)

    axs[0].add_patch(circle1)
    axs[1].add_patch(circle2)
    axs[2].add_patch(circle3)

    if output_name is not None:
        plt.savefig(output_name, transparent=True)


def plot_nn_rot_coord_df(df, output_name=None, displ_threshold=None, title=None, marker_size=20):
    """Plot 2D scatter plots of nearest neighbor coordinates that are rotated to the reference frame.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the coordinates with columns 'coord_x', 'coord_y', and 'coord_z'.
    output_name : str, optional
        Name of the file to save the figure. If None, the figure will not be saved. Default is None.
    displ_threshold : float, optional
        Threshold for displaying limits on the axes. If None, limits will not be set. Default is None.
    title : str, optional
        Title for the entire figure. If None, no title will be set. Default is None.
    marker_size : int, default=20
        Size of the markers in the scatter plots. Default is 20.

    Returns
    -------
    None
        The function displays the plots and optionally saves the figure to a file.
    """
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title("XY distribution")
    sns.scatterplot(ax=axs[0], data=df, x="coord_rx", y="coord_ry", hue="type", s=marker_size)
    axs[1].set_title("XZ distribution")
    sns.scatterplot(ax=axs[1], data=df, x="coord_rx", y="coord_rz", hue="type", s=marker_size)
    axs[2].set_title("YZ distribution")
    sns.scatterplot(ax=axs[2], data=df, x="coord_ry", y="coord_rz", hue="type", s=marker_size)

    sns.move_legend(axs[0], "upper right")
    sns.move_legend(axs[1], "upper right")
    sns.move_legend(axs[2], "upper right")

    axs[0].set_aspect("equal", "box")
    axs[1].set_aspect("equal", "box")
    axs[2].set_aspect("equal", "box")

    if displ_threshold is not None:
        limits = [-displ_threshold, displ_threshold]
        axs[0].set(xlim=limits, ylim=limits)
        axs[1].set(xlim=limits, ylim=limits)
        axs[2].set(xlim=limits, ylim=limits)

    if title is not None:
        fig.suptitle(title)

    plt.tight_layout()
    if output_name is not None:
        plt.savefig(output_name, transparent=True)


def plot_nn_coord(coord, displ_threshold=None, marker_size=20):
    """Plot 2D scatter distributions of 3D coordinates.

    Parameters
    ----------
    coord : numpy.ndarray
        A 2D array of shape (n, 3) where n is the number of points and each row represents a point in 3D space (x, y, z).
    displ_threshold : float, optional
        A threshold value to set the limits of the scatter plots. If provided, the x and y limits of each subplot will
        be set to [-displ_threshold, displ_threshold]. Default is None.
    marker_size : int, default=20
        The size of the markers in the scatter plots. Default is 20.

    Returns
    -------
    None
        This function does not return any value. It displays the scatter plots of the coordinates.
    """

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].set_title("XY distribution")
    axs[0].scatter(coord[:, 0], coord[:, 1], s=marker_size)
    axs[1].set_title("XZ distribution")
    axs[1].scatter(coord[:, 0], coord[:, 2], s=marker_size)
    axs[2].set_title("YZ distribution")
    axs[2].scatter(coord[:, 1], coord[:, 2], s=marker_size)

    axs[0].set_aspect("equal", "box")
    axs[1].set_aspect("equal", "box")
    axs[2].set_aspect("equal", "box")

    if displ_threshold is not None:
        limits = [-displ_threshold, displ_threshold]
        axs[0].set(xlim=limits, ylim=limits)
        axs[1].set(xlim=limits, ylim=limits)
        axs[2].set(xlim=limits, ylim=limits)


def plot_nn_rot_coord_df_plotly(df, displ_threshold=None, title=None, marker_size=5, output_name=None):
    """
    Create 2D scatter plots of rotated NN coordinates using Plotly (XY, XZ, YZ).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns 'coord_rx', 'coord_ry', 'coord_rz', and 'type'.
    displ_threshold : float, optional
        Axis limit for all plots (symmetric), default: None.
    title : str, optional
        Overall figure title.
    marker_size : int
        Size of scatter markers.
    output_name : str, optional
        If given, saves the figure as HTML or image.

    Returns
    -------
    plotly.graph_objects.Figure
        A Plotly figure with 3 subplots.
    """
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["XY Distribution", "XZ Distribution", "YZ Distribution"],
        shared_yaxes=False,
        horizontal_spacing=0.08,
    )

    # Add XY
    fig.add_trace(
        go.Scattergl(
            x=df["coord_rx"],
            y=df["coord_ry"],
            mode="markers",
            marker=dict(size=marker_size),
            name="XY",
            text=df["type"],
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Add XZ
    fig.add_trace(
        go.Scattergl(
            x=df["coord_rx"],
            y=df["coord_rz"],
            mode="markers",
            marker=dict(size=marker_size),
            name="XZ",
            text=df["type"],
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Add YZ
    fig.add_trace(
        go.Scattergl(
            x=df["coord_ry"],
            y=df["coord_rz"],
            mode="markers",
            marker=dict(size=marker_size),
            name="YZ",
            text=df["type"],
            showlegend=False,
        ),
        row=1,
        col=3,
    )

    if displ_threshold is not None:
        limits = [-displ_threshold, displ_threshold]
        fig.update_xaxes(range=limits, row=1, col=1)
        fig.update_yaxes(range=limits, row=1, col=1)
        fig.update_xaxes(range=limits, row=1, col=2)
        fig.update_yaxes(range=limits, row=1, col=2)
        fig.update_xaxes(range=limits, row=1, col=3)
        fig.update_yaxes(range=limits, row=1, col=3)

    fig.update_layout(
        title=title or "Rotated Coordinate Distributions",
        height=400,
        margin=dict(t=40, b=30, l=30, r=30),
        plot_bgcolor="white",
    )

    # Save to file if needed
    if output_name:
        if output_name.endswith(".html"):
            fig.write_html(output_name)
        else:
            fig.write_image(output_name)

    return fig
