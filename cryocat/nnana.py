import numpy as np
import pandas as pd
from cryocat import cryomotl
from scipy.spatial.transform import Rotation as srot
from cryocat import geom
import seaborn as sns
from scipy.spatial import KDTree
import sklearn.neighbors as sn
import matplotlib.pyplot as plt


def get_nn_within_distance(feature_motl, radius):
    coord = feature_motl.get_coordinates()
    kdt_nn = sn.KDTree(coord)
    nn_idx = kdt_nn.query_radius(coord, radius)

    # create id array to check where the NN was the same particle
    ordered_idx = np.arange(0, nn_idx.shape[0], 1)

    # each particle is also part of the results -> keep only those that have more than 1 particle
    ordered_idx = [i for i, row in zip(ordered_idx, nn_idx) if len(row) > 1]
    nn_indices = nn_idx[ordered_idx]

    # remove duplicates: first sort in each row, then find the unique idx
    sorted_idx = [np.sort(row) for row in nn_indices]
    nn_indices = np.unique([tuple(row) for row in sorted_idx], axis=0)

    center_idx = np.array([row[0] for row in nn_indices])
    nn_idx = np.array([row[1:] for row in nn_indices])
    # nn_indices = np.column_stack((first_elements, remaining_elements))

    return center_idx, nn_idx


def get_feature_nn_indices(fm_a, fm_nn, nn_number=1):
    coord_a = fm_a.get_coordinates()
    coord_nn = fm_nn.get_coordinates()

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


def get_nn_stats(motl_a, motl_nn, pixel_size=1.0, feature_id="tomo_id", nn_number=1):
    (
        centered_coord,
        rotated_coord,
        nn_dist,
        ang_dst,
        subtomo_idx,
        subtomo_idx_nn,
    ) = get_nn_distances(
        motl_a,
        motl_nn,
        nn_number=nn_number,
        pixel_size=pixel_size,
        feature=feature_id,
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


def get_nn_distances(
    motl_a,
    motl_nn,
    pixel_size=1.0,
    nn_number=1,
    feature="tomo_id",
):
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
            angular_distances.append(geom.angular_distance(rotations, rotations_nn)[0])

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
        angles_ref_to_zero = -fm_a.get_features(["psi", "theta", "phi"])
        rot_to_zero = srot.from_euler("zxz", angles=angles_ref_to_zero[idx, :], degrees=True)

        for i in range(nn_count):
            rot_nn = srot.from_euler("zxz", angles=angles_nn[idx_nn[:, i], :], degrees=True)
            nn_rotations.append(rot_to_zero * rot_nn)

    nn_rotations = srot.concatenate(nn_rotations)
    points_on_sphere = geom.visualize_rotations(nn_rotations, plot_rotations=False)
    angles = nn_rotations.as_euler("zxz", degrees=True)

    return points_on_sphere, angles


def get_nn_within_radius(
    motl_a,
    motl_nn,
    nn_radius,
    pixel_size=1.0,
    feature="tomo_id",
):
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
    # if displ_threshold is not None:
    #   coord[:,0] = coord[:, (coord#[:,0] >= -displ_threshold) & (coord[:,0] <= displ_threshold) ]

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
