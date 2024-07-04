import numpy as np
import pandas as pd
import cryocat
from cryocat import cryomotl
from scipy.spatial.transform import Rotation as srot
from cryocat import geom
import seaborn as sns
from scipy.spatial import KDTree
import sklearn.neighbors as sn
import matplotlib.pyplot as plt


def add_occupancy(
    motl,
    output_motl=None,
    occupancy_id="geom1",
    object_id="object_id",
    feature="tomo_id",
    order_id="geom2",
):
    if isinstance(motl, str):
        motl = cryomotl.Motl.load(motl)

    motl.df[occupancy_id] = motl.df.groupby([feature, object_id])[order_id].transform("max")

    if output_motl is not None:
        motl.write_out(output_motl)

    return motl


def get_feature_nn(fm_entry, fm_exit, remove_duplicates=True):
    coord_entry = fm_entry.get_coordinates()
    coord_exit = fm_exit.get_coordinates()

    kdt_entry = sn.KDTree(coord_entry)

    # return 2 NN for each point
    dist, idx = kdt_entry.query(coord_exit, k=1)

    return dist, idx


def get_feature_nn_indices(fm_entry, fm_exit, remove_duplicates=True):
    coord_entry = fm_entry.get_coordinates()
    coord_exit = fm_exit.get_coordinates()

    if coord_entry.size <= 3:
        return [], [], []

    kdt_entry = sn.KDTree(coord_entry)

    # return 2 NN for each point
    dist, idx = kdt_entry.query(coord_exit, k=2)

    # create id array to check where the NN was the same particle
    ordered_idx = np.arange(0, idx.shape[0], 1)

    # if the first NN was the same particle, return the second one; same for distances
    nn_indices = np.where(ordered_idx == idx[:, 0], idx[:, 1], idx[:, 0])
    nn_distances = np.where(ordered_idx == idx[:, 0], dist[:, 1], dist[:, 0])

    # remove duplicates: first sort in each row, then find the unique idx
    if remove_duplicates:
        sorted_idx = np.sort(
            np.hstack((ordered_idx.reshape(idx.shape[0], 1), nn_indices.reshape(idx.shape[0], 1))),
            axis=1,
        )
        unique_idx = np.sort(np.unique(sorted_idx, return_index=True, axis=0)[1])

        return ordered_idx[unique_idx], nn_indices[unique_idx], nn_distances[unique_idx]
    else:
        return ordered_idx, nn_indices, nn_distances


def get_rotations(df):
    angles = -df[["psi", "theta", "phi"]].values
    angles_nn = df[["phi", "theta", "psi"]].values
    angles = angles[0:-1, :]
    angles_nn = angles_nn[1:, :]

    rot_nn = srot.from_euler("zxz", angles=angles_nn, degrees=True)
    rot_to_zero = srot.from_euler("zxz", angles=angles, degrees=True)
    final_rot = rot_to_zero * rot_nn
    points_on_sphere = geom.visualize_rotations(final_rot, plot_rotations=False)

    zero_angles = np.tile([0.0, 0.0, 0.0], (angles_nn.shape[0], 1))
    zero_rot = srot.from_euler("zxz", angles=zero_angles, degrees=True)
    ang_dist = geom.angular_distance(final_rot, zero_rot)[0]
    rotated_angles = final_rot.as_euler("zxz", degrees=True)

    rot_stats = np.hstack((ang_dist.reshape(ang_dist.shape[0], 1), points_on_sphere, rotated_angles))

    return rot_stats


def get_chain_distances(df, pixel_size, feature):
    entry_coord = (df[["x", "y", "z"]].values + df[["shift_x", "shift_y", "shift_z"]].values) * pixel_size
    exit_coord = df[["exit_x", "exit_y", "exit_z"]].values * pixel_size
    entry_coord = entry_coord[1:, :]
    exit_coord = exit_coord[0:-1, :]

    chain_dist = np.linalg.norm(entry_coord - exit_coord, axis=1).reshape(entry_coord.shape[0], 1)
    centered_coord = entry_coord - exit_coord

    angles = -df[["psi", "theta", "phi"]].values
    angles = angles[0:-1, :]
    rot = srot.from_euler("zxz", angles=angles, degrees=True)
    rotated_points = rot.apply(centered_coord)

    dist_stats = np.hstack(
        (
            df[feature].values[:-1].reshape(chain_dist.shape[0], 1),
            chain_dist,
            centered_coord,
            rotated_points,
        )
    )

    return dist_stats


def get_polysome_stats(motl_entry, motl_exit, pixel_size=1.0, feature="geom1"):
    if isinstance(motl_entry, str):
        motl_entry = cryomotl.Motl.load(motl_entry)

    if isinstance(motl_exit, str):
        motl_exit = cryomotl.Motl.load(motl_exit)

    poly_entry_df = motl_entry.df[motl_entry.df[feature] > 1].copy()
    poly_exit_df = motl_exit.df[motl_exit.df[feature] > 1].copy()

    poly_entry_df.sort_values(["tomo_id", "object_id", "geom2"], inplace=True)
    poly_exit_df.sort_values(["tomo_id", "object_id", "geom2"], inplace=True)

    poly_entry_df[["exit_x", "exit_y", "exit_z"]] = (
        poly_exit_df[["x", "y", "z"]].values + poly_exit_df[["shift_x", "shift_y", "shift_z"]].values
    )

    chain_dist_stats = poly_entry_df.groupby(["tomo_id", "object_id"]).apply(get_chain_distances, pixel_size, feature)
    chain_rot_stats = poly_entry_df.groupby(["tomo_id", "object_id"]).apply(get_rotations)

    chain_dist_stats = np.vstack(chain_dist_stats.values)
    chain_rot_stats = np.vstack(chain_rot_stats.values)

    stats_df = pd.DataFrame(
        np.hstack((chain_dist_stats, chain_rot_stats)),
        columns=[
            "polysome_length",
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

    stats_df["type"] = "polysome"

    return stats_df


def get_monosome_stats(motl_entry, motl_exit, pixel_size=1.0, feature="geom1"):
    centered_coord, rotated_coord, nn_dist, ang_dst = get_nn_distances(
        motl_entry, motl_exit, pixel_size=pixel_size, monosomes_only=True
    )
    coord_rot, angles = get_nn_rotations(motl_entry, motl_exit, monosomes_only=True)

    monosome_stats = pd.DataFrame(
        np.hstack(
            (
                nn_dist.reshape((nn_dist.shape[0], 1)),
                centered_coord,
                rotated_coord,
                ang_dst.reshape((nn_dist.shape[0], 1)),
                coord_rot,
                angles,
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
        ],
    )

    monosome_stats["type"] = "monosome"

    return monosome_stats


def get_nn_stats(
    motl_entry, motl_exit, pixel_size=1.0, feature_id="tomo_id", angular_dist_type="full", remove_duplicates=True
):
    (
        centered_coord,
        rotated_coord,
        nn_dist,
        ang_dst,
        subtomo_idx,
        subtomo_idx_nn,
    ) = get_nn_distances(
        motl_entry,
        motl_exit,
        pixel_size=pixel_size,
        feature=feature_id,
        monosomes_only=False,
        angular_dist_type=angular_dist_type,
        remove_duplicates=remove_duplicates,
    )
    coord_rot, angles = get_nn_rotations(
        motl_entry, motl_exit, feature=feature_id, monosomes_only=False, remove_duplicates=remove_duplicates
    )

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
    motl_entry,
    motl_exit,
    pixel_size=1.0,
    feature="tomo_id",
    monosomes_only=False,
    type_id="geom1",
    angular_dist_type="full",
    remove_duplicates=True,
):
    if isinstance(motl_entry, str):
        motl_entry = cryomotl.Motl.load(motl_entry)

    if isinstance(motl_exit, str):
        motl_exit = cryomotl.Motl.load(motl_exit)

    if angular_dist_type == "full":
        res_id = 0
    elif angular_dist_type == "cone":
        res_id = 1
    elif angular_dist_type == "inplane":
        res_id = 2
    else:
        raise ValueError(f"Specified angular distance type is not supported: {angular_dist_type}!")

    features = np.unique(motl_entry.df.loc[:, feature].values)
    centered_coord = []
    nn_dist = []
    angular_distances = []
    rotated_coord = []
    subtomo_idx = []
    subtomo_idx_nn = []

    for f in features:
        fm_entry = motl_entry.get_motl_subset(f, feature_id=feature, reset_index=False)
        fm_exit = motl_exit.get_motl_subset(f, feature_id=feature, reset_index=False)

        if monosomes_only:
            fm_entry = fm_entry.get_motl_subset(1, type_id, reset_index=False)
            fm_exit = fm_exit.get_motl_subset(1, type_id, reset_index=False)

        idx, nn_idx, dist = get_feature_nn_indices(fm_entry, fm_exit, remove_duplicates)

        if len(idx) == 0:
            continue

        # ordered_idx = np.arange(0, idx.shape[0], 1)

        coord_entry = fm_entry.get_coordinates() * pixel_size
        coord_exit = fm_exit.get_coordinates() * pixel_size

        c_coord = coord_entry[nn_idx, :] - coord_exit[idx, :]
        centered_coord.append(c_coord)
        nn_dist.append(dist * pixel_size)

        # rotations = fm_entry.get_rotations()
        all_angles = fm_entry.get_angles()
        angles = all_angles[idx, :]
        angles_nn = all_angles[nn_idx, :]
        rotations = srot.from_euler("zxz", angles=angles, degrees=True)
        rotations_nn = srot.from_euler("zxz", angles=angles_nn, degrees=True)
        angular_distances.append(geom.compare_rotations(rotations, rotations_nn)[res_id])

        angles = -fm_entry.df[["psi", "theta", "phi"]].values
        angles = angles[idx, :]
        rot = srot.from_euler("zxz", angles=angles, degrees=True)
        rotated_coord.append(rot.apply(c_coord))
        subtomos_all = fm_entry.df["subtomo_id"].to_numpy()
        subtomo_idx_nn.append(subtomos_all[nn_idx])
        subtomo_idx.append(subtomos_all[idx])

    return (
        np.vstack(centered_coord),
        np.vstack(rotated_coord),
        np.concatenate(nn_dist),
        np.concatenate(angular_distances),
        np.concatenate(subtomo_idx),
        np.concatenate(subtomo_idx_nn),
    )


def get_nn_rotations(
    motl_entry, motl_exit, feature="tomo_id", monosomes_only=False, type_id="geom1", remove_duplicates=True
):
    if isinstance(motl_entry, str):
        motl_entry = cryomotl.Motl.load(motl_entry)

    if isinstance(motl_exit, str):
        motl_exit = cryomotl.Motl.load(motl_exit)

    features = np.unique(motl_entry.df.loc[:, feature].values)

    nn_rotations = []

    for f in features:
        fm_entry = motl_entry.get_motl_subset(f, feature_id=feature, reset_index=False)
        fm_exit = motl_exit.get_motl_subset(f, feature_id=feature, reset_index=False)

        if monosomes_only:
            fm_entry = fm_entry.get_motl_subset(1, type_id, reset_index=False)
            fm_exit = fm_exit.get_motl_subset(1, type_id, reset_index=False)

        idx, idx_nn, _ = get_feature_nn_indices(fm_entry, fm_exit, remove_duplicates)

        if len(idx) == 0:
            continue

        angles_nn = fm_entry.get_angles()
        rot_nn = srot.from_euler("zxz", angles=angles_nn[idx_nn, :], degrees=True)

        angles_ref_to_zero = -fm_entry.get_feature(["psi", "theta", "phi"])
        rot_to_zero = srot.from_euler("zxz", angles=angles_ref_to_zero[idx, :], degrees=True)

        nn_rotations.append(rot_to_zero * rot_nn)

    nn_rotations = srot.concatenate(nn_rotations)
    points_on_sphere = geom.visualize_rotations(nn_rotations, plot_rotations=False)
    angles = nn_rotations.as_euler("zxz", degrees=True)

    return points_on_sphere, angles


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


def get_class_polysome_occupancies_mdp(motl, occupancy_id="geom1"):
    u_classes = np.unique(motl.df.loc[:, "class"].values)

    class_df = pd.DataFrame(columns=["class", "ribosome_number", "chain_type"])

    for c in u_classes:
        monosomes = motl.df[(motl.df["class"] == c) & (motl.df[occupancy_id] == 1)].shape[0]
        disomes = motl.df[(motl.df["class"] == c) & (motl.df[occupancy_id] == 2)].shape[0]
        polysomes = motl.df[(motl.df["class"] == c) & (motl.df[occupancy_id] > 2)].shape[0]
        new_entry = [c, monosomes, "monosomes"]
        class_df.loc[len(class_df)] = new_entry
        new_entry = [c, disomes, "disomes"]
        class_df.loc[len(class_df)] = new_entry
        new_entry = [c, polysomes, "polysomes"]
        class_df.loc[len(class_df)] = new_entry

    return class_df


def get_class_polysome_occupancies_mp(motl, occupancy_id="geom1"):
    u_classes = np.unique(motl.df.loc[:, "class"].values)

    number_of_particles = motl.df.shape[0]

    class_df = pd.DataFrame(columns=["class", "ribosome_number", "chain_type", "percentage"])

    for c in u_classes:
        monosomes = motl.df[(motl.df["class"] == c) & (motl.df[occupancy_id] == 1)].shape[0]
        polysomes = motl.df[(motl.df["class"] == c) & (motl.df[occupancy_id] > 1)].shape[0]
        new_entry = [c, monosomes, "monosomes", (monosomes / number_of_particles * 100)]
        class_df.loc[len(class_df)] = new_entry
        new_entry = [c, polysomes, "polysomes", (polysomes / number_of_particles * 100)]
        class_df.loc[len(class_df)] = new_entry

    return class_df


def assign_class(
    motl_unassigned,
    motl_list,
    starting_class=1,
    dist_threshold=20,
    output_motl=None,
    unassigned_class=0,
    update_coord=False,
):
    motl = cryomotl.Motl.load(motl_unassigned)

    motl.df["class"] = unassigned_class

    classified_particles = 0
    overlaps = 0
    cl = starting_class

    for m in motl_list:
        cm = cryomotl.Motl.load(m)
        classified_particles += cm.df.shape[0]
        tomos = np.unique(cm.df.loc[:, "tomo_id"].values)

        for t in tomos:
            tm_coord = cm.get_coordinates(t)
            all_coord = motl.get_coordinates(t)
            tm = cm.get_motl_subset(t, return_df=True, reset_index=False)
            tm_all = motl.get_motl_subset(t, return_df=True, reset_index=False)

            tm_idx = np.arange(0, tm.shape[0], 1)

            kdt = KDTree(all_coord)
            dist, idx = kdt.query(tm_coord)

            th = dist < dist_threshold  # kick out particles too far away
            idx = idx[th]
            tm_idx = tm_idx[th]

            unique_idx, unique_idx_counts = np.unique(idx, return_counts=True)
            duplicates = unique_idx[unique_idx_counts > 1]

            if duplicates.size > 0:
                identical_idx = []
                for d in duplicates:
                    identical_idx.append(np.where(idx == d))
                identical_idx = np.concatenate(identical_idx).flatten()
                subtomo_idx = tm.loc[tm.index[identical_idx], ["subtomo_id"]].values.flatten()
                print(f"Following particles in motl {m} are identical: {subtomo_idx}")
                overlaps += np.sum(unique_idx_counts) - unique_idx_counts.size

            tm_all.loc[tm_all.index[idx], ["geom1"]] += 1
            tm_all.loc[tm_all["geom1"] > 1, ["geom2"]] = tm_all.loc[
                tm_all["geom1"] > 1, ["class"]
            ]  # store previous class in the case of overlap
            tm_all.loc[tm_all.index[idx], ["class"]] = cl

            if update_coord:
                tm_all.loc[tm_all.index[idx], ["phi", "psi", "theta"]] = tm.loc[
                    tm.index[tm_idx], ["phi", "psi", "theta"]
                ].values
                tm_all.loc[tm_all.index[idx], ["geom3", "geom4", "geom5"]] = (
                    tm.loc[tm.index[tm_idx], ["x", "y", "z"]].values
                    + tm.loc[tm.index[tm_idx], ["shift_x", "shift_y", "shift_z"]].values
                )

            motl.df.loc[motl.df["tomo_id"] == t] = tm_all

        cl += 1

    assigned_aprticles = motl.df.loc[motl.df["geom1"] > 0].shape[0]
    print(
        f"Particles in classified motls: {classified_particles}, number of assigned particles: {assigned_aprticles}, number of overlaps: {overlaps}"
    )

    if update_coord:
        motl.df.loc[motl.df["class"] != unassigned_class, ["x", "y", "z"]] = motl.df.loc[
            motl.df["class"] != unassigned_class, ["geom3", "geom4", "geom5"]
        ].values
        motl.df.loc[motl.df["class"] != unassigned_class, ["shift_x", "shift_y", "shift_z"]] = 0.0
        motl.df["geom3"] = 0.0

    motl.df["geom4"] = motl.df["geom2"].values
    motl.df["geom5"] = motl.df["geom1"].values
    motl.df[["geom1", "geom2"]] = 0.0

    motl.update_coordinates()

    if output_motl is not None:
        motl.write_to_emfile(output_motl)

    return motl


# def get_nn_dist(kdt,query_point,query_size,active_points,test_value):
# search the nearest points
# first get all points sorted by distance
# then find the first one that is still unused
#    entry_dist, np_idx = kdt.query(query_point,query_size)
#    rp_idx = np.argmax(active_points[np_idx]==test_value)
#    entry_dist = entry_dist[rp_idx]
#    np_idx = np_idx[rp_idx]
#
#    return entry_dist, np_idx

# def get_nn_dist2(kdt,query_point,dist_max,dist_min,active_points,test_value):

#     id_max = kdt.query_ball_point(query_point, dist_max, p=2.0, eps=0, return_sorted=False)
#     entry_dist, np_idx = kdt.query(query_point,2)

#     if len(id_max) == 0:
#         return -1

#     if dist_min > 0:
#         id_min = kdt.query_ball_point(query_point, dist_min, p=2.0, eps=0, return_sorted=False)
#         id_max = np.setdiff1d(id_max,id_min)
#     else:
#         id_max = np.array(id_max)

#     ia=id_max[active_points[id_max]==test_value]

#     if ia.size != 0:
#         nn_id=ia[0]
#     else:
#         nn_id=-1

#     return nn_id


def add_traced_info(traced_motl, input_motl, output_motl_path=None, sort_by_subtomo=True):
    if isinstance(traced_motl, str):
        traced_motl = motl = cryomotl.Motl.load(traced_motl)

    if isinstance(input_motl, str):
        input_motl = motl = cryomotl.Motl.load(input_motl)

    if sort_by_subtomo:
        traced_motl.df.sort_values(["subtomo_id"], inplace=True)
        input_motl.df.sort_values(["subtomo_id"], inplace=True)

    if not np.array_equal(traced_motl.df["subtomo_id"].values, traced_motl.df["subtomo_id"].values):
        raise ValueError("The input motl has different subtomograms as the traced motl.")

    input_motl.df[["geom1", "geom2", "geom4", "object_id"]] = traced_motl.df[
        ["geom1", "geom2", "geom4", "object_id"]
    ].values
    input_motl.df.sort_values(["tomo_id", "object_id", "geom2"], inplace=True)

    if output_motl_path is not None:
        input_motl.write_out(output_motl_path)

    return input_motl


def get_nn_dist(kdt, query_point, dist_max, dist_min, active_points, test_value):
    id_max, dist = kdt.query_radius(query_point, dist_max, return_distance=True, sort_results=True)
    # id_max, dist = [a[0] for a in kdt.query_radius(query_point, dist_max, return_distance=True, sort_results=True)]
    id_max = id_max[0]
    dist = dist[0]
    if id_max.size == 0:
        return -1, []

    rp_idx = id_max[active_points[id_max] == test_value]
    rp_dist = dist[active_points[id_max] == test_value]

    if rp_idx.size == 0:
        return -1, []
    elif dist_min > 0:
        rp_idx = rp_idx[rp_dist > dist_min]
        rp_dist = rp_dist[rp_dist > dist_min]

    if rp_idx.size == 0:
        return -1, []
    else:
        return rp_idx[0], rp_dist[0]


def add_chain_suffix(
    chain_df,
    motl,
    traced_df,
    subtomo_id,
    current_dist,
    store_idx1="object_id",
    store_idx2="geom2",
    store_dist="geom4",
):
    particle_id = motl.df.loc[motl.df.index[subtomo_id], "subtomo_id"]

    temp_cl_id, order_id, previous_dist = traced_df.loc[
        traced_df["subtomo_id"] == particle_id, [store_idx1, store_idx2, store_dist]
    ].values[0]
    chain_max_order = np.max(traced_df.loc[traced_df[store_idx1] == temp_cl_id, [store_idx2]].values)

    if chain_max_order != order_id:  # the closest particle is not the last one
        if previous_dist <= current_dist:  # the original chain holds, do nothing
            return False
        else:  # the new chain is better, cut of the tail of the existing one
            current_class = chain_df[store_idx1].values[0]
            traced_df.loc[
                (traced_df[store_idx1] == temp_cl_id) & (traced_df[store_idx2] > order_id),
                store_idx1,
            ] = current_class
            new_chain_size = traced_df.loc[(traced_df[store_idx1] == current_class), store_idx2].shape[0]
            traced_df.loc[(traced_df[store_idx1] == current_class), store_idx2] = np.arange(1, new_chain_size + 1)
            chain_max_order = np.max(
                traced_df.loc[traced_df[store_idx1] == temp_cl_id, [store_idx2]].values
            )  # max changed in the meantime so has to be fetched again

    traced_df.loc[traced_df["subtomo_id"] == particle_id, store_dist] = (
        current_dist  # add distance to the last traced element from the chain (should be 0 before)
    )
    chain_df[store_idx1] = temp_cl_id
    chain_df[store_idx2] += chain_max_order

    return True  # chain was changed


def add_chain_prefix(
    chain_df,
    motl,
    traced_df,
    subtomo_id,
    current_dist,
    store_idx1="object_id",
    store_idx2="geom2",
    store_dist="geom4",
    class_max=None,
):
    # finding out class of the chain that should be appended to the current chain
    particle_id = motl.df.loc[motl.df.index[subtomo_id], "subtomo_id"]
    class_to_change = traced_df.loc[traced_df["subtomo_id"] == particle_id, store_idx1].values[0]

    order_id = traced_df.loc[traced_df["subtomo_id"] == particle_id, store_idx2].values[0]

    current_class = chain_df[store_idx1].values[0]
    cut_off_size = 0

    if order_id != 1:  # the closest particle is NOT the first one in the chain!
        # take the previous particle distance
        previous_dist = traced_df.loc[
            (traced_df[store_idx1] == class_to_change) & (traced_df[store_idx2] == order_id - 1),
            store_dist,
        ].values[0]

        if previous_dist <= current_dist:  # original particle closer -> do not append
            return -1
        else:  # the new particle is closer - change the class/object_id to the one from the current particle
            cut_off_size = traced_df.loc[
                (traced_df[store_idx1] == class_to_change) & (traced_df[store_idx2] < order_id)
            ].shape[0]
            if (
                class_max is None
            ):  # Only appending, the chain object_id value is not used and can be assing to the cut chain
                traced_df.loc[
                    (traced_df[store_idx1] == class_to_change) & (traced_df[store_idx2] < order_id),
                    store_idx1,
                ] = current_class
            else:  # Connectiong from both sides, the chain object_id was changed in the previoius append and cannot be used -> the input from current is used
                traced_df.loc[
                    (traced_df[store_idx1] == class_to_change) & (traced_df[store_idx2] < order_id),
                    store_idx1,
                ] = -1  # class_max[1]

    if class_max is None:
        chain_df[store_idx1] = class_to_change
        class_max = np.max(chain_df[store_idx2].values)
        traced_df.loc[traced_df[store_idx1] == class_to_change, [store_idx2]] += class_max - cut_off_size
    else:
        temp_cl_id = chain_df[store_idx1][0]
        traced_df.loc[traced_df[store_idx1] == class_to_change, [store_idx2]] += class_max[0] - cut_off_size
        traced_df.loc[traced_df[store_idx1] == class_to_change, [store_idx1]] = temp_cl_id
        if order_id != 1:
            traced_df.loc[traced_df[store_idx1] == -1, [store_idx1]] = class_max[1]  # class_to_change

    chain_df.loc[chain_df.index[-1], store_dist] = current_dist


def trace_chains(
    motl_entry,
    motl_exit,
    max_distance,
    min_distance=0,
    feature="tomo_id",
    output_motl=None,
    store_idx1="object_id",
    store_idx2="geom2",
    store_dist="geom4",
):
    motl_entry = cryomotl.Motl.load(motl_entry)
    motl_exit = cryomotl.Motl.load(motl_exit)

    features1 = np.unique(motl_entry.df.loc[:, feature])
    features2 = np.unique(motl_exit.df.loc[:, feature])

    if ~np.all(np.equal(features1, features2)):
        ValueError("Provided motls have different features sets!!!")

    traced_motl = cryomotl.Motl.create_empty_motl_df()

    for f in features1:
        # for f in np.array([2,274,405,423]):
        # for f in np.array([423]):
        # print(f)
        fm_entry = motl_entry.get_motl_subset(f, feature, reset_index=False)
        fm_exit = motl_exit.get_motl_subset(f, feature, reset_index=False)

        nfm_df = cryomotl.Motl.create_empty_motl_df()

        fm_size = fm_entry.df.shape[0]
        remain_entry = np.full((fm_size,), True)
        remain_exit = np.full((fm_size,), True)

        class_c = 1

        coord_entry = fm_entry.get_coordinates()
        coord_exit = fm_exit.get_coordinates()

        kdt_entry = sn.KDTree(coord_entry)
        kdt_exit = sn.KDTree(coord_exit)

        for i, current_point in enumerate(coord_exit):
            if ~remain_exit[i]:
                continue
            else:
                ch_m = cryomotl.Motl.create_empty_motl_df()  # create new chain motl df
                chain_id = 1  # assign chain id
                trace_chain = True
                p_idx = i
                used_idx = []
                # print(i)
                while trace_chain:
                    # take the particle from the exit list
                    # part_process = fm_exit.df.iloc[p_idx]

                    # add the same processed particle from entry list to the chain
                    ch_m = pd.concat([ch_m, fm_entry.df.iloc[[p_idx]]], ignore_index=True)

                    ch_m.loc[ch_m.index[-1], [store_idx2]] = chain_id
                    chain_id += 1

                    # remove currently processed point from both entry and exit
                    remain_entry[p_idx] = False
                    remain_exit[p_idx] = False
                    used_idx.append(p_idx)

                    # prepare coordinates
                    p_coord = coord_exit[p_idx, None, :]

                    if np.all(remain_entry == False):  # no remaining particles, end the chain
                        # np_idx = p_idx
                        np_idx = -1
                    else:
                        # search for the nearest active point
                        np_idx, np_dist = get_nn_dist(
                            kdt_entry,
                            p_coord,
                            max_distance,
                            min_distance,
                            remain_entry,
                            True,
                        )

                    if np_idx != -1:  # continue tracing
                        p_idx = np_idx
                        ch_m.loc[ch_m.index[-1], [store_dist]] = np_dist
                    else:  # end chain
                        ch_m.loc[:, store_idx1] = class_c
                        class_c += 1

                        if nfm_df.size != 0:  # check existing chains for connections
                            first_coord = (
                                ch_m.loc[ch_m.index[0], ["x", "y", "z"]].values
                                + ch_m.loc[ch_m.index[0], ["shift_x", "shift_y", "shift_z"]].values
                            )  # entry point
                            first_coord = first_coord.reshape(1, 3)
                            remain_entry[used_idx] = True
                            remain_exit[used_idx] = True
                            # check if this chain cannot be connected to already an existing one
                            # This can happen if the chain is started "in the middle"
                            nm_idx, nm_dist = get_nn_dist(
                                kdt_entry,
                                p_coord,
                                max_distance,
                                min_distance,
                                remain_entry,
                                False,
                            )
                            first_idx, first_dist = get_nn_dist(
                                kdt_exit,
                                first_coord,
                                max_distance,
                                min_distance,
                                remain_exit,
                                False,
                            )

                            remain_entry[used_idx] = False
                            remain_exit[used_idx] = False

                            # rather rare case where a single particle wants to connect to the same particle in a chain
                            if first_idx == nm_idx and first_idx != -1 and ch_m.shape[0] == 1:
                                if first_dist <= nm_dist:
                                    nm_idx = -1  # add only suffix
                                else:
                                    first_idx = -1  # add only prefix
                            elif first_idx != -1 and nm_idx != -1:
                                part1 = fm_exit.df.loc[fm_exit.df.index[first_idx], "subtomo_id"]
                                part2 = fm_entry.df.loc[fm_entry.df.index[nm_idx], "subtomo_id"]
                                cl1 = nfm_df.loc[nfm_df["subtomo_id"] == part1, store_idx1].values[0]
                                cl2 = nfm_df.loc[nfm_df["subtomo_id"] == part2, store_idx1].values[0]
                                if cl1 == cl2:
                                    if first_dist <= nm_dist:
                                        nm_idx = -1  # add only suffix
                                    else:
                                        first_idx = -1  # add only prefix

                            ch_changed = False  # default is no chain change

                            if first_idx != -1:  # appneding the chain after an existing one
                                ch_changed = add_chain_suffix(
                                    ch_m,
                                    fm_exit,
                                    nfm_df,
                                    first_idx,
                                    first_dist,
                                    store_idx1,
                                    store_idx2,
                                )

                            if nm_idx != -1:  # connecting the chain before an existing one

                                class_max = None

                                # they connect from both sides
                                if ch_changed:
                                    current_class = class_c - 1
                                    cl_max = np.max(ch_m[store_idx2].values)
                                    if cl_max > 1:
                                        class_max = (cl_max, current_class)

                                add_chain_prefix(
                                    ch_m,
                                    fm_entry,
                                    nfm_df,
                                    nm_idx,
                                    nm_dist,
                                    store_idx1,
                                    store_idx2,
                                    class_max=class_max,
                                )

                        nfm_df = pd.concat([nfm_df, ch_m])
                        trace_chain = False

        traced_motl = pd.concat([traced_motl, nfm_df])

    traced_motl = cryomotl.Motl(motl_df=traced_motl)

    if output_motl is not None:
        traced_motl.write_to_emfile(output_motl)

    return traced_motl
