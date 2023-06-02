import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as srot
from cryocat.exceptions import UserInputError
import matplotlib.pyplot as plt

ANGLE_DEGREES_TOL = 10e-12


def change_handedness_coordinates(coordinates, dimensions):
    new_z = dimensions[2] - coordinates["z"]
    coordinates.loc[:, "z"] = new_z

    return coordinates


def change_handedness_orientation(orientation, convention="ZXZ"):
    # is actually dependent on the convention and what should be flipped
    # the method from cryomotl should be used instead...
    quats = orientation.as_quat()
    quats[:, 0:3] = -quats[:, 0:3]
    return srot.from_quat(quats)


def euler_angles_to_normals(angles):
    points = visualize_angles(angles, plot_rotations=False)
    n_length = np.linalg.norm(points)
    normalized_normal_vectors = points / n_length

    return normalized_normal_vectors


def normals_to_euler_angles(input_normals, output_order="zzx"):
    if isinstance(input_normals, pd.DataFrame):
        normals = input_normals.loc[:, ["x", "y", "z"]]
    elif isinstance(input_normals, np.ndarray):
        normals = input_normals
    else:
        raise UserInputError(
            "The input_normals have to be either pandas dataFrame or numpy array"
        )

    theta = np.degrees(
        np.arctan2(np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2), normals[:, 2])
    )

    psi = 90 + np.degrees(np.arctan2(normals[:, 1], normals[:, 0]))
    b_idx = np.where(np.arctan2(normals[:, 1], normals[:, 0]) == 0)
    psi[b_idx] = 0

    phi = np.random.rand(normals.shape[0]) * 360

    if output_order == "zzx":
        angles = np.column_stack((phi, psi, theta))
    else:
        angles = np.column_stack((phi, theta, psi))

    return angles


def quaternion_mult(qs1, qs2):
    mutliplied = []
    for q, q1 in enumerate(qs1):
        q2 = qs2[q, :]
        w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        i = q1[3] * q2[0] + q1[0] * q2[3] - q1[1] * q2[2] + q1[2] * q2[1]
        j = q1[3] * q2[1] + q1[0] * q2[2] + q1[1] * q2[3] - q1[2] * q2[0]
        k = q1[3] * q2[2] - q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3]
        mutliplied.append(np.array([i, j, k, w]))

    return np.vstack(mutliplied)


def quaternion_log(q):
    v_norm = np.linalg.norm(q[:, :3], axis=1)
    q_norm = np.linalg.norm(q, axis=1)

    tolerance = 10e-14

    new_scalar = []
    new_vector = []

    for i, _ in enumerate(q):
        if q_norm[i] < tolerance:
            # 0 quaternion - undefined
            new_scalar.append(0.0)
            new_vector.append([0.0, 0.0, 0.0])

        elif v_norm[i] < tolerance:
            # real quaternions - no imaginary part
            new_scalar.append(np.log(q_norm[i]))
            new_vector.append([0, 0, 0])
        else:
            vec = q[i, :3] / v_norm[i]
            new_scalar.append(np.log(q_norm[i]))
            vector = np.arccos(q[i, 3] / q_norm[i])
            vector = vec * vector
            new_vector.append(vector)

    new_vector = np.vstack(new_vector)

    new_scalar = np.array(new_scalar).reshape(q.shape[0], 1)
    return np.hstack([new_vector, new_scalar])


def cone_distance(input_rot1, input_rot2):
    point = [0, 0, 1.0]

    vec1 = np.array(input_rot1.apply(point), ndmin=2)
    vec2 = np.array(input_rot2.apply(point), ndmin=2)

    vec1_n = np.linalg.norm(vec1, axis=1)
    vec1 = vec1 / vec1_n[:, np.newaxis]
    vec2_n = np.linalg.norm(vec2, axis=1)
    vec2 = vec2 / vec2_n[:, np.newaxis]
    cone_angle = np.degrees(
        np.arccos(np.maximum(np.minimum(np.sum(vec1 * vec2, axis=1), 1.0), -1.0))
    )

    return cone_angle


def inplane_distance(
    input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1
):
    phi1 = np.array(input_rot1.as_euler(convention, degrees=degrees), ndmin=2)[:, 0]
    phi2 = np.array(input_rot2.as_euler(convention, degrees=degrees), ndmin=2)[:, 0]

    # Remove flot precision errors during conversion
    phi1 = np.where(phi1 < ANGLE_DEGREES_TOL, 0.0, phi1)
    phi2 = np.where(phi2 < ANGLE_DEGREES_TOL, 0.0, phi2)

    # From Scipy the phi is from [-180,180] -> change to [0.0,360]
    phi1 += 180.0
    phi2 += 180.0

    # Get the angular range for symmetry and divide the angles to be only in that range
    if c_symmetry > 1:
        sym_div = 360.0 / c_symmetry
        phi1 = np.mod(phi1, sym_div)
        phi2 = np.mod(phi2, sym_div)

    inplane_angle = np.abs(phi1 - phi2)

    inplane_angle = np.where(
        inplane_angle > 180.0, np.abs(inplane_angle - 360.0), inplane_angle
    )

    return inplane_angle


def cone_inplane_distance(
    input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1
):
    if isinstance(input_rot1, np.ndarray):
        rot1 = srot.from_euler(convention, input_rot1, degrees=degrees)
    else:
        rot1 = input_rot1

    if isinstance(input_rot2, np.ndarray):
        rot2 = srot.from_euler(convention, input_rot2, degrees=degrees)
    else:
        rot2 = input_rot2

    cone_angle = cone_distance(rot1, rot2)
    inplane_angle = inplane_distance(rot1, rot2, convention, degrees, c_symmetry)

    return cone_angle, inplane_angle


def angular_distance(
    input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1
):
    # Computes angular distance between 2 quaternions or two sets of Euler
    # angles with ZXZ convention. Formula is based on this post
    # https://math.stackexchange.com/questions/90081/quaternion-distance
    # Unlike other distance metrics based on ||x-y||=0 this one correctly
    # returns 1 if the two orientations are 180 degrees apart and 0 if they
    # represent same orientation (i.e. if q1=q and q2=-q the distance should be
    # 0, not 2, and the angle should be also 0).

    if isinstance(input_rot1, np.ndarray):
        rot1 = srot.from_euler(convention, input_rot1, degrees=degrees)
    else:
        rot1 = input_rot1

    if isinstance(input_rot2, np.ndarray):
        rot2 = srot.from_euler(convention, input_rot2, degrees=degrees)
    else:
        rot2 = input_rot2

    if c_symmetry > 1:
        angles1 = rot1.as_euler(convention, degrees=degrees)
        angles2 = rot2.as_euler(convention, degrees=degrees)
        sym_div = 360.0 / c_symmetry
        angles1[:, 0] = np.mod(angles1[:, 0], sym_div)
        angles2[:, 0] = np.mod(angles2[:, 0], sym_div)
        rot1 = srot.from_euler(convention, angles1, degrees=degrees)
        rot2 = srot.from_euler(convention, angles2, degrees=degrees)

    q1 = np.array(rot1.as_quat(), ndmin=2)
    q2 = np.array(rot2.as_quat(), ndmin=2)

    if q1.shape != q2.shape:
        print("The size of input rotations differ!!!")
        return

    angle = np.degrees(2 * np.arccos(np.abs(np.sum(q1 * q2, axis=1))))
    angle = angle.astype(float)

    dist = 1 - np.power(np.sum(q1 * q2, 1), 2)

    dist[dist < 10e-8] = 0

    return angle, dist


def number_of_cone_rotations(cone_angle, cone_sampling):
    # Theta steps
    theta_max = cone_angle / 2
    temp_steps = theta_max / cone_sampling
    theta_array = np.linspace(0, theta_max, round(temp_steps) + 1)
    arc = 2.0 * np.pi * (cone_sampling / 360.0)

    number_of_rotations = 2  # starting and ending angle

    # Generate psi angles
    for i, theta in enumerate(theta_array[(theta_array > 0) & (theta_array < 180)]):
        radius = np.sin(theta * np.pi / 180.0)  # Radius of circle
        circ = 2.0 * np.pi * radius  # Circumference
        number_of_rotations += np.ceil(circ / arc) + 1  # Number of psi steps

    return number_of_rotations


def sample_cone(
    cone_angle, cone_sampling, center=np.array([0.0, 0.0, 0.0]), radius=1.0
):
    # Creates "even" distribution on sphere. This is in many regards
    # just an approximation. However, it seems to work for not too extreme cases.
    # Source:
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012

    number_of_points = number_of_cone_rotations(cone_angle, cone_sampling)

    # golden angle in radians
    phi = np.pi * (3 - np.sqrt(5))
    cone_size = cone_angle / 180.0

    sampled_points = [[0.0, 0.0, 1.0]]
    for i in np.arange(1, number_of_points, 1):
        # z goes from 1 to -1 for 360 degrees (i.e., a full sphere), is less
        z = 1 - (i / (number_of_points - 1)) * cone_size

        sp_radius = np.sqrt(1 - z * z)

        # golden angle increment
        theta = phi * i
        x = np.cos(theta) * sp_radius
        y = np.sin(theta) * sp_radius

        x = x * radius + center[0]
        y = y * radius + center[1]
        z = z * radius + center[2]

        sampled_points.append(np.array([x, y, z]))

    return np.stack(sampled_points, axis=0)


def generate_angles(
    cone_angle,
    cone_sampling,
    inplane_angle=360.0,
    inplane_sampling=None,
    starting_angles=None,
    symmetry=1.0,
):
    points = sample_cone(cone_angle, cone_sampling)
    angles = normals_to_euler_angles(points, output_order="zxz")
    angles[:, 0] = 0.0

    starting_phi = 0.0

    # if case of no starting angles one can directly do cone_angles = angles
    # but going through rotation object will set the angles to the canonical set
    cone_rotations = srot.from_euler("zxz", angles=angles, degrees=True)

    if starting_angles is not None:
        starting_rot = srot.from_euler("zxz", angles=starting_angles, degrees=True)
        cone_rotations = (
            starting_rot * cone_rotations
        )  # swapped order w.r.t. the quat_mult in matlab!
        starting_phi = starting_angles[0]

    cone_angles = cone_rotations.as_euler("zxz", degrees=True)
    cone_angles = cone_angles[:, 1:3]

    # Calculate phi angles
    if inplane_sampling is None:
        inplane_sampling = cone_sampling

    if inplane_angle != 360.0:
        phi_max = min(360.0 / symmetry, inplane_angle)
    else:
        phi_max = inplane_angle / symmetry

    phi_steps = phi_max / inplane_sampling
    phi_array = np.linspace(0, phi_max, round(phi_steps) + 1)
    phi_array = phi_array[:-1]  # Final angle is redundant

    if phi_array.size == 0:
        phi_array = np.array([[0.0]])

    n_phi = np.size(phi_array)
    phi_array = phi_array + starting_phi

    # Generate angle list
    angular_array = np.concatenate(
        [
            np.tile(phi_array[:, np.newaxis], (cone_angles.shape[0], 1)),
            np.repeat(cone_angles, n_phi, axis=0),
        ],
        axis=1,
    )

    return angular_array


def visualize_rotations(
    rotations,
    plot_rotations=True,
    color_map=None,
    marker_size=20,
    alpha=1.0,
    radius=1.0,
):
    starting_point = np.array([0.0, 0.0, radius])
    new_points = np.array(rotations.apply(starting_point), ndmin=2)

    if plot_rotations:
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        if color_map is None:
            ax.scatter(
                new_points[:, 0],
                new_points[:, 1],
                new_points[:, 2],
                s=marker_size,
                alpha=alpha,
            )
        else:
            ax.scatter(
                new_points[:, 0],
                new_points[:, 1],
                new_points[:, 2],
                s=marker_size,
                alpha=alpha,
                c=color_map,
            )
            # plt.colorbar(color_map)

        ax.set_xlim3d(-radius, radius)
        ax.set_ylim3d(-radius, radius)
        ax.set_zlim3d(-radius, radius)

    return new_points


def visualize_angles(angles, plot_rotations=True, color_map=None):
    rotations = srot.from_euler("zxz", angles=angles, degrees=True)
    new_points = visualize_rotations(rotations, plot_rotations, color_map)

    return new_points


def fill_ellipsoid(box_size, ellipsoid_parameters):
    # Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0

    if isinstance(box_size, int):
        box_size = np.full((3,), box_size)
    elif isinstance(box_size, (tuple, list)):
        box_size = np.asarray(box_size)

    x_array = np.arange(0, box_size[0], 1)
    y_array = np.arange(0, box_size[1], 1)
    z_array = np.arange(0, box_size[2], 1)
    x, y, z = np.meshgrid(x_array, y_array, z_array, indexing="ij")

    A, B, C, D, E, F, G, H, I, J = ellipsoid_parameters
    vals = (
        A * x * x
        + B * y * y
        + C * z * z
        + 2 * D * x * y
        + 2 * E * x * z
        + 2 * F * y * z
        + 2 * G * x
        + 2 * H * y
        + 2 * I * z
        + J
    )

    mask = vals >= 0

    return mask


# http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit
# for arbitrary axes
def fit_ellipsoid(coord):
    x = coord[:, 0]
    y = coord[:, 1]
    z = coord[:, 2]
    D = np.array(
        [
            x * x + y * y - 2 * z * z,
            x * x + z * z - 2 * y * y,
            2 * x * y,
            2 * x * z,
            2 * y * z,
            2 * x,
            2 * y,
            2 * z,
            1 - 0 * x,
        ]
    )
    d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array(
        [
            [v[0], v[3], v[4], v[6]],
            [v[3], v[1], v[5], v[7]],
            [v[4], v[5], v[2], v[8]],
            [v[6], v[7], v[8], v[9]],
        ]
    )

    center = np.linalg.solve(-A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1.0 / np.abs(evals))
    radii *= np.sign(evals)

    return center, radii, evecs, v


def point_pairwise_dist(coord_1, coord_2):
    if coord_1.shape[0] == 1 and coord_2.shape[0] != 1:
        coord_1 = np.tile(coord_1, (coord_2.shape[0], 1))

    # Squares of the distances
    pairwise_dist = np.linalg.norm(coord_1 - coord_2, axis=1)

    pairwise_dist = np.where(isinstance(pairwise_dist, complex), 0.0, pairwise_dist)

    return pairwise_dist
