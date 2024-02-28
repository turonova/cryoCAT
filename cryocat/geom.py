import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as srot
from cryocat.exceptions import UserInputError
import matplotlib.pyplot as plt
import os
from scipy.interpolate import InterpolatedUnivariateSpline

ANGLE_DEGREES_TOL = 10e-12


def load_dimensions(dims):
    """Loads tomogram dimensions from a file or nd.array.

    Parameters
    ----------
    dims : str
        Either a path to a file with the dimensions or nd.array. The shape of the input should
        be 1x3 (x y z) in case of one tomogram or Nx4 for multiple tomograms (tomo_id x y z). In case of file the
        separator is a space.

    Returns
    -------
    nd.array
        Dimensions of a tomogram in x, y, z (shape 1x3) or tomogram idx and corresponding dimensions
        (shape Nx4 where N is the number of tomograms)

    Raises
    ------
    UserInputError
        Wrong size of the input.

    """

    if os.path.isfile(dims):
        dimensions = pd.read_csv(dims, sep="\s+", header=None, dtype=float)
    elif isinstance(dims, pd.DataFrame):
        dimensions = dims
    else:
        dimensions = pd.DataFrame(dims)

    if dimensions.shape == (1, 3):
        dimensions.columns = ["x", "y", "z"]
    elif dimensions.shape[1] == 4:
        dimensions.columns = ["tomo_id", "x", "y", "z"]
    else:
        raise UserInputError(
            f"The dimensions should have shape of 1x3 or Nx4, where N is number of tomograms."
            f"Instead following shape was specified: {dimensions.shape}."
        )

    return dimensions


def load_angles(input_angles, angles_order="zxz"):
    """_summary_

    Parameters
    ----------
    input_angles : _type_
        _description_

    Returns
    -------
    _type_
        _description_

    Raises
    ------
    ValueError
        _description_

    """
    if isinstance(input_angles, str):
        angles = pd.read_csv(input_angles, header=None)
        if angles_order == "zzx":
            angles.columns = ["phi", "psi", "theta"]
        else:
            angles.columns = ["phi", "theta", "psi"]

        angles = angles.loc[:, ["phi", "theta", "psi"]].to_numpy()

    elif isinstance(input_angles, np.ndarray):
        angles = input_angles.copy()
    else:
        raise ValueError("The input_angles have to be either a valid path to a file or numpy array!!!")

    return angles


def spline_sampling(coords, sampling_distance):
    # Samples a spline specified by coordinates with a given sampling distance
    # Input:  coords - coordinates of the spline
    #         sampling_distance: sampling frequency in pixels
    # Output: coordinates of points on the spline

    # spline = UnivariateSpline(np.arange(0, len(coords), 1), coords.to_numpy())
    spline = InterpolatedUnivariateSpline(np.arange(0, len(coords), 1), coords.to_numpy())

    # Keep track of steps across whole tube
    totalsteps = 0

    for i, row in coords.iterrows():
        if i == 0:
            continue
        # Calculate projected distance between each point
        # TODO: check this issue
        dist = point_pairwise_dist(row, coords.iloc[i - 1])

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


def compare_rotations(angles1, angles2, c_symmetry=1, rotation_type="all"):
    """Compare the rotations between two sets of angles.

    Parameters
    ----------
    angles1 : list
        The first set of angles.
    angles2 : list
        The second set of angles.
    c_symmetry : int
        The degree of rotational symmetry. Defaults to 1.

    Returns
    -------
    tuple
        A tuple containing the following distances:
        - dist_degrees (float): The overall angular distance between the two sets of angles.
        - dist_degrees_normals (float): The angular distance between the normal vectors of the two sets of angles.
        - dist_degrees_inplane (float): The angular distance within the plane of rotation between the two sets of angles.

    """

    dist_degrees = angular_distance(angles1, angles2, c_symmetry=c_symmetry)[0]
    dist_degrees_normals, dist_degrees_inplane = cone_inplane_distance(angles1, angles2, c_symmetry=c_symmetry)

    if rotation_type == "all":
        return dist_degrees, dist_degrees_normals, dist_degrees_inplane
    elif rotation_type == "angular_distance":
        return dist_degrees
    elif rotation_type == "cone_distance":
        return dist_degrees_normals
    elif rotation_type == "in_plane_distance":
        return dist_degrees_inplane
    else:
        raise UserInputError(f"The rotation type {rotation_type} is not supported.")


def change_handedness_coordinates(coordinates, dimensions):
    """The change_handedness_coordinates function takes in a pandas dataframe of coordinates and the dimensions of the
        coordinate system. It then changes the handedness of those coordinates by subtracting each z-coordinate from
        dimension[2]. This is done because we want to change our coordinate system so that it has its origin at the top left
        corner, with positive x going right and positive y going down. The original coordinate system had its origin at
        bottom left, with positive x going right and positive y going up.

    Parameters
    ----------
    coordinates :
        Store the coordinates of the voxels
    dimensions :
        Determine the new z value

    Returns
    -------

        The coordinates with the z axis inverted
        Doc Author:
        Trelent

    """
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
        raise UserInputError("The input_normals have to be either pandas dataFrame or numpy array")

    # normalize vectors
    normals = normals / np.linalg.norm(normals, axis=1)[:, np.newaxis]
    theta = np.degrees(np.arctan2(np.sqrt(normals[:, 0] ** 2 + normals[:, 1] ** 2), normals[:, 2]))

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
    cone_angle = np.degrees(np.arccos(np.maximum(np.minimum(np.sum(vec1 * vec2, axis=1), 1.0), -1.0)))

    return cone_angle


def inplane_distance(input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1):
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

    inplane_angle = np.where(inplane_angle > 180.0, np.abs(inplane_angle - 360.0), inplane_angle)

    return inplane_angle


def cone_inplane_distance(input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1):
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


def angular_distance(input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1):
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


def sample_cone(cone_angle, cone_sampling, center=None, radius=1.0):
    # Creates "even" distribution on sphere. This is in many regards
    # just an approximation. However, it seems to work for not too extreme cases.
    # Source:
    # https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012

    if center is None:
        center = np.array([0.0, 0.0, 0.0])

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
    angles_order="zxz",
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
        cone_rotations = starting_rot * cone_rotations  # swapped order w.r.t. the quat_mult in matlab!
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


def angle_between_vectors(vectors1, vectors2):
    dot_products = np.einsum("ij,ij->i", vectors1, vectors2)
    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)

    cosines = dot_products / (norms1 * norms2)
    radians = np.arccos(np.clip(cosines, -1.0, 1.0))
    degrees = np.degrees(radians)

    return degrees


def compute_pairwise_angles(angles1, angles2, coord1, coord2, axis="z"):
    angles1 = np.atleast_2d(angles1)
    angles2 = np.atleast_2d(angles2)
    rot_angles1 = srot.from_euler("zxz", angles=angles1, degrees=True)

    angles_ref_to_zero = -angles1[:, [2, 1, 0]]  # swap psi and theta -> ["psi", "theta", "phi"]
    rot_to_zero = srot.from_euler("zxz", angles=angles_ref_to_zero, degrees=True)

    angles2_as_rotations = srot.from_euler("zxz", angles=angles2, degrees=True)
    # rot_angles2 = rot_to_zero * angles2_as_rotations
    rot_angles2 = rot_to_zero

    coord_vector = coord2 - coord1

    if axis.endswith("x"):
        vector = [1, 0, 0]
    elif axis.endswith("y"):
        vector = [0, 1, 0]
    elif axis.endswith("z"):
        vector = [0, 0, 1]
    else:
        raise UserInputError(f"Invalid axis epcification: {axis}. Allowed inputs are x, -x, y, -y, z, -z!")

    if axis.startswith("-"):
        vector = vector * -1

    rot_vectors1 = rot_angles1.apply(vector)
    rot_vectors2 = rot_angles2.apply(coord_vector)

    angles = angle_between_vectors(rot_vectors1, rot_vectors2)
    ##nn_rotations = srot.concatenate(nn_rotations)
    ##points_on_sphere = geom.visualize_rotations(nn_rotations, plot_rotations=False)
    ##angles = nn_rotations.as_euler("zxz", degrees=True)

    return angles


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


# cite https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on
def area_triangle(coords):
    # The cross product of two sides is a normal vector
    triangles = np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0], axis=1)
    # The norm of the cross product of two sides is twice the area
    return np.linalg.norm(triangles, axis=1) / 2


def ray_ray_intersection_3d(starting_points, ending_points):
    # N lines described as vectors
    Si = ending_points - starting_points

    # Normalize vectors
    ni = Si / np.linalg.norm(Si, axis=1)[:, np.newaxis]

    nx, ny, nz = ni[:, 0], ni[:, 1], ni[:, 2]

    # Calculate sums
    SXX = np.sum(nx**2 - 1)
    SYY = np.sum(ny**2 - 1)
    SZZ = np.sum(nz**2 - 1)
    SXY = np.sum(nx * ny)
    SXZ = np.sum(nx * nz)
    SYZ = np.sum(ny * nz)

    # Matrix S
    S = np.array([[SXX, SXY, SXZ], [SXY, SYY, SYZ], [SXZ, SYZ, SZZ]])

    CX = np.sum(
        starting_points[:, 0] * (nx**2 - 1) + starting_points[:, 1] * (nx * ny) + starting_points[:, 2] * (nx * nz)
    )
    CY = np.sum(
        starting_points[:, 0] * (nx * ny) + starting_points[:, 1] * (ny**2 - 1) + starting_points[:, 2] * (ny * nz)
    )
    CZ = np.sum(
        starting_points[:, 0] * (nx * nz) + starting_points[:, 1] * (ny * nz) + starting_points[:, 2] * (nz**2 - 1)
    )

    C = np.array([CX, CY, CZ])

    # Solve the system of linear equations
    P_intersect = np.linalg.solve(S, C)

    distances = np.zeros(starting_points.shape[0])

    for i in range(starting_points.shape[0]):
        ui = np.dot(P_intersect - starting_points[i, :], Si[i, :]) / np.dot(Si[i, :], Si[i, :])
        distances[i] = np.linalg.norm(P_intersect - starting_points[i, :] - ui * Si[i, :])

    return P_intersect, distances


def fit_circle_3d_pratt(coord):
    # Projection of coordinates on 2D plane
    coord = coord.T
    U, S, V = np.linalg.svd(coord, full_matrices=False)
    coord_proj = np.dot(U.T, coord)

    # Pratt method
    x = coord_proj[0, :]
    y = coord_proj[1, :]
    a = np.linalg.lstsq(np.vstack([x, y, np.ones_like(x)]).T, -(x**2 + y**2), rcond=None)[0]
    xc = -a[0]
    yc = -a[1]

    center_2d = np.array([xc, yc])

    # 3d center
    c_si = np.concatenate([center_2d, [0]])
    circle_center = np.matmul(U, c_si) / 2.0

    # Compute a radius
    coord = coord.T
    center_coord = np.tile(circle_center, (coord.shape[0], 1))
    euc_dist = np.sqrt(np.sum((coord - center_coord) ** 2, axis=0))
    circle_radius = np.mean(euc_dist)

    # confidence
    if np.all(center_coord == 0):
        confidence = -1
    else:
        confidence = 1

    return circle_center, circle_radius, confidence


def fit_circle_3d_taubin(coord):
    # Projection of coordinates on 2D plane
    coord = coord.T
    U, S, V = np.linalg.svd(coord, full_matrices=False)
    coord_proj = np.dot(U.T, coord)

    a = np.array([[0, 1], [0, 2], [1, 2]])
    cs = []
    cr = []

    for b in range(a.shape[0]):
        cp = coord_proj[a[b, :], :]
        center_2d, rr, confidence = fit_circle_2d(cp)

        # Check if the fit was successful
        if center_2d[0] == np.inf:
            circle_center = np.inf
            circle_radius = np.inf
            confidence = -1
            return circle_center, circle_radius, confidence

        # Estimated center
        cc = np.zeros(3)
        cc[a[b, 0]] = center_2d[0]
        cc[a[b, 1]] = center_2d[1]
        circle_center = np.matmul(U, cc)

        cs.append(circle_center)
        cr.append(rr)

    circle_radius, c_idx = min(zip(cr, range(len(cr))))

    circle_center = cs[c_idx]

    return circle_center, circle_radius, confidence


def fit_circle_2d(coord):
    XY = coord.T  # Transpose to match MATLAB's column-wise data format

    n = XY.shape[0]  # number of data points
    centroid = np.mean(XY, axis=0)  # the centroid of the data set

    # computing moments (note: all moments will be normed, i.e. divided by n)
    Mxx = 0
    Myy = 0
    Mxy = 0
    Mxz = 0
    Myz = 0
    Mzz = 0

    for i in range(n):
        Xi = XY[i, 0] - centroid[0]  # centering data
        Yi = XY[i, 1] - centroid[1]  # centering data
        Zi = Xi * Xi + Yi * Yi
        Mxy = Mxy + Xi * Yi
        Mxx = Mxx + Xi * Xi
        Myy = Myy + Yi * Yi
        Mxz = Mxz + Xi * Zi
        Myz = Myz + Yi * Zi
        Mzz = Mzz + Zi * Zi

    Mxx = Mxx / n
    Myy = Myy / n
    Mxy = Mxy / n
    Mxz = Mxz / n
    Myz = Myz / n
    Mzz = Mzz / n

    # computing the coefficients of the characteristic polynomial
    Mz = Mxx + Myy
    Cov_xy = Mxx * Myy - Mxy * Mxy
    A3 = 4 * Mz
    A2 = -3 * Mz * Mz - Mzz
    A1 = Mzz * Mz + 4 * Cov_xy * Mz - Mxz * Mxz - Myz * Myz - Mz * Mz * Mz
    A0 = Mxz * Mxz * Myy + Myz * Myz * Mxx - Mzz * Cov_xy - 2 * Mxz * Myz * Mxy + Mz * Mz * Cov_xy
    A22 = A2 + A2
    A33 = A3 + A3 + A3
    xnew = 0
    ynew = 1e20
    epsilon = 1e-12
    IterMax = 20

    # set default to 1
    confidence = 1

    # Newton's method starting at x=0
    for _ in range(IterMax):
        yold = ynew
        ynew = A0 + xnew * (A1 + xnew * (A2 + xnew * A3))
        if abs(ynew) > abs(yold):
            xnew = 0
            confidence = -1
            break
        Dy = A1 + xnew * (A22 + xnew * A33)
        xold = xnew
        xnew = xold - ynew / Dy
        if abs((xnew - xold) / xnew) < epsilon:
            break
        if _ >= IterMax:
            xnew = 0
            confidence = -1
        if xnew < 0:
            xnew = 0
            confidence = -1

    # computing the circle parameters
    DET = xnew * xnew - xnew * Mz + Cov_xy
    Center = np.array([(Mxz * (Myy - xnew) - Myz * Mxy), (Myz * (Mxx - xnew) - Mxz * Mxy)]) / DET / 2
    Par = np.concatenate([Center + centroid, [np.sqrt(np.dot(Center, Center) + Mz)]])
    circle_center = Par[:2]
    circle_radius = Par[2]

    return circle_center, circle_radius, confidence


def normalize_vector(vector):
    """Returns the unit vector of the vector."""
    return vector / np.linalg.norm(vector)


def vector_angular_distance(v1, v2):
    """Returns the angle in degrees between vectors 'v1' and 'v2'::"""
    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
