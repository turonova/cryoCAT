import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as srot
from cryocat.exceptions import UserInputError
import matplotlib.pyplot as plt
import os
import math
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.interpolate import splprep, splev
from scipy.optimize import fsolve

ANGLE_DEGREES_TOL = 10e-12


class Line:
    def __init__(self, starting_point, line_dir):
        self.p = starting_point
        self.dir = line_dir


class LineSegment(Line):
    def __init__(self, point1, point2):
        self.p = point1
        self.dir = normalize_vectors(point2 - point1)
        self.p_end = point2
        self.length = np.linalg.norm(point2 - point1)


def project_points_on_plane_with_preserved_distance(starting_point, normal, nn_points):
    """Project approximately coplanar points around a starting_point onto the 
    plane perpendicular to normal vector. The distances between projected nearest neighbors and
    starting point are preserved.

    Parameters
    ----------
    starting_point : ndarray 
        origin of plane specified by normal vector
    normal : ndarray
        normal vector to plane 
    nn_points : ndarray 
        nearest neighbors of starting_point

    Returns
    -------
    ndarray
        projected points on plane specified by starting_point and normal
    """

    # Compute the projection of each neighbor point onto the plane defined by the starting point and normal vector
    projection_lengths = np.dot(nn_points - starting_point, normal) / np.linalg.norm(normal)
    projected_points = nn_points - np.outer(projection_lengths, normal)

    # Compute the distances between the neighbor points and the starting point
    distances_to_starting_point = np.linalg.norm(nn_points - starting_point, axis=1)

    # Compute the distances between the projected points and the starting point
    distances_to_projected_point = np.linalg.norm(projected_points - starting_point, axis=1)

    # Compute the adjustment vectors
    adjustment_vectors = projected_points - starting_point

    # Compute the shifted points
    shifted_points = (
        starting_point
        + adjustment_vectors * (distances_to_starting_point / distances_to_projected_point)[:, np.newaxis]
    )

    # distances_to_projected_point = np.linalg.norm(shifted_points - starting_point, axis=1)

    return shifted_points


def align_points_to_xy_plane(points_on_plane, plane_normal=None):
    """Plane is rotated to be aligned with xy-plane.

    Parameters
    ----------
    points_on_plane : ndarray
        coplanar points
    plane_normal : ndarray, optional
        Plane normal. Defaults to None.
        If None, plane normal is estimated from points_on_plane

    Raises
    ------
    ValueError 
        One needs at least 3 points to specify a plane if plane normal is not given.

    Returns
    -------
    ndarray (n,3), ndarray (3,3)
        Points in xy-plane, corresponding rotation matrix
    """

    if plane_normal is None:
        if points_on_plane.shape[0] >= 3:
            # Calculate the normal vector of the plane
            v1 = points_on_plane[1] - points_on_plane[0]
            v2 = points_on_plane[2] - points_on_plane[0]
            normal = np.cross(v1, v2)
        else:
            raise ValueError(
                f"The plane has to be specified either by plane normal or at least three points lying on the plane!"
            )
    else:
        normal = plane_normal

    normal = normal / np.linalg.norm(normal)

    # Find the rotation matrix to align normal with z-axis
    z_axis = np.array([0, 0, 1])
    axis = np.cross(normal, z_axis)
    axis = axis / np.linalg.norm(axis)
    angle = np.arccos(np.dot(normal, z_axis))
    rotation_matrix = srot.from_rotvec(angle * axis).as_matrix()  # Construct rotation matrix

    # Apply rotation matrix to all points
    rotated_points = np.dot(rotation_matrix, points_on_plane.T).T
    # rotated_points = np.dot(rotation_matrix, plane_normal.T).T

    # return rotated_points[:, 0:2], rotation_matrix
    return rotated_points, rotation_matrix


def spline_sampling(coords, sampling_distance):
    """Samples a spline specified by coordinates with a given sampling distance

    Parameters
    ----------
    coords : ndarray
        coordinates of the spline
    sampling_distance : float
        sampling frequency in pixels

    Returns
    -------
    ndarray 
        coordinates of points on the spline
    """

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


def euler_angles_to_normals(angles):
    """Compute normal vectors pointing in z-direction from Euler angles.

    Parameers
    ---------
    angles : ndarray (n,3)
        n triplets of Euler angles

    Returns
    -------
    ndarray (n,3)
        Unit length z-normal vectors associated to input Euler angles.
    """
    points = visualize_angles(angles, plot_rotations=False)
    n_length = np.linalg.norm(points)
    normalized_normal_vectors = points / n_length

    return normalized_normal_vectors


def normals_to_euler_angles(input_normals, output_order="zxz"):
    """Given normal vectors pointing in z-direction in particle frames,
    compute choice of Euler angles.

    Parameters
    ----------
    input_normals : ndarray, pandas dataFrame
        z-normal vectors
    output_order : str, optional
        Euler angle convention. Defaults to "zxz".

    Raises
    ------
    UserInputError
        input_normals have to be either pandas dataFrame or numpy array.

    Returns
    -------
    ndarray : (n,3)
        n triplets of Euler angles in choses convention.
    """
    if isinstance(input_normals, pd.DataFrame):
        normals = input_normals.loc[:, ["x", "y", "z"]].values
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
    """Given arrays of quaternions in scalar-last convention, compute 
    array of products of unit quaternions.

    Parameters
    ----------
    qs1 : ndarray (n,4)
        n quaternions in scalar-last convention.
    qs2 : ndarray (n,4) 
        n quaternions in scalar-last convention.

    Returns
    -------
    ndarray (n,4)
        n quaternions in scalar-last convention.
        Row i is product of qs1[i] and qs2[i].
    
    """
    mutliplied = []
    for q, q1 in enumerate(qs1):
        q2 = qs2[q, :]
        w = q1[3] * q2[3] - q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2]
        i = q1[3] * q2[0] + q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1]
        j = q1[3] * q2[1] - q1[0] * q2[2] + q1[1] * q2[3] + q1[2] * q2[0]
        k = q1[3] * q2[2] + q1[0] * q2[1] - q1[1] * q2[0] + q1[2] * q2[3]
        mutliplied.append(np.array([i, j, k, w]))

    return np.vstack(mutliplied)


def quaternion_log(q):
    """Given array of unit scalar-last quaternions, compute array of
    unit-quaternion logarithms.

    Parameters
    ----------
    q : ndarray (n,4)
        Array of n unit quaternions in scalar-last convention.

    Returns
    -------
    ndarray (n,4)
        Array of n quaternion logarithms.
    """
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
    """Compute great-circle distance between z-normals corresponding to orientations
    as represented by input rotations. This corresponds to angular distance between cone-rotation
    portions of respective input rotations.

    Parameters
    ----------
    input_rot1 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle
    input_rot2 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle

    Returns
    -------
    float
        cone-distance in degrees
    """
    point = [0, 0, 1.0]

    vec1 = np.array(input_rot1.apply(point), ndmin=2)
    vec2 = np.array(input_rot2.apply(point), ndmin=2)

    vec1_n = np.linalg.norm(vec1, axis=1)
    vec1 = vec1 / vec1_n[:, np.newaxis]
    vec2_n = np.linalg.norm(vec2, axis=1)
    vec2 = vec2 / vec2_n[:, np.newaxis]
    cone_angle = np.degrees(np.arccos(np.maximum(np.minimum(np.sum(vec1 * vec2, axis=1), 1.0), -1.0)))

    return cone_angle


def get_axis_from_rotation(input_rotation, axis="z"):
    """Given an input rotation, compute the desired unit normal vector
    from the coordinate frame associated to the rotation.

    Parameters
    ----------
    input_rotation : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle
    axis : str, optional
        Desired coordinate direction. Defaults to "z".

    Raises
    ------
    ValueError
        Input must be valid scipy rotation object.

    Returns
    -------
    ndarray
        unit vector
    """

    matrix_rep = input_rotation.as_matrix()

    axes_dict = {"x": 0, "y": 1, "z": 2}

    if matrix_rep.shape == (3, 3):  # Single (3, 3) matrix
        ret_axis = matrix_rep[:, axes_dict[axis]]  # Extract column 1 for a single matrix
    elif matrix_rep.shape[1:] == (3, 3):  # Multiple (N, 3, 3) matrices
        ret_axis = matrix_rep[:, :, axes_dict[axis]]  # Extract column 1 for each (N, 3, 3) matrix
    else:
        raise ValueError("Input must be valid scipy rotation object.")

    return ret_axis


def inplane_distance(input_rot1, input_rot2, convention="zxz", degrees=True, c_symmetry=1):
    """Compute the angular distance between inplane-rotation portion of two given rotations.

    Parameters
    ----------
    input_rot1 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle.
    input_rot2 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle.
    convention : str, optional
        Euler angle convention. Defaults to "zxz".
    degrees : bool, optional
        Return angular distance in degrees (True) or radians (False). Defaults to True.
    c_symmetry : int, optional
        Rotational symmetry of underlying particles. Defaults to 1.

    Returns
    -------
    float
        Angular distance between inplane rotations.
    """
    phi1 = np.array(input_rot1.as_euler(convention, degrees=degrees), ndmin=2)[:, 0]
    phi2 = np.array(input_rot2.as_euler(convention, degrees=degrees), ndmin=2)[:, 0]

    # Remove flot precision errors during conversion
    phi1 = np.where(abs(phi1) < ANGLE_DEGREES_TOL, 0.0, phi1)
    phi2 = np.where(abs(phi2) < ANGLE_DEGREES_TOL, 0.0, phi2)

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
    """Compute angular distance between cone-rotations and inplane-rotations, respectively.

    Parameters
    ----------
    input_rot1 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle.
    input_rot2 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle.
    convention : str, optional
        Euler angle convention. Defaults to "zxz".
    degrees :bool, optional
        Return angular distance in degrees (True) or radians (False). Defaults to True.
    c_symmetry : int, optional
        Rotational symmetry of underlying particles. Defaults to 1.

    Returns
    -------
    float
        Angular distance between cone-rotations
    float
        angular distance between inplane rotations.
    """
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
    """Compute angular distance between two rotations. 
    Formula is based on this post
    https://math.stackexchange.com/questions/90081/quaternion-distance

    Parameters
    ----------
    input_rot1 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle.
    input_rot2 : scipy.spatial.transform.Rotation object
        Rotation object describing orientation of particle.
    convention : str, optional
        Euler angle convention. Defaults to "zxz".
    degrees : bool, optional
        Return angular distance in degrees (True) or radians (False). Defaults to True.
    c_symmetry : int, optional
        Rotational symmetry of underlying particles. Defaults to 1.

    Returns
    -------
    float
        Angular distance between input rotations.

    Examples
    --------
    >>> rot1 = srot.from_euler("zxz", [0, 0, 0], degrees=True)
    >>> rot2 = srot.from_euler("zxz", [45, 45, 0], degrees=True)
    >>> angular_distance(rot1, rot2)
    45.0
    """

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
    """Calculates the number of rotations required for a sampling process
    of cone-angles based on a sampling interval.

    Parameters
    ----------
    cone_angle : float
        The total cone-angle in degrees.
    cone_sampling : float
        The angular sampling interval in degrees.

    Returns
    -------
    int
        The total number of rotations required for the sampling process.
    """
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
    """Creates an "even" distibution on sphere. Works for tame cases.
    Source:
    https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere/26127012#26127012

    Parameters
    ----------
    cone_angle : float
        Angle for sampling of cone-angles (refers to range of z-normals of particles).
    cone_sampling : float
        Frequency for cone sampling.
    center : ndarray, optional
        Center of sphere to be sampled. Defaults to None.
    radius : float, optional
        Radius of sphere to be sampled. Defaults to 1.0.

    Returns
    -------
    ndarray
        Samples on sphere.
    """

    if center is None:
        center = np.array([0.0, 0.0, 0.0])

    number_of_points = number_of_cone_rotations(cone_angle, cone_sampling)

    # golden angle in radians
    phi = np.pi * (3 - np.sqrt(5))
    cone_size = cone_angle / 180.0

    north_pole = [0.0, 0.0, radius]
    if center is not None:
        north_pole = north_pole + center
    sampled_points = [north_pole]
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
    angle_order="zxz",
):
    """Compute Euler angles from sample for normal vectors on sphere.
    Sphere sample corresponds to cone-angles.

    Parameters
    ----------
    cone_angle : float
        Angle for sampling of cone-angles (refers to range of z-normals of particles).
    cone_sampling : float
        Frequency for cone sampling.
    inplane_angle : float, optional
        Desired inplane-angles for particle orientations. Defaults to 360.0.
    inplane_sampling : float, optional
        Frequency for sampling of inplane-angles. Defaults to None.
    starting_angles : ndarray , optional
        Triplet of Euler angles in convention as spefified by angle_order. Defaults to None.
    symmetry : float, optional
        Refers to rotational symmetry of particles. Defaults to 1.0.
    angle_order : str, optional
        Convention for Euler angles. Defaults to "zxz".

    Returns
    -------
    ndarray
        Sample of Euler angles.
    """
    points = sample_cone(cone_angle, cone_sampling)
    angles = normals_to_euler_angles(points, output_order=angle_order)
    angles[:, 0] = 0.0

    starting_phi = 0.0

    # if case of no starting angles one can directly do cone_angles = angles
    # but going through rotation object will set the angles to the canonical set
    cone_rotations = srot.from_euler(angle_order, angles=angles, degrees=True)

    if starting_angles is not None:
        starting_rot = srot.from_euler(angle_order, angles=starting_angles, degrees=True)
        cone_rotations = starting_rot * cone_rotations  # swapped order w.r.t. the quat_mult in matlab!
        starting_phi = starting_angles[0]

    cone_angles = cone_rotations.as_euler(angle_order, degrees=True)
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
    """Compute z-normals of input rotations. 
    If desried, generate plot depicting z-normals of input rotations.

    Parameters
    ----------
    rotations : array of scipy.spatial.transform.Rotation objects
        Orientations to be visualized
    plot_rotations : bool, optional
        If True, plot is generated. Defaults to True.
    color_map : str, optional
        Specify colormap for plot. Defaults to None.
    marker_size : int, optional
        Specify marker size for plot. Defaults to 20.
    alpha : float, optional
        Specify alpha parameter for plot. Defaults to 1.0.
    radius : float, optional
        Specify size of sphere for visualization. Defaults to 1.0.

    Returns
    -------
    ndarray (n,3)
        Array of z-normals.
    """
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
    """Compute the angle (in degrees) between corresponding pairs of vectors in two arrays.

    Parameters
    ----------
    vectors1 : ndarray (n, d)
        Each row represents a d-dimensional vector.
    vectors2 : ndarray (n, d)
        Each row represents a d-dimensional vector.

    Returns
    -------
    ndarray (n,)
        Array containing the angles (in degrees) between corresponding vectors.
     
    Examples
    --------
    >>> vectors1 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    >>> vectors2 = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
    >>> angle_between_vectors(vectors1, vectors2)
    array([90., 90., 90.])
    """
    dot_products = np.einsum("ij,ij->i", vectors1, vectors2)
    norms1 = np.linalg.norm(vectors1, axis=1)
    norms2 = np.linalg.norm(vectors2, axis=1)

    cosines = dot_products / (norms1 * norms2)
    radians = np.arccos(np.clip(cosines, -1.0, 1.0))
    degrees = np.degrees(radians)

    return degrees


def visualize_angles(angles, plot_rotations=True, color_map=None):
    """Compute z-normals of input orientations as described using Euler angles in zxz-convention. 
    If desried, generate plot depicting z-normals of input orientations.
    
    Parameters
    ---------- 
    angles : ndarray (n, 3) 
        Array of triplets of Euler angles in zxz-convention.
    plot_rotations : bool, optional
        If True, plot is generated. Defaults to True.
    color_map : str, optional
        Specify colormap for plot. Defaults to None.

    Returns
    -------
    ndarray (n,3)
        Array of z-normals.
    """
    rotations = srot.from_euler("zxz", angles=angles, degrees=True)
    new_points = visualize_rotations(rotations, plot_rotations, color_map)

    return new_points


def fill_ellipsoid(box_size, ellipsoid_parameters):
    """Fills a 3D space defined by `box_size` with a boolean mask where an ellipsoid defined by `ellipsoid_parameters`
    is located.

    Parameters
    ----------
    box_size : int or array_like
        Size of the box in which the ellipsoid will be placed. If an integer is provided, it is
        interpreted as the size for all three dimensions. If a tuple or list is provided, it should contain
        three integers defining the dimensions of the box.
    ellipsoid_parameters : array_like
        Coefficients for the general ellipsoid equation:
        Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz + J = 0
        Should contain ten elements corresponding to A, B, C, D, E, F, G, H, I, J respectively.

    Returns
    -------
    numpy.ndarray
        A 3D boolean array where True values represent the points inside or on the surface of the ellipsoid.

    Examples
    --------
    >>> box_size = 10
    >>> ellipsoid_parameters = (1, 1, 1, 0, 0, 0, 0, 0, 0, -100)
    >>> mask = fill_ellipsoid(box_size, ellipsoid_parameters)
    >>> mask.shape
    (10, 10, 10)
    """

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


def fit_ellipsoid(coord):
    """Fit an ellipsoid to a set of 3D coordinates. It is based on
    http://www.mathworks.com/matlabcentral/fileexchange/24693-ellipsoid-fit

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) where each row represents the x, y, z coordinates.

    Returns
    -------
    center : ndarray
        The center of the ellipsoid (x, y, z coordinates).
    radii : ndarray
        Radii of the ellipsoid along the principal axes.
    evecs : ndarray
        The eigenvectors corresponding to the principal axes of the ellipsoid.
    v : ndarray
        The 1D array of the ellipsoid parameters used to form the quadratic form.

    Notes
    -----
    This function fits an ellipsoid to a set of points by solving a linear least squares problem to estimate the
    parameters of the ellipsoid's equation in its algebraic form.

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> center, radii, evecs, _ = fit_ellipsoid(points)
    >>> center
    array([x_center, y_center, z_center])
    >>> radii
    array([radius_x, radius_y, radius_z])
    >>> evecs
    array([[evec1_x, evec1_y, evec1_z],
           [evec2_x, evec2_y, evec2_z],
           [evec3_x, evec3_y, evec3_z]])
    """

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


def point_ellipsoid_distance(p, params):
    """Computes the shortest distance from a point p to the surface of an ellipsoid.

    Parameters:
    -----------
    p : ndarray (3,)
        The 3D point in space.
    params : ndarray (26,)
        The ellipsoid parameters in the following order:
        ["cx", "cy", "cz", "rx", "ry", "rz",
         "ev1x", "ev1y", "ev1z", "ev2x", "ev2y", "ev2z",
         "ev3x", "ev3y", "ev3z", "p1", ..., "p10"]

    Returns:
    --------
    float
        The shortest distance from the point to the ellipsoid surface.
    """
    # Extract ellipsoid parameters
    center = np.array(params[:3])  # (cx, cy, cz)
    radii = np.array(params[3:6])  # (rx, ry, rz)
    evecs = np.array(params[6:15]).reshape(3, 3)  # 3x3 eigenvector matrix

    # Transform point to local ellipsoid coordinates
    p_local = np.dot(evecs.T, (p - center))

    # Function to solve for lambda (scaling factor)
    def scale_equation(lmbda):
        scaled = p_local / (1 + lmbda)
        return np.sum((scaled / radii) ** 2) - 1

    # Solve for λ numerically
    lambda_solution = fsolve(scale_equation, 0)[0]

    # Compute closest point on ellipsoid in local space
    closest_local = p_local / (1 + lambda_solution)

    # Transform back to global coordinates
    closest_global = np.dot(evecs, closest_local) + center

    # Compute Euclidean distance from point to closest surface point
    return np.linalg.norm(p - closest_global)


def point_pairwise_dist(coord_1, coord_2):
    """Calculate the pairwise Euclidean distance between two sets of coordinates.

    Parameters
    ----------
    coord_1 : ndarray
        An array of shape (N, D) where N is the number of points and D is the dimensionality of each point.
        If N=1, the single point is broadcasted to match the number of points in coord_2.
    coord_2 : ndarray
        An array of shape (M, D) where M is the number of points and D is the dimensionality of each point. If coord_1
        has N>1, then M has to be equal to N.

    Returns
    -------
    pairwise_dist : ndarray
        An array of shape (max(N, M),) containing the Euclidean distances between each pair of points from coord_1
        and coord_2.

    Notes
    -----
    If the input arrays have complex numbers, the distance calculation defaults to 0.0 for those pairs.
    """

    if coord_1.shape[0] == 1 and coord_2.shape[0] != 1:
        coord_1 = np.tile(coord_1, (coord_2.shape[0], 1))

    coord_1 = np.atleast_2d(coord_1)
    coord_2 = np.atleast_2d(coord_2)
    # Squares of the distances
    pairwise_dist = np.linalg.norm(coord_1 - coord_2, axis=1)

    pairwise_dist = np.where(isinstance(pairwise_dist, complex), 0.0, pairwise_dist)

    return pairwise_dist


def area_triangle(coords):
    """Calculate the area of a triangle given its vertex coordinates. See
    https://stackoverflow.com/questions/71346322/numpy-area-of-triangle-and-equation-of-a-plane-on-which-triangle-lies-on

    Parameters
    ----------
    coords : ndarray
        An array of shape (3, 3) where each row represents a vertex of the triangle, and each vertex is given by three
        coordinates (x, y, z).

    Returns
    -------
    float
        The area of the triangle.

    Examples
    --------
    >>> coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    >>> area_triangle(coords)
    0.5
    """

    # The cross product of two sides is a normal vector
    triangles = np.cross(coords[:, 1] - coords[:, 0], coords[:, 2] - coords[:, 0])
    # The norm of the cross product of two sides is twice the area
    return np.linalg.norm(triangles) / 2


def ray_ellipsoid_intersection_3d(point, normal, ellipsoid_params):
    """Compute the intersection between a ray starting at point in direction of normal and 
    an ellipsoid specified by ellipsoid_params.

    Parameters
    ----------
    point : ndarray (3,)
        Point in 3D describing origin of ray.
    normal : ndarray (3,)
        Normal vector describing direction of ray.
    ellipsoid_params : ndarray, list or tuple of 10 floats
        Coefficients describing quadratic form of ellipsoid.

    Returns
    -------
    tuple (p1, p2, d1, d2, is_inside) with:
        p1: ndarray describing closest intersection, or NaN.
        p2: ndarray describing intersection, or NaN.
        d1: float (distance between point and p1), or NaN.
        d2: float (distance between point and p2), or NaN.
        is_inside: bool, true if point lies inside the ellipsoid.
    """
    # Extract line parameters
    x, y, z = point[0], point[1], point[2]
    n1, n2, n3 = normal[0], normal[1], normal[2]

    # Extract ellipsoid parameters
    a, b, c, d, e, f, g, h, i, j = ellipsoid_params

    # Calculate coefficients A, B, C for the quadratic equation
    A = a * n1**2 + b * n2**2 + c * n3**2 + 2 * d * n1 * n2 + 2 * e * n1 * n3 + 2 * f * n3 * n2
    B = 2 * (
        a * x * n1
        + b * y * n2
        + c * z * n3
        + d * x * n2
        + d * y * n1
        + e * x * n3
        + e * z * n1
        + f * z * n2
        + f * y * n3
        + g * n1
        + h * n2
        + i * n3
    )
    C = j + a * x**2 + b * y**2 + c * z**2 + 2 * (i * z + h * y + g * x + d * x * y + e * x * z + f * z * y)

    # Discriminant
    D = B**2 - 4 * A * C

    # Initialize results
    p1, p2 = None, None
    is_inside = False

    if D < 0:  # No intersection
        p1, p2 = np.nan, np.nan
        d1, d2 = np.nan, np.nan
    elif D == 0:  # One intersection point
        t1 = -B / (2 * A)
        p1 = point + t1 * normal
        d1 = np.sign(t1) * np.linalg.norm(p1 - point)  # assigning the correct sign
        p2 = np.nan
        d2 = np.nan
    else:  # Two intersection points
        t1 = (-B + np.sqrt(D)) / (2 * A)
        t2 = (-B - np.sqrt(D)) / (2 * A)

        p1 = point + t1 * normal
        p2 = point + t2 * normal

        d1 = np.sign(t1) * np.linalg.norm(p1 - point)  # assigning the correct sign
        d2 = np.sign(t2) * np.linalg.norm(p2 - point)  # assigning the correct sign

        ps = np.column_stack((p1, p2))
        distances = np.asarray([d1, d2])

        if d1 < 0 and d2 > 0:
            is_inside = True
            pi = 1
        elif d1 > 0 and d2 < 0:
            is_inside = True
            pi = 0
        elif d1 < 0 and d2 < 0:
            pi = np.argmin(abs(distances))
        else:
            pi = np.argmin(distances)

        p1 = ps[:, pi]
        p2 = ps[:, 1 - pi]
        d1 = distances[pi]
        d2 = distances[1 - pi]

    return p1, p2, d1, d2, is_inside


def ray_ray_intersection_3d(starting_points, ending_points):
    """Calculate the intersection point and distances from the intersection to each line for a set of 3D rays.

    Parameters
    ----------
    starting_points : ndarray
        An array of shape (N, 3) representing the starting points of N lines in 3D space.
    ending_points : ndarray
        An array of shape (N, 3) representing the ending points of N lines in 3D space.

    Returns
    -------
    P_intersect : ndarray
        A 1D array of shape (3,) containing the coordinates of the intersection point.
    distances : ndarray
        A 1D array of shape (N,) containing the distances from the intersection point to each line.

    Notes
    -----
    This function assumes that all lines are somewhat close to intersecting at a common point and uses a least squares
    approach to find the best intersection point. The function may not be suitable for parallel lines or lines that
    do not converge.
    """

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


def rotate_points_rodrigues(P, n0, n1):
    """Rotates points by the rotation defined by two vectors.

    Parameters
    ----------
    P : ndarray
        Array containing point(s) to be rotated. Can be a 1D array for a single point or a 2D array for multiple points.
    n0 : ndarray
        Initial vector, before rotation. Must be a 1D array of 3 elements.
    n1 : ndarray
        Final vector, after rotation. Must be a 1D array of 3 elements.

    Returns
    -------
    P_rot : ndarray
        Array of rotated points. Same shape as input array P.

    Notes
    -----
    This function computes the rotation matrix that rotates vector n0 to align with vector n1 and applies this rotation
    to point(s) P. The rotation is performed using the Rodrigues' rotation formula, facilitated by scipy's spatial
    transformations.

    Examples
    --------
    >>> P = np.array([1, 0, 0])
    >>> n0 = np.array([1, 0, 0])
    >>> n1 = np.array([0, 1, 0])
    >>> rotate_points_rodrigues(P, n0, n1)
    array([[0., 1., 0.]])
    """

    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[np.newaxis, :]

    # Normalize vectors n0 and n1
    n0 = n0 / np.linalg.norm(n0)
    n1 = n1 / np.linalg.norm(n1)

    # Calculate the axis of rotation (k) and angle (theta)
    k = np.cross(n0, n1)
    k = k / np.linalg.norm(k)
    theta = np.arccos(np.clip(np.dot(n0, n1), -1.0, 1.0))  # clip to avoid floating point errors

    # Create the scipy rotation object
    r = srot.from_rotvec(theta * k)

    # Apply rotation to each point in P
    P_rot = r.apply(P)

    return P_rot


def project_3d_points_on_2d_plane_normal_aligned(coord, target_direction=None):
    """Projects 3D points onto a 2D plane that is aligned with a specified normal direction.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) containing N points in 3D space.
    target_direction : array_like, optional
        A 3-element array specifying the direction to which the normal of the 2D plane should be aligned.
        If None, defaults to [0, 0, 1], aligning the plane with the z-axis. Defaults to None.

    Returns
    -------
    coord_proj : ndarray
        An array of shape (N, 2) containing the 2D coordinates of the projected points.
    coord_mean : ndarray
        A 1D array of length 3 representing the mean of the original coordinates.
    normal : ndarray
        A 1D array of length 3 representing the normal vector of the plane onto which the points are projected.

    Notes
    -----
    The function first centers the points by subtracting their mean. It then uses Singular Value Decomposition (SVD)
    to find the principal components of the points. The smallest singular vector (normal to the plane of best fit)
    is used. The points are then rotated to align this normal with
    the target direction, effectively projecting them onto a new 2D plane.
    """

    coord_mean = coord.mean(axis=0)
    coord_centered = coord - coord_mean
    _, _, V = np.linalg.svd(coord_centered)
    normal = V[2, :]

    if target_direction is None:
        target_direction = [0, 0, 1]
    coord_proj = rotate_points_rodrigues(coord_centered, normal, target_direction)

    return coord_proj, coord_mean, normal


def project_3d_points_on_2d_plane_variance_based(coord):
    """Projects 3D points onto a 2D plane using variance-based method via Singular Value Decomposition (SVD).

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) where n is the number of 3D points.

    Returns
    -------
    coord_proj : ndarray
        The projected coordinates of the points onto the 2D plane, of shape (N, 2).
    U : ndarray
        The matrix containing the left singular vectors of the decomposition, used to project the points.

    Notes
    -----
    The function performs dimensionality reduction by projecting the original 3D points onto the 2D plane that captures
    the most variance in the data. This is achieved using SVD, which decomposes the input matrix into its singular
    vectors and singular values.

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> projected_points, _ = project_3d_points_on_2d_plane_variance_based(points)
    >>> print(projected_points)
    """

    coord = coord.T
    U, _, _ = np.linalg.svd(coord, full_matrices=False)
    coord_proj = np.dot(U.T, coord)

    return coord_proj, U


def fit_circle_3d_lsq(coord):
    """Fit a circle to 3D points using least squares optimization.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) containing the 3D coordinates of the points.

    Returns
    -------
    circle_center : ndarray
        A 1D array of length 3 containing the x, y, z coordinates of the fitted circle's center.
    circle_radius : float
        The radius of the fitted circle.
    residual_error : float
        Sum of the squared residuals of the fit.

    Notes
    -----
    This function projects 3D points onto a 2D plane that is normal to their average normal vector. It then fits a
    circle in 2D and transforms the center back to the 3D space.
    """

    coord_xy, coord_mean, normal = project_3d_points_on_2d_plane_normal_aligned(coord)
    xc, yc, circle_radius, residual_error = fit_circle_2d_lsq(coord_xy[:, 0], coord_xy[:, 1])

    circle_center = rotate_points_rodrigues(np.array([xc, yc, 0]), [0, 0, 1], normal) + coord_mean
    circle_center = circle_center.flatten()

    return circle_center, circle_radius, residual_error


def fit_circle_2d_lsq(x, y, w=None):
    """Fit a circle to 2D points using the least squares method. The method was taken from
    https://meshlogic.github.io/posts/jupyter/curve-fitting/fitting-a-circle-to-cluster-of-3d-points/

    Parameters
    ----------
    x : array_like
        X-coordinates of the points.
    y : array_like
        Y-coordinates of the points.
    w : array_like, optional
        Weights for each point. If provided, must be the same length as `x` and `y`. Defaults to None.

    Returns
    -------
    xc : float
        X-coordinate of the fitted circle's center.
    yc : float
        Y-coordinate of the fitted circle's center.
    r : float
        Radius of the fitted circle.
    error : float
        Sum of the squared residuals of the fit.

    Notes
    -----
    This function fits a circle in 2D to a set of points (x, y) by solving the
    weighted least squares problem if weights `w` are provided. If no weights are
    provided, it solves the ordinary least squares problem.

    Examples
    --------
    >>> x = np.array([1, 2, 3])
    >>> y = np.array([4, 5, 6])
    >>> xc, yc, r, error = fit_circle_2d_lsq(x, y)
    """

    if w is None:
        w = []

    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2

    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W, A)
        b = np.dot(W, b)

    # Solve by method of least squares
    # c = np.linalg.lstsq(A,b,rcond=None)[0]
    c, residuals, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # Get circle parameters from solution c
    xc = c[0] / 2
    yc = c[1] / 2
    r = np.sqrt(c[2] + xc**2 + yc**2)

    # Calculate least squares error (sum of squared residuals)
    error = np.sum(residuals)

    return xc, yc, r, error


def fit_circle_3d_pratt(coord):
    """Fit a circle to a set of 3D points using Pratt's method after projecting them onto a 2D plane.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) containing N points in 3D space.

    Returns
    -------
    circle_center : ndarray
        A 1D array of length 3 representing the center of the circle in 3D space.
    circle_radius : float
        The radius of the fitted circle.
    confidence : int
        Confidence indicator, returns 1 if the center is not at the origin, otherwise -1.

    Notes
    -----
    The function first projects the 3D points onto a 2D plane using a variance-based method. It then applies
    Pratt's method to these 2D points to fit a circle. The center of the circle is then transformed back to
    3D space. The radius is calculated as the mean Euclidean distance from the 3D points to the estimated
    center. The confidence is a simple check on the location of the center.

    Examples
    --------
    >>> points = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> center, radius, conf = fit_circle_3d_pratt(points)
    >>> print(center, radius, conf)
    """

    # Projection of coordinates on 2D plane
    coord_proj, U = project_3d_points_on_2d_plane_variance_based(coord)

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
    """Fit a circle to 3D points using Taubin's method projected onto a 2D plane.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 3) representing the coordinates of the 3D points.

    Returns
    -------
    circle_center : ndarray
        A 1D array of length 3 representing the center of the fitted circle in 3D space.
    circle_radius : float
        The radius of the fitted circle.
    confidence : float
        A confidence measure for the circle fitting. Returns -1 if the fitting fails.

    Notes
    -----
    The function projects 3D points onto a 2D plane using a variance-based method,
    then fits a circle in 2D using Newton's method. The best fitting circle is selected
    based on the minimum radius criterion among possible circle fits.

    Examples
    --------
    >>> coord = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    >>> center, radius, conf = fit_circle_3d_taubin(coord)
    >>> print(center, radius, conf)
    """

    # Projection of coordinates on 2D plane
    coord_proj, U = project_3d_points_on_2d_plane_variance_based(coord)

    a = np.array([[0, 1], [0, 2], [1, 2]])
    cs = []
    cr = []

    for b in range(a.shape[0]):
        cp = coord_proj[a[b, :], :]
        center_2d, rr, confidence = fit_circle_2d_newton(cp)

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


def fit_circle_2d_newton(coord):
    """Fit a circle to 2D data using Newton's method.

    Parameters
    ----------
    coord : ndarray
        An array of shape (N, 2) containing the x and y coordinates of the data points.

    Returns
    -------
    circle_center : ndarray
        A 1D array containing the x and y coordinates of the circle's center.
    circle_radius : float
        The radius of the fitted circle.
    confidence : int
        Confidence level of the fit, 1 if successful, -1 if the fit failed.

    Notes
    -----
    The function implements a numerical method to fit a circle to a set of 2D points by minimizing the algebraic
    distance to the circle. The method used is based on the algebraic form of a circle and Newton's optimization
    method to find the circle parameters that best fit the data.

    Examples
    --------
    >>> points = np.array([[1, 2], [3, 4], [5, 6]])
    >>> center, radius, conf = fit_circle_2d_newton(points)
    >>> print(center, radius, conf)
    """

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
    """Normalize a vector.

    Parameters
    ----------
    vector : array_like
        Input vector to be normalized.

    Returns
    -------
    ndarray
        Normalized vector with the same direction but with a norm of 1.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([2, 3, 6])
    >>> normalize_vector(v)
    array([0.26726124, 0.40089186, 0.80278373])
    """

    return vector / np.linalg.norm(vector)


def normalize_vectors(v):
    """Normalize each vector, handling both single vectors and arrays of vectors.

    Parameters
    ----------
    v : ndarray (n,d)

    Returns
    -------
    ndarray (n,d)
        Array of normalized vectors.

    Examples
    --------
    >>> import numpy as np
    >>> v = np.array([[1, 2, 3], [4, 5, 6]])
    >>> normalize_vectors(v)
    array([[0.26726124, 0.53452248, 0.80178373],^
              [0.45584231, 0.56980288, 0.68376346]])
    """
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / norm


# TODO: This function should not be necessary due to "angle_between_vectors"
def angle_between_n_vectors(v1, v2):
    """Compute the angle (in degrees) between corresponding pairs of vectors in two arrays.

    Parameters
    ----------
    v1 : ndarray (n, d)
        Each row represents d-dimensional vector.
    v2 : ndarray (n, d)
        Each row represents d-dimensional vector.

    Returns
    -------
    ndarray (n,)
        Array containing the angles (in degrees) between corresponding vectors.

    Examples
    --------
    >>> import numpy as np
    >>> v1 = np.array([[1, 0, 0], [0, 1, 0]])
    >>> v2 = np.array([[0, 1, 0], [1, 0, 0]])
    >>> angle_between_n_vectors(v1, v2)
    array([90., 90.])
    """

    # Ensure both vectors are normalized
    v1_u = normalize_vectors(v1)
    v2_u = normalize_vectors(v2)

    # Compute dot product element-wise, handling both (3,) and (N, 3) shapes
    dot_product = np.einsum("ij,ij->i", v1_u, v2_u) if v1.ndim > 1 else np.dot(v1_u, v2_u)

    # Clip values to avoid numerical errors outside the valid range for arccos
    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    return angle


def vector_angular_distance(v1, v2):
    """Calculate the angular distance between two vectors in degrees.

    Parameters
    ----------
    v1 : array_like
        First input vector.
    v2 : array_like
        Second input vector.

    Returns
    -------
    float
        The angular distance between `v1` and `v2` in degrees.

    Examples
    --------
    >>> v1 = [1, 0, 0]
    >>> v2 = [0, 1, 0]
    >>> vector_angular_distance(v1, v2)
    90.0
    """

    v1_u = normalize_vector(v1)
    v2_u = normalize_vector(v2)
    return np.degrees(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))


def vector_angular_distance_signed(u, v, n=None):
    """Compute the signed angular distance between two vectors.

    Parameters
    ----------
    u : array_like
        First input vector.
    v : array_like
        Second input vector.
    n : array_like, optional
        Normal vector to the plane containing `u` and `v`. If not provided, the function
        computes the unsigned angular distance.

    Returns
    -------
    float
        The signed angular distance between vectors `u` and `v`. This is the angle in radians
        between the two vectors, signed according to the direction given by `n`. If `n` is not
        provided, the result is the unsigned angle.

    Notes
    -----
    The signed angle is computed based on the right-hand rule. If `n` is provided, the sign
    of the angle is determined by the direction of `n` relative to the cross product of `u` and `v`.
    If `n` is not provided, the function returns the magnitude of the angle only.

    Examples
    --------
    >>> u = np.array([1, 0, 0])
    >>> v = np.array([0, 1, 0])
    >>> vector_angular_distance_signed(u, v)
    1.5707963267948966  # 90 degrees in radians

    >>> n = np.array([0, 0, 1])
    >>> vector_angular_distance_signed(u, v, n)
    1.5707963267948966  # 90 degrees in radians, positive as per right-hand rule with `n` as z-axis

    >>> n = np.array([0, 0, -1])
    >>> vector_angular_distance_signed(u, v, n)
    -1.5707963267948966  # 90 degrees in radians, negative as per right-hand rule with `n` as -z-axis
    """

    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))
    else:
        return np.arctan2(np.dot(n, np.cross(u, v)), np.dot(u, v))


def oversample_spline(coords, target_spacing):
    """Fit a spline through 3D coordinates and oversample so that the distance between points is approximately `target_spacing`.

    Parameters
    ----------
    coords : ndarray
        Array of shape (n, 3) representing the input points.
    target_spacing : float
        Desired distance between points on the spline.

    Returns
    -------
    ndarray
        Oversampled coordinates along the spline.
    """
    # Fit a parametric spline to the data
    tck, u = splprep(coords.T, s=0)  # `s=0` ensures an exact fit to the input points

    # Compute cumulative arc length
    distances = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
    total_length = np.sum(distances)

    # Determine number of samples based on target spacing
    num_samples = int(total_length / target_spacing) + 1

    # Generate evenly spaced parameter values along the spline
    u_fine = np.linspace(0, 1, num_samples)

    # Evaluate the spline to get oversampled points
    oversampled_points = np.array(splev(u_fine, tck)).T

    return oversampled_points

def distance_array(vol):
    shell_grid = np.arange(math.floor(-len(vol[0]) / 2), math.ceil(len(vol[0]) / 2), 1)
    xv, yv, zv = shell_grid, shell_grid, shell_grid
    shell_space = np.meshgrid(xv, yv, zv, indexing="xy")  ## 'ij' denominates matrix indexing, 'xy' cartesian
    distance_v = np.sqrt(shell_space[0] ** 2 + shell_space[1] ** 2 + shell_space[2] ** 2)

    return distance_v