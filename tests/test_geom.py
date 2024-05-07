import cryocat.geom as geom

import numpy as np

import pytest 

from scipy.spatial.transform import Rotation as srot

tolerance = 10e-12

def test_project_points_on_plane_with_preserved_distance():
    start_point = np.array([0, 0, 0])
    normal = np.array([1,0,1])
    normal = normal / np.linalg.norm(normal)
    nn_points = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

    shifted_points = geom.project_points_on_plane_with_preserved_distance(start_point, normal, nn_points)

    # Shifted points are supposed to be in plane perpendicular to normal, so their dot-product with normal should be 0

    assert np.linalg.norm(np.dot(shifted_points, normal)) < tolerance, "Projected points are not in desired plane!"

    # Distances should be preserved. in this case: Distances are 1 

    assert np.linalg.norm(shifted_points[0]) < 1 + tolerance and np.linalg.norm(shifted_points[0]) > 1 - tolerance, "Length of projection of first point is off"
    assert np.linalg.norm(shifted_points[1]) < 1 + tolerance and np.linalg.norm(shifted_points[1]) > 1 - tolerance, "Length of projection of second point is off"
    assert np.linalg.norm(shifted_points[2]) < 1 + tolerance and np.linalg.norm(shifted_points[2]) > 1 - tolerance, "Length of projection of third point is off"

def test_align_points_to_xy_plane():
    test_normal = np.array([0,1,0])
    test_points = np.array([
        [1,0,0],
        [0,0,1],
        [-1,0,0]
    ])

    rotated_points, rot_matrix = geom.align_points_to_xy_plane(test_points, test_normal)

    expected_points = np.array([
        [1,0,0],
        [0,-1,0],
        [-1,0,0]
    ])

    expected_rotation = np.array([
        [1,0,0],
        [0,0,-1],
        [0,1,0]
    ])

    assert np.linalg.norm(rotated_points[:, 2]) < tolerance, "Projected points are not in xy-plane!"
    assert np.sum(rotated_points - expected_points) < tolerance, "Points were wrongly rotated!"
    assert np.sum(rot_matrix - expected_rotation) < tolerance, "Rotation matrix is not as expected!"

def test_quaternion_mult():
    # scalar last convention
    unit_i = np.array([[1,0,0,0]])
    unit_j = np.array([[0,1,0,0]])
    unit_k = np.array([[0,0,1,0]])

    # in quaternion notation q = i* b + j* c + k *d + a
    # multiplication of quaternions is not commutative: i*j = k, j*i = -k, 
    # j*k = i, k*j = -i, k*i = j, i*k = -j

    q1 = np.array([[1, 2, 3, 4]])
    q2 = np.array([[4, 3, 2, 1]])

    # check for q1 * q2, q2 * q1 and products of units

    prod_ij = geom.quaternion_mult(unit_i, unit_j)
    prod_ji = geom.quaternion_mult(unit_j, unit_i)

    prod_jk = geom.quaternion_mult(unit_j, unit_k)
    prod_kj = geom.quaternion_mult(unit_k, unit_j)

    prod_ki = geom.quaternion_mult(unit_k, unit_i)
    prod_ik = geom.quaternion_mult(unit_i, unit_k)

    # product of two distinct units should behave as indicated

    assert np.linalg.norm(prod_ij - unit_k) < tolerance, "Product of i and j should be k"
    assert np.linalg.norm(prod_ji + unit_k) < tolerance, "Product of j and i should be -k"
    assert np.linalg.norm(prod_jk - unit_i) < tolerance, "Product of j and k should be i"
    assert np.linalg.norm(prod_kj + unit_i) < tolerance, "Product of k and j should be -i"
    assert np.linalg.norm(prod_ki - unit_j) < tolerance, "Product of k and i should be j"
    assert np.linalg.norm(prod_ik + unit_i) < tolerance, "Product of i and k should be -j"

    # square of any unit should be -1

    unit_stack = np.hstack((np.eye(3), np.zeros((3,1))))

    stack_squared = geom.quaternion_mult(unit_stack, unit_stack)

    negative_1_stack = np.hstack((np.zeros((3,3)), - np.ones((3,1))))

    assert np.linalg.norm(stack_squared - negative_1_stack) < tolerance, "Quaternion units should square to -1"

    # look at products of q1, q2:

    result_q1q2 = np.array([[12, 24, 6, -12]])
    result_q2q1 = np.array([[22, 4, 16, -12]])

    assert np.linalg.norm(geom.quaternion_mult(q1, q2) - result_q1q2) < tolerance, "(4 + i + 2j + 3k)(1 + 4i + 3j + 2k) = -12 + 12i + 24j + 6k"
    assert np.linalg.norm(geom.quaternion_mult(q2, q1) - result_q2q1) < tolerance, "(1 + 4i + 3j + 2k)(4 + i + 2j + 3k) = -12 + 22i +4j +16k"

def test_angular_distance():

    rot_1 = srot.from_matrix([
        [1, 0, 0],
        [0, np.cos(np.pi), - np.sin(np.pi)],
        [0, np.sin(np.pi), np.cos(np.pi)]
    ])

    rot_2 = srot.from_matrix([
        [np.cos(np.pi / 2), 0, - np.sin(np.pi / 2)],
        [0,1,0],
        [np.sin(np.pi/2), 0, np.cos(np.pi / 2)]
    ])

    assert geom.angular_distance(rot_1, rot_2)[0][0] - 180 < tolerance, "Result should be 180°; function does not work with rotation matrices!"
    assert geom.angular_distance(rot_1, rot_2)[1][0] - 1 < tolerance, "Result should be 1; function does not work with rotation matrices!"
    assert geom.angular_distance(rot_1, rot_1)[0][0] < tolerance, "Result should be 0°"
    assert geom.angular_distance(rot_1, rot_1)[1][0] < tolerance, "Result should be 0"
    assert geom.angular_distance(np.array([0, 180, 0]), np.array([90, 90, - 90]))[0][0] - 180 < tolerance, "Result should be 180°; function does not work with Euler angles!"
    assert geom.angular_distance(np.array([0, 180, 0]), np.array([90, 90, - 90]))[1][0] - 1 < tolerance, "Result should be 1; function does not work with Euler angles!"
    assert geom.angular_distance(np.array([0, 180, 0]), np.array([90, 90, - 90]))[0][0] < tolerance, "Result should be 0°"
    assert geom.angular_distance(np.array([0, 180, 0]), np.array([90, 90, - 90]))[1][0] < tolerance, "Result should be 0"

def test_quaternion_log():
    unit_stack = np.hstack((np.eye(3), np.zeros((3,1))))
    log_stack = np.pi / 2 * unit_stack

    assert np.linalg.norm(log_stack - geom.quaternion_log(unit_stack)) < tolerance, "log(i) = pi / 2 * i, log(j) = pi/2 * j, log(k) = pi/2 * k"
    # there are no error messages / nan for undefined input!
    q = np.array([[1,1,1,2]])

    imaginary_summand = np.arccos(2 / np.sqrt(7)) / np.sqrt(3)

    result = np.array([[imaginary_summand, imaginary_summand, imaginary_summand, np.log(np.sqrt(7))]])

    assert np.linalg.norm(geom.quaternion_log(q) - result) < tolerance, "Result is not as required."


def test_normalize_vector():

    vector = np.array([1,2,3])
    assert geom.normalize_vector(vector) - 1 < tolerance, "Result does not have norm 1"

def test_vector_angular_distance():

    vector_1 = np.array([1,0,0])
    vector_2 = np.array([0,1,0])

    vector_3 =  np.array([1,1,0])

    result_1 = geom.vector_angular_distance(vector_1, vector_2) - 90
    result_2 = geom.vector_angular_distance(vector_1, vector_3) - 45

    assert result_1 < tolerance and - tolerance < result_1, "Input vectors should be at angle of 90°."
    assert result_2 < tolerance and - tolerance < result_2, "Input vectors should be at angle of 45°."

def test_angle_between_vectors():

    vectors_1 = np.array([
        [1,0,0],
        [1,0,0]
    ])

    vectors_2 = np.array([
        [0,1,0],
        [1,1,0]
    ])
    
    result = geom.angle_between_vectors(vectors_1, vectors_2) - np.array([90, 45])

    assert np.linalg.norm(result) < tolerance, "Result is not as required."


def test_area_triangle():

    coords_colin = np.array([
        [1,0,0],
        [2,0,0],
        [3,0,0]
    ])

    assert geom.area_triangle(coords_colin) < tolerance, "Colinear points don't span a 2-dimensional area."

    coords_planar = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0]
    ])

    result = geom.area_triangle(coords_planar) - 0.5

    assert result < tolerance and - tolerance < result, "Area of triangle spanned by input points should be 0.5."

def test_ray_ray_intersection_3d():
    starting_points = np.array([
        [1,0,0],
        [0,0,1]
    ])

    ending_points = np.array([
        [-1,0,0]
        [0,0,-1]
    ])

    p_intersect, distances = geom.ray_ray_intersection_3d(starting_points, ending_points)

    assert np.linalg.norm(p_intersect) < tolerance, "The given lines should intersect in the origin."
    assert np.linalg.norm(distances - np.ones(shape= distances.shape)) < tolerance, "Distances to point of intersection should be 1."

    # No error message for non-intersecting rays was built into the function -- you get, however, a message from python
 



    



