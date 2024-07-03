from cryocat.geom import *

import numpy as np

import pytest 

from scipy.spatial.transform import Rotation as srot

import sys
sys.path.append('.')

TOLERANCE = 10e-12

def identity():
    return np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1]
    ])

def unit_stack():
    return np.hstack((np.eye(3), np.zeros((3,1))))

def unit_i():
    return np.array([[1,0,0,0]])

def unit_j():
    return np.array([[0,1,0,0]])

def unit_k():
    return np.array([[0,0,1,0]])

def rot_x_pi():
    rot = srot.from_matrix([
        [1, 0, 0],
        [0, np.cos(np.pi), - np.sin(np.pi)],
        [0, np.sin(np.pi), np.cos(np.pi)]
    ])
    return rot

def rot_y_pi_2():
    rot = srot.from_matrix([
        [np.cos(np.pi / 2), 0, - np.sin(np.pi / 2)],
        [0,1,0],
        [np.sin(np.pi/2), 0, np.cos(np.pi / 2)]
    ])
    return rot

def test_project_points_in_plane():
    start_point = np.array([0, 0, 0])
    normal = np.array([1,0,1])
    normal = normal / np.linalg.norm(normal)
    nn_points = identity()

    shifted_points = project_points_on_plane_with_preserved_distance(start_point, normal, nn_points)

    # Shifted points are supposed to be in plane perpendicular to normal, so their dot-product with normal should be 0

    assert np.linalg.norm(np.dot(shifted_points, normal)) < TOLERANCE

def test_project_points_preserved_distance():
    start_point = np.array([0, 0, 0])
    normal = np.array([1,0,1])
    normal = normal / np.linalg.norm(normal)
    nn_points = identity()

    shifted_points = project_points_on_plane_with_preserved_distance(start_point, normal, nn_points)

    # Distances should be preserved. in this case: Distances are 1 

    assert np.allclose(np.linalg.norm(shifted_points, axis= 1), np.ones(shifted_points.shape))


def test_align_points_to_xy_plane_in_plane():
    test_normal = np.array([0,1,0])
    test_points = np.array([
        [1,0,0],
        [0,0,1],
        [-1,0,0]
    ])

    rotated_points, _ = align_points_to_xy_plane(test_points, test_normal)

    assert np.linalg.norm(rotated_points[:, 2]) < TOLERANCE


def test_align_points_to_xy_plane_correctly_rotated():
    test_normal = np.array([0,1,0])
    test_points = np.array([
        [1,0,0],
        [0,0,1],
        [-1,0,0]
    ])

    rotated_points, _ = align_points_to_xy_plane(test_points, test_normal)

    expected_points = np.array([
        [1,0,0],
        [0,-1,0],
        [-1,0,0]
    ])

    assert np.allclose(rotated_points, expected_points)

@pytest.mark.parametrize("quat_1, quat_2, result", [
    (unit_i(), unit_j(), unit_k()), 
    (unit_j(), unit_i(), - unit_k()),
    (unit_j(), unit_k(), unit_i()),
    (unit_k(), unit_j(), - unit_i()),
    (unit_k(), unit_i(), unit_j()),
    (unit_i(), unit_k(), - unit_j()),
    (unit_i(), unit_i(), np.array([0,0,0,-1])),
    (unit_j(), unit_j(), np.array([0,0,0,-1])),
    (unit_k(), unit_k(), np.array([0,0,0,-1])),
    (np.array([[1, 2, 3, 4]]), np.array([[4, 3, 2, 1]]), np.array([[12, 24, 6, -12]])),
    (np.array([[4, 3, 2, 1]]), np.array([[1, 2, 3, 4]]), np.array([[22, 4, 16, -12]]))
])
def test_quaternion_mult(quat_1, quat_2, result):

    res = quaternion_mult(quat_1, quat_2)

    assert np.allclose(res, result)

@pytest.mark.parametrize("input_1, input_2, result_angle", [
    (rot_x_pi(), rot_y_pi_2(), np.array([180])),
    (np.array([0, 180, 0]), np.array([90, 90, - 90]), np.array([180]))
])
def test_angular_distance_angle(input_1, input_2, result_angle):
    result = angular_distance(input_1, input_2)[0]
    
    assert np.allclose(result, result_angle)

@pytest.mark.parametrize("input_1, input_2, result_dist", [
    (rot_x_pi(), rot_y_pi_2(), np.array([1])),
    (np.array([0, 180, 0]), np.array([90, 90, - 90]), np.array([1]))
])
def test_angular_distance_dist(input_1, input_2, result_dist):
    result = angular_distance(input_1, input_2)[1]
    
    assert np.allclose(result, result_dist)


@pytest.mark.parametrize("quat_stack, log_stack", [
    (unit_stack(), np.pi / 2 * unit_stack()), 
    (np.array([[1,1,1,2]]), np.array([[np.arccos(2 / np.sqrt(7)) / np.sqrt(3), np.arccos(2 / np.sqrt(7)) / np.sqrt(3), np.arccos(2 / np.sqrt(7)) / np.sqrt(3), np.log(np.sqrt(7))]]))
])
def test_quaternion_log(quat_stack, log_stack):
    
    assert np.allclose(quaternion_log(quat_stack), log_stack)


def test_normalize_vector():

    vector = np.array([1,2,3])

    assert np.allclose(np.linalg.norm(normalize_vector(vector)), 1)
    

@pytest.mark.parametrize("vector_1, vector_2, result", [
    (np.array([1,0,0]), np.array([0,1,0]), 90), 
    (np.array([1,0,0]), np.array([1,1,0]), 45)
])
def test_vector_angular_distance(vector_1, vector_2, result):

    assert np.allclose(vector_angular_distance(vector_1, vector_2), result)


def test_angle_between_vectors():

    vectors_1 = np.array([
        [1,0,0],
        [1,0,0]
    ])

    vectors_2 = np.array([
        [0,1,0],
        [1,1,0]
    ])

    assert np.allclose(angle_between_vectors(vectors_1, vectors_2), np.array([90, 45]))


def test_area_triangle_colinear():

    coords_colin = np.array([
        [1,0,0],
        [2,0,0],
        [3,0,0]
    ])

    assert area_triangle(coords_colin) < TOLERANCE


def test_area_triangle():

    coords_planar = np.array([
        [0,0,0],
        [1,0,0],
        [0,1,0]
    ])

    result = area_triangle(coords_planar)

    assert np.allclose(result, 0.5)

@pytest.mark.parametrize("starting_points, end_points, intersection", [
    (np.array([
        [1,0,0],
        [0,0,1]
    ]),
    np.array([
        [-1,0,0],
        [0,0,-1]
    ]),
    np.zeros(shape= (3,)))
])
def test_ray_ray_intersection_3d_intersection(starting_points, end_points, intersection):

    p_intersect, _ = ray_ray_intersection_3d(starting_points, end_points)

    assert np.allclose(p_intersect, intersection)


    # No error message for non-intersecting rays was built into the function -- you get, however, a message from python

@pytest.mark.parametrize("starting_points, end_points, distance_result", [
    (np.array([
        [1,0,0],
        [0,0,1]
    ]),
    np.array([
        [-1,0,0],
        [0,0,-1]
    ]),
    np.zeros(shape = (2,)))
])
def test_ray_ray_intersection_3d_intersection(starting_points, end_points, distance_result):

    _, distances = ray_ray_intersection_3d(starting_points, end_points)

    assert np.allclose(distances, distance_result)


# TODO: change_handedness_coordinates, change_handedness_orientation, euler_angles_to_normals, normals_to_euler_angles, ...
def test_change_handedness_coordinates():
    pass

def test_change_handedness_orientation():
    random_orientation = srot.random().as_matrix()
    changed_handedness = change_handedness_orientation(random_orientation)
    changed_mat = changed_handedness.as_mat()
    pass
    

 
 



    



