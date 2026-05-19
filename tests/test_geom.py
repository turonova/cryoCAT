from cryocat.utils.geom import *

import numpy as np

import pytest

from scipy.spatial.transform import Rotation as srot

import sys

sys.path.append(".")

TOLERANCE = 10e-12


def identity():
    return np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])


def unit_stack():
    return np.hstack((np.eye(3), np.zeros((3, 1))))


def unit_i():
    return np.array([[1, 0, 0, 0]])


def unit_j():
    return np.array([[0, 1, 0, 0]])


def unit_k():
    return np.array([[0, 0, 1, 0]])


def rot_x_pi():
    rot = srot.from_matrix([[1, 0, 0], [0, np.cos(np.pi), -np.sin(np.pi)], [0, np.sin(np.pi), np.cos(np.pi)]])
    return rot


def rot_y_pi_2():
    rot = srot.from_matrix(
        [[np.cos(np.pi / 2), 0, -np.sin(np.pi / 2)], [0, 1, 0], [np.sin(np.pi / 2), 0, np.cos(np.pi / 2)]]
    )
    return rot


def test_project_points_in_plane():
    start_point = np.array([0, 0, 0])
    normal = np.array([1, 0, 1])
    normal = normal / np.linalg.norm(normal)
    nn_points = identity()

    shifted_points = project_points_on_plane_with_preserved_distance(start_point, normal, nn_points)

    # Shifted points are supposed to be in plane perpendicular to normal, so their dot-product with normal should be 0

    assert np.linalg.norm(np.dot(shifted_points, normal)) < TOLERANCE


def test_project_points_preserved_distance():
    start_point = np.array([0, 0, 0])
    normal = np.array([1, 0, 1])
    normal = normal / np.linalg.norm(normal)
    nn_points = identity()

    shifted_points = project_points_on_plane_with_preserved_distance(start_point, normal, nn_points)

    # Distances should be preserved. in this case: Distances are 1

    assert np.allclose(np.linalg.norm(shifted_points, axis=1), np.ones(shifted_points.shape))


def test_align_points_to_xy_plane_in_plane():
    test_normal = np.array([0, 1, 0])
    test_points = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])

    rotated_points, _ = align_points_to_xy_plane(test_points, test_normal)

    assert np.linalg.norm(rotated_points[:, 2]) < TOLERANCE


def test_align_points_to_xy_plane_correctly_rotated():
    test_normal = np.array([0, 1, 0])
    test_points = np.array([[1, 0, 0], [0, 0, 1], [-1, 0, 0]])

    rotated_points, _ = align_points_to_xy_plane(test_points, test_normal)

    expected_points = np.array([[1, 0, 0], [0, -1, 0], [-1, 0, 0]])

    assert np.allclose(rotated_points, expected_points)


@pytest.mark.parametrize(
    "quat_1, quat_2, result",
    [
        (unit_i(), unit_j(), unit_k()),
        (unit_j(), unit_i(), -unit_k()),
        (unit_j(), unit_k(), unit_i()),
        (unit_k(), unit_j(), -unit_i()),
        (unit_k(), unit_i(), unit_j()),
        (unit_i(), unit_k(), -unit_j()),
        (unit_i(), unit_i(), np.array([0, 0, 0, -1])),
        (unit_j(), unit_j(), np.array([0, 0, 0, -1])),
        (unit_k(), unit_k(), np.array([0, 0, 0, -1])),
        (np.array([[1, 2, 3, 4]]), np.array([[4, 3, 2, 1]]), np.array([[12, 24, 6, -12]])),
        (np.array([[4, 3, 2, 1]]), np.array([[1, 2, 3, 4]]), np.array([[22, 4, 16, -12]])),
    ],
)
def test_quaternion_mult(quat_1, quat_2, result):

    res = quaternion_mult(quat_1, quat_2)

    assert np.allclose(res, result)


@pytest.mark.parametrize(
    "input_1, input_2, result_angle",
    [(rot_x_pi(), rot_y_pi_2(), np.array([180])), (np.array([0, 180, 0]), np.array([90, 90, -90]), np.array([180]))],
)
def test_angular_distance_angle(input_1, input_2, result_angle):
    result = angular_distance(input_1, input_2)[0]

    assert np.allclose(result, result_angle)


@pytest.mark.parametrize(
    "input_1, input_2, result_dist",
    [(rot_x_pi(), rot_y_pi_2(), np.array([1])), (np.array([0, 180, 0]), np.array([90, 90, -90]), np.array([1]))],
)
def test_angular_distance_dist(input_1, input_2, result_dist):
    result = angular_distance(input_1, input_2)[1]

    assert np.allclose(result, result_dist)


@pytest.mark.parametrize(
    "quat_stack, log_stack",
    [
        (unit_stack(), np.pi / 2 * unit_stack()),
        (
            np.array([[1, 1, 1, 2]]),
            np.array(
                [
                    [
                        np.arccos(2 / np.sqrt(7)) / np.sqrt(3),
                        np.arccos(2 / np.sqrt(7)) / np.sqrt(3),
                        np.arccos(2 / np.sqrt(7)) / np.sqrt(3),
                        np.log(np.sqrt(7)),
                    ]
                ]
            ),
        ),
    ],
)
def test_quaternion_log(quat_stack, log_stack):

    assert np.allclose(quaternion_log(quat_stack), log_stack)


def test_normalize_vector():

    vector = np.array([1, 2, 3])

    assert np.allclose(np.linalg.norm(normalize_vector(vector)), 1)


@pytest.mark.parametrize(
    "vector_1, vector_2, result",
    [(np.array([1, 0, 0]), np.array([0, 1, 0]), 90), (np.array([1, 0, 0]), np.array([1, 1, 0]), 45)],
)
def test_vector_angular_distance(vector_1, vector_2, result):

    assert np.allclose(vector_angular_distance(vector_1, vector_2), result)


def test_angle_between_vectors():

    vectors_1 = np.array([[1, 0, 0], [1, 0, 0]])

    vectors_2 = np.array([[0, 1, 0], [1, 1, 0]])

    assert np.allclose(angle_between_vectors(vectors_1, vectors_2), np.array([90, 45]))


def test_area_triangle_colinear():

    coords_colin = np.array([[1, 0, 0], [2, 0, 0], [3, 0, 0]])

    assert area_triangle(coords_colin) < TOLERANCE


def test_area_triangle():

    coords_planar = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])

    result = area_triangle(coords_planar)

    assert np.allclose(result, 0.5)


@pytest.mark.parametrize(
    "starting_points, end_points, intersection",
    [(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[-1, 0, 0], [0, 0, -1]]), np.zeros(shape=(3,)))],
)
def test_ray_ray_intersection_3d_intersection(starting_points, end_points, intersection):

    p_intersect, _ = ray_ray_intersection_3d(starting_points, end_points)

    assert np.allclose(p_intersect, intersection)

    # No error message for non-intersecting rays was built into the function -- you get, however, a message from python


@pytest.mark.parametrize(
    "starting_points, end_points, distance_result",
    [(np.array([[1, 0, 0], [0, 0, 1]]), np.array([[-1, 0, 0], [0, 0, -1]]), np.zeros(shape=(2,)))],
)
def test_ray_ray_intersection_3d_intersection(starting_points, end_points, distance_result):

    _, distances = ray_ray_intersection_3d(starting_points, end_points)

    assert np.allclose(distances, distance_result)


# TODO: change_handedness_coordinates, change_handedness_orientation, euler_angles_to_normals, normals_to_euler_angles, ...
def test_change_handedness_coordinates():
    pass


@pytest.mark.parametrize(
    "input_value, reference_size, expected",
    [
        ([1, 2, 3], None, np.array([1, 2, 3])),
        ([1, 2], None, None),
        (
            [
                1,
            ],
            None,
            np.array([1, 1, 1]),
        ),
        (1, None, np.array([1, 1, 1])),
        ((1, 2, 3), None, np.array([1, 2, 3])),
        ((1, 2), None, None),
        ((1,), None, np.array([1, 1, 1])),
        ((1), None, np.array([1, 1, 1])),
        ((1.5, 5.3, 3), None, np.array([1, 5, 3])),
        (np.array([1, 5, 3]), None, np.array([1, 5, 3])),
        (np.array([1.5, 5.3, 3]), None, np.array([1, 5, 3])),
    ],
)
def test_as_triplet(input_value, reference_size, expected):
    if expected is None:
        with pytest.raises(ValueError):
            as_triplet(input_value, reference_size)
    else:
        assert np.array_equal(as_triplet(input_value, reference_size), expected)


# ---------------------------------------------------------------------------
# Line / LineSegment
# ---------------------------------------------------------------------------

def test_line_stores_point_and_direction():
    p = np.array([1.0, 2.0, 3.0])
    d = np.array([0.0, 0.0, 1.0])
    line = Line(p, d)
    assert np.allclose(line.p, p)
    assert np.allclose(line.dir, d)


def test_line_segment_length():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([3.0, 4.0, 0.0])
    seg = LineSegment(p1, p2)
    assert seg.length == pytest.approx(5.0)


def test_line_segment_unit_direction():
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([0.0, 0.0, 7.0])
    seg = LineSegment(p1, p2)
    assert np.allclose(np.linalg.norm(seg.dir), 1.0)
    assert np.allclose(seg.dir, [0.0, 0.0, 1.0])


def test_line_segment_end_point():
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([4.0, 6.0, 3.0])
    seg = LineSegment(p1, p2)
    assert np.allclose(seg.p_end, p2)


# ---------------------------------------------------------------------------
# Point3D
# ---------------------------------------------------------------------------

def test_point3d_coords():
    p = Point3D(1.0, 2.0, 3.0)
    assert p.x == 1.0 and p.y == 2.0 and p.z == 3.0


def test_point3d_add():
    p1 = Point3D(1.0, 2.0, 3.0)
    p2 = Point3D(4.0, 5.0, 6.0)
    result = p1 + p2
    assert np.allclose(np.array(result), [5.0, 7.0, 9.0])


def test_point3d_sub():
    p1 = Point3D(4.0, 5.0, 6.0)
    p2 = Point3D(1.0, 2.0, 3.0)
    result = p1 - p2
    assert np.allclose(np.array(result), [3.0, 3.0, 3.0])


def test_point3d_mul_scalar():
    p = Point3D(1.0, 2.0, 3.0)
    result = p * 2.0
    assert np.allclose(np.array(result), [2.0, 4.0, 6.0])


def test_point3d_equality():
    assert Point3D(1.0, 2.0, 3.0) == Point3D(1.0, 2.0, 3.0)
    assert not (Point3D(1.0, 2.0, 3.0) == Point3D(0.0, 0.0, 0.0))


def test_point3d_len():
    assert len(Point3D(1.0, 2.0, 3.0)) == 3


def test_point3d_numpy_array():
    p = Point3D(1.0, 2.0, 3.0)
    arr = np.asarray(p)
    assert arr.shape == (3,)
    assert np.allclose(arr, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Triangle
# ---------------------------------------------------------------------------

def test_triangle_area_right():
    t = Triangle([0, 0, 0], [1, 0, 0], [0, 1, 0])
    assert t.area() == pytest.approx(0.5)


def test_triangle_area_colinear_zero():
    t = Triangle([0, 0, 0], [1, 0, 0], [2, 0, 0])
    assert t.area() == pytest.approx(0.0, abs=1e-12)


def test_triangle_inner_angles_equilateral():
    s = np.sqrt(3) / 2
    t = Triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, s, 0.0])
    a, b, c = t.inner_angles()
    assert a == pytest.approx(60.0, rel=1e-5)
    assert b == pytest.approx(60.0, rel=1e-5)
    assert c == pytest.approx(60.0, rel=1e-5)


def test_triangle_inner_angles_sum_180():
    t = Triangle([0, 0, 0], [3, 0, 0], [1, 2, 0])
    a, b, c = t.inner_angles()
    assert a + b + c == pytest.approx(180.0, rel=1e-5)


def test_triangle_circumcircle_equilateral():
    s = np.sqrt(3) / 2
    t = Triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, s, 0.0])
    center, radius = t.circumcircle()
    assert radius == pytest.approx(1.0 / np.sqrt(3), rel=1e-5)


def test_triangle_inscribed_circle_equilateral():
    s = np.sqrt(3) / 2
    t = Triangle([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, s, 0.0])
    center, radius = t.inscribed_circle()
    assert radius == pytest.approx(s / 3, rel=1e-5)


# ---------------------------------------------------------------------------
# Matrix
# ---------------------------------------------------------------------------

def test_matrix_default_is_identity():
    m = Matrix()
    assert np.allclose(m.m, np.eye(3))


def test_matrix_is_so3_identity():
    assert Matrix().is_SO3()


def test_matrix_is_so3_rejects_non_orthogonal():
    m = Matrix(np.ones((3, 3)))
    assert not m.is_SO3()


def test_matrix_is_se3_identity_block():
    rot = np.eye(3)
    t = np.array([1.0, 2.0, 3.0])
    se3 = np.eye(4)
    se3[:3, :3] = rot
    se3[:3, 3] = t
    assert Matrix(se3).is_SE3()


def test_matrix_is_se3_rejects_bad_bottom_row():
    se3 = np.eye(4)
    se3[3, 0] = 1.0
    assert not Matrix(se3).is_SE3()


def test_matrix_power_zero_is_identity():
    rot = srot.from_euler("zxz", [30, 45, 60], degrees=True).as_matrix()
    m = Matrix(rot)
    assert np.allclose(m.matrix_power(0), np.eye(3))


def test_matrix_power_one_is_self():
    rot = srot.from_euler("zxz", [30, 45, 60], degrees=True).as_matrix()
    m = Matrix(rot)
    assert np.allclose(m.matrix_power(1), rot)


def test_matrix_power_negative_raises():
    with pytest.raises(ValueError):
        Matrix().matrix_power(-1)


def test_matrix_dual_basis_so3():
    skew = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]], dtype=float)
    m = Matrix(skew)
    assert np.allclose(m.dual_basis_so3(), [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# Platonic solid vertex functions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "fn, expected_count",
    [
        (tetrahedron, 4),
        (octahedron, 6),
        (cube, 8),
        (icosahedron, 12),
        (dodecahedron, 20),
    ],
)
def test_platonic_vertex_count(fn, expected_count):
    v = fn()
    assert v.shape == (expected_count, 3)


@pytest.mark.parametrize("fn", [tetrahedron, octahedron, cube, icosahedron])
def test_platonic_vertices_on_unit_sphere(fn):
    v = fn()
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# normalize_vectors
# ---------------------------------------------------------------------------

def test_normalize_vectors_unit_norms():
    v = np.array([[1.0, 2.0, 3.0], [4.0, 0.0, 0.0]])
    n = normalize_vectors(v)
    norms = np.linalg.norm(n, axis=1)
    assert np.allclose(norms, 1.0)


def test_normalize_vectors_direction_preserved():
    v = np.array([[3.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    n = normalize_vectors(v)
    assert np.allclose(n[0], [1.0, 0.0, 0.0])
    assert np.allclose(n[1], [0.0, 1.0, 0.0])


# ---------------------------------------------------------------------------
# angle_between_n_vectors
# ---------------------------------------------------------------------------

def test_angle_between_n_vectors_orthogonal():
    v1 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    angles = angle_between_n_vectors(v1, v2)
    assert np.allclose(angles, [90.0, 90.0])


def test_angle_between_n_vectors_parallel():
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[2.0, 0.0, 0.0]])
    angles = angle_between_n_vectors(v1, v2)
    assert np.allclose(angles, [0.0], atol=1e-10)


def test_angle_between_n_vectors_radians():
    v1 = np.array([[1.0, 0.0, 0.0]])
    v2 = np.array([[0.0, 1.0, 0.0]])
    angle_rad = angle_between_n_vectors(v1, v2, degrees=False)
    assert np.allclose(angle_rad, [np.pi / 2])


# ---------------------------------------------------------------------------
# vector_angular_distance_signed
# ---------------------------------------------------------------------------

def test_vector_angular_distance_signed_no_normal():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    d = vector_angular_distance_signed(u, v)
    assert d == pytest.approx(np.pi / 2, rel=1e-6)


def test_vector_angular_distance_signed_with_normal():
    u = np.array([1.0, 0.0, 0.0])
    v = np.array([0.0, 1.0, 0.0])
    n_pos = np.array([0.0, 0.0, 1.0])
    n_neg = np.array([0.0, 0.0, -1.0])
    assert vector_angular_distance_signed(u, v, n_pos) == pytest.approx(np.pi / 2, rel=1e-6)
    assert vector_angular_distance_signed(u, v, n_neg) == pytest.approx(-np.pi / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# as_rotation
# ---------------------------------------------------------------------------

def test_as_rotation_from_euler():
    r = as_rotation([0.0, 0.0, 0.0])
    assert np.allclose(r.as_matrix(), np.eye(3))


def test_as_rotation_from_matrix():
    rot = srot.from_euler("zxz", [30, 45, 60], degrees=True).as_matrix()
    r = as_rotation(rot)
    assert np.allclose(r.as_matrix(), rot, atol=1e-12)


def test_as_rotation_from_quaternion():
    q = np.array([0.0, 0.0, 0.0, 1.0])
    r = as_rotation(q)
    assert np.allclose(r.as_matrix(), np.eye(3), atol=1e-12)


def test_as_rotation_passthrough():
    r = srot.from_euler("zxz", [10, 20, 30], degrees=True)
    assert as_rotation(r) is r


def test_as_rotation_invalid_raises():
    with pytest.raises(ValueError):
        as_rotation(np.zeros(5))


# ---------------------------------------------------------------------------
# as_symmetry
# ---------------------------------------------------------------------------

def test_as_symmetry_cyclic_string():
    assert as_symmetry("C5") == ("C", 5)


def test_as_symmetry_dihedral_string_lowercase():
    assert as_symmetry("d3") == ("D", 3)


def test_as_symmetry_integer():
    assert as_symmetry(7) == ("C", 7)


def test_as_symmetry_float_whole():
    assert as_symmetry(4.0) == ("C", 4)


def test_as_symmetry_invalid_string_raises():
    with pytest.raises(ValueError):
        as_symmetry("X5")


def test_as_symmetry_float_non_whole_raises():
    with pytest.raises(ValueError):
        as_symmetry(2.5)


# ---------------------------------------------------------------------------
# point_inside_triangle
# ---------------------------------------------------------------------------

def test_point_inside_triangle_centroid():
    tri = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    centroid = tri.mean(axis=0)
    assert point_inside_triangle(centroid, tri)


def test_point_inside_triangle_outside():
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    outside = np.array([2.0, 2.0, 0.0])
    assert not point_inside_triangle(outside, tri)


def test_point_inside_triangle_vertex():
    tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert point_inside_triangle(tri[0], tri)


# ---------------------------------------------------------------------------
# distance_array
# ---------------------------------------------------------------------------

def test_distance_array_shape():
    vol = np.zeros((10, 10, 10))
    d = distance_array(vol)
    assert d.shape == (10, 10, 10)


def test_distance_array_center_is_zero():
    vol = np.zeros((10, 10, 10))
    d = distance_array(vol)
    center = tuple([5] * 3)
    assert d[center] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# order_points_on_circle
# ---------------------------------------------------------------------------

def test_order_points_on_circle_sorted_angles():
    angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    pts = np.column_stack([np.cos(angles), np.sin(angles), np.zeros(8)])
    shuffled = pts[[4, 2, 6, 0, 7, 3, 5, 1]]
    ordered, _ = order_points_on_circle(shuffled)
    ordered_angles = np.arctan2(ordered[:, 1], ordered[:, 0])
    assert np.all(np.diff(ordered_angles) >= 0)


# ---------------------------------------------------------------------------
# cartesian_to_spherical
# ---------------------------------------------------------------------------

def test_cartesian_to_spherical_z_axis():
    coord = np.array([[0.0, 0.0, 1.0]])
    phi, theta = cartesian_to_spherical(coord, normalize=False)
    assert theta == pytest.approx(0.0, abs=1e-10)


def test_cartesian_to_spherical_shape():
    coord = np.random.randn(20, 3)
    norms = np.linalg.norm(coord, axis=1, keepdims=True)
    coord = coord / norms
    phi, theta = cartesian_to_spherical(coord)
    assert phi.shape == theta.shape
    assert len(phi) <= 20


def test_cartesian_to_spherical_invalid_shape_raises():
    with pytest.raises(ValueError):
        cartesian_to_spherical(np.ones((5, 4)))


# ---------------------------------------------------------------------------
# project_points_on_sphere
# ---------------------------------------------------------------------------

def test_project_points_on_sphere_stereo_shape():
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    polar, xy = project_points_on_sphere(pts, projection_type="stereo")
    assert polar.shape == (3, 2)
    assert xy.shape == (3, 2)


def test_project_points_on_sphere_lambert_shape():
    pts = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    polar, xy = project_points_on_sphere(pts, projection_type="lambert")
    assert polar.shape == (2, 2)


def test_project_points_on_sphere_invalid_raises():
    pts = np.array([[0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        project_points_on_sphere(pts, projection_type="gnomonic")
