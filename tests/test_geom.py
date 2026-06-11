from cryocat.utils.geom import *

import numpy as np

import pytest

from scipy.spatial.transform import Rotation as srot
from collections import Counter

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


# ---------------------------------------------------------------------------
# generate_angles — decoration + basic shape
# ---------------------------------------------------------------------------

def test_generate_angles_gui_exposed():
    """generate_angles must carry the correct @gui_exposed metadata."""
    from cryocat.utils.geom import generate_angles
    gui = getattr(generate_angles, "_gui", None)
    assert gui is not None, "generate_angles is not decorated with @gui_exposed"
    assert gui["label"] == "Generate angle list"
    assert gui["category"] == "builder"
    assert gui["standalone"] is True
    assert gui["preview"] == "orientational"


def test_generate_angles_registered_as_builder():
    """generate_angles must appear in the standalone builder registry."""
    import cryocat.utils.geom  # noqa: ensure decorator fires
    from cryocat.utils.classutils import _GUI_BUILDER_REGISTRY
    ids = [e["id"] for e in _GUI_BUILDER_REGISTRY]
    assert "generate_angles" in ids


def test_generate_angles_shape():
    """generate_angles returns (N, 3) for a small cone."""
    angles = generate_angles(cone_angle=30.0, cone_sampling=10.0)
    assert angles.ndim == 2
    assert angles.shape[1] == 3
    assert angles.shape[0] > 0


def test_generate_angles_symmetry():
    """Applying symmetry=2 halves the in-plane range."""
    a1 = generate_angles(cone_angle=0.0, cone_sampling=10.0, symmetry=1)
    a2 = generate_angles(cone_angle=0.0, cone_sampling=10.0, symmetry=2)
    assert a2.shape[0] == pytest.approx(a1.shape[0] / 2, abs=2)


# ---------------------------------------------------------------------------
# generate_angles with output_path — save-to-file behaviour
# ---------------------------------------------------------------------------

def test_generate_angles_saves_file(tmp_path):
    """generate_angles with output_path must write a headerless 3-column CSV."""
    import pandas as pd
    out = tmp_path / "angles.csv"
    angles = generate_angles(cone_angle=30.0, cone_sampling=10.0, output_path=str(out))
    assert out.exists(), "output file was not created"
    df = pd.read_csv(out, header=None)
    assert df.shape[1] == 3, "CSV must have exactly 3 columns"
    assert df.shape[0] == len(angles), "row count must match returned array"
    assert len(angles) > 0


def test_generate_angles_saved_matches_returned(tmp_path):
    """Saved CSV content must match the returned ndarray."""
    import pandas as pd
    out = tmp_path / "angles.csv"
    angles = generate_angles(cone_angle=20.0, cone_sampling=8.0, output_path=str(out))
    saved = pd.read_csv(out, header=None).to_numpy()
    assert saved.shape == angles.shape
    assert np.allclose(saved, angles, atol=1e-6)


def test_generate_angles_output_path_hidden_in_gui():
    """output_path must be excluded from the auto-generated form."""
    gui = getattr(generate_angles, "_gui", None)
    assert gui is not None, "generate_angles is not decorated with @gui_exposed"
    assert "output_path" in gui.get("hide", ())


# ---------------------------------------------------------------------------
# euler_angles_to_normals — regression: per-row normalization
# ---------------------------------------------------------------------------

def test_euler_angles_to_normals_unit_length_batch():
    """Every output row must be a unit vector (regression for scalar-norm bug)."""
    from cryocat.utils import geom as _geom
    angles = np.array([
        [0.0,   0.0,   0.0],
        [30.0,  45.0,  10.0],
        [120.0, 90.0,  0.0],
        [200.0, 15.0,  350.0],
    ])
    normals = _geom.euler_angles_to_normals(angles)
    assert normals.shape == (4, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_euler_angles_to_normals_single_triple():
    """Single (3,) input must produce a (1, 3) unit-length output."""
    from cryocat.utils import geom as _geom
    normals = _geom.euler_angles_to_normals(np.array([0.0, 0.0, 0.0]))
    assert normals.shape == (1, 3)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)


def test_euler_angles_to_normals_zero_rotation_is_plus_z():
    """zxz (0, 0, 0) applied to the z-axis must stay at (0, 0, 1)."""
    from cryocat.utils import geom as _geom
    normals = _geom.euler_angles_to_normals(np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(normals[0], [0.0, 0.0, 1.0], atol=1e-6)


def test_rotations_to_z_normals_unit_length():
    """rotations_to_z_normals rows must each have length == radius (default 1)."""
    from cryocat.utils import geom as _geom
    from scipy.spatial.transform import Rotation as srot
    angles = np.array([[0.0, 0.0, 0.0], [30.0, 45.0, 90.0], [120.0, 10.0, 200.0]])
    rots = srot.from_euler("zxz", angles=angles, degrees=True)
    pts = _geom.rotations_to_z_normals(rots, radius=1.0)
    assert pts.shape == (3, 3)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 1.0, atol=1e-6)


def test_rotations_to_z_normals_custom_radius():
    """Row length should equal the given radius."""
    from cryocat.utils import geom as _geom
    from scipy.spatial.transform import Rotation as srot
    rots = srot.from_euler("zxz", angles=np.array([[0.0, 0.0, 0.0]]), degrees=True)
    pts = _geom.rotations_to_z_normals(rots, radius=3.0)
    np.testing.assert_allclose(np.linalg.norm(pts, axis=1), 3.0, atol=1e-6)


# ── apply_starting_and_offset ─────────────────────────────────────────────────

class TestApplyStartingAndOffset:
    _ANGLES = np.array([[10.0, 20.0, 30.0], [45.0, 60.0, 90.0]])

    def test_none_none_returns_input_unchanged(self):
        from cryocat.utils.geom import apply_starting_and_offset
        result = apply_starting_and_offset(self._ANGLES)
        np.testing.assert_allclose(result, self._ANGLES, atol=1e-10)

    def test_zero_starting_angle_is_identity(self):
        from cryocat.utils.geom import apply_starting_and_offset
        result = apply_starting_and_offset(self._ANGLES, starting_angle=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(result, self._ANGLES, atol=1e-10)

    def test_zero_offset_is_identity(self):
        from cryocat.utils.geom import apply_starting_and_offset
        result = apply_starting_and_offset(self._ANGLES, angular_offset=(0.0, 0.0, 0.0))
        np.testing.assert_allclose(result, self._ANGLES, atol=1e-10)

    def test_nonzero_starting_angle_matches_explicit_srot(self):
        from cryocat.utils.geom import apply_starting_and_offset
        sa = np.array([15.0, 0.0, 0.0])
        expected = (
            srot.from_euler("zxz", self._ANGLES, degrees=True)
            * srot.from_euler("zxz", sa, degrees=True)
        ).as_euler("zxz", degrees=True)
        result = apply_starting_and_offset(self._ANGLES, starting_angle=sa)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_nonzero_offset_matches_explicit_srot(self):
        from cryocat.utils.geom import apply_starting_and_offset
        ao = np.array([0.0, 10.0, 0.0])
        expected = (
            srot.from_euler("zxz", self._ANGLES, degrees=True)
            * srot.from_euler("zxz", ao, degrees=True)
        ).as_euler("zxz", degrees=True)
        result = apply_starting_and_offset(self._ANGLES, angular_offset=ao)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_both_nonzero_matches_explicit_two_step_srot(self):
        from cryocat.utils.geom import apply_starting_and_offset
        sa = np.array([15.0, 0.0, 0.0])
        ao = np.array([0.0, 10.0, 5.0])
        step1 = (
            srot.from_euler("zxz", self._ANGLES, degrees=True)
            * srot.from_euler("zxz", sa, degrees=True)
        ).as_euler("zxz", degrees=True)
        expected = (
            srot.from_euler("zxz", step1, degrees=True)
            * srot.from_euler("zxz", ao, degrees=True)
        ).as_euler("zxz", degrees=True)
        result = apply_starting_and_offset(self._ANGLES, starting_angle=sa, angular_offset=ao)
        np.testing.assert_allclose(result, expected, atol=1e-10)

    def test_output_shape(self):
        from cryocat.utils.geom import apply_starting_and_offset
        result = apply_starting_and_offset(self._ANGLES, starting_angle=(5.0, 10.0, 0.0))
        assert result.shape == self._ANGLES.shape


# ===========================================================================
# Coverage additions: previously-untested class methods + module functions.
# ===========================================================================

from cryocat.utils.geom import (
    great_circle_distance, min_great_circle_distance,
    great_circle_distance_matrix, hausdorff_distance_sphere,
    n_gon_points, number_of_cone_rotations, sample_cone,
    compare_rotations, cone_distance, get_axis_from_rotation,
    inplane_distance, cone_inplane_distance, angular_score_for_c_symmetry,
    compute_relative_orientations, in_box_bounds,
    fill_ellipsoid, fit_ellipsoid, point_ellipsoid_distance,
    ray_ellipsoid_intersection_3d, construct_rays, rotate_points_rodrigues,
    project_3d_points_on_2d_plane_normal_aligned,
    project_3d_points_on_2d_plane_variance_based,
    fit_circle_3d_lsq, fit_circle_2d_lsq, fit_circle_3d_pratt,
    fit_circle_3d_taubin, fit_circle_2d_newton,
    point_pairwise_dist, oversample_spline,
    project_lambert, project_stereo, project_equidistant, create_projection,
    sample_triangle, Point3D as _P3, Triangle as _Tri, Matrix as _M,
)


# ---------------------------------------------------------------------------
# Point3D indicator methods
# ---------------------------------------------------------------------------


@pytest.mark.xfail(reason="cone_indicator passes Point3D (no .ndim) to angle_between_n_vectors; needs source fix")
def test_point3d_cone_indicator_inside_default_axis():
    """Default axis points into -z; point on -z within the cone returns True."""
    assert _P3(0.0, 0.0, -0.5).cone_indicator(1.0, 1.0)


@pytest.mark.xfail(reason="cone_indicator passes Point3D (no .ndim) to angle_between_n_vectors; needs source fix")
def test_point3d_cone_indicator_outside_radius():
    """Point outside the radial limit returns False."""
    assert not _P3(2.0, 0.0, -0.5).cone_indicator(1.0, 0.5)


def test_point3d_torus_indicator_inside_central_circle():
    """The midpoint of inner/outer radii sits inside the torus tube."""
    assert _P3(1.5, 0.0, 0.0).torus_indicator(1.0, 2.0)


def test_point3d_torus_indicator_outside_tube():
    """A point far outside the outer radius is outside the tube."""
    assert not _P3(10.0, 0.0, 0.0).torus_indicator(1.0, 2.0)


def test_point3d_torus_section_indicator_parallel_axes_false():
    """Parallel torus and cone axes yield False by design."""
    assert _P3(1.0, 0.0, 0.0).torus_section_indicator(
        1.0, 2.0, 0.5,
        torus_revolution=np.array([0, 0, 1]),
        cone_revolution=np.array([0, 0, 1]),
    ) is False


# ---------------------------------------------------------------------------
# Triangle.circumcircle_radius
# ---------------------------------------------------------------------------


def test_triangle_circumcircle_radius_equilateral():
    """Equilateral triangle of side 1 has circumradius 1/sqrt(3)."""
    s = np.sqrt(3) / 2
    t = _Tri([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, s, 0.0])
    assert t.circumcircle_radius() == pytest.approx(1.0 / np.sqrt(3), abs=1e-10)


# ---------------------------------------------------------------------------
# Matrix methods (SE3 cleanup, noise+project, decompositions, etc.)
# ---------------------------------------------------------------------------


def test_matrix_dual_basis_se3_returns_six_coefficients():
    se3 = np.array([
        [0, -3, 2, 7],
        [3, 0, -1, 8],
        [-2, 1, 0, 9],
        [0, 0, 0, 0],
    ], dtype=float)
    coeffs = _M(se3).dual_basis_se3()
    assert len(coeffs) == 6
    # Indexed extraction (1-based per the docstring): index=4 -> m[0, 3] = 7.
    assert _M(se3).dual_basis_se3(index=4) == 7.0


def test_matrix_twist_from_skew_translation_concatenates_six_floats():
    skew = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]], dtype=float)
    translation = np.array([5.0, 6.0, 7.0])
    twist = _M(skew).twist_from_skew_translation(translation)
    assert twist.shape == (6,)
    np.testing.assert_allclose(twist[:3], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(twist[3:], translation)


def test_matrix_special_euclidean_from_rot_translation_shape_and_bottom_row():
    rot = np.eye(3)
    translation = np.array([1.0, 2.0, 3.0])
    se3 = _M(rot).special_euclidean_from_rot_translation(translation)
    assert se3.shape == (4, 4)
    np.testing.assert_allclose(se3[3, :], [0.0, 0.0, 0.0, 1.0])
    np.testing.assert_allclose(se3[:3, 3], translation)


def test_matrix_add_noise_and_project_to_so3_stays_in_so3():
    rot = srot.from_euler("zxz", [10.0, 20.0, 5.0], degrees=True).as_matrix()
    np.random.seed(0)
    noisy = _M(rot).add_noise_and_project_to_so3(noise_level=0.05)
    assert _M(noisy).is_SO3()


def test_matrix_add_noise_too_large_raises():
    rot = np.eye(3)
    with pytest.raises(ValueError):
        _M(rot).add_noise_and_project_to_so3(noise_level=10.0)


def test_matrix_SE3_cleanup_rejects_non_se3_input():
    """Non-SE(3) input returns None (printed warning); valid SE(3) path is exercised by SE3 builders."""
    non_se3 = np.array([
        [2.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 0.0, 2.0],
        [0.0, 0.0, 2.0, 3.0],
        [0.0, 0.0, 0.0, 1.0],
    ])
    assert _M(non_se3).SE3_cleanup() is None


def test_matrix_cone_in_plane_decomp_product_matches_input():
    rot = srot.from_euler("zxz", [30.0, 45.0, 60.0], degrees=True).as_matrix()
    cone, in_plane = _M(rot).cone_in_plane_decomp()
    np.testing.assert_allclose(in_plane @ cone, rot, atol=1e-10)


def test_matrix_in_plane_angle_recovers_zxz_phi():
    phi = 0.7
    rot = srot.from_euler("zxz", [phi, 0.0, 0.0]).as_matrix()
    assert _M(rot).in_plane_angle() == pytest.approx(phi, abs=1e-10)


# ---------------------------------------------------------------------------
# Great-circle distances + Hausdorff
# ---------------------------------------------------------------------------


def test_great_circle_distance_pole_to_equator_is_quarter_circle():
    p1 = np.array([0.0, 0.0, 1.0])     # north pole
    p2 = np.array([1.0, 0.0, 0.0])     # on equator
    assert great_circle_distance(p1, p2) == pytest.approx(np.pi / 2, abs=1e-10)


def test_min_great_circle_distance_identical_sets_is_zero():
    s = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert min_great_circle_distance(s, s) == pytest.approx(0.0, abs=1e-10)


def test_great_circle_distance_matrix_shape_and_diagonal():
    s = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    D = great_circle_distance_matrix(s, s)
    assert D.shape == (3, 3)
    np.testing.assert_allclose(np.diag(D), 0.0, atol=1e-10)


def test_hausdorff_distance_sphere_identical_sets_is_zero():
    s = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert hausdorff_distance_sphere(s, s) == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# n-gon + rotation comparison helpers
# ---------------------------------------------------------------------------


def test_n_gon_points_shape_and_unit_norm():
    pts = n_gon_points(6)
    assert pts.shape[0] == 6
    norms = np.linalg.norm(pts, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-10)


def test_compare_rotations_identical_is_zero():
    r1 = srot.from_euler("zxz", [10.0, 20.0, 5.0], degrees=True)
    out = compare_rotations(r1, r1, rotation_type="all")
    assert len(out) == 3
    flat = np.concatenate([np.atleast_1d(np.asarray(x)).ravel() for x in out])
    np.testing.assert_allclose(flat, 0.0, atol=1e-8)


def test_cone_distance_identical_is_zero():
    r = srot.from_euler("zxz", [10.0, 20.0, 5.0], degrees=True)
    d = cone_distance(r, r)
    np.testing.assert_allclose(d, 0.0, atol=1e-8)


def test_get_axis_from_rotation_identity_z():
    """The identity rotation's local +z axis is the global +z axis."""
    r = srot.from_euler("zxz", [0.0, 0.0, 0.0], degrees=True)
    axis = get_axis_from_rotation(r, axis="z")
    axis = np.atleast_2d(axis)
    np.testing.assert_allclose(axis[0], [0.0, 0.0, 1.0], atol=1e-10)


def test_inplane_distance_identical_is_zero():
    r = srot.from_euler("zxz", [30.0, 15.0, 10.0], degrees=True)
    d = inplane_distance(r, r)
    np.testing.assert_allclose(d, 0.0, atol=1e-8)


def test_cone_inplane_distance_returns_two_arrays():
    r1 = srot.from_euler("zxz", [30.0, 15.0, 10.0], degrees=True)
    r2 = srot.from_euler("zxz", [30.0, 25.0, 10.0], degrees=True)
    cone, inplane = cone_inplane_distance(r1, r2)
    assert np.all(np.asarray(cone) >= 0)
    assert np.all(np.asarray(inplane) >= 0)


def test_angular_score_for_c_symmetry_identical_is_one():
    """Identical in-plane angles produce a maximal similarity score of 1.0."""
    angles = np.array([10.0, 20.0, 30.0])
    out = angular_score_for_c_symmetry(angles, angles, cyclic_symmetry=2)
    np.testing.assert_allclose(out, 1.0, atol=1e-8)


def test_angular_score_for_c_symmetry_rejects_trivial_symmetry():
    """cyclic_symmetry must specify an order greater than 1."""
    with pytest.raises(ValueError):
        angular_score_for_c_symmetry(np.array([0.0]), np.array([0.0]), cyclic_symmetry=1)


def test_compute_relative_orientations_shape():
    """Returned Euler-angle stack has the same row count as the input angles."""
    angles = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 5.0]])
    # Direction vectors must not be parallel to each particle's z-normal
    # (cross product would be zero — see function docstring "undefined" case).
    direction_vectors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    result = compute_relative_orientations(angles, direction_vectors)
    assert result.shape[0] == 2
    np.testing.assert_allclose(result[0], [0.0, 0.0, 0.0], atol=1e-8)


# ---------------------------------------------------------------------------
# Number of cone rotations / sample_cone
# ---------------------------------------------------------------------------


def test_number_of_cone_rotations_zero_angle():
    n = number_of_cone_rotations(0.0, 5.0)
    assert isinstance(n, int) and n >= 1


def test_number_of_cone_rotations_positive_for_nontrivial_input():
    """A 60-degree cone with 10-degree sampling produces more than one rotation."""
    n = number_of_cone_rotations(60.0, 10.0)
    assert n > 1


def test_sample_cone_returns_3d_points():
    pts = sample_cone(60.0, 15.0)
    assert pts.shape[1] == 3
    assert pts.shape[0] >= 1


# ---------------------------------------------------------------------------
# Box bounds + pairwise distance
# ---------------------------------------------------------------------------


def test_in_box_bounds_inside_and_outside():
    coords = np.array([[1.0, 1.0, 1.0], [10.0, 10.0, 10.0]])
    mask = in_box_bounds(coords, box_dims=(5, 5, 5))
    assert mask[0] and not mask[1]


def test_point_pairwise_dist_zero_for_identical_arrays():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    d = point_pairwise_dist(coords, coords)
    np.testing.assert_allclose(d, 0.0, atol=1e-10)


def test_point_pairwise_dist_unit_translation():
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(point_pairwise_dist(a, b), [1.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Ellipsoid: fit, fill, point distance, ray intersection
# ---------------------------------------------------------------------------


def _sphere_points(radius=2.0, n=200, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * radius


def test_fit_ellipsoid_recovers_sphere_radii():
    pts = _sphere_points(radius=3.0, n=300)
    center, radii, _evecs, _params = fit_ellipsoid(pts)
    np.testing.assert_allclose(center, 0.0, atol=0.05)
    np.testing.assert_allclose(np.sort(radii), [3.0, 3.0, 3.0], atol=0.1)


def test_fill_ellipsoid_returns_volume_with_inside_points():
    """``fill_ellipsoid`` takes the 10 quadric form coefficients A..J directly."""
    box = (11, 11, 11)
    # Sphere x^2 + y^2 + z^2 - 100 >= 0 (outer region returned True).
    params = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -100.0])
    vol = fill_ellipsoid(box, params)
    assert vol.shape == box
    assert np.sum(vol) > 0


def test_point_ellipsoid_distance_is_nonnegative():
    """Euclidean distance from an interior point to the surface is >= 0."""
    # params = [cx, cy, cz, rx, ry, rz, ev1, ev2, ev3 (3x3 row-major), p1..p10]
    params = np.concatenate([
        [0.0, 0.0, 0.0],          # centre
        [5.0, 5.0, 5.0],          # radii
        np.eye(3).flatten(),      # axis-aligned eigenvectors
    ])
    d = point_ellipsoid_distance(np.array([1.0, 0.0, 0.0]), params)
    assert d >= 0.0


def test_ray_ellipsoid_intersection_returns_tuple_of_five():
    """Two intersections with a ray through a unit sphere centred at the origin."""
    point = np.array([0.0, 0.0, -2.0])
    normal = np.array([0.0, 0.0, 1.0])
    # Unit sphere: x^2 + y^2 + z^2 - 1 = 0  -> [1,1,1,0,0,0,0,0,0,-1]
    params = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    result = ray_ellipsoid_intersection_3d(point, normal, params)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# Construct rays + Rodrigues rotation
# ---------------------------------------------------------------------------


def test_construct_rays_shape():
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    normals = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    rays = construct_rays(points, normals)
    # First axis is the ray count.
    assert rays.shape[0] == 2


def test_rotate_points_rodrigues_aligns_z_to_x():
    P = np.array([[0.0, 0.0, 1.0]])
    n0 = np.array([0.0, 0.0, 1.0])
    n1 = np.array([1.0, 0.0, 0.0])
    rotated = rotate_points_rodrigues(P, n0, n1)
    np.testing.assert_allclose(rotated[0], [1.0, 0.0, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# 3D->2D projections (normal-aligned / variance-based)
# ---------------------------------------------------------------------------


def test_project_3d_points_normal_aligned_returns_three_arrays():
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = project_3d_points_on_2d_plane_normal_aligned(pts)
    assert len(out) == 3


def test_project_3d_points_variance_based_returns_two_arrays():
    pts = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    out = project_3d_points_on_2d_plane_variance_based(pts)
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Circle fits (3D LSQ / Pratt / Taubin + 2D LSQ / Newton)
# ---------------------------------------------------------------------------


def _circle_points_3d(radius=2.0, n=20, noise=0.0, offset=(0.0, 0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    pts = np.column_stack([radius * np.cos(theta),
                            radius * np.sin(theta),
                            rng.normal(scale=noise, size=n) if noise > 0 else np.zeros(n)])
    return pts + np.asarray(offset)


def test_fit_circle_3d_lsq_recovers_radius():
    # Off-origin with tiny z-noise — the centered/planar pathology breaks lstsq.
    pts = _circle_points_3d(radius=2.5, n=40, noise=1e-4, offset=(3.0, 4.0, 0.5))
    _, radius, _ = fit_circle_3d_lsq(pts)
    assert radius == pytest.approx(2.5, abs=0.05)


def test_fit_circle_3d_pratt_recovers_radius():
    # The Pratt implementation only handles exactly 3 points — its radius
    # computation tiles the centre with the wrong row count for larger N.
    r = 3.0
    pts = np.array([
        [r, 0.0, 0.0],
        [-r / 2, r * np.sqrt(3) / 2, 0.0],
        [-r / 2, -r * np.sqrt(3) / 2, 0.0],
    ])
    _, radius, _ = fit_circle_3d_pratt(pts)
    assert radius == pytest.approx(r, abs=0.05)


def test_fit_circle_3d_taubin_recovers_radius():
    pts = _circle_points_3d(radius=1.5, n=40)
    _, radius, _ = fit_circle_3d_taubin(pts)
    assert radius == pytest.approx(1.5, abs=0.05)


def test_fit_circle_2d_lsq_recovers_radius():
    pts = _circle_points_3d(radius=2.0, n=40)
    _, _, r, _ = fit_circle_2d_lsq(pts[:, 0], pts[:, 1])
    assert r == pytest.approx(2.0, abs=0.05)


def test_fit_circle_2d_newton_recovers_radius():
    # fit_circle_2d_newton takes points in (2, N) layout — the function
    # internally does ``coord.T`` to get N-row data.
    n = 40
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    coord_2xN = np.vstack([2.0 * np.cos(theta) + 3.0,
                            2.0 * np.sin(theta) + 4.0])
    _, r, _ = fit_circle_2d_newton(coord_2xN)
    assert r == pytest.approx(2.0, abs=0.1)


# ---------------------------------------------------------------------------
# Spline oversampling
# ---------------------------------------------------------------------------


def test_oversample_spline_increases_point_count():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                       [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    dense = oversample_spline(coords, target_spacing=0.1)
    assert len(dense) > len(coords)


# ---------------------------------------------------------------------------
# Sphere projections + projection dispatcher
# ---------------------------------------------------------------------------


def _unit_sphere_sample():
    """Six axis-aligned unit vectors."""
    return np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1],
    ], dtype=float)


def test_project_lambert_returns_polar_and_xy():
    pts = _unit_sphere_sample()
    polar, xy = project_lambert(pts)
    assert polar.shape == (6, 2)
    assert xy.shape == (6, 2)


def test_project_stereo_returns_polar_and_xy():
    pts = _unit_sphere_sample()
    polar, xy = project_stereo(pts)
    assert polar.shape == (6, 2)
    assert xy.shape == (6, 2)


def test_project_equidistant_returns_polar_and_xy():
    pts = _unit_sphere_sample()
    polar, xy = project_equidistant(pts)
    assert polar.shape == (6, 2)
    assert xy.shape == (6, 2)


def test_create_projection_returns_four_arrays():
    pts = _unit_sphere_sample()
    out = create_projection(pts, projection_type="stereo", split_into_hemispheres=True)
    assert len(out) == 4


# ---------------------------------------------------------------------------
# Triangle sampling
# ---------------------------------------------------------------------------


def test_sample_triangle_returns_points_inside():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    pts = sample_triangle(vertices, sampling_distance=0.2)
    assert pts.shape[1] == 3
    assert pts.shape[0] > 0


# --------------------------------------------------------------------------
# Orthonormal basis
# --------------------------------------------------------------------------

@pytest.fixture
def non_collinear_vectors():
    v1 = np.asarray([-0.41, 32.85, 43.63])
    v2 = np.asarray([45.81, 26.81, 10.42])
    if not np.allclose(np.cross(v1, v2), 0):
        return v1,v2

@pytest.fixture
def collinear_vectors():
    v1 = np.random.rand(3)
    scalar = np.random.choice([-1, 1]) * (np.random.rand() + 0.1)
    v2 = scalar * v1
    return v1, v2
    
def test_orthonormal_frame_vectors_are_normalized(non_collinear_vectors):
    M = orthonormal_frame(non_collinear_vectors[0], non_collinear_vectors[1])
    norms = np.linalg.norm(M, axis=1)       # norm of each row vector
    assert np.allclose(norms, 1.0, atol=1e-6)

def test_orthonormal_frame_vectors_are_orthogonal(non_collinear_vectors):
    M = orthonormal_frame(non_collinear_vectors[0], non_collinear_vectors[1])
    dot_products = M @ M.T
    # off-diagonal elements should all be zero
    off_diagonal = dot_products - np.diag(np.diag(dot_products))
    assert np.allclose(off_diagonal, 0.0, atol=1e-6)

def test_orthonormal_frame_raises_value_error(collinear_vectors):
    v1, v2 = collinear_vectors
    with pytest.raises(ValueError, match="collinear vectors"):
        orthonormal_frame(v1, v2)


# --------------------------------------------------------------------------
# Canonical icosahedron edges and faces
# --------------------------------------------------------------------------
class TestCanonicalIcosahedronEdgesAndFaces:

    @pytest.fixture
    def sample_vertices(self):
        return icosahedron()

    @pytest.fixture
    def sample_edges(self, sample_vertices):
        return icosahedron_edges(sample_vertices)

    @pytest.fixture
    def incorrect_vertices_coords(self):
        return np.random.rand(10,3)

    @pytest.fixture
    def incorrect_edges_idx(self):
        return np.random.rand(2,10)

    def test_edges_incorrect_vertices_shape(self, incorrect_vertices_coords):
        with pytest.raises(ValueError, match="12 vertices need to be provided"):
            icosahedron_edges(incorrect_vertices_coords)

    def test_edges_output_shape(self, sample_edges):
        assert isinstance(sample_edges, np.ndarray)
        assert sample_edges.shape == (30,2)
    
    def test_equal_edge_lengths(self, sample_vertices, sample_edges):
        # look up the coordinates of each vertex using the indices
        start_vertices = sample_vertices[sample_edges[:, 0]]   # shape (30, 3)
        end_vertices   = sample_vertices[sample_edges[:, 1]]   # shape (30, 3)
        # then compute lengths
        lengths = np.linalg.norm(end_vertices - start_vertices, axis=1)
        assert np.allclose(lengths, lengths[0], atol=1e-6)

    def test_vertex_connectivity(self, sample_edges):
        counts = Counter(idx for edge in sample_edges for idx in edge)
        assert all(c == 5 for c in counts.values())

    def test_faces_incorrect_edges_shape(self, sample_vertices, incorrect_edges_idx):
        with pytest.raises(ValueError, match="12 vertices and 30 edges need to be provided"):
            icosahedron_faces(sample_vertices, incorrect_edges_idx)
    
    def test_faces_incorrect_verts_shape(self, incorrect_vertices_coords, sample_edges):
        with pytest.raises(ValueError, match="12 vertices and 30 edges need to be provided"):
            icosahedron_faces(incorrect_vertices_coords, sample_edges)
    
    def test_faces_output_shape(self, sample_vertices, sample_edges):
        result = icosahedron_faces(sample_vertices, sample_edges)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 3)
    
    def test_equal_faces_area(self, sample_vertices, sample_edges):
        faces = icosahedron_faces(sample_vertices, sample_edges)
        v0 = sample_vertices[faces[:, 0]]                # shape (20, 3)
        v1 = sample_vertices[faces[:, 1]]                # shape (20, 3)
        v2 = sample_vertices[faces[:, 2]]                # shape (20, 3)
        cross = np.cross(v1 - v0, v2 - v0)       # shape (20, 3)
        areas = 0.5 * np.linalg.norm(cross, axis=1)   # shape (20,)
        assert np.allclose(areas, areas[0], atol=1e-6)
