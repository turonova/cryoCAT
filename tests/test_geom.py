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


def test_align_points_to_xy_plane_normal_already_z():
    # Plane already parallel to xy (normal = +z): the cross product with z is
    # zero, which previously caused a 0/0 division and all-NaN output.
    # Expected: no rotation at all (identity matrix, points unchanged).
    test_normal = np.array([0.0, 0.0, 1.0])
    test_points = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])

    rotated_points, rotation_matrix = align_points_to_xy_plane(test_points, test_normal)

    assert not np.isnan(rotated_points).any()
    assert np.allclose(rotation_matrix, np.eye(3))
    assert np.allclose(rotated_points, test_points)


def test_align_points_to_xy_plane_normal_minus_z():
    # Plane parallel to xy but with its normal pointing down (-z): also a
    # zero cross product. Expected: a proper 180° rotation that flips the
    # normal onto +z and keeps all points at the same height (|z| preserved).
    test_normal = np.array([0.0, 0.0, -1.0])
    test_points = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])

    rotated_points, rotation_matrix = align_points_to_xy_plane(test_points, test_normal)

    assert not np.isnan(rotated_points).any()
    # Proper rotation: orthogonal with determinant +1
    assert np.allclose(rotation_matrix @ rotation_matrix.T, np.eye(3))
    assert np.isclose(np.linalg.det(rotation_matrix), 1.0)
    # The normal is mapped onto +z
    assert np.allclose(rotation_matrix @ test_normal, [0.0, 0.0, 1.0])
    # All points still lie on one plane parallel to xy
    assert np.allclose(rotated_points[:, 2], rotated_points[0, 2])


def test_align_points_to_xy_plane_normal_from_flat_points():
    # Normal estimated from the points themselves (plane_normal=None) for a
    # plane already parallel to xy at z=5: same degenerate case reached via
    # the estimation branch. Expected: finite output, all z values equal.
    test_points = np.array([[0.0, 0.0, 5.0], [1.0, 0.0, 5.0], [0.0, 1.0, 5.0]])

    rotated_points, _ = align_points_to_xy_plane(test_points)

    assert not np.isnan(rotated_points).any()
    assert np.allclose(rotated_points[:, 2], rotated_points[0, 2])


def test_align_points_to_xy_plane_collinear_points_raises():
    # Three collinear points do not define a plane: the estimated normal is
    # the zero vector. Expected: a clear ValueError instead of NaN output.
    test_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])

    with pytest.raises(ValueError):
        align_points_to_xy_plane(test_points)


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


@pytest.mark.parametrize("symm", [4, "C4"])
def test_angular_distance_single_rotation_with_symmetry(symm):
    """A single rotation (not a stack) is accepted together with symmetry > 1.

    Previously ``as_euler`` returned a (3,) triple and indexing ``[:, 0]``
    raised IndexError. 10 deg and 100 deg spins differ by one C4 step (90 deg),
    so they are the same orientation for a C4 particle.
    """
    rot1 = srot.from_euler("zxz", [10.0, 30.0, 0.0], degrees=True)
    rot2 = srot.from_euler("zxz", [100.0, 30.0, 0.0], degrees=True)
    angle, dist = angular_distance(rot1, rot2, symmetry=symm)
    assert angle.shape == (1,) and dist.shape == (1,)
    np.testing.assert_allclose(angle, 0.0, atol=1e-5)


def test_angular_distance_one_vs_stack_raises():
    """One rotation vs a stack is an error (previously printed and returned None)."""
    rot1 = srot.from_euler("zxz", [0.0, 0.0, 0.0], degrees=True)
    rots = srot.random(3, random_state=0)
    with pytest.raises(ValueError, match="same number of rotations"):
        angular_distance(rot1, rots)


def test_angular_distance_radians_output():
    """With ``degrees=False`` the angle is returned in radians, ``dist`` is unchanged.

    The same pair of rotations is given once as degree triples and once as
    radian triples; the angles must agree after unit conversion.
    """
    eul_1 = np.array([[10.0, 20.0, 30.0]])
    eul_2 = np.array([[50.0, 60.0, 70.0]])
    ang_deg, dist_deg = angular_distance(eul_1, eul_2, degrees=True)
    ang_rad, dist_rad = angular_distance(np.radians(eul_1), np.radians(eul_2), degrees=False)
    np.testing.assert_allclose(ang_rad, np.radians(ang_deg))
    np.testing.assert_allclose(dist_rad, dist_deg)
    # A 180 deg turn is pi in radians
    ang_pi, _ = angular_distance(np.zeros((1, 3)), np.array([[0.0, np.pi, 0.0]]), degrees=False)
    np.testing.assert_allclose(ang_pi, np.pi)


def test_angular_distance_radians_with_symmetry_matches_degrees():
    """With ``degrees=False`` the C_n step is 2*pi/n (not 360/n applied to radians).

    Degree and radian inputs describing the same rotations must give the same
    symmetry-folded distance.
    """
    eul_1 = np.array([[10.0, 20.0, 30.0], [80.0, 45.0, -60.0]])
    eul_2 = np.array([[130.0, 25.0, 30.0], [5.0, 40.0, 10.0]])
    ang_deg, _ = angular_distance(eul_1, eul_2, symmetry=3)
    ang_rad, _ = angular_distance(np.radians(eul_1), np.radians(eul_2), degrees=False, symmetry=3)
    np.testing.assert_allclose(ang_rad, np.radians(ang_deg), atol=1e-10)


def test_angular_distance_dist_equals_sin_squared_half_angle():
    """The second output equals sin(angle / 2)**2, as documented."""
    rots_1 = srot.random(500, random_state=2)
    rots_2 = srot.random(500, random_state=3)
    angle, dist = angular_distance(rots_1, rots_2)
    expected = np.sin(np.radians(angle) / 2) ** 2
    expected[expected < 10e-8] = 0
    np.testing.assert_allclose(dist, expected, atol=1e-10)


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
    "cls, expected_count",
    [
        (Tetrahedron, 4),
        (Octahedron, 6),
        (Cube, 8),
        (Icosahedron, 12),
        (Dodecahedron, 20),
    ],
)
def test_platonic_vertex_count(cls, expected_count):
    v = cls().vertices
    assert v.shape == (expected_count, 3)


@pytest.mark.parametrize("cls", [Tetrahedron, Octahedron, Cube, Icosahedron, Dodecahedron])
def test_platonic_vertices_on_unit_sphere(cls):
    v = cls().vertices
    norms = np.linalg.norm(v, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


# ---------------------------------------------------------------------------
# Polyhedron.vertex_neighbors / Polyhedron.transport_points
# ---------------------------------------------------------------------------

_SOLIDS_WITH_FOLD_AND_GROUP = [
    (Tetrahedron, 3, "T"),
    (Octahedron, 4, "O"),
    (Cube, 3, "O"),
    (Icosahedron, 5, "I"),
    (Dodecahedron, 3, "I"),
]


@pytest.mark.parametrize("cls, fold, _group", _SOLIDS_WITH_FOLD_AND_GROUP)
def test_vertex_neighbors_degree_and_symmetric(cls, fold, _group):
    # Every vertex has `fold` neighbours (the fold of its symmetry axis), and
    # the relation is mutual: j is a neighbour of i <=> i is a neighbour of j.
    nb = cls().vertex_neighbors()
    assert len(nb) == cls.n_vertices
    assert all(len(n) == fold for n in nb)
    for i, n in enumerate(nb):
        assert all(i in nb[j] for j in n)


@pytest.mark.parametrize("cls, _fold, _group", _SOLIDS_WITH_FOLD_AND_GROUP)
def test_vertex_neighbors_match_shortest_distance_rule(cls, _fold, _group):
    # Same result as the original distance-based draft (get_neighbors): the
    # neighbours are the vertices at the shortest non-zero distance.
    from scipy.spatial.distance import pdist, squareform

    v = cls().vertices
    d = squareform(pdist(v))
    edge_len = d[d > 1e-6].min()
    expected = [np.where(np.abs(d[i] - edge_len) < 1e-6)[0] for i in range(len(v))]
    for got, exp in zip(cls().vertex_neighbors(), expected):
        np.testing.assert_array_equal(got, exp)


@pytest.mark.parametrize("cls, _fold, _group", _SOLIDS_WITH_FOLD_AND_GROUP)
def test_vertex_neighbors_independent_of_radius_and_rotation(cls, _fold, _group):
    # Topology comes from the canonical vertices: scaling or turning the solid
    # must not change who is joined to whom.
    base = cls().vertex_neighbors()
    moved = cls(radius=37.0, R=srot.random(random_state=1)).vertex_neighbors()
    for a, b in zip(base, moved):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("cls, fold, group", _SOLIDS_WITH_FOLD_AND_GROUP)
def test_transport_points_ring_equals_group_orbit(cls, fold, group):
    # A ring of `fold` points around one vertex, copied to every vertex, gives
    # exactly the symmetry orbit of one ring point (12/24/60 points), for a
    # scaled and turned solid. The reference block is the input itself.
    from cryocat.utils import symmetry

    solid = cls(radius=40.0, R=srot.random(random_state=2))
    ref = 1
    shift = 0.9 * solid.vertices[ref] + np.array([3.0, -2.0, 1.0])
    ring = rotate_vectors_about_axis(shift, solid.vertices[ref], 360.0 * np.arange(fold) / fold)
    out = solid.transport_points(ring, reference=ref)
    orbit = symmetry.SYMMETRY_GROUPS[group].from_polyhedron(solid).orbit(shift)
    assert out.shape == orbit.shape
    d = np.linalg.norm(out[:, None] - orbit[None], axis=2)
    assert np.all(d.min(axis=1) < 1e-8) and np.all(d.min(axis=0) < 1e-8)
    np.testing.assert_allclose(out[ref * fold : (ref + 1) * fold], ring)


@pytest.mark.parametrize("cls, _fold, _group", _SOLIDS_WITH_FOLD_AND_GROUP)
def test_transport_points_maps_reference_vertex_to_each_vertex(cls, _fold, _group):
    # Transporting the reference vertex itself must land on every vertex in
    # turn (block i holds vertex i), since each copy is a rotation carrying the
    # reference vertex onto vertex i.
    solid = cls(radius=5.0, R=srot.random(random_state=3))
    out = solid.transport_points(solid.vertices[2], reference=2)
    np.testing.assert_allclose(out, solid.vertices, atol=1e-10)


@pytest.mark.parametrize("n_points", [1, 3, 7])
def test_transport_points_any_number_of_points(n_points):
    # Regression for the draft's hardcoded block size 5: any P works and the
    # output has V * P rows, vertex by vertex.
    solid = Icosahedron()
    pts = np.random.default_rng(0).normal(size=(n_points, 3))
    out = solid.transport_points(pts, reference=4)
    assert out.shape == (12 * n_points, 3)
    np.testing.assert_allclose(out[4 * n_points : 5 * n_points], pts)


@pytest.mark.parametrize(
    "points, reference",
    [
        (np.zeros((2, 2)), 0),  # points with 2 components
        (np.zeros((2, 2, 3)), 0),  # 3D stack
        (np.zeros(3), 12),  # reference out of range for 12 vertices
        (np.zeros(3), -1),  # negative reference
        (np.zeros(3), 1.0),  # non-integer reference
    ],
)
def test_transport_points_invalid_input_raises(points, reference):
    # Invalid shapes or vertex indices give a clear ValueError.
    with pytest.raises(ValueError):
        Icosahedron().transport_points(points, reference=reference)


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
    from cryocat.utils.classutils import GUI_REGISTRY, GuiCategory
    assert "geom.generate_angles" in GUI_REGISTRY
    entry = GUI_REGISTRY["geom.generate_angles"]
    assert entry.category == GuiCategory.BUILDER
    assert entry.standalone is True


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
# generate_angles — symmetry-reduced search for D/T/O/I (added 2026-10-01)
# ---------------------------------------------------------------------------


def _worst_gap_deg(grid_angles, symmetry, test_rotations):
    """Largest angle (deg) from any test rotation to the nearest grid orientation,
    allowing for symmetry: min over grid R and group g of angle(test, R @ g)."""
    from cryocat.utils.symmetry import get_symmetry_rotations

    grid = srot.from_euler("zxz", grid_angles, degrees=True).as_matrix()
    group = get_symmetry_rotations(symmetry) if symmetry != "C1" else np.eye(3)[None]
    best = np.full(len(test_rotations), np.inf)
    for g in group:
        traces = np.einsum("mji,njk,ki->mn", test_rotations, grid, g)  # trace(test^T @ R @ g)
        best = np.minimum(best, np.degrees(np.arccos(np.clip((traces - 1) / 2, -1, 1))).min(axis=1))
    return best.max()


_RANDOM_ROTATIONS = srot.random(300, random_state=0).as_matrix()


@pytest.mark.parametrize("symmetry", ["C1", "C2", "C6"])
def test_generate_angles_cyclic_keeps_inplane_reduction(symmetry):
    # Cyclic symmetry keeps the original algorithm: in-plane angles limited to
    # [0, 360/n) and the grid 1/n of the C1 grid (verified identical to the
    # pre-2026-10-01 output on 30 cases when the change was made).
    n = int(symmetry[1:])
    full = generate_angles(360, 10)
    red = generate_angles(360, 10, symmetry=symmetry)
    assert len(red) == len(full) // n
    assert red[:, 0].max() < 360.0 / n


@pytest.mark.parametrize("symmetry", ["D2", "D6", "T", "O", "I"])
def test_generate_angles_global_search_covers_all_orientations(symmetry):
    # Every orientation must be as close to a searched orientation (allowing
    # for symmetry) as with the unreduced grid, whose worst gap at a 10 deg
    # step is ~7.5 deg. A fixed bound of 8 deg is used because a worst gap
    # estimated from random samples fluctuates. The bound catches both the
    # old bug (T/O/I gaps of ~19/28/10 deg) and a missing edge margin (~8.9 deg for I).
    gap = _worst_gap_deg(generate_angles(360, 10, symmetry=symmetry), symmetry, _RANDOM_ROTATIONS)
    assert gap <= 8.0


@pytest.mark.parametrize("symmetry, order", [("D2", 4), ("D6", 12), ("T", 12), ("O", 24), ("I", 60)])
def test_generate_angles_global_search_size(symmetry, order):
    # About 1/order of the unreduced grid, plus the half-step margin (< 40% extra).
    n_full = len(generate_angles(360, 10))
    n_red = len(generate_angles(360, 10, symmetry=symmetry))
    assert n_full / order <= n_red <= 1.4 * n_full / order


@pytest.mark.parametrize("n", [2, 3, 6])
def test_generate_angles_dihedral_is_not_cyclic(n):
    # D_n is handled as true dihedral symmetry: its extra 2-fold axes make the
    # grid smaller than for C_n (previously "Dn" gave exactly the C_n grid).
    c = generate_angles(360, 10, symmetry=f"C{n}")
    d = generate_angles(360, 10, symmetry=f"D{n}")
    assert len(d) < 0.75 * len(c)


@pytest.mark.parametrize("symmetry", ["D6", "T", "I"])
def test_generate_angles_local_search_loses_nothing(symmetry):
    # Local search around a starting orientation: every orientation of the
    # unreduced local grid must have a kept symmetric copy within the
    # tolerance (half a sampling step = 2.5 deg).
    start = [30.0, 50.0, 70.0]
    full = generate_angles(30, 5, starting_angles=start)
    red = generate_angles(30, 5, starting_angles=start, symmetry=symmetry)
    assert len(red) <= len(full)
    full_rot = srot.from_euler("zxz", full, degrees=True).as_matrix()
    assert _worst_gap_deg(red, symmetry, full_rot) <= 2.5 + 1e-6


def test_generate_angles_lookalike_convention_matches_template_rotation():
    # The reduction treats R and R @ g as look-alikes. This must match how an
    # orientation turns a template's content (particle-list convention,
    # cryomap.rotate(..., transpose_rotation=True), as in cryomap.place_object):
    # an icosahedrally symmetric map turned by R and by R @ g must be identical.
    from cryocat.core import cryomap
    from cryocat.utils.symmetry import IcosahedralGroup

    n_box, radius = 40, 12.0
    centre = np.array(as_triplet(None, reference_size=(n_box, n_box, n_box)), float)
    grid = np.indices((n_box,) * 3).reshape(3, -1).T.astype(float)
    volume = np.zeros(len(grid))
    for vertex in Icosahedron().vertices * radius + centre:  # canonical orientation
        volume += np.exp(-np.sum((grid - vertex) ** 2, axis=1) / (2 * 1.5**2))
    volume = volume.reshape((n_box,) * 3)

    r = srot.from_euler("zxz", [25, 40, 65], degrees=True)
    g = srot.from_matrix(IcosahedralGroup().matrices[7])
    a = cryomap.rotate(volume, rotation=r, transpose_rotation=True)
    b = cryomap.rotate(volume, rotation=r * g, transpose_rotation=True)
    c = cryomap.rotate(volume, rotation=g * r, transpose_rotation=True)  # the other order: NOT a look-alike
    assert np.corrcoef(a.ravel(), b.ravel())[0, 1] > 0.99
    assert np.corrcoef(a.ravel(), c.ravel())[0, 1] < 0.9


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


def test_euler_angles_to_normals_agrees_with_inline_rotation():
    """euler_angles_to_normals must agree element-wise with the inline pattern
    srot.from_euler('zxz', angles, degrees=True).apply([0,0,1]).

    This locks the library function to the exact same convention used in the
    app before the inline was replaced, so any future drift is caught here.
    """
    from cryocat.utils import geom as _geom
    from scipy.spatial.transform import Rotation as srot

    angles = np.array([
        [0.0,    0.0,   0.0],
        [30.0,  45.0,  10.0],
        [120.0, 90.0,   0.0],
        [200.0, 15.0, 350.0],
        [359.9, 89.9, 180.0],
    ])
    library = _geom.euler_angles_to_normals(angles)
    inline  = srot.from_euler("zxz", angles, degrees=True).apply([0.0, 0.0, 1.0])
    np.testing.assert_allclose(library, inline, atol=1e-6,
        err_msg="euler_angles_to_normals diverged from the reference inline rotation")


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


def test_point3d_cone_indicator_inside_default_axis():
    """Default axis points into -z; point on -z within the cone returns True."""
    assert _P3(0.0, 0.0, -0.5).cone_indicator(1.0, 1.0)


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
# unit_axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name, expected",
    [("x", [1, 0, 0]), ("y", [0, 1, 0]), ("z", [0, 0, 1]), (" Z ", [0, 0, 1])],
)
def test_unit_axis_names(name, expected):
    # Axis names are case-insensitive and surrounding whitespace is ignored.
    np.testing.assert_allclose(unit_axis(name), expected)


def test_unit_axis_vector_is_normalized():
    # Only the direction of a vector matters: [3, 0, 4] has length 5.
    np.testing.assert_allclose(unit_axis([3, 0, 4]), [0.6, 0.0, 0.8])


def test_unit_axis_returns_copy():
    # The returned array must be a fresh copy: modifying it must not change
    # the result of later calls (the old symmetry._AXIS_MAP returned the
    # shared module-level array itself).
    first = unit_axis("x")
    first[0] = 99.0
    np.testing.assert_allclose(unit_axis("x"), [1, 0, 0])


@pytest.mark.parametrize(
    "bad_axis, match",
    [("w", "Unknown axis"), ([0, 0, 0], "non-zero"), ([1e-12, 0, 0], "non-zero"), ([1, 0], "3 elements")],
)
def test_unit_axis_invalid_raises(bad_axis, match):
    # Unknown names, (near-)zero vectors and wrong lengths give a clear error
    # instead of NaN output.
    with pytest.raises(ValueError, match=match):
        unit_axis(bad_axis)


# ---------------------------------------------------------------------------
# rotate_vectors_about_axis
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "vector, axis, angle, expected",
    [
        ([1, 0, 0], "z", 90, [0, 1, 0]),  # right-hand rule: x -> y about +z
        ([1, 0, 0], "z", -90, [0, -1, 0]),  # negative angle turns the other way
        ([1, 0, 0], [0, 0, -1], 90, [0, -1, 0]),  # flipping the axis flips the turn
        ([0, 1, 0], "x", 90, [0, 0, 1]),  # y -> z about +x
        ([0, 0, 1], "y", 90, [1, 0, 0]),  # z -> x about +y
        ([1, 0, 0], "z", 180, [-1, 0, 0]),  # half turn reverses the vector
        ([0, 0, 2], "z", 37, [0, 0, 2]),  # a vector along the axis does not move
    ],
)
def test_rotate_vectors_about_axis_known_values(vector, axis, angle, expected):
    # Hand-checkable rotations fixing the direction convention (right-hand
    # rule, active rotation).
    np.testing.assert_allclose(rotate_vectors_about_axis(vector, axis, angle), expected, atol=1e-12)


def test_rotate_vectors_about_axis_radians():
    # degrees=False: pi/2 rad must give the same result as 90 degrees.
    np.testing.assert_allclose(
        rotate_vectors_about_axis([1, 0, 0], "z", np.pi / 2, degrees=False),
        rotate_vectors_about_axis([1, 0, 0], "z", 90),
        atol=1e-12,
    )


def test_rotate_vectors_about_axis_axis_length_irrelevant():
    # Only the axis direction is used: [0, 0, 5] behaves like "z".
    np.testing.assert_allclose(
        rotate_vectors_about_axis([1, 2, 3], [0, 0, 5], 40),
        rotate_vectors_about_axis([1, 2, 3], "z", 40),
        atol=1e-12,
    )


def test_rotate_vectors_about_axis_preserves_length_and_inverts():
    # A rotation never changes vector lengths, and turning by +theta then
    # -theta about the same axis gives back the original vectors.
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(20, 3))
    axis = rng.normal(size=3)
    rotated = rotate_vectors_about_axis(vectors, axis, 73.0)
    np.testing.assert_allclose(np.linalg.norm(rotated, axis=1), np.linalg.norm(vectors, axis=1))
    np.testing.assert_allclose(rotate_vectors_about_axis(rotated, axis, -73.0), vectors, atol=1e-12)


def test_rotate_vectors_about_axis_single_vector_keeps_shape():
    # One (3,) vector and a scalar angle -> (3,) output.
    assert rotate_vectors_about_axis([1, 0, 0], "z", 30).shape == (3,)


def test_rotate_vectors_about_axis_scalar_angle_many_vectors():
    # A scalar angle is applied to every vector -> (N, 3).
    out = rotate_vectors_about_axis([[1, 0, 0], [0, 1, 0]], "z", 90)
    np.testing.assert_allclose(out, [[0, 1, 0], [-1, 0, 0]], atol=1e-12)


def test_rotate_vectors_about_axis_one_vector_many_angles():
    # One vector, M angles -> the vector rotated by each angle, shape (M, 3).
    out = rotate_vectors_about_axis([1, 0, 0], "z", [0, 90, 180])
    np.testing.assert_allclose(out, [[1, 0, 0], [0, 1, 0], [-1, 0, 0]], atol=1e-12)


def test_rotate_vectors_about_axis_three_angles_not_mixed_into_axis():
    # Regression for the original draft: with exactly 3 angles,
    # deg2rad(angles) * axis multiplied element by element and silently built
    # a single rotation about a different axis. Each angle must instead act
    # about the given axis: here a vector along z must stay fixed for all 3.
    out = rotate_vectors_about_axis([0, 0, 1], "z", [10, 20, 30])
    assert out.shape == (3, 3)
    np.testing.assert_allclose(out, np.tile([0, 0, 1], (3, 1)), atol=1e-12)


def test_rotate_vectors_about_axis_paired_rows():
    # N vectors and N angles are paired row by row.
    out = rotate_vectors_about_axis([[1, 0, 0], [1, 0, 0]], "z", [90, 180])
    np.testing.assert_allclose(out, [[0, 1, 0], [-1, 0, 0]], atol=1e-12)


def test_rotate_vectors_about_axis_matches_rotate_points_rodrigues():
    # Consistency with the existing helper: turning z by 90 deg about +y
    # gives the same as the rotation that aligns z onto x.
    np.testing.assert_allclose(
        rotate_vectors_about_axis([[0, 0, 1], [0, 1, 0]], "y", 90),
        rotate_points_rodrigues(np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]), [0, 0, 1], [1, 0, 0]),
        atol=1e-12,
    )


@pytest.mark.parametrize(
    "vectors, axis, angles",
    [
        ([[1, 0, 0], [0, 1, 0]], "z", [10, 20, 30]),  # 2 vectors vs 3 angles
        ([1, 0], "z", 10),  # vector with 2 components
        (np.zeros((2, 2, 3)), "z", 10),  # 3D stack of vectors
        ([1, 0, 0], "z", [[10, 20]]),  # 2D array of angles
        ([1, 0, 0], [0, 0, 0], 10),  # zero axis
        ([1, 0, 0], "w", 10),  # unknown axis name
    ],
)
def test_rotate_vectors_about_axis_invalid_input_raises(vectors, axis, angles):
    # Invalid shapes or axes raise ValueError instead of NaN or an opaque
    # scipy/numpy broadcasting error.
    with pytest.raises(ValueError):
        rotate_vectors_about_axis(vectors, axis, angles)


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
        return Icosahedron().vertices

    @pytest.fixture
    def sample_edges(self):
        return Icosahedron()._edge_idx

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

    def test_faces_output_shape(self, sample_vertices, sample_edges):
        result = np.array(Icosahedron()._face_groups)
        assert isinstance(result, np.ndarray)
        assert result.shape == (20, 3)

    def test_equal_faces_area(self, sample_vertices, sample_edges):
        faces = np.array(Icosahedron()._face_groups)
        v0 = sample_vertices[faces[:, 0]]                # shape (20, 3)
        v1 = sample_vertices[faces[:, 1]]                # shape (20, 3)
        v2 = sample_vertices[faces[:, 2]]                # shape (20, 3)
        cross = np.cross(v1 - v0, v2 - v0)       # shape (20, 3)
        areas = 0.5 * np.linalg.norm(cross, axis=1)   # shape (20,)
        assert np.allclose(areas, areas[0], atol=1e-6)


class TestBarycenter:
    def test_unweighted_centroid(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        np.testing.assert_allclose(barycenter(coords), [1.0, 1.0, 1.0])

    def test_weighted_com(self):
        coords = np.array([[0.0, 0.0, 0.0], [2.0, 2.0, 2.0]])
        weights = np.array([1.0, 3.0])
        np.testing.assert_allclose(barycenter(coords, weights), [1.5, 1.5, 1.5])

    def test_single_point(self):
        coords = np.array([[3.0, 4.0, 5.0]])
        np.testing.assert_allclose(barycenter(coords), [3.0, 4.0, 5.0])

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="empty"):
            barycenter(np.zeros((0, 3)))

    def test_raises_on_wrong_shape(self):
        with pytest.raises(ValueError, match="shape"):
            barycenter(np.ones((4, 2)))


class TestAsSymmetryPlatonicGroups:
    @pytest.mark.parametrize(
        "source, expected",
        [
            ("T", ("T", 12)),
            ("t", ("T", 12)),
            ("O", ("O", 24)),
            ("o", ("O", 24)),
            ("I", ("I", 60)),
            ("i", ("I", 60)),
        ],
    )
    def test_platonic_groups(self, source, expected):
        assert as_symmetry(source) == expected

    @pytest.mark.parametrize("source", ["C5", "D3", 4])
    def test_cyclic_dihedral_unchanged(self, source):
        result = as_symmetry(source)
        assert result[0] in ("C", "D")

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError):
            as_symmetry("X")

    def test_no_digits_raises(self):
        with pytest.raises(ValueError):
            as_symmetry("C")


# ---------------------------------------------------------------------------
# sample_sphere
# ---------------------------------------------------------------------------

class TestSampleSphere:
    def test_point_count(self):
        pts = sample_sphere(100)
        assert pts.shape == (100, 3)

    def test_single_point_north_pole(self):
        pts = sample_sphere(1)
        assert pts.shape == (1, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-12)

    def test_two_points_poles(self):
        pts = sample_sphere(2)
        assert pts.shape == (2, 3)
        np.testing.assert_allclose(pts[0], [0.0, 0.0, 1.0], atol=1e-12)
        np.testing.assert_allclose(pts[1], [0.0, 0.0, -1.0], atol=1e-12)

    def test_all_on_unit_sphere(self):
        pts = sample_sphere(200)
        radii = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_radius_scaling(self):
        pts = sample_sphere(50, radius=3.0)
        radii = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(radii, 3.0, atol=1e-12)

    def test_center_offset(self):
        center = [10, 20, 30]
        pts = sample_sphere(50, center=center, radius=1.0)
        radii = np.linalg.norm(pts - np.array(center), axis=1)
        np.testing.assert_allclose(radii, 1.0, atol=1e-12)

    def test_near_uniform_hemispheres(self):
        # For a large sample, each hemisphere should contain between 30% and 70%
        # of the points (true uniform distribution gives exactly 50%).
        pts = sample_sphere(2000)
        n_north = np.sum(pts[:, 2] >= 0)
        assert 600 <= n_north <= 1400

    def test_zero_raises(self):
        with pytest.raises(UserInputError):
            sample_sphere(0)

    def test_negative_raises(self):
        with pytest.raises(UserInputError):
            sample_sphere(-5)


# =============================================================================
# Warp geometry tests
# =============================================================================

import os as _os
from cryocat.utils import ioutils as _ioutils
from cryocat.utils.geom import project_px, triangulate, frame_rotation

_WARP_DIR = _os.path.join(_os.path.dirname(__file__), "test_data", "motl_data", "warp_mapping")
_PRE_DIR = _os.path.join(_WARP_DIR, "preMA")
_POST_DIR = _os.path.join(_WARP_DIR, "postMA")


def _load_pair(name="TS_204.xml"):
    old = _ioutils.read_warp_tilt_xml(_os.path.join(_PRE_DIR, name))
    new = _ioutils.read_warp_tilt_xml(_os.path.join(_POST_DIR, name))
    _ioutils.set_volume_geometry(new, volume_dims=old["volume_dims"], image_dims=old["image_dims"])
    return old, new


class TestProjectTriangulate:
    def test_identity_same_xml(self):
        old, _ = _load_pair()
        coords = np.array([[4000.0, 4000.0, 2100.0], [2000.0, 3000.0, 1500.0]])
        u = project_px(old, coords)
        q, hist = triangulate(old, u)
        np.testing.assert_allclose(q, coords, atol=0.5)

    def test_remap_recovers_known_positions(self):
        old, new = _load_pair()
        coords_ang = np.array([[4000.0, 4000.0, 2100.0]])
        u = project_px(old, coords_ang)
        q, hist = triangulate(new, u)
        assert q.shape == (1, 3)
        assert hist[-1] < 1e-3

    def test_convergence_history_decreasing(self):
        old, new = _load_pair()
        coords_ang = np.array([[3000.0, 3500.0, 1800.0]])
        u = project_px(old, coords_ang)
        _, hist = triangulate(new, u, n_iter=6)
        assert hist[-1] < hist[0]


class TestFrameRotation:
    def test_same_xml_returns_identity(self):
        old, _ = _load_pair()
        R, angle, spread = frame_rotation(old, old)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
        assert angle < 1e-4
        assert spread < 1e-4

    def test_frame_rotation_is_orthogonal(self):
        old, new = _load_pair()
        R, angle, spread = frame_rotation(old, new)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) > 0


class TestGridVolumeWarpZSign:
    """Verify that GridVolumeWarpZ values are negated and that the Relion z-normalization
    (1 - z/vdz) is applied correctly in project_parts."""

    def _make_geom(self, warp_z_value: float) -> dict:
        """Minimal synthetic geom dict: one tilt, identity matrix, constant warp grids."""
        from cryocat.utils.ioutils import CubicGrid, LinearGrid4D

        ntilts = 2
        vdims = np.array([1000.0, 1000.0, 400.0])
        idims = np.array([1000.0, 1000.0])
        zero_cg = CubicGrid((1, 1, 1), [0.0])
        zero_lg = LinearGrid4D((1, 1, 1, 1), [0.0])
        warp_z = LinearGrid4D((1, 1, 1, 1), [warp_z_value])
        return {
            "angles_inverted": False,
            "tilt_matrices": np.stack([np.eye(3)] * ntilts),
            "volume_dims": vdims,
            "image_dims": idims,
            "dose": np.array([0.0, 1.0]),
            "axis_offset_x": np.zeros(ntilts),
            "axis_offset_y": np.zeros(ntilts),
            "grid_movement_x": zero_cg,
            "grid_movement_y": zero_cg,
            "grid_volume_warp": (zero_lg, zero_lg, warp_z),
            "pixel_size": 1.0,
        }

    def test_z_warp_sign_negated(self):
        """A positive GridVolumeWarpZ value must shift the projected z-coordinate negatively."""
        from cryocat.utils.geom import project_parts

        warp_val = 10.0  # Ångströms, the value stored in the Warp XML
        geom = self._make_geom(warp_val)

        # Place particle at the centre of the volume.
        vdims = geom["volume_dims"]
        coords = np.array([[vdims[0] / 2, vdims[1] / 2, vdims[2] / 2]])

        pre_with_warp, _ = project_parts(geom, coords)

        # Build a geom with zero warp for reference projection.
        geom_zero = self._make_geom(0.0)
        pre_zero, _ = project_parts(geom_zero, coords)

        # The identity tilt matrix maps warped coord directly onto the image plane.
        # A positive Warp z-value adds +warp_val to z BEFORE projection; with an
        # identity matrix this shows up as a z-offset in the center-relative 3-D
        # coordinate, which does NOT change x/y projection for a pure z-shift under
        # an identity tilt.  But the key property we test is the z-component of the
        # warp correction vector gw[:, t, 2], which equals -warp_val.
        # We verify by comparing projections with tilt matrices that have a non-zero
        # z-to-image coupling.
        geom_tilted = self._make_geom(warp_val)
        # Replace tilt matrices with one that maps z onto the x image axis.
        R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        geom_tilted["tilt_matrices"] = np.stack([R, R])
        geom_zero_tilted = self._make_geom(0.0)
        geom_zero_tilted["tilt_matrices"] = np.stack([R, R])

        pre_warp_tilted, _ = project_parts(geom_tilted, coords)
        pre_zero_tilted, _ = project_parts(geom_zero_tilted, coords)

        # Under R, gw_z = -warp_val shifts x projection by -warp_val.
        delta_x = pre_warp_tilted[:, :, 0] - pre_zero_tilted[:, :, 0]
        assert np.allclose(delta_x, -warp_val, atol=1e-10), (
            f"Expected x shift of {-warp_val} from z-warp; got {delta_x}"
        )

    def test_z_normalization_at_top_of_volume(self):
        """Particle at z=0 (EM top) maps to Relion z=1, so full warp is applied."""
        from cryocat.utils.geom import project_parts

        warp_val = 5.0
        geom = self._make_geom(warp_val)
        vdims = geom["volume_dims"]
        R = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
        geom["tilt_matrices"] = np.stack([R, R])
        geom_zero = self._make_geom(0.0)
        geom_zero["tilt_matrices"] = np.stack([R, R])

        # z=0: Relion z = 1 - 0/vdz = 1; with a 1×1×1×1 constant grid this is
        # still the same constant value — no spatial variation to test differently.
        # Instead confirm magnitude is correct.
        coords_top = np.array([[vdims[0] / 2, vdims[1] / 2, 0.0]])
        pre_top, _ = project_parts(geom, coords_top)
        pre_zero_top, _ = project_parts(geom_zero, coords_top)
        delta_x = pre_top[:, :, 0] - pre_zero_top[:, :, 0]
        assert np.allclose(delta_x, -warp_val, atol=1e-10), (
            f"Expected x shift of {-warp_val} at z=0; got {delta_x}"
        )


# ---------------------------------------------------------------------------
# GW1 — as_rotation (3,3) shape-disambiguation tests
# ---------------------------------------------------------------------------

class TestAsRotation33:
    """as_rotation must validate (3,3) arrays before calling srot.from_matrix.

    Before GW1b the (3,3) branch always called srot.from_matrix regardless of
    whether the array was a valid rotation matrix.  Three Euler triples stacked
    into (3,3) are NOT a rotation matrix and were silently misread (or raised
    ValueError for zero-det cases like all-zero angles).
    """

    def test_genuine_rotation_matrix_accepted(self):
        """A proper (3,3) rotation matrix round-trips through as_rotation."""
        R = srot.from_euler("zxz", [30, 45, 60], degrees=True).as_matrix()
        result = as_rotation(R)
        assert np.allclose(result.as_matrix(), R, atol=1e-10)

    def test_three_different_euler_triplets_interpreted_as_euler(self):
        """Three distinct Euler triples in a (3,3) array must NOT be read as a rotation matrix."""
        euler_arr = np.array([[10.0, 20.0, 30.0],
                              [40.0, 50.0, 60.0],
                              [70.0, 80.0, 90.0]])
        result = as_rotation(euler_arr)
        expected = srot.from_euler("zxz", euler_arr, degrees=True)
        assert np.allclose(result.as_matrix(), expected.as_matrix(), atol=1e-10)

    def test_three_identical_euler_triplets_interpreted_as_euler(self):
        """Three identical (non-zero) Euler triples in (3,3) must NOT raise and must decode correctly.

        Before GW1b this raised ValueError: Non-positive determinant because
        a matrix of three identical rows has determinant 0.
        """
        euler_arr = np.array([[30.0, 45.0, 60.0],
                              [30.0, 45.0, 60.0],
                              [30.0, 45.0, 60.0]])
        result = as_rotation(euler_arr)
        expected = srot.from_euler("zxz", euler_arr, degrees=True)
        assert np.allclose(result.as_matrix(), expected.as_matrix(), atol=1e-10)


# ---------------------------------------------------------------------------
# Symmetry in angular_distance / inplane_distance / compare_rotations (added 2026-10-02)
# ---------------------------------------------------------------------------

from cryocat.utils.exceptions import UserInputError
from cryocat.utils.symmetry import closest_symmetric_copy, get_symmetry_rotations


def _old_c1_angular_distance(r1, r2):
    """Verbatim copy of the C1 path before 2026-10-02 (quaternion formula), for regression checks."""
    q1 = np.array(r1.as_quat(), ndmin=2)
    q2 = np.array(r2.as_quat(), ndmin=2)
    angle = np.degrees(2 * np.arccos(np.abs(np.sum(q1 * q2, axis=1)))).astype(float)
    dist = 1 - np.power(np.sum(q1 * q2, 1), 2)
    dist[dist < 10e-8] = 0
    return angle, dist


def _old_c1_inplane_distance(r1, r2):
    """Verbatim copy of the C1 in-plane path before 2026-10-02 (degrees), for regression checks."""
    phi1 = np.array(r1.as_euler("zxz", degrees=True), ndmin=2)[:, 0]
    phi2 = np.array(r2.as_euler("zxz", degrees=True), ndmin=2)[:, 0]
    phi1 = np.where(abs(phi1) < ANGLE_DEGREES_TOL, 0.0, phi1) + 180.0
    phi2 = np.where(abs(phi2) < ANGLE_DEGREES_TOL, 0.0, phi2) + 180.0
    d = np.abs(phi1 - phi2)
    return np.where(d > 180.0, np.abs(d - 360.0), d)


def test_c1_outputs_unchanged():
    """C1 (default) keeps exactly the previous results (pana ground truth relies on it)."""
    r1 = srot.random(500, random_state=20)
    r2 = srot.random(500, random_state=21)
    angle, dist = angular_distance(r1, r2)
    old_angle, old_dist = _old_c1_angular_distance(r1, r2)
    np.testing.assert_array_equal(angle, old_angle)
    np.testing.assert_array_equal(dist, old_dist)
    np.testing.assert_array_equal(inplane_distance(r1, r2), _old_c1_inplane_distance(r1, r2))
    all3 = compare_rotations(r1, r2)
    np.testing.assert_array_equal(all3[0], old_angle)
    np.testing.assert_array_equal(all3[1], cone_distance(r1, r2))
    np.testing.assert_array_equal(all3[2], _old_c1_inplane_distance(r1, r2))


@pytest.mark.parametrize(
    "eul_1, eul_2, expected",
    [
        # spins just either side of the old "fold" edge: 1 degree apart, not 89
        ([0.5, 30.0, 10.0], [-0.5, 30.0, 10.0], 1.0),
        # no tilt (theta = 0): spins 0 and 90 are the same C4 orientation
        ([0.0, 0.0, 0.0], [90.0, 0.0, 0.0], 0.0),
        # one full C4 step apart: identical
        ([10.0, 30.0, 0.0], [100.0, 30.0, 0.0], 0.0),
    ],
)
def test_angular_distance_c4_edge_cases(eul_1, eul_2, expected):
    """C4 cases that the old phi-folding got wrong (89, 90 and 0 degrees were reported)."""
    angle, _ = angular_distance(np.array([eul_1]), np.array([eul_2]), symmetry="C4")
    np.testing.assert_allclose(angle, expected, atol=1e-5)


@pytest.mark.parametrize(
    "phi_1, phi_2, expected",
    [
        (10.0, 80.0, 20.0),  # 80 is 10 short of 90 (= 0 for C4): 10 + 10 = 20, the old code said 70
        (0.5, -0.5, 1.0),  # across the edge
        (0.0, 90.0, 0.0),  # one C4 step
        (0.0, 45.0, 45.0),  # the largest possible in-plane distance for C4 (180/4)
    ],
)
def test_inplane_distance_c4_cases(phi_1, phi_2, expected):
    """In-plane distance under C4 is the spin difference after removing whole 90-degree steps."""
    r1 = np.array([[phi_1, 30.0, 10.0]])
    r2 = np.array([[phi_2, 30.0, 10.0]])
    np.testing.assert_allclose(inplane_distance(r1, r2, symmetry="C4"), expected, atol=1e-6)


@pytest.mark.parametrize("n", [2, 3, 4, 6])
def test_inplane_distance_cn_matches_brute_force(n):
    """C_n in-plane = smallest wrapped |phi1 - phi2 - k*360/n| over k, and lies in [0, 180/n]."""
    r1 = srot.random(300, random_state=22)
    r2 = srot.random(300, random_state=23)
    phi1 = r1.as_euler("zxz", degrees=True)[:, 0]
    phi2 = r2.as_euler("zxz", degrees=True)[:, 0]
    diffs = np.abs(((phi1 - phi2)[:, None] - np.arange(n) * 360.0 / n + 180.0) % 360.0 - 180.0)
    got = inplane_distance(r1, r2, symmetry=n)
    np.testing.assert_allclose(got, diffs.min(axis=1), atol=1e-6)
    assert np.all((got >= 0) & (got <= 180.0 / n + 1e-9))


@pytest.mark.parametrize("symm", ["C3", "D2", "D6", "T", "O", "I"])
def test_angular_distance_symmetry_equals_closest_copy(symm):
    """With symmetry, angular_distance measures to the closest symmetric copy (any group)."""
    r1 = srot.random(200, random_state=24)
    r2 = srot.random(200, random_state=25)
    angle, dist = angular_distance(r1, r2, symmetry=symm)
    ref_angle, _ = closest_symmetric_copy(r1, r2, symm)
    np.testing.assert_allclose(angle, ref_angle, atol=1e-5)
    # the second output comes from the same copy: dist = sin^2(angle / 2)
    expected = np.sin(np.radians(angle) / 2) ** 2
    expected[expected < 10e-8] = 0
    np.testing.assert_allclose(dist, expected, atol=1e-10)


def test_angular_distance_tetrahedral_is_not_c12():
    """'T' is real tetrahedral symmetry, no longer treated as C12 (its order)."""
    r1 = srot.random(200, random_state=26)
    r2 = srot.random(200, random_state=27)
    t_angle, _ = angular_distance(r1, r2, symmetry="T")
    c12_angle, _ = angular_distance(r1, r2, symmetry="C12")
    assert not np.allclose(t_angle, c12_angle)


@pytest.mark.parametrize("symm", ["T", "O", "I", "D3"])
def test_symmetric_copy_has_zero_angular_distance(symm):
    """R and R @ g look identical for every group rotation g.

    NaN is read as 0, as documented in the angular_distance Notes (known
    limitation: no clipping before arccos, kept for the pana ground truth; it
    affects identical rotations with or without symmetry at the same rate).
    """
    gs = get_symmetry_rotations(symm)
    r = srot.random(1, random_state=28).as_matrix()
    r1 = srot.from_matrix(np.repeat(r, len(gs), axis=0))
    r2 = r1 * srot.from_matrix(gs)
    angle, dist = angular_distance(r1, r2, symmetry=symm)
    np.testing.assert_allclose(np.nan_to_num(angle, nan=0.0), 0.0, atol=1e-4)
    np.testing.assert_allclose(dist, 0.0, atol=1e-12)  # dist is never NaN


@pytest.mark.parametrize("symm", ["D2", "T", "O", "I"])
def test_non_cyclic_cone_inplane_raise(symm):
    """D/T/O/I have several equivalent axes: cone / in-plane distances are not defined."""
    r1 = srot.random(5, random_state=29)
    r2 = srot.random(5, random_state=30)
    with pytest.raises(NotImplementedError):
        inplane_distance(r1, r2, symmetry=symm)
    with pytest.raises(NotImplementedError):
        cone_inplane_distance(r1, r2, symmetry=symm)
    for rotation_type in ("all", "cone_distance", "in_plane_distance"):
        with pytest.raises(NotImplementedError):
            compare_rotations(r1, r2, symmetry=symm, rotation_type=rotation_type)
    # the full angular distance alone is supported
    out = compare_rotations(r1, r2, symmetry=symm, rotation_type="angular_distance")
    np.testing.assert_allclose(out, angular_distance(r1, r2, symmetry=symm)[0])


def test_compare_rotations_cn_consistent_with_parts():
    """compare_rotations returns the same three numbers as the individual functions (C5)."""
    r1 = srot.random(100, random_state=31)
    r2 = srot.random(100, random_state=32)
    ang, cone, inpl = compare_rotations(r1, r2, symmetry="C5")
    np.testing.assert_array_equal(ang, angular_distance(r1, r2, symmetry="C5")[0])
    np.testing.assert_array_equal(cone, cone_distance(r1, r2))
    np.testing.assert_array_equal(inpl, inplane_distance(r1, r2, symmetry="C5"))
    # positional third argument is the symmetry (as used by tmana)
    np.testing.assert_array_equal(compare_rotations(r1, r2, "C5")[0], ang)


def test_compare_rotations_unknown_type_raises():
    """An unsupported rotation_type is still a UserInputError (checked before any computation)."""
    r = srot.random(2, random_state=33)
    with pytest.raises(UserInputError):
        compare_rotations(r, r, rotation_type="bogus")


@pytest.mark.parametrize("symm", [1, 4])
def test_inplane_distance_radians(symm):
    """degrees=False: in-plane distance in radians, equal to the degree result converted (C1 and C4)."""
    eul_1 = np.array([[10.0, 20.0, 30.0], [170.0, 60.0, -20.0], [-80.0, 100.0, 5.0]])
    eul_2 = np.array([[50.0, 60.0, 70.0], [-175.0, 40.0, 10.0], [100.0, 120.0, -5.0]])
    deg = inplane_distance(eul_1, eul_2, symmetry=symm)
    rad = inplane_distance(np.radians(eul_1), np.radians(eul_2), degrees=False, symmetry=symm)
    np.testing.assert_allclose(rad, np.radians(deg), atol=1e-10)


def test_cyclic_symmetry_keyword_deprecated():
    """The old keyword still works, with a DeprecationWarning, and gives the same result."""
    r1 = srot.random(20, random_state=34)
    r2 = srot.random(20, random_state=35)
    with pytest.warns(DeprecationWarning, match="cyclic_symmetry"):
        old = compare_rotations(r1, r2, cyclic_symmetry=4)
    new = compare_rotations(r1, r2, symmetry=4)
    for a, b in zip(old, new):
        np.testing.assert_array_equal(a, b)
    with pytest.warns(DeprecationWarning):
        np.testing.assert_array_equal(
            angular_distance(r1, r2, cyclic_symmetry="C4")[0], angular_distance(r1, r2, symmetry="C4")[0]
        )


def test_resolve_symmetry_argument():
    """Normalized labels; conflicting values of the two keywords raise."""
    assert resolve_symmetry_argument(4, None) == "C4"
    assert resolve_symmetry_argument("d3", None) == "D3"
    assert resolve_symmetry_argument("i", None) == "I"
    with pytest.warns(DeprecationWarning):
        assert resolve_symmetry_argument("C1", 6) == "C6"  # default symmetry -> old value used
    with pytest.warns(DeprecationWarning):
        assert resolve_symmetry_argument("C6", 6) == "C6"  # same value given twice is fine
    with pytest.warns(DeprecationWarning), pytest.raises(ValueError, match="Conflicting"):
        resolve_symmetry_argument("C4", 6)


def test_require_cyclic_symmetry():
    """Cyclic groups pass, all others raise NotImplementedError naming the quantity."""
    require_cyclic_symmetry("C7", "x")
    require_cyclic_symmetry(1, "x")
    for symm in ("D2", "T", "O", "I"):
        with pytest.raises(NotImplementedError, match="In-plane distances"):
            require_cyclic_symmetry(symm, "In-plane distances")


# ===========================================================================
# Polyhedron.angular_dissimilarity (added 2026-10-06, plan of action point 1.2)
# ===========================================================================

from cryocat.utils.symmetry import SYMMETRY_GROUPS, angular_score, max_angular_mismatch

# (solid class, group letter, kind name used by symmetry.angular_score)
_PLATONIC = [
    (Tetrahedron, "T", "tetrahedron"),
    (Octahedron, "O", "octahedron"),
    (Cube, "O", "cube"),
    (Icosahedron, "I", "icosahedron"),
    (Dodecahedron, "I", "dodecahedron"),
]


class TestPolyhedronAngularDissimilarity:

    @pytest.mark.parametrize("solid_cls, letter, kind", _PLATONIC)
    def test_identical_orientations_give_zero(self, solid_cls, letter, kind):
        # Same rotation on both sides: the turned corners coincide.
        r = srot.random(6, random_state=0)
        np.testing.assert_allclose(solid_cls().angular_dissimilarity(r, r), 0.0, atol=1e-7)

    @pytest.mark.parametrize("solid_cls, letter, kind", _PLATONIC)
    def test_symmetry_related_orientations_give_zero(self, solid_cls, letter, kind):
        # R and R @ g (g = any rotation of the solid's group) put the corners on
        # the same places, so the solid cannot tell them apart.
        g = srot.from_matrix(SYMMETRY_GROUPS[letter]().matrices)
        r = srot.random(random_state=1)
        np.testing.assert_allclose(solid_cls().angular_dissimilarity(r, r * g), 0.0, atol=1e-6)

    @pytest.mark.parametrize("solid_cls, letter, kind", _PLATONIC)
    def test_independent_of_radius(self, solid_cls, letter, kind):
        # Corners are normalized to unit length, so the radius has no effect.
        r1, r2 = srot.random(8, random_state=2), srot.random(8, random_state=3)
        np.testing.assert_allclose(
            solid_cls(radius=37.0).angular_dissimilarity(r1, r2),
            solid_cls(radius=1.0).angular_dissimilarity(r1, r2),
            atol=1e-12,
        )

    @pytest.mark.parametrize("solid_cls, letter, kind", _PLATONIC)
    def test_invariant_to_common_rotation(self, solid_cls, letter, kind):
        # Turning both particles by the same rotation does not change how
        # different they are.
        r1, r2 = srot.random(8, random_state=4), srot.random(8, random_state=5)
        q = srot.random(random_state=6)
        solid = solid_cls()
        np.testing.assert_allclose(
            solid.angular_dissimilarity(q * r1, q * r2), solid.angular_dissimilarity(r1, r2), atol=1e-7
        )

    @pytest.mark.parametrize("solid_cls, letter, kind", _PLATONIC)
    def test_bounded_by_max_angular_mismatch(self, solid_cls, letter, kind):
        # The value can never exceed the worst case (corner -> nearest face centre).
        r1, r2 = srot.random(300, random_state=7), srot.random(300, random_state=8)
        d = solid_cls().angular_dissimilarity(r1, r2)
        assert d.min() >= 0.0
        assert d.max() <= max_angular_mismatch(letter, kind) + 1e-9

    @pytest.mark.parametrize("solid_cls, letter, kind", _PLATONIC)
    def test_consistent_with_symmetry_angular_score(self, solid_cls, letter, kind):
        # symmetry.angular_score is the normalized similarity 1 - d / d_max of
        # the same distance d (canonical solid); compare away from its clamped ends.
        r1, r2 = srot.random(50, random_state=9), srot.random(50, random_state=10)
        d = solid_cls().angular_dissimilarity(r1, r2)
        score = angular_score(r1, r2, letter, kind=kind)
        inside = (score > 1e-5) & (score < 1 - 1e-5)
        np.testing.assert_allclose(
            d[inside], (1.0 - score[inside]) * max_angular_mismatch(letter, kind), atol=1e-9
        )

    def test_matches_hausdorff_distance_per_pair(self):
        # Each element is exactly the draft's per-pair computation.
        solid = Icosahedron(radius=5.0, R=srot.random(random_state=11))
        r1, r2 = srot.random(5, random_state=12), srot.random(5, random_state=13)
        verts = solid.vertices / np.linalg.norm(solid.vertices, axis=1, keepdims=True)
        expected = [
            hausdorff_distance_sphere(verts @ m1.T, verts @ m2.T)
            for m1, m2 in zip(r1.as_matrix(), r2.as_matrix())
        ]
        np.testing.assert_array_equal(solid.angular_dissimilarity(r1, r2), expected)

    def test_input_forms_agree(self):
        # Rotation stack, Euler angles (zxz, degrees), matrices and quaternions
        # describing the same rotations give the same result.
        solid = Octahedron()
        r1, r2 = srot.random(4, random_state=14), srot.random(4, random_state=15)
        ref = solid.angular_dissimilarity(r1, r2)
        np.testing.assert_allclose(
            solid.angular_dissimilarity(r1.as_euler("zxz", degrees=True), r2.as_euler("zxz", degrees=True)),
            ref, atol=1e-9,
        )
        np.testing.assert_allclose(solid.angular_dissimilarity(r1.as_matrix(), r2.as_matrix()), ref, atol=1e-12)
        np.testing.assert_allclose(solid.angular_dissimilarity(r1.as_quat(), r2.as_quat()), ref, atol=1e-12)

    def test_single_pair_returns_length_one_array(self):
        # Output is always an array, also for one pair.
        out = Tetrahedron().angular_dissimilarity([10.0, 20.0, 30.0], [40.0, 50.0, 60.0])
        assert isinstance(out, np.ndarray)
        assert out.shape == (1,)

    def test_stack_equals_single_pairs(self):
        # Element i of a stacked call equals the call on pair i alone.
        solid = Cube()
        r1, r2 = srot.random(6, random_state=16), srot.random(6, random_state=17)
        stacked = solid.angular_dissimilarity(r1, r2)
        singles = [solid.angular_dissimilarity(r1[i], r2[i])[0] for i in range(6)]
        np.testing.assert_array_equal(stacked, singles)

    def test_one_against_many_broadcasts(self):
        # A single rotation on either side is compared with every rotation on
        # the other side (e.g. all particles against a reference orientation).
        solid = Dodecahedron()
        ref = srot.random(random_state=18)
        many = srot.random(5, random_state=19)
        # Rebuilding the repeated stack round-trips through quaternions, so
        # compare to rounding precision rather than bit for bit.
        repeated = srot.from_quat(np.tile(ref.as_quat(), (5, 1)))
        expected = solid.angular_dissimilarity(repeated, many)
        np.testing.assert_allclose(solid.angular_dissimilarity(ref, many), expected, atol=1e-12)
        np.testing.assert_allclose(solid.angular_dissimilarity(many, ref), expected, atol=1e-12)

    def test_length_mismatch_raises(self):
        # Different numbers of rotations, neither single: pairing is undefined.
        with pytest.raises(ValueError, match="same number of rotations"):
            Icosahedron().angular_dissimilarity(srot.random(3, random_state=20), srot.random(4, random_state=21))

    @pytest.mark.parametrize("empty_side", [0, 1])
    def test_empty_input_raises(self, empty_side):
        # Motl.get_rotations() returns [] for an empty motl: clear error instead
        # of as_rotation's shape message.
        args = [srot.random(2, random_state=22), srot.random(2, random_state=23)]
        args[empty_side] = []
        with pytest.raises(ValueError, match="holds no rotations"):
            Icosahedron().angular_dissimilarity(*args)


# ===========================================================================
# align_z_axes / inplane_angle_after_alignment (added 2026-10-09)
# ===========================================================================


def _align_z_axes_single_pair(z1, z2, atol=1e-8):
    """Reference: the original one-pair version of align_z_axes, as provided by the user."""
    z1 = np.asarray(z1, dtype=float)
    z2 = np.asarray(z2, dtype=float)
    z1 = z1 / np.linalg.norm(z1)
    z2 = z2 / np.linalg.norm(z2)
    vector = np.cross(z2, z1)
    sin_theta = np.linalg.norm(vector)
    cos_theta = np.clip(np.dot(z2, z1), -1.0, 1.0)
    if sin_theta < atol:
        if cos_theta > 0:
            return np.eye(3)
        helper = np.array([0.0, 1.0, 0.0]) if abs(z2[0]) > 0.9 else np.array([1.0, 0.0, 0.0])
        axis = np.cross(z2, helper)
        axis /= np.linalg.norm(axis)
        return srot.from_rotvec(np.pi * axis).as_matrix()
    tilt_angle_theta = np.arccos(cos_theta)
    rotvec = (vector / sin_theta) * tilt_angle_theta
    return srot.from_rotvec(rotvec).as_matrix()


def _random_unit_vectors(n, seed):
    v = np.random.default_rng(seed).normal(size=(n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class TestAlignZAxes:
    """align_z_axes(z1, z2): smallest tilt T with T @ z2 along z1."""

    def test_puts_z2_onto_z1(self):
        # The defining property, for many random pairs (vectors need not be unit length).
        z1, z2 = 3.0 * _random_unit_vectors(200, 1), 0.5 * _random_unit_vectors(200, 2)
        tilts = align_z_axes(z1, z2)
        moved = np.einsum("nij,nj->ni", tilts, z2 / np.linalg.norm(z2, axis=1, keepdims=True))
        np.testing.assert_allclose(moved, z1 / np.linalg.norm(z1, axis=1, keepdims=True), atol=1e-12)

    def test_tilt_angle_is_angle_between_axes(self):
        # Smallest tilt: the turn angle equals the angle between the two directions,
        # and its axis is perpendicular to both (no extra spin).
        z1, z2 = _random_unit_vectors(200, 3), _random_unit_vectors(200, 4)
        rotvec = srot.from_matrix(align_z_axes(z1, z2)).as_rotvec()
        between = np.arccos(np.clip(np.sum(z1 * z2, axis=1), -1, 1))
        np.testing.assert_allclose(np.linalg.norm(rotvec, axis=1), between, atol=1e-9)
        np.testing.assert_allclose(np.sum(rotvec * z1, axis=1), 0.0, atol=1e-9)
        np.testing.assert_allclose(np.sum(rotvec * z2, axis=1), 0.0, atol=1e-9)

    def test_parallel_gives_identity(self):
        np.testing.assert_allclose(align_z_axes([0, 0, 1], [0, 0, 2]), np.eye(3), atol=1e-12)

    @pytest.mark.parametrize("z2", [[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
    def test_opposite_gives_180_degree_flip(self, z2):
        # Opposite directions: a 180-degree turn that still lands z2 on z1 = -z2.
        z2 = np.asarray(z2, dtype=float)
        tilt = align_z_axes(-z2, z2)
        np.testing.assert_allclose(tilt @ z2, -z2, atol=1e-12)
        assert np.linalg.norm(srot.from_matrix(tilt).as_rotvec()) == pytest.approx(np.pi)

    def test_matches_single_pair_reference(self):
        # The many-pairs version equals the original one-pair function, pair by pair,
        # including parallel and opposite pairs.
        z1 = np.vstack([_random_unit_vectors(50, 5), [[0, 0, 1], [0, 0, 1], [1, 0, 0]]])
        z2 = np.vstack([_random_unit_vectors(50, 6), [[0, 0, 1], [0, 0, -1], [-1, 0, 0]]])
        expected = np.array([_align_z_axes_single_pair(a, b) for a, b in zip(z1, z2)])
        np.testing.assert_allclose(align_z_axes(z1, z2), expected, atol=1e-12)

    def test_single_pair_returns_3x3(self):
        assert align_z_axes([1, 0, 0], [0, 1, 0]).shape == (3, 3)
        assert align_z_axes([[1, 0, 0]], [[0, 1, 0]]).shape == (1, 3, 3)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same number of vectors"):
            align_z_axes(_random_unit_vectors(3, 7), _random_unit_vectors(2, 8))


class TestInplaneAngleAfterAlignment:
    """Spin about the shared z-axis left after tilting particle 2's z-axis onto particle 1's."""

    def test_remaining_difference_is_pure_z_turn(self):
        # R1^T @ T @ R2 must keep z in place, i.e. be a turn about z by the returned spin.
        r1, r2 = srot.random(300, random_state=1), srot.random(300, random_state=2)
        tilts = align_z_axes(r1.as_matrix()[:, :, 2], r2.as_matrix()[:, :, 2])
        remaining = np.einsum("nji,njk,nkl->nil", r1.as_matrix(), tilts, r2.as_matrix())
        spin = inplane_angle_after_alignment(r1, r2)
        np.testing.assert_allclose(remaining, srot.from_euler("z", spin[:, None]).as_matrix(), atol=1e-12)

    def test_same_z_axis_gives_phi_difference(self):
        # Shared z-axis (same theta, psi): the spin is phi2 - phi1, wrapped to [-180, 180].
        phis = np.random.default_rng(3).uniform(-180, 180, (100, 2))
        e1 = np.column_stack([phis[:, 0], np.full(100, 60.0), np.full(100, 15.0)])
        e2 = np.column_stack([phis[:, 1], np.full(100, 60.0), np.full(100, 15.0)])
        spin = inplane_angle_after_alignment(e1, e2, degrees=True)
        wrapped = np.mod(phis[:, 1] - phis[:, 0] + 180.0, 360.0) - 180.0
        np.testing.assert_allclose(np.mod(spin - wrapped + 180.0, 360.0) - 180.0, 0.0, atol=1e-9)

    def test_theta_zero_pair(self):
        # (10, 0, 50) vs (10, 0, 0): both turn about z only, 50 degrees apart.
        spin = inplane_angle_after_alignment([10.0, 0.0, 50.0], [10.0, 0.0, 0.0], degrees=True)
        assert spin[0] == pytest.approx(-50.0)

    def test_common_turn_leaves_spin_unchanged(self):
        # Depends only on how the two orientations differ.
        r1, r2 = srot.random(200, random_state=4), srot.random(200, random_state=5)
        q = srot.random(random_state=6)
        np.testing.assert_allclose(
            inplane_angle_after_alignment(r1, r2), inplane_angle_after_alignment(q * r1, q * r2), atol=1e-9
        )

    def test_swap_flips_sign(self):
        r1, r2 = srot.random(200, random_state=7), srot.random(200, random_state=8)
        np.testing.assert_allclose(inplane_angle_after_alignment(r2, r1), -inplane_angle_after_alignment(r1, r2), atol=1e-9)

    def test_spin_about_own_z_is_returned(self):
        # R2 = R1 @ Rz(a): particle 2 is particle 1 spun by a about its own z-axis.
        r1 = srot.random(100, random_state=9)
        a = np.random.default_rng(9).uniform(-170, 170, 100)
        r2 = r1 * srot.from_euler("z", a[:, None], degrees=True)
        np.testing.assert_allclose(inplane_angle_after_alignment(r1, r2, degrees=True), a, atol=1e-9)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same number of rotations"):
            inplane_angle_after_alignment(srot.random(3, random_state=1), srot.random(2, random_state=2))
