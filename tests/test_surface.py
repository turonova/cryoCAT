"""Tests for Ellipsoid in cryocat.core.surface."""
from __future__ import annotations


import numpy as np
import pytest
from cryocat.core.surface import Ellipsoid

@pytest.fixture
def axis_aligned_ellipsoid():
    """Ellipsoid with known geometry, non-zero centre, distinct semi-axes."""
    a, b, c = 5.0, 3.0, 2.0
    center = np.array([10.0, 5.0, -3.0])

    # Dense grid of surface points
    theta = np.linspace(0, 2 * np.pi, 80)
    phi = np.linspace(-np.pi / 2, np.pi / 2, 80)
    T, P = np.meshgrid(theta, phi)
    pts = np.column_stack([
        center[0] + a * np.cos(T).ravel() * np.cos(P).ravel(),
        center[1] + b * np.sin(T).ravel() * np.cos(P).ravel(),
        center[2] + c * np.sin(P).ravel(),
    ])
    return pts, center, np.array([a, b, c])


@pytest.fixture
def fitted_ellipsoid(axis_aligned_ellipsoid):
    pts, center, radii = axis_aligned_ellipsoid
    return Ellipsoid.fit_to_points(pts), center, radii

def test_fit_returns_ellipsoid_instance(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    assert isinstance(el, Ellipsoid)


def test_fit_not_singular(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    assert not el.singular


def test_fit_center(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    assert np.allclose(el.center, center, atol=0.5)


def test_fit_radii(fitted_ellipsoid):
    el, _, radii = fitted_ellipsoid
    # radii may be returned in any order; compare sorted values
    assert np.allclose(np.sort(np.abs(el.radii)), np.sort(radii), atol=0.5)


def test_fit_params_length(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    assert el.params.shape == (10,)


def test_fit_singular_on_too_few_points():
    el = Ellipsoid.fit_to_points(np.random.rand(2, 3))
    assert el.singular


def test_get_props_as_ndarray_shape(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    arr = el.get_props_as_ndarray()
    assert arr.shape == (25,), f"expected (25,), got {arr.shape}"


def test_get_props_as_ndarray_contains_center(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    arr = el.get_props_as_ndarray()
    assert np.allclose(arr[:3], el.center, atol=1e-10)


def test_get_props_as_ndarray_contains_params(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    arr = el.get_props_as_ndarray()
    assert np.allclose(arr[15:], el.params, atol=1e-10)


def test_get_props_as_df_columns(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    df = el.get_props_as_df()
    assert list(df.columns) == Ellipsoid.columns


def test_get_props_as_df_eigenvectors_distinct(fitted_ellipsoid):
    """ev2 and ev3 rows must differ from ev1 (tests the old copy-paste bug)."""
    el, _, _ = fitted_ellipsoid
    df = el.get_props_as_df()
    ev1 = df[["ev1x", "ev1y", "ev1z"]].values.flatten()
    ev2 = df[["ev2x", "ev2y", "ev2z"]].values.flatten()
    ev3 = df[["ev3x", "ev3y", "ev3z"]].values.flatten()
    assert not np.allclose(ev1, ev2), "ev2 is identical to ev1 — copy-paste bug present"
    assert not np.allclose(ev1, ev3), "ev3 is identical to ev1 — copy-paste bug present"
    assert not np.allclose(ev2, ev3), "ev2 and ev3 are identical"


def test_get_props_as_df_center(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    df = el.get_props_as_df()
    assert np.allclose(df[["cx", "cy", "cz"]].values.flatten(), el.center, atol=1e-10)


def test_roundtrip_ndarray(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    arr = el.get_props_as_ndarray()
    el2 = Ellipsoid.from_array_like(arr)
    assert np.allclose(el2.center, el.center, atol=1e-10)
    assert np.allclose(el2.radii, el.radii, atol=1e-10)


def test_roundtrip_params_only(fitted_ellipsoid):
    """from_array_like with 10-element params vector recomputes geometry."""
    el, center, _ = fitted_ellipsoid
    el2 = Ellipsoid.from_array_like(el.params)
    assert np.allclose(el2.center, center, atol=0.5)


def test_roundtrip_dict(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    d = el.get_props_as_dict()
    el2 = Ellipsoid.from_dict(d)
    assert np.allclose(el2.center, el.center, atol=1e-10)
    assert np.allclose(el2.radii, el.radii, atol=1e-10)


def test_roundtrip_df(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    df = el.get_props_as_df()
    el2 = Ellipsoid.from_df(df)
    assert np.allclose(el2.center, el.center, atol=1e-10)
    assert np.allclose(el2.radii, el.radii, atol=1e-10)
    assert np.allclose(el2.e_vec1, el.e_vec1, atol=1e-10)
    assert np.allclose(el2.e_vec2, el.e_vec2, atol=1e-10)
    assert np.allclose(el2.e_vec3, el.e_vec3, atol=1e-10)


def test_roundtrip_points(axis_aligned_ellipsoid):
    """from_array_like with (N,3) points fits and recovers center."""
    pts, center, _ = axis_aligned_ellipsoid
    el = Ellipsoid.from_array_like(pts)
    assert np.allclose(el.center, center, atol=0.5)


def test_distance_surface_point_is_zero(fitted_ellipsoid):
    """A point sitting on the ellipsoid tip along the major axis is at distance ~0."""
    el, center, radii = fitted_ellipsoid
    a = np.max(np.abs(el.radii))
    # find which eigenvector corresponds to the largest radius
    idx = np.argmax(np.abs(el.radii))
    evecs = np.column_stack([el.e_vec1, el.e_vec2, el.e_vec3])
    surface_point = el.center + a * evecs[:, idx]
    dist = el.distance_point_surface(surface_point)
    assert dist < 0.1, f"expected ~0, got {dist:.4f}"


def test_distance_offset_from_surface(fitted_ellipsoid):
    """Point offset by d beyond the major-axis tip should have distance ~d."""
    el, center, _ = fitted_ellipsoid
    a = np.max(np.abs(el.radii))
    idx = np.argmax(np.abs(el.radii))
    evecs = np.column_stack([el.e_vec1, el.e_vec2, el.e_vec3])
    d = 2.0
    outside_point = el.center + (a + d) * evecs[:, idx]
    dist = el.distance_point_surface(outside_point)
    assert abs(dist - d) < 0.2, f"expected ~{d}, got {dist:.4f}"


def test_distance_is_positive(fitted_ellipsoid):
    """Any point outside returns a positive distance."""
    el, center, radii = fitted_ellipsoid
    far_point = el.center + np.array([100.0, 0.0, 0.0])
    assert el.distance_point_surface(far_point) > 0

def test_translate_shifts_center(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    v = np.array([1.0, -2.0, 3.0])
    el.translate(v)
    assert np.allclose(el.center, center + v, atol=1e-10)


def test_translate_does_not_change_radii(fitted_ellipsoid):
    el, _, radii = fitted_ellipsoid
    original_radii = el.radii.copy()
    el.translate(np.array([5.0, 5.0, 5.0]))
    assert np.allclose(el.radii, original_radii, atol=1e-10)


def test_rotate_identity_leaves_center(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    el.rotate(np.eye(3))
    assert np.allclose(el.center, np.eye(3).dot(center), atol=1e-10)


def test_rotate_90_degrees(fitted_ellipsoid):
    """90° rotation around z-axis swaps x and y of centre."""
    el, center, _ = fitted_ellipsoid
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    el.rotate(R)
    expected_center = R.dot(center)
    assert np.allclose(el.center, expected_center, atol=1e-10)


def test_rotate_preserves_eigenvector_orthogonality(fitted_ellipsoid):
    el, _, _ = fitted_ellipsoid
    theta = np.pi / 6
    R = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0, 0, 1],
    ])
    el.rotate(R)
    assert abs(np.dot(el.e_vec1, el.e_vec2)) < 1e-6
    assert abs(np.dot(el.e_vec1, el.e_vec3)) < 1e-6
    assert abs(np.dot(el.e_vec2, el.e_vec3)) < 1e-6

def test_transform_identity_leaves_center(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    el.transform(np.eye(4))
    assert np.allclose(el.center, center, atol=1e-10)


def test_transform_pure_translation(fitted_ellipsoid):
    el, center, _ = fitted_ellipsoid
    M = np.eye(4)
    M[:3, 3] = [1.0, 2.0, 3.0]
    el.transform(M)
    assert np.allclose(el.center, center + np.array([1.0, 2.0, 3.0]), atol=1e-10)
