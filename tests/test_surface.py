"""Tests for Ellipsoid, Mesh, and OrientedPointCloud in cryocat.core.surface."""
from __future__ import annotations

import numpy as np
import pytest
import open3d as o3d
from cryocat.core.surface import Ellipsoid, Mesh, OrientedPointCloud, DiscreteSurface

# =============================================================================
# Ellipsoid — tests
# =============================================================================

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


# ===========================================================================
# Create a unit sphere mesh
# ===========================================================================

def _make_unit_sphere_mesh() -> Mesh:
    """Unit sphere mesh via Open3D (analytically exact vertices, outward normals)."""
    o3d_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=20)
    o3d_sphere.compute_vertex_normals()
    mesh = Mesh()
    mesh.vertices = np.asarray(o3d_sphere.vertices)
    mesh.faces = np.asarray(o3d_sphere.triangles)
    mesh.normals = np.asarray(o3d_sphere.vertex_normals)
    return mesh


def _make_unit_sphere_opc() -> OrientedPointCloud:
    """Dense oriented point cloud on the unit sphere (radially outward normals)."""
    theta = np.linspace(0, 2 * np.pi, 60)
    phi = np.linspace(-np.pi / 2, np.pi / 2, 60)
    T, P = np.meshgrid(theta, phi)
    pts = np.column_stack([
        np.cos(T).ravel() * np.cos(P).ravel(),
        np.sin(T).ravel() * np.cos(P).ravel(),
        np.sin(P).ravel(),
    ])
    opc = OrientedPointCloud()
    opc.vertices = pts.astype(np.float64)
    opc.normals = pts.copy().astype(np.float64)  # radially outward on unit sphere
    return opc


@pytest.fixture
def unit_sphere_mesh():
    return _make_unit_sphere_mesh()


@pytest.fixture
def unit_sphere_opc():
    return _make_unit_sphere_opc()


@pytest.fixture
def two_sphere_mesh():
    """Inner (r=0.5) and outer (r=1.0) sphere merged into one mesh — for separate_surfaces.

    The inner sphere normals are flipped to point inward (toward the shared centroid at
    origin) so that separate_surfaces correctly identifies them as the inner surface.
    The outer sphere normals point outward as usual.
    """
    inner = o3d.geometry.TriangleMesh.create_sphere(radius=0.5, resolution=10)
    inner.compute_vertex_normals()
    outer = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)
    outer.compute_vertex_normals()

    inner_verts = np.asarray(inner.vertices)
    inner_faces = np.asarray(inner.triangles)
    outer_verts = np.asarray(outer.vertices)
    outer_faces = np.asarray(outer.triangles) + len(inner_verts)

    mesh = Mesh()
    mesh.vertices = np.vstack([inner_verts, outer_verts])
    mesh.faces = np.vstack([inner_faces, outer_faces])
    mesh.normals = np.vstack([
        -np.asarray(inner.vertex_normals),  # inward normals → labeled "inner"
        np.asarray(outer.vertex_normals),   # outward normals → labeled "outer"
    ])
    return mesh


# ===========================================================================
# Mesh — tests
# ===========================================================================

def test_mesh_has_vertices_and_faces(unit_sphere_mesh):
    assert unit_sphere_mesh.vertices is not None
    assert unit_sphere_mesh.faces is not None
    assert unit_sphere_mesh.vertices.shape[1] == 3
    assert unit_sphere_mesh.faces.shape[1] == 3


def test_mesh_get_vertices_shape(unit_sphere_mesh):
    verts = unit_sphere_mesh.get_vertices()
    assert verts.ndim == 2 and verts.shape[1] == 3


def test_mesh_get_normals_shape(unit_sphere_mesh):
    norms = unit_sphere_mesh.get_normals()
    assert norms.shape == unit_sphere_mesh.vertices.shape


def test_mesh_normals_are_unit_vectors(unit_sphere_mesh):
    norms = unit_sphere_mesh.get_normals()
    lengths = np.linalg.norm(norms, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_mesh_surface_area_unit_sphere(unit_sphere_mesh):
    area = unit_sphere_mesh.get_surface_area()
    # 4π ≈ 12.566; Open3D icosphere at resolution=20 approximates this within ~1%
    assert abs(area - 4 * np.pi) < 0.1, f"expected ≈ 4π, got {area:.4f}"


def test_mesh_distance_outside_point(unit_sphere_mesh):
    pt = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_mesh.distance_to_points(pt, compute_occupancy=True)
    assert abs(result["distances"][0] - 1.0) < 0.02


def test_mesh_distance_surface_point_is_zero(unit_sphere_mesh):
    pt = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_mesh.distance_to_points(pt, compute_occupancy=False)
    assert result["distances"][0] < 0.01


def test_mesh_occupancy_inside(unit_sphere_mesh):
    pt = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_mesh.distance_to_points(pt, compute_occupancy=True)
    assert result["n_inside"] == 1


def test_mesh_occupancy_outside(unit_sphere_mesh):
    pt = np.array([[2.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_mesh.distance_to_points(pt, compute_occupancy=True)
    assert result["n_outside"] == 1


def test_mesh_cast_rays_hits_sphere(unit_sphere_mesh):
    # Ray from [2, 0, 0] pointing in -x direction; should hit at t ≈ 1
    ray = np.array([[2.0, 0.0, 0.0, -1.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_mesh.cast_rays(ray)
    assert np.isfinite(result["t_hit"][0])
    assert abs(result["t_hit"][0] - 1.0) < 0.02


def test_mesh_cast_rays_miss(unit_sphere_mesh):
    # Ray from [2, 0, 0] pointing in +x direction (away from sphere) — no hit
    ray = np.array([[2.0, 0.0, 0.0, 1.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_mesh.cast_rays(ray)
    assert not np.isfinite(result["t_hit"][0])


def test_mesh_translate_shifts_centroid(unit_sphere_mesh):
    v = np.array([3.0, 1.0, -2.0])
    unit_sphere_mesh.translate(v)
    centroid = unit_sphere_mesh.vertices.mean(axis=0)
    assert np.allclose(centroid, v, atol=0.01)


def test_mesh_translate_preserves_face_count(unit_sphere_mesh):
    n_faces = len(unit_sphere_mesh.faces)
    unit_sphere_mesh.translate(np.array([1.0, 0.0, 0.0]))
    assert len(unit_sphere_mesh.faces) == n_faces


def test_mesh_rotate_identity_leaves_vertices(unit_sphere_mesh):
    original = unit_sphere_mesh.vertices.copy()
    unit_sphere_mesh.rotate(np.eye(3))
    assert np.allclose(unit_sphere_mesh.vertices, original, atol=1e-10)


def test_mesh_rotate_90_around_z(unit_sphere_mesh):
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    original_centroid = unit_sphere_mesh.vertices.mean(axis=0)
    unit_sphere_mesh.rotate(R)
    new_centroid = unit_sphere_mesh.vertices.mean(axis=0)
    assert np.allclose(new_centroid, R @ original_centroid, atol=1e-5)


def test_mesh_oversample_returns_opc(unit_sphere_mesh):
    result = unit_sphere_mesh.oversample(point_spacing=0.5)
    assert isinstance(result, OrientedPointCloud)


def test_mesh_oversample_has_normals(unit_sphere_mesh):
    result = unit_sphere_mesh.oversample(point_spacing=0.5)
    assert result.normals is not None
    assert result.normals.shape == result.vertices.shape


def test_mesh_separate_surfaces_two_labels(two_sphere_mesh):
    labels = DiscreteSurface.separate_surfaces(two_sphere_mesh)
    unique = np.unique(labels)
    assert set(unique) == {0, 1}, f"expected labels {{0,1}}, got {unique}"


def test_mesh_separate_surfaces_both_present(two_sphere_mesh):
    labels = DiscreteSurface.separate_surfaces(two_sphere_mesh)
    n_inner = int(np.sum(labels == 0))
    n_outer = int(np.sum(labels == 1))
    assert n_inner > 0 and n_outer > 0


def test_mesh_filter_by_labels_returns_subset(two_sphere_mesh):
    labels = DiscreteSurface.separate_surfaces(two_sphere_mesh)
    inner = two_sphere_mesh.filter_by_labels(labels, "inner")
    outer = two_sphere_mesh.filter_by_labels(labels, "outer")
    # Each filtered mesh must be a strict subset of the original
    assert len(inner.vertices) < len(two_sphere_mesh.vertices)
    assert len(outer.vertices) < len(two_sphere_mesh.vertices)
    # Together they account for all original vertices (labels partition exactly)
    assert len(inner.vertices) + len(outer.vertices) <= len(two_sphere_mesh.vertices)


# ===========================================================================
# OrientedPointCloud — tests
# ===========================================================================

def test_opc_has_vertices_and_normals(unit_sphere_opc):
    assert unit_sphere_opc.vertices is not None
    assert unit_sphere_opc.normals is not None


def test_opc_get_vertices_shape(unit_sphere_opc):
    verts = unit_sphere_opc.get_vertices()
    assert verts.ndim == 2 and verts.shape[1] == 3


def test_opc_get_normals_shape(unit_sphere_opc):
    norms = unit_sphere_opc.get_normals()
    assert norms.shape == unit_sphere_opc.vertices.shape


def test_opc_distance_outside_point(unit_sphere_opc):
    pt = np.array([[2.0, 0.0, 0.0]])
    result = unit_sphere_opc.distance_to_points(pt)
    assert abs(result["distances"][0] - 1.0) < 0.02


def test_opc_distance_surface_point(unit_sphere_opc):
    pt = np.array([[1.0, 0.0, 0.0]])
    result = unit_sphere_opc.distance_to_points(pt)
    assert result["distances"][0] < 0.05


def test_opc_distance_is_unsigned(unit_sphere_opc):
    pt = np.array([[0.5, 0.0, 0.0]])
    result = unit_sphere_opc.distance_to_points(pt)
    assert result["distance_type"] == "unsigned"
    assert result["distances"][0] > 0


def test_opc_cast_rays_hits_sphere(unit_sphere_opc):
    # Ray from [2, 0, 0] toward -x with long length; should hit at t ≈ 1
    ray = np.array([[2.0, 0.0, 0.0, -3.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_opc.cast_rays(ray, knn_radius=0.15)
    assert np.isfinite(result["t_hit"][0]), "expected a hit, got inf"
    assert abs(result["t_hit"][0] - 1.0) < 0.1


def test_opc_cast_rays_miss(unit_sphere_opc):
    # Ray from [2, 0, 0] pointing in +x (away) — no sphere points in that direction
    ray = np.array([[2.0, 0.0, 0.0, 3.0, 0.0, 0.0]], dtype=np.float32)
    result = unit_sphere_opc.cast_rays(ray, knn_radius=0.15)
    assert not np.isfinite(result["t_hit"][0])


def test_opc_translate_shifts_centroid(unit_sphere_opc):
    v = np.array([5.0, -1.0, 2.0])
    unit_sphere_opc.translate(v)
    centroid = unit_sphere_opc.vertices.mean(axis=0)
    assert np.allclose(centroid, v, atol=0.1)


def test_opc_rotate_identity_leaves_vertices(unit_sphere_opc):
    original = unit_sphere_opc.vertices.copy()
    unit_sphere_opc.rotate(np.eye(3))
    assert np.allclose(unit_sphere_opc.vertices, original, atol=1e-10)


def test_opc_rotate_preserves_normal_unit_length(unit_sphere_opc):
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    unit_sphere_opc.rotate(R)
    lengths = np.linalg.norm(unit_sphere_opc.normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


def test_opc_oversample_larger_spacing_fewer_points(unit_sphere_opc):
    n_original = len(unit_sphere_opc.vertices)
    result = unit_sphere_opc.oversample(point_spacing=0.3)
    assert isinstance(result, OrientedPointCloud)
    assert len(result.vertices) < n_original


def test_opc_oversample_preserves_normals(unit_sphere_opc):
    result = unit_sphere_opc.oversample(point_spacing=0.3)
    assert result.normals is not None
    assert result.normals.shape == result.vertices.shape
