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
    inner_mask, outer_mask = two_sphere_mesh.separate_closed_surface()
    assert inner_mask.any(), "inner surface mask is empty"
    assert outer_mask.any(), "outer surface mask is empty"


def test_mesh_separate_surfaces_both_present(two_sphere_mesh):
    inner_mask, outer_mask = two_sphere_mesh.separate_closed_surface()
    n_inner = int(inner_mask.sum())
    n_outer = int(outer_mask.sum())
    assert n_inner > 0 and n_outer > 0


def test_mesh_filter_by_labels_returns_subset(two_sphere_mesh):
    inner_mask, outer_mask = two_sphere_mesh.separate_closed_surface()
    inner = two_sphere_mesh.apply_vertex_mask(inner_mask)
    outer = two_sphere_mesh.apply_vertex_mask(outer_mask)
    # Each filtered mesh must be a strict subset of the original
    assert len(inner.vertices) < len(two_sphere_mesh.vertices)
    assert len(outer.vertices) < len(two_sphere_mesh.vertices)
    # Together they account for all original vertices (masks partition exactly)
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


# ===========================================================================
# DiscreteSurface — tests for methods not covered above
# ===========================================================================

# ---------------------------------------------------------------------------
# Fixture: two parallel planes (ideal for separate_planar_surface)
# ---------------------------------------------------------------------------

@pytest.fixture
def two_plane_mesh():
    """Upper plane (z=+0.5, normals +z) and lower plane (z=−0.5, normals −z).

    This is the simplest possible bilayer proxy for testing ``separate_planar_surface``:
    a clear PCA axis separates the two groups.
    """
    n = 10
    xs, ys = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
    xs, ys = xs.ravel(), ys.ravel()
    npts = len(xs)

    upper_v = np.column_stack([xs, ys,  np.full(npts, 0.5)])
    lower_v = np.column_stack([xs, ys, -np.full(npts, 0.5)])
    upper_n = np.tile([0.0, 0.0,  1.0], (npts, 1))
    lower_n = np.tile([0.0, 0.0, -1.0], (npts, 1))

    mesh = Mesh()
    mesh.vertices = np.vstack([upper_v, lower_v])
    mesh.normals  = np.vstack([upper_n, lower_n])
    mesh.faces    = np.zeros((0, 3), dtype=int)
    return mesh


# ---------------------------------------------------------------------------
# separate_planar_surface
# ---------------------------------------------------------------------------

def test_separate_planar_surface_returns_two_groups(two_plane_mesh):
    s1, s2 = two_plane_mesh.separate_planar_surface()
    assert s1 is not None and s2 is not None
    assert s1.sum() > 0 and s2.sum() > 0


def test_separate_planar_surface_no_overlap(two_plane_mesh):
    s1, s2 = two_plane_mesh.separate_planar_surface()
    assert not np.any(s1 & s2), "A vertex is assigned to both surfaces"


def test_separate_planar_surface_covers_all_vertices(two_plane_mesh):
    s1, s2 = two_plane_mesh.separate_planar_surface()
    # Together the two masks should partition all vertices (union = all True)
    assert np.all(s1 | s2), "Some vertices are assigned to neither surface"


def test_separate_planar_surface_equal_split(two_plane_mesh):
    """Upper and lower planes have the same number of points → 50/50 split."""
    s1, s2 = two_plane_mesh.separate_planar_surface()
    assert s1.sum() == s2.sum()


# ---------------------------------------------------------------------------
# orient_normals_globally
# ---------------------------------------------------------------------------

def test_orient_normals_globally_consistent(unit_sphere_opc):
    """After orientation, most normals should point radially outward (unit sphere)."""
    opc = unit_sphere_opc
    opc.orient_normals_globally(inplace=True)
    # On a unit sphere, outward normals have positive dot product with position
    dots = (opc.vertices * opc.normals).sum(axis=1)
    fraction_outward = (dots > 0).mean()
    assert fraction_outward > 0.8, (
        f"Only {fraction_outward:.1%} of normals point outward after global orientation"
    )


def test_orient_normals_globally_preserves_shape(unit_sphere_opc):
    original_shape = unit_sphere_opc.normals.shape
    unit_sphere_opc.orient_normals_globally(inplace=True)
    assert unit_sphere_opc.normals.shape == original_shape


# ---------------------------------------------------------------------------
# refine_normals / refine_normals_from_arrays
# ---------------------------------------------------------------------------

def test_refine_normals_does_not_change_shape(unit_sphere_mesh):
    before_shape = unit_sphere_mesh.normals.shape
    unit_sphere_mesh.refine_normals(radius_hit=0.3, n_iter=1, inplace=True)
    assert unit_sphere_mesh.normals.shape == before_shape


def test_refine_normals_keeps_unit_length(unit_sphere_mesh):
    unit_sphere_mesh.refine_normals(radius_hit=0.3, n_iter=1, inplace=True)
    lengths = np.linalg.norm(unit_sphere_mesh.normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-4)


def test_refine_normals_from_arrays_returns_unit_normals(unit_sphere_mesh):
    refined = DiscreteSurface.refine_normals_from_arrays(
        unit_sphere_mesh.vertices,
        unit_sphere_mesh.normals,
        radius_hit=0.3,
        n_iter=1,
    )
    assert refined.shape == unit_sphere_mesh.normals.shape
    lengths = np.linalg.norm(refined, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-4)


def test_refine_normals_from_arrays_smooths_toward_neighbors(unit_sphere_opc):
    """Flipping one normal and then refining should pull it back toward its neighbors."""
    opc = unit_sphere_opc
    flipped_normals = opc.normals.copy()
    flipped_normals[0] = -flipped_normals[0]   # flip one normal

    refined = DiscreteSurface.refine_normals_from_arrays(
        opc.vertices, flipped_normals, radius_hit=0.15, n_iter=3
    )
    # After refinement, the flipped normal should be more aligned with the original
    dot_before = np.dot(opc.normals[0], flipped_normals[0])
    dot_after  = np.dot(opc.normals[0], refined[0])
    assert dot_after > dot_before, "Refinement should pull the flipped normal toward its neighbors"


# ---------------------------------------------------------------------------
# apply_normals_mask
# ---------------------------------------------------------------------------

def test_apply_normals_mask_reduces_vertex_count(unit_sphere_opc):
    """A 45° threshold removes roughly half the vertices of a sphere."""
    original_n = len(unit_sphere_opc.vertices)
    filtered = unit_sphere_opc.apply_normals_mask(angle_threshold=45.0, inplace=False)
    assert filtered is not None
    assert len(filtered.vertices) < original_n
    assert len(filtered.vertices) > 0


def test_apply_normals_mask_tight_threshold_few_survivors(unit_sphere_opc):
    """Very tight threshold (5°) keeps only normals close to the mean."""
    filtered = unit_sphere_opc.apply_normals_mask(angle_threshold=5.0, inplace=False)
    assert len(filtered.vertices) < len(unit_sphere_opc.vertices)


def test_apply_normals_mask_wide_threshold_keeps_all(unit_sphere_opc):
    """180° threshold keeps every vertex."""
    original_n = len(unit_sphere_opc.vertices)
    filtered = unit_sphere_opc.apply_normals_mask(angle_threshold=180.0, inplace=False)
    assert len(filtered.vertices) == original_n


def test_apply_normals_mask_normals_consistent_after_filter(unit_sphere_opc):
    """Filtered normals should remain unit-length."""
    filtered = unit_sphere_opc.apply_normals_mask(angle_threshold=60.0, inplace=False)
    lengths = np.linalg.norm(filtered.normals, axis=1)
    assert np.allclose(lengths, 1.0, atol=1e-5)


# =============================================================================
# Phase-1 signature tightenings (kwargs -> explicit) + to_motl helper
# =============================================================================
import inspect


def test_opc_flip_normals_signature_drops_var_kwargs():
    """flip_normals exposes only inplace; no var-kwargs catch-all."""
    sig = inspect.signature(OrientedPointCloud.flip_normals)
    assert list(sig.parameters) == ["self", "inplace"]
    has_var_kw = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    assert not has_var_kw


def test_opc_flip_normals_rejects_unknown_kwarg(unit_sphere_opc):
    """An unknown kwarg now raises TypeError directly (no custom message needed)."""
    with pytest.raises(TypeError):
        unit_sphere_opc.flip_normals(flip_faces=True)


def test_opc_flip_normals_inplace_still_works(unit_sphere_opc):
    """The behaviour we care about (flip in place) is unchanged."""
    before = unit_sphere_opc.normals.copy()
    unit_sphere_opc.flip_normals(inplace=True)
    assert np.allclose(unit_sphere_opc.normals, -before)


def test_opc_save_signature_explicit_kwargs():
    """save now names every format-specific kwarg: write_ascii / input_dict /
    subtomo_ids / tomo_id (no catch-all)."""
    sig = inspect.signature(OrientedPointCloud.save)
    expected = {"self", "output_path", "format",
                "write_ascii", "input_dict", "subtomo_ids", "tomo_id"}
    assert set(sig.parameters) == expected
    assert not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())


def test_opc_save_rejects_unknown_kwarg(unit_sphere_opc, tmp_path):
    """Unknown kwargs must now fail at the signature, not silently slip into the
    private helper."""
    out = tmp_path / "out.ply"
    with pytest.raises(TypeError):
        unit_sphere_opc.save(out, format="ply", bogus=True)


def test_opc_from_mrc_annotation_is_mapsource():
    """The PathOrStr -> MapSource tightening matches Mesh.from_mrc so an
    in-memory ndarray segmentation is also a valid input.

    surface.py uses ``from __future__ import annotations`` so the annotation
    is a string at runtime; compare textually."""
    sig = inspect.signature(OrientedPointCloud.from_mrc)
    assert sig.parameters["input_path"].annotation == "MapSource"


def test_opc_to_motl_returns_motl_with_one_row_per_vertex(unit_sphere_opc):
    """to_motl is the new pure helper: build a Motl in memory without writing."""
    motl = unit_sphere_opc.to_motl()
    assert motl is not None
    assert len(motl.df) == len(unit_sphere_opc.vertices)
    for col in ("x", "y", "z", "phi", "theta", "psi", "subtomo_id", "class"):
        assert col in motl.df.columns
    # Sequential 1..N by default.
    assert np.allclose(
        motl.df["subtomo_id"].to_numpy(),
        np.arange(1, len(unit_sphere_opc.vertices) + 1, dtype=float),
    )
    assert np.allclose(motl.df["class"].to_numpy(), 1.0)


def test_opc_to_motl_scalar_tomo_id_broadcasts(unit_sphere_opc):
    motl = unit_sphere_opc.to_motl(tomo_id=42)
    assert np.allclose(motl.df["tomo_id"].to_numpy(), 42.0)


def test_opc_to_motl_subtomo_ids_remap_to_sequential(unit_sphere_opc):
    """Object-style subtomo_ids get remapped to sequential per-object IDs."""
    n = len(unit_sphere_opc.vertices)
    ids = np.where(np.arange(n) % 2 == 0, 100, 200)
    motl = unit_sphere_opc.to_motl(subtomo_ids=ids)
    out = motl.df["subtomo_id"].to_numpy()
    assert set(np.unique(out)) == {1.0, 2.0}
    assert len(out) == n


def test_opc_to_motl_input_dict_forbidden_keys(unit_sphere_opc):
    """Coords/angles/subtomo_id/tomo_id can't be smuggled in via input_dict."""
    with pytest.raises(ValueError):
        unit_sphere_opc.to_motl(input_dict={"x": np.zeros(len(unit_sphere_opc.vertices))})


def test_opc_save_em_writes_motl_file(unit_sphere_opc, tmp_path):
    """save(format='em') still works -- public path now routes through to_motl."""
    out = tmp_path / "pc.em"
    unit_sphere_opc.save(out, format="em")
    assert out.exists() and out.stat().st_size > 0


# ── Mesh.from_alpha_shape / suggest_alpha_range / alpha_shape_tetra ──────────


def _sphere_points(n=500, radius=10.0, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * radius


def _two_cluster_points(n=200, sep=50.0):
    rng = np.random.default_rng(1)
    a = rng.standard_normal((n, 3))
    b = rng.standard_normal((n, 3)) + np.array([sep, 0.0, 0.0])
    return np.vstack([a, b])


class TestFromAlphaShape:
    def test_returns_mesh_with_geometry(self):
        from cryocat.core.surface import Mesh
        pts = _sphere_points()
        lo, hi = Mesh.suggest_alpha_range(pts)
        mesh = Mesh.from_alpha_shape(pts, alpha=(lo + hi) / 2)
        assert mesh.vertices is not None
        assert mesh.faces is not None
        assert len(mesh.vertices) > 0
        assert len(mesh.faces) > 0

    def test_precomputed_tetra_gives_same_result(self):
        from cryocat.core.surface import Mesh
        pts = _sphere_points(n=300, seed=42)
        alpha = 3.0
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(pts)
        mesh_with = Mesh.from_alpha_shape(pts, alpha, tetra_mesh, pt_map)
        mesh_without = Mesh.from_alpha_shape(pts, alpha)
        np.testing.assert_array_equal(mesh_with.vertices, mesh_without.vertices)
        np.testing.assert_array_equal(mesh_with.faces, mesh_without.faces)

    def test_large_alpha_matches_convex_hull_vertex_count(self):
        from cryocat.core.surface import Mesh
        import open3d as o3d
        pts = _sphere_points(n=200, seed=7)
        alpha = 1e9
        mesh = Mesh.from_alpha_shape(pts, alpha)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        hull, _ = pcd.compute_convex_hull()
        assert len(mesh.vertices) == len(np.asarray(hull.vertices))

    def test_small_alpha_fragments_two_clusters(self):
        from cryocat.core.surface import Mesh
        pts = _two_cluster_points(n=300, sep=50.0)
        lo, _ = Mesh.suggest_alpha_range(pts)
        mesh = Mesh.from_alpha_shape(pts, alpha=lo * 2)
        n_comp = mesh.get_connected_component_count()
        assert n_comp > 1

    def test_fewer_than_four_points_raises(self):
        from cryocat.core.surface import Mesh
        with pytest.raises(ValueError, match="at least 4"):
            Mesh.from_alpha_shape(np.zeros((3, 3)), alpha=1.0)

    def test_coplanar_points_raise_via_tetra(self):
        from cryocat.core.surface import Mesh
        pts = np.column_stack([
            np.linspace(0, 1, 20), np.linspace(0, 1, 20), np.zeros(20)
        ])
        with pytest.raises(ValueError, match="coplanar"):
            Mesh.alpha_shape_tetra(pts)


class TestSuggestAlphaRange:
    def test_returns_positive_pair_ordered(self):
        from cryocat.core.surface import Mesh
        pts = _sphere_points()
        lo, hi = Mesh.suggest_alpha_range(pts)
        assert lo > 0
        assert hi > lo

    def test_scales_linearly_with_cloud(self):
        from cryocat.core.surface import Mesh
        pts = _sphere_points(n=400, seed=3)
        lo1, hi1 = Mesh.suggest_alpha_range(pts)
        lo2, hi2 = Mesh.suggest_alpha_range(pts * 10.0)
        assert lo2 == pytest.approx(lo1 * 10.0, rel=1e-6)
        assert hi2 == pytest.approx(hi1 * 10.0, rel=1e-6)

    def test_fewer_than_four_points_raises(self):
        from cryocat.core.surface import Mesh
        with pytest.raises(ValueError, match="at least 4"):
            Mesh.suggest_alpha_range(np.zeros((3, 3)))


class TestAlphaShapeTetra:
    def test_fewer_than_four_points_raises(self):
        from cryocat.core.surface import Mesh
        with pytest.raises(ValueError, match="at least 4"):
            Mesh.alpha_shape_tetra(np.zeros((3, 3)))

    def test_coplanar_raises(self):
        from cryocat.core.surface import Mesh
        pts = np.column_stack([
            np.linspace(0, 1, 10), np.linspace(0, 1, 10), np.zeros(10)
        ])
        with pytest.raises(ValueError, match="coplanar"):
            Mesh.alpha_shape_tetra(pts)

    def test_returns_two_objects(self):
        from cryocat.core.surface import Mesh
        pts = _sphere_points(n=100)
        result = Mesh.alpha_shape_tetra(pts)
        assert len(result) == 2


class TestConnectivityMetrics:
    def test_empty_mesh_component_count_zero(self):
        from cryocat.core.surface import Mesh
        m = Mesh()
        assert m.get_connected_component_count() == 0

    def test_empty_mesh_not_watertight(self):
        from cryocat.core.surface import Mesh
        m = Mesh()
        assert m.is_watertight() is False

    def test_sphere_single_component(self):
        from cryocat.core.surface import Mesh
        pts = _sphere_points(n=500)
        lo, hi = Mesh.suggest_alpha_range(pts)
        mesh = Mesh.from_alpha_shape(pts, alpha=(lo + hi) / 2)
        n_comp = mesh.get_connected_component_count()
        assert n_comp == 1

    def test_two_clusters_two_components(self):
        from cryocat.core.surface import Mesh
        pts = _two_cluster_points(n=300, sep=50.0)
        lo, _ = Mesh.suggest_alpha_range(pts)
        mesh = Mesh.from_alpha_shape(pts, alpha=lo * 2)
        assert mesh.get_connected_component_count() > 1


# =============================================================================
# Mesh.from_ordered_path — tube constructor
# =============================================================================

class TestFromOrderedPath:
    _AXIS = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 50.0], [0.0, 0.0, 100.0]])
    _R = 5.0

    def _straight_tube(self, radius=None):
        return Mesh.from_ordered_path(self._AXIS, radius=radius if radius is not None else self._R)

    def test_returns_mesh_with_vertices_and_faces(self):
        m = self._straight_tube()
        assert m.vertices is not None and m.faces is not None
        assert m.vertices.shape[1] == 3
        assert m.faces.shape[1] == 3

    def test_bounding_box_matches_axis_and_radius(self):
        m = self._straight_tube()
        v = m.vertices
        # z must span 0..100
        assert v[:, 2].min() == pytest.approx(0.0, abs=0.1)
        assert v[:, 2].max() == pytest.approx(100.0, abs=0.1)
        # x and y must not exceed the radius
        assert abs(v[:, 0].min()) == pytest.approx(self._R, abs=0.1)
        assert abs(v[:, 0].max()) == pytest.approx(self._R, abs=0.1)

    def test_all_vertices_within_radius_of_axis(self):
        m = self._straight_tube()
        radial = np.sqrt(m.vertices[:, 0] ** 2 + m.vertices[:, 1] ** 2)
        # End caps add vertices on the axis itself (radial ~ 0); lateral band at ~R.
        assert radial.max() == pytest.approx(self._R, abs=0.1)

    def test_per_point_radius_wider_at_midpoint(self):
        radii = np.array([self._R, 15.0, self._R])  # bulge at z=50
        m = Mesh.from_ordered_path(self._AXIS, radius=radii)
        v = m.vertices
        # Near the midpoint (z ≈ 50) the tube should be wider than at the ends.
        near_mid = v[np.abs(v[:, 2] - 50.0) < 5.0]
        near_end = v[v[:, 2] < 5.0]
        radial_mid = np.sqrt(near_mid[:, 0] ** 2 + near_mid[:, 1] ** 2).max()
        radial_end = np.sqrt(near_end[:, 0] ** 2 + near_end[:, 1] ** 2).max()
        assert radial_mid > radial_end + 5.0

    def test_two_points_works(self):
        pts = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 100.0]])
        m = Mesh.from_ordered_path(pts, radius=self._R)
        assert len(m.vertices) > 0

    def test_one_point_raises_with_count(self):
        with pytest.raises(ValueError, match="1"):
            Mesh.from_ordered_path(np.array([[0.0, 0.0, 0.0]]), radius=self._R)


# =============================================================================
# Mesh.from_motl_filaments — tubes from a motl
# =============================================================================

def _make_filament_motl(tmp_path):
    """Two-chain motl (chain 1: 3 pts, chain 2: 2 pts) plus a single-point chain 3."""
    import pandas as pd
    from cryocat.core import cryomotl

    cols = [
        "phi", "theta", "psi", "x", "y", "z",
        "shift_x", "shift_y", "shift_z",
        "tomo_id", "object_id", "subtomo_id", "class", "score",
        "geom1", "geom2", "geom3", "geom4", "geom5", "subtomo_mean",
    ]
    df = pd.DataFrame(0.0, index=range(6), columns=cols)
    df["tomo_id"] = 1
    df["object_id"] = [1, 1, 1, 2, 2, 3]   # chain id
    df["geom2"]    = [1, 2, 3, 1, 2, 1]     # position within chain
    df["x"] = [0, 0, 0, 50, 50, 200]
    df["y"] = [0, 0, 0, 50, 50, 200]
    df["z"] = [0, 10, 20, 0, 10, 0]
    motl = cryomotl.Motl(df)
    path = str(tmp_path / "filaments.em")
    motl.write_out(path)
    return path


class TestFromMotlFilaments:
    def test_two_chains_two_tubes(self, tmp_path):
        path = _make_filament_motl(tmp_path)
        tubes = Mesh.from_motl_filaments(path, radius=5.0)
        assert len(tubes) == 2

    def test_keys_match_chain_ids(self, tmp_path):
        path = _make_filament_motl(tmp_path)
        tubes = Mesh.from_motl_filaments(path, radius=5.0)
        assert set(tubes.keys()) == {"1.0", "2.0"}

    def test_single_point_chain_skipped_with_warning(self, tmp_path):
        import warnings
        path = _make_filament_motl(tmp_path)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            tubes = Mesh.from_motl_filaments(path, radius=5.0)
        assert "3.0" not in tubes
        assert any("skipped" in str(x.message).lower() for x in w)

    def test_each_tube_is_a_mesh(self, tmp_path):
        path = _make_filament_motl(tmp_path)
        tubes = Mesh.from_motl_filaments(path, radius=5.0)
        for t in tubes.values():
            assert isinstance(t, Mesh)
            assert t.vertices is not None and t.faces is not None


# =============================================================================
# Mesh.to_segmentation — binary volume from mesh
# =============================================================================

class TestToSegmentation:
    _R = 8.0
    # Tube centred at (15,15) in x/y, from z=2..18, pixel_size=2 -> 30x30x15 vox.
    _AXIS = np.array([[15.0, 15.0, 2.0], [15.0, 15.0, 18.0]])
    _PS = 2.0
    _VDIMS = (15, 15, 12)  # covers 30x30x24 in world units

    @pytest.fixture
    def tube_seg(self):
        tube = Mesh.from_ordered_path(self._AXIS, radius=self._R, n_spline_points=80)
        return tube, tube.to_segmentation(self._VDIMS, pixel_size=self._PS)

    def test_shape_matches_volume_dims(self, tube_seg):
        _, seg = tube_seg
        assert seg.shape == self._VDIMS

    def test_dtype_is_bool(self, tube_seg):
        _, seg = tube_seg
        assert seg.dtype == bool

    def test_axis_voxel_inside(self, tube_seg):
        _, seg = tube_seg
        # Axis midpoint in world: x=15, y=15, z=10 → voxel (7, 7, 5)
        assert seg[7, 7, 5]

    def test_far_outside_radius_excluded(self, tube_seg):
        _, seg = tube_seg
        # World (0, 0, 10) is 15√2 ≈ 21 units from axis centre — well outside R=8
        assert not seg[0, 0, 5]

    def test_round_trip_stable(self, tmp_path):
        """Tube → segmentation → from_mrc gives surface within one voxel of original."""
        import os
        from cryocat.core import cryomap

        tube = Mesh.from_ordered_path(self._AXIS, radius=self._R, n_spline_points=80)
        seg = tube.to_segmentation(self._VDIMS, pixel_size=self._PS)

        mrc_path = str(tmp_path / "tube_seg.mrc")
        cryomap.write(seg.astype(np.float32), mrc_path, pixel_size=self._PS)
        recovered = Mesh.from_mrc(mrc_path, pixel_size=self._PS)

        # x/y extent of recovered mesh should match the original within one pixel.
        tol = self._PS + 0.5
        orig_xmax = tube.vertices[:, 0].max()
        rec_xmax = recovered.vertices[:, 0].max()
        assert abs(rec_xmax - orig_xmax) < tol, (
            f"x extent mismatch: original {orig_xmax:.1f}, recovered {rec_xmax:.1f}"
        )


# =============================================================================
# HH2c — from_ball_pivoting, triangle_sizes, filter_triangles
# =============================================================================

def _sphere_point_cloud(n_pts=800, radius=5.0):
    """Return (points, normals) on a sphere with outward normals."""
    rng = np.random.default_rng(42)
    pts = rng.standard_normal((n_pts, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= radius
    nrm = pts / radius
    return pts, nrm


def _hemisphere_point_cloud(n_pts=400, radius=5.0):
    """Return (points, normals) on the upper hemisphere (z >= 0)."""
    pts, nrm = _sphere_point_cloud(n_pts * 3, radius)
    mask = pts[:, 2] >= 0
    return pts[mask][:n_pts], nrm[mask][:n_pts]


class TestFromBallPivoting:
    def test_full_sphere_produces_mesh(self):
        pts, nrm = _sphere_point_cloud(n_pts=1000, radius=5.0)
        mesh = Mesh.from_ball_pivoting(pts, normals=nrm)
        assert mesh.vertices is not None and len(mesh.vertices) > 0
        assert mesh.faces is not None and len(mesh.faces) > 0

    def test_hemisphere_not_watertight(self):
        pts, nrm = _hemisphere_point_cloud(n_pts=500, radius=5.0)
        mesh = Mesh.from_ball_pivoting(pts, normals=nrm)
        assert mesh.vertices is not None
        assert len(mesh.faces) > 0
        assert not mesh.is_watertight(), "hemisphere should have boundary edges (not watertight)"

    def test_accepts_oriented_point_cloud(self):
        pts, nrm = _sphere_point_cloud(n_pts=800, radius=5.0)
        opc = OrientedPointCloud()
        opc.vertices = pts
        opc.normals = nrm
        mesh = Mesh.from_ball_pivoting(opc)
        assert len(mesh.faces) > 0

    def test_raises_without_normals(self):
        pts, _ = _sphere_point_cloud(n_pts=100)
        with pytest.raises(ValueError, match="normals"):
            Mesh.from_ball_pivoting(pts)

    def test_custom_radii(self):
        pts, nrm = _sphere_point_cloud(n_pts=800, radius=5.0)
        mesh = Mesh.from_ball_pivoting(pts, normals=nrm, radii=[0.5, 1.0, 2.0])
        assert len(mesh.faces) > 0


class TestTriangleSizes:
    @pytest.fixture
    def simple_mesh(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=float)
        f = np.array([[0, 1, 2], [0, 3, 2]])
        m = Mesh()
        m.vertices = v
        m.faces = f
        return m

    def test_returns_dataframe_with_correct_columns(self, simple_mesh):
        df = simple_mesh.triangle_sizes()
        assert set(df.columns) == {"triangle_index", "longest_edge", "area"}

    def test_row_count_equals_face_count(self, simple_mesh):
        df = simple_mesh.triangle_sizes()
        assert len(df) == len(simple_mesh.faces)

    def test_small_triangle_edge_and_area(self, simple_mesh):
        df = simple_mesh.triangle_sizes()
        row = df[df["triangle_index"] == 0].iloc[0]
        # triangle [0,0,0],[1,0,0],[0,1,0]: hypotenuse = sqrt(2), area = 0.5
        np.testing.assert_allclose(row["longest_edge"], np.sqrt(2), atol=1e-9)
        np.testing.assert_allclose(row["area"], 0.5, atol=1e-9)

    def test_large_triangle_has_bigger_longest_edge(self, simple_mesh):
        df = simple_mesh.triangle_sizes()
        assert df.loc[df["triangle_index"] == 1, "longest_edge"].iloc[0] > \
               df.loc[df["triangle_index"] == 0, "longest_edge"].iloc[0]

    def test_raises_on_empty_mesh(self):
        m = Mesh()
        with pytest.raises(ValueError):
            m.triangle_sizes()


class TestFilterTriangles:
    @pytest.fixture
    def two_triangle_mesh(self):
        v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [10, 0, 0]], dtype=float)
        f = np.array([[0, 1, 2], [0, 3, 2]])
        m = Mesh()
        m.vertices = v
        m.faces = f
        return m

    def test_removes_large_triangle_by_edge(self, two_triangle_mesh):
        result = two_triangle_mesh.filter_triangles(max_edge=2.0)
        assert len(result.faces) == 1

    def test_removes_large_triangle_by_area(self, two_triangle_mesh):
        result = two_triangle_mesh.filter_triangles(max_area=1.0)
        assert len(result.faces) == 1

    def test_threshold_above_all_keeps_all(self, two_triangle_mesh):
        result = two_triangle_mesh.filter_triangles(max_edge=100.0)
        assert len(result.faces) == len(two_triangle_mesh.faces)

    def test_no_criteria_keeps_all(self, two_triangle_mesh):
        result = two_triangle_mesh.filter_triangles()
        assert len(result.faces) == len(two_triangle_mesh.faces)

    def test_isolated_vertices_removed(self, two_triangle_mesh):
        result = two_triangle_mesh.filter_triangles(max_edge=2.0)
        used_vertices = np.unique(result.faces)
        # vertex at index 3 (the [10,0,0] point) must be gone
        assert len(result.vertices) == len(used_vertices)

    def test_raises_on_empty_mesh(self):
        m = Mesh()
        with pytest.raises(ValueError):
            m.filter_triangles(max_edge=1.0)

    def test_returns_new_mesh_not_in_place(self, two_triangle_mesh):
        result = two_triangle_mesh.filter_triangles(max_edge=2.0)
        assert result is not two_triangle_mesh
        assert len(two_triangle_mesh.faces) == 2  # original unchanged


# =============================================================================
# HI3 — get_curvature_table
# =============================================================================

@pytest.fixture(scope="module")
def sphere_mesh_with_curvatures():
    """Unit sphere mesh with curvatures computed."""
    pts = _sphere_points(n=800, radius=5.0, seed=7)
    lo, hi = Mesh.suggest_alpha_range(pts)
    m = Mesh.from_alpha_shape(pts, alpha=(lo + hi) / 2)
    m.compute_curvatures()
    return m


class TestGetCurvatureTable:
    EXPECTED_COLUMNS = {
        "index", "x", "y", "z",
        "mean_curvature", "gaussian_curvature", "k1", "k2",
        "curvature_anisotropy", "shape_index", "curvedness",
        "shape_category", "shape_category_label",
    }

    def test_raises_without_curvatures(self):
        pts = _sphere_points(n=100, radius=3.0, seed=0)
        lo, hi = Mesh.suggest_alpha_range(pts)
        m = Mesh.from_alpha_shape(pts, alpha=(lo + hi) / 2)
        # curvatures NOT computed
        with pytest.raises(ValueError, match="compute_curvatures"):
            m.get_curvature_table()

    def test_invalid_element_raises(self, sphere_mesh_with_curvatures):
        with pytest.raises(ValueError, match="element"):
            sphere_mesh_with_curvatures.get_curvature_table(element="edge")

    def test_vertex_mode_row_count(self, sphere_mesh_with_curvatures):
        m = sphere_mesh_with_curvatures
        df = m.get_curvature_table(element="vertex")
        assert len(df) == len(m.vertices)

    def test_triangle_mode_row_count(self, sphere_mesh_with_curvatures):
        m = sphere_mesh_with_curvatures
        df = m.get_curvature_table(element="triangle")
        assert len(df) == len(m.faces)

    def test_all_columns_present_vertex(self, sphere_mesh_with_curvatures):
        df = sphere_mesh_with_curvatures.get_curvature_table(element="vertex")
        assert self.EXPECTED_COLUMNS <= set(df.columns)

    def test_all_columns_present_triangle(self, sphere_mesh_with_curvatures):
        df = sphere_mesh_with_curvatures.get_curvature_table(element="triangle")
        assert self.EXPECTED_COLUMNS <= set(df.columns)

    def test_shape_category_label_is_string(self, sphere_mesh_with_curvatures):
        df = sphere_mesh_with_curvatures.get_curvature_table(element="vertex")
        # dtype may be 'object' or pandas StringDtype; check element type
        assert isinstance(df["shape_category_label"].iloc[0], str)
        valid_labels = set(Mesh.SURFACE_TYPE_LABELS.values())
        assert set(df["shape_category_label"].unique()).issubset(valid_labels)

    def test_curvature_anisotropy_equals_abs_k1_minus_k2(self, sphere_mesh_with_curvatures):
        df = sphere_mesh_with_curvatures.get_curvature_table(element="vertex")
        expected = np.abs(df["k1"].values - df["k2"].values)
        np.testing.assert_allclose(df["curvature_anisotropy"].values, expected, atol=1e-10)
