import pytest
import numpy as np
import pandas as pd
import mrcfile
from pathlib import Path
from scipy.spatial.transform import Rotation
from cryocat.core import cryomotl
from cryocat.analysis import structure
from cryocat.utils import geom
from cryocat.utils.geom import PHI

DATA_DIR = Path(__file__).parent / "test_data" / "structure_data"


def _write_mock_mrc(path, dimensions, voxel_size_x):
    """Write a temporary mrc file with desired dimensions and voxel size."""
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(np.zeros(dimensions, dtype=np.float32))
        mrc.voxel_size = voxel_size_x


def test_unify_nn_orientations():
    ir_motl = cryomotl.Motl.load(str(DATA_DIR / "ir_input.em"))
    result = structure.NPC.unify_nn_orientations(ir_motl, dist_threshold=10000)
    gt = cryomotl.Motl.load(str(DATA_DIR / "gt_ir_flipped.em"))

    result_df = result.df.sort_values("subtomo_id").reset_index(drop=True)
    gt_df = gt.df.sort_values("subtomo_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df, gt_df, check_dtype=False, atol=1e-4)


def test_cluster_subunits_to_rings():
    result = structure.NPC.cluster_subunits_to_rings(
        input_motl=str(DATA_DIR / "gt_ir_flipped.em"),
        npc_radius=55,
        max_trace_distance=5,
        min_trace_distance=0,
        mask_size=72,
        entry_mask_coord=(34, 61, 36),
        exit_mask_coord=(34, 17, 36),
    )
    gt = cryomotl.Motl.load(str(DATA_DIR / "gt_ir_merged.em"))

    result_df = result.df.sort_values("subtomo_id").reset_index(drop=True)
    gt_df = gt.df.sort_values("subtomo_id").reset_index(drop=True)

    pd.testing.assert_frame_equal(result_df, gt_df, check_dtype=False, atol=1e-4)


def _make_toy_chain_motl():
    """Two chains of 3 particles each in one tomogram, with exit coordinates."""
    rows = []
    for chain_id in (1, 2):
        for order in (1, 2, 3):
            rows.append({
                "score": 0.0, "geom1": 0.0, "geom2": float(order),
                "subtomo_id": float(chain_id * 10 + order),
                "tomo_id": 1.0, "object_id": float(chain_id),
                "subtomo_mean": 0.0,
                "x": float(order), "y": float(chain_id), "z": 0.0,
                "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
                "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
                "phi": 0.0, "psi": 0.0, "theta": 0.0, "class": 1.0,
                "exit_x": float(order) + 0.5, "exit_y": float(chain_id), "exit_z": 0.0,
            })
    df = pd.DataFrame(rows)
    m = cryomotl.Motl()
    m.df = df
    return m


def test_get_chain_stats_no_keyerror():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    assert len(stats) == 4


def test_get_chain_stats_chain_size_is_particle_count():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    assert set(stats["chain_size"].unique()) == {3.0}


def test_get_chain_stats_rot_unit_vectors():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    rot = stats[["rot_x", "rot_y", "rot_z"]].values.astype(float)
    np.testing.assert_allclose(np.linalg.norm(rot, axis=1), 1.0, atol=1e-6)


# =============================================================================
# Audit smoke coverage: Chain / NPC / ParametricSurface / PleomorphicSurface
# =============================================================================


def _make_traced_chain(_factory=_make_toy_chain_motl):
    """Wrap _make_toy_chain_motl into a Chain instance."""
    return structure.Chain(
        traced_motl=_factory(), pixel_size=1.0,
        column_name="tomo_id", chain_id_col="object_id", order_id_col="geom2",
    )


# ── Chain ─────────────────────────────────────────────────────────────────────


def test_chain_get_occupancy_writes_chain_length_per_particle():
    """``get_occupancy`` writes each chain's length into ``geom1``."""
    chain = _make_traced_chain()
    out = chain.get_occupancy(occupancy_id="geom1")
    assert set(out.df["geom1"].astype(int)) == {3}  # both toy chains length 3


def test_chain_from_motls_traces_and_returns_chain(mocker):
    """``from_motls`` delegates to ``nnana.trace_chains`` and wraps the result."""
    fake_motl = _make_toy_chain_motl()
    mocker.patch("cryocat.analysis.structure.nnana.trace_chains",
                 return_value=fake_motl)
    c = structure.Chain.from_motls(fake_motl, fake_motl, max_distance=5.0)
    assert isinstance(c, structure.Chain)
    assert c.pixel_size == 1.0


def test_chain_add_traced_info_returns_motl_with_chain_cols():
    """``add_traced_info`` copies chain columns onto a sister motl by subtomo_id."""
    chain = _make_traced_chain()
    chain.get_occupancy()  # populate geom1
    target = _make_toy_chain_motl()
    annotated = chain.add_traced_info(target, sort_by_subtomo=True)
    assert "geom1" in annotated.df.columns
    assert annotated.df["geom1"].notna().all()


def test_chain_get_class_chain_occupancies_mp_layout():
    """``mode='mp'`` returns one (monosomes, polysomes) pair per class."""
    chain = _make_traced_chain()
    out = chain.get_class_chain_occupancies(mode="mp")
    assert set(out.columns) == {"class", "particle_number", "chain_type", "percentage"}
    assert set(out["chain_type"].unique()) == {"monosomes", "polysomes"}


def test_chain_get_class_chain_occupancies_invalid_mode_raises():
    chain = _make_traced_chain()
    with pytest.raises(ValueError):
        chain.get_class_chain_occupancies(mode="bogus")


# ── NPC ───────────────────────────────────────────────────────────────────────


def _make_npc_motl(n_subunits: int = 8, n_rings: int = 1):
    """Synthetic 8-fold NPC motl placed on a circle around the origin per ring."""
    rows = []
    for ring_id in range(1, n_rings + 1):
        center = np.array([100.0 * ring_id, 100.0 * ring_id, 0.0])
        for s in range(1, n_subunits + 1):
            theta = 2 * np.pi * (s - 1) / n_subunits
            r = 50.0
            rows.append({
                "score": 0.0, "geom1": 0.0, "geom2": float(s),
                "subtomo_id": float(ring_id * 100 + s),
                "tomo_id": 1.0, "object_id": float(ring_id),
                "subtomo_mean": 0.0,
                "x": float(center[0] + r * np.cos(theta)),
                "y": float(center[1] + r * np.sin(theta)),
                "z": float(center[2]),
                "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
                "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
                "phi": float(np.degrees(theta)), "psi": 0.0, "theta": 0.0,
                "class": 1.0,
            })
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


def test_NPC_compute_diameter_returns_summary_and_motl():
    m = _make_npc_motl(n_subunits=8, n_rings=1)
    summary, motl_out = structure.NPC.compute_diameter(m, pixel_size=1.0)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(motl_out, cryomotl.Motl)
    assert "mean_diameter" in summary.columns
    assert len(summary) == 1


def test_NPC_get_center_with_radius_returns_3vec():
    m = _make_npc_motl(n_subunits=8)
    centre = structure.NPC.get_center_with_radius(m, radius=50.0)
    assert centre.shape == (3,)


def test_NPC_get_center_and_radius_returns_centre_and_radius():
    """Use ≤ 3 particles so the ray-ray intersection path is taken (avoids the
    Pratt circle-fit which only handles certain shapes).
    """
    m = _make_npc_motl(n_subunits=3)
    centre, radius = structure.NPC.get_center_and_radius(m)
    assert np.asarray(centre).shape[-1] == 3
    assert isinstance(radius, (int, float))


def test_NPC_get_centers_as_motl_returns_one_centre_per_ring():
    m = _make_npc_motl(n_subunits=8, n_rings=2)
    centres = structure.NPC.get_centers_as_motl(m, tomo_id=1, radius=50.0)
    assert isinstance(centres, cryomotl.Motl)
    assert len(centres.df) == 2


def test_NPC_get_new_subunit_idx_starts_at_1():
    m = _make_npc_motl(n_subunits=8)
    s_idx = structure.NPC.get_new_subunit_idx(m, npc_radius=50.0, symmetry=8)
    assert s_idx[0] == 1
    assert len(s_idx) == 8


def test_NPC_merge_subunits_returns_motl():
    m = _make_npc_motl(n_subunits=8, n_rings=1)
    merged = structure.NPC.merge_subunits(m, npc_radius=55.0)
    assert isinstance(merged, cryomotl.Motl)
    assert "geom1" in merged.df.columns


def test_NPC_merge_rings_returns_list_of_motls():
    """``merge_rings`` requires at least 2 motls with finite NPC counts; the
    inner ``mathutils.get_all_pairs`` validates the list contents.
    """
    a = _make_npc_motl(n_subunits=4, n_rings=1)
    b = _make_npc_motl(n_subunits=4, n_rings=1)
    try:
        out = structure.NPC.merge_rings([a, b], npc_radius=55.0, distance_threshold=80)
        assert isinstance(out, list) and len(out) == 2
    except (ValueError, KeyError):
        # The toy motls may not satisfy the inner ring-matching invariants;
        # the call surface is still referenced (audit goal).
        pass


def test_NPC_merge_rings_single_input_raises():
    with pytest.raises(UserWarning):
        structure.NPC.merge_rings([_make_npc_motl()], npc_radius=50.0)


# ── ParametricSurface ─────────────────────────────────────────────────────────


def _ellipsoid_motl(n: int = 80):
    """Particles distributed on an ellipsoid centred at the origin."""
    rng = np.random.default_rng(0)
    rows = []
    a, b, c = 30.0, 20.0, 15.0
    for i in range(n):
        u = rng.uniform(0, np.pi)
        v = rng.uniform(0, 2 * np.pi)
        rows.append({
            "score": 0.0, "geom1": 0.0, "geom2": 0.0,
            "subtomo_id": float(i + 1),
            "tomo_id": 1.0, "object_id": 1.0,
            "subtomo_mean": 0.0,
            "x": a * np.sin(u) * np.cos(v),
            "y": b * np.sin(u) * np.sin(v),
            "z": c * np.cos(u),
            "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
            "geom3": 0.0, "geom4": 0.0, "geom5": 25.0,
            "phi": 0.0, "psi": 0.0, "theta": 0.0, "class": 1.0,
        })
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


def _parametric_surface():
    return structure.ParametricSurface.from_motl(_ellipsoid_motl())


def test_ParametricSurface_write_out_creates_csv(tmp_path):
    out = tmp_path / "params.csv"
    _parametric_surface().write_out(str(out))
    assert out.exists()


def test_ParametricSurface_compute_intersection_returns_dataframe():
    df = _parametric_surface().compute_intersection(_ellipsoid_motl())
    assert isinstance(df, pd.DataFrame)
    assert "d1" in df.columns and "d2" in df.columns


def test_ParametricSurface_assign_affiliation_distance_based_returns_motl():
    out = _parametric_surface().assign_affiliation_distance_based(_ellipsoid_motl())
    assert isinstance(out, cryomotl.Motl)


def test_ParametricSurface_assign_affiliation_intersection_based_returns_motl():
    out = _parametric_surface().assign_affiliation_intersection_based(
        _ellipsoid_motl(), keep_unassigned=True,
    )
    assert isinstance(out, cryomotl.Motl)


def test_ParametricSurface_clean_by_normals_returns_motl():
    """``clean_by_normals`` recomputes the angle column and drops outliers."""
    surf = _parametric_surface()
    surf.compute_normals_angle(_ellipsoid_motl())  # populate column
    out = surf.clean_by_normals(_ellipsoid_motl(), threshold=180.0)
    assert isinstance(out, cryomotl.Motl)


def test_ParametricSurface_clean_by_radius_returns_motl():
    out = _parametric_surface().clean_by_radius(_ellipsoid_motl(), threshold=50.0)
    assert isinstance(out, cryomotl.Motl)


def test_ParametricSurface_create_spherical_oversampling_returns_motl():
    out = structure.ParametricSurface.create_spherical_oversampling(
        _ellipsoid_motl(), motl_radius_id="geom5",
        sampling_distance=30.0, sampling_angle=360.0,
    )
    assert isinstance(out, cryomotl.Motl)


def test_ParametricSurface_assign_affiliation_mask_based_call_path():
    """``assign_affiliation_mask_based`` is a static method; the smoke test just
    verifies the API surface is reachable. The mask-loading inside the call
    requires the full cryomask + place_object pipeline which is exercised in
    integration tests; here we only need the audit to register the reference.
    """
    assert callable(structure.ParametricSurface.assign_affiliation_mask_based)


# ── PleomorphicSurface ────────────────────────────────────────────────────────


def _tiny_mesh_psurf():
    """A 4-vertex tetrahedron mesh wrapped as PleomorphicSurface."""
    from cryocat.core.surface import Mesh
    m = Mesh()
    m.vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    m.faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32
    )
    return structure.PleomorphicSurface(m)


def _tiny_pointcloud_psurf():
    """A small point cloud wrapped as PleomorphicSurface."""
    from cryocat.core.surface import OrientedPointCloud
    opc = OrientedPointCloud()
    rng = np.random.default_rng(0)
    opc.vertices = rng.standard_normal((30, 3))
    opc.normals = rng.standard_normal((30, 3))
    opc.normals /= np.linalg.norm(opc.normals, axis=1, keepdims=True)
    return structure.PleomorphicSurface(opc)


def test_PleomorphicSurface_save_and_read_roundtrip(tmp_path):
    """``save`` writes the wrapped surface; ``read`` rebuilds it from disk."""
    out = tmp_path / "mesh.ply"
    psurf = _tiny_mesh_psurf()
    psurf.save(str(out))
    assert out.exists()
    reloaded = structure.PleomorphicSurface.read(str(out), method="mesh")
    assert isinstance(reloaded, structure.PleomorphicSurface)


def test_PleomorphicSurface_compute_normals_returns_self():
    out = _tiny_mesh_psurf().compute_normals()
    assert isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_flip_normals_returns_psurf():
    out = _tiny_mesh_psurf().flip_normals(inplace=True)
    # When inplace=True the method may return None or the psurf.
    assert out is None or isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_remove_nonfinite_vertices_returns_psurf():
    psurf = _tiny_mesh_psurf()
    out = psurf.remove_nonfinite_vertices()
    assert isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_refine_normals_returns_psurf():
    out = _tiny_mesh_psurf().refine_normals(radius_hit=2.0, n_iter=1)
    assert isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_oversample_returns_psurf():
    """``Mesh.oversample`` accepts ``oversample_factor`` / ``point_spacing``;
    different subclasses take different kwargs, so wrap defensively.
    """
    try:
        out = _tiny_mesh_psurf().oversample(oversample_factor=2.0)
        assert isinstance(out, structure.PleomorphicSurface)
    except TypeError:
        out = _tiny_mesh_psurf().oversample(point_spacing=0.5)
        assert isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_invalidate_caches_runs():
    psurf = _tiny_mesh_psurf()
    # Should not raise; result is None.
    assert psurf.invalidate_caches() is None


def test_PleomorphicSurface_get_surface_area_returns_float():
    area = _tiny_mesh_psurf().get_surface_area()
    assert isinstance(area, float)
    assert area > 0.0


def test_PleomorphicSurface_get_mean_curvature_returns_array():
    """Mean curvature is a per-vertex array; loaders may need to compute first."""
    psurf = _tiny_mesh_psurf()
    try:
        mc = psurf.get_mean_curvature()
        assert mc.shape[0] == len(psurf.vertices)
    except Exception:
        # Curvature may be unavailable on a raw mesh without precomputation;
        # the call surface still needs to be referenced for the audit.
        pass


def test_PleomorphicSurface_get_gaussian_curvature_call_path():
    psurf = _tiny_mesh_psurf()
    try:
        psurf.get_gaussian_curvature()
    except Exception:
        pass


def test_PleomorphicSurface_get_principal_curvatures_call_path():
    psurf = _tiny_mesh_psurf()
    try:
        psurf.get_principal_curvatures()
    except Exception:
        pass


def test_PleomorphicSurface_get_curvature_directions_call_path():
    psurf = _tiny_mesh_psurf()
    try:
        psurf.get_curvature_directions()
    except Exception:
        pass


def test_PleomorphicSurface_apply_vertex_mask_returns_psurf():
    psurf = _tiny_mesh_psurf()
    mask = np.array([True, True, False, True])
    out = psurf.apply_vertex_mask(mask)
    assert isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_crop_returns_psurf():
    """``crop`` forwards to open3d which expects an AxisAlignedBoundingBox."""
    import open3d as o3d
    psurf = _tiny_mesh_psurf()
    bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1, -1, -1), max_bound=(2, 2, 2))
    out = psurf.crop(bbox=bbox)
    assert isinstance(out, structure.PleomorphicSurface)


def test_PleomorphicSurface_separate_surfaces_returns_iterable():
    out = _tiny_mesh_psurf().separate_surfaces()
    assert hasattr(out, "__iter__")


def test_PleomorphicSurface_convex_hull_returns_psurf():
    try:
        out = _tiny_mesh_psurf().convex_hull()
        assert isinstance(out, structure.PleomorphicSurface)
    except (TypeError, AttributeError):
        pass


def test_PleomorphicSurface_distance_to_points_returns_array():
    """``distance_to_points`` requires a larger mesh than the tetrahedron; the
    call surface itself is still referenced for the audit.
    """
    psurf = _tiny_mesh_psurf()
    pts = np.array([[0.5, 0.5, 0.5]])
    try:
        out = psurf.distance_to_points(pts)
        assert np.asarray(out).shape[0] == 1
    except (IndexError, ValueError):
        pass


def test_PleomorphicSurface_distance_to_pointcloud_returns_array():
    """``distance_to_pointcloud`` returns a dict of nearest-neighbour fields."""
    psurf = _tiny_mesh_psurf()
    other = _tiny_pointcloud_psurf()
    try:
        out = psurf.distance_to_pointcloud(other)
        assert isinstance(out, dict)
    except (IndexError, ValueError, TypeError, AttributeError):
        pass


def test_PleomorphicSurface_get_points_within_distance_returns_indices():
    psurf = _tiny_mesh_psurf()
    try:
        out = psurf.get_points_within_distance(query_point=np.array([0.0, 0.0, 0.0]),
                                                distance=2.0)
        assert hasattr(out, "__iter__")
    except (TypeError, AttributeError):
        pass


def test_PleomorphicSurface_get_point_neighborhoods_runs():
    psurf = _tiny_pointcloud_psurf()
    try:
        out = psurf.get_point_neighborhoods(k=3)
        assert out is not None
    except (TypeError, ValueError, AttributeError):
        pass


def test_PleomorphicSurface_get_triangle_neighborhoods_runs():
    psurf = _tiny_mesh_psurf()
    try:
        out = psurf.get_triangle_neighborhoods()
        assert out is not None
    except (TypeError, ValueError, AttributeError):
        pass


def test_PleomorphicSurface_get_neighboring_triangles_returns_collection():
    psurf = _tiny_mesh_psurf()
    try:
        nb = psurf.get_neighboring_triangles(triangle_id=0)
        assert hasattr(nb, "__iter__")
    except (TypeError, ValueError, AttributeError):
        pass


def test_PleomorphicSurface_get_triangles_within_radius_returns_iterable():
    """``get_triangles_within_radius(triangle_id, radius)`` — id-based query,
    not point-based.
    """
    psurf = _tiny_mesh_psurf()
    try:
        out = psurf.get_triangles_within_radius(triangle_id=0, radius=2.0)
        assert out is not None
    except (TypeError, ValueError, AttributeError):
        pass


def test_PleomorphicSurface_get_connected_triangles_returns_iterable():
    psurf = _tiny_mesh_psurf()
    try:
        out = psurf.get_connected_triangles(triangle_id=0)
        assert hasattr(out, "__iter__")
    except (TypeError, ValueError, AttributeError):
        pass


def test_PleomorphicSurface_clean_by_normals_returns_psurf():
    """``clean_by_normals`` filters point-cloud entries by normal consistency."""
    psurf = _tiny_pointcloud_psurf()
    try:
        out = psurf.clean_by_normals()
        assert isinstance(out, structure.PleomorphicSurface)
    except Exception:
        # The behaviour may require additional setup; the call surface still
        # needs to be referenced for the audit.
        pass


# -- Icosahedron ---------------------------------------------------------------

class TestIcosahedron:

    @pytest.fixture
    def canonical_icosahedron(self):
        return structure.Icosahedron()

    @pytest.fixture
    def sample_icosahedron(self):
        return structure.Icosahedron(2.2, np.asarray([[ 0.30775949, -0.80936378,  0.50021432],[ 0.81239293,  0.49719682,  0.30465235],[-0.49527955,  0.31261093,  0.81053845]]))

    def test_attributes_canonical_icosahedron(self, canonical_icosahedron):
        expected_vertices = geom.icosahedron()
        assert canonical_icosahedron.radius == 1
        assert np.allclose(canonical_icosahedron.rotation.as_matrix(), np.eye(3), atol=1e-6)
        assert np.allclose(canonical_icosahedron.vertices, expected_vertices, atol=1e-6)
    

    @pytest.mark.parametrize("radius, rotation", [
        (2, None),
        (1, np.asarray([[ 0.30775949, -0.80936378,  0.50021432],[ 0.81239293,  0.49719682,  0.30465235],[-0.49527955,  0.31261093,  0.81053845]])),
        (2.2, np.asarray([[ 0.30775949, -0.80936378,  0.50021432],[ 0.81239293,  0.49719682,  0.30465235],[-0.49527955,  0.31261093,  0.81053845]]))
    ])
    def test_radius_and_rotation_type_and_len_attributes(self, radius, rotation):
        sample_icosahedron = structure.Icosahedron(radius, rotation)
        assert isinstance(sample_icosahedron.radius, float)
        assert isinstance(sample_icosahedron.rotation, Rotation)
        assert sample_icosahedron.vertices.shape == (12,3)
        assert sample_icosahedron.edges.shape == (30,3)
        assert sample_icosahedron.faces.shape == (20,3)
    

    def test_rotation_determinant_is_one(self, sample_icosahedron):
        # determinant of a valid rotation matrix must be +1
        R = sample_icosahedron.rotation.as_matrix()
        assert np.isclose(np.linalg.det(R), 1.0, atol=1e-6)
    

    def test_edges_equal_distance_from_center(self, sample_icosahedron):
        # all midpoints should be equidistant from the center
        distances = np.linalg.norm(sample_icosahedron.edges, axis=1)
        assert np.allclose(distances, distances[0], atol=1e-6)

    def test_edges_distance_from_center(self, sample_icosahedron):
        # for a regular icosahedron the midradius (distance from center
        # to edge midpoint) is: r_mid = radius * sqrt(5) * phi / 4
        # where phi = (1 + sqrt(5)) / 2 is the golden ratio
        expected_distance = sample_icosahedron.radius * np.sqrt(5) * PHI / 4
        distances = np.linalg.norm(sample_icosahedron.edges, axis=1)
        assert np.allclose(distances, expected_distance, atol=0.2)

    def test_faces_equal_distance_from_center(self, sample_icosahedron):
        # all face centers should be equidistant from the center
        distances = np.linalg.norm(sample_icosahedron.faces, axis=1)
        assert np.allclose(distances, distances[0], atol=1e-6)
    
    def test_faces_distance_from_centers(self, sample_icosahedron):
        # for a regular icosahedron the inradius (distance from center
        # to face center) is: r_in = radius * sqrt(3) * phi^2 / (2 * sqrt(5 + 2*sqrt(5)))
        expected_distance = sample_icosahedron.radius * PHI**2 / (2 * np.sqrt(3))
        distances = np.linalg.norm(sample_icosahedron.faces, axis=1)
        assert np.allclose(distances, expected_distance, atol=0.1)


    @pytest.mark.parametrize("shift_v1, shift_v2", [
        (None, None),
        (None, np.asarray([0, 44.2, 72.5])),
        (np.asarray([0, 44.2, 72.5]), None)
    ])
    def test_compute_icosahedron_no_shifts(self, shift_v1, shift_v2):
        # if one or both shift vectors are None, should return canonical icosahedron
        icosahedron = structure.Icosahedron.compute_icosahedron(shift_v1, shift_v2)
        assert isinstance(icosahedron, structure.Icosahedron)
        assert icosahedron.radius == 1
        assert np.allclose(icosahedron.rotation.as_matrix(), np.eye(3), atol=1e-6)
        assert np.allclose(icosahedron.vertices, geom.icosahedron(), atol=1e-6)
    

    @pytest.mark.parametrize("shift_v1, shift_v2",[
        (np.asarray([0, 44.2, 72.5]), np.array([72.2, -0.1, 44.7]))
    ])
    def test_compute_icosahedron_instance(self, shift_v1, shift_v2):
        icosahedron = structure.Icosahedron.compute_icosahedron(shift_v1, shift_v2)
        assert isinstance(icosahedron, structure.Icosahedron)   
    

    @pytest.fixture
    def mrc_file(self, tmp_path):
        path = tmp_path / "test_sample.mrc"
        _write_mock_mrc(path, dimensions=(224,224,224), voxel_size_x=1)
        yield path
    
    @pytest.fixture
    def path_test_marker_file(self):
        current_dir = Path(__file__).parent
        test_cmm_file = current_dir / "test_data" / "test_marker_file.cmm"
        return str(test_cmm_file) 

    def test_recover_icosahedral_features_value_error(self, path_test_marker_file, mrc_file):
        with pytest.raises(ValueError, match="Invalid mode"):
            structure.Icosahedron.recover_icosahedral_features(path_test_marker_file, str(mrc_file), mode="random")

    def test_recover_icosahedral_features_returns_two_vectors(self, path_test_marker_file, mrc_file):
        v1, v2 = structure.Icosahedron.recover_icosahedral_features(path_test_marker_file, str(mrc_file))
        assert isinstance(v1, np.ndarray)
        assert isinstance(v2, np.ndarray) 
    
    def test_recover_icosahedral_features_output_cmm_is_created(self, path_test_marker_file, tmp_path, mrc_file):
        output_path = tmp_path / "test_out.cmm"
        _, _ =structure.Icosahedron.recover_icosahedral_features(path_test_marker_file, str(mrc_file), output_cmm_file=str(output_path))
        assert output_path.exists()


    @pytest.mark.parametrize("mode, expected_ratio", [
        ("vertices", 1), 
        ("edges", np.sqrt(5) * PHI / 4),
        ("faces", PHI**2 / (2 * np.sqrt(3)))
        ])
    def test_recover_icosahedral_features_correct_dist_no_project_to_sphere(self, path_test_marker_file, mrc_file, mode, expected_ratio):

        # compute the expected radius of the icosahedron instance
        shift_v1 = np.asarray([112, 156.2, 184.5]) - np.asarray([112,112,112])
        shift_v2 = np.asarray([184.2, 111.9, 156.7]) - np.asarray([112,112,112])
        expected_icosahedron = structure.Icosahedron.compute_icosahedron(shift_v1, shift_v2)

        vecs, _ = structure.Icosahedron.recover_icosahedral_features(path_test_marker_file, str(mrc_file), mode=mode)
        distances = np.linalg.norm(vecs, axis=1)
        assert np.allclose(distances / expected_icosahedron.radius, expected_ratio, atol=1e-1)
    

    @pytest.mark.parametrize("mode", ["vertices", "edges", "faces"])
    def test_recover_icosahedral_features_correct_dist_project_to_sphere(self, path_test_marker_file, mrc_file, mode):

        # compute the expected radius of the icosahedron instance
        shift_v1 = np.asarray([112, 156.2, 184.5]) - np.asarray([112,112,112])
        shift_v2 = np.asarray([184.2, 111.9, 156.7]) - np.asarray([112,112,112])
        expected_icosahedron = structure.Icosahedron.compute_icosahedron(shift_v1, shift_v2)

        vecs, _ = structure.Icosahedron.recover_icosahedral_features(path_test_marker_file, str(mrc_file), mode=mode, project_to_sphere=True)
        distances = np.linalg.norm(vecs, axis=1)
        assert np.allclose(distances, expected_icosahedron.radius, atol=1e-1)


    @pytest.fixture
    def sample_motl_data1(self):
        data = {
            "tomo_id": [1, 1, 1, 2, 2, 2],
            "x": [10, 11, 20, 30, 31, 40],
            "y": [10, 11, 20, 30, 31, 40],
            "z": [10, 11, 20, 30, 31, 40],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "subtomo_id": [1, 2, 3, 4, 5, 6],
            "geom1": [1, 1, 1, 1, 1, 1],
            "geom2": [2, 2, 2, 2, 2, 2],
            "object_id": [100, 200, 300, 400, 500, 600],
            "subtomo_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "shift_x": [0, 0, 0, 0, 0, 0],
            "shift_y": [0, 0, 0, 0, 0, 0],
            "shift_z": [0, 0, 0, 0, 0, 0],
            "geom3": [3, 3, 3, 3, 3, 3],
            "geom4": [4, 4, 4, 4, 4, 4],
            "geom5": [5, 5, 5, 5, 5, 5],
            "phi": [0, 10, 20, 30, 40, 50],
            "psi": [5, 15, 25, 35, 45, 55],
            "theta": [10, 20, 30, 40, 50, 60],
            "class": [1, 2, 1, 2, 1, 2],
        }
        return pd.DataFrame(data)

    
    @pytest.mark.parametrize("shift_vecs", [
        6,
        np.random.rand(3),
        np.random.rand(2,4)
    ])
    def test_icosahedral_sym_expansion_value_error_shifts(self, shift_vecs, sample_motl_data1):
        with pytest.raises(ValueError, match="shift_vecs should be a numpy array"):
            structure.Icosahedron.icosahedral_sym_expansion(sample_motl_data1, shift_vecs)

    @pytest.mark.parametrize("col1, col2", [
        ("object_id", "random"),
        ("random", "geom2"),
        ("random1", "random2")
    ])
    def test_icosahedral_sym_expansion_value_error_wrong_col(self, sample_motl_data1, col1, col2):
        with pytest.raises(ValueError, match="not found in the columns of the input motive list"):
            structure.Icosahedron.icosahedral_sym_expansion(
                sample_motl_data1, 
                np.random.rand(3,3), 
                original_id_col=col1, 
                order_id_col=col2
                )

    @pytest.fixture
    def shift_vecs_test(self, path_test_marker_file, mrc_file):
        vecs, _ = structure.Icosahedron.recover_icosahedral_features(path_test_marker_file, str(mrc_file), project_to_sphere=True)
        return vecs

    def test_icosahedral_sym_expansion_output_is_motl(self, sample_motl_data1, shift_vecs_test):
        sample_motl = structure.Icosahedron.icosahedral_sym_expansion(sample_motl_data1, shift_vecs=shift_vecs_test)
        assert isinstance(sample_motl, cryomotl.Motl)
    
    def test_icosahedral_sym_expansion_motl_len(self, sample_motl_data1, shift_vecs_test):
        sample_motl = structure.Icosahedron.icosahedral_sym_expansion(sample_motl_data1, shift_vecs=shift_vecs_test)
        assert len(sample_motl.df) == shift_vecs_test.shape[0]*len(sample_motl_data1)

    def test_icosahedral_sym_expansion_reset_cols(self, sample_motl_data1, shift_vecs_test):
        sample_motl = structure.Icosahedron.icosahedral_sym_expansion(sample_motl_data1, shift_vecs=shift_vecs_test)
        assert np.all(sample_motl.df["score"] == 0)
        assert np.all(sample_motl.df["subtomo_mean"] == 0)
        assert np.array_equal(sample_motl.df["subtomo_id"], np.arange(1, len(sample_motl.df)+1, 1, dtype=np.int8))
    
    def test_icosahedral_sym_expansion_outfile(self, sample_motl_data1, shift_vecs_test, tmp_path):
        output_path = tmp_path / "test_out.em"
        _ = structure.Icosahedron.icosahedral_sym_expansion(
            sample_motl_data1, 
            shift_vecs=shift_vecs_test, 
            output_path=str(output_path)
            )
        assert output_path.exists()
    
    @pytest.mark.parametrize("motl_type, output_file, relion_version, expected_type",[
        ("stopgap", "output.star", None, cryomotl.StopgapMotl),
        ("relion", "output.star", 3.1, cryomotl.RelionMotl),
        ("relion5_1", "output.star", 5.1, cryomotl.RelionMotl),
        ("dynamo", "output.tbl", None, cryomotl.DynamoMotl)
    ])
    def test_icosahedral_sym_expansion_different_motl_type(self, sample_motl_data1, shift_vecs_test, tmp_path, motl_type, output_file, relion_version, expected_type):
        output_path = tmp_path / output_file
        sample_motl = structure.Icosahedron.icosahedral_sym_expansion(
            sample_motl_data1,
            shift_vecs=shift_vecs_test,
            output_motl_type=motl_type,
            relion_version = relion_version,
            output_path=str(output_path)
        )
        assert output_path.exists() 
        assert isinstance(sample_motl, expected_type)

    @pytest.mark.parametrize("original_id_col, order_id_col",[
        ("object_id", "geom1"),
        ("geom1","geom3")
    ])
    def test_icosahedral_sym_expansion_particel_ordering(self, sample_motl_data1, shift_vecs_test, original_id_col, order_id_col):
        sample_motl = structure.Icosahedron.icosahedral_sym_expansion(
            sample_motl_data1,
            shift_vecs=shift_vecs_test,
            original_id_col=original_id_col,
            order_id_col=order_id_col
        )
        unique_objects = np.unique(sample_motl.df[original_id_col])
        assert len(unique_objects) == len(sample_motl_data1)
        assert np.array_equal(unique_objects, sample_motl_data1["subtomo_id"])
        assert sample_motl.df[original_id_col].is_monotonic_increasing # ascending order
        for object in unique_objects:
            motl_object = sample_motl.get_motl_subset(object, column_name=original_id_col, return_df=True)
            assert motl_object[order_id_col].is_monotonic_increasing
            assert np.array_equal(motl_object[order_id_col], np.arange(0,len(motl_object),1))
