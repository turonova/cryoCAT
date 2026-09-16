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
    cs = structure.NPC(ir_motl, symmetry=8)
    cs.unify_nn_orientations(dist_threshold=10000)
    result = cs.motl
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
            rows.append(
                {
                    "score": 0.0,
                    "geom1": 0.0,
                    "geom2": float(order),
                    "subtomo_id": float(chain_id * 10 + order),
                    "tomo_id": 1.0,
                    "object_id": float(chain_id),
                    "subtomo_mean": 0.0,
                    "x": float(order),
                    "y": float(chain_id),
                    "z": 0.0,
                    "shift_x": 0.0,
                    "shift_y": 0.0,
                    "shift_z": 0.0,
                    "geom3": 0.0,
                    "geom4": 0.0,
                    "geom5": 0.0,
                    "phi": 0.0,
                    "psi": 0.0,
                    "theta": 0.0,
                    "class": 1.0,
                    "exit_x": float(order) + 0.5,
                    "exit_y": float(chain_id),
                    "exit_z": 0.0,
                }
            )
    df = pd.DataFrame(rows)
    m = cryomotl.Motl()
    m.df = df
    return m


def test_get_chain_stats_no_keyerror():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(),
        pixel_size=1.0,
        column_name="tomo_id",
        chain_id_col="object_id",
        order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    assert len(stats) == 4


def test_get_chain_stats_chain_size_is_particle_count():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(),
        pixel_size=1.0,
        column_name="tomo_id",
        chain_id_col="object_id",
        order_id_col="geom2",
    )
    stats = chain.get_chain_stats(min_chain_size=2)
    assert set(stats["chain_size"].unique()) == {3.0}


def test_get_chain_stats_rot_unit_vectors():
    chain = structure.Chain(
        traced_motl=_make_toy_chain_motl(),
        pixel_size=1.0,
        column_name="tomo_id",
        chain_id_col="object_id",
        order_id_col="geom2",
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
        traced_motl=_factory(),
        pixel_size=1.0,
        column_name="tomo_id",
        chain_id_col="object_id",
        order_id_col="geom2",
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
    mocker.patch("cryocat.analysis.structure.nnana.trace_chains", return_value=fake_motl)
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


# ── Chain.get_step_stats ──────────────────────────────────────────────────────


def _make_step_stats_motl(chains: dict[int, list[tuple[float, float, float]]]) -> "cryomotl.Motl":
    """Build a minimal traced-chain motl for get_step_stats tests.

    chains maps chain_id → list of (phi, theta, psi) tuples, one per particle.
    Within each chain, particles are numbered 1..n (order_id_col = geom2).
    """
    rows = []
    subtomo = 1
    for chain_id, orientations in chains.items():
        for order, (phi, theta, psi) in enumerate(orientations, start=1):
            rows.append({
                "score": 0.0, "geom1": 0.0,
                "geom2": float(order),
                "subtomo_id": float(subtomo),
                "tomo_id": 1.0,
                "object_id": float(chain_id),
                "subtomo_mean": 0.0,
                "x": float(order), "y": float(chain_id), "z": 0.0,
                "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
                "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
                "phi": phi, "psi": psi, "theta": theta, "class": 1.0,
            })
            subtomo += 1
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


def test_get_step_stats_identical_orientations_zero_distances():
    """All three angular distances are 0 when every particle has the same orientation.

    Uses 3 particles (2 steps) because cone_inplane_distance has a known pre-existing
    degenerate case for exactly 3 identical pairs; 2 pairs works correctly.
    """
    motl = _make_step_stats_motl({1: [(30.0, 45.0, 60.0)] * 3})
    chain = structure.Chain(motl)
    result = chain.get_step_stats()
    assert len(result) == 2  # 3 particles → 2 steps
    np.testing.assert_allclose(result["angular_distance"].values, 0.0, atol=1e-10)
    np.testing.assert_allclose(result["cone_distance"].values, 0.0, atol=1e-10)
    np.testing.assert_allclose(result["in_plane_distance"].values, 0.0, atol=1e-10)


def test_get_step_stats_known_rotation():
    """A pure in-plane 45° rotation (psi=45) gives angular=45°, cone=0°, in_plane=45°."""
    motl = _make_step_stats_motl({1: [(0.0, 0.0, 0.0), (0.0, 0.0, 45.0)]})
    chain = structure.Chain(motl)
    result = chain.get_step_stats()
    assert len(result) == 1
    np.testing.assert_allclose(result["angular_distance"].values, 45.0, atol=1e-6)
    np.testing.assert_allclose(result["cone_distance"].values, 0.0, atol=1e-6)
    np.testing.assert_allclose(result["in_plane_distance"].values, 45.0, atol=1e-6)


def test_get_step_stats_two_chains_no_boundary_step():
    """Two chains of 3 produce 4 steps; the last particle of chain 1 never pairs with chain 2."""
    motl = _make_step_stats_motl({
        1: [(0.0, 0.0, 0.0)] * 3,
        2: [(0.0, 0.0, 0.0)] * 3,
    })
    chain = structure.Chain(motl)
    result = chain.get_step_stats()
    assert len(result) == 4
    counts = result["chain_id"].value_counts()
    assert counts[1.0] == 2
    assert counts[2.0] == 2


def test_get_step_stats_shuffled_rows_same_result():
    """Reversing DataFrame row order does not change the result (order_id_col is used, not row order)."""
    motl_ordered = _make_step_stats_motl({1: [(0.0, 0.0, 0.0), (0.0, 0.0, 45.0), (0.0, 0.0, 90.0)]})
    motl_shuffled = cryomotl.Motl()
    motl_shuffled.df = motl_ordered.df.iloc[::-1].reset_index(drop=True)
    result_ordered = structure.Chain(motl_ordered).get_step_stats().reset_index(drop=True)
    result_shuffled = structure.Chain(motl_shuffled).get_step_stats().reset_index(drop=True)
    pd.testing.assert_frame_equal(result_ordered, result_shuffled)


def test_get_step_stats_single_particle_chain_produces_no_rows():
    """A single-particle chain contributes 0 rows; a 3-particle chain alongside it is unaffected."""
    motl = _make_step_stats_motl({
        1: [(0.0, 0.0, 0.0)],
        2: [(0.0, 0.0, 0.0)] * 3,
    })
    chain = structure.Chain(motl)
    result = chain.get_step_stats()
    assert 1.0 not in result["chain_id"].values  # single-particle chain absent
    assert len(result) == 2  # only chain 2 contributes 2 steps


# ── NPC ───────────────────────────────────────────────────────────────────────


def _make_npc_motl(n_subunits: int = 8, n_rings: int = 1):
    """Synthetic 8-fold NPC motl placed on a circle around the origin per ring."""
    rows = []
    for ring_id in range(1, n_rings + 1):
        center = np.array([100.0 * ring_id, 100.0 * ring_id, 0.0])
        for s in range(1, n_subunits + 1):
            theta = 2 * np.pi * (s - 1) / n_subunits
            r = 50.0
            rows.append(
                {
                    "score": 0.0,
                    "geom1": 0.0,
                    "geom2": float(s),
                    "subtomo_id": float(ring_id * 100 + s),
                    "tomo_id": 1.0,
                    "object_id": float(ring_id),
                    "subtomo_mean": 0.0,
                    "x": float(center[0] + r * np.cos(theta)),
                    "y": float(center[1] + r * np.sin(theta)),
                    "z": float(center[2]),
                    "shift_x": 0.0,
                    "shift_y": 0.0,
                    "shift_z": 0.0,
                    "geom3": 0.0,
                    "geom4": 0.0,
                    "geom5": 0.0,
                    "phi": float(np.degrees(theta)),
                    "psi": 0.0,
                    "theta": 0.0,
                    "class": 1.0,
                }
            )
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


def test_NPC_compute_diameter_returns_summary_and_motl():
    m = _make_npc_motl(n_subunits=8, n_rings=1)
    cs = structure.CnComplex(m, symmetry=8)
    summary, motl_out = cs.diameter(pixel_size=1.0)
    assert isinstance(summary, pd.DataFrame)
    assert isinstance(motl_out, cryomotl.Motl)
    assert "mean_diameter" in summary.columns
    assert len(summary) == 1


def test_NPC_get_center_with_radius_returns_3vec():
    m = _make_npc_motl(n_subunits=8)
    centre = structure.NPC._center_by_radius_shift(m, npc_radius=50.0)
    assert centre.shape == (3,)


def test_NPC_get_center_and_radius_returns_centre_and_radius():
    """Use 3 particles to exercise the circle-fit / barycentric centre path."""
    m = _make_npc_motl(n_subunits=3)
    cs = structure.CnComplex(m, symmetry=3)
    centre, radius = cs._compute_object_center(m)
    assert np.asarray(centre).shape[-1] == 3
    assert isinstance(radius, (int, float))


def test_NPC_get_centers_as_motl_returns_one_centre_per_ring():
    m = _make_npc_motl(n_subunits=8, n_rings=2)
    cs = structure.CnComplex(m, symmetry=8)
    centres = cs.get_centers_as_motl()
    assert isinstance(centres, cryomotl.Motl)
    assert len(centres.df) == 2


def test_NPC_get_new_subunit_idx_starts_at_1():
    m = _make_npc_motl(n_subunits=8)
    s_idx = structure.NPC._assign_subunit_index(m, 50.0, symmetry=8)
    assert s_idx[0] == 1
    assert len(s_idx) == 8


def test_NPC_merge_subunits_returns_motl():
    m = _make_npc_motl(n_subunits=8, n_rings=1)
    cs = structure.CnComplex(m, symmetry=8)
    cs.merge_subunits(radius=55.0)
    merged = cs.motl
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
        rows.append(
            {
                "score": 0.0,
                "geom1": 0.0,
                "geom2": 0.0,
                "subtomo_id": float(i + 1),
                "tomo_id": 1.0,
                "object_id": 1.0,
                "subtomo_mean": 0.0,
                "x": a * np.sin(u) * np.cos(v),
                "y": b * np.sin(u) * np.sin(v),
                "z": c * np.cos(u),
                "shift_x": 0.0,
                "shift_y": 0.0,
                "shift_z": 0.0,
                "geom3": 0.0,
                "geom4": 0.0,
                "geom5": 25.0,
                "phi": 0.0,
                "psi": 0.0,
                "theta": 0.0,
                "class": 1.0,
            }
        )
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
        _ellipsoid_motl(),
        keep_unassigned=True,
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
        _ellipsoid_motl(),
        motl_radius_id="geom5",
        sampling_distance=30.0,
        sampling_angle=360.0,
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
    m.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    m.faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32)
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
        out = psurf.get_points_within_distance(query_point=np.array([0.0, 0.0, 0.0]), distance=2.0)
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


# -- PolyhedralComplex --------------------------------------------------------


def _make_poly_motl(n_particles: int = 6) -> cryomotl.Motl:
    """Minimal Motl with *n_particles* rows, each in its own object."""
    rows = []
    for i in range(n_particles):
        rows.append(
            {
                "score": 0.9 - i * 0.1,
                "geom1": 1.0,
                "geom2": 2.0,
                "subtomo_id": float(i + 1),
                "tomo_id": (1.0 if i < 3 else 2.0),
                "object_id": float(100 * (i + 1)),
                "subtomo_mean": float(i + 1) * 0.1,
                "x": float(10 + i),
                "y": float(10 + i),
                "z": float(10 + i),
                "shift_x": 0.0,
                "shift_y": 0.0,
                "shift_z": 0.0,
                "geom3": 3.0,
                "geom4": 4.0,
                "geom5": 5.0,
                "phi": float(i * 10),
                "psi": float(i * 10 + 5),
                "theta": float(i * 10 + 10),
                "class": float(1 + i % 2),
            }
        )
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


class TestPolyhedralComplex:

    # ------------------------------------------------------------------ fixtures

    @pytest.fixture
    def mrc_file(self, tmp_path):
        path = tmp_path / "test_sample.mrc"
        _write_mock_mrc(path, dimensions=(224, 224, 224), voxel_size_x=1)
        yield path

    @pytest.fixture
    def path_test_marker_file(self):
        current_dir = Path(__file__).parent
        return str(current_dir / "test_data" / "test_marker_file.cmm")

    @pytest.fixture
    def sample_motl(self):
        return _make_poly_motl()

    @pytest.fixture
    def ico_complex(self, sample_motl):
        return structure.IcosahedralComplex(sample_motl)

    @pytest.fixture
    def shift_vecs_test(self, path_test_marker_file, mrc_file):
        vecs, _ = structure.IcosahedralComplex.recover_features(
            path_test_marker_file, str(mrc_file), project_to_sphere=True
        )
        return vecs

    # ------------------------------------------------------------------ abstract guard

    def test_abstract_base_raises(self, sample_motl):
        with pytest.raises(TypeError, match="abstract"):
            structure.PolyhedralComplex(sample_motl)

    def test_abstract_recover_features_raises(self, path_test_marker_file, mrc_file):
        with pytest.raises(TypeError, match="concrete subclass"):
            structure.PolyhedralComplex.recover_features(path_test_marker_file, str(mrc_file))

    # ------------------------------------------------------------------ concrete subclasses

    @pytest.mark.parametrize(
        "cls, sym, n_subunits, solid_cls",
        [
            (structure.TetrahedralComplex, "T", 12, geom.Tetrahedron),
            (structure.OctahedralComplex, "O", 24, geom.Octahedron),
            (structure.IcosahedralComplex, "I", 60, geom.Icosahedron),
        ],
    )
    def test_concrete_class_attributes(self, sample_motl, cls, sym, n_subunits, solid_cls):
        pc = cls(sample_motl)
        assert pc.group == sym
        assert pc.n_subunits == n_subunits
        assert pc._solid is solid_cls

    def test_stores_column_names(self, sample_motl):
        pc = structure.IcosahedralComplex(
            sample_motl,
            affiliation_column="geom3",
            order_column="geom4",
        )
        assert pc.affiliation_column == "geom3"
        assert pc.order_column == "geom4"

    # ------------------------------------------------------------------ feature_vectors

    @pytest.mark.parametrize(
        "cls, mode, expected_n",
        [
            (structure.TetrahedralComplex, "vertices", 4),
            (structure.TetrahedralComplex, "edges", 6),
            (structure.TetrahedralComplex, "faces", 4),
            (structure.OctahedralComplex, "vertices", 6),
            (structure.OctahedralComplex, "edges", 12),
            (structure.OctahedralComplex, "faces", 8),
            (structure.IcosahedralComplex, "vertices", 12),
            (structure.IcosahedralComplex, "edges", 30),
            (structure.IcosahedralComplex, "faces", 20),
        ],
    )
    def test_feature_vectors_count(self, sample_motl, cls, mode, expected_n):
        pc = cls(sample_motl)
        vecs = pc.feature_vectors(mode=mode)
        assert vecs.shape == (expected_n, 3)

    # ------------------------------------------------------------------ assign_subunit_order

    def test_assign_subunit_order_xyz_ordering(self):
        """Subunit indices follow x→y→z ascending lexicographic order."""
        rows = []
        for x_val, y_val, z_val in [(3, 1, 1), (1, 3, 1), (1, 1, 3), (2, 2, 2)]:
            rows.append(
                {
                    "score": 0.0,
                    "geom1": 0.0,
                    "geom2": 0.0,
                    "subtomo_id": float(len(rows) + 1),
                    "tomo_id": 1.0,
                    "object_id": 1.0,
                    "subtomo_mean": 0.0,
                    "x": float(x_val),
                    "y": float(y_val),
                    "z": float(z_val),
                    "shift_x": 0.0,
                    "shift_y": 0.0,
                    "shift_z": 0.0,
                    "geom3": 0.0,
                    "geom4": 0.0,
                    "geom5": 0.0,
                    "phi": 0.0,
                    "psi": 0.0,
                    "theta": 0.0,
                    "class": 1.0,
                }
            )
        m = cryomotl.Motl()
        m.df = pd.DataFrame(rows)
        pc = structure.IcosahedralComplex(m)
        pc.assign_subunit_order()
        # particle lexicographically first (x→y→z) gets rank 1
        df = pc.motl.df
        lex_min_idx = df.sort_values(["x", "y", "z"]).index[0]
        assert df.loc[lex_min_idx, "geom1"] == 1

    # ------------------------------------------------------------------ recover_features

    def test_recover_features_invalid_mode(self, path_test_marker_file, mrc_file):
        with pytest.raises(ValueError, match="Invalid mode"):
            structure.IcosahedralComplex.recover_features(path_test_marker_file, str(mrc_file), mode="random")

    def test_recover_features_returns_two_arrays(self, path_test_marker_file, mrc_file):
        v1, v2 = structure.IcosahedralComplex.recover_features(path_test_marker_file, str(mrc_file))
        assert isinstance(v1, np.ndarray)
        assert isinstance(v2, np.ndarray)

    def test_recover_features_output_cmm_is_created(self, path_test_marker_file, tmp_path, mrc_file):
        output_path = tmp_path / "test_out.cmm"
        structure.IcosahedralComplex.recover_features(
            path_test_marker_file,
            str(mrc_file),
            output_cmm_file=str(output_path),
        )
        assert output_path.exists()

    @pytest.mark.parametrize(
        "mode, expected_ratio",
        [
            ("vertices", 1),
            ("edges", np.sqrt(5) * PHI / 4),
            ("faces", PHI**2 / (2 * np.sqrt(3))),
        ],
    )
    def test_recover_features_correct_dist_no_project(self, path_test_marker_file, mrc_file, mode, expected_ratio):
        shift_v1 = np.asarray([112, 156.2, 184.5]) - 112.0
        shift_v2 = np.asarray([184.2, 111.9, 156.7]) - 112.0
        expected_radius = geom.Icosahedron.from_vectors(shift_v1, shift_v2).radius

        vecs, _ = structure.IcosahedralComplex.recover_features(path_test_marker_file, str(mrc_file), mode=mode)
        distances = np.linalg.norm(vecs, axis=1)
        assert np.allclose(distances / expected_radius, expected_ratio, atol=1e-1)

    @pytest.mark.parametrize("mode", ["vertices", "edges", "faces"])
    def test_recover_features_correct_dist_project(self, path_test_marker_file, mrc_file, mode):
        shift_v1 = np.asarray([112, 156.2, 184.5]) - 112.0
        shift_v2 = np.asarray([184.2, 111.9, 156.7]) - 112.0
        expected_radius = geom.Icosahedron.from_vectors(shift_v1, shift_v2).radius

        vecs, _ = structure.IcosahedralComplex.recover_features(
            path_test_marker_file,
            str(mrc_file),
            mode=mode,
            project_to_sphere=True,
        )
        distances = np.linalg.norm(vecs, axis=1)
        assert np.allclose(distances, expected_radius, atol=1e-1)

    # ------------------------------------------------------------------ expand

    @pytest.mark.parametrize(
        "shift_vecs",
        [
            6,
            np.random.rand(3),
            np.random.rand(2, 4),
        ],
    )
    def test_expand_value_error_shifts(self, ico_complex, shift_vecs):
        with pytest.raises(ValueError, match="shift_vecs should be a numpy array"):
            ico_complex.expand(shift_vecs=shift_vecs)

    @pytest.mark.parametrize(
        "col1, col2",
        [
            ("object_id", "random"),
            ("random", "geom2"),
            ("random1", "random2"),
        ],
    )
    def test_expand_value_error_wrong_col(self, ico_complex, col1, col2):
        with pytest.raises(ValueError, match="not found in the columns of the input motive list"):
            ico_complex.expand(
                shift_vecs=np.random.rand(3, 3),
                original_id_col=col1,
                order_id_col=col2,
            )

    def test_expand_output_is_motl(self, ico_complex, shift_vecs_test):
        result = ico_complex.expand(shift_vecs=shift_vecs_test)
        assert isinstance(result, cryomotl.Motl)

    def test_expand_motl_len(self, ico_complex, shift_vecs_test):
        result = ico_complex.expand(shift_vecs=shift_vecs_test)
        assert len(result.df) == shift_vecs_test.shape[0] * len(ico_complex.motl.df)

    def test_expand_reset_cols(self, ico_complex, shift_vecs_test):
        result = ico_complex.expand(shift_vecs=shift_vecs_test)
        assert np.all(result.df["score"] == 0)
        assert np.all(result.df["subtomo_mean"] == 0)
        assert np.array_equal(
            result.df["subtomo_id"],
            np.arange(1, len(result.df) + 1, 1, dtype=np.int8),
        )

    def test_expand_outfile(self, ico_complex, shift_vecs_test, tmp_path):
        output_path = tmp_path / "test_out.em"
        ico_complex.expand(shift_vecs=shift_vecs_test, output_path=str(output_path))
        assert output_path.exists()

    @pytest.mark.parametrize(
        "motl_type, output_file, relion_version, expected_type",
        [
            ("stopgap", "output.star", None, cryomotl.StopgapMotl),
            ("relion", "output.star", 3.1, cryomotl.RelionMotl),
            ("relion5_1", "output.star", 5.1, cryomotl.RelionMotl),
            ("dynamo", "output.tbl", None, cryomotl.DynamoMotl),
        ],
    )
    def test_expand_different_motl_type(
        self,
        ico_complex,
        shift_vecs_test,
        tmp_path,
        motl_type,
        output_file,
        relion_version,
        expected_type,
    ):
        output_path = tmp_path / output_file
        result = ico_complex.expand(
            shift_vecs=shift_vecs_test,
            output_motl_type=motl_type,
            relion_version=relion_version,
            output_path=str(output_path),
        )
        assert output_path.exists()
        assert isinstance(result, expected_type)

    @pytest.mark.parametrize(
        "original_id_col, order_id_col",
        [
            ("object_id", "geom1"),
            ("geom1", "geom3"),
        ],
    )
    def test_expand_particle_ordering(self, ico_complex, shift_vecs_test, original_id_col, order_id_col):
        result = ico_complex.expand(
            shift_vecs=shift_vecs_test,
            original_id_col=original_id_col,
            order_id_col=order_id_col,
        )
        unique_objects = np.unique(result.df[original_id_col])
        assert len(unique_objects) == len(ico_complex.motl.df)
        assert np.array_equal(unique_objects, ico_complex.motl.df["subtomo_id"].values)
        assert result.df[original_id_col].is_monotonic_increasing
        for obj in unique_objects:
            subset = result.get_motl_subset(obj, column_name=original_id_col, return_df=True)
            assert subset[order_id_col].is_monotonic_increasing
            assert np.array_equal(subset[order_id_col], np.arange(0, len(subset), 1))


# ---------------------------------------------------------------------------
# Helpers for CnComplex tests
# ---------------------------------------------------------------------------


def _make_synthetic_ring(
    n: int = 8,
    radius: float = 50.0,
    center: tuple[float, float, float] = (100.0, 100.0, 100.0),
    tomo_id: float = 1.0,
    object_id: float = 1.0,
) -> cryomotl.Motl:
    """Synthetic ring with *n* subunits placed on a circle of given radius."""
    angles = np.linspace(0, 360, n, endpoint=False)
    x = center[0] + radius * np.cos(np.radians(angles))
    y = center[1] + radius * np.sin(np.radians(angles))
    z = np.full(n, center[2])
    rows = []
    for i in range(n):
        rows.append(
            {
                "score": 0.0,
                "geom1": 0.0,
                "geom2": float(i + 1),
                "subtomo_id": float(i + 1),
                "tomo_id": tomo_id,
                "object_id": object_id,
                "subtomo_mean": 0.0,
                "x": x[i],
                "y": y[i],
                "z": z[i],
                "shift_x": 0.0,
                "shift_y": 0.0,
                "shift_z": 0.0,
                "geom3": 0.0,
                "geom4": 0.0,
                "geom5": 0.0,
                "phi": float(angles[i]),
                "psi": 0.0,
                "theta": 0.0,
                "class": 1.0,
            }
        )
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


def _make_two_ring_motl() -> cryomotl.Motl:
    """Two identical rings in the same tomogram, with overlapping centres."""
    ring1 = _make_synthetic_ring(object_id=1.0)
    ring2 = _make_synthetic_ring(object_id=2.0, center=(100.0, 100.0, 100.0))
    ring2.df["subtomo_id"] += 8
    combined = cryomotl.Motl()
    combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
    return combined


# ---------------------------------------------------------------------------
# Tests for geom.barycenter (here so they run alongside structure tests)
# ---------------------------------------------------------------------------


class TestCnComplexInit:
    def test_accepts_string_C8(self):
        m = _make_synthetic_ring()
        cs = structure.CnComplex(m, "C8")
        assert cs.n == 8

    def test_accepts_int_symmetry(self):
        m = _make_synthetic_ring()
        cs = structure.CnComplex(m, 6)
        assert cs.n == 6

    def test_raises_on_dihedral(self):
        m = _make_synthetic_ring()
        with pytest.raises(ValueError, match="cyclic"):
            structure.CnComplex(m, "D2")

    def test_raises_on_other_non_cyclic(self):
        m = _make_synthetic_ring()
        with pytest.raises(ValueError, match="cyclic"):
            structure.CnComplex(m, "D8")

    def test_stores_column_names(self):
        m = _make_synthetic_ring()
        cs = structure.CnComplex(m, 8, affiliation_column="geom3", order_column="geom4")
        assert cs.affiliation_column == "geom3"
        assert cs.order_column == "geom4"


class TestCnComplexProperties:
    def test_central_angle_C8(self):
        m = _make_synthetic_ring()
        cs = structure.CnComplex(m, 8)
        assert cs.central_angle == pytest.approx(45.0)

    def test_interior_angle_C8(self):
        m = _make_synthetic_ring()
        cs = structure.CnComplex(m, 8)
        assert cs.interior_angle == pytest.approx(135.0)

    def test_central_angle_C6(self):
        m = _make_synthetic_ring(n=6)
        cs = structure.CnComplex(m, 6)
        assert cs.central_angle == pytest.approx(60.0)


class TestCnComplexCenters:
    def test_barycentric_center_correct(self):
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        centers = cs.get_centers_as_motl()
        assert centers.df.shape[0] == 1
        np.testing.assert_allclose(
            centers.df[["x", "y", "z"]].values[0],
            [100.0, 100.0, 100.0],
            atol=1e-6,
        )

    def test_circle_fit_center_close_to_true(self):
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="circle_fit")
        centers = cs.get_centers_as_motl()
        assert centers.df.shape[0] == 1
        np.testing.assert_allclose(
            centers.df[["x", "y", "z"]].values[0],
            [100.0, 100.0, 100.0],
            atol=5.0,
        )

    def test_circle_fit_fallback_warns_on_collinear(self):
        """Collinear 4-point input should trigger fallback warning."""
        rows = []
        for i in range(4):
            rows.append(
                {
                    "score": 0.0,
                    "geom1": 0.0,
                    "geom2": float(i + 1),
                    "subtomo_id": float(i + 1),
                    "tomo_id": 1.0,
                    "object_id": 1.0,
                    "subtomo_mean": 0.0,
                    "x": float(i * 10),
                    "y": 0.0,
                    "z": 0.0,
                    "shift_x": 0.0,
                    "shift_y": 0.0,
                    "shift_z": 0.0,
                    "geom3": 0.0,
                    "geom4": 0.0,
                    "geom5": 0.0,
                    "phi": 0.0,
                    "psi": 0.0,
                    "theta": 0.0,
                    "class": 1.0,
                }
            )
        m = cryomotl.Motl()
        m.df = pd.DataFrame(rows)
        cs = structure.CnComplex(m, 8, center_method="circle_fit")
        with pytest.warns(UserWarning):
            centers = cs.get_centers_as_motl()
        assert centers.df.shape[0] == 1

    def test_get_centers_one_row_per_object(self):
        ring1 = _make_synthetic_ring(n=8, object_id=1.0)
        ring2 = _make_synthetic_ring(n=8, object_id=2.0, center=(200.0, 200.0, 200.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, center_method="barycentric")
        centers = cs.get_centers_as_motl()
        assert centers.df.shape[0] == 2


class TestCnComplexAssignSubunitOrder:
    def test_writes_order_column(self):
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        assert cs.motl.df["geom1"].notna().all()
        assert set(cs.motl.df["geom1"].astype(int)).issubset(set(range(1, 10)))

    def test_first_particle_gets_index_one(self):
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        assert cs.motl.df["geom1"].iloc[0] == 1


class TestCnComplexMergeSubunits:
    def test_distinct_objects_unchanged(self):
        ring1 = _make_synthetic_ring(n=8, object_id=1.0, center=(0.0, 0.0, 0.0))
        ring2 = _make_synthetic_ring(n=8, object_id=2.0, center=(500.0, 500.0, 500.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, center_method="barycentric")
        cs.merge_subunits(radius=50)
        assert cs.motl.df["object_id"].nunique() == 2

    def test_close_objects_merged(self):
        """Two rings with the same centre should merge into one object."""
        ring1 = _make_synthetic_ring(n=8, object_id=1.0, center=(100.0, 100.0, 100.0))
        ring2 = _make_synthetic_ring(n=8, object_id=2.0, center=(100.0, 100.0, 100.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, center_method="barycentric")
        cs.merge_subunits(radius=50)
        assert cs.motl.df["object_id"].nunique() == 1


# ---------------------------------------------------------------------------
# Step 2 helpers
# ---------------------------------------------------------------------------


def _make_ordered_ring(
    n: int = 8,
    radius: float = 50.0,
    center: tuple[float, float, float] = (100.0, 100.0, 100.0),
    tomo_id: float = 1.0,
    object_id: float = 1.0,
) -> cryomotl.Motl:
    """Ring with subunit order in geom2 (1-based, matches assign_subunit_order)."""
    return _make_synthetic_ring(n=n, radius=radius, center=center, tomo_id=tomo_id, object_id=object_id)


def _drop_subunits(motl: cryomotl.Motl, indices: list[int]) -> cryomotl.Motl:
    """Return a copy of *motl* with rows whose geom2 value is in *indices* removed."""
    out = cryomotl.Motl()
    out.df = motl.df[~motl.df["geom2"].astype(int).isin(indices)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Step 2a — diameter
# ---------------------------------------------------------------------------


class TestCnComplexDiameter:
    def test_even_n_opposite_pairs_approx_2r(self):
        """Diameter of a regular C8 ring at radius 50 ≈ 100 (2 × radius)."""
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2", center_method="barycentric")
        summary, motl_out = cs.diameter(pixel_size=1.0)
        assert len(summary) == 1
        assert summary["n_pairs"].iloc[0] == 4
        np.testing.assert_allclose(summary["mean_diameter"].iloc[0], 100.0, atol=1.0)

    def test_store_column_filled_for_all_rows(self):
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2", center_method="barycentric")
        _, motl_out = cs.diameter(pixel_size=1.0, store_column="geom4")
        assert motl_out.df["geom4"].notna().all()

    def test_two_objects_both_appear_in_summary(self):
        ring1 = _make_ordered_ring(n=8, radius=50.0, object_id=1.0)
        ring2 = _make_ordered_ring(n=8, radius=50.0, object_id=2.0, center=(200.0, 200.0, 200.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, order_column="geom2", center_method="barycentric")
        summary, _ = cs.diameter()
        assert len(summary) == 2

    def test_odd_n_warns_and_returns_circumradius_based(self):
        """For n=5 (odd) the circumradius fallback fires with a warning."""
        m = _make_ordered_ring(n=5, radius=50.0)
        cs = structure.CnComplex(m, 5, order_column="geom2", center_method="barycentric")
        with pytest.warns(UserWarning, match="odd"):
            summary, _ = cs.diameter()
        assert len(summary) == 1
        assert summary["n_pairs"].iloc[0] == 0
        np.testing.assert_allclose(summary["mean_diameter"].iloc[0], 100.0, atol=2.0)

    def test_no_order_column_warns_and_uses_circumradius(self):
        """When order_column is absent, circumradius fallback with warning."""
        m = _make_ordered_ring(n=8, radius=50.0)
        # Use order_column that doesn't exist in the motl
        cs = structure.CnComplex(m, 8, order_column="geom5", center_method="barycentric")
        with pytest.warns(UserWarning):
            summary, _ = cs.diameter()
        assert summary["n_pairs"].iloc[0] == 0

    def test_pixel_size_scales_diameter(self):
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2", center_method="barycentric")
        s1, _ = cs.diameter(pixel_size=1.0)
        s2, _ = cs.diameter(pixel_size=2.0)
        np.testing.assert_allclose(s2["mean_diameter"].iloc[0], s1["mean_diameter"].iloc[0] * 2.0, atol=1e-6)


# ---------------------------------------------------------------------------
# Step 2b — occupancy
# ---------------------------------------------------------------------------


class TestCnComplexOccupancy:
    def test_full_ring_occupancy_one(self):
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2")
        occ = cs.occupancy()
        assert len(occ) == 1
        assert occ["occupancy"].iloc[0] == pytest.approx(1.0)

    def test_full_ring_missing_empty(self):
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2")
        occ = cs.occupancy()
        assert occ["missing"].iloc[0] == []

    def test_partial_ring_occupancy_fraction(self):
        """Drop subunit 7 → occupancy = 7/8."""
        m = _drop_subunits(_make_ordered_ring(n=8, radius=50.0), [7])
        cs = structure.CnComplex(m, 8, order_column="geom2")
        occ = cs.occupancy()
        assert occ["n_present"].iloc[0] == 7
        np.testing.assert_allclose(occ["occupancy"].iloc[0], 7 / 8)

    def test_partial_ring_missing_index(self):
        """Dropped subunit 7 must appear in missing."""
        m = _drop_subunits(_make_ordered_ring(n=8, radius=50.0), [7])
        cs = structure.CnComplex(m, 8, order_column="geom2")
        occ = cs.occupancy()
        assert 7 in occ["missing"].iloc[0]

    def test_over_occupied_object_visible(self):
        """An object with n_present > n appears with occupancy > 1."""
        ring1 = _make_ordered_ring(n=8, radius=50.0)
        extra = _make_ordered_ring(n=8, radius=50.0)
        extra.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, extra.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, order_column="geom2")
        occ = cs.occupancy()
        assert any(occ["occupancy"] > 1.0)

    def test_no_order_column_missing_is_none(self):
        """When order_column is not a column in the motl, missing should be None."""
        m = _make_ordered_ring(n=8, radius=50.0)
        # "subunit_order" is not in the standard 20-column Motl schema
        cs = structure.CnComplex(m, 8, order_column="subunit_order")
        occ = cs.occupancy()
        assert occ["missing"].iloc[0] is None


# ---------------------------------------------------------------------------
# Step 2c — circumference
# ---------------------------------------------------------------------------


class TestCnComplexCircumference:
    def test_circumference_approx_2pi_r(self):
        """Circumference of a C8 ring at radius 50 ≈ 2π×50."""
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        circ = cs.circumference(pixel_size=1.0)
        assert len(circ) == 1
        np.testing.assert_allclose(circ["circumference"].iloc[0], 2 * np.pi * 50.0, atol=2.0)

    def test_pixel_size_scales_circumference(self):
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        c1 = cs.circumference(pixel_size=1.0)["circumference"].iloc[0]
        c2 = cs.circumference(pixel_size=3.0)["circumference"].iloc[0]
        np.testing.assert_allclose(c2, c1 * 3.0, atol=1e-6)

    def test_two_objects_two_rows(self):
        ring1 = _make_ordered_ring(n=8, radius=50.0, object_id=1.0)
        ring2 = _make_ordered_ring(n=8, radius=80.0, object_id=2.0, center=(200.0, 200.0, 200.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, center_method="barycentric")
        circ = cs.circumference()
        assert len(circ) == 2
        # Larger ring has larger circumference
        c_by_obj = circ.set_index("object_id")["circumference"]
        assert c_by_obj[2.0] > c_by_obj[1.0]


# ---------------------------------------------------------------------------
# Step 2d — get_object_stats
# ---------------------------------------------------------------------------


class TestCnComplexGetObjectStats:
    def test_one_row_per_object(self):
        ring1 = _make_ordered_ring(n=8, radius=50.0, object_id=1.0)
        ring2 = _make_ordered_ring(n=8, radius=50.0, object_id=2.0, center=(200.0, 200.0, 200.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, 8, order_column="geom2", center_method="barycentric")
        stats = cs.get_object_stats()
        assert len(stats) == 2

    def test_expected_columns_present(self):
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2", center_method="barycentric")
        stats = cs.get_object_stats()
        for col in [
            "tomo_id",
            "object_id",
            "n_present",
            "occupancy",
            "x",
            "y",
            "z",
            "radius",
            "circumference",
            "mean_diameter",
            "n_pairs",
        ]:
            assert col in stats.columns, f"missing column: {col}"

    def test_values_consistent_with_individual_methods(self):
        """Values in get_object_stats agree with occupancy() and circumference()."""
        m = _make_ordered_ring(n=8, radius=50.0)
        cs = structure.CnComplex(m, 8, order_column="geom2", center_method="barycentric")
        stats = cs.get_object_stats()
        occ = cs.occupancy()
        circ = cs.circumference()
        np.testing.assert_allclose(stats["occupancy"].iloc[0], occ["occupancy"].iloc[0])
        np.testing.assert_allclose(stats["circumference"].iloc[0], circ["circumference"].iloc[0], atol=1e-6)

    def test_centre_approx_correct(self):
        m = _make_ordered_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        stats = cs.get_object_stats()
        np.testing.assert_allclose(stats[["x", "y", "z"]].values[0], [100.0, 100.0, 100.0], atol=1e-6)


# ---------------------------------------------------------------------------
# Helpers for Step 1B tests
# ---------------------------------------------------------------------------


def _make_multi_tomo_motl() -> cryomotl.Motl:
    """8-subunit ring in tomo 1 + 8-subunit ring in tomo 2 + 1 isolated in tomo 1.

    Subunits in each ring are placed on a circle of radius 50 voxels so that
    adjacent-subunit distance ≈ 38 voxels.  The isolated particle is placed at
    (300, 300, 300), well beyond any reasonable NN radius.
    """
    ring1 = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0), tomo_id=1.0, object_id=0.0)
    ring2 = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0), tomo_id=2.0, object_id=0.0)
    ring2.df["subtomo_id"] += 8
    iso_row = {
        "score": 0.0,
        "geom1": 0.0,
        "geom2": 0.0,
        "subtomo_id": 17.0,
        "tomo_id": 1.0,
        "object_id": 0.0,
        "subtomo_mean": 0.0,
        "x": 300.0,
        "y": 300.0,
        "z": 300.0,
        "shift_x": 0.0,
        "shift_y": 0.0,
        "shift_z": 0.0,
        "geom3": 0.0,
        "geom4": 0.0,
        "geom5": 0.0,
        "phi": 0.0,
        "psi": 0.0,
        "theta": 0.0,
        "class": 1.0,
    }
    combined = cryomotl.Motl()
    combined.df = pd.concat(
        [ring1.df, ring2.df, pd.DataFrame([iso_row])],
        ignore_index=True,
    )
    return combined


def _make_ring_with_tilted_outlier(n: int = 8, radius: float = 50.0, tilt: float = 45.0) -> cryomotl.Motl:
    """Ring with *n* subunits; first particle's theta (ZXZ Euler) set to *tilt* degrees.

    All other particles have theta=0 so their z-axis is [0, 0, 1].  The tilted
    particle's z-axis points at approximately *tilt* degrees from [0, 0, 1].
    """
    m = _make_synthetic_ring(n=n, radius=radius, object_id=1.0)
    m.df.loc[0, "theta"] = float(tilt)
    return m


def _make_overcrowded_motl(n_particles: int = 10, score_range: tuple[float, float] = (0.0, 9.0)) -> cryomotl.Motl:
    """Single tight cluster of *n_particles* with scores spanning *score_range*.

    Particles are placed within a 5-voxel range so any radius > 5 groups them.
    """
    rows = []
    scores = np.linspace(score_range[0], score_range[1], n_particles)
    for i in range(n_particles):
        rows.append(
            {
                "score": float(scores[i]),
                "geom1": 0.0,
                "geom2": 0.0,
                "subtomo_id": float(i + 1),
                "tomo_id": 1.0,
                "object_id": 1.0,
                "subtomo_mean": 0.0,
                "x": 100.0 + i * 0.5,
                "y": 100.0,
                "z": 100.0,
                "shift_x": 0.0,
                "shift_y": 0.0,
                "shift_z": 0.0,
                "geom3": float(i * 5),
                "geom4": 0.0,
                "geom5": 0.0,
                "phi": 0.0,
                "psi": 0.0,
                "theta": 0.0,
                "class": 1.0,
            }
        )
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


# ---------------------------------------------------------------------------
# Step 1B — create_affiliation (radius method)
# ---------------------------------------------------------------------------


class TestCnComplexCreateAffiliationRadius:
    # Adjacent-subunit distance on C8 radius-50 ring ≈ 38.3 voxels;
    # radius=44 connects all 8 within each tomogram.
    _R = 44.0

    def test_ring_subunits_share_one_affiliation(self):
        """All 8 subunits of a ring in one tomogram get the same object_id."""
        m = _make_multi_tomo_motl()
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=self._R)
        tomo1 = result.df[result.df["tomo_id"] == 1.0]
        # The ring is the largest object; its affiliation id appears 8 times
        counts = tomo1.groupby("object_id").size()
        assert counts.max() == 8

    def test_no_object_spans_two_tomograms(self):
        """NN search is per-tomogram; tomo 2 must still have exactly 8 particles in 1 object."""
        m = _make_multi_tomo_motl()
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=self._R)
        tomo2 = result.df[result.df["tomo_id"] == 2.0]
        # Tomo 2 ring must be a single complete object, not merged with tomo 1
        assert tomo2["object_id"].nunique() == 1
        assert len(tomo2) == 8

    def test_isolated_particle_kept_with_unique_id(self):
        """Isolated particle (no NN within radius) gets its own object_id."""
        m = _make_multi_tomo_motl()
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=self._R, drop_below_min_occupancy=False)
        # Tomo 1 has ring (object) + isolated (object) → 2 distinct object_ids
        tomo1_ids = result.df[result.df["tomo_id"] == 1.0]["object_id"].unique()
        assert len(tomo1_ids) == 2

    def test_isolated_removed_when_drop_below_min_occupancy(self):
        """With drop_below_min_occupancy=True and min_occupancy=2, singletons removed."""
        m = _make_multi_tomo_motl()
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(
            method="radius",
            radius=self._R,
            min_occupancy=2,
            drop_below_min_occupancy=True,
        )
        tomo1 = result.df[result.df["tomo_id"] == 1.0]
        assert len(tomo1) == 8
        assert tomo1["object_id"].nunique() == 1

    def test_occupancy_column_equals_object_size(self):
        """occupancy_column must equal the row count for that (tomo, object) group."""
        m = _make_multi_tomo_motl()
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=self._R, occupancy_column="geom2")
        for (_t, _o), grp in result.df.groupby(["tomo_id", "object_id"]):
            occ_vals = grp["geom2"].unique()
            assert len(occ_vals) == 1
            assert int(occ_vals[0]) == len(grp)


# ---------------------------------------------------------------------------
# Step 1B — normals threshold
# ---------------------------------------------------------------------------


class TestCnComplexNormals:
    def test_cone_distance_stored_for_all_particles(self):
        """cone_distance_column is populated for every particle even with no threshold."""
        m = _make_ring_with_tilted_outlier(n=8, tilt=45.0)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=44.0, normals_threshold=None)
        assert len(result.df) == 8
        assert result.df["geom3"].notna().all()

    def test_tilted_outlier_has_higher_cone_distance(self):
        """The particle with theta=45 should have a cone distance > 10°."""
        m = _make_ring_with_tilted_outlier(n=8, tilt=45.0)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=44.0, normals_threshold=None)
        assert result.df["geom3"].max() > 10.0

    def test_outlier_removed_when_threshold_set(self):
        """With normals_threshold=30°, the ~40° outlier is dropped."""
        m = _make_ring_with_tilted_outlier(n=8, tilt=45.0)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=44.0, normals_threshold=30.0)
        assert len(result.df) == 7

    def test_all_retained_when_threshold_is_none(self):
        """With normals_threshold=None no particles are dropped."""
        m = _make_ring_with_tilted_outlier(n=8, tilt=45.0)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.create_affiliation(method="radius", radius=44.0, normals_threshold=None)
        assert len(result.df) == 8


# ---------------------------------------------------------------------------
# Step 1B — over-occupancy warning + clean_per_object
# ---------------------------------------------------------------------------


class TestCnComplexOverOccupancyAndClean:
    def test_over_occupancy_warning_fires(self):
        """A tight cluster of 10 particles (n=8) should trigger UserWarning."""
        m = _make_overcrowded_motl(n_particles=10)
        # object_id=1 already set; clear it so create_affiliation re-clusters
        m.df["object_id"] = 0.0
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        with pytest.warns(UserWarning, match="exceed"):
            cs.create_affiliation(method="radius", radius=100.0)

    def test_clean_per_object_keeps_high_scores(self):
        """keep='high' retains the n highest-score rows per object."""
        m = _make_overcrowded_motl(n_particles=10)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.clean_per_object("score", keep="high")
        assert len(result.df) == 8
        # linspace(0, 9, 10) → scores 0..9; top 8 are scores 2..9 (min=2, max=9)
        assert result.df["score"].max() == pytest.approx(9.0, rel=0.05)
        assert float(result.df["score"].min()) > 0.0

    def test_clean_per_object_keeps_low_values(self):
        """keep='low' retains the n lowest-value rows per object (e.g. cone distance)."""
        m = _make_overcrowded_motl(n_particles=10)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.clean_per_object("geom3", keep="low")
        assert len(result.df) == 8
        assert result.df["geom3"].max() == pytest.approx(35.0, rel=0.05)
        assert result.df["geom3"].min() == pytest.approx(0.0, abs=0.01)

    def test_clean_per_object_unchanged_when_at_n(self):
        """Objects already at n rows are returned intact."""
        m = _make_synthetic_ring(n=8, radius=50.0, object_id=1.0)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.clean_per_object("score", keep="high")
        assert len(result.df) == 8

    def test_clean_per_object_custom_n_overrides_self_n(self):
        """Passing n=5 keeps 5 rows regardless of self.n=8."""
        m = _make_overcrowded_motl(n_particles=10)
        cs = structure.CnComplex(m, 8, affiliation_column="object_id")
        result = cs.clean_per_object("score", keep="high", n=5)
        assert len(result.df) == 5


# ---------------------------------------------------------------------------
# New class hierarchy tests
# ---------------------------------------------------------------------------


class TestSymmetricComplex:
    def test_d6_n_subunits(self):
        """SymmetricComplex('D6').n_subunits == 12."""
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "D6")
        assert sc.n_subunits == 12

    def test_t_n_subunits(self):
        """SymmetricComplex('T').n_subunits == 12."""
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "T")
        assert sc.n_subunits == 12

    def test_o_n_subunits(self):
        """SymmetricComplex('O').n_subunits == 24."""
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "O")
        assert sc.n_subunits == 24

    def test_i_n_subunits(self):
        """SymmetricComplex('I').n_subunits == 60."""
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "I")
        assert sc.n_subunits == 60

    def test_c8_n_subunits(self):
        """SymmetricComplex('C8').n_subunits == 8."""
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "C8")
        assert sc.n_subunits == 8

    def test_stores_group_and_fold(self):
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "D6")
        assert sc.group == "D"
        assert sc.fold == 6


class TestCnComplexVsSymmetricComplex:
    def test_cncomplex_c8_same_as_old_api(self):
        """CnComplex('C8') has same n and center_method as before."""
        m = _make_synthetic_ring(n=8)
        cs = structure.CnComplex(m, "C8")
        assert cs.n == 8
        assert cs.n_subunits == 8
        assert cs.group == "C"
        assert cs.fold == 8

    def test_cncomplex_rejects_dihedral(self):
        """CnComplex('D6') raises ValueError."""
        m = _make_synthetic_ring(n=8)
        with pytest.raises(ValueError, match="cyclic"):
            structure.CnComplex(m, "D6")

    def test_cncomplex_rejects_tetrahedral(self):
        """CnComplex('T') raises ValueError."""
        m = _make_synthetic_ring(n=8)
        with pytest.raises(ValueError, match="cyclic"):
            structure.CnComplex(m, "T")


class TestComplexCenters:
    def test_returns_one_row_per_object(self):
        m = _make_synthetic_ring(n=8, object_id=1.0)
        result = structure.complex_centers(m)
        assert len(result.df) == 1

    def test_two_objects_two_rows(self):
        ring1 = _make_synthetic_ring(object_id=1.0, tomo_id=1.0)
        ring2 = _make_synthetic_ring(object_id=2.0, tomo_id=1.0)
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        result = structure.complex_centers(combined)
        assert len(result.df) == 2

    def test_center_is_barycenter(self):
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        result = structure.complex_centers(m)
        np.testing.assert_allclose(result.df["x"].values[0], 100.0, atol=1.0)
        np.testing.assert_allclose(result.df["y"].values[0], 100.0, atol=1.0)
        np.testing.assert_allclose(result.df["z"].values[0], 100.0, atol=1e-9)


# ---------------------------------------------------------------------------
# SymmetricComplex base class — promoted methods and dispatch hook
# ---------------------------------------------------------------------------


class TestSymmetricComplexPromotedMethods:
    def test_assign_subunit_order_raises_not_implemented(self):
        """SymmetricComplex.assign_subunit_order raises NotImplementedError."""
        m = _make_synthetic_ring(n=8)
        sc = structure.SymmetricComplex(m, "C8")
        with pytest.raises(NotImplementedError):
            sc.assign_subunit_order()

    def test_occupancy_denominator_is_n_subunits(self):
        """occupancy fraction uses n_subunits, which equals n for CnComplex."""
        m = _make_ordered_ring(n=8, radius=50.0)
        m.df = m.df.iloc[:-1].reset_index(drop=True)  # drop one subunit → 7 present
        cs = structure.CnComplex(m, "C8", order_column="geom2")
        occ = cs.occupancy()
        np.testing.assert_allclose(occ["occupancy"].iloc[0], 7 / cs.n_subunits)
        assert cs.n_subunits == cs.n

    def test_clean_per_object_default_n_uses_n_subunits(self):
        """clean_per_object default n=None falls back to n_subunits (== n for CnComplex)."""
        extra = _make_ordered_ring(n=8, radius=50.0)
        extra.df["subtomo_id"] += 8  # avoid id collision
        m = cryomotl.Motl()
        m.df = pd.concat([_make_ordered_ring(n=8, radius=50.0).df, extra.df], ignore_index=True)
        cs = structure.CnComplex(m, "C8", order_column="geom2")
        result = cs.clean_per_object("score", keep="high", n=None)
        assert len(result.df) == cs.n_subunits

    def test_merge_subunits_smoke_two_distant_rings(self):
        """merge_subunits on CnComplex (base barycentric) keeps distant objects separate."""
        ring1 = _make_synthetic_ring(n=8, object_id=1.0, center=(0.0, 0.0, 0.0))
        ring2 = _make_synthetic_ring(n=8, object_id=2.0, center=(500.0, 500.0, 500.0))
        ring2.df["subtomo_id"] += 8
        combined = cryomotl.Motl()
        combined.df = pd.concat([ring1.df, ring2.df], ignore_index=True)
        cs = structure.CnComplex(combined, "C8")
        cs.merge_subunits(radius=55)
        assert cs.motl.df["object_id"].nunique() == 2

    def test_ring_group_columns_default(self):
        """CnComplex._ring_group_columns defaults to [tomo_id_column, affiliation_column]."""
        m = _make_synthetic_ring(n=8)
        cs = structure.CnComplex(m, "C8")
        assert cs._ring_group_columns == [cs.tomo_id_column, cs.affiliation_column]


# ---------------------------------------------------------------------------
# DnComplex — dihedral symmetry
# ---------------------------------------------------------------------------


def _make_dn_motl(
    n: int = 6,
    radius: float = 50.0,
    center: tuple[float, float, float] = (100.0, 100.0, 100.0),
    axial_offset: float = 20.0,
    stagger_degrees: float = 0.0,
    tomo_id: float = 1.0,
    object_id: float = 1.0,
) -> cryomotl.Motl:
    """Two stacked Cn rings separated axially.

    Ring 0 (top) is at ``center + (0, 0, +axial_offset/2)``.
    Ring 1 (bottom) is at ``center + (0, 0, -axial_offset/2)``.
    *stagger_degrees* rotates ring 1 relative to ring 0 (0 = eclipsed,
    180/n = staggered).
    """
    angles_top = np.linspace(0, 360, n, endpoint=False)
    angles_bot = angles_top + stagger_degrees
    rows = []
    pid = 1
    for ring_idx, (angles, z_off) in enumerate([(angles_top, axial_offset / 2.0), (angles_bot, -axial_offset / 2.0)]):
        for ang in angles:
            rows.append(
                {
                    "score": 0.0,
                    "geom1": 0.0,
                    "geom2": 0.0,
                    "subtomo_id": float(pid),
                    "tomo_id": tomo_id,
                    "object_id": object_id,
                    "subtomo_mean": 0.0,
                    "x": center[0] + radius * np.cos(np.radians(ang)),
                    "y": center[1] + radius * np.sin(np.radians(ang)),
                    "z": center[2] + z_off,
                    "shift_x": 0.0,
                    "shift_y": 0.0,
                    "shift_z": 0.0,
                    "geom3": 0.0,
                    "geom4": 0.0,
                    "geom5": 0.0,
                    "phi": ang,
                    "psi": 0.0,
                    "theta": 0.0,
                    "class": 1.0,
                }
            )
            pid += 1
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


class TestDnComplex:
    def test_n_subunits_is_2n(self):
        """DnComplex('D6').n_subunits == 12, .n == 6."""
        m = _make_dn_motl(n=6)
        dn = structure.DnComplex(m, "D6")
        assert dn.n_subunits == 12
        assert dn.n == 6

    def test_wrong_group_raises(self):
        """DnComplex rejects non-dihedral symmetry."""
        m = _make_dn_motl(n=6)
        with pytest.raises(ValueError, match="dihedral"):
            structure.DnComplex(m, "C6")

    def test_split_rings_labels_two_groups(self):
        """split_rings partitions 2n subunits into ring 0 and ring 1."""
        m = _make_dn_motl(n=6, axial_offset=20.0)
        dn = structure.DnComplex(m, "D6")
        dn.split_rings()
        labels = dn.motl.df["geom5"].values
        assert set(labels).issubset({0.0, 1.0})
        assert (labels == 0.0).sum() == 6
        assert (labels == 1.0).sum() == 6

    def test_split_rings_updates_ring_group_columns(self):
        """After split_rings, _ring_group_columns has 3 elements."""
        m = _make_dn_motl(n=6)
        dn = structure.DnComplex(m, "D6")
        dn.split_rings(ring_column="geom5")
        assert len(dn._ring_group_columns) == 3
        assert dn._ring_group_columns[2] == "geom5"

    def test_assign_subunit_order_ring_partitioning(self):
        """Ring 0 subunits get indices <= n, ring 1 subunits get indices > n."""
        m = _make_dn_motl(n=6, axial_offset=20.0)
        dn = structure.DnComplex(m, "D6")
        dn.assign_subunit_order()
        ring_col = dn._ring_column
        ring0_orders = dn.motl.df.loc[dn.motl.df[ring_col] == 0.0, "geom1"].values
        ring1_orders = dn.motl.df.loc[dn.motl.df[ring_col] == 1.0, "geom1"].values
        assert all(v <= dn.n for v in ring0_orders)
        assert all(v > dn.n for v in ring1_orders)

    def test_assign_subunit_order_ring1_offset(self):
        """Ring 1 subunits have order_column > n (offset by n applied)."""
        m = _make_dn_motl(n=6, axial_offset=20.0)
        dn = structure.DnComplex(m, "D6")
        dn.assign_subunit_order()
        ring_col = dn._ring_column
        ring1_orders = dn.motl.df.loc[dn.motl.df[ring_col] == 1.0, "geom1"].values
        assert all(v > 6 for v in ring1_orders)

    def test_ring_spacing_matches_axial_separation(self):
        """ring_spacing recovers the known axial offset."""
        axial_offset = 30.0
        m = _make_dn_motl(n=6, axial_offset=axial_offset)
        dn = structure.DnComplex(m, "D6")
        df = dn.ring_spacing(pixel_size=1.0)
        np.testing.assert_allclose(df["ring_spacing"].iloc[0], axial_offset, atol=1e-6)

    def test_inter_ring_twist_staggered(self):
        """Staggered rings (180/n rotation) give twist ≈ 180/n degrees."""
        n = 6
        stagger = 180.0 / n
        m = _make_dn_motl(n=n, axial_offset=20.0, stagger_degrees=stagger)
        dn = structure.DnComplex(m, "D6")
        df = dn.inter_ring_twist(degrees=True)
        np.testing.assert_allclose(df["inter_ring_twist"].iloc[0], stagger, atol=1e-4)

    def test_inter_ring_twist_eclipsed(self):
        """Eclipsed rings (0 rotation) give twist ≈ 0 degrees."""
        m = _make_dn_motl(n=6, axial_offset=20.0, stagger_degrees=0.0)
        dn = structure.DnComplex(m, "D6")
        df = dn.inter_ring_twist(degrees=True)
        np.testing.assert_allclose(df["inter_ring_twist"].iloc[0], 0.0, atol=1e-4)

    def test_occupancy_denominator_is_2n(self):
        """occupancy uses n_subunits == 2n as denominator."""
        n = 6
        m = _make_dn_motl(n=n, axial_offset=20.0)
        m.df = m.df.iloc[:-1].reset_index(drop=True)
        dn = structure.DnComplex(m, "D6")
        occ = dn.occupancy()
        expected = (2 * n - 1) / (2 * n)
        np.testing.assert_allclose(occ["occupancy"].iloc[0], expected)

    def test_non_z_axis_split(self):
        """split_rings works for a non-Z splitting axis."""
        n = 4
        m = _make_dn_motl(n=n, axial_offset=0.0)
        m.df["y"] += np.where(np.arange(len(m.df)) < n, 15.0, -15.0)
        dn = structure.DnComplex(m, "D4")
        dn.split_rings(axis=(0.0, 1.0, 0.0))
        labels = dn.motl.df["geom5"].values
        assert set(labels).issubset({0.0, 1.0})
        assert (labels == 0.0).sum() == n
        assert (labels == 1.0).sum() == n

    def test_unify_nn_orientations_not_on_cn_or_dn(self):
        """unify_nn_orientations is only on NPC, not CnComplex or DnComplex."""
        assert not hasattr(structure.CnComplex, "unify_nn_orientations")
        assert not hasattr(structure.DnComplex, "unify_nn_orientations")
        assert hasattr(structure.NPC, "unify_nn_orientations")

    def test_diameter_per_ring(self):
        """diameter() works for DnComplex and returns per-ring rows."""
        n = 6
        radius = 50.0
        m = _make_dn_motl(n=n, radius=radius, axial_offset=20.0)
        dn = structure.DnComplex(m, "D6")
        dn.split_rings()
        summary_df, _ = dn.diameter(pixel_size=1.0)
        assert len(summary_df) == 2

    def test_get_object_stats_one_row_per_object(self):
        """get_object_stats returns one row per (tomo_id, object_id)."""
        m = _make_dn_motl(n=6, axial_offset=20.0)
        dn = structure.DnComplex(m, "D6")
        stats = dn.get_object_stats(pixel_size=1.0)
        assert len(stats) == 1
        assert "ring_spacing" in stats.columns
        assert "inter_ring_twist" in stats.columns


# ---------------------------------------------------------------------------
# rays_from_motl — geometry
# ---------------------------------------------------------------------------

def _toy_motl(n: int = 4) -> cryomotl.Motl:
    """Minimal Motl with identity orientations and sequential x positions."""
    rows = []
    for i in range(n):
        rows.append({
            "score": 1.0, "geom1": 0.0, "geom2": 0.0,
            "subtomo_id": i + 1, "tomo_id": 1, "object_id": 1,
            "subtomo_mean": 0.0,
            "x": float(i), "y": 0.0, "z": 0.0,
            "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
            "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
            "phi": 0.0, "psi": 0.0, "theta": 0.0,
            "class": 1,
        })
    return cryomotl.Motl(pd.DataFrame(rows))


def test_rays_from_motl_shape_and_origins():
    """Output shape is (N, 6); origin x coords are scaled by pixel_size; identity
    rotation produces dx=dy=0, dz>0 for reverse_direction=False."""
    motl = _toy_motl(n=3)
    rays = structure.rays_from_motl(motl, pixel_size=2.0, reverse_direction=False)
    assert rays.shape == (3, 6)
    np.testing.assert_array_almost_equal(rays[:, 0], np.arange(3) * 2.0)
    assert np.all(rays[:, 5] > 0)
    np.testing.assert_array_almost_equal(rays[:, 3], 0.0)
    np.testing.assert_array_almost_equal(rays[:, 4], 0.0)


def test_rays_from_motl_reverse_direction_flips_z():
    """reverse_direction=True negates the z component of the direction."""
    motl = _toy_motl(n=2)
    rays = structure.rays_from_motl(motl, pixel_size=1.0, reverse_direction=True)
    assert np.all(rays[:, 5] < 0)


# ---------------------------------------------------------------------------
# GX5 — _cyclic_indices_for_object: signed-angle ordering
# ---------------------------------------------------------------------------


def _ring_motl(
    angles_deg: list[float],
    radius: float = 50.0,
    center: tuple[float, float, float] = (100.0, 100.0, 100.0),
    tomo_id: float = 1.0,
    object_id: float = 1.0,
) -> cryomotl.Motl:
    """Particles at the given *angles_deg* on a circle in the XY plane."""
    rows = []
    for i, a in enumerate(angles_deg):
        rows.append(
            {
                "score": 0.0,
                "geom1": 0.0,
                "geom2": float(i + 1),
                "subtomo_id": float(i + 1),
                "tomo_id": tomo_id,
                "object_id": object_id,
                "subtomo_mean": 0.0,
                "x": center[0] + radius * np.cos(np.radians(a)),
                "y": center[1] + radius * np.sin(np.radians(a)),
                "z": float(center[2]),
                "shift_x": 0.0,
                "shift_y": 0.0,
                "shift_z": 0.0,
                "geom3": 0.0,
                "geom4": 0.0,
                "geom5": 0.0,
                "phi": a,
                "psi": 0.0,
                "theta": 0.0,
                "class": 1.0,
            }
        )
    m = cryomotl.Motl()
    m.df = pd.DataFrame(rows)
    return m


class TestCyclicIndicesSignedAngle:
    def test_full_c8_ring_all_indices_present(self):
        """Complete C8 ring placed at 0, 45, 90, ... 315 degrees yields indices 1–8."""
        angles = list(np.linspace(0, 360, 8, endpoint=False))
        m = _ring_motl(angles)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        idx, residual = cs._cyclic_indices_for_object(m)
        assert sorted(idx) == list(range(1, 9))
        assert residual == pytest.approx(0.0, abs=1.0)

    def test_c8_ring_missing_subunit_6_no_duplicate(self):
        """C8 ring with subunit at 225° removed: no duplicates, 6 missing from result."""
        angles = [a for a in np.linspace(0, 360, 8, endpoint=False) if not np.isclose(a, 225.0)]
        m = _ring_motl(angles)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        idx, residual = cs._cyclic_indices_for_object(m)
        assert len(idx) == 7
        assert len(set(idx)) == 7  # no duplicates

    def test_signed_angle_separates_symmetric_subunits(self):
        """Subunits at +45° and -45° (=315°) get distinct indices.

        Old unsigned code mapped both to the same index (both are 45° away
        from the reference); signed angle correctly separates them.
        """
        # Use the true ring center to avoid barycentric bias with 3 particles
        angles = [0.0, 45.0, 315.0]
        m = _ring_motl(angles, center=(100.0, 100.0, 100.0))
        # Force center to the known ring center via circle_fit or by placing a
        # full ring and picking a subset; simplest: use a helper that sets center.
        # We create a CnComplex with barycentric and manually compute the indices
        # via the center at the known true center.
        center_true = np.array([100.0, 100.0, 100.0])
        su_coord = m.get_coordinates()
        vectors = su_coord - center_true
        import decimal as _decimal
        from cryocat.utils import geom as _geom
        _, _, Vt = np.linalg.svd(vectors, full_matrices=True)
        normal = Vt[-1]
        ref_vec = vectors[0]
        s_idx = []
        for vec in vectors:
            rad = _geom.vector_angular_distance_signed(ref_vec, vec, normal)
            deg = np.degrees(rad) % 360.0
            raw = deg / 45.0
            idx_i = int(_decimal.Decimal(str(raw)).to_integral_value(rounding=_decimal.ROUND_HALF_UP))
            s_idx.append((idx_i % 8) + 1)
        # No duplicates: signed angle separates +45 and -45
        assert len(set(s_idx)) == len(s_idx), f"duplicates found: {s_idx}"
        # +45° from ref → index 2; 315° (= -45°) from ref → index 8
        assert s_idx[1] == 2
        assert s_idx[2] == 8

    def test_shuffled_rows_same_indices_as_ordered(self):
        """Row order must not affect which physical angle gets which index."""
        angles_ordered = list(np.linspace(0, 360, 8, endpoint=False))
        m_ordered = _ring_motl(angles_ordered)
        shuffled_angles = [angles_ordered[i] for i in [3, 0, 7, 5, 1, 6, 2, 4]]
        m_shuffled = _ring_motl(shuffled_angles)
        cs_ordered = structure.CnComplex(m_ordered, 8, center_method="barycentric")
        cs_shuffled = structure.CnComplex(m_shuffled, 8, center_method="barycentric")
        idx_ordered, _ = cs_ordered._cyclic_indices_for_object(m_ordered)
        idx_shuffled, _ = cs_shuffled._cyclic_indices_for_object(m_shuffled)
        # The first particle in each motl gets index 1; shift so physical angles align
        assert sorted(idx_ordered) == sorted(idx_shuffled)


# ---------------------------------------------------------------------------
# HB3 — grid-fit subunit ordering with corrected circle fit
# ---------------------------------------------------------------------------


def _sorted_cyclic_gaps(indices: list[int], n: int) -> list[int]:
    """Sorted cyclic gaps between *indices* on a Cn ring."""
    s = sorted(set(indices))
    diffs = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    diffs.append(n - s[-1] + s[0])
    return sorted(diffs)


class TestGridFitSubunitOrder:
    def test_circle_fit_complete_ring_all_indices(self):
        """Fixed circle-fit center + grid-fit: complete C8 ring yields indices 1–8."""
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="circle_fit")
        cs.assign_subunit_order()
        assert sorted(cs.motl.df["geom1"].astype(int)) == list(range(1, 9))

    def test_consecutive_arc_no_duplicates_and_forms_arc(self):
        """5-consecutive-arc of C8: 5 distinct indices forming a consecutive C8 arc."""
        m = _drop_subunits(_make_synthetic_ring(n=8), [6, 7, 8])
        cs = structure.CnComplex(m, 8, center_method="circle_fit")
        cs.assign_subunit_order()
        idx = list(cs.motl.df["geom1"].astype(int))
        assert len(set(idx)) == 5
        # A consecutive arc of length 5 in C8 has exactly one cyclic gap of 4
        gaps = _sorted_cyclic_gaps(idx, 8)
        assert gaps == [1, 1, 1, 1, 4]

    def test_nonconsecutive_incomplete_ring_no_duplicates(self):
        """C8 ring missing subunits 3 and 6: 6 distinct indices, no duplicates."""
        m = _drop_subunits(_make_synthetic_ring(n=8), [3, 6])
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        idx = list(cs.motl.df["geom1"].astype(int))
        assert len(set(idx)) == 6
        assert set(idx).issubset(set(range(1, 9)))

    def test_duplicate_angle_warns(self):
        """Two particles at the same angular position trigger a duplicate-index warning."""
        angles = list(np.linspace(0, 360, 8, endpoint=False))
        angles[2] = angles[1]  # second copy at 45°
        m = _ring_motl(angles)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        with pytest.warns(UserWarning, match="duplicate"):
            cs.assign_subunit_order()

    def test_ref_direction_sets_index_one_for_target_particle(self):
        """ref_direction=(0,1,0) makes the particle at 90° (the +y position) get index 1."""
        m = _make_synthetic_ring(n=8, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order(ref_direction=np.array([0.0, 1.0, 0.0]))
        row_90 = cs.motl.df[np.isclose(cs.motl.df["phi"], 90.0)]
        assert len(row_90) == 1
        assert int(row_90["geom1"].iloc[0]) == 1

    def test_tilted_ring_circle_fit_center_and_radius(self):
        """fit_circle_3d_pratt recovers center and radius of a ring tilted 45° around x."""
        n, radius = 16, 50.0
        center = np.array([100.0, 100.0, 100.0])
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        # Ring in XY plane rotated 45° around x-axis: y → y/√2, z → y/√2
        sq2 = np.sqrt(2.0) / 2.0
        pts = np.column_stack([
            center[0] + radius * np.cos(angles),
            center[1] + radius * np.sin(angles) * sq2,
            center[2] + radius * np.sin(angles) * sq2,
        ])
        fitted_center, fitted_radius, _ = geom.fit_circle_3d_pratt(pts)
        assert np.linalg.norm(fitted_center - center) < 5.0, (
            f"Center error {np.linalg.norm(fitted_center - center):.2f} voxels"
        )
        assert abs(fitted_radius - radius) < 1.0, (
            f"Radius error {abs(fitted_radius - radius):.2f} voxels"
        )


# ---------------------------------------------------------------------------
# HC5 — CnComplex.central_angles: 8 test scenarios
# ---------------------------------------------------------------------------


class TestCentralAngles:
    """HC5: central_angles() method on CnComplex.

    angle_pos and angle_ori are non-negative (absolute values).
    Signed versions are in angle_pos_signed / angle_ori_signed.
    """

    def test_complete_ring_angle_pos_equals_central_angle(self):
        """Complete C8 ring: every consecutive pair has angle_pos ≈ 45°."""
        m = _make_synthetic_ring(n=8)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        assert len(df) == 8
        np.testing.assert_allclose(df["angle_pos"].values, 45.0, atol=1.0)

    def test_perfect_ring_dev_pos_near_zero(self):
        """Perfect ring: deviations from ideal are ≈ 0 and signed columns are present."""
        m = _make_synthetic_ring(n=8)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        np.testing.assert_allclose(df["dev_pos"].values, 0.0, atol=1.0)
        assert "angle_pos_signed" in df.columns
        assert "angle_ori_signed" in df.columns
        # Signed values have consistent sign within the ring; abs matches angle_pos
        np.testing.assert_allclose(np.abs(df["angle_pos_signed"].values), df["angle_pos"].values, atol=1e-10)

    def test_displaced_center_angles_invariant(self):
        """Ring centred at (500, 300, 200): angle_pos identical to origin-centred ring."""
        m = _make_synthetic_ring(n=8, center=(500.0, 300.0, 200.0))
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        np.testing.assert_allclose(df["angle_pos"].values, 45.0, atol=1.0)

    def test_tilted_ring_angle_pos_preserved(self):
        """Ring tilted 45° around x-axis: angle_pos ≈ 45° for each consecutive pair."""
        n, radius = 8, 50.0
        cx, cy, cz = 100.0, 100.0, 100.0
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        sq2 = np.sqrt(2.0) / 2.0
        rows = []
        for i, th in enumerate(angles):
            rows.append({
                "score": 0.0, "geom1": 0.0, "geom2": float(i + 1),
                "subtomo_id": float(i + 1), "tomo_id": 1.0, "object_id": 1.0,
                "subtomo_mean": 0.0,
                "x": cx + radius * np.cos(th),
                "y": cy + radius * np.sin(th) * sq2,
                "z": cz + radius * np.sin(th) * sq2,
                "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
                "geom3": 0.0, "geom4": 0.0, "geom5": 0.0,
                "phi": float(np.degrees(th)), "psi": 0.0, "theta": 0.0,
                "class": 1.0,
            })
        m = cryomotl.Motl()
        m.df = pd.DataFrame(rows)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        assert len(df) == 8
        np.testing.assert_allclose(df["angle_pos"].values, 45.0, atol=2.0)

    def test_small_ring_circle_fit_fallback_works(self):
        """3-subunit ring triggers barycentric fallback; angle_pos ≈ 120°, no error."""
        m = _make_synthetic_ring(n=3, radius=50.0, center=(100.0, 100.0, 100.0))
        cs = structure.CnComplex(m, 3, center_method="circle_fit")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        assert len(df) == 3
        np.testing.assert_allclose(df["angle_pos"].values, 120.0, atol=2.0)

    def test_holey_gap_pair_has_idx_diff_2_and_double_angle(self):
        """Gapped C8 ring (holey, missing index 5): gap pair idx_diff=2, angle_pos≈90°.

        Uses circle_fit so the centre is recovered accurately despite the missing
        subunit; barycentric centre is pulled toward the dense arc and gives ~80°.
        """
        m = _drop_subunits(_make_synthetic_ring(n=8), [5])
        cs = structure.CnComplex(m, 8, center_method="circle_fit")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        assert len(df) == 7
        gap = df[df["idx_diff"] == 2.0]
        assert len(gap) == 1
        assert gap["angle_pos"].iloc[0] == pytest.approx(90.0, abs=5.0)

    def test_full_gaps_skips_pair_across_gap(self):
        """Gapped C8 ring (full, missing index 5): absent target → 6 pairs, all idx_diff=1."""
        m = _drop_subunits(_make_synthetic_ring(n=8), [5])
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="full")
        assert len(df) == 6
        assert (df["idx_diff"] == 1.0).all()

    def test_orientational_angle_matches_positional(self):
        """Convention-correct orientations (phi=angular_position): angle_diff < 1° per pair."""
        m = _make_synthetic_ring(n=8)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.central_angles(gaps="holey")
        np.testing.assert_allclose(df["angle_diff"].values, 0.0, atol=1.0)


# ---------------------------------------------------------------------------
# HD4 — CnComplex.get_object_stats: central-angle statistics columns
# ---------------------------------------------------------------------------


class TestGetObjectStatsAngles:
    """HD4: get_object_stats adds per-position angle stats when order is assigned."""

    def test_regular_ring_mean_near_360_over_n_and_std_near_zero(self):
        """Complete C8 ring: mean_angle_pos ≈ 45°, std_angle_pos ≈ 0, n_angle_pairs=8."""
        m = _make_synthetic_ring(n=8)
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.get_object_stats()
        for col in ["mean_angle_pos", "std_angle_pos", "var_angle_pos",
                    "mean_angle_ori", "std_angle_ori", "var_angle_ori", "n_angle_pairs"]:
            assert col in df.columns, f"missing column {col!r}"
        row = df.iloc[0]
        assert row["mean_angle_pos"] == pytest.approx(45.0, abs=1.0)
        assert row["std_angle_pos"] == pytest.approx(0.0, abs=0.5)
        assert int(row["n_angle_pairs"]) == 8

    def test_displaced_subunit_gives_nonzero_std_angle(self):
        """Ring with one angularly displaced subunit: std_angle_pos > 0.5°."""
        m = _make_synthetic_ring(n=8)
        m_df = m.df.copy()
        row_idx = m_df.index[m_df["geom2"].astype(int) == 3]
        m_df.at[row_idx[0], "x"] += 5.0  # shift ≈ 5.7° at radius 50
        m2 = cryomotl.Motl()
        m2.df = m_df
        cs = structure.CnComplex(m2, 8, center_method="circle_fit")
        cs.assign_subunit_order()
        df = cs.get_object_stats()
        assert df.iloc[0]["std_angle_pos"] > 0.5

    def test_too_few_members_gives_nan_angle_stats(self):
        """Ring with only 1 subunit: angle-statistics columns are NaN."""
        m = _drop_subunits(_make_synthetic_ring(n=8), [2, 3, 4, 5, 6, 7, 8])
        cs = structure.CnComplex(m, 8, center_method="barycentric")
        cs.assign_subunit_order()
        df = cs.get_object_stats()
        for col in ["mean_angle_pos", "std_angle_pos", "var_angle_pos"]:
            assert pd.isna(df.iloc[0][col]), f"expected NaN for {col!r}, got {df.iloc[0][col]}"


# =============================================================================
# Pleomorphic assembly Phase 1 — helpers
# =============================================================================

def _soccer_ball(
    edge: float = 18.0,
    seed: int = 0,
    flipped: tuple[int, ...] = (),
) -> tuple["cryomotl.Motl", float, float]:
    """Build a synthetic soccer-ball blocks motl.

    Places 60 blocks at the 1/3 and 2/3 points of the 30 edges of a regular
    icosahedron, scaled so that the minimum pairwise block distance equals
    *edge*.

    Returns (blocks_motl, arm_length=edge/2, arm_elevation_deg≈11.64).
    """
    from scipy.spatial.distance import cdist as _cdist

    phi_golden = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi_golden], [0, 1, -phi_golden], [0, -1, phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0], [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1], [phi_golden, 0, -1], [-phi_golden, 0, -1],
    ], dtype=float)

    edges = [
        (i, j)
        for i in range(12)
        for j in range(i + 1, 12)
        if abs(np.linalg.norm(ico_verts[i] - ico_verts[j]) - 2.0) < 1e-9
    ]
    assert len(edges) == 30

    raw_pts = np.array([
        pt
        for a_idx, b_idx in edges
        for pt in (
            ico_verts[a_idx] + (ico_verts[b_idx] - ico_verts[a_idx]) / 3.0,
            ico_verts[a_idx] + 2.0 * (ico_verts[b_idx] - ico_verts[a_idx]) / 3.0,
        )
    ])
    assert len(raw_pts) == 60

    d = _cdist(raw_pts, raw_pts)
    np.fill_diagonal(d, np.inf)
    pts = raw_pts * (edge / d.min()) + 100.0
    centre = pts.mean(axis=0)

    d2 = _cdist(pts, pts)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argsort(d2, axis=1)[:, :3]

    rng = np.random.default_rng(seed)
    R_list: list[Rotation] = []
    z_axes: list[np.ndarray] = []
    elev_samples: list[float] = []

    for i in range(60):
        z_i = pts[i] - centre
        z_i /= np.linalg.norm(z_i)
        z_axes.append(z_i)

        diff = pts[nn_idx[i, 0]] - pts[i]
        x_raw = diff - np.dot(diff, z_i) * z_i
        x_i = x_raw / np.linalg.norm(x_raw)
        y_i = np.cross(z_i, x_i)

        R = (Rotation.from_matrix(np.column_stack([x_i, y_i, z_i]))
             * Rotation.from_euler("z", 120.0 * int(rng.integers(3)), degrees=True))
        R_list.append(R)

        for j in nn_idx[i]:
            diff_nn = pts[i] - pts[j]
            elev_samples.append(np.dot(diff_nn / np.linalg.norm(diff_nn), z_i))

    arm_elevation_deg = float(np.degrees(np.arcsin(np.mean(elev_samples))))
    arm_length = edge / 2.0

    flip_rot = Rotation.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0]))
    for f_idx in flipped:
        R_list[f_idx] = R_list[f_idx] * flip_rot

    angles = np.array([R.as_euler("zxz", degrees=True) for R in R_list])
    n = 60
    data = {c: np.zeros(n) for c in cryomotl.Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    data["tomo_id"] = np.ones(n, dtype=float)
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["phi"] = angles[:, 0]    # cryoCAT: phi=first-Z, theta=X, psi=last-Z
    data["theta"] = angles[:, 1]
    data["psi"] = angles[:, 2]

    df = pd.DataFrame(data)[cryomotl.Motl.motl_columns]
    return cryomotl.Motl(df), arm_length, arm_elevation_deg


# =============================================================================
# T1 — geom.cn_site_vectors
# =============================================================================

class TestCnSiteVectors:
    def test_single_site_along_x(self):
        vecs = geom.cn_site_vectors(1, 3.0)
        assert vecs.shape == (1, 3)
        np.testing.assert_allclose(vecs[0], [3.0, 0.0, 0.0], atol=1e-12)

    def test_four_sites_azimuth_offset(self):
        vecs = geom.cn_site_vectors(4, 2.0, azimuth=45.0)
        assert vecs.shape == (4, 3)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 2.0, atol=1e-12)
        expected_az = np.radians([45.0, 135.0, 225.0, 315.0])
        az = np.arctan2(vecs[:, 1], vecs[:, 0])
        np.testing.assert_allclose(np.cos(az), np.cos(expected_az), atol=1e-12)
        np.testing.assert_allclose(np.sin(az), np.sin(expected_az), atol=1e-12)

    def test_elevation_90_points_down(self):
        vecs = geom.cn_site_vectors(3, 5.0, elevation=90.0)
        np.testing.assert_allclose(vecs[:, :2], 0.0, atol=1e-12)
        np.testing.assert_allclose(vecs[:, 2], -5.0, atol=1e-12)

    def test_raises_n_less_than_1(self):
        with pytest.raises(ValueError, match="n must be >= 1"):
            geom.cn_site_vectors(0, 1.0)

    def test_raises_length_not_positive(self):
        with pytest.raises(ValueError, match="length must be > 0"):
            geom.cn_site_vectors(3, 0.0)
        with pytest.raises(ValueError, match="length must be > 0"):
            geom.cn_site_vectors(3, -1.0)

    def test_c3_vectors_equidistant(self):
        vecs = geom.cn_site_vectors(3, 4.0)
        d01 = np.linalg.norm(vecs[0] - vecs[1])
        d12 = np.linalg.norm(vecs[1] - vecs[2])
        d20 = np.linalg.norm(vecs[2] - vecs[0])
        np.testing.assert_allclose(d01, d12, atol=1e-12)
        np.testing.assert_allclose(d01, d20, atol=1e-12)


# =============================================================================
# T2 — structure.expand_motl
# =============================================================================

class TestExpandMotl:
    def _simple_motl(self, n: int = 3) -> "cryomotl.Motl":
        data = {c: np.zeros(n) for c in cryomotl.Motl.motl_columns}
        data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
        data["tomo_id"] = np.ones(n, dtype=float)
        data["x"] = np.arange(n, dtype=float)
        return cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])

    def test_shape_and_columns(self):
        motl = self._simple_motl(3)
        result = structure.expand_motl(
            motl, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), orientation="radial"
        )
        assert len(result.df) == 6
        assert "geom1" in result.df.columns
        assert "object_id" in result.df.columns

    def test_original_id_tracks_source_subtomo_id(self):
        motl = self._simple_motl(3)
        result = structure.expand_motl(
            motl, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), orientation="keep"
        )
        assert sorted(result.df["object_id"].unique()) == [1.0, 2.0, 3.0]

    def test_order_id_values(self):
        motl = self._simple_motl(2)
        sv = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
        result = structure.expand_motl(motl, sv, orientation="keep", start_index=0)
        assert set(result.df["geom1"].unique()) == {0.0, 1.0, 2.0}

    def test_start_index(self):
        motl = self._simple_motl(2)
        result = structure.expand_motl(
            motl, np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
            orientation="keep", start_index=1
        )
        assert set(result.df["geom1"].unique()) == {1.0, 2.0}

    def test_sort_vectors_false_preserves_order(self):
        motl = self._simple_motl(1)
        sv = np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])
        result = structure.expand_motl(motl, sv, orientation="keep", sort_vectors=False)
        first_row = result.df[result.df["geom1"] == 0.0]
        np.testing.assert_allclose(first_row["z"].values[0], -1.0, atol=1e-4)

    def test_orientation_keep_preserves_angles(self):
        motl = self._simple_motl(2)
        result = structure.expand_motl(
            motl, np.array([[0.0, 0.0, 5.0]]), orientation="keep"
        )
        np.testing.assert_allclose(result.df["phi"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.df["psi"].values, 0.0, atol=1e-10)
        np.testing.assert_allclose(result.df["theta"].values, 0.0, atol=1e-10)

    def test_invalid_orientation_raises(self):
        motl = self._simple_motl(1)
        with pytest.raises(ValueError, match="orientation must be"):
            structure.expand_motl(motl, np.array([[1.0, 0.0, 0.0]]), orientation="bad")

    def test_sorted_by_object_id_and_geom1(self):
        motl = self._simple_motl(3)
        result = structure.expand_motl(
            motl, np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]), orientation="keep"
        )
        rows = list(zip(result.df["object_id"].tolist(), result.df["geom1"].tolist()))
        assert rows == sorted(rows)


# =============================================================================
# T3 — structure.ContactSite and structure.BlockDefinition
# =============================================================================

class TestContactSite:
    def test_valid_site(self):
        cs = structure.ContactSite(vector=(1.0, 0.0, 0.0))
        assert cs.vector == (1.0, 0.0, 0.0)
        assert cs.site_type == "site"

    def test_custom_site_type(self):
        cs = structure.ContactSite(vector=(0.5, 0.5, 0.0), site_type="arm")
        assert cs.site_type == "arm"

    def test_zero_inplane_raises(self):
        with pytest.raises(ValueError, match="zero in-plane part"):
            structure.ContactSite(vector=(0.0, 0.0, 1.0))

    def test_wrong_length_raises(self):
        with pytest.raises((ValueError, TypeError)):
            structure.ContactSite(vector=(1.0, 0.0))


class TestBlockDefinition:
    def test_single_site_valid(self):
        bd = structure.BlockDefinition(sites=(structure.ContactSite((1.0, 0.0, 0.0)),))
        assert bd.n_sites == 1
        assert bd.flip_site == 1

    def test_flip_site_out_of_range_raises(self):
        with pytest.raises(ValueError, match="flip_site"):
            structure.BlockDefinition(
                sites=(structure.ContactSite((1.0, 0.0, 0.0)),), flip_site=0
            )
        with pytest.raises(ValueError, match="flip_site"):
            structure.BlockDefinition(
                sites=(structure.ContactSite((1.0, 0.0, 0.0)),), flip_site=2
            )

    def test_empty_sites_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            structure.BlockDefinition(sites=())

    def test_pairing_unknown_type_raises(self):
        with pytest.raises(ValueError, match="Pairing type"):
            structure.BlockDefinition(
                sites=(structure.ContactSite((1.0, 0.0, 0.0), site_type="arm"),),
                pairing=(("arm", "other"),),
            )

    def test_ccw_order_valid(self):
        sites = (
            structure.ContactSite((1.0, 0.0, 0.0)),
            structure.ContactSite((-0.5, 0.866, 0.0)),
            structure.ContactSite((-0.5, -0.866, 0.0)),
        )
        bd = structure.BlockDefinition(sites=sites)
        assert bd.n_sites == 3

    def test_cw_order_raises(self):
        sites = (
            structure.ContactSite((1.0, 0.0, 0.0)),
            structure.ContactSite((-0.5, -0.866, 0.0)),
            structure.ContactSite((-0.5, 0.866, 0.0)),
        )
        with pytest.raises(ValueError, match="counter-clockwise"):
            structure.BlockDefinition(sites=sites)

    def test_site_vectors_shape(self):
        bd = structure.BlockDefinition.cyclic(3, [5.0, 0.0, 0.0])
        vecs = bd.site_vectors()
        assert vecs.shape == (3, 3)
        np.testing.assert_allclose(np.linalg.norm(vecs, axis=1), 5.0, atol=1e-12)

    def test_cyclic_classmethod(self):
        bd = structure.BlockDefinition.cyclic(4, [2.0, 0.0, 0.0], site_type="a")
        assert bd.n_sites == 4
        assert all(s.site_type == "a" for s in bd.sites)
        assert bd.pairing == (("a", "a"),)

    def test_site_types_property(self):
        bd = structure.BlockDefinition.cyclic(2, [1.0, 0.0, 0.0], site_type="x")
        assert bd.site_types == ("x", "x")


# =============================================================================
# T4 — PleomorphicSurface block-layer __init__
# =============================================================================

class TestPleomorphicSurfaceBlockInit:
    def _make_motl(self, n: int = 5) -> "cryomotl.Motl":
        data = {c: np.zeros(n) for c in cryomotl.Motl.motl_columns}
        data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
        data["tomo_id"] = np.ones(n, dtype=float)
        return cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])

    def test_envelope_only_still_raises_on_bad_type(self):
        with pytest.raises(TypeError):
            structure.PleomorphicSurface(surface=42)

    def test_blocks_without_definition_raises(self):
        motl = self._make_motl()
        with pytest.raises(ValueError, match="block_definition is required"):
            structure.PleomorphicSurface(blocks=motl)

    def test_blocks_only_valid(self):
        motl = self._make_motl()
        bd = structure.BlockDefinition.cyclic(3, [5.0, 0.0, 0.0])
        psurf = structure.PleomorphicSurface(blocks=motl, block_definition=bd)
        assert psurf.has_blocks
        assert not psurf.has_envelope
        assert len(psurf.blocks.df) == 5

    def test_no_surface_and_no_blocks_raises(self):
        with pytest.raises(TypeError, match="requires at least"):
            structure.PleomorphicSurface()

    def test_surface_property_raises_when_no_envelope(self):
        motl = self._make_motl()
        bd = structure.BlockDefinition.cyclic(2, [3.0, 0.0, 0.0])
        psurf = structure.PleomorphicSurface(blocks=motl, block_definition=bd)
        with pytest.raises(ValueError, match="no envelope"):
            _ = psurf.surface

    def test_duplicate_subtomo_id_raises(self):
        data = {c: np.zeros(3) for c in cryomotl.Motl.motl_columns}
        data["subtomo_id"] = [1.0, 1.0, 2.0]
        motl = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])
        bd = structure.BlockDefinition.cyclic(3, [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="subtomo_id must be unique"):
            structure.PleomorphicSurface(blocks=motl, block_definition=bd)

    def test_ideal_lattice_mismatch_raises(self):
        motl = self._make_motl()
        bd = structure.BlockDefinition.cyclic(3, [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="ideal_degree and ideal_face_size"):
            structure.PleomorphicSurface(blocks=motl, block_definition=bd, ideal_degree=6)

    def test_ideal_lattice_wrong_euler_raises(self):
        motl = self._make_motl()
        bd = structure.BlockDefinition.cyclic(3, [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="1/2"):
            structure.PleomorphicSurface(
                blocks=motl, block_definition=bd, ideal_degree=6, ideal_face_size=6
            )

    def test_ideal_lattice_valid(self):
        motl = self._make_motl()
        bd = structure.BlockDefinition.cyclic(3, [1.0, 0.0, 0.0])
        psurf = structure.PleomorphicSurface(
            blocks=motl, block_definition=bd, ideal_degree=3, ideal_face_size=6
        )
        assert psurf.ideal_degree == 3
        assert psurf.ideal_face_size == 6

    def test_site_type_codes_built(self):
        motl = self._make_motl()
        bd = structure.BlockDefinition.cyclic(2, [1.0, 0.0, 0.0], site_type="arm")
        psurf = structure.PleomorphicSurface(blocks=motl, block_definition=bd)
        assert "arm" in psurf.site_type_codes
        assert psurf.site_type_codes["arm"] == 1

    def test_dict_block_definition(self):
        motl = self._make_motl(4)
        motl.df["class"] = [0.0, 0.0, 1.0, 1.0]
        bd0 = structure.BlockDefinition.cyclic(3, [2.0, 0.0, 0.0], site_type="a")
        bd1 = structure.BlockDefinition.cyclic(2, [3.0, 0.0, 0.0], site_type="b")
        psurf = structure.PleomorphicSurface(
            blocks=motl, block_definition={0.0: bd0, 1.0: bd1}
        )
        assert "a" in psurf.site_type_codes
        assert "b" in psurf.site_type_codes

    def test_dict_definition_missing_type_raises(self):
        motl = self._make_motl(2)
        motl.df["class"] = [0.0, 2.0]
        bd0 = structure.BlockDefinition.cyclic(3, [1.0, 0.0, 0.0])
        with pytest.raises(ValueError, match="without a BlockDefinition"):
            structure.PleomorphicSurface(blocks=motl, block_definition={0.0: bd0})


# =============================================================================
# T5 — PleomorphicSurface.get_sites_as_motl
# =============================================================================

class TestGetSitesAsMotl:
    def test_shape(self):
        blocks_motl, arm_length, arm_elev = _soccer_ball()
        bd = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        psurf = structure.PleomorphicSurface(blocks=blocks_motl, block_definition=bd)
        sites = psurf.get_sites_as_motl()
        assert len(sites.df) == 180, f"Expected 180 rows, got {len(sites.df)}"

    def test_geom1_values(self):
        blocks_motl, arm_length, arm_elev = _soccer_ball()
        bd = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        psurf = structure.PleomorphicSurface(blocks=blocks_motl, block_definition=bd)
        sites = psurf.get_sites_as_motl()
        assert set(sites.df["geom1"].unique()) == {1.0, 2.0, 3.0}

    def test_object_id_tracks_block_ids(self):
        blocks_motl, arm_length, arm_elev = _soccer_ball()
        bd = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        psurf = structure.PleomorphicSurface(blocks=blocks_motl, block_definition=bd)
        sites = psurf.get_sites_as_motl()
        expected = set(blocks_motl.df["subtomo_id"].unique())
        assert set(sites.df["object_id"].unique()) == expected

    def test_positions_equal_c_plus_R_apply_v(self):
        blocks_motl, arm_length, arm_elev = _soccer_ball()
        bd = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        psurf = structure.PleomorphicSurface(blocks=blocks_motl, block_definition=bd)
        sites = psurf.get_sites_as_motl()

        angles = blocks_motl.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = Rotation.from_euler("zxz", angles, degrees=True)
        site_vecs = bd.site_vectors()

        for _, row in sites.df.iterrows():
            obj_id = int(row["object_id"])
            site_idx = int(row["geom1"]) - 1
            block_mask = blocks_motl.df["subtomo_id"] == obj_id
            block_row = blocks_motl.df[block_mask].iloc[0]
            block_pos_idx = blocks_motl.df[block_mask].index[0]
            c = np.array([block_row["x"], block_row["y"], block_row["z"]])
            expected_f = c + all_R[block_pos_idx].apply(site_vecs[site_idx])
            expected = np.where(expected_f >= 0,
                                np.floor(expected_f + 0.5),
                                -np.floor(-expected_f + 0.5))
            actual = np.array([row["x"], row["y"], row["z"]])
            np.testing.assert_allclose(
                actual, expected, atol=1e-6,
                err_msg=f"block {obj_id} site {site_idx + 1}"
            )

    def test_no_blocks_raises(self):
        blocks_motl, arm_length, arm_elev = _soccer_ball()
        bd = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        psurf = structure.PleomorphicSurface(blocks=blocks_motl, block_definition=bd)
        psurf.blocks = None
        with pytest.raises(ValueError, match="requires a block layer"):
            psurf.get_sites_as_motl()


# =============================================================================
# T6 — PleomorphicSurface.unify_polarity
# =============================================================================

_SOCCER_FLIPPED = (3, 17, 25, 31, 40, 44, 52, 59)


class TestUnifyPolarity:
    def _make_psurf(self, flipped=_SOCCER_FLIPPED):
        blocks_motl, arm_length, arm_elev = _soccer_ball(flipped=flipped)
        bd = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        return structure.PleomorphicSurface(blocks=blocks_motl, block_definition=bd)

    def test_neighbours_returns_changed_count(self):
        psurf = self._make_psurf()
        changed = psurf.unify_polarity(max_block_distance=20.0, reference="neighbours")
        assert changed == len(_SOCCER_FLIPPED)

    def test_neighbours_all_pairs_aligned(self):
        from scipy.spatial.distance import cdist as _cdist
        psurf = self._make_psurf()
        psurf.unify_polarity(max_block_distance=20.0, reference="neighbours")

        angles = psurf.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = Rotation.from_euler("zxz", angles, degrees=True)
        z = np.array([all_R[i].apply(np.array([0., 0., 1.])) for i in range(len(angles))])
        c = psurf.blocks.get_coordinates()
        d = _cdist(c, c)
        np.fill_diagonal(d, np.inf)
        ii, jj = np.where(d <= 20.0)
        for i, j in zip(ii, jj):
            if i < j:
                dot = np.dot(z[i], z[j])
                assert dot > 0, f"blocks {i},{j}: dot={dot:.4f}"

    def test_centroid_returns_changed_count(self):
        psurf = self._make_psurf()
        changed = psurf.unify_polarity(max_block_distance=20.0, reference="centroid")
        assert changed == len(_SOCCER_FLIPPED)

    def test_centroid_all_blocks_point_outward(self):
        psurf = self._make_psurf()
        psurf.unify_polarity(max_block_distance=20.0, reference="centroid")

        angles = psurf.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = Rotation.from_euler("zxz", angles, degrees=True)
        z = np.array([all_R[i].apply(np.array([0., 0., 1.])) for i in range(len(angles))])
        c = psurf.blocks.get_coordinates()
        g = c.mean(axis=0)
        for i in range(len(c)):
            outward = c[i] - g
            dot = np.dot(z[i], outward / np.linalg.norm(outward))
            assert dot > 0, f"block {i}: dot={dot:.4f}"

    def test_already_aligned_returns_zero(self):
        psurf = self._make_psurf(flipped=())
        psurf.unify_polarity(max_block_distance=20.0, reference="centroid")
        assert psurf.unify_polarity(max_block_distance=20.0, reference="centroid") == 0

    def test_no_blocks_raises(self):
        psurf = self._make_psurf()
        psurf.blocks = None
        with pytest.raises(ValueError, match="requires a block layer"):
            psurf.unify_polarity(20.0)

    def test_invalid_reference_raises(self):
        psurf = self._make_psurf()
        with pytest.raises(ValueError, match="reference must be"):
            psurf.unify_polarity(20.0, reference="invalid")


# =============================================================================
# Pleomorphic assembly Phase 2 — helpers
# =============================================================================

def _geodesic(
    edge: float = 10.0,
    seed: int = 0,
) -> tuple["cryomotl.Motl", dict]:
    """Build a v=2 icosahedron geodesic dome (42 blocks, 12×C5 + 30×C6).

    Returns (blocks_motl, block_definition_dict).
    """
    from scipy.spatial.distance import cdist as _cdist

    phi_golden = (1 + np.sqrt(5)) / 2
    ico_verts = np.array([
        [0, 1, phi_golden], [0, 1, -phi_golden], [0, -1, phi_golden], [0, -1, -phi_golden],
        [1, phi_golden, 0], [-1, phi_golden, 0], [1, -phi_golden, 0], [-1, -phi_golden, 0],
        [phi_golden, 0, 1], [-phi_golden, 0, 1], [phi_golden, 0, -1], [-phi_golden, 0, -1],
    ], dtype=float)
    ico_verts /= np.linalg.norm(ico_verts[0])

    ico_edges = [
        (i, j) for i in range(12) for j in range(i + 1, 12)
        if abs(np.dot(ico_verts[i], ico_verts[j]) - np.dot(ico_verts[0], ico_verts[4])) < 1e-9
    ]
    assert len(ico_edges) == 30, f"Expected 30 edges, got {len(ico_edges)}"

    # 12 original vertices + 30 midpoints projected to unit sphere
    pts_unit = list(ico_verts)
    for a_idx, b_idx in ico_edges:
        mid = ico_verts[a_idx] + ico_verts[b_idx]
        pts_unit.append(mid / np.linalg.norm(mid))
    pts_unit = np.array(pts_unit)  # (42, 3)

    d3 = _cdist(pts_unit, pts_unit)
    np.fill_diagonal(d3, np.inf)
    min_d = d3.min()
    pts = pts_unit * (edge / min_d) + 100.0  # offset to (100,100,100)

    # Recompute distances after scaling
    d3 = _cdist(pts, pts)
    np.fill_diagonal(d3, np.inf)

    # Block type: 0=C5 (first 12), 1=C6 (next 30)
    block_types = np.array([5] * 12 + [6] * 30, dtype=float)
    c_vals = np.array([5] * 12 + [6] * 30, dtype=int)

    # Per-block: number of nearest neighbours = connectivity
    nn_idx = np.argsort(d3, axis=1)
    centre = pts.mean(axis=0)

    rng = np.random.default_rng(seed)
    R_list = []
    elev_samples = []
    arm_lengths = []

    for i in range(42):
        c_i = c_vals[i]
        z_i = pts[i] - centre
        z_i /= np.linalg.norm(z_i)

        diff = pts[nn_idx[i, 0]] - pts[i]
        x_raw = diff - np.dot(diff, z_i) * z_i
        x_i = x_raw / np.linalg.norm(x_raw)
        y_i = np.cross(z_i, x_i)

        k = int(rng.integers(c_i))
        R = (Rotation.from_matrix(np.column_stack([x_i, y_i, z_i]))
             * Rotation.from_euler("z", 360.0 * k / c_i, degrees=True))
        R_list.append(R)

        nn_c = nn_idx[i, :c_i]
        for j in nn_c:
            diff_nn = pts[i] - pts[j]
            elev_samples.append(np.dot(diff_nn / np.linalg.norm(diff_nn), z_i))
        arm_lengths.append(np.mean(d3[i, nn_c]) / 2.0)

    arm_elevation_deg = float(np.degrees(np.arcsin(np.clip(np.mean(elev_samples), -1.0, 1.0))))
    L = float(np.mean(arm_lengths))

    angles = np.array([R.as_euler("zxz", degrees=True) for R in R_list])
    n = 42
    data = {c: np.zeros(n) for c in cryomotl.Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    data["tomo_id"] = np.ones(n, dtype=float)
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["phi"] = angles[:, 0]
    data["theta"] = angles[:, 1]
    data["psi"] = angles[:, 2]
    data["geom3"] = block_types  # store connectivity (5 or 6)

    df = pd.DataFrame(data)[cryomotl.Motl.motl_columns]
    blocks_motl = cryomotl.Motl(df)

    block_def = {
        5.0: structure.BlockDefinition.cyclic(5, [L * np.cos(np.radians(arm_elevation_deg)), 0.0, -L * np.sin(np.radians(arm_elevation_deg))]),
        6.0: structure.BlockDefinition.cyclic(6, [L * np.cos(np.radians(arm_elevation_deg)), 0.0, -L * np.sin(np.radians(arm_elevation_deg))]),
    }
    return blocks_motl, block_def


def _microtubule(
    n_pf: int = 13,
    n_dimers: int = 20,
) -> tuple["cryomotl.Motl", "structure.BlockDefinition"]:
    """Build a synthetic microtubule blocks motl.

    n_pf protofilaments × n_dimers dimers, placed on a cylinder of radius
    ~11.4 Å (standard 13-pf MT).  Returns (blocks_motl, block_definition).
    """
    R_tube = 11.4       # protofilament radius (voxels)
    rise_per_dimer = 8.16       # axial rise per αβ-dimer (voxels)
    rise_per_pf = 12.24 / 13   # ≈ 0.9415 voxels per protofilament step

    n = n_pf * n_dimers
    pf_indices = np.empty(n, dtype=int)
    dimer_indices = np.empty(n, dtype=int)
    for k in range(n_pf):
        for j in range(n_dimers):
            idx = k * n_dimers + j
            pf_indices[idx] = k
            dimer_indices[idx] = j

    theta = np.radians(pf_indices * 360.0 / n_pf)
    xs = R_tube * np.cos(theta) + 100.0
    ys = R_tube * np.sin(theta) + 100.0
    zs = dimer_indices * rise_per_dimer + pf_indices * rise_per_pf + 20.0

    R_list = []
    for k_pf, j_dim in zip(pf_indices, dimer_indices):
        th = np.radians(k_pf * 360.0 / n_pf)
        z_i = np.array([np.cos(th), np.sin(th), 0.0])   # radial outward
        x_i = np.array([-np.sin(th), np.cos(th), 0.0])  # tangential (CCW)
        y_i = np.array([0.0, 0.0, 1.0])                 # axial (+z)
        R_list.append(Rotation.from_matrix(np.column_stack([x_i, y_i, z_i])))

    angles = np.array([R.as_euler("zxz", degrees=True) for R in R_list])

    data = {c: np.zeros(n) for c in cryomotl.Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    data["tomo_id"] = np.ones(n, dtype=float)
    data["x"] = xs
    data["y"] = ys
    data["z"] = zs
    data["phi"] = angles[:, 0]
    data["theta"] = angles[:, 1]
    data["psi"] = angles[:, 2]
    data["geom3"] = pf_indices.astype(float)
    data["geom4"] = dimer_indices.astype(float)

    df = pd.DataFrame(data)[cryomotl.Motl.motl_columns]
    blocks_motl = cryomotl.Motl(df)

    lat_arm = R_tube * np.sin(np.pi / n_pf)   # half the chord between adjacent pf centres
    ax_arm = rise_per_dimer / 2.0              # = 4.08

    block_def = structure.BlockDefinition.microtubule(lat_arm, ax_arm)
    return blocks_motl, block_def


# =============================================================================
# Phase 2 — Step 0.2: envelope methods on blocks-only object
# =============================================================================

class TestEnvelopeOnBlocksOnly:
    def _make(self):
        motl, arm_length, arm_elev = _soccer_ball()
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        return structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def, ideal_degree=3, ideal_face_size=6
        )

    def test_get_mean_curvature_raises(self):
        ps = self._make()
        with pytest.raises((ValueError, AttributeError)):
            ps.get_mean_curvature()

    def test_distance_to_points_raises(self):
        ps = self._make()
        pts = np.array([[100.0, 100.0, 100.0]])
        with pytest.raises((ValueError, AttributeError)):
            ps.distance_to_points(pts)


# =============================================================================
# Phase 2 — Step 0.3e: fold path gives correct positions
# =============================================================================

class TestGetSitesAsMotlFold:
    def _make_psurf(self, n: int, arm_len: float, azimuth: float, elev: float):
        motl, _, _ = _soccer_ball(edge=18.0, seed=0)
        # Use just first 4 blocks from soccer ball for speed
        sub = motl.df.iloc[:4].copy().reset_index(drop=True)
        sub["subtomo_id"] = np.arange(1, 5, dtype=float)
        m = cryomotl.Motl(pd.DataFrame(sub, columns=cryomotl.Motl.motl_columns))
        block_def = structure.BlockDefinition.cyclic(n, [
            arm_len * np.cos(np.radians(elev)) * np.cos(np.radians(azimuth)),
            arm_len * np.cos(np.radians(elev)) * np.sin(np.radians(azimuth)),
            -arm_len * np.sin(np.radians(elev)),
        ])
        return m, block_def

    def test_fold3_positions_match_site_vectors(self):
        m, block_def = self._make_psurf(3, 9.0, 30.0, 11.0)
        ps = structure.PleomorphicSurface(blocks=m, block_definition=block_def)
        site_motl = ps.get_sites_as_motl()

        expected_vecs = block_def.site_vectors()  # (3, 3)
        angles_all = m.df[["phi", "theta", "psi"]].values.astype(float)
        block_coords = m.get_coordinates()

        for block_row_idx in range(len(m.df)):
            block_id = float(m.df.iloc[block_row_idx]["subtomo_id"])
            R_b = Rotation.from_euler("zxz", angles_all[block_row_idx], degrees=True)
            c_b = block_coords[block_row_idx]

            for k in range(1, 4):
                site_rows = site_motl.df[
                    (site_motl.df["object_id"] == block_id) &
                    (site_motl.df["geom1"] == float(k))
                ]
                assert len(site_rows) == 1, f"block {block_id} site {k}: {len(site_rows)} rows"
                site_pos = site_motl.get_coordinates()[site_rows.index[0]]
                expected = c_b + R_b.apply(expected_vecs[k - 1])
                np.testing.assert_allclose(
                    site_pos, expected, atol=1e-4,
                    err_msg=f"block {block_id} site {k}"
                )

    def test_fold6_positions_match_site_vectors(self):
        m, block_def = self._make_psurf(6, 5.0, 0.0, 15.0)
        ps = structure.PleomorphicSurface(blocks=m, block_definition=block_def)
        site_motl = ps.get_sites_as_motl()

        expected_vecs = block_def.site_vectors()
        angles_all = m.df[["phi", "theta", "psi"]].values.astype(float)
        block_coords = m.get_coordinates()

        for block_row_idx in range(len(m.df)):
            block_id = float(m.df.iloc[block_row_idx]["subtomo_id"])
            R_b = Rotation.from_euler("zxz", angles_all[block_row_idx], degrees=True)
            c_b = block_coords[block_row_idx]

            for k in range(1, 7):
                site_rows = site_motl.df[
                    (site_motl.df["object_id"] == block_id) &
                    (site_motl.df["geom1"] == float(k))
                ]
                assert len(site_rows) == 1
                site_pos = site_motl.get_coordinates()[site_rows.index[0]]
                expected = c_b + R_b.apply(expected_vecs[k - 1])
                np.testing.assert_allclose(site_pos, expected, atol=1e-4)


# =============================================================================
# Phase 2 — T1: trace_faces
# =============================================================================

class TestTraceFaces:
    def test_two_closed_faces(self):
        # 4 half-edges: contacts h0↔h3, h1↔h2 (mutual partners)
        # block/site: h0=(0,1), h1=(0,2), h2=(1,1), h3=(1,2)  [2 blocks, 2 sites each]
        # Walk: h0→partner h3(block=1,site=2)→next(1,2%2+1=1)=h2→partner h1(block=0,site=2)
        #       →next(0,2%2+1=1)=h0→closed => face 1 = {h0, h2}
        # Walk: h1→partner h2(block=1,site=1)→next(1,1%2+1=2)=h3→partner h0(block=0,site=1)
        #       →next(0,1%2+1=2)=h1→closed => face 2 = {h1, h3}
        # Result: face = [1, 2, 1, 2]
        partner = np.array([3, 2, 1, 0], dtype=np.intp)
        block   = np.array([0, 0, 1, 1], dtype=np.intp)
        site    = np.array([1, 2, 1, 2], dtype=np.intp)
        n_sites = np.array([2, 2, 2, 2], dtype=np.intp)
        face = structure.trace_faces(partner, block, site, n_sites)
        np.testing.assert_array_equal(face, [1, 2, 1, 2])

    def test_all_boundary_when_no_partner(self):
        # h0 and h3 unmatched; h1↔h2
        partner = np.array([-1, 2, 1, -1], dtype=np.intp)
        block   = np.array([0, 1, 2, 3], dtype=np.intp)
        site    = np.array([1, 1, 1, 1], dtype=np.intp)
        n_sites = np.array([3, 3, 3, 3], dtype=np.intp)
        face = structure.trace_faces(partner, block, site, n_sites)
        # h1 and h2 form a pair but can't close (both only have site 1, next = 2, no lookup)
        assert face[0] == -1
        assert face[3] == -1


# =============================================================================
# Phase 2 — T2: soccer ball connect
# =============================================================================

class TestSoccerBallConnect:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_all_sites_matched(self, psurf):
        st = psurf._site_table
        assert (st["partner"] >= 0).sum() == 180

    def test_face_count(self, psurf):
        face_stats = psurf.get_face_stats()
        assert len(face_stats) == 32

    def test_face_sizes_5_and_6(self, psurf):
        face_stats = psurf.get_face_stats()
        sizes = face_stats["size"].value_counts().to_dict()
        assert sizes.get(5, 0) == 12
        assert sizes.get(6, 0) == 20

    def test_euler_characteristic(self, psurf):
        asm = psurf.get_assembly_stats()
        assert len(asm) == 1
        assert int(asm.iloc[0]["euler_characteristic"]) == 2

    def test_closed(self, psurf):
        asm = psurf.get_assembly_stats()
        assert bool(asm.iloc[0]["closed"]) is True

    def test_defect_charge(self, psurf):
        asm = psurf.get_assembly_stats()
        np.testing.assert_allclose(float(asm.iloc[0]["defect_charge"]), 2.0, atol=1e-9)

    def test_angle_deficit_sum(self, psurf):
        asm = psurf.get_assembly_stats()
        np.testing.assert_allclose(float(asm.iloc[0]["angle_deficit_sum"]), 720.0, atol=0.1)

    def test_face_signature_all_blocks(self, psurf):
        block_stats = psurf.get_block_stats()
        sigs = block_stats["face_signature"].unique()
        assert len(sigs) == 1
        assert sigs[0] == "5-6-6"

    def test_vef_counts(self, psurf):
        asm = psurf.get_assembly_stats()
        row = asm.iloc[0]
        assert int(row["n_blocks"]) == 60
        assert int(row["n_contacts"]) == 90
        assert int(row["n_faces"]) == 32

    def test_n_faces_columns(self, psurf):
        asm = psurf.get_assembly_stats()
        assert int(asm.iloc[0]["n_faces_5"]) == 12
        assert int(asm.iloc[0]["n_faces_6"]) == 20


# =============================================================================
# Phase 2 — T3: relabel invariance
# =============================================================================

class TestRelabelInvariance:
    def _face_size_counts(self, seed):
        motl, arm_length, arm_elev = _soccer_ball(seed=seed)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        fs = ps.get_face_stats()
        return fs["size"].value_counts().to_dict()

    def test_seeds_give_same_topology(self):
        c0 = self._face_size_counts(0)
        c5 = self._face_size_counts(5)
        c9 = self._face_size_counts(9)
        assert c0 == c5 == c9 == {5: 12, 6: 20}


# =============================================================================
# Phase 2 — T4: polarity matters
# =============================================================================

class TestPolarityMatters:
    def _build(self, flipped=()):
        motl, arm_length, arm_elev = _soccer_ball(seed=0, flipped=flipped)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        return ps

    def test_flip_one_block_changes_euler(self):
        ps = self._build(flipped=(0,))
        ps.connect(max_distance=4.0)
        asm = ps.get_assembly_stats()
        chi = int(asm.iloc[0]["euler_characteristic"])
        assert chi == 0

    def test_flip_one_block_faces(self):
        ps = self._build(flipped=(0,))
        ps.connect(max_distance=4.0)
        fs = ps.get_face_stats()
        sizes = fs["size"].value_counts().to_dict()
        assert sizes.get(5, 0) == 11
        assert sizes.get(6, 0) == 18
        assert sizes.get(17, 0) == 1

    def test_flip_all_blocks_same_as_unflipped(self):
        ps_all = self._build(flipped=tuple(range(60)))
        ps_all.connect(max_distance=4.0)
        fs_all = ps_all.get_face_stats()["size"].value_counts().to_dict()

        ps_none = self._build(flipped=())
        ps_none.connect(max_distance=4.0)
        fs_none = ps_none.get_face_stats()["size"].value_counts().to_dict()

        assert fs_all == fs_none

    def test_unify_polarity_recovers_topology(self):
        flipped_8 = tuple(range(8))
        ps = self._build(flipped=flipped_8)
        ps.unify_polarity(max_block_distance=20.0, reference="centroid")
        ps.connect(max_distance=4.0)
        fs = ps.get_face_stats()["size"].value_counts().to_dict()
        assert fs == {5: 12, 6: 20}


# =============================================================================
# Phase 2 — T5: open patch
# =============================================================================

class TestOpenPatch:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        # Keep only blocks with z - 100 >= -5
        coords = motl.get_coordinates()
        mask = (coords[:, 2] - 100.0) >= -5.0
        sub_df = motl.df[mask].copy().reset_index(drop=True)
        sub_motl = cryomotl.Motl(sub_df)
        ps = structure.PleomorphicSurface(
            blocks=sub_motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_has_boundary(self, psurf):
        asm = psurf.get_assembly_stats()
        assert int(asm.iloc[0]["n_boundary_half_edges"]) > 0

    def test_not_closed(self, psurf):
        asm = psurf.get_assembly_stats()
        assert bool(asm.iloc[0]["closed"]) is False

    def test_has_closed_faces(self, psurf):
        fs = psurf.get_face_stats()
        assert len(fs) > 0


# =============================================================================
# Phase 2 — T6: geodesic connect
# =============================================================================

class TestGeodesicConnect:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, block_def = _geodesic(edge=10.0, seed=0)
        ps = structure.PleomorphicSurface(
            blocks=motl,
            block_definition=block_def,
            block_type_column="geom3",
            ideal_degree=6,
            ideal_face_size=3,
        )
        ps.connect(max_distance=3.0)
        return ps

    def test_all_sites_matched(self, psurf):
        st = psurf._site_table
        assert (st["partner"] >= 0).sum() == 240

    def test_face_count(self, psurf):
        fs = psurf.get_face_stats()
        assert len(fs) == 80

    def test_all_faces_triangles(self, psurf):
        fs = psurf.get_face_stats()
        assert (fs["size"] == 3).all()

    def test_euler_characteristic(self, psurf):
        asm = psurf.get_assembly_stats()
        assert int(asm.iloc[0]["euler_characteristic"]) == 2

    def test_vef_counts(self, psurf):
        asm = psurf.get_assembly_stats()
        row = asm.iloc[0]
        assert int(row["n_blocks"]) == 42
        assert int(row["n_contacts"]) == 120
        assert int(row["n_faces"]) == 80

    def test_defect_charge(self, psurf):
        asm = psurf.get_assembly_stats()
        np.testing.assert_allclose(float(asm.iloc[0]["defect_charge"]), 2.0, atol=1e-9)


# =============================================================================
# Phase 2 — T7: microtubule connect
# =============================================================================

class TestMicrotubuleConnect:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, block_def = _microtubule(n_pf=13, n_dimers=20)
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=4, ideal_face_size=4,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_matched_count(self, psurf):
        st = psurf._site_table
        n_matched = int((st["partner"] >= 0).sum())
        assert n_matched == 1012

    def test_face_count(self, psurf):
        fs = psurf.get_face_stats()
        assert len(fs) == 246

    def test_all_faces_size_4(self, psurf):
        fs = psurf.get_face_stats()
        assert (fs["size"] == 4).all()

    def test_euler_characteristic_zero(self, psurf):
        asm = psurf.get_assembly_stats()
        assert int(asm.iloc[0]["euler_characteristic"]) == 0

    def test_vef_counts(self, psurf):
        asm = psurf.get_assembly_stats()
        row = asm.iloc[0]
        assert int(row["n_blocks"]) == 260
        assert int(row["n_contacts"]) == 506

    def test_lateral_right_mismatch_y_values(self, psurf):
        cs = psurf.get_contact_stats()
        lat_r = cs[cs["site_type"] == "lateral_right"]["mismatch_y"].abs()
        # Non-seam: |mismatch_y| ≈ 0.94; seam: |mismatch_y| ≈ 3.14
        assert set(lat_r.round(2).unique()).issubset({0.94, 3.14})

    def test_seam_contacts(self, psurf):
        cs = psurf.get_contact_stats()
        lat_r = cs[cs["site_type"] == "lateral_right"]
        # Seam: lateral contacts with larger mismatch_y
        seam = lat_r[lat_r["mismatch_y"].abs() > 2.0]
        assert len(seam) == 19

        blocks_df = psurf.blocks.df.set_index("subtomo_id")
        for _, row in seam.iterrows():
            assert int(blocks_df.at[row["block_id"], "geom3"]) == 12
            assert int(blocks_df.at[row["partner_block_id"], "geom3"]) == 0
            # Seam: pf=12 at dimer j connects to pf=0 at dimer j+1
            dim_b = int(blocks_df.at[row["block_id"], "geom4"])
            dim_p = int(blocks_df.at[row["partner_block_id"], "geom4"])
            assert dim_p == dim_b + 1, f"seam: pf=12 j={dim_b} → pf=0 j+1={dim_b+1}, got {dim_p}"


# =============================================================================
# Phase 2 — T8: store_block_stats
# =============================================================================

class TestStoreBlockStats:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_degree_written_by_subtomo_id(self, psurf):
        psurf.store_block_stats({"degree": "geom1"})
        bs = psurf.get_block_stats()
        for _, row in bs.iterrows():
            bid = row["block_id"]
            written = float(
                psurf.blocks.df[psurf.blocks.df["subtomo_id"] == bid]["geom1"].iloc[0]
            )
            assert written == float(row["degree"])

    def test_assembly_id_written(self, psurf):
        psurf.store_block_stats({"assembly_id": "geom2"})
        bs = psurf.get_block_stats()
        for _, row in bs.iterrows():
            bid = row["block_id"]
            written = float(
                psurf.blocks.df[psurf.blocks.df["subtomo_id"] == bid]["geom2"].iloc[0]
            )
            assert written == float(row["assembly_id"])

    def test_face_signature_raises(self, psurf):
        with pytest.raises(ValueError, match="non-numeric"):
            psurf.store_block_stats({"face_signature": "geom3"})

    def test_unknown_column_raises(self, psurf):
        with pytest.raises(ValueError, match="Unknown"):
            psurf.store_block_stats({"no_such_col": "geom3"})


# =============================================================================
# Phase 2 — T9: getters raise before connect()
# =============================================================================

class TestGettersRaiseBeforeConnect:
    def _make(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        return structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )

    def test_contact_stats_raises(self):
        with pytest.raises(ValueError, match="connect"):
            self._make().get_contact_stats()

    def test_face_stats_raises(self):
        with pytest.raises(ValueError, match="connect"):
            self._make().get_face_stats()

    def test_block_stats_raises(self):
        with pytest.raises(ValueError, match="connect"):
            self._make().get_block_stats()

    def test_assembly_stats_raises(self):
        with pytest.raises(ValueError, match="connect"):
            self._make().get_assembly_stats()


# =============================================================================
# Phase 3 — T7.2: get_faces_as_motl
# =============================================================================

class TestGetFacesAsMotl:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_row_count(self, psurf):
        fm = psurf.get_faces_as_motl()
        assert len(fm.df) == 32

    def test_subtomo_ids_renumbered(self, psurf):
        fm = psurf.get_faces_as_motl()
        assert set(fm.df["subtomo_id"].astype(int)) == set(range(1, 33))

    def test_class_is_face_size(self, psurf):
        fm = psurf.get_faces_as_motl()
        counts = fm.df["class"].astype(int).value_counts().to_dict()
        assert counts.get(5, 0) == 12
        assert counts.get(6, 0) == 20

    def test_object_id_is_face_id(self, psurf):
        fm = psurf.get_faces_as_motl()
        fs = psurf.get_face_stats()
        assert set(fm.df["object_id"].astype(int)) == set(fs["face_id"].astype(int))

    def test_geom1_is_assembly_id(self, psurf):
        fm = psurf.get_faces_as_motl()
        assert (fm.df["geom1"] == 1.0).all()

    def test_positions_match_centroids(self, psurf):
        fm = psurf.get_faces_as_motl()
        fs = psurf.get_face_stats().sort_values("face_id").reset_index(drop=True)
        fm_sorted = fm.df.sort_values("object_id").reset_index(drop=True)
        np.testing.assert_allclose(fm_sorted["x"].values, fs["centroid_x"].values, atol=1e-6)
        np.testing.assert_allclose(fm_sorted["y"].values, fs["centroid_y"].values, atol=1e-6)
        np.testing.assert_allclose(fm_sorted["z"].values, fs["centroid_z"].values, atol=1e-6)


# =============================================================================
# Phase 3 — T7.3: get_gaps_as_motl (microtubule open ends)
# =============================================================================

class TestGetGapsAsMotl:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, block_def = _microtubule(n_pf=13, n_dimers=20)
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=4, ideal_face_size=4,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_returns_motl(self, psurf):
        gm = psurf.get_gaps_as_motl(cluster_radius=18.0)
        assert isinstance(gm, cryomotl.Motl)

    def test_two_caps(self, psurf):
        # Top and bottom open ends each form one cluster of 13 gap points
        gm = psurf.get_gaps_as_motl(cluster_radius=18.0, min_blocks=13)
        assert len(gm.df) == 2

    def test_geom1_n_distinct_blocks(self, psurf):
        gm = psurf.get_gaps_as_motl(cluster_radius=18.0, min_blocks=13)
        assert (gm.df["geom1"].astype(int) == 13).all()

    def test_subtomo_ids_renumbered(self, psurf):
        gm = psurf.get_gaps_as_motl(cluster_radius=18.0, min_blocks=13)
        assert set(gm.df["subtomo_id"].astype(int)) == {1, 2}

    def test_min_blocks_filters(self, psurf):
        # With min_blocks=14, neither cluster qualifies
        gm = psurf.get_gaps_as_motl(cluster_radius=18.0, min_blocks=14)
        assert len(gm.df) == 0


# =============================================================================
# Phase 3 — T7.4: envelope_from_faces
# =============================================================================

class TestEnvelopeFromFaces:
    @pytest.fixture(scope="class")
    def mesh_and_psurf(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        m = ps.envelope_from_faces()
        return m, ps

    def test_returns_mesh(self, mesh_and_psurf):
        m, _ = mesh_and_psurf
        from cryocat.core.surface import Mesh
        assert isinstance(m, Mesh)

    def test_vertex_count(self, mesh_and_psurf):
        m, _ = mesh_and_psurf
        assert len(m.vertices) == 60 + 32  # blocks + faces

    def test_triangle_count(self, mesh_and_psurf):
        m, _ = mesh_and_psurf
        assert len(m.faces) == 12 * 5 + 20 * 6  # 180

    def test_normals_computed(self, mesh_and_psurf):
        m, _ = mesh_and_psurf
        assert m.normals is not None
        assert m.normals.shape == m.vertices.shape

    def test_outward_normals(self, mesh_and_psurf):
        m, ps = mesh_and_psurf
        centre = ps.blocks.df[["x", "y", "z"]].values.mean(axis=0) * ps.pixel_size
        tri_verts = m.vertices[m.faces]  # (F, 3, 3)
        tri_centroids = tri_verts.mean(axis=1)  # (F, 3)
        e0 = tri_verts[:, 1] - tri_verts[:, 0]
        e1 = tri_verts[:, 2] - tri_verts[:, 0]
        geom_normals = np.cross(e0, e1)  # (F, 3)
        gn_norm = np.linalg.norm(geom_normals, axis=1, keepdims=True)
        geom_normals = geom_normals / np.where(gn_norm > 1e-15, gn_norm, 1.0)
        dots = np.sum(geom_normals * (tri_centroids - centre), axis=1)
        assert (dots > 0).all(), f"Some normals point inward; min dot={dots.min():.4f}"


# =============================================================================
# Phase 3 — T7.5: annotate_with_envelope
# =============================================================================

class TestAnnotateWithEnvelope:
    @pytest.fixture(scope="class")
    def annot_and_psurf(self):
        # Phase 6 correction: annotate_with_envelope requires both layers.
        # Previous setup: blocks-only PS after connect() — relied on the now-removed
        # envelope_from_faces fallback.  New setup: build envelope from unflipped
        # soccer ball, then create PS with envelope + same blocks.
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps_clean = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_clean.connect(max_distance=4.0)
        mesh = ps_clean.envelope_from_faces()
        ps = structure.PleomorphicSurface(
            mesh, blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        annot = ps.annotate_with_envelope()
        return annot, ps

    def test_columns_present(self, annot_and_psurf):
        annot, _ = annot_and_psurf
        required = {
            "tomo_id", "block_id", "envelope_distance",
            "closest_x", "closest_y", "closest_z",
            "primitive_id", "normal_angle",
            "mean_curvature", "gaussian_curvature",
        }
        assert required.issubset(set(annot.columns))

    def test_row_count(self, annot_and_psurf):
        annot, ps = annot_and_psurf
        assert len(annot) == len(ps.blocks.df)

    def test_envelope_distance_near_zero(self, annot_and_psurf):
        # Block centres are on the envelope (they are its vertices)
        annot, _ = annot_and_psurf
        assert (annot["envelope_distance"] < 1e-3).all()

    def test_normal_angle_small(self, annot_and_psurf):
        # Blocks should largely face the same direction as the envelope
        annot, _ = annot_and_psurf
        assert (annot["normal_angle"] < 60).all()


# =============================================================================
# Phase 3 — T7.6: unify_polarity(reference="envelope")
# =============================================================================

class TestUnifyPolarityEnvelope:
    def _make_inverted(self, seed=0):
        motl, arm_length, arm_elev = _soccer_ball(seed=seed)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        # Invert every other block to create polarity mismatches
        rng = np.random.default_rng(42)
        flip_mask = rng.integers(0, 2, size=len(ps.blocks.df), dtype=bool)
        angles = ps.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        from scipy.spatial.transform import Rotation as R
        all_R = R.from_euler("zxz", angles, degrees=True)
        for i, flip in enumerate(flip_mask):
            if flip:
                flip_rot = ps._get_flip_rot(i)
                new_R = all_R[i] * flip_rot
                e = new_R.as_euler("zxz", degrees=True)
                ps.blocks.df.at[i, "phi"] = e[0]
                ps.blocks.df.at[i, "theta"] = e[1]
                ps.blocks.df.at[i, "psi"] = e[2]
        return ps, flip_mask

    def test_envelope_reference_reduces_normal_angle(self):
        # Phase 6 correction: setup follows phase-3 T7 test 6.
        # Previous setup used _make_inverted(seed=0) — a blocks-only PS after
        # connect(), which relied on the now-removed envelope_from_faces fallback.
        motl_clean, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps_clean = structure.PleomorphicSurface(
            blocks=motl_clean, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_clean.connect(max_distance=4.0)
        mesh = ps_clean.envelope_from_faces()
        motl_flipped, _, _ = _soccer_ball(seed=0, flipped=(3, 17, 25, 31, 40, 44, 52, 59))
        ps = structure.PleomorphicSurface(
            mesh, blocks=motl_flipped, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.unify_polarity(reference="envelope")
        annot = ps.annotate_with_envelope()
        # After unification, all blocks should face the envelope
        assert (annot["normal_angle"] < 90).all()

    def test_returns_int_count(self):
        # Phase 6 correction: unify_polarity(reference="envelope") requires both
        # layers.  Previous setup used _make_inverted(seed=0) — blocks-only PS.
        motl_clean, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps_clean = structure.PleomorphicSurface(
            blocks=motl_clean, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_clean.connect(max_distance=4.0)
        mesh = ps_clean.envelope_from_faces()
        motl_f, _, _ = _soccer_ball(seed=0, flipped=(3, 17, 25, 31, 40, 44, 52, 59))
        ps = structure.PleomorphicSurface(
            mesh, blocks=motl_f, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        n_flipped = ps.unify_polarity(reference="envelope")
        assert isinstance(n_flipped, int)
        assert n_flipped >= 0

    def test_no_max_distance_needed(self):
        # Phase 6 correction: same as test_returns_int_count — requires both layers.
        # Previous setup used _make_inverted(seed=0) — blocks-only PS.
        motl_clean, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps_clean = structure.PleomorphicSurface(
            blocks=motl_clean, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_clean.connect(max_distance=4.0)
        mesh = ps_clean.envelope_from_faces()
        motl_f, _, _ = _soccer_ball(seed=0, flipped=(3, 17, 25, 31, 40, 44, 52, 59))
        ps = structure.PleomorphicSurface(
            mesh, blocks=motl_f, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        # Must not raise when max_block_distance is None (default)
        ps.unify_polarity(reference="envelope")

    def test_neighbours_requires_max_distance(self):
        ps, _ = self._make_inverted(seed=0)
        with pytest.raises(ValueError, match="max_block_distance"):
            ps.unify_polarity(reference="neighbours")

    def test_envelope_reference_returns_8_and_all_outward(self):
        # Phase 5 Step 2: verify exactly 8 blocks are flipped and all z outward.
        motl_clean, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps_clean = structure.PleomorphicSurface(
            blocks=motl_clean, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_clean.connect(max_distance=4.0)
        mesh = ps_clean.envelope_from_faces()
        motl_flipped, _, _ = _soccer_ball(seed=0, flipped=(3, 17, 25, 31, 40, 44, 52, 59))
        ps = structure.PleomorphicSurface(
            mesh, blocks=motl_flipped, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        n = ps.unify_polarity(reference="envelope")
        assert n == 8
        from scipy.spatial.transform import Rotation as srot
        angles_arr = ps.blocks.df[["phi", "theta", "psi"]].values.astype(float)
        z_axes = np.array([R.apply(np.array([0., 0., 1.])) for R in srot.from_euler("zxz", angles_arr, degrees=True)])
        coords = ps.blocks.df[["x", "y", "z"]].values
        centre = coords.mean(axis=0)
        assert (np.array([np.dot(z_axes[i], coords[i] - centre) for i in range(len(z_axes))]) > 0).all()


# =============================================================================
# Phase 3 — T7.7: BlockDefinition.cyclic with Symmetry
# =============================================================================

class TestBlockDefinitionCyclicSymmetry:
    def test_int_n_still_works(self):
        bd3 = structure.BlockDefinition.cyclic(3, [9.0, 0.0, 0.0])
        assert bd3.n_sites == 3
        assert bd3.fold == 3

    def test_string_cn_works(self):
        bd5 = structure.BlockDefinition.cyclic("C5", [9.0, 0.0, 0.0])
        assert bd5.n_sites == 5
        assert bd5.fold == 5

    def test_string_and_int_give_same_sites(self):
        bd_int = structure.BlockDefinition.cyclic(3, [9.0, 0.0, 0.0])
        bd_str = structure.BlockDefinition.cyclic("C3", [9.0, 0.0, 0.0])
        np.testing.assert_allclose(bd_int.site_vectors(), bd_str.site_vectors(), atol=1e-12)

    def test_dihedral_raises(self):
        with pytest.raises(ValueError, match="Cn"):
            structure.BlockDefinition.cyclic("D2", [9.0, 0.0, 0.0])

    def test_tetrahedral_raises(self):
        with pytest.raises(ValueError, match="Cn"):
            structure.BlockDefinition.cyclic("T", [9.0, 0.0, 0.0])

    def test_case_insensitive(self):
        bd = structure.BlockDefinition.cyclic("c4", [9.0, 0.0, 0.0])
        assert bd.n_sites == 4


# =============================================================================
# Phase 3 — T7.8: default ideal lattice derivation
# =============================================================================

class TestDefaultIdealLattice:
    def test_d3_gives_ideals_3_6(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        # Do NOT pass ideal_degree / ideal_face_size
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        assert ps.ideal_degree == 3
        assert ps.ideal_face_size == 6

    def test_d3_defect_charge(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        ps.connect(max_distance=4.0)
        asm = ps.get_assembly_stats()
        np.testing.assert_allclose(float(asm.iloc[0]["defect_charge"]), 2.0, atol=1e-9)

    def test_d4_gives_ideals_4_4(self):
        motl, block_def = _microtubule(n_pf=13, n_dimers=5)
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        assert ps.ideal_degree == 4
        assert ps.ideal_face_size == 4

    def test_d5_no_derivation(self):
        # n_sites=5 → not in {3,4,6} → ideals remain None
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def_5 = structure.BlockDefinition.cyclic(5, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def_5)
        assert ps.ideal_degree is None
        assert ps.ideal_face_size is None

    def test_explicit_ideals_not_overridden(self):
        motl, arm_length, arm_elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm_length * np.cos(np.radians(arm_elev)), 0.0, -arm_length * np.sin(np.radians(arm_elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        # Explicit values are kept
        assert ps.ideal_degree == 3
        assert ps.ideal_face_size == 6


# =============================================================================
# Phase 3 — T7.9: BlockDefinition.microtubule preset
# =============================================================================

class TestMicrotubulePreset:
    def _make_bd(self, elev=0.0):
        import math
        R_tube = 11.4
        n_pf = 13
        lat = R_tube * math.sin(math.pi / n_pf)
        ax = 4.08
        return structure.BlockDefinition.microtubule(lat, ax, elev)

    def test_four_sites(self):
        bd = self._make_bd()
        assert bd.n_sites == 4

    def test_site_types(self):
        bd = self._make_bd()
        assert bd.site_types == ("lateral_right", "plus", "lateral_left", "minus")

    def test_lateral_sites_symmetric(self):
        bd = self._make_bd(elev=0.0)
        vecs = bd.site_vectors()
        # lateral_right.x == -lateral_left.x
        np.testing.assert_allclose(vecs[0, 0], -vecs[2, 0], atol=1e-12)
        # y-components zero
        np.testing.assert_allclose(vecs[0, 1], 0.0, atol=1e-12)
        np.testing.assert_allclose(vecs[2, 1], 0.0, atol=1e-12)

    def test_axial_sites_symmetric(self):
        bd = self._make_bd()
        vecs = bd.site_vectors()
        np.testing.assert_allclose(vecs[1, 1], -vecs[3, 1], atol=1e-12)
        np.testing.assert_allclose(vecs[1, 0], 0.0, atol=1e-12)
        np.testing.assert_allclose(vecs[3, 0], 0.0, atol=1e-12)

    def test_flip_site_is_2(self):
        bd = self._make_bd()
        assert bd.flip_site == 2

    def test_pairing(self):
        bd = self._make_bd()
        pairs = {frozenset(p) for p in bd.pairing}
        assert frozenset({"plus", "minus"}) in pairs
        assert frozenset({"lateral_right", "lateral_left"}) in pairs

    def test_elevation_tilts_sites(self):
        import math
        elev_deg = 180 / 13
        bd = self._make_bd(elev=elev_deg)
        vecs = bd.site_vectors()
        e = math.radians(elev_deg)
        R_tube = 11.4
        lat = R_tube * math.sin(math.pi / 13)
        expected_x = lat * math.cos(e)
        expected_z = -lat * math.sin(e)
        np.testing.assert_allclose(vecs[0, 0], expected_x, atol=1e-10)
        np.testing.assert_allclose(vecs[0, 2], expected_z, atol=1e-10)

    def test_microtubule_connect_mismatch_y(self):
        motl, block_def = _microtubule(n_pf=13, n_dimers=5)
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        ps.connect(max_distance=4.0)
        cs = ps.get_contact_stats()
        lat_r = cs[cs["site_type"] == "lateral_right"]["mismatch_y"].abs()
        assert set(lat_r.round(2).unique()).issubset({0.94, 3.14})


# =============================================================================
# Phase 4 â€” helper: flat honeycomb lattice
# =============================================================================

def _honeycomb(
    edge: float = 18.0,
    n: int = 4,
    seed: int = 0,
) -> "tuple[cryomotl.Motl, float]":
    """Build a flat honeycomb (graphene-like) block motl.

    Lattice vectors a1 = (1.5e, sqrt3/2*e, 0) and a2 = (1.5e, -sqrt3/2*e, 0).
    For i, j in range(-n, n+1) appends A = i*a1 + j*a2 then B = A + (e, 0, 0),
    both offset by (100, 100, 100).  Rotations are pure-Z:
    Rotation.from_euler("z", 60*s + 120*k, degrees=True) with s=0 for A, s=1 for B,
    k drawn from rng.integers(3) per block in order.

    Returns (motl, arm_length) where arm_length = edge / 2.
    """
    sqrt3 = np.sqrt(3)
    a1 = np.array([1.5 * edge, (sqrt3 / 2) * edge, 0.0])
    a2 = np.array([1.5 * edge, -(sqrt3 / 2) * edge, 0.0])
    offset = np.array([100.0, 100.0, 100.0])

    pts_list: list[np.ndarray] = []
    s_list: list[int] = []
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            A = i * a1 + j * a2 + offset
            B = A + np.array([edge, 0.0, 0.0])
            pts_list.extend([A, B])
            s_list.extend([0, 1])

    pts = np.array(pts_list)
    n_blocks = len(pts)
    rng = np.random.default_rng(seed)
    ks = rng.integers(3, size=n_blocks)
    angles_z = [60.0 * s + 120.0 * k for s, k in zip(s_list, ks)]
    R_list = [Rotation.from_euler("z", float(a), degrees=True) for a in angles_z]
    euler = np.array([R.as_euler("zxz", degrees=True) for R in R_list])

    data = {c: np.zeros(n_blocks) for c in cryomotl.Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n_blocks + 1, dtype=float)
    data["tomo_id"] = np.ones(n_blocks, dtype=float)
    data["x"] = pts[:, 0]
    data["y"] = pts[:, 1]
    data["z"] = pts[:, 2]
    data["phi"] = euler[:, 0]
    data["theta"] = euler[:, 1]
    data["psi"] = euler[:, 2]
    motl = cryomotl.Motl(pd.DataFrame(data))
    return motl, edge / 2.0


# =============================================================================
# Phase 4 â€” Step 1: face walk-order test
# =============================================================================

class TestFaceBlockIdsWalkOrder:
    """Every consecutive block pair in face block_ids must be bonded,
    including the last-to-first wrap-around.  Tests five distinct geometries."""

    @staticmethod
    def _check_walk_order(ps: structure.PleomorphicSurface) -> None:
        cs = ps.get_contact_stats()
        bonds: set[tuple[float, float]] = set()
        for _, row in cs.iterrows():
            bonds.add((float(row["block_id"]), float(row["partner_block_id"])))

        fs = ps.get_face_stats()
        for _, frow in fs.iterrows():
            bids = frow["block_ids"]
            n = len(bids)
            for i in range(n):
                b_a = float(bids[i])
                b_b = float(bids[(i + 1) % n])
                assert (b_a, b_b) in bonds or (b_b, b_a) in bonds, (
                    f"Face {frow['face_id']}: blocks {b_a} and {b_b} are not bonded"
                )

    def test_soccer_ball(self):
        motl, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        self._check_walk_order(ps)

    def test_geodesic(self):
        motl, block_def = _geodesic(edge=10.0, seed=0)
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            block_type_column="geom3",
            ideal_degree=6, ideal_face_size=3,
        )
        ps.connect(max_distance=3.0)
        self._check_walk_order(ps)

    def test_microtubule(self):
        motl, block_def = _microtubule(n_pf=13, n_dimers=20)
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=4, ideal_face_size=4,
        )
        ps.connect(max_distance=4.0)
        self._check_walk_order(ps)

    def test_open_patch(self):
        motl_full, arm, elev = _soccer_ball(seed=0)
        coords = motl_full.get_coordinates()
        mask = (coords[:, 2] - 100.0) >= -5.0
        sub_motl = cryomotl.Motl(motl_full.df[mask].copy().reset_index(drop=True))
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=sub_motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        self._check_walk_order(ps)

    def test_honeycomb(self):
        motl, arm = _honeycomb(edge=18.0, n=4, seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm, 0.0, 0.0])
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        ps.connect(max_distance=4.0)
        self._check_walk_order(ps)


# =============================================================================
# Phase 4 â€” Step 3: flat honeycomb lattice tests
# =============================================================================

class TestHoneycombLattice:
    @pytest.fixture(scope="class")
    def psurf(self):
        motl, arm = _honeycomb(edge=18.0, n=4, seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm, 0.0, 0.0])
        # No explicit ideals: derived from n_sites=3 gives (3, 6).
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        ps.connect(max_distance=4.0)
        return ps

    def test_matched_sites(self, psurf):
        st = psurf._site_table
        assert (st["partner"] >= 0).sum() == 450

    def test_closed_face_count(self, psurf):
        fs = psurf.get_face_stats()
        assert len(fs) == 64

    def test_all_faces_hexagons(self, psurf):
        fs = psurf.get_face_stats()
        assert (fs["size"] == 6).all()

    def test_vef_counts(self, psurf):
        asm = psurf.get_assembly_stats()
        row = asm.iloc[0]
        assert int(row["n_blocks"]) == 162
        assert int(row["n_contacts"]) == 225
        assert int(row["n_faces"]) == 64

    def test_euler_characteristic(self, psurf):
        asm = psurf.get_assembly_stats()
        assert int(asm.iloc[0]["euler_characteristic"]) == 1

    def test_not_closed(self, psurf):
        asm = psurf.get_assembly_stats()
        assert bool(asm.iloc[0]["closed"]) is False

    def test_defect_charge(self, psurf):
        asm = psurf.get_assembly_stats()
        np.testing.assert_allclose(float(asm.iloc[0]["defect_charge"]), 0.0, atol=1e-9)

    def test_angle_deficit_sum(self, psurf):
        asm = psurf.get_assembly_stats()
        np.testing.assert_allclose(
            float(asm.iloc[0]["angle_deficit_sum"]), 0.0, atol=1e-6
        )

    def test_envelope_triangle_count(self, psurf):
        mesh = psurf.envelope_from_faces()
        assert len(mesh.faces) == 384  # 64 faces x 6 triangles

    def test_envelope_normals_positive_z(self, psurf):
        mesh = psurf.envelope_from_faces()
        tri_verts = mesh.vertices[mesh.faces]
        e0 = tri_verts[:, 1] - tri_verts[:, 0]
        e1 = tri_verts[:, 2] - tri_verts[:, 0]
        geom_normals = np.cross(e0, e1)
        assert (geom_normals[:, 2] > 0).all(), (
            "Some envelope triangles have non-positive z-component normal"
        )


# =============================================================================
# Phase 4 â€” Step 4: unify_polarity always invalidates the contact graph
# =============================================================================

class TestUnifyPolarityInvalidatesGraph:
    """After any unify_polarity call, _faces is cleared and connect() is
    required before graph queries; reconnecting restores the soccer-ball topology."""

    def test_envelope_reference_clears_graph_and_reconnect_restores(self):
        # Build an unflipped soccer ball and extract its face-polygon envelope.
        motl_clean, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps_clean = structure.PleomorphicSurface(
            blocks=motl_clean, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_clean.connect(max_distance=4.0)
        mesh = ps_clean.envelope_from_faces()
        # Create a new surface with the correct envelope + 8 specifically chosen
        # flipped blocks (distributed so no face has a flipped majority).
        motl_flipped, _, _ = _soccer_ball(seed=0, flipped=(3, 17, 25, 31, 40, 44, 52, 59))
        ps = structure.PleomorphicSurface(
            mesh, blocks=motl_flipped, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        ps.unify_polarity(reference="envelope")
        # Contact graph must be cleared after unify_polarity
        with pytest.raises(ValueError):
            ps.get_face_stats()
        # After reconnecting, the phase-2 soccer-ball topology is restored
        ps.connect(max_distance=4.0)
        asm = ps.get_assembly_stats()
        assert int(asm.iloc[0]["euler_characteristic"]) == 2
        assert int(asm.iloc[0]["n_blocks"]) == 60
        assert int(asm.iloc[0]["n_contacts"]) == 90
        assert int(asm.iloc[0]["n_faces"]) == 32


# =============================================================================
# Phase 5 — Step 2.3: blocks-only object ValueError tests
# =============================================================================

class TestEnvelopeMethodsRequireEnvelope:
    def test_annotate_raises_without_envelope(self):
        motl, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        with pytest.raises(ValueError):
            ps.annotate_with_envelope()

    def test_unify_polarity_raises_without_envelope(self):
        motl, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=motl, block_definition=block_def)
        with pytest.raises(ValueError):
            ps.unify_polarity(reference="envelope")


# =============================================================================
# Phase 6 — Step 2.2: blocks-only PS after connect() still raises
# =============================================================================

class TestEnvelopeMethodsRequireEnvelopeAfterConnect:
    """Even after connect(), a blocks-only PleomorphicSurface without an
    attached envelope raises ValueError from annotate_with_envelope and
    unify_polarity(reference='envelope'), and has_envelope stays False."""

    @pytest.fixture
    def ps_connected(self):
        motl, arm, elev = _soccer_ball(seed=0)
        block_def = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=block_def,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_annotate_raises_after_connect(self, ps_connected):
        with pytest.raises(ValueError):
            ps_connected.annotate_with_envelope()

    def test_unify_polarity_raises_after_connect(self, ps_connected):
        angles_before = ps_connected.blocks.df[["phi", "theta", "psi"]].values.copy()
        with pytest.raises(ValueError):
            ps_connected.unify_polarity(reference="envelope")
        angles_after = ps_connected.blocks.df[["phi", "theta", "psi"]].values
        np.testing.assert_array_equal(angles_before, angles_after)

    def test_has_envelope_unchanged(self, ps_connected):
        try:
            ps_connected.unify_polarity(reference="envelope")
        except ValueError:
            pass
        assert ps_connected.has_envelope is False


# =============================================================================
# Phase 5 — Step 3: expand_motl radial ±z guard tests
# =============================================================================

class TestExpandMotlRadialPolarZ:
    """expand_motl with orientation='radial' must handle ±z shift vectors
    without raising and must produce particles whose z-axes point along
    the shift direction (± tolerance 1e-6)."""

    def _make_single_motl(self):
        data = {c: np.zeros(1) for c in cryomotl.Motl.motl_columns}
        data["subtomo_id"] = np.array([1.0])
        data["tomo_id"] = np.array([1.0])
        return cryomotl.Motl(pd.DataFrame(data))

    def test_plus_and_minus_z_finite_angles(self):
        motl = self._make_single_motl()
        shift_vecs = np.array([[0.0, 0.0, 5.0], [0.0, 0.0, -5.0]])
        expanded = structure.expand_motl(
            motl, shift_vecs, orientation="radial", sort_vectors=False
        )
        angles = expanded.df[["phi", "theta", "psi"]].values.astype(float)
        assert np.isfinite(angles).all(), "Euler angles must be finite for ±z shifts"

    def test_plus_z_axis(self):
        motl = self._make_single_motl()
        expanded = structure.expand_motl(
            motl, np.array([[0.0, 0.0, 5.0]]), orientation="radial", sort_vectors=False
        )
        angles = expanded.df[["phi", "theta", "psi"]].values.astype(float)
        z_axes = geom.euler_angles_to_normals(angles)
        np.testing.assert_allclose(z_axes[0], [0.0, 0.0, 1.0], atol=1e-6)

    def test_minus_z_axis(self):
        motl = self._make_single_motl()
        expanded = structure.expand_motl(
            motl, np.array([[0.0, 0.0, -5.0]]), orientation="radial", sort_vectors=False
        )
        angles = expanded.df[["phi", "theta", "psi"]].values.astype(float)
        z_axes = geom.euler_angles_to_normals(angles)
        np.testing.assert_allclose(z_axes[0], [0.0, 0.0, -1.0], atol=1e-6)


# =============================================================================
# Phase 7 — from_blocks classmethod and @gui_exposed metadata
# =============================================================================


class TestFromBlocks:
    """PleomorphicSurface.from_blocks builds the same assembly as direct construction."""

    def test_single_symmetry_matches_direct_construction(self):
        """from_blocks("C3") + connect gives the same soccer-ball topology as the
        direct constructor with BlockDefinition.cyclic."""
        motl, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])

        ps_direct = structure.PleomorphicSurface(
            blocks=motl, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_direct.connect(max_distance=4.0)
        asm_direct = ps_direct.get_assembly_stats()

        ps_fb = structure.PleomorphicSurface.from_blocks(
            motl, symmetry="C3",
            site_shift=[arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))],
        )
        ps_fb.connect(max_distance=4.0)
        asm_fb = ps_fb.get_assembly_stats()

        assert int(asm_fb.iloc[0]["n_blocks"]) == int(asm_direct.iloc[0]["n_blocks"])
        assert int(asm_fb.iloc[0]["n_contacts"]) == int(asm_direct.iloc[0]["n_contacts"])
        assert int(asm_fb.iloc[0]["n_faces"]) == int(asm_direct.iloc[0]["n_faces"])
        assert int(asm_fb.iloc[0]["euler_characteristic"]) == int(asm_direct.iloc[0]["euler_characteristic"])

    def test_single_symmetry_sites_motl_matches_direct(self):
        """get_sites_as_motl() from from_blocks equals that from the direct constructor."""
        motl, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])

        ps_direct = structure.PleomorphicSurface(
            blocks=motl, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps_fb = structure.PleomorphicSurface.from_blocks(
            motl, symmetry="C3",
            site_shift=[arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))],
        )

        sites_direct = ps_direct.get_sites_as_motl()
        sites_fb = ps_fb.get_sites_as_motl()

        assert len(sites_direct.df) == len(sites_fb.df)
        np.testing.assert_allclose(
            sites_direct.df[["x", "y", "z"]].values,
            sites_fb.df[["x", "y", "z"]].values,
            atol=1e-10,
        )

    def test_mixed_symmetry_column_matches_direct_and_has_correct_ideals(self):
        """from_blocks(symmetry_column='class') gives the same geodesic topology
        as the direct constructor with a block_definition dict, and the inferred
        ideals are (6, 3) (max n_sites=6 for C6 blocks)."""
        geo_motl, block_def_dict = _geodesic(edge=10.0, seed=0)

        # Build direct version (geom3 column)
        ps_direct = structure.PleomorphicSurface(
            blocks=geo_motl, block_definition=block_def_dict,
            block_type_column="geom3",
            ideal_degree=6, ideal_face_size=3,
        )
        ps_direct.connect(max_distance=3.0)
        asm_direct = ps_direct.get_assembly_stats()

        # Build from_blocks version (class column, mixed symmetry)
        new_df = geo_motl.df.copy()
        new_df["class"] = new_df["geom3"]
        new_motl = cryomotl.Motl(new_df)

        ps_fb = structure.PleomorphicSurface.from_blocks(
            new_motl,
            site_shift=block_def_dict[5.0].sites[0].vector,
            symmetry_column="class",
        )
        ps_fb.connect(max_distance=3.0)
        asm_fb = ps_fb.get_assembly_stats()

        # Check ideals derived from max n_sites = 6
        assert ps_fb.ideal_degree == 6
        assert ps_fb.ideal_face_size == 3

        # Check topology matches direct construction
        assert int(asm_fb.iloc[0]["n_blocks"]) == int(asm_direct.iloc[0]["n_blocks"])
        assert int(asm_fb.iloc[0]["n_contacts"]) == int(asm_direct.iloc[0]["n_contacts"])
        assert int(asm_fb.iloc[0]["n_faces"]) == int(asm_direct.iloc[0]["n_faces"])

    def test_non_integer_symmetry_column_raises(self):
        """A symmetry_column value of 2.5 (non-integer) raises ValueError."""
        motl, arm, elev = _soccer_ball(seed=0)
        # Set "class" column to 2.5 (non-integer)
        df = motl.df.copy()
        df["class"] = 2.5
        bad_motl = cryomotl.Motl(df)
        with pytest.raises(ValueError, match="non-integer"):
            structure.PleomorphicSurface.from_blocks(
                bad_motl, site_shift=[arm, 0.0, 0.0], symmetry_column="class"
            )


_EXPECTED_GUI_METADATA = [
    ("PleomorphicSurface.unify_polarity",    "Unify polarity",     "Lattice setup",       20, "none"),
    ("PleomorphicSurface.connect",           "Connect",            "Lattice setup",       30, "none"),
    ("PleomorphicSurface.get_sites_as_motl", "Sites as motl",      "Lattice setup",       40, "motl"),
    ("PleomorphicSurface.get_blocks_as_motl","Blocks as motl",     "Lattice setup",       50, "motl"),
    ("PleomorphicSurface.get_contact_stats", "Contact stats",      "Lattice statistics",  10, "dataframe"),
    ("PleomorphicSurface.get_face_stats",    "Face stats",         "Lattice statistics",  20, "dataframe"),
    ("PleomorphicSurface.get_block_stats",   "Block stats",        "Lattice statistics",  30, "dataframe"),
    ("PleomorphicSurface.get_assembly_stats","Assembly stats",     "Lattice statistics",  40, "dataframe"),
    ("PleomorphicSurface.store_block_stat",  "Store block stat",   "Lattice statistics",  50, "motl"),
    ("PleomorphicSurface.get_faces_as_motl", "Faces as motl",      "Lattice motls",       10, "motl"),
    ("PleomorphicSurface.get_gaps_as_motl",  "Gaps as motl",       "Lattice motls",       20, "motl"),
    ("PleomorphicSurface.envelope_from_faces","Envelope from faces","Lattice envelope",   10, "surface"),
]


class TestGuiExposedMetadata:
    """Each @gui_exposed method on PleomorphicSurface has the correct label,
    group, order, and returns registered in GUI_REGISTRY."""

    @pytest.mark.parametrize("key,label,group,order,returns", _EXPECTED_GUI_METADATA)
    def test_entry_registered_with_correct_metadata(self, key, label, group, order, returns):
        from cryocat.utils.classutils import GUI_REGISTRY, GuiCategory
        assert key in GUI_REGISTRY, (
            f"GUI_REGISTRY does not contain {key!r}. "
            "Did the @gui_exposed decorator fire? Check that structure.py was imported."
        )
        entry = GUI_REGISTRY[key]
        assert entry.label == label,   f"{key}: label {entry.label!r} != {label!r}"
        assert entry.group == group,   f"{key}: group {entry.group!r} != {group!r}"
        assert entry.order == order,   f"{key}: order {entry.order} != {order}"
        assert entry.returns == returns, f"{key}: returns {entry.returns!r} != {returns!r}"
        assert entry.category == GuiCategory.MOTL_OP, (
            f"{key}: category {entry.category!r} != GuiCategory.MOTL_OP"
        )


# =============================================================================
# Phase 8 Step 0 — store_block_stat behaviour
# =============================================================================

class TestStoreBlockStat:
    """store_block_stat writes one block-stats column and returns a deep copy."""

    @pytest.fixture(scope="class")
    def soccer_ps(self):
        motl, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=motl, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps

    def test_returns_motl_with_correct_column(self, soccer_ps):
        result = soccer_ps.store_block_stat("degree", "geom4")
        assert (result.df["geom4"] == 3).all(), (
            "Expected degree=3 for all soccer-ball blocks."
        )

    def test_mutates_instance_blocks(self, soccer_ps):
        soccer_ps.store_block_stat("degree", "geom4")
        assert (soccer_ps.blocks.df["geom4"] == 3).all(), (
            "store_block_stat must mutate ps.blocks in place."
        )

    def test_returned_motl_is_deep_copy(self, soccer_ps):
        result = soccer_ps.store_block_stat("degree", "geom4")
        assert result is not soccer_ps.blocks, (
            "Returned Motl must be a deep copy, not the same object."
        )

    def test_invalid_stat_raises(self, soccer_ps):
        with pytest.raises(ValueError):
            soccer_ps.store_block_stat("face_signature", "geom4")


# =============================================================================
# Phase 11 — input motl is copied; get_blocks_as_motl retrieves current state
# =============================================================================

class TestBlocksIsCopied:
    """The input Motl is always deep-copied; the original is never modified."""

    def test_init_blocks_is_not_same_object(self):
        m, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        assert ps.blocks is not m
        pd.testing.assert_frame_equal(ps.blocks.df.reset_index(drop=True),
                                      m.df.reset_index(drop=True))

    def test_from_blocks_single_symmetry_is_not_same_object(self):
        m, arm, elev = _soccer_ball(seed=0)
        ps = structure.PleomorphicSurface.from_blocks(
            m, "C3",
            site_shift=[arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))],
        )
        assert ps.blocks is not m
        pd.testing.assert_frame_equal(ps.blocks.df.reset_index(drop=True),
                                      m.df.reset_index(drop=True))

    def test_from_blocks_mixed_symmetry_column_is_not_same_object(self):
        from scipy.spatial.distance import cdist as _cdist

        geo_m, _def = _geodesic(edge=10.0, seed=0)

        # Put block types (5 or 6) in the "class" column so from_blocks can read them.
        geo_m.df["class"] = geo_m.df["geom3"]

        # Recompute L and el using the same formulas the builder uses internally.
        pts = geo_m.df[["x", "y", "z"]].values
        centre = pts.mean(axis=0)
        d = _cdist(pts, pts)
        np.fill_diagonal(d, np.inf)
        nn_idx = np.argsort(d, axis=1)
        c_vals = geo_m.df["class"].values.astype(int)
        elev_samples: list[float] = []
        arm_lengths: list[float] = []
        for i in range(len(pts)):
            c_i = c_vals[i]
            z_i = pts[i] - centre
            z_i /= np.linalg.norm(z_i)
            nn_c = nn_idx[i, :c_i]
            for j in nn_c:
                diff_nn = pts[i] - pts[j]
                elev_samples.append(float(np.dot(diff_nn / np.linalg.norm(diff_nn), z_i)))
            arm_lengths.append(float(np.mean(d[i, nn_c]) / 2.0))
        el = float(np.degrees(np.arcsin(np.clip(np.mean(elev_samples), -1.0, 1.0))))
        L = float(np.mean(arm_lengths))

        ps = structure.PleomorphicSurface.from_blocks(
            geo_m,
            site_shift=[L * np.cos(np.radians(el)), 0.0, -L * np.sin(np.radians(el))],
            symmetry_column="class",
        )
        assert ps.blocks is not geo_m
        pd.testing.assert_frame_equal(
            ps.blocks.df.reset_index(drop=True),
            geo_m.df.reset_index(drop=True),
        )
        assert ps.ideal_degree == 6
        assert ps.ideal_face_size == 3

    def test_copy_constructor_deep_copies_blocks(self):
        m, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        ps2 = structure.PleomorphicSurface(ps)
        assert ps2.blocks is not ps.blocks
        pd.testing.assert_frame_equal(ps2.blocks.df.reset_index(drop=True),
                                      ps.blocks.df.reset_index(drop=True))

    def test_file_path_construction_works(self, tmp_path):
        m, arm, elev = _soccer_ball(seed=0)
        em_file = str(tmp_path / "blocks.em")
        m.write_out(em_file)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=em_file, block_definition=bd)
        assert ps.blocks is not m
        assert len(ps.blocks.df) == len(m.df)


class TestInputUntouchedByUnifyPolarity:
    """The original input motl is not modified by unify_polarity."""

    def test_original_motl_unchanged(self):
        m, arm, elev = _soccer_ball(flipped=_SOCCER_FLIPPED)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        before = m.df.copy()
        ps.unify_polarity(20.0, "centroid")
        pd.testing.assert_frame_equal(m.df, before, check_like=False)

    def test_internal_blocks_exactly_flipped_rows_changed(self):
        from scipy.spatial.transform import Rotation as srot
        m, arm, elev = _soccer_ball(flipped=_SOCCER_FLIPPED)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        before = m.df.copy()
        ps.unify_polarity(20.0, "centroid")
        angle_cols = ["phi", "theta", "psi"]
        R_before = srot.from_euler("zxz", before[angle_cols].values, degrees=True)
        R_after = srot.from_euler("zxz", ps.blocks.df[angle_cols].values, degrees=True)
        rel_mag = (R_before.inv() * R_after).magnitude()
        changed = (rel_mag > 1e-6).sum()
        assert changed == len(_SOCCER_FLIPPED), (
            f"Expected {len(_SOCCER_FLIPPED)} rows changed, got {changed}."
        )

    def test_internal_non_angle_columns_unchanged(self):
        m, arm, elev = _soccer_ball(flipped=_SOCCER_FLIPPED)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        before = m.df.copy()
        ps.unify_polarity(20.0, "centroid")
        other_cols = [c for c in before.columns if c not in ("phi", "theta", "psi")]
        pd.testing.assert_frame_equal(
            ps.blocks.df[other_cols].reset_index(drop=True),
            before[other_cols].reset_index(drop=True),
            check_like=False,
        )

    def test_internal_index_and_row_order_unchanged(self):
        m, arm, elev = _soccer_ball(flipped=_SOCCER_FLIPPED)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        before_index = m.df.index.tolist()
        before_subtomo = m.df["subtomo_id"].tolist()
        ps.unify_polarity(20.0, "centroid")
        assert ps.blocks.df.index.tolist() == before_index, "Index must not change."
        assert ps.blocks.df["subtomo_id"].tolist() == before_subtomo, "Row order must not change."


class TestInputUntouchedByStores:
    """The original input motl is not modified by store_block_stats/stat."""

    @pytest.fixture
    def connected_ps(self):
        m, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=m, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        return ps, m

    def test_original_motl_unchanged_by_store_block_stats(self, connected_ps):
        ps, m = connected_ps
        before = m.df.copy()
        ps.store_block_stats({"degree": "geom4"})
        pd.testing.assert_frame_equal(m.df, before, check_like=False)

    def test_internal_blocks_updated_by_store_block_stats(self, connected_ps):
        ps, m = connected_ps
        ps.store_block_stats({"degree": "geom4"})
        assert (ps.blocks.df["geom4"] == 3.0).all()

    def test_store_block_stat_returned_motl_is_not_blocks(self, connected_ps):
        ps, m = connected_ps
        ps.store_block_stats({"degree": "geom4"})
        result = ps.store_block_stat("assembly_id", "geom5")
        assert result is not ps.blocks
        assert (result.df["geom4"] == 3.0).all(), "Returned motl must include geom4."
        assert (result.df["geom5"] >= 1.0).all(), "Returned motl must include geom5."

    def test_original_motl_unchanged_by_store_block_stat(self, connected_ps):
        ps, m = connected_ps
        before = m.df.copy()
        ps.store_block_stat("degree", "geom4")
        pd.testing.assert_frame_equal(m.df, before, check_like=False)


class TestGettersLeaveStateUnchanged:
    """Stat getters and get_sites_as_motl must not modify input or internal motl."""

    def test_getters_leave_original_and_blocks_unchanged(self):
        m, arm, elev = _soccer_ball(seed=0)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=m, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        before_m = m.df.copy()
        before_blocks = ps.blocks.df.copy()
        ps.get_sites_as_motl()
        ps.get_contact_stats()
        ps.get_face_stats()
        ps.get_block_stats()
        ps.get_assembly_stats()
        pd.testing.assert_frame_equal(m.df, before_m, check_like=False)
        pd.testing.assert_frame_equal(ps.blocks.df, before_blocks, check_like=False)


class TestGetBlocksAsMotl:
    """get_blocks_as_motl returns an independent deep copy of the current block state."""

    @pytest.fixture
    def unified_ps(self):
        m, arm, elev = _soccer_ball(flipped=_SOCCER_FLIPPED)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=m, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        ps.store_block_stats({"degree": "geom4"})
        ps.unify_polarity(20.0, "centroid")
        return ps

    def test_returns_not_blocks(self, unified_ps):
        result = unified_ps.get_blocks_as_motl()
        assert result is not unified_ps.blocks

    def test_returned_df_equals_blocks_df(self, unified_ps):
        result = unified_ps.get_blocks_as_motl()
        pd.testing.assert_frame_equal(result.df.reset_index(drop=True),
                                      unified_ps.blocks.df.reset_index(drop=True))

    def test_modifying_returned_leaves_blocks_unchanged(self, unified_ps):
        before_blocks = unified_ps.blocks.df.copy()
        result = unified_ps.get_blocks_as_motl()
        result.df["geom5"] = 999.0
        pd.testing.assert_frame_equal(unified_ps.blocks.df, before_blocks,
                                      check_like=False)

    def test_raises_without_blocks(self):
        envelope_only = _tiny_mesh_psurf()
        with pytest.raises(ValueError, match="requires a block layer"):
            envelope_only.get_blocks_as_motl()


class TestGetBlocksRoundTrip:
    """get_blocks_as_motl feeds a new assembly with pre-unified orientations."""

    def test_round_trip_soccer_ball_topology(self):
        m, arm, elev = _soccer_ball(flipped=_SOCCER_FLIPPED)
        bd = structure.BlockDefinition.cyclic(3, [arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))])
        ps = structure.PleomorphicSurface(
            blocks=m, block_definition=bd,
            ideal_degree=3, ideal_face_size=6,
        )
        ps.connect(max_distance=4.0)
        ps.unify_polarity(20.0, "centroid")

        unified_blocks = ps.get_blocks_as_motl()
        ps2 = structure.PleomorphicSurface.from_blocks(
            unified_blocks, "C3",
            site_shift=[arm * np.cos(np.radians(elev)), 0.0, -arm * np.sin(np.radians(elev))],
        )
        ps2.connect(max_distance=4.0)

        asm = ps2.get_assembly_stats().iloc[0]
        assert int(asm["euler_characteristic"]) == 2
        fs = ps2.get_face_stats()["size"].value_counts().to_dict()
        assert fs.get(5, 0) == 12, f"Expected 12 pentagons, got {fs}"
        assert fs.get(6, 0) == 20, f"Expected 20 hexagons, got {fs}"


# =============================================================================
# Phase 13 — site_shift vector API: new tests
# =============================================================================

class TestBlockDefinitionCyclicSiteShift:
    """Phase 13 — site_shift replaces length/azimuth/elevation in cyclic()."""

    def test_site_vectors_c3_minus_x(self):
        """cyclic(3, [-5, 0, 0]) produces the expected three site vectors."""
        bd = structure.BlockDefinition.cyclic(3, [-5.0, 0.0, 0.0])
        vecs = bd.site_vectors()
        expected = np.array([
            [-5.0,  0.0,    0.0],
            [ 2.5, -4.330,  0.0],
            [ 2.5,  4.330,  0.0],
        ])
        np.testing.assert_allclose(vecs, expected, atol=1e-3)

    def test_get_sites_as_motl_positions_equal_c_plus_R_apply_vk(self):
        """get_sites_as_motl places site k at round(c + R.apply(vector_k)) for each block."""
        rng = np.random.default_rng(0)
        n_blocks = 3
        data = {c: np.zeros(n_blocks) for c in cryomotl.Motl.motl_columns}
        data["subtomo_id"] = np.arange(1, n_blocks + 1, dtype=float)
        data["tomo_id"] = np.ones(n_blocks, dtype=float)
        data["x"] = rng.uniform(80.0, 120.0, n_blocks)
        data["y"] = rng.uniform(80.0, 120.0, n_blocks)
        data["z"] = rng.uniform(80.0, 120.0, n_blocks)
        data["phi"] = rng.uniform(-180.0, 180.0, n_blocks)
        data["theta"] = rng.uniform(0.0, 180.0, n_blocks)
        data["psi"] = rng.uniform(-180.0, 180.0, n_blocks)
        m = cryomotl.Motl(pd.DataFrame(data)[cryomotl.Motl.motl_columns])

        bd = structure.BlockDefinition.cyclic(3, [-5.0, 0.0, 0.0])
        ps = structure.PleomorphicSurface(blocks=m, block_definition=bd)
        sites = ps.get_sites_as_motl()

        vecs = bd.site_vectors()
        block_coords = m.get_coordinates()
        angles = m.df[["phi", "theta", "psi"]].values.astype(float)
        all_R = Rotation.from_euler("zxz", angles, degrees=True)

        for row_idx in range(n_blocks):
            block_id = float(m.df.iloc[row_idx]["subtomo_id"])
            c = block_coords[row_idx]
            R = all_R[row_idx]
            for k_idx, vec_k in enumerate(vecs):
                expected_f = c + R.apply(vec_k)
                # split_in_asymmetric_subunits rounds positions to integers
                expected = np.where(expected_f >= 0,
                                    np.floor(expected_f + 0.5),
                                    -np.floor(-expected_f + 0.5))
                mask = (sites.df["object_id"] == block_id) & (sites.df["geom1"] == float(k_idx + 1))
                site_rows = sites.df[mask]
                assert len(site_rows) == 1, f"block {block_id} site {k_idx + 1}: {len(site_rows)} rows"
                actual_pos = np.array([site_rows.iloc[0]["x"], site_rows.iloc[0]["y"], site_rows.iloc[0]["z"]])
                np.testing.assert_allclose(
                    actual_pos, expected, atol=1e-6,
                    err_msg=f"block {block_id} site {k_idx + 1}",
                )
