"""Tests for the Surfaces page wiring: LOAD_OPS / OPERATIONS registries,
has_curvatures flag, and the surfaceview vertex-field helper.

The page itself is exercised in the live Dash app; here we cover the pure
helpers and registry interactions, per guideline §2.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from cryocat.app.suite.pages import pstructure
from cryocat.app.components import surface_registry as sr
from cryocat.app.components.surfaceview import (
    _vertex_field, _mesh_traces, _build_figure, COLOR_BY_OPTIONS,
)
from cryocat.analysis.structure import PleomorphicSurface
from cryocat.core.surface import Mesh, OrientedPointCloud


@pytest.fixture(autouse=True)
def _reset_registry():
    sr.clear_registry()
    yield
    sr.clear_registry()


@pytest.fixture
def tiny_mesh_psurf():
    """A 4-vertex tetrahedron mesh wrapped as PleomorphicSurface."""
    m = Mesh()
    m.vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float
    )
    m.faces = np.array(
        [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]], dtype=np.int32
    )
    return PleomorphicSurface(m)


# ── Registry layout ──────────────────────────────────────────────────────────

def test_load_ops_includes_vtp_loader():
    """The VTP loader moved into LOAD_OPS during the Loading/Operations split."""
    assert "mesh_vtp" in pstructure.LOAD_OPS


def test_operations_includes_phase_2a_mesh_ops():
    """The 4 Phase-2a mesh-only ops are registered in OPERATIONS."""
    for op_id in ("cleanup_mesh", "smooth", "compute_curvatures",
                  "surface_area"):
        assert op_id in pstructure.OPERATIONS, f"missing op: {op_id}"


def test_compute_curvatures_marked_field_source():
    """The op's kind drives the runtime branch that flips has_curvatures."""
    assert pstructure.OPERATIONS["compute_curvatures"]["kind"] == "field-source"


def test_surface_area_marked_scalar():
    """Scalar kind reports the result to the op status instead of the pool."""
    assert pstructure.OPERATIONS["surface_area"]["kind"] == "scalar"


def test_smooth_marked_inplace():
    """smooth returns None / mutates the mesh; kind reflects that."""
    assert pstructure.OPERATIONS["smooth"]["kind"] == "unary-inplace"


def test_mesh_only_helper_rejects_point_cloud(tiny_mesh_psurf):
    """The _mesh_only gate returns None for non-mesh surfaces."""
    assert pstructure._mesh_only(tiny_mesh_psurf) is tiny_mesh_psurf.surface
    opc = OrientedPointCloud()
    opc.vertices = np.zeros((1, 3))
    opc.normals = np.array([[0, 0, 1.0]])
    assert pstructure._mesh_only(PleomorphicSurface(opc)) is None
    assert pstructure._mesh_only(None) is None


def test_compute_curvatures_method_for_returns_none_on_point_cloud():
    """The mesh-only op binds to None when the selected surface is a cloud."""
    opc = OrientedPointCloud()
    opc.vertices = np.zeros((1, 3))
    opc.normals = np.array([[0, 0, 1.0]])
    bound = pstructure.OPERATIONS["compute_curvatures"]["method_for"](
        PleomorphicSurface(opc)
    )
    assert bound is None


# ── has_curvatures handle flag ───────────────────────────────────────────────

def test_make_handle_no_curvatures_by_default(tiny_mesh_psurf):
    h = sr.make_handle(tiny_mesh_psurf, label="x")
    assert h["has_curvatures"] is False


def test_make_handle_flips_when_curvatures_present(tiny_mesh_psurf):
    """Populating _mean_curvature manually mimics what compute_curvatures does."""
    tiny_mesh_psurf.surface._mean_curvature = np.zeros(4)
    tiny_mesh_psurf.surface._gaussian_curvature = np.zeros(4)
    tiny_mesh_psurf.surface._principal_curvatures = np.zeros((4, 2))
    h = sr.make_handle(tiny_mesh_psurf, label="curved")
    assert h["has_curvatures"] is True


def test_make_handle_point_cloud_never_has_curvatures():
    opc = OrientedPointCloud()
    opc.vertices = np.zeros((3, 3))
    opc.normals = np.array([[0, 0, 1.0]] * 3)
    h = sr.make_handle(PleomorphicSurface(opc), label="cloud")
    assert h["has_curvatures"] is False


# ── surfaceview color-by helpers ─────────────────────────────────────────────

def test_color_by_options_have_expected_values():
    """Field names must match plot_vtp_mesh(color_by=...) for parity."""
    values = {opt["value"] for opt in COLOR_BY_OPTIONS}
    assert {
        "none", "mean_curvature", "gaussian_curvature",
        "k1", "k2", "curvature_anisotropy",
    } == values


def test_vertex_field_returns_none_without_curvatures(tiny_mesh_psurf):
    """No fields populated -> no intensity available -> None."""
    assert _vertex_field(tiny_mesh_psurf, "mean_curvature") is None


def test_vertex_field_returns_none_for_point_cloud():
    opc = OrientedPointCloud()
    opc.vertices = np.zeros((3, 3))
    opc.normals = np.array([[0, 0, 1.0]] * 3)
    assert _vertex_field(PleomorphicSurface(opc), "mean_curvature") is None


def test_vertex_field_returns_none_when_color_by_is_none(tiny_mesh_psurf):
    tiny_mesh_psurf.surface._mean_curvature = np.arange(4, dtype=float)
    tiny_mesh_psurf.surface._principal_curvatures = np.zeros((4, 2))
    assert _vertex_field(tiny_mesh_psurf, "none") is None


def test_vertex_field_returns_mean_curvature_array(tiny_mesh_psurf):
    """mean_curvature getter is forwarded through the facade."""
    tiny_mesh_psurf.surface._mean_curvature = np.arange(4, dtype=float)
    tiny_mesh_psurf.surface._principal_curvatures = np.column_stack(
        [np.arange(4, dtype=float), np.arange(4, dtype=float) * 0.5]
    )
    tiny_mesh_psurf.surface._gaussian_curvature = (
        tiny_mesh_psurf.surface._principal_curvatures[:, 0]
        * tiny_mesh_psurf.surface._principal_curvatures[:, 1]
    )
    out = _vertex_field(tiny_mesh_psurf, "mean_curvature")
    assert out is not None
    assert out.shape == (4,)
    assert np.allclose(out, np.arange(4, dtype=float))


def test_vertex_field_k1_and_k2_pull_from_principal_curvatures(tiny_mesh_psurf):
    tiny_mesh_psurf.surface._mean_curvature = np.zeros(4)
    tiny_mesh_psurf.surface._principal_curvatures = np.column_stack(
        [np.arange(4, dtype=float), -np.arange(4, dtype=float)]
    )
    tiny_mesh_psurf.surface._gaussian_curvature = np.zeros(4)
    np.testing.assert_array_equal(
        _vertex_field(tiny_mesh_psurf, "k1"), np.arange(4, dtype=float)
    )
    np.testing.assert_array_equal(
        _vertex_field(tiny_mesh_psurf, "k2"), -np.arange(4, dtype=float)
    )


def test_vertex_field_anisotropy_in_unit_interval(tiny_mesh_psurf):
    tiny_mesh_psurf.surface._mean_curvature = np.zeros(4)
    tiny_mesh_psurf.surface._principal_curvatures = np.column_stack(
        [np.array([1.0, 2.0, 3.0, 0.0]), np.array([1.0, -2.0, 0.0, 0.0])]
    )
    tiny_mesh_psurf.surface._gaussian_curvature = np.zeros(4)
    out = _vertex_field(tiny_mesh_psurf, "curvature_anisotropy")
    assert out is not None
    assert np.all((out >= 0) & (out <= 1))


# ── _mesh_traces intensity passthrough ───────────────────────────────────────

def test_mesh_traces_flat_color_without_intensity(tiny_mesh_psurf):
    traces = _mesh_traces(tiny_mesh_psurf, color="#abcdef", name="x", selected=True)
    assert len(traces) == 1
    assert traces[0].color == "#abcdef"
    assert traces[0].intensity is None


def test_mesh_traces_uses_intensity_when_provided(tiny_mesh_psurf):
    intensity = np.array([0.1, 0.2, 0.3, 0.4])
    traces = _mesh_traces(
        tiny_mesh_psurf, color="#abcdef", name="x", selected=True,
        intensity=intensity,
    )
    assert len(traces) == 1
    np.testing.assert_array_equal(traces[0].intensity, intensity)
    assert traces[0].intensitymode == "vertex"
    assert traces[0].cmin is not None and traces[0].cmax is not None


def test_mesh_traces_falls_back_to_flat_when_intensity_shape_mismatched(tiny_mesh_psurf):
    """Wrong-length intensity arrays must not be silently accepted."""
    traces = _mesh_traces(
        tiny_mesh_psurf, color="#abcdef", name="x", selected=False,
        intensity=np.array([0.1, 0.2]),  # mesh has 4 vertices
    )
    assert traces[0].color == "#abcdef"
    assert traces[0].intensity is None


# ── End-to-end build_figure with handles and color_by ────────────────────────

def test_build_figure_applies_color_by_only_to_selected_curvatured_mesh(tiny_mesh_psurf):
    tiny_mesh_psurf.surface._mean_curvature = np.arange(4, dtype=float)
    tiny_mesh_psurf.surface._principal_curvatures = np.zeros((4, 2))
    tiny_mesh_psurf.surface._gaussian_curvature = np.zeros(4)
    sid = sr.register_surface(tiny_mesh_psurf)
    handle = sr.make_handle(tiny_mesh_psurf, label="curved")
    fig = _build_figure(
        handles={sid: handle},
        selected_id=sid,
        gs=None,
        color_by="mean_curvature",
    )
    assert len(fig.data) == 1
    trace = fig.data[0]
    assert trace.intensity is not None
    assert trace.intensitymode == "vertex"


def test_build_figure_skips_color_by_when_no_curvatures(tiny_mesh_psurf):
    sid = sr.register_surface(tiny_mesh_psurf)  # no curvatures populated
    handle = sr.make_handle(tiny_mesh_psurf, label="bare")
    fig = _build_figure(
        handles={sid: handle},
        selected_id=sid,
        gs=None,
        color_by="mean_curvature",
    )
    assert fig.data[0].intensity is None


# ── Phase 2b: backend signature tightenings ──────────────────────────────────

import inspect


def test_extract_region_signature_drops_var_kwargs():
    """extract_region no longer takes **kwargs; preserve_curvatures is explicit."""
    sig = inspect.signature(PleomorphicSurface.extract_region)
    assert "preserve_curvatures" in sig.parameters
    assert not any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )


def test_extract_region_element_is_literal():
    """element annotation is the inline Literal of the three valid values."""
    sig = inspect.signature(PleomorphicSurface.extract_region)
    ann = sig.parameters["element"].annotation
    # surface.py uses `from __future__ import annotations` so annotation is a string;
    # check for the literal-shaped text.
    assert "Literal" in str(ann) or "triangles" in str(ann)


def test_intersection_data_surface_seeds_is_literal():
    sig = inspect.signature(PleomorphicSurface.intersection_data)
    assert "Literal" in str(sig.parameters["surface_seeds"].annotation)
    assert "Literal" in str(sig.parameters["surface_element"].annotation)


def test_ray_intersections_target_orientation_is_literal_or_callable():
    sig = inspect.signature(PleomorphicSurface.ray_intersections)
    ann = str(sig.parameters["target_orientation"].annotation)
    assert "Literal" in ann
    assert "Callable" in ann


def test_extract_region_triangle_path(tiny_mesh_psurf):
    """End-to-end: extract the first triangle as a 3-vertex submesh."""
    sub = tiny_mesh_psurf.extract_region(
        np.array([0]), element="triangles"
    )
    assert sub.is_mesh
    assert len(sub.surface.faces) == 1


def test_extract_region_rejects_unknown_element(tiny_mesh_psurf):
    with pytest.raises((ValueError, TypeError)):
        tiny_mesh_psurf.extract_region(np.array([0]), element="bogus")


# ── Phase 2b: helpers ────────────────────────────────────────────────────────

from cryocat.app.suite.pages._pstructure_intersect import (
    motl_from_rows,
    motl_rows_to_rays,
    subset_motl_rows,
    hits_summary_dataframe,
)


def _toy_motl_rows(n=4):
    """Minimal pool-rows shape (the columns cryomotl.Motl expects)."""
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
    return rows


def test_motl_from_rows_empty_raises():
    with pytest.raises(ValueError):
        motl_from_rows([])


def test_motl_rows_to_rays_shape_and_origins():
    rows = _toy_motl_rows(n=3)
    rays = motl_rows_to_rays(rows, pixel_size=2.0, reverse_direction=False)
    assert rays.shape == (3, 6)
    # Origins should match coords * pixel_size (no shifts in fixture).
    np.testing.assert_array_almost_equal(rays[:, 0], np.arange(3) * 2.0)
    # Default zero rotations -> z-normal points along +z (construct_rays
    # may scale by ray_length; check the SIGN and axis only).
    assert np.all(rays[:, 5] > 0)
    np.testing.assert_array_almost_equal(rays[:, 3], 0.0)
    np.testing.assert_array_almost_equal(rays[:, 4], 0.0)


def test_motl_rows_to_rays_reverse_direction_flips_z():
    rows = _toy_motl_rows(n=2)
    rays = motl_rows_to_rays(rows, pixel_size=1.0, reverse_direction=True)
    # Reversed normals point along -z (regardless of construct_rays scale).
    assert np.all(rays[:, 5] < 0)


def test_subset_motl_rows_keeps_order_and_dedupes():
    rows = _toy_motl_rows(n=5)
    out = subset_motl_rows(rows, [3, 1, 1, 3, 7])  # 7 out of range, 1 & 3 dup
    assert [r["subtomo_id"] for r in out] == [4, 2]  # order = first occurrence


def test_subset_motl_rows_empty_input_yields_empty():
    rows = _toy_motl_rows(n=3)
    assert subset_motl_rows(rows, []) == []


def test_hits_summary_dataframe_returns_region_summary_when_present():
    rs = pd.DataFrame({"region": ["a", "b"], "n_hits": [3, 7]})
    out = hits_summary_dataframe({"region_summary": rs})
    assert isinstance(out, pd.DataFrame)
    assert list(out["n_hits"]) == [3, 7]


def test_hits_summary_dataframe_falls_back_to_hits_overview():
    hits = pd.DataFrame({"distance_nm": [1.0, 2.0, 3.0]})
    out = hits_summary_dataframe({"hits": hits, "region_summary": None})
    assert "n_hits" in out.columns
    assert int(out.iloc[0]["n_hits"]) == 3


def test_hits_summary_dataframe_no_data_returns_empty_frame():
    out = hits_summary_dataframe({})
    assert isinstance(out, pd.DataFrame)
    assert out.empty


# ── Phase 2b: result-to-store snapshot helper ─────────────────────────────────

from cryocat.app.suite.pages import pstructure as ps


def test_result_to_store_serialises_dataframes_to_records():
    snap = ps._result_to_store(
        {
            "hits": pd.DataFrame({"source_id": [0, 1], "distance_nm": [0.5, 1.5]}),
            "region_summary": pd.DataFrame({"region": ["x"], "n_hits": [2]}),
            "regions": {"x": np.array([0, 1, 2])},
        },
        particle_ids_seen=[0, 1],
    )
    assert snap["hits_records"] == [
        {"source_id": 0, "distance_nm": 0.5},
        {"source_id": 1, "distance_nm": 1.5},
    ]
    assert snap["region_summary_records"][0]["n_hits"] == 2
    assert snap["regions"]["x"] == [0, 1, 2]
    assert snap["hit_source_ids"] == [0, 1]


# ── Phase 3: ParametricSurface rename + type tightening ──────────────────────

from cryocat.analysis.structure import ParametricSurface
from cryocat.app.components import parametric_registry as pr


def test_parametric_surface_renames_feature_id_to_column_name():
    """Constructor + classmethods now expose column_name, not feature_id."""
    sig_init = inspect.signature(ParametricSurface.__init__)
    sig_from_motl = inspect.signature(ParametricSurface.from_motl)
    sig_from_csv = inspect.signature(ParametricSurface.from_csv)
    for sig in (sig_init, sig_from_motl, sig_from_csv):
        assert "column_name" in sig.parameters
        assert "feature_id" not in sig.parameters


def test_parametric_surface_stores_column_name_attribute():
    """Instance attribute renamed from feature_id to column_name."""

    class _FakeQM:
        dict = {}

    p = ParametricSurface(_FakeQM(), column_name="object_id")
    assert hasattr(p, "column_name")
    assert p.column_name == "object_id"
    assert not hasattr(p, "feature_id")


def test_compute_point_surface_distance_uses_store_column_name():
    sig = inspect.signature(ParametricSurface.compute_point_surface_distance)
    assert "store_column_name" in sig.parameters
    assert "store_id" not in sig.parameters


def test_compute_normals_angle_uses_store_column_name():
    sig = inspect.signature(ParametricSurface.compute_normals_angle)
    assert "store_column_name" in sig.parameters
    assert "store_id" not in sig.parameters


def test_surface_type_annotation_is_literal_ellipsoid():
    """surface_type is narrowed to Literal['ellipsoid']."""
    sig = inspect.signature(ParametricSurface.from_motl)
    ann = str(sig.parameters["surface_type"].annotation)
    assert "Literal" in ann
    assert "ellipsoid" in ann


# ── Phase 3: parametric_registry ─────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _reset_parametric_slot():
    pr.clear_active_fit()
    yield
    pr.clear_active_fit()


class _StubPSurf:
    """Stand-in PleomorphicSurface-like object for registry tests."""

    def __init__(self, n_quadrics=3, column_name="object_id"):
        self.column_name = column_name

        class _Q:
            dict = {(1, i): None for i in range(n_quadrics)}

        self.quadrics = _Q()


def test_set_active_fit_round_trip():
    s = _StubPSurf(n_quadrics=2)
    handle = pr.set_active_fit(s, source="motl:abc")
    assert pr.get_active_fit() is s
    assert handle["n_quadrics"] == 2
    assert handle["column_name"] == "object_id"
    assert handle["source"] == "motl:abc"


def test_set_active_fit_replaces_previous():
    a = _StubPSurf(n_quadrics=1, column_name="object_id")
    b = _StubPSurf(n_quadrics=5, column_name="tomo_id")
    pr.set_active_fit(a, source="motl:a")
    pr.set_active_fit(b, source="motl:b")
    assert pr.get_active_fit() is b


def test_clear_active_fit():
    pr.set_active_fit(_StubPSurf(), source="x")
    pr.clear_active_fit()
    assert pr.get_active_fit() is None


def test_make_handle_reflects_quadrics_dict():
    s = _StubPSurf(n_quadrics=7, column_name="tomo_id")
    h = pr.make_handle(s, source="csv:/tmp/x.csv")
    assert h["n_quadrics"] == 7
    assert h["column_name"] == "tomo_id"
    assert h["surface_type"] == "ellipsoid"


# ── Phase 3: page-side helpers ───────────────────────────────────────────────


def test_motl_from_pool_rows_round_trip():
    """The Motl-rebuild helper handles both populated and empty inputs."""
    rows = _toy_motl_rows(n=3)
    motl = ps._motl_from_pool_rows({"m1": rows}, "m1")
    assert motl is not None
    assert len(motl.df) == 3
    # Empty / missing returns None.
    assert ps._motl_from_pool_rows({"m1": []}, "m1") is None
    assert ps._motl_from_pool_rows({}, "m1") is None
    assert ps._motl_from_pool_rows({"m1": rows}, None) is None


def test_parametric_ops_table_keys_and_kinds():
    """Every parametric op in OPERATIONS has the structural fields the dispatcher uses."""
    expected = {
        "param_distance", "param_assign_distance", "param_assign_intersection",
        "param_intersection", "param_normal_angle", "param_clean_normals",
        "param_clean_radius", "param_assign_mask", "param_oversample_spherical",
    }
    parametric_ops = {
        op_id: op for op_id, op in ps.OPERATIONS.items()
        if op.get("category") == "parametric"
    }
    assert set(parametric_ops) == expected
    for op_id, op in parametric_ops.items():
        for key in ("label", "method_name", "needs_active_fit",
                    "result_kind", "extra_pickers"):
            assert key in op, f"{op_id} missing {key}"


def test_static_ops_do_not_need_active_fit():
    """assign_affiliation_mask_based + create_spherical_oversampling are static."""
    assert ps.OPERATIONS["param_assign_mask"]["needs_active_fit"] is False
    assert ps.OPERATIONS["param_oversample_spherical"]["needs_active_fit"] is False
    assert ps.OPERATIONS["param_assign_mask"]["extra_pickers"] == ["object_motl"]


def test_intersection_op_is_dataframe_kind():
    """compute_intersection's result is rendered as a table, not pushed to pool."""
    assert ps.OPERATIONS["param_intersection"]["result_kind"] == "dataframe"
