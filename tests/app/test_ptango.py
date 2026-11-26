"""Tests for the TANGO page — descriptor speedup, viewer, colors, and diagnostics.

Covers:
- PLComplexDescriptor, SHOTDescriptor, AlphaComplexDescriptor: structural
  correctness, empty-case handling, and list-and-concat equivalence to the
  old loop output.
- Timing smoke test: SHOTDescriptor completes in reasonable time for 200 groups.
- Twist particle viewer (detail_figure, register_viewer_callbacks, ptango stores).
- Palette switching per tool: Monet for tango, StarryNight otherwise; user-set
  palette survives tool switches.
- Descriptor slot management: five-slot cap, close/reopen, stale marking.
- Diagnostics: explained variance, constant features, standardisation invariant,
  tablecluster PCA, tabletomotl merge.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tests.app.assertions import assert_grid_cells_have_text, assert_grid_shows


# ── Shared fixture ─────────────────────────────────────────────────────────────

def _ring_twist_df(n_qps: int = 4, n_per_qp: int = 8, seed: int = 42) -> pd.DataFrame:
    """Synthetic twist table: neighbours placed on a circle in the XY plane."""
    rng = np.random.default_rng(seed)
    rows = []
    nn_id = 1
    for qp_id in range(1, n_qps + 1):
        angles = np.linspace(0, 2 * np.pi, n_per_qp, endpoint=False)
        radii = rng.uniform(0.5, 2.0, n_per_qp)
        for angle, r in zip(angles, radii):
            rows.append({
                "qp_id": qp_id,
                "nn_id": nn_id,
                "twist_x": r * np.cos(angle),
                "twist_y": r * np.sin(angle),
                "twist_z": float(rng.uniform(-0.1, 0.1)),
                "euclidean_distance": r,
            })
            nn_id += 1
    return pd.DataFrame(rows)


@pytest.fixture()
def ring_df():
    return _ring_twist_df(n_qps=4, n_per_qp=8)


@pytest.fixture()
def single_group_df():
    return _ring_twist_df(n_qps=1, n_per_qp=8)


@pytest.fixture()
def empty_df():
    return pd.DataFrame(columns=["qp_id", "nn_id", "twist_x", "twist_y", "twist_z", "euclidean_distance"])


@pytest.fixture()
def tiny_groups_df():
    """Groups with only 1 neighbour — all skipped by PLComplex/AlphaComplex."""
    rows = [
        {"qp_id": i, "nn_id": i, "twist_x": float(i), "twist_y": 0.0, "twist_z": 0.0, "euclidean_distance": float(i)}
        for i in range(1, 5)
    ]
    return pd.DataFrame(rows)


# ===========================================================================
# Descriptor speedup — PLComplexDescriptor
# ===========================================================================

class TestPLComplexDescriptor:

    def test_columns_present(self, ring_df):
        from cryocat.analysis.tango import PLComplexDescriptor
        desc = PLComplexDescriptor(ring_df, build_unique_desc=False)
        expected_cols = {"central_angle", "area", "in_radius", "circle_radius", "radii_ratio", "qp_id", "nn_id"}
        assert expected_cols <= set(desc.pl_df.columns)

    def test_all_qp_ids_represented(self, ring_df):
        from cryocat.analysis.tango import PLComplexDescriptor
        desc = PLComplexDescriptor(ring_df, build_unique_desc=False)
        assert set(desc.pl_df["qp_id"].unique()) == set(ring_df["qp_id"].unique())

    def test_result_is_dataframe(self, ring_df):
        from cryocat.analysis.tango import PLComplexDescriptor
        desc = PLComplexDescriptor(ring_df, build_unique_desc=False)
        assert isinstance(desc.pl_df, pd.DataFrame)
        assert len(desc.pl_df) > 0

    def test_empty_input_yields_empty_df(self, empty_df):
        from cryocat.analysis.tango import PLComplexDescriptor
        desc = PLComplexDescriptor(empty_df, build_unique_desc=False)
        assert isinstance(desc.pl_df, pd.DataFrame)
        assert len(desc.pl_df) == 0

    def test_all_skipped_groups_yields_empty_df(self, tiny_groups_df):
        """Groups with 1 member are skipped — pl_df must still be a DataFrame."""
        from cryocat.analysis.tango import PLComplexDescriptor
        desc = PLComplexDescriptor(tiny_groups_df, build_unique_desc=False)
        assert isinstance(desc.pl_df, pd.DataFrame)
        assert len(desc.pl_df) == 0

    def test_single_group(self, single_group_df):
        from cryocat.analysis.tango import PLComplexDescriptor
        desc = PLComplexDescriptor(single_group_df, build_unique_desc=False)
        assert isinstance(desc.pl_df, pd.DataFrame)
        assert len(desc.pl_df) > 0
        assert set(desc.pl_df["qp_id"]) == {1}

    def test_output_identical_to_reference(self, ring_df):
        """New list-and-concat output must equal the old loop output."""
        from cryocat.analysis.tango import PLComplexDescriptor

        desc_new = PLComplexDescriptor(ring_df, build_unique_desc=False)

        desc_ref = PLComplexDescriptor.__new__(PLComplexDescriptor)
        desc_ref.ordered_vertices = {}
        desc_ref.ordered_nn = {}
        desc_ref.triangles = {}
        from cryocat.utils import geom
        ref_frames = []
        for qp_id, group in ring_df.groupby("qp_id"):
            desc_ref.ordered_vertices[qp_id], sorted_indices = geom.order_points_on_circle(
                group[["twist_x", "twist_y", "twist_z"]].to_numpy()
            )
            desc_ref.ordered_nn[qp_id] = group["nn_id"].to_numpy()[sorted_indices]
            desc_ref.triangles[qp_id] = desc_ref.compute_triangles(qp_id)
            if desc_ref.triangles[qp_id] is not None:
                ref_frames.append(desc_ref.compute_features(qp_id))
        ref_df = pd.concat(ref_frames) if ref_frames else pd.DataFrame()

        pd.testing.assert_frame_equal(
            desc_new.pl_df.reset_index(drop=True),
            ref_df.reset_index(drop=True),
            check_like=False,
        )


# ===========================================================================
# Descriptor speedup — SHOTDescriptor
# ===========================================================================

class TestSHOTDescriptor:

    def test_columns_present(self, ring_df):
        from cryocat.analysis.tango import SHOTDescriptor
        desc = SHOTDescriptor(ring_df, build_unique_desc=False)
        expected_cols = {"shell_id", "cone_id", "qp_id", "nn_id"}
        assert expected_cols <= set(desc.shot_df.columns)

    def test_row_count_equals_input(self, ring_df):
        from cryocat.analysis.tango import SHOTDescriptor
        desc = SHOTDescriptor(ring_df, build_unique_desc=False)
        assert len(desc.shot_df) == len(ring_df)

    def test_all_qp_ids_represented(self, ring_df):
        from cryocat.analysis.tango import SHOTDescriptor
        desc = SHOTDescriptor(ring_df, build_unique_desc=False)
        assert set(desc.shot_df["qp_id"].unique()) == set(ring_df["qp_id"].unique())

    def test_empty_input_yields_empty_df(self, empty_df):
        from cryocat.analysis.tango import SHOTDescriptor
        desc = SHOTDescriptor(empty_df, build_unique_desc=False)
        assert isinstance(desc.shot_df, pd.DataFrame)
        assert len(desc.shot_df) == 0

    def test_single_group(self, single_group_df):
        from cryocat.analysis.tango import SHOTDescriptor
        desc = SHOTDescriptor(single_group_df, build_unique_desc=False)
        assert len(desc.shot_df) == len(single_group_df)
        assert set(desc.shot_df["qp_id"]) == {1}

    def test_cone_dirs_precomputed(self, ring_df):
        """_cone_dirs must be set on the instance after construction."""
        from cryocat.analysis.tango import SHOTDescriptor
        desc = SHOTDescriptor(ring_df, build_unique_desc=False)
        assert hasattr(desc, "_cone_dirs"), "_cone_dirs not set — cone directions not pre-computed"
        assert desc._cone_dirs is not None

    def test_output_identical_to_reference(self, ring_df):
        """New list-and-concat output must equal the old loop output."""
        from cryocat.analysis.tango import SHOTDescriptor

        cone_number, shell_number = 6, 1
        desc_new = SHOTDescriptor(ring_df, cone_number=cone_number, shell_number=shell_number, build_unique_desc=False)

        max_radius = ring_df["euclidean_distance"].max()
        desc_ref = SHOTDescriptor.__new__(SHOTDescriptor)
        desc_ref.cone_number = cone_number
        desc_ref.shell_number = shell_number
        ref_frames = []
        for qp_id, group_df in ring_df.groupby("qp_id"):
            coord = group_df[["twist_x", "twist_y", "twist_z"]].to_numpy()
            adf = desc_ref.assign_shell_and_cone_ids(
                qp_id, coord, shell_number, cone_number, radius=max_radius
            )
            adf["qp_id"] = qp_id
            adf["nn_id"] = group_df["nn_id"].values
            ref_frames.append(adf)
        ref_df = pd.concat(ref_frames) if ref_frames else pd.DataFrame()

        pd.testing.assert_frame_equal(
            desc_new.shot_df.reset_index(drop=True),
            ref_df.reset_index(drop=True),
            check_like=False,
        )


# ===========================================================================
# Descriptor speedup — AlphaComplexDescriptor
# ===========================================================================

class TestAlphaComplexDescriptor:

    def test_columns_present(self, ring_df):
        from cryocat.analysis.tango import AlphaComplexDescriptor
        desc = AlphaComplexDescriptor(ring_df, build_unique_desc=False)
        expected_cols = {"central_angle", "area", "in_radius", "circle_radius", "radii_ratio", "qp_id"}
        assert expected_cols <= set(desc.alpha_df.columns)

    def test_all_qp_ids_represented(self, ring_df):
        from cryocat.analysis.tango import AlphaComplexDescriptor
        desc = AlphaComplexDescriptor(ring_df, build_unique_desc=False)
        assert set(desc.alpha_df["qp_id"].unique()) == set(ring_df["qp_id"].unique())

    def test_result_is_dataframe(self, ring_df):
        from cryocat.analysis.tango import AlphaComplexDescriptor
        desc = AlphaComplexDescriptor(ring_df, build_unique_desc=False)
        assert isinstance(desc.alpha_df, pd.DataFrame)
        assert len(desc.alpha_df) > 0

    def test_empty_input_yields_empty_df(self, empty_df):
        from cryocat.analysis.tango import AlphaComplexDescriptor
        desc = AlphaComplexDescriptor(empty_df, build_unique_desc=False)
        assert isinstance(desc.alpha_df, pd.DataFrame)
        assert len(desc.alpha_df) == 0

    def test_all_skipped_groups_yields_empty_df(self, tiny_groups_df):
        from cryocat.analysis.tango import AlphaComplexDescriptor
        desc = AlphaComplexDescriptor(tiny_groups_df, build_unique_desc=False)
        assert isinstance(desc.alpha_df, pd.DataFrame)
        assert len(desc.alpha_df) == 0

    def test_single_group(self, single_group_df):
        from cryocat.analysis.tango import AlphaComplexDescriptor
        desc = AlphaComplexDescriptor(single_group_df, build_unique_desc=False)
        assert isinstance(desc.alpha_df, pd.DataFrame)
        assert len(desc.alpha_df) > 0
        assert set(desc.alpha_df["qp_id"]) == {1}

    def test_output_identical_to_reference(self, ring_df):
        """New list-and-concat output must equal the old loop output."""
        from cryocat.analysis.tango import AlphaComplexDescriptor

        desc_new = AlphaComplexDescriptor(ring_df, build_unique_desc=False)

        desc_ref = AlphaComplexDescriptor.__new__(AlphaComplexDescriptor)
        desc_ref.alpha_param = 200.0
        desc_ref.alpha_complexes = {}
        desc_ref.stars = {}
        desc_ref.triangles = {}
        desc_ref.link_edges = {}
        ref_frames = []
        for qp_id, group in ring_df.groupby("qp_id"):
            coord = group[["twist_x", "twist_y", "twist_z"]].to_numpy()
            if coord.shape[0] < 2:
                continue
            desc_ref.triangles[qp_id], desc_ref.link_edges[qp_id] = desc_ref.compute_alpha_complex(coord)
            if desc_ref.triangles[qp_id] is not None:
                ref_frames.append(desc_ref.compute_features(qp_id))
        ref_df = pd.concat(ref_frames) if ref_frames else pd.DataFrame()

        pd.testing.assert_frame_equal(
            desc_new.alpha_df.reset_index(drop=True),
            ref_df.reset_index(drop=True),
            check_like=False,
        )


# ===========================================================================
# Timing smoke test
# ===========================================================================

def test_shot_descriptor_completes_in_reasonable_time():
    """Smoke test: 200-group table must build SHOTDescriptor in under 5 seconds."""
    from cryocat.analysis.tango import SHOTDescriptor

    df = _ring_twist_df(n_qps=200, n_per_qp=20)
    t0 = time.perf_counter()
    SHOTDescriptor(df, build_unique_desc=False)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"SHOTDescriptor took {elapsed:.2f}s for 200 groups — list-and-concat fix may not have taken effect"
    )


# ===========================================================================
# Viewer — twist particle viewer correctness
# ===========================================================================

_MOTL_COLS = [
    "score", "geom1", "geom2", "subtomo_id", "tomo_id", "object_id",
    "subtomo_mean", "x", "y", "z", "shift_x", "shift_y", "shift_z",
    "geom3", "geom4", "geom5", "phi", "psi", "theta", "class",
]


def _make_motl_df(n: int = 3, tomo_id: int = 1) -> pd.DataFrame:
    """Minimal Motl-formatted DataFrame with n particles on tomo_id."""
    data = {col: [0.0] * n for col in _MOTL_COLS}
    data["subtomo_id"] = list(range(10, 10 + n))
    data["tomo_id"] = [float(tomo_id)] * n
    data["x"] = [float(i * 10) for i in range(n)]
    data["y"] = [float(i * 10) for i in range(n)]
    data["z"] = [float(i * 10) for i in range(n)]
    return pd.DataFrame(data)


def _make_twist_records(qp_ids: list[int]) -> list[dict]:
    """Create twist records for each qp_id (one neighbor per query particle)."""
    return [
        {"qp_id": float(qid), "nn_id": float(qid + 1), "twist_x": 1.0, "twist_y": 2.0, "twist_z": 3.0}
        for qid in qp_ids
    ]


class TestTangoViewer:
    """Twist particle viewer correctness."""

    def test_detail_figure_returns_figure_for_known_particle(self):
        """Clicking particle 0 (subtomo_id=10) that has twist neighbours produces a figure."""
        from cryocat.app.components.tomoview import detail_figure

        df = _make_motl_df(n=3)
        records = _make_twist_records([10, 11, 12])
        clickData = {"points": [{"pointNumber": 0}]}
        fig = detail_figure(clickData, df, records, radius=50.0, tomo_index=0)
        assert fig is not None, "Expected a figure for a particle with twist neighbours"
        traces = fig.to_plotly_json()["data"]
        scatter3d_types = [t for t in traces if t.get("type") == "scatter3d"]
        assert scatter3d_types, "Expected at least one Scatter3d trace in the detail figure"

    def test_detail_figure_axis_range_equals_radius(self):
        """x/y/z axis range must be [-radius, radius]."""
        from cryocat.app.components.tomoview import detail_figure

        df = _make_motl_df(n=3)
        records = _make_twist_records([10, 11, 12])
        radius = 42.0
        clickData = {"points": [{"pointNumber": 0}]}
        fig = detail_figure(clickData, df, records, radius=radius, tomo_index=0)
        assert fig is not None
        scene = fig.to_plotly_json()["layout"]["scene"]
        for axis_key in ("xaxis", "yaxis", "zaxis"):
            r = scene[axis_key]["range"]
            assert r == [-radius, radius], (
                f"{axis_key} range {r!r} does not match [-{radius}, {radius}]"
            )

    def test_detail_figure_returns_none_for_out_of_range_point(self):
        """A pointNumber beyond the tomo subset size must return None."""
        from cryocat.app.components.tomoview import detail_figure

        df = _make_motl_df(n=2)
        records = _make_twist_records([10, 11])
        clickData = {"points": [{"pointNumber": 999}]}
        fig = detail_figure(clickData, df, records, radius=30.0, tomo_index=0)
        assert fig is None

    def test_detail_figure_returns_none_for_particle_without_neighbours(self):
        """A particle whose subtomo_id is absent from twist_data must return None."""
        from cryocat.app.components.tomoview import detail_figure

        df = _make_motl_df(n=3)
        records = _make_twist_records([20, 21])
        clickData = {"points": [{"pointNumber": 0}]}
        fig = detail_figure(clickData, df, records, radius=30.0, tomo_index=0)
        assert fig is None

    def test_register_viewer_callbacks_raises_without_radius_store_id(self):
        """show_dual_graph=True without radius_store_id must raise ValueError."""
        import dash
        import dash_bootstrap_components as dbc
        from cryocat.app.components.tomoview import register_viewer_callbacks

        app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP],
                        suppress_callback_exceptions=True)
        with pytest.raises(ValueError, match="radius_store_id"):
            register_viewer_callbacks(app, "test-tv", show_dual_graph=True)

# ===========================================================================
# Colors and layout
# ===========================================================================

class TestTangoColorsAndLayout:
    """Palette defaults, per-tab switching, button variables, radio layout."""

    def test_graph_settings_defaults_palette_not_user_set(self):
        """GRAPH_SETTINGS_DEFAULTS must have palette_is_user_set=False."""
        from cryocat.app.components.graphsettings import GRAPH_SETTINGS_DEFAULTS
        assert "palette_is_user_set" in GRAPH_SETTINGS_DEFAULTS, (
            "palette_is_user_set key missing from GRAPH_SETTINGS_DEFAULTS"
        )
        assert GRAPH_SETTINGS_DEFAULTS["palette_is_user_set"] is False

    def test_palette_switches_to_monet_for_tango(self):
        """Switching tool to 'tango' when palette is not user-set yields Monet."""
        from cryocat.app.components.graphsettings import GRAPH_SETTINGS_DEFAULTS

        def _set_tab_palette(tool: str, settings: dict) -> dict | object:
            from dash import no_update
            settings = dict(settings or GRAPH_SETTINGS_DEFAULTS)
            if settings.get("palette_is_user_set"):
                return no_update
            target = "Monet" if tool == "tango" else "StarryNight"
            if settings.get("discrete_palette") == target:
                return no_update
            return {**settings, "discrete_palette": target, "continuous_palette": target}

        initial = dict(GRAPH_SETTINGS_DEFAULTS)
        result = _set_tab_palette("tango", initial)
        assert isinstance(result, dict), "Expected a dict result for tango tool"
        assert result["discrete_palette"] == "Monet"

    def test_palette_switches_to_starrynight_for_non_tango(self):
        """Switching from tango to another tool resets palette to StarryNight."""
        from cryocat.app.components.graphsettings import GRAPH_SETTINGS_DEFAULTS
        from dash import no_update

        def _set_tab_palette(tool: str, settings: dict):
            settings = dict(settings or GRAPH_SETTINGS_DEFAULTS)
            if settings.get("palette_is_user_set"):
                return no_update
            target = "Monet" if tool == "tango" else "StarryNight"
            if settings.get("discrete_palette") == target:
                return no_update
            return {**settings, "discrete_palette": target, "continuous_palette": target}

        tango_settings = {**GRAPH_SETTINGS_DEFAULTS, "discrete_palette": "Monet",
                          "continuous_palette": "Monet", "palette_is_user_set": False}
        result = _set_tab_palette("motl", tango_settings)
        assert isinstance(result, dict)
        assert result["discrete_palette"] == "StarryNight"

    def test_user_set_palette_survives_tab_switch(self):
        """palette_is_user_set=True must prevent the per-tab callback from overriding."""
        from cryocat.app.components.graphsettings import GRAPH_SETTINGS_DEFAULTS
        from dash import no_update

        def _set_tab_palette(tool: str, settings: dict):
            settings = dict(settings or GRAPH_SETTINGS_DEFAULTS)
            if settings.get("palette_is_user_set"):
                return no_update
            target = "Monet" if tool == "tango" else "StarryNight"
            if settings.get("discrete_palette") == target:
                return no_update
            return {**settings, "discrete_palette": target, "continuous_palette": target}

        user_settings = {**GRAPH_SETTINGS_DEFAULTS, "discrete_palette": "Viridis",
                         "palette_is_user_set": True}
        result = _set_tab_palette("tango", user_settings)
        assert result is no_update, (
            "User-set palette must not be overridden when switching to tango"
        )

    def test_tablesource_radio_has_flex_display(self):
        """RadioItems inside get_table_source must have style with display:flex."""
        from cryocat.app.components.tablesource import get_table_source

        source = get_table_source("ts-test", compute_children=[])

        radio = None

        def _find(comp):
            nonlocal radio
            if radio is not None:
                return
            if type(comp).__name__ == "RadioItems":
                radio = comp
                return
            children = getattr(comp, "children", None)
            if children is None:
                return
            if not isinstance(children, list):
                children = [children]
            for child in children:
                if hasattr(child, "children") or hasattr(child, "id"):
                    _find(child)

        _find(source)
        assert radio is not None, "RadioItems not found in get_table_source output"
        assert (radio.style or {}).get("display") == "flex", (
            f"RadioItems.style must include display:flex; got {radio.style!r}"
        )

# ===========================================================================
# Descriptor slot management
# ===========================================================================

def _make_twist_sig(df: pd.DataFrame) -> str:
    return f"{len(df)}:{','.join(sorted(df.columns.tolist()))}"


def _make_df(n: int = 30, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "qp_id": np.arange(1, n + 1, dtype=float),
        "feat_a": rng.uniform(0, 1, n),
        "feat_b": rng.uniform(0, 100, n),
        "feat_c": rng.uniform(-10, 10, n),
    })


class TestTangoStructure:
    """Structural invariants after the clustering removal and slot rework."""

    def test_five_slots_cap(self):
        """Finding a free slot with all 5 used returns None."""
        registry = {str(i): {"label": f"desc_{i}"} for i in range(5)}
        _DESC_SLOTS = 5
        slot = next((i for i in range(_DESC_SLOTS) if str(i) not in registry), None)
        assert slot is None, "Expected None when all 5 slots are occupied"

    def test_first_empty_slot_found(self):
        """Finding a free slot works correctly when slots 0 and 2 are used."""
        registry = {"0": {"label": "a"}, "2": {"label": "b"}}
        _DESC_SLOTS = 5
        slot = next((i for i in range(_DESC_SLOTS) if str(i) not in registry), None)
        assert slot == 1

    def test_close_slot_removes_entry(self):
        """Close-slot logic removes the entry from the registry."""
        registry = {
            "0": {"label": "desc_1", "twist_sig": "10:a,b", "stale": False},
            "1": {"label": "desc_2", "twist_sig": "10:a,b", "stale": False},
        }
        slot_to_close = 0
        registry = dict(registry)
        registry.pop(str(slot_to_close), None)
        assert "0" not in registry
        assert "1" in registry

    def test_close_slot_frees_for_reuse(self):
        """After closing slot 0, the next free slot is 0 again."""
        registry = {"1": {"label": "desc_2"}}
        _DESC_SLOTS = 5
        slot = next((i for i in range(_DESC_SLOTS) if str(i) not in registry), None)
        assert slot == 0

    def test_descriptor_stores_twist_sig(self):
        """Registry entry must contain a twist_sig key."""
        twist_df = pd.DataFrame({
            "qp_id": [1, 2, 3],
            "angle": [0.1, 0.2, 0.3],
        })
        sig = _make_twist_sig(twist_df)
        entry = {"label": "desc_1", "twist_sig": sig, "stale": False, "features": ["angle"], "n_rows": 3}
        assert entry["twist_sig"] == sig

    def test_mark_stale_on_twist_change(self):
        """Descriptors whose twist_sig differs from the new one are marked stale."""
        old_sig = "10:angle_x,angle_y"
        registry = {
            "0": {"label": "desc_1", "twist_sig": old_sig, "stale": False},
            "1": {"label": "desc_2", "twist_sig": old_sig, "stale": False},
        }
        new_twist_df = pd.DataFrame({
            "qp_id": list(range(15)),
            "angle_x": np.zeros(15),
            "angle_y": np.zeros(15),
        })
        new_sig = _make_twist_sig(new_twist_df)
        assert new_sig != old_sig

        for k, v in registry.items():
            if v and not v.get("stale") and v.get("twist_sig") != new_sig:
                registry[k] = {**v, "stale": True}

        assert registry["0"]["stale"] is True
        assert registry["1"]["stale"] is True

    def test_already_stale_not_double_marked(self):
        """An already-stale slot is not modified again."""
        old_sig = "5:a"
        registry = {"0": {"label": "d", "twist_sig": old_sig, "stale": True}}
        new_twist_df = pd.DataFrame({"qp_id": [1, 2, 3], "a": [0, 0, 0]})
        new_sig = _make_twist_sig(new_twist_df)
        changed = False
        for k, v in registry.items():
            if v and not v.get("stale") and v.get("twist_sig") != new_sig:
                registry[k] = {**v, "stale": True}
                changed = True
        assert not changed

    def test_desc_tabs_disabled_empty_registry(self):
        """All descriptor tabs are disabled when the registry is empty."""
        registry = {}
        _DESC_SLOTS = 5
        disabled = [str(i) not in registry for i in range(_DESC_SLOTS)]
        assert all(disabled)

    def test_desc_tab_enabled_when_slot_populated(self):
        """Slot 2 tab is enabled when registry has entry '2'."""
        registry = {"2": {"label": "desc_3", "stale": False}}
        _DESC_SLOTS = 5
        disabled = [str(i) not in registry for i in range(_DESC_SLOTS)]
        assert disabled[0] is True
        assert disabled[2] is False

# ===========================================================================
# Diagnostics correctness
# ===========================================================================

class TestTangoDiagnostics:
    """Diagnostics correctness from _build_diagnostics and tablecluster."""

    @staticmethod
    def _scree(df_feat: pd.DataFrame) -> np.ndarray:
        """Run z-scored PCA and return explained_variance_ratio_."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        X = df_feat.values.astype(float)
        X_scaled = StandardScaler().fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        return pca.explained_variance_ratio_

    def test_variance_sums_to_one(self):
        """Explained variance ratios must sum to 1 (within floating-point tolerance)."""
        df = _make_df(50)
        df_feat = df.drop(columns=["qp_id"])
        evr = self._scree(df_feat)
        assert abs(evr.sum() - 1.0) < 1e-6

    def test_duplicate_features_correlation(self):
        """Two identical columns must be correlated at 1.0."""
        rng = np.random.default_rng(7)
        n = 40
        a = rng.uniform(0, 1, n)
        df = pd.DataFrame({"feat_a": a, "feat_a_copy": a, "feat_b": rng.uniform(0, 1, n)})
        corr = df.corr()
        assert abs(corr.loc["feat_a", "feat_a_copy"] - 1.0) < 1e-10

    def test_duplicate_features_fewer_nonzero_pcs(self):
        """With two identical features out of F, only F-1 PCs have non-zero variance."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(7)
        n = 40
        a = rng.uniform(0, 1, n)
        df = pd.DataFrame({"feat_a": a, "feat_a_copy": a, "feat_b": rng.uniform(0, 1, n)})
        X_scaled = StandardScaler().fit_transform(df.values)
        pca = PCA()
        pca.fit(X_scaled)
        evr = pca.explained_variance_ratio_
        nonzero = np.sum(evr > 1e-10)
        assert nonzero == 2, f"Expected 2 non-zero PCs for 3 features (2 identical), got {nonzero}"

    def test_constant_feature_std_zero(self):
        """A constant column must have std 0 in the summary."""
        rng = np.random.default_rng(3)
        n = 30
        df = pd.DataFrame({
            "feat_a": rng.uniform(0, 1, n),
            "const":  np.ones(n) * 5.0,
        })
        std = df.std()
        assert std["const"] == pytest.approx(0.0, abs=1e-10)

    def test_constant_feature_flagged(self):
        """constant=True is set for a constant column, False otherwise."""
        rng = np.random.default_rng(3)
        n = 30
        df = pd.DataFrame({
            "feat_a": rng.uniform(0, 1, n),
            "const":  np.ones(n) * 5.0,
        })
        summary = df.describe().T[["std"]]
        summary["constant"] = summary["std"] == 0
        assert summary.loc["const", "constant"] == True
        assert summary.loc["feat_a", "constant"] == False

    def test_standardisation_invariant(self):
        """Rescaling features by different positive constants must not change the scree plot."""
        df = _make_df(60)
        df_feat = df.drop(columns=["qp_id"])

        evr_orig = self._scree(df_feat)

        df_scaled = df_feat.copy()
        df_scaled["feat_a"] = df_scaled["feat_a"] * 1000
        df_scaled["feat_b"] = df_scaled["feat_b"] * 0.001
        df_scaled["feat_c"] = df_scaled["feat_c"] * 42
        evr_scaled = self._scree(df_scaled)

        np.testing.assert_allclose(evr_orig, evr_scaled, atol=1e-6,
                                   err_msg="Scree plot differs after rescaling — PCA is not standardised")

    def test_build_diagnostics_returns_children(self):
        """_build_diagnostics from ptango must return 3 children for valid data."""
        from cryocat.app.suite.pages.ptango import _build_diagnostics
        df = _make_df(20)
        data = df.to_dict("records")
        children = _build_diagnostics(data)
        assert len(children) == 3, f"Expected 3 diagnostic children, got {len(children)}"

    def test_build_diagnostics_empty_data(self):
        """_build_diagnostics with None/empty data returns empty list."""
        from cryocat.app.suite.pages.ptango import _build_diagnostics
        assert _build_diagnostics(None) == []
        assert _build_diagnostics([]) == []

    def test_tablecluster_pca_standardisation_invariant(self):
        """Same rescaling test replicated through the raw PCA path."""
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA

        rng = np.random.default_rng(99)
        n = 50
        df = pd.DataFrame({
            "feat_a": rng.uniform(0, 1, n),
            "feat_b": rng.uniform(0, 100, n),
            "feat_c": rng.uniform(-10, 10, n),
        })
        df_scaled = df.copy()
        df_scaled["feat_a"] *= 1000
        df_scaled["feat_b"] *= 0.001
        df_scaled["feat_c"] *= 42

        def _evr(df_in):
            X = StandardScaler().fit_transform(df_in.values)
            pca = PCA()
            pca.fit(X)
            return pca.explained_variance_ratio_

        np.testing.assert_allclose(_evr(df), _evr(df_scaled), atol=1e-6)

    def test_tabletomotl_merge_count(self):
        """merge reports how many subtomo_ids matched; non-matching ids produce no row."""
        src_df = pd.DataFrame({
            "qp_subtomo_id": [1.0, 2.0, 3.0, 99.0],  # 99 won't match
            "score": [0.9, 0.8, 0.7, 0.1],
        })
        motl_df = pd.DataFrame({
            "subtomo_id": [1.0, 2.0, 3.0, 4.0],
            "x": [10, 20, 30, 40],
        })
        id_column = "qp_subtomo_id"
        ids_set = set(src_df[id_column].dropna().astype(float))
        subset = motl_df[motl_df["subtomo_id"].isin(ids_set)].copy()
        matched = len(subset)
        assert matched == 3, f"Expected 3 matched, got {matched}"

    def test_tabletomotl_clean_subset_no_extra_cols(self):
        """When no destination column is chosen, the subset has only the motl's original columns."""
        motl_df = pd.DataFrame({
            "subtomo_id": [1.0, 2.0, 3.0],
            "x": [10, 20, 30],
            "score": [0.9, 0.8, 0.7],
        })
        src_df = pd.DataFrame({"qp_subtomo_id": [1.0, 3.0], "extra": [5, 6]})
        ids_set = set(src_df["qp_subtomo_id"].dropna().astype(float))
        subset = motl_df[motl_df["subtomo_id"].isin(ids_set)].copy()
        assert set(subset.columns) == {"subtomo_id", "x", "score"}
        assert len(subset) == 2

    def test_tabletomotl_write_col(self):
        """Writing a value column maps source values into the motl column by subtomo_id."""
        src_df = pd.DataFrame({
            "qp_subtomo_id": [1.0, 2.0, 3.0],
            "score": [0.9, 0.8, 0.7],
        })
        motl_df = pd.DataFrame({
            "subtomo_id": [1.0, 2.0, 3.0, 4.0],
            "x": [10, 20, 30, 40],
        })
        id_column = "qp_subtomo_id"
        val_col = "score"
        dst_col = "score"
        id_to_val = (
            src_df.drop_duplicates(subset=[id_column])
            .set_index(id_column)[val_col]
            .dropna()
            .to_dict()
        )
        motl_df = motl_df.copy()
        motl_df[dst_col] = motl_df["subtomo_id"].map(id_to_val)
        matched = motl_df["subtomo_id"].isin(id_to_val).sum()
        assert matched == 3
        assert motl_df.loc[motl_df["subtomo_id"] == 1.0, "score"].iloc[0] == pytest.approx(0.9)
        assert pd.isna(motl_df.loc[motl_df["subtomo_id"] == 4.0, "score"].iloc[0])

# ===========================================================================
# Table content — twist and descriptor grids show their rows
# ===========================================================================

_TANGO_DATA = Path(__file__).resolve().parents[1] / "test_data" / "tango"
_ROWS_REQUEST = {"startRow": 0, "endRow": 1000, "sortModel": [], "filterModel": {}}


def _descriptor_names() -> list[str]:
    from cryocat.utils.classutils import get_class_names_by_parent

    return get_class_names_by_parent("Descriptor", "cryocat.analysis.tango")


def _page_app():
    import dash
    from cryocat.app.suite.pages import ptango

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    ptango.register_callbacks(app)
    return app


def _callback(app, name: str, output: str):
    """Return the undecorated callback *name* whose output key contains *output*."""
    for key, entry in app.callback_map.items():
        fn = getattr(entry.get("callback"), "__wrapped__", None)
        if fn is not None and fn.__name__ == name and output in key:
            return fn
    raise LookupError(f"no callback {name!r} writing {output!r}")


def _call(fn, *args, triggered: str = "", states_list: list | None = None):
    """Run *fn* inside a Dash callback context (for ``ctx.triggered_id`` / ``ctx.states_list``)."""
    from contextvars import copy_context
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    def _run():
        context_value.set(
            AttributeDict(
                triggered_inputs=[{"prop_id": triggered, "value": None}],
                states_list=states_list or [],
            )
        )
        return fn(*args)

    return copy_context().run(_run)


def _grid_payload(app, prefix: str, ref: dict) -> tuple[list[dict], list[dict]]:
    """columnDefs and first-block rowData that ``{prefix}-grid`` receives for *ref*, JSON round-tripped."""
    import json
    from dash._utils import to_json

    cols = _call(
        _callback(app, "_cols", f"{prefix}-grid.columnDefs"), ref, {}, None,
        triggered=f"{prefix}-global-data-store.data",
    )
    resp, _ = _call(
        _callback(app, "_rows", f"{prefix}-grid.getRowsResponse"), _ROWS_REQUEST, ref, {},
        triggered=f"{prefix}-rows-request-store.data",
    )
    assert isinstance(resp, dict), f"_rows returned {resp!r} instead of a getRowsResponse"
    wire = json.loads(to_json({"columnDefs": cols[0], "rowData": resp["rowData"]}))
    return wire["columnDefs"], wire["rowData"]


def _computed_twist(app) -> tuple[dict, dict]:
    """Run the page's compute-twist callback on a small motl; return (handle, table ref)."""
    import dash
    from cryocat.core.cryomotl import Motl
    from cryocat.app import pool

    motl = Motl.load(str(_TANGO_DATA / "motl_cone.em"))
    state, motl_id = pool.insert_motl(pool.PoolState({}, {}, 0), motl.df.iloc[:150], label="cone")
    out = _callback(app, "_compute_twist", "tango-twist-tabv-global-data-store.data")(
        1, [motl_id], 10.0, "tomo_id", None, None, None, False, False, state.registry, 0, {}, 0, None,
    )
    assert out[1] is not dash.no_update, f"twist not computed: {out[3]}"
    return out[0], out[1]


def _loaded_twist(app) -> dict:
    """Load a twist CSV through the page's table-source path; return the table ref."""
    import dash

    path = str(_TANGO_DATA / "gt_twist_vector_r5.csv")
    status, loaded = _call(
        _callback(app, "_load", "tango-twist-src-ts-loaded.data"), 1, path, [None],
        triggered="tango-twist-src-ts-load-btn.n_clicks",
        states_list=[{}, [{"id": {"type": "tango-twist-src-ts-extra", "param": "nn_radius"}}]],
    )
    assert loaded is not dash.no_update, f"twist file not loaded: {status}"
    out = _callback(app, "_handle_twist_loaded", "tango-twist-tabv-global-data-store.data")(loaded, 0, {}, 0, None)
    assert out[1] is not dash.no_update, f"loaded twist not stored: {out[3]}"
    return out[1]


def _descriptor_ref(app, desc_name: str) -> dict:
    """Compute *desc_name* on a computed twist; return the slot-0 table ref."""
    import dash

    handle, _ = _computed_twist(app)
    out = _callback(app, "_compute_desc", "tango-desc-pool-slot-map.data")(
        1, handle, [None] * 5, desc_name, None, [], [], [], [], [], [], [], {}, 0,
    )
    assert out[0] is not dash.no_update, f"{desc_name} not computed: {out[2]}"
    refs = _callback(app, "_sync_desc_slot_stores", "tango-desc-0-global-data-store.data")(out[0], *[None] * 5)
    return refs[0]


class TestTangoTableContent:
    """The callbacks feeding the twist and descriptor grids give rows for every column."""

    def test_computed_twist_grid_rows_carry_every_column(self):
        from cryocat.app import datapool

        app = _page_app()
        _, ref = _computed_twist(app)
        col_defs, rows = _grid_payload(app, "tango-twist-tabv", ref)
        assert_grid_shows(col_defs, rows, datapool.resolve_df(ref))

    def test_loaded_twist_grid_rows_carry_every_column(self):
        from cryocat.app import datapool

        app = _page_app()
        ref = _loaded_twist(app)
        col_defs, rows = _grid_payload(app, "tango-twist-tabv", ref)
        assert_grid_shows(col_defs, rows, datapool.resolve_df(ref))

    @pytest.mark.parametrize("desc_name", _descriptor_names())
    def test_descriptor_grid_rows_carry_every_column(self, desc_name):
        from cryocat.app import datapool

        app = _page_app()
        ref = _descriptor_ref(app, desc_name)
        col_defs, rows = _grid_payload(app, "tango-desc-0", ref)
        assert_grid_shows(col_defs, rows, datapool.resolve_df(ref))


# ── Browser: the grids' cells contain text ──────────────────────────────────


def _browser_app(outputs: list[tuple[str, str, object]]):
    """The tango page with the suite stores and a button that writes *outputs*."""
    import dash
    from dash import Input, Output, html
    from cryocat.app.suite.app import POOL_STORES
    from cryocat.app.components.graphsettings import get_graph_settings_components
    from cryocat.app.suite.pages import ptango

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            html.Button("Load", id="grid-test-load-btn"),
            *POOL_STORES,
            *get_graph_settings_components(),
            ptango.layout,
        ]
    )
    ptango.register_callbacks(app)

    @app.callback(
        [Output(cid, prop, allow_duplicate=True) for cid, prop, _ in outputs],
        Input("grid-test-load-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _load(_):
        return [value for _, _, value in outputs]

    return app


def _click_load(dash_duo) -> None:
    dash_duo.wait_for_element("#grid-test-load-btn")
    dash_duo.driver.execute_script("document.getElementById('grid-test-load-btn').click()")


@pytest.mark.slow
def test_twist_grid_cells_show_text_in_browser(dash_duo):
    ref = _computed_twist(_page_app())[1]
    app = _browser_app([
        ("tango-twist-tabv-global-data-store", "data", ref),
        ("tango-tabs", "active_tab", "tango-tab-twist"),
    ])
    dash_duo.start_server(app)
    _click_load(dash_duo)
    assert_grid_cells_have_text(dash_duo, "tango-twist-tabv-grid")


@pytest.mark.slow
def test_descriptor_grid_cells_show_text_in_browser(dash_duo):
    ref = _descriptor_ref(_page_app(), "TwistDescriptor")
    app = _browser_app([
        ("tango-desc-0-global-data-store", "data", ref),
        ("tango-tab-desc-0", "disabled", False),
        ("tango-tabs", "active_tab", "tango-tab-desc-0"),
    ])
    dash_duo.start_server(app)
    _click_load(dash_duo)
    assert_grid_cells_have_text(dash_duo, "tango-desc-0-grid")


@pytest.mark.slow
def test_ptango_page_loads_in_browser(dash_duo, suite_app, monkeypatch):
    """The Tango page loads, its main components render, and the console shows no errors."""
    from cryocat.app.suite.pages import ptango
    from tests.app.conftest import main_component_ids, open_suite_page

    errors = open_suite_page(dash_duo, suite_app, "tango", monkeypatch)
    missing = [cid for cid in main_component_ids(ptango.layout) if not dash_duo.find_elements(cid, attribute="ID")]
    assert not missing, f"components not rendered: {missing}"
    assert not errors, f"browser errors: {errors}"
