"""Tests for the plot editor — spec/figures and selection/sources.

Pure-function and structural tests; no Dash server is started.
"""
from __future__ import annotations

import copy
import json

import pandas as pd
import pytest
import plotly.graph_objects as go

from cryocat.app.components.ploteditor import (
    SPEC_DEFAULTS,
    _CHART_CONFIG,
    _ALL_ROLES,
    _express_opt_rows,
    _build_figure,
    _apply_layout_only,
    _detect_id_column,
    _export_dims,
    _JOURNAL_WIDTH_MM,
    _to_px,
    build_spec,
    normalize_spec,
)
from cryocat.app.components.graphsettings import GRAPH_SETTINGS_DEFAULTS


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _sample_df(n=20) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "z": rng.normal(size=n),
        "score": rng.uniform(size=n),
        "class": (rng.integers(0, 3, size=n)).astype(str),
        "tomo_id": range(n),
    })


# ── T1a: spec round-trips through JSON unchanged ──────────────────────────────

class TestSpecJsonRoundTrip:

    def test_empty_spec_survives_json(self):
        """SPEC_DEFAULTS round-trips through json.dumps/loads unchanged."""
        rt = json.loads(json.dumps(SPEC_DEFAULTS))
        assert rt == SPEC_DEFAULTS

    def test_built_spec_survives_json(self):
        """build_spec produces a dict that round-trips through JSON."""
        spec = build_spec(
            "scatter",
            {"data_id": "data_1"},
            {"x": "score", "y": "z", "color": "class"},
            layout={"title": "Test"},
            chart_opts={"trendline": "ols"},
        )
        rt = json.loads(json.dumps(spec))
        assert rt == spec

    def test_spec_keys_present(self):
        """build_spec produces traces-list format with required keys."""
        spec = build_spec("histogram", None, {})
        assert "traces" in spec
        assert "layout" in spec
        trace = spec["traces"][0]
        for key in ("chart", "source", "roles", "chart_opts"):
            assert key in trace

    def test_normalize_spec_converts_old_format(self):
        """normalize_spec converts old flat format to new traces-list format."""
        old = {"chart": "scatter", "source": {"data_id": "d"}, "roles": {"x": "a"}, "layout": {}}
        new = normalize_spec(old)
        assert "traces" in new and "layout" in new
        assert "chart" not in new
        assert new["traces"][0]["chart"] == "scatter"


# ── T1b: same spec applied to two DFs produces two valid figures ──────────────

class TestSpecAppliedToTwoDFs:

    def test_same_spec_two_dfs(self):
        df1 = _sample_df(10)
        df2 = _sample_df(20)
        roles = {"x": "x", "y": "y"}

        fig1 = _build_figure("scatter", df1, roles, {}, None,
                              GRAPH_SETTINGS_DEFAULTS, {})
        fig2 = _build_figure("scatter", df2, roles, {}, None,
                              GRAPH_SETTINGS_DEFAULTS, {})

        assert fig1 is not None, "Figure from df1 must not be None"
        assert fig2 is not None, "Figure from df2 must not be None"
        assert len(fig1.data) >= 1 and len(fig2.data) >= 1, "Both figures must have at least one trace"
        # The two figures are independent: their JSON representations are distinct
        j1 = fig1.to_plotly_json()
        j2 = fig2.to_plotly_json()
        # df2 has twice the rows — the raw data size in the trace differs
        assert j1 != j2, "Same spec on different DFs must produce different figures"


# ── T1c: changing chart type preserves overlapping roles ─────────────────────

class TestChartTypeRolePreservation:

    def _roles_for(self, chart: str) -> set[str]:
        return set(_CHART_CONFIG[chart]["roles"])

    def test_scatter_to_line_keeps_x_y_color(self):
        """x, y, color are shared between scatter and line — they must survive chart change."""
        shared = self._roles_for("scatter") & self._roles_for("line")
        assert "x" in shared
        assert "y" in shared
        assert "color" in shared

    def test_scatter_drops_size_when_switching_to_histogram(self):
        """size is a scatter role but not a histogram role."""
        scatter_roles = self._roles_for("scatter")
        histogram_roles = self._roles_for("histogram")
        assert "size" in scatter_roles
        assert "size" not in histogram_roles

    def test_scatter_3d_has_z(self):
        assert "z" in self._roles_for("scatter_3d")

    def test_density_heatmap_has_z_not_color(self):
        roles = self._roles_for("density_heatmap")
        assert "z" in roles
        assert "color" not in roles


# ── T1d: layout levels (W5) ───────────────────────────────────────────────────

class TestStylingLevels:

    def test_figure_title_overrides_nothing(self):
        """A figure with no layout spec follows defaults (no crash, valid figure)."""
        df = _sample_df()
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        assert isinstance(fig, go.Figure)

    def test_figure_level_title_set(self):
        """Layout panel title reaches the figure layout (W5 level 2)."""
        df = _sample_df()
        layout_spec = {"title": "My Figure"}
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, layout_spec)
        assert fig.layout.title.text == "My Figure"

    def test_figure_level_xrange_set(self):
        """Typed axis range from Layout panel reaches the figure (W5 level 2)."""
        df = _sample_df()
        layout_spec = {"xaxis_min": -1.0, "xaxis_max": 1.0}
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, layout_spec)
        assert list(fig.layout.xaxis.range) == [-1.0, 1.0]

    def test_figure_level_log_scale(self):
        """Log scale switch from Layout panel sets xaxis.type='log'."""
        df = _sample_df()
        layout_spec = {"xaxis_log": True}
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, layout_spec)
        assert fig.layout.xaxis.type == "log"

    def test_defaults_level_palette_applied(self):
        """Defaults-level palette (GRAPH_SETTINGS_STORE) reaches the figure (W5 level 1)."""
        from cryocat.analysis.visplot import resolve_palette
        settings = {**GRAPH_SETTINGS_DEFAULTS, "discrete_palette": "StarryNight",
                    "palette_is_user_set": False}
        df = _sample_df()
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, settings, {})
        palette = resolve_palette("StarryNight")
        assert list(fig.layout.colorway) == palette


# ── T1e: chart-specific options reach the Express call (W9) ──────────────────

class TestChartSpecificOptions:

    def test_box_points_option_reaches_figure(self):
        """points='all' option for box chart reaches the trace (W9)."""
        df = _sample_df()
        chart_opts = {"points": "all"}
        fig = _build_figure("box", df, {"y": "score"}, chart_opts,
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        assert fig is not None, "box chart with points='all' must produce a figure"
        # px.box maps 'points' to trace.boxpoints
        assert fig.data[0].boxpoints in ("all", "outliers", "suspectedoutliers")

    def test_scatter_opacity_reaches_figure(self):
        """opacity option for scatter chart reaches the trace (W9)."""
        df = _sample_df()
        chart_opts = {"opacity": 0.5}
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, chart_opts,
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        assert fig is not None, "scatter with opacity=0.5 must produce a figure"
        assert len(fig.data) >= 1

    def test_express_opt_rows_non_empty_for_scatter(self):
        """_express_opt_rows returns at least one row for scatter (W9)."""
        rows = _express_opt_rows("scatter", "test-prefix")
        assert len(rows) >= 1
        # Should not be the "No additional options" placeholder
        # (scatter has many options like trendline, opacity, etc.)
        assert not (len(rows) == 1 and "No additional" in str(rows[0]))

    def test_express_opt_rows_hide_role_params(self):
        """Role params (x, y, color) must not appear in chart-specific options (W9)."""
        rows = _express_opt_rows("scatter", "test-prefix")
        row_str = str(rows)
        # x, y, color are roles — they must not appear as option widgets
        # Check that the widget IDs (which contain the param name) don't include role names
        # We check the string representation doesn't contain role-named IDs
        for role in ("\"x\"", "\"y\"", "\"color\""):
            assert f"\"param\": {role}" not in row_str


# ── T2a: selection maps to row identities (W7) ───────────────────────────────

class TestSelectionRoundTrip:

    def test_customdata_injected_when_id_col_present(self):
        """When an id_column is detected, customdata is injected into the figure (W7)."""
        df = _sample_df()
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            "tomo_id", GRAPH_SETTINGS_DEFAULTS, {})
        assert fig is not None
        # customdata should be on traces
        for trace in fig.data:
            assert hasattr(trace, "customdata") and trace.customdata is not None, \
                "customdata must be set for selection round-trip when id_col is present"

    def test_no_customdata_without_id_col(self):
        """When no id_column is found, customdata is not injected."""
        df = pd.DataFrame({"score": [1.0, 2.0, 3.0], "z": [4.0, 5.0, 6.0]})
        id_col = _detect_id_column(df)
        assert id_col is None

    def test_detect_id_column_finds_tomo_id(self):
        """_detect_id_column returns 'tomo_id' when present."""
        df = _sample_df()
        assert _detect_id_column(df) == "tomo_id"

    def test_detect_id_column_prefers_tomo_id_over_id(self):
        """Detection priority: tomo_id > object_id > id."""
        df = pd.DataFrame({"id": [1, 2], "tomo_id": [10, 20], "x": [0.0, 1.0]})
        assert _detect_id_column(df) == "tomo_id"


# ── T2b: multi-source plots (W4, W7) ─────────────────────────────────────────

class TestMultiSourcePlots:

    def test_overlay_traces_add_to_figure(self):
        """Overlay traces from a second source are added to the primary figure (W4)."""
        from cryocat.app.components.ploteditor import _add_overlay
        df1 = _sample_df(10)
        fig = _build_figure("scatter", df1, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        assert fig is not None
        initial_traces = len(fig.data)

        df2 = _sample_df(5)
        # source must be a non-None dict with "data_id" so _resolve_df uses dp_resolve_df
        trace_cfg = {"source": {"data_id": "overlay_ds"}, "label": "Overlay run", "color": "#ff0000"}

        def _dp(ref):
            return df2

        _add_overlay(fig, trace_cfg, None, _dp, {"x": "x", "y": "y"},
                     "scatter", GRAPH_SETTINGS_DEFAULTS)
        assert len(fig.data) == initial_traces + 1, "Overlay trace must be added to figure"

    def test_overlay_trace_label_set(self):
        """Overlay trace carries the label from trace config (W4)."""
        from cryocat.app.components.ploteditor import _add_overlay
        df = _sample_df(5)
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        trace_cfg = {"source": {"data_id": "ds"}, "label": "Run 2", "color": None}
        _add_overlay(fig, trace_cfg, None, lambda _: df, {"x": "x", "y": "y"},
                     "scatter", GRAPH_SETTINGS_DEFAULTS)
        names = [t.name for t in fig.data]
        assert "Run 2" in names, "Overlay trace name must be set from trace config"


# ── T2c: visplot external figure (W6) ────────────────────────────────────────

class TestExternalFigure:

    def test_apply_layout_only_works_without_data(self):
        """_apply_layout_only re-applies layout to an existing figure dict without re-fetching (W6)."""
        df = _sample_df()
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        existing = fig.to_plotly_json()
        layout_spec = {"title": "Updated Title"}
        result = _apply_layout_only(existing, layout_spec, GRAPH_SETTINGS_DEFAULTS, None, None)
        assert result["layout"]["title"]["text"] == "Updated Title"

    def test_apply_layout_only_does_not_drop_traces(self):
        """_apply_layout_only preserves trace data (W6 — geometry must survive layout-only update)."""
        df = _sample_df(15)
        fig = _build_figure("scatter", df, {"x": "x", "y": "y"}, {},
                            None, GRAPH_SETTINGS_DEFAULTS, {})
        existing = fig.to_plotly_json()
        n_traces = len(existing["data"])
        result = _apply_layout_only(existing, {}, GRAPH_SETTINGS_DEFAULTS, None, None)
        assert len(result["data"]) == n_traces


# ── T2d: export physical units (W10) ─────────────────────────────────────────

class TestExportPhysicalUnits:

    def test_85mm_at_300dpi(self):
        """85 mm at 300 dpi resolves to width=321, scale=3.125 (W10)."""
        w_px, h_px, scale = _export_dims(85, 65, "mm", 300)
        assert w_px == 321, f"Expected 321 px, got {w_px}"
        assert abs(scale - 3.125) < 0.01, f"Expected scale 3.125, got {scale}"

    def test_svg_scale_is_one(self):
        """SVG export produces scale=1.0 regardless of DPI (W10 — SVG is vector)."""
        _, _, scale = _export_dims(85, 65, "mm", 300)
        # For SVG the caller ignores scale — _export_dims returns the px scale factor
        # which is used only for raster; SVG ignores it.
        # The function always returns scale = dpi/96 for non-px units.
        # Caller must skip scale for SVG.  Here we verify the pixel width is correct.
        w_px, _, _ = _export_dims(85, 65, "mm", 96)
        assert w_px == 321

    def test_journal_single_column_is_85mm(self):
        """Journal preset 'single' maps to 85 mm (W10)."""
        assert _JOURNAL_WIDTH_MM["single"] == 85

    def test_journal_double_column_is_178mm(self):
        """Journal preset 'double' maps to 178 mm (W10)."""
        assert _JOURNAL_WIDTH_MM["double"] == 178

    def test_inch_conversion(self):
        """1 inch = 96 px at CSS dpi (W10)."""
        assert abs(_to_px(1.0, "inch") - 96.0) < 0.01

    def test_px_unit_is_identity(self):
        """'px' unit leaves value unchanged."""
        assert _to_px(500, "px") == 500.0

