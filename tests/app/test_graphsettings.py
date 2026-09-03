"""Tests for cryocat.app.components.graphsettings — T4 acceptance suite."""

import pytest
import plotly.graph_objects as go

from cryocat.app.components.graphsettings import (
    apply_settings_to_figure,
    styled_figure,
    error_figure,
    GRAPH_SETTINGS_DEFAULTS,
    _is_dark,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _scatter_fig(marker_size=None, line_dash=None):
    """Simple scatter with optional explicit marker.size / line.dash."""
    trace = {"type": "scatter", "x": [1, 2], "y": [3, 4]}
    if marker_size is not None:
        trace["marker"] = {"size": marker_size}
    if line_dash is not None:
        trace["line"] = {"dash": line_dash}
    return {"data": [trace], "layout": {}}


# ── T4a: fill-only semantics ──────────────────────────────────────────────────

class TestFillOnly:
    def test_marker_size_not_overwritten_when_explicit(self):
        fig = _scatter_fig(marker_size=20)
        apply_settings_to_figure(fig, {"marker_size": 6})
        assert fig["data"][0]["marker"]["size"] == 20

    def test_marker_size_filled_when_absent(self):
        fig = _scatter_fig()
        apply_settings_to_figure(fig, {"marker_size": 6})
        assert fig["data"][0]["marker"]["size"] == 6

    def test_line_dash_not_overwritten_when_explicit(self):
        fig = _scatter_fig(line_dash="dot")
        apply_settings_to_figure(fig, {"line_dash": "solid"})
        assert fig["data"][0]["line"]["dash"] == "dot"

    def test_line_dash_filled_when_absent(self):
        fig = _scatter_fig()
        apply_settings_to_figure(fig, {"line_dash": "dash"})
        assert fig["data"][0]["line"]["dash"] == "dash"

    def test_override_replaces_explicit_marker_size(self):
        fig = _scatter_fig(marker_size=20)
        apply_settings_to_figure(fig, {"marker_size": 6}, override=True)
        assert fig["data"][0]["marker"]["size"] == 6

    def test_override_replaces_explicit_line_dash(self):
        fig = _scatter_fig(line_dash="dot")
        apply_settings_to_figure(fig, {"line_dash": "solid"}, override=True)
        assert fig["data"][0]["line"]["dash"] == "solid"


# ── T4a: continuous trace types ───────────────────────────────────────────────

class TestContinuousTypes:
    def test_mesh3d_gets_colorscale(self):
        fig = {"data": [{"type": "mesh3d", "x": [0], "y": [0], "z": [0]}], "layout": {}}
        apply_settings_to_figure(fig, {"continuous_palette": "Plasma"})
        cs = fig["data"][0].get("colorscale")
        assert cs, "mesh3d must receive a colorscale"

    def test_scatter3d_gets_discrete_palette_not_colorscale(self):
        fig = {"data": [{"type": "scatter3d", "x": [0], "y": [0], "z": [0]}], "layout": {}}
        apply_settings_to_figure(fig, {"discrete_palette": "Monet", "continuous_palette": "Plasma"})
        assert "colorscale" not in fig["data"][0]

    def test_histogram2d_gets_colorscale(self):
        fig = {"data": [{"type": "histogram2d", "x": [1], "y": [1]}], "layout": {}}
        apply_settings_to_figure(fig, {"continuous_palette": "Viridis"})
        cs = fig["data"][0].get("colorscale")
        assert cs, "histogram2d must receive a colorscale"


# ── T4a: dark background → light font ────────────────────────────────────────

class TestDarkBackground:
    def test_dark_hex_detected(self):
        assert _is_dark("#1e1e1e") is True

    def test_light_hex_not_dark(self):
        assert _is_dark("#ffffff") is False

    def test_white_name_not_dark(self):
        assert _is_dark("white") is False

    def test_dark_bg_sets_axis_gridcolor(self):
        fig = {"data": [], "layout": {}}
        apply_settings_to_figure(fig, {"bg_color": "#1e1e1e"})
        assert fig["layout"].get("xaxis", {}).get("gridcolor") == "#444444"

    def test_light_bg_no_dark_axis_overrides(self):
        fig = {"data": [], "layout": {}}
        apply_settings_to_figure(fig, {"bg_color": "white"})
        assert "gridcolor" not in fig["layout"].get("xaxis", {})


# ── T4b: styled_figure ────────────────────────────────────────────────────────

class TestStyledFigure:
    def test_requires_uirevision_keyword(self):
        with pytest.raises(TypeError):
            styled_figure(go.Figure(), {})  # uirevision is keyword-only, must be named

    def test_stamps_uirevision(self):
        result = styled_figure(go.Figure(), {}, uirevision="my-view")
        assert result.layout.uirevision == "my-view"

    def test_stamps_height(self):
        result = styled_figure(go.Figure(), {}, uirevision="v", height=500)
        assert result.layout.height == 500

    def test_stamps_margin(self):
        m = {"t": 0, "b": 0, "l": 0, "r": 0}
        result = styled_figure(go.Figure(), {}, uirevision="v", margin=m)
        assert result.layout.margin.t == 0

    def test_stamps_title_string(self):
        result = styled_figure(go.Figure(), {}, uirevision="v", title="My Plot")
        assert result.layout.title.text == "My Plot"

    def test_stamps_title_dict(self):
        result = styled_figure(go.Figure(), {}, uirevision="v", title={"text": "X", "font": {"size": 10}})
        assert result.layout.title.text == "X"

    def test_applies_settings(self):
        fig = go.Figure(go.Scatter(x=[1], y=[2]))
        result = styled_figure(fig, {"marker_size": 12}, uirevision="v")
        assert result.data[0].marker.size == 12

    def test_returns_go_figure(self):
        result = styled_figure(go.Figure(), {}, uirevision="v")
        assert isinstance(result, go.Figure)

    def test_empty_settings_noop(self):
        result = styled_figure(go.Figure(), {}, uirevision="v")
        assert result.layout.uirevision == "v"


# ── T4b: error_figure ────────────────────────────────────────────────────────

class TestErrorFigure:
    def test_returns_go_figure(self):
        assert isinstance(error_figure("boom"), go.Figure)

    def test_zero_traces(self):
        assert len(error_figure("boom").data) == 0

    def test_one_annotation(self):
        fig = error_figure("boom")
        assert len(fig.layout.annotations) == 1

    def test_annotation_text(self):
        fig = error_figure("my error")
        assert fig.layout.annotations[0].text == "my error"

    def test_annotation_centered(self):
        ann = error_figure("x").layout.annotations[0]
        assert ann.xref == "paper"
        assert ann.yref == "paper"


# ── palette invariant (regression guard) ─────────────────────────────────────

class TestPaletteInvariant:
    def test_discrete_palette_applied_to_layout_colorway(self):
        fig = {"data": [], "layout": {}}
        apply_settings_to_figure(fig, {"discrete_palette": "Monet"})
        assert isinstance(fig["layout"].get("colorway"), list)
        assert len(fig["layout"]["colorway"]) > 0

    def test_continuous_palette_applied_to_coloraxis(self):
        fig = {"data": [], "layout": {}}
        apply_settings_to_figure(fig, {"continuous_palette": "Viridis"})
        cs = fig["layout"]["coloraxis"]["colorscale"]
        assert cs, "layout.coloraxis.colorscale must be set"
