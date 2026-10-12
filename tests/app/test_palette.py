"""Tests for palette application and palette loader utilities.

Covers:
- palette_is_user_set flag controls fill-only vs clear-and-apply.
- Continuous palette reaches coloraxis-bound traces via layout path.
- Every palette registered in visplot appears in the chooser (paletteloader).
- Pure-function tests for paletteloader: _validate, _make_swatch.
"""
from __future__ import annotations

import copy
import pytest

from cryocat.app.components.graphsettings import apply_settings_to_figure, GRAPH_SETTINGS_DEFAULTS
from cryocat.app.components.paletteloader import _validate, _make_swatch


# ===========================================================================
# Helpers
# ===========================================================================

def _scatter_fig(marker_color=None) -> dict:
    trace: dict = {"type": "scatter", "x": [1, 2], "y": [3, 4]}
    if marker_color is not None:
        trace["marker"] = {"color": marker_color}
    return {"data": [trace], "layout": {}}


def _coloraxis_fig() -> dict:
    return {
        "data": [
            {
                "type": "scatter",
                "x": [1, 2, 3],
                "y": [4, 5, 6],
                "marker": {
                    "color": [0.1, 0.5, 0.9],
                    "coloraxis": "coloraxis",
                },
            }
        ],
        "layout": {"coloraxis": {"colorscale": []}},
    }


def _heatmap_fig() -> dict:
    return {
        "data": [
            {
                "type": "heatmap",
                "z": [[1, 2], [3, 4]],
            }
        ],
        "layout": {},
    }


def _settings(*, palette_is_user_set: bool, discrete="StarryNight", continuous="StarryNight") -> dict:
    return {
        **GRAPH_SETTINGS_DEFAULTS,
        "discrete_palette": discrete,
        "continuous_palette": continuous,
        "palette_is_user_set": palette_is_user_set,
    }


# ===========================================================================
# palette_is_user_set=True — explicit string colour is replaced
# ===========================================================================

class TestUserSetPalette:

    def test_explicit_string_color_is_replaced(self):
        fig = _scatter_fig(marker_color="#explicit")
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=True))
        applied = result["data"][0]["marker"]["color"]
        assert applied != "#explicit"
        assert isinstance(applied, str)

    def test_no_explicit_color_is_filled(self):
        fig = _scatter_fig()
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=True))
        assert isinstance(result["data"][0].get("marker", {}).get("color"), str)


# ===========================================================================
# palette_is_user_set=False — explicit string colour is kept
# ===========================================================================

class TestFillOnlyPalette:

    def test_explicit_string_color_is_kept(self):
        original_color = "#explicit"
        fig = _scatter_fig(marker_color=original_color)
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=False))
        applied = result["data"][0]["marker"]["color"]
        assert applied == original_color

    def test_no_explicit_color_is_filled(self):
        fig = _scatter_fig()
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=False))
        assert isinstance(result["data"][0].get("marker", {}).get("color"), str)


# ===========================================================================
# Per-point array is never overwritten
# ===========================================================================

class TestArrayColorNeverOverwritten:

    def test_array_color_untouched_user_set_true(self):
        array_color = ["#aaa", "#bbb", "#ccc"]
        fig = _scatter_fig(marker_color=array_color)
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=True))
        assert result["data"][0]["marker"]["color"] == array_color

    def test_array_color_untouched_fill_only(self):
        array_color = ["#aaa", "#bbb", "#ccc"]
        fig = _scatter_fig(marker_color=array_color)
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=False))
        assert result["data"][0]["marker"]["color"] == array_color

    def test_numeric_array_untouched(self):
        fig = _coloraxis_fig()
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=True))
        assert result["data"][0]["marker"]["color"] == [0.1, 0.5, 0.9]


# ===========================================================================
# Coloraxis path receives the continuous palette
# ===========================================================================

class TestColoraxisPath:

    def test_coloraxis_layout_receives_scale(self):
        fig = _coloraxis_fig()
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=False))
        colorscale = result["layout"]["coloraxis"]["colorscale"]
        assert colorscale
        assert isinstance(colorscale, list)

    def test_coloraxis_trace_does_not_get_trace_colorscale(self):
        fig = _coloraxis_fig()
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=False))
        assert "colorscale" not in result["data"][0]

    def test_standalone_heatmap_gets_trace_colorscale(self):
        fig = _heatmap_fig()
        result = apply_settings_to_figure(copy.deepcopy(fig), _settings(palette_is_user_set=False))
        assert "colorscale" in result["data"][0]


# ===========================================================================
# Every visplot-registered palette appears in the chooser
# ===========================================================================

class TestChooserMatchesRegistry:

    def test_all_custom_discrete_palettes_in_presets(self):
        from cryocat.analysis import visplot
        from cryocat.app.components.paletteloader import _DISCRETE_PRESETS
        for name in visplot.CUSTOM_PALETTE_NAMES:
            assert name in _DISCRETE_PRESETS

    def test_all_custom_continuous_scales_in_presets(self):
        from cryocat.analysis import visplot
        from cryocat.app.components.paletteloader import _CONTINUOUS_PRESETS
        for name in visplot.CUSTOM_SCALE_NAMES:
            assert name in _CONTINUOUS_PRESETS


# ===========================================================================
# Pure-function tests for paletteloader
# ===========================================================================

class TestValidate:
    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _validate("", "discrete")

    def test_empty_continuous_raises(self):
        with pytest.raises(ValueError):
            _validate("", "continuous")

    def test_known_discrete_preset(self):
        _validate("Plotly", "discrete")

    def test_known_continuous_preset(self):
        _validate("Viridis", "continuous")

    def test_unknown_discrete_raises(self):
        with pytest.raises(Exception):
            _validate("NotARealPalette_xyz", "discrete")

    def test_unknown_continuous_raises(self):
        with pytest.raises(Exception):
            _validate("NotARealPalette_xyz", "continuous")


class TestMakeSwatch:
    def test_empty_name_returns_empty(self):
        result = _make_swatch("", "discrete")
        assert result == []

    def test_none_like_returns_empty(self):
        result = _make_swatch(None, "discrete")
        assert result == []

    def test_valid_discrete_returns_swatch(self):
        result = _make_swatch("Plotly", "discrete")
        assert result is not None and result != []

    def test_valid_continuous_returns_swatch(self):
        result = _make_swatch("Viridis", "continuous")
        assert result is not None and result != []

    def test_invalid_name_returns_empty(self):
        result = _make_swatch("__not_a_real_palette__", "discrete")
        assert result == []
