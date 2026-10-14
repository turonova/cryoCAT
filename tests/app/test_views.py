"""Tests for tomo view, volume view, and wedge preview pure functions.

Covers:
- tomoview: hover_template, aspect_ratio_dict.
- volumeview: mesh_at, _scene_layout.
- wedgepreview: wedge_xz_figure.
"""
from __future__ import annotations

import numpy as np
import pytest

from cryocat.app.components.tomoview import hover_template, aspect_ratio_dict
from cryocat.app.components.volumeview import mesh_at, _scene_layout
from cryocat.app.components.wedgepreview import wedge_xz_figure


# ===========================================================================
# tomoview — hover_template
# ===========================================================================

def test_hover_template_full_includes_all_columns():
    cols = ["x", "y", "score"]
    tmpl = hover_template(cols, "full")
    assert "x: %{customdata[0]}" in tmpl
    assert "y: %{customdata[1]}" in tmpl
    assert "score: %{customdata[2]}" in tmpl
    assert tmpl.endswith("<extra></extra>")


def test_hover_template_full_delimiter():
    cols = ["a", "b"]
    tmpl = hover_template(cols, "full")
    assert "<br>" in tmpl


def test_hover_template_partial_filters_to_requested():
    cols = ["x", "y", "z", "score"]
    tmpl = hover_template(cols, ["x", "score"])
    assert "x: %{customdata[0]}" in tmpl
    assert "score: %{customdata[3]}" in tmpl
    assert "y:" not in tmpl
    assert "z:" not in tmpl


def test_hover_template_partial_skips_missing_columns():
    cols = ["x", "y"]
    tmpl = hover_template(cols, ["x", "nonexistent"])
    assert "x: %{customdata[0]}" in tmpl
    assert "nonexistent" not in tmpl


def test_hover_template_empty_columns_full():
    tmpl = hover_template([], "full")
    assert tmpl == "<extra></extra>"


# ===========================================================================
# tomoview — aspect_ratio_dict
# ===========================================================================

def test_aspect_ratio_unit_cube():
    coords = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    ar = aspect_ratio_dict(coords)
    assert ar == {"x": 1.0, "y": 1.0, "z": 1.0}


def test_aspect_ratio_2x1x1():
    coords = np.array([[0, 0, 0], [2, 1, 1]], dtype=float)
    ar = aspect_ratio_dict(coords)
    assert ar["x"] == pytest.approx(1.0)
    assert ar["y"] == pytest.approx(0.5)
    assert ar["z"] == pytest.approx(0.5)


def test_aspect_ratio_no_divide_by_zero():
    coords = np.array([[5, 5, 5]], dtype=float)
    ar = aspect_ratio_dict(coords)
    assert ar == {"x": 0.0, "y": 0.0, "z": 0.0}


def test_aspect_ratio_values_between_0_and_1():
    rng = np.random.default_rng(0)
    coords = rng.uniform(0, 100, size=(50, 3))
    ar = aspect_ratio_dict(coords)
    for v in ar.values():
        assert 0.0 <= v <= 1.0


# ===========================================================================
# volumeview — mesh_at
# ===========================================================================

class TestMeshAt:
    def _blob(self, shape=(16, 16, 16)):
        c = [s // 2 for s in shape]
        idx = np.indices(shape)
        r2 = sum((idx[i] - c[i]) ** 2 for i in range(3))
        return np.exp(-r2 / 4.0).astype(np.float32)

    def test_returns_dict_for_valid_surface(self):
        vol = self._blob()
        result = mesh_at(vol, level=0.5)
        assert result is not None
        for key in ("x", "y", "z", "i", "j", "k"):
            assert key in result

    def test_returns_none_for_empty_faces(self):
        vol = np.zeros((16, 16, 16), dtype=np.float32)
        result = mesh_at(vol, level=1.0)
        assert result is None

    def test_bad_level_returns_none(self):
        vol = np.ones((8, 8, 8), dtype=np.float32)
        result = mesh_at(vol, level=10.0)
        assert result is None

    def test_result_is_json_serialisable(self):
        import json
        vol = self._blob()
        result = mesh_at(vol, level=0.5)
        if result is None:
            pytest.skip("no surface at this level")
        json.dumps(result)


# ===========================================================================
# volumeview — _scene_layout
# ===========================================================================

class TestSceneLayout:
    def test_basic(self):
        layout = _scene_layout((10, 20, 40))
        assert layout["xaxis"]["range"] == [0, 10]
        assert layout["yaxis"]["range"] == [0, 20]
        assert layout["zaxis"]["range"] == [0, 40]

    def test_aspect_ratio_normalised(self):
        layout = _scene_layout((10, 20, 40))
        ar = layout["aspectratio"]
        assert ar["z"] == pytest.approx(1.0)
        assert ar["x"] == pytest.approx(10 / 40)
        assert ar["y"] == pytest.approx(20 / 40)

    def test_aspectmode(self):
        layout = _scene_layout((8, 8, 8))
        assert layout["aspectmode"] == "manual"


# ===========================================================================
# wedgepreview — wedge_xz_figure
# ===========================================================================

class TestWedgeXzFigure:
    def make_mask(self, shape=(16, 16, 16)):
        arr = np.zeros(shape)
        arr[: shape[0] // 2, :, :] = 1.0
        return arr

    def test_basic_3d_runs(self):
        mask = self.make_mask()
        fig = wedge_xz_figure(mask)
        assert fig is not None
        assert len(fig.data) == 1

    def test_square_aspect_via_scaleanchor(self):
        mask = self.make_mask()
        fig = wedge_xz_figure(mask)
        yaxis = fig.layout.yaxis
        assert yaxis.scaleanchor == "x"
        assert yaxis.scaleratio == 1

    def test_raises_on_2d(self):
        with pytest.raises(ValueError, match="3D"):
            wedge_xz_figure(np.zeros((16, 16)))

    def test_raises_on_1d(self):
        with pytest.raises(ValueError):
            wedge_xz_figure(np.zeros(16))

    def test_uirevision_set(self):
        mask = self.make_mask()
        fig = wedge_xz_figure(mask)
        assert fig.layout.uirevision is not None

    def test_non_cubic_shape(self):
        mask = self.make_mask(shape=(8, 12, 16))
        fig = wedge_xz_figure(mask)
        assert len(fig.data) == 1
