import numpy as np
import pandas as pd
import pytest
from copy import deepcopy

import cryocat.analysis.visplot as vp
from cryocat.utils import geom


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit(v):
    """Return unit vector(s)."""
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


# ---------------------------------------------------------------------------
# register_palette / resolve_palette
# ---------------------------------------------------------------------------

class TestPaletteRegistry:
    def setup_method(self):
        vp.CUSTOM_PALETTES.pop("_testpal", None)

    def teardown_method(self):
        vp.CUSTOM_PALETTES.pop("_testpal", None)

    def test_register_and_resolve(self):
        vp.register_palette("_TestPal", ["#aabbcc", "#112233"])
        result = vp.resolve_palette("_TestPal")
        assert result == ["#aabbcc", "#112233"]

    def test_register_case_insensitive(self):
        vp.register_palette("_TestPal", ["#aabbcc"])
        assert vp.resolve_palette("_testpal") == ["#aabbcc"]

    def test_register_empty_raises(self):
        with pytest.raises(ValueError):
            vp.register_palette("_TestPal", [])

    def test_resolve_builtin(self):
        result = vp.resolve_palette("D3")
        assert isinstance(result, list)
        assert len(result) > 0

    def test_resolve_none_returns_default(self):
        result = vp.resolve_palette(None)
        assert isinstance(result, list)
        assert len(result) > 0

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError):
            vp.resolve_palette("__nonexistent_palette__")

    def test_resolve_explicit_list(self):
        colors = ["red", "green", "blue"]
        assert vp.resolve_palette(colors) == colors


# ---------------------------------------------------------------------------
# register_colorscale / resolve_colorscale
# ---------------------------------------------------------------------------

class TestColorscaleRegistry:
    def setup_method(self):
        vp.CUSTOM_SCALES.pop("_testscale", None)

    def teardown_method(self):
        vp.CUSTOM_SCALES.pop("_testscale", None)

    def test_register_two_colors(self):
        vp.register_colorscale("_testscale", ["#000000", "#ffffff"])
        result = vp.resolve_colorscale("_testscale")
        assert result[0] == (0.0, "#000000")
        assert result[-1] == (1.0, "#ffffff")

    def test_register_single_color(self):
        vp.register_colorscale("_testscale", ["#abcdef"])
        result = vp.resolve_colorscale("_testscale")
        assert len(result) == 1
        assert result[0][0] == 0.0

    def test_register_empty_raises(self):
        with pytest.raises(ValueError):
            vp.register_colorscale("_testscale", [])

    def test_resolve_builtin(self):
        result = vp.resolve_colorscale("Viridis")
        assert isinstance(result, list)
        pos_vals = [p for p, _ in result]
        assert pos_vals[0] == pytest.approx(0.0)
        assert pos_vals[-1] == pytest.approx(1.0)

    def test_resolve_none_returns_viridis(self):
        result = vp.resolve_colorscale(None)
        assert isinstance(result, list)
        assert all(isinstance(p, float) for p, _ in result)

    def test_resolve_unknown_raises(self):
        with pytest.raises(KeyError):
            vp.resolve_colorscale("__nonexistent_scale__")

    def test_resolve_hex_list_auto_stops(self):
        hexes = ["#000000", "#888888", "#ffffff"]
        result = vp.resolve_colorscale(hexes)
        assert result[0][0] == pytest.approx(0.0)
        assert result[-1][0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# set_defaults / use_defaults
# ---------------------------------------------------------------------------

class TestDefaults:
    def setup_method(self):
        self._saved = deepcopy(vp.DEFAULTS)

    def teardown_method(self):
        vp.DEFAULTS = self._saved

    def test_set_defaults_height(self):
        vp.set_defaults(height=800)
        assert vp.DEFAULTS.height == 800

    def test_set_defaults_template(self):
        vp.set_defaults(template="seaborn")
        assert vp.DEFAULTS.template == "seaborn"

    def test_set_defaults_extra_layout_merged(self):
        vp.set_defaults(extra_layout={"key1": 1})
        vp.set_defaults(extra_layout={"key2": 2})
        assert vp.DEFAULTS.extra_layout.get("key2") == 2

    def test_use_defaults_context_reverts(self):
        original_height = vp.DEFAULTS.height
        with vp.use_defaults(height=9999):
            assert vp.DEFAULTS.height == 9999
        assert vp.DEFAULTS.height == original_height

    def test_use_defaults_reverts_on_exception(self):
        original_height = vp.DEFAULTS.height
        with pytest.raises(RuntimeError):
            with vp.use_defaults(height=7777):
                raise RuntimeError("test error")
        assert vp.DEFAULTS.height == original_height


# ---------------------------------------------------------------------------
# format_input_data_id
# ---------------------------------------------------------------------------

class TestFormatInputDataId:
    def test_dataframe_no_id(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = vp._format_input_data_id(df, None)
        assert list(result) == ["a", "b"]

    def test_ndarray_1d_no_id(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = vp._format_input_data_id(arr, None)
        assert result == ["Value"]

    def test_ndarray_2d_no_id(self):
        arr = np.zeros((5, 3))
        result = vp._format_input_data_id(arr, None)
        assert result == ["Value", "Value", "Value"]

    def test_explicit_id_returned_unchanged(self):
        df = pd.DataFrame({"x": [1]})
        result = vp._format_input_data_id(df, ["x"])
        assert result == ["x"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            vp._format_input_data_id([1, 2, 3], None)

    def test_custom_default_name(self):
        arr = np.zeros((4, 2))
        result = vp._format_input_data_id(arr, None, default_name="Col")
        assert result == ["Col", "Col"]


# ---------------------------------------------------------------------------
# format_input_data
# ---------------------------------------------------------------------------

class TestFormatInputData:
    def test_dataframe_returns_numpy(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        data, ids = vp._format_input_data(df, ["a", "b"], 2)
        assert isinstance(data, np.ndarray)
        assert data.shape == (2, 2)
        assert ids == ["a", "b"]

    def test_dataframe_drops_missing_columns(self):
        df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]})
        data, ids = vp._format_input_data(df, ["a", "z"], 2)
        assert ids == ["a"]

    def test_dataframe_no_matching_columns_raises(self):
        df = pd.DataFrame({"a": [1.0]})
        with pytest.raises(ValueError):
            vp._format_input_data(df, ["z"], 1)

    def test_ndarray_returns_data(self):
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])
        data, ids = vp._format_input_data(arr, ["x", "y"], 2)
        np.testing.assert_array_equal(data, arr)
        assert ids == ["x", "y"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            vp._format_input_data([1, 2, 3], ["x"], 1)


# ---------------------------------------------------------------------------
# project_lambert  (moved to geom)
# ---------------------------------------------------------------------------

class TestProjectLambert:
    def test_north_pole_maps_to_origin(self):
        coord = np.array([[0.0, 0.0, 1.0]])
        _, xy = geom.project_lambert(coord)
        np.testing.assert_allclose(xy[0], [0.0, 0.0], atol=1e-10)

    def test_output_shapes(self):
        coord = _unit(np.random.randn(15, 3))
        tr, xy = geom.project_lambert(coord)
        assert tr.shape == (15, 2)
        assert xy.shape == (15, 2)

    def test_equator_r_equals_sqrt2(self):
        coord = np.array([[1.0, 0.0, 0.0]])
        tr, _ = geom.project_lambert(coord)
        # At equator (theta=pi/2): r = 2*cos((pi - pi/2)/2) = 2*cos(pi/4) = sqrt(2)
        assert tr[0, 1] == pytest.approx(np.sqrt(2), rel=1e-6)


# ---------------------------------------------------------------------------
# project_stereo  (moved to geom)
# ---------------------------------------------------------------------------

class TestProjectStereo:
    def test_north_pole_polar_r_is_zero(self):
        # At the north pole (z=1) xy is 0/0 (singularity), but polar r should be 0
        coord = np.array([[0.0, 0.0, 1.0]])
        tr, _ = geom.project_stereo(coord)
        assert tr[0, 1] == pytest.approx(0.0, abs=1e-10)

    def test_output_shapes(self):
        coord = _unit(np.random.randn(12, 3))
        # avoid south pole (z close to 1) to prevent division by zero
        coord = coord[coord[:, 2] > -0.9]
        tr, xy = geom.project_stereo(coord)
        assert tr.shape[1] == 2
        assert xy.shape[1] == 2


# ---------------------------------------------------------------------------
# project_equidistant  (moved to geom)
# ---------------------------------------------------------------------------

class TestProjectEquidistant:
    def test_output_shapes(self):
        coord = _unit(np.random.randn(10, 3))
        tr, xy = geom.project_equidistant(coord)
        assert tr.shape == (10, 2)
        assert xy.shape == (10, 2)


# ---------------------------------------------------------------------------
# project_points_on_sphere dispatch  (moved to geom)
# ---------------------------------------------------------------------------

class TestProjectPointsOnSphere:
    @pytest.mark.parametrize("proj", ["stereo", "lambert", "equidistant"])
    def test_dispatch(self, proj):
        coord = _unit(np.random.randn(8, 3))
        coord = coord[coord[:, 2] > -0.8]  # avoid south-pole singularity for stereo
        tr, xy = geom.project_points_on_sphere(coord, projection_type=proj)
        assert tr.shape[1] == 2
        assert xy.shape[1] == 2


# ---------------------------------------------------------------------------
# create_projection  (moved to geom)
# ---------------------------------------------------------------------------

class TestCreateProjection:
    def test_split_hemispheres(self):
        np.random.seed(0)
        coord = _unit(np.random.randn(30, 3))
        tr_pos, xy_pos, tr_neg, xy_neg = geom.create_projection(coord, "lambert", split_into_hemispheres=True)
        n_pos = np.sum(coord[:, 2] >= 0)
        n_neg = np.sum(coord[:, 2] < 0)
        assert tr_pos.shape[0] == n_pos
        assert tr_neg.shape[0] == n_neg

    def test_no_split(self):
        coord = _unit(np.random.randn(20, 3))
        tr, xy, tr_neg, xy_neg = geom.create_projection(coord, "lambert", split_into_hemispheres=False)
        assert tr.shape[0] == 20
        assert tr_neg.shape == (0, 2)
        assert xy_neg.shape == (0, 2)

    def test_all_northern_hemisphere(self):
        coord = _unit(np.random.randn(10, 3))
        coord[:, 2] = np.abs(coord[:, 2])  # force z >= 0
        tr_pos, _, tr_neg, _ = geom.create_projection(coord, "lambert")
        assert tr_pos.shape[0] == 10
        assert tr_neg.shape[0] == 0


# ---------------------------------------------------------------------------
# get_colors_from_palette
# ---------------------------------------------------------------------------

class TestGetColorsFromPalette:
    def test_returns_correct_count(self):
        result = vp.get_colors_from_palette(5)
        assert len(result) == 5

    def test_returns_hex_strings(self):
        result = vp.get_colors_from_palette(3)
        for c in result:
            assert c.startswith("#")
            assert len(c) == 7

    def test_custom_palette(self):
        result = vp.get_colors_from_palette(4, pallete_name="viridis")
        assert len(result) == 4


# ---------------------------------------------------------------------------
# plot_scatter_xyz_panels
# ---------------------------------------------------------------------------

class TestPlotScatterXyzPanels:
    def _make_df(self, n=20):
        rng = np.random.default_rng(42)
        return pd.DataFrame({"x": rng.standard_normal(n),
                             "y": rng.standard_normal(n),
                             "z": rng.standard_normal(n),
                             "group": np.tile(["a", "b"], n // 2)})

    def test_returns_figure(self):
        import plotly.graph_objects as go
        df = self._make_df()
        fig = vp.plot_scatter_xyz_panels(df, coord_columns=["x", "y", "z"])
        assert isinstance(fig, go.Figure)

    def test_three_subplots(self):
        df = self._make_df()
        fig = vp.plot_scatter_xyz_panels(df, coord_columns=["x", "y", "z"])
        assert len(fig.data) == 3

    def test_group_by_creates_legend_groups(self):
        df = self._make_df()
        fig = vp.plot_scatter_xyz_panels(df, coord_columns=["x", "y", "z"], group_by="group")
        legend_groups = {t.legendgroup for t in fig.data}
        assert legend_groups == {"a", "b"}

    def test_displ_threshold_applied(self):
        df = self._make_df()
        fig = vp.plot_scatter_xyz_panels(df, coord_columns=["x", "y", "z"], displ_threshold=2.0)
        assert tuple(fig.layout.xaxis.range) == (-2.0, 2.0)

    def test_wrong_coord_columns_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError):
            vp.plot_scatter_xyz_panels(df, coord_columns=["x", "y"])

    def test_accepts_numpy_array(self):
        import plotly.graph_objects as go
        arr = np.zeros((10, 3))
        fig = vp.plot_scatter_xyz_panels(arr)
        assert isinstance(fig, go.Figure)


# ---------------------------------------------------------------------------
# plot_scatter_3d
# ---------------------------------------------------------------------------

class TestPlotScatter3d:
    def _make_df(self, n=15):
        rng = np.random.default_rng(0)
        return pd.DataFrame({"x": rng.standard_normal(n),
                             "y": rng.standard_normal(n),
                             "z": rng.standard_normal(n),
                             "val": rng.uniform(0, 1, n)})

    def test_returns_figure(self):
        import plotly.graph_objects as go
        df = self._make_df()
        fig = vp.plot_scatter_3d(df, coord_columns=["x", "y", "z"])
        assert isinstance(fig, go.Figure)

    def test_single_trace(self):
        df = self._make_df()
        fig = vp.plot_scatter_3d(df, coord_columns=["x", "y", "z"])
        assert len(fig.data) == 1

    def test_color_column_sets_marker_color(self):
        df = self._make_df()
        fig = vp.plot_scatter_3d(df, coord_columns=["x", "y", "z"], color_column="val")
        assert fig.data[0].marker.color is not None

    def test_wrong_coord_columns_raises(self):
        df = self._make_df()
        with pytest.raises(ValueError):
            vp.plot_scatter_3d(df, coord_columns=["x", "y"])


# ---------------------------------------------------------------------------
# plot_grouped_box
# ---------------------------------------------------------------------------

class TestPlotGroupedBox:
    def _make_df(self, n=30):
        rng = np.random.default_rng(7)
        return pd.DataFrame({"group": np.tile(["A", "B", "C"], n // 3),
                             "value": rng.standard_normal(n)})

    def test_returns_figure(self):
        import plotly.graph_objects as go
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column="group", value_column="value")
        assert isinstance(fig, go.Figure)

    def test_one_box_per_group(self):
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column="group", value_column="value")
        assert len(fig.data) == 3

    def test_group_names_match(self):
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column="group", value_column="value")
        names = {t.name for t in fig.data}
        assert names == {"A", "B", "C"}

    def test_title_applied(self):
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column="group", value_column="value",
                                  title="My Title")
        assert fig.layout.title.text == "My Title"


# ---------------------------------------------------------------------------
# add_xyz_heatmap_row
# ---------------------------------------------------------------------------

class TestAddXyzHeatmapRow:
    def test_adds_three_traces(self):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=3)
        slices = [np.zeros((4, 4)), np.ones((4, 4)), np.eye(4)]
        vp.add_xyz_heatmap_row(fig, slices, row=1)
        assert len(fig.data) == 3

    def test_wrong_slice_count_raises(self):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=3)
        with pytest.raises(ValueError):
            vp.add_xyz_heatmap_row(fig, [np.zeros((4, 4)), np.zeros((4, 4))], row=1)

    def test_coloraxis_propagated(self):
        from plotly.subplots import make_subplots
        fig = make_subplots(rows=1, cols=3)
        slices = [np.zeros((3, 3))] * 3
        vp.add_xyz_heatmap_row(fig, slices, row=1, coloraxis="coloraxis2")
        for trace in fig.data:
            assert trace.coloraxis == "coloraxis2"
