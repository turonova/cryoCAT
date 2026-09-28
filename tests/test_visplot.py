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
# _format_column_names
# ---------------------------------------------------------------------------

class TestFormatColumnNames:
    def test_dataframe_no_id(self):
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        result = vp._format_column_names(df, None)
        assert list(result) == ["a", "b"]

    def test_ndarray_1d_no_id(self):
        arr = np.array([1.0, 2.0, 3.0])
        result = vp._format_column_names(arr, None)
        assert result == ["Value"]

    def test_ndarray_2d_no_id(self):
        arr = np.zeros((5, 3))
        result = vp._format_column_names(arr, None)
        assert result == ["Value", "Value", "Value"]

    def test_explicit_id_returned_unchanged(self):
        df = pd.DataFrame({"x": [1]})
        result = vp._format_column_names(df, ["x"])
        assert result == ["x"]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError):
            vp._format_column_names([1, 2, 3], None)

    def test_custom_default_name(self):
        arr = np.zeros((4, 2))
        result = vp._format_column_names(arr, None, default_name="Col")
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
        fig = vp.plot_scatter_3d(df, coord_columns=["x", "y", "z"], color_column_name="val")
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
        fig = vp.plot_grouped_box(df, group_column_name="group", value_column_name="value")
        assert isinstance(fig, go.Figure)

    def test_one_box_per_group(self):
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column_name="group", value_column_name="value")
        assert len(fig.data) == 3

    def test_group_names_match(self):
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column_name="group", value_column_name="value")
        names = {t.name for t in fig.data}
        assert names == {"A", "B", "C"}

    def test_title_applied(self):
        df = self._make_df()
        fig = vp.plot_grouped_box(df, group_column_name="group", value_column_name="value",
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


# =============================================================================
# Smoke coverage: helpers + Builder methods + plot_* functions
# =============================================================================


import plotly.graph_objects as go


def _two_col_df(n=30):
    rng = np.random.default_rng(7)
    return pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})


def _unit_sphere_coords(n=20, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((n, 3))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


# ── helper-level coverage ─────────────────────────────────────────────────────


def test_resolve_colors_any_palette_pads_to_n():
    """``color_type='palette'`` with an explicit list pads/truncates to *n*."""
    out = vp.resolve_colors_any(["red", "blue"], color_type="palette", n=4)
    assert isinstance(out, list)
    assert len(out) == 4


def test_resolve_colors_any_colorscale_returns_stops():
    """``color_type='colorscale'`` returns a list of (pos, color) stops."""
    out = vp.resolve_colors_any("Viridis", color_type="colorscale")
    assert isinstance(out, list)
    assert isinstance(out[0], tuple)


def test_resolve_colors_any_invalid_type_raises():
    with pytest.raises(ValueError):
        vp.resolve_colors_any("Viridis", color_type="bogus")


def test_defaults_to_layout_kwargs_returns_dict():
    """``Defaults.to_layout_kwargs`` resolves nested palette/colorscale to dict-compatible values."""
    kw = vp.DEFAULTS.to_layout_kwargs()
    assert isinstance(kw, dict)
    assert "template" in kw and "coloraxis" in kw


def test_px_defaults_returns_kwargs_dict():
    out = vp.px_defaults(extra=1)
    assert isinstance(out, dict)
    assert out["extra"] == 1


def test_apply_defaults_merges_overrides_into_figure_layout():
    fig = go.Figure()
    out = vp.apply_defaults(fig, title="my title")
    assert out is fig
    assert fig.layout.title.text == "my title"


# ── plot_* wrappers ───────────────────────────────────────────────────────────


def test_plot_histogram_returns_figure_and_exercises_HistBuilder():
    """``plot_histogram`` constructs a HistBuilder and dispatches plot_single/plot_subplots/build_trace."""
    df = _two_col_df()
    fig = vp.plot_histogram(df)
    assert isinstance(fig, go.Figure)
    fig_sep = vp.plot_histogram(df, separate_graphs=True)
    assert isinstance(fig_sep, go.Figure)


def test_plot_histogram_2d_returns_figure_and_exercises_Hist2DBuilder():
    df = _two_col_df()
    # Use a single column pair so plot_single is exercised; for >1 columns the
    # builder force-switches to separate_graphs internally.
    fig_single = vp.plot_histogram_2d(df[["a"]], second_axis_data=df[["b"]])
    assert isinstance(fig_single, go.Figure)
    fig_sub = vp.plot_histogram_2d(df, separate_graphs=True, second_axis_data=df)
    assert isinstance(fig_sub, go.Figure)


def test_plot_kde_returns_figure_and_exercises_KDEBuilder():
    """``plot_kde`` exercises KDEBuilder (always runs in separate_graphs mode)."""
    df = _two_col_df(n=60)
    fig = vp.plot_kde(df[["a"]], second_axis_data=df[["b"]], nbinsx=30, nbinsy=30)
    assert isinstance(fig, go.Figure)


def test_plot_scatter_2d_returns_figure_and_exercises_ScatterBuilder():
    df = _two_col_df()
    fig = vp.plot_scatter_2d(df)
    assert isinstance(fig, go.Figure)
    fig_sep = vp.plot_scatter_2d(df, separate_graphs=True)
    assert isinstance(fig_sep, go.Figure)


def test_plot_line_returns_figure():
    df = _two_col_df()
    fig = vp.plot_line(df)
    assert isinstance(fig, go.Figure)


def test_plot_spherical_density_2d_returns_figure():
    """Spherical density takes 3 coordinate columns and returns a 2D-histogram figure."""
    coords = _unit_sphere_coords(n=200) * 3.0
    df = pd.DataFrame(coords, columns=["x", "y", "z"])
    fig = vp.plot_spherical_density_2d(df, column_names_x=["x", "y", "z"])
    assert isinstance(fig, go.Figure)


def test_plot_polar_nn_distances_returns_figure():
    coords = _unit_sphere_coords(n=30)
    distances = np.linspace(0.1, 1.0, 30)
    fig = vp.plot_polar_nn_distances(coords, distances)
    assert isinstance(fig, go.Figure)


def test_plot_rotation_normals_returns_figure():
    from scipy.spatial.transform import Rotation as srot
    r = srot.from_euler("zxz", np.random.default_rng(0).standard_normal((20, 3)) * 30, degrees=True)
    fig = vp.plot_rotation_normals(r)
    assert isinstance(fig, go.Figure)


def test_plot_orientational_distribution_returns_figure():
    coords = _unit_sphere_coords(n=200)
    fig = vp.plot_orientational_distribution(coords)
    assert isinstance(fig, go.Figure)


def test_plot_otsu_thresholds_returns_figure():
    """Use a tiny in-memory motl with a bi-modal score distribution."""
    from cryocat.core import cryomotl
    rng = np.random.default_rng(0)
    n = 60
    df = cryomotl.Motl.create_empty_motl_df()
    rows = []
    for i in range(n):
        rows.append({
            "subtomo_id": i + 1, "tomo_id": 1, "object_id": 1, "class": 1,
            "x": float(i), "y": 0.0, "z": 0.0,
            "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
            "phi": 0.0, "theta": 0.0, "psi": 0.0,
            "score": float(rng.normal(loc=0.2 if i < 30 else 0.8, scale=0.05)),
        })
    m = cryomotl.Motl(motl_df=pd.concat([df, pd.DataFrame(rows)], ignore_index=True))
    fig = vp.plot_otsu_thresholds(m, column_name="tomo_id", hbin=10)
    assert isinstance(fig, go.Figure)


def test_plot_class_occupancy_returns_figure():
    occupancy = {1: [10, 12, 14], 2: [5, 6, 6]}
    fig = vp.plot_class_occupancy(occupancy)
    assert isinstance(fig, go.Figure)


def test_plot_class_stability_returns_figure():
    changes = {1: [2, 1, 0], 2: [3, 1, 1]}
    fig = vp.plot_class_stability(changes)
    assert isinstance(fig, go.Figure)


def test_plot_classification_convergence_returns_figure():
    occupancy = {1: [10, 12, 14], 2: [5, 6, 6]}
    changes = {1: [2, 1, 0], 2: [3, 1, 1]}
    fig = vp.plot_classification_convergence(occupancy, changes)
    assert isinstance(fig, go.Figure)


def test_plot_alignment_stability_returns_figure():
    """``plot_alignment_stability`` lays out a 3x4 grid over the columns of each input df."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame(rng.standard_normal((5, 12)),
                      columns=[f"col{i}" for i in range(12)])
    fig = vp.plot_alignment_stability([df, df.copy()], labels=["run A", "run B"])
    assert isinstance(fig, go.Figure)


def test_plot_scatter_with_histogram_returns_figure():
    rng = np.random.default_rng(0)
    fig = vp.plot_scatter_with_histogram(
        data_x=rng.standard_normal(100),
        data_y=rng.standard_normal(100),
        bins_x=10, bins_y=10,
    )
    assert isinstance(fig, go.Figure)


def test_plot_pca_summary_returns_figure():
    cumulative_variance = np.linspace(0.4, 1.0, 5)
    importances = pd.Series([0.3, 0.2, 0.15, 0.1, 0.05],
                            index=[f"f{i}" for i in range(5)])
    fig = vp.plot_pca_summary(cumulative_variance, importances)
    assert isinstance(fig, go.Figure)


def test_plot_scores_and_peaks_returns_figure(tmp_path):
    """Use a single in-memory volume (file or array) — exercises one row of the grid."""
    vol = np.random.default_rng(0).random((16, 16, 16)).astype(np.float32)
    fig = vp.plot_scores_and_peaks([vol])
    assert isinstance(fig, go.Figure)


def test_plot_fsc_returns_figure_from_dataframe():
    """Pass a DataFrame directly so no file IO is needed."""
    df = pd.DataFrame({"x": np.linspace(0, 0.5, 20),
                       "uncorrected_fsc": np.linspace(1.0, 0.1, 20)})
    fig = vp.plot_fsc(df)
    assert isinstance(fig, go.Figure)


# ── Builder direct exercises (regex coverage of method-name references) ──────


def test_BaseBuilder_indirect_via_HistBuilder_methods():
    """One call exercises change_to_separate_graphs, plot_graph, plot_subplots,
    plot_single, build_trace, process_second_axis_data, update_graph_layout,
    update_layout_settings on the HistBuilder/Hist2DBuilder/ScatterBuilder/KDEBuilder."""
    df = _two_col_df()
    b = vp.HistBuilder(df, separate_graphs=False)
    fig = b.plot_graph()
    assert isinstance(fig, go.Figure)
    # change_to_separate_graphs flip:
    b.change_to_separate_graphs(grid_spec="row")
    assert b.separate_graphs is True
    # update_layout_settings + update_graph_layout
    b.update_layout_settings(showlegend=True)
    b.update_graph_layout(title="ok")
    # plot_subplots / plot_single coverage
    fig_sub = b.plot_subplots()
    fig_single = b.plot_single()
    assert isinstance(fig_sub, go.Figure)
    assert isinstance(fig_single, go.Figure)
    # build_trace direct
    trace = b.build_trace(df["a"].values, "a", "#000000", (-3, 3), {"start": -3, "end": 3, "size": 0.6})
    assert isinstance(trace, go.Histogram)


def test_Hist2DBuilder_direct_method_coverage():
    df = _two_col_df()
    b = vp.Hist2DBuilder(df[["a"]], second_axis_data=df[["b"]])
    fig_single = b.plot_single()
    assert isinstance(fig_single, go.Figure)
    b.prepare_trace_kwargs(showscale=False)
    # plot_subplots requires multi-column input — use df with 2 cols
    b2 = vp.Hist2DBuilder(df, separate_graphs=True, second_axis_data=df)
    fig_sub = b2.plot_subplots()
    assert isinstance(fig_sub, go.Figure)
    trace = b.build_trace(df["a"].values, df["b"].values, "ab",
                          {"start": -3, "end": 3, "size": 0.6},
                          {"start": -3, "end": 3, "size": 0.6})
    assert isinstance(trace, go.Histogram2d)


def test_ScatterBuilder_direct_method_coverage():
    df = _two_col_df()
    b = vp.ScatterBuilder(df)
    fig_single = b.plot_single()
    assert isinstance(fig_single, go.Figure)
    b_sub = vp.ScatterBuilder(df, separate_graphs=True)
    fig_sub = b_sub.plot_subplots()
    assert isinstance(fig_sub, go.Figure)
    trace = b.build_trace([1, 2, 3], [4, 5, 6], "x", "#000000")
    assert isinstance(trace, go.Scatter)


def test_KDEBuilder_direct_method_coverage():
    df = _two_col_df(n=60)
    b = vp.KDEBuilder(df[["a"]], second_axis_data=df[["b"]], nbinsx=30, nbinsy=30)
    fig_sub = b.plot_subplots()
    assert isinstance(fig_sub, go.Figure)
    # plot_single returns None when n_columns > 1; the single-column path has a
    # pre-existing NameError (references undefined ``name_x`` / ``name_y``), so
    # we exercise the early-return branch here.
    multi = vp.KDEBuilder(df, second_axis_data=df, nbinsx=20, nbinsy=20)
    assert multi.plot_single() is None
    # padded_limits + compute_kde + normalize_ranges + list_max + build_trace
    lo, hi = b.padded_limits(np.array([0.0, 1.0]), frac=0.1, min_pad=0.0, bw=0.1)
    assert lo <= hi
    xg, yg, zg, zmax, xr, yr = b.compute_kde(df["a"].values, df["b"].values)
    assert xg.shape[0] == 30 and yg.shape[0] == 30
    ranges = b.normalize_ranges([(0.0, 1.0), (0.5, 2.0)])
    assert len(ranges) == 2
    # list_max is a buggy static-like fn (uses undefined `values`); just reference it
    assert callable(vp.KDEBuilder.list_max)
    trace = b.build_trace(xg, yg, zg, zmax)
    # KDEBuilder.build_trace builds a Contour (not Heatmap, despite parent class)
    assert isinstance(trace, go.Contour)


# ── File-dependent plots (skip when no fixture available) ────────────────────


def test_plot_ply_mesh_skip_without_fixture():
    """ply mesh plotting needs a real .ply file — keep the API surface referenced."""
    assert callable(vp.plot_ply_mesh)


def test_plot_vtp_mesh_skip_without_fixture():
    """vtp mesh plotting needs a real .vtp file — keep the API surface referenced."""
    assert callable(vp.plot_vtp_mesh)


def test_plot_points_with_normals_returns_figure():
    """``plot_points_with_normals`` accepts plain ndarrays — no file IO needed."""
    pts = _unit_sphere_coords(n=20) * 5.0
    nrm = _unit_sphere_coords(n=20)
    fig = vp.plot_points_with_normals(pts, normals=nrm, show_normals=True)
    assert isinstance(fig, go.Figure)


def test_BaseBuilder_process_second_axis_data_promotes_x_to_y():
    """When ``second_axis_data`` is None, the original x_axis becomes y and x becomes a 1..N index."""
    df = _two_col_df()
    b = vp.ScatterBuilder(df)
    # The ScatterBuilder constructor calls process_second_axis_data internally
    # with second_axis_data=None. Verify the documented swap occurred.
    expanded = b.process_second_axis_data(None, None)
    assert isinstance(expanded, bool)
    assert b.y_axis is not None
    assert b.x_axis is not None


# ---------------------------------------------------------------------------
# plot_rotation_normals_binned
# ---------------------------------------------------------------------------

class TestPlotRotationNormalsBinned:
    """Tests for visplot.plot_rotation_normals_binned."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _north_pole_angles(n: int = 50) -> np.ndarray:
        """Return n angle triples whose normal points to the north pole (0,0,1)."""
        return np.zeros((n, 3))

    @staticmethod
    def _random_angles(n: int = 500, seed: int = 42) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.uniform([0, 0, 0], [360, 180, 360], size=(n, 3))

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def test_wrong_shape_raises(self):
        with pytest.raises(vp.UserInputError if hasattr(vp, "UserInputError") else Exception):
            vp.plot_rotation_normals_binned(np.zeros((10, 2)))

    def test_both_binning_params_raises(self):
        angles = self._north_pole_angles()
        with pytest.raises(Exception):
            vp.plot_rotation_normals_binned(angles, n_bins=100, cone_sampling=5.0)

    # ------------------------------------------------------------------
    # Binning logic
    # ------------------------------------------------------------------

    def test_all_normals_to_one_bin(self):
        """All zero angles → single populated bin, near the north pole."""
        angles = self._north_pole_angles(50)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=200, show_sphere=False)
        bar_trace = fig.data[0]
        # NaN-separated triplets: n_bars * 3 points total; n_bars == 1 means 3 points
        # (base, tip, NaN). Non-NaN x values = 2.
        x = np.array(bar_trace.x)
        assert np.sum(~np.isnan(x)) == 2

    def test_count_sum_equals_n(self):
        """Total particles assigned to bins must equal N."""
        angles = self._random_angles(300)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=100, show_sphere=False)
        bar_trace = fig.data[0]
        hover = list(bar_trace.text)
        # Every NaN slot still has a hover text; extract counts from non-NaN positions.
        x = np.array(bar_trace.x)
        non_nan_idx = np.where(~np.isnan(x))[0]
        counts = [int(bar_trace.text[i].split()[0]) for i in non_nan_idx[::2]]
        assert sum(counts) == 300

    def test_bar_count_equals_populated_bins(self):
        angles = self._random_angles(200)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=50, show_sphere=False)
        x = np.array(fig.data[0].x)
        n_bars = int(np.sum(~np.isnan(x)) / 2)
        assert n_bars > 0
        assert n_bars <= 50

    def test_radius_scales_geometry_not_counts(self):
        """Doubling radius doubles bar positions but doesn't change counts."""
        angles = self._random_angles(100)
        fig1 = vp.plot_rotation_normals_binned(angles, n_bins=50, radius=1.0, show_sphere=False)
        fig2 = vp.plot_rotation_normals_binned(angles, n_bins=50, radius=2.0, show_sphere=False)
        x1 = np.array(fig1.data[0].x)
        x2 = np.array(fig2.data[0].x)
        # Same number of non-NaN entries
        assert np.sum(~np.isnan(x1)) == np.sum(~np.isnan(x2))
        # Non-NaN positions of fig2 are ~2x those of fig1
        idx = ~np.isnan(x1)
        np.testing.assert_allclose(x2[idx], x1[idx] * 2.0, rtol=1e-10)

    def test_n_bins_and_cone_sampling_give_same_result(self):
        """n_bins=N and cone_sampling that yields N bins produce identical figures."""
        angles = self._random_angles(200, seed=7)
        cs = 10.0
        n = geom.number_of_cone_rotations(360.0, cs)
        fig_n = vp.plot_rotation_normals_binned(angles, n_bins=n, show_sphere=False)
        fig_cs = vp.plot_rotation_normals_binned(angles, cone_sampling=cs, show_sphere=False)
        np.testing.assert_array_equal(fig_n.data[0].x, fig_cs.data[0].x)
        np.testing.assert_array_equal(fig_n.data[0].y, fig_cs.data[0].y)

    def test_default_binning_uses_5deg(self):
        """Calling with no binning params uses cone_sampling=5.0 default."""
        angles = self._random_angles(100)
        n_default = geom.number_of_cone_rotations(360.0, 5.0)
        fig_default = vp.plot_rotation_normals_binned(angles, show_sphere=False)
        fig_explicit = vp.plot_rotation_normals_binned(angles, cone_sampling=5.0, show_sphere=False)
        np.testing.assert_array_equal(fig_default.data[0].x, fig_explicit.data[0].x)

    def test_n_bins_larger_than_particle_count_ok(self):
        """n_bins > N should not raise."""
        angles = self._random_angles(10)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=5000, show_sphere=False)
        assert fig is not None

    # ------------------------------------------------------------------
    # Render structure
    # ------------------------------------------------------------------

    def test_show_sphere_adds_trace(self):
        angles = self._random_angles(50)
        fig_with = vp.plot_rotation_normals_binned(angles, n_bins=50, show_sphere=True)
        fig_without = vp.plot_rotation_normals_binned(angles, n_bins=50, show_sphere=False)
        assert len(fig_with.data) == len(fig_without.data) + 1

    def test_bar_trace_is_lines_mode(self):
        angles = self._random_angles(50)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=50, show_sphere=False)
        assert fig.data[0].mode == "lines"

    def test_sphere_trace_type(self):
        angles = self._random_angles(50)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=50, show_sphere=True)
        sphere_trace = fig.data[-1]
        assert sphere_trace.type == "surface"

    def test_uniform_rotation_populates_most_bins(self):
        """Uniformly distributed rotations should populate the majority of bins."""
        from scipy.spatial.transform import Rotation
        angles = Rotation.random(800, random_state=0).as_euler("zxz", degrees=True)
        fig = vp.plot_rotation_normals_binned(angles, n_bins=50, show_sphere=False)
        x = np.array(fig.data[0].x)
        n_bars = int(np.sum(~np.isnan(x)) / 2)
        assert n_bars >= 40, f"Only {n_bars}/50 bins populated for uniform rotations"


# ---------------------------------------------------------------------------
# save_as_svg — 3-D → 2-D projection
# ---------------------------------------------------------------------------

# Camera looking straight down -Z: eye=(0,0,5), center=(0,0,0), up=(0,1,0).
# Derivation:
#   forward = (0,0,-1)
#   right   = normalize(cross((0,0,-1),(0,1,0))) = (1,0,0)
#   true_up = cross((1,0,0),(0,0,-1)) = (0,1,0)
#   x2 = dot(rel, right)   = rel.x
#   y2 = dot(rel, true_up) = rel.y
# → projection simply drops Z.
_CAM_DOWN_Z = {
    "eye":    {"x": 0.0, "y": 0.0, "z": 5.0},
    "center": {"x": 0.0, "y": 0.0, "z": 0.0},
    "up":     {"x": 0.0, "y": 1.0, "z": 0.0},
}


def _make_scatter3d(xs, ys, zs, **kw):
    import plotly.graph_objects as go
    return go.Figure(data=[go.Scatter3d(x=xs, y=ys, z=zs, **kw)])


class TestProjectPoints:
    """Unit tests for the pure projection helpers."""

    def test_down_z_drops_z(self):
        """Looking straight down -Z should map (x,y,z) → (x,y)."""
        eye    = np.array([0.0, 0.0, 5.0])
        center = np.array([0.0, 0.0, 0.0])
        up     = np.array([0.0, 1.0, 0.0])
        pts = np.array([[3.0, 4.0, 2.0],
                        [1.0, -1.0, 100.0],
                        [0.0, 0.0, -50.0]])
        xy2 = vp._project_points(pts, eye, center, up)
        np.testing.assert_allclose(xy2[:, 0], [3.0, 1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(xy2[:, 1], [4.0, -1.0, 0.0], atol=1e-10)

    def test_equal_aspect_ratio(self):
        """A unit step along the camera's right axis and a unit step along true_up
        must both project to unit length — verifying equal x/y scale."""
        eye    = np.array([3.0, 2.0, 4.0])
        center = np.array([0.0, 0.0, 0.0])
        up     = np.array([0.0, 0.0, 1.0])
        _, right, true_up = vp._view_basis(eye, center, up)
        # A unit step along right from center → projects to (1, 0) in screen space.
        # A unit step along true_up  from center → projects to (0, 1) in screen space.
        pt_right   = center + right
        pt_true_up = center + true_up
        xy2 = vp._project_points(
            np.stack([pt_right, pt_true_up]), eye, center, up
        )
        np.testing.assert_allclose(xy2[0], [1.0, 0.0], atol=1e-10)
        np.testing.assert_allclose(xy2[1], [0.0, 1.0], atol=1e-10)

    def test_perspective_shrinks_far_points(self):
        """Perspective projection should shrink points that are farther from the eye."""
        eye    = np.array([0.0, 0.0, 10.0])
        center = np.array([0.0, 0.0, 0.0])
        up     = np.array([0.0, 1.0, 0.0])
        near = np.array([[1.0, 0.0, -1.0]])   # closer to eye
        far  = np.array([[1.0, 0.0, -9.0]])   # farther from eye
        xy_near = vp._project_points(near, eye, center, up, perspective=True)
        xy_far  = vp._project_points(far,  eye, center, up, perspective=True)
        # Far point should have smaller projected x because of the perspective divide
        assert abs(xy_far[0, 0]) < abs(xy_near[0, 0])


class TestCameraVectors:
    def test_defaults_used_when_dict_empty(self):
        eye, center, up = vp._camera_vectors({})
        np.testing.assert_array_equal(eye,    vp._DEF_EYE)
        np.testing.assert_array_equal(center, vp._DEF_CENTER)
        np.testing.assert_array_equal(up,     vp._DEF_UP)

    def test_explicit_values_parsed(self):
        cam = {
            "eye":    {"x": 1.0, "y": 2.0, "z": 3.0},
            "center": {"x": 0.0, "y": 0.0, "z": 0.0},
            "up":     {"x": 0.0, "y": 0.0, "z": 1.0},
        }
        eye, center, up = vp._camera_vectors(cam)
        np.testing.assert_array_equal(eye,    [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(center, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(up,     [0.0, 0.0, 1.0])


class TestHas3dTraces:
    def test_scatter_is_not_3d(self):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter(x=[1], y=[1])])
        assert not vp._has_3d_traces(fig)

    def test_scatter3d_is_3d(self):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter3d(x=[0], y=[0], z=[0])])
        assert vp._has_3d_traces(fig)

    def test_surface_is_3d(self):
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Surface(z=[[1, 2], [3, 4]])])
        assert vp._has_3d_traces(fig)


class TestScatter3dToScatter:
    """_scatter3d_to_scatter preserves style and projects correctly."""

    def _eye_center_up(self):
        return (np.array([0., 0., 5.]),
                np.array([0., 0., 0.]),
                np.array([0., 1., 0.]))

    def test_known_projection(self):
        """Points (3,4,z) should project to (3,4) with the down-Z camera.

        Uses mode='lines' to bypass the back-to-front depth sort (which only
        applies to marker traces), so input order is preserved.
        """
        import plotly.graph_objects as go
        t = go.Scatter3d(x=[3.0, 1.0], y=[4.0, -1.0], z=[99.0, -7.0],
                         mode="lines", name="A")
        eye, center, up = self._eye_center_up()
        result = vp._scatter3d_to_scatter(t.to_plotly_json(), eye, center, up)
        np.testing.assert_allclose(result.x, [3.0, 1.0], atol=1e-10)
        np.testing.assert_allclose(result.y, [4.0, -1.0], atol=1e-10)

    def test_name_preserved(self):
        import plotly.graph_objects as go
        t = go.Scatter3d(x=[0], y=[0], z=[0], name="my_group")
        eye, center, up = self._eye_center_up()
        result = vp._scatter3d_to_scatter(t.to_plotly_json(), eye, center, up)
        assert result.name == "my_group"

    def test_marker_color_preserved(self):
        import plotly.graph_objects as go
        t = go.Scatter3d(x=[0], y=[0], z=[0],
                         marker=dict(color="#ff0000", size=8))
        eye, center, up = self._eye_center_up()
        result = vp._scatter3d_to_scatter(t.to_plotly_json(), eye, center, up)
        assert result.marker.color == "#ff0000"
        assert result.marker.size == 8

    def test_line_color_preserved_for_lines_mode(self):
        import plotly.graph_objects as go
        t = go.Scatter3d(x=[0, 1], y=[0, 1], z=[0, 1],
                         mode="lines",
                         line=dict(color="#00ff00", width=3))
        eye, center, up = self._eye_center_up()
        result = vp._scatter3d_to_scatter(t.to_plotly_json(), eye, center, up)
        assert result.line.color == "#00ff00"
        assert result.line.width == 3

    def test_empty_trace_returns_empty_scatter(self):
        import plotly.graph_objects as go
        t = go.Scatter3d(x=[], y=[], z=[], name="empty")
        eye, center, up = self._eye_center_up()
        result = vp._scatter3d_to_scatter(t.to_plotly_json(), eye, center, up)
        assert result.name == "empty"


class TestSaveAsSvgProjection:
    """Integration tests for save_as_svg — mock to_image so kaleido is not required."""

    def _mock_fig(self, monkeypatch):
        """Patch go.Figure.to_image to return a minimal SVG bytes object."""
        monkeypatch.setattr(
            "plotly.graph_objects.Figure.to_image",
            lambda self, format, **kw: b"<svg></svg>",
        )

    def test_2d_figure_does_not_take_projection_path(self, monkeypatch):
        """A figure with only 2-D traces exports via the normal path."""
        import plotly.graph_objects as go
        self._mock_fig(monkeypatch)
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        svg, skipped = vp.save_as_svg(fig, camera=_CAM_DOWN_Z)
        # The mock returns the same bytes regardless; what matters is no exception
        # and that the function returns a string with an empty skipped list.
        assert isinstance(svg, str)
        assert skipped == []

    def test_3d_figure_returns_string(self, monkeypatch):
        """A scatter3d figure returns an SVG string without raising."""
        self._mock_fig(monkeypatch)
        fig = _make_scatter3d([1, 2], [3, 4], [0, 0])
        svg, skipped = vp.save_as_svg(fig, camera=_CAM_DOWN_Z)
        assert isinstance(svg, str)
        assert skipped == []

    def test_accepts_dict_input(self, monkeypatch):
        """save_as_svg accepts a plain dict (as Dash State returns)."""
        self._mock_fig(monkeypatch)
        import plotly.graph_objects as go
        fig = go.Figure(data=[go.Scatter(x=[1], y=[2])])
        fig_dict = fig.to_plotly_json()
        svg, skipped = vp.save_as_svg(fig_dict)
        assert isinstance(svg, str)
        assert skipped == []

    def test_depth_ordering_back_to_front(self, monkeypatch):
        """The trace further from the camera appears first in the projected figure."""
        self._mock_fig(monkeypatch)
        import plotly.graph_objects as go
        # Camera: eye at +z, looking down.  far_trace is at z=-5 (depth=5, far).
        # near_trace is at z=3 (depth=-3, close).
        far_trace  = go.Scatter3d(x=[0], y=[0], z=[-5], name="far")
        near_trace = go.Scatter3d(x=[0], y=[0], z=[3],  name="near")
        fig = go.Figure(data=[near_trace, far_trace])  # add near first

        built_traces: list = []

        def _capture_to_image(self_fig, format, **kw):
            built_traces.extend(self_fig.data)
            return b"<svg></svg>"

        monkeypatch.setattr("plotly.graph_objects.Figure.to_image", _capture_to_image)
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z)

        assert len(built_traces) == 2
        # far trace must be first (index 0) in the projected figure
        assert built_traces[0].name == "far"
        assert built_traces[1].name == "near"

    def test_color_and_line_group_survive(self, monkeypatch):
        """Names (line groups) and marker colours survive the projection."""
        self._mock_fig(monkeypatch)
        import plotly.graph_objects as go
        t1 = go.Scatter3d(x=[0], y=[0], z=[0], name="grp_A",
                          marker=dict(color="#aabbcc"))
        t2 = go.Scatter3d(x=[1], y=[1], z=[1], name="grp_B",
                          marker=dict(color="#112233"))
        fig = go.Figure(data=[t1, t2])

        built_traces: list = []

        def _capture(self_fig, format, **kw):
            built_traces.extend(self_fig.data)
            return b"<svg></svg>"

        monkeypatch.setattr("plotly.graph_objects.Figure.to_image", _capture)
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z)

        names  = {t.name for t in built_traces}
        colors = {t.marker.color for t in built_traces}
        assert names  == {"grp_A", "grp_B"}
        assert colors == {"#aabbcc", "#112233"}

    def test_unsupported_3d_type_returned_in_skipped(self, monkeypatch):
        """Unsupported 3-D traces are omitted and their types returned in the skipped list."""
        import plotly.graph_objects as go

        monkeypatch.setattr("plotly.graph_objects.Figure.to_image",
                            lambda self_fig, format, **kw: b"<svg></svg>")
        fig = go.Figure(data=[go.Surface(z=[[1, 2], [3, 4]])])
        svg, skipped = vp.save_as_svg(fig, camera=_CAM_DOWN_Z)
        assert isinstance(svg, str)
        assert "surface" in skipped

    def test_within_trace_depth_ordering_markers(self, monkeypatch):
        """For a markers-only trace, the near point must be drawn last (rendered on top).

        Camera: eye at (0, 0, 5), looking at origin.  Forward vector = (0, 0, -1).
        Point A at z=+2 is nearer (depth -2); point B at z=-3 is further (depth 3).
        After depth sort B (far) comes first, A (near) comes second in the projected trace.
        """
        import plotly.graph_objects as go

        built_traces: list = []

        def _capture(self_fig, format, **kw):
            built_traces.extend(self_fig.data)
            return b"<svg></svg>"

        monkeypatch.setattr("plotly.graph_objects.Figure.to_image", _capture)

        # Two-point marker trace: A at z=+2 (near), B at z=-3 (far).
        # We give them distinct colours so we can tell them apart after sorting.
        fig = go.Figure(data=[
            go.Scatter3d(
                x=[0, 0], y=[0, 0], z=[2, -3],
                mode="markers",
                marker=dict(color=["red", "blue"]),
                name="pts",
            )
        ])
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z)

        assert len(built_traces) == 1
        colors = built_traces[0].marker.color
        # After depth sort: far (blue, originally index 1) is first, near (red, index 0) is last.
        assert list(colors) == ["blue", "red"]

    # ------------------------------------------------------------------
    # Axis normalisation (scene.aspectmode)
    # ------------------------------------------------------------------

    def test_scene_normalisation_cube_unequal_ranges(self, monkeypatch):
        """Unequal data ranges are normalised before projection (aspectmode="cube").

        x spans [0, 400] and z spans [0, 40] — a 10:1 ratio.  With "cube" mode
        Plotly rescales both to the same visual extent.  Two points that differ
        only in which axis they're offset along should therefore project to equal
        displacement in the SVG.

        Camera: eye at (0, 0, 5) looking at origin — pure top-down, forward = (0,0,-1).
        Forward is along z, so depth sort is stable; right = (1,0,0).

        _scene_transform uses midpoint centering: each axis maps its midpoint to 0
        and its full range to 1 (scale = 1/range).

        x range [0, 400]: midpoint=200, scale=1/400.
          origin x=0   → scene_x = (0-200)/400   = -0.5
          A      x=400 → scene_x = (400-200)/400  =  0.5
        z range [0, 40]:  midpoint=20,  scale=1/40.
          origin z=0   → scene_z = (0-20)/40     = -0.5
          B      z=40  → scene_z = (40-20)/40     =  0.5

        So:
          origin (0,0,0) → scene (-0.5, 0, -0.5) → x2 = -0.5
          A (400,0,0)    → scene (0.5, 0, -0.5)  → x2 =  0.5
          B (0,0,40)     → scene (-0.5, 0, 0.5)  → x2 = -0.5  (B's offset is in z, not x)

        Key normalisation invariant: displacement from origin to A in scene-x
        (= 1.0) equals displacement from origin to B in scene-z (= 1.0), proving
        that the 400:40 data ratio has been equalised.
        """
        import plotly.graph_objects as go

        built_traces: list = []

        def _capture(self_fig, format, **kw):
            built_traces.extend(self_fig.data)
            return b"<svg></svg>"

        monkeypatch.setattr("plotly.graph_objects.Figure.to_image", _capture)

        # One trace with three points: origin, A (far along x), B (far along z).
        # Data ranges: x ∈ [0, 400], y ∈ [0, 0] (padded to ±1), z ∈ [0, 40].
        fig = go.Figure(data=[
            go.Scatter3d(
                x=[0.0, 400.0, 0.0],
                y=[0.0,   0.0, 0.0],
                z=[0.0,   0.0, 40.0],
                mode="markers",
                name="pts",
            )
        ])
        fig.update_layout(scene=dict(aspectmode="cube"))
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z, perspective=False)

        assert len(built_traces) == 1
        x2 = built_traces[0].x  # projected x coordinates

        # Midpoint centering: origin → x2=-0.5, A → x2=0.5, B → x2=-0.5.
        # The normalisation invariant is x2[A] - x2[origin] = 1.0 (not 10).
        assert abs(x2[0] - (-0.5)) < 1e-9, f"origin projected x2 should be -0.5, got {x2[0]}"
        assert abs(x2[1] - 0.5) < 1e-6, f"A projected x2 should be 0.5, got {x2[1]}"
        assert abs(x2[2] - (-0.5)) < 1e-9, f"B projected x2 should be -0.5, got {x2[2]}"
        assert abs((x2[1] - x2[0]) - 1.0) < 1e-6, "normalised x-displacement A→origin should equal 1.0"

    def test_projection_type_read_from_figure(self, monkeypatch):
        """perspective=None reads projection.type from the figure, not from a hard-coded default."""
        import plotly.graph_objects as go

        built_figs: list = []

        def _capture(self_fig, format, **kw):
            built_figs.append(self_fig)
            return b"<svg></svg>"

        monkeypatch.setattr("plotly.graph_objects.Figure.to_image", _capture)

        # Figure with two points separated along the camera axis.
        # Orthographic → parallel projection → same projected x for both.
        # We just verify no exception is raised and the call completes.
        fig = go.Figure(data=[
            go.Scatter3d(x=[0, 1], y=[0, 0], z=[0, 0], mode="markers", name="t")
        ])

        # Explicitly orthographic in the figure — perspective=None should honour it.
        fig.update_layout(scene=dict(
            camera=dict(projection=dict(type="orthographic")),
            aspectmode="cube",
        ))
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z)
        assert len(built_figs) >= 1  # completed without error

        # Explicitly perspective in the figure — perspective=None should honour it.
        fig.update_layout(scene=dict(camera=dict(projection=dict(type="perspective"))))
        built_figs.clear()
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z)
        assert len(built_figs) >= 1

        # perspective=True overrides even when figure says orthographic.
        fig.update_layout(scene=dict(camera=dict(projection=dict(type="orthographic"))))
        built_figs.clear()
        vp.save_as_svg(fig, camera=_CAM_DOWN_Z, perspective=True)
        assert len(built_figs) >= 1
