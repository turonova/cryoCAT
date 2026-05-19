import numpy as np
import pytest
from cryocat.analysis import tmana

# IMPORTANT: pytest-mock needs to be installed within environment to run these tests


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def cube_volume():
    """20x20x20 volume with a 4x4x4 cube at [8:12, 8:12, 8:12]."""
    vol = np.zeros((20, 20, 20))
    vol[8:12, 8:12, 8:12] = 1.0
    return vol


@pytest.fixture
def peak_volume():
    """20x20x20 volume with a single voxel peak at the centre."""
    vol = np.zeros((20, 20, 20))
    vol[10, 10, 10] = 5.0
    return vol


# ── compute_scores_map_threshold_triangle ─────────────────────────────────────

class TestComputeScoresMapThresholdTriangle:
    def test_returns_scalar(self):
        arr = np.concatenate([np.zeros(90), np.ones(10)])
        assert np.ndim(tmana.compute_scores_map_threshold_triangle(arr)) == 0

    def test_threshold_within_data_range(self):
        arr = np.concatenate([np.full(90, 0.1), np.full(10, 1.0)])
        result = tmana.compute_scores_map_threshold_triangle(arr)
        assert arr[arr > 0].min() <= result <= arr.max()

    def test_2d_input_works(self):
        arr = np.concatenate([np.zeros(90), np.ones(10)]).reshape(10, 10)
        assert np.isfinite(tmana.compute_scores_map_threshold_triangle(arr))

    def test_3d_input_works(self):
        arr = np.zeros((10, 10, 10))
        arr[7:, :, :] = 1.0
        assert np.isfinite(tmana.compute_scores_map_threshold_triangle(arr))

    def test_all_equal_nonzero_returns_that_value(self):
        result = tmana.compute_scores_map_threshold_triangle(np.ones(100))
        assert result == pytest.approx(1.0)

    def test_threshold_does_not_exceed_max(self):
        rng = np.random.default_rng(0)
        arr = rng.uniform(0.1, 2.0, 500)
        assert tmana.compute_scores_map_threshold_triangle(arr) <= arr.max()

    def test_threshold_is_finite_for_random_data(self):
        rng = np.random.default_rng(42)
        arr = rng.uniform(0.0, 1.0, 1000)
        assert np.isfinite(tmana.compute_scores_map_threshold_triangle(arr))

    @pytest.mark.parametrize("n_background,background_val,n_signal,signal_val", [
        (900, 0.05, 100, 1.0),
        (800, 0.1,  200, 0.8),
    ])
    def test_bimodal_threshold_below_signal(self, n_background, background_val, n_signal, signal_val):
        arr = np.concatenate([np.full(n_background, background_val), np.full(n_signal, signal_val)])
        assert tmana.compute_scores_map_threshold_triangle(arr) <= signal_val


# ── create_starting_parameters_1D ─────────────────────────────────────────────

class TestCreateStartingParameters1D:
    def test_returns_three_values(self, peak_volume):
        assert len(tmana.create_starting_parameters_1D(peak_volume, peak_tolerance=6)) == 3

    def test_peak_center_detected(self, peak_volume):
        pc, _, _ = tmana.create_starting_parameters_1D(peak_volume, peak_tolerance=6)
        assert pc == (10, 10, 10)

    def test_peak_height_is_global_max(self, peak_volume):
        _, ph, _ = tmana.create_starting_parameters_1D(peak_volume, peak_tolerance=6)
        assert ph == pytest.approx(5.0)

    def test_profiles_shape(self, peak_volume):
        _, _, profiles = tmana.create_starting_parameters_1D(peak_volume, peak_tolerance=6)
        assert profiles.shape == (peak_volume.shape[0], 3)

    def test_profiles_contain_peak_value(self, peak_volume):
        _, _, profiles = tmana.create_starting_parameters_1D(peak_volume, peak_tolerance=6)
        assert np.any(np.isclose(profiles, 5.0))

    def test_profiles_are_finite(self, peak_volume):
        _, _, profiles = tmana.create_starting_parameters_1D(peak_volume, peak_tolerance=6)
        assert np.all(np.isfinite(profiles))


# ── create_starting_parameters_2D ─────────────────────────────────────────────

class TestCreateStartingParameters2D:
    def test_returns_three_values(self, peak_volume):
        assert len(tmana.create_starting_parameters_2D(peak_volume, peak_tolerance=6)) == 3

    def test_peak_center_auto_detected(self, peak_volume):
        pc, _, _ = tmana.create_starting_parameters_2D(peak_volume, peak_tolerance=6)
        assert pc == (10, 10, 10)

    def test_peak_height_is_global_max_when_no_center_given(self, peak_volume):
        _, ph, _ = tmana.create_starting_parameters_2D(peak_volume, peak_tolerance=6)
        assert ph == pytest.approx(5.0)

    def test_slices_shape(self, peak_volume):
        n = peak_volume.shape[0]
        _, _, slices = tmana.create_starting_parameters_2D(peak_volume, peak_tolerance=6)
        assert slices.shape == (n, n, 3)

    def test_provided_peak_center_respected(self, peak_volume):
        pc, _, _ = tmana.create_starting_parameters_2D(peak_volume, peak_center=(10, 10, 10))
        assert pc == (10, 10, 10)

    def test_provided_peak_center_height_from_masked_map(self, peak_volume):
        _, ph, _ = tmana.create_starting_parameters_2D(peak_volume, peak_center=(10, 10, 10))
        assert ph == pytest.approx(5.0)

    def test_slices_contain_peak(self, peak_volume):
        _, _, slices = tmana.create_starting_parameters_2D(peak_volume, peak_tolerance=6)
        assert np.any(np.isclose(slices, 5.0))


# ── get_central_label ─────────────────────────────────────────────────────────

class TestGetCentralLabel:
    def test_returns_two_values(self, cube_volume):
        assert len(tmana.get_central_label(cube_volume, (10, 10, 10))) == 2

    def test_labeled_mask_shape(self, cube_volume):
        labeled, _ = tmana.get_central_label(cube_volume, (10, 10, 10))
        assert labeled.shape == cube_volume.shape

    def test_cube_sizes(self, cube_volume):
        _, sizes = tmana.get_central_label(cube_volume, (10, 10, 10))
        assert sizes == (4, 4, 4)

    def test_peak_is_inside_labeled_region(self, cube_volume):
        labeled, _ = tmana.get_central_label(cube_volume, (10, 10, 10))
        assert labeled[10, 10, 10] == 1.0

    def test_background_is_zero(self, cube_volume):
        labeled, _ = tmana.get_central_label(cube_volume, (10, 10, 10))
        assert labeled[0, 0, 0] == 0.0

    def test_disconnected_region_excluded(self):
        vol = np.zeros((20, 20, 20))
        vol[2:4, 2:4, 2:4] = 1.0   # remote cube
        vol[8:12, 8:12, 8:12] = 1.0  # central cube
        labeled, _ = tmana.get_central_label(vol, (10, 10, 10))
        assert labeled[3, 3, 3] == 0.0
        assert labeled[10, 10, 10] == 1.0

    def test_asymmetric_region_sizes(self):
        vol = np.zeros((20, 20, 20))
        vol[8:12, 9:11, 10] = 1.0  # 4 x 2 x 1 slab
        _, sizes = tmana.get_central_label(vol, (10, 10, 10))
        assert sizes == (4, 2, 1)

    def test_labeled_mask_binary(self, cube_volume):
        labeled, _ = tmana.get_central_label(cube_volume, (10, 10, 10))
        assert set(np.unique(labeled)).issubset({0.0, 1.0})


# ── filter_dist_maps ──────────────────────────────────────────────────────────

class TestFilterDistMaps:
    def test_returns_two_arrays(self):
        shape = (8, 8, 8)
        result = tmana.filter_dist_maps(np.ones((*shape, 1)), np.ones(shape), 1)
        assert len(result) == 2

    def test_output_shapes_preserved(self):
        shape = (10, 10, 10)
        dist = np.ones((*shape, 2))
        mask = np.ones(shape)
        out_dist, out_mask = tmana.filter_dist_maps(dist.copy(), mask.copy(), 1)
        assert out_dist.shape == (10, 10, 10, 2)
        assert out_mask.shape == (10, 10, 10)

    def test_small_threshold_keeps_region(self):
        shape = (10, 10, 10)
        _, out_mask = tmana.filter_dist_maps(np.ones((*shape, 2)), np.ones(shape), 1)
        assert out_mask.sum() > 0

    def test_large_threshold_removes_all(self):
        shape = (10, 10, 10)
        _, out_mask = tmana.filter_dist_maps(np.ones((*shape, 2)), np.ones(shape), 2000)
        assert out_mask.sum() == 0

    def test_dist_maps_zeroed_when_everything_removed(self):
        shape = (10, 10, 10)
        out_dist, _ = tmana.filter_dist_maps(np.ones((*shape, 2)), np.ones(shape), 2000)
        assert out_dist.sum() == 0.0

    def test_zero_mask_leaves_everything_zero(self):
        shape = (8, 8, 8)
        dist = np.ones((*shape, 1))
        mask = np.zeros(shape)
        out_dist, out_mask = tmana.filter_dist_maps(dist.copy(), mask.copy(), 1)
        assert out_mask.sum() == 0
        assert out_dist.sum() == 0

    @pytest.mark.parametrize("n_maps", [1, 2, 3])
    def test_multiple_dist_maps(self, n_maps):
        shape = (8, 8, 8)
        dist = np.ones((*shape, n_maps))
        mask = np.ones(shape)
        out_dist, _ = tmana.filter_dist_maps(dist.copy(), mask.copy(), 1)
        assert out_dist.shape[-1] == n_maps


# ── evaluate_scores_map ───────────────────────────────────────────────────────

class TestEvaluateScoresMap:
    @pytest.fixture
    def block_volume(self):
        vol = np.zeros((20, 20, 20))
        vol[9:12, 9:12, 9:12] = 1.0
        return vol

    def test_invalid_threshold_type_raises(self, block_volume):
        with pytest.raises(ValueError):
            tmana.evaluate_scores_map(block_volume, threshold_type="invalid")

    def test_returns_five_values_hard_central(self, block_volume):
        result = tmana.evaluate_scores_map(block_volume, label_type="central", threshold_type="hard")
        assert len(result) == 5

    def test_peak_height_positive(self, block_volume):
        _, _, ph, _, _ = tmana.evaluate_scores_map(block_volume, label_type="central", threshold_type="hard")
        assert ph > 0

    def test_labeled_map_nonnegative(self, block_volume):
        labeled_map, _, _, _, _ = tmana.evaluate_scores_map(
            block_volume, label_type="central", threshold_type="hard"
        )
        assert np.all(labeled_map >= 0)

    def test_surface_is_empty_for_central_label(self, block_volume):
        _, _, _, _, surface = tmana.evaluate_scores_map(
            block_volume, label_type="central", threshold_type="hard"
        )
        assert surface == []

    def test_surface_is_empty_for_plane_label(self, block_volume):
        _, _, _, _, surface = tmana.evaluate_scores_map(
            block_volume, label_type="plane", threshold_type="hard"
        )
        assert surface == []

    def test_thresholded_map_shape_matches_input(self, block_volume):
        _, _, _, th_map, _ = tmana.evaluate_scores_map(
            block_volume, label_type="central", threshold_type="hard"
        )
        assert th_map.shape == block_volume.shape


# ── scores_extract_particles ──────────────────────────────────────────────────

class TestScoresExtractParticles:
    """Tests use mocker to avoid file I/O for scores/angles maps."""

    def _make_inputs(self, shape=(20, 20, 20)):
        scores = np.zeros(shape)
        scores[10, 10, 10] = 0.9
        scores[5, 5, 5] = 0.8
        angles_map = np.zeros(shape)
        angles_map[10, 10, 10] = 1
        angles_map[5, 5, 5] = 2
        anglist = np.zeros((3, 3))  # rows 0-2; ang_idx will be 1 and 2
        return scores, angles_map, anglist

    def _patch(self, mocker, scores, amap, anglist):
        mocker.patch("cryocat.core.cryomap.read", side_effect=[scores, amap])
        mocker.patch("cryocat.utils.ioutils.rot_angles_load", return_value=anglist)

    def test_returns_motl_above_threshold(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3, scores_threshold=0.7
        )
        assert motl is not None
        assert len(motl.df) == 2

    def test_returns_none_when_nothing_above_threshold(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3, scores_threshold=1.5
        )
        assert motl is None

    def test_tomo_id_assigned(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=7, particle_diameter=3, scores_threshold=0.7
        )
        assert (motl.df["tomo_id"] == 7).all()

    def test_object_id_defaults_to_1(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3, scores_threshold=0.7
        )
        assert (motl.df["object_id"] == 1).all()

    def test_n_particles_limits_output(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3,
            scores_threshold=0.7, n_particles=1
        )
        assert len(motl.df) == 1

    def test_sigma_threshold_very_high_returns_none(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3, sigma_threshold=1000.0
        )
        assert motl is None

    def test_non_c_symmetry_issues_warning(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        with pytest.warns(UserWarning):
            tmana.scores_extract_particles(
                "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3,
                scores_threshold=0.7, symmetry="d2"
            )

    def test_c1_symmetry_runs_without_phi_change(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3,
            scores_threshold=0.7, symmetry="c1"
        )
        assert motl is not None

    def test_c2_symmetry_runs_without_error(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3,
            scores_threshold=0.7, symmetry="c2"
        )
        assert motl is not None

    def test_scores_column_present(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3, scores_threshold=0.7
        )
        assert "score" in motl.df.columns

    def test_scores_above_threshold(self, mocker):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=3, scores_threshold=0.7
        )
        assert (motl.df["score"] > 0.7).all()

    def test_large_particle_diameter_merges_clusters(self, mocker):
        # Both peaks within diameter=15 of each other → only the highest-score one survives
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles(
            "s.em", "a.em", "al.npy", tomo_id=1, particle_diameter=15, scores_threshold=0.7
        )
        assert len(motl.df) == 1
        assert motl.df["score"].iloc[0] == pytest.approx(0.9)
