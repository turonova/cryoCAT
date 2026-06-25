import numpy as np
import pytest
from unittest.mock import patch
from cryocat.analysis import tmana
from cryocat.core import cryomotl

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

    @pytest.mark.parametrize("threshold_type", ["hard", "triangle", "gauss"])
    def test_returns_five_values(self, block_volume, threshold_type):
        result = tmana.evaluate_scores_map(block_volume, label_type="central", threshold_type=threshold_type)
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

    @pytest.mark.parametrize("threshold_type", ["hard", "triangle", "gauss"])
    def test_thresholded_map_shape_matches_input(self, block_volume, threshold_type):
        _, _, _, th_map, _ = tmana.evaluate_scores_map(
            block_volume, label_type="central", threshold_type=threshold_type
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
        _file_returns = iter([scores, amap])

        def _fake_read(x, *_args, **_kwargs):
            # Mirrors cryomap.read pass-through: ndarrays are returned as-is.
            if isinstance(x, np.ndarray):
                return x
            return next(_file_returns)

        mocker.patch("cryocat.core.cryomap.read", side_effect=_fake_read)
        mocker.patch("cryocat.utils.ioutils.euler_angles_load", return_value=anglist)

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


# ── compute_gaussian_threshold ────────────────────────────────────────────────

class TestComputeGaussianThreshold:
    @pytest.fixture
    def gaussian_volume(self):
        """30x30x30 volume with a 4x4x4 block of 1s at the centre."""
        vol = np.zeros((30, 30, 30))
        vol[13:17, 13:17, 13:17] = 1.0
        return vol

    def test_returns_finite_float(self, gaussian_volume):
        result = tmana.compute_gaussian_threshold(gaussian_volume)
        assert isinstance(result, float)
        assert np.isfinite(result)

    def test_threshold_positive(self, gaussian_volume):
        result = tmana.compute_gaussian_threshold(gaussian_volume)
        assert result > 0

    def test_threshold_plausible_magnitude(self, gaussian_volume):
        result = tmana.compute_gaussian_threshold(gaussian_volume)
        assert result < 10 * gaussian_volume.max()


# ── get_ellipsoid_label ───────────────────────────────────────────────────────

class TestGetEllipsoidLabel:
    @pytest.fixture
    def blob_volume(self):
        """30x30x30 volume with a 10x10x10 cube at [10:20, 10:20, 10:20]."""
        vol = np.zeros((30, 30, 30))
        vol[10:20, 10:20, 10:20] = 1.0
        return vol

    def test_returns_four_values(self, blob_volume):
        result = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert len(result) == 4

    def test_fitted_label_shape(self, blob_volume):
        fitted_label, _, _, _ = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert fitted_label.shape == blob_volume.shape

    def test_radii_shape(self, blob_volume):
        _, radii, _, _ = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert radii.shape == (3,)

    def test_radii_positive(self, blob_volume):
        _, radii, _, _ = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert np.all(radii > 0)

    def test_surface_fit_shape(self, blob_volume):
        _, _, surface_fit, _ = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert surface_fit.shape == blob_volume.shape

    def test_th_map_shape(self, blob_volume):
        _, _, _, th_map = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert th_map.shape == blob_volume.shape

    def test_th_map_binary(self, blob_volume):
        _, _, _, th_map = tmana.get_ellipsoid_label(blob_volume, (15, 15, 15))
        assert set(np.unique(th_map)).issubset({0.0, 1.0})

    def test_custom_threshold_background(self):
        vol = np.zeros((30, 30, 30))
        vol[10:20, 10:20, 10:20] = 2.0
        fitted_label, _, _, _ = tmana.get_ellipsoid_label(vol, (15, 15, 15), map_threshold=0.0)
        assert fitted_label.shape == vol.shape


# ── get_central_plane_labels ──────────────────────────────────────────────────

class TestGetCentralPlaneLabels:
    @pytest.fixture
    def cubic_blob(self):
        """20x20x20 volume with a 4x4x4 cube at [8:12, 8:12, 8:12]."""
        vol = np.zeros((20, 20, 20))
        vol[8:12, 8:12, 8:12] = 1.0
        return vol

    def test_returns_two_values(self, cubic_blob):
        result = tmana.get_central_plane_labels(cubic_blob, (10, 10, 10))
        assert len(result) == 2

    def test_mask_shape_matches_input(self, cubic_blob):
        mask, _ = tmana.get_central_plane_labels(cubic_blob, (10, 10, 10))
        assert mask.shape == cubic_blob.shape

    def test_mask_is_binary(self, cubic_blob):
        mask, _ = tmana.get_central_plane_labels(cubic_blob, (10, 10, 10))
        assert set(np.unique(mask)).issubset({0.0, 1.0})

    def test_half_lengths_are_three_values(self, cubic_blob):
        _, half_lengths = tmana.get_central_plane_labels(cubic_blob, (10, 10, 10))
        assert len(half_lengths) == 3

    def test_half_lengths_positive(self, cubic_blob):
        _, half_lengths = tmana.get_central_plane_labels(cubic_blob, (10, 10, 10))
        assert all(h > 0 for h in half_lengths)

    def test_mask_nonzero_near_peak(self, cubic_blob):
        mask, _ = tmana.get_central_plane_labels(cubic_blob, (10, 10, 10))
        assert mask.sum() > 0


# -- extract_peak_orientations ------------------------------------------------------------

class TestExtractPeakOrientations:

    @pytest.fixture
    def peak_coords(self):
        """Numpy ndarray of shape (2, 3) with two sets of 3D coordinates."""
        peak_coords = np.asarray([[5, 5, 5],[10, 10, 10]])
        return peak_coords

    def _make_inputs(self, peak_coords, shape=(20, 20, 20)):
        #scores = np.zeros(shape)
        #scores[10, 10, 10] = 0.9
        #scores[5, 5, 5] = 0.8
        angles_map = np.zeros(shape)
        angles_map[peak_coords[0]] = 1
        angles_map[peak_coords[1]] = 1
        anglist = np.zeros((3, 3))  # rows 0-2
        anglist[1] = [10, 20, 30]
        return angles_map, anglist
    
    def _patch(self, mocker, amap, anglist):
        mocker.patch("cryocat.core.cryomap.read", return_value=amap)
        mocker.patch("cryocat.utils.ioutils.euler_angles_load", return_value=anglist)
    
    def test_returns_orientations_for_peaks(self, mocker, peak_coords):
        angles_map, anglist = self._make_inputs(peak_coords)
        self._patch(mocker, angles_map, anglist)
        orientations = tmana.extract_peak_orientations(peak_coords, "a.em", "al.npy")
        assert orientations[0].shape == (2,)
        assert np.allclose(orientations[0], 10)
        assert orientations[1].shape == (2,)
        assert np.allclose(orientations[1], 20)
        assert orientations[2].shape == (2,)
        assert np.allclose(orientations[2], 30)

    def test_warning_for_non_c_symmetry(self, mocker, peak_coords):
        angles_map, anglist = self._make_inputs(peak_coords)
        self._patch(mocker, angles_map, anglist)
        with pytest.warns(UserWarning):
            tmana.extract_peak_orientations(peak_coords, "a.em", "al.npy", symmetry="d2")


# -- scores_extract_particles_around_positions ------------------------------------------------------------

class TestScoresExtractParticlesAroundPositions:

    @pytest.fixture
    def input_motl_data(self):
        """Motl.df with coordinates of particles to extract around."""
        input_motl = cryomotl.Motl()
        input_motl.fill(
            {
            "x": [6.5, 12],
            "y": [6.5, 12],
            "z": [6.5, 12],
            "class": 1,
            "subtomo_id":[1,2]
            }
        )
        return input_motl
    
    def _make_inputs(self, shape=(20, 20, 20)):
        scores = np.zeros(shape)
        scores[10, 10, 10] = 0.9
        scores[5, 5, 5] = 0.8
        angles_map = np.zeros(shape)
        angles_map[5, 5, 5] = 1
        angles_map[10, 10, 10] = 1
        anglist = np.zeros((3, 3))  # rows 0-2
        anglist[1] = [10, 20, 30]
        return scores, angles_map, anglist

    def _patch(self, mocker, scores, amap, anglist):
        mocker.patch("cryocat.core.cryomap.read", side_effect=[scores, amap])
        mocker.patch("cryocat.utils.ioutils.euler_angles_load", return_value=anglist) 

    def test_extracts_particles_around_positions(self, mocker, input_motl_data):
        scores, amap, anglist = self._make_inputs()
        self._patch(mocker, scores, amap, anglist)
        motl = tmana.scores_extract_particles_around_positions(
            "s.em", "a.em", "al.npy", input_motl_data, radius=3, tomo_id=1
        )
        assert motl.df.shape[0] == 2
        assert (np.all(motl.df["tomo_id"] == 1))
        assert np.array_equal(motl.df["score"], [0.8, 0.9])
        assert np.array_equal(motl.df["x"], [6, 11])
        assert np.array_equal(motl.df["y"], [6, 11])
        assert np.array_equal(motl.df["z"], [6, 11])
        assert (np.all(motl.df["phi"] == 10))
        assert (np.all(motl.df["psi"] == 30))
        assert (np.all(motl.df["theta"] == 20))


# ── create_angular_distance_maps ──────────────────────────────────────────────

class TestCreateAngularDistanceMaps:
    """Verify 0-based index convention and -1 sentinel handling."""

    def _run(self, angles_map_arr, angles):
        with patch("cryocat.analysis.tmana.cryomap.read", return_value=angles_map_arr), \
             patch("cryocat.analysis.tmana.ioutils.euler_angles_load", return_value=angles), \
             patch("cryocat.analysis.tmana.cryomap.write"):
            return tmana.create_angular_distance_maps(
                angles_map_arr, angles, write_out_maps=False
            )

    def test_identity_angle_gives_zero_distance(self):
        angles = np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0]])
        amap = np.full((3, 3, 3), -1, dtype=int)
        amap[1, 1, 1] = 0
        dist_all, dist_normals, dist_inplane = self._run(amap, angles)
        assert dist_all[1, 1, 1] == pytest.approx(0.0, abs=1e-6)
        assert dist_normals[1, 1, 1] == pytest.approx(0.0, abs=1e-6)
        assert dist_inplane[1, 1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_nonzero_angle_gives_nonzero_distance(self):
        angles = np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0]])
        amap = np.full((3, 3, 3), -1, dtype=int)
        amap[1, 1, 1] = 1
        dist_all, _, _ = self._run(amap, angles)
        assert dist_all[1, 1, 1] > 1.0

    def test_sentinel_voxels_get_zero_distance(self):
        angles = np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0]])
        amap = np.full((3, 3, 3), -1, dtype=int)
        dist_all, dist_normals, dist_inplane = self._run(amap, angles)
        assert np.all(dist_all == 0.0)
        assert np.all(dist_normals == 0.0)
        assert np.all(dist_inplane == 0.0)

    def test_output_shape_matches_input(self):
        angles = np.array([[0.0, 0.0, 0.0], [45.0, 0.0, 0.0]])
        amap = np.zeros((5, 6, 7), dtype=int)
        dist_all, dist_normals, dist_inplane = self._run(amap, angles)
        assert dist_all.shape == (5, 6, 7)
        assert dist_normals.shape == (5, 6, 7)
        assert dist_inplane.shape == (5, 6, 7)

    
