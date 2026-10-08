import numpy as np
import pytest
import warnings
from unittest.mock import patch
from cryocat.analysis import tmana
from cryocat.utils import geom
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
        # approx, not ==: C1 angles are now round-tripped through scipy to canonicalize the range
        # (2026-10-08 fix), which can introduce float noise at the ~1e-15 level on already-in-range input
        np.testing.assert_allclose(motl.df["phi"], 10, atol=1e-9)
        np.testing.assert_allclose(motl.df["psi"], 30, atol=1e-9)
        np.testing.assert_allclose(motl.df["theta"], 20, atol=1e-9)


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

    


# ── create_angular_distance_maps with symmetry (added 2026-10-02) ─────────────

class TestCreateAngularDistanceMapsSymmetry:
    """The symmetry reaches compare_rotations with its group letter, and the old keyword is deprecated."""

    # index 0 = reference; index 1 = one C4 step (90 deg spin) away, same look for C4; index 2 = 10 deg more tilt
    ANGLES = np.array([[20.0, 30.0, 10.0], [110.0, 30.0, 10.0], [20.0, 40.0, 10.0]])

    def _amap(self):
        amap = np.full((3, 3, 3), -1, dtype=int)
        amap[0, 0, 0], amap[1, 1, 1], amap[2, 2, 2] = 0, 1, 2
        return amap

    def _run(self, **kwargs):
        amap = self._amap()
        with patch("cryocat.analysis.tmana.cryomap.read", return_value=amap), \
             patch("cryocat.analysis.tmana.ioutils.euler_angles_load", return_value=self.ANGLES), \
             patch("cryocat.analysis.tmana.cryomap.write"):
            return tmana.create_angular_distance_maps(amap, self.ANGLES, write_out_maps=False, **kwargs)

    def test_c4_matches_compare_rotations(self):
        # Map values equal compare_rotations(reference, angles, "C4") at the indexed voxels
        maps = self._run(symmetry="C4")
        ref = np.tile(self.ANGLES[0], (3, 1))
        expected = geom.compare_rotations(ref, self.ANGLES, symmetry="C4")
        for got, exp in zip(maps, expected):
            got_vals = np.array([got[0, 0, 0], got[1, 1, 1], got[2, 2, 2]])
            np.testing.assert_allclose(np.nan_to_num(got_vals), np.nan_to_num(exp), atol=1e-6)
        # one C4 step away looks identical -> 0 (NaN read as 0, see angular_distance Notes)
        dist_all, _, dist_inplane = maps
        assert np.nan_to_num(dist_all[1, 1, 1]) == pytest.approx(0.0, abs=1e-4)
        assert dist_inplane[1, 1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_non_cyclic_raises(self):
        # Cone/in-plane maps are not defined for T (previously silently computed as C12)
        with pytest.raises(NotImplementedError):
            self._run(symmetry="T")

    def test_deprecated_keyword(self):
        # cyclic_symmetry still works, with a DeprecationWarning, and gives the same maps as symmetry=
        with pytest.warns(DeprecationWarning, match="cyclic_symmetry"):
            old = self._run(cyclic_symmetry=4)
        new = self._run(symmetry=4)
        for a, b in zip(old, new):
            np.testing.assert_array_equal(a, b)


# ── extract_peak_orientations with any symmetry (added 2026-10-02) ───────────

from scipy.spatial.transform import Rotation as srot
from cryocat.utils.symmetry import closest_symmetric_copy, get_symmetry_rotations


def _peak_inputs(n_peaks=200, seed=0, angles=None):
    """Peaks on distinct voxels of a 10^3 map, each pointing to its own row of the angle list.

    Default angle list: random rotations in cryoCAT's canonical Euler form (zxz, degrees).
    """
    if angles is None:
        angles = srot.random(n_peaks, random_state=seed).as_euler("zxz", degrees=True)
    n_peaks = len(angles)
    flat = np.random.default_rng(seed).choice(1000, size=n_peaks, replace=False)
    coords = np.column_stack(np.unravel_index(flat, (10, 10, 10)))
    amap = np.full((10, 10, 10), -1.0)
    amap[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(n_peaks)
    return coords, amap, angles


def _stored_and_returned(coords, amap, angles, symmetry, seed=7):
    """Run extract_peak_orientations with a fixed seed; return (stored, returned) as Rotations."""
    np.random.seed(seed)
    phi, theta, psi = tmana.extract_peak_orientations(coords, amap, angles, symmetry=symmetry)
    stored = srot.from_euler("zxz", angles, degrees=True)
    returned = srot.from_euler("zxz", np.column_stack([phi, theta, psi]), degrees=True)
    return stored, returned, np.column_stack([phi, theta, psi])


class TestExtractPeakOrientationsSymmetry:
    """Each particle gets a random symmetric copy R @ g; canonical Euler ranges; frame warning for D/T/O/I."""

    @pytest.mark.parametrize("symm", ["C4", "D2", "T", "O", "I"])
    def test_outputs_are_symmetric_copies(self, symm):
        # Every returned orientation looks identical to the stored one (distance 0 up to symmetry)
        coords, amap, angles = _peak_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)  # frame reminder for D/T/O/I
            stored, returned, _ = _stored_and_returned(coords, amap, angles, symm)
        dist, _ = closest_symmetric_copy(stored, returned, symm)
        np.testing.assert_allclose(dist, 0.0, atol=1e-4)

    @pytest.mark.parametrize("symm", ["C4", "T", "I"])
    def test_copies_actually_vary(self, symm):
        # The copies are spread: with 200 particles more than one copy (not only the identity) is used
        coords, amap, angles = _peak_inputs()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            stored, returned, _ = _stored_and_returned(coords, amap, angles, symm)
        _, copy_idx = closest_symmetric_copy(returned, stored, symm)
        assert len(np.unique(copy_idx)) > 1

    @pytest.mark.parametrize("symm", ["C3", "C6", "D3", "O"])
    def test_canonical_euler_ranges(self, symm):
        # phi, psi in [-180, 180] and theta in [0, 180] (cryoCAT's conventions), even for input outside them
        coords, amap, angles = _peak_inputs()
        angles = angles.copy()
        angles[:, 2] = np.mod(angles[:, 2], 360.0)  # psi stored in [0, 360)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            _, _, out = _stored_and_returned(coords, amap, angles, symm)
        assert np.all((out[:, [0, 2]] >= -180.0) & (out[:, [0, 2]] <= 180.0))
        assert np.all((out[:, 1] >= 0.0) & (out[:, 1] <= 180.0))

    @pytest.mark.parametrize("n", [2, 3, 4, 6])
    def test_cn_same_copies_as_former_phi_shift(self, n):
        # For a given seed the same copy is picked as by the former code (phi + random k*360/n):
        # same rotations, and for in-range input the same Euler numbers once phi is wrapped
        coords, amap, angles = _peak_inputs()
        _, returned, out = _stored_and_returned(coords, amap, angles, f"C{n}", seed=11)
        np.random.seed(11)
        old_phi = angles[:, 0] + np.random.choice(np.linspace(0, 360, n + 1)[:-1], size=len(angles))
        old = srot.from_euler("zxz", np.column_stack([old_phi, angles[:, 1], angles[:, 2]]), degrees=True)
        np.testing.assert_allclose((old.inv() * returned).magnitude(), 0.0, atol=1e-7)
        wrapped = np.mod(old_phi + 180.0, 360.0) - 180.0
        diff = np.abs(out[:, 0] - wrapped)
        np.testing.assert_allclose(np.minimum(diff, 360.0 - diff), 0.0, atol=1e-9)  # 180 and -180 are the same
        np.testing.assert_allclose(out[:, 1:], angles[:, 1:], atol=1e-9)

    def test_cn_no_tilt_moves_spin_into_phi(self):
        # theta = 0: same rotation, written canonically with the whole spin in phi and psi = 0
        angles = np.array([[10.0, 0.0, 30.0], [-50.0, 0.0, 20.0]])
        coords, amap, angles = _peak_inputs(angles=angles)
        stored, returned, out = _stored_and_returned(coords, amap, angles, "C4")
        dist, _ = closest_symmetric_copy(stored, returned, "C4")
        np.testing.assert_allclose(dist, 0.0, atol=1e-4)
        np.testing.assert_allclose(out[:, 2], 0.0, atol=1e-9)

    def test_c1_returns_same_rotation_in_canonical_range(self):
        # No symmetry: no copy is assigned, but the angles are still canonicalized (fixed 2026-10-08),
        # since angles_list entries are not guaranteed to already be in cryoCAT's ranges
        angles = np.array([[10.0, 20.0, 300.0], [400.0, 0.0, 30.0]])  # psi=300, phi=400 are out of range
        coords, amap, angles = _peak_inputs(angles=angles)
        phi, theta, psi = tmana.extract_peak_orientations(coords, amap, angles, symmetry="C1")
        out = np.column_stack([phi, theta, psi])
        assert np.all((out[:, [0, 2]] >= -180.0) & (out[:, [0, 2]] <= 180.0))
        assert np.all((out[:, 1] >= 0.0) & (out[:, 1] <= 180.0))
        # same rotation as the stored (out-of-range) angles, only rewritten canonically
        stored = srot.from_euler("zxz", angles, degrees=True)
        returned = srot.from_euler("zxz", out, degrees=True)
        np.testing.assert_allclose((stored.inv() * returned).magnitude(), 0.0, atol=1e-9)

    def test_c1_in_range_angles_unchanged(self):
        # Angles already in cryoCAT's canonical ranges are returned unchanged (no spurious rewriting)
        angles = np.array([[10.0, 20.0, -30.0], [-170.0, 150.0, 175.0]])
        coords, amap, angles = _peak_inputs(angles=angles)
        phi, theta, psi = tmana.extract_peak_orientations(coords, amap, angles, symmetry="C1")
        np.testing.assert_allclose(np.column_stack([phi, theta, psi]), angles, atol=1e-9)

    @pytest.mark.parametrize("symm", ["D2", "T", "O", "I"])
    def test_non_cyclic_warns_about_canonical_frame(self, symm):
        # D/T/O/I: reminder that the template must be in cryoCAT's canonical frame
        coords, amap, angles = _peak_inputs(n_peaks=5)
        with pytest.warns(UserWarning, match="canonical frame"):
            tmana.extract_peak_orientations(coords, amap, angles, symmetry=symm)

    @pytest.mark.parametrize("symm", ["C1", "C5"])
    def test_cyclic_does_not_warn(self, symm):
        # C_n: the n-fold axis is z by convention, no reminder needed
        coords, amap, angles = _peak_inputs(n_peaks=5)
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            tmana.extract_peak_orientations(coords, amap, angles, symmetry=symm)

    def test_empty_peak_list(self):
        # No peaks: empty arrays, no error
        _, amap, angles = _peak_inputs(n_peaks=5)
        with pytest.warns(UserWarning):
            phi, theta, psi = tmana.extract_peak_orientations(np.empty((0, 3), dtype=int), amap, angles, symmetry="T")
        assert phi.shape == theta.shape == psi.shape == (0,)

    def test_scores_extract_particles_octahedral(self):
        # The caller passes the symmetry through: particle-list orientations are octahedral copies
        scores = np.zeros((20, 20, 20))
        scores[10, 10, 10], scores[5, 5, 5] = 0.9, 0.8
        amap = np.zeros((20, 20, 20))
        amap[10, 10, 10], amap[5, 5, 5] = 1, 2
        anglist = np.array([[0.0, 0.0, 0.0], [10.0, 20.0, 30.0], [-40.0, 70.0, 120.0]])
        np.random.seed(3)
        with pytest.warns(UserWarning, match="canonical frame"):
            motl = tmana.scores_extract_particles(
                scores, amap, anglist, tomo_id=1, particle_diameter=3, scores_threshold=0.7, symmetry="O"
            )
        got = srot.from_euler("zxz", motl.df[["phi", "theta", "psi"]].values, degrees=True)
        # peak at (10,10,10) -> list row 1, peak at (5,5,5) -> row 2; match each particle by its score
        rows = np.where(motl.df["score"].values > 0.85, 1, 2)
        dist, _ = closest_symmetric_copy(srot.from_euler("zxz", anglist[rows], degrees=True), got, "O")
        np.testing.assert_allclose(dist, 0.0, atol=1e-4)
