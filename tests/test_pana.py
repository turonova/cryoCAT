import numpy as np
import pandas as pd
import os
from glob import glob
from cryocat.analysis import pana
from cryocat.utils import imageutils
import pytest
from pathlib import Path
from scipy.spatial.transform import Rotation as srot

# IMPORTANT: pytest-mock needs to be installed within environment to run these tests

#test_csv_folder = os.path.join("tests", "test_data", "pana_data", "test_template_lists", "*.csv")
test_data_dir = Path(__file__).parent / "test_data" / "pana_data" / "test_template_lists"
test_template_lists = list(test_data_dir.glob("*.csv"))


@pytest.mark.parametrize("csv_file", test_template_lists)
def test_create_subtomograms(mocker, csv_file):

    parent_path = test_data_dir
    subvolume_sh = np.ones((64, 64, 64))
    angles = [1, 2, 3]

    mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo", return_value=(subvolume_sh, angles))
    mocker.patch("pandas.DataFrame.to_csv")

    df = pana.create_subtomograms_for_tm(str(csv_file), str(parent_path))

    assert df["Tomo created"].to_list() == [True] * len(df)
    assert all([isinstance(i, str) for i in df["Tomo map"].tolist()]) == True
    assert (df[["Phi", "Theta", "Psi"]].to_numpy() == np.full((len(df), len(angles)), angles)).all()


# ── Path / name building ───────────────────────────────────────────────────────

class TestCreateStructurePath:
    def test_basic(self):
        assert pana.create_structure_path("/data/", "ribosome") == "/data/ribosome/"

    def test_no_trailing_slash_in_base(self):
        assert pana.create_structure_path("/data", "ribosome") == "/dataribosome/"

    def test_empty_structure_name(self):
        assert pana.create_structure_path("/data/", "") == "/data//"

    def test_returns_string(self):
        assert isinstance(pana.create_structure_path("/a/", "b"), str)


class TestCreateEmPath:
    def test_basic(self):
        assert pana.create_em_path("/data/", "ribosome", "template") == "/data/ribosome/template.em"

    def test_em_extension_appended(self):
        assert pana.create_em_path("/base/", "struct", "myfile").endswith(".em")

    def test_structure_folder_in_path(self):
        assert "mystructure" in pana.create_em_path("/base/", "mystructure", "myfile")


class TestCreateSubtomoName:
    def test_full_format(self):
        assert (
            pana.create_subtomo_name("ribosome", "motl1", "001", 80)
            == "subtomo_ribosome_mmotl1_t001_s80.em"
        )

    def test_ends_with_em(self):
        assert pana.create_subtomo_name("s", "m", "1", 64).endswith(".em")

    @pytest.mark.parametrize("boxsize", [32, 64, 80, 128])
    def test_boxsize_in_name(self, boxsize):
        assert f"_s{boxsize}.em" in pana.create_subtomo_name("s", "m", "1", boxsize)

    def test_tomo_id_in_name(self):
        assert "_t007_" in pana.create_subtomo_name("s", "m", "007", 64)


class TestCreateTomoName:
    def test_basic(self):
        assert pana.create_tomo_name("/data/", "tomo_001") == "/data/tomo_001.mrc"

    def test_mrc_extension(self):
        assert pana.create_tomo_name("/p/", "name").endswith(".mrc")


class TestCreateWedgeNames:
    def test_default_filter_is_half_boxsize(self):
        tomo_w, tmpl_w = pana.create_wedge_names("/w/", 1, 80, 4)
        assert "_f40.em" in tomo_w
        assert "_f40.em" in tmpl_w

    def test_custom_filter(self):
        tomo_w, tmpl_w = pana.create_wedge_names("/w/", 5, 64, 2, filter=20)
        assert "_f20.em" in tomo_w
        assert "_f20.em" in tmpl_w

    def test_tomo_number_in_name(self):
        tomo_w, _ = pana.create_wedge_names("/w/", 7, 80, 3)
        assert "_t7_" in tomo_w

    def test_binning_in_name(self):
        tomo_w, _ = pana.create_wedge_names("/w/", 1, 80, 3)
        assert "_b3_" in tomo_w

    def test_tile_and_tmpl_prefixes(self):
        tomo_w, tmpl_w = pana.create_wedge_names("/w/", 1, 64, 2)
        assert "tile_filt_" in tomo_w
        assert "tmpl_filt_" in tmpl_w

    def test_returns_two_strings(self):
        result = pana.create_wedge_names("/w/", 1, 64, 2)
        assert len(result) == 2 and all(isinstance(s, str) for s in result)


class TestCreateOutputNames:
    @pytest.mark.parametrize("idx,expected", [(0, "id_0"), (5, "id_5"), (42, "id_42")])
    def test_base_name(self, idx, expected):
        assert pana.create_output_base_name(idx) == expected

    @pytest.mark.parametrize("idx,expected", [(0, "id_0_results"), (3, "id_3_results")])
    def test_folder_name(self, idx, expected):
        assert pana.create_output_folder_name(idx) == expected

    def test_folder_path_int_spec(self):
        assert pana.create_output_folder_path("/base/", "ribosome", 2) == "/base/ribosome/id_2_results/"

    def test_folder_path_string_spec(self):
        assert pana.create_output_folder_path("/base/", "ribosome", "my_run") == "/base/ribosome/my_run/"

    def test_folder_path_ends_with_slash(self):
        assert pana.create_output_folder_path("/b/", "s", 0).endswith("/")
        assert pana.create_output_folder_path("/b/", "s", "x").endswith("/")


# ── CTF (private helper) ───────────────────────────────────────────────────────

class TestCtf:
    _DEF = np.array([[2.0], [2.5], [3.0]])
    _PSH = np.zeros((3, 1))
    _F = np.linspace(0.0, 0.5, 20)

    def test_output_shape(self):
        result = imageutils.compute_ctf_2d(self._DEF, self._PSH, 0.07, 2.7, 300, self._F)
        assert result.shape == (3, 20)

    def test_values_in_minus1_to_1(self):
        result = imageutils.compute_ctf_2d(self._DEF, self._PSH, 0.07, 2.7, 300, self._F)
        assert np.all(result >= -1.0) and np.all(result <= 1.0)

    def test_finite_at_zero_frequency(self):
        f = np.linspace(0.0, 0.5, 10)
        result = imageutils.compute_ctf_2d(self._DEF, self._PSH, 0.07, 2.7, 300, f)
        assert np.all(np.isfinite(result[:, 0]))

    def test_identical_defocus_same_ctf(self):
        defocus = np.array([[2.0], [2.0]])
        result = imageutils.compute_ctf_2d(defocus, np.zeros((2, 1)), 0.07, 2.7, 300, self._F)
        np.testing.assert_array_almost_equal(result[0], result[1])

    def test_different_defocus_different_ctf(self):
        defocus = np.array([[1.0], [4.0]])
        result = imageutils.compute_ctf_2d(defocus, np.zeros((2, 1)), 0.07, 2.7, 300, self._F)
        assert not np.allclose(result[0], result[1])


# ── rotate_image ───────────────────────────────────────────────────────────────

class TestRotateImage:
    def test_output_shape_preserved_2d(self):
        img = np.random.rand(32, 32)
        assert imageutils.rotate_2d(img, 45).shape == img.shape

    def test_output_shape_preserved_3d(self):
        vol = np.random.rand(8, 8, 8)
        assert imageutils.rotate_2d(vol, 30).shape == vol.shape

    def test_zero_rotation_is_identity(self):
        img = np.random.rand(16, 16)
        np.testing.assert_array_almost_equal(imageutils.rotate_2d(img, 0), img)

    def test_360_rotation_is_identity(self):
        img = np.random.rand(16, 16)
        np.testing.assert_array_almost_equal(imageutils.rotate_2d(img, 360), img, decimal=5)

    def test_fill_value_at_corners(self):
        img = np.ones((20, 20))
        rotated = imageutils.rotate_2d(img, 45, fill_value=0.0)
        assert rotated[0, 0] == pytest.approx(0.0)

    def test_180_rotation_roundtrip(self):
        img = np.random.rand(16, 16)
        np.testing.assert_array_almost_equal(
            imageutils.rotate_2d(imageutils.rotate_2d(img, 180), 180), img, decimal=5
        )


# ── Mask stats ─────────────────────────────────────────────────────────────────

class TestMaskVoxelCountAndBbox:
    def test_voxel_count_all_ones(self):
        mask = np.ones((10, 10, 10))
        n_voxels, _ = imageutils.mask_voxel_count_and_bbox(mask)
        assert n_voxels == 1000

    def test_voxel_count_partial(self):
        mask = np.zeros((10, 10, 10))
        mask[2:5, 2:5, 2:5] = 1.0
        n_voxels, _ = imageutils.mask_voxel_count_and_bbox(mask)
        assert n_voxels == 27

    def test_zero_mask(self):
        mask = np.zeros((8, 8, 8))
        n_voxels, bbox = imageutils.mask_voxel_count_and_bbox(mask)
        assert n_voxels == 0
        np.testing.assert_array_equal(bbox, [0, 0, 0])

    def test_bbox_partial_mask(self):
        mask = np.zeros((10, 10, 10))
        mask[2:5, 2:5, 2:5] = 1.0
        _, bbox = imageutils.mask_voxel_count_and_bbox(mask)
        np.testing.assert_array_equal(bbox, [3, 3, 3])

    def test_returns_two_values(self):
        assert len(imageutils.mask_voxel_count_and_bbox(np.ones((8, 8, 8)))) == 2

    def test_soft_mask_threshold(self):
        mask = np.zeros((10, 10, 10))
        mask[2:6, 2:6, 2:6] = 0.8   # 4x4x4 = 64 voxels above 0.5
        mask[6:8, 6:8, 6:8] = 0.3   # below threshold, ignored
        n_voxels, _ = imageutils.mask_voxel_count_and_bbox(mask, threshold=0.5)
        assert n_voxels == 64

    def test_soft_all_below_threshold(self):
        mask = np.full((8, 8, 8), 0.3)
        n_voxels, _ = imageutils.mask_voxel_count_and_bbox(mask, threshold=0.5)
        assert n_voxels == 0

    def test_soft_all_above_threshold(self):
        mask = np.full((8, 8, 8), 0.9)
        n_voxels, _ = imageutils.mask_voxel_count_and_bbox(mask, threshold=0.5)
        assert n_voxels == 512

    def test_soft_exactly_threshold_excluded(self):
        mask = np.full((4, 4, 4), 0.5)
        n_voxels, _ = imageutils.mask_voxel_count_and_bbox(mask, threshold=0.5)
        assert n_voxels == 0


# ── get_indices ────────────────────────────────────────────────────────────────

class TestGetIndices:
    def test_filter_single_structure(self):
        idx = pana.get_indices(str(test_data_dir / "test_template_list_1.csv"), {"Structure": "ribosome"})
        assert list(idx) == [0, 1]

    def test_filter_other_structure(self):
        idx = pana.get_indices(str(test_data_dir / "test_template_list_1.csv"), {"Structure": "tric_open"})
        assert list(idx) == [2, 3]

    def test_no_match_returns_empty(self):
        idx = pana.get_indices(str(test_data_dir / "test_template_list_1.csv"), {"Structure": "nonexistent"})
        assert len(idx) == 0

    def test_multiple_conditions(self):
        idx = pana.get_indices(
            str(test_data_dir / "test_template_list_1.csv"),
            {"Structure": "ribosome", "Tomo created": False},
        )
        assert list(idx) == [0, 1]

    def test_sort_by_ascending(self):
        # list_1: ribosome boxsize=80 (rows 0,1), tric_open boxsize=64 (rows 2,3)
        # sort all by Boxsize → tric_open (64) first
        csv = str(test_data_dir / "test_template_list_1.csv")
        idx = pana.get_indices(csv, {"Done": False}, sort_by="Boxsize")
        df = pd.read_csv(csv, index_col=0)
        boxes = df.loc[idx, "Boxsize"].tolist()
        assert boxes == sorted(boxes)

    def test_unordered_csv_returns_results(self):
        idx = pana.get_indices(str(test_data_dir / "test_template_list_2.csv"), {"Structure": "ribosome"})
        assert len(idx) > 0

    def test_non_zero_start_index(self):
        idx = pana.get_indices(str(test_data_dir / "test_template_list_5.csv"), {"Structure": "ribosome"})
        assert all(i >= 5 for i in idx)

    def test_tomo_created_true_filter(self):
        # list_3: last entry has Tomo created = True
        idx = pana.get_indices(str(test_data_dir / "test_template_list_3.csv"), {"Tomo created": True})
        assert len(idx) == 1

    def test_single_structure_csv(self):
        idx = pana.get_indices(str(test_data_dir / "test_template_list_0.csv"), {"Structure": "ribosome"})
        assert len(idx) == 2


# ── Wedge mask generation ──────────────────────────────────────────────────────

@pytest.fixture
def simple_wedgelist():
    return pd.DataFrame({"tilt_angle": np.arange(-30, 31, 10, dtype=float)})


@pytest.fixture
def cubic_filter():
    return np.ones((8, 8, 8))


class TestGenerateWedgeMaskSlicesTemplate:
    def test_returns_three_values(self, simple_wedgelist, cubic_filter):
        assert len(pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)) == 3

    def test_active_slices_count_matches_tilts(self, simple_wedgelist, cubic_filter):
        active_slices, _, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        assert len(active_slices) == len(simple_wedgelist)

    def test_wedge_slices_binary(self, simple_wedgelist, cubic_filter):
        _, _, wedge_slices = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        assert set(np.unique(wedge_slices)).issubset({0.0, 1.0})

    def test_output_shapes_match_filter(self, simple_wedgelist, cubic_filter):
        _, weights, wedge_slices = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        assert weights.shape == cubic_filter.shape
        assert wedge_slices.shape == cubic_filter.shape

    def test_weights_nonnegative(self, simple_wedgelist, cubic_filter):
        _, weights, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        assert np.all(weights >= 0)

    def test_non_cubic_filter_raises(self, simple_wedgelist):
        with pytest.raises(AssertionError):
            pana.generate_wedgemask_slices_template(simple_wedgelist, np.ones((8, 8, 4)))

    def test_2d_filter_raises(self, simple_wedgelist):
        with pytest.raises(AssertionError):
            pana.generate_wedgemask_slices_template(simple_wedgelist, np.ones((8, 8)))

    def test_more_tilts_more_coverage(self, cubic_filter):
        wl_few = pd.DataFrame({"tilt_angle": [-30.0, 0.0, 30.0]})
        wl_many = pd.DataFrame({"tilt_angle": np.arange(-60, 61, 3, dtype=float)})
        _, _, ws_few = pana.generate_wedgemask_slices_template(wl_few, cubic_filter)
        _, _, ws_many = pana.generate_wedgemask_slices_template(wl_many, cubic_filter)
        assert ws_many.sum() >= ws_few.sum()

    def test_zero_filter_gives_empty_mask(self, simple_wedgelist):
        _, _, wedge_slices = pana.generate_wedgemask_slices_template(
            simple_wedgelist, np.zeros((8, 8, 8))
        )
        assert wedge_slices.sum() == 0

    def test_nonempty_for_ones_filter(self, simple_wedgelist, cubic_filter):
        _, _, wedge_slices = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        assert wedge_slices.sum() > 0


class TestGenerateWedgeMaskSlicesTile:
    def test_returns_ndarray(self, simple_wedgelist, cubic_filter):
        assert isinstance(pana.generate_wedgemask_slices_tile(simple_wedgelist, cubic_filter), np.ndarray)

    def test_output_shape_matches_filter(self, simple_wedgelist, cubic_filter):
        result = pana.generate_wedgemask_slices_tile(simple_wedgelist, cubic_filter)
        assert result.shape == cubic_filter.shape

    def test_binary_output(self, simple_wedgelist, cubic_filter):
        result = pana.generate_wedgemask_slices_tile(simple_wedgelist, cubic_filter)
        assert set(np.unique(result)).issubset({0.0, 1.0})

    def test_non_cubic_raises(self, simple_wedgelist):
        with pytest.raises(AssertionError):
            pana.generate_wedgemask_slices_tile(simple_wedgelist, np.ones((8, 8, 4)))

    def test_2d_filter_raises(self, simple_wedgelist):
        with pytest.raises(AssertionError):
            pana.generate_wedgemask_slices_tile(simple_wedgelist, np.ones((8, 8)))

    def test_nonempty_for_nonzero_tilts(self, simple_wedgelist, cubic_filter):
        assert pana.generate_wedgemask_slices_tile(simple_wedgelist, cubic_filter).sum() > 0

    def test_full_tilt_range_high_coverage(self, cubic_filter):
        wl_full = pd.DataFrame({"tilt_angle": np.arange(-90, 90, 1, dtype=float)})
        result = pana.generate_wedgemask_slices_tile(wl_full, cubic_filter)
        assert result.mean() > 0.5


# ── generate_exposure ──────────────────────────────────────────────────────────

@pytest.fixture
def exposure_wedgelist():
    n = 5
    return pd.DataFrame({
        "tilt_angle": np.linspace(-20, 20, n),
        "pixelsize": np.full(n, 1.35),
        "exposure": np.linspace(1.0, 10.0, n),
    })


class TestGenerateExposure:
    def test_output_shape(self, exposure_wedgelist):
        filt = np.ones((8, 8, 8))
        slices, weights, _ = pana.generate_wedgemask_slices_template(exposure_wedgelist, filt)
        result = pana.generate_exposure(exposure_wedgelist, slices, weights, binning=1)
        assert result.shape == filt.shape

    def test_output_nonnegative(self, exposure_wedgelist):
        filt = np.ones((8, 8, 8))
        slices, weights, _ = pana.generate_wedgemask_slices_template(exposure_wedgelist, filt)
        result = pana.generate_exposure(exposure_wedgelist, slices, weights, binning=1)
        assert np.all(result >= 0)

    def test_output_bounded_by_one(self, exposure_wedgelist):
        filt = np.ones((8, 8, 8))
        slices, weights, _ = pana.generate_wedgemask_slices_template(exposure_wedgelist, filt)
        result = pana.generate_exposure(exposure_wedgelist, slices, weights, binning=1)
        assert np.all(result <= 1.0 + 1e-9)

    def test_higher_exposure_lower_filter(self):
        filt = np.ones((8, 8, 8))
        wl_low = pd.DataFrame({"tilt_angle": [0.0], "pixelsize": [1.35], "exposure": [1.0]})
        wl_high = pd.DataFrame({"tilt_angle": [0.0], "pixelsize": [1.35], "exposure": [200.0]})
        sl_low, w_low, _ = pana.generate_wedgemask_slices_template(wl_low, filt)
        sl_high, w_high, _ = pana.generate_wedgemask_slices_template(wl_high, filt)
        r_low = pana.generate_exposure(wl_low, sl_low, w_low, binning=1)
        r_high = pana.generate_exposure(wl_high, sl_high, w_high, binning=1)
        assert r_low.sum() >= r_high.sum()

    def test_binning_affects_result(self, exposure_wedgelist):
        filt = np.ones((8, 8, 8))
        slices, weights, _ = pana.generate_wedgemask_slices_template(exposure_wedgelist, filt)
        r1 = pana.generate_exposure(exposure_wedgelist, slices, weights, binning=1)
        r4 = pana.generate_exposure(exposure_wedgelist, slices, weights, binning=4)
        assert not np.allclose(r1, r4)


# ── generate_ctf ───────────────────────────────────────────────────────────────

@pytest.fixture
def ctf_wedgelist(simple_wedgelist):
    wl = simple_wedgelist.copy()
    wl["tomo_x"] = 8
    wl["tomo_y"] = 8
    wl["tomo_z"] = 8
    wl["pixelsize"] = 1.35
    wl["defocus"] = 2.0
    wl["amp_contrast"] = 0.07
    wl["cs"] = 2.7
    wl["voltage"] = 300.0
    return wl


class TestGenerateCtf:
    def test_output_shape(self, ctf_wedgelist, simple_wedgelist, cubic_filter):
        slices, weights, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        result = imageutils.generate_ctf_slice(ctf_wedgelist, slices, weights, binning=1)
        assert result.shape == cubic_filter.shape

    def test_output_nonnegative(self, ctf_wedgelist, simple_wedgelist, cubic_filter):
        slices, weights, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        result = imageutils.generate_ctf_slice(ctf_wedgelist, slices, weights, binning=1)
        assert np.all(result >= 0)

    def test_output_finite(self, ctf_wedgelist, simple_wedgelist, cubic_filter):
        slices, weights, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        result = imageutils.generate_ctf_slice(ctf_wedgelist, slices, weights, binning=1)
        assert np.all(np.isfinite(result))

    def test_binning_affects_result(self, ctf_wedgelist, simple_wedgelist, cubic_filter):
        slices, weights, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        r1 = imageutils.generate_ctf_slice(ctf_wedgelist, slices, weights, binning=1)
        r2 = imageutils.generate_ctf_slice(ctf_wedgelist, slices, weights, binning=2)
        assert not np.allclose(r1, r2)

    def test_pshift_column_optional(self, ctf_wedgelist, simple_wedgelist, cubic_filter):
        wl_no_pshift = ctf_wedgelist.drop(columns=["pshift"], errors="ignore")
        slices, weights, _ = pana.generate_wedgemask_slices_template(simple_wedgelist, cubic_filter)
        result = imageutils.generate_ctf_slice(wl_no_pshift, slices, weights, binning=1)
        assert result.shape == cubic_filter.shape


# ── Layer 1 pure-compute functions ────────────────────────────────────────────


def _gaussian_map(size=32, peak_sigma=3.0):
    """Synthetic 3D Gaussian-peaked volume centred in the array."""
    center = size // 2
    coords = np.arange(size) - center
    x, y, z = np.meshgrid(coords, coords, coords, indexing="ij")
    return np.exp(-(x**2 + y**2 + z**2) / (2.0 * peak_sigma**2))


class TestFilterTemplateDf:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "Structure": ["ribo", "ribo", "tric", "tric"],
            "Boxsize": [80, 80, 64, 64],
            "Done": [True, False, True, False],
        })

    def test_single_condition(self, df):
        idx = pana.filter_template_df(df.copy(), {"Structure": "ribo"})
        assert list(idx) == [0, 1]

    def test_multiple_conditions(self, df):
        idx = pana.filter_template_df(df.copy(), {"Structure": "ribo", "Done": True})
        assert list(idx) == [0]

    def test_empty_result(self, df):
        idx = pana.filter_template_df(df.copy(), {"Structure": "none"})
        assert len(idx) == 0

    def test_sort_by(self, df):
        idx = pana.filter_template_df(df.copy(), {"Done": False}, sort_by="Boxsize")
        boxes = df.loc[idx, "Boxsize"].tolist()
        assert boxes == sorted(boxes)


class TestMaskStats:
    def test_returns_required_keys(self):
        soft = np.ones((10, 10, 10)) * 0.8
        sharp = np.ones((10, 10, 10))
        result = pana.mask_stats(soft, sharp)
        assert set(result.keys()) == {"voxels_soft", "voxels_sharp", "bbox", "solidity"}

    def test_voxels_sharp_positive(self):
        sharp = np.zeros((10, 10, 10))
        sharp[3:7, 3:7, 3:7] = 1.0
        result = pana.mask_stats(sharp.copy(), sharp)
        assert result["voxels_sharp"] > 0

    def test_solidity_in_range(self):
        sharp = np.zeros((10, 10, 10))
        sharp[3:7, 3:7, 3:7] = 1.0
        result = pana.mask_stats(sharp.copy(), sharp)
        assert 0.0 <= result["solidity"] <= 1.0

    def test_soft_voxels_uses_threshold(self):
        soft = np.zeros((10, 10, 10))
        soft[2:6, 2:6, 2:6] = 0.8  # above 0.5 → counted
        soft[6:8, 6:8, 6:8] = 0.3  # below 0.5 → ignored
        sharp = (soft > 0.4).astype(float)
        result = pana.mask_stats(soft, sharp)
        assert result["voxels_soft"] == 64  # 4×4×4


class TestSharpMaskOverlap:
    def test_returns_array_of_correct_length(self):
        mask = np.zeros((16, 16, 16))
        mask[5:11, 5:11, 5:11] = 1.0
        rotations = [
            srot.from_euler("z", 0, degrees=True),
            srot.from_euler("z", 45, degrees=True),
            srot.from_euler("z", 90, degrees=True),
        ]
        result = pana.sharp_mask_overlap(mask, rotations)
        assert isinstance(result, np.ndarray)
        assert len(result) == 3

    def test_non_negative_values(self):
        mask = np.zeros((16, 16, 16))
        mask[5:11, 5:11, 5:11] = 1.0
        rotations = [
            srot.from_euler("z", 0, degrees=True),
            srot.from_euler("z", 45, degrees=True),
        ]
        result = pana.sharp_mask_overlap(mask, rotations)
        assert np.all(result >= 0)

    def test_identity_rotation_max_overlap(self):
        mask = np.zeros((16, 16, 16))
        mask[5:11, 5:11, 5:11] = 1.0
        ov_id = pana.sharp_mask_overlap(mask, [srot.from_euler("z", 0, degrees=True)])
        ov_45 = pana.sharp_mask_overlap(mask, [srot.from_euler("z", 45, degrees=True)])
        assert ov_id[0] >= ov_45[0]


class TestFindMatchingOverlapRow:
    @pytest.fixture
    def df(self):
        return pd.DataFrame({
            "Tight mask": ["a", "a", "b", "a"],
            "Degrees": [5.0, 5.0, 5.0, 10.0],
            "Done": [True, False, True, True],
        })

    def test_finds_matching_rows(self, df):
        idx = pana.find_matching_overlap_row(df, 0)
        assert 0 in idx

    def test_excludes_done_false(self, df):
        idx = pana.find_matching_overlap_row(df, 0)
        assert 1 not in idx  # row 1 has Done=False

    def test_empty_when_no_match(self):
        df = pd.DataFrame({
            "Tight mask": ["unique"],
            "Degrees": [7.0],
            "Done": [False],
        })
        idx = pana.find_matching_overlap_row(df, 0)
        assert len(idx) == 0

    def test_different_mask_excluded(self, df):
        idx = pana.find_matching_overlap_row(df, 0)
        assert 2 not in idx  # row 2 has mask="b"


class TestDistMapStats:
    @pytest.fixture
    def simple_dist_map(self):
        dm = np.full((30, 30, 30), 20.0)
        dm[13:18, 13:18, 13:18] = 3.0
        dm[15, 15, 15] = 1.0
        return dm, (15, 15, 15)

    def test_returns_required_keys(self, simple_dist_map):
        dm, center = simple_dist_map
        result = pana.dist_map_stats(dm, center, degrees=5.0)
        assert set(result.keys()) == {"vc", "solidity", "label", "vco", "open_label", "dim"}

    def test_label_is_binary(self, simple_dist_map):
        dm, center = simple_dist_map
        result = pana.dist_map_stats(dm, center, degrees=5.0)
        assert set(np.unique(result["label"])).issubset({0.0, 1.0})

    def test_open_label_is_binary(self, simple_dist_map):
        dm, center = simple_dist_map
        result = pana.dist_map_stats(dm, center, degrees=5.0)
        assert set(np.unique(result["open_label"])).issubset({0.0, 1.0})

    def test_is_all_doubles_threshold(self):
        # Outer ring: values 8 (5 < 8 ≤ 10), visible with is_all but not single
        dm = np.full((30, 30, 30), 20.0)
        dm[12:19, 12:19, 12:19] = 8.0   # within 2*degrees=10, not within degrees=5
        dm[13:18, 13:18, 13:18] = 3.0   # within degrees=5
        dm[15, 15, 15] = 1.0
        res_single = pana.dist_map_stats(dm, (15, 15, 15), degrees=5.0, is_all=False)
        res_all = pana.dist_map_stats(dm, (15, 15, 15), degrees=5.0, is_all=True)
        assert res_all["vco"] >= res_single["vco"]


class TestPeakStatsAndProfiles:
    @pytest.fixture
    def scores_vol(self):
        return _gaussian_map(size=32, peak_sigma=3.0)

    def test_returns_required_keys(self, scores_vol):
        center = (16, 16, 16)
        result = pana.peak_stats_and_profiles(scores_vol, center, 1.0)
        expected_keys = {"peak_value", "line_profiles", "drop_x", "drop_y", "drop_z",
                         "peak_x", "peak_y", "peak_z"}
        for r in range(1, 6):
            expected_keys |= {f"mean_{r}", f"median_{r}", f"var_{r}"}
        assert expected_keys.issubset(set(result.keys()))

    def test_line_profiles_shape(self, scores_vol):
        center = (16, 16, 16)
        result = pana.peak_stats_and_profiles(scores_vol, center, 1.0)
        lp = result["line_profiles"]
        assert lp.ndim == 2 and lp.shape[1] == 3

    def test_spherical_stats_for_all_radii(self, scores_vol):
        center = (16, 16, 16)
        result = pana.peak_stats_and_profiles(scores_vol, center, 1.0)
        for r in range(1, 6):
            assert f"mean_{r}" in result and f"median_{r}" in result and f"var_{r}" in result


class TestPeakShapes:
    @pytest.fixture
    def scores_vol(self):
        return _gaussian_map(size=32, peak_sigma=4.0)

    def test_returns_required_keys(self, scores_vol):
        result = pana.peak_shapes(scores_vol)
        assert {"tp_shape", "gp_shape", "hp_shape", "peak_value"}.issubset(set(result.keys()))

    def test_peak_value_is_float(self, scores_vol):
        result = pana.peak_shapes(scores_vol)
        assert isinstance(result["peak_value"], float)

    def test_shape_arrays_length_3(self, scores_vol):
        result = pana.peak_shapes(scores_vol)
        for key in ("tp_shape", "gp_shape", "hp_shape"):
            assert len(result[key]) == 3


class TestShapeStats:
    def test_returns_dataframe(self):
        mask = np.zeros((20, 20, 20))
        mask[5:10, 5:10, 5:10] = 1.0
        result = pana.shape_stats(mask)
        assert isinstance(result, pd.DataFrame)

    def test_has_expected_columns(self):
        mask = np.zeros((20, 20, 20))
        mask[5:10, 5:10, 5:10] = 1.0
        result = pana.shape_stats(mask)
        assert "label" in result.columns and "solidity" in result.columns

    def test_one_component(self):
        mask = np.zeros((20, 20, 20))
        mask[5:10, 5:10, 5:10] = 1.0
        assert len(pana.shape_stats(mask)) == 1

    def test_two_components(self):
        mask = np.zeros((30, 30, 30))
        mask[2:6, 2:6, 2:6] = 1.0
        mask[20:25, 20:25, 20:25] = 1.0
        assert len(pana.shape_stats(mask)) == 2


class TestBuildSummaryFigure:
    @pytest.fixture
    def minimal_inputs(self):
        np.random.seed(0)
        n = 10
        rot_info = pd.DataFrame({
            "Tight mask overlap": np.arange(n, dtype=float),
            "ang_dist": np.linspace(0, 90, n),
            "ccc_masked": np.random.rand(n),
        })
        line_profiles = pd.DataFrame({
            "x": np.random.rand(8), "y": np.random.rand(8), "z": np.random.rand(8),
        })
        s2 = np.random.rand(8, 8)
        cross_slices = [[s2, s2, s2] for _ in range(6)]
        dicts = [[["k1", "v1"], ["k2", "v2"]], [["k3", "v3"]], [["k4", "v4"]]]
        return dicts, rot_info, line_profiles, cross_slices

    def test_returns_figure(self, minimal_inputs):
        import plotly.graph_objects as go
        dicts, rot_info, line_profiles, cross_slices = minimal_inputs
        fig = pana.build_summary_figure("Test", dicts, rot_info, line_profiles, cross_slices, 1.0)
        assert isinstance(fig, go.Figure)

    def test_with_hist_returns_figure(self, minimal_inputs):
        import plotly.graph_objects as go
        dicts, rot_info, line_profiles, cross_slices = minimal_inputs
        hist_info = pd.DataFrame({
            "ang_dist": np.random.rand(100),
            "cone_dist": np.random.rand(100),
            "inplane_dist": np.random.rand(100),
        })
        hist_info2 = pd.DataFrame({
            "ccc_masked": np.random.rand(359),
            "cone_ccc_masked": np.random.rand(359),
            "inplane_ccc_masked": np.random.rand(359),
        })
        fig = pana.build_summary_figure(
            "Test", dicts, rot_info, line_profiles, cross_slices, 1.0,
            hist_info=hist_info, hist_info2=hist_info2,
        )
        assert isinstance(fig, go.Figure)


class TestRunAnalysisArgsFromRow:
    @pytest.fixture
    def tmpl_row(self):
        return pd.Series({
            "Structure": "ribosome",
            "Template": "mytemplate",
            "Mask": "mymask",
            "Compare": "tmpl",
            "Phi": 0.0, "Theta": 45.0, "Psi": 90.0,
            "Symmetry": 1,
        })

    @pytest.fixture
    def subtomo_row(self):
        return pd.Series({
            "Structure": "ribosome",
            "Template": "mytemplate",
            "Mask": "mymask",
            "Compare": "subtomo",
            "Tomo map": "tomofile",
            "Tomogram": "ts_001",
            "Apply wedge": False,
            "Boxsize": 80,
            "Binning": 4,
            "Phi": 0.0, "Theta": 0.0, "Psi": 0.0,
            "Symmetry": 1,
        })

    def test_returns_required_keys(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert set(result.keys()) == {
            "structure_name", "template", "mask", "tomo",
            "wedge_tomo", "wedge_tmpl", "starting_angle", "cyclic_symmetry",
        }

    def test_tmpl_compare_tomo_equals_template(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert result["tomo"] == result["template"]

    def test_no_wedge_when_apply_wedge_false(self, subtomo_row):
        result = pana._run_analysis_args_from_row(subtomo_row, "/base/", "/wedge/")
        assert result["wedge_tomo"] is None
        assert result["wedge_tmpl"] is None

    def test_starting_angle_shape(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert result["starting_angle"].shape == (1, 3)

    def test_structure_name_in_result(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert result["structure_name"] == "ribosome"
