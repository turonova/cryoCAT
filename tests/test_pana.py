import numpy as np
import pandas as pd
import os
from glob import glob
from cryocat.analysis import pana
from cryocat.utils import imageutils
from cryocat.utils import wedgeutils
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


class TestComputeSharpMaskOverlapSingle:
    @pytest.fixture
    def mask_and_angles(self):
        mask = np.zeros((16, 16, 16))
        mask[5:11, 5:11, 5:11] = 1.0
        angles = np.array([[0.0, 0.0, 0.0], [45.0, 0.0, 0.0]])
        return mask, angles

    def test_returns_dict_with_overlap(self, mask_and_angles):
        mask, angles = mask_and_angles
        result = pana.compute_sharp_mask_overlap_single(mask, angles)
        assert "overlap" in result
        assert "angles" in result

    def test_overlap_length_matches_angles(self, mask_and_angles):
        mask, angles = mask_and_angles
        result = pana.compute_sharp_mask_overlap_single(mask, angles)
        assert len(result["overlap"]) == len(angles)

    def test_identity_angle_has_max_overlap(self, mask_and_angles):
        mask, angles = mask_and_angles
        result = pana.compute_sharp_mask_overlap_single(mask, angles)
        assert result["overlap"][0] >= result["overlap"][1]


class TestComputeShapeStatsSingle:
    @pytest.fixture
    def scores_vol(self):
        return _gaussian_map(size=32, peak_sigma=4.0)

    def test_returns_dict_with_required_keys(self, scores_vol):
        result = pana.compute_shape_stats_single(scores_vol)
        assert {"tp_shape", "gp_shape", "hp_shape", "peak_value", "shape_type"}.issubset(result.keys())

    def test_peak_value_is_float(self, scores_vol):
        result = pana.compute_shape_stats_single(scores_vol)
        assert isinstance(result["peak_value"], float)

    def test_shape_arrays_have_length_3(self, scores_vol):
        result = pana.compute_shape_stats_single(scores_vol)
        for k in ("tp_shape", "gp_shape", "hp_shape"):
            assert len(result[k]) == 3

    def test_shape_type_echoed(self, scores_vol):
        result = pana.compute_shape_stats_single(scores_vol, shape_type="loose")
        assert result["shape_type"] == "loose"


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
            "structure_name", "template", "mask", "target_map",
            "wedge_target", "wedge_tmpl", "starting_angle", "cyclic_symmetry",
        }

    def test_tmpl_compare_tomo_equals_template(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert result["target_map"] == result["template"]

    def test_no_wedge_when_apply_wedge_false(self, subtomo_row):
        result = pana._run_analysis_args_from_row(subtomo_row, "/base/", "/wedge/")
        assert result["wedge_target"] is None
        assert result["wedge_tmpl"] is None

    def test_starting_angle_shape(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert result["starting_angle"].shape == (1, 3)

    def test_structure_name_in_result(self, tmpl_row):
        result = pana._run_analysis_args_from_row(tmpl_row, "/base/", "/wedge/")
        assert result["structure_name"] == "ribosome"


# ── Layer 2 single-case functions ─────────────────────────────────────────────


def _make_angles_map(size=24, n_angles=5, peak_angle_0based=2, cc_radius=5):
    """Synthetic angles_map: peak_angle_0based at centre cube, -1 elsewhere."""
    amap = np.full((size, size, size), -1.0)
    c = size // 2
    r = cc_radius
    amap[c - r : c + r + 1, c - r : c + r + 1, c - r : c + r + 1] = float(peak_angle_0based)
    # a few corner voxels with a different (non-peak) angle
    other = float((peak_angle_0based + 1) % n_angles)
    amap[0:2, 0:2, 0:2] = other
    return amap


class TestResolveWriteDir:
    def test_overwrite_creates_dir(self, tmp_path):
        d = pana._resolve_write_dir(tmp_path, "mycase", "overwrite")
        assert d.is_dir() and d.name == "mycase"

    def test_overwrite_idempotent(self, tmp_path):
        d1 = pana._resolve_write_dir(tmp_path, "mycase", "overwrite")
        d2 = pana._resolve_write_dir(tmp_path, "mycase", "overwrite")
        assert d1 == d2

    def test_error_raises_when_artifact_exists(self, tmp_path):
        case_dir = tmp_path / "mycase"
        case_dir.mkdir()
        (case_dir / "scores.em").write_bytes(b"")
        with pytest.raises(FileExistsError):
            pana._resolve_write_dir(tmp_path, "mycase", "error")

    def test_error_succeeds_when_no_artifacts(self, tmp_path):
        d = pana._resolve_write_dir(tmp_path, "fresh_case", "error")
        assert d.is_dir()

    def test_timestamp_creates_run_subdir(self, tmp_path):
        d = pana._resolve_write_dir(tmp_path, "mycase", "timestamp")
        assert d.name.startswith("run_") and d.parent.name == "mycase"

    def test_timestamp_two_calls_differ(self, tmp_path):
        import time
        d1 = pana._resolve_write_dir(tmp_path, "mycase", "timestamp")
        time.sleep(1.1)
        d2 = pana._resolve_write_dir(tmp_path, "mycase", "timestamp")
        assert d1 != d2


class TestComputeDistanceMap:
    N_ANGLES = 5
    SIZE = 24
    PEAK_0BASED = 2

    @pytest.fixture
    def angles(self):
        a = np.zeros((self.N_ANGLES, 3))
        for i in range(self.N_ANGLES):
            a[i] = [i * 15, 0, 0]
        return a

    @pytest.fixture
    def angles_map(self):
        return _make_angles_map(
            size=self.SIZE,
            n_angles=self.N_ANGLES,
            peak_angle_0based=self.PEAK_0BASED,
        )

    def test_returns_dict_with_required_keys(self, angles_map, angles):
        result = pana.compute_distance_map(angles_map, angles)
        assert isinstance(result, dict)
        for k in ("dist_all", "dist_normals", "dist_inplane"):
            assert k in result
            assert isinstance(result[k], np.ndarray)

    def test_output_shape_matches_input(self, angles_map, angles):
        result = pana.compute_distance_map(angles_map, angles)
        assert result["dist_all"].shape == angles_map.shape
        assert result["dist_normals"].shape == angles_map.shape
        assert result["dist_inplane"].shape == angles_map.shape

    def test_unset_voxels_have_zero_distance(self, angles_map, angles):
        result = pana.compute_distance_map(angles_map, angles)
        assert np.all(result["dist_all"][angles_map < 0] == 0.0)

    def test_default_reference_is_zero_rotation(self):
        """starting_angle=None → reference is (0,0,0); voxels with idx=0 get dist_all=0."""
        angles = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
        amap = np.full((8, 8, 8), -1.0)
        c = 4
        amap[c-1:c+2, c-1:c+2, c-1:c+2] = 0.0  # index 0 → (0°,0,0)
        amap[0, 0, 0] = 1.0                       # index 1 → (30°,0,0)

        result = pana.compute_distance_map(amap, angles)  # starting_angle=None
        np.testing.assert_allclose(result["dist_all"][amap == 0.0], 0.0, atol=1e-5)
        assert result["dist_all"][0, 0, 0] > 0.0

    def test_explicit_starting_angle_yields_zero_at_matching_angle(self):
        """Voxels whose angle equals starting_angle have dist_all ≈ 0."""
        angles = np.array([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
        amap = np.full((8, 8, 8), -1.0)
        c = 4
        amap[c-1:c+2, c-1:c+2, c-1:c+2] = 1.0  # index 1 → (30°,0,0)
        amap[0, 0, 0] = 0.0                       # index 0 → (0°,0,0)

        result = pana.compute_distance_map(amap, angles, starting_angle=[30.0, 0.0, 0.0])
        np.testing.assert_allclose(result["dist_all"][amap == 1.0], 0.0, atol=1e-5)
        assert result["dist_all"][0, 0, 0] > 0.0

    def test_writes_three_distance_files_with_output_dir(self, tmp_path, angles_map, angles):
        pana.compute_distance_map(angles_map, angles, output_dir=str(tmp_path))
        assert (tmp_path / "distance_map_all.em").exists()
        assert (tmp_path / "distance_map_normals.em").exists()
        assert (tmp_path / "distance_map_inplane.em").exists()

    def test_labels_none_without_scores(self, angles_map, angles):
        result = pana.compute_distance_map(angles_map, angles)
        assert result["labels_all"] is None
        assert result["labels_normals"] is None
        assert result["labels_inplane"] is None

    def test_no_cc_radius_parameter(self, angles_map, angles):
        """cc_radius was removed; passing it must raise TypeError."""
        with pytest.raises(TypeError):
            pana.compute_distance_map(angles_map, angles, cc_radius=5)


class TestComputePeakStats:
    SIZE = 32

    @pytest.fixture
    def scores_vol(self):
        return _gaussian_map(size=self.SIZE, peak_sigma=3.0)

    @pytest.fixture
    def dist_maps(self):
        dm = np.full((self.SIZE, self.SIZE, self.SIZE), 20.0)
        c = self.SIZE // 2
        dm[c - 3 : c + 4, c - 3 : c + 4, c - 3 : c + 4] = 2.0
        return dm, dm.copy(), dm.copy()

    def test_returns_required_top_level_keys(self, scores_vol, dist_maps):
        da, dn, di = dist_maps
        result = pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0)
        assert "peak_stats" in result and "dist_maps" in result

    def test_dist_maps_subkeys(self, scores_vol, dist_maps):
        da, dn, di = dist_maps
        result = pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0)
        assert set(result["dist_maps"].keys()) == {"dist_all", "dist_normals", "dist_inplane"}

    def test_peak_stats_has_required_fields(self, scores_vol, dist_maps):
        da, dn, di = dist_maps
        result = pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0)
        ps = result["peak_stats"]
        required = {"peak_value", "peak_x", "peak_y", "peak_z",
                    "drop_x", "drop_y", "drop_z"}
        for r in range(1, 6):
            required |= {f"mean_{r}", f"median_{r}", f"var_{r}"}
        assert required.issubset(set(ps.keys()))

    def test_peak_value_positive(self, scores_vol, dist_maps):
        da, dn, di = dist_maps
        result = pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0)
        assert result["peak_stats"]["peak_value"] > 0.0

    def test_each_dist_entry_has_required_fields(self, scores_vol, dist_maps):
        da, dn, di = dist_maps
        result = pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0)
        for name in ["dist_all", "dist_normals", "dist_inplane"]:
            assert {"vc", "solidity", "vco", "dim"}.issubset(result["dist_maps"][name].keys())

    def test_writes_files_with_output_dir(self, tmp_path, scores_vol, dist_maps):
        import json as _json
        da, dn, di = dist_maps
        pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0, output_dir=str(tmp_path))
        stats_file = tmp_path / "stats.json"
        assert stats_file.is_file()
        with open(stats_file) as f:
            data = _json.load(f)
        assert "peak_stats" in data and "dist_maps" in data
        assert (tmp_path / "peak_line_profiles.csv").is_file()

    def test_returns_peak_line_profiles_dataframe(self, scores_vol, dist_maps):
        import pandas as pd
        da, dn, di = dist_maps
        result = pana.compute_peak_stats(scores_vol, da, dn, di, degrees=5.0)
        assert "peak_line_profiles" in result
        assert isinstance(result["peak_line_profiles"], pd.DataFrame)
        assert list(result["peak_line_profiles"].columns) == ["x", "y", "z"]


class TestVisualizeResults:
    SIZE = 32

    @pytest.fixture
    def scores_vol(self):
        return _gaussian_map(size=self.SIZE, peak_sigma=3.0)

    @pytest.fixture
    def angles_map_vol(self):
        amap = np.full((self.SIZE,) * 3, -1.0)
        c = self.SIZE // 2
        amap[c - 3 : c + 4, c - 3 : c + 4, c - 3 : c + 4] = 2.0
        return amap

    @pytest.fixture
    def dist_vol(self):
        return _gaussian_map(size=self.SIZE, peak_sigma=3.0) * 20.0

    def test_returns_dict(self, scores_vol):
        result = pana.visualize_results(scores=scores_vol)
        assert isinstance(result, dict)

    def test_no_args_returns_empty_dict(self):
        result = pana.visualize_results()
        assert result == {}

    def test_score_panels_present(self, scores_vol):
        result = pana.visualize_results(scores=scores_vol)
        assert "score_slices" in result
        assert "line_profiles" in result

    def test_distance_panel_present(self, dist_vol):
        result = pana.visualize_results(dist_all_map=dist_vol)
        assert "distance_slices_all" in result

    def test_angle_distribution_present(self, angles_map_vol):
        result = pana.visualize_results(angles_map=angles_map_vol)
        assert "angle_distribution" in result

    def test_loads_peak_stats_from_json(self, tmp_path, scores_vol):
        import json as _json
        c = self.SIZE // 2
        stats = {"peak_stats": {"peak_x": c, "peak_y": c, "peak_z": c}}
        stats_path = tmp_path / "stats.json"
        with open(stats_path, "w") as f:
            _json.dump(stats, f)
        result = pana.visualize_results(scores=scores_vol, peak_stats=stats_path)
        assert "score_slices" in result


class TestRunSingleCase:
    SIZE = 16

    @pytest.fixture
    def syn_data(self):
        size = self.SIZE
        scores = _gaussian_map(size=size)
        n_angles = 4
        amap = np.full((size, size, size), -1.0)
        c = size // 2
        amap[c - 2 : c + 3, c - 2 : c + 3, c - 2 : c + 3] = 2.0
        angles = np.array([[i * 20.0, 0.0, 0.0] for i in range(n_angles)])
        return scores, amap.astype(float), angles

    def test_creates_case_subdir(self, tmp_path, mocker, syn_data):
        scores, amap, angles = syn_data
        mocker.patch("cryocat.analysis.pana.analyze_rotations",
                     return_value=(pd.DataFrame(), scores, amap, scores))
        mocker.patch("cryocat.core.cryomap.write")
        pana.run_single_case(
            target_map=scores, template=scores, template_mask=scores,
            input_angles=angles, output_dir=tmp_path, case_name="run1",
            compute_distance_map=False, compute_peak_stats=False,
        )
        assert (tmp_path / "run1").is_dir()

    def test_returns_required_keys(self, tmp_path, mocker, syn_data):
        scores, amap, angles = syn_data
        mocker.patch("cryocat.analysis.pana.analyze_rotations",
                     return_value=(pd.DataFrame(), scores, amap, scores))
        mocker.patch("cryocat.core.cryomap.write")
        result = pana.run_single_case(
            target_map=scores, template=scores, template_mask=scores,
            input_angles=angles, output_dir=tmp_path, case_name="run2",
            compute_distance_map=False, compute_peak_stats=False,
        )
        assert {"res_table", "scores_map", "angles_map", "write_dir"}.issubset(result.keys())

    def test_dist_map_keys_in_result_when_enabled(self, tmp_path, mocker, syn_data):
        scores, amap, angles = syn_data
        mocker.patch("cryocat.analysis.pana.analyze_rotations",
                     return_value=(pd.DataFrame(), scores, amap, scores))
        mocker.patch("cryocat.core.cryomap.write")
        result = pana.run_single_case(
            target_map=scores, template=scores, template_mask=scores,
            input_angles=angles, output_dir=tmp_path, case_name="run3",
            compute_distance_map=True, compute_peak_stats=False,
        )
        assert {"dist_all_map", "dist_normals_map", "dist_inplane_map"}.issubset(result.keys())

    def test_if_exists_error_raises_on_existing_artifact(self, tmp_path, syn_data):
        scores, amap, angles = syn_data
        case_dir = tmp_path / "run_err"
        case_dir.mkdir()
        (case_dir / "scores.em").write_bytes(b"")
        with pytest.raises(FileExistsError):
            pana.run_single_case(
                target_map=scores, template=scores, template_mask=scores,
                input_angles=angles, output_dir=tmp_path, case_name="run_err",
                compute_distance_map=False, compute_peak_stats=False,
                if_exists="error",
            )

    def test_if_exists_timestamp_creates_run_subdir(self, tmp_path, mocker, syn_data):
        scores, amap, angles = syn_data
        mocker.patch("cryocat.analysis.pana.analyze_rotations",
                     return_value=(pd.DataFrame(), scores, amap, scores))
        mocker.patch("cryocat.core.cryomap.write")
        result = pana.run_single_case(
            target_map=scores, template=scores, template_mask=scores,
            input_angles=angles, output_dir=tmp_path, case_name="run_ts",
            compute_distance_map=False, compute_peak_stats=False,
            if_exists="timestamp",
        )
        write_dir = result["write_dir"]
        assert write_dir.name.startswith("run_") and write_dir.parent.name == "run_ts"

    def test_starting_angle_forwarded_to_compute_distance_map(self, tmp_path, mocker, syn_data):
        """run_single_case must pass its starting_angle to _compute_distance_map."""
        scores, amap, angles = syn_data
        starting_angle = np.array([30.0, 0.0, 0.0])
        mocker.patch("cryocat.analysis.pana.analyze_rotations",
                     return_value=(pd.DataFrame(), scores, amap, scores))
        mocker.patch("cryocat.core.cryomap.write")
        mock_cdm = mocker.patch(
            "cryocat.analysis.pana._compute_distance_map",
            return_value={
                "dist_all": amap, "dist_normals": amap, "dist_inplane": amap,
                "labels_all": None, "labels_normals": None, "labels_inplane": None,
                "labels_all_open": None, "labels_normals_open": None, "labels_inplane_open": None,
                "output_dir": None,
            },
        )
        pana.run_single_case(
            target_map=scores, template=scores, template_mask=scores,
            input_angles=angles, output_dir=tmp_path, case_name="fwd_sa",
            starting_angle=starting_angle,
            compute_distance_map=True, compute_peak_stats=False,
        )
        assert mock_cdm.called
        call_kwargs = mock_cdm.call_args[1]
        np.testing.assert_array_equal(call_kwargs["starting_angle"], starting_angle)


# ── angles.em 0-based indexing invariant ─────────────────────────────────────

def test_analyze_rotations_angles_map_is_zero_based(tmp_path):
    """angles.em must use 0-based indices; index 0 must appear for the first angle."""
    size = 16
    tomogram = np.zeros((size, size, size), dtype=np.single)
    # Place signal at centre so the first angle (identity) is preferred
    c = size // 2
    tomogram[c, c, c] = 1.0
    reference = tomogram.copy()
    # Two angles: identity [0,0,0] and 90° rotation — identity should win
    angles = np.array([[0.0, 0.0, 0.0], [0.0, 90.0, 0.0]])
    angles_path = tmp_path / "angles.csv"
    import pandas as pd
    pd.DataFrame(angles).to_csv(str(angles_path), header=False, index=False)
    mask = np.ones((size, size, size), dtype=np.single)
    _, _, angles_map, _ = pana.analyze_rotations(
        target_map=tomogram,
        template=reference,
        template_mask=mask,
        input_angles=str(angles_path),
        output_path=str(tmp_path / "out"),
        starting_angle=[0.0, 0.0, 0.0],
    )
    valid = angles_map[angles_map >= 0]
    assert len(valid) > 0, "No voxels were updated in angles_map"
    assert valid.min() >= 0, "angles.em contains negative values (1-based off-by-one?)"
    assert valid.max() < len(angles), "angles.em index exceeds number of angles"


# ── extract_best_subtomogram ──────────────────────────────────────────────────


class TestExtractBestSubtomogram:
    SIZE = 24

    @pytest.fixture
    def tomo_arr(self):
        arr = np.zeros((self.SIZE,) * 3, dtype=np.single)
        c = self.SIZE // 2
        arr[c, c, c] = 1.0
        return arr

    @pytest.fixture
    def motl_df(self):
        return pd.DataFrame({
            "x": [self.SIZE // 2], "y": [self.SIZE // 2], "z": [self.SIZE // 2],
            "shift_x": [0.0], "shift_y": [0.0], "shift_z": [0.0],
            "phi": [0.0], "theta": [0.0], "psi": [0.0],
            "score": [1.0],
        })

    def test_returns_required_keys(self, tomo_arr, motl_df, mocker):
        mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo",
                     return_value=(tomo_arr, np.array([0.0, 0.0, 0.0])))
        result = pana.extract_best_subtomogram(tomo_arr, motl_df, box_size=16)
        assert set(result.keys()) == {"subtomogram", "rotation", "output_path"}

    def test_output_path_none_by_default(self, tomo_arr, motl_df, mocker):
        mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo",
                     return_value=(tomo_arr, np.array([0.0, 0.0, 0.0])))
        result = pana.extract_best_subtomogram(tomo_arr, motl_df, box_size=16)
        assert result["output_path"] is None

    def test_output_path_forwarded(self, tomo_arr, motl_df, mocker, tmp_path):
        out = str(tmp_path / "subtomo.em")
        mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo",
                     return_value=(tomo_arr, np.array([0.0, 0.0, 0.0])))
        result = pana.extract_best_subtomogram(tomo_arr, motl_df, box_size=16, output_path=out)
        assert result["output_path"] == out

    def test_subtomogram_is_ndarray(self, tomo_arr, motl_df, mocker):
        mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo",
                     return_value=(tomo_arr, np.array([10.0, 20.0, 30.0])))
        result = pana.extract_best_subtomogram(tomo_arr, motl_df, box_size=16)
        assert isinstance(result["subtomogram"], np.ndarray)

    def test_rotation_is_ndarray(self, tomo_arr, motl_df, mocker):
        rot = np.array([10.0, 20.0, 30.0])
        mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo",
                     return_value=(tomo_arr, rot))
        result = pana.extract_best_subtomogram(tomo_arr, motl_df, box_size=16)
        np.testing.assert_array_equal(result["rotation"], rot)

    def test_scalar_box_size_treated_as_cubic(self, tomo_arr, motl_df, mocker):
        captured = {}
        def _fake_cut(tomo, motl, shape, path):
            captured["shape"] = shape
            return tomo_arr, np.zeros(3)
        mocker.patch("cryocat.analysis.pana.cut_the_best_subtomo", side_effect=_fake_cut)
        pana.extract_best_subtomogram(tomo_arr, motl_df, box_size=20)
        np.testing.assert_array_equal(captured["shape"], [20, 20, 20])

# ── run_single_case: wedgelist kwarg pinned as removed ───────────────────────


def test_run_single_case_rejects_wedgelist_kwarg():
    """run_single_case must not accept wedgelist/tomo_number — those belong
    to generate_wedge_masks_single or the batch path."""
    arr = np.ones((8,) * 3, dtype=np.single)
    angles = np.zeros((2, 3))
    with pytest.raises(TypeError):
        pana.run_single_case(arr, arr, arr, angles,
                             output_dir="/tmp", case_name="x",
                             wedgelist="fake.star", tomo_number=1)


# ── wedgeutils.generate_wedge_mask parity ─────────────────────────────────────

_WL_STAR = Path(__file__).parent / "test_data" / "wedgeutils_data" / "wedge_list.star"


def test_generate_wedge_mask_template_side_shape_and_range():
    # generate_wedge_mask replaced the template-side branch of the old
    # pana.generate_wedge_masks.  Verify it produces a correctly-shaped,
    # unit-interval float mask for a cubic template volume.
    template_size = 32
    result = wedgeutils.generate_wedge_mask(template_size, str(_WL_STAR), 17)
    mask = result["mask"]
    assert mask.shape == (template_size,) * 3
    assert mask.dtype in (np.float32, np.float64)
    assert mask.min() >= 0.0
    assert mask.max() <= 1.0 + 1e-6


# ── Docstring alias invariant ─────────────────────────────────────────────────

import inspect
import re

_ALIAS_TOKENS = {"MapSource", "DataSource", "EulerAngles", "Symmetry", "PathOrStr", "TripletLike"}


def _params_section(doc):
    m = re.search(
        r"Parameters\n\s*-{3,}\n(.*?)(?:\n\s*(?:Returns|Raises|Notes|Examples)\n\s*-{3,}|$)",
        doc or "",
        re.S,
    )
    return m.group(1) if m else ""


def test_docstrings_use_type_aliases():
    for name in (
        "analyze_rotations",
        "run_single_case",
        "compute_distance_map",
        "compute_peak_stats",
        "visualize_results",
    ):
        fn = getattr(pana, name)
        sig = inspect.signature(fn)
        doc_params = _params_section(fn.__doc__)
        for p in sig.parameters.values():
            ann = str(p.annotation)
            for alias in _ALIAS_TOKENS:
                if alias in ann:
                    assert alias in doc_params, (
                        f"{name}: parameter '{p.name}' uses {alias} in its annotation "
                        f"but the docstring Parameters section does not name it."
                    )
