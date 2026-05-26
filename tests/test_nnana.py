import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from cryocat.analysis import nnana
from cryocat.analysis.structure import Chain
from cryocat.core import cryomotl

"""
# for test creation
for f in ["tomo_id", "object_id"]:
    for i in ["angular_distance", "cone_distance", "in_plane_distance"]:
        df = nnana.get_nn_stats(m, m, column_name=f, nn_number=2, rotation_type=i)
        if f == "object_id":
            df=df.sort_values(by="distance")
        df = df.iloc[31:]
        df=df.sort_values(by="subtomo_idx")
        df = df.round(4)
        df.reset_index(inplace=True, drop=True)
        df.to_csv(f"./tests/test_data/nnana_data/nn_{f}_{i}.csv", index=False)

# for radius
for f in ["tomo_id", "object_id"]:
    for r in [0.1, 0.51, 1.0]:
        df = nnana.get_nn_stats_within_radius(m, nn_radius=r, column_name=f)
        #df = df.iloc[31:]
        #df=df.sort_values(by="subtomo_idx")
        df = df.round(4)
        #df.reset_index(inplace=True, drop=True)
        df.to_csv(f"./tests/test_data/nnana_data/nn_stats_radius_{f}_{str(r)}.csv", index=False)

"""


@pytest.fixture
def motl():
    motl = cryomotl.Motl.load(Path(__file__).parent / "test_data" / "nn_test_motl.em")
    return motl


@pytest.mark.parametrize(
    "column_name, expected_res",
    [
        (
            "tomo_id",
            np.asarray([7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 3, 2, 4, 5, 6, 6, 6, 7]),
        ),
        (
            "object_id",
            np.asarray([7, 4, 5, 6, 6, 6, 7, 4, 4, 4, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 4, 3, 3, 2]),
        ),
    ],
)
def test_get_nn_within_radius(motl, column_name, expected_res):

    # expected_res = np.asarray(
    #    [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 3, 2, 4, 5, 6, 6, 6, 7]
    # )
    res = nnana.get_nn_within_radius(motl, motl, nn_radius=0.6, column_name=column_name)

    np.testing.assert_array_equal(expected_res, res)


@pytest.mark.parametrize(
    "column_name, rotation_type",
    [
        ("tomo_id", "angular_distance"),
        ("tomo_id", "cone_distance"),
        ("tomo_id", "in_plane_distance"),
        ("object_id", "angular_distance"),
        ("object_id", "cone_distance"),
        ("object_id", "in_plane_distance"),
    ],
)
def test_get_nn_stats(motl, column_name, rotation_type):

    df = nnana.get_nn_stats(motl, motl, column_name=column_name, nn_number=2, rotation_type=rotation_type)
    exp_df = pd.read_csv(f"./tests/test_data/nnana_data/nn_{column_name}_{rotation_type}.csv")
    if column_name == "object_id":
        df = df.sort_values(by="distance")
    df = df.iloc[31:]
    df = df.sort_values(by="subtomo_idx")
    df = df.round(4)
    df.reset_index(inplace=True, drop=True)

    pd.testing.assert_frame_equal(df, exp_df, atol=1e-10)


@pytest.mark.parametrize(
    "column_name, radius",
    [
        ("tomo_id", 0.1),
        ("tomo_id", 0.51),
        ("tomo_id", 1.0),
        ("object_id", 0.1),
        ("object_id", 0.51),
        ("object_id", 1.0),
    ],
)
def test_get_nn_stats_within_radius(motl, column_name, radius):

    df = nnana.get_nn_stats_within_radius(motl, nn_radius=radius, column_name=column_name)
    exp_df = pd.read_csv(f"./tests/test_data/nnana_data/nn_stats_radius_{column_name}_{str(radius)}.csv")
    df = df.round(4)
    pd.testing.assert_frame_equal(df, exp_df, atol=1e-10, check_dtype=False)


def test_trace_chains():
    data_dir = Path(__file__).parent / "test_data" / "nnana_data"
    pixel_size = 0.1971
    max_distance = 20 / pixel_size

    motl_entry_path = str(data_dir / "n_entry_subset.em")
    motl_exit_path = str(data_dir / "n_exit_subset.em")
    motl_path = str(data_dir / "n_subset.em")

    chain = Chain.from_motls(motl_entry_path, motl_exit_path, max_distance=max_distance, min_distance=0)
    chain.traced_motl.df.sort_values(["tomo_id", "object_id", "geom2"], inplace=True)

    chain.get_occupancy()
    m_entry_traced = chain.add_traced_info(motl_entry_path)
    m_exit_traced = chain.add_traced_info(motl_exit_path)
    m_traced = chain.add_traced_info(motl_path)

    exact_cols = ["geom1", "geom2", "object_id"]
    float_cols = ["geom4"]

    for result, gt_file in [
        (m_entry_traced, "gt_n_entry_subset.em"),
        (m_exit_traced, "gt_n_exit_subset.em"),
        (m_traced, "gt_n_subset.em"),
    ]:
        gt = cryomotl.Motl.load(str(data_dir / gt_file))
        result_df = result.df.sort_values("subtomo_id").reset_index(drop=True)
        gt_df = gt.df.sort_values("subtomo_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(result_df[exact_cols], gt_df[exact_cols], check_dtype=False)
        pd.testing.assert_frame_equal(result_df[float_cols], gt_df[float_cols], check_dtype=False, atol=1e-6)


# =============================================================================
# Layer-1 helpers
# =============================================================================


class TestFindNnIndices:
    def test_returns_four_values(self):
        coords = np.random.rand(5, 3)
        result = nnana.find_nn_indices(coords, coords, k=1)
        assert len(result) == 4

    def test_qp_idx_is_arange(self):
        coords = np.random.rand(6, 3)
        qp, _, _, _ = nnana.find_nn_indices(coords, coords, k=1, remove_qp=True)
        np.testing.assert_array_equal(qp, np.arange(6))

    def test_remove_qp_drops_self_match(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        _, _, dist, _ = nnana.find_nn_indices(coords, coords, k=1, remove_qp=True)
        assert np.all(dist > 0)

    def test_k_neighbors_shape(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.], [3., 0., 0.]])
        _, nn, _, k = nnana.find_nn_indices(coords, coords, k=2, remove_qp=True)
        assert k == 2
        assert nn.shape == (4, 2)

    def test_nearest_is_closest(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.]])
        ref = np.array([[0., 0., 0.]])
        _, nn, dist, _ = nnana.find_nn_indices(ref, coords, k=1)
        assert nn[0, 0] == 0
        assert dist[0, 0] == pytest.approx(0.0)


class TestFindNnWithinRadius:
    def test_no_neighbors_far_apart(self):
        q = np.array([[0., 0., 0.]])
        n = np.array([[100., 0., 0.]])
        qp, nn = nnana.find_nn_within_radius(q, n, radius=1.0)
        assert len(qp) == 0

    def test_neighbors_within_radius_found(self):
        coords = np.array([[0., 0., 0.], [0.5, 0., 0.], [5., 0., 0.]])
        qp, nn = nnana.find_nn_within_radius(coords[:1], coords, radius=1.0)
        assert 0 in nn[0] and 1 in nn[0]

    def test_remove_qp_excludes_self(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.]])
        qp, nn = nnana.find_nn_within_radius(coords, coords, radius=2.0, remove_qp=True)
        for center, neighbors in zip(qp, nn):
            assert center not in neighbors

    def test_returns_sorted_neighbors(self):
        coords = np.array([[0., 0., 0.], [2., 0., 0.], [1., 0., 0.]])
        qp, nn = nnana.find_nn_within_radius(coords[:1], coords, radius=3.0)
        assert list(nn[0]) == sorted(nn[0])


class TestFindNnWithinSelf:
    def test_basic_pairs_found(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [10., 0., 0.]])
        center, nn = nnana.find_nn_within_self(coords, radius=2.0)
        assert len(center) > 0

    def test_unique_only_reduces_count(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        c_uniq, _ = nnana.find_nn_within_self(coords, radius=1.5, unique_only=True)
        c_all, _ = nnana.find_nn_within_self(coords, radius=1.5, unique_only=False)
        assert len(c_uniq) <= len(c_all)

    def test_self_not_in_neighbors(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        centers, nn_list = nnana.find_nn_within_self(coords, radius=1.5, unique_only=False)
        for c, neighbors in zip(centers, nn_list):
            assert c not in neighbors

    def test_large_radius_finds_at_least_one(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        center, _ = nnana.find_nn_within_self(coords, radius=100.0, unique_only=True)
        assert len(center) >= 1


class TestNmsByDistance:
    def test_returns_boolean_mask(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [5., 0., 0.]])
        scores = np.array([0.9, 0.5, 0.8])
        mask = nnana.nms_by_distance(coords, scores, distance=2.0)
        assert mask.dtype == bool
        assert mask.shape == (3,)

    def test_keep_greater_keeps_highest_scorer(self):
        coords = np.array([[0., 0., 0.], [0.5, 0., 0.]])
        scores = np.array([0.1, 0.9])
        mask = nnana.nms_by_distance(coords, scores, distance=1.0, keep_greater=True)
        assert mask[1] and not mask[0]

    def test_keep_lesser_keeps_lowest_scorer(self):
        coords = np.array([[0., 0., 0.], [0.5, 0., 0.]])
        scores = np.array([0.1, 0.9])
        mask = nnana.nms_by_distance(coords, scores, distance=1.0, keep_greater=False)
        assert mask[0] and not mask[1]

    def test_distant_points_both_kept(self):
        coords = np.array([[0., 0., 0.], [100., 0., 0.]])
        scores = np.array([0.5, 0.5])
        mask = nnana.nms_by_distance(coords, scores, distance=1.0)
        assert np.all(mask)

    def test_single_point_always_kept(self):
        coords = np.array([[0., 0., 0.]])
        mask = nnana.nms_by_distance(coords, np.array([1.0]), distance=1.0)
        assert mask[0]


class TestCenteredNnCoords:
    def test_output_shape(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]])
        qp_idx = np.array([0, 1])
        nn_idx = np.array([[1], [2]])
        out = nnana.centered_nn_coords(coords, qp_idx, coords, nn_idx)
        assert out.shape == (2, 3)

    def test_direction(self):
        coords = np.array([[0., 0., 0.], [3., 4., 0.]])
        qp_idx = np.array([0])
        nn_idx = np.array([[1]])
        out = nnana.centered_nn_coords(coords, qp_idx, coords, nn_idx)
        np.testing.assert_allclose(out[0], [3., 4., 0.])

    def test_pixel_size_scales_output(self):
        coords = np.array([[0., 0., 0.], [1., 0., 0.]])
        qp_idx = np.array([0])
        nn_idx = np.array([[1]])
        out1 = nnana.centered_nn_coords(coords, qp_idx, coords, nn_idx, pixel_size=1.0)
        out2 = nnana.centered_nn_coords(coords, qp_idx, coords, nn_idx, pixel_size=2.5)
        np.testing.assert_allclose(out2, out1 * 2.5)


class TestRotatedNnCoords:
    def test_output_shape_preserved(self):
        centered = np.random.rand(5, 3)
        angles = np.zeros((5, 3))
        out = nnana.rotated_nn_coords(centered, angles)
        assert out.shape == centered.shape

    def test_zero_angles_identity(self):
        centered = np.random.rand(4, 3)
        angles = np.zeros((4, 3))
        out = nnana.rotated_nn_coords(centered, angles)
        np.testing.assert_allclose(out, centered, atol=1e-10)

    def test_output_finite(self):
        centered = np.random.rand(10, 3)
        angles = np.random.uniform(-180, 180, (10, 3))
        out = nnana.rotated_nn_coords(centered, angles)
        assert np.all(np.isfinite(out))


class TestAngularDistances:
    @pytest.mark.parametrize("rotation_type", [
        "angular_distance", "cone_distance", "in_plane_distance"
    ])
    def test_returns_array_of_correct_length(self, rotation_type):
        angles = np.zeros((5, 3))
        result = nnana.angular_distances(angles, angles, rotation_type=rotation_type)
        assert len(np.atleast_1d(result)) == 5

    def test_identical_angles_zero_distance(self):
        angles = np.array([[10., 20., 30.]] * 4)
        dist = nnana.angular_distances(angles, angles, rotation_type="angular_distance")
        np.testing.assert_allclose(np.atleast_1d(dist), 0.0, atol=1e-10)


class TestRelativeRotations:
    def test_returns_rotation_object(self):
        from scipy.spatial.transform import Rotation
        angles = np.zeros((3, 3))
        rel = nnana.relative_rotations(angles, angles)
        assert isinstance(rel, Rotation)

    def test_identity_to_identity_gives_identity(self):
        angles = np.zeros((4, 3))
        rel = nnana.relative_rotations(angles, angles)
        for mat in rel.as_matrix():
            np.testing.assert_allclose(mat, np.eye(3), atol=1e-10)

    def test_length_matches_input(self):
        angles = np.random.uniform(-180, 180, (7, 3))
        rel = nnana.relative_rotations(angles, angles)
        assert len(rel) == 7


class TestRotationsToUnitVectors:
    def test_returns_two_arrays(self):
        from scipy.spatial.transform import Rotation
        rot = Rotation.identity(5)
        pts, eul = nnana.rotations_to_unit_vectors(rot)
        assert pts.shape == (5, 3) and eul.shape == (5, 3)

    def test_unit_vectors_have_unit_norm(self):
        from scipy.spatial.transform import Rotation
        rot = Rotation.random(10, random_state=42)
        pts, _ = nnana.rotations_to_unit_vectors(rot)
        norms = np.linalg.norm(pts, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-10)

    def test_euler_angles_shape(self):
        from scipy.spatial.transform import Rotation
        rot = Rotation.random(6, random_state=0)
        _, eul = nnana.rotations_to_unit_vectors(rot)
        assert eul.shape == (6, 3)


# =============================================================================
# NearestNeighbors class
# =============================================================================


class TestNearestNeighbors:
    def test_init_no_args_gives_none_df(self):
        nn = nnana.NearestNeighbors()
        assert nn.df is None
        assert nn.features is None

    def test_init_single_motl_closest_dist(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="closest_dist", type_param=1)
        assert nn.df is not None
        assert "nn_dist" in nn.df.columns

    def test_init_single_motl_radius(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="radius", type_param=100)
        assert nn.df is not None
        assert "nn_dist" not in nn.df.columns

    def test_invalid_nn_type_raises(self, motl):
        with pytest.raises(ValueError):
            nnana.NearestNeighbors(motl, nn_type="invalid_type")

    def test_get_normalized_coord_shape(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="closest_dist", type_param=1)
        norm = nn.get_normalized_coord()
        assert norm.shape[1] == 3
        assert norm.shape[0] == len(nn.df)

    def test_get_rotated_coord_shape(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="closest_dist", type_param=1)
        rot = nn.get_rotated_coord()
        assert rot.shape[1] == 3

    def test_to_stats_dataframe_has_expected_columns(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="closest_dist", type_param=1)
        df = nn.to_stats_dataframe()
        for col in ("distance", "angular_distance", "coord_x", "coord_y", "coord_z"):
            assert col in df.columns

    def test_to_stats_dataframe_radius_raises(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="radius", type_param=100)
        with pytest.raises(ValueError):
            nn.to_stats_dataframe()

    def test_get_unique_values_nonempty(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="closest_dist", type_param=1)
        assert len(nn.get_unique_values()) > 0

    def test_drop_symmetric_duplicates_reduces_rows(self, motl):
        nn = nnana.NearestNeighbors(motl, nn_type="closest_dist", type_param=1)
        deduped = nn.drop_symmetric_duplicates()
        assert len(deduped) <= len(nn.df)


# =============================================================================
# filter_nn_radial_stats
# =============================================================================


class TestFilterNnRadialStats:
    _STATS = pd.DataFrame({
        "coord_rx": [0.0, 1.0, -1.0],
        "coord_ry": [0.0, 0.0,  0.0],
        "coord_rz": [0.0, 0.0,  0.0],
        "value":    [1,   2,    3],
    })

    def test_all_kept_ones_mask(self):
        mask = np.ones((10, 10, 10))
        result = nnana.filter_nn_radial_stats(self._STATS, mask)
        assert len(result) == 3

    def test_all_dropped_zeros_mask(self):
        mask = np.zeros((10, 10, 10))
        result = nnana.filter_nn_radial_stats(self._STATS, mask)
        assert len(result) == 0

    def test_temp_integer_columns_removed(self):
        mask = np.ones((10, 10, 10))
        result = nnana.filter_nn_radial_stats(self._STATS, mask)
        for col in ("x_int", "y_int", "z_int"):
            assert col not in result.columns

    def test_out_of_bounds_dropped(self):
        stats = pd.DataFrame({
            "coord_rx": [1000.0],
            "coord_ry": [1000.0],
            "coord_rz": [1000.0],
        })
        mask = np.ones((10, 10, 10))
        result = nnana.filter_nn_radial_stats(stats, mask)
        assert len(result) == 0

    def test_index_reset(self):
        mask = np.ones((10, 10, 10))
        result = nnana.filter_nn_radial_stats(self._STATS, mask)
        assert list(result.index) == list(range(len(result)))
