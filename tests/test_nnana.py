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

    pd.testing.assert_frame_equal(df, exp_df, atol=1e-10, check_dtype=False)


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


# =============================================================================
# Direct coverage for module-level wrappers + NearestNeighbors lazy accessors
# =============================================================================


def _two_particle_motl():
    """Tiny 2-particle Motl, same tomo, useful for k=1 NN smoke tests."""
    df = cryomotl.Motl.create_empty_motl_df()
    rows = [
        {"subtomo_id": 1, "tomo_id": 1, "object_id": 1, "class": 1,
         "x": 10.0, "y": 0.0, "z": 0.0, "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
         "phi": 0.0, "theta": 0.0, "psi": 0.0, "score": 1.0},
        {"subtomo_id": 2, "tomo_id": 1, "object_id": 1, "class": 1,
         "x": 13.0, "y": 0.0, "z": 0.0, "shift_x": 0.0, "shift_y": 0.0, "shift_z": 0.0,
         "phi": 30.0, "theta": 0.0, "psi": 0.0, "score": 1.0},
    ]
    return cryomotl.Motl(motl_df=pd.concat([df, pd.DataFrame(rows)], ignore_index=True))


# ── module-level layer-2 wrappers ────────────────────────────────────────────


def test_get_feature_nn_indices_returns_4_tuple():
    """``get_feature_nn_indices`` is a thin wrapper that returns (qp_idx, nn_idx, nn_dist, k_eff)."""
    m = _two_particle_motl()
    qp_idx, nn_idx, nn_dist, k_eff = nnana.get_feature_nn_indices(m, m, nn_number=1, remove_qp=True)
    assert qp_idx.shape == (2,)
    assert nn_idx.shape == (2, 1)
    assert nn_dist.shape == (2, 1)
    assert k_eff == 1


def test_get_feature_nn_within_radius_returns_2_tuple():
    """``get_feature_nn_within_radius`` returns (qp_idx_list, nn_idx_list)."""
    m = _two_particle_motl()
    qp_idx, nn_idx = nnana.get_feature_nn_within_radius(m, m, radius=5.0, remove_qp=True)
    assert isinstance(qp_idx, list) and isinstance(nn_idx, list)
    assert len(qp_idx) == len(nn_idx)


def test_get_nn_within_distance_returns_self_nn_pairs():
    """Self-NN within a radius returns (center_idx, nn_idx_list)."""
    m = _two_particle_motl()
    center_idx, nn_idx = nnana.get_nn_within_distance(m, radius=5.0)
    assert hasattr(center_idx, "__len__")
    assert len(center_idx) == len(nn_idx)


def test_get_nn_distances_returns_six_arrays():
    """``get_nn_distances`` returns the 6-tuple (centered, rotated, nn_dist, ang, qp_id, nn_id)."""
    m = _two_particle_motl()
    centered, rotated, nn_dist, ang, qp_id, nn_id = nnana.get_nn_distances(
        m, m, pixel_size=2.0, nn_number=1, paired=False, remove_duplicates=False,
    )
    assert centered.shape[1] == 3
    assert rotated.shape == centered.shape
    assert nn_dist.shape[0] == centered.shape[0]
    assert ang.shape[0] == centered.shape[0]
    assert qp_id.shape == nn_id.shape


def test_get_nn_rotations_returns_unit_vectors_and_angles():
    """``get_nn_rotations`` returns (points_on_sphere, euler_angles)."""
    m = _two_particle_motl()
    points, eulers = nnana.get_nn_rotations(m, m, nn_number=1)
    assert points.shape[1] == 3
    assert eulers.shape[1] == 3
    np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-6)


def test_assign_class_by_nn_labels_matched_particles():
    """A motl matched to itself gets ``starting_class`` for every particle."""
    m = _two_particle_motl()
    out = nnana.assign_class_by_nn(m, [m], starting_class=7, dist_threshold=1.0)
    assert isinstance(out, cryomotl.Motl)
    # Every particle in `m` was matched by itself → class label is `starting_class`.
    assert set(out.df["class"].unique()) == {7}


def test_trace_chains_links_two_particles():
    """A 2-particle motl ⇒ one chain of length 2 within the threshold."""
    m = _two_particle_motl()
    out = nnana.trace_chains(m, max_distance=5.0, min_distance=0.0)
    assert isinstance(out, cryomotl.Motl)
    # ``object_id`` is the chain identifier; both rows must share it.
    assert out.df["object_id"].nunique() == 1
    assert len(out.df) == 2


def test_trace_chains_no_max_distance_raises():
    """``max_distance=None`` is the only required arg; omitting it raises."""
    m = _two_particle_motl()
    with pytest.raises(ValueError):
        nnana.trace_chains(m)


# ── NearestNeighbors lazy accessors ──────────────────────────────────────────


def _nn_from_two_particle_motl():
    return nnana.NearestNeighbors(_two_particle_motl(), nn_type="closest_dist", type_param=1)


def test_nn_get_qp_rotations_returns_rotation_object():
    """``get_qp_rotations`` returns a scipy Rotation matching the qp-angle stack."""
    from scipy.spatial.transform import Rotation as srot
    nn = _nn_from_two_particle_motl()
    rot = nn.get_qp_rotations()
    assert isinstance(rot, srot)


def test_nn_get_nn_rotations_returns_rotation_object():
    from scipy.spatial.transform import Rotation as srot
    nn = _nn_from_two_particle_motl()
    assert isinstance(nn.get_nn_rotations(), srot)


def test_nn_get_relative_rotations_returns_rotation_object():
    from scipy.spatial.transform import Rotation as srot
    nn = _nn_from_two_particle_motl()
    assert isinstance(nn.get_relative_rotations(), srot)


def test_nn_get_angular_distances_returns_array():
    nn = _nn_from_two_particle_motl()
    d = nn.get_angular_distances(rotation_type="angular_distance")
    assert hasattr(d, "__len__")


def test_nn_get_nn_subset_filters_by_motl_id_and_feature():
    """Subsetting by an existing (motl_id, tomo_id) pair returns a non-empty NearestNeighbors."""
    nn = _nn_from_two_particle_motl()
    sub = nn.get_nn_subset(motl_id_values=1, column_values=1)
    assert isinstance(sub, nnana.NearestNeighbors)
    assert sub.df is not None and len(sub.df) > 0


# =============================================================================
# Helpers for new tests
# =============================================================================


def _four_particle_motl_two_objects():
    """4 particles: 2 in object 1, 2 in object 2, all tomo_id=1.
    Particle distances: p1-p2=3 (same obj), p1-p3=4 (diff obj), p2-p4=4 (diff obj).
    """
    df = cryomotl.Motl.create_empty_motl_df()
    rows = [
        {"subtomo_id": 1, "tomo_id": 1, "object_id": 1, "class": 1,
         "x": 0., "y": 0., "z": 0., "shift_x": 0., "shift_y": 0., "shift_z": 0.,
         "phi": 0., "theta": 0., "psi": 0., "score": 1.},
        {"subtomo_id": 2, "tomo_id": 1, "object_id": 1, "class": 1,
         "x": 3., "y": 0., "z": 0., "shift_x": 0., "shift_y": 0., "shift_z": 0.,
         "phi": 10., "theta": 0., "psi": 0., "score": 1.},
        {"subtomo_id": 3, "tomo_id": 1, "object_id": 2, "class": 1,
         "x": 4., "y": 0., "z": 0., "shift_x": 0., "shift_y": 0., "shift_z": 0.,
         "phi": 20., "theta": 0., "psi": 0., "score": 1.},
        {"subtomo_id": 4, "tomo_id": 1, "object_id": 2, "class": 1,
         "x": 7., "y": 0., "z": 0., "shift_x": 0., "shift_y": 0., "shift_z": 0.,
         "phi": 30., "theta": 0., "psi": 0., "score": 1.},
    ]
    return cryomotl.Motl(motl_df=pd.concat([df, pd.DataFrame(rows)], ignore_index=True))


# =============================================================================
# TestDtypes
# =============================================================================


class TestDtypes:
    """int32 ids, float32 geometry after construction."""

    def test_int_columns_are_int32(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        for col in ("motl_id", "tomo_id", "qp_id", "qp_subtomo_id", "nn_id", "nn_subtomo_id"):
            assert nn.df[col].dtype == np.int32, f"{col} should be int32"

    def test_float_columns_are_float32(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        for col in ("qp_angles_phi", "qp_coord_x", "nn_angles_phi", "nn_coord_x", "nn_dist"):
            assert nn.df[col].dtype == np.float32, f"{col} should be float32"

    def test_radius_mode_no_nn_dist(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="radius", type_param=5.0)
        assert "nn_dist" not in nn.df.columns
        for col in ("qp_angles_phi", "qp_coord_x"):
            assert nn.df[col].dtype == np.float32


# =============================================================================
# TestMotls
# =============================================================================


class TestMotls:
    """self.motls stores live references."""

    def test_motls_set_after_construction(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        assert nn.motls is not None
        assert len(nn.motls) == 2  # [query, nn_motl] (same object for single-motl)

    def test_motls_none_when_no_input(self):
        nn = nnana.NearestNeighbors()
        assert nn.motls is None


# =============================================================================
# TestAddMotlColumns
# =============================================================================


class TestAddMotlColumns:
    def test_qp_object_id_matches_source(self):
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        nn.add_motl_columns("object_id")
        expected = m.df.set_index("subtomo_id")["object_id"]
        for _, row in nn.df.iterrows():
            assert row["qp_object_id"] == expected[row["qp_subtomo_id"]]

    def test_nn_object_id_matches_source(self):
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        nn.add_motl_columns("object_id")
        expected = m.df.set_index("subtomo_id")["object_id"]
        for _, row in nn.df.iterrows():
            assert row["nn_object_id"] == expected[row["nn_subtomo_id"]]

    def test_qp_side_only(self):
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        nn.add_motl_columns("object_id", sides="qp")
        assert "qp_object_id" in nn.df.columns
        assert "nn_object_id" not in nn.df.columns

    def test_raises_when_motls_is_none(self):
        nn = nnana.NearestNeighbors()
        nn.motls = None
        with pytest.raises(RuntimeError, match="No source motls"):
            nn.add_motl_columns("object_id")

    def test_raises_when_column_missing(self):
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        with pytest.raises(KeyError):
            nn.add_motl_columns("nonexistent_column")

    def test_idempotent_overwrite(self):
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        nn.add_motl_columns("object_id")
        nn.add_motl_columns("object_id")  # should not raise or duplicate columns
        assert "qp_object_id" in nn.df.columns


# =============================================================================
# TestExcludeColumnName
# =============================================================================


class TestExcludeColumnName:
    """Particles in the same object are excluded from NN candidates."""

    def test_closest_dist_excludes_same_object(self):
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(
            m, nn_type="closest_dist", type_param=1,
            exclude_column_name="object_id",
        )
        nn.add_motl_columns("object_id")
        # Every returned NN must belong to a different object than qp
        assert (nn.df["qp_object_id"] != nn.df["nn_object_id"]).all()

    def test_radius_excludes_same_object(self):
        m = _four_particle_motl_two_objects()
        # radius=5 would normally catch same-object neighbors (distance=3)
        nn = nnana.NearestNeighbors(
            m, nn_type="radius", type_param=5.0,
            exclude_column_name="object_id",
        )
        nn.add_motl_columns("object_id")
        assert (nn.df["qp_object_id"] != nn.df["nn_object_id"]).all()

    def test_same_column_warns(self):
        m = _two_particle_motl()
        with pytest.warns(UserWarning):
            nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1,
                                   exclude_column_name="tomo_id")


# =============================================================================
# TestGetNnSubsetFixed
# =============================================================================


class TestGetNnSubsetFixed:
    """get_nn_subset: feature_id bug fixed, motls propagated."""

    def test_subset_returns_nonempty(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        sub = nn.get_nn_subset(motl_id_values=1, column_values=1)
        assert sub.df is not None and len(sub.df) > 0

    def test_subset_carries_motls(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        sub = nn.get_nn_subset(motl_id_values=1, column_values=1)
        assert sub.motls is nn.motls

    def test_scalar_and_list_give_same_result(self):
        m = _two_particle_motl()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        sub1 = nn.get_nn_subset(motl_id_values=1, column_values=1)
        sub2 = nn.get_nn_subset(motl_id_values=[1], column_values=[1])
        pd.testing.assert_frame_equal(sub1.df.reset_index(drop=True),
                                      sub2.df.reset_index(drop=True))

    def test_subset_add_motl_columns_works(self):
        """add_motl_columns should work on subsets since motls are propagated."""
        m = _four_particle_motl_two_objects()
        nn = nnana.NearestNeighbors(m, nn_type="closest_dist", type_param=1)
        sub = nn.get_nn_subset(motl_id_values=1, column_values=1)
        sub.add_motl_columns("object_id")  # must not raise
        assert "qp_object_id" in sub.df.columns
