import numpy as np
import pandas as pd
import pytest
from sklearn.decomposition import PCA

from cryocat.analysis import clustering
from cryocat.analysis.tango import Descriptor, TwistDescriptor, CustomDescriptor
from cryocat.core.cryomotl import Motl


def _make_desc_df(n=40, seed=0):
    """Synthetic descriptor DataFrame: qp_id + three feature columns."""
    rng = np.random.default_rng(seed)
    base = rng.standard_normal(n)
    return pd.DataFrame({
        "qp_id": np.arange(n, dtype=float),
        "feat_a": base + rng.standard_normal(n) * 0.1,
        "feat_b": -base + rng.standard_normal(n) * 0.1,
        "feat_c": rng.standard_normal(n),
    })


def _make_descriptor(n=30):
    d = Descriptor()
    d.desc = _make_desc_df(n=n)
    d.pca_components = 2
    return d, d.desc.copy()


# ---------------------------------------------------------------------------
# drop_nans
# ---------------------------------------------------------------------------

def test_drop_nans_row():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result = clustering.drop_nans(df, "row")
    assert len(result) == 2
    assert result["a"].notna().all()


def test_drop_nans_column():
    df = pd.DataFrame({"a": [1.0, np.nan, 3.0], "b": [4.0, 5.0, 6.0]})
    result = clustering.drop_nans(df, "column")
    assert list(result.columns) == ["b"]


def test_drop_nans_invalid_axis():
    with pytest.raises(ValueError):
        clustering.drop_nans(pd.DataFrame({"a": [1]}), "diagonal")


# ---------------------------------------------------------------------------
# filter_feature_columns
# ---------------------------------------------------------------------------

def test_filter_all_returns_full_df():
    df = _make_desc_df()
    result = clustering.filter_feature_columns(df, feature_ids="all")
    assert list(result.columns) == list(df.columns)


def test_filter_single_feature():
    df = _make_desc_df()
    result = clustering.filter_feature_columns(df, feature_ids="feat_a")
    assert set(result.columns) == {"feat_a", "qp_id"}


def test_filter_list_of_features():
    df = _make_desc_df()
    result = clustering.filter_feature_columns(df, feature_ids=["feat_a", "feat_b"])
    assert set(result.columns) == {"feat_a", "feat_b", "qp_id"}


def test_filter_invalid_string_raises():
    df = _make_desc_df()
    with pytest.raises(ValueError):
        clustering.filter_feature_columns(df, feature_ids="nonexistent")


def test_filter_list_all_invalid_raises():
    df = _make_desc_df()
    with pytest.raises(ValueError):
        clustering.filter_feature_columns(df, feature_ids=["nonexistent"])


def test_filter_non_string_non_list_raises():
    df = _make_desc_df()
    with pytest.raises(ValueError):
        clustering.filter_feature_columns(df, feature_ids=42)


# ---------------------------------------------------------------------------
# pca_feature_importance
# ---------------------------------------------------------------------------

def test_pca_feature_importance_dominant_feature_ranks_first():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 3))
    X[:, 0] *= 10  # feat0 dominates
    pca = PCA(n_components=1).fit(X)
    imp = clustering.pca_feature_importance(pca, ["dominant", "b", "c"])
    assert imp.index[0] == "dominant"


def test_pca_feature_importance_scores_sum_to_n_components():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((30, 4))
    pca = PCA(n_components=2).fit(X)
    imp = clustering.pca_feature_importance(pca, ["a", "b", "c", "d"])
    np.testing.assert_allclose(imp.sum(), 2.0, atol=1e-6)


def test_pca_feature_importance_sorted_descending():
    rng = np.random.default_rng(7)
    X = rng.standard_normal((40, 3))
    pca = PCA(n_components=2).fit(X)
    imp = clustering.pca_feature_importance(pca, ["x", "y", "z"])
    assert list(imp.values) == sorted(imp.values, reverse=True)


# ---------------------------------------------------------------------------
# compute_pca
# ---------------------------------------------------------------------------

def test_compute_pca_output_shape():
    df = _make_desc_df(n=30)
    pca_df, qp_ids = clustering.compute_pca(df, n_components=2)
    assert pca_df.shape == (30, 2)
    assert len(qp_ids) == 30


def test_compute_pca_qp_ids_match_input():
    df = _make_desc_df(n=20)
    _, qp_ids = clustering.compute_pca(df, n_components=1)
    np.testing.assert_array_equal(qp_ids, df["qp_id"].to_numpy())


def test_compute_pca_drops_nn_id():
    df = _make_desc_df()
    df["nn_id"] = np.arange(len(df), dtype=float)
    pca_df, _ = clustering.compute_pca(df, n_components=1)
    assert "nn_id" not in pca_df.columns
    assert "qp_id" not in pca_df.columns


def test_compute_pca_n_components_clamped():
    df = _make_desc_df(n=10)  # 3 feature cols
    pca_df, _ = clustering.compute_pca(df, n_components=100)
    assert pca_df.shape[1] <= 3


# ---------------------------------------------------------------------------
# kmeans_cluster
# ---------------------------------------------------------------------------

def test_kmeans_cluster_returns_expected_columns():
    df = _make_desc_df(n=20)
    result = clustering.kmeans_cluster(df, n_clusters=2)
    assert "cluster" in result.columns
    assert "qp_id" in result.columns


def test_kmeans_cluster_two_blobs():
    rng = np.random.default_rng(7)
    n = 40
    X = np.vstack([
        rng.standard_normal((n // 2, 2)),
        rng.standard_normal((n // 2, 2)) + 100.0,
    ])
    df = pd.DataFrame({"qp_id": np.arange(n, dtype=float), "x": X[:, 0], "y": X[:, 1]})
    result = clustering.kmeans_cluster(df, n_clusters=2)
    assert set(result["cluster"].unique()) == {0, 1}


def test_kmeans_cluster_with_pca_dict():
    df = _make_desc_df(n=30)
    result = clustering.kmeans_cluster(df, n_clusters=2, pca_dict={"n_components": 1})
    assert "cluster" in result.columns
    assert len(result) == 30


def test_kmeans_cluster_no_scale():
    df = _make_desc_df(n=20)
    result = clustering.kmeans_cluster(df, n_clusters=2, scale_data=False)
    assert "cluster" in result.columns


# ---------------------------------------------------------------------------
# connected_component_clusters
# ---------------------------------------------------------------------------

def test_connected_components_returns_one():
    qp = [1, 2, 3, 4, 5, 6]
    nn = [2, 3, 1, 5, 6, 4]
    result = clustering.connected_component_clusters(qp, nn, num_components=1)
    assert len(result) == 1


def test_connected_components_clamped_to_graph_count():
    qp = [1, 2, 3, 4, 5, 6]
    nn = [2, 3, 1, 5, 6, 4]
    result = clustering.connected_component_clusters(qp, nn, num_components=100)
    assert len(result) == 2


def test_connected_components_min_size():
    # Three components: {1,2,3}, {4,5,6}, {7} (self-loop, size 1)
    qp = [1, 2, 3, 4, 5, 6, 7]
    nn = [2, 3, 1, 5, 6, 4, 7]
    result = clustering.connected_component_clusters(qp, nn, min_size=3)
    assert len(result) == 2
    for g in result:
        assert len(g.nodes) >= 3


def test_connected_components_non_int_min_size_falls_back_to_num():
    qp = [1, 2, 3, 4, 5, 6]
    nn = [2, 3, 1, 5, 6, 4]
    result = clustering.connected_component_clusters(qp, nn, num_components=1, min_size="all")
    assert len(result) == 1


# ---------------------------------------------------------------------------
# Descriptor wrapper equivalence
# ---------------------------------------------------------------------------

def test_wrapper_remove_nans_equivalence():
    df = _make_desc_df()
    df.iloc[0, 1] = np.nan
    d = Descriptor()
    assert Descriptor.remove_nans(df, "row").equals(clustering.drop_nans(df, "row"))


def test_wrapper_filter_features_equivalence():
    df = _make_desc_df()
    d = Descriptor()
    wrap = d.filter_features(df, feature_ids="feat_a")
    pure = clustering.filter_feature_columns(df, feature_ids="feat_a", id_columns=("qp_id",))
    pd.testing.assert_frame_equal(wrap, pure)


def test_wrapper_compute_pca_equivalence():
    d, df = _make_descriptor()
    wrap_pca, wrap_ids = d.compute_pca(pca_components=2)
    pure_pca, pure_ids = clustering.compute_pca(df, n_components=2)
    pd.testing.assert_frame_equal(wrap_pca, pure_pca)
    np.testing.assert_array_equal(wrap_ids, pure_ids)


def test_wrapper_kmeans_columns_present():
    d, _ = _make_descriptor(n=40)
    result = d.k_means_clustering(n_clusters=2, scale_data=False)
    assert "cluster" in result.columns
    assert "qp_id" in result.columns


def test_wrapper_proximity_clustering_returns_list():
    d = Descriptor()
    d.df = pd.DataFrame({
        "qp_id": [1, 2, 3, 4, 5, 6],
        "nn_id": [2, 3, 1, 5, 6, 4],
    })
    result = d.proximity_clustering(num_connected_components=2)
    assert isinstance(result, list)
    assert len(result) == 2


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_blobs_df(n_per_blob=40, noise=5, sep=20.0, seed=0):
    """Three well-separated blobs plus optional isolated noise points.

    Noise points are placed on a circle of radius ``sep * 1.5`` centred on
    the centroid of the three blob centres, evenly spaced in angle.  This
    guarantees each noise point is (a) far from every blob and (b) far from
    every other noise point, so DBSCAN cannot recruit them into a cluster
    regardless of ``eps`` or ``min_samples``.
    """
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [sep, 0], [0, sep]])
    parts = []
    for cx, cy in centers:
        pts = rng.standard_normal((n_per_blob, 2)) + np.array([cx, cy])
        parts.append(pts)
    blobs = np.vstack(parts)
    if noise > 0:
        angles = np.linspace(0, 2 * np.pi, noise, endpoint=False)
        radius = sep * 1.5
        centroid = centers.mean(axis=0)
        noise_pts = np.column_stack([
            radius * np.cos(angles) + centroid[0],
            radius * np.sin(angles) + centroid[1],
        ])
        data = np.vstack([blobs, noise_pts])
    else:
        data = blobs
    n = len(data)
    return pd.DataFrame({
        "qp_id": np.arange(n, dtype=float),
        "x": data[:, 0],
        "y": data[:, 1],
    })


# ---------------------------------------------------------------------------
# dbscan_cluster
# ---------------------------------------------------------------------------

def test_dbscan_cluster_columns_match_kmeans():
    df = _make_desc_df(n=30)
    k_result = clustering.kmeans_cluster(df, n_clusters=2)
    d_result = clustering.dbscan_cluster(df, eps=5.0)
    assert set(k_result.columns) == set(d_result.columns)


def test_dbscan_cluster_three_blobs():
    df = _make_blobs_df(n_per_blob=40, noise=5, sep=20.0)
    result = clustering.dbscan_cluster(df, eps=1.5, min_samples=3)
    real_clusters = [c for c in result["cluster"].unique() if c != -1]
    assert len(real_clusters) == 3
    assert -1 in result["cluster"].values


def test_dbscan_cluster_large_eps_one_cluster():
    df = _make_blobs_df()
    result = clustering.dbscan_cluster(df, eps=100.0, min_samples=2)
    real_clusters = set(result["cluster"].unique()) - {-1}
    assert len(real_clusters) == 1
    assert -1 not in result["cluster"].values


def test_dbscan_cluster_tiny_eps_all_noise():
    df = _make_blobs_df()
    result = clustering.dbscan_cluster(df, eps=1e-10, min_samples=2)
    assert set(result["cluster"].unique()) == {-1}


def test_dbscan_cluster_scale_invariant():
    df = _make_blobs_df(n_per_blob=40, noise=0, sep=20.0)
    result_orig = clustering.dbscan_cluster(df, eps=1.5, min_samples=3)

    df_scaled = df.copy()
    df_scaled["x"] = df_scaled["x"] * 100  # rescale one feature
    result_scaled = clustering.dbscan_cluster(df_scaled, eps=1.5, min_samples=3)

    np.testing.assert_array_equal(
        result_orig["cluster"].values, result_scaled["cluster"].values,
        err_msg="Labels changed after rescaling — standardisation missing",
    )


# ---------------------------------------------------------------------------
# k_distance_curve
# ---------------------------------------------------------------------------

def test_k_distance_curve_monotonic():
    df = _make_blobs_df()
    curve = clustering.k_distance_curve(df, min_samples=5)
    assert np.all(np.diff(curve) >= 0), "k-distance curve must be non-decreasing"


def test_k_distance_curve_length():
    df = _make_blobs_df(n_per_blob=20, noise=3)
    curve = clustering.k_distance_curve(df, min_samples=4)
    assert len(curve) == len(df)


# ---------------------------------------------------------------------------
# screen_feature_columns
# ---------------------------------------------------------------------------

def _make_screen_df():
    n = 201
    rng = np.random.default_rng(42)
    return pd.DataFrame({
        "good":        rng.standard_normal(n),
        "constant":    np.ones(n),
        "all_nan":     np.full(n, np.nan),
        "all_zero":    np.zeros(n),
        "non_numeric": ["a"] * n,
        "near_const":  np.array([1.0] * (n - 1) + [2.0]),   # 2/201 ≈ 0.995 % < 1 %
    })


def test_screen_constant_column():
    excluded, _ = clustering.screen_feature_columns(_make_screen_df())
    assert excluded.get("constant") == "constant"


def test_screen_all_nan_column():
    excluded, _ = clustering.screen_feature_columns(_make_screen_df())
    assert excluded.get("all_nan") == "all NaN"


def test_screen_all_zero_column():
    excluded, _ = clustering.screen_feature_columns(_make_screen_df())
    assert excluded.get("all_zero") == "all zero"


def test_screen_non_numeric_column():
    excluded, _ = clustering.screen_feature_columns(_make_screen_df())
    assert excluded.get("non_numeric") == "non-numeric"


def test_screen_good_column_not_excluded():
    excluded, warnings = clustering.screen_feature_columns(_make_screen_df())
    assert "good" not in excluded
    assert "good" not in warnings


def test_screen_near_constant_is_warning_not_exclusion():
    excluded, warnings = clustering.screen_feature_columns(_make_screen_df())
    assert "near_const" not in excluded
    assert warnings.get("near_const") == "near-constant"


def test_screen_near_constant_at_boundary_not_warned():
    # 2 distinct out of 200 = exactly 1 % — the boundary of the < 1 % rule.
    # The docstring specifies this is NOT warned (strict less-than).
    n = 200
    df = pd.DataFrame({"col": np.array([1.0] * (n - 1) + [2.0])})
    _, warnings = clustering.screen_feature_columns(df)
    assert "col" not in warnings


# ---------------------------------------------------------------------------
# default_feature_columns
# ---------------------------------------------------------------------------

def _make_twist_df():
    n = 10
    rng = np.random.default_rng(0)
    cols = TwistDescriptor.get_all_feature_ids()
    data = {c: rng.standard_normal(n) for c in cols}
    return pd.DataFrame(data)


def test_default_feature_columns_twist():
    df = _make_twist_df()
    result = TwistDescriptor.default_feature_columns(df)
    assert result == TwistDescriptor.get_all_feature_ids()


def test_default_feature_columns_motl_excludes_identifiers():
    n = 5
    rng = np.random.default_rng(1)
    motl_data = {c: rng.standard_normal(n) for c in Motl.motl_columns}
    df = pd.DataFrame(motl_data)
    result = Motl.default_feature_columns(df)
    _excluded = {"subtomo_id", "tomo_id", "object_id", "class",
                 "geom1", "geom2", "geom3", "geom4", "geom5"}
    for col in _excluded:
        assert col not in result, f"Expected {col!r} to be excluded from motl defaults"
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Regression: _prepare_features identical for both callers
# ---------------------------------------------------------------------------

def test_prepare_features_identical_for_both():
    df = _make_desc_df(n=30)
    feat_km, ids_km = clustering._prepare_features(
        df, feature_ids="all", id_columns=("qp_id",), nan_drop="row", pca_dict=None
    )
    feat_db, ids_db = clustering._prepare_features(
        df, feature_ids="all", id_columns=("qp_id",), nan_drop="row", pca_dict=None
    )
    pd.testing.assert_frame_equal(feat_km, feat_db)
    np.testing.assert_array_equal(ids_km, ids_db)


# ---------------------------------------------------------------------------
# Descriptor wrapper for DBSCAN
# ---------------------------------------------------------------------------

def test_wrapper_dbscan_columns_present():
    d, _ = _make_descriptor(n=40)
    result = d.dbscan_clustering(eps=5.0, min_samples=3)
    assert "cluster" in result.columns
    assert "qp_id" in result.columns
