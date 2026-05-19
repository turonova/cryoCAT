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
        df = nnana.get_nn_stats(m, m, feature_id=f, nn_number=2, rotation_type=i)
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
        df = nnana.get_nn_stats_within_radius(m, nn_radius=r, feature=f)
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
    "feature_id, expected_res",
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
def test_get_nn_within_radius(motl, feature_id, expected_res):

    # expected_res = np.asarray(
    #    [7, 4, 4, 4, 4, 4, 4, 4, 4, 4, 2, 3, 3, 4, 4, 4, 4, 4, 4, 2, 3, 3, 3, 3, 2, 4, 5, 6, 6, 6, 7]
    # )
    res = nnana.get_nn_within_radius(motl, motl, nn_radius=0.6, column_name=feature_id)

    np.testing.assert_array_equal(expected_res, res)


@pytest.mark.parametrize(
    "feature_id, rotation_type",
    [
        ("tomo_id", "angular_distance"),
        ("tomo_id", "cone_distance"),
        ("tomo_id", "in_plane_distance"),
        ("object_id", "angular_distance"),
        ("object_id", "cone_distance"),
        ("object_id", "in_plane_distance"),
    ],
)
def test_get_nn_stats(motl, feature_id, rotation_type):

    df = nnana.get_nn_stats(motl, motl, column_name=feature_id, nn_number=2, rotation_type=rotation_type)
    exp_df = pd.read_csv(f"./tests/test_data/nnana_data/nn_{feature_id}_{rotation_type}.csv")
    if feature_id == "object_id":
        df = df.sort_values(by="distance")
    df = df.iloc[31:]
    df = df.sort_values(by="subtomo_idx")
    df = df.round(4)
    df.reset_index(inplace=True, drop=True)

    pd.testing.assert_frame_equal(df, exp_df, atol=1e-10)


@pytest.mark.parametrize(
    "feature_id, radius",
    [
        ("tomo_id", 0.1),
        ("tomo_id", 0.51),
        ("tomo_id", 1.0),
        ("object_id", 0.1),
        ("object_id", 0.51),
        ("object_id", 1.0),
    ],
)
def test_get_nn_stats_within_radius(motl, feature_id, radius):

    df = nnana.get_nn_stats_within_radius(motl, nn_radius=radius, column_name=feature_id)
    exp_df = pd.read_csv(f"./tests/test_data/nnana_data/nn_stats_radius_{feature_id}_{str(radius)}.csv")
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
