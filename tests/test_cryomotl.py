import tempfile
import mrcfile
import copy
import numpy as np
import pandas as pd
import pytest

from cryocat import ioutils
from cryocat.cryomotl import Motl
from cryocat.exceptions import UserInputError
from scipy.spatial.transform import Rotation as rot
@pytest.fixture
def motl():
    motl = Motl.load("./test_data/au_1.em")
    return motl


@pytest.mark.parametrize("feature_id", ["missing"])
def test_get_feature_not_existing(motl, feature_id):
    with pytest.raises(UserInputError):
        motl.get_feature(feature_id)

def test_get_feature(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with valid feature_id (single string)
    feature_values = motl.get_feature("tomo_id")
    expected_values = sample_motl_data1["tomo_id"].values.reshape(-1, 1) # Reshape to 2D
    assert np.array_equal(feature_values, expected_values)

    # Test with valid feature_id (list of strings)
    feature_values_list = motl.get_feature(["tomo_id", "x"])
    expected_values_list = sample_motl_data1[["tomo_id", "x"]].values
    assert np.array_equal(feature_values_list, expected_values_list)

    # Test with valid feature_id (single string, other column)
    feature_values_score = motl.get_feature("score")
    expected_values_score = sample_motl_data1["score"].values.reshape(-1, 1) # Reshape to 2D
    assert np.array_equal(feature_values_score, expected_values_score)

    # Test with invalid feature_id
    with pytest.raises(UserInputError):
        motl.get_feature("non_existent_column")

    # Test with invalid feature_id within a list
    with pytest.raises(UserInputError):
        motl.get_feature(["tomo_id", "non_existent_column"])



@pytest.mark.parametrize("feature", ["geom1"])
def test_remove_feature_existing(motl, feature):
    motl.remove_feature(feature, 1)
    assert 1 not in motl.df.loc[:, feature].values


def check_emmotl(motl):
    assert np.array_equal(
        motl.df.columns,
        [
            "score",
            "geom1",
            "geom2",
            "subtomo_id",
            "tomo_id",
            "object_id",
            "subtomo_mean",
            "x",
            "y",
            "z",
            "shift_x",
            "shift_y",
            "shift_z",
            "geom3",
            "geom4",
            "geom5",
            "phi",
            "psi",
            "theta",
            "class",
        ],
    )
    assert all(dt == "float64" for dt in motl.df.dtypes.values)


@pytest.mark.parametrize("m", ["./tests/test_data/au_1.em", "./tests/test_data/au_2.em"])
def test_read_from_emfile(m):
    motl = Motl.load(m)
    check_emmotl(motl)


@pytest.mark.parametrize("m", ["./tests/test_data/col_missing.em", "./tests/test_data/extra_col.em"])
# TODO did not manage to write out corrupted em file '/test/na_values.em', '/test/bad_values.em'
def test_read_from_emfile_wrong(m):
    with pytest.raises(UserInputError):
        Motl.load(m)


@pytest.mark.parametrize(
    "motl_list",
    [
        ["./tests/test_data/au_1.em", "./tests/test_data/au_2.em"],
        ["./tests/test_data/au_1.em", "./tests/test_data/au_1.em"],
    ],
)
def test_merge_and_renumber(motl_list):
    # TODO how should we check the 'object_id' is numbered correctly ?
    combined_len = 0
    for m in motl_list:
        combined_len += len(Motl.load(m).df)
    merged_motl = Motl.merge_and_renumber(motl_list)
    assert len(merged_motl.df) == combined_len


@pytest.mark.parametrize(
    "motl_list", ["./tests/test_data/au_1.em", [], (), "not_a_list", 42, ["./tests/test_data/au_1.em", None]]
)
def test_merge_and_renumber_wrong(motl_list):
    with pytest.raises((ValueError, UserInputError)):
        Motl.merge_and_renumber(motl_list)


@pytest.mark.parametrize("f", ["subtomo_id", "geom2"])
def test_split_by_feature(motl, f):
    motls = motl.split_by_feature(f)
    for motl in motls:
        check_emmotl(motl)


@pytest.mark.parametrize(
    "m, ref",
    [
        ("./tests/test_data/recenter/allmotl_sp_cl1_1.em", "./tests/test_data/recenter/ref1.em"),
        ("./tests/test_data/recenter/allmotl_sp_cl1_2.em", "./tests/test_data/recenter/ref2.em"),
    ],
)
def test_recenter_particles(m, ref):
    motl = Motl.load(m)
    ref_motl = Motl.load(ref)
    motl.update_coordinates()
    assert motl.df.equals(ref_motl.df)


@pytest.mark.parametrize(
    "m, shift, ref",
    [
        (
            "./tests/test_data/shift_positions/allmotl_sp_cl1_1.em",
            [1, 2, 3],
            "./tests/test_data/shift_positions/ref1.em",
        ),
        (
            "./tests/test_data/shift_positions/allmotl_sp_cl1_1.em",
            [-10, 200, 3.5],
            "./tests/test_data/shift_positions/ref2.em",
        ),
        (
            "./tests/test_data/shift_positions/allmotl_sp_cl1_1.em",
            [0, 0, 0],
            "./tests/test_data/shift_positions/ref3.em",
        ),
        (
            "./tests/test_data/shift_positions/allmotl_sp_cl1_1.em",
            [1, 1, 1],
            "./tests/test_data/shift_positions/ref4.em",
        ),
        (
            "./tests/test_data/shift_positions/allmotl_sp_cl1_5.em",
            [-10, 10, 100],
            "./tests/test_data/shift_positions/ref5.em",
        ),
    ],
)
def test_shift_positions(m, shift, ref):
    motl = Motl.load(m)
    ref_motl = Motl.load(ref)
    motl.shift_positions(shift)
    assert np.allclose(
        motl.df.iloc[0, :].values, ref_motl.df.iloc[0, :].values, rtol=1e-05, atol=1e-08, equal_nan=False
    )

#TODO
def test_emmotl2relion():
    pass
#TODO
def test_relion2emmotl():
    pass
#TODO
def test_stopgap2emmotl():
    pass
#TODO
def test_emmotl2stopgap():
    pass
#TODO
def test_relion2stopgap():
    pass
#TODO
def test_stopgap2relion():
    pass

def test_check_df_correct_format():
    motl_columns = [
        "score",
        "geom1",
        "geom2",
        "subtomo_id",
        "tomo_id",
        "object_id",
        "subtomo_mean",
        "x",
        "y",
        "z",
        "shift_x",
        "shift_y",
        "shift_z",
        "geom3",
        "geom4",
        "geom5",
        "phi",
        "psi",
        "theta",
        "class",
    ]
    motl_columns1 = [
        "geom1",
        "geom2",
        "subtomo_id",
        "tomo_id",
        "object_id",
        "subtomo_mean",
        "x",
        "y",
        "z",
        "shift_x",
        "shift_y",
        "shift_z",
        "geom3",
        "geom4",
        "geom5",
        "phi",
        "psi",
        "theta",
        "class",
    ]
    df1 = pd.DataFrame(columns=motl_columns)
    df2 = pd.DataFrame(columns=motl_columns1)
    assert Motl.check_df_correct_format(df1) == True
    assert Motl.check_df_correct_format(df2) == False

def test_create_empty_motl_df():
    df = Motl.create_empty_motl_df()
    # Check if the return type is a DataFrame
    assert isinstance(df, pd.DataFrame), "Returned object is not a pandas DataFrame"
    # Check if the DataFrame is empty
    assert df.empty, "DataFrame should be empty"
    # Check if the columns match Motl.motl_columns
    assert list(df.columns) == Motl.motl_columns, "Column names do not match expected"
    # Check if all values are initialized to 0.0
    assert (df.fillna(0) == 0.0).all().all(), "All values should be initialized to 0.0"

@pytest.fixture
def sample_data():
    sample_data = {
        "score": [0.5, 0.7, 0.9, 1.0],
        "geom1": [1, 1, 1, 1],
        "geom2": [2, 2, 2, 2],
        "subtomo_id": [1, 2, 3, 4],
        "tomo_id": [1, 1, 1, 1],
        "object_id": [100, 200, 300, 400],
        "subtomo_mean": [0.2, 0.4, 0.6, 0.8],
        "x": [10, 50, 100, 200],  # X coordinates
        "y": [20, 60, 110, 250],  # Y coordinates
        "z": [30, 70, 120, 300],  # Z coordinates
        "shift_x": [0, 0, 0, 0],
        "shift_y": [0, 0, 0, 0],
        "shift_z": [0, 0, 0, 0],
        "geom3": [3, 3, 3, 3],
        "geom4": [4, 4, 4, 4],
        "geom5": [5, 5, 5, 5],
        "phi": [0, 10, 20, 30],
        "psi": [5, 15, 25, 35],
        "theta": [10, 20, 30, 40],
        "class": [1, 2, 1, 2],
    }
    return sample_data

def test_adapt_to_trimming(sample_data):
    motl_df = pd.DataFrame(sample_data)
    motl = Motl(motl_df)  # This should pass check_df_correct_format()

    # Define trimming parameters
    trim_start = np.array([10, 20, 30])  # Start trim coordinates (x, y, z)
    trim_end = np.array([150, 150, 150])  # End trim coordinates (x, y, z)

    # Apply trimming
    motl.adapt_to_trimming(trim_start, trim_end)

    # Expected shifted coordinates
    expected_x = [1, 41, 91]  # Last particle (200) should be removed
    expected_y = [1, 41, 91]  # Last particle (250) should be removed
    expected_z = [1, 41, 91]  # Last particle (300) should be removed

    # Check that the remaining particles have the correct new coordinates
    assert list(motl.df["x"]) == expected_x, "X coordinates were not adjusted correctly"
    assert list(motl.df["y"]) == expected_y, "Y coordinates were not adjusted correctly"
    assert list(motl.df["z"]) == expected_z, "Z coordinates were not adjusted correctly"

    # Check that only 3 particles remain (one should be removed)
    assert motl.df.shape[0] == 3, "Particles outside the trimming range were not removed correctly"

    # Check that all required columns are still present after trimming
    assert set(motl.df.columns) == set(Motl.motl_columns), "Some columns are missing after trimming"

def test_apply_rotation(sample_data):
    motl_df = pd.DataFrame(sample_data)
    motl = Motl(motl_df)

    rotation = rot.from_euler('zxz', [45, 45, 45], degrees=True)

    motl.apply_rotation(rotation)

    rotated_phi = motl.df["phi"].to_numpy()
    rotated_theta = motl.df["theta"].to_numpy()
    rotated_psi = motl.df["psi"].to_numpy()

    initial_angles = np.array([sample_data["phi"], sample_data["theta"], sample_data["psi"]]).T
    initial_rotations = rot.from_euler('zxz', initial_angles, degrees=True)
    final_rotations = initial_rotations * rotation
    expected_angles = final_rotations.as_euler('zxz', degrees=True)

    expected_phi = expected_angles[:, 0]
    expected_theta = expected_angles[:, 1]
    expected_psi = expected_angles[:, 2]

    assert np.allclose(rotated_phi, expected_phi, atol=1e-6), f"Expected {expected_phi}, but got {rotated_phi.tolist()}"
    assert np.allclose(rotated_theta, expected_theta, atol=1e-6), f"Expected {expected_theta}, but got {rotated_theta.tolist()}"
    assert np.allclose(rotated_psi, expected_psi, atol=1e-6), f"Expected {expected_psi}, but got {rotated_psi.tolist()}"

@pytest.fixture
def sample_motl():
    motl_data = {
        "score": [0.5, 0.7],
        "geom1": [1, 1],
        "geom2": [2, 2],
        "subtomo_id": [1, 2],
        "tomo_id": [1, 1],
        "object_id": [100, 200],
        "subtomo_mean": [0.2, 0.4],
        "x": [10, 50],
        "y": [20, 60],
        "z": [30, 70],
        "shift_x": [0, 0],
        "shift_y": [0, 0],
        "shift_z": [0, 0],
        "geom3": [3, 3],
        "geom4": [4, 4],
        "geom5": [5, 5],
        "phi": [0, 10],
        "psi": [5, 15],
        "theta": [10, 20],
        "class": [1, 2],
    }
    return Motl(pd.DataFrame(motl_data))

@pytest.fixture
def sample_input_df():
    input_data = {
        "input_tomo": [3, 4],
        "input_subtomo": [5, 6],
        "other_col": [1000, 2000],
    }
    return pd.DataFrame(input_data)

def test_assign_column(sample_motl, sample_input_df):
    motl = sample_motl
    input_df = sample_input_df

    column_pairs = {"tomo_id": "input_tomo", "subtomo_id": "input_subtomo"}

    motl.assign_column(input_df, column_pairs)

    expected_tomo_id = input_df["input_tomo"].tolist()
    expected_subtomo_id = input_df["input_subtomo"].tolist()

    assert motl.df["tomo_id"].tolist() == expected_tomo_id
    assert motl.df["subtomo_id"].tolist() == expected_subtomo_id

def test_assign_column_missing_input_column(sample_motl, sample_input_df):
    motl = sample_motl
    input_df = sample_input_df

    column_pairs = {"tomo_id": "nonexistent_column", "subtomo_id": "input_subtomo"}

    motl.assign_column(input_df, column_pairs)

    expected_subtomo_id = input_df["input_subtomo"].tolist()

    assert motl.df["subtomo_id"].tolist() == expected_subtomo_id
    # Ensure that the tomo_id column was not changed because the input column did not exist.
    assert motl.df["tomo_id"].tolist() == [1,1]

def test_assign_column_empty_column_pairs(sample_motl, sample_input_df):
    motl = sample_motl
    input_df = sample_input_df

    column_pairs = {}

    motl.assign_column(input_df, column_pairs)

    # Ensure that the motl dataframe was not changed
    assert motl.df["tomo_id"].tolist() == [1,1]
    assert motl.df["subtomo_id"].tolist() == [1,2]



@pytest.fixture
def sample_motl_data1():
    data = {
        "tomo_id": [1, 1, 1, 2, 2, 2],
        "x": [10, 11, 20, 30, 31, 40],
        "y": [10, 11, 20, 30, 31, 40],
        "z": [10, 11, 20, 30, 31, 40],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
        "subtomo_id": [1, 2, 3, 4, 5, 6],
        "geom1": [1, 1, 1, 1, 1, 1],
        "geom2": [2, 2, 2, 2, 2, 2],
        "object_id": [100, 200, 300, 400, 500, 600],
        "subtomo_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "shift_x": [0, 0, 0, 0, 0, 0],
        "shift_y": [0, 0, 0, 0, 0, 0],
        "shift_z": [0, 0, 0, 0, 0, 0],
        "geom3": [3, 3, 3, 3, 3, 3],
        "geom4": [4, 4, 4, 4, 4, 4],
        "geom5": [5, 5, 5, 5, 5, 5],
        "phi": [0, 10, 20, 30, 40, 50],
        "psi": [5, 15, 25, 35, 45, 55],
        "theta": [10, 20, 30, 40, 50, 60],
        "class": [1, 2, 1, 2, 1, 2],
    }
    return pd.DataFrame(data)

def test_clean_by_distance_basic(sample_motl_data1):
    motl = Motl(sample_motl_data1.copy())
    motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="score")
    assert motl.df.shape[0] == 4
    assert np.all(motl.df["tomo_id"].values == np.array([1, 1, 2, 2]))

def test_clean_by_distance_grouping(sample_motl_data1):
    motl = Motl(sample_motl_data1.copy())
    motl.clean_by_distance(distance_in_voxels=20, feature_id="tomo_id", metric_id="score")
    assert motl.df.shape[0] == 2
    assert np.all(motl.df["tomo_id"].values == np.array([1, 2]))

def test_clean_by_distance_metric(sample_motl_data1):
    motl = Motl(sample_motl_data1.copy())
    motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="x")
    assert motl.df.shape[0] == 4
    assert np.all(motl.df["x"].values == np.array([11, 20, 31, 40]))

def test_clean_by_distance_empty():
    motl = Motl(Motl.create_empty_motl_df())
    motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="score")
    assert motl.df.shape[0] == 0

def test_clean_by_distance_dist_mask(sample_motl_data1):
    mask = np.zeros((5, 5, 5), dtype=bool)
    mask[2, 2, 2] = True
    mask[1, 2, 2] = True
    mask[3, 2, 2] = True
    mask[2, 1, 2] = True
    mask[2, 3, 2] = True
    mask[2, 2, 1] = True
    mask[2, 2, 3] = True

    motl = Motl(sample_motl_data1.copy())
    motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="score", dist_mask=mask)

    assert motl.df.shape[0] == 5

def test_clean_by_distance_to_points(sample_motl_data1):
    points_data = {
        "tomo_id": [1, 2],
        "x": [10, 30],
        "y": [10, 30],
        "z": [10, 30],
    }
    points_df = pd.DataFrame(points_data)

    motl = Motl(sample_motl_data1.copy())
    original_particle_count = motl.df.shape[0]

    # Clean with a radius of 2 voxels
    motl.clean_by_distance_to_points(points_df, radius_in_voxels=2)

    # Verify that the correct particles were removed
    assert motl.df.shape[0] == 2
    assert np.all(motl.df["tomo_id"].values == np.array([1, 2]))

    # Clean with a radius of 10 voxels
    motl2 = Motl(sample_motl_data1.copy())
    motl2.clean_by_distance_to_points(points_df, radius_in_voxels=10)
    assert motl2.df.shape[0] == 2
    assert np.all(motl2.df["tomo_id"].values == np.array([1, 2]))

    # Test inplace=False
    motl3 = Motl(sample_motl_data1.copy())
    cleaned_motl = motl3.clean_by_distance_to_points(points_df, radius_in_voxels=2, inplace=False)
    assert cleaned_motl.df.shape[0] == 2
    assert motl3.df.shape[0] == original_particle_count

    # Test with different feature_id
    points_data2 = {
        "class": [1, 2],
        "x": [10, 30],
        "y": [10, 30],
        "z": [10, 30],
    }
    points_df2 = pd.DataFrame(points_data2)
    motl4 = Motl(sample_motl_data1.copy())
    motl4.clean_by_distance_to_points(points_df2, radius_in_voxels=2, feature_id='class')
    assert motl4.df.shape[0] == 4

def test_clean_by_tomo_mask(sample_motl_data1):
    # Create sample tomogram masks (NumPy arrays)
    mask1 = np.ones((100, 100, 100), dtype=np.int8)
    mask1[10:20, 10:20, 10:20] = 0  # Create a masked-out region

    mask2 = np.ones((100, 100, 100), dtype=np.int8)
    mask2[30:40, 30:40, 30:40] = 0

    # Create Motl object
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Clean by tomo mask
    motl.clean_by_tomo_mask([1, 2], [mask1, mask2])

    # Verify the result
    assert motl.df.shape[0] == 2
    assert np.all(motl.df["tomo_id"].values == np.array([1, 2]))

    # Test with a single mask
    motl2 = Motl(copy.deepcopy(sample_motl_data1))
    motl2.clean_by_tomo_mask([1, 2], mask1)
    assert motl2.df.shape[0] == 4

    # Test inplace=False
    motl3 = Motl(copy.deepcopy(sample_motl_data1))
    cleaned_motl = motl3.clean_by_tomo_mask([1, 2], [mask1, mask2], inplace=False)
    assert cleaned_motl.df.shape[0] == 2
    assert motl3.df.shape[0] == 6

    # Test ValueError
    with pytest.raises(ValueError):
        motl.clean_by_tomo_mask([1, 2], [mask1])

def test_clean_by_otsu(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Clean by Otsu's thresholding (tomo_id)
    motl.clean_by_otsu("tomo_id")

    # Verify the result
    assert motl.df.shape[0] == 4
    assert np.all(motl.df["tomo_id"].values == np.array([1, 1, 2, 2]))

    # Clean by Otsu's thresholding (class)
    motl2 = Motl(copy.deepcopy(sample_motl_data1))
    motl2.clean_by_otsu("class", histogram_bin=20) #Added Histogram bin
    assert motl2.df.shape[0] == 2

    # Clean by Otsu's thresholding (histogram_bin)
    motl3 = Motl(copy.deepcopy(sample_motl_data1))
    motl3.clean_by_otsu("tomo_id", histogram_bin=20)
    assert motl3.df.shape[0] == 4

def test_check_df_type(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with a correct DataFrame
    correct_df = pd.DataFrame(sample_motl_data1.copy())
    motl.check_df_type(correct_df)
    assert motl.df.equals(correct_df.fillna(0.0).reset_index(drop=True))

    # Test with an incorrect DataFrame (will trigger conversion)
    class MockMotl(Motl):
        def convert_to_motl(self, input_df):
            # Mock conversion: add a new column
            input_df["new_column"] = 1
            self.df = input_df.copy()
            self.df.reset_index(inplace=True, drop=True)
            self.df = self.df.fillna(0.0)

    incorrect_df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_motl = MockMotl(sample_motl_data1.copy())
    mock_motl.check_df_type(incorrect_df)
    assert "new_column" in mock_motl.df.columns
    assert mock_motl.df["new_column"].equals(pd.Series([1, 1]))

    # Test that original dataframe is unchanged.
    assert sample_motl_data1.equals(pd.DataFrame(sample_motl_data1))

def test_fill(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))
    input_dict1 = {"score": 1.0}
    motl.fill(input_dict1)
    assert np.all(motl.df["score"] == 1.0)

    # Test coord update
    input_dict2 = {"coord": [[100, 200, 300]] * len(motl.df)}
    motl.fill(input_dict2)
    assert np.all(motl.df["x"] == 100)
    assert np.all(motl.df["y"] == 200)
    assert np.all(motl.df["z"] == 300)

    # Test angles update
    input_dict3 = {"angles": [[10, 20, 30]] * len(motl.df)}
    motl.fill(input_dict3)
    assert np.all(motl.df["phi"] == 10)
    assert np.all(motl.df["theta"] == 20)
    assert np.all(motl.df["psi"] == 30)

    # Test shifts update
    input_dict4 = {"shifts": [[1, 2, 3]] * len(motl.df)}
    motl.fill(input_dict4)
    assert np.all(motl.df["shift_x"] == 1)
    assert np.all(motl.df["shift_y"] == 2)
    assert np.all(motl.df["shift_z"] == 3)

    # Test NaN filling
    motl_nan = Motl(copy.deepcopy(sample_motl_data1))
    motl_nan.df.loc[0, "score"] = np.nan
    motl_nan.fill({})  # Fill with empty dict to trigger NaN filling
    assert motl_nan.df.loc[0, "score"] == 0.0

    #Test multiple updates.
    motl2 = Motl(copy.deepcopy(sample_motl_data1))
    input_dict5 = {"score": 0.5, "coord": [[10,10,10]]*len(motl2.df)}
    motl2.fill(input_dict5)
    assert np.all(motl2.df["score"] == 0.5)
    assert np.all(motl2.df["x"] == 10)

def test_get_random_subset(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with a valid number of particles
    subset_motl = motl.get_random_subset(3)
    assert isinstance(subset_motl, Motl)
    assert subset_motl.df.shape[0] == 3
    assert all(index in motl.df.index for index in subset_motl.df.index)

    # Test with the same number of particles as the original Motl
    subset_motl2 = motl.get_random_subset(len(motl.df))
    assert subset_motl2.df.shape[0] == len(motl.df)
    assert subset_motl2.df.sort_index().equals(motl.df.sort_index())

    # Test with a number of particles greater than the original Motl
    with pytest.raises(ValueError):
        motl.get_random_subset(len(motl.df) + 1)

    # Test with 0 particles.
    subset_motl3 = motl.get_random_subset(0)
    assert subset_motl3.df.shape[0] == 0

def test_assign_random_classes(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with a valid number of classes
    number_of_classes = 3
    motl.assign_random_classes(number_of_classes)

    # Verify that the "class" column is created
    assert "class" in motl.df.columns

    # Verify that the "class" column has an integer data type
    assert np.issubdtype(motl.df["class"].dtype, np.integer)

    # Verify that the "class" column has the correct range of values
    assert all(1 <= c <= number_of_classes for c in motl.df["class"])

    # Verify that the "class" column has the correct length
    assert len(motl.df["class"]) == len(motl.df)

    # Test with a different number of classes
    number_of_classes2 = 5
    motl.assign_random_classes(number_of_classes2)
    assert all(1 <= c <= number_of_classes2 for c in motl.df["class"])

    # Test with 1 class
    number_of_classes3 = 1
    motl.assign_random_classes(number_of_classes3)
    assert all(c == 1 for c in motl.df["class"])

def test_flip_handedness(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test orientation flip only
    original_theta = motl.df["theta"].copy()
    motl.flip_handedness()
    assert np.all(motl.df["theta"] == -original_theta)

    # Test position flip with single tomogram dimensions
    tomo_dims_single = pd.DataFrame({"x": [100], "y": [100], "z": [100]})

    motl2 = Motl(copy.deepcopy(sample_motl_data1))
    original_z = motl2.df["z"].copy()
    motl2.flip_handedness(tomo_dims_single)
    assert np.all(motl2.df["theta"] == -original_theta)
    assert np.all(motl2.df["z"] == 101 - original_z)

    # Test position flip with multiple tomogram dimensions
    tomo_dims_multiple = pd.DataFrame({"tomo_id": [1, 2], "x": [100, 200], "y": [100, 200], "z": [100, 200]})

    motl3 = Motl(copy.deepcopy(sample_motl_data1))
    original_z_tomo1 = motl3.df.loc[motl3.df["tomo_id"] == 1, "z"].copy()
    original_z_tomo2 = motl3.df.loc[motl3.df["tomo_id"] == 2, "z"].copy()
    motl3.flip_handedness(tomo_dims_multiple)
    assert np.all(motl3.df["theta"] == -original_theta)
    assert np.all(motl3.df.loc[motl3.df["tomo_id"] == 1, "z"] == 101 - original_z_tomo1)
    assert np.all(motl3.df.loc[motl3.df["tomo_id"] == 2, "z"] == 201 - original_z_tomo2)

def test_get_angles(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with no tomo_number
    all_angles = motl.get_angles()
    expected_all_angles = motl.df.loc[:, ["phi", "theta", "psi"]].values
    assert np.array_equal(all_angles, expected_all_angles)
    assert all_angles.ndim == 2

    # Test with tomo_number
    tomo_number = 1
    tomo_angles = motl.get_angles(tomo_number)
    expected_tomo_angles = motl.df.loc[motl.df["tomo_id"] == tomo_number, ["phi", "theta", "psi"]].values
    assert np.array_equal(tomo_angles, expected_tomo_angles)
    assert tomo_angles.ndim == 2

    # Test with a tomo_number that doesn't exist.
    tomo_number2 = 5
    tomo_angles2 = motl.get_angles(tomo_number2)
    assert np.array_equal(tomo_angles2, np.empty((0, 3)))
    assert tomo_angles2.ndim == 2

    # Test with an empty motl
    empty_motl = Motl.create_empty_motl_df()
    empty_motl = Motl(empty_motl)
    empty_angles = empty_motl.get_angles()
    assert np.array_equal(empty_angles, np.empty((0, 3)))
    assert empty_angles.ndim == 2

def test_get_coordinates(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with no tomo_number
    all_coords = motl.get_coordinates()
    expected_all_coords = (
        motl.df.loc[:, ["x", "y", "z"]].values
        + motl.df.loc[:, ["shift_x", "shift_y", "shift_z"]].values
    )
    assert np.array_equal(all_coords, expected_all_coords)

    # Test with tomo_number
    tomo_number = 1
    tomo_coords = motl.get_coordinates(tomo_number)
    expected_tomo_coords = (
        motl.df.loc[motl.df["tomo_id"] == tomo_number, ["x", "y", "z"]].values
        + motl.df.loc[
            motl.df["tomo_id"] == tomo_number, ["shift_x", "shift_y", "shift_z"]
        ].values
    )
    assert np.array_equal(tomo_coords, expected_tomo_coords)

    # Test with a tomo_number that doesn't exist.
    tomo_number2 = 5
    tomo_coords2 = motl.get_coordinates(tomo_number2)
    assert np.array_equal(tomo_coords2, np.empty((0, 3)))

    # Test with an empty motl
    empty_motl = Motl.create_empty_motl_df()
    empty_motl = Motl(empty_motl)
    empty_coords = empty_motl.get_coordinates()
    assert np.array_equal(empty_coords, np.empty((0, 3)))

@pytest.fixture
def sample_motl_object(sample_motl_data1):
    return Motl(copy.deepcopy(sample_motl_data1))

def test_get_max_number_digits(sample_motl_object):
    # Test with default feature_id ("tomo_id")
    max_digits_tomo_id = sample_motl_object.get_max_number_digits()
    assert max_digits_tomo_id == 1  # max tomo_id is 2

    # Test with a different feature_id ("subtomo_id")
    max_digits_subtomo_id = sample_motl_object.get_max_number_digits(feature_id="subtomo_id")
    assert max_digits_subtomo_id == 1  # max subtomo_id is 6

    # Test with a different feature_id ("score")
    max_digits_score = sample_motl_object.get_max_number_digits(feature_id="score")
    assert max_digits_score == 3 # max score is 0.9

    # Test with a different feature_id ("x")
    max_digits_x = sample_motl_object.get_max_number_digits(feature_id="x")
    assert max_digits_x == 2 # max x is 40

    # Test with a different feature_id ("y")
    max_digits_y = sample_motl_object.get_max_number_digits(feature_id="y")
    assert max_digits_y == 2 # max y is 40

    # Test with a different feature_id ("z")
    max_digits_z = sample_motl_object.get_max_number_digits(feature_id="z")
    assert max_digits_z == 2 # max z is 40

def test_get_rotations(sample_motl_data1):

    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with no tomo_number
    all_rotations = motl.get_rotations()
    all_angles = motl.get_angles()
    expected_all_rotations = rot.from_euler("zxz", all_angles, degrees=True)
    assert np.array_equal(all_rotations.as_matrix(), expected_all_rotations.as_matrix())

    # Test with tomo_number
    tomo_number = 1
    tomo_rotations = motl.get_rotations(tomo_number)
    tomo_angles = motl.get_angles(tomo_number)
    expected_tomo_rotations = rot.from_euler("zxz", tomo_angles, degrees=True)
    assert np.array_equal(tomo_rotations.as_matrix(), expected_tomo_rotations.as_matrix())

    # Test with a tomo_number that doesn't exist.
    tomo_number2 = 5
    tomo_rotations2 = motl.get_rotations(tomo_number2)
    assert len(tomo_rotations2) == 0

def test_make_angles_canonical_precise(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    original_angles = motl.df[["phi", "theta", "psi"]].values

    motl.make_angles_canonical()

    new_angles = motl.df[["phi", "theta", "psi"]].values

    expected_canonical_angles = []
    for phi, theta, psi in original_angles:
        # Example: Keep phi and psi within [-180, 180], theta within [0, 180]
        phi_canonical = (phi + 180) % 360 - 180
        theta_canonical = abs(theta % 360)
        if theta_canonical > 180:
            theta_canonical = 360 - theta_canonical
        psi_canonical = (psi + 180) % 360 - 180
        expected_canonical_angles.append([phi_canonical, theta_canonical, psi_canonical])
    expected_canonical_angles = np.array(expected_canonical_angles)

    # Compare calculated and actual angles
    assert np.allclose(new_angles, expected_canonical_angles)

def test_get_barycentric_motl():
    pass
    #TODO
