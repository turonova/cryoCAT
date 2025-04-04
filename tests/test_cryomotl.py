import shutil
import tempfile
import warnings
from cryocat import starfileio, cryomotl
import mrcfile
import copy
import numpy as np
import pandas as pd
import pytest
import os

from cryocat import ioutils, cryomap, cryomask, geom
from cryocat.cryomotl import Motl, EmMotl, RelionMotl, StopgapMotl, DynamoMotl, ModMotl, stopgap2emmotl, emmotl2stopgap
from cryocat.exceptions import UserInputError
from scipy.spatial.transform import Rotation as rot


@pytest.fixture
def motl():
    motl = Motl.load("./test_data/au_1.em")
    return motl

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

def test_remove_feature(sample_motl_data1):
    motl = Motl(copy.deepcopy(sample_motl_data1))

    # Test with single value removal
    motl_single = copy.deepcopy(motl)
    motl_single.remove_feature("tomo_id", 1)
    expected_single = sample_motl_data1[sample_motl_data1["tomo_id"] != 1].reset_index(drop=True)
    assert np.array_equal(motl_single.df.values, expected_single.values)

    # Test with multiple value removal (list)
    motl_list = copy.deepcopy(motl)
    motl_list.remove_feature("tomo_id", [1, 2])
    expected_list = sample_motl_data1[~sample_motl_data1["tomo_id"].isin([1, 2])].reset_index(drop=True)
    assert np.array_equal(motl_list.df.values, expected_list.values)

    # Test with multiple value removal (NumPy array)
    motl_array = copy.deepcopy(motl)
    motl_array.remove_feature("tomo_id", np.array([1, 2]))
    expected_array = sample_motl_data1[~sample_motl_data1["tomo_id"].isin(np.array([1, 2]))].reset_index(drop=True)
    assert np.array_equal(motl_array.df.values, expected_array.values)

    # Test with empty feature_values
    motl_empty_values = copy.deepcopy(motl)
    motl_empty_values.remove_feature("tomo_id", [])
    assert np.array_equal(motl_empty_values.df.values, sample_motl_data1.values)

    # Test with non-existent feature_id
    with pytest.raises(KeyError):
        motl.remove_feature("non_existent_feature", 1)


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


@pytest.mark.parametrize("m", ["./test_data/au_1.em", "./test_data/au_2.em"])
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
        ["./test_data/au_1.em", "./test_data/au_2.em"],
        ["./test_data/au_1.em", "./test_data/au_1.em"],
    ],
)
def test_merge_and_renumber(motl_list):
    combined_len = 0
    for m in motl_list:
        combined_len += len(Motl.load(m).df)
    merged_motl = Motl.merge_and_renumber(motl_list)
    assert len(merged_motl.df) == combined_len


@pytest.mark.parametrize(
    "motl_list", ["./test_data/au_1.em", [], (), "not_a_list", 42, ["./test_data/au_1.em", None]]
)
def test_merge_and_renumber_wrong(motl_list):
    with pytest.raises((ValueError, UserInputError)):
        Motl.merge_and_renumber(motl_list)

def test_merge_and_renumber_empty_motls():
    empty_motl1 = Motl(Motl.create_empty_motl_df())
    empty_motl2 = Motl(Motl.create_empty_motl_df())
    merged_motl = Motl.merge_and_renumber([empty_motl1, empty_motl2])
    assert len(merged_motl.df) == 0


@pytest.mark.parametrize("f", ["subtomo_id", "geom2"])
def test_split_by_feature(motl, f):
    motls = motl.split_by_feature(f)
    for motl in motls:
        check_emmotl(motl)

def test_split_by_feature_tomo_id(sample_motl_data1):
    motl = Motl(sample_motl_data1)
    motls = motl.split_by_feature("tomo_id")
    assert len(motls) == 2
    assert len(motls[0].df) == 3
    assert len(motls[1].df) == 3
    assert motls[0].df["tomo_id"].unique() == [1]
    assert motls[1].df["tomo_id"].unique() == [2]

def test_split_by_feature_object_id(sample_motl_data1):
    motl = Motl(sample_motl_data1)
    motls = motl.split_by_feature("object_id")
    assert len(motls) == 6
    for i, submotl in enumerate(motls):
        assert len(submotl.df) == 1
        assert submotl.df["object_id"].unique() == [100 + i * 100]

def test_split_by_feature_class(sample_motl_data1):
    motl = Motl(sample_motl_data1)
    motls = motl.split_by_feature("class")
    assert len(motls) == 2
    assert len(motls[0].df) == 3
    assert len(motls[1].df) == 3
    assert motls[0].df["class"].unique() == [1]
    assert motls[1].df["class"].unique() == [2]

def test_split_by_feature_nonexistent_feature(sample_motl_data1):
    motl = Motl(sample_motl_data1)
    with pytest.raises(KeyError):
        motl.split_by_feature("nonexistent_feature")


@pytest.mark.parametrize(
    "m, ref",
    [
        ("./test_data/recenter/allmotl_sp_cl1_1.em", "./test_data/recenter/ref1.em"),
        ("./test_data/recenter/allmotl_sp_cl1_2.em", "./test_data/recenter/ref2.em"),
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
            "./test_data/shift_positions/allmotl_sp_cl1_1.em",
            [1, 2, 3],
            "./test_data/shift_positions/ref1.em",
        ),
        (
            "./test_data/shift_positions/allmotl_sp_cl1_1.em",
            [-10, 200, 3.5],
            "./test_data/shift_positions/ref2.em",
        ),
        (
            "./test_data/shift_positions/allmotl_sp_cl1_1.em",
            [0, 0, 0],
            "./test_data/shift_positions/ref3.em",
        ),
        (
            "./test_data/shift_positions/allmotl_sp_cl1_1.em",
            [1, 1, 1],
            "./test_data/shift_positions/ref4.em",
        ),
        (
            "./test_data/shift_positions/allmotl_sp_cl1_5.em",
            [-10, 10, 100],
            "./test_data/shift_positions/ref5.em",
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

@pytest.fixture
def sample_stopgap_data():
        data = {
            "motl_idx": [1],
            "tomo_num": [1],
            "object": [1],
            "subtomo_num": [1],
            "halfset": ["A"],
            "orig_x": [10.0],
            "orig_y": [11.0],
            "orig_z": [12.0],
            "score": [0.9],
            "x_shift": [0.0],
            "y_shift": [0.0],
            "z_shift": [0.0],
            "phi": [0.0],
            "psi": [0.0],
            "the": [0.0],
            "class": [1],
        }
        return pd.DataFrame(data)

def test_stopgap2emmotl_1(sample_stopgap_data, tmp_path):
    # Verify that the converted_to_motl df respects the checks of Motl class
    sg_motl = StopgapMotl(sample_stopgap_data)
    #print('\n',sorted(Motl.motl_columns))
    #print(sorted(sg_motl.df.columns))
    assert Motl.check_df_correct_format(sg_motl.df)
    em_motl = stopgap2emmotl(sg_motl)

    expected_em_data = pd.DataFrame({
        "score": [0.9],
        "geom1": [0.0],
        "geom2": [0.0],
        "subtomo_id": [1],
        "tomo_id": [1],
        "object_id": [1],
        "subtomo_mean": [0.0],
        "x": [10.0],
        "y": [11.0],
        "z": [12.0],
        "shift_x": [0.0],
        "shift_y": [0.0],
        "shift_z": [0.0],
        "geom3": [0.0],
        "geom4": [0.0],
        "geom5": [0.0],
        "phi": [0.0],
        "psi": [0.0],
        "theta": [0.0],
        "class": [1],
    })

    assert isinstance(em_motl, EmMotl)
    print(em_motl.df.columns)
    pd.testing.assert_frame_equal(em_motl.df, expected_em_data)

def test_stopgap2emmotl_file():
    tmp = "./test_data/motl_data/class6_er_mr1_1_sg.star"
    stopgap_motl = StopgapMotl(tmp)
    em_motl_c = cryomotl.stopgap2emmotl(tmp, update_coordinates=False)
    assert isinstance(em_motl_c, cryomotl.EmMotl)
    em_motl_c1 = cryomotl.stopgap2emmotl(stopgap_motl, update_coordinates=False)
    assert isinstance(em_motl_c1, cryomotl.EmMotl)
    em_motl_c2 = cryomotl.stopgap2emmotl(stopgap_motl.df, update_coordinates=False)
    assert isinstance(em_motl_c2, cryomotl.EmMotl)

def test_emmotl2stopgap_basic(tmp_path):
    em_data = pd.DataFrame({
        "score": [0.9, 0.8],
        "geom1": [0.0, 0.0],
        "geom2": [0.0, 0.0],
        "subtomo_id": [1, 2],
        "tomo_id": [1, 1],
        "object_id": [1, 2],
        "subtomo_mean": [0.0, 0.0],
        "x": [10.0, 11.0],
        "y": [11.0, 12.0],
        "z": [12.0, 13.0],
        "shift_x": [0.0, 0.1],
        "shift_y": [0.0, 0.2],
        "shift_z": [0.0, 0.3],
        "geom3": [1.0, 0.0],
        "geom4": [0.0, 0.0],
        "geom5": [0.0, 0.0],
        "phi": [0.0, 10.0],
        "psi": [0.0, 20.0],
        "theta": [0.0, 30.0],
        "class": [1, 2],
    })
    em_motl = EmMotl(em_data)

    sg_motl = emmotl2stopgap(em_motl)

    assert isinstance(sg_motl, StopgapMotl)
    expected_sg_df = pd.DataFrame({
        "score": [0.9, 0.8],
        "geom1": [0.0, 0.0],
        "geom2": [0.0, 0.0],
        "subtomo_id": [1, 2],
        "tomo_id": [1, 1],
        "object_id": [1, 2],
        "subtomo_mean": [0.0, 0.0],
        "x": [10.0, 11.0],
        "y": [11.0, 12.0],
        "z": [12.0, 13.0],
        "shift_x": [0.0, 0.1],
        "shift_y": [0.0, 0.2],
        "shift_z": [0.0, 0.3],
        "geom3": [1.0, 0.0],
        "geom4": [0.0, 0.0],
        "geom5": [0.0, 0.0],
        "phi": [0.0, 10.0],
        "psi": [0.0, 20.0],
        "theta": [0.0, 30.0],
        "class": [1, 2],
    })

    pd.testing.assert_frame_equal(sg_motl.df, expected_sg_df)


#TODO
def test_relion2stopgap():
    pass
#TODO
def test_stopgap2relion():
    pass


class TestMotl:

    def test_check_df_correct_format(self):
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

    def test_create_empty_motl_df(self):
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
    def sample_data(self):
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

    def test_adapt_to_trimming(self, sample_data):
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

    def test_apply_rotation(self, sample_data):
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
    def sample_motl(self):
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
    def sample_input_df(self):
        input_data = {
            "input_tomo": [3, 4],
            "input_subtomo": [5, 6],
            "other_col": [1000, 2000],
        }
        return pd.DataFrame(input_data)

    def test_assign_column(self, sample_motl, sample_input_df):
        motl = sample_motl
        input_df = sample_input_df

        column_pairs = {"tomo_id": "input_tomo", "subtomo_id": "input_subtomo"}

        motl.assign_column(input_df, column_pairs)

        expected_tomo_id = input_df["input_tomo"].tolist()
        expected_subtomo_id = input_df["input_subtomo"].tolist()

        assert motl.df["tomo_id"].tolist() == expected_tomo_id
        assert motl.df["subtomo_id"].tolist() == expected_subtomo_id

    def test_assign_column_missing_input_column(self, sample_motl, sample_input_df):
        motl = sample_motl
        input_df = sample_input_df

        column_pairs = {"tomo_id": "nonexistent_column", "subtomo_id": "input_subtomo"}

        motl.assign_column(input_df, column_pairs)

        expected_subtomo_id = input_df["input_subtomo"].tolist()

        assert motl.df["subtomo_id"].tolist() == expected_subtomo_id
        # Ensure that the tomo_id column was not changed because the input column did not exist.
        assert motl.df["tomo_id"].tolist() == [1,1]

    def test_assign_column_empty_column_pairs(self, sample_motl, sample_input_df):
        motl = sample_motl
        input_df = sample_input_df

        column_pairs = {}

        motl.assign_column(input_df, column_pairs)

        # Ensure that the motl dataframe was not changed
        assert motl.df["tomo_id"].tolist() == [1,1]
        assert motl.df["subtomo_id"].tolist() == [1,2]



    @pytest.fixture
    def sample_motl_data1(self):
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

    def test_clean_by_distance_basic(self, sample_motl_data1):
        motl = Motl(sample_motl_data1.copy())
        motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="score")
        assert motl.df.shape[0] == 4
        assert np.all(motl.df["tomo_id"].values == np.array([1, 1, 2, 2]))

    def test_clean_by_distance_grouping(self, sample_motl_data1):
        motl = Motl(sample_motl_data1.copy())
        motl.clean_by_distance(distance_in_voxels=20, feature_id="tomo_id", metric_id="score")
        assert motl.df.shape[0] == 2
        assert np.all(motl.df["tomo_id"].values == np.array([1, 2]))

    def test_clean_by_distance_metric(self, sample_motl_data1):
        motl = Motl(sample_motl_data1.copy())
        motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="x")
        assert motl.df.shape[0] == 4
        assert np.all(motl.df["x"].values == np.array([11, 20, 31, 40]))

    def test_clean_by_distance_empty(self):
        motl = Motl(Motl.create_empty_motl_df())
        motl.clean_by_distance(distance_in_voxels=2, feature_id="tomo_id", metric_id="score")
        assert motl.df.shape[0] == 0

    def test_clean_by_distance_dist_mask(self, sample_motl_data1):
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

    def test_clean_by_distance_to_points(self, sample_motl_data1):
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

    def test_clean_by_tomo_mask(self, sample_motl_data1):
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

    def test_clean_by_otsu(self, sample_motl_data1):
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

    def test_check_df_type(self, sample_motl_data1):
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

    def test_fill(self, sample_motl_data1):
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

    def test_get_random_subset(self, sample_motl_data1):
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

    def test_assign_random_classes(self, sample_motl_data1):
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

    def test_flip_handedness(self, sample_motl_data1):
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

    def test_get_angles(self, sample_motl_data1):
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

    def test_get_coordinates(self, sample_motl_data1):
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
    def sample_motl_object(self, sample_motl_data1):
        return Motl(copy.deepcopy(sample_motl_data1))

    def test_get_max_number_digits(self, sample_motl_object):
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

    def test_get_rotations(self, sample_motl_data1):

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

    def test_make_angles_canonical_precise(self, sample_motl_data1):
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

    def test_get_barycentric_motl(self, sample_motl_data1):
        motl = Motl(sample_motl_data1.copy())

        # Test with 2 points (barycenter between two points)
        idx = np.array([0, 2])
        nn_idx = np.array([[1], [3]])
        bary_motl_2pt = motl.get_barycentric_motl(idx, nn_idx)

        coord1 = motl.get_coordinates()[idx, :]
        coord2 = motl.get_coordinates()[nn_idx[:, 0], :]
        expected_coords_2pt = (coord1 + coord2) / 2

        assert np.allclose(bary_motl_2pt.get_coordinates(), expected_coords_2pt)
        assert np.array_equal(bary_motl_2pt.df['tomo_id'].values, motl.df['tomo_id'].values[idx])
        assert np.array_equal(bary_motl_2pt.df['object_id'].values, motl.df['object_id'].values[idx])

        # Test with 3 points (barycenter of a triangle)
        idx = np.array([0, 1])
        nn_idx = np.array([[2, 3], [0, 2]])
        bary_motl_3pt = motl.get_barycentric_motl(idx, nn_idx)

        coord1 = motl.get_coordinates()[idx, :]
        coord2 = motl.get_coordinates()[nn_idx[:, 0], :]
        coord3 = motl.get_coordinates()[nn_idx[:, 1], :]
        expected_coords_3pt = (coord1 + coord2 + coord3) / 3

        assert np.allclose(bary_motl_3pt.get_coordinates(), expected_coords_3pt)
        assert np.array_equal(bary_motl_3pt.df['tomo_id'].values, motl.df['tomo_id'].values[idx])
        assert np.array_equal(bary_motl_3pt.df['object_id'].values, motl.df['object_id'].values[idx])

        # Test with angles
        idx = np.array([0, 1])
        nn_idx = np.array([[2, 3], [0, 2]])
        bary_motl_angles = motl.get_barycentric_motl(idx, nn_idx)

        angles_original = motl.df[["phi", "theta", "psi"]].values[idx]
        coord_diff = bary_motl_angles.get_coordinates() - motl.get_coordinates()[idx]

        w1 = geom.euler_angles_to_normals(angles_original[0, :])
        w2 = coord_diff[0, :] / np.linalg.norm(coord_diff[0, :])
        w3 = np.cross(w1, w2)
        w3 = (w3 / np.linalg.norm(w3)).reshape(3,)
        w_base_mat = np.asarray([w1.reshape((3,)), w2, w3]).T

        v1 = geom.euler_angles_to_normals(angles_original)
        rot_angles = np.zeros(angles_original.shape)

        for i in range(1, angles_original.shape[0]):
            v2 = coord_diff[i, :] / np.linalg.norm(coord_diff[i, :])
            v3 = np.cross(v1[i, :], v2)
            v3 = (v3 / np.linalg.norm(v3)).reshape(3,)
            v_base_mat = np.asarray([v1[i, :].reshape((3,)), v2, v3])
            final_mat = np.matmul(w_base_mat, v_base_mat)
            final_rot = rot.from_matrix(final_mat)
            rot_angles[i, :] = final_rot.as_euler("zxz", degrees=True)

        assert np.allclose(bary_motl_angles.df[["phi", "theta", "psi"]].values[1:], rot_angles[1:])

        # Test with empty input
        with pytest.raises(IndexError):
            motl.get_barycentric_motl(np.array([]), np.array([]))

    def test_get_motl_subset(self, sample_motl_data1):
        motl = Motl(copy.deepcopy(sample_motl_data1))

        # Test with single feature_value
        subset_motl_single = motl.get_motl_subset(1)
        expected_df_single = sample_motl_data1[sample_motl_data1["tomo_id"] == 1].reset_index(drop=True)

        # Reorder columns to match
        subset_motl_single_df = subset_motl_single.df[expected_df_single.columns]

        assert np.allclose(subset_motl_single_df.values, expected_df_single.values)

        # Test with list of feature_values
        subset_motl_list = motl.get_motl_subset([1, 2])
        expected_df_list = sample_motl_data1[sample_motl_data1["tomo_id"].isin([1, 2])].reset_index(drop=True)

        subset_motl_list_df = subset_motl_list.df[expected_df_list.columns]

        assert np.allclose(subset_motl_list_df.values, expected_df_list.values)

        # Test with different feature_id
        subset_motl_score = motl.get_motl_subset(0.9, feature_id="score")
        expected_df_score = sample_motl_data1[sample_motl_data1["score"] == 0.9].reset_index(drop=True)

        subset_motl_score_df = subset_motl_score.df[expected_df_score.columns]

        assert np.allclose(subset_motl_score_df.values, expected_df_score.values)

        # Test with return_df=True
        subset_df = motl.get_motl_subset(1, return_df=True)
        subset_df = subset_df[expected_df_single.columns] # Reorder columns
        assert np.allclose(subset_df.values, expected_df_single.values)

        # Test with reset_index=False
        subset_motl_no_reset = motl.get_motl_subset(1, reset_index=False)
        expected_df_no_reset = sample_motl_data1[sample_motl_data1["tomo_id"] == 1]

        subset_motl_no_reset_df = subset_motl_no_reset.df[expected_df_no_reset.columns]

        assert np.allclose(subset_motl_no_reset_df.values, expected_df_no_reset.values)

        # Test with an empty subset
        subset_motl_empty = motl.get_motl_subset(3)
        assert len(subset_motl_empty.df) == 0

    @pytest.fixture
    def sample_motl_data2(self):
        """Fixture providing a sample DataFrame for Motl object."""
        data = {
            "tomo_id": [1, 2, 3, 4, 5, 6],
            "x": [10, 20, 30, 40, 50, 60],
            "y": [11, 21, 31, 41, 51, 61],
            "z": [12, 22, 32, 42, 52, 62],
            "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4],
            "subtomo_id": [1, 2, 3, 7, 8, 9],
            "geom1": [1, 1, 1, 2, 2, 2],
            "geom2": [2, 2, 2, 3, 3, 3],
            "geom3": [100, 200, 300, 400, 500, 600],
            "object_id": [1, 2, 3, 4, 5, 6], # Added
            "subtomo_mean": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6], # Added
            "shift_x": [0, 0, 0, 0, 0, 0], # Added
            "shift_y": [0, 0, 0, 0, 0, 0], # Added
            "shift_z": [0, 0, 0, 0, 0, 0], # Added
            "geom4": [3, 3, 3, 4, 4, 4],
            "geom5": [4, 4, 4, 5, 5, 5],
            "phi": [0, 10, 20, 30, 40, 50],
            "psi": [5, 15, 25, 35, 45, 55],
            "theta": [10, 20, 30, 40, 50, 60],
            "class": [1, 2, 1, 2, 1, 2],
        }
        return pd.DataFrame(data)

    def test_get_motl_intersection(self, sample_motl_data1, sample_motl_data2):
        motl1 = Motl(copy.deepcopy(sample_motl_data1))
        motl2 = Motl(copy.deepcopy(sample_motl_data2))

        # Test with default feature_id ("subtomo_id")
        intersection_motl = Motl.get_motl_intersection(motl1, motl2)
        expected_df = pd.merge(sample_motl_data1, sample_motl_data2[["subtomo_id"]], how="inner").reset_index(drop=True)
        assert np.allclose(intersection_motl.df.values, expected_df.values)

        # Test with different feature_id ("tomo_id")
        intersection_motl_tomo = Motl.get_motl_intersection(motl1, motl2, feature_id="tomo_id")
        expected_df_tomo = pd.merge(sample_motl_data1, sample_motl_data2[["tomo_id"]], how="inner").reset_index(drop=True)
        assert np.allclose(intersection_motl_tomo.df.values, expected_df_tomo.values)

    def test_renumber_objects_sequentially(self, sample_motl_data1):
        motl = Motl(sample_motl_data1.copy())

        # Test with default starting number (1)
        motl.renumber_objects_sequentially()
        expected_object_ids_default = pd.Series([1, 2, 3, 4, 5, 6])
        assert motl.df["object_id"].equals(expected_object_ids_default)

        # Test with a different starting number (10)
        motl2 = Motl(sample_motl_data1.copy())
        motl2.renumber_objects_sequentially(starting_number=10)
        expected_object_ids_10 = pd.Series([10, 11, 12, 13, 14, 15])
        assert motl2.df["object_id"].equals(expected_object_ids_10)

        # Test with multiple tomograms
        data_multiple_tomos = {
            "tomo_id": [1, 1, 2, 2, 3, 3],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
            "z": [1, 2, 3, 4, 5, 6],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "subtomo_id": [1, 2, 3, 4, 5, 6],
            "shift_x": [0, 0, 0, 0, 0, 0],
            "shift_y": [0, 0, 0, 0, 0, 0],
            "shift_z": [0, 0, 0, 0, 0, 0],
            "phi": [0, 0, 0, 0, 0, 0],
            "psi": [0, 0, 0, 0, 0, 0],
            "theta": [0, 0, 0, 0, 0, 0],
            "geom1": [0, 0, 0, 0, 0, 0],
            "geom2": [0, 0, 0, 0, 0, 0],
            "geom3": [0, 0, 0, 0, 0, 0],
            "geom4": [0, 0, 0, 0, 0, 0],
            "geom5": [0, 0, 0, 0, 0, 0],
            "object_id": [10, 20, 30, 40, 50, 60],
            "class": [1, 1, 1, 1, 1, 1],
        }
        df_multiple_tomos = pd.DataFrame(data_multiple_tomos)
        df_multiple_tomos["subtomo_mean"] = [0, 0, 0, 0, 0, 0] # explicitly create column
        print(sample_motl_data1.columns)
        print(df_multiple_tomos.columns)
        df_multiple_tomos = df_multiple_tomos[sample_motl_data1.columns]
        motl3 = Motl(df_multiple_tomos.copy())
        motl3.renumber_objects_sequentially()
        expected_object_ids_multiple_tomos = pd.Series([1, 2, 3, 4, 5, 6])
        assert motl3.df["object_id"].equals(expected_object_ids_multiple_tomos)

        # Test with multiple tomograms and custom start number
        motl4 = Motl(df_multiple_tomos.copy())
        motl4.renumber_objects_sequentially(starting_number=100)
        expected_object_ids_multiple_tomos_100 = pd.Series([100, 101, 102, 103, 104, 105])
        assert motl4.df["object_id"].equals(expected_object_ids_multiple_tomos_100)

        # test with duplicate object ids within a tomo id.
        data_duplicate_object_ids = {
            "tomo_id": [1, 1, 1, 2, 2, 2],
            "x": [1, 2, 3, 4, 5, 6],
            "y": [1, 2, 3, 4, 5, 6],
            "z": [1, 2, 3, 4, 5, 6],
            "score": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "subtomo_id": [1, 2, 3, 4, 5, 6],
            "shift_x": [0, 0, 0, 0, 0, 0],
            "shift_y": [0, 0, 0, 0, 0, 0],
            "shift_z": [0, 0, 0, 0, 0, 0],
            "phi": [0, 0, 0, 0, 0, 0],
            "psi": [0, 0, 0, 0, 0, 0],
            "theta": [0, 0, 0, 0, 0, 0],
            "geom1": [0, 0, 0, 0, 0, 0],
            "geom2": [0, 0, 0, 0, 0, 0],
            "geom3": [0, 0, 0, 0, 0, 0],
            "geom4": [0, 0, 0, 0, 0, 0],
            "geom5": [0, 0, 0, 0, 0, 0],
            "object_id": [10, 10, 20, 30, 30, 40],
            "class": [1, 1, 1, 1, 1, 1],
        }
        df_duplicate_object_ids = pd.DataFrame(data_duplicate_object_ids)
        df_duplicate_object_ids["subtomo_mean"] = [0, 0, 0, 0, 0, 0]
        df_duplicate_object_ids = df_duplicate_object_ids[sample_motl_data1.columns]
        motl5 = Motl(df_duplicate_object_ids.copy())
        motl5.renumber_objects_sequentially()
        expected_object_ids_duplicate_object_ids = pd.Series([1, 1, 2, 3, 3, 4])
        assert motl5.df["object_id"].equals(expected_object_ids_duplicate_object_ids)

    @pytest.fixture
    def sample_relative_position_df(self):
        """Create a sample motl DataFrame for testing relative positions."""
        data = {
            'score': [1.0, 2.0, 3.0, 4.0],
            'geom1': [0, 0, 0, 0],
            'geom2': [0, 0, 0, 0],
            'subtomo_id': [1, 2, 3, 4],
            'tomo_id': [1, 1, 2, 2],
            'object_id': [1, 1, 2, 2],
            'subtomo_mean': [0, 0, 0, 0],
            'x': [10, 20, 30, 40],
            'y': [10, 20, 30, 40],
            'z': [10, 20, 30, 40],
            'shift_x': [0, 0, 0, 0],
            'shift_y': [0, 0, 0, 0],
            'shift_z': [0, 0, 0, 0],
            'geom3': [0, 0, 0, 0],
            'geom4': [0, 0, 0, 0],
            'geom5': [0, 0, 0, 0],
            'phi': [0, 45, 90, 135],
            'psi': [0, 0, 0, 0],
            'theta': [0, 0, 0, 0],
            'class': [1, 1, 2, 2]
        }
        return pd.DataFrame(data)

    def test_get_relative_position(self, sample_relative_position_df):
        motl = Motl(sample_relative_position_df)

        # Test with simple indices
        idx = np.array([0, 1])
        nn_idx = np.array([2, 3])
        rel_motl, rotated_coord = motl.get_relative_position(idx, nn_idx)

        # Check output type and shape
        assert isinstance(rel_motl, Motl)
        assert len(rel_motl.df) == len(idx)
        assert rotated_coord.shape == (len(idx), 3)

        # Check relative positions are calculated correctly (center between particles)
        expected_coords = np.array([[20, 20, 20], [30, 30, 30]])
        np.testing.assert_array_almost_equal(rel_motl.get_coordinates(), expected_coords)

        # Check Euler angles are calculated correctly (we need to calculate them manually)
        angles = motl.df[["phi", "theta", "psi"]].values[idx, :]
        coord1 = motl.get_coordinates()[idx, :]
        coord2 = motl.get_coordinates()[nn_idx, :]
        c_coord = coord2 - coord1

        w1 = geom.euler_angles_to_normals(angles[0, :])
        w2 = c_coord[0, :] / np.linalg.norm(c_coord[0, :])
        w3 = np.cross(w1, w2)
        w3 = (w3 / np.linalg.norm(w3)).reshape(3,)
        w_base_mat = np.asarray([w1.reshape((3,)), w2, w3]).T

        v1 = geom.euler_angles_to_normals(angles)
        rot_angles = np.zeros(angles.shape)

        for i in range(1, angles.shape[0]):
            v2 = c_coord[i, :] / np.linalg.norm(c_coord[i, :])
            v3 = np.cross(v1[i, :], v2)
            v3 = (v3 / np.linalg.norm(v3)).reshape(3,)
            v_base_mat = np.asarray([v1[i, :].reshape((3,)), v2, v3])
            final_mat = np.matmul(w_base_mat, v_base_mat)
            final_rot = rot.from_matrix(final_mat)
            rot_angles[i, :] = final_rot.as_euler("zxz", degrees=True)
        np.testing.assert_array_almost_equal(rel_motl.df[["phi", "theta", "psi"]].values, rot_angles)

        # Check rotated coordinates
        rot_coord = rot.from_euler("zxz", angles=angles, degrees=True)
        expected_rotated_coord = rot_coord.apply(c_coord)
        np.testing.assert_array_almost_equal(rotated_coord, expected_rotated_coord)


    def test_get_unique_values(self, sample_motl_data1):
        motl = Motl(copy.deepcopy(sample_motl_data1))

        # Test with a simple column
        unique_tomo_ids = motl.get_unique_values("tomo_id")
        expected_tomo_ids = np.array([1, 2])
        assert np.array_equal(unique_tomo_ids, expected_tomo_ids)

        # Test with a different data type (float)
        unique_scores = motl.get_unique_values("score")
        expected_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        assert np.array_equal(np.sort(unique_scores), np.sort(expected_scores)) #order does not matter.

        # Test with an empty Motl DataFrame
        empty_motl = Motl(pd.DataFrame(columns=motl.motl_columns))
        unique_empty = empty_motl.get_unique_values("tomo_id")
        assert np.array_equal(unique_empty, np.array([]))

        # Test with column not found
        with pytest.raises(KeyError):
            motl.get_unique_values("non_existent_column")


    def test_renumber_particles(self, sample_motl_data1):
        motl = Motl(copy.deepcopy(sample_motl_data1))

        # Test basic renumbering
        motl_renumbered = copy.deepcopy(motl)
        motl_renumbered.renumber_particles()
        expected_renumbered = sample_motl_data1.copy()
        expected_renumbered["subtomo_id"] = list(range(1, len(expected_renumbered) + 1))
        assert np.array_equal(motl_renumbered.df.values, expected_renumbered.values)

        # Test with empty Motl
        empty_motl = Motl(pd.DataFrame(columns=motl.motl_columns))
        empty_motl.renumber_particles()
        assert len(empty_motl.df) == 0

        # Test with DataFrame with rows removed
        motl_removed = copy.deepcopy(motl)
        motl_removed.remove_feature("tomo_id", 1)
        motl_removed.renumber_particles()
        expected_removed = sample_motl_data1[sample_motl_data1["tomo_id"] != 1].copy()
        expected_removed["subtomo_id"] = list(range(1, len(expected_removed) + 1))
        assert np.array_equal(motl_removed.df.values, expected_removed.values)

        #Test already renumbered
        motl_double_renumbered = copy.deepcopy(motl)
        motl_double_renumbered.renumber_particles()
        motl_double_renumbered.renumber_particles()
        assert np.array_equal(motl_double_renumbered.df.values, expected_renumbered.values)

    def test_scale_coordinates(self, sample_motl_data1):
        motl = Motl(copy.deepcopy(sample_motl_data1))

        # Test with positive scaling factor
        motl_positive = copy.deepcopy(motl)
        scaling_factor_positive = 2.0
        motl_positive.scale_coordinates(scaling_factor_positive)
        expected_positive = sample_motl_data1.copy()
        for coord in ("x", "y", "z"):
            expected_positive[coord] = expected_positive[coord] * scaling_factor_positive
            shift_column = "shift_" + coord
            expected_positive[shift_column] = expected_positive[shift_column] * scaling_factor_positive
        assert np.array_equal(motl_positive.df.values, expected_positive.values)

        # Test with negative scaling factor
        motl_negative = copy.deepcopy(motl)
        scaling_factor_negative = -1.0
        motl_negative.scale_coordinates(scaling_factor_negative)
        expected_negative = sample_motl_data1.copy()
        for coord in ("x", "y", "z"):
            expected_negative[coord] = expected_negative[coord] * scaling_factor_negative
            shift_column = "shift_" + coord
            expected_negative[shift_column] = expected_negative[shift_column] * scaling_factor_negative
        assert np.array_equal(motl_negative.df.values, expected_negative.values)

        # Test with zero scaling factor
        motl_zero = copy.deepcopy(motl)
        scaling_factor_zero = 0.0
        motl_zero.scale_coordinates(scaling_factor_zero)
        expected_zero = sample_motl_data1.copy()
        for coord in ("x", "y", "z"):
            expected_zero[coord] = expected_zero[coord] * scaling_factor_zero
            shift_column = "shift_" + coord
            expected_zero[shift_column] = expected_zero[shift_column] * scaling_factor_zero
        assert np.array_equal(motl_zero.df.values, expected_zero.values)

        # Test with scaling factor of 1
        motl_one = copy.deepcopy(motl)
        scaling_factor_one = 1.0
        motl_one.scale_coordinates(scaling_factor_one)
        assert np.array_equal(motl_one.df.values, sample_motl_data1.values)


    def test_write_to_model_file_tomo_id(self, sample_motl_data1):
        motl = Motl(sample_motl_data1)
        temp_dir = "test_temp_dir"  # Define a temporary directory name
        try:
            os.makedirs(temp_dir, exist_ok=True)  # Create the directory
            output_base = os.path.join(temp_dir, "test")
            motl.write_to_model_file("tomo_id", output_base, 5, binning=2.0)

            # Verify files are created
            assert os.path.exists(f"{output_base}_tomo_id_1_model.txt")
            assert os.path.exists(f"{output_base}_tomo_id_1.mod")
            assert os.path.exists(f"{output_base}_tomo_id_2_model.txt")
            assert os.path.exists(f"{output_base}_tomo_id_2.mod")

            # Verify content of text file (basic check)
            with open(f"{output_base}_tomo_id_1_model.txt", "r") as f:
                content = f.read()
                assert "1\t1\t20.0\t20.0\t20.0\n2\t1\t22.0\t22.0\t22.0\n1\t1\t40.0\t40.0\t40.0\n" in content

        finally:
            shutil.rmtree(temp_dir)  # Clean up the directory

    def test_write_to_model_file_object_id(self, sample_motl_data1):
        motl = Motl(sample_motl_data1)
        temp_dir = "test_temp_dir2"
        try:
            os.makedirs(temp_dir, exist_ok=True)
            output_base = os.path.join(temp_dir, "object")
            motl.write_to_model_file("object_id", output_base, 3, zero_padding=3)
            assert os.path.exists(f"{output_base}_object_id_100.mod")
            assert os.path.exists(f"{output_base}_object_id_200.mod")
            assert os.path.exists(f"{output_base}_object_id_300.mod")
            assert os.path.exists(f"{output_base}_object_id_400.mod")
        finally:
            shutil.rmtree(temp_dir)


    def test_update_coordinates(self):
        data = {
            "x": [10.2, 20.5, 30.8, 40.0, 50.5],
            "y": [11.3, 21.6, 31.9, 41.0, 51.5],
            "z": [12.4, 22.7, 32.0, 42.0, 52.5],
            "shift_x": [0.3, -0.5, 0.2, 0.0, -0.5],
            "shift_y": [-0.4, 0.4, -0.1, 0.0, 0.5],
            "shift_z": [0.6, -0.7, 0.0, 0.0, -0.5],
            "tomo_id": [1, 1, 2, 2, 3],
            "object_id": [1, 2, 3, 4, 5],
            "score": [1, 1, 1, 1, 1],
            "subtomo_id": [1, 2, 3, 4, 5],
            "geom1": [0,0,0,0,0],
            "geom2": [0,0,0,0,0],
            "geom3": [0,0,0,0,0],
            "geom4": [0,0,0,0,0],
            "geom5": [0,0,0,0,0],
            "phi": [0,0,0,0,0],
            "psi": [0,0,0,0,0],
            "theta": [0,0,0,0,0],
            "class": [1,1,1,1,1],
            "subtomo_mean": [0,0,0,0,0]
        }
        df = pd.DataFrame(data)
        motl = Motl(df)

        with warnings.catch_warnings(record=True) as w:
            motl.update_coordinates()
            assert len(w) == 1
            assert "The coordinates for subtomogram extraction were changed, new extraction is necessary!" in str(w[0].message)

        expected_x = [11, 20, 31, 40, 50]
        expected_y = [11, 22, 32, 41, 52]
        expected_z = [13, 22, 32, 42, 52]
        expected_shift_x = [-0.5, 0.0, 0.0, 0.0, 0.0]
        expected_shift_y = [-0.1, 0.0, -0.2, 0.0, 0.0]
        expected_shift_z = [0.0, 0.0, 0.0, 0.0, 0.0]

        assert list(motl.df["x"]) == expected_x
        assert list(motl.df["y"]) == expected_y
        assert list(motl.df["z"]) == expected_z
        assert np.allclose(motl.df["shift_x"], expected_shift_x, atol=1e-8) # added atol
        assert np.allclose(motl.df["shift_y"], expected_shift_y, atol=1e-8) # added atol
        assert np.allclose(motl.df["shift_z"], expected_shift_z, atol=1e-8) # added atol


    @pytest.mark.parametrize(
        "motl_list, expected_shape",
        [
            # Test merging two identical motls - should drop duplicates
            (lambda df: [Motl(df.copy()), Motl(df.copy())], (4, 20)),
            # Test merging empty motl with non-empty
            (lambda df: [Motl(df.copy()), Motl(Motl.create_empty_motl_df())], (4, 20)),
            # Test merging three motls with some overlapping data
            (lambda df: [Motl(df.iloc[0:2]), Motl(df.iloc[1:3]), Motl(df.iloc[2:4])], (4, 20)),
        ],
    )
    def test_merge_and_drop_duplicates(self, sample_motl_df, motl_list, expected_shape):
        test_motls = motl_list(sample_motl_df)

        # Merge motls
        merged = Motl.merge_and_drop_duplicates(test_motls)

        # Check the merged result has expected shape
        assert merged.df.shape == expected_shape

        # Check no duplicates exist
        assert merged.df.duplicated().sum() == 0

        # Check all original data is preserved when merging non-duplicates
        if len(test_motls) == 2 and test_motls[1].df.empty:
            merged.df = merged.df.astype(test_motls[0].df.dtypes)  # Ensure dtype consistency
            pd.testing.assert_frame_equal(merged.df, test_motls[0].df)


    @pytest.fixture
    def sample_motl_df(self):
        """Create a sample motl DataFrame for testing."""
        data = {
            'score': [1.0, 2.0, 3.0, 4.0],
            'geom1': [0, 0, 0, 0],
            'geom2': [0, 0, 0, 0],
            'subtomo_id': [1, 2, 3, 4],
            'tomo_id': [1, 1, 2, 2],
            'object_id': [1, 1, 2, 2],
            'subtomo_mean': [0, 0, 0, 0],
            'x': [10, 20, 500, 50],  # Third coordinate out of bounds
            'y': [10, 20, 30, 150],  # Fourth coordinate out of bounds
            'z': [10, 20, 30, 30],
            'shift_x': [0, 0, 0, 0],
            'shift_y': [0, 0, 0, 0],
            'shift_z': [0, 0, 0, 0],
            'geom3': [0, 0, 0, 0],
            'geom4': [0, 0, 0, 0],
            'geom5': [0, 0, 0, 0],
            'phi': [0, 0, 0, 0],
            'psi': [0, 0, 0, 0],
            'theta': [0, 0, 0, 0],
            'class': [1, 1, 2, 2]
        }
        return pd.DataFrame(data)


    @pytest.fixture
    def sample_dimensions(self):
        """Create sample dimensions DataFrame for testing."""
        data = {
            'tomo_id': [1, 2],
            'x': [100, 100],
            'y': [100, 100],
            'z': [100, 100]
        }
        return pd.DataFrame(data)


    def test_remove_out_of_bounds_particles(self, sample_motl_df, sample_dimensions):
        # Create test Motl instance
        motl = Motl(sample_motl_df)

        # Test center boundary type
        motl.remove_out_of_bounds_particles(sample_dimensions, boundary_type="center")
        assert len(motl.df) == 2  # Two particles should be removed

        # Check specific remaining particles
        expected_subtomo_ids = [1, 2]  # Only first two particles should remain
        assert list(motl.df['subtomo_id']) == expected_subtomo_ids

        # Test whole boundary type with box_size=20
        motl = Motl(sample_motl_df)  # Reset motl
        motl.remove_out_of_bounds_particles(sample_dimensions, boundary_type="whole", box_size=20)

        # With box_size=20, particles need 10 pixels clearance from edges
        assert len(motl.df) == 2
        assert list(motl.df['subtomo_id']) == expected_subtomo_ids

        # Test whole boundary type with larger box_size=40
        motl = Motl(sample_motl_df)
        motl.remove_out_of_bounds_particles(sample_dimensions, boundary_type="whole", box_size=40)
        assert len(motl.df) == 2  # Same result but tests different boundary condition

        # Test invalid boundary type
        with pytest.raises(UserInputError, match="Unknown type of boundaries:"):
            motl.remove_out_of_bounds_particles(sample_dimensions, boundary_type="invalid")

        # Test missing box_size
        with pytest.raises(UserInputError, match="You need to specify box_size"):
            motl.remove_out_of_bounds_particles(sample_dimensions, boundary_type="whole")

    def test_drop_duplicates(self, sample_motl_data1):
        df = pd.concat([sample_motl_data1, sample_motl_data1.iloc[[0]]]).reset_index(drop=True)
        motl = Motl(df)

        # Test default behavior (drop duplicates, keep highest score)
        motl.drop_duplicates()
        assert len(motl.df) == len(sample_motl_data1)
        assert motl.df["score"].iloc[0] == sample_motl_data1["score"].iloc[0]

        # Test drop duplicates, keep lowest score
        motl = Motl(df.copy())
        motl.drop_duplicates(decision_sort_ascending=True)
        assert len(motl.df) == len(sample_motl_data1)
        assert motl.df["score"].iloc[0] == sample_motl_data1["score"].iloc[0]

        # Test drop duplicates, keep lowest geom1
        motl = Motl(df.copy())
        motl.drop_duplicates(decision_column="geom1", decision_sort_ascending=True)
        assert len(motl.df) == len(sample_motl_data1)
        assert motl.df["geom1"].iloc[0] == sample_motl_data1["geom1"].iloc[0]

        # Test drop duplicates object_id, keep lowest geom1
        df2 = df.copy()
        df2['object_id'] = [1,1,2,3,4,5,1]
        motl = Motl(df2.copy())
        motl.drop_duplicates(duplicates_column="object_id", decision_column="geom1", decision_sort_ascending=True)
        assert len(motl.df) == 5 # Corrected assertion
        assert motl.df["geom1"].iloc[0] == sample_motl_data1["geom1"].iloc[0]

        #Test that indecies are reset
        assert motl.df.index.tolist() == list(range(5))

    def test_recenter_to_subparticle(self, sample_motl_data1, tmp_path):
        # Create a simple binary mask
        mask_data = np.zeros((10, 10, 10), dtype=np.uint8)
        mask_data[3:7, 3:7, 3:7] = 1  # Create a cube in the center
        mask_path = str(tmp_path / "mask.mrc")
        cryomap.write(mask_data, mask_path)

        # Create a Motl instance
        motl = Motl(sample_motl_data1.copy())

        # Recenter the Motl
        recentered_motl = Motl.recenter_to_subparticle(motl, mask_path)

        # Calculate the expected shifts
        old_center = np.array(mask_data.shape) / 2
        mask_center = cryomask.get_mass_center(mask_data)
        expected_shifts = mask_center - old_center

        # Check that the positions are shifted correctly
        assert np.allclose(recentered_motl.df["x"] - motl.df["x"], expected_shifts[0])
        assert np.allclose(recentered_motl.df["y"] - motl.df["y"], expected_shifts[1])
        assert np.allclose(recentered_motl.df["z"] - motl.df["z"], expected_shifts[2])

        # Test with rotation
        rotation = rot.from_euler("xyz", [np.pi / 4, 0, 0])
        rotated_motl = Motl.recenter_to_subparticle(motl, mask_path, rotation=rotation)

        # Apply the same rotation to the shifts and check
        rotated_shifts = rotation.apply(expected_shifts)
        rotated_shifted_positions = motl.df[["x", "y", "z"]].values + rotated_shifts

        assert np.allclose(rotated_motl.df[["x", "y", "z"]].values, rotated_shifted_positions)

    def test_apply_tomo_rotation(self, sample_motl_data1):
        motl = Motl(sample_motl_data1)
        rotation_angles = [90, 0, 0]
        tomo_id = 1
        tomo_dim = [100, 100, 100]

        subset_motl = motl.get_motl_subset(tomo_id, feature_id="tomo_id")
        coord = subset_motl.get_coordinates()

        coord_rot = rot.from_euler(
            "zyx", angles=[rotation_angles[2], rotation_angles[1], rotation_angles[0]], degrees=True
        )

        def rotate_points(points, rot, tomo_dim):
            dim = np.asarray(tomo_dim)
            points = points - dim / 2
            points = rot.apply(points) + dim / 2
            return points

        expected_coord = rotate_points(coord, coord_rot, tomo_dim)

        shift_x_coord = subset_motl.shift_positions([1, 0, 0], inplace=False).get_coordinates()
        shift_y_coord = subset_motl.shift_positions([0, 1, 0], inplace=False).get_coordinates()
        shift_z_coord = subset_motl.shift_positions([0, 0, 1], inplace=False).get_coordinates()

        x_vector = rotate_points(shift_x_coord, coord_rot, tomo_dim) - expected_coord
        y_vector = rotate_points(shift_y_coord, coord_rot, tomo_dim) - expected_coord
        phi_angle = geom.angle_between_vectors(x_vector, y_vector)
        rot_angles = geom.normals_to_euler_angles(
            rotate_points(shift_z_coord, coord_rot, tomo_dim) - expected_coord, output_order="zxz"
        )
        rot_angles[:, 0] = phi_angle

        result = motl.apply_tomo_rotation(rotation_angles, tomo_id, tomo_dim)

        np.testing.assert_allclose(result.get_coordinates(), expected_coord, atol=1e-5)
        np.testing.assert_allclose(result.df[["phi", "theta", "psi"]].values, rot_angles, atol=1e-5)

    def test_apply_tomo_rotation_zero_rotation(self, sample_motl_data1):
        motl = Motl(sample_motl_data1)
        rotation_angles = [0, 0, 0]
        tomo_id = 1
        tomo_dim = [100, 100, 100]

        subset_motl = motl.get_motl_subset(tomo_id, feature_id="tomo_id")
        result = motl.apply_tomo_rotation(rotation_angles, tomo_id, tomo_dim)

        np.testing.assert_allclose(result.get_coordinates(), subset_motl.get_coordinates(), atol=1e-5)

    def test_apply_tomo_rotation_single_particle(self, sample_motl_data1):
        motl = Motl(sample_motl_data1[sample_motl_data1["tomo_id"] == 1].iloc[[0]])
        rotation_angles = [45, 0, 0]
        tomo_id = 1
        tomo_dim = [100, 100, 100]

        result = motl.apply_tomo_rotation(rotation_angles, tomo_id, tomo_dim)
        assert len(result.df) == 1

    def test_apply_tomo_rotation_nonexistent_tomo_id(self, sample_motl_data1):
        motl = Motl(sample_motl_data1)
        rotation_angles = [45, 0, 0]
        tomo_id = 3
        tomo_dim = [100, 100, 100]

        with pytest.raises(Exception):
            result = motl.apply_tomo_rotation(rotation_angles, tomo_id, tomo_dim)

    @pytest.fixture
    def single_row_motl(self):
        data = {
            "shift_x": [5.0],
            "shift_y": [-3.0],
            "shift_z": [2.0],
            "phi": [30.0],
            "theta": [45.0],
            "psi": [60.0],
            "tomo_id": [1],
            "x": [10.0],
            "y": [20.0],
            "z": [30.0],
            "score": [0.8],
            "subtomo_id": [1],
            "geom1": [1],
            "geom2": [1],
            "object_id": [1],
            "subtomo_mean": [1.5],
            "geom3": [1],
            "geom4": [1],
            "geom5": [1],
            "class": [2],
        }
        return pd.DataFrame(data)

    def test_split_in_asymmetric_subunits_c2(self, single_row_motl):
        motl = Motl(single_row_motl)
        result = motl.split_in_asymmetric_subunits("C2", np.array([10, 0, 0]))

        # Corrected expected values
        expected_x = [16, 14]
        expected_y = [26, 8]
        expected_z = [36, 28]
        expected_shift_x = [ 0.268265, -0.268265]
        expected_shift_y = [ 0.267767, -0.267767]
        expected_shift_z = [-0.464466,  0.464466]
        expected_phi = [30, -150]
        expected_theta = [45, 45]
        expected_psi = [60, 60]

        assert len(result.df) == 2
        assert np.array_equal(result.df["subtomo_id"].values, [1, 2])
        assert np.array_equal(result.df["geom2"].values, [1, 2])
        np.testing.assert_allclose(result.df["x"].values, expected_x, atol=1e-5)
        np.testing.assert_allclose(result.df["y"].values, expected_y, atol=1e-5)
        np.testing.assert_allclose(result.df["z"].values, expected_z, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_x"].values, expected_shift_x, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_y"].values, expected_shift_y, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_z"].values, expected_shift_z, atol=1e-5)
        np.testing.assert_allclose(result.df["phi"].values, expected_phi, atol=1e-5)
        np.testing.assert_allclose(result.df["theta"].values, expected_theta, atol=1e-5)
        np.testing.assert_allclose(result.df["psi"].values, expected_psi, atol=1e-5)

    def test_split_in_asymmetric_subunits_d2(self, single_row_motl):
        motl = Motl(single_row_motl)
        result = motl.split_in_asymmetric_subunits("D2", np.array([10, 0, 0]))

        # Calculate expected values
        expected_x = [16, 7, 14, 23]
        expected_y = [26, 16, 8, 18]
        expected_z = [36, 38, 28, 26]
        expected_shift_x = [ 0.268265,  0.196699, -0.268265, -0.196699]
        expected_shift_y = [ 0.267767, -0.268265, -0.267767,  0.268265]
        expected_shift_z = [-0.464466,  0.123724,  0.464466, -0.123724]
        expected_phi = [30, -120, -150, 60]
        expected_theta = [45, 135, 45, 135]
        expected_psi = [60, -120, 60, -120]

        assert len(result.df) == 4
        assert np.array_equal(result.df["subtomo_id"].values, [1, 2, 3, 4])
        assert np.array_equal(result.df["geom2"].values, [1, 2, 3, 4])
        np.testing.assert_allclose(result.df["x"].values, expected_x, atol=1e-5)
        np.testing.assert_allclose(result.df["y"].values, expected_y, atol=1e-5)
        np.testing.assert_allclose(result.df["z"].values, expected_z, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_x"].values, expected_shift_x, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_y"].values, expected_shift_y, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_z"].values, expected_shift_z, atol=1e-5)
        np.testing.assert_allclose(result.df["phi"].values, expected_phi, atol=1e-5)
        np.testing.assert_allclose(result.df["theta"].values, expected_theta, atol=1e-5)
        np.testing.assert_allclose(result.df["psi"].values, expected_psi, atol=1e-5)

    def test_split_in_asymmetric_subunits_c3(self, single_row_motl):
        motl = Motl(single_row_motl)
        result = motl.split_in_asymmetric_subunits("C3", np.array([10, 0, 0]))

        # Calculate expected values
        expected_x = [16, 8, 21]
        expected_y = [26, 11, 13]
        expected_z = [36, 36, 25]
        expected_shift_x = [0.268265, -0.391989,  0.123724]
        expected_shift_y = [0.267767, 0.267767, 0.464466]
        expected_shift_z = [-0.464466, -0.464466, -0.071068]
        expected_phi = [30, 150, -90]
        expected_theta = [45, 45, 45]
        expected_psi = [60, 60, 60]

        assert len(result.df) == 3
        assert np.array_equal(result.df["subtomo_id"].values, [1, 2, 3])
        assert np.array_equal(result.df["geom2"].values, [1, 2, 3])
        np.testing.assert_allclose(result.df["x"].values, expected_x, atol=1e-5)
        np.testing.assert_allclose(result.df["y"].values, expected_y, atol=1e-5)
        np.testing.assert_allclose(result.df["z"].values, expected_z, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_x"].values, expected_shift_x, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_y"].values, expected_shift_y, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_z"].values, expected_shift_z, atol=1e-5)
        np.testing.assert_allclose(result.df["phi"].values, expected_phi, atol=1e-5)
        np.testing.assert_allclose(result.df["theta"].values, expected_theta, atol=1e-5)
        np.testing.assert_allclose(result.df["psi"].values, expected_psi, atol=1e-5)

    def test_split_in_asymmetric_subunits_int_symmetry(self, single_row_motl):
        motl = Motl(single_row_motl)
        result = motl.split_in_asymmetric_subunits(2, np.array([10, 0, 0]))

        # Calculate expected values
        expected_x = [16, 14]
        expected_y = [26, 8]
        expected_z = [36, 28]
        expected_shift_x = [ 0.268265, -0.268265]
        expected_shift_y = [ 0.267767, -0.267767]
        expected_shift_z = [-0.464466,  0.464466]
        expected_phi = [30, -150]
        expected_theta = [45, 45]
        expected_psi = [60, 60]

        assert len(result.df) == 2
        assert np.array_equal(result.df["subtomo_id"].values, [1, 2])
        assert np.array_equal(result.df["geom2"].values, [1, 2])
        np.testing.assert_allclose(result.df["x"].values, expected_x, atol=1e-5)
        np.testing.assert_allclose(result.df["y"].values, expected_y, atol=1e-5)
        np.testing.assert_allclose(result.df["z"].values, expected_z, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_x"].values, expected_shift_x, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_y"].values, expected_shift_y, atol=1e-5)
        np.testing.assert_allclose(result.df["shift_z"].values, expected_shift_z, atol=1e-5)
        np.testing.assert_allclose(result.df["phi"].values, expected_phi, atol=1e-5)
        np.testing.assert_allclose(result.df["theta"].values, expected_theta, atol=1e-5)
        np.testing.assert_allclose(result.df["psi"].values, expected_psi, atol=1e-5)

class TestEmMotl:
    @pytest.fixture
    def sample_em_data(self):
        data = {
            "score": [0.9],
            "geom1": [0.0],
            "geom2": [0.0],
            "subtomo_id": [1],
            "tomo_id": [1],
            "object_id": [1],
            "subtomo_mean": [0.0],
            "x": [10.0],
            "y": [11.0],
            "z": [12.0],
            "shift_x": [0.0],
            "shift_y": [0.0],
            "shift_z": [0.0],
            "geom3": [0.0],
            "geom4": [0.0],
            "geom5": [0.0],
            "phi": [0.0],
            "psi": [0.0],
            "theta": [0.0],
            "class": [1],
        }
        return pd.DataFrame(data)


    @pytest.mark.parametrize("m", ["./test_data/au_1.em", "./test_data/au_2.em"])
    def test_read_from_emfile(self, m):
        motl = EmMotl(m)
        check_emmotl(motl)

    def test_emmotl_init_emmotl(self, sample_em_data):
        em_motl1 = EmMotl(sample_em_data)
        em_motl2 = EmMotl(em_motl1)
        assert em_motl1.df.equals(em_motl2.df)

    def test_emmotl_init_dataframe(self, sample_em_data):
        em_motl = EmMotl(sample_em_data)
        assert em_motl.df.equals(sample_em_data)

    def test_emmotl_init_invalid_file(self, tmp_path):
        file_path = tmp_path / "nonexistent.em"
        with pytest.raises(UserInputError):
            EmMotl(str(file_path))

    def test_emmotl_init_invalid_input(self):
        with pytest.raises(UserInputError):
            EmMotl(123)

    def test_emmotl_init_no_input(self):
        em_motl = EmMotl()
        assert len(em_motl.df) == 0

    def test_emmotl_convert_to_motl(self, sample_em_data):
        em_motl = EmMotl(sample_em_data)
        with pytest.raises(ValueError):
            em_motl.convert_to_motl(sample_em_data)

    def test_write_out_em(self, sample_em_data, tmp_path):
        stopgap_motl = EmMotl(sample_em_data)
        output_path = tmp_path / "output.em"

        # Call the write_out method
        EmMotl.write_out(str(output_path))

        # Check if the file was created
        assert os.path.exists(output_path)

        # Check if the correct data was written.
        loaded_motl = EmMotl(output_path)
        pd.testing.assert_frame_equal(loaded_motl.df, stopgap_motl.df)

#TODO
class TestRelionMotl:
    def test_set_version_already_set(self):
        motl = RelionMotl(version=4.0)
        motl.set_version(pd.DataFrame())
        assert motl.version == 4.0

    def test_set_version_argument(self):
        motl = RelionMotl()
        motl.set_version(pd.DataFrame(), version=3.0)
        assert motl.version == 3.1 #Should not be changed: default is 3.1 and after set
        #It can't be changed!

    def test_set_version_from_dataframe_v4(self):
        df = pd.DataFrame({"rlnTomoName": [1]})
        motl = RelionMotl()
        motl.set_version(df)
        assert motl.version == 3.1 #As test before

    def test_set_version_default(self):
        df = pd.DataFrame()
        motl = RelionMotl()
        with warnings.catch_warnings(record=True) as w:
            motl.set_version(df)
            assert len(w) == 0
            assert motl.version == 3.1

class TestStopgapMotl:
    @pytest.fixture
    def sample_stopgap_data(self):
        data = {
            "motl_idx": [1],
            "tomo_num": [1],
            "object": [1],
            "subtomo_num": [1],
            "halfset": ["A"],
            "orig_x": [10.0],
            "orig_y": [11.0],
            "orig_z": [12.0],
            "score": [0.9],
            "x_shift": [0.0],
            "y_shift": [0.0],
            "z_shift": [0.0],
            "phi": [0.0],
            "psi": [0.0],
            "the": [0.0],
            "class": [1],
        }
        return pd.DataFrame(data)

    def test_init(self, sample_stopgap_data):
        sg = StopgapMotl(sample_stopgap_data)
        assert isinstance(sg, StopgapMotl)
        # fixme: issue with Stopgap.convert_to_motl: geom1,geom2,geom4,geom5,subtomo_mean are missing
        assert Motl.check_df_correct_format(sg.df) == True
    def test_read_in(self, tmp_path, sample_stopgap_data):
        # Create a temporary STAR file with the sample data
        star_path = tmp_path / "test.star"
        starfileio.Starfile.write([sample_stopgap_data], str(star_path), specifiers=["data_stopgap_motivelist"])


        # Assert that the read DataFrame is equal to the original sample data
        pd.testing.assert_frame_equal(StopgapMotl.read_in(str(star_path)), sample_stopgap_data)

    def test_convert_to_motl_halfsets(self):
        stopgap_motl = StopgapMotl()
        data = {
            "motl_idx": [1, 2, 3, 4],
            "tomo_num": [1, 1, 1, 1],
            "object": [1, 1, 1, 1],
            "subtomo_num": [1, 2, 3, 4],
            "halfset": ["A", "B", "A", "B"],
            "orig_x": [10.0, 11.0, 12.0, 13.0],
            "orig_y": [11.0, 12.0, 13.0, 14.0],
            "orig_z": [12.0, 13.0, 14.0, 15.0],
            "score": [0.9, 0.8, 0.7, 0.6],
            "x_shift": [0.0, 0.1, 0.2, 0.3],
            "y_shift": [0.0, 0.1, 0.2, 0.3],
            "z_shift": [0.0, 0.1, 0.2, 0.3],
            "phi": [0.0, 1.0, 2.0, 3.0],
            "psi": [0.0, 1.0, 2.0, 3.0],
            "the": [0.0, 1.0, 2.0, 3.0],
            "class": [1, 2, 1, 2],
        }
        stopgap_df = pd.DataFrame(data)
        stopgap_motl.convert_to_motl(stopgap_df, keep_halfsets=True)

        # Verify "geom3" column
        assert list(stopgap_motl.df["geom3"]) == [1, 2, 3, 4]

        # Verify "subtomo_id" renumbering
        assert list(stopgap_motl.df["subtomo_id"]) == [1, 2, 3, 4]

        # Verify original subtomo_num moved to geom3
        assert list(stopgap_motl.df["geom3"]) == [1, 2, 3, 4]

        # Verify that self.sg_df stores the original data
        pd.testing.assert_frame_equal(stopgap_motl.sg_df, stopgap_df)


    def test_convert_to_motl_no_halfsets(self, sample_stopgap_data):
        stopgap_motl = StopgapMotl()
        stopgap_motl.convert_to_motl(sample_stopgap_data, keep_halfsets=True)

        # Verify that geom3 is not added
        assert "geom3" not in stopgap_motl.df.columns

        # Verify subtomo_id is not changed
        assert stopgap_motl.df["subtomo_id"].iloc[0] == sample_stopgap_data["subtomo_num"].iloc[0]

        # Verify that self.sg_df stores the original data
        pd.testing.assert_frame_equal(stopgap_motl.sg_df, sample_stopgap_data)

    def test_convert_to_motl_one_halfset(self):
        stopgap_motl = StopgapMotl()
        data = {
            "motl_idx": [1, 2, 3],
            "tomo_num": [1, 1, 1],
            "object": [1, 1, 1],
            "subtomo_num": [1, 2, 3],
            "halfset": ["A", "A", "A"],
            "orig_x": [10.0, 11.0, 12.0],
            "orig_y": [11.0, 12.0, 13.0],
            "orig_z": [12.0, 13.0, 14.0],
            "score": [0.9, 0.8, 0.7],
            "x_shift": [0.0, 0.1, 0.2],
            "y_shift": [0.0, 0.1, 0.2],
            "z_shift": [0.0, 0.1, 0.2],
            "phi": [0.0, 1.0, 2.0],
            "psi": [0.0, 1.0, 2.0],
            "the": [0.0, 1.0, 2.0],
            "class": [1, 2, 1],
        }
        stopgap_df = pd.DataFrame(data)
        stopgap_motl.convert_to_motl(stopgap_df, keep_halfsets=True)

        # Verify that geom3 is not added
        assert "geom3" not in stopgap_motl.df.columns

        # Verify subtomo_id is not changed
        assert list(stopgap_motl.df["subtomo_id"]) == [1, 2, 3]

        # Verify that self.sg_df stores the original data
        pd.testing.assert_frame_equal(stopgap_motl.sg_df, stopgap_df)

    @pytest.fixture
    def sample_em_data(self):
        data = {
            "score": [0.9],
            "geom1": [0.0],
            "geom2": [0.0],
            "subtomo_id": [1],
            "tomo_id": [1],
            "object_id": [1],
            "subtomo_mean": [0.0],
            "x": [10.0],
            "y": [11.0],
            "z": [12.0],
            "shift_x": [0.0],
            "shift_y": [0.0],
            "shift_z": [0.0],
            "geom3": [0.0],
            "geom4": [0.0],
            "geom5": [0.0],
            "phi": [0.0],
            "psi": [0.0],
            "theta": [0.0],
            "class": [1],
        }
        return pd.DataFrame(data)
    def test_convert_to_sg_motl_basic(self, sample_em_data):
        sg_df = StopgapMotl.convert_to_sg_motl(sample_em_data)

        # Verify that the Stopgap DataFrame has the correct columns and data
        for em_key, star_key in StopgapMotl.pairs.items():
            assert star_key in sg_df.columns
            assert sg_df[star_key].iloc[0] == sample_em_data[em_key].iloc[0]

        # Verify halfset assignment
        assert sg_df["halfset"].iloc[0] == "B"  # subtomo_id is 1 (odd)

        # Verify motl_idx assignment
        assert sg_df["motl_idx"].iloc[0] == sample_em_data["subtomo_id"].iloc[0]

    def test_sg_df_reset_index_false(self, sample_stopgap_data):
        # Create a copy of the sample data to avoid modifying the original
        df = sample_stopgap_data.copy()

        # Call the method with reset_index=False
        result_df = StopgapMotl.sg_df_reset_index(df, reset_index=False)

        # Assert that the DataFrame is unchanged
        pd.testing.assert_frame_equal(result_df, df)

    def test_sg_df_reset_index_true(self):
        # Create a DataFrame with multiple rows and arbitrary motl_idx values
        data = {
            "motl_idx": [10, 20, 30],
            "tomo_num": [1, 1, 1],
            "object": [1, 1, 1],
            "subtomo_num": [1, 2, 3],
            "halfset": ["A", "B", "A"],
            "orig_x": [10.0, 11.0, 12.0],
            "orig_y": [11.0, 12.0, 13.0],
            "orig_z": [12.0, 13.0, 14.0],
            "score": [0.9, 0.8, 0.7],
            "x_shift": [0.0, 0.1, 0.2],
            "y_shift": [0.0, 0.1, 0.2],
            "z_shift": [0.0, 0.1, 0.2],
            "phi": [0.0, 1.0, 2.0],
            "psi": [0.0, 1.0, 2.0],
            "the": [0.0, 1.0, 2.0],
            "class": [1, 2, 1],
        }
        df = pd.DataFrame(data)

        # Call the method with reset_index=True
        result_df = StopgapMotl.sg_df_reset_index(df, reset_index=True)

        # Assert that the motl_idx column is correctly reset
        expected_motl_idx = [1, 2, 3]
        assert list(result_df["motl_idx"]) == expected_motl_idx

    def test_write_out_star_reset_index(self, sample_em_data, tmp_path):
        stopgap_motl = StopgapMotl(sample_em_data)
        output_path = tmp_path / "output.star"
        stopgap_motl.write_out(str(output_path), reset_index=True)

        # Read the STAR file and verify the data
        read_df = starfileio.Starfile.read(str(output_path))[0][0]

        # Verify motl_idx
        assert list(read_df["motl_idx"]) == [1]

    def test_write_out_star_reset_index(self, sample_em_data, tmp_path):
        stopgap_motl = StopgapMotl(sample_em_data)
        output_path = tmp_path / "output.star"
        stopgap_motl.write_out(str(output_path), reset_index=True)

        # Read the STAR file and verify the data
        read_df = starfileio.Starfile.read(str(output_path))[0][0]

        # Verify motl_idx
        assert list(read_df["motl_idx"]) == [1]

    def test_write_out_star_basic(self, sample_stopgap_data, tmp_path):
        stopgap_motl = StopgapMotl(sample_stopgap_data)
        output_path = tmp_path / "output.star"
        stopgap_motl.write_out(str(output_path))

        # Read the written STAR file using read_in
        read_motl = StopgapMotl(str(output_path))

        # Verify that the data is correct
        pd.testing.assert_frame_equal(read_motl.df, stopgap_motl.df)

    def test_write_out_star_reset_index(self, sample_stopgap_data, tmp_path):
        stopgap_motl = StopgapMotl(sample_stopgap_data)
        output_path = tmp_path / "output.star"
        stopgap_motl.write_out(str(output_path), reset_index=True)

        # Read the written STAR file using read_in
        read_motl = StopgapMotl(str(output_path))

        # Verify motl_idx
        assert list(read_motl.df["subtomo_id"]) == [1]

    def test_write_out_em(self, sample_stopgap_data, tmp_path):
        stopgap_motl = StopgapMotl(sample_stopgap_data)
        output_path = tmp_path / "output.em"
        stopgap_motl.write_out(str(output_path))

        # Read the written EM file using EmMotl
        read_motl = EmMotl(str(output_path))

        # Convert subtomo_id to int64
        read_motl.df["subtomo_id"] = read_motl.df["subtomo_id"].astype("int64")

        # Verify that the data is correct
        pd.testing.assert_frame_equal(read_motl.df, stopgap_motl.df)

class TestDynamoMotl:
    def test_constructor1(self):
        pathfile = "./test_data/motl_data/b4_motl_CR_tm_topbott_clean600_1_dynamo.tbl"
        test1 = DynamoMotl(pathfile)
        print(test1.df)
        print(test1.dynamo_df)

        test2 = DynamoMotl(pathfile)
        print(test2.df)
        print(test2.dynamo_df)

        with pytest.raises(Exception):
            excp = DynamoMotl("test")

    def test_read_in(self):
        pathfile = "./test_data/motl_data/b4_motl_CR_tm_topbott_clean600_1_dynamo.tbl"
        expected3_entries = [
            [1, 1, 1, 1.0651, 1.1602, 0.6847, 144.13, -79.159, -180.0, 0.1593, 0, 0, 0, -50.01, 43.99, 0, 0, 0, 0, 2, 2,
             1, 0, 507, 548, 167],
            [2, 1, 1, 0.4504, 0.4036, 0.9178, 135.93, -90.484, -86.0, 0.1426, 0, 0, 0, -50.01, 43.99, 0, 0, 0, 0, 2, 2,
             1, 0, 476, 512, 218],
            [3, 1, 1, 1.1498, 1.0941, 0.568, 142.6, -81.436, -1.0, 0.1376, 0, 0, 0, -50.01, 43.99, 0, 0, 0, 0, 2, 2, 1,
             0, 432, 477, 169]
        ]
        rows_subset = DynamoMotl.read_in(pathfile).iloc[:3]
        pd.testing.assert_frame_equal(pd.DataFrame(expected3_entries), rows_subset)

    def test_convert_to_motl(self):
        pathfile = "./test_data/motl_data/b4_motl_CR_tm_topbott_clean600_1_dynamo.tbl"
        dynamo_motl = DynamoMotl(pathfile)
        dynamo_df = dynamo_motl.dynamo_df
        motl_df = dynamo_motl.df

        # Expected values for the first three rows
        expected_score = dynamo_df.loc[:2, 9].tolist()
        expected_subtomo_id = dynamo_df.loc[:2, 0].tolist()
        expected_tomo_id = dynamo_df.loc[:2, 19].tolist()
        expected_object_id = dynamo_df.loc[:2, 20].tolist()
        expected_x = dynamo_df.loc[:2, 23].tolist()
        expected_y = dynamo_df.loc[:2, 24].tolist()
        expected_z = dynamo_df.loc[:2, 25].tolist()
        expected_shift_x = dynamo_df.loc[:2, 3].tolist()
        expected_shift_y = dynamo_df.loc[:2, 4].tolist()
        expected_shift_z = dynamo_df.loc[:2, 5].tolist()
        expected_phi = (-dynamo_df.loc[:2, 8]).tolist()
        expected_psi = (-dynamo_df.loc[:2, 6]).tolist()
        expected_theta = (-dynamo_df.loc[:2, 7]).tolist()
        expected_class = dynamo_df.loc[:2, 21].tolist()

        assert motl_df["score"].tolist()[:3] == expected_score
        assert motl_df["subtomo_id"].tolist()[:3] == expected_subtomo_id
        assert motl_df["tomo_id"].tolist()[:3] == expected_tomo_id
        assert motl_df["object_id"].tolist()[:3] == expected_object_id
        assert motl_df["x"].tolist()[:3] == expected_x
        assert motl_df["y"].tolist()[:3] == expected_y
        assert motl_df["z"].tolist()[:3] == expected_z
        assert motl_df["shift_x"].tolist()[:3] == expected_shift_x
        assert motl_df["shift_y"].tolist()[:3] == expected_shift_y
        assert motl_df["shift_z"].tolist()[:3] == expected_shift_z
        assert motl_df["phi"].tolist()[:3] == expected_phi
        assert motl_df["psi"].tolist()[:3] == expected_psi
        assert motl_df["theta"].tolist()[:3] == expected_theta
        assert motl_df["class"].tolist()[:3] == expected_class

        assert Motl.check_df_correct_format(motl_df) == True

#TODO
class TestModMotl:
    def test_read_in_valid_file(self):
        test_file_path = "./test_data/motl_data/two_contour_001.mod"
        mod_df = ModMotl.read_in(test_file_path)
        assert isinstance(mod_df, pd.DataFrame)
        assert not mod_df.empty
        assert all(col in mod_df.columns for col in ['object_id', 'x', 'y', 'z', 'mod_id', 'contour_id'])
        assert (mod_df["mod_id"] == "001").all()
        print(mod_df)

    def test_read_in_invalid_file(self):
        with pytest.raises(ValueError):
            ModMotl.read_in("non_existent_file.mod")

    def test_convert_to_motl(self):
        test_file_path = "./test_data/motl_data/two_contour_001.mod"
        mod_motl = ModMotl(test_file_path)
        assert all(col in mod_motl.df.columns for col in
                   ['subtomo_id', 'tomo_id', 'object_id', 'x', 'y', 'z', 'phi', 'psi', 'theta'])
        # test for correct ranges of euler angles.
        assert all((mod_motl.df["phi"] >= -180) & (mod_motl.df["phi"] <= 180))
        assert all((mod_motl.df["psi"] >= -90) & (mod_motl.df["psi"] <= 90))
        assert all((mod_motl.df["theta"] >= -180) & (mod_motl.df["theta"] <= 180))


