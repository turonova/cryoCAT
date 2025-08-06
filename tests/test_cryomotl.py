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
from cryocat.cryomotl import Motl, EmMotl, RelionMotl, RelionMotlv5, StopgapMotl, DynamoMotl, ModMotl, stopgap2emmotl, emmotl2stopgap
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

def test_emmotl2relion():
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
    rln_motl = cryomotl.emmotl2relion(
        input_motl=em_motl
    )
    assert isinstance(rln_motl, cryomotl.RelionMotl)
    #print(rln_motl.relion_df)
    assert list(rln_motl.relion_df.columns) == []
    #print(rln_motl.df)

    relion_written = "./test_data/motl_data/relionfromem1.star"
    rln_motl = cryomotl.emmotl2relion(
        input_motl=em_motl,
        output_motl_path=relion_written,
        #load_kwargs={"pixel_size":1.0, "binning":1.0}
        #load_kwargs={"version": 4.0}
    )
    #fixme: Only once having written, and loaded the new written file, we can access
    #relion_df attribute correctly!
    rln_motl_written = RelionMotl(relion_written)
    print(rln_motl_written.relion_df)

    assert list(rln_motl_written.relion_df.columns) == RelionMotl.columns_v3_1 + ["ccSubtomoID"]
    assert isinstance(rln_motl_written, cryomotl.RelionMotl)

    """if os.path.exists(relion_written):
        os.remove(relion_written)"""

def test_relion2emmotl():
    relion="./test_data/motl_data/relion_3.1_optics2.star"
    written_em = "./test_data/motl_data/written.em"
    rln = RelionMotl(input_motl=relion, version=3.1)
    em = cryomotl.relion2emmotl(
        input_motl=rln,
        output_motl_path=written_em
    )
    assert isinstance(em, EmMotl)
    print(em.df)
    assert list(em.df.columns) == Motl.motl_columns
    em_loaded = EmMotl(written_em)
    em.df['class'] = em.df['class'].astype('float64')
    pd.testing.assert_frame_equal(em_loaded.df, em.df)

    if os.path.exists(written_em):
        os.remove(written_em)

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

def test_relion2stopgap():
    relion3 = "./test_data/motl_data/relion_3.0.star"
    stopgap = "./test_data/motl_data/sg_from_relion.star"
    sg = cryomotl.relion2stopgap(input_motl=relion3, output_motl_path=stopgap)
    sg1 = StopgapMotl(input_motl=stopgap)
    assert not list(sg.sg_df.columns) == StopgapMotl.columns
    assert list(sg1.sg_df.columns) == StopgapMotl.columns
    pd.testing.assert_frame_equal(sg1.df.dropna(), RelionMotl(relion3).df.dropna())

    if os.path.exists(stopgap):
        os.remove(stopgap)

def test_relion2stopgap_write():
    relion3= "./test_data/motl_data/relion_3.0.star"
    relion3written = "./test_data/motl_data/stopgap_written.star"
    sg = cryomotl.relion2stopgap(
        input_motl=relion3,
        output_motl_path=relion3written
    )
    sg_written = StopgapMotl(relion3written)
    assert list(sg_written.sg_df.columns) == StopgapMotl.columns
    print(sg_written.sg_df)

    if os.path.exists(relion3written):
        os.remove(relion3written)

def test_stopgap2relion():
    sg = "./test_data/motl_data/class6_er_mr1_1_sg.star"
    #relion_optics = "./test_data/motl_data/relion_3.1_optics2.star"
    relion = cryomotl.stopgap2relion(input_motl=sg)
    #of course the same as before
    relion.relion_df = relion.create_relion_df(
        version=3.1,
        pixel_size=1.0
    )
    print(relion.relion_df)
    print(relion.df)
    assert list(relion.relion_df.columns) == RelionMotl.columns_v3_1

def test_stopgap2relion_write():
    sg = "./test_data/motl_data/class6_er_mr1_1_sg.star"
    relion_written = "./test_data/motl_data/rln31_written.star"
    relion = cryomotl.stopgap2relion(
        input_motl=sg,
        output_motl_path=relion_written
    )
    assert isinstance(relion, RelionMotl)
    assert list(relion.relion_df) == []
    relion_motl_written = RelionMotl(relion_written)
    assert list(relion_motl_written.relion_df.columns) == RelionMotl.columns_v3_1 + ["ccSubtomoID"]
    print(relion_motl_written.relion_df)
    # of course the same as before: relion_df only populated when we write the motl out

    if os.path.exists(relion_written):
        os.remove(relion_written)

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

    @pytest.fixture
    def get_sample_data1(self):
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
        return pd.DataFrame(sample_data)

    def test_add_f(self, get_sample_data1):
        df1 = get_sample_data1
        df2 = get_sample_data1
        motl1 = Motl(df1)
        print(len(motl1))
        motl2 = Motl(df2)
        combined_motl = motl1.__add__(motl2)
        assert isinstance(combined_motl, Motl)
        assert len(combined_motl.df) == 8  # 4 rows + 4 rows
        pd.testing.assert_frame_equal(
            combined_motl.df.reset_index(drop=True),
            pd.concat([df1, df2]).reset_index(drop=True)
        )

        with pytest.raises(ValueError):
            combined_motl = motl1.__add__("not a motl object")
    def test_len(self, get_sample_data1):
        motl1 = Motl(get_sample_data1)
        assert motl1.__len__() == 4
        empty = Motl()
        assert empty.__len__() == 0

    def test_getitem(self, get_sample_data1):
        #TODO
        pass

    def test_load(self):
        #emmotldf = EmMotl(input_motl="./test_data/au_1.em").df
        relionmotldf = RelionMotl(input_motl="./test_data/motl_data/relion_3.0.star").df
        #stopgapmotldf = StopgapMotl(input_motl="./test_data/motl_data/bin1_1deg_500.star").sg_df
        #modmotldf = ModMotl(input_motl="./test_data/motl_data/modMotl/correct111.mod").mod_df
        #dynamomotldf = DynamoMotl(input_motl="./test_data/motl_data/crop.tbl").dynamo_df



        #relionmotl
        relionmotl = Motl.load(
            input_motl=relionmotldf,
            motl_type="relion",
            version = 3.0,
            pixel_size = 6,
            binning  = 2.0
        )
        assert relionmotl.pixel_size == 6
        assert relionmotl.binning==2.0
        assert relionmotl.version == 3.0

        #passing not existing arguments should throw an exception
        with pytest.raises(Exception):
            relionmotl2 = Motl.load(
                input_motl=relionmotldf,
                motl_type="relion",
                version=3.0,
                random=5
            )

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
        mmotl = EmMotl(sample_em_data)
        output_path = tmp_path / "output.em"

        # Call the write_out method
        mmotl.write_out(output_path=str(output_path))

        # Check if the file was created
        assert os.path.exists(output_path)

        # Check if the correct data was written.
        loaded_motl = EmMotl(output_path)
        pd.testing.assert_frame_equal(loaded_motl.df, mmotl.df, check_dtype=False)

class TestRelionMotl:
    @pytest.fixture(scope="class", autouse=True)
    def cleanup_after_class(self, request):
        yield
        print("cleanup")

        output_files_to_remove = [
            "./test_data/motl_data/out_3_0.star",
            "./test_data/motl_data/two_out_3_0.star",
            "./test_data/motl_data/out_3_1.star",
            "./test_data/motl_data/out_4_0.star",
            "./test_data/motl_data/out_3_1_optics.star",
            "./test_data/motl_data/out_4_0_optics.star"
        ]
        for file_path in output_files_to_remove:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                    print(f"Removed: {file_path}")
                except OSError as e:
                    print(f"Error removing {file_path}: {e}")
        print("cleanup finished")

    @pytest.fixture(scope="class")
    def relion_paths(self):
        return {
            "relion30_path": "./test_data/motl_data/relion_3.0.star",
            "relion31_path": "./test_data/motl_data/relion_3.1_optics2.star",
            "relion40_path": "./test_data/motl_data/relion_4.0.star",
            "relionbroken_path": "./test_data/motl_data/bin1_1deg_500.star"
        }

    def test_set_version_already_set(self):
        motl = RelionMotl(version=4.0, binning=1.0)
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

    def test_read_in_real_files(self, relion_paths):
        relion_v3_0 = [
            "rlnMicrographName",
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnMagnification",
            "rlnDetectorPixelSize",
            "rlnCtfMaxResolution",
            "rlnImageName",
            "rlnCtfImage",
            "rlnPixelSize",
            "rlnVoltage",
            "rlnSphericalAberration",
            "ccSubtomoID"
        ]

        # Test with v3.0 file
        relion_motl_v30 = RelionMotl(relion_paths['relion30_path'])
        assert isinstance(relion_motl_v30.relion_df, pd.DataFrame)
        assert relion_motl_v30.version == 3.0
        assert relion_motl_v30.optics_data is None
        #more
        """print("3.0df",relion_motl_v30.df)
        print("3.0rdf",relion_motl_v30.relion_df)
        print("3.0op",relion_motl_v30.optics_data)"""
        assert sorted(relion_motl_v30.relion_df.columns) == sorted(relion_v3_0)
        assert Motl.check_df_correct_format(relion_motl_v30.df) == True


        # Test with v3.1 file (with / without optics)
        relion_v3_1 = [
            "rlnMicrographName",
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnCtfMaxResolution",
            "rlnImageName",
            "rlnCtfImage",
            "rlnPixelSize",
            "rlnOpticsGroup",
            "rlnGroupNumber",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnOriginZAngst",
            "rlnClassNumber",
            "rlnNormCorrection",
            "rlnRandomSubset",
            "rlnLogLikeliContribution",
            "rlnMaxValueProbDistribution",
            "rlnNrOfSignificantSamples",
            "ccSubtomoID"
        ]
        relion_v3_1_op = [
            "rlnOpticsGroup",
            "rlnOpticsGroupName",
            "rlnSphericalAberration",
            "rlnVoltage",
            "rlnImagePixelSize",
            "rlnImageSize",
            "rlnImageDimensionality"
        ]
        relion_motl_v31 = RelionMotl(relion_paths['relion31_path'])
        assert isinstance(relion_motl_v31.relion_df, pd.DataFrame)
        assert relion_motl_v31.version == 3.1
        assert isinstance(relion_motl_v31.optics_data, pd.DataFrame)
        """print("3.1df", relion_motl_v31.df.columns)
        print("3.1rdf", relion_motl_v31.relion_df.columns)
        print("3.1op", relion_motl_v31.optics_data)"""
        assert sorted(relion_motl_v31.relion_df.columns) == sorted(relion_v3_1)
        _, list, _ = starfileio.Starfile.read(relion_paths['relion31_path'])
        if "data_optics" in list:
            assert relion_motl_v31.optics_data is not None
            assert sorted(relion_motl_v31.optics_data.columns) == sorted(relion_v3_1_op)
            """print(relion_motl_v31.pixel_size)
            print(relion_motl_v31.optics_data)"""
            #This file contains 2 optics_data entries, and 3 data_particles
            assert np.array_equal(relion_motl_v31.pixel_size, np.array([2.446, 3.446, 3.446]))
        else:
            assert relion_motl_v31.optics_data is None


        # Test with v4.0 file
        relion_motl_v40 = RelionMotl(relion_paths['relion40_path'], binning=1.0)
        relion_v4_0 = [
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnTomoName",
            "rlnTomoParticleName",
            "rlnRandomSubset",
            "rlnOpticsGroup",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnOriginZAngst",
            "rlnImageName",
            "rlnCtfImage",
            "rlnGroupNumber",
            "rlnClassNumber",
            "rlnNormCorrection",
            "rlnLogLikeliContribution",
            "rlnMaxValueProbDistribution",
            "rlnNrOfSignificantSamples",
            "ccSubtomoID"
        ]
        relion_v4_0_op = [
            "rlnOpticsGroup",
            "rlnOpticsGroupName",
            "rlnSphericalAberration",
            "rlnVoltage",
            "rlnTomoTiltSeriesPixelSize",
            "rlnCtfDataAreCtfPremultiplied",
            "rlnImageDimensionality",
            "rlnTomoSubtomogramBinning",
            "rlnImagePixelSize",
            "rlnImageSize"
        ]
        assert isinstance(relion_motl_v40.relion_df, pd.DataFrame)
        assert relion_motl_v40.version == 4.0
        assert sorted(relion_motl_v40.relion_df.columns) == sorted(relion_v4_0)
        """print("4.0df", relion_motl_v40.df.columns)
        print("4.0rdf", relion_motl_v40.relion_df.columns)
        print("4.0op", relion_motl_v40.optics_data)"""
        _, list, _ = starfileio.Starfile.read(relion_paths['relion40_path'])
        if "data_optics" in list:
            assert relion_motl_v40.optics_data is not None
            assert sorted(relion_motl_v40.optics_data.columns) == sorted(relion_v4_0_op)
            #This file contains 1 optics_data entry
            assert relion_motl_v40.pixel_size == 3.942
        else:
            assert relion_motl_v40.optics_data is None

        #Test with "data_images" or "data_..."
        relion123 = RelionMotl(relion_paths["relionbroken_path"])

        #Test2 with "data_****"
        relion123 = RelionMotl("./test_data/motl_data/bin1_1deg_500_2.star")
        print(relion123.relion_df)

    def test_get_version_from_file(self, relion_paths):
        frames3_0, spec3_0, _ = starfileio.Starfile.read(relion_paths['relion30_path'])
        frames3_1, spec3_1, _ = starfileio.Starfile.read(relion_paths['relion31_path'])
        frames4, spec4, _ = starfileio.Starfile.read(relion_paths['relion40_path'])
        assert RelionMotl.get_version_from_file(frames3_0, spec3_0) == 3.0
        assert RelionMotl.get_version_from_file(frames3_1, spec3_1) == 3.1
        assert RelionMotl.get_version_from_file(frames4, spec4) == 4.0

    def test_get_data_particles_id(self, relion_paths):
        frames3_0, spec3_0, _ = starfileio.Starfile.read(relion_paths['relion30_path'])
        frames3_1, spec3_1, _ = starfileio.Starfile.read(relion_paths['relion31_path'])
        frames4, spec4, _ = starfileio.Starfile.read(relion_paths['relion40_path'])
        assert "data_" in spec3_0
        assert RelionMotl._get_data_particles_id(spec3_0) == spec3_0.index("data_")
        assert "data_particles" in spec3_1
        assert RelionMotl._get_data_particles_id(spec3_1) == spec3_1.index("data_particles")
        assert "data_particles" in spec4
        assert RelionMotl._get_data_particles_id(spec4) == spec4.index("data_particles")

    def test_get_optics_id(self, relion_paths):
        input_list_present_start = ["data_optics", "data_particles"]
        assert RelionMotl._get_optics_id(input_list_present_start) == 0

        input_list_not_present = ["data_particles", "data_"]
        assert RelionMotl._get_optics_id(input_list_not_present) is None

        input_list_empty = []
        assert RelionMotl._get_optics_id(input_list_empty) is None

        input_list_present_end = ["data_particles", "data_optics"]
        assert RelionMotl._get_optics_id(input_list_present_end) == 1

        input_list_substring = ["adata_opticsb", "data_particles"]
        assert RelionMotl._get_optics_id(input_list_substring) is None

        _, spec3_1, _ = starfileio.Starfile.read(relion_paths['relion31_path'])
        index_31 = RelionMotl._get_optics_id(spec3_1)
        assert "data_optics" in spec3_1
        assert index_31 == spec3_1.index("data_optics")

        _, spec4, _ = starfileio.Starfile.read(relion_paths['relion40_path'])
        index_4 = RelionMotl._get_optics_id(spec4)
        assert "data_optics" in spec4
        assert index_4 == spec4.index("data_optics")

        _, spec3_0, _ = starfileio.Starfile.read(relion_paths['relion30_path'])
        index_30 = RelionMotl._get_optics_id(spec3_0)
        assert "data_optics" not in spec3_0
        assert index_30 is None

    def test_convert_angles_from_relion(self, relion_paths):
        relion_motl = RelionMotl(relion_paths['relion30_path'])  # Use the 3.0 path
        relion_motl.convert_angles_from_relion(relion_motl.relion_df.copy())

        # Print the raw zxz angles
        relion_angles_np = relion_motl.relion_df.loc[0, ["rlnAngleRot", "rlnAngleTilt", "rlnAnglePsi"]].to_numpy().reshape(1, 3)
        r = rot.from_euler("ZYZ", relion_angles_np, degrees=True)
        zxz_angles_raw = r.as_euler("zxz", degrees=True)
        print(f"Raw zxz angles (psi, theta, phi): {zxz_angles_raw}")

        expected_phi = np.array([-29.80023])
        expected_psi = np.array([3.911201])
        expected_theta = np.array([-120.999041])

        assert np.allclose(relion_motl.df.loc[0, "phi"], expected_phi)
        assert np.allclose(relion_motl.df.loc[0, "theta"], expected_theta)
        assert np.allclose(relion_motl.df.loc[0, "psi"], expected_psi)

    def test_convert_to_relion_synthetic(self):
        # Create synthetic cryoCAT (zxz) angles in degrees
        phi = np.array([10.0, -45.0, 120.5])
        theta = np.array([30.0, 60.0, -90.0])
        psi = np.array([20.0, -30.0, 180.0])
        angles_cryocat = np.stack([psi, theta, phi], axis=1)  # zxz order

        # Initialize RelionMotl and set the angles
        relion_motl = RelionMotl()
        relion_motl.df = pd.DataFrame({'phi': phi, 'theta': theta, 'psi': psi})

        # Create a dummy relion_df to pass to the function
        relion_df_out = pd.DataFrame()

        # Call the conversion function
        relion_df_converted = relion_motl.convert_angles_to_relion(relion_df_out.copy())

        # Calculate expected Relion (ZYZ) angles using scipy
        expected_relion_angles = []
        for i in range(len(phi)):
            zxz_angles = np.array([[psi[i], theta[i], phi[i]]])
            r_zxz = rot.from_euler("zxz", zxz_angles, degrees=True)
            zyz_angles = r_zxz.as_euler("ZYZ", degrees=True)
            expected_relion_angles.append((-zyz_angles[0, 0], zyz_angles[0, 1], -zyz_angles[0, 2]))

        expected_rlnAngleRot = np.array([angles[0] for angles in expected_relion_angles])
        expected_rlnAngleTilt = np.array([angles[1] for angles in expected_relion_angles])
        expected_rlnAnglePsi = np.array([angles[2] for angles in expected_relion_angles])

        # Assert the results
        assert "rlnAngleRot" in relion_df_converted.columns
        assert "rlnAngleTilt" in relion_df_converted.columns
        assert "rlnAnglePsi" in relion_df_converted.columns
        assert np.allclose(relion_df_converted["rlnAngleRot"].to_numpy(), expected_rlnAngleRot, atol=1e-6)
        assert np.allclose(relion_df_converted["rlnAngleTilt"].to_numpy(), expected_rlnAngleTilt, atol=1e-6)
        assert np.allclose(relion_df_converted["rlnAnglePsi"].to_numpy(), expected_rlnAnglePsi, atol=1e-6)

    def test_convert_to_relion_single_angle(self):
        # Create a single set of cryoCAT angles
        phi = np.array([60.0])
        theta = np.array([45.0])
        psi = np.array([30.0])
        angles_cryocat = np.stack([psi, theta, phi], axis=1)

        relion_motl = RelionMotl()
        relion_motl.df = pd.DataFrame({'phi': phi, 'theta': theta, 'psi': psi})
        relion_df_out = pd.DataFrame()
        relion_df_converted = relion_motl.convert_angles_to_relion(relion_df_out.copy())

        zxz_angles = np.array([[psi[0], theta[0], phi[0]]])
        r_zxz = rot.from_euler("zxz", zxz_angles, degrees=True)
        zyz_angles = r_zxz.as_euler("ZYZ", degrees=True)
        expected_rlnAngleRot = np.array([-zyz_angles[0, 0]])
        expected_rlnAngleTilt = np.array([zyz_angles[0, 1]])
        expected_rlnAnglePsi = np.array([-zyz_angles[0, 2]])

        assert np.allclose(relion_df_converted["rlnAngleRot"].to_numpy(), expected_rlnAngleRot, atol=1e-6)
        assert np.allclose(relion_df_converted["rlnAngleTilt"].to_numpy(), expected_rlnAngleTilt, atol=1e-6)
        assert np.allclose(relion_df_converted["rlnAnglePsi"].to_numpy(), expected_rlnAnglePsi, atol=1e-6)

    def test_convert_shifts_relion_30(self):
        # Synthetic Relion 3.0 DataFrame (shifts in pixels)
        data_30 = {
            'rlnOriginX': [1.5, -2.0, 0.0],
            'rlnOriginY': [0.5, 3.0, -1.0],
            'rlnOriginZ': [-0.5, 1.0, 2.0]
        }
        relion_df_30 = pd.DataFrame(data_30)

        # Initialize RelionMotl for version 3.0
        relion_motl_30 = RelionMotl()
        relion_motl_30.version = 3.0
        relion_motl_30.shifts_id_names = ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]
        relion_motl_30.df = pd.DataFrame()  # Initialize self.df

        # Call the conversion function
        relion_motl_30.convert_shifts(relion_df_30.copy())

        # Expected shifts in cryoCAT format (negative of Relion, no pixel size division)
        expected_shift_x = np.array([-1.5, 2.0, -0.0])
        expected_shift_y = np.array([-0.5, -3.0, 1.0])
        expected_shift_z = np.array([0.5, -1.0, -2.0])

        assert np.allclose(relion_motl_30.df["shift_x"].to_numpy(), expected_shift_x)
        assert np.allclose(relion_motl_30.df["shift_y"].to_numpy(), expected_shift_y)
        assert np.allclose(relion_motl_30.df["shift_z"].to_numpy(), expected_shift_z)
        assert all(relion_motl_30.df["shift_x"].fillna(0) == relion_motl_30.df["shift_x"]) # Check for no NaN

    def test_convert_shifts_relion_31_angstroms(self):
        # Synthetic Relion 3.1 DataFrame (shifts in Angstroms)
        data_31 = {
            'rlnOriginXAngst': [15.0, -20.0, 0.0],
            'rlnOriginYAngst': [5.0, 30.0, -10.0],
            'rlnOriginZAngst': [-5.0, 10.0, 20.0]
        }
        relion_df_31 = pd.DataFrame(data_31)
        pixel_size = 1.0  # Angstroms per pixel

        # Initialize RelionMotl for version 3.1
        relion_motl_31 = RelionMotl()
        relion_motl_31.version = 3.1
        relion_motl_31.pixel_size = pixel_size
        relion_motl_31.shifts_id_names = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        relion_motl_31.df = pd.DataFrame()  # Initialize self.df

        # Call the conversion function
        relion_motl_31.convert_shifts(relion_df_31.copy())

        # Expected shifts in cryoCAT format (negative of Relion, divided by pixel size)
        expected_shift_x = np.array([-15.0 / pixel_size, 20.0 / pixel_size, -0.0 / pixel_size])
        expected_shift_y = np.array([-5.0 / pixel_size, -30.0 / pixel_size, 10.0 / pixel_size])
        expected_shift_z = np.array([5.0 / pixel_size, -10.0 / pixel_size, -20.0 / pixel_size])

        assert np.allclose(relion_motl_31.df["shift_x"].to_numpy(), expected_shift_x)
        assert np.allclose(relion_motl_31.df["shift_y"].to_numpy(), expected_shift_y)
        assert np.allclose(relion_motl_31.df["shift_z"].to_numpy(), expected_shift_z)
        assert all(relion_motl_31.df["shift_x"].fillna(0) == relion_motl_31.df["shift_x"]) # Check for no NaN

    def test_convert_shifts_relion_40_angstroms_different_pixel_size(self):
        # Synthetic Relion 4.0 DataFrame (shifts in Angstroms)
        data_40 = {
            'rlnOriginXAngst': [22.5, -35.0],
            'rlnOriginYAngst': [7.5, 42.0],
            'rlnOriginZAngst': [-2.5, 18.0]
        }
        relion_df_40 = pd.DataFrame(data_40)
        pixel_size = 0.5  # Angstroms per pixel

        # Initialize RelionMotl for version 4.0
        relion_motl_40 = RelionMotl()
        relion_motl_40.version = 4.0
        relion_motl_40.pixel_size = pixel_size
        relion_motl_40.shifts_id_names = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        relion_motl_40.df = pd.DataFrame()  # Initialize self.df

        # Call the conversion function
        relion_motl_40.convert_shifts(relion_df_40.copy())

        # Expected shifts in cryoCAT format
        expected_shift_x = np.array([-22.5 / pixel_size, 35.0 / pixel_size])
        expected_shift_y = np.array([-7.5 / pixel_size, -42.0 / pixel_size])
        expected_shift_z = np.array([2.5 / pixel_size, -18.0 / pixel_size])

        assert np.allclose(relion_motl_40.df["shift_x"].to_numpy(), expected_shift_x)
        assert np.allclose(relion_motl_40.df["shift_y"].to_numpy(), expected_shift_y)
        assert np.allclose(relion_motl_40.df["shift_z"].to_numpy(), expected_shift_z)
        assert all(relion_motl_40.df["shift_x"].fillna(0) == relion_motl_40.df["shift_x"]) # Check for no NaN

    def test_convert_shifts_with_nan(self):
        # Synthetic Relion 3.1 DataFrame with NaN values
        data_nan = {
            'rlnOriginXAngst': [10.0, np.nan, 20.0],
            'rlnOriginYAngst': [np.nan, 15.0, np.nan],
            'rlnOriginZAngst': [5.0, np.nan, 25.0]
        }
        relion_df_nan = pd.DataFrame(data_nan)
        pixel_size = 1.0

        relion_motl_nan = RelionMotl()
        relion_motl_nan.version = 3.1
        relion_motl_nan.pixel_size = pixel_size
        relion_motl_nan.shifts_id_names = ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        relion_motl_nan.df = pd.DataFrame()

        relion_motl_nan.convert_shifts(relion_df_nan.copy())

        # Expected shifts with NaN filled as 0
        expected_shift_x = np.array([-10.0, 0.0, -20.0]) / pixel_size
        expected_shift_y = np.array([0.0, -15.0, 0.0]) / pixel_size
        expected_shift_z = np.array([-5.0, 0.0, -25.0]) / pixel_size

        assert np.allclose(relion_motl_nan.df["shift_x"].to_numpy(), expected_shift_x)
        assert np.allclose(relion_motl_nan.df["shift_y"].to_numpy(), expected_shift_y)
        assert np.allclose(relion_motl_nan.df["shift_z"].to_numpy(), expected_shift_z)

    def test_convert_shifts_relion_30_specific_values(self, relion_paths):
        # Create a RelionMotl object from the 3.0 file
        relion_motl = RelionMotl(relion_paths['relion30_path'])

        # Simulate reading only the shift-related columns based on your provided data
        shift_data = {
            'rlnOriginX': [1028.455, 793.955, 115.955],
            'rlnOriginY': [733.955, 948.455, 930.455],
            'rlnOriginZ': [422.955, 387.955, 366.455]
        }
        relion_df_specific = pd.DataFrame(shift_data)
        relion_motl.relion_df = relion_df_specific  # Manually set relion_df

        # Explicitly set version and shift names
        relion_motl.version = 3.0
        relion_motl.shifts_id_names = ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]
        relion_motl.df = pd.DataFrame()  # Initialize self.df

        # Call the conversion function
        relion_motl.convert_shifts(relion_motl.relion_df.copy())

        # Expected precise output (negated values)
        expected_shift_x = np.array([-1028.455, -793.955, -115.955])
        expected_shift_y = np.array([-733.955, -948.455, -930.455])
        expected_shift_z = np.array([-422.955, -387.955, -366.455])

        # Assert the converted shifts with precise equality
        assert np.array_equal(relion_motl.df["shift_x"].to_numpy(), expected_shift_x)
        assert np.array_equal(relion_motl.df["shift_y"].to_numpy(), expected_shift_y)
        assert np.array_equal(relion_motl.df["shift_z"].to_numpy(), expected_shift_z)

        # Check for no NaN values
        assert not relion_motl.df[["shift_x", "shift_y", "shift_z"]].isnull().any().any()

    def test_parse_tomo_id_relion_30_from_file(self, relion_paths):
        # Create a RelionMotl object using the path from the fixture
        relion_motl = RelionMotl(relion_paths['relion30_path'])

        # Assert that the 'tomo_id' column was created
        assert "tomo_id" in relion_motl.df.columns

        # Extract the 'rlnMicrographName' column to determine expected tomo_ids
        micrograph_names = relion_motl.relion_df["rlnMicrographName"].tolist()
        expected_tomo_ids = np.array([float(name.split("/")[-1].split("_")[0]) for name in micrograph_names])

        # Assert the parsed 'tomo_id' values
        assert np.array_equal(relion_motl.df["tomo_id"].to_numpy(), expected_tomo_ids)

    def test_parse_tomo_id_relion_30_from_rlnMicrographName(self):
        data = {
            'rlnMicrographName': [
                '/path/to/tomograms/tomo1_1.mrc',
                '/another/path/tomo2_2.mrc',
                '/yet/another/tomo3_3.mrc'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 3.0
        relion_motl.df = pd.DataFrame()  # Initialize self.df
        relion_motl.relion_df = relion_df  # Manually set relion_df
        relion_motl.tomo_id_name = "rlnMicrographName"
        relion_motl.subtomo_id_name = "rlnImageName" # Dummy

        relion_motl.parse_tomo_id(relion_df.copy())

        expected_tomo_ids = np.array([1.0, 2.0, 3.0])
        assert "tomo_id" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["tomo_id"].to_numpy(), expected_tomo_ids)

    def test_parse_tomo_id_relion_40_from_rlnTomoName(self):
        data = {
            'rlnTomoName': [
                'TS_4',
                'TS_5',
                'TS_6'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 4.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.tomo_id_name = "rlnTomoName"
        relion_motl.subtomo_id_name = "rlnTomoParticleName" # Dummy

        relion_motl.parse_tomo_id(relion_df.copy())

        expected_tomo_ids = np.array([4.0, 5.0, 6.0])
        assert "tomo_id" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["tomo_id"].to_numpy(), expected_tomo_ids)

    def test_parse_tomo_id_relion_30_from_rlnImageName(self):
        data = {
            'rlnImageName': [
                '/path/tomo7_sub1_1.mrc',
                '/another/tomo8_sub2_2.mrc',
                '/yet/another/tomo9_sub3_3.mrc'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 3.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.tomo_id_name = "rlnMicrographName" # Missing
        relion_motl.subtomo_id_name = "rlnImageName"

        relion_motl.parse_tomo_id(relion_df.copy())

        expected_tomo_ids = np.array([7.0, 8.0, 9.0])
        assert "tomo_id" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["tomo_id"].to_numpy(), expected_tomo_ids)

    def test_parse_tomo_id_relion_40_from_rlnTomoParticleName(self):
        data = {
            'rlnTomoParticleName': [
                'TS_10/particle1',
                'TS_11/particle2',
                'TS_12/particle3'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 4.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.tomo_id_name = "rlnTomoName" # Missing
        relion_motl.subtomo_id_name = "rlnTomoParticleName"

        relion_motl.parse_tomo_id(relion_df.copy())

        expected_tomo_ids = np.array([10.0, 11.0, 12.0])
        assert "tomo_id" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["tomo_id"].to_numpy(), expected_tomo_ids)

    def test_parse_subtomo_id_relion_30_unique(self):
        data = {
            'rlnImageName': [
                '/path/tomo1_1_1.mrc',
                '/another/tomo1_2_1.mrc',
                '/yet/another/tomo1_3_1.mrc'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 3.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.subtomo_id_name = "rlnImageName"

        relion_motl.parse_subtomo_id(relion_df.copy())

        expected_subtomo_ids = np.array([1.0, 2.0, 3.0])
        expected_geom3 = np.array([1.0, 2.0, 3.0])
        assert "subtomo_id" in relion_motl.df.columns
        assert "geom3" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["subtomo_id"].to_numpy(), expected_subtomo_ids)
        assert np.array_equal(relion_motl.df["geom3"].to_numpy(), expected_geom3)

    def test_parse_subtomo_id_relion_40_unique(self):
        data = {
            'rlnTomoParticleName': [
                'TS_1/1',
                'TS_1/2',
                'TS_1/3'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 4.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.subtomo_id_name = "rlnTomoParticleName"

        relion_motl.parse_subtomo_id(relion_df.copy())

        expected_subtomo_ids = np.array([1.0, 2.0, 3.0])
        expected_geom3 = np.array([1.0, 2.0, 3.0])
        assert "subtomo_id" in relion_motl.df.columns
        assert "geom3" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["subtomo_id"].to_numpy(), expected_subtomo_ids)
        assert np.array_equal(relion_motl.df["geom3"].to_numpy(), expected_geom3)

    def test_parse_subtomo_id_relion_30_non_unique(self):
        data = {
            'rlnImageName': [
                '/path/tomo1_1_1.mrc',
                '/another/tomo1_1_1.mrc',
                '/yet/another/tomo1_2_1.mrc'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 3.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.subtomo_id_name = "rlnImageName"

        relion_motl.parse_subtomo_id(relion_df.copy())

        expected_subtomo_ids = np.array([1, 2, 3])
        expected_geom3 = np.array([1.0, 1.0, 2.0])
        assert "subtomo_id" in relion_motl.df.columns
        assert "geom3" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["subtomo_id"].to_numpy(), expected_subtomo_ids)
        assert np.array_equal(relion_motl.df["geom3"].to_numpy(), expected_geom3)

    def test_parse_subtomo_id_relion_40_non_unique(self):
        data = {
            'rlnTomoParticleName': [
                'TS_1/1',
                'TS_1/1',
                'TS_1/2'
            ]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 4.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.subtomo_id_name = "rlnTomoParticleName"

        relion_motl.parse_subtomo_id(relion_df.copy())

        expected_subtomo_ids = np.array([1, 2, 3])
        expected_geom3 = np.array([1.0, 1.0, 2.0])
        assert "subtomo_id" in relion_motl.df.columns
        assert "geom3" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["subtomo_id"].to_numpy(), expected_subtomo_ids)
        assert np.array_equal(relion_motl.df["geom3"].to_numpy(), expected_geom3)

    def test_parse_subtomo_id_with_half_sets(self):
        data = {
            'rlnImageName': [
                '/path/tomo1_1_1.mrc',
                '/another/tomo1_2_1.mrc',
                '/yet/another/tomo1_3_1.mrc',
                '/another/tomo1_1_1.mrc',
                '/yet/another/tomo1_2_1.mrc',
                '/path/tomo1_4_1.mrc'
            ],
            'rlnRandomSubset': [1, 2, 1, 2, 1, 2]
        }
        relion_df = pd.DataFrame(data)
        relion_motl = RelionMotl()
        relion_motl.version = 3.0
        relion_motl.df = pd.DataFrame()
        relion_motl.relion_df = relion_df
        relion_motl.subtomo_id_name = "rlnImageName"

        relion_motl.parse_subtomo_id(relion_df.copy())

        expected_subtomo_ids = np.array([1, 2, 3, 4, 5, 6]) # Renumbered due to non-unique and half-sets
        expected_geom3 = np.array([1.0, 2.0, 3.0, 1.0, 2.0, 4.0]) # Original non-unique values
        assert "subtomo_id" in relion_motl.df.columns
        assert "geom3" in relion_motl.df.columns
        assert np.array_equal(relion_motl.df["subtomo_id"].to_numpy(), expected_subtomo_ids)
        assert np.array_equal(relion_motl.df["geom3"].to_numpy(), expected_geom3)

        # Test half-set renumbering more specifically
        relion_motl_hs = RelionMotl()
        relion_motl_hs.version = 3.0
        relion_motl_hs.df = pd.DataFrame()
        relion_motl_hs.relion_df = relion_df
        relion_motl_hs.subtomo_id_name = "rlnImageName"
        relion_motl_hs.parse_subtomo_id(relion_df.copy())

        expected_subtomo_ids_hs = np.array([1, 2, 3, 4, 5, 6]) # Still renumbered due to non-unique
        assert np.array_equal(relion_motl_hs.df["subtomo_id"].to_numpy(), expected_subtomo_ids_hs)

        # Let's try with unique IDs and half-sets
        data_hs_unique = {
            'rlnImageName': [
                '/path/tomo1_1_1.mrc',
                '/another/tomo1_2_1.mrc',
                '/yet/another/tomo1_3_1.mrc',
                '/another/tomo1_4_1.mrc'
            ],
            'rlnRandomSubset': [1, 2, 1, 2]
        }
        relion_df_hs_unique = pd.DataFrame(data_hs_unique)
        relion_motl_hs_unique = RelionMotl()
        relion_motl_hs_unique.version = 3.0
        relion_motl_hs_unique.df = pd.DataFrame()
        relion_motl_hs_unique.relion_df = relion_df_hs_unique
        relion_motl_hs_unique.subtomo_id_name = "rlnImageName"
        relion_motl_hs_unique.parse_subtomo_id(relion_df_hs_unique.copy())

        expected_subtomo_ids_hs_unique = np.array([1, 2, 3, 4]) # Should remain unique if they were
        expected_geom3_hs_unique = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.array_equal(relion_motl_hs_unique.df["subtomo_id"].to_numpy(), expected_subtomo_ids_hs_unique)
        assert np.array_equal(relion_motl_hs_unique.df["geom3"].to_numpy(), expected_geom3_hs_unique)

    def test_parse_subtomo_id_relion_30_from_file(self, relion_paths):
        # Create a RelionMotl object from the 3.0 file
        relion_motl = RelionMotl(relion_paths['relion30_path'])

        # The subtomo IDs should be parsed from the 'rlnImageName' column
        # The format is '/path/tomoID_subtomoID_pixelSize.mrc', so we expect the second number

        # Based on the content you provided:
        # /../01006/01006_0000000_5.36A.mrc -> 0
        # /../01006/01006_0000001_5.36A.mrc -> 1
        # /../01006/01006_0000002_5.36A.mrc -> 2

        expected_subtomo_ids = np.array([0.0, 1.0, 2.0])
        expected_geom3_values = np.array([0.0, 1.0, 2.0])  # Initially, geom3 should be the same

        # Call the parse_subtomo_id function
        relion_motl.parse_subtomo_id(relion_motl.relion_df.copy())

        # Assert that the 'subtomo_id' and 'geom3' columns were created
        assert "subtomo_id" in relion_motl.df.columns
        assert "geom3" in relion_motl.df.columns

        # Assert the parsed 'subtomo_id' values
        assert np.array_equal(relion_motl.df["subtomo_id"].to_numpy(), expected_subtomo_ids)

        # Assert the 'geom3' values (should be the same as subtomo_id initially)
        assert np.array_equal(relion_motl.df["geom3"].to_numpy(), expected_geom3_values)

    def test_convert_to_motl_relion_30_from_file(self, relion_paths):
        # Create a RelionMotl object by loading the Relion 3.0 file
        relion_motl = RelionMotl(relion_paths['relion30_path'])

        # Assert that the 'df' attribute is now a populated DataFrame
        assert isinstance(relion_motl.df, pd.DataFrame)
        assert not relion_motl.df.empty
        assert len(relion_motl.df) == len(relion_motl.relion_df)

        # Assert the presence of key columns in the 'df' attribute
        expected_columns = [
            'x', 'y', 'z',
            'shift_x', 'shift_y', 'shift_z',
            'phi', 'theta', 'psi',
            'tomo_id', 'subtomo_id', 'geom3',
            'class', 'score'
        ]
        for col in expected_columns:
            assert col in relion_motl.df.columns

        # Optionally, you can also check if the 'ccSubtomoID' column exists in the 'relion_df'
        assert 'ccSubtomoID' in relion_motl.relion_df.columns
        assert len(relion_motl.relion_df['ccSubtomoID']) == len(relion_motl.df)

    def test_adapt_original_entries_no_change(self):
        relion_data = {'ccSubtomoID': [1, 2, 3], 'rlnOriginXAngst': [1, 2, 3], 'rlnOriginYAngst': [4, 5, 6],
                       'rlnOriginZAngst': [7, 8, 9]}
        motl_data = {'subtomo_id': [1, 2, 3]}
        relion_df = pd.DataFrame(relion_data)
        motl_df = pd.DataFrame(motl_data)

        relion_motl = RelionMotl()
        relion_motl.relion_df = relion_df.copy()
        relion_motl.df = motl_df.copy()

        updated_df = relion_motl.adapt_original_entries()

        assert len(updated_df) == 3
        assert np.allclose(updated_df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].values,
                           np.zeros((3, 3)))
        assert 'ccSubtomoID' not in updated_df.columns  # Changed assertion
        assert np.array_equal(updated_df.index.to_numpy() + 1,
                              motl_df['subtomo_id'].to_numpy())  # Check index indirectly

    def test_adapt_original_entries_particle_removal(self):
        relion_data = {'ccSubtomoID': [1, 2, 3, 4], 'rlnOriginXAngst': [1, 2, 3, 4], 'rlnOriginYAngst': [5, 6, 7, 8],
                       'rlnOriginZAngst': [9, 10, 11, 12]}
        motl_data = {'subtomo_id': [1, 3]}
        relion_df = pd.DataFrame(relion_data)
        motl_df = pd.DataFrame(motl_data)

        relion_motl = RelionMotl()
        relion_motl.relion_df = relion_df.copy()
        relion_motl.df = motl_df.copy()

        updated_df = relion_motl.adapt_original_entries()

        assert len(updated_df) == 2
        assert np.allclose(updated_df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].values,
                           np.zeros((2, 3)))
        assert 'ccSubtomoID' not in updated_df.columns

        # Check if the rows in updated_df correspond to the subtomo_id in motl_df
        merged_df = updated_df.reset_index().merge(motl_df.reset_index(), left_index=True, right_index=True)
        assert np.array_equal(merged_df['subtomo_id_y'].to_numpy(), motl_df['subtomo_id'].to_numpy())

    def test_adapt_original_entries_particle_reordering(self):
        relion_data = {'ccSubtomoID': [1, 2, 3], 'rlnOriginX': [1, 2, 3], 'rlnOriginY': [4, 5, 6],
                       'rlnOriginZ': [7, 8, 9]}
        motl_data = {'subtomo_id': [3, 1, 2]}
        relion_df = pd.DataFrame(relion_data)
        motl_df = pd.DataFrame(motl_data)

        relion_motl = RelionMotl()
        relion_motl.relion_df = relion_df.copy()
        relion_motl.df = motl_df.copy()

        updated_df = relion_motl.adapt_original_entries()

        assert len(updated_df) == 3
        assert np.allclose(updated_df[['rlnOriginX', 'rlnOriginY', 'rlnOriginZ']].values, np.zeros((3, 3)))
        assert 'ccSubtomoID' not in updated_df.columns

        # Check if the rows in updated_df are reordered according to motl_df['subtomo_id']
        merged_df = updated_df.reset_index().merge(motl_df.reset_index(), left_index=True, right_index=True)
        assert np.array_equal(merged_df['subtomo_id_y'].to_numpy(), motl_df['subtomo_id'].to_numpy())

    def test_adapt_original_entries_no_relion_df(self):
        relion_motl = RelionMotl()
        relion_motl.df = pd.DataFrame({'subtomo_id': [1]})

        with pytest.raises(UserInputError) as excinfo:
            relion_motl.adapt_original_entries()
        assert "There are no original entries for this relion motl" in str(excinfo.value)

    def test_adapt_original_entries_different_shift_column(self):
        relion_data = {'ccSubtomoID': [1, 2], 'rlnOriginX': [1, 2], 'rlnOriginY': [3, 4], 'rlnOriginZ': [5, 6]}
        motl_data = {'subtomo_id': [1, 2]}
        relion_df = pd.DataFrame(relion_data)
        motl_df = pd.DataFrame(motl_data)

        relion_motl = RelionMotl()
        relion_motl.relion_df = relion_df.copy()
        relion_motl.df = motl_df.copy()

        updated_df = relion_motl.adapt_original_entries()

        assert len(updated_df) == 2
        assert np.allclose(updated_df[['rlnOriginX', 'rlnOriginY', 'rlnOriginZ']].values, np.zeros((2, 3)))
        assert 'ccSubtomoID' not in updated_df.columns
        assert np.array_equal(updated_df.index.to_numpy() + 1, motl_df['subtomo_id'].to_numpy())

    def test_adapt_original_entries_v30(self, relion_paths):
        # Load the Relion 3.0 star file
        relion_motl_orig = RelionMotl(relion_paths['relion30_path'])

        # Assume the original relion_df has at least these columns and rows
        original_data = {
            'rlnImageName': ['/path/01006_0000000_5.36A.mrc', '/path/01006_0000001_5.36A.mrc',
                             '/path/01006_0000002_5.36A.mrc'],
            'rlnOriginXAngst': [1.0, 2.0, 3.0],
            'rlnOriginYAngst': [4.0, 5.0, 6.0],
            'rlnOriginZAngst': [7.0, 8.0, 9.0],
            # Add other necessary columns if the function relies on them
            'ccSubtomoID': [0.0, 1.0, 2.0]  # We'll manually add this based on rlnImageName for this test
        }
        relion_df_orig_manual = pd.DataFrame(original_data)

        # Create a modified df with reordered and subsetted subtomo_ids
        modified_subtomo_ids = np.array([2.0, 0.0])
        modified_df = pd.DataFrame({'subtomo_id': modified_subtomo_ids})

        # Create a new RelionMotl object and set its relion_df and df
        relion_motl = RelionMotl()
        relion_motl.relion_df = relion_df_orig_manual.copy()  # Use our manual df
        relion_motl.df = modified_df.copy()

        # Call adapt_original_entries
        updated_df = relion_motl.adapt_original_entries()

        # Expected output DataFrame
        expected_data = {
            'rlnImageName': ['/path/01006_0000002_5.36A.mrc', '/path/01006_0000000_5.36A.mrc'],
            'rlnOriginXAngst': [0.0, 0.0],
            'rlnOriginYAngst': [0.0, 0.0],
            'rlnOriginZAngst': [0.0, 0.0],
            # Other columns should be preserved in their original order corresponding to the kept particles
        }
        expected_df = pd.DataFrame(expected_data)

        # Assertions on the updated_df
        assert len(updated_df) == len(expected_df)
        assert 'ccSubtomoID' not in updated_df.columns

        # Check if the shift columns are zero
        assert np.allclose(updated_df[['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']].values,
                           np.zeros((len(expected_df), 3)), atol=1e-6)

        # Check the order and values of 'rlnImageName' (or another preserved column)
        assert np.array_equal(updated_df['rlnImageName'].to_numpy(), expected_df['rlnImageName'].to_numpy())

        # You can add more assertions for other columns if needed, based on what should be preserved.

    def test_get_version_specific_names(self):
        # Test Relion 3.0
        tomo_id, subtomo_id, shifts, data_spec = RelionMotl.get_version_specific_names(3.0)
        assert tomo_id == "rlnMicrographName"
        assert subtomo_id == "rlnImageName"
        assert shifts == ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]
        assert data_spec == "data_"

        # Test Relion 3.1
        tomo_id, subtomo_id, shifts, data_spec = RelionMotl.get_version_specific_names(3.1)
        assert tomo_id == "rlnMicrographName"
        assert subtomo_id == "rlnImageName"
        assert shifts == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        assert data_spec == "data_particles"

        # Test Relion 4.0 (and higher)
        tomo_id, subtomo_id, shifts, data_spec = RelionMotl.get_version_specific_names(4.0)
        assert tomo_id == "rlnTomoName"
        assert subtomo_id == "rlnTomoParticleName"
        assert shifts == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        assert data_spec == "data_particles"

        # Test a higher version
        tomo_id, subtomo_id, shifts, data_spec = RelionMotl.get_version_specific_names(4.1)
        assert tomo_id == "rlnTomoName"
        assert subtomo_id == "rlnTomoParticleName"
        assert shifts == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        assert data_spec == "data_particles"

        # Test default version (if you have it set) - adjust the expected values if your default is different
        RelionMotl.default_version = 3.1  # Set a default for this test
        tomo_id, subtomo_id, shifts, data_spec = RelionMotl.get_version_specific_names(None)
        assert tomo_id == "rlnMicrographName"
        assert subtomo_id == "rlnImageName"
        assert shifts == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        assert data_spec == "data_particles"
        RelionMotl.default_version = 3.0  # Reset the default

    def test_set_version_specific_names(self):
        relion_motl = RelionMotl()

        relion_motl.version = 3.0
        relion_motl.set_version_specific_names()
        assert relion_motl.tomo_id_name == "rlnMicrographName"
        assert relion_motl.subtomo_id_name == "rlnImageName"
        assert relion_motl.shifts_id_names == ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]
        assert relion_motl.data_spec == "data_"

        relion_motl.version = 3.1
        relion_motl.set_version_specific_names()
        assert relion_motl.tomo_id_name == "rlnMicrographName"
        assert relion_motl.subtomo_id_name == "rlnImageName"
        assert relion_motl.shifts_id_names == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        assert relion_motl.data_spec == "data_particles"

        relion_motl.version = 4.0
        relion_motl.set_version_specific_names()
        assert relion_motl.tomo_id_name == "rlnTomoName"
        assert relion_motl.subtomo_id_name == "rlnTomoParticleName"
        assert relion_motl.shifts_id_names == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
        assert relion_motl.data_spec == "data_particles"

    def test_set_version_specific_names_1(self, relion_paths):
        # Test with a Relion 3.0 file
        relion_motl_30 = RelionMotl(relion_paths['relion30_path'])
        assert relion_motl_30.version <= 3.0
        assert relion_motl_30.tomo_id_name == "rlnMicrographName"
        assert relion_motl_30.subtomo_id_name == "rlnImageName"
        assert relion_motl_30.shifts_id_names == ["rlnOriginX", "rlnOriginY", "rlnOriginZ"]
        assert relion_motl_30.data_spec == "data_"

        # Test with a Relion 3.1 file (you'll need a sample 3.1 file)
        if os.path.exists(relion_paths['relion31_path']):
            relion_motl_31 = RelionMotl(relion_paths['relion31_path'])
            assert relion_motl_31.version == 3.1
            assert relion_motl_31.tomo_id_name == "rlnMicrographName"
            assert relion_motl_31.subtomo_id_name == "rlnImageName"
            assert relion_motl_31.shifts_id_names == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
            assert relion_motl_31.data_spec == "data_particles"
        else:
            pytest.skip("Relion 3.1 test file not found.")

        # Test with a Relion 4.0 file (you'll need a sample 4.0 file)
        if os.path.exists(relion_paths['relion40_path']):
            relion_motl_40 = RelionMotl(relion_paths['relion40_path'], binning=1.0)
            assert relion_motl_40.version >= 4.0
            assert relion_motl_40.tomo_id_name == "rlnTomoName"
            assert relion_motl_40.subtomo_id_name == "rlnTomoParticleName"
            assert relion_motl_40.shifts_id_names == ["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]
            assert relion_motl_40.data_spec == "data_particles"
        else:
            pytest.skip("Relion 4.0 test file not found.")

    def test_create_particles_data_version_30(self):
        relion_motl = RelionMotl()
        num_particles = 5
        relion_motl.df = pd.DataFrame({'subtomo_id': range(num_particles)})

        particles_df = relion_motl.create_particles_data(version=3.0)

        assert isinstance(particles_df, pd.DataFrame)
        assert len(particles_df) == num_particles
        assert list(particles_df.columns) == RelionMotl.columns_v3_0
        assert np.all(particles_df.values == 0)

    def test_create_particles_data_version_31(self):
        relion_motl = RelionMotl()
        num_particles = 3
        relion_motl.df = pd.DataFrame({'subtomo_id': range(num_particles)})

        particles_df = relion_motl.create_particles_data(version=3.1)

        assert isinstance(particles_df, pd.DataFrame)
        assert len(particles_df) == num_particles
        assert list(particles_df.columns) == RelionMotl.columns_v3_1
        assert np.all(particles_df.values == 0)

    def test_create_particles_data_version_4_or_higher(self):
        relion_motl = RelionMotl()
        num_particles = 7
        relion_motl.df = pd.DataFrame({'subtomo_id': range(num_particles)})

        particles_df = relion_motl.create_particles_data(version=4.0)
        assert isinstance(particles_df, pd.DataFrame)
        assert len(particles_df) == num_particles
        assert list(particles_df.columns) == RelionMotl.columns_v4
        assert np.all(particles_df.values == 0)

        with pytest.raises(Exception):
            particles_df_higher = relion_motl.create_particles_data(version=4.2)

    class TestPrepareOpticsData:
        @pytest.fixture
        def optics_file_paths(self):
            motl_dir = "./test_data/motl_data/"
            return {
                'optics30_path': os.path.join(motl_dir, "optics_v30.star"),
                'optics31_path': os.path.join(motl_dir, "optics_v31.star"),
            }

        @pytest.fixture
        def create_dummy_optics_files(self, optics_file_paths):
            # Create a dummy Relion 3.0 optics file
            optics_content_v30 = """
                data_
            
                loop_
                _rlnMagnification #1
                _rlnVoltage #2
                10000 300
            """
            with open(optics_file_paths['optics30_path'], 'w') as f:
                f.write(optics_content_v30)

            # Create a dummy Relion 3.1+ optics file
            optics_content_v31 = """
                data_optics
            
                loop_
                _rlnMagnification #1
                _rlnVoltage #2
                12000 200
            """
            with open(optics_file_paths['optics31_path'], 'w') as f:
                f.write(optics_content_v31)

        def test_prepare_optics_data_use_original_entries_existing(self):
            relion_motl = RelionMotl()
            optics_df_orig = pd.DataFrame({
                'rlnOpticsGroup': [1, 2],
                'rlnOpticsGroupName': ['opticsGroup1', 'opticsGroup2'],
                'rlnSphericalAberration': [2.7, 2.7],
                'rlnVoltage': [300.0, 300.0],
                'rlnImagePixelSize': [2.446, 3.446],
                'rlnImageSize': [168, 168],
                'rlnImageDimensionality': [3, 3]
            })
            relion_motl.optics_data = optics_df_orig.copy()
            optics_df = relion_motl.prepare_optics_data()
            pd.testing.assert_frame_equal(optics_df, optics_df_orig)

        def test_prepare_optics_data_use_original_entries_missing_warning(self):
            relion_motl = RelionMotl()
            with pytest.raises(Warning):
                relion_motl.prepare_optics_data()

        def test_prepare_optics_data_from_starfile_v30(self, optics_file_paths, create_dummy_optics_files):
            relion_motl = RelionMotl()
            relion_motl.version = 3.0
            optics_df = relion_motl.prepare_optics_data(use_original_entries=False,
                                                        optics_data=optics_file_paths['optics30_path'])
            expected_df = pd.DataFrame({'rlnMagnification': [10000], 'rlnVoltage': [300]})
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_from_starfile_v31(self, optics_file_paths, create_dummy_optics_files):
            relion_motl = RelionMotl()
            relion_motl.version = 3.1
            optics_df = relion_motl.prepare_optics_data(use_original_entries=False,
                                                        optics_data=optics_file_paths['optics31_path'])
            expected_df = pd.DataFrame({'rlnMagnification': [12000], 'rlnVoltage': [200]})
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_from_dict(self):
            relion_motl = RelionMotl()
            optics_dict = {'rlnMagnification': [12000], 'rlnVoltage': [200]}
            optics_df = relion_motl.prepare_optics_data(use_original_entries=False, optics_data=optics_dict)
            expected_df = pd.DataFrame(optics_dict)
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_invalid_optics_data_type_error(self):
            relion_motl = RelionMotl()
            with pytest.raises(UserInputError,
                               match="Optics has to be specified as a dictionary or as a path to the starfile."):
                relion_motl.prepare_optics_data(use_original_entries=False, optics_data=123)

        def test_prepare_optics_data_no_optics_data_v31_calls_create(self, monkeypatch):
            relion_motl = RelionMotl()
            relion_motl.version = 3.1
            expected_df = pd.DataFrame({'rlnOpticsGroup': [1]})
            with monkeypatch.context() as m:
                m.setattr(relion_motl, 'create_optics_group_v3_1', lambda: pd.DataFrame({'rlnOpticsGroup': [1]}))
                optics_df = relion_motl.prepare_optics_data(use_original_entries=False, optics_data=None)
                pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_no_optics_data_v_greater_than_31_calls_create(self, monkeypatch):
            relion_motl = RelionMotl()
            relion_motl.version = 4.0
            expected_df = pd.DataFrame({'rlnOpticsGroup': [1]})
            with monkeypatch.context() as m:
                m.setattr(relion_motl, 'create_optics_group_v4', lambda: pd.DataFrame({'rlnOpticsGroup': [1]}))
                optics_df = relion_motl.prepare_optics_data(use_original_entries=False, optics_data=None)
                pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_no_optics_data_v30_warning(self):
            relion_motl = RelionMotl()
            relion_motl.version = 3.0
            with pytest.raises(Warning):
                relion_motl.prepare_optics_data(use_original_entries=False, optics_data=None)

        def test_prepare_optics_data_explicit_version_from_file(self, optics_file_paths, create_dummy_optics_files):
            relion_motl = RelionMotl()
            optics_df_v31 = relion_motl.prepare_optics_data(use_original_entries=False,
                                                            optics_data=optics_file_paths['optics31_path'], version=3.1)
            expected_df_v31 = pd.DataFrame({'rlnMagnification': [12000], 'rlnVoltage': [200]})
            pd.testing.assert_frame_equal(optics_df_v31, expected_df_v31)

            relion_motl.version = 4.0  # Set an instance version, but override with explicit version
            optics_df_v30 = relion_motl.prepare_optics_data(use_original_entries=False,
                                                            optics_data=optics_file_paths['optics30_path'], version=3.0)
            expected_df_v30 = pd.DataFrame({'rlnMagnification': [10000], 'rlnVoltage': [300]})
            pd.testing.assert_frame_equal(optics_df_v30, expected_df_v30)

        def test_prepare_optics_data_no_optics_data_no_version_uses_self_version_v31(self, monkeypatch):
            relion_motl = RelionMotl()
            relion_motl.version = 3.1
            expected_df = pd.DataFrame({'rlnOpticsGroup': [1]})
            with monkeypatch.context() as m:
                m.setattr(relion_motl, 'create_optics_group_v3_1', lambda: pd.DataFrame({'rlnOpticsGroup': [1]}))
                optics_df = relion_motl.prepare_optics_data(use_original_entries=False, optics_data=None, version=None)
                pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_no_optics_data_no_version_uses_self_version_v4(self, monkeypatch):
            relion_motl = RelionMotl()
            relion_motl.version = 4.0
            expected_df = pd.DataFrame({'rlnOpticsGroup': [1]})
            with monkeypatch.context() as m:
                m.setattr(relion_motl, 'create_optics_group_v4', lambda: pd.DataFrame({'rlnOpticsGroup': [1]}))
                optics_df = relion_motl.prepare_optics_data(use_original_entries=False, optics_data=None, version=None)
                pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_prepare_optics_data_use_original_entries_true_optics_data_none_warning(self):
            relion_motl = RelionMotl()
            relion_motl.optics_data = None
            with pytest.raises(Warning):
                relion_motl.prepare_optics_data(use_original_entries=True)  # Default is True, but let's be explicit

        def test_prepare_optics_data_use_original_entries_false_optics_data_none_version_30_warning(self):
            relion_motl = RelionMotl()
            with pytest.raises(Warning):
                relion_motl.prepare_optics_data(use_original_entries=False, optics_data=None, version=3.0)
        def test_cleanup(self):
            if os.path.exists("./test_data/motl_data/optics_v30.star"):
                os.remove("./test_data/motl_data/optics_v30.star")
            if os.path.exists("./test_data/motl_data/optics_v31.star"):
                os.remove("./test_data/motl_data/optics_v31.star")

    def test_prepare_particles_data(self, relion_paths):
        # Initialize a RelionMotl object by loading from a Relion .star file
        rln_motl = RelionMotl(input_motl=relion_paths["relion30_path"], pixel_size=1.5, version=3.1)

        # 1: empty format strings
        df_empty_format = rln_motl.prepare_particles_data()
        assert "rlnMicrographName" in df_empty_format.columns
        assert "rlnImageName" in df_empty_format.columns
        assert len(df_empty_format) == 3
        assert "rlnPixelSize" in df_empty_format.columns
        assert df_empty_format["rlnPixelSize"].iloc[0] == 1.5


        assert "rlnOriginXAngst" in df_empty_format.columns
        assert "rlnOriginYAngst" in df_empty_format.columns
        assert "rlnOriginZAngst" in df_empty_format.columns
        assert np.all(df_empty_format[["rlnOriginXAngst", "rlnOriginYAngst", "rlnOriginZAngst"]].values == 0)

        # 2: basic formatting with padding
        df_padded_format = rln_motl.prepare_particles_data(
            tomo_format="/path/to/tomo_$xxxx.rec",
            subtomo_format="/path/to/subtomo_$yyyy.mrc"
        )
        assert list(df_padded_format["rlnMicrographName"]) == ["/path/to/tomo_1006.rec"] * 3
        assert list(df_padded_format["rlnImageName"]) == ["/path/to/subtomo_0000.mrc", "/path/to/subtomo_0001.mrc",
                                                          "/path/to/subtomo_0002.mrc"]

        # 3: combined
        df_combined_format = rln_motl.prepare_particles_data(
            tomo_format="/base/tomo_$xx.rec",
            subtomo_format="/base/tomo_$xx_sub_$yy.mrc"
        )
        assert list(df_combined_format["rlnMicrographName"]) == ["/base/tomo_1006.rec"] * 3
        assert list(df_combined_format["rlnImageName"]) == ["/base/tomo_1006_sub_00.mrc", "/base/tomo_1006_sub_01.mrc",
                                                            "/base/tomo_1006_sub_02.mrc"]

        #--
        rln_motl_no_pixelsize = RelionMotl(input_motl=relion_paths["relion30_path"], version=3.1)
        df_pixelsize_arg = rln_motl_no_pixelsize.prepare_particles_data(pixel_size=2.0)
        assert "rlnPixelSize" in df_pixelsize_arg.columns
        assert df_pixelsize_arg["rlnPixelSize"].iloc[0] == 2.0

        rln_motl_v4_no_pixelsize = RelionMotl(input_motl=relion_paths["relion40_path"], version=4.0, binning=1.0)
        df_v4_pixelsize_arg = rln_motl_v4_no_pixelsize.prepare_particles_data(pixel_size=2.0)
        assert "rlnPixelSize" not in df_v4_pixelsize_arg.columns

        #different padding lengths
        df_diff_padding = rln_motl.prepare_particles_data(
            tomo_format="/data/tomo_$x.rec",
            subtomo_format="/data/subtomo_$yy.mrc"
        )
        assert list(df_diff_padding["rlnMicrographName"]) == [
            "/data/tomo_1006.rec"] * 3  # Assuming single 'x' doesn't trigger padding
        assert list(df_diff_padding["rlnImageName"]) == ["/data/subtomo_00.mrc", "/data/subtomo_01.mrc",
                                                         "/data/subtomo_02.mrc"]

        #multiple sequences
        """df_multiple_sequences = rln_motl.prepare_particles_data(
            tomo_format="/a_$xx$_b_$xxxx$_c.rec",
            subtomo_format="/p_$yyy$_q_$y$_r.mrc"
        )
        assert list(df_multiple_sequences["rlnMicrographName"]) == ["/a_10$_b_1006$_c.rec"] * 3
        assert list(df_multiple_sequences["rlnImageName"]) == ["/p_000$_q_0$_r.mrc", "/p_001$_q_1$_r.mrc",
                                                               "/p_002$_q_2$_r.mrc"]"""

        #incorrect format but not empty
        with pytest.raises(ValueError):
            rln_motl.prepare_particles_data(subtomo_format="/path/to/subtomo.mrc")
        with pytest.raises(ValueError):
            rln_motl.prepare_particles_data(tomo_format="/path/to/tomo.rec")

        #v4
        rln_motl_v4 = RelionMotl(input_motl=relion_paths["relion40_path"], version=4.0, binning=1.0)
        df_v4 = rln_motl_v4.prepare_particles_data(
            tomo_format="tomo_{$xx}",
            subtomo_format="particle_{$yyy}"
        )
        assert "rlnTomoName" in df_v4.columns
        assert "rlnTomoParticleName" in df_v4.columns
        assert "rlnMicrographName" not in df_v4.columns
        assert "rlnImageName" not in df_v4.columns

        #pixel size overriding?
        df_override_pixelsize = rln_motl.prepare_particles_data(pixel_size=2.5)
        assert "rlnPixelSize" in df_override_pixelsize.columns
        assert df_override_pixelsize["rlnPixelSize"].iloc[0] == 2.5

        # empty tomo, not empty subtomo
        df_empty_tomo_format = rln_motl.prepare_particles_data(
            tomo_format="",
            subtomo_format="/tomo_$x/subtomo_$yy.mrc"
        )
        assert list(df_empty_tomo_format["rlnMicrographName"]) == [1006] * 3  # Expecting integers
        assert list(df_empty_tomo_format["rlnImageName"]) == ["/tomo_1006/subtomo_00.mrc", "/tomo_1006/subtomo_01.mrc",
                                                              "/tomo_1006/subtomo_02.mrc"]

        #empty subtomo, not empty tomo
        df_empty_subtomo_format = rln_motl.prepare_particles_data(
            tomo_format="/tomo_$xx.rec",
            subtomo_format=""
        )
        assert list(df_empty_subtomo_format["rlnMicrographName"]) == ["/tomo_1006.rec"] * 3
        assert list(df_empty_subtomo_format["rlnImageName"]) == [0, 1, 2]  # subtomo_id defaults to index

        #pixelsetting
        rln_motl_v4_no_init_pixelsize = RelionMotl(input_motl=relion_paths["relion40_path"], version=4.0, binning=1.0)
        df_v4_pixelsize_arg = rln_motl_v4_no_init_pixelsize.prepare_particles_data(pixel_size=3.0)
        assert "rlnPixelSize" not in df_v4_pixelsize_arg.columns

    class TestCreateOpticsGroupV31:
        def test_create_optics_group_v31_defaults(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 1.5  # Set a default pixel size
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnImagePixelSize": [1.5],
                "rlnImageSize": ["NaN"],
                "rlnImageDimensionality": [3],
            })
            optics_df = relion_motl.create_optics_group_v3_1()
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v31_with_parameters(self):
            relion_motl = RelionMotl()
            pixel_size = 2.0
            subtomo_size = 128
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnImagePixelSize": [2.0],
                "rlnImageSize": [128],
                "rlnImageDimensionality": [3],
            })
            optics_df = relion_motl.create_optics_group_v3_1(pixel_size=pixel_size, subtomo_size=subtomo_size)
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v31_pixel_size_none(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 3.0
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnImagePixelSize": [3.0],
                "rlnImageSize": ["NaN"],
                "rlnImageDimensionality": [3],
            })

            optics_df = relion_motl.create_optics_group_v3_1(pixel_size=None)
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v31_subtomo_size_none(self):
            relion_motl = RelionMotl()
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnImagePixelSize": [1.0],
                "rlnImageSize": ["NaN"],
                "rlnImageDimensionality": [3],
            })
            optics_df = relion_motl.create_optics_group_v3_1(subtomo_size=None)
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v31_pixel_size_none_instance_none(self):
            relion_motl = RelionMotl()
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnImagePixelSize": [1.0],
                "rlnImageSize": ["NaN"],
                "rlnImageDimensionality": [3],
            })
            optics_df = relion_motl.create_optics_group_v3_1(pixel_size=None, subtomo_size=None)
            pd.testing.assert_frame_equal(optics_df, expected_df)

    class TestCreateOpticsGroupV4:
        def test_create_optics_group_v4_with_parameters(self):
            relion_motl = RelionMotl()
            pixel_size = 2.0
            subtomo_size = 128
            binning = 2
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnTomoTiltSeriesPixelSize": [1.0],  # 2.0 / 2
                "rlnCtfDataAreCtfPremultiplied": [1],
                "rlnImageDimensionality": [3],
                "rlnTomoSubtomogramBinning": [2],
                "rlnImagePixelSize": [2.0],
                "rlnImageSize": [128],
            })
            optics_df = relion_motl.create_optics_group_v4(
                pixel_size=pixel_size, subtomo_size=subtomo_size, binning=binning
            )
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v4_defaults(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 1.5
            relion_motl.binning = 1
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnTomoTiltSeriesPixelSize": [1.5],  # 1.5 / 1
                "rlnCtfDataAreCtfPremultiplied": [1],
                "rlnImageDimensionality": [3],
                "rlnTomoSubtomogramBinning": [1],
                "rlnImagePixelSize": [1.5],
                "rlnImageSize": ["NaN"],
            })
            optics_df = relion_motl.create_optics_group_v4()
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v4_pixel_size_none(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 3.0
            relion_motl.binning = 2
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnTomoTiltSeriesPixelSize": [1.5],  # 3.0 / 2
                "rlnCtfDataAreCtfPremultiplied": [1],
                "rlnImageDimensionality": [3],
                "rlnTomoSubtomogramBinning": [2],
                "rlnImagePixelSize": [3.0],
                "rlnImageSize": ["NaN"],
            })
            optics_df = relion_motl.create_optics_group_v4(pixel_size=None)
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v4_binning_none(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 2.0  # <--- Setting self.pixel_size here
            relion_motl.binning = 3  # <--- Setting self.binning here
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnTomoTiltSeriesPixelSize": [2.0 / 3],  # <--- Direct float here
                "rlnCtfDataAreCtfPremultiplied": [1],
                "rlnImageDimensionality": [3],
                "rlnTomoSubtomogramBinning": [3],
                "rlnImagePixelSize": [2.0],
                "rlnImageSize": ["NaN"],
            })
            optics_df = relion_motl.create_optics_group_v4(binning=None)  # <--- binning argument is None
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v4_subtomo_size_none(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 1.0
            relion_motl.binning = 1
            expected_df = pd.DataFrame({
                "rlnOpticsGroup": [1],
                "rlnOpticsGroupName": ["opticsGroup1"],
                "rlnSphericalAberration": [2.7],
                "rlnVoltage": [300.0],
                "rlnTomoTiltSeriesPixelSize": [1.0],
                "rlnCtfDataAreCtfPremultiplied": [1],
                "rlnImageDimensionality": [3],
                "rlnTomoSubtomogramBinning": [1],
                "rlnImagePixelSize": [1.0],
                "rlnImageSize": ["NaN"],
            })
            optics_df = relion_motl.create_optics_group_v4(subtomo_size=None)
            pd.testing.assert_frame_equal(optics_df, expected_df)

        def test_create_optics_group_v4_pixel_size_and_binning_none(self):
            relion_motl = RelionMotl()
            with pytest.raises(TypeError):
                relion_motl.create_optics_group_v4(pixel_size=None, binning=None)

        def test_create_optics_group_v4_pixel_size_set_binning_none(self):
            relion_motl = RelionMotl()
            relion_motl.pixel_size = 2.0
            with pytest.raises(TypeError):
                relion_motl.create_optics_group_v4(binning=None)

    def test_create_final_output_no_optics(self):
        relion_motl = RelionMotl()
        relion_motl.data_spec = "particle_data"
        sample_relion_df = pd.DataFrame({"particleId": [1, 2]})
        frames, specifiers = relion_motl.create_final_output(sample_relion_df)
        assert len(frames) == 1
        pd.testing.assert_frame_equal(frames[0], sample_relion_df)
        assert specifiers == ["particle_data"]

    def test_create_final_output_v31_or_higher(self):
        relion_motl = RelionMotl()
        relion_motl.version = 3.1
        relion_motl.data_spec = "particle_data"
        sample_optics_df = pd.DataFrame({"opticsId": [1]})
        sample_relion_df = pd.DataFrame({"particleId": [1, 2]})
        frames, specifiers = relion_motl.create_final_output(sample_relion_df, optics_df=sample_optics_df)
        assert len(frames) == 2
        pd.testing.assert_frame_equal(frames[0], sample_optics_df)
        pd.testing.assert_frame_equal(frames[1], sample_relion_df)
        assert specifiers == ["data_optics", "particle_data"]

    def test_create_final_output_v30_or_lower(self):
        relion_motl = RelionMotl()
        relion_motl.version = 3.0
        relion_motl.data_spec = "combined_data"
        sample_optics_df = pd.DataFrame({"commonCol": [1, 2], "opticsSpec": ["a", "b"]})
        sample_relion_df = pd.DataFrame({"commonCol": [2, 3], "particleSpec": ["x", "y"]})
        expected_combined_df = pd.DataFrame(
            {"commonCol": [1, 2, 2, 3], "opticsSpec": ["a", "b", None, None], "particleSpec": [None, None, "x", "y"]}
        ).reset_index(drop=True)
        frames, specifiers = relion_motl.create_final_output(sample_relion_df, optics_df=sample_optics_df)
        assert len(frames) == 1
        pd.testing.assert_frame_equal(frames[0].sort_index(axis=1), expected_combined_df.sort_index(axis=1))
        assert specifiers == ["combined_data"]

        # Test with duplicates
        sample_optics_df_dup = pd.DataFrame({"commonCol": [1, 2], "opticsSpec": ["a", "b"]})
        sample_relion_df_dup = pd.DataFrame({"commonCol": [2, 3], "particleSpec": ["x", "y"]})
        expected_combined_df_dup = pd.DataFrame(
            {"commonCol": [1, 2, 2, 3], "opticsSpec": ["a", "b", None, None], "particleSpec": [None, None, "x", "y"]}
        ).drop_duplicates().reset_index(drop=True)
        frames_dup, specifiers_dup = relion_motl.create_final_output(sample_relion_df_dup,
                                                                     optics_df=sample_optics_df_dup)
        assert len(frames_dup) == 1
        pd.testing.assert_frame_equal(frames_dup[0].sort_index(axis=1), expected_combined_df_dup.sort_index(axis=1))
        assert specifiers_dup == ["combined_data"]

    def test_create_relion_df_default_v30(self, relion_paths):
        # Load the relion_3.0.star file using RelionMotl
        motl = RelionMotl(input_motl=relion_paths["relion30_path"], version=3.0)
        relion_df = motl.create_relion_df()
        assert list(relion_df.columns) == motl.columns_v3_0
        expected_num_particles = 3
        assert len(relion_df) == expected_num_particles

        assert relion_df["rlnMicrographName"].iloc[0] == 1006

        expected_coordinate_x_1 = 793.955
        assert np.isclose(relion_df["rlnCoordinateX"].iloc[1], expected_coordinate_x_1)
        assert relion_df["rlnImageName"].iloc[2] == 2
        expected_pixel_size = 5.36400
        assert np.allclose(relion_df["rlnPixelSize"].astype(float), expected_pixel_size)
        assert np.allclose(relion_df[["rlnOriginX", "rlnOriginY", "rlnOriginZ"]].values, 0)

        #case2
        relion_df_formatted = motl.create_relion_df(tomo_format="tomo_$xxxx.mrc", subtomo_format="particle_$yyy.mrc")
        assert "tomo_1006.mrc" in relion_df_formatted["rlnMicrographName"].iloc[0]
        assert "particle_00" in relion_df_formatted["rlnImageName"].iloc[0]

        #original entries
        relion_original = motl.create_relion_df(use_original_entries=True)
        expected_cols = ['rlnMicrographName', 'rlnCoordinateX', 'rlnCoordinateY',
                         'rlnCoordinateZ', 'rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi',
                         'rlnMagnification', 'rlnDetectorPixelSize', 'rlnCtfMaxResolution',
                         'rlnImageName', 'rlnCtfImage', 'rlnPixelSize', 'rlnVoltage',
                         'rlnSphericalAberration', 'rlnOriginX', 'rlnOriginY', 'rlnOriginZ',
                         'rlnClassNumber']

        assert list(relion_original.columns) == expected_cols
        assert relion_original.loc[0, "rlnMicrographName"] == "/STORAGE/anschwar/cryoET/bpbrain/rat/processing/E4/80S/warp/mrc/reconstruction/01006_10.73Apx.mrc"
        assert (relion_original["rlnVoltage"] == 300.00).all()
        assert (relion_original["rlnSphericalAberration"] == 2.70).all()
        assert relion_original.loc[0, "rlnCtfImage"] == "../warp/mrc/subtomo/01006/01006_0000000_ctf_5.36A.mrc"
        #all entries
        relion_original_all = motl.create_relion_df(use_original_entries=True, keep_all_entries=True)

        #If keep_all_entries = True, then rlnClassNumber should not be inside the df
        relion_original_without_class = relion_original.drop(columns=['rlnClassNumber'], errors='ignore')
        pd.testing.assert_frame_equal(relion_original_without_class, relion_original_all)
        #all entries adapt
        motl = RelionMotl(input_motl=relion_paths["relion30_path"], version=3.0)
        relion_original_all_adapt = motl.create_relion_df(use_original_entries=True, keep_all_entries=True, adapt_object_attr=True)
        pd.testing.assert_frame_equal(relion_original_without_class, relion_original_all_adapt)
        #the output should not have rlnClassNumber, and self.relion_df should have subtomo_id
        # Compare the relevant columns after adaptation
        columns_to_compare = relion_original_without_class.columns
        pd.testing.assert_frame_equal(motl.relion_df[columns_to_compare], relion_original_without_class)
        assert "subtomo_id" in motl.relion_df.columns

    def test_write_out(self, relion_paths):
        output_path30 = "./test_data/motl_data/out_3_0.star"
        two_output_path30 = "./test_data/motl_data/two_out_3_0.star"
        output_path31 = "./test_data/motl_data/out_3_1.star"
        output_path40 = "./test_data/motl_data/out_4_0.star"
        motl40 = RelionMotl(relion_paths["relion40_path"], binning=1.0)
        motl31 = RelionMotl(relion_paths["relion31_path"])
        motl30 = RelionMotl(relion_paths["relion30_path"])

        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', None)

        #print(motl3.relion_df)
        #print(motl3.df["class"].to_numpy())
        motl30.write_out(
            output_path=output_path30,
            write_optics=False,
            version=3.0
        )
        #3.0 does not support optics_Data reading: should throw a warning
        with pytest.raises(Warning):
            motl30.write_out(
                output_path=two_output_path30,
                use_original_entries=True,
                version=3.0
            )

        motl30.write_out(
            output_path=two_output_path30,
            use_original_entries=True,
            write_optics=False,
            version=3.0
        )
        # FIXME: class column contains nan: wait for explanation from beata
        #two_loaded_motl30= RelionMotl(input_motl=two_output_path30, version=3.0)
        #loaded_motl30 = RelionMotl(input_motl=output_path30, version=3.0)
        #assert list(loaded_motl.relion_df.columns) == RelionMotl.columns_v3_0
        # rlnClassNumber values are all nan: it means they aren't specified i guess
        # print(loaded_motl.df)

        motl31.write_out(
            output_path=output_path31,
            write_optics=False,
            version=3.1
        )
        loaded_motl31 = RelionMotl(input_motl=output_path31, version=3.1)
        assert list(loaded_motl31.relion_df.columns) == RelionMotl.columns_v3_1 + ["ccSubtomoID"]
        motl40.write_out(
            output_path=output_path40,
            write_optics=False,
            version=4.0,
            binning= 2
        )
        loaded_motl40 = RelionMotl(input_motl=output_path40, version=4.0, binning=1.0)
        assert list(loaded_motl40.relion_df.columns) == RelionMotl.columns_v4 + ["ccSubtomoID"]

        #Test write_optics=True
        output_path31_2 = "./test_data/motl_data/out_3_1_optics.star"
        motl31 = RelionMotl(relion_paths["relion31_path"])
        print("\n")
        print(motl31.optics_data)
        motl31.write_out(
            output_path=output_path31_2,
            write_optics=True,
            version=3.1,
            optics_data=relion_paths["relion31_path"]
        )
        loaded_motl31_2 = RelionMotl(input_motl=output_path31_2, version=3.1)

        output_path40_2 = "./test_data/motl_data/out_4_0_optics.star"
        motl40 = RelionMotl(relion_paths["relion40_path"], binning=2.0)
        print(motl40.binning)
        #BINNING MUST BE SPECIFIED FOR WRITING 4.0 ! In general if reading a file
        #binning is never assigned!
        # to fix
        motl40.write_out(
            output_path=output_path40_2,
            write_optics=True,
            version=4.0
        )
        loaded_motl40_2 = RelionMotl(input_motl=output_path40_2, version=4.0, binning=1.0)

        # Test if to keep original relion the dataframes are the same?
        motl31 = RelionMotl(relion_paths["relion31_path"])
        output_path31_1 = "./test_data/motl_data/out_3_1_optics.star"
        motl31.write_out(
            output_path=output_path31_1,
            write_optics=True,
            use_original_entries=True,
            optics_data=relion_paths["relion31_path"],
            version=3.1
        )
        loaded_motl31_1 = RelionMotl(input_motl=output_path31_1, version=3.1)
        assert list(loaded_motl31_1.relion_df.columns) == [
            "rlnMicrographName",
            "rlnCoordinateX",
            "rlnCoordinateY",
            "rlnCoordinateZ",
            "rlnAngleRot",
            "rlnAngleTilt",
            "rlnAnglePsi",
            "rlnCtfMaxResolution",
            "rlnImageName",
            "rlnCtfImage",
            "rlnPixelSize",
            "rlnOpticsGroup",
            "rlnGroupNumber",
            "rlnOriginXAngst",
            "rlnOriginYAngst",
            "rlnOriginZAngst",
            "rlnClassNumber",
            "rlnNormCorrection",
            "rlnRandomSubset",
            "rlnLogLikeliContribution",
            "rlnMaxValueProbDistribution",
            "rlnNrOfSignificantSamples",
            "ccSubtomoID"
        ]

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
        assert Motl.check_df_correct_format(sg.df) == True
        assert list(sg.sg_df.columns) == StopgapMotl.columns
        print(sg.sg_df)
    def test_read_in(self, tmp_path, sample_stopgap_data):
        # Create a temporary STAR file with the sample data
        star_path = tmp_path / "test.star"
        starfileio.Starfile.write([sample_stopgap_data], str(star_path), specifiers=["data_stopgap_motivelist"])


        # Assert that the read DataFrame is equal to the original sample data
        pd.testing.assert_frame_equal(StopgapMotl.read_in(str(star_path)), sample_stopgap_data)

        #read real file
        stopgap_motl = StopgapMotl("./test_data/motl_data/class6_er_mr1_1_sg.star")
        assert list(stopgap_motl.sg_df.columns) == StopgapMotl.columns
        print(stopgap_motl.sg_df)

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
        assert "geom3" not in stopgap_motl.sg_df.columns

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
        assert "geom3" not in stopgap_motl.sg_df.columns

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
        pd.testing.assert_frame_equal(read_motl.df, stopgap_motl.df.fillna(0), check_dtype=False)

    @pytest.fixture
    def motl_df(self):
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
        return pd.DataFrame(sample_data)
    def test_reconvert(self, motl_df):
        pd.set_option('display.max_rows', None)  # Print all rows
        pd.set_option('display.max_columns', None)  # Print all columns

        test_file_path = "./test_data/motl_data/"
        motl_df_sg = StopgapMotl(motl_df)
        print("\n")
        print(motl_df_sg.df)
        assert motl_df_sg.sg_df.empty  # OK - normal

        motl_df_sg.write_out(output_path=test_file_path + "test1_sg.star")
        mod11 = StopgapMotl(input_motl=test_file_path + "test1_sg.star")
        # print(mod11.mod_df)
        print(mod11.df)
        # fixme: the issue is when reloading?
        columns_to_compare = motl_df_sg.df.columns[
            ~(motl_df_sg.df.isna().all() | mod11.df.isna().all())
        ]
        pd.testing.assert_frame_equal(
            motl_df_sg.df[columns_to_compare],
            mod11.df[columns_to_compare],
            check_dtype=False
        )
        if os.path.exists(test_file_path + "test1_sg.star"):
            os.remove(test_file_path + "test1_sg.star")

class TestDynamoMotl:
    def test_constructor1(self):
        pd.set_option('display.max_rows', None)  # Print all rows
        pd.set_option('display.max_columns', None)  # Print all columns

        pathfile = "./test_data/motl_data/b4_motl_CR_tm_topbott_clean600_1_dynamo.tbl"
        test1 = DynamoMotl(pathfile)
        print(test1.df)
        #print(test1.dynamo_df)


        with pytest.raises(Exception):
            excp = DynamoMotl("test")

        pathfile = "./test_data/motl_data/bin1.6_SG_j013_afterM_24k_subtomos_MN_Dyn_v2.tbl"
        test2 = DynamoMotl(pathfile)
        print(test2.df)

        #print(test2.dynamo_df)


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

    def test_write_out(self):
        # Test using a real EM file pandas dataframe as input
        em_motl_test = EmMotl(input_motl="./test_data/au_1.em")
        em_dynamo_motl = DynamoMotl(input_motl=em_motl_test.df)
        print("\n")
        # (em_mod_motl.df)
        assert em_dynamo_motl.dynamo_df.empty
        em_dynamo_motl.write_out(output_path="./test_data/motl_data/test_dynamo.tbl")
        em_dynamo_motl1 = DynamoMotl(input_motl="./test_data/motl_data/test_dynamo.tbl")
        pd.set_option('display.max_rows', None)  # Print all rows
        pd.set_option('display.max_columns', None)  # Print all columns
        print(em_dynamo_motl1.dynamo_df)

        if os.path.exists("./test_data/motl_data/test_dynamo.tbl"):
            os.remove("./test_data_motl_data/test_dynamo.tbl")

    def test_convert_to_dynamo(self):
        #Create a dynamo using real - confirmed - dynamo df
        #and then try to create the same object with dynamo.df
        dynamo_test = DynamoMotl(input_motl="./test_data/motl_data/crop.tbl")
        dynamo_test_reconvert = DynamoMotl(input_motl=dynamo_test.df)
        dynamo_test_reconvert.write_out(output_path="./test_data/motl_data/crop2.tbl")
        if os.path.exists("./test_data/motl_data/crop2.tbl"):
            os.remove("./test_data/motl_data/crop2.tbl")

class TestModMotl:
    def test_init(self):
        test_file_path = "./test_data/motl_data/modMotl/"
        mod_df = ModMotl(input_motl=test_file_path)
        """pd.set_option('display.max_rows', None)  # Print all rows
        pd.set_option('display.max_columns', None)  # Print all columns"""
        print("\n")
        print(mod_df.df)
        print(mod_df.mod_df)

    def test_read_in_valid_file(self):
        test_file_path = "./test_data/motl_data/modMotl/"
        mod_df = ModMotl.read_in(input_path=test_file_path)
        assert isinstance(mod_df, pd.DataFrame)
        assert not mod_df.empty
        assert all(col in mod_df.columns for col in ['object_id', 'x', 'y', 'z', 'mod_id', 'contour_id'])
        unique_mod_ids = mod_df["mod_id"].unique()
        expected_ids = ["correct189", "correct111"]
        assert all(uid in unique_mod_ids for uid in expected_ids)
        assert len(unique_mod_ids) == len(expected_ids)



        test_file_path = "./test_data/motl_data/empty089.mod"
        mod_df = ModMotl.read_in(input_path=test_file_path)
        assert isinstance(mod_df, pd.DataFrame)
        assert not mod_df.empty
        assert all(col in mod_df.columns for col in ['object_id', 'x', 'y', 'z', 'mod_id', 'contour_id'])
        assert mod_df["mod_id"].unique() == "089"

    def test_read_in_invalid_file(self):
        with pytest.raises(ValueError):
            ModMotl.read_in("non_existent_file.mod")

    def test_check_tomo_id_type_string_filenames(self):
        def check_tomo_id_type(df):
            if df["mod_id"].apply(lambda x: isinstance(x, str)).all():
                # If all values are strings, extract digits and convert to integers
                df["mod_id"] = df["mod_id"].str.extract("(\d+)")[0].astype(int)
            elif df["mod_id"].apply(lambda x: isinstance(x, int)).all():
                # If all values are integers, do nothing or keep as is
                pass
            else:
                raise ValueError("Column contains mixed types or unexpected data.")

            return df
        df = pd.DataFrame({"mod_id": ["empty089", "empty111"]})
        expected_df = pd.DataFrame({"mod_id": [89, 111]})
        result = check_tomo_id_type(df.copy())
        pd.testing.assert_frame_equal(result[["mod_id"]], expected_df)

    def test_subtract_rows(self):
        test_df_two_points_per_contour = pd.DataFrame({
            "object_id": [1, 1, 1, 1, 2, 2, 2, 2],
            "contour_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "x": [1, 2, 3, 4, 5, 6, 7, 8],
            "y": [9, 10, 11, 12, 13, 14, 15, 16],
            "z": [17, 18, 19, 20, 21, 22, 23, 24],
            "mod_id": ["100", "100", "100", "100", "101", "101", "101", "101"],
            "object_radius": [10, 10, 10, 10, 15, 15, 15, 15],
        })

    def test_convert_to_motl(self):
        # Test case 1: Single contour per object (should pass)
        input_df_test_single = pd.DataFrame({
            "object_id": [1, 2, 3],
            "contour_id": [1, 1, 1],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
            "mod_id": ["100", "101", "102"],
            "object_radius": [10, 15, 20],
        })
        mod_motl1 = ModMotl(input_motl=input_df_test_single)
        expected_df_single = pd.DataFrame({
            'score': [0.0, 0.0, 0.0],
            'geom1': [0.0, 0.0, 0.0],
            'geom2': [1.0, 1.0, 1.0],
            'subtomo_id': [1.0, 2.0, 3.0],
            'tomo_id': [100.0, 101.0, 102.0],
            'object_id': [1.0, 2.0, 3.0],
            'subtomo_mean': [0.0, 0.0, 0.0],
            'x': [1.0, 2.0, 3.0],
            'y': [4.0, 5.0, 6.0],
            'z': [7.0, 8.0, 9.0],
            'shift_x': [0.0, 0.0, 0.0],
            'shift_y': [0.0, 0.0, 0.0],
            'shift_z': [0.0, 0.0, 0.0],
            'geom3': [0.0, 0.0, 0.0],
            'geom4': [0.0, 0.0, 0.0],
            'geom5': [10.0, 15.0, 20.0],
            'phi': [0.0, 0.0, 0.0],
            'psi': [0.0, 0.0, 0.0],
            'theta': [0.0, 0.0, 0.0],
            'class': [0.0, 0.0, 0.0]
        })
        pd.testing.assert_frame_equal(mod_motl1.df, expected_df_single)



        # Test case 3: Incorrect number of countours per object (should raise ValueError)
        data = {'object_id': [1, 1, 2, 2],
                'contour_id': [1, 2, 1, 2],
                'x': [1.0, 4.0, 7.0, 10.0],
                'y': [2.0, 5.0, 8.0, 11.0],
                'z': [3.0, 6.0, 9.0, 12.0],
                'mod_id': ["100", "101", "102", "103"],
                'object_radius': [10, 15, 20, 25]}

        mod_df_incorrect = pd.DataFrame(data)

        # Now contours_per_object is {1: 2, 2: 2}, and there are 2 contours per object
        contours_per_object = mod_df_incorrect["object_id"].value_counts(sort=False)

    @pytest.fixture
    def motl_df(self):
        sample_data = {
            "score": [0.5, 0.7, 0.9, 1.0],
            "geom1": [1, 1, 1, 1],
            "geom2": [1, 2, 1, 1],
            "subtomo_id": [1, 2, 3, 4],
            "tomo_id": [888, 888, 888, 888],
            "object_id": [1, 1, 2, 3],
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
        return pd.DataFrame(sample_data)

    def test_write_out(self, motl_df):
        pd.set_option('display.max_rows', None)  # Print all rows
        pd.set_option('display.max_columns', None)  # Print all columns

        test_file_path = "./test_data/motl_data/modMotl/"
        mod_motl = ModMotl(test_file_path)
        mod_motl.write_out(output_path=test_file_path + "test999.mod")
        mod = ModMotl(test_file_path + "test999.mod")
        """print(mod.mod_df)
        pd.testing.assert_frame_equal(
            ModMotl(test_file_path + "test999.mod").mod_df.drop(columns='mod_id'),
            mod_motl.mod_df.drop(columns='mod_id'))"""
        if os.path.exists(test_file_path + "test999.mod"):
            os.remove(test_file_path + "test999.mod")

        input_df_test_single = pd.DataFrame({
            "object_id": [1, 2, 3],
            "contour_id": [1, 1, 1],
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "z": [7, 8, 9],
            "object_radius": [3, 3, 3],
            "mod_id": ["669", "669", "669"]
        })
        mod = ModMotl(input_motl=input_df_test_single)
        mod.write_out(output_path="./test_data/motl_data/test669.mod")
        mod1 = ModMotl(input_motl="./test_data/motl_data/test669.mod")

        pd.testing.assert_frame_equal(mod.mod_df, mod1.mod_df, check_dtype=False)
        pd.testing.assert_frame_equal(mod.df, mod1.df, check_dtype=False)
        if os.path.exists("./test_data/motl_data/test669.mod"):
            os.remove("./test_data/motl_data/test669.mod")

        #create a ModMotl using a Motl df
        motl_df_mod = ModMotl(motl_df)
        """print("\n")
        print(motl_df_mod.df)"""
        assert motl_df_mod.mod_df.empty #OK - normal

        motl_df_mod.write_out(output_path= test_file_path + "test888.mod")
        mod11 = ModMotl(input_motl = test_file_path + "test888.mod")
        print(mod11.mod_df)
        #print(mod11.df)
        #print(mod11.mod_df)
        #print(motl_df_mod.mod_df)
        #pd.testing.assert_frame_equal(motl_df_mod.df, mod11.df)
        pd.testing.assert_frame_equal(motl_df_mod.mod_df, mod11.mod_df, check_dtype=False)

        if os.path.exists(test_file_path + "test888.mod"):
            os.remove(test_file_path + "test888.mod")


        #Test using a real EM file pandas dataframe as input
        em_motl_test = EmMotl(input_motl = "./test_data/au_1.em")
        em_mod_motl = ModMotl(input_motl = em_motl_test.df)
        print("\n")
        #(em_mod_motl.df)
        assert em_mod_motl.mod_df.empty
        em_mod_motl.write_out(output_path = test_file_path + "test567.mod")
        em_mod_motl1 = ModMotl(input_motl = test_file_path + "test567.mod")
        print(em_mod_motl1.mod_df)
        #print(em_mod_motl1.df)

        """if os.path.exists(test_file_path + "test567.mod"):
            os.remove(test_file_path + "test567.mod")
        """

class TestRelionMotlv5:
    warp_tomo_path = "./test_data/motl_data/relion5/clean/warp2_matching_tomograms.star"
    warp_particles_path = "test_data/motl_data/relion5/clean/warp2_particles_matching.star"
    relion_tomo_path = "./test_data/motl_data/relion5/clean/R5_tutorial_run_tomograms.star"
    relion_particles_path = "./test_data/motl_data/relion5/clean/R5_tutorial_run_data.star"

    def test_getpd(self):
        result = RelionMotlv5.read_in_tomograms(self, self.warp_tomo_path)
        print(result)
    def test_write2(self):
        motl5 = RelionMotlv5(
            input_particles = self.warp_particles_path,
            input_tomograms = self.warp_tomo_path,
            pixel_size = 1.0,
        )
        motl5.write_out(
            output_path = "./test_data/motl_data/relion5/clean/warp2test.star",
            type="relion",
            write_optics=True,
            optics_data= self.warp_particles_path,
        )

    def test_readInWarpStar(self):
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option("display.width", 2000)
        pd.set_option("display.expand_frame_repr", False)
        #relion files
        motl5_relion = RelionMotlv5(
            input_particles = self.relion_particles_path,
            input_tomograms = self.relion_tomo_path,
        ) #ok
        #print(motl5_relion.relion_df)
        #print(motl5_relion.optics_data)
        #print(motl5_relion.tomo_df)
        # print(motl5_relion.df) #todo: test conversion, how?

        with pytest.raises(Exception):
            motl5_relion1 = RelionMotlv5(
                input_particles = self.relion_particles_path,
            )

        motl5_relion2 = RelionMotlv5(
            input_tomograms = self.relion_tomo_path
        ) #empty object
        assert motl5_relion2.relion_df.empty
        assert not motl5_relion2.tomo_df.empty
        assert motl5_relion2.df.empty

        #test: input_tomo with file
        data_rows = [
            [1, 4000, 4000, 2000],
            [3, 4000, 4000, 2000],
            [43, 4000, 4000, 2000],
            [45, 4000, 4000, 2000],
            [54, 4000, 4000, 2000]
        ]
        column_names = ['rlnTomoName', 'rlnTomoSizeX', 'rlnTomoSizeY', 'rlnTomoSizeZ']
        dim_tomo_df = pd.DataFrame(data_rows, columns=column_names)
        motl5_relion4 = RelionMotlv5(
            input_particles=self.relion_particles_path,
            input_tomograms=dim_tomo_df
        )
        pd.testing.assert_frame_equal(dim_tomo_df, motl5_relion.tomo_df)

        #test: use relion object as input
        motl5_relion3 = RelionMotlv5(
            input_particles=motl5_relion2
        )
        pd.testing.assert_frame_equal(motl5_relion2.relion_df, motl5_relion3.relion_df)

        #warp file test -- same tests
        relion5motl_warp = RelionMotlv5(
            input_particles=self.warp_particles_path,
            input_tomograms=self.warp_tomo_path
        )
        with pytest.raises(Exception):
            relion5motl_warp1 = RelionMotlv5(
                input_particles = self.warp_particles_path,
            )
        relion5motl_warp2 = RelionMotlv5(
            input_tomograms=self.warp_tomo_path
        )  # empty object
        assert relion5motl_warp2.relion_df.empty
        assert not relion5motl_warp2.tomo_df.empty
        assert relion5motl_warp2.df.empty


