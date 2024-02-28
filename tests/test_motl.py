import numpy as np
import pandas as pd
import pytest

from cryocat.cryomotl import Motl
from cryocat.exceptions import UserInputError


@pytest.fixture
def motl():
    motl = Motl.load("./tests/test_data/au_1.em")
    return motl


@pytest.mark.parametrize("feature_id", ["missing"])
def test_get_feature_not_existing(motl, feature_id):
    with pytest.raises(UserInputError):
        motl.get_feature(feature_id)


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
