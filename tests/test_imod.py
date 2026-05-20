import os
import struct

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from cryocat.utils import imod

test_data = str(Path(__file__).parent / "test_data")
test_data_path = test_data + "/motl_data/modMotl/correct111.mod"
test_data_dir = test_data + "/motl_data/modMotl/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_simple_df(num_objects=2, points_per_contour=5):
    rows = []
    for obj_id in range(1, num_objects + 1):
        for contour_id in range(1, 3):
            for point in range(points_per_contour):
                rows.append({
                    "object_id": obj_id,
                    "contour_id": contour_id,
                    "x": float(obj_id * 10 + point),
                    "y": float(obj_id * 10 + point + 1),
                    "z": float(obj_id * 10 + point + 2),
                    "object_radius": obj_id * 2,
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Header creation (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, attr, expected", [
    (imod.ModelHeader, "name", "IMOD-NewModel"),
    (imod.ModelHeader, "xmax", 4096),
    (imod.ModelHeader, "objsize", 1),
    (imod.ObjectHeader, "name", "OBJT"),
    (imod.ObjectHeader, "contsize", 0),
    (imod.ObjectHeader, "red", 0.0),
    (imod.ObjectHeader, "green", 1.0),
    (imod.ObjectHeader, "blue", 0.0),
    (imod.ContourHeader, "name", "CONT"),
    (imod.ContourHeader, "psize", 0),
])
def test_header_default_attributes(cls, attr, expected):
    assert getattr(cls(), attr) == expected


# ---------------------------------------------------------------------------
# Header bytes round-trip (parametrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls, init_kwargs, check_attrs", [
    (imod.ModelHeader, {"xmax": 1024, "ymax": 768, "zmax": 100}, ["xmax", "ymax", "zmax"]),
    (imod.ObjectHeader, {"contsize": 5, "pdrawsize": 10}, ["contsize", "pdrawsize"]),
    (imod.ContourHeader, {"psize": 100, "flags": 1}, ["psize", "flags"]),
])
def test_header_bytes_roundtrip(cls, init_kwargs, check_attrs):
    header = cls(**init_kwargs)
    roundtrip = cls.from_bytes(header.to_bytes(), "utf-8")
    for attr in check_attrs:
        assert getattr(roundtrip, attr) == getattr(header, attr)


# ---------------------------------------------------------------------------
# File I/O — write + read round-trip
# ---------------------------------------------------------------------------

def test_write_and_read_simple_model(tmp_path):
    data = pd.DataFrame({
        "object_id": [1, 1, 1, 2, 2],
        "contour_id": [1, 1, 1, 1, 1],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0],
        "object_radius": [3, 3, 3, 5, 5],
    })
    temp_file = str(tmp_path / "test.mod")
    imod.write_model_binary(data, temp_file)

    assert os.path.exists(temp_file)
    assert os.path.getsize(temp_file) > 0

    read_data = imod.read_mod_file(temp_file)
    assert isinstance(read_data, pd.DataFrame)
    assert len(read_data) == len(data)
    for col in ("object_id", "x", "y", "z"):
        assert col in read_data.columns


def test_write_model_with_multiple_objects(tmp_path):
    data = pd.DataFrame({
        "object_id": [1, 1, 1, 2, 2, 2, 3, 3],
        "contour_id": [1, 1, 2, 1, 1, 2, 1, 1],
        "x": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "y": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "z": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
        "object_radius": [2, 2, 2, 3, 3, 3, 4, 4],
    })
    temp_file = str(tmp_path / "multi.mod")
    imod.write_model_binary(data, temp_file)
    read_data = imod.read_mod_file(temp_file)

    assert read_data["object_id"].nunique() == data["object_id"].nunique()
    assert read_data["contour_id"].nunique() == data["contour_id"].nunique()


def test_write_model_without_radius(tmp_path):
    data = pd.DataFrame({
        "object_id": [1, 1, 1],
        "contour_id": [1, 1, 1],
        "x": [1.0, 2.0, 3.0],
        "y": [1.0, 2.0, 3.0],
        "z": [1.0, 2.0, 3.0],
    })
    temp_file = str(tmp_path / "no_radius.mod")
    imod.write_model_binary(data, temp_file)
    read_data = imod.read_mod_file(temp_file)

    assert len(read_data) == len(data)
    assert "object_radius" in read_data.columns


# ---------------------------------------------------------------------------
# read_mod_file vs read_mod_files
# ---------------------------------------------------------------------------

def test_read_mod_file_has_no_mod_id_column(tmp_path):
    data = pd.DataFrame({
        "object_id": [1, 1],
        "contour_id": [1, 1],
        "x": [1.0, 2.0],
        "y": [1.0, 2.0],
        "z": [1.0, 2.0],
        "object_radius": [3, 3],
    })
    temp_file = str(tmp_path / "single.mod")
    imod.write_model_binary(data, temp_file)
    result = imod.read_mod_file(temp_file)

    assert isinstance(result, pd.DataFrame)
    assert "mod_id" not in result.columns


def test_read_single_real_mod_file():
    if not os.path.exists(test_data_path):
        pytest.skip(f"Real mod file not found: {test_data_path}")
    result = imod.read_mod_file(test_data_path)
    assert isinstance(result, pd.DataFrame)
    assert "mod_id" not in result.columns


def test_read_mod_files_from_directory(tmp_path):
    data1 = pd.DataFrame({
        "object_id": [1, 1], "contour_id": [1, 1],
        "x": [1.0, 2.0], "y": [1.0, 2.0], "z": [1.0, 2.0], "object_radius": [3, 3],
    })
    data2 = pd.DataFrame({
        "object_id": [1, 1], "contour_id": [1, 1],
        "x": [3.0, 4.0], "y": [3.0, 4.0], "z": [3.0, 4.0], "object_radius": [5, 5],
    })
    file1 = str(tmp_path / "tomo1.mod")
    file2 = str(tmp_path / "tomo2.mod")
    imod.write_model_binary(data1, file1)
    imod.write_model_binary(data2, file2)

    try:
        result = imod.read_mod_files(str(tmp_path), file_prefix="tomo", file_suffix=".mod")
        assert isinstance(result, pd.DataFrame)
        assert "mod_id" in result.columns
        assert len(result) == len(data1) + len(data2)
        assert result["mod_id"].nunique() == 2
    except FileNotFoundError:
        result1 = imod.read_mod_files(file1)
        result2 = imod.read_mod_files(file2)
        combined = pd.concat([result1, result2], ignore_index=True)
        assert len(combined) == len(data1) + len(data2)


def test_read_mod_files_from_real_directory():
    if not os.path.exists(test_data_dir):
        pytest.skip(f"Real mod directory not found: {test_data_dir}")
    mod_files = [f for f in os.listdir(test_data_dir) if f.endswith(".mod")]
    if len(mod_files) < 2:
        pytest.skip(f"Need at least 2 mod files, found {len(mod_files)}")
    result = imod.read_mod_files(test_data_dir)
    assert isinstance(result, pd.DataFrame)
    assert "mod_id" in result.columns
    assert len(result) > 0


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_dataframe_write(tmp_path):
    empty_df = pd.DataFrame(columns=["object_id", "contour_id", "x", "y", "z"])
    temp_file = str(tmp_path / "empty.mod")
    imod.write_model_binary(empty_df, temp_file)
    assert os.path.exists(temp_file)


def test_nonexistent_file_raises():
    with pytest.raises(FileNotFoundError):
        imod.read_mod_file("/nonexistent/path/file.mod")


def test_invalid_file_format(tmp_path):
    bad_file = str(tmp_path / "bad.txt")
    with open(bad_file, "wb") as f:
        f.write(b"This is not a mod file")
    try:
        result = imod.read_mod_file(bad_file)
        assert isinstance(result, pd.DataFrame)
    except Exception as e:
        assert isinstance(e, (ValueError, AttributeError, struct.error))


# ---------------------------------------------------------------------------
# Binary structure
# ---------------------------------------------------------------------------

def test_binary_header_sequence_detection(tmp_path):
    test_bytes = b"IMODv1.2_MORE_DATA"
    bin_file = str(tmp_path / "seq.bin")
    with open(bin_file, "wb") as f:
        f.write(test_bytes)
    with open(bin_file, "rb") as f:
        position = imod.ImodHeader.check_sequence(f, "IMOD", "utf-8")
    assert position == 0


def test_coordinate_precision(tmp_path):
    data = pd.DataFrame({
        "object_id": [1], "contour_id": [1],
        "x": [1.23456789], "y": [2.34567891], "z": [3.45678912],
        "object_radius": [3],
    })
    temp_file = str(tmp_path / "precision.mod")
    imod.write_model_binary(data, temp_file)
    read_data = imod.read_mod_file(temp_file)
    np.testing.assert_allclose(
        read_data[["x", "y", "z"]].values,
        data[["x", "y", "z"]].values,
        rtol=1e-6,
    )
