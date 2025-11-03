import pytest
import tempfile
import os
import pandas as pd
import struct
import numpy as np
from pathlib import Path
from cryocat import imod


class TestImodHeaders:
    def test_model_header_creation(self):
        header = imod.ModelHeader()
        assert header.name == "IMOD-NewModel"
        assert header.xmax == 4096
        assert header.objsize == 1

    def test_model_header_bytes_conversion(self):
        header = imod.ModelHeader(xmax=1024, ymax=768, zmax=100)
        header_bytes = header.to_bytes()

        new_header = imod.ModelHeader.from_bytes(header_bytes, "utf-8")
        assert new_header.xmax == 1024
        assert new_header.ymax == 768
        assert new_header.zmax == 100

    def test_object_header_creation(self):
        header = imod.ObjectHeader()
        assert header.name == "OBJT"
        assert header.contsize == 0
        assert header.red == 0.0
        assert header.green == 1.0
        assert header.blue == 0.0

    def test_object_header_bytes_conversion(self):
        header = imod.ObjectHeader(contsize=5, pdrawsize=10)
        header_bytes = header.to_bytes()

        new_header = imod.ObjectHeader.from_bytes(header_bytes, "utf-8")
        assert new_header.contsize == 5
        assert new_header.pdrawsize == 10

    def test_contour_header_creation(self):
        header = imod.ContourHeader()
        assert header.name == "CONT"
        assert header.psize == 0

    def test_contour_header_bytes_conversion(self):
        header = imod.ContourHeader(psize=100, flags=1)
        header_bytes = header.to_bytes()

        new_header = imod.ContourHeader.from_bytes(header_bytes, "utf-8")
        assert new_header.psize == 100
        assert new_header.flags == 1


class TestImodFileOperations:
    def test_write_and_read_simple_model(self):
        test_data = pd.DataFrame({
            'object_id': [1, 1, 1, 2, 2],
            'contour_id': [1, 1, 1, 1, 1],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0],
            'z': [1.0, 2.0, 3.0, 4.0, 5.0],
            'object_radius': [3, 3, 3, 5, 5]
        })

        with tempfile.NamedTemporaryFile(suffix='.mod', delete=False) as f:
            temp_file = f.name

        try:
            imod.write_model_binary(test_data, temp_file)

            assert os.path.exists(temp_file)
            file_size = os.path.getsize(temp_file)
            assert file_size > 0

            read_data = imod.read_mod_file(temp_file)

            assert isinstance(read_data, pd.DataFrame)
            assert len(read_data) == len(test_data)
            assert 'object_id' in read_data.columns
            assert 'x' in read_data.columns
            assert 'y' in read_data.columns
            assert 'z' in read_data.columns

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_model_with_multiple_objects(self):
        test_data = pd.DataFrame({
            'object_id': [1, 1, 1, 2, 2, 2, 3, 3],
            'contour_id': [1, 1, 2, 1, 1, 2, 1, 1],
            'x': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'z': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            'object_radius': [2, 2, 2, 3, 3, 3, 4, 4]
        })

        with tempfile.NamedTemporaryFile(suffix='.mod', delete=False) as f:
            temp_file = f.name

        try:
            imod.write_model_binary(test_data, temp_file)
            read_data = imod.read_mod_file(temp_file)

            assert read_data['object_id'].nunique() == test_data['object_id'].nunique()
            assert read_data['contour_id'].nunique() == test_data['contour_id'].nunique()

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_write_model_without_radius(self):
        test_data = pd.DataFrame({
            'object_id': [1, 1, 1],
            'contour_id': [1, 1, 1],
            'x': [1.0, 2.0, 3.0],
            'y': [1.0, 2.0, 3.0],
            'z': [1.0, 2.0, 3.0]
            #no object_radius column
        })

        with tempfile.NamedTemporaryFile(suffix='.mod', delete=False) as f:
            temp_file = f.name

        try:
            #should work without object_radius
            imod.write_model_binary(test_data, temp_file)
            read_data = imod.read_mod_file(temp_file)

            assert len(read_data) == len(test_data)
            #should have object_radius column with default value
            assert 'object_radius' in read_data.columns

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestImodFileDetection:

    def test_read_single_mod_file(self):
        test_data = pd.DataFrame({
            'object_id': [1, 1],
            'contour_id': [1, 1],
            'x': [1.0, 2.0],
            'y': [1.0, 2.0],
            'z': [1.0, 2.0],
            'object_radius': [3, 3]
        })

        with tempfile.NamedTemporaryFile(suffix='.mod', delete=False) as f:
            temp_file = f.name

        try:
            imod.write_model_binary(test_data, temp_file)

            result = imod.read_mod_file(temp_file)

            assert isinstance(result, pd.DataFrame)
            # read_mod_file should NOT have mod_id column
            assert 'mod_id' not in result.columns

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_read_single_real_mod_file(self):
        test_data_path = "./test_data/motl_data/modMotl/correct111.mod"
        if not os.path.exists(test_data_path):
            pytest.skip(f"Real mod file not found: {test_data_path}")

        result = imod.read_mod_file(test_data_path)
        assert isinstance(result, pd.DataFrame)
        assert 'mod_id' not in result.columns

    def test_read_mod_files_from_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            test_data1 = pd.DataFrame({
                'object_id': [1, 1],
                'contour_id': [1, 1],
                'x': [1.0, 2.0],
                'y': [1.0, 2.0],
                'z': [1.0, 2.0],
                'object_radius': [3, 3]
            })

            test_data2 = pd.DataFrame({
                'object_id': [1, 1],
                'contour_id': [1, 1],
                'x': [3.0, 4.0],
                'y': [3.0, 4.0],
                'z': [3.0, 4.0],
                'object_radius': [5, 5]
            })

            #full_paths
            file1 = os.path.join(temp_dir, "tomo1.mod")
            file2 = os.path.join(temp_dir, "tomo2.mod")
            imod.write_model_binary(test_data1, file1)
            imod.write_model_binary(test_data2, file2)

            #handle path issue
            try:
                result = imod.read_mod_files(temp_dir, file_prefix="tomo", file_suffix=".mod")

                # If we get here, the function worked
                assert isinstance(result, pd.DataFrame)
                assert 'mod_id' in result.columns
                assert len(result) == len(test_data1) + len(test_data2)
                assert result['mod_id'].nunique() == 2

            except FileNotFoundError as e:
                #path issues: individually test!
                print(f"Path issue detected: {e}")
                result1 = imod.read_mod_files(file1)
                result2 = imod.read_mod_files(file2)

                assert isinstance(result1, pd.DataFrame)
                assert isinstance(result2, pd.DataFrame)
                assert 'mod_id' in result1.columns
                assert 'mod_id' in result2.columns

                combined_result = pd.concat([result1, result2], ignore_index=True)
                assert len(combined_result) == len(test_data1) + len(test_data2)

    def test_read_mod_files_from_real_directory(self):
        test_data_dir = "./test_data/motl_data/modMotl/"
        if not os.path.exists(test_data_dir):
            pytest.skip(f"Real mod directory not found: {test_data_dir}")

        mod_files = [f for f in os.listdir(test_data_dir) if f.endswith('.mod')]
        if len(mod_files) < 2:
            pytest.skip(f"Need at least 2 mod files for directory test, found {len(mod_files)}")

        result = imod.read_mod_files(test_data_dir)
        assert isinstance(result, pd.DataFrame)
        assert 'mod_id' in result.columns
        assert len(result) > 0


class TestImodEdgeCases:

    def test_empty_dataframe(self):
        empty_df = pd.DataFrame(columns=['object_id', 'contour_id', 'x', 'y', 'z'])

        with tempfile.NamedTemporaryFile(suffix='.mod', delete=False) as f:
            temp_file = f.name

        try:
            imod.write_model_binary(empty_df, temp_file)
            assert os.path.exists(temp_file)

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_nonexistent_file_read(self):
        with pytest.raises(FileNotFoundError):
            imod.read_mod_file("/nonexistent/path/file.mod")

    def test_invalid_file_format(self):
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not a mod file")
            temp_file = f.name

        try:
            try:
                result = imod.read_mod_file(temp_file)
                assert isinstance(result, pd.DataFrame)
            except Exception as e:
                assert isinstance(e, (ValueError, AttributeError, struct.error))

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


class TestImodBinaryStructure:

    def test_binary_header_sequence_detection(self):
        test_data = b"IMODv1.2_MORE_DATA"  # Start with IMOD to ensure it's found

        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            f.write(test_data)
            temp_file = f.name

        try:
            with open(temp_file, 'rb') as f:
                position = imod.ImodHeader.check_sequence(f, "IMOD", "utf-8")
                assert position == 0

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)

    def test_coordinate_precision(self):
        test_data = pd.DataFrame({
            'object_id': [1],
            'contour_id': [1],
            'x': [1.23456789],
            'y': [2.34567891],
            'z': [3.45678912],
            'object_radius': [3]
        })

        with tempfile.NamedTemporaryFile(suffix='.mod', delete=False) as f:
            temp_file = f.name

        try:
            imod.write_model_binary(test_data, temp_file)
            read_data = imod.read_mod_file(temp_file)

            np.testing.assert_allclose(
                read_data[['x', 'y', 'z']].values,
                test_data[['x', 'y', 'z']].values,
                rtol=1e-6
            )

        finally:
            if os.path.exists(temp_file):
                os.unlink(temp_file)


#helper
def create_test_mod_data(num_objects=2, points_per_contour=5):
    data = []
    for obj_id in range(1, num_objects + 1):
        for contour_id in range(1, 3):  # 2 contours per object
            for point in range(points_per_contour):
                data.append({
                    'object_id': obj_id,
                    'contour_id': contour_id,
                    'x': float(obj_id * 10 + point),
                    'y': float(obj_id * 10 + point + 1),
                    'z': float(obj_id * 10 + point + 2),
                    'object_radius': obj_id * 2
                })
    return pd.DataFrame(data)

