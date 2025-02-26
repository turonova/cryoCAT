import sys
from io import StringIO

from cryocat.cryomotl import Motl

from cryocat.cryomask import get_correct_format
from matplotlib import pyplot as plt
from scipy.fft import fftn, ifftn, fftshift
from cryocat.cryomap import *
import pytest
import numpy as np
import os
from skimage import data
from pathlib import Path
import h5py

def test_scale():
    #test1: use camera image from skimage, using antialias
    scaled_up_camera = scale(data.camera(), 2.0)
    expected_shape_up = (data.camera().shape[0] * 2.0, data.camera().shape[1] * 2.0)
    assert scaled_up_camera.shape == expected_shape_up
    assert scaled_up_camera.dtype == np.float32

    #test2: use camera image from skimage, without antialias
    scaled_up_camera1 = scale(data.camera(), 0.5)
    expected_shape1_up = (data.camera().shape[0] * 0.5, data.camera().shape[1] * 0.5)
    assert scaled_up_camera1.shape == expected_shape1_up
    assert scaled_up_camera1.dtype == np.float32

    #test3: check if output file is being generated
    output_dir = str(Path(__file__).parent / "test_data" / "test_scale")
    scaled_up_camera2 = scale(data.camera(), 0.5, output_dir)
    assert os.path.exists(output_dir)
    if os.path.exists(output_dir):
        os.remove(output_dir)

def test_pixels2resolution():
    #test1
    res = pixels2resolution(100, 200, 1.5, print_out=False)
    expected_res = 200 * 1.5 / 100
    assert res == expected_res, f"Expected {expected_res}, but got {res}"

    #test2
    captured_output = StringIO()
    sys.stdout = captured_output  # Redirect stdout to capture print statements
    pixels2resolution(100, 200, 1.5, print_out=True)
    sys.stdout = sys.__stdout__  # Reset redirect
    expected_print_output = f"The target resolution is {expected_res} Angstroms.\n"
    assert captured_output.getvalue() == expected_print_output, f"Expected print output: {expected_print_output}, but got {captured_output.getvalue()}"

    #test3
    res_small = pixels2resolution(10, 50, 0.5, print_out=False)
    expected_res_small = 50 * 0.5 / 10
    assert res_small == expected_res_small, f"Expected {expected_res_small}, but got {res_small}"

def test_resolution2pixels():
    #test1
    res = resolution2pixels(3.0, 200, 1.5, print_out=False)
    expected_res = round(200 * 1.5 / 3.0)
    assert res == expected_res, f"Expected {expected_res}, but got {res}"

    #test2: intercept print out
    captured_output = StringIO()
    sys.stdout = captured_output
    resolution2pixels(3.0, 200, 1.5, print_out=True)
    sys.stdout = sys.__stdout__
    expected_print_output = f"The target resolution corresponds to {expected_res} pixels.\n"
    assert captured_output.getvalue() == expected_print_output, f"Expected print output: {expected_print_output}, but got {captured_output.getvalue()}"

    #test3
    res_small = resolution2pixels(0.5, 50, 0.5, print_out=False)
    expected_res_small = round(50 * 0.5 / 0.5)
    assert res_small == expected_res_small, f"Expected {expected_res_small}, but got {res_small}"

    #test4: testing types
    res_non_integer = resolution2pixels(2.7, 150, 1.2, print_out=False)
    expected_res_non_integer = round(150 * 1.2 / 2.7)
    assert res_non_integer == expected_res_non_integer, f"Expected {expected_res_non_integer}, but got {res_non_integer}"

def test_binarize():
    #test1
    input_map = np.array([0.2, 0.6, 0.4, 0.8])
    result = binarize(input_map)
    expected_result = np.array([0, 1, 0, 1])
    np.testing.assert_array_equal(result, expected_result, "The binarized map does not match the expected result.")

    #test2
    result = binarize(input_map, threshold=0.7)
    expected_result = np.array([0, 0, 0, 1])  # Expected result based on threshold 0.7
    np.testing.assert_array_equal(result, expected_result,
                                  "The binarized map does not match the expected result with threshold 0.7.")

    #test3
    input_map_high = np.array([0.9, 1.0, 0.8])
    result = binarize(input_map_high, threshold=0.5)
    expected_result = np.array([1, 1, 1])
    np.testing.assert_array_equal(result, expected_result, "All values should be 1 since they are greater than 0.5.")

    #test4
    input_map_low = np.array([0.1, 0.3, 0.2])
    result = binarize(input_map_low, threshold=0.5)
    expected_result = np.array([0, 0, 0])
    np.testing.assert_array_equal(result, expected_result, "All values should be 0 since they are less than 0.5.")

    #test5
    input_map_exact = np.array([0.5, 0.5, 0.5])
    result = binarize(input_map_exact, threshold=0.5)
    expected_result = np.array([0, 0, 0])  # 0 for values <= threshold
    np.testing.assert_array_equal(result, expected_result, "Values equal to threshold should be set to 0.")

    #test6
    input_map_empty = np.array([])
    result = binarize(input_map_empty, threshold=0.5)
    expected_result_empty = np.array([])
    np.testing.assert_array_equal(result, expected_result_empty,
                                  "The result for an empty input map should be an empty array.")

    #test7
    with pytest.raises(ValueError):
        binarize("invalid_input")

@pytest.mark.parametrize(
    "edge_size, fourier_pixels, target_resolution, pixel_size, expected_result, expect_exception",
    [   #edge_size, (fourier_pixels), (target_resolution, pixel_size)
        (100, 50, None, 1.5, 50, None),
        (100, None, 2.0, 1.5, resolution2pixels(2.0, edge_size=100, pixel_size=1.5), None),
        (100, None, None, None, None, ValueError),
        (1000, 500, None, 0.5, 500, None),
        (1, None, 2.0, 1.0, resolution2pixels(2.0, edge_size=1, pixel_size=1.0), None),
        (200, 100, None, None, 100, None),
    ])
def test_get_filter_radius(edge_size, fourier_pixels, target_resolution, pixel_size, expected_result, expect_exception):
    if expect_exception:
        with pytest.raises(expect_exception):
            get_filter_radius(edge_size, fourier_pixels, target_resolution, pixel_size)
    else:
        result = get_filter_radius(edge_size, fourier_pixels, target_resolution, pixel_size)
        assert result == expected_result, f"Expected {expected_result}, but got {result}"

def test_read():
    # ⚠ Modifica questi percorsi con file reali
    MRC_FILE_PATH = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    EM_FILE_PATH = str(Path(__file__).parent / "test_data" / "au_1.em")

    @pytest.mark.parametrize("file_path", [MRC_FILE_PATH, EM_FILE_PATH])
    def test_read_function(file_path):
        data = read(file_path)
        assert isinstance(data, np.ndarray)
        assert data.ndim == 3

        data_no_transpose = read(file_path, transpose=False)
        data_transposed = read(file_path, transpose=True)
        assert data_transposed.shape == (data_no_transpose.shape[2],
                                         data_no_transpose.shape[1],
                                         data_no_transpose.shape[0])

        data_float16 = read(file_path, data_type=np.float16)
        assert data_float16.dtype == np.float16

        array = np.random.rand(5, 5, 5)
        data_from_array = read(array)
        assert isinstance(data_from_array, np.ndarray)
        assert data_from_array.shape == (5, 5, 5)

        with pytest.raises(ValueError, match="is neither em or mrc file"):
            read("invalid.txt")

        with pytest.raises(ValueError, match="Input map must be path to valid file or nparray"):
            read(1234)

MRC_TEST_FILE = str(Path(__file__).parent / "test_data" / "tilt_stack1.mrc")
EM_TEST_FILE = str(Path(__file__).parent / "test_data" / "au_11.em")
@pytest.mark.parametrize("file_path", [MRC_TEST_FILE, EM_TEST_FILE])
def test_write(file_path):
    """Comprehensive test for the write function using real output files."""

    # 1. Create test data
    data = np.random.rand(10, 10, 10).astype(np.float32)

    # 2. Test writing to MRC/EM file
    write(data, file_path)
    assert os.path.exists(file_path), f"File {file_path} was not created successfully"

    # 3. Test reading and validating content
    read_data = read(file_path, transpose=False)
    assert isinstance(read_data, np.ndarray), "The written file was not read correctly"
    assert read_data.shape == (10, 10, 10), "The shape of the written data is incorrect"

    # 4. Test transposition
    write(data, file_path, transpose=True)
    read_transposed = read(file_path, transpose=False)
    assert read_transposed.shape == (10, 10, 10)[::-1], "Transposition did not occur correctly"

    # 5. Test data type conversion
    write(data, file_path, data_type=np.float64)
    read_float64 = read(file_path)
    assert read_float64.dtype == np.float32, "must be converted to float32"

    # 6. Test error handling
    with pytest.raises(ValueError, match="has to end with .mrc, .rec or .em"):
        write(data, "invalid.txt")

    # 7. Cleanup (remove the file after the test)
    os.remove(file_path)

def test_invert_contrast(tmp_path):
    # Assuming we have a real .mrc file for this test
    input_file = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    output_file = str(Path(__file__).parent / "test_data" / "test_invert.mrc")
    # Read the original map
    original_map = read(input_file)
    # Invert contrast and save to a new file
    inverted_map = invert_contrast(input_file, output_name=output_file)
    # Read the saved inverted map
    saved_inverted_map = read(output_file)

    # Check if the inverted map matches the manually inverted one
    np.testing.assert_array_equal(inverted_map, saved_inverted_map)
    np.testing.assert_array_equal(inverted_map, original_map * -1)

    # Ensure the datatype is handled correctly
    if original_map.dtype == np.float64:
        assert saved_inverted_map.dtype == np.single
    else:
        assert saved_inverted_map.dtype == original_map.dtype

    if os.path.exists(output_file):
        os.remove(output_file)

def test_em2mrc():
    input_file = str(Path(__file__).parent / "test_data" / "au_1.em")
    output_file = str(Path(__file__).parent / "test_data" / "au_12.mrc")

    em2mrc(input_file, output_name=output_file)
    assert os.path.exists(output_file), "Output file was not created"

    input_data = read(input_file)
    output_data = read(output_file)
    assert np.allclose(input_data, output_data, atol=1e-6, rtol=1e-6), "Output data does not match input data"

    if os.path.exists(output_file):
        os.remove(output_file)

    # Test default output naming
    em2mrc(input_file)
    expected_path = str(Path(__file__).parent / "test_data" / "au_1.mrc")
    assert os.path.exists(expected_path)
    assert np.allclose(read(input_file), read(expected_path), atol=1e-6, rtol=1e-6)

    if os.path.exists(expected_path):
        os.remove(expected_path)

    # Test inversion
    em2mrc(input_file, invert=True)
    assert os.path.exists(expected_path)

    inverted_output_data = read(expected_path)  # Read the newly created inverted file
    assert np.allclose(inverted_output_data, -input_data, atol=1e-6, rtol=1e-6), "Inversion did not work correctly"

    if os.path.exists(expected_path):
        os.remove(expected_path)

def test_mrc2em():
    input_file = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    output_file = str(Path(__file__).parent / "test_data" / "tilt_stack1.em")

    mrc2em(input_file, output_name=output_file)
    assert os.path.exists(output_file)
    input_data = read(input_file)
    output_data = read(output_file)
    assert np.allclose(input_data, output_data, atol=1e-6, rtol=1e-6)

    if os.path.exists(output_file):
        os.remove(output_file)

    mrc2em(input_file)
    expected_path = str(Path(__file__).parent / "test_data" / "tilt_stack.em")
    assert os.path.exists(expected_path)
    assert np.allclose(read(input_file), read(expected_path), atol=1e-6, rtol=1e-6)
    if os.path.exists(expected_path):
        os.remove(expected_path)

    mrc2em(input_file, invert=True)
    assert os.path.exists(expected_path)
    inverted_output_data = read(expected_path)
    assert np.allclose(inverted_output_data, -input_data, atol=1e-6, rtol=1e-6)

    if os.path.exists(expected_path):
        os.remove(expected_path)

def write_hdf5():
    test_dir = Path(__file__).parent / "test_data"
    mrc_file = test_dir / "tilt_stack.mrc"
    em_file = test_dir / "au_1.em"
    hdf5_file_em = test_dir / "au_1.hdf5"
    hdf5_file = test_dir / "tilt_stack.hdf5"
    hdf5_file1 = test_dir / "tilt_stack1.hdf5"

    test_dir.mkdir(parents=True, exist_ok=True)
    assert mrc_file.exists(), f"Test MRC file '{mrc_file}' does not exist!"

    write_hdf5(str(mrc_file))
    assert hdf5_file.exists(), f"HDF5 file '{hdf5_file}' was not created!"

    original_data = read(str(mrc_file))
    with h5py.File(hdf5_file, "r") as f:
        assert "raw" in f, "Dataset 'raw' not found in HDF5 file!"
        converted_data = np.array(f["raw"])

    assert isinstance(converted_data, np.ndarray), "Output data is not a NumPy array."
    assert converted_data.shape == original_data.shape, f"Data shape mismatch: expected {original_data.shape}, got {converted_data.shape}"
    assert np.allclose(converted_data, original_data), "Data values mismatch between MRC and HDF5."
    hdf5_file.unlink()

    write_hdf5(str(mrc_file), output_name=str(hdf5_file1))
    assert hdf5_file1.exists()
    hdf5_file1.unlink()

    write_hdf5(str(em_file))
    assert hdf5_file_em.exists()

    with pytest.raises(ValueError):
        write_hdf5("1")
    with pytest.raises(Exception):
        write_hdf5("1.em")
    with pytest.raises(ValueError):
        write_hdf5(1)

    hdf5_file_em.unlink()


def create_test_hdf5(data, file_path, dataset_name="predictions"):
    file_path = Path(file_path)
    if file_path.suffix != ".hdf5":
        file_path = file_path.with_suffix(".hdf5")
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(file_path, "w") as f:
        f.create_dataset(dataset_name, data=data)
@pytest.mark.parametrize("data", [
    np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32),  # Known dataset
    np.random.rand(10, 10)  # Random dataset
])
def test_read_hdf5(data):
    test_dir = Path(__file__).parent / "test_data"
    test_file = test_dir / "test.hdf5"
    create_test_hdf5(data, str(test_file))

    assert test_file.exists(), f"Test HDF5 file was not created at {test_file}"

    read_data = read_hdf5(str(test_file))

    assert isinstance(read_data, np.ndarray), "Returned data is not a numpy array."
    assert read_data.shape == data.shape, f"Data shape mismatch: expected {data.shape}, got {read_data.shape}"
    assert np.allclose(read_data, data), "Data values mismatch."

    test_file.unlink()

def test_normalize():
    mrcinput = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    expected_tonormalize = read(mrcinput)
    assert np.allclose(((expected_tonormalize-np.mean(expected_tonormalize))/np.std(expected_tonormalize)), normalize(mrcinput))

def test_rotate():
    input_filename = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    output_filename = str(Path(__file__).parent / "test_data" / "tilt_stack_o1.mrc")

    input_data = read(input_filename)
    rotation_angles = (90, 0, 0)

    rotated_data = rotate(input_filename, rotation_angles=rotation_angles)
    assert isinstance(rotated_data, np.ndarray)

    assert rotated_data.shape == input_data.shape
    write(rotated_data, output_filename)
    output_data = read(output_filename)
    assert output_data.shape == input_data.shape

    test_volume = np.zeros((5, 5, 5), dtype=np.float32)
    test_volume[2, 1, 0] = 1.0

    rotated_test_volume = rotate(test_volume, rotation_angles=(0, 0, 90))

    expected_result = np.zeros((5, 5, 5), dtype=np.float32)
    expected_result[1, 2, 0] = 1.0

    assert np.allclose(rotated_test_volume, expected_result, atol=1e-6), "Rotation result is incorrect"

    if os.path.exists(output_filename):
        os.remove(output_filename)

def test_crop():
    MRC_TEST_FILE = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    input_map = read(MRC_TEST_FILE)
    original_shape = np.array(input_map.shape)

    crop_size = original_shape // 2
    crop_coord = get_correct_format(original_shape) // 2
    cropped_volume = crop(MRC_TEST_FILE, crop_size)

    vs, ve, _, _ = get_start_end_indices(crop_coord, original_shape, crop_size)
    obtained_center = vs + get_correct_format(cropped_volume.shape) // 2
    assert cropped_volume.shape == tuple(crop_size), "Cropped volume has incorrect shape"

    assert np.any(cropped_volume), "Cropped volume is empty"

    assert np.allclose(crop_coord, obtained_center, atol=1), "Center alignment incorrect"

    output_file = str(Path(__file__).parent / "test_output.mrc")
    crop(MRC_TEST_FILE, crop_size, output_file)
    with mrcfile.open(output_file, mode='r') as mrc:
        assert mrc.data.shape == tuple(crop_size)[::-1], "Saved cropped file has incorrect shape"

    if os.path.exists(output_file):
        os.remove(output_file)

def test_shift():
    test_map = np.zeros((5, 5, 5))
    test_map[2, 2, 2] = 1
    delta = np.array([0.5, 0.5, 0.5])
    shifted_map = shift(test_map, delta)
    assert shifted_map.shape == test_map.shape
    assert np.count_nonzero(shifted_map) > 0
    delta = np.array([0, 0, 0])
    shifted_map = shift(test_map, delta)
    assert np.allclose(shifted_map, test_map, atol=1e-5)
    delta = np.array([1, 0, 0])
    shifted_map = shift(test_map, delta)
    assert np.count_nonzero(shifted_map) > 0
    delta = np.array([0.1, 0.1, 0.1])
    shifted_map = shift(test_map, delta)
    assert np.count_nonzero(shifted_map) > 0
    delta = np.array([0.9, 0.9, 0.9])
    shifted_map = shift(test_map, delta)
    assert np.count_nonzero(shifted_map) > 0

def test_recenter():
    test_map = np.zeros((5, 5, 5))
    test_map[2, 2, 2] = 1
    new_center = np.array([3, 3, 3])
    recentered_map = recenter(test_map, new_center)
    assert recentered_map.shape == test_map.shape
    assert np.isclose(recentered_map[3, 3, 3], 1)
    recentered_map = recenter(test_map, np.array([2, 2, 2]))
    assert np.allclose(recentered_map, test_map, atol=1e-8)

def test_normalize_under_mask():
    ref = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=float)
    mask = np.array([[[1, 0], [1, 1]], [[0, 1], [1, 0]]], dtype=int)
    norm_ref = normalize_under_mask(ref, mask)

    masked_values = norm_ref[mask > 0]
    not_masked_values = norm_ref[mask == 0]
    assert np.isclose(np.mean(masked_values), 0, atol=1e-6), "Mean of masked elements is not zero"
    assert np.isclose(np.std(masked_values), 1, atol=1e-6), "Standard deviation of masked elements is not one"
    expected_values = np.array([-1.498, -0.561, -0.093, 0.843, 1.309])
    #expected_not_masked_values = np.array([2,5,8])
    assert np.allclose(masked_values, expected_values, atol=1e-2), "Normalized values do not match expected output"
    #assert np.allclose(not_masked_values, expected_not_masked_values, atol=1e-2)

def test_get_start_end_indices():
    coord = np.array([5, 5, 5])
    volume_shape = (10, 10, 10)
    subvolume_shape = (4, 4, 4)
    expected_volume_start = np.array([3, 3, 3])
    expected_volume_end = np.array([7, 7, 7])
    expected_subvolume_start = np.array([0, 0, 0])
    expected_subvolume_end = np.array([4, 4, 4])
    volume_start, volume_end, subvolume_start, subvolume_end = get_start_end_indices(coord, volume_shape,subvolume_shape)
    assert np.array_equal(volume_start,expected_volume_start)
    assert np.array_equal(volume_end, expected_volume_end)
    assert np.array_equal(subvolume_start,expected_subvolume_start)
    assert np.array_equal(subvolume_end,expected_subvolume_end)

@pytest.mark.parametrize("coord, volume_shape, subvolume_shape, expected", [
    #case1: subvolume bigger than principal volume
    (np.array([5, 5, 5]), (10, 10, 10), (20, 20, 20),
     (np.array([0, 0, 0]), np.array([10, 10, 10]), np.array([5, 5, 5]), np.array([15, 15, 15]))),
    #case2: subvolume totally outside the main volume
    (np.array([15, 15, 15]), (10, 10, 10), (4, 4, 4),
     (np.array([10, 10, 10]), np.array([10, 10, 10]), np.array([0, 0, 0]), np.array([0, 0, 0]))),
    #case3: sobvolume on boarder of main volume
    (np.array([0, 0, 0]), (10, 10, 10), (4, 4, 4),
     (np.array([0, 0, 0]), np.array([4, 4, 4]), np.array([0, 0, 0]), np.array([4, 4, 4]))),
    #case4: subvolume with odd
    (np.array([5, 5, 5]), (10, 10, 10), (3, 3, 3),
     (np.array([4, 4, 4]), np.array([7, 7, 7]), np.array([0, 0, 0]), np.array([3, 3, 3]))),
    #case5: minimal subvolume
    (np.array([0, 0, 0]), (1, 1, 1), (1, 1, 1),
     (np.array([0, 0, 0]), np.array([1, 1, 1]), np.array([0, 0, 0]), np.array([1, 1, 1])))])
def test_edge_get_start_end_indices(coord, volume_shape, subvolume_shape, expected):
    result = get_start_end_indices(coord, volume_shape, subvolume_shape)
    for res, exp in zip(result, expected):
        assert np.array_equal(res,exp)

def test_extract_subvolume():
    # Create a 3D volume with unique values for verification
    volume = np.arange(5 * 5 * 5).reshape(5, 5, 5)
    # Define the center coordinate and subvolume shape
    coordinates = (2, 2, 2)
    subvolume_shape = (3, 3, 3)
    expected = np.array([[[31, 32, 33],
                          [36, 37, 38],
                          [41, 42, 43]],

                         [[56, 57, 58],
                          [61, 62, 63],
                          [66, 67, 68]],

                         [[81, 82, 83],
                          [86, 87, 88],
                          [91, 92, 93]]])
    # Extract subvolume
    result = extract_subvolume(volume, coordinates, subvolume_shape)
    assert result.shape == subvolume_shape, "Extracted subvolume shape is incorrect."
    # Check values
    np.testing.assert_array_equal(result, expected, "Extracted subvolume values are incorrect.")

def test_pad():
    input_volume = np.ones((3, 3, 3))
    new_size = (5, 5, 5)
    padded_volume = pad(input_volume, new_size)
    assert padded_volume.shape == new_size
    assert np.all(padded_volume == 1)

    padded_volume_custom = pad(input_volume, new_size, fill_value=5)
    assert padded_volume_custom.shape == new_size
    assert np.all(padded_volume_custom[0] == 5)  # Top slice
    assert np.all(padded_volume_custom[-1] == 5)  # Bottom slice
    assert np.all(padded_volume_custom[:, 0] == 5)  # Left slice
    assert np.all(padded_volume_custom[:, -1] == 5)  # Right slice
    assert np.all(padded_volume_custom[0, 0] == 5)  # Front top corner
    assert np.all(padded_volume_custom[-1, -1] == 5)  # Back bottom corner
    assert np.all(padded_volume_custom[1:4, 1:4, 1:4] == 1)


    input_volume_same_size = np.ones((5, 5, 5))
    new_size_same = (5, 5, 5)
    padded_volume_same = pad(input_volume_same_size, new_size_same)
    assert np.array_equal(input_volume_same_size, padded_volume_same)

    input_volume_edge_case = np.ones((1, 1, 1))
    new_size_edge_case = (3, 3, 3)
    padded_volume_edge = pad(input_volume_edge_case, new_size_edge_case)
    assert padded_volume_edge.shape == new_size_edge_case
    assert np.all(padded_volume_edge == 1)

    try:
        new_size_invalid = (5, 5)  # New size has only 2 dimensions
        pad(input_volume, new_size_invalid)
        assert False, "Expected ValueError for invalid new_size"
    except IndexError:
        pass
    try:
        new_size_invalid_dim = (5, 5, 5, 5)  # New size has 4 dimensions
        pad(input_volume, new_size_invalid_dim)
        assert False, "Expected ValueError for invalid new_size"
    except Exception:
        pass
    try:
        new_size_small = (2, 2, 2)  # New size is smaller than the original volume
        pad(input_volume, new_size_small)
        assert False, "Expected ValueError for new_size smaller than volume size"
    except Exception:
        pass

def test_deconvolve():
    # Parameters for the synthetic test
    input_volume = np.random.rand(64, 64, 64).astype(np.float32)  # Random 3D volume
    pixel_size_a = 3.42  # Angstroms
    defocus = 2.5  # micrometers
    snr_falloff = 1.2
    deconv_strength = 1.0
    highpass_nyquist = 0.02
    phase_flipped = False
    phaseshift = 0.0

    # Apply deconvolution to the synthetic 3D volume
    output_volume = deconvolve(
        input_volume, pixel_size_a, defocus, snr_falloff, deconv_strength,
        highpass_nyquist, phase_flipped, phaseshift
    )

    assert output_volume.shape == input_volume.shape
    assert np.isfinite(output_volume).all()
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(input_volume[32, :, :], cmap="gray")  # Slice through the middle
    plt.title("Original Volume Slice")
    plt.subplot(1, 2, 2)
    plt.imshow(output_volume[32, :, :], cmap="gray")  # Slice through the middle
    plt.title("Deconvolved Volume Slice")
    plt.show()
    # Compare FFT of input and output volumes (using log scale)
    input_fft = np.abs(fftshift(fftn(input_volume)))
    output_fft = np.abs(fftshift(fftn(output_volume)))
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log1p(np.mean(input_fft, axis=0)), cmap="inferno")
    plt.title("FFT Input")
    plt.subplot(1, 2, 2)
    plt.imshow(np.log1p(np.mean(output_fft, axis=0)), cmap="inferno")
    plt.title("FFT Output (Deconvolved)")
    plt.show()
    assert not np.allclose(output_volume, input_volume)
    assert np.max(output_volume) <= np.max(input_volume) * 2
    assert np.isfinite(output_volume).all()

def test_compute_ctf_1d():
    length = 128  # Numero di punti
    pixel_size = 1.32e-10  # 1.32 Å per pixel
    voltage = 300000  # 300 kV
    cs = 2.7e-3  # 2.7 mm
    defocus = 1.5e-6  # 1.5 µm
    amplitude = 0.07
    phaseshift = 0.0
    bfactor = 10  # B-factor in Å²
    ctf = compute_ctf_1d(length, pixel_size, voltage, cs, defocus, amplitude, phaseshift, bfactor)
    assert ctf.shape == (length,)
    assert np.all(np.abs(ctf) <= 1.0)
    expected_ctf_path = Path(__file__).parent / "test_data" / "expected_ctf.npy"
    if not expected_ctf_path.exists():
        np.save(expected_ctf_path, ctf)
        raise RuntimeError

    expected_ctf = np.load(str(expected_ctf_path))
    assert np.allclose(ctf, expected_ctf, atol=1e-6)

def test_trim():
    input_map = np.arange(27).reshape(3, 3, 3)
    trim_start = [1, 1, 1]
    trim_end = [3, 3, 3]
    expected = input_map[1:3, 1:3, 1:3]
    trimmed = trim(input_map, trim_start, trim_end)
    assert trimmed.shape == expected.shape
    assert np.array_equal(trimmed, expected)
    trimmed = trim(input_map, [0, 0, 0], [3, 3, 3])
    assert np.array_equal(trimmed, input_map)
    trimmed = trim(input_map, [1, 1, 1], [1, 1, 1])
    assert trimmed.size == 0
    trimmed = trim(input_map, [2, 2, 2], [3, 3, 3])
    assert np.array_equal(trimmed, input_map[2:3, 2:3, 2:3])
    trimmed = trim(input_map, [-1, -1, -1], [2, 2, 2])
    assert np.array_equal(trimmed, input_map[0:2, 0:2, 0:2])
    trimmed = trim(input_map, [1, 1, 1], [10, 10, 10])
    assert np.array_equal(trimmed, input_map[1:3, 1:3, 1:3])
    trimmed = trim(input_map, [1, 0, 0], [2, 3, 3])
    assert np.array_equal(trimmed, input_map[1:2, 0:3, 0:3])
    input_map_nan = input_map.astype(float)
    input_map_nan[1, 1, 1] = np.nan
    trimmed = trim(input_map_nan, [1, 1, 1], [3, 3, 3])
    assert np.isnan(trimmed[0, 0, 0])
    input_map_inf = input_map.astype(float)
    input_map_inf[2, 2, 2] = np.inf
    trimmed = trim(input_map_inf, [2, 2, 2], [3, 3, 3])
    assert np.isinf(trimmed[0, 0, 0])

    output_name = str(Path(__file__).parent / "test_output.mrc")
    trimmed = trim(input_map, [1, 1, 1], [3, 3, 3], output_name)
    assert os.path.exists(output_name)
    os.remove(output_name)

def test_flip():
    input_filename = str(Path(__file__).parent / "test_data" / "tilt_stack.mrc")
    output_filename = str(Path(__file__).parent / "test_data" / "tilt_stack_o1.mrc")
    # Read original data
    original_data = read(input_filename)

    # Test flipping along x-axis
    flipped_x = flip(input_filename, axis="x", output_name=output_filename)
    expected_x = np.flip(original_data, 0)
    np.testing.assert_array_equal(flipped_x, expected_x)

    # Test flipping along y-axis
    flipped_y = flip(input_filename, axis="y", output_name=output_filename)
    expected_y = np.flip(original_data, 1)
    np.testing.assert_array_equal(flipped_y, expected_y)

    # Test flipping along z-axis
    flipped_z = flip(input_filename, axis="z", output_name=output_filename)
    expected_z = np.flip(original_data, 2)
    np.testing.assert_array_equal(flipped_z, expected_z)

    # Test flipping along multiple axes
    flipped_xy = flip(input_filename, axis="xy", output_name=output_filename)
    expected_xy = np.flip(original_data, (0, 1))
    np.testing.assert_array_equal(flipped_xy, expected_xy)

    with pytest.raises(ValueError):
        flip(None)

    with pytest.raises(ValueError):
        flip("invalid_map.txt")

    with pytest.raises(ValueError):
        flip(np.array([1, 2, 3]))

    if os.path.exists(output_filename):
        os.remove(output_filename)

def test_calculate_conjugates():
    vol = np.random.rand(4, 4, 4)
    test_filter = np.ones((4, 4, 4))
    conj_target, conj_target_sq = calculate_conjugates(vol, filter=test_filter)

    assert conj_target.shape == vol.shape, "conj_target shape mismatch"
    assert conj_target_sq.shape == vol.shape, "conj_target_sq shape mismatch"
    vol_fft = np.fft.fftn(vol) * test_filter
    vol_fft[0, 0, 0] = 0
    expected_conj_target = np.conj(vol_fft)
    np.testing.assert_array_almost_equal(conj_target, expected_conj_target,
                                         err_msg="conj_target does not match expected result")

    filtered_volume = np.fft.ifftn(vol_fft).real
    expected_conj_target_sq = np.conj(np.fft.fftn(np.power(filtered_volume, 2)))
    np.testing.assert_array_almost_equal(conj_target_sq, expected_conj_target_sq,
                                         err_msg="conj_target_sq does not match expected result")

def test_calculate_flcf():
    vol1 = np.random.rand(4, 4, 4)
    mask = np.ones((4, 4, 4))
    vol2 = np.random.rand(4, 4, 4)
    filter = np.ones((4, 4, 4))
    conj_target, conj_target_sq = calculate_conjugates(vol2, filter)

    cc_map = calculate_flcf(vol1, mask, vol2=vol2, filter=filter)
    assert cc_map.shape == vol1.shape
    assert np.all(cc_map >= 0.0) and np.all(cc_map <= 1.0)

    cc_map_precomputed = calculate_flcf(vol1, mask, conj_target=conj_target, conj_target_sq=conj_target_sq)
    assert np.allclose(cc_map, cc_map_precomputed, atol=1e-5)

    with pytest.raises(ValueError):
        calculate_flcf(vol1, mask)

    vol1_different_shape = np.random.rand(5, 5, 5)
    with pytest.raises(ValueError):
        calculate_flcf(vol1_different_shape, mask, vol2=vol2)

    mask_partial = np.zeros((4, 4, 4))
    mask_partial[1:3, 1:3, 1:3] = 1
    cc_map_partial = calculate_flcf(vol1, mask_partial, vol2=vol2, filter=filter)
    assert cc_map_partial.shape == vol1.shape

    """mask_zero = np.zeros((4, 4, 4))
    cc_map_zero_mask = calculate_flcf(vol1, mask_zero, vol2=vol2, filter=filter)
    assert np.all(np.isnan(cc_map_zero_mask)) or np.all(
        cc_map_zero_mask == 0), "Expected NaN or zero values for zero mask"""

    vol1_nan = vol1.copy()
    vol1_nan[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        calculate_flcf(vol1_nan, mask, vol2=vol2, filter=filter)

def calculate_flcf_with_todo_changes(vol1, mask, vol2=None, conj_target=None, conj_target_sq=None, filter=None):
    """
    using fftshift followed by transpose
    to check the effect on the output.
    """

    # get the size of the box and number of voxels contributing to the calculations
    if np.isnan(vol1).any() or np.isnan(mask).any():
        raise ValueError("Input volumes or mask contain NaN values")
    box_size = np.array(vol1.shape)
    n_pix = mask.sum()

    # Calculate initial Fourier transforms
    vol1 = np.fft.fftn(vol1)
    mask = np.fft.fftn(mask)

    if vol2 is not None:
        conj_target, conj_target_sq = calculate_conjugates(vol2, filter)

    elif conj_target is None or conj_target_sq is None:
        raise ValueError(
            "If the second volume is NOT provided, both conj_target and conj_target_sq have to be passed as parameters."
        )

    # Calculate numerator of equation
    numerator = np.fft.ifftn(vol1 * conj_target).real

    # Calculate denominator in three steps
    A = np.fft.ifftn(mask * conj_target_sq)
    B = np.fft.ifftn(mask * conj_target)
    denominator = np.sqrt(n_pix * A - B * B).real

    # FLCC map
    flcc_map = (numerator / denominator).real

    # Apply fftshift to the map (to align the frequency components)
    flcc_map = np.fft.ifftshift(flcc_map)

    # Adjust orientation (transpose if necessary)
    flcc_map = np.transpose(flcc_map, (2, 1, 0))  # Optional: transpose dimensions if required

    return np.clip(flcc_map, 0.0, 1.0)
def test_flcf_changes_with_todo():
    vol1 = np.random.rand(32, 32, 32)
    mask = np.ones((32, 32, 32))
    vol2 = np.random.rand(32, 32, 32)
    filter = None

    original_output = calculate_flcf(vol1, mask, vol2, filter=filter)
    new_output = calculate_flcf_with_todo_changes(vol1, mask, vol2, filter=filter)
    if np.allclose(original_output, new_output, atol=1e-5):
        print("The outputs of both functions are identical.")
    else:
        print("The outputs of the two functions are different.")


@pytest.mark.parametrize(
    "input_map, lp_fourier_pixels, hp_fourier_pixels, lp_target_resolution, hp_target_resolution, pixel_size, lp_gaussian, hp_gaussian, output_name, expect_exception, expected_output",
    [
        # sinusoidal input → check fourier effect
        (np.sin(np.linspace(0, 2 * np.pi, 32)).reshape(32, 1, 1).repeat(32, axis=1).repeat(32, axis=2),
         None, 5, 20, None, 1.5, 3, 2, None, None, "check_fourier_effect"),
        # normal case with random input, result not manually predictable but should be a numpy array
        (np.random.rand(32, 32, 32), 10, 5, None, None, 1.5, 3, 2, None, None, None),
        # file input case (simulated, does not check file content)
        (str(Path(__file__).parent / "test_data" / "tilt_stack.mrc"), 50, 30, None, None, 1.5, 3, 2,
         str(Path(__file__).parent / "test_data" / "test1.mrc"), None, None),
        # missing low-pass parameter but target resolution given → should work
        (np.random.rand(100, 100, 100), None, 30, 2.0, None, 1.5, 3, 2, None, None, None),
        # missing high-pass parameter but target resolution given → should work
        (np.random.rand(100, 100, 100), 50, None, None, 2.0, 1.5, 3, 2, None, None, None),
        # missing both lp and hp parameters → should raise ValueError
        (np.random.rand(100, 100, 100), None, None, None, None, 1.5, 3, 2, None, ValueError, None),
        # invalid input (empty string) → should raise ValueError
        ('', 50, 30, None, None, 1.5, 3, 2, None, ValueError, None),
    ]
)
def test_bandpass(
    input_map, lp_fourier_pixels, hp_fourier_pixels, lp_target_resolution, hp_target_resolution,
    pixel_size, lp_gaussian, hp_gaussian, output_name, expect_exception, expected_output
):
    """
    Bandpass filter implements both lowpass and highpass filter, so that only those frequences between a range
    can actually pass.
    """
    if expect_exception:
        with pytest.raises(expect_exception):
            bandpass(
                input_map,
                lp_fourier_pixels=lp_fourier_pixels,
                hp_fourier_pixels=hp_fourier_pixels,
                lp_target_resolution=lp_target_resolution,
                hp_target_resolution=hp_target_resolution,
                pixel_size=pixel_size,
                lp_gaussian=lp_gaussian,
                hp_gaussian=hp_gaussian,
                output_name=output_name
            )
    else:
        result = bandpass(
            input_map,
            lp_fourier_pixels=lp_fourier_pixels,
            hp_fourier_pixels=hp_fourier_pixels,
            lp_target_resolution=lp_target_resolution,
            hp_target_resolution=hp_target_resolution,
            pixel_size=pixel_size,
            lp_gaussian=lp_gaussian,
            hp_gaussian=hp_gaussian,
            output_name=output_name
        )

        # check if the output is a numpy array
        assert isinstance(result, np.ndarray)

        # if an expected output is defined, compare results
        if expected_output is not None:
            if isinstance(expected_output, np.ndarray):
                # compare with expected output (e.g., zero array for uniform input)
                np.testing.assert_allclose(result, expected_output, atol=1e-5)
            elif expected_output == "check_fourier_effect":
                # verify that the filter affects the fourier domain
                original_fft = np.abs(np.fft.fftn(input_map))
                filtered_fft = np.abs(np.fft.fftn(result))
                assert np.any(filtered_fft < original_fft), "filter did not attenuate frequencies"

        # if an output file is specified, check that it was created
        if output_name:
            assert os.path.exists(output_name), f"output file {output_name} was not created"

            # optional: read and compare saved output
            saved_output = read(output_name)
            np.testing.assert_allclose(result, saved_output, atol=1e-5)

            # cleanup: remove file after test
            os.remove(output_name)


@pytest.mark.parametrize(
    "input_map, fourier_pixels, target_resolution, pixel_size, gaussian, output_name, expected_output",
    [
        (np.ones((32, 32, 32)), None, 10, 1.5, 3, None, np.ones((32, 32, 32))),
        (np.sin(np.linspace(0, 2 * np.pi, 32)).reshape(32, 1, 1).repeat(32, axis=1).repeat(32, axis=2),
         None, 20, 1.5, 3, None, "check_fourier_effect"),
        (np.random.rand(32, 32, 32), 10, None, 1.5, 3, None, None),
        (str(Path(__file__).parent / "test_data" / "tilt_stack.mrc"), 50, None, 1.5, 3, 'output_map.mrc', None)
    ]
)
def test_lowpass(input_map, fourier_pixels, target_resolution, pixel_size, gaussian, output_name, expected_output):
    if isinstance(input_map, str):
        input_map = read(input_map)

    result = lowpass(
        input_map,
        fourier_pixels=fourier_pixels,
        target_resolution=target_resolution,
        pixel_size=pixel_size,
        gaussian=gaussian,
        output_name=output_name
    )

    assert result.shape == input_map.shape, f"Expected output shape {input_map.shape}, but got {result.shape}"
    assert isinstance(result, np.ndarray), "The result should be a numpy array"
    if os.path.exists(output_name):
        os.remove(output_name)

@pytest.mark.parametrize(
    "input_map, fourier_pixels, target_resolution, pixel_size, gaussian, output_name, expected_output",
    [
        (np.ones((32, 32, 32)), None, 10, 1.5, 3, None, np.ones((32, 32, 32))),
        (np.sin(np.linspace(0, 2 * np.pi, 32)).reshape(32, 1, 1).repeat(32, axis=1).repeat(32, axis=2),
         None, 20, 1.5, 3, None, "check_highpass_effect"),
        (np.random.rand(32, 32, 32), 10, None, 1.5, 3, None, None),
        (str(Path(__file__).parent / "test_data" / "tilt_stack.mrc"), 50, None, 1.5, 3, 'output_map_highpass.mrc', None)
    ]
)
def test_highpass(input_map, fourier_pixels, target_resolution, pixel_size, gaussian, output_name, expected_output):
    # Se l'input è un file, lo carico
    if isinstance(input_map, str):
        input_map = read(input_map)

    result = highpass(
        input_map,
        fourier_pixels=fourier_pixels,
        target_resolution=target_resolution,
        pixel_size=pixel_size,
        gaussian=gaussian,
        output_name=output_name
    )

    assert result.shape == input_map.shape, f"Expected output shape {input_map.shape}, but got {result.shape}"
    assert isinstance(result, np.ndarray), "The result should be a numpy array"

    if output_name:
        assert output_name.endswith('.mrc'), "Expected file to have '.mrc' extension"
    if os.path.exists(output_name):
        os.remove(output_name)

def test_get_cross_slices():
    # Create a 3D array (ensure the array is 3D)
    input_map = np.random.rand(10, 10, 10)

    # Basic test to generate slices and check their shape
    result = get_cross_slices(input_map)
    assert len(result) == 3
    assert result[0].shape == (10, 10)
    assert result[1].shape == (10, 10)
    assert result[2].shape == (10, 10)

    # Test with slice_half_dim=2 and check the shape
    result = get_cross_slices(input_map, slice_half_dim=2)
    assert result[0].shape == (5, 5)  # Correct size based on the function's logic
    assert result[1].shape == (5, 5)
    assert result[2].shape == (5, 5)

    # Test with specified slice_numbers
    result = get_cross_slices(input_map, slice_numbers=[5, 5, 5])
    assert result[0].shape == (10, 10)
    assert result[1].shape == (10, 10)
    assert result[2].shape == (10, 10)

    # Test with valid axes (0, 1, 2) - ensure that the array is 3D
    result = get_cross_slices(input_map, axis=[0, 1, 2])
    assert len(result) == 3
    assert result[0].shape == (10, 10)
    assert result[1].shape == (10, 10)
    assert result[2].shape == (10, 10)

#TODO, to create instance of Motl object
def test_place_object():
    pass
    #to test
    #motl = Motl.load(str(Path(__file__).parent / "test_data" / "wedge_list.em"))