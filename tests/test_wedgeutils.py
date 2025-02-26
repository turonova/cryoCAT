from cryocat.wedgeutils import *
from cryocat.ioutils import *
import numpy as np
import pytest
from pathlib import Path
import emfile


def test_check_data_consistency():
    #case1, normal correct input
    data1 = np.asarray([1,2,3])
    data2 = np.asarray([4,5,6])
    try:
        check_data_consistency(data1, data2, "int", "int")
    except Exception as e:
        pytest.fail()

    #case2, different shape
    data1 = np.asarray([[1,2],[3,4]]) # 2x2
    data2 = np.asarray([[5,6,7],[8,9,10]]) # 2x3
    with pytest.raises(ValueError):
        check_data_consistency(data1, data2, "int", "int")

    #case3, different type, should be correct
    data1 = np.asarray([1,2,3])
    data2 = np.asarray([1.1,2.1,3.1])
    try:
        check_data_consistency(data1, data2, "int", "float")
    except Exception as e:
        pytest.fail()

    #case4: not an npndarray as input
    data1 = [1,2,3]
    data2 = np.asarray([1,2,3])
    with pytest.raises(TypeError):
        check_data_consistency(data1, data2, "int", "int")

    #case5:

def compare_star_files(file1, file2):
    try:
        if isinstance(file1, str):
            df1 = sf.Starfile.read(file1, data_id=0)[0]
        if isinstance(file2, str):
            df2 = sf.Starfile.read(file2, data_id=0)[0]
        return df1.equals(df2)
    except Exception as e:
        return False
def compare_pd_and_star(df1, file1):
    try:
        df1 = df1
        df2 = sf.Starfile.read(file1, data_id=0)[0]
        for col in df1.columns:
            if not np.allclose(df1[col].astype(float), df2[col].astype(float), rtol=1e-5, atol=1e-8, equal_nan=True):
                return False
        return True

    except Exception as e:
        return False

def test_create_wedge_list_sg():
    directory = Path(__file__).parent / "test_data"
    ts_017 = directory / "TS_017"
    ts_018 = directory / "TS_018"
    wedge_output = directory / "wedgeutils_data"

    pixel_size = 2.4
    #Case1: tomo17, TILT.COM, TLT, GCTF, no DOSE_FILE

    tomo_id = 17
    tomo_dim = dimensions_load(str(ts_017 / "tilt.com"))
    tlt_file = str(ts_017 / "017.tlt")
    ctf_file = str(ts_017 / "017_gctf.star")
    #dose_file = str(ts_017 / "017_corrected_dose.txt")
    df_17 = create_wedge_list_sg(
        tomo_id = tomo_id,
        tomo_dim = tomo_dim,
        pixel_size = pixel_size,
        tlt_file = tlt_file,
        ctf_file = ctf_file,
        output_file= str(wedge_output / "wedge_list_017_1.star")
    )
    #check if output file and given file are equal
    # tlt_angle column from dataframe is float32, but in output file type is float64
    assert compare_star_files(str(wedge_output / "wedge_list_017_1.star"), str(wedge_output / "wedge_list_017.star"))
    #check if returned pd is equal to given file
    assert compare_pd_and_star(df_17, str(wedge_output / "wedge_list_017.star"))
    if os.path.exists(str(wedge_output / "wedge_list_017_1.star")):
        os.remove(str(wedge_output / "wedge_list_017_1.star"))

    #Case2: tomo18, TILT.COM, TLT, GCTF, no DOSE_FILE
    tomo_id = 18
    tomo_dim = dimensions_load(str(ts_018 / "tilt.com"))
    tlt_file = str(ts_018 / "018.tlt")
    ctf_file = str(ts_018 / "018_gctf.star")
    #dose_file = str(ts_018 / "018_corrected_dose.txt")
    df_18 = create_wedge_list_sg(
        tomo_id = tomo_id,
        tomo_dim = tomo_dim,
        pixel_size = pixel_size,
        tlt_file = tlt_file,
        ctf_file = ctf_file,
        z_shift = 100.0,
        output_file= str(wedge_output / "wedge_list_018_1.star")
    )
    # check if given em file and output are the same
    assert compare_star_files(str(wedge_output / "wedge_list_018_1.star"), str(wedge_output / "wedge_list_018.star"))
    #check if returned pd is equal to given file
    assert compare_pd_and_star(df_18, str(wedge_output / "wedge_list_018.star"))
    if os.path.exists(str(wedge_output / "wedge_list_018_1.star")):
        os.remove(str(wedge_output / "wedge_list_018_1.star"))



def test_create_wedge_list_sg_batch():
    directory = Path(__file__).parent / "test_data"
    wedgeutils_datad = directory / "wedgeutils_data"
    pixel_size = 2.4
    tomo_list = wedgeutils_datad / "tomo_list.txt"
    tlt_file_format = directory / "TS_$xxx" / "$xxx.tlt"
    tomo_dim_file_format = directory / "TS_$xxx" / "tilt.com"
    z_shift_file_format = directory / "TS_$xxx" / "tilt.com"
    ctf_file_format = directory / "TS_$xxx" / "$xxx_gctf.star"
    output_file = directory / "wedgeutils_data" / "wedge_list_1.star"
    df17_18 = create_wedge_list_sg_batch(
        tomo_list = str(tomo_list),
        pixel_size = pixel_size,
        tlt_file_format = str(tlt_file_format),
        tomo_dim_file_format = str(tomo_dim_file_format),
        z_shift_file_format = str(z_shift_file_format),
        ctf_file_format = str(ctf_file_format),
        output_file = str(output_file)
    )
    assert compare_star_files(str(output_file), str(wedgeutils_datad / "wedge_list.star"))
    # check if returned pd is equal to given file
    assert compare_pd_and_star(df17_18, str(wedgeutils_datad / "wedge_list.star"))
    if os.path.exists(str(output_file)):
        os.remove(str(output_file))

    #case2: testing if not passing either tomo_dim_file_format / tomo_dim raises and exception
    with pytest.raises(ValueError):
        df17_18_1 = create_wedge_list_sg_batch(
            tomo_list = str(tomo_list),
            pixel_size = pixel_size,
            tlt_file_format = str(tlt_file_format),
            z_shift_file_format = str(z_shift_file_format),
            ctf_file_format = str(ctf_file_format)
        )
    #case3: passing a tomo_list of only 1 tomogram
    df17_18_2 = create_wedge_list_sg_batch(
        tomo_list = np.asarray(["018"]),
        pixel_size = pixel_size,
        tlt_file_format = str(tlt_file_format),
        tomo_dim_file_format = str(tomo_dim_file_format),
        z_shift_file_format = str(z_shift_file_format),
        ctf_file_format = str(ctf_file_format)
    )
    assert compare_pd_and_star(df17_18_2, str(wedgeutils_datad / "wedge_list_018.star"))

    #case4: passing empty tomo_list :
    #ioutils tlt_load has to be changed, to check that input isn't an empty list or an empty nparray
    #otherwise an empty df is created
    with pytest.raises(ValueError):
        df17_18_3 = create_wedge_list_sg_batch(
            tomo_list=np.asarray([]),
            pixel_size=pixel_size,
            tlt_file_format=str(tlt_file_format),
            tomo_dim_file_format=str(tomo_dim_file_format),
            z_shift_file_format=str(z_shift_file_format),
            ctf_file_format=str(ctf_file_format)
        )


    #case5: not passing either z_shift either z_shift_file_format results into df with Default z_shift = 0 value for all entries
    df17_18_4 = create_wedge_list_sg_batch(
        tomo_list = np.asarray(["017", "018"]),
        pixel_size = pixel_size,
        tlt_file_format = str(tlt_file_format),
        tomo_dim_file_format = str(tomo_dim_file_format),
        ctf_file_format = str(ctf_file_format),
    )
    assert df17_18_4["z_shift"].all() == 0

    #case5: not passing tomo_dim_file_format, but passing tomo_dim as dataframe
    tomo_dim_df = pd.DataFrame([[int("017"),4096.0,4096.0,1500.0],[int("018"),4096.0,4096.0,1800.0]])
    df_17_18_5 = create_wedge_list_sg_batch(
        tomo_list = str(tomo_list),
        pixel_size = pixel_size,
        z_shift_file_format=str(z_shift_file_format),
        tlt_file_format = str(tlt_file_format),
        tomo_dim = tomo_dim_df,
        ctf_file_format= str(ctf_file_format)
    )
    pd.testing.assert_frame_equal(df_17_18_5, df17_18)



def test_create_wedge_list_em_batch():
    directory = Path(__file__).parent / "test_data"
    wedgeutils_datad = directory / "wedgeutils_data"

    #case1: correct input to test given em file
    tomograms = wedgeutils_datad / "tomo_list.txt"
    tlt_file_format = str(directory / "TS_$xxx" / "$xxx.tlt")
    df_17 = create_wedge_list_em_batch(
        tomo_list=str(tomograms),
        tlt_file_format=tlt_file_format,
        output_file=str(wedgeutils_datad / "wedge_list_1.em")
    )
    emfile1 = emfile.read(str(wedgeutils_datad / "wedge_list_1.em"))[1]  # Get the data part
    emfile2 = emfile.read(str(wedgeutils_datad / "wedge_list.em"))[1]
    #assert output em file is the same as test em file
    np.testing.assert_equal(emfile1, emfile.read(str(wedgeutils_datad / "wedge_list.em"))[1])
    #assert dataframe returned is the same as what is written in output file
    assert np.allclose(df_17.to_numpy(), emfile1.squeeze(), rtol=1e-5, atol=1e-8)
    #assert dataframe returned is the same as test em file
    assert np.allclose(df_17.to_numpy(), emfile2.squeeze(), rtol=1e-5, atol=1e-8)
    if os.path.exists(str(wedgeutils_datad / "wedge_list_1.em")):
        os.remove(str(wedgeutils_datad / "wedge_list_1.em"))


def test_load_wedge_list_sg():
    filename = str(Path(__file__).parent / "test_data" / "TS_017" / "017_gctf.star")
    result = load_wedge_list_sg(filename)
    assert isinstance(result, pd.DataFrame)
    assert not result.empty
    input_df = pd.DataFrame({"col1": [10, 20], "col2": [30, 40]})
    result = load_wedge_list_sg(input_df)
    assert result.equals(input_df)
    with pytest.raises(ValueError,match="Inavlid input - only strings \(file names\) or pandas data frames are supported."):
        load_wedge_list_sg(123)

    if os.path.exists(filename):
        os.remove(filename)


def test_load_wedge_list_em():
    directory = Path(__file__).parent / "test_data"
    wedgeutils_datad = directory / "wedgeutils_data"
    wedge_listem = wedgeutils_datad / "wedge_list.em"

    # Test loading from file
    df_file = load_wedge_list_em(str(wedge_listem))
    assert isinstance(df_file, pd.DataFrame)
    assert list(df_file.columns) == ["tomo_id", "min_tilt_angle", "max_tilt_angle"]

    # Test loading from valid numpy array
    valid_array = np.array([[1, -60, 60], [2, -50, 50]])
    df_numpy = load_wedge_list_em(valid_array)
    assert isinstance(df_numpy, pd.DataFrame)
    assert df_numpy.shape == (2, 3)

    # Test loading from invalid numpy array
    invalid_array = np.array([[1, -60], [2, -50]])  # Wrong shape
    with pytest.raises(ValueError, match="correct shape"):
        load_wedge_list_em(invalid_array)

    # Test loading from valid DataFrame
    valid_df = pd.DataFrame({"tomo_id": [1, 2], "min_tilt_angle": [-60, -50], "max_tilt_angle": [60, 50]})
    df_dataframe = load_wedge_list_em(valid_df)
    assert isinstance(df_dataframe, pd.DataFrame)
    assert df_dataframe.shape == (2, 3)

    """# Test loading from invalid DataFrame (wrong column names)
    invalid_df = pd.DataFrame({"A": [1, 2], "B": [-60, -50], "C": [60, 50]})
    with pytest.raises(ValueError, match="correct shape"):
        load_wedge_list_em(invalid_df)"""

    # Test invalid input type
    with pytest.raises(ValueError, match="Invalid input"):
        load_wedge_list_em(12345)


def test_create_wg_mask():
    #print(sample_wedge_list)
    sample_wedge_list = str(Path(__file__).parent / "test_data" / "wedgeutils_data" / "wedge_list.star")
    sample_wedge_list = load_wedge_list_sg(sample_wedge_list)
    #sample_tomo_list = str(Path(__file__).parent / "test_data" / "wedgeutils_data" / "tomo_list.txt")
    sample_tomo_list = np.asarray([17,18])
    box_size = 64
    mask = create_wg_mask(sample_wedge_list, sample_tomo_list, box_size)

    assert isinstance(mask, np.ndarray)
    assert mask.shape == (box_size, box_size, box_size)

    box_size_array = np.array([64, 64, 64])
    mask = create_wg_mask(sample_wedge_list, sample_tomo_list, box_size_array)
    assert mask.shape == tuple(box_size_array), f"Forma errata: {mask.shape}"

    output_file = str(Path(__file__).parent / "test_data" / "wedgeutils_data" / "wedge_mask.em")
    mask = create_wg_mask(sample_wedge_list, sample_tomo_list, box_size, output_path=str(output_file))
    assert os.path.exists(output_file)

    with pytest.raises(ValueError):
        create_wg_mask(12345, sample_tomo_list, box_size)

    with pytest.raises(ValueError):
        create_wg_mask(sample_wedge_list, "wrong_tomo_list.txt", box_size)

    with pytest.raises(ValueError):
        create_wg_mask(sample_wedge_list, sample_tomo_list, [64, 64])

    """if os.path.exists(output_file):
        os.remove(output_file)"""


#TODO, to fix
def test_apply_wedge_mask():
    # Prepare sample data paths
    directory = Path(__file__).parent / "test_data"
    wedge_mask_path = str(directory / "wedgeutils_data" / "wedge_mask.em")
    in_map_path = str(directory /  "wedgeutils_data" / "wedge_list.em")  # Example input map file

    # Create a sample input map (you can mock this if needed for isolated testing)
    # Make sure the map is in a compatible format
    in_map = cryomap.read(in_map_path)  # This should return a numpy array

    # Apply the wedge mask without rotation or output path
    result_map = apply_wedge_mask(wedge_mask_path, in_map_path)

    # Test the type of the result
    assert isinstance(result_map, np.ndarray), "Output is not a numpy array"

    # Check that the output map has the same shape as the input map
    assert result_map.shape == in_map.shape, f"Expected shape {in_map.shape}, but got {result_map.shape}"

    # Test apply with rotation (example: rotate the map by 30 degrees)
    rotation = [30, 0, 0]  # Example rotation (adjust as needed)
    result_map_rot = apply_wedge_mask(wedge_mask_path, in_map_path, rotation_zxz=rotation)

    # Assert that the rotated map is still a valid numpy array
    assert isinstance(result_map_rot, np.ndarray), "Rotated map is not a numpy array"
    assert result_map_rot.shape == in_map.shape, f"Expected rotated map shape {in_map.shape}, but got {result_map_rot.shape}"

    # Test apply with output path (check if the file is created)
    output_file = str(Path(__file__).parent / "test_data" / "output_map.em")
    result_map_with_output = apply_wedge_mask(wedge_mask_path, in_map_path, output_path=output_file)

    # Ensure the output file was created
    assert os.path.exists(output_file), f"Output file {output_file} was not created"

    # Optionally, you can also check the contents of the output file if needed
    output_map = cryomap.read(output_file)
    assert isinstance(output_map, np.ndarray), "Output file does not contain a valid numpy array"

    # Test with invalid inputs
    with pytest.raises(ValueError):
        apply_wedge_mask(wedge_mask_path, "invalid_map_path", rotation_zxz=rotation)

    with pytest.raises(ValueError):
        apply_wedge_mask("invalid_wedge_mask.em", in_map_path, rotation_zxz=rotation)

    # Clean up output file after the test
    if os.path.exists(output_file):
        os.remove(output_file)

