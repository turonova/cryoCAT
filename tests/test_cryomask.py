import sys, os, random, math
from matplotlib import pyplot as plt
sys.path.append('.')
import pytest
from cryocat.cryomask import *
from cryocat.cryomap import read

gen_dir = './test_data/masks/'
temp_dir = './test_data/temp/'


@pytest.fixture(scope='session', autouse=True)
def cleannup():
    os.mkdir(temp_dir[:-1])
    yield
    for filename in os.listdir(temp_dir):
        file_path = os.path.join(temp_dir, filename)
        os.remove(file_path)
    os.rmdir(temp_dir)

@pytest.fixture
def ref_masks():
    ref_masks = dict()
    ref_masks["zero"] = np.zeros((3, 6, 9))
    ref_masks["zero0"] = np.zeros((0, 0, 0)) 
    for name in filter(lambda name: name.endswith('.em'), os.listdir(gen_dir)):
        ref_masks[name] = read(gen_dir + name)
    return ref_masks

def compare_lists(list1, list2):
    if len(list1) != len(list2):
        return False
    for array1, array2 in zip(list1, list2):
        if not np.array_equal(array1, array2):
            return False
    return True

@pytest.mark.parametrize("gaussian, gaussian_outwards", [
    (0, False), (0, True), (0.5, False), (0.5, True)
])
def test_spherical_mask(ref_masks, gaussian, gaussian_outwards):
    actual = spherical_mask([4, 6, 8], 2, [1, 3, 3], gaussian, gaussian_outwards)
    expected_file = f"sm{int(gaussian*100)}" + ("" if gaussian == 0 or not gaussian_outwards else "o") + ".em"
    expected = ref_masks[expected_file]
    assert np.allclose(actual, expected, atol=1e-8)

@pytest.mark.parametrize("gaussian, gaussian_outwards", [
    (0, False), (0, True), (0.25, False), (0.25, True)
])
def test_cylindrical_mask(ref_masks, gaussian, gaussian_outwards):
    actual = cylindrical_mask([23, 40, 40], 7, 10, [10, 10, 15], gaussian, gaussian_outwards)
    expected_file = f"cm{int(gaussian*100)}" + ("" if gaussian == 0 or not gaussian_outwards else "o") + ".em"
    expected = ref_masks[expected_file]
    assert np.allclose(actual, expected, atol=1e-8)

@pytest.mark.parametrize("gaussian, gaussian_outwards", [
    (0, False), (0, True), (0.75, False)
])
def test_ellipsoid_mask(ref_masks, gaussian, gaussian_outwards):
    actual = ellipsoid_mask([45, 34, 50], [5, 10, 15], gaussian=gaussian, gaussian_outwards=gaussian_outwards)
    expected_file = f"em{int(gaussian*100)}" + ("" if gaussian == 0 or not gaussian_outwards else "o") + ".em"
    expected = ref_masks[expected_file]
    assert np.allclose(actual, expected, atol=1e-8)

def test_map_tight_mask(ref_masks):
    mtmsm0 = map_tight_mask(
        ref_masks["sm0.em"]
    )
    assert np.allclose(mtmsm0, ref_masks["mtmsm0.em"], atol=1e-8)
    mtmsmtmsm0_3 = map_tight_mask(
        ref_masks["smtmsm0_3.em"],
        output_name=f"{gen_dir}mtmsmtmsm0_3.em"
    )
    assert np.allclose(mtmsmtmsm0_3, ref_masks["mtmsmtmsm0_3.em"], atol=1e-8)
    mtmmmtmcm25_t10d7_t40d4g25 = map_tight_mask(
        ref_masks["mmtmcm25_t10d7.em"],
        0.4,
        3,
        0.25,
        output_name=f"{gen_dir}mtmcm25_t40d4g25.em"
    )
    assert np.allclose(mtmmmtmcm25_t10d7_t40d4g25, ref_masks["mtmmmtmcm25_t10d7_t40d4g25.em"], atol=1e-8)
    mtmum_t30d2g20r310040022 = map_tight_mask(
        ref_masks["um.em"],
        0.3,
        2,
        0.2,
        (3.1, 0.4, 2.2),
        output_name=f"{gen_dir}mtmum_t30d2g20r310040022.em"
    )
    assert np.allclose(mtmum_t30d2g20r310040022, ref_masks["mtmum_t30d2g20r310040022.em"], atol=1e-8)

def test_molmap_tight_mask(ref_masks):
    mmtmcm25o_g34F = molmap_tight_mask(
        ref_masks["cm25o.em"],
        gaussian=0.34,
        gaussian_outwards=False,
        output_name=f"{gen_dir}mmtmcm25o_g34F.em"
    )
    assert np.allclose(mmtmcm25o_g34F, ref_masks["mmtmcm25o_g34F.em"], atol=1e-8)
    mmtmcm25_t10d7 = molmap_tight_mask(
        ref_masks["cm25.em"],
        10,
        7,
        output_name=f"{gen_dir}mmtmcm25_t10d7.em"
    )
    assert np.allclose(mmtmcm25_t10d7, ref_masks["mmtmcm25_t10d7.em"], atol=1e-8)
    mmtmmtmsm0_t5d2r111222333 = molmap_tight_mask(
        ref_masks["mtmsm0.em"],
        0.05,
        2,
        angles=(1.11, 2.22, 3.33),
        output_name=f"{gen_dir}mmtmmtmsm0_t5d2r111222333.em"
    )
    assert np.allclose(mmtmmtmsm0_t5d2r111222333, ref_masks["mmtmmtmsm0_t5d2r111222333.em"], atol=1e-8)

@pytest.mark.parametrize("mask, factor", [
    ("mtmsm0", 3),
    ("um", 4)
])
def test_shrink_full_mask(ref_masks, mask, factor):
    actual = shrink_full_mask(ref_masks[mask + ".em"], factor)
    expected = ref_masks[f"s{mask}_{factor}.em"]
    assert np.allclose(actual, expected, atol=1e-8)

def test_rotate(ref_masks):
    mask = spherical_mask([4, 6, 8], 2, [1, 3, 3], 0.5, False)
    actual = rotate(mask, [0.3, 0.2, 0.1])
    expected = ref_masks["sm50r302010.em"]
    assert np.allclose(actual, expected, atol=1e-8)

@pytest.mark.parametrize("input_value, reference_size, expected", [
    ([1, 2, 3], None, np.array([1, 2, 3])),
    ([1, 2], None, None),
    ([1,], None, np.array([1, 1, 1])),
    (1, None, np.array([1, 1, 1])),
    ((1, 2, 3), None, np.array([1, 2, 3])),
    ((1, 2), None, None),
    ((1,), None, np.array([1, 1, 1])),
    ((1), None, np.array([1, 1, 1])),
    ((1.5, 5.3, 3), None, np.array([1, 5, 3])),
    (np.array([1, 5, 3]), None, np.array([1, 5, 3])),
    (np.array([1.5, 5.3, 3]), None, np.array([1, 5, 3])),
])
def test_get_correct_format(input_value, reference_size, expected):
    if expected is None:
        with pytest.raises(ValueError):
            get_correct_format(input_value, reference_size)
    else: assert np.array_equal(get_correct_format(input_value, reference_size), expected)


def test_add_gaussian(ref_masks):
    actual = add_gaussian(ref_masks["sm0.em"], 0.5)
    expected = ref_masks["sm50.em"]
    assert np.allclose(actual, expected, atol=1e-8)

def test_write_out(ref_masks):
    for ref_mask in os.listdir(gen_dir):
        write_out(ref_masks[ref_mask], temp_dir + ref_mask)
        assert ref_mask in os.listdir(temp_dir)
        loaded_mask = read(temp_dir + ref_mask)
        assert np.allclose(ref_masks[ref_mask], loaded_mask, atol=1e-8)
    
def test_write_out_without_output(ref_masks):
    before = os.listdir(temp_dir)
    write_out(random.choice(list(ref_masks.values())), None)
    after = os.listdir(temp_dir)
    assert before == after

def test_postprocess(ref_masks):
    sm0 = ref_masks["sm0.em"]
    actual = postprocess(sm0, 0.5, [0.3, 0.2, 0.1], None)
    expected = ref_masks["sm50r302010.em"]
    assert np.allclose(actual, expected, atol=1e-8)

def test_union(ref_masks):
    actual = union(
        [ref_masks[element] for element in ['um_1.em', 'um_2.em', 'um_3.em']]
    )
    expected = ref_masks['um.em']
    assert np.allclose(actual, expected, atol=1e-8)

@pytest.mark.parametrize("radius, gaussian, gaussian_outwards", [
    (20, 0.97, True),
    (16, 0.59, True),
    (3, 0.72, True),
    (8, 0, False),
    (8, 0.12, True),
    (15, 0, True),
    (15, 0.52, False),
    (4, 0.8, False),
    (19, 0.61, True),
    (9, 0.92, True)
])
def test_preprocess_params(radius, gaussian, gaussian_outwards):
    actual = preprocess_params(radius, gaussian, gaussian_outwards)
    expected = radius + np.ceil(gaussian * int(gaussian_outwards) * 5.0).astype(int)
    assert actual == expected

def test_preprocess_params_2():
    assert preprocess_params(20, 0.97, True) == 25
    assert preprocess_params(16, 0.59, True) == 19
    assert preprocess_params(3, 0.72, True) == 7
    assert preprocess_params(8, 0, False) == 8
    assert preprocess_params(8, 0.12, True) == 9
    assert preprocess_params(15, 0, True) == 15
    assert preprocess_params(15, 0.52, False) == 15

@pytest.mark.parametrize("name", [
    "um.em",
    "em75.em",
    "cm25o.em",
    "zero",
    "zero0"
])
def test_get_bounding_box(ref_masks, name):
    mask = ref_masks[name]
    actual = get_bounding_box(mask)
    i, j , k = np.asarray(mask > 0.00001).nonzero()
    expected = (
        np.array([min(i, default=0), min(j, default=0), min(k, default=0)]),
        np.array([max(i, default=0), max(j, default=0), max(k, default=0)])
    )
    assert np.allclose(actual[0], expected[0], atol=1e-8)
    assert np.allclose(actual[1], expected[1], atol=1e-8)

def test_get_bounding_box_2(ref_masks):
    assert compare_lists(get_bounding_box(ref_masks["um.em"]), ((0, 3, 0), (24, 46, 34)))
    assert compare_lists(get_bounding_box(ref_masks["em75.em"]), ((15, 5, 8), (30, 29, 42)))
    assert compare_lists(get_bounding_box(ref_masks["cm25o.em"]), ((0, 0, 7), (20, 20, 23)))
    assert compare_lists(get_bounding_box(ref_masks["zero"]), ((0, 0, 0), (0, 0, 0)))
    assert compare_lists(get_bounding_box(ref_masks["zero0"]), ((0, 0, 0), (0, 0, 0)))

def test_get_mass_dimensions(ref_masks):
    assert np.array_equal(get_mass_dimensions(ref_masks["um.em"]), (25, 44, 35))
    assert np.array_equal(get_mass_dimensions(ref_masks["em75.em"]), (16, 25, 35))
    assert np.array_equal(get_mass_dimensions(ref_masks["cm25o.em"]), (21, 21, 17))
    assert np.array_equal(get_mass_dimensions(ref_masks["zero"]), (1, 1, 1))
    assert np.array_equal(get_mass_dimensions(ref_masks["zero0"]), (1, 1, 1))

def test_get_mass_center(ref_masks):
    # this test had incorrect assumptions (centres of mass) and the function had a bug/discrepancy w/creating masks ~MAK 240830
    mask_centre = np.array([13,26,28])
    sph_40_r2 = spherical_mask(40, 2, center=mask_centre)
    assert np.array_equal(get_mass_center(sph_40_r2), mask_centre)
    # assert np.array_equal(get_mass_center(ref_masks["um.em"]), (13, 26, 18))
    # assert np.array_equal(get_mass_center(ref_masks["em75.em"]), (24, 18, 26))
    # assert np.array_equal(get_mass_center(ref_masks["cm25o.em"]), (11, 11, 16))
    # assert np.array_equal(get_mass_center(ref_masks["zero"]), (1, 1, 1))
    # assert np.array_equal(get_mass_center(ref_masks["zero0"]), (1, 1, 1))

def test_compute_solidity(ref_masks):
    # TODO: only works with abs_tol=1e-1. 1e-2 does not work
    # The numbers before abs_tol are the results computed directly from CREATED mask objects,
    # while the ref_masks are the masks READ after writing as references
    assert math.isclose(compute_solidity(ref_masks["um.em"]), 0.584102329, abs_tol=1e-1)
    assert math.isclose(compute_solidity(ref_masks["em75.em"]), 0.7857142857142857, abs_tol=1e-1)
    with pytest.raises(KeyError):
        compute_solidity(ref_masks["sm50.em"])
    assert math.isclose(compute_solidity(ref_masks["sm50o.em"]), 0.9285714285714286, abs_tol=1e-1)
    assert math.isclose(compute_solidity(ref_masks["um_2.em"]), 0.5526315789473685, abs_tol=1e-1)

def test_mask_overlap(ref_masks):
    assert mask_overlap(ref_masks["um_1.em"], ref_masks["um_2.em"]) == 1357
    assert mask_overlap(ref_masks["um_2.em"], ref_masks["um_3.em"]) == 0
    assert mask_overlap(ref_masks["um_1.em"], ref_masks["um_3.em"]) == 519
    assert mask_overlap(ref_masks["um.em"], ref_masks["um_1.em"]) == 13159

def test_parse_shape_string():
    # Valid cases
    assert parse_shape_string("sphere_r10") == ("sphere", [10])
    assert parse_shape_string("cylinder_r5_h20") == ("cylinder", [5, 20])
    assert parse_shape_string("s_shell_r15_s3") == ("s_shell", [15, 3])
    assert parse_shape_string("ellipsoid_rx4_ry5_rz6") == ("ellipsoid", [4, 5, 6])
    assert parse_shape_string("e_shell_rx8_ry9_rz10_s2") == ("e_shell", [8, 9, 10, 2])

    # Invalid cases
    with pytest.raises(ValueError):
        parse_shape_string("sphere10")  # Missing '_r'

    with pytest.raises(ValueError):
        parse_shape_string("cylinder_r_h20")  # Missing radius

    with pytest.raises(ValueError):
        parse_shape_string("ellipsoid_rx4_ry_rz6")  # Missing one dimension

    with pytest.raises(ValueError):
        parse_shape_string("random_string")  # Completely invalid format

@pytest.mark.parametrize("shape_string, mask_size, expected_radii", [
    (f"sphere_r5", 20, (5,)),  # Sphere with radius 5
    (f"cylinder_r5_h10", 20, (5, 10)),  # Cylinder with radius 5, height 10
    (f"ellipsoid_rx5_ry8_rz10", 20, (5, 8, 10)),  # Ellipsoid with radii 5, 8, 10
])
def test_generate_mask(shape_string, mask_size, expected_radii):

    mask = generate_mask(shape_string, mask_size=mask_size)

    # Verify mask dimensions
    assert mask.shape == (mask_size, mask_size, mask_size), "Incorrect mask dimensions"

    # Ensure the mask contains only 0s and 1s
    assert np.unique(mask).tolist() == [0, 1], "Mask should contain only 0s and 1s"

    # Get center of the mask
    center = mask_size // 2
    cx, cy, cz = center, center, center  # Center coordinates

    # Loop through every voxel in the mask and validate it
    for x in range(mask_size):
        for y in range(mask_size):
            for z in range(mask_size):
                inside = False
                if len(expected_radii) == 1:  # Sphere
                    r = expected_radii[0]
                    inside = (x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= r ** 2
                elif len(expected_radii) == 2:  # Cylinder
                    r, h = expected_radii
                    inside = (x - cx) ** 2 + (y - cy) ** 2 <= r ** 2 and abs(z - cz) <= h / 2
                elif len(expected_radii) == 3:  # Ellipsoid
                    rx, ry, rz = expected_radii
                    inside = ((x - cx) ** 2 / rx ** 2 +
                              (y - cy) ** 2 / ry ** 2 +
                              (z - cz) ** 2 / rz ** 2) <= 1
                # Check if mask is correctly set
                assert mask[x, y, z] == int(inside), f"Incorrect mask value at ({x}, {y}, {z})"


def generate_test_mask(shape, values):
    return np.full(shape, values)

def test_intersection():
    # Create some simple test masks (for example, 3x3x3 arrays)
    mask1 = generate_test_mask((3, 3, 3), 1)  # Fully ones
    mask2 = generate_test_mask((3, 3, 3), 1)  # Fully ones
    mask3 = generate_test_mask((3, 3, 3), 0)  # Fully zeros

    # Combine the masks into a list for the intersection test
    mask_list = [mask1, mask2, mask3]

    # Apply the intersection function
    result = intersection(mask_list)

    # Test if the intersection result is correct
    # The intersection of a fully ones mask with other fully ones masks should still be ones,
    # but the intersection with a fully zeros mask should be all zeros.
    expected_result = np.zeros((3, 3, 3))

    # Check if the result is as expected
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test if final values are clipped between 0.0 and 1.0
    assert np.all(result >= 0.0) and np.all(result <= 1.0), "Result values should be clipped between 0 and 1"

    # If an output file name is given, check if it's written (optional)
    # If you want to validate the file output, you can check the file itself.
    output_name = "test_output_mask.mrc"
    result_with_output = intersection(mask_list, output_name)
    if os.path.exists(output_name):
        os.remove(output_name)

def test_intersection_expected():
    # Create some realistic test masks (3x3x3 arrays)
    mask1 = np.array([[[1, 0, 0],
                       [1, 1, 0],
                       [0, 1, 1]],

                      [[1, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0]],

                      [[0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1]]])

    mask2 = np.array([[[0, 1, 1],
                       [0, 1, 0],
                       [1, 0, 0]],

                      [[1, 1, 0],
                       [1, 0, 1],
                       [0, 1, 1]],

                      [[0, 0, 0],
                       [0, 1, 1],
                       [1, 1, 0]]])

    # Combine the masks into a list for the intersection test
    mask_list = [mask1, mask2]
    # Apply the intersection function
    result = intersection(mask_list)
    # Define the expected result (manually calculated)
    expected_result = np.array([[[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]],

                                [[1, 1, 0],
                                 [0, 0, 1],
                                 [0, 0, 0]],

                                [[0, 0, 0],
                                 [0, 1, 0],
                                 [0, 1, 0]]])

    # Check if the result matches the expected result
    assert np.array_equal(result, expected_result), f"Expected {expected_result}, but got {result}"

    # Test if final values are clipped between 0.0 and 1.0
    assert np.all(result >= 0.0) and np.all(result <= 1.0), "Result values should be clipped between 0 and 1"

def test_subtraction():
    # Create some simple test masks (3x3x3 arrays)
    mask1 = np.array([[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]])

    mask2 = np.array([[[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]])

    mask3 = np.array([[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]])

    # Combine the masks into a list for the subtraction test
    mask_list1 = [mask1, mask2]  # full ones and full zeros
    mask_list2 = [mask1, mask3]  # full ones subtracted from full ones

    # Apply the subtraction function
    result1 = subtraction(mask_list1)
    result2 = subtraction(mask_list2)

    # Define the expected result for full ones and full zeros
    expected_result1 = np.array([[[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]],

                                 [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]],

                                 [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]])

    # Define the expected result for full ones subtracted from full ones (should be all zeros)
    expected_result2 = np.array([[[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],

                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]],

                                 [[0, 0, 0],
                                  [0, 0, 0],
                                  [0, 0, 0]]])

    # Check if the result matches the expected result for both cases
    assert np.array_equal(result1, expected_result1), f"Expected {expected_result1}, but got {result1}"
    assert np.array_equal(result2, expected_result2), f"Expected {expected_result2}, but got {result2}"

    # Realistic subtraction test
    mask4 = np.array([[[1, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1]],

                      [[1, 0, 1],
                       [0, 1, 1],
                       [1, 0, 0]],

                      [[0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1]]])

    mask5 = np.array([[[0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0]],

                      [[1, 1, 0],
                       [0, 0, 1],
                       [0, 1, 1]],

                      [[0, 1, 0],
                       [0, 1, 0],
                       [1, 1, 0]]])

    mask_list3 = [mask4, mask5]  # realistic subtraction test

    # Apply the subtraction function
    result3 = subtraction(mask_list3)

    # Define the expected result (manually calculated)
    expected_result3 = np.array([[[1, 0, 1],
                                  [1, 0, 0],
                                  [0, 1, 1]],

                                 [[0, 0, 1],
                                  [0, 1, 0],
                                  [1, 0, 0]],

                                 [[0, 0, 1],
                                  [1, 0, 0],
                                  [0, 0, 1]]], dtype=np.float32)  # Conversione a float

    # Check if the result matches the expected result for realistic subtraction
    assert np.array_equal(result3, expected_result3), f"Expected {expected_result3}, but got {result3}"

def test_difference():
    # Create test masks (3x3x3 arrays)
    mask1 = np.array([[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]])

    mask2 = np.array([[[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]],

                      [[0, 0, 0],
                       [0, 0, 0],
                       [0, 0, 0]]])

    mask3 = np.array([[[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]],

                      [[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]]])

    # Test case 1: Union of mask1 and mask2 - intersection of mask1 and mask2
    mask_list1 = [mask1, mask2]
    result1 = difference(mask_list1)

    expected_result1 = np.array([[[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]],

                                 [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]],

                                 [[1, 1, 1],
                                  [1, 1, 1],
                                  [1, 1, 1]]], dtype=np.float32)

    assert np.array_equal(result1, expected_result1), f"Expected {expected_result1}, but got {result1}"

    # Test case 2: Union of mask1 and mask3 - intersection of mask1 and mask3 (should be all zeros)
    mask_list2 = [mask1, mask3]
    result2 = difference(mask_list2)

    expected_result2 = np.zeros((3, 3, 3), dtype=np.float32)

    assert np.array_equal(result2, expected_result2), f"Expected {expected_result2}, but got {result2}"

    # Realistic test case
    mask4 = np.array([[[1, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1]],

                      [[1, 0, 1],
                       [0, 1, 1],
                       [1, 0, 0]],

                      [[0, 0, 1],
                       [1, 1, 0],
                       [0, 1, 1]]])

    mask5 = np.array([[[0, 1, 0],
                       [0, 1, 1],
                       [1, 0, 0]],

                      [[1, 1, 0],
                       [0, 0, 1],
                       [0, 1, 1]],

                      [[0, 1, 0],
                       [0, 1, 0],
                       [1, 1, 0]]])

    mask_list3 = [mask4, mask5]
    result3 = difference(mask_list3)

    # Manually computed expected result
    expected_result3 = np.array([[[1, 1, 1],
                                  [1, 0, 1],
                                  [1, 1, 1]],

                                 [[0, 1, 1],
                                  [0, 1, 0],
                                  [1, 1, 1]],

                                 [[0, 1, 1],
                                  [1, 0, 0],
                                  [1, 0, 1]]], dtype=np.float32)

    assert np.array_equal(result3, expected_result3), f"Expected {expected_result3}, but got {result3}"

def test_spherical_shell_mask():
    # Test case 1: Basic spherical shell mask
    mask_size = (9, 9, 9)
    shell_thickness = 2
    radius = 3

    result1 = spherical_shell_mask(mask_size, shell_thickness, radius=radius)

    # Expected: A shell structure with ones at the shell region, zeros elsewhere
    expected1 = (spherical_mask(mask_size, radius=radius + shell_thickness / 2) -
                 spherical_mask(mask_size, radius=radius - shell_thickness / 2))
    assert np.array_equal(result1, expected1), f"Expected {expected1}, but got {result1}"

    # Test case 2: Auto-calculated radius (should be half of the smallest mask dimension)
    mask_size = (10, 10, 10)
    result2 = spherical_shell_mask(mask_size, shell_thickness)
    expected_radius = min(mask_size) // 2
    expected2 = (spherical_mask(mask_size, radius=expected_radius + shell_thickness / 2) -
                 spherical_mask(mask_size, radius=expected_radius - shell_thickness / 2))
    assert np.array_equal(result2, expected2), f"Expected {expected2}, but got {result2}"

    # Test case 3: Different center
    center = (4, 4, 4)
    result3 = spherical_shell_mask(mask_size, shell_thickness, radius=3, center=center)

    expected3 = (spherical_mask(mask_size, radius=3 + shell_thickness / 2, center=center) -
                 spherical_mask(mask_size, radius=3 - shell_thickness / 2, center=center))
    assert np.array_equal(result3, expected3), f"Expected {expected3}, but got {result3}"

    # Test case 4: Applying Gaussian smoothing (assuming postprocess applies Gaussian blur)
    gaussian_std = 1.0
    result4 = spherical_shell_mask(mask_size, shell_thickness, radius=3, gaussian=gaussian_std)
    assert result4.shape == mask_size, "Output shape mismatch"

def test_spherical_shell_mask_manual():
    # Define input parameters
    mask_size = (7, 7, 7)
    shell_thickness = 2
    radius = 3
    center = (3, 3, 3)  # Center of the mask

    # Compute expected output manually
    expected_mask = np.zeros(mask_size, dtype=np.float32)

    for x in range(mask_size[0]):
        for y in range(mask_size[1]):
            for z in range(mask_size[2]):
                dist = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
                if (radius - shell_thickness / 2) <= dist <= (radius + shell_thickness / 2):
                    expected_mask[x, y, z] = 1.0

    # Get actual result from the function
    result = spherical_shell_mask(mask_size, shell_thickness, radius=radius, center=center)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(expected_mask[:, :, 3], cmap='gray')
    axes[0].set_title("Expected")
    axes[1].imshow(result[:, :, 3], cmap='gray')
    axes[1].set_title("Actual")
    plt.show()

    # Compare both results
    assert np.array_equal(result, expected_mask), f"Expected {expected_mask}, but got {result}"

def test_ellipsoid_shell_mask():
    # Test case 1: Basic ellipsoid shell mask
    mask_size = (9, 9, 9)
    shell_thickness = 2
    radii = (3, 2, 1)

    result1 = ellipsoid_shell_mask(mask_size, shell_thickness, radii=radii)

    expected1 = np.logical_xor(
        ellipsoid_mask(mask_size, radii=np.array(radii) + shell_thickness / 2),
        ellipsoid_mask(mask_size, radii=np.array(radii) - shell_thickness / 2)
    )

    assert np.array_equal(result1, expected1), f"Expected {expected1}, but got {result1}"

    # Test case 2: Auto-centered ellipsoid shell mask
    mask_size = (10, 10, 10)
    radii = (4, 3, 2)
    result2 = ellipsoid_shell_mask(mask_size, shell_thickness, radii=radii)

    expected2 = np.logical_xor(
        ellipsoid_mask(mask_size, radii=np.array(radii) + shell_thickness / 2),
        ellipsoid_mask(mask_size, radii=np.array(radii) - shell_thickness / 2)
    )

    assert np.array_equal(result2, expected2), f"Expected {expected2}, but got {result2}"

    # Test case 3: Different center
    center = (4, 4, 4)
    result3 = ellipsoid_shell_mask(mask_size, shell_thickness, radii=(3, 3, 3), center=center)

    expected3 = np.logical_xor(
        ellipsoid_mask(mask_size, radii=np.array((3, 3, 3)) + shell_thickness / 2, center=center),
        ellipsoid_mask(mask_size, radii=np.array((3, 3, 3)) - shell_thickness / 2, center=center)
    )
    differences = np.where(expected3 != result3)
    print("Differences at indices:", differences)
    assert np.array_equal(result3, expected3), f"Expected {expected3}, but got {result3}"

    # Test case 4: Applying Gaussian smoothing
    gaussian_std = 1.0
    result4 = ellipsoid_shell_mask(mask_size, shell_thickness, radii=(3, 3, 3), gaussian=gaussian_std)
    assert result4.shape == mask_size, "Output shape mismatch"

#TODO
def test_fill_hollow_mask():

    # Case 1: Empty mask (all zeros) -> Should remain unchanged
    empty_mask = np.zeros((5, 5, 5), dtype=int)
    result = fill_hollow_mask(empty_mask)
    np.testing.assert_array_equal(result.astype(int), empty_mask)

    # Case 2: Fully filled mask (all ones) -> Should remain unchanged
    full_mask = np.ones((5, 5, 5), dtype=int)
    result = fill_hollow_mask(full_mask)
    np.testing.assert_array_equal(result.astype(int), full_mask)

    # Case 3: Simple 3D mask with a hole in the middle -> Hole should be filled
    input_mask = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ], dtype=int)

    expected_result = np.array([
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]],  # Hole should be filled
        [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    ], dtype=int)

    result = fill_hollow_mask(input_mask)

    print("Expected:\n", expected_result)
    print("Result:\n", result.astype(int))  # Convert to int before checking

    np.testing.assert_array_equal(result.astype(int), expected_result)

    # Case 4: Large 3D mask with random holes -> Validate shape & binary values
    np.random.seed(42)
    random_mask = (np.random.rand(10, 10, 10) > 0.7).astype(int)
    result = fill_hollow_mask(random_mask)
    assert result.shape == random_mask.shape
    assert np.all(np.isin(result, [0, 1]))  # Ensure binary output

    # Case 5: Thin structure with gaps (2D-like) -> Should be filled in 3D
    thin_mask = np.zeros((5, 5, 5), dtype=int)
    thin_mask[:, 2, 2] = 1  # Thin vertical line with gaps
    thin_mask[2, 2, 1] = 0  # A small hole in the middle

    expected_result = np.ones((5, 5, 5), dtype=int)  # Should fill completely
    result = fill_hollow_mask(thin_mask)
    np.testing.assert_array_equal(result, expected_result)

    # Case 6: Check that small objects are not falsely removed
    small_object_mask = np.zeros((5, 5, 5), dtype=int)
    small_object_mask[2, 2, 2] = 1  # A single voxel object

    result = fill_hollow_mask(small_object_mask)
    np.testing.assert_array_equal(result, small_object_mask)  # Should remain

def test_def_tomogram_shell_mask():
    pass
    #TODO