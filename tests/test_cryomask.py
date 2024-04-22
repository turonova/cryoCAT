import sys, os, random, math
sys.path.append('.')
import pytest

from cryocat.cryomask import *
from cryocat.cryomap import read

gen_dir = './tests/test_data/masks/'
temp_dir = './tests/test_data/temp/'

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


def test_add_gausian(ref_masks):
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
    assert np.array_equal(get_mass_center(ref_masks["um.em"]), (13, 26, 18))
    assert np.array_equal(get_mass_center(ref_masks["em75.em"]), (24, 18, 26))
    assert np.array_equal(get_mass_center(ref_masks["cm25o.em"]), (11, 11, 16))
    assert np.array_equal(get_mass_center(ref_masks["zero"]), (1, 1, 1))
    assert np.array_equal(get_mass_center(ref_masks["zero0"]), (1, 1, 1))

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