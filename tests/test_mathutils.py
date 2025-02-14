from cryocat.mathutils import *
import numpy as np
import pytest
from skimage import filters, data, io, color
from pathlib import Path


def test_compute_rmse():
    #Case1:
    nparray1 = np.array([
        [5, 6],
        [7, 8],
        [1, 2],
        [3, 4]
    ])
    nparray2 = np.array([
        [8, 9],
        [7, 6],
        [4, 3],
        [0, 0]
    ])
    np.testing.assert_allclose(
        compute_rmse(nparray1, nparray2),
        np.array([2.598076, 2.738613]),
        rtol=1e-6,  # Relative tolerance
        atol=0  # Absolute tolerance
    )

    #Case2: floats
    nparray3 = np.array([
        [3.745401, 9.507143],
        [7.319939, 5.986585],
        [1.560186, 1.559945],
        [0.580836, 8.661761],
        [6.011150, 7.080726],
        [0.205845, 9.699099],
        [8.324426, 2.123391],
        [1.818250, 1.834045],
        [3.042422, 5.247564],
        [4.319450, 2.912291],
        [6.118529, 1.394939],
        [2.921446, 3.663618],
        [4.560700, 7.851760],
        [1.996738, 5.142344],
        [5.924146, 0.464504]
    ])
    nparray4 = np.array([
        [6.075449, 1.705241],
        [0.650516, 9.488855],
        [9.656320, 8.083973],
        [3.046138, 0.976721],
        [6.842330, 4.401525],
        [1.220382, 4.951769],
        [0.343885, 9.093204],
        [2.587800, 6.625223],
        [3.117111, 5.200680],
        [5.467103, 1.848545],
        [9.695846, 7.751328],
        [9.394989, 8.948273],
        [5.979000, 9.218742],
        [0.884925, 1.959829],
        [0.452273, 3.253303]
    ])
    np.testing.assert_allclose(
        compute_rmse(nparray3, nparray4),
        np.array([4.295366, 4.93455 ]),
        rtol=1e-6,
        atol=0
    )

    #test raise exception 1)
    with pytest.raises(ValueError):
        rmse = compute_rmse(nparray1, [1,2,3])

    with pytest.raises(ValueError):
        rmse = compute_rmse(nparray1, np.array(["str"]))

    with pytest.raises(ValueError):
        rmse = compute_rmse(np.array([1,2,3]), np.array([[1,2,3],[1,2,3]]))

    with pytest.raises(ValueError):
        rmse = compute_rmse(np.array([1,2,3]), np.array([1,2,3,4]))


def test_get_all_pairs():
    # Test with a list of 5 elements
    result = get_all_pairs([1, 2.3, 3, 4, 5])
    expected = [
        (1, 2.3), (1, 3), (1, 4), (1, 5),
        (2.3, 3), (2.3, 4), (2.3, 5),
        (3, 4), (3, 5),
        (4, 5)
    ]
    assert result == expected

    assert(get_all_pairs([1,2])==[(1,2)])

    with pytest.raises(ValueError):
        result = get_all_pairs(np.array([1,2,3]))
    with pytest.raises(ValueError):
        result = get_all_pairs(["str", "str1", 1, 2])

    #one element
    assert(get_all_pairs([1])==[])

def test_get_numbers_of_digits():
    assert get_number_of_digits(12345) == 5  # 12345 has 5 digits
    assert get_number_of_digits(0) == 1  # 0 has 1 digit
    assert get_number_of_digits(-987) == 3  # -987 has 3 digits (ignoring the minus sign)
    assert get_number_of_digits(3.14) == 3  # 3.14 has 3 digits (excluding the decimal point)
    assert get_number_of_digits(0.00123) == 6  # 0.00123 has 5 digits
    assert get_number_of_digits(-1.2345) == 5  # -1.2345 has 5 digits (ignoring the minus sign)
    with pytest.raises(ValueError):
        exc = get_number_of_digits("wrong")

def test_get_similar_size_factors():
    # Test with smaller numbers
    assert get_similar_size_factors(28, order="ascending") == (4, 7)
    assert get_similar_size_factors(28, order="descending") == (7, 4)
    assert get_similar_size_factors(7, order="ascending") == (1, 7)
    assert get_similar_size_factors(7, order="descending") == (7, 1)
    assert get_similar_size_factors(36, order="ascending") == (6, 6)
    assert get_similar_size_factors(36, order="descending") == (6, 6)
    assert get_similar_size_factors(10007, order="ascending") == (1, 10007)
    assert get_similar_size_factors(10007, order="descending") == (10007, 1)
    assert get_similar_size_factors(100003, order="ascending") == (1, 100003)
    assert get_similar_size_factors(100003, order="descending") == (100003, 1)
    assert get_similar_size_factors(100000, order="ascending") == (250, 400)
    assert get_similar_size_factors(100000, order="descending") == (400, 250)
    assert get_similar_size_factors(123456, order="ascending") == (192, 643)
    assert get_similar_size_factors(123456, order="descending") == (643, 192)

    # Test invalid input (non-integer)
    with pytest.raises(ValueError):
        get_similar_size_factors(100.5)
    with pytest.raises(ValueError):
        get_similar_size_factors("wrong")
    with pytest.raises(ValueError):
        get_similar_size_factors(-1)

    assert get_similar_size_factors(1, order="ascending") == (1, 1)
    assert get_similar_size_factors(1, order="descending") == (1, 1)
    assert get_similar_size_factors(999999, order="ascending") == (999, 1001)
    assert get_similar_size_factors(999999, order="descending") == (1001, 999)



def generate_cc_values(size=1000):
    """
    Simulate cross-correlation values with a bimodal-like histogram.
    """
    group1 = np.random.normal(0.2, 0.05, size // 2)  # Non-significant group
    group2 = np.random.normal(0.7, 0.1, size // 2)  # Significant group
    return np.concatenate([group1, group2])
@pytest.mark.parametrize("size", [500, 1000, 5000, 50000])
def test_otsu_threshold_cc(size):
    """
    Test the custom Otsu threshold function using simulated cross-correlation (CC) values.
    """
    #generate cc values
    #Case 50000 takes a bit of time but eventually is ok
    values = generate_cc_values(size)
    otsu_skimage = filters.threshold_otsu(values)
    otsu_custom = otsu_threshold(values)
    print(f"thr difference for size={size}: skimage={otsu_skimage}, custom={otsu_custom}, diff={abs(otsu_custom-otsu_skimage)}")
    assert abs(otsu_custom - otsu_skimage) < 1
    # binarize values using both thresholds
    binary_skimage = (values > otsu_skimage).astype(int)
    binary_custom = (values > otsu_custom).astype(int)
    #percentage difference in binarization results
    diff_percentage = np.sum(binary_skimage != binary_custom) / len(values) * 100
    print(f"difference for size={size}: {diff_percentage:.2f}%")
    assert diff_percentage < 1

def test_compute_frequency_array():
    test_cases = [
        ((4, 4), 1.0),
        ((5, 5), 0.5),
        ((3, 3), 2.0),
        ((4, 6), 1.0),
        ((2, 2), 0.1),
    ]

    for shape, pixel_size in test_cases:
        freq_array = compute_frequency_array(shape, pixel_size)
        assert freq_array.shape == shape
        assert np.all(freq_array >= 0)
        center = tuple(s // 2 for s in shape)
        assert freq_array[center] == 0
        # Check symmetry
        for idx in np.ndindex(freq_array.shape):
            mirror_idx = tuple((2 * c - i) % s for i, c, s in zip(idx, center, shape))
            assert np.isclose(freq_array[idx], freq_array[mirror_idx])

    shape = (5,)
    pixel_size = 1.0
    expected_magnitudes = np.array([0.4, 0.2, 0.0, 0.2, 0.4])  # Precomputed expected values
    freqs = compute_frequency_array(shape, pixel_size)
    np.testing.assert_allclose(freqs, expected_magnitudes, rtol=1e-6)
    assert np.allclose(freqs, np.flip(freqs))

    shape = (3, 3)
    pixel_size = 1.0
    expected_magnitudes = np.array([
        [np.sqrt(2) * 1 / 3, 1 / 3, np.sqrt(2) * 1 / 3],
        [1 / 3, 0.0, 1 / 3],
        [np.sqrt(2) * 1 / 3, 1 / 3, np.sqrt(2) * 1 / 3]
    ])
    freqs = compute_frequency_array(shape, pixel_size)
    np.testing.assert_allclose(freqs, expected_magnitudes, rtol=1e-6)
    assert np.allclose(freqs, np.flip(freqs))
