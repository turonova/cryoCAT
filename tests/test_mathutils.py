import numpy as np
import pytest
from skimage import filters

from cryocat.utils.mathutils import (
    compute_frequency_array,
    compute_rmse,
    get_all_pairs,
    get_number_of_digits,
    get_similar_size_factors,
    otsu_threshold,
    randomize_phases,
)


# ---------------------------------------------------------------------------
# compute_rmse
# ---------------------------------------------------------------------------

_RMSE_ARR1 = np.array([[5, 6], [7, 8], [1, 2], [3, 4]])
_RMSE_ARR2 = np.array([[8, 9], [7, 6], [4, 3], [0, 0]])
_RMSE_ARR3 = np.array([
    [3.745401, 9.507143], [7.319939, 5.986585], [1.560186, 1.559945],
    [0.580836, 8.661761], [6.011150, 7.080726], [0.205845, 9.699099],
    [8.324426, 2.123391], [1.818250, 1.834045], [3.042422, 5.247564],
    [4.319450, 2.912291], [6.118529, 1.394939], [2.921446, 3.663618],
    [4.560700, 7.851760], [1.996738, 5.142344], [5.924146, 0.464504],
])
_RMSE_ARR4 = np.array([
    [6.075449, 1.705241], [0.650516, 9.488855], [9.656320, 8.083973],
    [3.046138, 0.976721], [6.842330, 4.401525], [1.220382, 4.951769],
    [0.343885, 9.093204], [2.587800, 6.625223], [3.117111, 5.200680],
    [5.467103, 1.848545], [9.695846, 7.751328], [9.394989, 8.948273],
    [5.979000, 9.218742], [0.884925, 1.959829], [0.452273, 3.253303],
])


@pytest.mark.parametrize("arr1, arr2, expected", [
    (_RMSE_ARR1, _RMSE_ARR2, np.array([2.598076, 2.738613])),
    (_RMSE_ARR3, _RMSE_ARR4, np.array([4.295366, 4.93455])),
])
def test_compute_rmse(arr1, arr2, expected):
    np.testing.assert_allclose(compute_rmse(arr1, arr2), expected, rtol=1e-6)


@pytest.mark.parametrize("arr1, arr2", [
    (_RMSE_ARR1, [1, 2, 3]),
    (_RMSE_ARR1, np.array(["str"])),
    (np.array([1, 2, 3]), np.array([[1, 2, 3], [1, 2, 3]])),
    (np.array([1, 2, 3]), np.array([1, 2, 3, 4])),
])
def test_compute_rmse_raises(arr1, arr2):
    with pytest.raises(ValueError):
        compute_rmse(arr1, arr2)


# ---------------------------------------------------------------------------
# get_all_pairs
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("input_list, expected", [
    (
        [1, 2.3, 3, 4, 5],
        [(1, 2.3), (1, 3), (1, 4), (1, 5), (2.3, 3), (2.3, 4), (2.3, 5), (3, 4), (3, 5), (4, 5)],
    ),
    ([1, 2], [(1, 2)]),
    ([1], []),
])
def test_get_all_pairs(input_list, expected):
    assert get_all_pairs(input_list) == expected


@pytest.mark.parametrize("input_val", [
    np.array([1, 2, 3]),
    ["str", "str1", 1, 2],
])
def test_get_all_pairs_raises(input_val):
    with pytest.raises(ValueError):
        get_all_pairs(input_val)


# ---------------------------------------------------------------------------
# get_number_of_digits
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, expected", [
    (12345, 5),
    (0, 1),
    (-987, 3),
    (3.14, 3),
    (0.00123, 6),
    (-1.2345, 5),
])
def test_get_number_of_digits(n, expected):
    assert get_number_of_digits(n) == expected


def test_get_number_of_digits_raises():
    with pytest.raises(ValueError):
        get_number_of_digits("wrong")


# ---------------------------------------------------------------------------
# get_similar_size_factors
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n, order, expected", [
    (28, "ascending", (4, 7)),
    (28, "descending", (7, 4)),
    (7, "ascending", (1, 7)),
    (7, "descending", (7, 1)),
    (36, "ascending", (6, 6)),
    (36, "descending", (6, 6)),
    (10007, "ascending", (1, 10007)),
    (10007, "descending", (10007, 1)),
    (100003, "ascending", (1, 100003)),
    (100003, "descending", (100003, 1)),
    (100000, "ascending", (250, 400)),
    (100000, "descending", (400, 250)),
    (123456, "ascending", (192, 643)),
    (123456, "descending", (643, 192)),
    (1, "ascending", (1, 1)),
    (1, "descending", (1, 1)),
    (999999, "ascending", (999, 1001)),
    (999999, "descending", (1001, 999)),
])
def test_get_similar_size_factors(n, order, expected):
    assert get_similar_size_factors(n, order=order) == expected


@pytest.mark.parametrize("n", [100.5, "wrong", -1])
def test_get_similar_size_factors_raises(n):
    with pytest.raises(ValueError):
        get_similar_size_factors(n)


# ---------------------------------------------------------------------------
# otsu_threshold
# ---------------------------------------------------------------------------

def _generate_cc_values(size=1000):
    group1 = np.random.normal(0.2, 0.05, size // 2)
    group2 = np.random.normal(0.7, 0.1, size // 2)
    return np.concatenate([group1, group2])


@pytest.mark.parametrize("size", [500, 1000, 5000])
def test_otsu_threshold_cc(size):
    values = _generate_cc_values(size)
    otsu_skimage = filters.threshold_otsu(values)
    otsu_custom = otsu_threshold(values)
    assert abs(otsu_custom - otsu_skimage) < 1
    binary_skimage = (values > otsu_skimage).astype(int)
    binary_custom = (values > otsu_custom).astype(int)
    diff_percentage = np.sum(binary_skimage != binary_custom) / len(values) * 100
    assert diff_percentage < 1


# ---------------------------------------------------------------------------
# compute_frequency_array
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape, pixel_size", [
    ((4, 4), 1.0),
    ((5, 5), 0.5),
    ((3, 3), 2.0),
    ((4, 6), 1.0),
    ((2, 2), 0.1),
])
def test_compute_frequency_array_properties(shape, pixel_size):
    freq_array = compute_frequency_array(shape, pixel_size)
    assert freq_array.shape == shape
    assert np.all(freq_array >= 0)
    center = tuple(s // 2 for s in shape)
    assert freq_array[center] == 0
    for idx in np.ndindex(freq_array.shape):
        mirror_idx = tuple((2 * c - i) % s for i, c, s in zip(idx, center, shape))
        assert np.isclose(freq_array[idx], freq_array[mirror_idx])


def test_compute_frequency_array_1d():
    expected = np.array([0.4, 0.2, 0.0, 0.2, 0.4])
    freqs = compute_frequency_array((5,), 1.0)
    np.testing.assert_allclose(freqs, expected, rtol=1e-6)
    assert np.allclose(freqs, np.flip(freqs))


def test_compute_frequency_array_3x3():
    expected = np.array([
        [np.sqrt(2) / 3, 1 / 3, np.sqrt(2) / 3],
        [1 / 3, 0.0, 1 / 3],
        [np.sqrt(2) / 3, 1 / 3, np.sqrt(2) / 3],
    ])
    freqs = compute_frequency_array((3, 3), 1.0)
    np.testing.assert_allclose(freqs, expected, rtol=1e-6)
    assert np.allclose(freqs, np.flip(freqs))


# ---------------------------------------------------------------------------
# randomize_phases
# ---------------------------------------------------------------------------

def test_randomize_phases():
    np.random.seed(42)
    box = 16
    vol = np.random.rand(box, box, box).astype(np.float64)
    fourier_cutoff = 4

    result = randomize_phases(vol, fourier_cutoff)

    assert result.shape == vol.shape
    assert np.isrealobj(result)
    assert not np.allclose(result, vol)

    ft_orig = np.fft.fftshift(np.fft.fftn(vol))
    ft_rand = np.fft.fftshift(np.fft.fftn(result))
    dist = compute_frequency_array(vol.shape, 1) * box
    low_mask = dist < fourier_cutoff
    np.testing.assert_allclose(
        np.angle(ft_orig[low_mask]),
        np.angle(ft_rand[low_mask]),
        atol=1e-10,
    )


def test_randomize_phases_identity_at_large_cutoff():
    np.random.seed(0)
    vol = np.random.rand(16, 16, 16).astype(np.float64)
    result = randomize_phases(vol, fourier_cutoff=100)
    np.testing.assert_allclose(result, vol, atol=1e-10)


def test_randomize_phases_constant_volume():
    const_vol = np.ones((8, 8, 8)) * 2.5
    np.testing.assert_allclose(randomize_phases(const_vol, fourier_cutoff=2), const_vol, atol=1e-10)
