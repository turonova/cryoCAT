import numpy as np
import pandas as pd
import pytest

from cryocat.utils.imageutils import (
    generate_ctf_slice,
    apply_bandpass,
    apply_highpass,
    apply_lowpass,
    apply_wiener_radial,
    calculate_conjugates,
    compute_wiener_1d,
    extract_orthogonal_lines_1d,
    extract_orthogonal_slices_2d,
    find_peak_3d,
    gaussian_smooth,
    gaussian_threshold,
    get_label_bounding_box,
    label_at_point,
    label_connected_components,
    morphology_open_close,
    triangle_threshold,
    calculate_flcf,
    compute_ctf_1d,
    compute_ctf_2d,
    compute_frequency_array,
    get_filter_radius,
    mask_voxel_count_and_bbox,
    otsu_threshold,
    randomize_phases,
    rotate_2d,
    normalize_array,
    normalize_under_mask,
    binarize,
    rotate_volume,
    shift_array,
    flip_array,
    recenter_volume,
    pad_volume,
    symmetrize_volume,
    equalize_histogram_2d,
    apply_fft_filter,
    mask_overlap,
)


# ---------------------------------------------------------------------------
# compute_frequency_array
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("shape, pixel_size", [
    ((4, 4), 1.0),
    ((5, 5), 0.5),
    ((4, 6), 1.0),
])
def test_compute_frequency_array_shape(shape, pixel_size):
    arr = compute_frequency_array(shape, pixel_size)
    assert arr.shape == shape
    assert np.all(arr >= 0)


def test_compute_frequency_array_center_zero():
    arr = compute_frequency_array((8, 8, 8), 1.0)
    assert arr[4, 4, 4] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# randomize_phases
# ---------------------------------------------------------------------------

def test_randomize_phases_shape():
    np.random.seed(0)
    vol = np.random.rand(16, 16, 16)
    result = randomize_phases(vol, fourier_cutoff=4)
    assert result.shape == vol.shape
    assert np.isrealobj(result)


def test_randomize_phases_preserves_low_freq():
    np.random.seed(42)
    vol = np.random.rand(16, 16, 16).astype(np.float64)
    result = randomize_phases(vol, fourier_cutoff=4)
    ft_orig = np.fft.fftshift(np.fft.fftn(vol))
    ft_rand = np.fft.fftshift(np.fft.fftn(result))
    dist = compute_frequency_array(vol.shape, 1) * vol.shape[0]
    low_mask = dist < 4
    np.testing.assert_allclose(
        np.angle(ft_orig[low_mask]), np.angle(ft_rand[low_mask]), atol=1e-10
    )


# ---------------------------------------------------------------------------
# otsu_threshold
# ---------------------------------------------------------------------------

def test_otsu_threshold_bimodal():
    np.random.seed(1)
    low = np.random.normal(0.2, 0.05, 500)
    high = np.random.normal(0.7, 0.1, 500)
    values = np.concatenate([low, high])
    threshold = otsu_threshold(values)
    assert 0.3 < threshold < 0.6


# ---------------------------------------------------------------------------
# get_filter_radius
# ---------------------------------------------------------------------------

def test_get_filter_radius_fourier_pixels():
    assert get_filter_radius(100, fourier_pixels=20, target_resolution=None, pixel_size=None) == 20


def test_get_filter_radius_from_resolution():
    # 100 * 1.5 / 15 = 10
    assert get_filter_radius(100, fourier_pixels=None, target_resolution=15.0, pixel_size=1.5) == 10


def test_get_filter_radius_raises_without_params():
    with pytest.raises(ValueError):
        get_filter_radius(100, fourier_pixels=None, target_resolution=None, pixel_size=None)


# ---------------------------------------------------------------------------
# rotate_2d
# ---------------------------------------------------------------------------

def test_rotate_2d_shape():
    img = np.random.rand(32, 32)
    assert rotate_2d(img, 45).shape == img.shape


def test_rotate_2d_zero_is_identity():
    img = np.random.rand(16, 16)
    np.testing.assert_array_almost_equal(rotate_2d(img, 0), img)


def test_rotate_2d_360_is_identity():
    img = np.random.rand(16, 16)
    np.testing.assert_array_almost_equal(rotate_2d(img, 360), img, decimal=5)


# ---------------------------------------------------------------------------
# compute_ctf_1d
# ---------------------------------------------------------------------------

def test_compute_ctf_1d_shape():
    ctf = compute_ctf_1d(256, 1e-10, 300e3, 2.7e-3, -2e-6, 0.07, 0.0, 0)
    assert ctf.shape == (256,)


def test_compute_ctf_1d_finite():
    ctf = compute_ctf_1d(256, 1e-10, 300e3, 2.7e-3, -2e-6, 0.07, 0.0, 0)
    assert np.all(np.isfinite(ctf))


# ---------------------------------------------------------------------------
# compute_ctf_2d
# ---------------------------------------------------------------------------

def test_compute_ctf_2d_shape():
    defocus = np.array([[2.0], [3.0]])
    pshift = np.zeros((2, 1))
    f = np.linspace(0, 0.5, 20)
    result = compute_ctf_2d(defocus, pshift, 0.07, 2.7, 300, f)
    assert result.shape == (2, 20)


def test_compute_ctf_2d_range():
    defocus = np.array([[2.0]])
    pshift = np.zeros((1, 1))
    f = np.linspace(0, 0.5, 20)
    result = compute_ctf_2d(defocus, pshift, 0.07, 2.7, 300, f)
    assert np.all(result >= -1.0) and np.all(result <= 1.0)


# ---------------------------------------------------------------------------
# calculate_conjugates
# ---------------------------------------------------------------------------

def test_calculate_conjugates_shapes():
    vol = np.random.rand(8, 8, 8)
    conj, conj_sq = calculate_conjugates(vol)
    assert conj.shape == vol.shape
    assert conj_sq.shape == vol.shape


def test_calculate_conjugates_zero_dc():
    vol = np.random.rand(8, 8, 8)
    conj, _ = calculate_conjugates(vol)
    # DC component (0,0,0) must be zero → conj[0,0,0] == conj(0) = 0
    assert conj.flat[0] == pytest.approx(0.0)


def test_calculate_conjugates_with_filter():
    vol = np.random.rand(8, 8, 8)
    filt = np.ones((8, 8, 8))
    conj_no_filt, _ = calculate_conjugates(vol)
    conj_filt, _ = calculate_conjugates(vol, filter=filt)
    np.testing.assert_array_almost_equal(conj_no_filt, conj_filt)


# ---------------------------------------------------------------------------
# calculate_flcf
# ---------------------------------------------------------------------------

def test_calculate_flcf_shape():
    np.random.seed(7)
    vol1 = np.random.rand(8, 8, 8).astype(np.float32)
    mask = np.ones((8, 8, 8), dtype=np.float32)
    vol2 = np.random.rand(8, 8, 8).astype(np.float32)
    result = calculate_flcf(vol1, mask, vol2=vol2)
    assert result.shape == (8, 8, 8)


def test_calculate_flcf_range():
    np.random.seed(8)
    vol1 = np.random.rand(8, 8, 8).astype(np.float32)
    mask = np.ones((8, 8, 8), dtype=np.float32)
    vol2 = np.random.rand(8, 8, 8).astype(np.float32)
    result = calculate_flcf(vol1, mask, vol2=vol2)
    assert np.all(result >= 0.0) and np.all(result <= 1.0)


def test_calculate_flcf_nan_raises():
    vol1 = np.full((4, 4, 4), np.nan)
    mask = np.ones((4, 4, 4))
    with pytest.raises(ValueError):
        calculate_flcf(vol1, mask, vol2=np.ones((4, 4, 4)))


def test_calculate_flcf_no_vol2_no_conj_raises():
    vol1 = np.random.rand(4, 4, 4)
    mask = np.ones((4, 4, 4))
    with pytest.raises(ValueError):
        calculate_flcf(vol1, mask)


# ---------------------------------------------------------------------------
# mask_voxel_count_and_bbox
# ---------------------------------------------------------------------------

def test_mask_voxel_count_and_bbox_full():
    mask = np.ones((8, 8, 8))
    n, bbox = mask_voxel_count_and_bbox(mask)
    assert n == 512
    np.testing.assert_array_equal(bbox, [8, 8, 8])


def test_mask_voxel_count_and_bbox_partial():
    mask = np.zeros((10, 10, 10))
    mask[2:5, 2:5, 2:5] = 1.0
    n, bbox = mask_voxel_count_and_bbox(mask)
    assert n == 27
    np.testing.assert_array_equal(bbox, [3, 3, 3])


def test_mask_voxel_count_and_bbox_empty():
    mask = np.zeros((6, 6, 6))
    n, bbox = mask_voxel_count_and_bbox(mask)
    assert n == 0
    np.testing.assert_array_equal(bbox, [0, 0, 0])


def test_mask_voxel_count_and_bbox_soft():
    mask = np.full((8, 8, 8), 0.9)
    n, _ = mask_voxel_count_and_bbox(mask, threshold=0.5)
    assert n == 512


def test_mask_voxel_count_and_bbox_soft_excluded():
    mask = np.full((4, 4, 4), 0.5)
    n, _ = mask_voxel_count_and_bbox(mask, threshold=0.5)
    assert n == 0


# ---------------------------------------------------------------------------
# apply_lowpass / apply_highpass / apply_bandpass
# ---------------------------------------------------------------------------

def test_apply_lowpass_shape():
    vol = np.random.rand(16, 16, 16)
    result = apply_lowpass(vol, radius=4)
    assert result.shape == vol.shape
    assert np.isrealobj(result)


def test_apply_lowpass_attenuates_high_freq():
    """A sharp high-freq signal should be suppressed by lowpass."""
    np.random.seed(10)
    vol = np.random.rand(16, 16, 16)
    lp = apply_lowpass(vol, radius=2, gaussian=0)
    # Power in the high-freq region of lp should be less than in original
    ft_orig = np.abs(np.fft.fftshift(np.fft.fftn(vol)))
    ft_lp = np.abs(np.fft.fftshift(np.fft.fftn(lp)))
    # outer shell (radius > 6): lowpass should have less power
    from cryocat.utils.imageutils import compute_frequency_array
    dist = compute_frequency_array(vol.shape, 1.0) * vol.shape[0]
    outer = dist > 6
    assert ft_lp[outer].sum() < ft_orig[outer].sum()


def test_apply_highpass_shape():
    vol = np.random.rand(16, 16, 16)
    result = apply_highpass(vol, radius=4)
    assert result.shape == vol.shape
    assert np.isrealobj(result)


def test_apply_highpass_suppresses_dc():
    """Highpass should remove DC (mean ≈ 0 for zero-mean input)."""
    np.random.seed(11)
    vol = np.random.rand(16, 16, 16)
    hp = apply_highpass(vol, radius=2, gaussian=0)
    assert abs(hp.mean()) < abs(vol.mean())


def test_apply_bandpass_shape():
    vol = np.random.rand(16, 16, 16)
    result = apply_bandpass(vol, lp_radius=6, hp_radius=2)
    assert result.shape == vol.shape
    assert np.isrealobj(result)


def test_apply_bandpass_is_lp_minus_hp():
    """bandpass(lp, hp) should equal lowpass(lp) − highpass_complement(lp, hp)."""
    np.random.seed(12)
    vol = np.random.rand(16, 16, 16).astype(np.float64)
    bp = apply_bandpass(vol, lp_radius=6, hp_radius=2, lp_gaussian=0, hp_gaussian=0)
    lp = apply_lowpass(vol, radius=6, gaussian=0)
    hp_of_lp = apply_highpass(lp, radius=2, gaussian=0)
    np.testing.assert_allclose(bp, hp_of_lp, atol=1e-10)


def test_apply_lowpass_gaussian_zero_is_sharp():
    """With gaussian=0, the filter should be a sharp step (no values between 0 and 1 at non-edge)."""
    vol = np.ones((8, 8, 8))
    lp = apply_lowpass(vol, radius=3, gaussian=0)
    # All values should be very close to 0 or 1
    assert np.all((lp < 0.01) | (lp > 0.99))


# ---------------------------------------------------------------------------
# gaussian_smooth
# ---------------------------------------------------------------------------

def test_gaussian_smooth_zero_sigma_identity():
    arr = np.random.rand(8, 8, 8)
    assert gaussian_smooth(arr, 0) is arr


def test_gaussian_smooth_shape():
    arr = np.random.rand(12, 12, 12)
    result = gaussian_smooth(arr, sigma=1.5)
    assert result.shape == arr.shape


# ---------------------------------------------------------------------------
# generate_ctf_slice
# ---------------------------------------------------------------------------

def _make_wl(tomo_size=64, pixelsize=3.5, defocus=2.0, pshift=None):
    data = {
        "tomo_x": [tomo_size], "tomo_y": [tomo_size], "tomo_z": [tomo_size],
        "pixelsize": [pixelsize], "defocus": [defocus],
        "amp_contrast": [0.07], "cs": [2.7], "voltage": [300.0],
    }
    if pshift is not None:
        data["pshift"] = [pshift]
    return pd.DataFrame(data)


class TestGenerateCtfSlice:
    _SZ = 16

    @pytest.fixture
    def simple_setup(self):
        sw = np.ones((self._SZ, self._SZ, self._SZ))
        idx = np.nonzero(sw)
        wl = _make_wl(tomo_size=64)
        return wl, [idx], sw

    def test_output_shape_matches_slice_weight(self, simple_setup):
        wl, slice_idx, sw = simple_setup
        result = generate_ctf_slice(wl, slice_idx, sw, binning=1)
        assert result.shape == sw.shape

    def test_output_is_finite(self, simple_setup):
        wl, slice_idx, sw = simple_setup
        result = generate_ctf_slice(wl, slice_idx, sw, binning=1)
        assert np.all(np.isfinite(result))

    def test_without_pshift_no_error(self):
        wl = _make_wl(tomo_size=32, pshift=None)
        sw = np.ones((8, 8, 8))
        idx = np.nonzero(sw)
        result = generate_ctf_slice(wl, [idx], sw, binning=1)
        assert result.shape == sw.shape

    def test_with_pshift_no_error(self):
        wl = _make_wl(tomo_size=32, pshift=30.0)
        sw = np.ones((8, 8, 8))
        idx = np.nonzero(sw)
        result = generate_ctf_slice(wl, [idx], sw, binning=1)
        assert result.shape == sw.shape

    def test_ctf_modifies_weights(self):
        wl = _make_wl(tomo_size=64)
        sw = np.ones((self._SZ, self._SZ, self._SZ))
        idx = np.nonzero(sw)
        result = generate_ctf_slice(wl, [idx], sw, binning=1)
        assert not np.allclose(result, sw)


def test_gaussian_smooth_reduces_variance():
    np.random.seed(20)
    arr = np.random.rand(16, 16, 16)
    smoothed = gaussian_smooth(arr, sigma=2)
    assert smoothed.var() < arr.var()


def test_gaussian_smooth_constant_input_unchanged():
    arr = np.full((8, 8, 8), 3.0)
    result = gaussian_smooth(arr, sigma=1)
    np.testing.assert_allclose(result, arr, atol=1e-10)


# ---------------------------------------------------------------------------
# compute_wiener_1d / apply_wiener_radial
# ---------------------------------------------------------------------------

def test_compute_wiener_1d_shape():
    w = compute_wiener_1d(2048, 3.42, 2.5, 1.2, 1.0, 0.02)
    assert w.shape == (2048,)


def test_compute_wiener_1d_finite():
    w = compute_wiener_1d(2048, 3.42, 2.5, 1.2, 1.0, 0.02)
    assert np.all(np.isfinite(w))


def test_compute_wiener_1d_phase_flipped_differs():
    w_normal = compute_wiener_1d(512, 3.42, 2.5, 1.2, 1.0, 0.02, phase_flipped=False)
    w_flipped = compute_wiener_1d(512, 3.42, 2.5, 1.2, 1.0, 0.02, phase_flipped=True)
    assert not np.allclose(w_normal, w_flipped)


def test_compute_wiener_1d_zero_at_dc():
    """Wiener filter must be 0 at DC because the highpass ramp zeroes it."""
    w = compute_wiener_1d(2048, 3.42, 2.5, 1.2, 1.0, 0.02)
    assert w[0] == pytest.approx(0.0)


def test_apply_wiener_radial_shape_3d():
    np.random.seed(30)
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    w = compute_wiener_1d(2048, 3.42, 2.5, 1.2, 1.0, 0.02)
    result = apply_wiener_radial(vol, w, 2048)
    assert result.shape == vol.shape
    assert np.isrealobj(result)


def test_apply_wiener_radial_shape_2d():
    np.random.seed(31)
    img = np.random.rand(32, 32).astype(np.float32)
    w = compute_wiener_1d(2048, 3.42, 2.5, 1.2, 1.0, 0.02)
    result = apply_wiener_radial(img, w, 2048)
    assert result.shape == img.shape
    assert np.isrealobj(result)


def test_apply_wiener_radial_changes_volume():
    np.random.seed(32)
    vol = np.random.rand(16, 16, 16).astype(np.float64)
    w = compute_wiener_1d(2048, 3.42, 2.5, 1.2, 1.0, 0.02)
    result = apply_wiener_radial(vol, w, 2048)
    assert not np.allclose(result, vol)


# ---------------------------------------------------------------------------
# triangle_threshold
# ---------------------------------------------------------------------------

def test_triangle_threshold_scalar():
    arr = np.concatenate([np.zeros(90), np.ones(10)])
    assert np.ndim(triangle_threshold(arr)) == 0


def test_triangle_threshold_within_range():
    arr = np.concatenate([np.full(90, 0.1), np.full(10, 1.0)])
    result = triangle_threshold(arr)
    assert arr[arr > 0].min() <= result <= arr.max()


def test_triangle_threshold_finite_random():
    np.random.seed(99)
    arr = np.random.uniform(0.0, 1.0, 500)
    assert np.isfinite(triangle_threshold(arr))


def test_triangle_threshold_3d():
    arr = np.zeros((10, 10, 10))
    arr[7:, :, :] = 1.0
    assert np.isfinite(triangle_threshold(arr))


# ---------------------------------------------------------------------------
# find_peak_3d
# ---------------------------------------------------------------------------

def test_find_peak_3d_known_peak():
    vol = np.zeros((20, 20, 20))
    vol[10, 10, 10] = 5.0
    assert find_peak_3d(vol, search_radius=6) == (10, 10, 10)


def test_find_peak_3d_returns_tuple():
    vol = np.random.rand(16, 16, 16)
    center = find_peak_3d(vol)
    assert len(center) == 3


# ---------------------------------------------------------------------------
# extract_orthogonal_lines_1d
# ---------------------------------------------------------------------------

def test_extract_orthogonal_lines_1d_shape():
    vol = np.zeros((20, 20, 20))
    profiles = extract_orthogonal_lines_1d(vol, (10, 10, 10))
    assert profiles.shape == (20, 3)


def test_extract_orthogonal_lines_1d_contains_peak():
    vol = np.zeros((20, 20, 20))
    vol[10, 10, 10] = 7.0
    profiles = extract_orthogonal_lines_1d(vol, (10, 10, 10))
    assert np.any(np.isclose(profiles, 7.0))


# ---------------------------------------------------------------------------
# extract_orthogonal_slices_2d
# ---------------------------------------------------------------------------

def test_extract_orthogonal_slices_2d_shape():
    vol = np.zeros((20, 20, 20))
    slices = extract_orthogonal_slices_2d(vol, (10, 10, 10))
    assert slices.shape == (20, 20, 3)


def test_extract_orthogonal_slices_2d_contains_peak():
    vol = np.zeros((20, 20, 20))
    vol[10, 10, 10] = 7.0
    slices = extract_orthogonal_slices_2d(vol, (10, 10, 10))
    assert np.any(np.isclose(slices, 7.0))


# ---------------------------------------------------------------------------
# gaussian_threshold
# ---------------------------------------------------------------------------

def test_gaussian_threshold_returns_finite_float():
    vol = np.zeros((30, 30, 30))
    vol[13:17, 13:17, 13:17] = 1.0
    result = gaussian_threshold(vol)
    assert isinstance(result, float)
    assert np.isfinite(result)


def test_gaussian_threshold_positive():
    vol = np.zeros((30, 30, 30))
    vol[13:17, 13:17, 13:17] = 1.0
    assert gaussian_threshold(vol) > 0


# ---------------------------------------------------------------------------
# label_connected_components
# ---------------------------------------------------------------------------

def test_label_connected_components_two_blobs():
    vol = np.zeros((10, 10, 10))
    vol[1:3, 1:3, 1:3] = 1.0
    vol[7:9, 7:9, 7:9] = 1.0
    labeled = label_connected_components(vol)
    assert len(np.unique(labeled)) == 3  # background + 2 labels


def test_label_connected_components_background_zero():
    vol = np.zeros((8, 8, 8))
    vol[3:5, 3:5, 3:5] = 1.0
    labeled = label_connected_components(vol)
    assert labeled[0, 0, 0] == 0


# ---------------------------------------------------------------------------
# label_at_point
# ---------------------------------------------------------------------------

def test_label_at_point_correct():
    vol = np.zeros((10, 10, 10))
    vol[1:4, 1:4, 1:4] = 1.0
    vol[6:9, 6:9, 6:9] = 1.0
    labeled = label_connected_components(vol)
    l1 = label_at_point(labeled, (2, 2, 2))
    l2 = label_at_point(labeled, (7, 7, 7))
    assert l1 != 0
    assert l2 != 0
    assert l1 != l2


# ---------------------------------------------------------------------------
# get_label_bounding_box
# ---------------------------------------------------------------------------

def test_get_label_bounding_box_returns_dataframe():
    import pandas as pd
    vol = np.zeros((10, 10, 10))
    vol[2:5, 2:5, 2:5] = 1.0
    labeled = label_connected_components(vol)
    df = get_label_bounding_box(labeled)
    assert isinstance(df, pd.DataFrame)
    assert "label" in df.columns
    assert "bbox-0" in df.columns


# ---------------------------------------------------------------------------
# morphology_open_close
# ---------------------------------------------------------------------------

def test_morphology_open_close_removes_isolated():
    vol = np.zeros((10, 10, 10), dtype=bool)
    vol[5, 5, 5] = True  # isolated voxel
    result = morphology_open_close(vol, operation="open")
    assert not result[5, 5, 5]


def test_morphology_open_close_invalid_raises():
    with pytest.raises(ValueError):
        morphology_open_close(np.zeros((4, 4, 4)), operation="erode")


# ---------------------------------------------------------------------------
# normalize_array
# ---------------------------------------------------------------------------

def test_normalize_array_zero_mean_unit_std():
    rng = np.random.default_rng(0)
    vol = rng.normal(5.0, 3.0, (16, 16, 16)).astype(np.float32)
    out = normalize_array(vol)
    assert np.isclose(np.mean(out), 0.0, atol=1e-5)
    assert np.isclose(np.std(out), 1.0, atol=1e-5)


def test_normalize_array_shape_preserved():
    vol = np.random.rand(8, 10, 12).astype(np.float32)
    assert normalize_array(vol).shape == vol.shape


def test_normalize_array_no_finite_warns():
    vol = np.full((4, 4, 4), np.nan)
    with pytest.warns(UserWarning, match="No finite"):
        out = normalize_array(vol)
    assert out.shape == vol.shape


def test_normalize_array_zero_std_warns():
    vol = np.ones((4, 4, 4), dtype=float)
    with pytest.warns(UserWarning, match="Standard deviation"):
        out = normalize_array(vol)
    assert out.shape == vol.shape


# ---------------------------------------------------------------------------
# normalize_under_mask
# ---------------------------------------------------------------------------

def test_normalize_under_mask_shape():
    vol = np.random.rand(8, 8, 8).astype(np.float32)
    mask = np.zeros((8, 8, 8))
    mask[2:6, 2:6, 2:6] = 1.0
    out = normalize_under_mask(vol, mask)
    assert out.shape == vol.shape


def test_normalize_under_mask_mean_under_mask():
    rng = np.random.default_rng(1)
    vol = rng.normal(10.0, 2.0, (16, 16, 16)).astype(np.float32)
    mask = np.zeros_like(vol)
    mask[4:12, 4:12, 4:12] = 1.0
    out = normalize_under_mask(vol, mask)
    idx = mask > 0
    assert np.isclose(np.mean(out[idx]), 0.0, atol=1e-4)


# ---------------------------------------------------------------------------
# binarize
# ---------------------------------------------------------------------------

def test_binarize_default_threshold():
    vol = np.array([0.0, 0.4, 0.5, 0.6, 1.0])
    out = binarize(vol)
    np.testing.assert_array_equal(out, [0, 0, 0, 1, 1])


def test_binarize_custom_threshold():
    vol = np.array([0.1, 0.5, 0.9])
    out = binarize(vol, threshold=0.4)
    np.testing.assert_array_equal(out, [0, 1, 1])


def test_binarize_dtype_is_int():
    out = binarize(np.random.rand(4, 4, 4))
    assert np.issubdtype(out.dtype, np.integer)


def test_binarize_shape_preserved():
    vol = np.random.rand(5, 6, 7)
    assert binarize(vol).shape == vol.shape


# ---------------------------------------------------------------------------
# rotate_volume
# ---------------------------------------------------------------------------

def test_rotate_volume_shape():
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    out = rotate_volume(vol, rotation_angles=[0, 0, 45])
    assert out.shape == vol.shape


def test_rotate_volume_identity_angles():
    vol = np.random.rand(12, 12, 12).astype(np.float32)
    out = rotate_volume(vol, rotation_angles=[0, 0, 0])
    np.testing.assert_allclose(out, vol, atol=1e-5)


def test_rotate_volume_nonzero_changes_volume():
    rng = np.random.default_rng(42)
    vol = rng.random((12, 12, 12)).astype(np.float32)
    out = rotate_volume(vol, rotation_angles=[0, 0, 45])
    assert not np.allclose(out, vol, atol=1e-3)


def test_rotate_volume_no_rotation_raises():
    with pytest.raises(ValueError):
        rotate_volume(np.zeros((8, 8, 8)))


def test_rotate_volume_with_rotation_object():
    from scipy.spatial.transform import Rotation as srot
    vol = np.random.rand(12, 12, 12).astype(np.float32)
    r = srot.from_euler("zxz", [0, 0, 0], degrees=True)
    out = rotate_volume(vol, rotation=r)
    np.testing.assert_allclose(out, vol, atol=1e-5)


# ---------------------------------------------------------------------------
# shift_array
# ---------------------------------------------------------------------------

def test_shift_array_shape():
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    out = shift_array(vol, np.array([2, 0, 0]))
    assert out.shape == vol.shape


def test_shift_array_zero_is_identity():
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    out = shift_array(vol, np.array([0, 0, 0]))
    np.testing.assert_allclose(out, vol, atol=1e-5)


def test_shift_array_moves_content():
    vol = np.zeros((16, 16, 16), dtype=np.float32)
    vol[8, 8, 8] = 1.0
    out = shift_array(vol, np.array([2, 0, 0]))
    # With T[:3,-1] = -delta, output[i] = input[i - delta],
    # so the peak at input[8] appears at output[10].
    assert out[8, 8, 8] < 0.5
    assert out[10, 8, 8] > 0.5


# ---------------------------------------------------------------------------
# flip_array
# ---------------------------------------------------------------------------

def test_flip_array_shape():
    vol = np.random.rand(8, 10, 12)
    assert flip_array(vol, "z").shape == vol.shape


def test_flip_array_double_flip_identity():
    vol = np.random.rand(8, 8, 8)
    assert np.array_equal(flip_array(flip_array(vol, "z"), "z"), vol)


def test_flip_array_all_axes():
    vol = np.arange(24).reshape(2, 3, 4).astype(float)
    out = flip_array(vol, "xyz")
    expected = np.flip(np.flip(np.flip(vol, 0), 1), 2)
    np.testing.assert_array_equal(out, expected)


def test_flip_array_empty_axis_unchanged():
    vol = np.random.rand(8, 8, 8)
    out = flip_array(vol, "")
    np.testing.assert_array_equal(out, vol)


# ---------------------------------------------------------------------------
# recenter_volume
# ---------------------------------------------------------------------------

def test_recenter_volume_shape():
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    out = recenter_volume(vol, np.array([8, 8, 8]))
    assert out.shape == vol.shape


def test_recenter_volume_box_center_is_identity():
    vol = np.random.rand(16, 16, 16).astype(np.float32)
    center = np.array(vol.shape) // 2
    out = recenter_volume(vol, center)
    np.testing.assert_allclose(out, vol, atol=1e-5)


# ---------------------------------------------------------------------------
# pad_volume
# ---------------------------------------------------------------------------

def test_pad_volume_output_shape():
    vol = np.ones((8, 8, 8), dtype=np.float32)
    out = pad_volume(vol, np.array([16, 16, 16]))
    assert out.shape == (16, 16, 16)


def test_pad_volume_content_preserved():
    vol = np.ones((4, 4, 4), dtype=np.float32) * 5.0
    out = pad_volume(vol, np.array([8, 8, 8]), fill_value=0.0)
    # centre region should be 5.0
    assert out[2:6, 2:6, 2:6].mean() == pytest.approx(5.0)


def test_pad_volume_fill_value():
    vol = np.zeros((4, 4, 4), dtype=np.float32)
    out = pad_volume(vol, np.array([8, 8, 8]), fill_value=99.0)
    assert out[0, 0, 0] == pytest.approx(99.0)


def test_pad_volume_default_fill_is_mean():
    vol = np.full((4, 4, 4), 3.0, dtype=np.float32)
    out = pad_volume(vol, np.array([8, 8, 8]))
    assert out[0, 0, 0] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# symmetrize_volume
# ---------------------------------------------------------------------------

def test_symmetrize_volume_shape():
    vol = np.random.rand(12, 12, 12).astype(np.float32)
    out = symmetrize_volume(vol, "C2")
    assert out.shape == vol.shape


def test_symmetrize_volume_c1_close_to_original():
    vol = np.random.rand(12, 12, 12).astype(np.float32)
    out = symmetrize_volume(vol, 1)
    # C1 = single 360° rotation → same as original
    np.testing.assert_allclose(out, vol, atol=1e-4)


def test_symmetrize_volume_invalid_raises():
    with pytest.raises((ValueError, TypeError)):
        symmetrize_volume(np.zeros((8, 8, 8)), [1, 2])


# ---------------------------------------------------------------------------
# equalize_histogram_2d
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method", ["contrast_stretching", "equalization", "adaptive_eq"])
def test_equalize_histogram_2d_shape(method):
    img = np.random.rand(32, 32).astype(np.float32)
    out = equalize_histogram_2d(img, method=method)
    assert out.shape == img.shape


def test_equalize_histogram_2d_contrast_stretching_range():
    rng = np.random.default_rng(0)
    img = rng.normal(0, 1, (64, 64)).astype(np.float32)
    out = equalize_histogram_2d(img, "contrast_stretching")
    # skimage rescale_intensity maps float images to the dtype range [-1, 1]
    assert out.min() >= -1.0 - 1e-6
    assert out.max() <= 1.0 + 1e-6


def test_equalize_histogram_2d_adaptive_range():
    img = np.random.rand(32, 32).astype(np.float32)
    out = equalize_histogram_2d(img, "adaptive_eq")
    assert np.all(out >= 0) and np.all(out <= 1 + 1e-6)


def test_equalize_histogram_2d_invalid_raises():
    with pytest.raises(ValueError):
        equalize_histogram_2d(np.random.rand(8, 8), method="unknown")


# ---------------------------------------------------------------------------
# apply_fft_filter
# ---------------------------------------------------------------------------

def test_apply_fft_filter_shape():
    img = np.random.rand(16, 16).astype(np.float32)
    filt = np.ones((16, 16))
    out = apply_fft_filter(img, filt)
    assert out.shape == img.shape


def test_apply_fft_filter_ones_roundtrip():
    img = np.random.rand(16, 16).astype(np.float32)
    filt = np.ones((16, 16))
    out = apply_fft_filter(img, filt)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_apply_fft_filter_zero_filter_zeros_output():
    img = np.random.rand(16, 16).astype(np.float32)
    filt = np.zeros((16, 16))
    out = apply_fft_filter(img, filt)
    np.testing.assert_allclose(out, np.zeros((16, 16)), atol=1e-10)


def test_apply_fft_filter_attenuates():
    img = np.random.rand(32, 32).astype(np.float32)
    filt = np.full((32, 32), 0.5)
    out = apply_fft_filter(img, filt)
    assert np.std(out) < np.std(img)


# ---------------------------------------------------------------------------
# mask_overlap
# ---------------------------------------------------------------------------

def test_mask_overlap_returns_int():
    a = np.ones((4, 4, 4))
    b = np.ones((4, 4, 4))
    assert isinstance(mask_overlap(a, b), int)


def test_mask_overlap_non_overlapping():
    a = np.zeros((8, 8, 8))
    b = np.zeros((8, 8, 8))
    a[:4, :, :] = 1.0
    b[4:, :, :] = 1.0
    assert mask_overlap(a, b) == 0


def test_mask_overlap_identical_masks():
    a = np.ones((4, 4, 4))
    result = mask_overlap(a, a)
    assert result == 4 * 4 * 4


def test_mask_overlap_threshold():
    a = np.ones((4, 4, 4))
    b = np.ones((4, 4, 4)) * 0.5
    # sum = 1.5, default threshold 1.9 → no overlap
    assert mask_overlap(a, b, threshold=1.9) == 0
    # with lower threshold → all voxels overlap
    assert mask_overlap(a, b, threshold=1.0) == 4 * 4 * 4
