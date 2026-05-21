from __future__ import annotations

import numpy as np
import pandas as pd
from lmfit import models as _lmfit_models
from scipy.interpolate import RegularGridInterpolator
from skimage import measure as _skimage_measure
from skimage import morphology as _skimage_morphology
from skimage.filters import gaussian as _skimage_gaussian
from skimage.transform import rotate as skimage_rotate


# ---------------------------------------------------------------------------
# §3.1  Frequency / phase
# ---------------------------------------------------------------------------

def compute_frequency_array(shape: tuple[int, ...], pixel_size: float) -> np.ndarray:
    """Compute the frequency array for a given shape and pixel size.

    Parameters
    ----------
    shape : tuple of int
        The shape of the array for which the frequency array is computed.
    pixel_size : float
        The size of each pixel in the spatial domain.

    Returns
    -------
    numpy.ndarray
        An array representing the frequency magnitudes corresponding to the
        input shape.

    Notes
    -----
    The function uses the Fast Fourier Transform (FFT) to compute the
    frequency bins and then calculates the magnitude of the frequency vector
    for each point in the frequency domain.
    """

    freqs = np.array(
        np.meshgrid(
            *[np.fft.fftshift(np.fft.fftfreq(n, pixel_size)) for n in shape],
            indexing="ij",
        )
    )
    return np.sqrt(np.sum(freqs**2, axis=0))


def randomize_phases(volume: np.ndarray, fourier_cutoff: int) -> np.ndarray:
    """Return a real-space map with phases randomized beyond fourier_cutoff Fourier pixels.

    Phases of the input volume's Fourier transform are replaced by a random
    permutation of the original phases at all shells with radial pixel distance
    >= *fourier_cutoff*.  Amplitudes are preserved exactly.

    Parameters
    ----------
    volume : ndarray
        3-D real-space volume.
    fourier_cutoff : int
        Radial threshold in Fourier pixels.  Phases at shells with distance
        >= this value are randomly permuted.

    Returns
    -------
    ndarray
        Real-space volume (float64) with randomized high-frequency phases.
    """
    ft = np.fft.fftshift(np.fft.fftn(volume))
    amp = np.abs(ft)
    phase = np.angle(ft)
    dist = compute_frequency_array(volume.shape, 1) * volume.shape[0]
    pr_phase = phase.copy()
    idx = np.where(dist >= fourier_cutoff)
    pr_phase[idx] = np.random.permutation(phase[idx])
    return np.fft.ifftn(np.fft.ifftshift(amp * np.exp(1j * pr_phase))).real


# ---------------------------------------------------------------------------
# §3.2  Thresholding
# ---------------------------------------------------------------------------

def otsu_threshold(input_values: np.ndarray) -> float:
    """Calculate the Otsu threshold for binarization based on the histogram of input values.

    Parameters
    ----------
    input_values : ndarray
        An array of input values for which the histogram and threshold need to
        be computed.

    Returns
    -------
    float
        The computed threshold value according to Otsu's method.

    Notes
    -----
    Otsu's method is used to automatically perform histogram shape-based image
    thresholding.  The algorithm assumes that the data contains two classes of
    values following a bimodal histogram, it then calculates the optimum
    threshold separating the two classes so that their combined spread
    (intra-class variance) is minimal.

    References
    ----------
    Taken from: https://www.kdnuggets.com/2018/10/basic-image-analysis-python-p4.html
    """

    stats_bins = np.histogram(input_values, bins=input_values.shape[0])
    bin_counts = stats_bins[0]
    s_max = (0, 0)

    for threshold in range(len(bin_counts)):
        w_0 = sum(bin_counts[:threshold])
        w_1 = sum(bin_counts[threshold:])

        mu_0 = sum([i * bin_counts[i] for i in range(0, threshold)]) / w_0 if w_0 > 0 else 0
        mu_1 = sum([i * bin_counts[i] for i in range(threshold, len(bin_counts))]) / w_1 if w_1 > 0 else 0

        s = w_0 * w_1 * (mu_0 - mu_1) ** 2

        if s > s_max[1]:
            s_max = (threshold, s)

    return stats_bins[1][s_max[0]]


# ---------------------------------------------------------------------------
# §3.3  Filter primitives
# ---------------------------------------------------------------------------

def get_filter_radius(
    edge_size: int,
    fourier_pixels: int | None,
    target_resolution: float | None,
    pixel_size: float | None,
) -> int:
    """Calculate the filter radius based on either direct Fourier pixel/voxel specification or target resolution.

    Parameters
    ----------
    edge_size : int
        Size of the edge of the image/map in pixels/voxels.
    fourier_pixels : int, optional
        Number of pixels in the Fourier space. Default is None.
    target_resolution : float, optional
        Desired resolution to achieve. Default is None.
    pixel_size : float, optional
        Size of a pixel/voxel in the image/map. Default is None.

    Returns
    -------
    int
        Calculated radius in pixels/voxels for the filter.

    Raises
    ------
    ValueError
        If neither `fourier_pixels` nor both `target_resolution` and `pixel_size` are specified.
    """

    if fourier_pixels is not None:
        radius = fourier_pixels
    elif target_resolution is not None and pixel_size is not None:
        radius = round(edge_size * pixel_size / target_resolution)
    else:
        raise ValueError(
            "Either target_voxels or target_resolution in combination with pixel_size have to be specified!"
        )

    return radius


# ---------------------------------------------------------------------------
# §3.4  Frequency-space filters
# ---------------------------------------------------------------------------

def gaussian_smooth(volume: np.ndarray, sigma: float) -> np.ndarray:
    """Apply Gaussian smoothing to an array.

    Parameters
    ----------
    volume : ndarray
        Input array (any shape and number of dimensions).
    sigma : float
        Standard deviation of the Gaussian kernel in voxels.
        Pass ``0`` to return *volume* unchanged.

    Returns
    -------
    ndarray
        Smoothed array with the same shape as *volume*.
        Returned unchanged (same object) when *sigma* == 0.
    """
    if sigma == 0:
        return volume
    return _skimage_gaussian(volume, sigma=sigma)


def _spherical_mask_nd(shape, radius, gaussian=0.0):
    """Create a centred spherical (or hyperspherical) binary mask with optional Gaussian softening.

    Implements the Dynamo convention (gaussian_outwards=False): the Gaussian
    blur is centred on the sphere surface, so the radius is NOT expanded.

    Parameters
    ----------
    shape : tuple of int
        Shape of the output array.  Works for 2-D, 3-D, and higher.
    radius : float
        Radius in voxels of the binary sphere before any blurring.
    gaussian : float, default=0.0
        Sigma of the Gaussian softening.  0 means no blur.

    Returns
    -------
    mask : ndarray, float64
        Values in [0, 1].  Shape matches *shape*.
    """
    shape = np.asarray(shape, dtype=int)
    center = shape // 2
    grids = np.mgrid[tuple(slice(0, s) for s in shape)]
    dist = np.sqrt(sum((g - c) ** 2 for g, c in zip(grids, center)))
    mask = (dist <= radius).astype(np.float64)
    return gaussian_smooth(mask, gaussian)


def apply_lowpass(volume: np.ndarray, radius: int, gaussian: float = 3) -> np.ndarray:
    """Apply a spherical lowpass filter to a volume in Fourier space.

    Parameters
    ----------
    volume : ndarray
        Input volume (any shape).
    radius : int
        Radius of the lowpass filter in Fourier pixels.
    gaussian : float, default=3
        Sigma of the Gaussian softening on the filter sphere surface.

    Returns
    -------
    ndarray
        Lowpass-filtered volume (real-valued, same shape as *volume*).
    """
    lp_mask = np.fft.ifftshift(_spherical_mask_nd(volume.shape, radius, gaussian=gaussian))
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * lp_mask))


def apply_highpass(volume: np.ndarray, radius: int, gaussian: float = 2) -> np.ndarray:
    """Apply a spherical highpass filter to a volume in Fourier space.

    Parameters
    ----------
    volume : ndarray
        Input volume (any shape).
    radius : int
        Radius of the highpass filter in Fourier pixels.
    gaussian : float, default=2
        Sigma of the Gaussian softening on the filter sphere surface.

    Returns
    -------
    ndarray
        Highpass-filtered volume (real-valued, same shape as *volume*).
    """
    hp_mask = np.fft.ifftshift(
        np.ones(volume.shape) - _spherical_mask_nd(volume.shape, radius, gaussian=gaussian)
    )
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * hp_mask))


def apply_bandpass(volume: np.ndarray, lp_radius: int, hp_radius: int, lp_gaussian: float = 3, hp_gaussian: float = 2) -> np.ndarray:
    """Apply a bandpass filter to a volume in Fourier space.

    Parameters
    ----------
    volume : ndarray
        Input volume (any shape).
    lp_radius : int
        Radius of the outer (lowpass) filter shell in Fourier pixels.
    hp_radius : int
        Radius of the inner (highpass) filter shell in Fourier pixels.
    lp_gaussian : float, default=3
        Sigma of the Gaussian softening on the outer shell.
    hp_gaussian : float, default=2
        Sigma of the Gaussian softening on the inner shell.

    Returns
    -------
    ndarray
        Bandpass-filtered volume (real-valued, same shape as *volume*).
    """
    outer = _spherical_mask_nd(volume.shape, lp_radius, gaussian=lp_gaussian)
    inner = _spherical_mask_nd(volume.shape, hp_radius, gaussian=hp_gaussian)
    band_mask = np.fft.ifftshift(outer - inner)
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * band_mask))


# ---------------------------------------------------------------------------
# §3.5  2-D image operations
# ---------------------------------------------------------------------------

def rotate_2d(image: np.ndarray, angle: float, fill_mode: str = "constant", fill_value: float = 0.0) -> np.ndarray:
    """Rotate an ndarray image by a specified angle.

    Uses 'skimage.transform.rotate' to rotate the input image without resizing
    the output. Pixels outside the boundaries of the input are filled
    according to the specified mode and fill value.

    Parameters
    ----------
    image : ndarray
        nD NumPy array representing the image to rotate.
    angle : float
        Angle of rotation in degrees. Positive values rotate counterclockwise.
    fill_mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, default='constant'
        Points outside the boundaries of the input are filled according to the
        given mode.
    fill_value : float, default=0.0
        Value used to fill points outside the boundaries when `fill_mode`
        is 'constant'.

    Returns
    -------
    ndarray
        Rotated image as a NumPy array with the same shape as the input.
    """

    return skimage_rotate(image, angle, resize=False, mode=fill_mode, cval=fill_value)


# ---------------------------------------------------------------------------
# §3.6  CTF
# ---------------------------------------------------------------------------

def compute_ctf_1d(
    length: int,
    pixel_size: float,
    voltage: float,
    cs: float,
    defocus: float,
    amplitude: float,
    phaseshift: float,
    bfactor: float,
) -> np.ndarray:
    """Compute the 1D Contrast Transfer Function (CTF) for a given set of parameters.

    Parameters
    ----------
    length : int
        The length of the 1D array for which the CTF is computed.
    pixel_size : float
        The size of the pixel in the image.
    voltage : float
        The voltage used in the microscope.
    cs : float
        The spherical aberration coefficient.
    defocus : float
        The defocus value.
    amplitude : float
        The amplitude contrast.
    phaseshift : float
        The phase shift value.
    bfactor : float
        The B-factor for the envelope function.

    Returns
    -------
    ctf : ndarray
        The computed 1D Contrast Transfer Function.
    """

    ny = 1 / pixel_size
    lambda_factor = 12.2643247 / np.sqrt(voltage * (1.0 + voltage * 0.978466e-6)) * 1e-10
    lambda2 = lambda_factor * 2

    points = np.arange(length)
    points = points / (2 * length) * ny
    k2 = points**2
    term1 = lambda_factor**3 * cs * k2**2

    w = np.pi / 2 * (term1 + lambda2 * defocus * k2) - phaseshift

    acurve = np.cos(w) * amplitude
    pcurve = -np.sqrt(1 - amplitude**2) * np.sin(w)
    bfactor = np.exp(-bfactor * k2 * 0.25)
    ctf = (pcurve + acurve) * bfactor

    return ctf


def compute_ctf_2d(defocus: np.ndarray, pshift: np.ndarray, famp: float, cs: float, evk: float, f: np.ndarray) -> np.ndarray:
    """Compute Contrast Transfer Function (CTF) for an acquisition scheme.

    This function evaluates the 1D CTF over a 1D spatial frequency array
    using image acquisition parameters such as defocus, spherical aberration,
    and accelerating voltage.

    Parameters
    ----------
    defocus : array_like, shape (N, 1)
        Defocus values (in μm) for each of the N tilts.
    pshift : array_like, shape (N, 1)
        Phase shift values (in degrees) for each tilt. Pass zeros if no
        phase plate is used.
    famp : float
        Amplitude contrast (typically between 0.0 and 0.2).
    cs : float
        Spherical aberration (in mm).
    evk : float
        Accelerating voltage (in kV).
    f : ndarray
        1D spatial frequency magnitude array (in Å⁻¹) at which to evaluate
        the CTF.  Must be 1D so it broadcasts correctly against the ``(N, 1)``
        defocus/pshift arrays.

    Returns
    -------
    ctf : ndarray, shape (N, len(f))
        CTF evaluated at each tilt (row) and each frequency (column).

    Notes
    -----
    The output combines both sine and cosine components weighted by phase and
    amplitude contrast.  The output has radial symmetry.
    """

    defocus = defocus * 1.0e4  # convert defocus distance from μm to Å
    cs = cs * 1.0e7  # spherical aberration from mm to Å
    pshift = pshift * np.pi / 180  # phase shift degree to radian

    h = 6.62606957e-34  # Planck's constant (m^2*kg/s)
    c = 299792458  # speed of light (m/s)
    erest = 511000  # electron rest energy (eV)
    v = evk * 1000  # accelerating voltage (V)
    e = 1.602e-19  # unit of e- charge (C)

    # de Broglie e- wavelength
    lam = (c * h) / np.sqrt(((2 * erest * v) + (v**2)) * (e**2)) * (10**10)

    # phase contrast weighting factor
    w = (1 - (famp**2)) ** 0.5

    # compute the ctf phase shift term / wave aberration equation
    chi = (np.pi * lam * (f**2)) * (defocus - 0.5 * (lam**2) * (f**2) * cs)
    chi += pshift

    # CTF function
    ctf = (w * np.sin(chi)) + (famp * np.cos(chi))

    return ctf


def generate_ctf_slice(wl: pd.DataFrame, slice_idx: list, slice_weight: np.ndarray, binning: int | float) -> np.ndarray:
    """Generate a CTF filter for a weighted volume that is subset of a full volume.

    This function computes defocus and phase shift-specific CTF filters for a
    template by:

    1. computing the frequency array of a tomogram;
    2. interpolating the CTF based on the full sized frequency array and
       defocus / pshift values;
    3. fourier cropping the full size CTF values to fit the size of a template,
       to keep only the lower frequencies;
    4. applying the CTF values to the given `slice_weight` array.

    Parameters
    ----------
    wl : pandas.DataFrame
        Wedge list dataframe for one tomogram.  Required columns:
        ``"tomo_x"``, ``"tomo_y"``, ``"tomo_z"`` (tomogram dimensions in
        voxels), ``"pixelsize"`` (unbinned pixel size in Å),
        ``"defocus"`` (in μm), ``"amp_contrast"``, ``"cs"`` (spherical
        aberration in mm), ``"voltage"`` (in kV).  An optional ``"pshift"``
        column (phase shift in degrees) defaults to zeros when absent.
    slice_idx : list of tuple of ndarray
        Per-tilt active voxel indices in Fourier space, as returned by
        ``generate_wedgemask_slices_template``.  Each element is a tuple of
        three 1-D index arrays ``(z, y, x)`` from ``np.nonzero``.
    slice_weight : ndarray
        3D array of normalised frequency-domain weights (missing wedge and
        bandpass combined), shape ``(depth, height, width)`` / ``(z, y, x)``.
        Applied as a final multiplicative factor after the per-tilt CTF
        accumulation.
    binning : int or float
        The binning factor of the tomogram. Used to scale the pixel size.

    Returns
    -------
    ctf_filt : ndarray
        The computed CTF filter applied on top of the weighted filter input
        (i.e. bandpass and missing wedge), in fourier space.
        Same shape as `slice_weight`.

    Notes
    -----
    If `pshift` is not provided in `wl`, it defaults to zeros.
    """

    # determine template and full tomogram sizes
    tmpl_size = slice_weight.shape[0]
    full_size = int(max(wl["tomo_x"].values[0], wl["tomo_y"].values[0], wl["tomo_z"].values[0]))
    pixelsize = wl["pixelsize"].values[0] * binning

    # calculate frequency arrays of tomogram and template sizes
    freqs_full = compute_frequency_array((full_size,), pixelsize)
    freqs_crop = compute_frequency_array((tmpl_size,), pixelsize)
    freqs_crop = freqs_crop[tmpl_size // 2:]  # only need half of the freq array

    # selects the central part (len = tmpl length) of the frequency array from
    # the full frequency array and wraps it to match FFT layout
    f_idx = np.zeros(full_size, dtype="bool")
    f_idx[:tmpl_size] = 1
    f_idx = np.roll(f_idx, -tmpl_size // 2)
    f_idx = np.nonzero(f_idx)[0]  # get the indices of selected part

    # acquisition parameters
    defocus = np.array(wl["defocus"])
    pshift = np.array(wl["pshift"]) if "pshift" in wl.columns else np.zeros_like(defocus)
    famp = wl["amp_contrast"].values[0]
    cs = wl["cs"].values[0]
    evk = wl["voltage"].values[0]

    # compute the 1D CTF as a function of full tomo freq magnitude array
    full_ctf = np.abs(compute_ctf_2d(defocus[:, None], pshift[:, None], famp, cs, evk, freqs_full))

    # cropping the ctf that only belongs to template size from the full size ctf
    # cropping is done in fourier space to remove high freq ctf content
    ft_ctf = np.fft.fft(full_ctf, axis=1)
    ft_ctf = ft_ctf[:, f_idx] * tmpl_size / full_size
    crop_ctf = np.real(np.fft.ifft(ft_ctf, axis=1))
    crop_ctf = crop_ctf[:, tmpl_size // 2:]

    # init an empty vol to save ctf values
    ctf_filt = np.zeros_like(slice_weight)

    # map weighted spatial components to their freq magnitudes
    x = np.fft.ifftshift(compute_frequency_array(slice_weight.shape, pixelsize))

    # for each tilt, interpolate the CTF value for each weighted voxel
    for ictf, sidx in zip(crop_ctf, slice_idx):
        ip = RegularGridInterpolator(
            (freqs_crop,),
            ictf,
            method="linear",
            bounds_error=False,
            fill_value=0,
        )
        ctf_filt[sidx] += ip(x[sidx])

    # apply the CTF filter to input weight filter
    ctf_filt *= slice_weight

    return ctf_filt


# ---------------------------------------------------------------------------
# §3.7  Wiener deconvolution
# ---------------------------------------------------------------------------

def compute_wiener_1d(
    interp_dim: int,
    pixel_size_a: float,
    defocus: float,
    snr_falloff: float,
    deconv_strength: float,
    highpass_nyquist: float,
    phase_flipped: bool = False,
    phaseshift: float = 0.0,
) -> np.ndarray:
    """Compute a 1-D Wiener deconvolution filter for cryo-EM/ET data.

    Implements the algorithm from tom_deconv_tomo by D. Tegunov
    (https://github.com/dtegunov/tom_deconv).

    Parameters
    ----------
    interp_dim : int
        Number of points in the filter (typically ``max(2048, volume_edge)``).
    pixel_size_a : float
        Pixel/voxel size in Angstroms.
    defocus : float
        Defocus in micrometers, positive = underfocus.
    snr_falloff : float
        How fast SNR falls off; higher values downweight high frequencies.
    deconv_strength : float
        Global scale for SNR; exponential scale (1.0 → SNR = 1000 at DC).
    highpass_nyquist : float
        Fraction of Nyquist frequency cut off at the low end.
    phase_flipped : bool, default=False
        Whether the data are already phase-flipped.
    phaseshift : float, default=0.0
        CTF phase shift in degrees.

    Returns
    -------
    wiener : ndarray, shape (interp_dim,)
        1-D Wiener filter values indexed by normalised frequency [0, 1).
    """
    highpass = np.arange(0, 1, 1 / interp_dim)
    highpass = np.minimum(1, highpass / highpass_nyquist) * np.pi
    highpass = 1 - np.cos(highpass)

    snr = (
        np.exp(np.arange(0, -1, -1 / interp_dim) * snr_falloff * 100 / pixel_size_a)
        * (10 ** (3 * deconv_strength))
        * highpass
    )
    ctf = compute_ctf_1d(
        interp_dim,
        pixel_size_a * 1e-10,
        300e3,
        2.7e-3,
        -defocus * 1e-6,
        0.07,
        phaseshift / 180 * np.pi,
        0,
    )
    if phase_flipped:
        ctf = np.abs(ctf)
    return ctf / (ctf * ctf + 1 / snr)


def apply_wiener_radial(volume: np.ndarray, wiener_1d: np.ndarray, interp_dim: int) -> np.ndarray:
    """Apply a radially-symmetric 1-D Wiener filter to a volume in Fourier space.

    Works for any number of dimensions (2-D tilt images or 3-D tomograms).

    Parameters
    ----------
    volume : ndarray
        Input array (any shape, any number of dimensions).
    wiener_1d : ndarray, shape (interp_dim,)
        1-D Wiener filter values indexed by normalised frequency [0, 1).
    interp_dim : int
        Length of *wiener_1d*.

    Returns
    -------
    ndarray
        Wiener-filtered volume (real-valued, same shape as *volume*).
    """
    s = volume.shape
    grids = np.meshgrid(*[np.arange(-d / 2, d / 2) for d in s], indexing="ij")
    normalized = [g / max(1, abs(d / 2)) for g, d in zip(grids, s)]
    r = np.minimum(1, np.sqrt(sum(g ** 2 for g in normalized)))
    r = np.fft.ifftshift(r)
    freq = np.arange(0, 1, 1 / interp_dim)
    ramp = np.interp(r.flatten(), freq, wiener_1d).reshape(r.shape)
    return np.real(np.fft.ifftn(np.fft.fftn(volume) * ramp))


# ---------------------------------------------------------------------------
# §3.8  Cross-correlation
# ---------------------------------------------------------------------------

def calculate_conjugates(
    volume: np.ndarray,
    filter: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate the complex conjugates of a volume and its square after applying a Fourier transform and an optional filter.

    Parameters
    ----------
    volume : ndarray
        The input volume to be transformed and filtered.
    filter : ndarray, optional
        The filter to be applied to the Fourier transform of the volume.
        If None, no filter is applied. Default is None.

    Returns
    -------
    conj_target : ndarray
        The complex conjugate of the Fourier transform of the input volume
        after applying the filter.
    conj_target_sq : ndarray
        The complex conjugate of the square of the filtered volume after
        applying the Fourier transform.

    Notes
    -----
    The 0-frequency peak of the Fourier transform of the input volume is set
    to zero before calculating the complex conjugates.
    """

    # Fourier transform tile
    vol_fft = np.fft.fftn(volume)

    # Apply filter
    if filter is not None:
        vol_fft = vol_fft * filter

    # Set 0-frequency peak to zero (generalised to work for any ndim input)
    vol_fft.flat[0] = 0

    # Store complex conjugate
    conj_target = np.conj(vol_fft)

    # Filtered volume
    filtered_volume = np.fft.ifftn(vol_fft).real

    # Store complex conjugate of square
    conj_target_sq = np.conj(np.fft.fftn(np.power(filtered_volume, 2)))

    return conj_target, conj_target_sq


def calculate_flcf(
    vol1: np.ndarray,
    mask: np.ndarray,
    vol2: np.ndarray | None = None,
    conj_target: np.ndarray | None = None,
    conj_target_sq: np.ndarray | None = None,
    filter: np.ndarray | None = None,
) -> np.ndarray:
    """Calculate the Fast Local Correlation Coefficient (FLCC) map between two volumes (3D arrays).

    Parameters
    ----------
    vol1 : ndarray
        The first volume for which the FLCC map is to be calculated.
    mask : ndarray
        The mask to be applied on the volumes.
    vol2 : ndarray, optional
        The second volume for which the FLCC map is to be calculated.
        If not provided, `conj_target` and `conj_target_sq` must be provided.
    conj_target : ndarray, optional
        The conjugate of the target volume. Required if `vol2` is not provided.
    conj_target_sq : ndarray, optional
        The square of the conjugate of the target volume. Required if `vol2`
        is not provided.
    filter : ndarray, optional
        The filter to be applied on the volumes.

    Raises
    ------
    ValueError
        If `vol2` is not provided, both `conj_target` and `conj_target_sq`
        must be provided.
        If input volumes or mask contain NaN values.

    Returns
    -------
    cc_map : ndarray
        The calculated FLCC map, clipped between 0.0 and 1.0.
    """

    if np.isnan(vol1).any() or np.isnan(mask).any():
        raise ValueError("Input volumes or mask contain NaN values")
    box_size = np.array(vol1.shape)
    n_pix = mask.sum()

    vol1 = np.fft.fftn(vol1)
    mask = np.fft.fftn(mask)

    if vol2 is not None:
        conj_target, conj_target_sq = calculate_conjugates(vol2, filter)

    elif conj_target is None or conj_target_sq is None:
        raise ValueError(
            "If the second volume is NOT provided, both conj_target and conj_target_sw have to be passed as parameters."
        )

    numerator = np.fft.ifftn(vol1 * conj_target).real

    A = np.fft.ifftn(mask * conj_target_sq)
    B = np.fft.ifftn(mask * conj_target)
    denominator = np.sqrt(n_pix * A - B * B).real

    cc_map = (numerator / denominator).real

    cen = np.floor(box_size / 2).astype(int) + 1
    cc_map = np.flip(cc_map)
    cc_map = np.roll(cc_map, cen, (0, 1, 2))

    return np.clip(cc_map, 0.0, 1.0)


# ---------------------------------------------------------------------------
# §3.9  Image analysis — thresholding, peak finding, profile extraction
# ---------------------------------------------------------------------------

def triangle_threshold(values: np.ndarray) -> float:
    """Compute a threshold from an array using the triangle / chord method.

    The array is flattened and sorted in ascending order.  A straight line is
    drawn from the first positive value to the maximum.  The threshold is the
    sorted value whose perpendicular distance from that line is greatest.

    Parameters
    ----------
    values : array_like
        Input array of any shape.  Flattened and sorted internally.

    Returns
    -------
    float
        Threshold value.
    """
    sp = np.sort(values, axis=None)
    nbins = len(sp)

    arg_peak_height = np.argmax(sp)
    peak_height = sp[arg_peak_height]
    arg_low_level, arg_high_level = np.where(sp > 0)[0][[0, -1]]

    flip = arg_peak_height - arg_low_level < arg_high_level - arg_peak_height
    if flip:
        sp = sp[::-1]
        arg_low_level = nbins - arg_high_level - 1
        arg_peak_height = nbins - arg_peak_height - 1

    del arg_high_level

    width = arg_peak_height - arg_low_level
    x1 = np.arange(width)
    y1 = sp[x1 + arg_low_level]

    norm = np.sqrt(peak_height**2 + width**2)
    peak_height /= norm
    width /= norm

    length = peak_height * x1 - width * y1
    arg_level = np.argmax(length) + arg_low_level

    if flip:
        arg_level = nbins - arg_level - 1

    return sp[arg_level]


def find_peak_3d(volume: np.ndarray, search_radius: int = 20) -> tuple[int, ...]:
    """Find the position of the highest voxel within a central spherical region.

    Parameters
    ----------
    volume : ndarray
        3-D input array.
    search_radius : int, default=20
        Radius in voxels of the spherical search region centred on the volume.

    Returns
    -------
    tuple of int
        Array indices (dim0, dim1, dim2) of the highest-valued voxel inside
        the search sphere.
    """
    mask = _spherical_mask_nd(volume.shape, search_radius)
    masked = volume * mask
    return np.unravel_index(np.argmax(masked), masked.shape)


def extract_orthogonal_lines_1d(volume: np.ndarray, center: tuple[int, ...]) -> np.ndarray:
    """Extract three 1-D intensity profiles through a point in a 3-D volume.

    Parameters
    ----------
    volume : ndarray
        3-D input array.
    center : tuple of int
        Array indices (dim0, dim1, dim2) of the point through which to slice.

    Returns
    -------
    ndarray of shape (N, 3)
        Column 0 varies along dim0 (fixed dim1=center[1], dim2=center[2]),
        column 1 along dim1, column 2 along dim2.  N is the length of the
        corresponding axis (requires a cubic volume for all columns to share N).
    """
    profiles = np.vstack([
        volume[:, center[1], center[2]],
        volume[center[0], :, center[2]],
        volume[center[0], center[1], :],
    ])
    return profiles.T


def extract_orthogonal_slices_2d(volume: np.ndarray, center: tuple[int, ...]) -> np.ndarray:
    """Extract three orthogonal 2-D slices through a point in a 3-D volume.

    Parameters
    ----------
    volume : ndarray
        3-D cubic array (all dimensions equal).
    center : tuple of int
        Array indices (dim0, dim1, dim2) of the point through which to slice.

    Returns
    -------
    ndarray of shape (N, N, 3)
        ``[..., 0]`` = XY plane (fixed dim2=center[2]),
        ``[..., 1]`` = YZ plane (fixed dim0=center[0]),
        ``[..., 2]`` = XZ plane (fixed dim1=center[1]).
    """
    return np.stack([
        volume[:, :, center[2]],
        volume[center[0], :, :],
        volume[:, center[1], :],
    ], axis=2)


def gaussian_threshold(volume: np.ndarray) -> float:
    """Estimate the peak intensity of a 3-D volume by Gaussian fitting of 1-D profiles.

    Three 1-D profiles through the detected peak (one per array axis) are each
    fitted with a Gaussian model.  The mean of the three fitted peak heights
    (``amplitude / (sigma * sqrt(2π))``) is returned.

    Parameters
    ----------
    volume : ndarray
        3-D array representing the map or volume to analyse.

    Returns
    -------
    float
        Mean Gaussian peak height averaged over the three orthogonal 1-D
        profiles through the map's central peak.
    """
    center = find_peak_3d(volume, search_radius=20)
    profiles = extract_orthogonal_lines_1d(volume, center)

    heights = []
    for i in range(3):
        line = profiles[:, i]
        x = np.linspace(0, line.shape[0], line.shape[0])
        mod = _lmfit_models.GaussianModel()
        params = mod.guess(line, x)
        params["amplitude"].min = 0
        params["sigma"].min = 0
        params["center"].min = center[i] - 1
        params["center"].max = center[i] + 1
        out = mod.fit(line, params, x=x)
        heights.append(out.params["height"].value)

    return float(np.mean(heights))


# ---------------------------------------------------------------------------
# §3.10  Labeling and morphology
# ---------------------------------------------------------------------------

def label_connected_components(volume: np.ndarray, connectivity: int = 1) -> np.ndarray:
    """Label connected regions in a binary or integer array.

    Parameters
    ----------
    volume : ndarray
        Input array.  Non-zero voxels are treated as foreground.
    connectivity : int, default=1
        Maximum number of orthogonal steps to consider connected
        (1 = face connectivity, 2 = edge, 3 = vertex for 3-D).

    Returns
    -------
    ndarray of int
        Integer label array.  Background is 0; each connected region gets a
        unique positive integer.
    """
    return _skimage_measure.label(volume, connectivity=connectivity)


def label_at_point(labeled: np.ndarray, point: tuple[int, ...]) -> int:
    """Return the label value at a specific voxel position.

    Parameters
    ----------
    labeled : ndarray of int
        Label array returned by :func:`label_connected_components`.
    point : sequence of int
        Array indices specifying the voxel position.

    Returns
    -------
    int
        Label value at *point*.
    """
    return labeled[tuple(point)]


def get_label_bounding_box(labeled: np.ndarray) -> pd.DataFrame:
    """Return a DataFrame of label ids and their bounding boxes.

    Parameters
    ----------
    labeled : ndarray of int
        Label array returned by :func:`label_connected_components`.

    Returns
    -------
    pandas.DataFrame
        One row per label with columns ``label`` and ``bbox-0`` … ``bbox-2N-1``
        where N is the number of dimensions.  For a 3-D array the bbox columns
        are ``bbox-0`` (x_min), ``bbox-1`` (y_min), ``bbox-2`` (z_min),
        ``bbox-3`` (x_max), ``bbox-4`` (y_max), ``bbox-5`` (z_max).
    """
    return pd.DataFrame(_skimage_measure.regionprops_table(labeled, properties=("label", "bbox")))


def morphology_open_close(volume: np.ndarray, footprint: np.ndarray | None = None, operation: str = "open") -> np.ndarray:
    """Apply binary morphological opening or closing to a volume.

    Parameters
    ----------
    volume : ndarray
        Binary input array.
    footprint : ndarray, optional
        Structuring element.  Defaults to the default footprint used by
        :func:`skimage.morphology.binary_opening` / ``binary_closing``.
    operation : {"open", "close"}, default="open"
        Morphological operation to apply.

    Returns
    -------
    ndarray of bool
        Result of the morphological operation.

    Raises
    ------
    ValueError
        If *operation* is not ``"open"`` or ``"close"``.
    """
    if operation == "open":
        return _skimage_morphology.binary_opening(volume, footprint=footprint)
    elif operation == "close":
        return _skimage_morphology.binary_closing(volume, footprint=footprint)
    else:
        raise ValueError(f"operation must be 'open' or 'close', got {operation!r}")


# ---------------------------------------------------------------------------
# §3.11  Mask statistics
# ---------------------------------------------------------------------------

def mask_voxel_count_and_bbox(mask: np.ndarray, threshold: float | None = None) -> tuple[int, np.ndarray]:
    """Count non-zero voxels and compute the bounding box of a mask.

    Parameters
    ----------
    mask : ndarray
        Input mask array (any number of dimensions).  When `threshold` is
        None the mask is treated as already binary.  When `threshold` is
        given, the mask is binarised with ``mask > threshold`` before
        counting.
    threshold : float, optional
        Binarisation threshold.  Pass ``0.5`` for a soft mask (reproduces
        the behaviour of the former ``get_soft_mask_stats``).  Default is
        None (sharp mask — no binarisation).

    Returns
    -------
    n_voxels : int
        Number of voxels where the (binarised) mask is non-zero.
    bbox : ndarray
        Size of the bounding box in each dimension (``[dim0, dim1, ...]``).
        All zeros if the mask is empty.
    """

    if threshold is not None:
        mask = np.where(mask > threshold, 1.0, 0.0)

    n_voxels = int(np.count_nonzero(mask))

    nonzero = np.asarray(mask > 1e-5).nonzero()
    if nonzero[0].shape[0] == 0:
        bbox = np.zeros(mask.ndim, dtype=int)
    else:
        bbox = np.array([int(nz.max() - nz.min() + 1) for nz in nonzero])

    return n_voxels, bbox


# ---------------------------------------------------------------------------
# §  Array-operation primitives (pure compute, no I/O)
# ---------------------------------------------------------------------------

def normalize_array(volume: np.ndarray) -> np.ndarray:
    """Z-score normalize *volume* using only finite values.

    Returns the original array unchanged (with a warning) when no finite
    values exist or when the standard deviation is zero.
    """
    import warnings

    values = volume[np.isfinite(volume)]
    if len(values) == 0:
        warnings.warn("No finite values found for normalization; returning original array unchanged.")
        return volume.copy()
    mean_v = np.mean(values)
    std_v = np.std(values)
    if std_v == 0:
        warnings.warn("Standard deviation is zero; returning original array unchanged.")
        return volume.copy()
    return (volume - mean_v) / std_v


def normalize_under_mask(volume: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalize *volume* using statistics computed under *mask* > 0."""
    m_idx = mask > 0
    ref_mean = np.mean(volume[m_idx])
    ref_std = np.std(volume[m_idx])
    return (volume - ref_mean) / ref_std


def binarize(volume: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Threshold *volume* to a binary integer array (0 or 1)."""
    return (volume > threshold).astype(int)


def rotate_volume(
    volume: np.ndarray,
    rotation=None,
    rotation_angles=None,
    coord_space: str = "zxz",
    transpose_rotation: bool = False,
    degrees: bool = True,
    spline_order: int = 3,
) -> np.ndarray:
    """Rotate a 3-D volume around its centre using an affine transform.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D array.
    rotation : scipy.spatial.transform.Rotation, optional
        Pre-built Rotation object.  Takes precedence over *rotation_angles*.
    rotation_angles : array-like, optional
        Euler angles in *coord_space* order.  Used only when *rotation* is
        ``None``.
    coord_space : str, default "zxz"
        Euler-angle convention passed to ``Rotation.from_euler``.
    transpose_rotation : bool, default False
        If True the transposed rotation matrix is applied.
    degrees : bool, default True
        Whether *rotation_angles* are in degrees.
    spline_order : int, default 3
        Interpolation order for ``scipy.ndimage.affine_transform``.

    Returns
    -------
    np.ndarray
        Rotated volume with the same shape as *volume*.
    """
    from scipy.ndimage import affine_transform
    from scipy.spatial.transform import Rotation as _srot

    T = np.eye(4)
    structure_center = np.asarray(volume.shape) // 2
    T[:3, -1] = structure_center

    rot_matrix = np.eye(4)
    if rotation is not None:
        if transpose_rotation:
            rot_matrix[:3, :3] = rotation.as_matrix().T
        else:
            rot_matrix[:3, :3] = rotation.as_matrix()
    elif rotation_angles is not None:
        rot = _srot.from_euler(coord_space, rotation_angles, degrees=degrees)
        rot_matrix[:3, :3] = rot.as_matrix().T
    else:
        raise ValueError("Either rotation or rotation_angles must be provided.")

    final_matrix = T @ rot_matrix @ np.linalg.inv(T)
    result = np.empty(volume.shape)
    affine_transform(input=volume, output=result, matrix=final_matrix, order=spline_order)
    return result


def shift_array(volume: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Shift *volume* by *delta* voxels using a grid-wrapped affine transform."""
    from scipy.ndimage import affine_transform

    delta = np.asarray(delta)
    T = np.eye(4)
    T[:3, -1] = -delta
    result = np.empty(volume.shape)
    affine_transform(input=volume, output=result, matrix=T, mode="grid-wrap")
    return result


def flip_array(volume: np.ndarray, axis: str = "z") -> np.ndarray:
    """Flip *volume* along named axes.

    Parameters
    ----------
    volume : np.ndarray
        Input array (any dimensionality, but axes are interpreted as xyz).
    axis : str, default "z"
        Any combination of ``'x'``, ``'y'``, ``'z'`` (case-insensitive).
    """
    out = volume.copy()
    if "z" in axis.lower():
        out = np.flip(out, 2)
    if "y" in axis.lower():
        out = np.flip(out, 1)
    if "x" in axis.lower():
        out = np.flip(out, 0)
    return out


def recenter_volume(volume: np.ndarray, new_center: np.ndarray) -> np.ndarray:
    """Shift *volume* so that the point *new_center* moves to the box centre.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D array.
    new_center : np.ndarray
        Target centre coordinates (3-element array-like).
    """
    from scipy.ndimage import affine_transform

    new_center = np.asarray(new_center)
    structure_center = np.asarray(volume.shape) // 2
    shift = new_center - structure_center
    T = np.eye(4)
    T[:3, -1] = -shift
    result = np.empty(volume.shape)
    affine_transform(input=volume, output=result, matrix=T)
    return result


def pad_volume(
    volume: np.ndarray,
    new_size: np.ndarray,
    fill_value: float | None = None,
) -> np.ndarray:
    """Pad *volume* to *new_size* by centring it in a larger array.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D array.
    new_size : np.ndarray
        Desired output shape (3-element array-like).
    fill_value : float, optional
        Value used to fill the padding region.  Defaults to the mean of *volume*.
    """
    new_size = np.asarray(new_size, dtype=int)
    if fill_value is None:
        padded = np.full(new_size, np.mean(volume))
    else:
        padded = np.full(new_size, fill_value)
    vol_size = np.asarray(volume.shape)
    x_start = int(np.ceil((new_size[0] - vol_size[0]) / 2))
    y_start = int(np.ceil((new_size[1] - vol_size[1]) / 2))
    z_start = int(np.ceil((new_size[2] - vol_size[2]) / 2))
    padded[
        x_start : x_start + vol_size[0],
        y_start : y_start + vol_size[1],
        z_start : z_start + vol_size[2],
    ] = volume
    return padded


def symmetrize_volume(volume: np.ndarray, symmetry) -> np.ndarray:
    """Average *volume* over the rotations implied by *symmetry*.

    Parameters
    ----------
    volume : np.ndarray
        Input 3-D array.
    symmetry : str or int or float
        Cyclic symmetry specifier, e.g. ``"C5"`` or ``5``.
    """
    import re

    if isinstance(symmetry, str):
        nfold = int(re.findall(r"\d+", symmetry)[-1])
    elif isinstance(symmetry, (int, float)):
        nfold = int(symmetry)
    else:
        raise ValueError("symmetry must be a string (e.g. 'C5') or a number.")

    inplane_step = 360.0 / nfold
    rotated_sum = np.zeros(volume.shape)
    for inplane in range(1, nfold + 1):
        angle = 360.0 % (inplane * inplane_step)
        rotated_sum += rotate_volume(volume, rotation_angles=[0, 0, angle])
    return rotated_sum / nfold


def equalize_histogram_2d(image: np.ndarray, method: str = "contrast_stretching") -> np.ndarray:
    """Equalize the histogram of a single 2-D image.

    Parameters
    ----------
    image : np.ndarray
        2-D input image.
    method : str, default "contrast_stretching"
        One of ``"contrast_stretching"``, ``"equalization"``, or
        ``"adaptive_eq"``.

    Returns
    -------
    np.ndarray
        Histogram-equalized image.

    Raises
    ------
    ValueError
        If *method* is not recognised.
    """
    from skimage import exposure

    if method == "contrast_stretching":
        p2, p98 = np.percentile(image, (2, 98))
        return exposure.rescale_intensity(image, in_range=(p2, p98))
    elif method == "equalization":
        return exposure.equalize_hist(image)
    elif method == "adaptive_eq":
        img = image.astype(float)
        img = (img - img.min()) / (img.max() - img.min())
        return exposure.equalize_adapthist(img, clip_limit=0.03)
    else:
        raise ValueError(f"Unknown histogram equalization method: {method!r}")


def apply_fft_filter(image: np.ndarray, filter_array: np.ndarray) -> np.ndarray:
    """Multiply the Fourier transform of *image* by *filter_array* and return the real part.

    Both *image* and *filter_array* must have the same shape.  The forward FFT
    uses ``numpy.fft.fftshift`` so that the filter is expected to be centred at
    zero frequency.
    """
    ft = np.fft.fftshift(np.fft.fft2(image))
    filtered = np.fft.ifft2(np.fft.ifftshift(ft * filter_array))
    return filtered.real


def mask_overlap(mask_1: np.ndarray, mask_2: np.ndarray, threshold: float = 1.9) -> int:
    """Return the number of voxels where ``mask_1 + mask_2 > threshold``.

    Parameters
    ----------
    mask_1, mask_2 : np.ndarray
        Binary (or soft) mask arrays with the same shape.
    threshold : float, default 1.9
        Voxels whose summed value exceeds this threshold are counted as
        overlapping.
    """
    return int(np.sum(np.where((mask_1 + mask_2) <= threshold, 0, 1)))
