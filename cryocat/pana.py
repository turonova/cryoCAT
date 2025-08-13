import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from cryocat import cryomap
from cryocat import geom
from cryocat import ioutils
from cryocat import cryomotl
from cryocat import tmana
from cryocat import mathutils
from cryocat import wedgeutils
from scipy.spatial.transform import Rotation as srot
import re
from pathlib import Path
from cryocat import cryomask
import os
from skimage import measure
from skimage import morphology
import matplotlib.gridspec as gridspec
from scipy.interpolate import RegularGridInterpolator
from skimage.transform import rotate as skimage_rotate

import warnings

warnings.filterwarnings("ignore")


def rotate_image(image, alpha, fill_mode="constant", fill_value=0.0):
    """Rotate an ndarray image by a specified angle.

    Uses 'skimage.transform.rotate' to rotate the input image without resizing
    the output. Pixels outside the boundaries of the input are filled
    according to the specified mode and fill value.

    Some descriptions are from `scikit-image page <https://scikit-image.org>`_.

    Parameters
    ----------
    image : ndarray
        nD NumPy array representing the image to rotate.
    alpha : float
        Angle of rotation in degrees. Positive values rotate counterclockwise.
    fill_mode : {'constant', 'edge', 'symmetric', 'reflect', 'wrap'}, optional
        Points outside the boundaries of the input are filled according to the
        given mode. Default is 'constant'.
    fill_value : float, optional
        Value used to fill points outside the boundaries when `fill_mode`
        is 'constant'. Default is 0.0.

    Returns
    -------
    ndarray
        Rotated image as a NumPy array with the same shape as the input.
    """

    return skimage_rotate(image, alpha, resize=False, mode=fill_mode, cval=fill_value)


def _ctf(defocus, pshift, famp, cs, evk, f):
    """Compute Contrast Transfer Function (CTF) for an acquisition scheme.

        This function evaluates the 1D CTF over a 1D spatial frequency array
        using image acquisition parameters such as defocus, spherical aberration, and
        accelerating voltage.

        Parameters
        ----------
        defocus : array_like
            Defocus values (in :math:`\mu\mathrm{m}`) for each tilt.
            Shape is '(len(defocus), 1)'.
        pshift : array_like
            Phase shift values (in degrees) for each tilt. 0 if no phase shift is added.
            Shape is '(len(pshift), 1)'.
        famp : float
            Amplitude contrast (typically between 0.0 and 0.2).
        cs : float
            Spherical aberration (in mm).
        evk : float
            Accelerating voltage (in kV).
        f : ndarray
            1D spatial frequency magnitude array (in :math:`\mathrm{\AA}^{-1}`) at which to evaluate the CTF.
            Must be 1D, as other shapes cannot be broadcasted with defocus /
            phase shift array of shape (len(defocus), 1).

        Returns
        -------
        ctf : ndarray
            The computed 1D CTF curve evaluated at each tilt as a function of
            frequency array `f`. Shape is '(len(defocus), len(f))'.

        Notes
        -----
        - The output combines both sine and cosine components weighted by
          phase and amplitude contrast:
    .. math::
       \mathrm{CTF}(f) = \sqrt{1 - f_\mathrm{amp}^2} \, \sin(\chi(f)) + f_\mathrm{amp} \, \cos(\chi(f))
           where :math:`\chi(f)` is the aberration phase function.
        - this output is of radial symmetry
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


def generate_ctf(wl, slice_idx, slice_weight, binning):
    """Generate a CTF filter for a weighted volume that is subset of a full volume.

    This function computes defocus and phase shift-specific CTF filters for a template.

    It does so by:
    1. computing the frequency array of a tomogram;

    2. interpolating the CTF based on the full sized frequency array and
       defocus / pshift values;

    3. fourier cropping the full size CTF values to fit the size of a template, to keep
       only the lower frequencies;

    4. applying the CTF values to the given `slice_weight` array.

    Parameters
    ----------
    wl : pandas.DataFrame
        Wedge list dataframe containing metadata and microscope parameters
        for one tomogram.
    slice_idx : array-like of int
        Indices of the frequency components (aka. points in fourier space)
        to which the CTF filter should be applied.
    slice_weight : ndarray
        A 3D array representing the frequency domain weighting (missing wedge +
        bandpass) of each tilt.
        The shape is assumed to be (depth, height, width) or (zyx).
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
    - If `pshift` is not provided in `wl`, it defaults to zeros.
    """

    # determine template and full tomogram sizes
    tmpl_size = slice_weight.shape[0]
    full_size = int(max(wl["tomo_x"].values[0], wl["tomo_y"].values[0], wl["tomo_z"].values[0]))
    pixelsize = wl["pixelsize"].values[0] * binning

    # calculate frequency arrays of tomogram and template sizes
    freqs_full = mathutils.compute_frequency_array((full_size,), pixelsize)
    freqs_crop = mathutils.compute_frequency_array((tmpl_size,), pixelsize)
    freqs_crop = freqs_crop[tmpl_size // 2 :]  # only need half of the freq array

    # selects the central part (len = tmpl length) of the frequency array from
    # the full frequency array and wraps it to match FFT layout
    f_idx = np.zeros(full_size, dtype="bool")
    f_idx[:tmpl_size] = 1
    f_idx = np.roll(f_idx, -tmpl_size // 2)
    f_idx = np.nonzero(f_idx)[0]  # get the indices of selected part

    # acquisition parameters
    defocus = np.array(wl["defocus"])
    pshift = wl.get("pshift", np.zeros_like(defocus))
    famp = wl["amp_contrast"].values[0]
    cs = wl["cs"].values[0]
    evk = wl["voltage"].values[0]

    # compute the 1D CTF as a function of full tomo freq magnitude array
    full_ctf = np.abs(_ctf(defocus[:, None], pshift[:, None], famp, cs, evk, freqs_full))

    # cropping the ctf that only belongs to template size from the full size ctf
    # cropping is done in fourier space to remove high freq ctf content
    ft_ctf = np.fft.fft(full_ctf, axis=1)
    ft_ctf = ft_ctf[:, f_idx] * tmpl_size / full_size
    crop_ctf = np.real(np.fft.ifft(ft_ctf, axis=1))
    crop_ctf = crop_ctf[:, tmpl_size // 2 :]

    # init an empty vol to save ctf values
    ctf_filt = np.zeros_like(slice_weight)

    # map weighted spatial components to their freq magnitudes
    x = np.fft.ifftshift(mathutils.compute_frequency_array(slice_weight.shape, pixelsize))

    # for each tilt, interpolate the CTF value for each weighted voxel
    for ictf, sidx in zip(crop_ctf, slice_idx):
        ip = RegularGridInterpolator(
            (freqs_crop,),  # Grid points
            ictf,  # Values on the grid
            method="linear",  # Interpolation method ('linear', 'nearest')
            bounds_error=False,  # Do not raise error for out-of-bound points
            fill_value=0,  # Fill with 0 for out-of-bound points
        )

        # write the interpolated CTF values for all weighted voxels to an empty vol
        ctf_filt[sidx] += ip(x[sidx])

    # ? np.nan_to_num(ctf_filt)

    # apply the CTF filter to input weight filter
    ctf_filt *= slice_weight

    return ctf_filt


def generate_exposure(wedgelist, slice_idx, slice_weight, binning):
    r"""Generate an exposure-based filter to account for frequency-dependent signal
    attenuation due to electron dose in cryo-electron tomography.

    This function models the decay of signal at different spatial frequencies
    based on cumulative electron exposure per tilt. The filter is computed per
    tilt exposure using an empirical decay function and returned as a 3D volume matching
    the input slice weights.

    Parameters
    ----------
    wedgelist : pandas.DataFrame
        Metadata table containing at least the following columns:
        - "exposure": float, cumulative electron exposure per tilt (in \mathrm{e^- / \AA^2})
        - "pixelsize": float, unbinned pixel size (in :math:`\mathrm{\AA}`)

    slice_idx : list of numpy.ndarray
        Indices of the frequency components (aka. points in fourier space)
        to which the exposure filter should be applied.

    slice_weight : numpy.ndarray
        A 3D array representing the frequency domain weighting (missing wedge +
        bandpass) of each tilt.
        The shape is assumed to be (depth, height, width) or (zyx).

    binning : int
        The binning factor of the tomogram. Used to scale the pixel size.

    Returns
    -------
    exp_filt : numpy.ndarray
        3D exposure filter of the same shape as `slice_weight`. Each voxel contains
        a multiplicative factor representing attenuation based on frequency and dose.

    Notes
    -----
    The attenuation function follows the empirical dose-dependent decay model:

    .. math::
        \exp\Bigg(-\frac{\mathrm{exposure}}{2 \, (a \, f^b + c)}\Bigg)

    where f is the spatial frequency in :math:`\mathrm{\AA}^{-1}` and a, b, c are empirical constants.
    See `this paper <https://elifesciences.org/articles/06980>`_ for more info.
    """

    expo = wedgelist["exposure"].values
    a, b, c = (0.245, -1.665, 2.81)  # values that best fit the function in the paper
    pixelsize = wedgelist["pixelsize"].values[0] * binning

    freq_array = np.fft.ifftshift(mathutils.compute_frequency_array(slice_weight.shape, pixelsize))

    exp_filt = np.zeros_like(slice_weight)

    # for each weighted component, find its corresponding freq then exposure
    for expi, idx in zip(expo, slice_idx):
        freqs = freq_array[idx]
        exp_filt[idx] += np.exp(-expi / (2 * ((a * freqs**b) + c)))

    exp_filt *= slice_weight

    return exp_filt


def generate_wedgemask_slices_template(wedgelist, template_filter):
    """Generate missing wedge masks and weights for a template in Fourier space.

    This function simulates the sampling of a 3D template volume in Fourier space,
    given a set of tilt angles. It computes:

    - Which voxels are covered based on tilt range.

    - A normalized weight to compensate for unequal coverage of tilts,
      due to rotation interpolation artifacts (some voxels may be covered by >1 tilt).

    - A binary template mask that shows active voxels
      after bandpass & missing wedge filters.

    Parameters
    ----------
    wedgelist : pandas.DataFrame
        A table with a column `tilt_angle` (in degrees) representing the tilt angles
        of the acquisition. This should only contain info for one tomogram.
    template_filter : ndarray
        A 3D frequency-filtered template in numpy array. The dimensions have to be equal,
        aka. a cube shape.

    Returns
    -------
    active_slices_idx : list of tuple of ndarrays
        A list of index tuples '[(zs, ys, xs)]' for each tilt, indicating where the
        projection intersects the bandpassed Fourier space.
        Each value inside the tuple is an array indicating where all active points
        is on each dimension. len(active_slices_idx) = len(wedgelist).
    wedge_slices_weights : ndarray
        A 3D array of the same shape as `template_filter`, containing weights that
        normalize the contribution of each voxel based on how frequently it was sampled.
        Voxels that are not sampled receive a weight of 0.
    wedge_slices : ndarray
        A binary 3D mask of the same shape as `template_filter` with 1s at all voxels
        that were sampled by at least one tilt.

    Notes
    -----
    - Since the wedge mask is manually built and assume zero frequency component in
      center of array, shifting of the mask from center of array to top-left corner is
      needed for later operations in fourier space.

    - Rotating the ray at specified degrees (not a continuous range) to match the real
      tilting scheme better (star-shaped).
    """

    template_size = np.array(template_filter.shape)
    assert len(template_size) == 3, "The template is not 3D!"
    assert len(np.unique(template_size)) == 1, "The template is not cubic in shape!"

    # get the x and z length of the filtered template
    # (original codes uses matlab axes 1 and 3 which correspond to x and z
    # according to the original gsg_mrcread; so with mrcfile this is 2 and 0)
    mx = np.max(template_size[[2, 0]])  # template is in zyx order

    # create a 2D (xz) image of a line going down through the middle
    # to simluate a projection ray at 0 deg tilt
    img = np.zeros((mx, mx))
    img[:, mx // 2] = 1.0

    # binary mask of where the bandpass filter is applied on the template
    bpf_idx = template_filter > 0  # shouldn't this be in fourier space?

    # initialize a few vars to store info
    active_slices_idx = []
    wedge_slices_weights = np.zeros_like(template_filter)
    weight = np.zeros_like(template_filter)

    for alpha in wedgelist["tilt_angle"]:

        # rotate the ray projection img by alpha deg
        r_img = rotate_image(img, alpha)

        # after rotation, smoothing effect thus need to turn interpolated values into binary
        crop_r_img = r_img > np.exp(-2)  # e^-2 is common effective support threshold for g-blur

        # repeat the 2D ray img into a 3D vol;
        # transpose it to the same dims as template (yxz to zyx);
        # then shift the zero freuqency voxels to origin to match FFT format
        slice_vol = np.fft.ifftshift(np.transpose(np.tile(crop_r_img, (mx, 1, 1)), (2, 0, 1)))

        # apply the actual bandpass filter mask to the 3D ray image
        slice_idx = slice_vol & bpf_idx

        # add together all tilts into one weight that indicates the complete missing wedge loc
        weight += slice_idx

        # store locations of nonzero voxels per tilt
        active_slices_idx.append(np.nonzero(slice_idx))

    # add together projection interporlations may lead to overlapping pixels;
    # invert the values to balance the over/under-sampling
    w_idx = np.nonzero(weight)
    wedge_slices_weights[w_idx] = 1.0 / weight[w_idx]

    # create a binary wedge mask
    wedge_slices = np.zeros_like(weight)
    wedge_slices[w_idx] = 1.0

    return active_slices_idx, wedge_slices_weights, wedge_slices


def generate_wedgemask_slices_tile(wedgelist, tile_filter):
    """Generate missing wedge masks and weights for a subtomo or a tile from tomogram
    in Fourier space.

    This function simulates the sampling of a 3D tile volume in Fourier space,
    given a set of tilt angles. It computes:

    - Which voxels are covered based on tilt range.

    - A normalized weight to compensate for unequal coverage of tilts,
      due to rotation interpolation artifacts (some voxels may be covered by >1 tilt).

    - A binary tile mask that shows active voxels
      after bandpass & missing wedge filters.

    Parameters
    ----------
    wedgelist : pandas.DataFrame
        A table with a column `tilt_angle` (in degrees) representing the tilt angles
        of the acquisition. This should only contain info for one tomogram.
    tile_filter : ndarray
        A 3D frequency-filtered tile in numpy array. The dimensions have to be equal,
        aka. a cube shape.
        The array itself is not used for its values, only its shape.

     Returns
    -------
    wedge_slices : ndarray
        A binary 3D mask of the same shape as `tile_filter` with 1s at all voxels
        that were sampled by at least one tilt.

    """

    tile_size = np.array(tile_filter.shape)

    # make sure that the tile size is cubic
    assert len(tile_size) == 3, "The tile is not 3D!"
    assert len(np.unique(tile_size)) == 1, "The tile is not cubic in shape!"

    # original codes uses matlab axes 1 and 3 which correspond to x and z
    # according to the original gsg_mrcread; so with mrcfile this is 2 and 0
    mx = np.max(tile_size[[2, 0]])  # tile is in zyx order
    img = np.zeros((mx, mx))
    img[:, mx // 2] = 1.0  # create projection line to simulate signal direction

    # initialize a vol to store missing wedge filter
    tile_bin_slice = np.zeros(tile_size[[2, 0]], dtype="float32")

    for alpha in wedgelist["tilt_angle"]:

        r_img = rotate_image(img, alpha)
        # tile filter?
        r_img = np.fft.fftshift(np.fft.fft2(r_img))
        new_img = np.real(np.fft.ifft2(np.fft.ifftshift(r_img)))
        new_img /= np.max(new_img)

        # rotation smoothing effect thus need to turn interpolated values into binary
        tile_bin_slice += new_img > np.exp(-2)

    # generate tile binary wedge filter
    tile_bin_slice = (tile_bin_slice > 0).astype("float32")

    # repeat the 2D ray img into a 3D vol;
    # transpose it to the same dims as tile (yxz to zyx);
    # then shift the zero freuqency voxels to origin
    wedge_slices = np.fft.ifftshift(np.transpose(np.tile(tile_bin_slice, (tile_size[1], 1, 1)), (2, 0, 1)))

    return wedge_slices


def generate_wedge_masks(
    template_size,
    tile_size,
    wedgelist,
    tomo_number,
    binning=1,
    low_pass_filter=None,
    high_pass_filter=None,
    ctf_weighting=False,
    exposure_weighting=False,
    output_template=None,
    output_tile=None,
):
    """Generates wedge masks for both template and subtomo tile volumes.

    This function computes frequency-space masks that account for the missing wedge
    artifacts based on a provided wedge list. Optionally applies low-pass and
    high-pass filters to the masks. CTF and exposure filtering may also be applied on
    top of the wedge masks.

    Parameters
    ----------
    template_size : int or array-like
        The size of the template. Could be a single int (assume cubic shape) or a
        tuple, list or numpy.ndarray of length of 3.

    tile_size : tuple of int
        The size of the subtomo or tile. Could be a single int (assume cubic shape) or a
        tuple, list or numpy.ndarray of length of 3.

    wedgelist : str or pandas.DataFrame
        Path to the STOPGAP wedge list file (.star) or a preloaded DataFrame.
        The list should contain entries specifying the missing wedge parameters
        per tomogram.

    tomo_number : int
        The tomogram number to select from the wedge list.

    binning : int, optional
        Binning factor used in tomogram. Default is 1 (no binning).

    low_pass_filter : int, optional
        If provided, applies a low-pass filter in Fourier space with the given cutoff
        in Fourier pixels.

    high_pass_filter : int, optional
        If provided, applies a high-pass filter in Fourier space with the given cutoff
        in Fourier pixels.

    ctf_weighting : bool, optional
        If True, applies CTF weighting. Default is False.

    exposure_weighting : bool, optional
        If True, applies exposure weighting. Default is False.

    output_template : str, optional
        Path to save the resulting template wedge mask. Can use `create_wedge_names`
        to generate this path.

    output_tile : str, optional
        Path to save the resulting tile wedge mask. Can use `create_wedge_names`
        to generate this path.

    Returns
    -------
    filter_template_t : ndarray
        The wedge-weighted and filtered Fourier mask for the template volume.

    filter_tile_t : ndarray
        The wedge-weighted and filtered Fourier mask for the tile volume.
    """

    # init mask volumes for template and tile
    filter_template = np.ones(cryomask.get_correct_format(template_size))
    filter_tile = np.ones(cryomask.get_correct_format(tile_size))

    # get relevant subset of the wedgelist
    wedgelist = wedgeutils.load_wedge_list_sg(wedgelist)
    wedgelist = wedgelist.loc[wedgelist["tomo_num"] == tomo_number]

    if low_pass_filter:
        filter_template = cryomap.lowpass(filter_template, fourier_pixels=low_pass_filter)
        filter_tile = cryomap.lowpass(filter_tile, fourier_pixels=low_pass_filter)

    if high_pass_filter:
        filter_template = cryomap.highpass(filter_template, fourier_pixels=low_pass_filter)
        filter_tile = cryomap.highpass(filter_tile, fourier_pixels=low_pass_filter)

    # compute wedge masks to inited masks
    active_slices_idx, wedge_slices_weights, wedge_slices_template = generate_wedgemask_slices_template(
        wedgelist, filter_template
    )
    wedge_slices_template_tile = generate_wedgemask_slices_tile(wedgelist, filter_tile)

    # apply wedge masks on initialized (or bandpassed) masks
    filter_template = wedge_slices_template * filter_template
    filter_tile = wedge_slices_template_tile * filter_tile

    # if true, apply exposure and ctf filtering to template filter only
    if exposure_weighting:
        filter_template *= generate_exposure(wedgelist, active_slices_idx, wedge_slices_weights, binning)

    if ctf_weighting:
        filter_template *= generate_ctf(wedgelist, active_slices_idx, wedge_slices_weights, binning)

    if output_template:
        cryomap.write(filter_template, output_template, transpose=False, data_type=np.single)

    if output_tile:
        cryomap.write(filter_tile, output_tile, transpose=False, data_type=np.single)

    filter_template_t = filter_template.transpose(2, 1, 0)  # zyx to xyz
    filter_tile_t = filter_tile.transpose(2, 1, 0)

    return filter_template_t, filter_tile_t


def create_structure_path(folder_path, structure_name):
    """Put together a path for the structure folder by combining a base folder path
    and the name of the structure.

    Parameters
    ----------
    folder_path : str
        The base directory path where the structure folder should be created.
        It should include a trailing slash if needed, otherwise the function
        will not insert a separator between `folder_path` and `structure_name`.
    structure_name : str
        The name of the structure of interest.

    Returns
    -------
    structure_folder : str
        Full path to the structure folder.
    """

    structure_folder = folder_path + structure_name + "/"
    return structure_folder


def create_em_path(folder_path, structure_name, em_filename):
    """Constructs the full path to an `.em` file within a specific structure folder.

    Parameters
    ----------
    folder_path : str
        The base directory path.
    structure_name : str
        The name of the structure, used to create a subdirectory under `folder_path`.
    em_filename : str
        The name of the `.em` file, without the file extension.

    Returns
    -------
    em_path : str
        The full path to the `.em` file, including the `.em` extension.
    """

    structure_folder_path = create_structure_path(folder_path, structure_name)
    em_path = structure_folder_path + em_filename + ".em"
    return em_path


def create_subtomo_name(structure_name, motl_name, tomo_id, boxsize):
    """Generate a standardized filename for a subtomogram.
    The generated file name is
    :code:`subtomo_<structure_name>_m<motl_name>_t<tomo_id>_s<boxsize>.em`

    Parameters
    ----------
    structure_name : str
        Name of the structure.
    motl_name : str
        Name of the motive list file containing particle information.
    tomo_id : str
        Tomogram id/number from which the subtomogram is extracted.
    boxsize : int
        Size of the subtomogram box in voxels.

    Returns
    -------
    subtomo_name : str
        The constructed filename for the subtomogram.
    """

    subtomo_name = "subtomo_" + structure_name + "_m" + motl_name + "_t" + tomo_id + "_s" + str(boxsize) + ".em"

    return subtomo_name


def create_tomo_name(
    folder_path,
    tomo,
):
    """Generate a full file path for a tomogram with an .mrc extension.

    Parameters
    ----------
    folder_path : str
        Path to the directory containing the tomogram.
    tomo : str
        Base name of the tomogram file (without extension).

    Returns
    -------
    tomo_name : str
        Full path to the tomogram file with .mrc extension.
    """

    tomo_name = folder_path + tomo + ".mrc"
    return tomo_name


def create_wedge_names(wedge_path, tomo_number, boxsize, binning, filter=None):
    """Generate filenames for tomogram and template wedge masks with filtering info.

    If no filter size is provided, it defaults to half of the box size.

    Parameters
    ----------
    wedge_path : str
        Directory path where the wedge files will be stored.
    tomo_number : int
        Number of the tomogram.
    boxsize : int
        Size of the subtomogram box in voxels.
    binning : int
        Binning level applied to the tomogram.
    filter : int, optional, default=boxsize // 2
        Size of the filter applied during processing.

    Returns
    -------
    tomo_wedge : str
        Filename for the filtered tomogram wedge mask.
    tmpl_wedge : str
        Filename for the filtered template wedge mask.
    """

    if filter is None:
        filter = boxsize // 2

    file_ending = str(boxsize) + "_t" + str(tomo_number) + "_b" + str(binning) + "_f" + str(filter) + ".em"
    tomo_wedge = wedge_path + "tile_filt_" + file_ending
    tmpl_wedge = wedge_path + "tmpl_filt_" + file_ending

    return tomo_wedge, tmpl_wedge


def create_output_base_name(tmpl_index):
    """Generates the base name for peak analysis output folders / files.

    Includes the index of the row analyzed from the template list csv.

    Parameters
    ----------
    tmpl_index : int
        The index of the row analyzed in the templatel list csv.

    Returns
    -------
    output_base : str
        Output file based name. It should be "id_<tmpl_index>".
    """

    output_base = "id_" + str(tmpl_index)
    return output_base


def create_output_folder_name(tmpl_index):
    """Generates the name of the folder (not the full path) where the peak analysis
    results will be stored, given the index of the row from the template list csv.

    Parameters
    ----------
    tmpl_index : int
        The index of the row analyzed in the templatel list csv.

    Returns
    -------
    str
        The name of result folder. It should be 'id_<tmpl_index>_results'.
    """

    return create_output_base_name(tmpl_index) + "_results"


def create_output_folder_path(folder_path, structure_name, folder_spec):
    """Constructs the full path of the output folder.

    Parameters
    ----------
    folder_path : str
        The path to the peak analysis base folder.
    structure_name : str
        The name of the structure.
    folder_spec : int or else
        Information about the output folder. If int (should be an index from the
        template list csv), the output folder name will be 'id_<folder_spec>_results'.
        If not int, the output folder name will be '<folder_spec>'.

    Returns
    -------
    output_path : str
        The full path to the output folder. Should be either 'id_<folder_spec>_results'
        or '<folder_spec>'.
    """

    if isinstance(folder_spec, int):
        output_path = create_structure_path(folder_path, structure_name) + create_output_folder_name(folder_spec) + "/"

    else:
        output_path = create_structure_path(folder_path, structure_name) + folder_spec + "/"

    return output_path


def get_indices(template_list, conditions, sort_by=None):
    """Get the indices of a filtered and optionally sorted template list csv file.

    Parameters
    ----------
    template_list : str
        Path to the template list csv file.
    conditions : dict
        Dictionary where keys are template list column names and values are the values to
        filter by. Only rows matching all conditions are retained.
    sort_by : str, optional
        Column name to sort the filtered DataFrame by. If None, no sorting
        is applied.

    Returns
    -------
    pandas.Index
        Index of the filtered (and optionally sorted) rows in the template list DataFrame.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for key, value in conditions.items():
        temp_df = temp_df.loc[temp_df[key] == value, :]
        # display(temp_df)

    if sort_by is not None:
        temp_df = temp_df.sort_values(by=sort_by, ascending=True)

    return temp_df.index


def get_sharp_mask_stats(input_mask):
    """Get the boxsize of the nonzero element inside the sharp mask and the total
    volume of the nonzero element.

    Parameters
    ----------
    input_mask : str or numpy.ndarray
         Input mask specified either by its path or already loaded as 3D numpy.ndarray.
         The nonzero element in this mask has sharp edges, i.e.: the values inside
         the mask are either 1 or 0.

    Returns
    -------
    n_voxels : int
        The number of voxels of where the input_mask is not zero.
    mask_bb : tuple of int
        The bounding box size in (x, y, z) of the nonzero element.
    """

    mask_bb = cryomask.get_mass_dimensions(input_mask)
    n_voxels = np.count_nonzero(input_mask)

    return n_voxels, mask_bb


def get_soft_mask_stats(input_mask):
    """Get the boxsize of the element (> 0.5) inside the soft mask and the total
    volume of the nonzero element.

    Parameters
    ----------
    input_mask : str or numpy.ndarray
         Input mask specified either by its path or already loaded as 3D numpy.ndarray.
         The nonzero element in this mask has soft edges, i.e.: the values inside
         the mask are between 0 and 1.

    Returns
    -------
    n_voxels : int
        The number of voxels of where the input_mask is bigger than 0.5.
    mask_bb : tuple of int
        The bounding box size in (x, y, z) of the element that has values bigger than 0.5.
    """

    # mask the mask where the values are bigger than 0.5
    mask_th = np.where(input_mask > 0.5, 1.0, 0.0)
    mask_bb = cryomask.get_mass_dimensions(mask_th)
    n_voxels = np.count_nonzero(mask_th)

    return n_voxels, mask_bb


def cut_the_best_subtomo(tomogram, motl_path, subtomo_shape, output_file):
    """Extract the highest-scoring subtomogram from a tomogram.

    Loads a tomogram and its corresponding particle motive list, identifies the entry
    with the highest score, and extracts the aligned subtomogram around that
    position. Optionally writes the result to a file.

    Parameters
    ----------
    tomogram : str
        Path to the tomogram file to extract from.
    motl_path : str
        Path to the motive list with extracted particle information (.csv or
        compatible format).
    subtomo_shape : tuple of int
        Shape of the subtomogram to extract, in (x, y, z) order.
    output_file : str or None
        Path to save the extracted subtomogram. If None, the file is not saved.

    Returns
    -------
    subvolume_sh : numpy.ndarray
        The extracted and shifted subtomogram.
    angles : numpy.ndarray
        The Euler angles (phi, theta, psi) rotation associated with the best subtomogram.
    """

    tomo = cryomap.read(tomogram)
    m = cryomotl.Motl.load(motl_path)
    m.update_coordinates()

    max_idx = m.df["score"].idxmax()  # get the dataframe idx where ccc is max

    coord = m.df.loc[m.df.index[max_idx], ["x", "y", "z"]].to_numpy() - 1
    shifts = -m.df.loc[m.df.index[max_idx], ["shift_x", "shift_y", "shift_z"]].to_numpy()
    angles = m.df.loc[m.df.index[max_idx], ["phi", "theta", "psi"]].to_numpy()

    subvolume = cryomap.extract_subvolume(tomo, coord, subtomo_shape)
    subvolume_sh = cryomap.shift2(subvolume, shifts)
    # subvolume_rot = cryomap.rotate(subvolume_sh,rotation_angles=angles)

    if output_file is not None:
        cryomap.write(subvolume_sh, output_file, data_type=np.single)

    return subvolume_sh, angles


def create_subtomograms_for_tm(template_list, parent_folder_path):
    """Generates subtomograms with highest ccc score from tomograms for each
    entry in template list csv.

    Updates the template list with orientation and status info, and saves
    the updated list.

    Parameters
    ----------
    template_list : str
        Path to the template list file with motl path info.
    parent_folder_path : str
        Path to the base directory for peak analysis.

    Returns
    -------
    temp_df : pandas.DataFrame
        The updated DataFrame containing subtomogram metadata, including
        creation status, orientation angles, and filenames.
    """

    temp_df = pd.read_csv(template_list, index_col=0)
    unique_entries = temp_df.groupby(["Structure", "Motl", "Tomogram", "Boxsize"]).groups
    entry_indices = list(unique_entries.values())

    for i, entry in enumerate(unique_entries):
        if np.all(temp_df.loc[entry_indices[i], "Tomo created"]):
            continue  # skip if subtomograms have been created
        else:
            motl = create_em_path(parent_folder_path, entry[0], entry[1])
            boxsize = entry[3]

            # find out which entries from template list have not had subtomos created
            not_created = temp_df.loc[temp_df["Tomo created"] == False, "Tomo created"].index
            create_idx = np.intersect1d(not_created, entry_indices[i])

            # cut the subtomos with best ccc scores and save them
            subtomo_name = create_subtomo_name(entry[0], entry[1], entry[2], boxsize)
            _, subtomo_rotation = cut_the_best_subtomo(
                create_tomo_name(parent_folder_path, entry[2]),
                motl,
                (boxsize, boxsize, boxsize),
                create_structure_path(parent_folder_path, entry[0]) + subtomo_name,
            )

            # updates the template list df after creating best subtomos
            temp_df.loc[create_idx, ["Phi", "Theta", "Psi"]] = np.tile(subtomo_rotation, (create_idx.shape[0], 1))
            temp_df.loc[create_idx, "Tomo created"] = True
            temp_df.loc[create_idx, "Tomo map"] = subtomo_name[0:-3]

    temp_df.to_csv(template_list)

    return temp_df


def get_mask_stats(template_list, indices, parent_folder_path):
    """Compute and update mask statistics for specified rows in a template list.

    Loads info about soft and tight (sharp) masks to computes volume-related statistics,
    including nonzero voxel counts, mask element bounding box dimensions, and solidity.
    Updates the CSV with these values: specifically:

        - 'Voxels': Number of nonzero voxels in the soft mask.

        - 'Voxels TM': Number of nonzero voxels in the tight mask.

        - 'Dim x', 'Dim y', 'Dim z': Dimensions of the bounding box enclosing the tight mask.

        - 'Solidity': Solidity metric of the tight mask (volume / convex hull volume).

    Parameters
    ----------
    template_list : str
        Path to the CSV template list file.
    indices : list of int
        List of row indices in the CSV to process.
    parent_folder_path : str
        Base directory where folders for all structures are.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        soft_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"]))
        sharp_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))

        # calculate how compactful the mask is
        solidity = cryomask.compute_solidity(sharp_mask)

        voxels, bbox = get_sharp_mask_stats(sharp_mask)
        voxels_soft, _ = get_soft_mask_stats(soft_mask)

        temp_df.at[i, "Voxels"] = voxels_soft  # total vol of soft mask
        temp_df.at[i, "Voxels TM"] = voxels  # total vol of sharp mask
        temp_df.at[i, "Dim x"] = bbox[0]  # bounding box x for mask element
        temp_df.at[i, "Dim y"] = bbox[1]  # bounding box y for mask element
        temp_df.at[i, "Dim z"] = bbox[2]  # bounding box z for mask element
        temp_df.at[i, "Solidity"] = solidity

        temp_df.to_csv(template_list)  # save new results back to template list


def compute_sharp_mask_overlap(template_list, indices, angle_list_path, parent_folder_path, angles_order="zxz"):
    """Compute the overlap between the original tight mask and rotated versions of it.

    For each template specified by its index in the template list, this function loads the
    corresponding tight mask and set of rotation angles, rotates the mask accordingly,
    computes the overlap (intersection) with the original mask, and writes the results
    to a CSV file.

    Parameters
    ----------
    template_list : str
        Path to the CSV template list file.
    indices : list of int
        List of row indices in the CSV to process.
    angle_list_path : str
        Path to the directory containing angle list files used for rotation.
    parent_folder_path : str
        Base directory where folders for all structures are.
    angles_order : str, optional
        The rotation order used to interpret the Euler angles (default is "zxz").
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    # only proceed for rows that haven't been analyzed
    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        mask_name = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"])
        mask = cryomap.read(mask_name)
        angle_list = angle_list_path + temp_df.at[i, "Angles"]
        angles = ioutils.rot_angles_load(angle_list, angles_order)
        rotations = srot.from_euler("zxz", angles, degrees=True)

        voxel_count = []

        # for each rotation, compute the overlapping vol w/ non-rotated mask
        for j in rotations:
            mask_rot = cryomap.rotate(mask, rotation=j, transpose_rotation=True)
            mask_rot = np.where(mask_rot > 0.1, 1.0, 0.0)
            voxel_count.append(np.count_nonzero(mask_rot * mask))

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        csv_name = output_folder + output_base + ".csv"
        info_df = pd.read_csv(csv_name, index_col=0)
        info_df["Tight mask overlap"] = np.asarray(voxel_count)
        info_df.to_csv(csv_name)


def check_existing_tight_mask_values(template_list, indices, parent_folder_path, angle_list_path, angles_order="zxz"):
    """Check and populate "Tight mask overlap" values for given rows in template list.

    This function verifies whether specified rows have "Tight mask overlap" values.
    If the values are missing, it attempts to find an analyzed row (with the same tight
    mask and degrees) from which the overlap values can be copied. If no such data is
    found, the function computes the overlap and saves it back to the output csvs.

    Parameters
    ----------
    template_list : str
        Path to the CSV template list file.
    indices : list of int
        List of row indices in `template_list` to check.
    parent_folder_path : str
        Base directory where structure folders are located.
    angle_list_path : str
        Path to the directory containing rotation angle list files.
    angles_order : str, optional
        Rotation order to interpret the Euler angles when computing overlaps.
        Default is "zxz".
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        rot_info = pd.read_csv(output_folder + output_base + ".csv", index_col=0)

        data_found = False

        # if the value does not exist, find and copy from other "same" analysis outputs
        if "Tight mask overlap" not in rot_info.columns:
            tm_spec = {"Done": True, "Tight mask": temp_df.at[i, "Tight mask"], "Degrees": temp_df.at[i, "Degrees"]}
            done_idx = get_indices(template_list, tm_spec)

            for j in done_idx:
                csv_file = (
                    create_output_folder_path(
                        parent_folder_path, temp_df.at[j, "Structure"], temp_df.at[j, "Output folder"]
                    )
                    + create_output_base_name(j)
                    + ".csv"
                )
                diff_info = pd.read_csv(csv_file, index_col=0)
                if "Tight mask overlap" in diff_info.columns:
                    rot_info["Tight mask overlap"] = diff_info["Tight mask overlap"].values
                    data_found = True
                    rot_info.to_csv(output_folder + output_base + ".csv")
                    break
        else:
            data_found = True

        # if the value is not found in "same" analysis, computes it
        if not data_found:
            print(f"Computing sharp mask overlap for index {i}")
            compute_sharp_mask_overlap(
                template_list, [i], angle_list_path, parent_folder_path, angles_order=angles_order
            )


def compute_dist_maps_voxels(template_list, indices, parent_folder_path, morph_footprint=(2, 2, 2)):
    """Compute a few morphology related measurements for areas with highest cc score.

    For each specified rows in the template list, this function processes angular distance
    maps to find patches (i.e. connected components) within the search angle and
    have highest cross correlation score. Morphological properties computed include
    voxel count, solidity, and bounding box dimensions. The results are stored back
    into the template list CSV file.

    Parameters
    ----------
    template_list : str
        Path to the CSV template list file.
    indices : list of int
        Row indices in `template_list` to process.
    parent_folder_path : str
        Base directory where structure folders are located.
    morph_footprint : tuple of int, optional
        Size of the structuring element used for binary opening during morphological
        processing of labeled regions. Default is (2, 2, 2).

    Notes
    -----
    For "dist_all", the threshold is set to '2.0 * degrees'; for the other maps,
      the threshold is 'degrees'.
        * degrees here is the search increment angle used.

        * the max angular distance (i.e. "dist_all") given angular combinations within
          the "degrees" value is just slightly above 2*degrees.

        * cc score decreases with increasing angular distance (given no symmetry),
          therefore only looking at places where the dist is no larger than search angles

    Labeled connected components are analyzed to extract:
        * Voxel count (VC) of components with highest cc score

        * Solidity

        * Morphologically opened voxel count (VCO)

        * Bounding box dimensions (O x, O y, O z)

    Labeled masks and morphologically opened masks are saved as `.em` files with `_label` and `_label_open` suffixes, respectively.

    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        degrees = temp_df.at[i, "Degrees"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        # only focus on central part of scores map (size of a template or subtomo)
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=10)
        scores_map *= cc_mask

        # find coordinates of centers of highest cc scores (peaks)
        peak_center, _, _ = tmana.create_starting_parameters_2D(scores_map)

        dist_names = ["dist_all", "dist_normals", "dist_inplane"]

        for j, value in enumerate(dist_names):
            dist_map = cryomap.read(output_folder + output_base + "_angles_" + value + ".em")

            # save peak coords so they don't disappear when masked (value could be > degrees)
            dist_map[peak_center[0], peak_center[1], peak_center[2]] = degrees

            # only save where the angular distance is within search angle degree
            if j == 0:
                dist_map = np.where(dist_map <= 2.0 * degrees, 1.0, 0.0)  #
            else:
                dist_map = np.where(dist_map <= degrees, 1.0, 0.0)

            # label and measure each connected component in dist_map
            dist_label = measure.label(dist_map, connectivity=1)
            dist_props = pd.DataFrame(measure.regionprops_table(dist_label, properties=("label", "area", "solidity")))

            # from all the connected components, find the ones that have highest ccc
            peak_label = dist_label[peak_center[0], peak_center[1], peak_center[2]]
            label_vc = dist_props.loc[dist_props["label"] == peak_label, "area"].values
            column_name = "VC " + value  # voxel count
            temp_df.at[i, column_name] = label_vc

            label_sol = dist_props.loc[dist_props["label"] == peak_label, "solidity"].values
            column_name = "Solidity " + value
            temp_df.at[i, column_name] = label_sol

            # save highest ccc connect components into a new vol as binary
            dist_label = np.where(dist_label == peak_label, 1.0, 0.0)
            cryomap.write(
                dist_label, output_folder + output_base + "_angles_" + value + "_label.em", data_type=np.single
            )

            # remove labels that are smaller than the footprint
            open_label = morphology.binary_opening(dist_label, footprint=np.ones(morph_footprint), out=None)
            open_label = measure.label(open_label, connectivity=1)
            peak_label = open_label[peak_center[0], peak_center[1], peak_center[2]]
            open_label = np.where(open_label == peak_label, 1.0, 0.0)

            # count the volumes of the size filtered peaklabels
            label_vc = np.count_nonzero(open_label)
            column_name = "VCO " + value
            temp_df.at[i, column_name] = label_vc
            cryomap.write(
                open_label, output_folder + output_base + "_angles_" + value + "_label_open.em", data_type=np.single
            )
            # print(column_name, label_vc)

            # get the dimensions of the bounding box for each open peak label
            open_dim = cryomask.get_mass_dimensions(open_label)
            for d, dim in enumerate(["x", "y", "z"]):
                column_name = "O " + value + " " + dim
                temp_df.at[i, column_name] = open_dim[d]

        temp_df.to_csv(template_list)  # to save what was finished in case of a crush


def compute_center_peak_stats_and_profiles(template_list, indices, parent_folder_path):
    """Compute statistics and line profiles for the cc peaks in a score map.

    For each specified template index, this function:
    1. Identifies the peak location and value.

    2. Saves the 1D line profiles through the peak along x, y, and z axes. Saved as
        '<output_base>_peak_line_profiles.csv' in the output folder.

    3. Computes the drop in score from the peak to its immediate neighbors along
        each axis.

    4. Calculates mean, median, and variance of scores in small areas where the peaks
        are centered (spherical of radius from 1 to 5 px).

    5. Updates the template list CSV file with the computed statistics:
        * "Peak value"
        * "Drop x", "Drop y", "Drop z"
        * "Peak x", "Peak y", "Peak z"
        * "Mean r", "Median r", "Var r" for r = 1..5

    Parameters
    ----------
    template_list : str
        Path to the CSV template list file.
    indices : list of int
        List of row indices in `template_list` to process.
    parent_folder_path : str
        Base directory containing structure folders.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")

        # only look at central area in scores map
        cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=10)
        masked_map = scores_map * cc_mask

        # find the coords and values of highest cc values
        peak_center, peak_value, _ = tmana.create_starting_parameters_2D(masked_map)

        # for each peak, find how the cc value changes in each x,y,z direction
        _, _, line_profiles = tmana.create_starting_parameters_1D(scores_map)

        temp_df.at[i, "Peak value"] = peak_value

        line_pd = pd.DataFrame(data=line_profiles, columns=["x", "y", "z"])
        line_pd.to_csv(output_folder + output_base + "_peak_line_profiles.csv")

        # for each axis, find out the value diff between peaks and surrounding pixels
        for j, dim in enumerate(["x", "y", "z"]):
            peak_difference = (
                peak_value - (line_profiles[peak_center[j] - 1, j] + line_profiles[peak_center[j] + 1, j]) / 2.0
            )
            temp_df.at[i, "Drop " + dim] = peak_difference
            temp_df.at[i, "Peak " + dim] = peak_center[j]  # peak coord in that axis

        # look at only the peak area (with 5 different sizes), and calculate some stats
        for r in range(1, 6):
            cc_mask = cryomask.spherical_mask(np.asarray(scores_map.shape), radius=r, center=peak_center)
            masked_map = scores_map[np.nonzero(scores_map * cc_mask)]
            temp_df.at[i, "Mean " + str(r)] = np.mean(masked_map)
            temp_df.at[i, "Median " + str(r)] = np.median(masked_map)
            temp_df.at[i, "Var " + str(r)] = np.var(masked_map)

        temp_df.to_csv(template_list)


def analyze_rotations(
    tomogram,
    template,
    template_mask,
    input_angles,
    wedge_mask_tomo=None,
    wedge_mask_tmpl=None,
    output_file=None,
    cc_radius=3,
    angular_offset=None,
    starting_angle=None,
    c_symmetry=1,
    angles_order="zxz",
):
    """Perform template matching between a tomogram and a reference template.

    This function rotates a reference template through a set of input Euler angles,
    computes the Fast Local Cross-Correlation Function (FLCF) between each rotated
    template and the target tomogram, and collects statistics for each rotation.
    Optionally, wedge masks can be applied to account for missing wedge artifacts,
    and a spherical correlation mask can be applied for localized measurements.

    Parameters
    ----------
    tomogram : str or numpy.ndarray
        File path (.mrc or .em) to or a numpy array of a tomogram of interest.
    template : str or numpy.ndarray
        File path to or an array of the reference template map.
    template_mask : str or numpy.ndarray
        File path to or an array of the binary mask for the template.
    input_angles : str or numpy.ndarray
        File path to an angle list or a numpy array of euler angles.
    wedge_mask_tomo : str or numpy.ndarray, optional
        File path to or a numpy array of the wedge mask for the tomogram.
    wedge_mask_tmpl : str or numpy.ndarray, optional
        File path to or an array of the wedge mask for the template.
    output_file : str, optional
        Base path for saving output CSV and EM maps. If None, results are not
        written to disk.
    cc_radius : int, optional
        Radius (in voxels) of the spherical mask applied to compute masked
        cross-correlation. Default is 3.
    angular_offset : float or array-like of shape (3,), optional
        Euler angles (degrees) to offset all input angles before matching.
    starting_angle : float or array-like of shape (3,), optional
        Euler angles (degrees) representing the reference orientation from which angular
        distances will be calculated. Default is (0, 0, 0).
    c_symmetry : int, optional
        C symmetry of the structure. Default is 1 (no symmetry).
    angles_order : {"zxz", ...}, optional
        Euler angle convention used for rotations. Default is "zxz".

    Returns
    -------
    res_table : pandas.DataFrame
        Table containing per-rotation statistics, including:
        - ang_dist: Angular distance from starting_angle (degrees)
        - cone_dist: Cone angle difference (degrees)
        - inplane_dist: In-plane rotation difference (degrees)
        - common_voxels: Overlap between mask and rotated mask
        - ccc: Maximum cross-correlation coefficient
        - ccc_masked: Maximum masked CCC within cc_radius
        - z_score: Maximum z-score across the full cc map
        - z_score_masked: Maximum z-score within the spherical mask
    final_ccc_map : ndarray
        3D array of the maximum CCC values observed across all rotations.
    final_angles_map : ndarray
        3D array of rotation indices (of angle rotation list) corresponding to the
        highest CCC at each voxel.
    final_ccc_map_masked : ndarray
        Masked CCC map showing only the central area of final_ccc_map.

    Notes
    -----
    - If the template and tomogram sizes differ, the smaller map is padded to match.
    - The function keeps track of the highest CCC per voxel across all rotations.
    """

    angles = ioutils.rot_angles_load(input_angles, angles_order)
    # angles = angles[0:4,:]

    if starting_angle is None:
        starting_angle = np.asarray([0, 0, 0])

    # adds the starting angle to every euler angle
    if np.any(starting_angle):
        rots = srot.from_euler("zxz", angles=angles, degrees=True)
        add_rot = srot.from_euler("zxz", angles=starting_angle, degrees=True)
        new_rot = rots * add_rot
        angles = new_rot.as_euler("zxz", degrees=True)

    if angular_offset is not None and np.any(angular_offset):
        rots = srot.from_euler("zxz", angles=angles, degrees=True)
        add_rot = srot.from_euler("zxz", angles=angular_offset, degrees=True)
        new_rot = rots * add_rot
        angles = new_rot.as_euler("zxz", degrees=True)

    # angles = angles[0:20,:]

    tomo = cryomap.read(tomogram)
    tmpl = cryomap.read(template)
    mask = cryomap.read(template_mask)

    # pad the maps so they are the same size
    # might be faster to pad tmpl2 and mask after the rotation, but less readible
    if np.any(tomo.shape < tmpl.shape):
        tomo = cryomap.pad(tomo, tmpl.shape)
        output_size = tmpl.shape
    elif np.any(tomo.shape > tmpl.shape):
        tmpl = cryomap.pad(tmpl, tomo.shape)
        mask = cryomap.pad(mask, tomo.shape)
        output_size = tomo.shape
    else:
        output_size = tomo.shape

    # a small central area where the ccc is relevant
    cc_mask = cryomask.spherical_mask(np.array(output_size), radius=cc_radius).astype(np.single)

    # calculates the complex conjugate of fourier transformed tomogram
    if wedge_mask_tomo is not None:
        wedge_tomo = cryomap.read(wedge_mask_tomo)
        conj_target, conj_target_sq = cryomap.calculate_conjugates(tomo, wedge_tomo)
    else:
        conj_target, conj_target_sq = cryomap.calculate_conjugates(tomo)

    if wedge_mask_tmpl is not None:
        wedge_tmpl = cryomap.read(wedge_mask_tmpl)

    # make an array of starting angles the same shape as angles
    starting_angles = np.tile(starting_angle, (angles.shape[0], 1))

    # calculates angular/cone/inplane distances
    ang_dist, cone, inplane = geom.compare_rotations(starting_angles, angles, c_symmetry=c_symmetry)

    res_table = pd.DataFrame(
        columns=[
            "ang_dist",
            "cone_dist",
            "inplane_dist",
            "common_voxels",
            "ccc",
            "ccc_masked",
            "z_score",
            "z_score_masked",
        ],
        dtype=float,
    )

    # init output maps
    final_ccc_map = np.full(output_size, -1)
    final_angles_map = np.full(output_size, -1)

    for i, a in enumerate(angles):

        # rotate the template and the mask
        rot_ref = cryomap.rotate(tmpl, rotation_angles=a, spline_order=1).astype(np.single)
        rot_mask = cryomap.rotate(mask, rotation_angles=a, spline_order=1).astype(np.single)

        rot_mask[rot_mask < 0.001] = 0.0  # Cutoff values for weird interpolated values
        rot_mask[rot_mask > 1.000] = 1.0  # Cutoff values

        # mask out missing wedge for template
        if wedge_mask_tmpl is not None:
            rot_ref = np.fft.ifftn(np.fft.fftn(rot_ref) * wedge_tmpl).real

        norm_ref = cryomap.normalize_under_mask(rot_ref, rot_mask)
        masked_ref = norm_ref * rot_mask

        # calculates fast local correlation coefficient
        cc_map = cryomap.calculate_flcf(masked_ref, rot_mask, conj_target=conj_target, conj_target_sq=conj_target_sq)
        z_score = (cc_map - np.mean(cc_map)) / np.std(cc_map)

        # find the indices where the current ccc is bigger than ccc in init/saved cc map
        max_idx = np.argmax((final_ccc_map, cc_map), 0).astype(bool)

        # overwrite in the same position with the bigger ccc (corresponding ang dist)
        final_ccc_map = np.maximum(final_ccc_map, cc_map)
        final_angles_map[max_idx] = i + 1

        masked_map = cc_map * cc_mask
        z_score_masked = z_score * cc_mask

        # update the table with a new row
        res_table.loc[len(res_table)] = [
            ang_dist[i],
            cone[i],
            inplane[i],
            cryomask.mask_overlap(mask, rot_mask),
            np.max(cc_map),
            np.max(masked_map),
            np.max(z_score),
            np.max(z_score_masked),
        ]

    # make sure the cc map value stays between 0 and 1
    final_ccc_map = np.clip(final_ccc_map, 0.0, 1.0)
    final_ccc_map_masked = final_ccc_map * cc_mask

    if output_file is not None:
        res_table.to_csv(output_file + ".csv", index=False)
        cryomap.write(file_name=output_file + "_scores.em", data_to_write=final_ccc_map, data_type=np.single)
        # cryomap.write(file_name=output_file + '_scores_masked.em',
        #               data_to_write = final_ccc_map_masked,
        #               data_type=np.single)
        cryomap.write(file_name=output_file + "_angles.em", data_to_write=final_angles_map, data_type=np.single)

    return res_table, final_ccc_map, final_angles_map, final_ccc_map_masked


def run_analysis(template_list, indices, angle_list_path, wedge_path, parent_folder_path, cc_radius_tol=10):
    """Run peak analysis based on a list with parameters and save results.

    This function iterates over the provided `indices` of a template list CSV,
    loads the corresponding tomogram, template, and mask files, and then calls
    `analyze_rotations` to perform rotation-based cross-correlation analysis.
    The results (score maps, angle maps, CSV stats) are written to an output
    folder for each index. It also generates angular distance maps for the
    resulting angle maps. The function updates the CSV in-place to record
    progress, ensuring partial results are saved in case of interruption.

    Parameters
    ----------
    template_list : str or path-like
        Path to a CSV file containing info about peak analysis to perform. The CSV
        must include at least the following columns:
        - Structure
        - Template
        - Mask
        - Angles
        - Compare (compare method; "tmpl": tmpl vs. tmpl, "subtomo": subtomo vs. sutomo, or else)
        - Tomo map (i.e. subtomo)
        - Tomogram
        - Apply wedge (bool)
        - Boxsize
        - Binning
        - Phi, Theta, Psi (starting Euler angles in degrees)
        - Apply angular offset (bool)
        - Degrees (search angle increment / offset magnitude)
        - Symmetry (C symmetry)
    indices : sequence of int
        List or array of row indices (0-based) in `template_list` to process.
    angle_list_path : str or path-like
        Base directory path where the angle list files are stored.
        The angle file name from the CSV's "Angles" column is appended to this path.
    wedge_path : str or path-like
        Base directory containing wedge mask files. Used only if "Apply wedge"
        is set for the current row.
    parent_folder_path : str or path-like
        Root folder containing all structure subfolders, templates, tomograms,
        and masks referenced in `template_list`.
    cc_radius_tol : float, optional
        Radius (in voxels) of the spherical mask used for computing local
        cross-correlation scores in `analyze_rotations`. Default is 10.

    Notes
    -------
    - Writes output files for each processed index:
        * '<output_base>_scores.em' (cross-correlation coefficient map)
        * '<output_base>_angles.em' (best-angle index map)
        * CSV file with per-angle statistics
    - Writes angular distance maps via 'tmana.create_angular_distance_maps'
    - Updates `template_list` CSV in place:
        * "Output folder" set for each processed index
        * "Done" flag set to True
    - Creates any necessary output directories.
    - For rows with `"Compare" == "tmpl"`, the tomogram is the same as the template.
    - For `"Compare" == "subtomo"`, a tomogram is loaded from the specified file,
      and wedge masks may be applied if `"Apply wedge"` is true.
    - `starting_angle` is read directly from `"Phi"`, `"Theta"`, `"Psi"` columns.
    - If `"Apply angular offset"` is true, `angular_offset` is set to half of
      `Degrees` for all three Euler components; otherwise it is [0, 0, 0].
    - Symmetry (`c_symmetry`) is passed to `analyze_rotations` to account for
      cyclic symmetry in angular distance calculations.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        tmpl_name = temp_df.at[i, "Template"]
        tmpl_folder = create_structure_path(parent_folder_path, structure_name)
        template = create_em_path(parent_folder_path, structure_name, tmpl_name)
        mask = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"])
        angle_list = angle_list_path + temp_df.at[i, "Angles"]

        wedge_tomo = None
        wedge_tmpl = None

        if temp_df.at[i, "Compare"] == "tmpl":
            tomo = template

        elif temp_df.at[i, "Compare"] == "subtomo":
            tomo = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tomo map"])
            tomo_number = re.findall(r"\d+", temp_df.at[i, "Tomogram"])[0]

            if temp_df.at[i, "Apply wedge"]:
                wedge_tomo, wedge_tmpl = create_wedge_names(
                    wedge_path, tomo_number, temp_df.at[i, "Boxsize"], temp_df.at[i, "Binning"]
                )
        else:
            tomo = create_em_path(parent_folder_path, temp_df.at[i, "Compare"], temp_df.at[i, "Tomo map"])

        starting_angle = temp_df.loc[[i], ["Phi", "Theta", "Psi"]].to_numpy()

        if temp_df.at[i, "Apply angular offset"]:
            angular_offset = np.full((3,), temp_df.at[i, "Degrees"] / 2.0)
        else:
            angular_offset = np.asarray([0, 0, 0])

        c_symmetry = temp_df.at[i, "Symmetry"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, i)

        temp_df.at[i, "Output folder"] = create_output_folder_name(i)

        Path(output_folder).mkdir(parents=True, exist_ok=True)

        _ = analyze_rotations(
            tomogram=tomo,
            template=template,
            template_mask=mask,
            input_angles=angle_list,
            wedge_mask_tomo=wedge_tomo,
            wedge_mask_tmpl=wedge_tmpl,
            output_file=output_folder + "/" + output_base,
            angular_offset=angular_offset,
            starting_angle=starting_angle,
            cc_radius=cc_radius_tol,
            c_symmetry=c_symmetry,
        )[0]

        angles_map = output_folder + "/" + output_base + "_angles.em"
        _, _, _ = tmana.create_angular_distance_maps(angles_map, angle_list, write_out_maps=True)

        temp_df.at[i, "Done"] = True
        temp_df.to_csv(template_list)  # save what was finished in case of a crush


def run_angle_analysis(
    template_list, indices, wedge_path, parent_folder_path, angular_range=359, write_output=False, cc_radius_tol=10
):
    """Perform a gradual rotation angular peak analysis.

    This function systematically evaluates the effect of varying Euler angles
    (3 kinds of rotations: full angular distance, cone rotation, and in-plane rotation)
    on the cross-correlation between a tomogram (or template) and a reference template.
    It iterates through a specified angular range 1 deg by 1 deg, computes
    correlation metrics, and optionally saves detailed analysis results and histograms.

    Parameters
    ----------
    template_list : str
        Path to a CSV file containing metadata and file paths for templates,
        tomograms, masks, and analysis parameters.
    indices : list of int
        List of row indices in `template_list` to process.
    wedge_path : str
        Directory path containing wedge mask files.
    parent_folder_path : str
        Root directory containing structure and template data.
    angular_range : int, optional
        Number of degrees to test in the rotation range. Default is 359, meaning
        all integer angles from 0 to 358 will be analyzed. 359 makes a full circle.
    write_output : bool, optional
        If True, saves the computed angular analysis results and histograms
        as CSV files in the corresponding output directory. Default is False.
    cc_radius_tol : int, optional
        Radius (in voxels) of the spherical mask applied to the cross-correlation
        map when evaluating local correlation. Default is 10.

    Notes
    -----
    - A histogram of CCC values across the angular range is also computed for each
      rotation type.

    - Output files are:

      - '<output_base>_gradual_angles_analysis.csv': Table containing detailed
        rotation metrics for all tested angles and rotation types

      - '<output_base>_gradual_angles_histograms.csv': Histograms of CCC values for
        each rotation type
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        tmpl_name = temp_df.at[i, "Template"]
        tmpl_folder = create_structure_path(parent_folder_path, structure_name)
        template = create_em_path(parent_folder_path, structure_name, tmpl_name)
        mask = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Mask"])

        wedge_tomo = None
        wedge_tmpl = None

        if temp_df.at[i, "Compare"] == "tmpl":
            tomo = template

        elif temp_df.at[i, "Compare"] == "subtomo":
            tomo = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tomo map"])
            tomo_number = re.findall(r"\d+", temp_df.at[i, "Tomogram"])[0]

            if temp_df.at[i, "Apply wedge"]:
                wedge_tomo, wedge_tmpl = create_wedge_names(
                    wedge_path, tomo_number, temp_df.at[i, "Boxsize"], temp_df.at[i, "Binning"]
                )
        else:
            tomo = create_em_path(parent_folder_path, temp_df.at[i, "Compare"], temp_df.at[i, "Tomo map"])

        starting_angle = temp_df.loc[[i], ["Phi", "Theta", "Psi"]].to_numpy()
        c_symmetry = temp_df.at[i, "Symmetry"]

        results = np.zeros((angular_range, 8, 3))

        n_bins = 100
        final_hist = np.zeros((n_bins, 3))

        # loop over each int deg in angular range
        for a in range(angular_range):
            angles = np.full((3, 3), a)
            angles[1, 0] = 0  # only cone rotation
            angles[2, 1:] = 0  # only inplane rotation
            # this is what "angles" contain: [[a, a, a],[0, a, a],[a, 0, 0]]
            # translates to [[both cone and inplane],[cone],[inplane]]

            for j in range(3):
                res_df, cc_map, _, _ = analyze_rotations(
                    tomogram=tomo,
                    template=template,
                    template_mask=mask,
                    input_angles=angles[j, :].reshape(1, 3),
                    wedge_mask_tomo=wedge_tomo,
                    wedge_mask_tmpl=wedge_tmpl,
                    output_file=None,
                    starting_angle=starting_angle,
                    cc_radius=cc_radius_tol,
                    c_symmetry=c_symmetry,
                )
                results[a, :, j] = res_df.values
                hist, _ = np.histogram(cc_map, bins=n_bins, range=(0.0, 1.0))

                # do cumulative final_hist for each type of rotation
                final_hist[:, j] += hist

        # save results separately for each type of rotation
        ang_dist = pd.DataFrame(
            data=results[:, :, 0],
            columns=[
                "ang_dist",
                "cone_dist",
                "inplane_dist",
                "common_voxels",
                "ccc",
                "ccc_masked",
                "z_score",
                "z_score_masked",
            ],
        )
        ang_cone = pd.DataFrame(
            data=results[:, :, 1],
            columns=[
                "cone_ang_dist",
                "cone_cone_dist",
                "cone_inplane_dist",
                "cone_common_voxels",
                "cone_ccc",
                "cone_ccc_masked",
                "cone_z_score",
                "cone_z_score_masked",
            ],
        )
        ang_inplane = pd.DataFrame(
            data=results[:, :, 2],
            columns=[
                "inplane_ang_dist",
                "inplane_cone_dist",
                "inplane_inplane_dist",
                "inplane_common_voxels",
                "inplane_ccc",
                "inplane_ccc_masked",
                "inplane_z_score",
                "inplane_z_score_masked",
            ],
        )
        final_df = pd.concat([ang_dist, ang_cone, ang_inplane], axis=1)

        if write_output:
            output_base = create_output_folder_path(
                parent_folder_path, structure_name, temp_df.at[i, "Output folder"]
            ) + create_output_base_name(i)
            final_df.to_csv(output_base + "_gradual_angles_analysis.csv")
            hist_df = pd.DataFrame(data=final_hist, columns=["ang_dist", "cone_dist", "inplane_dist"])
            hist_df.to_csv(output_base + "_gradual_angles_histograms.csv")


def create_summary_pdf(template_list, indices, parent_folder_path):
    """Generate a detailed summary PDF for a set of peak analysis results.

    This function reads metadata from a CSV file (`template_list`), retrieves
    volumetric map data, analysis results, and visualization slices for the
    specified `indices`, and compiles them into a structured multi-panel PDF
    report. Each report includes:

    - Template and processing parameters (symmetry, wedge application, voxel size, etc.)
    - Peak detection information (location, value, line profiles)
    - Distance map statistics and solidity/volume coverage measures
    - Scatter plots, histograms, and gradual rotation CCC analysis (if available)
    - Cross-sectional heatmaps of masks, score maps, and angular distance maps

    The output is saved as 'id_<index>_summary.pdf' in the corresponding
    output folder for each index.

    Parameters
    ----------
    template_list : str
        Path to the CSV file containing metadata for all templates and analyses.
    indices : list of int
        List of row indices from `template_list` to process.
    parent_folder_path : str
        Base directory containing the structure folders, output folders, and map files.

    Notes
    -----
    - If the "Done" column is False for a given index, that entry is skipped.
    - Gradual rotation histogram and CCC analysis are included if the corresponding
      '_gradual_angles_histograms.csv' and '_gradual_angles_analysis.csv' are found.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]
        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])

        if temp_df.at[i, "Compare"] == "tmpl":
            title_end = "self"
        elif temp_df.at[i, "Compare"] == "subtomo":
            title_end = "subtomo (" + temp_df.at[i, "Tomo map"] + ")"
        else:
            title_end = temp_df.at[i, "Compare"] + " (" + temp_df.at[i, "Tomo map"] + ")"

        figure_title = structure_name + " (" + temp_df.at[i, "Template"] + ") matched with " + title_end

        extenstion_list = ["_scores.em", "_angles_dist_all.em", "_angles_dist_normals.em", "_angles_dist_inplane.em"]

        tmpl_info = [
            "Symmetry",
            "Apply wedge",
            "Degrees",
            "Apply angular offset",
            "Binning",
            "Pixelsize",
            "Boxsize",
            "Voxels",
            "Voxels TM",
            "Solidity",
        ]

        temp_df = temp_df.round(decimals=4)
        tmpl_dict = pd.DataFrame(temp_df.loc[temp_df.index[i], tmpl_info]).to_dict()
        tmpl_dict = tmpl_dict.get(i)

        peak_info = ["Peak value"]
        peak_dict = pd.DataFrame(temp_df.loc[temp_df.index[i], peak_info]).to_dict()
        peak_dict = peak_dict.get(i)

        dist_names = ["dist_all", "dist_normals", "dist_inplane"]
        dist_dict = {}

        values_temp = np.zeros((3,))
        peak_center = np.zeros((3,))
        bb_dim = np.zeros((3,))
        peak_drop = np.zeros((3,))
        for d, dim in enumerate(["x", "y", "z"]):
            peak_center[d] = temp_df.at[i, "Peak " + dim]
            bb_dim[d] = temp_df.at[i, "Dim " + dim]
            peak_drop[d] = temp_df.at[i, "Drop " + dim]

        peak_dict["Peak center"] = peak_center
        tmpl_dict["Dimensions"] = bb_dim
        peak_dict["Drop"] = peak_drop

        dist_vc = np.zeros((3,))
        dist_sol = np.zeros((3,))
        dist_vco = np.zeros((3,))
        for d, dname in enumerate(dist_names):
            dist_vc[d] = temp_df.at[i, "VC " + dname]
            dist_sol[d] = temp_df.at[i, "Solidity " + dname]
            dist_vco[d] = temp_df.at[i, "VCO " + dname]

        # dist_dict['Dummy'] = 1
        dist_dict["Dist maps Solidity"] = dist_sol
        dist_dict["Dist maps VC"] = dist_vc
        dist_dict["Dist maps VC open"] = dist_vco

        for d, dname in enumerate(dist_names):
            for j, dim in enumerate(["x", "y", "z"]):
                values_temp[j] = temp_df.at[i, "O " + dname + " " + dim]

            dist_dict["Open " + dname] = values_temp.copy()

        values_temp = np.zeros((5,))

        for sts in ["Mean", "Median", "Var"]:
            for r in range(1, 6):
                values_temp[r - 1] = temp_df.at[i, sts + " " + str(r)]
            peak_dict[sts] = values_temp.copy()

        dicts = []
        with np.printoptions(
            precision=4,
            suppress=True,
        ):
            dicts.append([[k, str(v)] for k, v in tmpl_dict.items()])
            dicts.append([[k, str(v)] for k, v in peak_dict.items()])
            dicts.append([[k, str(v)] for k, v in dist_dict.items()])

        # extract the middle cross sections of tight mask
        cross_slices = []
        tight_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))
        cross_slices.append(cryomap.get_cross_slices(tight_mask))

        # also extract cross sections of output maps where the cc peak centers are
        for m, ext_name in enumerate(extenstion_list):
            input_map = cryomap.read(output_folder + output_base + ext_name)

            if m == 0:
                cross_slices.append(cryomap.get_cross_slices(input_map, slice_numbers=peak_center, axis=[0, 1, 2]))

            cross_slices.append(
                cryomap.get_cross_slices(input_map, slice_half_dim=5, slice_numbers=peak_center, axis=[0, 1, 2])
            )

        grid_rows_n = 2
        unit_size = 5
        grid_row_ratio = [1.6 * unit_size, 6 * unit_size]

        # check if file for histogram analysis exists or not
        hist_file = output_folder + output_base + "_gradual_angles_histograms.csv"
        add_hist = False
        last_row = 1

        # change the figure layout slightly if hist file exists
        if os.path.isfile(hist_file):
            grid_rows_n += 1
            add_hist = True
            last_row = 2
            grid_row_ratio = [1.6 * unit_size, 0.8 * unit_size, 6 * unit_size]

        fig_height = sum(grid_row_ratio) + 0.4
        widths = [unit_size, unit_size, unit_size, unit_size * 0.05]

        fig = plt.figure(layout="constrained", figsize=(sum(widths), fig_height))
        fig.suptitle(figure_title, fontsize=16, y=1.008)

        grid_base = gridspec.GridSpec(grid_rows_n, 1, figure=fig, height_ratios=grid_row_ratio)
        grid_rows = [gridspec.GridSpecFromSubplotSpec(2, 3, subplot_spec=grid_base[0])]

        if add_hist:
            grid_rows.append(gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=grid_base[1]))

        grid_rows.append(
            gridspec.GridSpecFromSubplotSpec(
                6,
                4,
                subplot_spec=grid_base[last_row],
                height_ratios=[unit_size, unit_size, unit_size, unit_size, unit_size, unit_size],
                width_ratios=widths,
            )
        )

        table_plots = []
        for j in range(3):
            table_plots.append(fig.add_subplot(grid_rows[0][0, j]))

        col_width = [[0.4, 0.6], [0.3, 0.7], [0.4, 0.6]]
        for tbl, dc, cw in zip(table_plots, dicts, col_width):
            tbl_h = tbl.table(colWidths=cw, cellText=dc, bbox=[0, 0, 1, 1])
            tbl_h.auto_set_font_size(False)
            tbl_h.set_fontsize(10)

        rot_info = pd.read_csv(output_folder + output_base + ".csv", index_col=0)
        line_profiles = pd.read_csv(output_folder + output_base + "_peak_line_profiles.csv", index_col=0)

        if "Tight mask overlap" not in rot_info.columns:
            print(i)
            continue

        ad_plt = sns.scatterplot(
            ax=fig.add_subplot(grid_rows[0][1, 0]),
            data=rot_info,
            x="Tight mask overlap",
            y="ccc_masked",
            linewidth=0,
            s=5,
        )
        ad_plt.set(ylabel="CCC", xlabel="Tight mask overlap (in voxels)")
        ad_plt = sns.scatterplot(
            ax=fig.add_subplot(grid_rows[0][1, 1]), data=rot_info, x="ang_dist", y="ccc_masked", linewidth=0, s=5
        )
        ad_plt.set(ylabel=None, xlabel="Angular distance (in degrees)")
        ad_plt = sns.lineplot(ax=fig.add_subplot(grid_rows[0][1, 2]), data=line_profiles[["x", "y", "z"]])
        ad_plt.set(ylabel=None, xlabel="Position (in voxels)")

        if add_hist:
            hist_info = pd.read_csv(hist_file, index_col=0)
            hist_plt = fig.add_subplot(grid_rows[1][0, 0])
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 1.0, num=100), y="ang_dist")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 1.0, num=100), y="cone_dist")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 1.0, num=100), y="inplane_dist")
            hist_plt.set(ylim=(0, 250), ylabel="Number of CCC values (bin size 0.1)", xlabel="CCC")

            hist_info = pd.read_csv(output_folder + output_base + "_gradual_angles_analysis.csv", index_col=0)
            hist_plt = fig.add_subplot(grid_rows[1][0, 1])
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 359.0, num=359), y="ccc_masked")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 359.0, num=359), y="cone_ccc_masked")
            sns.lineplot(ax=hist_plt, data=hist_info, x=np.linspace(0.0, 359.0, num=359), y="inplane_ccc_masked")
            hist_plt.set(ylabel="CCC", xlabel="Rotation (in degrees)")

        for c, slice in enumerate(cross_slices):
            if c <= 1:
                use_annot = False
            else:
                use_annot = True

            f_fmt = ".2f"
            if c == 0:
                data_max = 1.0
                c_scheme = "gray"
            elif c < 3:
                data_max = temp_df.at[i, "Peak value"]
                c_scheme = "viridis"
            else:
                data_max = 180
                c_scheme = "cividis"
                f_fmt = ".1f"

            hm1 = fig.add_subplot(grid_rows[last_row][c, 0])
            hm2 = fig.add_subplot(grid_rows[last_row][c, 1])
            hm3 = fig.add_subplot(grid_rows[last_row][c, 2])
            cb = fig.add_subplot(grid_rows[last_row][c, 3])

            sns.heatmap(
                ax=hm1,
                data=np.flipud(slice[0].T),
                annot=use_annot,
                fmt=f_fmt,
                square=True,
                cmap=c_scheme,
                yticklabels=False,
                xticklabels=False,
                vmin=0,
                vmax=data_max,
                cbar_ax=cb,
            )
            sns.heatmap(
                ax=hm2,
                data=np.flipud(slice[1].T),
                annot=use_annot,
                fmt=f_fmt,
                square=True,
                cmap=c_scheme,
                yticklabels=False,
                xticklabels=False,
                vmin=0,
                vmax=data_max,
                cbar=False,
            )
            sns.heatmap(
                ax=hm3,
                data=np.flipud(slice[2].T),
                annot=use_annot,
                fmt=f_fmt,
                square=True,
                cmap=c_scheme,
                yticklabels=False,
                xticklabels=False,
                vmin=0,
                vmax=data_max,
                cbar=False,
            )

        # plt.tight_layout() # or layout = "constrained" in figure
        plt.savefig(output_folder + output_base + "_summary.pdf", transparent=True, bbox_inches="tight")
        plt.close()


##########################################################################################################################
###### Following functions are not used in the current analysis but might come handy later ###############################
##########################################################################################################################


# Check what kind of descriptors skimage can offer
def get_shape_stats(template_list, indices, shape_type, parent_folder_path):
    """Compute and save shape statistics for specific shapes in a template list.

    This function reads path from a CSV file, loads corresponding tight masks, labels
    connected components, computes geometric and morphological
    properties for each labeled region, and saves the results to CSV files.

    Parameters
    ----------
    template_list : str or path-like
        Path to the CSV file containing metadata for all templates and analyses.
    indices : array_like of int
        Rows in `template_list` for which statistics should be computed.
    shape_type : str
        A descriptive label for the type of shape used for analysis. This string
        is appended to the output CSV filename.
    parent_folder_path : str or path-like
        Path to the root directory containing the structure and mask files.

    Notes
    -----
    The following region properties are computed for each labeled region:
        - 'label' : integer label ID
        - 'area' : voxel count
        - 'area_bbox' : bounding box volume
        - 'area_convex' : convex hull volume
        - 'equivalent_diameter_area' : diameter of a sphere with same volume
        - 'euler_number' : topological Euler characteristic
        - 'feret_diameter_max' : maximum caliper distance
        - 'inertia_tensor' : 3×3 inertia tensor matrix
        - 'solidity' : ratio of area to convex hull area

    Output files are named in the format:
        '<structure_path>/<output_folder>/id_<index>_shape_stats_<shape_type>.csv'
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]
        sharp_mask = cryomap.read(create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tight mask"]))

        mask_label = measure.label(sharp_mask, connectivity=1)
        mask_stats = pd.DataFrame(
            measure.regionprops_table(
                mask_label,
                properties=(
                    "label",
                    "area",
                    "area_bbox",
                    "area_convex",
                    "equivalent_diameter_area",
                    "euler_number",
                    "feret_diameter_max",
                    "inertia_tensor",
                    "solidity",
                ),
            )
        )

        output_base = (
            create_structure_path(parent_folder_path, structure_name)
            + temp_df.at[i, "Output folder"]
            + "/id_"
            + str(i)
            + "_shape_stats_"
        )
        mask_stats.to_csv(output_base + shape_type + ".csv")


def plot_scores_and_peaks(peak_files, plot_title=None, output_file=None):
    """
    Plot heatmaps of peak cross-sections for multiple peak-related data arrays.

    This function visualizes peak data from a list of files or arrays by generating
    heatmaps for three orthogonal 2D slices (X, Y, Z) centered at the main peak.
    All peaks are normalized to the same value range based on the first peak file.

    Parameters
    ----------
    peak_files : list of array_like or list of str
        List of 3D arrays or file paths containing peak-related data.
        Each entry is processed using
        `tmana.create_starting_parameters_2D` to extract peak-centered slices.
    plot_title : str, optional
        Title for the entire figure. If None, no title is added.
    output_file : str, optional
        Path to save the figure as an image file. If None, the figure is not saved.
    """

    n_rows = len(peak_files)
    row_size = 4 * n_rows
    fig, axs = plt.subplots(n_rows, 4, figsize=(row_size, 100), gridspec_kw={"width_ratios": [20, 20, 20, 1]})

    if plot_title is not None:
        fig.suptitle(plot_title, fontsize=28, y=1.005)

    peak_center, peak_height, _ = tmana.create_starting_parameters_2D(peak_files[0])

    for i, p in enumerate(peak_files):
        _, _, peaks = tmana.create_starting_parameters_2D(p, peak_center=peak_center)
        data_min = np.amin(p)

        sns.heatmap(
            ax=axs[i][0],
            data=np.flipud(peaks[:, :, 0].T),
            square=True,
            cmap="viridis",
            annot=False,
            yticklabels=False,
            xticklabels=False,
            vmin=data_min,
            vmax=peak_height,
            cbar_ax=axs[i][3],
        )
        sns.heatmap(
            ax=axs[i][1],
            data=np.flipud(peaks[:, :, 1].T),
            square=True,
            cmap="viridis",
            annot=False,
            yticklabels=False,
            xticklabels=False,
            vmin=data_min,
            vmax=peak_height,
            cbar=False,
        )
        sns.heatmap(
            ax=axs[i][2],
            data=np.flipud(peaks[:, :, 2].T),
            square=True,
            cmap="viridis",
            annot=False,
            yticklabels=False,
            xticklabels=False,
            vmin=data_min,
            vmax=peak_height,
            cbar=False,
        )

    plt.tight_layout()

    if output_file is not None:
        plt.savefig(output_file, transparent=True, bbox_inches="tight")


def compute_peak_shapes(template_list, indices, parent_folder_path):
    """
    Compute and record peak shape statistics from scores maps for selected structures.

    This function processes score maps for the given structures, evaluates the peak
    shapes using multiple thresholding methods, stores the results in the template
    list, and generates a visualization of the peaks and thresholds.

    Parameters
    ----------
    template_list : str
        Path to the CSV file containing metadata for all templates and analyses.
    indices : list of int
        List of row indices in the template list to process.
    parent_folder_path : str
        Path to the root directory containing structure data and score maps.

    Notes
    -----
    - Only rows marked as 'Done == True' in the template list are processed.
    - Structures named '"membrane"' are skipped.
    - peak shapes are measured using `tmana.evaluate_scores_map` with
      three thresholding methods:
      triangle, Gaussian, and hard threshold.
    - The three principal dimensions (x, y, z) of each peak shape are stored in
      the template list under the columns:
      'TP x/y/z', 'GP x/y/z', and 'HP x/y/z' (triangle, Gaussian, hard).
    - The maximum peak value is stored under 'Peak value'.
    - A PNG plot visualizing the scores and peaks is saved in the corresponding
      output folder under the name 'peaks.png'.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        if temp_df.at[i, "Structure"] == "membrane":
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_ouptut_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        scores_map = cryomap.read(output_folder + output_base + "_scores.em")

        t_map, tp_shape, peak_value, t_th_map, t_surf = tmana.evaluate_scores_map(
            scores_map, label_type="ellipsoid", threshold_type="triangle"
        )
        g_map, gp_shape, _, g_th_map, g_surf = tmana.evaluate_scores_map(
            scores_map, label_type="ellipsoid", threshold_type="gauss"
        )
        h_map, hp_shape, _, h_th_map, h_surf = tmana.evaluate_scores_map(
            scores_map, label_type="ellipsoid", threshold_type="hard"
        )

        for t, p in enumerate(["x", "y", "z"]):
            temp_df.at[i, "TP " + p] = np.round(tp_shape[t], 3)  # triangle
            temp_df.at[i, "GP " + p] = np.round(gp_shape[t], 3)  # gaussian
            temp_df.at[i, "HP " + p] = np.round(hp_shape[t], 3)  # hard

        temp_df.at[i, "Peak value"] = peak_value

        temp_df.to_csv(template_list)  # save what was finished in case of a crush

        plot_scores_and_peaks(
            [scores_map, t_th_map, t_surf, t_map, g_th_map, g_surf, g_map, h_th_map, h_surf, h_map],
            plot_title=structure_name + " id" + str(i),
            output_file=output_folder + "peaks.png",
        )


## Function to change the output folder base name
def rename_folders(template_list, indices, parent_folder_path):
    """
    Rename output folders for specified dataset entries and update metadata.

    This function updates the output folder names for given indices in a template
    list CSV file. Each folder is renamed to a new standardized name generated
    from its index, and the corresponding entry in the CSV file is updated to
    reflect the change.

    Parameters
    ----------
    template_list : str
        Path to the CSV file containing metadata for all templates and analyses.
    indices : iterable of int
        List or array of row indices in `template_list` to process.
    parent_folder_path : str
        Base directory containing all structure folders.

    Notes
    -----
    - Uses 'create_structure_path' to locate the parent folder for each structure.
    - Uses 'create_output_folder_name' to generate a new standardized folder name
      based on the index.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_path = create_structure_path(parent_folder_path, temp_df.at[i, "Structure"])
        current_folder_path = structure_path + temp_df.at[i, "Output folder"]
        new_folder_name = create_output_folder_name(i)
        new_folder_path = structure_path + new_folder_name

        os.rename(current_folder_path, new_folder_path)
        temp_df.at[i, "Output folder"] = new_folder_name
        temp_df.to_csv(template_list)  # to save what was finished in case of a crush


# Function to change the names of TM results -> facilitate reading later on
def rename_scores_angles(template_list, indices, parent_folder_path):
    """
    Rename score and angle-related output files for specified dataset entries.

    This function updates the filenames of score and angular analysis files for
    given indices in a template list CSV file. The files are renamed to a new
    standardized base name derived from the entry's index. File renaming is done
    in the filesystem without altering the CSV metadata.

    Parameters
    ----------
    template_list : str
        Path to the CSV file containing metadata for all templates and analyses.
    indices : iterable of int
        List or array of row indices in `template_list` to process.
    parent_folder_path : str
        Base directory containing all structure folders.

    Notes
    -----
    - The base name pattern of the old files depends on the value of the
      'Compare' column:

        * `"tmpl"` → `"tt_" + Map type`

        * `"subtomo"` → `"ts_t<tomogram_number>_" + Map type`

        * other → `"td_<Compare>_" + Map type`

    - The following files are renamed with new base name for each entry:

        * '<base>.csv'

        * '<base>_scores.em'

        * '<base>_angles.em'

        * '<base>_angles_dist_all.em'

        * '<base>_angles_dist_normals.em'

        * '<base>_angles_dist_inplane.em'

    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        structure_name = temp_df.at[i, "Structure"]

        if temp_df.at[i, "Compare"] == "tmpl":
            comp_type = "tt_"
        elif temp_df.at[i, "Compare"] == "subtomo":
            tomo = create_em_path(parent_folder_path, structure_name, temp_df.at[i, "Tomo map"])
            tomo_number = re.findall(r"\d+", temp_df.at[i, "Tomogram"])[0]
            comp_type = "ts_t" + tomo_number + "_"
        else:
            tomo = create_em_path(parent_folder_path, temp_df.at[i, "Compare"], temp_df.at[i, "Tomo map"])
            comp_type = "td_" + temp_df.at[i, "Compare"] + "_"

        output_base = comp_type + temp_df.at[i, "Map type"]
        new_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        csv_file = output_folder + output_base + ".csv"
        scores_map = output_folder + output_base + "_scores.em"
        angles_map1 = output_folder + output_base + "_angles.em"
        angles_map2 = output_folder + output_base + "_angles_dist_all.em"
        angles_map3 = output_folder + output_base + "_angles_dist_normals.em"
        angles_map4 = output_folder + output_base + "_angles_dist_inplane.em"

        new_csv_file = output_folder + new_base + ".csv"
        new_scores_map = output_folder + new_base + "_scores.em"
        new_angles_map1 = output_folder + new_base + "_angles.em"
        new_angles_map2 = output_folder + new_base + "_angles_dist_all.em"
        new_angles_map3 = output_folder + new_base + "_angles_dist_normals.em"
        new_angles_map4 = output_folder + new_base + "_angles_dist_inplane.em"

        os.rename(csv_file, new_csv_file)
        os.rename(scores_map, new_scores_map)
        os.rename(angles_map1, new_angles_map1)
        os.rename(angles_map2, new_angles_map2)
        os.rename(angles_map3, new_angles_map3)
        os.rename(angles_map4, new_angles_map4)


def correct_bbox(template_list, indices):
    """
    Increment specific bounding box-related columns by 1 for completed entries.

    This function reads a CSV file containing template metadata and, for the specified
    row indices where the "Done" column is True, increments the values in several
    bounding box-related columns by 1 along each spatial dimension ("x", "y", "z").
    The updated DataFrame is saved back to the same CSV file after all corrections.

    Parameters
    ----------
    template_list : str
        Path to the CSV file containing metadata for all templates and analyses.
        The file must include the following columns:
        - "Dim x", "Dim y", "Dim z"
        - "O dist_all x", "O dist_all y", "O dist_all z"
        - "O dist_normals x", "O dist_normals y", "O dist_normals z"
        - "O dist_inplane x", "O dist_inplane y", "O dist_inplane z"
    indices : iterable of int
        List or array of row indices to process within the CSV file.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        list_to_correct = ["Dim", "O dist_all", "O dist_normals", "O dist_inplane"]

        for l in list_to_correct:
            for d in ["x", "y", "z"]:
                temp_df.at[i, l + " " + d] += 1

        temp_df.to_csv(template_list)


def recompute_dist_maps(template_list, indices, parent_folder_path, angle_list_path):
    """
    Recompute angular distance maps for specified entries in a template list.

    This function reads a CSV file containing template metadata, and for each specified
    index where the "Done" flag is True, it recalculates angular distance maps using
    corresponding angle files. The updated maps are written to disk.

    Parameters
    ----------
    template_list : str
        Path to the CSV file containing metadata for all templates and analyses.
    indices : iterable of int
        List or array of row indices in the CSV file to process.
    parent_folder_path : str
        Base path to the parent folder where structure and output folders are.
    angle_list_path : str
        Base path to the directory containing angle list files referenced in the CSV.
    """

    temp_df = pd.read_csv(template_list, index_col=0)

    for i in indices:
        if not temp_df.at[i, "Done"]:
            continue

        structure_name = temp_df.at[i, "Structure"]

        output_base = create_output_base_name(i)
        output_folder = create_output_folder_path(parent_folder_path, structure_name, temp_df.at[i, "Output folder"])
        angles_map = output_folder + output_base + "_angles.em"
        angle_list = angle_list_path + temp_df.at[i, "Angles"]
        c_symmetry = temp_df.at[i, "Symmetry"]
        _, _, _ = tmana.create_angular_distance_maps(angles_map, angle_list, write_out_maps=True, c_symmetry=c_symmetry)
