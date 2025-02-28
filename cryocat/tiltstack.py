import os
import re
import numpy as np
from cryocat import cryomap
from cryocat import ioutils
from skimage.transform import downscale_local_mean
from skimage import exposure


class TiltStack:

    def __init__(self, tilt_stack, input_order="xyz", output_order="xyz"):

        if not isinstance(tilt_stack, np.ndarray):  # if loading necessary, load in zyx
            self.data = cryomap.read(tilt_stack, transpose=False)
            if self.data.shape == 2:
                self.data = np.expand_dims(
                    self.data, axis=0
                )  # ensure that it will always have three dimensions, for z=1 mrc returns 2d array
        else:
            self.data = tilt_stack.copy()
            if self.data.shape == 2:
                if input_order == "xyz":
                    self.data = np.expand_dims(self.data, axis=2)  # ensure that it will always have three dimensions
                else:
                    self.data = np.expand_dims(self.data, axis=0)  # ensure that it will always have three dimensions

            if input_order == "xyz":
                self.data = self.data.transpose(2, 1, 0)

        self.data_type = self.data.dtype

        self.input_order = input_order
        self.current_order = "zyx"
        self.output_order = output_order

        self.n_tilts, self.height, self.width = self.data.shape

    def write_out(self, output_file, new_data=None):

        if output_file:
            data_to_write = new_data if new_data is not None else self.data
            cryomap.write(data_to_write, output_file, data_type=self.data_type, transpose=False)

    def correct_order(self, new_data=None):

        return_data = new_data if new_data is not None else self.data

        if return_data.dtype != self.data_type:
            return_data = return_data.astype(self.data_type)

        if self.current_order != self.output_order:
            return return_data.transpose(2, 1, 0)
        else:
            return return_data


def crop(tilt_stack, new_width=None, new_height=None, output_file=None, input_order="xyz", output_order="xyz"):
    """Crop a tilt stack to a specified width and height, and optionally save the result to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be cropped specified either as a path or array-like data.
    new_width : int, optional
        The desired width of the cropped output. If None, the original width is used. Defaults to None.
    new_height : int, optional
        The desired height of the cropped output. If None, the original height is used. Defaults to None.
    output_file : str, optional
        The file path where the cropped tilt stack will be saved. If None, the output is not saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data in the desired order.

    Notes
    -----
    The cropping is performed around the center of the original tilt stack. The function modifies the tilt stack in
    place and saves the cropped data if an output file is specified.
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    if new_width is not None:
        if new_width > ts.width:
            raise ValueError(f"new_width cannot be greater than ts.width ({ts.width})")
    else:
        new_width = ts.width
    if new_height is not None:
        if new_height > ts.height:
            raise ValueError(f"new_height cannot be greater than ts.height ({ts.height})")
    else:
        new_height = ts.height

    # Calculate the center of the original array
    center_w, center_h = ts.width // 2, ts.height // 2

    # Calculate the cropping indices
    start_w = int(center_w - int(new_width) // 2)
    end_w = int(start_w + int(new_width))

    start_h = int(center_h - int(new_height) // 2)
    end_h = int(start_h + int(new_height))

    # crop the actual images
    ts.data = ts.data[:, start_h:end_h, start_w:end_w]

    ts.write_out(output_file)

    return ts.correct_order()


def sort_tilts_by_angle(tilt_stack, input_tilts, output_file=None, input_order="xyz", output_order="xyz"):
    """Sorts a stack of tilts by their angles and optionally writes the sorted data to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be sorted specified either as a path or array-like data.
    input_tilts : str or array-like
        The file path to the input tilt angles. See `ioutils.tlt_load` function for more info.
    output_file : str, optional
        The file path where the sorted tilt data will be saved. If None, the data will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data in the desired order.

    Notes
    -----
    The function loads tilt angles from the specified input file, sorts the tilt stack based on these angles,
    and writes the sorted data to the specified output file if provided. The input and output orders can be
    specified to accommodate different needs.
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    tilt_angles = ioutils.tlt_load(input_tilts, sort_angles=False)
    sorted_indices = np.argsort(tilt_angles)

    ts.data = ts.data[sorted_indices, :, :]
    ts.write_out(output_file)

    return ts.correct_order()


def remove_tilts(
    tilt_stack,
    idx_to_remove,
    numbered_from_1=True,
    output_file=None,
    input_order="xyz",
    output_order="xyz",
):
    """Remove specified tilts from a tilt stack and optionally save the result to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data from which tilts will be removed.
    idx_to_remove : array-like
        Indices of the tilts to remove. If `numbered_from_1` is True, the indices are 1-based.
    numbered_from_1 : bool, defaults=True
        If True, the indices in `idx_to_remove` are considered to be 1-based. Defaults to True.
    output_file : str, optional
        The file path where the modified tilt stack will be saved. If None, the result is not saved.  Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data with the specified tilts removed.

    Notes
    -----
    This function modifies the tilt stack in place and can save the result to a specified output file.
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    idx_to_remove_final = ioutils.indices_load(idx_to_remove, numbered_from_1=numbered_from_1)
    # Check bounds
    max_index = ts.data.shape[0]
    if any(idx < 0 or idx >= max_index for idx in idx_to_remove_final):
        raise IndexError(
            f"One or more indices in idx_to_remove exceed bounds. " f"Valid range: 0 to {max_index - 1} (0-based)."
        )
    ts.data = np.delete(ts.data, idx_to_remove_final, axis=0)
    ts.write_out(output_file)

    return ts.correct_order()


def bin(tilt_stack, binning_factor, output_file=None, input_order="xyz", output_order="xyz"):
    """Binning of a tilt stack using local mean downscaling.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be binned.
    binning_factor : int
        The factor by which to downscale the tilt stack.
    output_file : str, optional
        The file path to save the binned tilt stack. If None, the output will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack binned data in the specified output order.

    Notes
    -----
    This function utilizes local mean downscaling to reduce the size of the tilt stack
    by the specified binning factor. The output can be saved to a file if an output
    file path is provided.
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)
    ts.data = downscale_local_mean(ts.data, (1, binning_factor, binning_factor))
    ts.write_out(output_file)

    return ts.correct_order()


def equalize_histogram(
    tilt_stack, eh_method="contrast_stretching", output_file=None, input_order="xyz", output_order="xyz"
):
    """Equalizes the histogram of a tilt stack using specified methods.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be processed.
    eh_method : str, default='contrast_stretching'
        The method used for histogram equalization. Options are:
        - 'contrast_stretching': Applies contrast stretching.
        - 'equalization': Applies standard histogram equalization.
        - 'adaptive_eq': Applies adaptive histogram equalization.
        Defaults to 'contrast_stretching'.
    output_file : str, optional
        The file path where the output data will be saved. If None, the data will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data with equalized histograms.

    Raises
    ------
    ValueError
        If an unknown histogram equalization method is specified.
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)
    for z in range(ts.n_tilts):
        if eh_method == "contrast_stretching":
            p2, p98 = np.percentile(ts.data[z, :, :], (2, 98))
            ts.data[z, :, :] = exposure.rescale_intensity(ts.data[z, :, :], in_range=(p2, p98))
        elif eh_method == "equalization":
            # Equalization
            ts.data[z, :, :] = exposure.equalize_hist(ts.data[z, :, :])
        elif eh_method == "adaptive_eq":
            # Adaptive
            img = ts.data[z, :, :]
            img = (img - img.min()) / (img.max() - img.min())
            ts.data[z, :, :] = exposure.equalize_adapthist(img, clip_limit=0.03)
        else:
            raise ValueError(f"The {eh_method} is not known!")

    ts.write_out(output_file)
    return ts.correct_order()


def calculate_total_dose_batch(tomo_list, prior_dose_file_format, dose_per_image, output_file_format):
    """Calculate the total dose for a batch of tilt series and save the results to output files.

    Parameters
    ----------
    tomo_list : str or array-like
        A list of tilt series to be processed.
    prior_dose_file_format : str
        The file format string for the prior dose files, which should include a placeholder for the tilt series identifier.
    dose_per_image : float
        The dose value to be applied per image in the tilt series.
    output_file_format : str
        The file format string for the output files, which should include a placeholder for the tilt series identifier.

    Returns
    -------
    None
        The function saves the total dose results to files specified by the output_file_format.

    Notes
    -----
    The function uses the `ioutils` module to load the tilt series list and replace patterns in file names.
    The total dose is calculated using the `calculate_total_dose` function and results are saved using `numpy.savetxt`.
    """

    tomograms = ioutils.tlt_load(tomo_list).astype(int)

    for t in tomograms:
        file_name = ioutils.fileformat_replace_pattern(prior_dose_file_format, t, "x", raise_error=False)
        total_dose = ioutils.total_dose_load(file_name) + dose_per_image
        output_file = ioutils.fileformat_replace_pattern(output_file_format, t, "x", raise_error=False)
        np.savetxt(output_file, total_dose, fmt="%.6f")


def dose_filter(tilt_stack, pixel_size, total_dose, output_file=None, input_order="xyz", output_order="xyz"):
    """Apply a dose filter to a tilt stack of images.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data containing the images to be filtered.
    pixel_size : float
        The size of a pixel in the same units as the tilt stack in Angstroms.
    total_dose : str or array_like
        The total dose for each tilt image in the stack specified either by a file path or directly as an array.
    output_file : str, optional
        The file path to save the filtered tilt stack. If None, the output will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data with the dose filter applied.

    Notes
    -----
    This function calculates a frequency array based on the pixel size and applies a dose filter to each image in the
    tilt stack. The filtered images are then saved to the specified output file if provided.
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)
    pixel_size = float(pixel_size)
    total_dose = ioutils.total_dose_load(total_dose)

    # Precalculate frequency array
    frequency_array = np.zeros((ts.height, ts.width))
    cen_x = ts.width // 2  # Center for array is half the image size
    cen_y = ts.height // 2  # Center for array is half the image size

    rstep_x = 1 / (ts.width * pixel_size)  # reciprocal pixel size
    rstep_y = 1 / (ts.height * pixel_size)

    # Loop to fill array with frequency values
    for x in range(ts.width):
        for y in range(ts.height):
            d = np.sqrt(((x - cen_x) ** 2 * rstep_x**2) + ((y - cen_y) ** 2 * rstep_y**2))
            frequency_array[y, x] = d

    # Generate filtered stack
    ts.data = np.array(ts.data, copy=True)  # Make ts.data writeable
    for z in range(ts.n_tilts):
        image = ts.data[z, :, :]
        ts.data[z, :, :] = dose_filter_single_image(image, total_dose[z], frequency_array)

    ts.write_out(output_file)

    return ts.correct_order()


def dose_filter_single_image(image, dose, freq_array):
    """Filter a single image based on dose and frequency array using Fourier transform.

    Parameters
    ----------
    image : ndarray
        The input image to be filtered, represented as a 2D array.
    dose : float
        The dose value used to calculate the exposure-dependent amplitude attenuator.
    freq_array : ndarray
        The frequency array corresponding to the image, used in the calculation of the attenuator.

    Returns
    -------
    numpy.ndarray
        The filtered image, represented as a 2D array, with the same shape as the input image.

    Notes
    -----
    This function applies a frequency-dependent attenuation based on the dose, using parameters derived from
    the Grant and Grigorieff paper. The Fourier transform is utilized to perform the filtering in the frequency domain.
    """

    # Hard-coded resolution-dependent critical exposures
    # These parameters come from the fitted numbers in the Grant and Grigorieff paper.
    a = 0.245
    b = -1.665
    c = 2.81

    # Calculate Fourier transform
    ft = np.fft.fftshift(np.fft.fft2(image))

    # Calculate exposure-dependent amplitude attenuator
    q = np.exp((-dose) / (2 * ((a * (freq_array**b)) + c)))

    # Attenuate and inverse transform
    filtered_image = np.fft.ifft2(np.fft.ifftshift(ft * q))

    return filtered_image.real


def deconvolve(
    tilt_stack,
    pixel_size_a,
    defocus,
    defocus_file_type="gctf",
    snr_falloff=1.2,
    deconv_strength=1.0,
    highpass_nyquist=0.02,
    phase_flipped=False,
    phaseshift=0.0,
    output_file=None,
    input_order="xyz",
    output_order="xyz",
):
    """Deconvolution adapted from MATLAB script tom_deconv_tomo by D. Tegunov (https://github.com/dtegunov/tom_deconv)
    and adapted for the tilt series.
    Example for usage: deconvolve(my_map, 3.42, 6, 1.1, 1, 0.02, false, 0)

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be deconvolved.
    pixel_size_a : float
        Pixel size in Angstroms.
    defocus : float, int, str or array-like
        Defocus in micrometers, positive = underfocus, or file from CTF estimation.
    defocus_file_type : str, default='gctf'
        In case the defocus is specified as a file, the type of the file has to be specified (ctffind4, gctf, warp).
    snr_falloff : float, default=1.2
        How fast does SNR fall off, i. e. higher values will downweight high frequencies; values like 1.0 or 1.2 seem
        reasonable. Defaults to 1.2.
    deconv_strength : float, default=1.0
        How much will the signal be deconvoluted overall, i. e. a global scale for SNR; exponential scale: 1.0 is
        SNR = 1000 at zero frequency, 0.67 is SNR = 100, and so on. Defaults to 1.0.
    highpass_nyquist : float, default=0.02
        Fraction of Nyquist frequency to be cut off on the lower end (since it will be boosted the most). Defaults to 0.02.
    phase_flipped : bool, default=False
        Whether the data are already phase-flipped. Defaults to False.
    phaseshift : float, default=0
        CTF phase shift in degrees (e. g. from a phase plate). Defaults to 0.
    output_file : str, optional
        Name of the output file for the deconvolved stack. Defaults to None (tilt stack will be not written).  Defaults
        to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    deconvolved_stack : numpy.ndarray
        Deconvolved tilt stack data.

    """
    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    if not isinstance(defocus, (int, float)):
        defocus = ioutils.defocus_load(defocus, defocus_file_type)
        defocus = defocus["defocus_mean"].values
    else:
        defocus = np.full((ts.data.n_tilts,), defocus)

    for z in range(ts.n_tilts):
        tilt = ts.data[z, :, :]
        interp_dim = np.maximum(2048, tilt.shape[0])

        # Generate highpass filter
        highpass = np.arange(0, 1, 1 / interp_dim)
        highpass = np.minimum(1, highpass / highpass_nyquist) * np.pi
        highpass = 1 - np.cos(highpass)

        # Calculate SNR and Wiener filter
        snr = (
            np.exp(np.arange(0, -1, -1 / interp_dim) * snr_falloff * 100 / pixel_size_a)
            * (10 ** (3 * deconv_strength))
            * highpass
        )
        ctf = cryomap.compute_ctf_1d(
            interp_dim,
            pixel_size_a * 1e-10,
            300e3,
            2.7e-3,
            -defocus[z] * 1e-6,
            0.07,
            phaseshift / 180 * np.pi,
            0,
        )
        if phase_flipped:
            ctf = np.abs(ctf)
        wiener = ctf / (ctf * ctf + 1 / snr)

        # Generate ramp filter
        s = tilt.shape
        x, y = np.meshgrid(
            np.arange(-s[0] / 2, s[0] / 2),
            np.arange(-s[1] / 2, s[1] / 2),
            indexing="ij",
        )

        x /= abs(s[0] / 2)
        y /= abs(s[1] / 2)
        r = np.sqrt(x * x + y * y)
        r = np.minimum(1, r)
        r = np.fft.ifftshift(r)

        x = np.arange(0, 1, 1 / interp_dim)
        ramp_interp = cryomap.interp1d(x, wiener, fill_value="extrapolate")

        ramp = ramp_interp(r.flatten()).reshape(r.shape)
        # Perform deconvolution
        ts.data[z, :, :] = np.real(np.fft.ifftn(np.fft.fftn(tilt) * ramp))

    ts.write_out(output_file)

    return ts.correct_order()


def split_stack_even_odd(tilt_stack, output_file_prefix=None, input_order="xyz", output_order="xyz"):
    """Splits a given tilt stack into even and odd stacks.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data specified by its filename (including the path) or as 3d numpy array.
    output_file_prefix : str, optional
        The prefix for the output filenames. If provided, the function will save the even and odd stacks as files with
        this prefix followed by '_even.mrc' and '_odd.mrc', respectively. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays: the first array contains the even indexed tilts, and the second array contains
        the odd indexed tilts, both reordered according to `output_order`.

    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    even_stack = []
    odd_stack = []

    if not ts.n_tilts == 1:
        # For each tilt image in the stack
        for i in range(ts.n_tilts):

            # Split to even and odd by using modulo 2
            if i % 2 == 0:
                even_stack.append(ts.data[i, :, :])
            else:
                odd_stack.append(ts.data[i, :, :])

        even_stack = np.stack(even_stack, axis=0)
        odd_stack = np.stack(odd_stack, axis=0)

        if output_file_prefix:
            ts.write_out(output_file_prefix + "_even.mrc", new_data=even_stack)
            ts.write_out(output_file_prefix + "_odd.mrc", new_data=odd_stack)

        return ts.correct_order(even_stack), ts.correct_order(odd_stack)
    else:
        raise ValueError(f"Stack contains only 1 tilt.")


def merge(file_path_pattern, output_file=None, output_order="xyz"):
    """Merge multiple files matching a given pattern into a single stack.

    Parameters
    ----------
    file_path_pattern : str
        A pattern for file paths to match files that will be merged. This can include wildcards, i.e. tilt.mrc* will
        load all files from given folder that start with tilt.mrc followed by numbering such as tilt.mrc001, tilt.mrc2
        etc.
    output_file : str, optional
        The path to the output file where the merged stack will be saved. If None, the stack will not be saved to a file.
        Defaults to None.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    TiltStack
        A TiltStack object containing the merged data in the specified output order.

    Notes
    -----
    This function retrieves all files matching the specified pattern, sorts them, and then merges their contents into a
    single stack. The resulting stack is saved to the specified output file if provided. Since the data are always
    loaded first (having always 'zyx' order), the input_order is irrelevant and thus not required.

    Examples
    --------
    >>> merged_stack = merge("data/*.mrc", output_file="merged_output.mrc", output_order="xyz")
    """

    files, wildcards = ioutils.get_all_files_matching_pattern(file_path_pattern)
    sorted_files = ioutils.sort_files_by_idx(files, wildcards, order="ascending")

    all_stacks = []

    for sf in sorted_files:
        ts = TiltStack(sf, input_order="zyx", output_order=output_order)
        all_stacks.append(ts.data)

    final_stack = np.concatenate(all_stacks, axis=0)
    final_ts = TiltStack(final_stack, input_order="zyx", output_order=output_order)

    final_ts.write_out(output_file)

    return final_ts.correct_order()


def flip_along_axes(tilt_stack, axes, output_file=None, input_order="xyz", output_order="xyz"):
    """Flip the tilt stack along specified axes and optionally save the result to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be flipped along one or more axes.
    axes : list of str
        The axes along which to flip the tilt stack. Acceptable values are 'x', 'y', and 'z'.
    output_file : str, optional
        The file path to save the flipped tilt stack. If None, the result is not saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.

    Returns
    -------
    numpy.ndarray
        The flipped tilt stack data in the specified output order.

    Raises
    ------
    ValueError
        If the axes contains different values than 'x','y','z'.

    Notes
    -----
    The flipping correspond to IMOD's 'clip' function with options flipx, flipy, flipz. If multiple axes are specified
    it correspond to concatenation of those IMOD operations. For example, axes=['x','y'] will correspond to calling
    clip flipx input.mrc output_x.mrc and subsequently clip flipy ouput_x.mrc output_y.mrc. This is not equivalent to
    the result of calling clip flipxy input.mrc output.mrc!
    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    if not isinstance(axes, list):
        axes = [axes]

    for a in axes:
        if a == "x":
            ts.data = ts.data[:, ::-1, :]
        elif a == "y":
            ts.data = ts.data[:, :, ::-1]
        elif a == "z":
            ts.data = ts.data[::-1, :, :]
        else:
            raise ValueError(f"The axes can be 'x', 'y', or 'z'. Provided axis {a} not supported.")

    ts.write_out(output_file)

    return ts.correct_order()
