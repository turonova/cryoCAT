import os
import re
import numpy as np
from cryocat.core import cryomap
from cryocat.utils import ioutils
from cryocat.utils import imageutils
from cryocat.utils.classutils import as_list
from skimage.transform import downscale_local_mean


class TiltStack:

    def __init__(self, tilt_stack, input_order="xyz", output_order="xyz"):
        """Load or wrap a tilt series, storing data internally in ``zyx`` order.

        Parameters
        ----------
        tilt_stack : str or numpy.ndarray
            Path to an MRC file or a pre-loaded NumPy array.  When a path is
            given the file is read via :func:`cryomap.read`.  A 2-D input
            array is promoted to 3-D (one tilt).
        input_order : {'xyz', 'zyx'}, default='xyz'
            Axis convention of the input array.  Ignored when loading from a
            file (files are always read in ``zyx`` order).  For a 2-D array,
            determines which axis the singleton tilt dimension is added on.
        output_order : {'xyz', 'zyx'}, default='xyz'
            Axis convention used when returning data through
            :meth:`write_out` and :meth:`correct_order`.

        Attributes
        ----------
        data : numpy.ndarray
            Tilt series stored in ``zyx`` order with shape
            ``(n_tilts, height, width)``.
        data_type : numpy.dtype
            Data type of the loaded array.
        n_tilts : int
            Number of tilt images.
        height : int
            Height of each tilt image (y dimension in ``zyx``).
        width : int
            Width of each tilt image (x dimension in ``zyx``).
        """

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

    def write_out(self, output_path, new_data=None, **output_kwargs):
        """Writes data to a specified output file.

        Parameters
        ----------
        output_path : str
            The path to the output file where data will be written.
        new_data : optional
            The data to write to the output file. If not provided, the method will use the instance's data. Default is None.
        **output_kwargs
            Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

        Returns
        -------
        None

        Notes
        -----
        This method uses the `cryomap.write` function to perform the actual writing of data.
        """

        if output_path:
            data_to_write = new_data if new_data is not None else self.data
            write_kw = {"data_type": self.data_type, "transpose": False}
            write_kw.update(output_kwargs)
            cryomap.write(data_to_write, output_path, **write_kw)

    def correct_order(self, new_data=None):
        """Corrects the order of the data and ensures it is of the correct type.

        Parameters
        ----------
        new_data : array-like, optional
            The new data to be corrected. If None, the method will use the instance's data. Default is None.

        Returns
        -------
        array
            The corrected data, which is either the new data with the correct type and order,
            or the instance's data if no new data is provided.

        Notes
        -----
        The method checks if the data type of the provided or instance data matches the expected
        data type. If not, it converts the data to the expected type. Additionally, if the current
        order of the data does not match the desired output order, the data is transposed to
        the correct order.
        """

        return_data = new_data if new_data is not None else self.data

        if return_data.dtype != self.data_type:
            return_data = return_data.astype(self.data_type)

        if self.current_order != self.output_order:
            return return_data.transpose(2, 1, 0)
        else:
            return return_data


def crop(tilt_stack, new_width=None, new_height=None, output_path=None, input_order="xyz", output_order="xyz", **output_kwargs):
    """Crop a tilt stack to a specified width and height, and optionally save the result to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be cropped specified either as a path or array-like data.
    new_width : int, optional
        The desired width of the cropped output. If None, the original width is used. Defaults to None.
    new_height : int, optional
        The desired height of the cropped output. If None, the original height is used. Defaults to None.
    output_path : str, optional
        The file path where the cropped tilt stack will be saved. If None, the output is not saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data in the desired order.

    Notes
    -----
    The cropping is performed around the center of the original tilt stack. The function modifies the tilt stack in
    place and saves the cropped data if an output file is specified.
    """

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    print(f"Cropping of the tilt stack started...")

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    if new_width is not None:
        new_width = int(new_width)
        if new_width > ts.width:
            raise ValueError(f"new_width cannot be greater than ts.width ({ts.width})")
    else:
        new_width = ts.width
    if new_height is not None:
        new_height = int(new_height)
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

    ts.write_out(output_path, **output_kwargs)

    print(f"...cropping of the tilt stack successfully finished. New dimensions are {end_w-start_w}, {end_h-start_h}\n")

    return ts.correct_order()


def sort_tilts_by_angle(tilt_stack, input_tilts, output_path=None, input_order="xyz", output_order="xyz", **output_kwargs):
    """Sorts a stack of tilts by their angles and optionally writes the sorted data to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be sorted specified either as a path or array-like data.
    input_tilts : str or array-like
        The file path to the input tilt angles. See `ioutils.tlt_load` function for more info.
    output_path : str, optional
        The file path where the sorted tilt data will be saved. If None, the data will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

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

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    print(f"Reordering of the tilt stack started...")

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    tilt_angles = ioutils.tlt_load(input_tilts, sort_angles=False)
    sorted_indices = np.argsort(tilt_angles)

    ts.data = ts.data[sorted_indices, :, :]
    ts.write_out(output_path, **output_kwargs)

    print("...reordering of the tilt stack successfully finished.\n")

    return ts.correct_order()


def remove_tilts(
    tilt_stack,
    idx_to_remove,
    numbered_from_1=True,
    output_path=None,
    input_order="xyz",
    output_order="xyz",
    **output_kwargs,
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
    output_path : str, optional
        The file path where the modified tilt stack will be saved. If None, the result is not saved.  Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data with the specified tilts removed. If the tilts to be cleaned correspond to all the tilts (apart from those already cleaned), the numpy 3D array with the original tilt stack data is returned.

    Notes
    -----
    This function modifies the tilt stack in place and can save the result to a specified output file.
    """

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    print(f"Removing of specified tilts started...")

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    idx_to_remove_final = ioutils.indices_load(idx_to_remove, numbered_from_1=numbered_from_1)
    if idx_to_remove_final is not None:
        # Check bounds
        max_index = ts.data.shape[0]
        if any(idx < 0 or idx >= max_index for idx in idx_to_remove_final):
            raise IndexError(
                f"One or more indices in idx_to_remove exceed bounds. " f"Valid range: 0 to {max_index - 1} (0-based)."
            )
        ts.data = np.delete(ts.data, idx_to_remove_final, axis=0)
        ts.write_out(output_path, **output_kwargs)

        print(f"...removing of {idx_to_remove_final.shape[0]} tilts successfully finished.\n")

        return ts.correct_order()
    else:
        return ts


def bin(tilt_stack, binning_factor, output_path=None, input_order="xyz", output_order="xyz", **output_kwargs):
    """Binning of a tilt stack using local mean downscaling.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be binned.
    binning_factor : int
        The factor by which to downscale the tilt stack.
    output_path : str, optional
        The file path to save the binned tilt stack. If None, the output will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

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

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    print(f"Binning tilt stack with binning factor of {str(binning_factor)} started...")

    # cast in case of string
    binning_factor = int(binning_factor)

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)
    ts.data = downscale_local_mean(ts.data, (1, binning_factor, binning_factor))
    ts.write_out(output_path, **output_kwargs)

    print("...binning finished successfully.\n")
    return ts.correct_order()


def equalize_histogram(
    tilt_stack, eh_method="contrast_stretching", output_path=None, input_order="xyz", output_order="xyz", **output_kwargs
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
    output_path : str, optional
        The file path where the output data will be saved. If None, the data will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data with equalized histograms.

    Raises
    ------
    ValueError
        If an unknown histogram equalization method is specified.
    """

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)
    for z in range(ts.n_tilts):
        ts.data[z, :, :] = imageutils.equalize_histogram_2d(ts.data[z, :, :], method=eh_method)

    ts.write_out(output_path, **output_kwargs)
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
        output_path = ioutils.fileformat_replace_pattern(output_file_format, t, "x", raise_error=False)
        np.savetxt(output_path, total_dose, fmt="%.6f")


def dose_filter(tilt_stack, pixel_size, total_dose, output_path=None, input_order="xyz", output_order="xyz", **output_kwargs):
    """Apply a dose filter to a tilt stack of images.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data containing the images to be filtered.
    pixel_size : float
        The size of a pixel in the same units as the tilt stack in Angstroms.
    total_dose : str or array_like
        The total dose for each tilt image in the stack specified either by a file path or directly as an array.
    output_path : str, optional
        The file path to save the filtered tilt stack. If None, the output will not be saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

    Returns
    -------
    numpy.ndarray
        Numpy 3D array with tilt stack data with the dose filter applied.

    Notes
    -----
    This function calculates a frequency array based on the pixel size and applies a dose filter to each image in the
    tilt stack. The filtered images are then saved to the specified output file if provided.
    """

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    print(f"Dose-filtering started...")

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

    ts.write_out(output_path, **output_kwargs)

    print(f"...dose-filtering finished.")

    return ts.correct_order()


def compute_dose_attenuator(dose: float, freq_array: np.ndarray) -> np.ndarray:
    """Compute the exposure-dependent amplitude attenuator from Grant & Grigorieff.

    Parameters
    ----------
    dose : float
        Accumulated dose for this tilt image.
    freq_array : ndarray
        2-D frequency array with the same spatial shape as the tilt image.

    Returns
    -------
    ndarray
        Attenuator array with the same shape as *freq_array*.
    """
    a = 0.245
    b = -1.665
    c = 2.81
    return np.exp((-dose) / (2 * ((a * (freq_array**b)) + c)))


def dose_filter_single_image(image: np.ndarray, dose: float, freq_array: np.ndarray) -> np.ndarray:
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
    q = compute_dose_attenuator(dose, freq_array)
    return imageutils.apply_fft_filter(image, q)


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
    output_path=None,
    input_order="xyz",
    output_order="xyz",
    **output_kwargs,
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
    output_path : str, optional
        Name of the output file for the deconvolved stack. Defaults to None (tilt stack will be not written).  Defaults
        to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

    Returns
    -------
    deconvolved_stack : numpy.ndarray
        Deconvolved tilt stack data.

    """
    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    if not isinstance(defocus, (int, float)):
        defocus = ioutils.defocus_load(defocus, defocus_file_type)
        defocus = defocus["defocus_mean"].values
    else:
        defocus = np.full((ts.n_tilts,), defocus)

    for z in range(ts.n_tilts):
        tilt = ts.data[z, :, :]
        interp_dim = np.maximum(2048, tilt.shape[0])
        wiener = imageutils.compute_wiener_1d(
            interp_dim, pixel_size_a, defocus[z], snr_falloff, deconv_strength,
            highpass_nyquist, phase_flipped, phaseshift,
        )
        ts.data[z, :, :] = imageutils.apply_wiener_radial(tilt, wiener, interp_dim)

    ts.write_out(output_path, **output_kwargs)

    return ts.correct_order()


def split_stack_even_odd(tilt_stack, output_file_prefix=None, input_order="xyz", output_order="xyz", **output_kwargs):
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
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

    Returns
    -------
    tuple of numpy.ndarray
        A tuple containing two arrays: the first array contains the even indexed tilts, and the second array contains
        the odd indexed tilts, both reordered according to `output_order`.

    """

    if output_file_prefix is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_file_prefix is None.")

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
            ts.write_out(output_file_prefix + "_even.mrc", new_data=even_stack, **output_kwargs)
            ts.write_out(output_file_prefix + "_odd.mrc", new_data=odd_stack, **output_kwargs)

        return ts.correct_order(even_stack), ts.correct_order(odd_stack)
    else:
        raise ValueError(f"Stack contains only 1 tilt.")


def merge(file_path_pattern, output_path=None, output_order="xyz", **output_kwargs):
    """Merge multiple files matching a given pattern into a single stack.

    Parameters
    ----------
    file_path_pattern : str
        A pattern for file paths to match files that will be merged. This can include wildcards, i.e. tilt.mrc* will
        load all files from given folder that start with tilt.mrc followed by numbering such as tilt.mrc001, tilt.mrc2
        etc.
    output_path : str, optional
        The path to the output file where the merged stack will be saved. If None, the stack will not be saved to a file.
        Defaults to None.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

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
    >>> merged_stack = merge("data/*.mrc", output_path="merged_output.mrc", output_order="xyz")
    """

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    files, wildcards = ioutils.get_all_files_matching_pattern(file_path_pattern)
    sorted_files = ioutils.sort_files_by_idx(files, wildcards, order="ascending")

    all_stacks = []

    for sf in sorted_files:
        ts = TiltStack(sf, input_order="zyx", output_order=output_order)
        all_stacks.append(ts.data)

    final_stack = np.concatenate(all_stacks, axis=0)
    final_ts = TiltStack(final_stack, input_order="zyx", output_order=output_order)

    final_ts.write_out(output_path, **output_kwargs)

    return final_ts.correct_order()


def flip_along_axes(tilt_stack, axes, output_path=None, input_order="xyz", output_order="xyz", **output_kwargs):
    """Flip the tilt stack along specified axes and optionally save the result to a file.

    Parameters
    ----------
    tilt_stack : str or array-like
        The input tilt stack data to be flipped along one or more axes.
    axes : list of str
        The axes along which to flip the tilt stack. Acceptable values are 'x', 'y', and 'z'.
    output_path : str, optional
        The file path to save the flipped tilt stack. If None, the result is not saved. Defaults to None.
    input_order : str, default='xyz'
        The order of the input data dimensions. Relevant only if tilt_stack in numpy.ndarray. Defaults to 'xyz'.
    output_order : str, default='xyz'
        The order of the output data dimensions. It does not influence order for writing the stack out, just of the
        returned array. Defaults to 'xyz'.
    **output_kwargs
        Additional keyword arguments forwarded to :func:`cryocat.core.cryomap.write`.

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

    if output_path is None and output_kwargs:
        raise ValueError("output_kwargs provided but output_path is None.")

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    axes = as_list(axes)

    for a in axes:
        if a == "x":
            ts.data = ts.data[:, ::-1, :]
        elif a == "y":
            ts.data = ts.data[:, :, ::-1]
        elif a == "z":
            ts.data = ts.data[::-1, :, :]
        else:
            raise ValueError(f"The axes can be 'x', 'y', or 'z'. Provided axis {a} not supported.")

    ts.write_out(output_path, **output_kwargs)

    return ts.correct_order()
