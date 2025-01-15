from cryocat import cryomap
import numpy as np
import pandas as pd
from cryocat import mdoc
from cryocat import ioutils
from skimage.transform import downscale_local_mean
from skimage import exposure
import warnings
from functools import wraps


class TiltStack:

    def __init__(self, tilt_stack, input_order="xyz", output_order="xyz"):

        if not isinstance(tilt_stack, np.ndarray):  # if loading necessary, load in zyx
            self.data = cryomap.read(tilt_stack, transpose=False)
        else:
            self.data = tilt_stack.copy()
            if input_order == "xyz":
                self.data = self.data.transpose(2, 1, 0)

        self.data_type = self.data.dtype

        self.input_order = input_order
        self.current_order = "zyx"
        self.output_order = output_order

        self.n_tilts, self.height, self.width = self.data.shape

    def write_out(self, output_file, new_data=None):

        if output_file:
            cryomap.write(new_data or self.data, output_file, data_type=self.data_type, transpose=False)

    def correct_order(self, new_data=None):

        return_data = new_data or self.data

        if return_data.dtype != self.data_type:
            return_data = return_data.astype(self.data_type)

        if self.current_order != self.output_order:
            return return_data.transpose(2, 1, 0)
        else:
            return return_data


def crop(tilt_stack, new_width=None, new_height=None, output_file=None, input_order="xyz", output_order="xyz"):

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    new_width = new_width or ts.width
    new_height = new_height or ts.height

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

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    idx_to_remove_final = ioutils.indices_load(idx_to_remove, numbered_from_1=numbered_from_1)

    ts.data = np.delete(ts.data, idx_to_remove_final, axis=0)
    ts.write_out(output_file)

    return ts.correct_order()


def bin(tilt_stack, binning_factor, output_file=None, input_order="zyx", output_order="zyx"):

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)
    ts.data = downscale_local_mean(ts.data, (1, binning_factor, binning_factor))
    ts.write_out(output_file)

    return ts.correct_order()


def equalize_histogram(
    tilt_stack, eh_method="contrast_stretching", output_file=None, input_order="zyx", output_order="zyx"
):

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    for z in range(ts.n_tilts):
        if eh_method == "contrast_stretching":
            p2, p98 = np.percentile(ts.data[z, :, :], (2, 98))
            ts.data[z, :, :] = exposure.rescale_intensity(ts.data[z, :, :], in_range=(p2, p98))
        elif eh_method == "equalization":
            # Equalization
            ts.data[z, :, :] = exposure.equalize_hist(ts.data[z, :, :])
        elif eh_method == "adaptive_eq":
            # Adaptive Equalization
            ts.data[z, :, :] = exposure.equalize_adapthist(ts.data[z, :, :], clip_limit=0.03)
        else:
            raise ValueError(f"The {eh_method} is not known!")

    ts.write_out(output_file)

    return ts.correct_order()


def calculate_total_dose_batch(tomo_list, prior_dose_file_format, dose_per_image, output_file_format):
    tomograms = ioutils.tlt_load(tomo_list).astype(int)

    for t in tomograms:
        file_name = ioutils.fileformat_replace_pattern(prior_dose_file_format, t, "x", raise_error=False)
        total_dose = calculate_total_dose(file_name, dose_per_image)
        output_file = ioutils.fileformat_replace_pattern(output_file_format, t, "x", raise_error=False)
        np.savetxt(output_file, total_dose, fmt="%.6f")


def calculate_total_dose(prior_dose, dose_per_image):
    prior_dose = ioutils.total_dose_load(prior_dose)
    total_dose = prior_dose + dose_per_image

    return total_dose


def dose_filter(tilt_stack, pixel_size, total_dose, output_file=None, input_order="xyz", output_order="xyz"):
    # Input: mrc_file or path to it
    #        pixelsize: float, in Angstroms
    #        total_dose: ndarray or path to the .csv, .txt, or .mdoc file
    #        return_data_order: by default x y z (x,y,n_tilts), for napari compatible view use "zyx"

    # temporarily here until this function exist as an entry point on command line

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
    for z in range(ts.n_tilts):
        image = ts.data[z, :, :]
        ts.data[z, :, :] = dose_filter_single_image(image, total_dose[z], frequency_array)

    ts.write_out(output_file)

    return ts.correct_order()


def dose_filter_single_image(image, dose, freq_array):
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
    phaseshift=0,
    output_file=None,
    input_order="xyz",
    output_order="xyz",
):
    """Deconvolution adapted from MATLAB script tom_deconv_tomo by D. Tegunov (https://github.com/dtegunov/tom_deconv)
    and adapted for the tilt series.
    Example for usage: deconvolve(my_map, 3.42, 6, 1.1, 1, 0.02, false, 0)

    Parameters
    ----------
    tilt_stack : np.array or string
        tilt stack
    pixel_size_a : float
        pixel size in Angstroms
    defocus : float, int, str or array-like
        defocus in micrometers, positive = underfocus, or file from CTF estimation
    defocus_file_type : str, default=gctf
        in case the defocus is specified as a file, the type of the file has to be specified (ctffind4, gctf, warp)
    snr_falloff : float
        how fast does SNR fall off, i. e. higher values will downweight high frequencies; values like 1.0 or 1.2 seem reasonable
    deconv_strength : float
        how much will the signal be deconvoluted overall, i. e. a global scale for SNR; exponential scale: 1.0 is SNR = 1000 at zero frequency, 0.67 is SNR = 100, and so on
    highpass_nyquist : float
        fraction of Nyquist frequency to be cut off on the lower end (since it will be boosted the most)
    phase_flipped : bool
        whether the data are already phase-flipped. Defaults to False.
    phaseshift : int
        CTF phase shift in degrees (e. g. from a phase plate). Defaults to 0.
    output_file : str
        Name of the output file for the deconvolved stack. Defaults to None (tilt stack will be not written).
    input_order : str, default='xyz'
        The order of axes in the input tilt stack. Defaults to xyz.
    output_order : str, default='xyz'
        The desired order of axes for the output tilt stacks. Defaults to xyz.

    Returns
    -------
    deconvolved_stack : np.array
        deconvolved tilt stack

    """
    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    if not isinstance(defocus, (int, float)):
        defocus = ioutils.defocus_load(defocus, defocus_file_type)
        defocus = defocus["defocus_mean"].values
    else:
        defocus = np.full((ts.data.n_tilts,), defocus)

    for ts in range(ts.n_tilts):
        tilt = ts.data[ts, :, :]
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
            -defocus[ts] * 1e-6,
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
        ts.data[ts, :, :] = np.real(np.fft.ifftn(np.fft.fftn(tilt) * ramp))

    ts.write_out(output_file)

    return ts.correct_order()


def split_stack_even_odd(tilt_stack, output_file_prefix=None, input_order="xyz", output_order="xyz"):
    """Splits a given tilt stack into even and odd stacks.

    Parameters
    ----------
    tilt_stack : str or array_like
        The input stack of tilt images specified by its filename (including the path) or as 3d numpy array.
    output_file_prefix : str, optional
        The prefix for the output filenames. If provided, the function will save the even and odd stacks as files with
        this prefix followed by '_even.mrc' and '_odd.mrc', respectively.
    input_order : str, default='xyz'
        The order of axes in the input tilt stack. Defaults to xyz.
    output_order : str, default='xyz'
        The desired order of axes for the output tilt stacks. Defaults to xyz.

    Returns
    -------
    tuple of ndarray
        A tuple containing two arrays: the first array contains the even indexed tilts, and the second array contains
        the odd indexed tilts, both reordered according to `output_order`.

    """

    ts = TiltStack(tilt_stack=tilt_stack, input_order=input_order, output_order=output_order)

    even_stack = []
    odd_stack = []

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
