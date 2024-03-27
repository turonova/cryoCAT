from cryocat import cryomap
import numpy as np
import pandas as pd
from cryocat import mdoc
from cryocat import ioutils
from skimage.transform import downscale_local_mean
from skimage import exposure
import warnings
from functools import wraps


def suppress_warnings(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            result = func(*args, **kwargs)
        return result

    return wrapper


def bin(tilt_stack, binning_factor):

    # binned_stack = np.zeros((tilt_stack.shape[0],tilt_stack.shape[1],tilt_stack.shape[0]

    # for z in range(tilt_stack.shape[0]):

    #    binned_stack[z, :, :] = downscale_local_mean(tilt_stack[z, :, :], (1, binning_factor, binning_factor))

    binned_stack = downscale_local_mean(tilt_stack, (1, binning_factor, binning_factor))
    print(tilt_stack.dtype)
    return binned_stack.astype(tilt_stack.dtype)


def equalize_histogram(tilt_stack, eh_method="contrast_stretching"):

    equalized_titls = np.zeros(tilt_stack.shape)

    for z in range(tilt_stack.shape[0]):
        if eh_method == "contrast_stretching":
            p2, p98 = np.percentile(tilt_stack[z, :, :], (2, 98))
            equalized_titls[z, :, :] = exposure.rescale_intensity(tilt_stack[z, :, :], in_range=(p2, p98))
        elif eh_method == "equalization":
            # Equalization
            equalized_titls[z, :, :] = exposure.equalize_hist(tilt_stack[z, :, :])
        elif eh_method == "adaptive_eq":
            # Adaptive Equalization
            equalized_titls[z, :, :] = exposure.equalize_adapthist(tilt_stack[z, :, :], clip_limit=0.03)
        else:
            raise ValueError(f"The {eh_method} is not known!")

    return equalized_titls.astype(tilt_stack.dtype)


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


def dose_filter(mrc_file, pixel_size, total_dose, output_file=None, return_data_order="xyz"):
    # Input: mrc_file or path to it
    #        pixelsize: float, in Angstroms
    #        total_dose: ndarray or path to the .csv, .txt, or .mdoc file
    #        return_data_order: by default x y z (x,y,n_tilts), for napari compatible view use "zyx"

    # temporarily here until this function exist as an entry point on command line
    pixel_size = float(pixel_size)

    stack_data = cryomap.read(mrc_file)
    total_dose = ioutils.total_dose_load(total_dose)

    imgs_x = stack_data.shape[0]
    imgs_y = stack_data.shape[1]
    n_tilt_imgs = stack_data.shape[2]

    # Precalculate frequency array
    frequency_array = np.zeros((imgs_x, imgs_y))
    cen_x = imgs_x // 2  # Center for array is half the image size
    cen_y = imgs_y // 2  # Center for array is half the image size

    rstep_x = 1 / (imgs_x * pixel_size)  # reciprocal pixel size
    rstep_y = 1 / (imgs_y * pixel_size)

    # Loop to fill array with frequency values
    for x in range(imgs_x):
        for y in range(imgs_y):
            d = np.sqrt(((x - cen_x) ** 2 * rstep_x**2) + ((y - cen_y) ** 2 * rstep_y**2))
            frequency_array[x, y] = d

    # Generate filtered stack
    filtered_stack = np.zeros((imgs_x, imgs_y, n_tilt_imgs), dtype=np.single)
    for i in range(n_tilt_imgs):
        image = stack_data[:, :, i]
        filtered_stack[:, :, i] = dose_filter_single_image(image, total_dose[i], frequency_array)

    if return_data_order == "zyx":
        filtered_stack = filtered_stack.transpose(2, 1, 0)

    if output_file is not None:
        cryomap.write(filtered_stack, output_file)

    return filtered_stack


@suppress_warnings
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
