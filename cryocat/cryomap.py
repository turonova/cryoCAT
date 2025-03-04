import h5py
import emfile
import mrcfile
import re
import numpy as np
from numpy import fft
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as srot
from scipy.interpolate import interp1d
import re
from cryocat import cryomask
from skimage import transform


def scale(input_map, scaling_factor, output_name=None):
    """
    Scales an input map by a given scaling factor.
    
    Parameters
    ----------
    input_map : str or np.ndarray
        The input map to be scaled passed as a numpy.ndarray or as the path to its file.
    scaling_factor : float
        The factor by which to scale the input map (a value grater than one indicates upscaling, between 0 and downscaling). If the scaling factor is greater than 1, anti-aliasing is turned off.
    output_name : str, optional
        The name of the output file to which to write the scaled map. If not provided, the scaled map will not be saved.
    
    Returns
    -------
    scaled_map : ndarray
        The scaled map. If the dtype of the scaled map is float64, it is cast to float32 before being returned.

    Notes
    -----
    The function reads the input map, applies a scaling transformation using bicubic interpolation, and optionally
    saves the result to a specified output file. The output map is converted to `float32` if the original scaled map
    is of type `float64`.
    """

    input_map = read(input_map)

    if scaling_factor > 1:
        anti_alias = False
    else:
        anti_alias = True

    scaled_map = transform.rescale(input_map, scaling_factor, order=3, mode="reflect", anti_aliasing=anti_alias)

    if scaled_map.dtype == np.float64:
        scaled_map = scaled_map.astype(np.float32)

    if output_name is not None:
        write(scaled_map, output_name)

    return scaled_map


def pixels2resolution(fourier_pixels, edge_size, pixel_size, print_out=True):
    """Calculate the resolution in Angstroms based on Fourier pixel count, edge size, and pixel size.

    Parameters
    ----------
    fourier_pixels : int
        Number of pixels in the Fourier space.
    edge_size : float
        Size of the edge of the image in pixels/voxels.
    pixel_size : float
        Size of one pixel/voxel in Angstroms.
    print_out : bool, default=True
        Flag to determine whether to print the resolution. Default is True.

    Returns
    -------
    float
        The calculated resolution in Angstroms.

    Examples
    --------
    >>> pixels2resolution(100, 200, 1.5)
    The target resolution is 3.0 Angstroms.
    3.0
    """

    res = edge_size * pixel_size / fourier_pixels

    if print_out:
        print(f"The target resolution is {res} Angstroms.")

    return res


def resolution2pixels(resolution, edge_size, pixel_size, print_out=True):
    """Calculate the number of Fourier pixels/voxels corresponding to a given resolution for a specific edge size and
    pixel size.

    Parameters
    ----------
    resolution : float
        The target resolution to convert to Fourier pixels/voxels.
    edge_size : float
        The size of the edge in pixels/voxels.
    pixel_size : float
        The size of one pixel/voxel in Angstroms.
    print_out : bool, default=True
        Flag to determine whether to print the resolution. Default is True.

    Returns
    -------
    int
        The number of pixels/voxels corresponding to the given resolution.

    Notes
    -----
    This is the value one should use for low-pass and high-pass filters in STOPGAP, GAPSTOP(TM), and novaSTA.
    """

    pixels = round(edge_size * pixel_size / resolution)

    if print_out:
        print(f"The target resolution corresponds to {pixels} pixels.")

    return pixels


def binarize(input_map, threshold=0.5):
    """Converts a given input map to a binary map based on a specified threshold.

    Parameters
    ----------
    input_map : array_like or str
        The input map to binarize. Can be an array or a path to a file containing the map data.
    threshold : float, default=0.5
        The threshold value used for binarization. Values greater than this threshold will be set to 1,
        and values less than or equal to the threshold will be set to 0. Default is 0.5.

    Returns
    -------
    binary_map : ndarray
        The binarized map as a numpy array of integers (0s and 1s).

    Examples
    --------
    >>> input_map = np.array([0.2, 0.6, 0.4, 0.8])
    >>> binarize(input_map)
    array([0, 1, 0, 1])
    """

    input_map = read(input_map)

    binary_map = (input_map > threshold).astype(int)

    return binary_map


def get_filter_radius(edge_size, fourier_pixels, target_resolution, pixel_size):
    """Calculate the filter radius based on either direct Fourier pixel/voxel specification or target resolution.

    Parameters
    ----------
    edge_size : float
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

    Notes
    -----
    The function requires either a direct specification of the Fourier pixels or both the target resolution and pixel
    size to compute the filter radius.
    """

    if fourier_pixels is not None:
        radius = fourier_pixels
        if pixel_size is not None:
            _ = pixels2resolution(fourier_pixels=fourier_pixels, edge_size=edge_size, pixel_size=pixel_size)
    elif target_resolution is not None and pixel_size is not None:
        radius = resolution2pixels(target_resolution, edge_size=edge_size, pixel_size=pixel_size)
    else:
        raise ValueError(
            "Either target_voxels or target_resolution in combination with pixel_size have to be specified!"
        )

    return radius


def bandpass(
    input_map,
    lp_fourier_pixels=None,
    lp_target_resolution=None,
    hp_fourier_pixels=None,
    hp_target_resolution=None,
    pixel_size=None,
    lp_gaussian=3,
    hp_gaussian=2,
    output_name=None,
):
    """Apply a bandpass filter to an input map using specified low-pass and high-pass filter parameters.

    Parameters
    ----------
    input_map : str or array_like
        The input map to be filtered, either as a filename or as an array.
    lp_fourier_pixels : int, optional
        Number of pixels/voxels in Fourier space for the low-pass filter. Default is None.
    lp_target_resolution : float, optional
        Target resolution in Angstroms for the low-pass filter. Default is None.
    hp_fourier_pixels : int, optional
        Number of pixels/voxels in Fourier space for the high-pass filter. Default is None.
    hp_target_resolution : float, optional
        Target resolution in Angstroms for the high-pass filter. Default is None.
    pixel_size : float, optional
        Pixel/voxel size in Angstroms. Default is None.
    lp_gaussian : int, default=3
        Width of the Gaussian falloff for the low-pass filter. Default is 3.
    hp_gaussian : int, default=2
        Width of the Gaussian falloff for the high-pass filter. Default is 2.
    output_name : str, optional
        Filename to save the filtered output. If not provided, the filtered map is not saved. Default is None.

    Returns
    -------
    bandpass_filtered : ndarray
        The bandpass-filtered map.

    Notes
    -----
    The function reads an input map, applies a bandpass filter by creating a mask in Fourier space that combines
    a low-pass and a high-pass filter, and then applies this mask to the Fourier transform of the input map.
    The result is transformed back to real space. If an output filename is provided, the result is saved.
    """

    input_map = read(input_map)
    lp_radius = get_filter_radius(
        input_map.shape[0],
        fourier_pixels=lp_fourier_pixels,
        target_resolution=lp_target_resolution,
        pixel_size=pixel_size,
    )

    hp_radius = get_filter_radius(
        input_map.shape[0],
        fourier_pixels=hp_fourier_pixels,
        target_resolution=hp_target_resolution,
        pixel_size=pixel_size,
    )
    outer_mask = cryomask.spherical_mask(input_map.shape, lp_radius, gaussian=lp_gaussian, gaussian_outwards=False)
    inner_mask = cryomask.spherical_mask(input_map.shape, hp_radius, gaussian=hp_gaussian, gaussian_outwards=False)
    band_mask = fft.ifftshift(outer_mask - inner_mask)
    write(outer_mask - inner_mask, "band.em", data_type=np.single)
    bandpass_filtered = np.real(fft.ifftn(fft.fftn(input_map) * band_mask))

    # lowpass_filtered = lowpass(
    #     input_map=input_map,
    #     fourier_pixels=lp_fourier_pixels,
    #     target_resolution=lp_target_resolution,
    #     pixel_size=pixel_size,
    #     gaussian=lp_gaussian,
    # )

    # bandpass_filtered = highpass(
    #     input_map=lowpass_filtered,
    #     fourier_pixels=hp_fourier_pixels,
    #     target_resolution=hp_target_resolution,
    #     pixel_size=pixel_size,
    #     gaussian=hp_gaussian,
    # )

    if output_name is not None:
        write(bandpass_filtered, output_name, data_type=np.single)

    return bandpass_filtered


def lowpass(input_map, fourier_pixels=None, target_resolution=None, pixel_size=None, gaussian=3, output_name=None):
    """Apply a lowpass filter to a given input map using Fourier transform methods.

    Parameters
    ----------
    input_map : str or array_like
        The input map to be filtered, either as a file path or as an array.
    fourier_pixels : int, optional
        Number of pixels/voxels in the Fourier space representation. Default is None.
    target_resolution : float, optional
        The target resolution in Angstroms for the filtering process. Default is None.
    pixel_size : float, optional
        The size of each pixel/voxel in the input map in Angstroms.
    gaussian : int, default=3
        Width of the Gaussian falloff for the low-pass filter. Default is 3.
    output_name : str, optional
        The file name to save the filtered map. If not provided, the map is not saved. Default is None.

    Returns
    -------
    filtered_map : ndarray
        The filtered map as a numpy array.

    Examples
    --------
    >>> # For input map with box size 100 and pixel size 7.89
    >>> filtered = lowpass('input_map.mrc', target_resolution=20, pixel_size=7.89)
    The target resolution corresponds to 39 pixels.

    >>> # For input map with box size 100 and pixel size 7.89
    >>> filtered = lowpass('input_map.mrc', fourier_pixels=39, pixel_size=7.89)
    The target resolution is 20.23 Angstroms.
    """

    input_map = read(input_map)
    radius = get_filter_radius(
        input_map.shape[0], fourier_pixels=fourier_pixels, target_resolution=target_resolution, pixel_size=pixel_size
    )

    lowpass_filter = fft.ifftshift(
        cryomask.spherical_mask(input_map.shape, radius, gaussian=gaussian, gaussian_outwards=False)
    )
    # Apply filter
    filtered_map = np.real(fft.ifftn(fft.fftn(input_map) * lowpass_filter))

    if output_name is not None:
        write(filtered_map, output_name, data_type=np.single)

    return filtered_map


def highpass(input_map, fourier_pixels=None, target_resolution=None, pixel_size=None, gaussian=2, output_name=None):
    """Apply a highpass filter to a given input map using Fourier transform methods.

    Parameters
    ----------
    input_map : str or array_like
        The input map filename or its numpy array.
    fourier_pixels : int, optional
        Number of pixels/voxels to use in the Fourier space. Default is None.
    target_resolution : float, optional
        The target resolution in Angstroms for the highpass filter. Default is None.
    pixel_size : float, optional
        The size of each pixel/voxel in the input map in Angstroms. Default is None.
    gaussian : int, default=2
        The width of the Gaussian fall-off in pixels/voxels. Default is 2.
    output_name : str, optional
        The filename to save the filtered output. If None, the filtered map is not saved. Default is None.

    Returns
    -------
    filtered_map : ndarray
        The highpass filtered map as a numpy array.

    Notes
    -----
    The function reads an input map, calculates the necessary filter radius based on the provided parameters,
    applies a spherical highpass filter in Fourier space, and optionally saves the result to a file.
    """

    input_map = read(input_map)
    radius = get_filter_radius(
        input_map.shape[0], fourier_pixels=fourier_pixels, target_resolution=target_resolution, pixel_size=pixel_size
    )

    highpass_filter = fft.ifftshift(
        np.ones(input_map.shape)
        - cryomask.spherical_mask(input_map.shape, radius, gaussian=gaussian, gaussian_outwards=False)
    )

    # Apply filter
    filtered_map = np.real(fft.ifftn(fft.fftn(input_map) * highpass_filter))

    if output_name is not None:
        write(filtered_map, output_name, data_type=np.single)

    return filtered_map


def read(input_map, transpose=True, data_type=None):
    """Reads a map file (from the file or numpy array) and returns the data as a numpy array.

    Parameters
    ----------
    input_map : str or numpy.ndarray
        The input map file name or a numpy array containing the map data. The accepted formats are MRC and EM.
    transpose : bool, optional
        Whether to transpose the data. Default is True.
    data_type : numpy.dtype, optional
        The desired data type of the returned array. If None, the data type is not modified.

    Returns
    -------
    numpy.ndarray
        The map data as a numpy array.

    Raises
    ------
    ValueError
        If the input map file name does not have a valid extension.
        If the input map file is not a valid path or numpy array.

    Notes
    -----
    This function supports reading map files with the following extensions: .mrc, .rec, .st, .ali, .em.

    If the input_map is a string, the function will attempt to open the file and read the data.
    If the input_map is a numpy array, it will be directly used as the map data.

    If transpose is True, the data will be transposed using the transpose(2, 1, 0) method.

    If data_type is not None, the data will be cast to the specified data type using the astype method.

    Examples
    --------
    >>> data = read("map.mrc")
    >>> data = read("map.em", transpose=False, data_type=np.float32)
    >>> data = read(np.random.rand(10, 10, 10))
    """

    if isinstance(input_map, str):

        def valid_mrc(filename):
            pattern = r"\.(mrc|ali|rec|st)(\.\d+)?$"
            return bool(re.search(pattern, filename))

        if valid_mrc(input_map):
            data = mrcfile.open(input_map).data
        elif input_map.endswith(".em"):
            data = emfile.read(input_map)[1]
        else:
            raise ValueError("The input map file name", input_map, "is neither em or mrc file!")

        if transpose:
            data = data.transpose(2, 1, 0)
    elif isinstance(input_map, np.ndarray):
        data = np.array(input_map)
    else:
        raise ValueError(f"Input map must be path to valid file or nparray")

    data = np.array(data, copy=True)
    if data_type is not None:
        data = data.astype(data_type)

    return data


def write(data_to_write, file_name, transpose=True, data_type=None, overwrite=True):
    """Write data to a specified file in a given format.

    Parameters
    ----------
    data_to_write : numpy.ndarray
        The data array to be written to the file. It can be of any shape and type.

    file_name : str
        The name of the file to which the data will be written. The file extension must be
        one of the following: '.mrc', '.rec', or '.em'.

    transpose : bool, default=True
        If True (default), the data will be transposed before writing. The transposition
        will change the order of the axes to (2, 1, 0). Default is True.

    data_type : type, optional
        If specified, the data will be cast to this type before writing. If None (default),
        the original data type will be used.

    overwrite : bool, default=True
        If True (default), existing files will be overwritten. If False, an error will be
        raised if the file already exists. Default is True.

    Raises
    ------
    ValueError
        If the provided file name does not end with one of the allowed extensions
        ('.mrc', '.rec', or '.em').

    Notes
    -----
    The function will convert the data to float32 if the original data type is float64
    before writing to the file.
    """

    if data_type is not None:
        data_to_write = data_to_write.astype(data_type)

    if transpose:
        data_to_write = data_to_write.transpose(2, 1, 0)

    if data_to_write.dtype == np.float64:
        data_to_write = data_to_write.astype(np.float32)

    if file_name.endswith(".mrc") or file_name.endswith(".rec"):
        mrcfile.write(name=file_name, data=data_to_write, overwrite=overwrite)
    elif file_name.endswith(".em"):
        emfile.write(file_name, data=data_to_write, overwrite=overwrite)
    else:
        raise ValueError("The output file name", file_name, "has to end with .mrc, .rec or .em!")


def invert_contrast(input_map, output_name=None):
    """Invert the contrast of an input volume map.

    Parameters
    ----------
    input_map : str or numpy.ndarray
        The path to the input volume map file or the volume map data itself.
    output_name : str, optional
        The name of the output file where the inverted volume map will be saved.
        If not provided, the output will not be saved to a file.

    Returns
    -------
    numpy.ndarray
        The inverted volume map.

    Notes
    -----
    The contrast is inverted by multiplying the input map by -1. The data type
    of the output file will be set to single precision if the input map is of
    type float64; otherwise, it will retain the original data type.
    """

    input_map = read(input_map)
    inverted_map = input_map * (-1)

    if output_name is not None:
        if inverted_map.dtype == np.float64:
            data_type = np.single
        else:
            data_type = inverted_map.dtype

        write(inverted_map, output_name, data_type=data_type)

    return inverted_map


def em2mrc(map_name, invert=False, overwrite=True, output_name=None):
    """Convert a file in EM format to MRC format.

    Parameters
    ----------
    map_name : str
        The name of the input map file to be converted.
    invert : bool, default=False
        If True, the data will be inverted (multiplied by -1). Default is False.
    overwrite : bool, default=True
        If True, allows overwriting of the output file if it already exists. Default is True.
    output_name : str, optional
        The name of the output MRC file. If None, the output name will be derived from `map_name` by replacing the
        last two characters with 'mrc'.

    Returns
    -------
    None
        The function writes the converted data to the specified output file.

    Raises
    -------
    ValueError
        If input map_name is not a valid .em file path

    """
    if not isinstance(map_name, str):
        raise ValueError(f"Input file must be a string, valid path")
    elif not map_name.endswith(".em"):
        raise ValueError(f"Provided path must be .em file")
    data_to_write = read(map_name)

    if invert:
        data_to_write = data_to_write * (-1)

    if output_name is None:
        output_name = map_name[:-2] + "mrc"
    elif not output_name.endswith(".mrc"):
        raise ValueError(f"Specified output file name must end with .mrc")
    write(data_to_write, output_name, overwrite=overwrite)


def mrc2em(map_name, invert=False, overwrite=True, output_name=None):
    """Convert a file in MRC format to EM format.

    map_name : str
        The name of the input map file to be converted.
    invert : bool, default=False
        If True, the data will be inverted (multiplied by -1). Default is False.
    overwrite : bool, default=True
        If True, allows overwriting of the output file if it already exists. Default is True.
    output_name : str, optional
        The name of the output EM file. If None, the output name will be derived from `map_name` by replacing the
        last three characters with 'em'.

    Returns
    -------
    None
        The function writes the converted data to the specified output file.

    Raises
    -------
    ValueError
        If the provided file name does not end with .em extension.

    """
    if not isinstance(map_name, str):
        raise ValueError(f"Input is not a string")
    else:
        if not map_name.endswith(".mrc"):
            raise ValueError(f"Input file is not .mrc file")
    data_to_write = read(map_name)

    if invert:
        data_to_write = data_to_write * (-1)

    if output_name is None:
        output_name = map_name[:-3] + "em"
    elif not output_name.endswith(".em"):
        raise ValueError(f"Specified output_name is not .em file")

    write(data_to_write, output_name, overwrite=overwrite)


def write_hdf5(map_name, labels=None, weight=None, output_name=None):
    """Write data to an HDF5 file.

    Parameters
    ----------
    map_name : str
        The name of the input file containing the data to be written.
    labels : str, optional
        The name of the input file containing the labels to be written. If provided,
        the labels will be stored in the HDF5 file. Default is None.
    weight : str, optional
        The name of the input file containing the weights to be written. If provided, the weights will be stored in
        the HDF5 file. Default is None.
    output_name : str, optional
        The name of the output HDF5 file. If not provided, the output file will be named by replacing the last three
        characters of `map_name` with 'hdf5'. Default is None.

    Returns
    -------
    None
        This function does not return any value. It writes the data, labels, and weights to the specified HDF5 file.

    Raises
    -------
    ValueError
        If the provided file name does not end with .em or .mrc extension.

    """
    if not isinstance(map_name, str):
        raise ValueError(f"Input is not a string")
    else:
        if not map_name.endswith(".mrc") and not map_name.endswith(".em"):
            raise ValueError(f"Input file is not either .em either .mrc file")
    data_to_write = read(map_name)

    if output_name is None:
        if map_name.endswith(".mrc"):
            output_name= map_name[:-3] + "hdf5"
        elif map_name.endswith(".em"):
            output_name = map_name[:-2] + "hdf5"
    elif not output_name.endswith(".mrc") and not output_name.endswith(".em"):
        raise ValueError("Output file path must be .mrc or .em extension")

    f = h5py.File(output_name, "w")

    f.create_dataset("raw", data=data_to_write)

    if labels is not None:
        labels_to_write = read(labels)
        f.create_dataset("label", data=labels_to_write)
    if weight is not None:
        weight_to_write = read(weight)
        f.create_dataset("weight", data=weight_to_write)

    f.close()


def read_hdf5(hdf5_name, dataset_name="predictions", print_datasets=False):
    """Read a dataset from an HDF5 file.

    Parameters
    ----------
    hdf5_name : str
        The name of the HDF5 file to read from.
    dataset_name : str, defaults='predictions'
        The name of the dataset to read from the HDF5 file. Default is 'predictions'.
    print_datasets : bool, default=False
        If True, prints the names of available datasets in the HDF5 file. Default is False.

    Returns
    -------
    numpy.ndarray
        The data from the specified dataset as a NumPy array.

    Raises
    ------
    FileNotFoundError
        If the specified HDF5 file does not exist.
    KeyError
        If the specified dataset name does not exist in the HDF5 file.
    """

    f = h5py.File(hdf5_name, "r")

    if print_datasets:
        print(f"Available datasets: {f.keys()}")

    data = np.array(f[dataset_name][:])
    f.close()

    return data


def normalize(input_map):
    """Normalize a given map by standardizing its values.

    Parameters
    ----------
    map : str or numpy.ndarray
        The input map to be normalized.

    Returns
    -------
    numpy.ndarray
        The normalized map with zero mean and unit variance.

    Notes
    -----
    This function reads the input map, computes its mean and standard deviation,
    and then normalizes the map by subtracting the mean and dividing by the standard deviation.

    Examples
    --------
    >>> normalized_map = normalize(my_map)
    """

    norm_map = read(input_map)

    mean_v = np.mean(norm_map)
    std_v = np.std(norm_map)
    norm_map = (norm_map - mean_v) / std_v

    return norm_map


def rotate(
    input_map,
    rotation=None,
    rotation_angles=None,
    coord_space="zxz",
    transpose_rotation=False,
    degrees=True,
    spline_order=3,
    output_name=None,
):
    """Rotate a 3D input map using a specified rotation matrix or rotation angles.

    Parameters
    ----------
    input_map : str or numpy.ndarray
        The input 3D map to be rotated.

    rotation : Rotation, optional
        A rotation object representing the rotation to be applied. If provided,
        `rotation_angles` (if provided) will not be considered. Default is None.

    rotation_angles : array_like, optional
        Angles for rotation in the specified coordinate space. If provided, they will be considered only if
        `rotation` is not be specified. Default is None.

    coord_space : str, default='zxz'
        The coordinate space for the rotation angles. Default is 'zxz'.

    transpose_rotation : bool, default=False
        If True, the transpose of the rotation matrix will be used. Default is False.

    degrees : bool, default=True
        If True, the rotation angles are interpreted as degrees. Default is True.

    spline_order : int, default=3
        The order of the spline used for interpolation. Default is 3.

    output_name : str, optional
        If specified, the rotated structure will be written to this file. Default is None.

    Returns
    -------
    rot_struct : numpy.ndarray
        The rotated 3D map.

    Raises
    ------
    ValueError
        If neither `rotation` nor `rotation_angles` is specified.
    """

    input_map = read(input_map)
    # create translation to the center of the box
    T = np.eye(4)
    structure_center = np.asarray(input_map.shape) // 2
    T[:3, -1] = structure_center

    rot_matrix = np.eye(4)

    if rotation is not None:
        if transpose_rotation:
            rot_matrix[0:3, 0:3] = rotation.as_matrix().T
        else:
            rot_matrix[0:3, 0:3] = rotation.as_matrix()

    elif rotation_angles is not None:
        rot = srot.from_euler(coord_space, rotation_angles, degrees=degrees)
        rot_matrix[0:3, 0:3] = rot.as_matrix().T

    else:
        raise ValueError("Either rotation_angles or rotation has to be specified!!!")

    final_matrix = T @ rot_matrix @ np.linalg.inv(T)

    rot_struct = np.empty(input_map.shape)
    affine_transform(input=input_map, output=rot_struct, matrix=final_matrix, order=spline_order)

    if output_name is not None:
        write(rot_struct, output_name, data_type=np.single)

    return rot_struct


def crop(input_map, new_size, output_file=None, crop_coord=None):
    """
    This function crops a given input map to a new size. If no crop coordinates are provided, the function will crop from the center of the input map. If an output file is specified, the cropped volume will be written to this file.
    
    Parameters
    ----------
    input_map : str or np.array
        The input map to be cropped. If a string is provided, it is assumed to be the path to the input map file.
    new_size : tuple or str
        The desired size of the cropped volume. If a string is provided, it is assumed to be in the format 'x,y,z'.
    output_file : str, optional
        The path to the output file where the cropped volume will be written. If not provided, the cropped volume will not be written to a file.
    crop_coord : tuple or str, optional
        The coordinates from which to start cropping. If a string is provided, it is assumed to be in the format 'x,y,z'. If not provided, the function will crop from the center of the input map.
    
    Returns
    -------
    cropped_volume : np.array
        The cropped volume.
    
    Notes
    -----
    see also: trim
    """
    input_map = read(input_map)

    new_size = cryomask.get_correct_format(new_size)

    if crop_coord is None:
        crop_coord = cryomask.get_correct_format(input_map.shape) // 2
    else:
        crop_coord = cryomask.get_correct_format(crop_coord)

    vs, ve, _, _ = get_start_end_indices(crop_coord, input_map.shape, new_size)

    cropped_volume = input_map[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]

    if output_file is not None:
        write(cropped_volume, output_file, data_type=np.single)

    return cropped_volume


def shift(map, delta):
    """
    Shifts the input map by a given delta.
    
    Parameters
    ----------
    map : ndarray
        The input map to be shifted. It should be a 3D array.
    delta : float
        The amount by which to shift the map.
    
    Returns
    -------
    shifted_map : ndarray
        The shifted map. It is a 3D array of the same shape as the input map.
    
    Notes
    -----
    This function uses the Fourier shift theorem to perform the shift, which involves a Fourier transform, a multiplication by a phase shift factor, and an inverse Fourier transform.
    """
    
    dimx, dimy, dimz = map.shape

    x = np.arange(-dimx / 2, dimx / 2, 1)
    y = np.arange(-dimy / 2, dimy / 2, 1)
    z = np.arange(-dimz / 2, dimz / 2, 1)
    mx, my, mz = np.meshgrid(x, y, z, indexing="ij")

    delta = delta / np.array([dimx, dimy, dimz])
    sh = delta[0] * mx + delta[1] * my + delta[2] * mz
    fmap = np.fft.fftshift(np.fft.fftn(map))
    shifted_map = np.fft.ifftn(np.fft.ifftshift(fmap * np.exp(-2.0 * np.pi * 1j * sh))).real

    return shifted_map


def shift2(input_map, delta, output_name=None):
    """
    Shifts an input map by a specified delta.
    
    Parameters
    ----------
    input_map : str or np.ndarray
        The input map to be shifted. If a string is provided, it is assumed to be a filename from which to read the map.
    delta : array_like
        The shift to apply to the input map. Must be of length 3.
    output_name : str, optional
        If provided, the shifted map will be written to a file with this name. The data type of the output will be np.single.
    
    Returns
    -------
    shifted_map : np.ndarray
        The input map, shifted by the specified delta.
    
    Notes
    -----
    The shift is applied using an affine transformation with a matrix that is the identity except for the last column, which is set to the negative of the delta. The mode of the affine transformation is "grid-wrap".
    """
    
    input_map = read(input_map)

    T = np.eye(4)
    T[:3, -1] = -np.asarray(delta)

    shifted_map = np.empty(input_map.shape)
    affine_transform(input=input_map, output=shifted_map, matrix=T, mode="grid-wrap")

    if output_name is not None:
        write(shifted_map, output_name, data_type=np.single)

    return shifted_map


def recenter(map, new_center):
    """
    Recenter a given map around a new center.
    
    Parameters
    ----------
    map : str or ndarray
        The input map to be recentered.  If a string is passed, that is assumed to be the path to the map file.
    new_center : ndarray
        The new center coordinates for the map, with relation to the coordinate frame of the map (eg. box size).
    
    Returns
    -------
    trans_struct : ndarray
        The recentered map.
    
    Notes
    -----
    The function creates a transformation matrix, calculates the shift required to move the center of the map to the new center, applies the transformation to the map, and returns the recentered map.
    """
    
    original_map = read(map)
    T = np.eye(4)
    structure_center = np.asarray(original_map.shape) // 2
    shift = new_center - structure_center
    T[:3, -1] = -shift

    trans_struct = np.empty(original_map.shape)
    affine_transform(input=original_map, output=trans_struct, matrix=T)

    return trans_struct


def normalize_under_mask(ref, mask):
    """A function to take a reference volume and a mask, and normalize the area
    under the mask to 0-mean and standard deviation of 1.
    Based on stopgap code by W.Wan

    Parameters
    ----------
    ref : np.ndarray

    mask : np.ndarray


    Returns
    -------
    norm_ref : np.ndarray
        The map with the area under the mask normalized.
    """

    # Calculate mask parameters
    m_idx = mask > 0

    # Calculate stats
    ref_mean = np.mean(ref[m_idx])
    ref_std = np.std(ref[m_idx])

    # Normalize reference
    norm_ref = ref - ref_mean
    norm_ref = norm_ref / ref_std

    return norm_ref


def get_start_end_indices(coord, volume_shape, subvolume_shape):
    """
    This function calculates the start and end indices of a subvolume within a larger volume, given the center coordinate of the subvolume.
    
    Parameters
    ----------
    coord : array_like
        The center coordinate of the subvolume.
    volume_shape : array_like
        The shape of the larger volume.
    subvolume_shape : array_like
        The shape of the subvolume.
    
    Returns
    -------
    volume_start_clip : ndarray
        The start indices of the subvolume within the larger volume, after being clipped to ensure they are within the volume.
    volume_end_clip : ndarray
        The end indices of the subvolume within the larger volume, after being clipped to ensure they are within the volume.
    subvolume_start : ndarray
        The start indices of the subvolume, relative to its own shape.
    subvolume_end : ndarray
        The end indices of the subvolume, relative to its own shape.
    
    Notes
    -----
    The function first calculates the start and end indices of the subvolume within the larger volume, without considering whether they are within the volume. It then clips these indices to ensure they are within the volume. Finally, it calculates the start and end indices of the subvolume, relative to its own shape.
    """
    
    subvolume_shape = np.asarray(subvolume_shape)
    subvolume_half = subvolume_shape / 2

    volume_start = np.floor(coord - subvolume_half).astype(int)
    volume_end = (volume_start + subvolume_shape).astype(int)

    volume_start_clip = np.maximum([0, 0, 0], volume_start)
    volume_end_clip = np.minimum(np.asarray(volume_shape), volume_end)

    subvolume_start = volume_start_clip - volume_start
    subvolume_end = volume_end - volume_start
    subvolume_end = volume_end_clip - volume_end + subvolume_end

    return volume_start_clip, volume_end_clip, subvolume_start, subvolume_end


def extract_subvolume(volume, coordinates, subvolume_shape, enforce_shape=False, output_file=None):
    """
    Extracts a subvolume from a given volume.
    
    Parameters
    ----------
    volume : ndarray
        The 3D array from which to extract the subvolume.
    coordinates : tuple
        The (x, y, z) coordinates of the center of the subvolume to extract.
    subvolume_shape : tuple
        The (x, y, z) shape of the subvolume to extract.
    enforce_shape : bool, optional
        If True, the final volume will have the same shape as the original volume and the voxels outside the region of interest will be set to the mean value of the original volume. Default is False.
    output_file : str, optional
        If provided, the extracted subvolume will be written to this file.
    
    Returns
    -------
    subvolume : ndarray
        The extracted subvolume.
    
    Notes
    -----
    The function first calculates the start and end indices for the volume and subvolume. 
    Then it creates an empty subvolume with the same shape as the desired subvolume, 
    filled with the mean value of the original volume. 
    The values from the original volume are then copied into the subvolume. 
    If an output file is provided, the subvolume is written to this file.
    """
    
    vs, ve, ss, se = get_start_end_indices(coordinates, volume.shape, subvolume_shape)
    if enforce_shape is not False:
        subvolume = np.full(volume.shape, np.mean(volume))
        subvolume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]] = volume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]
    else:
        subvolume = np.full(subvolume_shape, np.mean(volume))
        subvolume[ss[0] : se[0], ss[1] : se[1], ss[2] : se[2]] = volume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]

    if output_file is not None:
        write(subvolume, output_file, data_type=np.single)

    return subvolume


def get_cross_slices(input_map, slice_half_dim=None, slice_numbers=None, axis=None):
    """
    This function generates cross slices across an axis from a given input map.
    
    Parameters
    ----------
    input_map : array_like
        The input map from which to generate cross slices.
    slice_half_dim : int, optional
        The half dimension of the slice. If None, the slice will cover the entire dimension of the input map.
    slice_numbers : array_like, optional
        The slice numbers to use. If None, the slice numbers will be calculated as the ceiling of half the shape of the input map.
    axis : array_like, optional
        The axis along which to generate the slices. If None, slices will be generated along all axes.
    
    Returns
    -------
    cross_slices : list
        A list of cross slices generated from the input map.
    
    Notes
    -----
    The function first reads the input map and checks the axis and slice numbers. If the axis is not provided, it defaults to [0, 1, 2]. If the slice numbers are not provided, they are calculated as the ceiling of half the shape of the input map. The function then generates the cross slices along the specified axis and returns them as a list.
    """
    
    cmap = read(input_map)

    if axis is None:
        axis = np.asarray([0, 1, 2])
    elif isinstance(axis, list):
        axis = np.asarray(axis)

    if slice_numbers is None:
        cs = np.ceil(np.asarray(cmap.shape) / 2).astype(int)
    else:
        cs = np.asarray(slice_numbers).astype(int)

    if axis.shape[0] > cs.shape[0]:
        cs = np.full(axis.shape, cs[0])
    elif axis.shape[0] < cs.shape[0]:
        cs = cs[: axis.shape[0]]

    cross_slices = []

    for i, a in enumerate(axis):
        if slice_half_dim is None:
            s = [0, 0, 0]
            e = cmap.shape
        else:
            s = cs - slice_half_dim
            e = cs + slice_half_dim + 1

        if a == 0:
            cross_slices.append(cmap[s[0] : e[0], s[1] : e[1], cs[2]])
        elif a == 1:
            cross_slices.append(cmap[s[0] : e[0], cs[1], s[2] : e[2]])
        else:
            cross_slices.append(cmap[cs[0], s[1] : e[1], s[2] : e[2]])

    return cross_slices


def pad(input_volume, new_size, fill_value=None):
    """
    Pads an input volume to a new size.
    
    This function reads an input volume, calculates the mean of the volume if no fill value is provided, and creates a new volume of the specified size filled with the mean or provided fill value. The original volume is then placed in the center of the new volume.
    
    Parameters
    ----------
    input_volume : ndarray
        The input volume to be padded.
    new_size : tuple
        The desired size of the new volume. Must be a 3-element tuple.
    fill_value : float, optional
        The value to fill the new volume with. If None, the new volume is filled with the mean of the input volume.
    
    Returns
    -------
    padded_volume : ndarray
        The padded volume of the new size.
    
    Notes
    -----
    The original volume is placed in the center of the new volume. If the new size is not an integer multiple of the original size, the original volume is placed such that there is an equal number of padding voxels on either side of the volume in each dimension.
    """
    
    volume = read(input_volume)

    if fill_value is None:
        padded_volume = np.full(new_size, np.mean(volume))
    else:
        padded_volume = np.full(new_size, fill_value)

    # Size of the original volume
    vol_size = volume.shape

    x_start = int(np.ceil((new_size[0] - vol_size[0]) / 2))
    y_start = int(np.ceil((new_size[1] - vol_size[1]) / 2))
    z_start = int(np.ceil((new_size[2] - vol_size[2]) / 2))

    x_end = int(x_start + vol_size[0])
    y_end = int(y_start + vol_size[1])
    z_end = int(z_start + vol_size[2])

    padded_volume[x_start:x_end, y_start:y_end, z_start:z_end] = volume

    return padded_volume


def place_object(input_object, motl, volume_shape=None, volume=None, feature_to_color="object_id"):
    """
    Places an object or a list of objects into a volume based on the given motion list (motl).
    
    Parameters
    ----------
    input_object : str or list
        The object or list of objects to be placed. If a string is provided, it is assumed to be a path to the object file.
    motl : object
        The motion list containing the rotations and coordinates for placing the objects.
    volume_shape : tuple, optional
        The shape of the volume in which the objects are to be placed. If not provided, the volume parameter must be provided.
    volume : str, optional
        The volume in which the objects are to be placed. If not provided, the volume_shape parameter must be provided.
    feature_to_color : str, default="object_id"
        The feature in the motion list dataframe to use for coloring the objects.
    
    Returns
    -------
    object_container : ndarray
        The volume with the objects placed.
    
    Notes
    -----
    The objects are rotated according to the rotations in the motl list and placed at the coordinates in the motl list.
    The objects are colored according to the feature_to_color parameter.
    If the input_object is a list, each object in the list is placed according to the corresponding rotation and coordinate in the motl list.
    If the input_object is a single object, it is placed at each rotation and coordinate in the motl list.
    """
    
    if not isinstance(input_object, list):
        input_object = read(input_object)

    if volume is not None:
        object_container = read(volume)
    elif volume_shape is not None:
        object_container = np.zeros(volume_shape)

    rotations = motl.get_rotations()
    coordinates = motl.get_coordinates() - 1.0
    colors = motl.df[feature_to_color]

    for i, coord in enumerate(coordinates):

        if isinstance(input_object, list):
            object_map = rotate(input_object[i], rotation=rotations[i], transpose_rotation=True)
        else:
            object_map = rotate(input_object, rotation=rotations[i], transpose_rotation=True)

        object_map = np.where(object_map > 0.1, 1.0, 0.0)

        ls, le, os, oe = get_start_end_indices(coord, object_container.shape, object_map.shape)

        object_shape = object_map[os[0] : oe[0], os[1] : oe[1], os[2] : oe[2]]
        object_container[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]] = np.where(
            object_shape == 1.0,
            colors[i],
            object_container[ls[0] : le[0], ls[1] : le[1], ls[2] : le[2]],
        )

    return object_container


def deconvolve(
    input_volume,
    pixel_size_a,
    defocus,
    snr_falloff,
    deconv_strength,
    highpass_nyquist,
    phase_flipped=False,
    phaseshift=0,
    output_name=None,
):
    """Deconvolution adapted from MATLAB script tom_deconv_tomo by D. Tegunov (https://github.com/dtegunov/tom_deconv).
    Example for usage: deconvolve(my_map, 3.42, 6, 1.1, 1, 0.02, false, 0)

    Parameters
    ----------
    input_volume : np.array or string
        tomogram volume
    pixel_size_a : float
        pixel size in Angstroms
    defocus : float
        defocus in micrometers, positive = underfocus
    snr_falloff : float
        how fast does SNR fall off, i. e. higher values will downweight high frequencies; values like 1.0 or 1.2 seem reasonable
    deconv_strength : float
        how much will the signal be deconvoluted overall, i. e. a global scale for SNR; exponential scale: 1.0 is SNR = 1000 at zero frequency, 0.67 is SNR = 100, and so on
    highpass_nyquist : float
        fraction of Nyquist frequency to be cut off on the lower end (since it will be boosted the most)
    phase_flipped : bool
        whether the data are already phase-flipped. Default is False.
    phaseshift : int
        CTF phase shift in degrees (e. g. from a phase plate). Default is 0.
    output_name : str
        Name of the output file for the deconvolved tomogram. Default is None (tomogram will be not written).

    Returns
    -------
    deconvolved_map : np.array
        deconvolved tomogram

    """
    input_map = read(input_volume)
    interp_dim = np.maximum(2048, input_map.shape[0])

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
    wiener = ctf / (ctf * ctf + 1 / snr)

    # Generate ramp filter
    s = input_map.shape
    x, y, z = np.meshgrid(
        np.arange(-s[0] / 2, s[0] / 2),
        np.arange(-s[1] / 2, s[1] / 2),
        np.arange(-s[2] / 2, s[2] / 2),
        indexing="ij",
    )

    x /= abs(s[0] / 2)
    y /= abs(s[1] / 2)
    z /= max(1, abs(s[2] / 2))
    r = np.sqrt(x * x + y * y + z * z)
    r = np.minimum(1, r)
    r = np.fft.ifftshift(r)

    x = np.arange(0, 1, 1 / interp_dim)
    ramp_interp = interp1d(x, wiener, fill_value="extrapolate")

    ramp = ramp_interp(r.flatten()).reshape(r.shape)
    # Perform deconvolution
    deconvolved_map = np.real(np.fft.ifftn(np.fft.fftn(input_map) * ramp))

    if output_name is not None:
        write(deconvolved_map, output_name, data_type=np.single)

    return deconvolved_map


def compute_ctf_1d(length, pixel_size, voltage, cs, defocus, amplitude, phaseshift, bfactor):
    """
    This function computes the 1D Contrast Transfer Function (CTF) for a given set of parameters.
    
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


def trim(input_map, trim_start, trim_end, output_name=None):
    """
    Trims a 3D map to a specified range.

    Parameters
    ----------
    input_map : ndarray
        The 3D map to be trimmed.
    trim_start : array_like
        The starting coordinates for the trim. Must be a 3-element array-like object.
    trim_end : array_like
        The ending coordinates for the trim. Must be a 3-element array-like object.
    output_name : str, optional
        If provided, the trimmed map will be written to a file with this name. The file will be written in single precision float format.
    
    Returns
    -------
    output_map : ndarray
        The trimmed 3D map.
    
    Notes
    -----
    The trim_start and trim_end parameters are inclusive. That is, the output_map will include the voxels at these coordinates.
    If either trim_start or trim_end is beyond the bounds of input_map, it will be adjusted to fit within the bounds.
    see also: crop
    """
    
    output_map = read(input_map)

    trim_start = np.asarray(trim_start)
    trim_end = np.asarray(trim_end)

    ts = np.maximum(trim_start, np.zeros((3,))).astype(int)
    te = np.minimum(trim_end, np.asarray(output_map.shape)).astype(int)

    output_map = output_map[ts[0] : te[0], ts[1] : te[1], ts[2] : te[2]]

    if output_name is not None:
        write(output_map, output_name, data_type=np.single)

    return output_map


def flip(input_map, axis="z", output_name=None):
    """
    Function to flip a given input map along specified axis.
    
    Parameters
    ----------
    input_map : array_like
        The input map to be flipped.
    axis : str, optional
        The axis along which to flip the input map. Default is "z".
    output_name : str, optional
        The name of the output file. If not provided, the function will return the flipped map.
    
    Returns
    -------
    output_map : array_like
        The flipped map.
    
    Notes
    -----
    The function reads the input map, flips it along the specified axis, and writes the output map to a file if an output name is provided.
    """
    
    output_map = read(input_map)

    if "z" in axis.lower():
        output_map = np.flip(output_map, 2)

    if "y" in axis.lower():
        output_map = np.flip(output_map, 1)

    if "x" in axis.lower():
        output_map = np.flip(output_map, 0)

    if output_name is not None:
        write(output_map, output_name, data_type=np.single)

    return output_map


def calculate_conjugates(vol, filter=None):
    """
    This function calculates the complex conjugates of a volume and its square after applying a Fourier transform and an optional filter.
    
    Parameters
    ----------
    vol : ndarray
        The input volume to be transformed and filtered.
    filter : ndarray, optional
        The filter to be applied to the Fourier transform of the volume. If None, no filter is applied. Default is None.
    
    Returns
    -------
    conj_target : ndarray
        The complex conjugate of the Fourier transform of the input volume after applying the filter.
    conj_target_sq : ndarray
        The complex conjugate of the square of the filtered volume after applying the Fourier transform.
    
    Notes
    -----
    The 0-frequency peak of the Fourier transform of the input volume is set to zero before calculating the complex conjugates.
    """
    
    # Fourier transform tile
    vol_fft = np.fft.fftn(vol)

    # Apply filter
    if filter is not None:
        vol_fft = vol_fft * filter

    # Set 0-frequency peak to zero
    vol_fft[0, 0, 0] = 0

    # Store complex conjugate
    conj_target = np.conj(vol_fft)

    # Filtered volume
    filtered_volume = np.fft.ifftn(vol_fft).real

    # Store complex conjugate of square
    conj_target_sq = np.conj(np.fft.fftn(np.power(filtered_volume, 2)))

    return conj_target, conj_target_sq


def calculate_flcf(vol1, mask, vol2=None, conj_target=None, conj_target_sq=None, filter=None):
    """
    This function calculates the Fast Local Correlation Coefficient (FLCC) map between two volumes (3D arrays). 
    
    Parameters
    ----------
    vol1 : ndarray
        The first volume for which the FLCC map is to be calculated.
    mask : ndarray
        The mask to be applied on the volumes.
    vol2 : ndarray, optional
        The second volume for which the FLCC map is to be calculated. If not provided, `conj_target` and `conj_target_sq` must be provided.
    conj_target : ndarray, optional
        The conjugate of the target volume. Required if `vol2` is not provided.
    conj_target_sq : ndarray, optional
        The square of the conjugate of the target volume. Required if `vol2` is not provided.
    filter : ndarray, optional
        The filter to be applied on the volumes. 
    
    Raises
    ------
    ValueError
        If `vol2` is not provided, both `conj_target` and `conj_target_sq` must be provided.
    
    Returns
    -------
    cc_map : ndarray
        The calculated FLCC map, clipped between 0.0 and 1.0.
    """
    
    # get the size of the box and number of voxels contributing to the calculations
    if np.isnan(vol1).any() or np.isnan(mask).any():
        raise ValueError("Input volumes or mask contain NaN values")
    box_size = np.array(vol1.shape)
    n_pix = mask.sum()

    # Calculate inital Fourier transfroms
    vol1 = np.fft.fftn(vol1)
    mask = np.fft.fftn(mask)

    if vol2 is not None:
        conj_target, conj_target_sq = calculate_conjugates(vol2, filter)

    elif conj_target is None or conj_target_sq is None:
        raise ValueError(
            "If the second volume is NOT provided, both conj_target and conj_target_sw have to be passed as parameters."
        )

    # Calculate numerator of equation
    numerator = np.fft.ifftn(vol1 * conj_target).real

    # Calculate denominator in three steps
    # sigma_a = np.fft.ifftn(mask*conj_target_sq).real/n_pix  # First part of denominator sigma
    # sigma_b = np.power(np.fft.ifftn(mask*conj_target).real/n_pix,2)   # Second part of denominator sigma
    A = np.fft.ifftn(mask * conj_target_sq)
    B = np.fft.ifftn(mask * conj_target)
    denominator = np.sqrt(n_pix * A - B * B).real

    # Shifted FLCL map
    cc_map = (numerator / denominator).real

    # Calculate map and do a much of flips to get orientation correct...
    # Note on the flips - normally, fftshift should directly work on cc_map, no flipping necessary
    # but for some reason the fftshift returns "mirrored" values, i.e. for shift of 6,6,6 the peak would be in -6,-6,-6
    # Following code corresponds to the original one from Matlab and was tested to be working despite looking ugly...
    # TODO: maybe check (on unflipped data) fftshift, followed by transpose - that could work. fftshift followed by flip
    # was having an offset of 1
    # TODO: check nomenclature: flcc, flcf, flcl
    cen = np.floor(box_size / 2).astype(int) + 1
    cc_map = np.flip(cc_map)
    cc_map = np.roll(cc_map, cen, (0, 1, 2))

    return np.clip(cc_map, 0.0, 1.0)

def symmterize_volume(vol, symmetry): 
    """
    Symmetrize the input volume based on the specified symmetry.

    Parameters:
    vol (ndarray): The input volume to be symmetrized.
    symmetry (str or int or float): The symmetry of the volume. If a string, it should start with 'C' followed by a number indicating the rotational symmetry. If an integer or float, it directly specifies the rotational symmetry.

    Returns:
    ndarray: The symmetrized volume.

    Raises:
    ValueError: If the symmetry is not specified correctly.

    """
    if isinstance(symmetry, str):
        nfold = int(re.findall(r"\d+", symmetry)[-1])
    elif isinstance(symmetry, (int, float)):
        nfold = symmetry
    else:
        raise ValueError("The symmetry has to be specified as a string (starting with C) or as a number (only for C)!")

    inplane_step = 360 / nfold
    rotated_sum = np.empty(vol.shape)

    for inplane in range(1, nfold+1):
        # print('inplane',inplane, inplane*inplane_step)
        rotated_volume = rotate(vol, rotation_angles=[0, 0, 360%(inplane*inplane_step)])
        # print('rot vol',rotated_volume[0][0][0:10])
        rotated_sum = np.add(rotated_sum, rotated_volume)
    sym_vol = np.divide(rotated_sum, nfold)

    return sym_vol