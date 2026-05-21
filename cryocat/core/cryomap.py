import h5py
import emfile
import mrcfile
import re
import os
import warnings
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as srot
from typing import TYPE_CHECKING
from cryocat.utils import ioutils
from cryocat.utils import geom
from cryocat.utils import imageutils
from skimage import transform
from cryocat._types import MapSource, PathOrStr, TripletLike, EulerAngles, Symmetry, ArrayLike

if TYPE_CHECKING:
    from cryocat.core import cryomotl


def scale(
    input_map: MapSource,
    scaling_factor: float,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """
    Scales an input map by a given scaling factor.

    Parameters
    ----------
    input_map : MapSource
        The input map to be scaled, either as an ndarray or a path to a map file
        (``.mrc``, ``.em``, ...). Normalized via :func:`read`.
    scaling_factor : float
        The factor by which to scale the input map (a value grater than one indicates upscaling, between 0 and downscaling). If the scaling factor is greater than 1, anti-aliasing is turned off.
    output_path : PathOrStr, optional
        Path to the output file to which to write the scaled map. If not provided,
        the scaled map will not be saved. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

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

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)

    if scaling_factor > 1:
        anti_alias = False
    else:
        anti_alias = True

    scaled_map = transform.rescale(input_map, scaling_factor, order=3, mode="reflect", anti_aliasing=anti_alias)

    if scaled_map.dtype == np.float64:
        scaled_map = scaled_map.astype(np.float32)

    if output_path is not None:
        write(scaled_map, output_path, **output_kwargs)

    return scaled_map


def pixels2resolution(
    fourier_pixels: int,
    edge_size: int,
    pixel_size: float,
    print_out: bool = True,
) -> float:
    """Calculate the resolution in Angstroms based on Fourier pixel count, edge size, and pixel size.

    Parameters
    ----------
    fourier_pixels : int
        Number of pixels in the Fourier space.
    edge_size : int
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


def resolution2pixels(
    resolution: float,
    edge_size: int,
    pixel_size: float,
    print_out: bool = True,
) -> int:
    """Calculate the number of Fourier pixels/voxels corresponding to a given resolution for a specific edge size and
    pixel size.

    Parameters
    ----------
    resolution : float
        The target resolution to convert to Fourier pixels/voxels.
    edge_size : int
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


def binarize(input_map: MapSource, threshold: float = 0.5) -> np.ndarray:
    """Converts a given input map to a binary map based on a specified threshold.

    Parameters
    ----------
    input_map : MapSource
        The input map to binarize, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
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
    return imageutils.binarize(input_map, threshold)


def get_filter_radius(
    edge_size: int,
    fourier_pixels: int | None,
    target_resolution: float | None,
    pixel_size: float | None,
) -> int:
    """Calculate the filter radius — delegates to :func:`imageutils.get_filter_radius`."""
    return imageutils.get_filter_radius(edge_size, fourier_pixels, target_resolution, pixel_size)


def bandpass(
    input_map: MapSource,
    lp_fourier_pixels: int | None = None,
    lp_target_resolution: float | None = None,
    hp_fourier_pixels: int | None = None,
    hp_target_resolution: float | None = None,
    pixel_size: float | None = None,
    lp_gaussian: int = 3,
    hp_gaussian: int = 2,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """Apply a bandpass filter to an input map using specified low-pass and high-pass filter parameters.

    Parameters
    ----------
    input_map : MapSource
        The input map to be filtered, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
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
    output_path : PathOrStr, optional
        Path to the output file to save the filtered result. If not provided, the
        filtered map is not saved. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

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

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

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
    bandpass_filtered = imageutils.apply_bandpass(
        input_map, lp_radius, hp_radius, lp_gaussian=lp_gaussian, hp_gaussian=hp_gaussian
    )

    if output_path is not None:
        write(bandpass_filtered, output_path, **{"data_type": np.single, **output_kwargs})

    return bandpass_filtered


def lowpass(
    input_map: MapSource,
    fourier_pixels: int | None = None,
    target_resolution: float | None = None,
    pixel_size: float | None = None,
    gaussian: int = 3,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """Apply a lowpass filter to a given input map using Fourier transform methods.

    Parameters
    ----------
    input_map : MapSource
        The input map to be filtered, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    fourier_pixels : int, optional
        Number of pixels/voxels in the Fourier space representation. Default is None.
    target_resolution : float, optional
        The target resolution in Angstroms for the filtering process. Default is None.
    pixel_size : float, optional
        The size of each pixel/voxel in the input map in Angstroms.
    gaussian : int, default=3
        Width of the Gaussian falloff for the low-pass filter. Default is 3.
    output_path : PathOrStr, optional
        Path to the output file to save the filtered map. If not provided, the map
        is not saved. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

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

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)
    radius = get_filter_radius(
        input_map.shape[0], fourier_pixels=fourier_pixels, target_resolution=target_resolution, pixel_size=pixel_size
    )

    filtered_map = imageutils.apply_lowpass(input_map, radius, gaussian=gaussian)

    if output_path is not None:
        write(filtered_map, output_path, **{"data_type": np.single, **output_kwargs})

    return filtered_map


def highpass(
    input_map: MapSource,
    fourier_pixels: int | None = None,
    target_resolution: float | None = None,
    pixel_size: float | None = None,
    gaussian: int = 2,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """Apply a highpass filter to a given input map using Fourier transform methods.

    Parameters
    ----------
    input_map : MapSource
        The input map to be filtered, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    fourier_pixels : int, optional
        Number of pixels/voxels to use in the Fourier space. Default is None.
    target_resolution : float, optional
        The target resolution in Angstroms for the highpass filter. Default is None.
    pixel_size : float, optional
        The size of each pixel/voxel in the input map in Angstroms. Default is None.
    gaussian : int, default=2
        The width of the Gaussian fall-off in pixels/voxels. Default is 2.
    output_path : PathOrStr, optional
        Path to the output file to save the filtered result. If None, the filtered
        map is not saved. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    filtered_map : ndarray
        The highpass filtered map as a numpy array.

    Notes
    -----
    The function reads an input map, calculates the necessary filter radius based on the provided parameters,
    applies a spherical highpass filter in Fourier space, and optionally saves the result to a file.
    """

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)
    radius = get_filter_radius(
        input_map.shape[0], fourier_pixels=fourier_pixels, target_resolution=target_resolution, pixel_size=pixel_size
    )

    filtered_map = imageutils.apply_highpass(input_map, radius, gaussian=gaussian)

    if output_path is not None:
        write(filtered_map, output_path, **{"data_type": np.single, **output_kwargs})

    return filtered_map


def read(
    input_map: MapSource,
    transpose: bool = True,
    data_type: np.dtype | None = None,
) -> np.ndarray:
    """Read a volume from disk or pass through an in-memory ndarray.

    Canonical normalizer for the :data:`cryocat._types.MapSource` type.
    Any function that accepts a MapSource should call ``cryomap.read(x)``
    at its entry point to get a plain ndarray to work with.

    Parameters
    ----------
    input_map : MapSource
        Path to an MRC/EM file (str, :class:`pathlib.Path`, or any
        :class:`os.PathLike`) or an ndarray containing the map data.
    transpose : bool, optional
        Whether to transpose loaded file data with ``(2, 1, 0)``. Has no
        effect on inputs that are already ndarrays. Default is True.
    data_type : numpy.dtype, optional
        Cast the result to this dtype. If None, the dtype is preserved.

    Returns
    -------
    numpy.ndarray
        Map data as an ndarray.

    Raises
    ------
    ValueError
        If the file extension is unsupported, or the input is neither a
        path nor an ndarray.

    Notes
    -----
    Supported file extensions: ``.mrc``, ``.rec``, ``.st``, ``.ali``,
    ``.em``. Numbered suffixes such as ``.mrc.5`` are accepted.

    Examples
    --------
    >>> data = read("map.mrc")
    >>> data = read(Path("map.em"), transpose=False, data_type=np.float32)
    >>> data = read(np.random.rand(10, 10, 10))
    """
    if isinstance(input_map, (str, os.PathLike)):
        path_str = os.fspath(input_map)

        def valid_mrc(input_path: str) -> bool:
            pattern = r"\.(mrc|ali|rec|st)(\.\d+)?$"
            return bool(re.search(pattern, input_path))

        if valid_mrc(path_str):
            data = mrcfile.open(path_str).data
        elif path_str.endswith(".em"):
            data = emfile.read(path_str)[1]
        else:
            raise ValueError(
                f"Unsupported file extension for input_map={path_str!r}. " "Expected .mrc, .ali, .rec, .st, or .em."
            )

        if transpose:
            data = data.transpose(2, 1, 0)
    elif isinstance(input_map, np.ndarray):
        data = np.array(input_map)
    else:
        raise ValueError(f"input_map must be a path or an ndarray, got {type(input_map).__name__}.")

    data = np.array(data, copy=True)
    if data_type is not None:
        data = data.astype(data_type)

    return data


def get_metadata(
    input_map: MapSource,
) -> tuple[tuple[int, ...], float, tuple[float, float, float]]:
    """Return MRC/EM header metadata without loading the data array.

    Parameters
    ----------
    input_map : MapSource
        Path to an MRC or EM file, or an ndarray. For ndarrays ``pixel_size_a``
        defaults to ``1.0`` and ``origin_a`` to ``(0.0, 0.0, 0.0)``.

    Returns
    -------
    shape : tuple[int, ...]
        Array dimensions (x, y, z) read from the file header. For MRC files
        no data is copied; for EM files the data is loaded (no header-only API).
        Shape is consistent with ``cryomap.read(path).shape``.
    pixel_size_a : float
        Pixel size in Ångströms (x component from MRC header).
        Defaults to ``1.0`` for EM files and ndarrays.
    origin_a : tuple[float, float, float]
        Origin (x, y, z) in Ångströms from MRC header.
        Defaults to ``(0.0, 0.0, 0.0)`` for EM files and ndarrays.

    Raises
    ------
    ValueError
        If the file extension is unsupported, or the input is neither a
        path nor an ndarray.
    """
    if isinstance(input_map, (str, os.PathLike)):
        path_str = os.fspath(input_map)

        if bool(re.search(r"\.(mrc|ali|rec|st)(\.\d+)?$", path_str)):
            with mrcfile.open(path_str, permissive=True) as mrc:
                shape = (int(mrc.header.nx), int(mrc.header.ny), int(mrc.header.nz))
                pixel_size_a = float(mrc.voxel_size.x)
                origin_a = (
                    float(mrc.header.origin.x),
                    float(mrc.header.origin.y),
                    float(mrc.header.origin.z),
                )
        elif path_str.endswith(".em"):
            data = np.array(emfile.read(path_str)[1])
            shape = data.shape
            pixel_size_a = 1.0
            origin_a = (0.0, 0.0, 0.0)
        else:
            raise ValueError(
                f"Unsupported file extension for input_map={path_str!r}. " "Expected .mrc, .ali, .rec, .st, or .em."
            )
    elif isinstance(input_map, np.ndarray):
        shape = input_map.shape
        pixel_size_a = 1.0
        origin_a = (0.0, 0.0, 0.0)
    else:
        raise ValueError(f"input_map must be a path or an ndarray, got {type(input_map).__name__}.")

    return shape, pixel_size_a, origin_a


def write(
    data_to_write: np.ndarray,
    output_path: PathOrStr,
    transpose: bool = True,
    data_type: np.dtype | None = None,
    pixel_size: float = 1.0,
    overwrite: bool = True,
) -> None:
    """Write data to a specified file in a given format.

    Parameters
    ----------
    data_to_write : numpy.ndarray
        The data array to be written to the file. It can be of any shape and type.
    output_path : PathOrStr
        Path to the output file (str, :class:`pathlib.Path`, or any
        :class:`os.PathLike`). The file extension must be one of: ``.mrc``,
        ``.rec``, or ``.em``.
    transpose : bool, default=True
        If True (default), the data will be transposed before writing. The transposition
        will change the order of the axes to (2, 1, 0). Default is True.
    data_type : type, optional
        If specified, the data will be cast to this type before writing. If None (default),
        the original data type will be used.
    pixel_size : float, default=1.0
        Pixel size in Angstroms to store in the header of MRC files. Note that .em files
        do not store this value in the header and it is therefore ignored. Defaults to 1.0.
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

    if not isinstance(output_path, (str, os.PathLike)):
        raise ValueError(f"output_path must be a path or str, got {type(output_path).__name__}.")
    file_name_str = os.fspath(output_path)

    if data_type is not None:
        data_to_write = data_to_write.astype(data_type)

    if transpose and data_to_write.ndim == 3:
        data_to_write = data_to_write.transpose(2, 1, 0)

    if data_to_write.dtype == np.float64:
        data_to_write = data_to_write.astype(np.float32)

    if file_name_str.endswith(".mrc") or file_name_str.endswith(".rec"):
        mrcfile.write(name=file_name_str, data=data_to_write, overwrite=overwrite, voxel_size=pixel_size)
    elif file_name_str.endswith(".em"):
        emfile.write(file_name_str, data=data_to_write, overwrite=overwrite)
    else:
        raise ValueError("The output file name", file_name_str, "has to end with .mrc, .rec or .em!")


def invert_contrast(input_map: MapSource, output_path: PathOrStr | None = None, **output_kwargs) -> np.ndarray:
    """Invert the contrast of an input volume map.

    Parameters
    ----------
    input_map : MapSource
        The input volume map, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    output_path : PathOrStr, optional
        Path to the output file where the inverted volume map will be saved.
        If not provided, the output will not be saved to a file. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

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

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)
    inverted_map = input_map * (-1)

    if output_path is not None:
        if inverted_map.dtype == np.float64:
            data_type = np.single
        else:
            data_type = inverted_map.dtype

        write(inverted_map, output_path, **{"data_type": data_type, **output_kwargs})

    return inverted_map


def em2mrc(
    input_path: PathOrStr,
    invert: bool = False,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> None:
    """Convert a file in EM format to MRC format.

    Parameters
    ----------
    input_path : PathOrStr
        Path to the input map file to be converted (must have ``.em`` extension).
    invert : bool, default=False
        If True, the data will be inverted (multiplied by -1). Default is False.
    output_path : PathOrStr, optional
        Path to the output MRC file. If None, the output name will be derived from
        ``input_path`` by replacing the ``.em`` extension with ``.mrc``.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    None
        The function writes the converted data to the specified output file.

    Raises
    -------
    ValueError
        If ``input_path`` is not a valid .em file path

    """
    if not isinstance(input_path, (str, os.PathLike)):
        raise ValueError(f"Input file must be a path or str")
    input_path_str = os.fspath(input_path)
    if not input_path_str.endswith(".em"):
        raise ValueError(f"Provided path must be .em file")
    data_to_write = read(input_path_str)

    if invert:
        data_to_write = data_to_write * (-1)

    if output_path is None:
        output_path_str = input_path_str[:-2] + "mrc"
    else:
        if not isinstance(output_path, (str, os.PathLike)):
            raise ValueError(f"output_path must be a path or str")
        output_path_str = os.fspath(output_path)
        if not output_path_str.endswith(".mrc"):
            raise ValueError(f"Specified output file name must end with .mrc")
    write(data_to_write, output_path_str, **output_kwargs)


def mrc2em(
    input_path: PathOrStr,
    invert: bool = False,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> None:
    """Convert a file in MRC format to EM format.

    input_path : PathOrStr
        Path to the input map file to be converted (must have ``.mrc`` extension).
    invert : bool, default=False
        If True, the data will be inverted (multiplied by -1). Default is False.
    output_path : PathOrStr, optional
        Path to the output EM file. If None, the output name will be derived from
        ``input_path`` by replacing the ``.mrc`` extension with ``.em``.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    None
        The function writes the converted data to the specified output file.

    Raises
    -------
    ValueError
        If the provided file name does not end with .em extension.

    """
    if not isinstance(input_path, (str, os.PathLike)):
        raise ValueError(f"Input must be a path or str")
    input_path_str = os.fspath(input_path)
    if not input_path_str.endswith(".mrc"):
        raise ValueError(f"Input file is not .mrc file")
    data_to_write = read(input_path_str)

    if invert:
        data_to_write = data_to_write * (-1)

    if output_path is None:
        output_path_str = input_path_str[:-3] + "em"
    else:
        if not isinstance(output_path, (str, os.PathLike)):
            raise ValueError(f"output_path must be a path or str")
        output_path_str = os.fspath(output_path)
        if not output_path_str.endswith(".em"):
            raise ValueError(f"Specified output_path is not .em file")

    write(data_to_write, output_path_str, **output_kwargs)


def write_hdf5(
    input_path: PathOrStr,
    labels: PathOrStr | None = None,
    weight: PathOrStr | None = None,
    output_path: PathOrStr | None = None,
) -> None:
    """Write data to an HDF5 file.

    Parameters
    ----------
    input_path : PathOrStr
        Path to the input file containing the data to be written.
    labels : PathOrStr, optional
        Path to the input file containing the labels to be written. If provided,
        the labels will be stored in the HDF5 file. Default is None.
    weight : PathOrStr, optional
        Path to the input file containing the weights to be written. If provided,
        the weights will be stored in the HDF5 file. Default is None.
    output_path : PathOrStr, optional
        Path to the output HDF5 file. If not provided, the output file will be named
        by replacing the ``.mrc``/``.em`` extension of ``input_path`` with ``.hdf5``.
        Default is None.

    Returns
    -------
    None
        This function does not return any value. It writes the data, labels, and weights to the specified HDF5 file.

    Raises
    -------
    ValueError
        If the provided file name does not end with .em or .mrc extension.

    """
    if not isinstance(input_path, (str, os.PathLike)):
        raise ValueError(f"Input must be a path or str")
    input_path_str = os.fspath(input_path)
    if not input_path_str.endswith(".mrc") and not input_path_str.endswith(".em"):
        raise ValueError(f"Input file is not either .em either .mrc file")
    data_to_write = read(input_path_str)

    if output_path is None:
        if input_path_str.endswith(".mrc"):
            output_path_str = input_path_str[:-3] + "hdf5"
        else:
            output_path_str = input_path_str[:-2] + "hdf5"
    else:
        if not isinstance(output_path, (str, os.PathLike)):
            raise ValueError(f"output_path must be a path or str")
        output_path_str = os.fspath(output_path)
        if not output_path_str.endswith(".mrc") and not output_path_str.endswith(".em"):
            raise ValueError("Output file path must be .mrc or .em extension")

    f = h5py.File(output_path_str, "w")

    f.create_dataset("raw", data=data_to_write)

    if labels is not None:
        labels_to_write = read(labels)
        f.create_dataset("label", data=labels_to_write)
    if weight is not None:
        weight_to_write = read(weight)
        f.create_dataset("weight", data=weight_to_write)

    f.close()


def read_hdf5(
    input_path: PathOrStr,
    dataset_name: str = "predictions",
    print_datasets: bool = False,
) -> np.ndarray:
    """Read a dataset from an HDF5 file.

    Parameters
    ----------
    input_path : PathOrStr
        Path to the HDF5 file to read from.
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

    f = h5py.File(input_path, "r")

    if print_datasets:
        print(f"Available datasets: {f.keys()}")

    data = np.array(f[dataset_name][:])
    f.close()

    return data


def normalize(input_map: MapSource) -> np.ndarray:
    """Normalize a given map by standardizing its values (z-score).

    Only finite values are used to compute the mean and standard deviation, so
    the result is robust to NaN/Inf pixels common in real tomograms.

    Parameters
    ----------
    input_map : MapSource
        The input map to be normalized, either as an ndarray or a path to a map
        file. Normalized via :func:`read`.

    Returns
    -------
    numpy.ndarray
        The normalized map with zero mean and unit variance. Returns a copy of
        the original map (unmodified) if no finite values are found or if the
        standard deviation is zero.

    Examples
    --------
    >>> normalized_map = normalize(my_map)
    """

    return imageutils.normalize_array(read(input_map))


def sample_line_profiles(
    p1: ArrayLike,
    p2: ArrayLike,
    input_map: MapSource,
    pixel_size_a: float | None = None,
    extension_half_width_a: float = 80.0,
) -> list[dict]:
    """Sample intensities along line segments connecting paired 3D coordinates.

    For each pair ``(p1[i], p2[i])``, a line segment is built from the midpoint,
    extended by ``extension_half_width_a`` on each side. Intensities are sampled
    with linear interpolation (``scipy.ndimage.map_coordinates``, ``order=1``,
    ``mode='nearest'``). Pairs with NaN coordinates or zero-length vectors are
    skipped silently.

    Parameters
    ----------
    p1, p2 : ArrayLike, shape (N, 3)
        Paired 3D coordinates in voxel units, XYZ order. Any array-coercible
        input is accepted; normalized internally with ``np.asarray``.
    input_map : MapSource
        3D volume in ZYX order, or a path to an MRC/EM file.
    pixel_size_a : float, optional
        Voxel size in Angstroms. Used to convert ``extension_half_width_a`` to
        voxels and stored in the output dicts for downstream rescaling. When
        ``None``, ``extension_half_width_a`` is treated as already in voxels.
    extension_half_width_a : float, default 80.0
        How far to extend the sampled segment beyond the midpoint on each side,
        in Angstroms (default 80 Å = 8 nm). Treated as voxels when no
        ``pixel_size_a`` is given.

    Returns
    -------
    list of dict
        One dict per valid pair with keys:

        * ``profile`` — 1-D intensity array (sampled values)
        * ``p1``, ``p2``, ``midpoint``, ``start``, ``end`` — geometry in voxels
        * ``pixel_size_a`` — voxel size in Angstroms (only when provided)
    """
    from scipy.ndimage import map_coordinates

    volume = read(input_map)
    if volume.ndim != 3:
        raise ValueError(f"input_map must be a 3D array, got shape {volume.shape}")

    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    if p1.ndim != 2 or p1.shape[1] != 3 or p2.shape != p1.shape:
        raise ValueError("p1 and p2 must both be (N, 3) arrays")

    if pixel_size_a is not None and np.isfinite(pixel_size_a) and pixel_size_a > 0:
        half_width_vox = extension_half_width_a / pixel_size_a
    else:
        half_width_vox = extension_half_width_a
        pixel_size_a = None  # normalise sentinel

    # Filter NaN pairs
    valid = ~(np.isnan(p1).any(axis=1) | np.isnan(p2).any(axis=1))
    p1, p2 = p1[valid], p2[valid]
    if len(p1) == 0:
        return []

    directions = p2 - p1
    lengths = np.linalg.norm(directions, axis=1)

    nonzero = lengths > 0
    p1, p2 = p1[nonzero], p2[nonzero]
    directions, lengths = directions[nonzero], lengths[nonzero]
    if len(p1) == 0:
        return []

    unit_vectors = directions / lengths[:, np.newaxis]
    midpoints = (p1 + p2) / 2.0
    starts = midpoints - unit_vectors * half_width_vox
    ends = midpoints + unit_vectors * half_width_vox

    profiles = []
    for i in range(len(p1)):
        num_points = int(np.ceil(2 * half_width_vox + lengths[i])) + 1
        line_points = np.linspace(starts[i], ends[i], num=num_points)

        coords_zyx = line_points[:, [2, 1, 0]].T
        intensities = map_coordinates(volume, coords_zyx, order=1, mode="nearest")

        entry = {
            "profile": intensities,
            "p1": p1[i],
            "p2": p2[i],
            "midpoint": midpoints[i],
            "start": starts[i],
            "end": ends[i],
        }
        if pixel_size_a is not None:
            entry["pixel_size_a"] = pixel_size_a

        profiles.append(entry)

    return profiles


def rotate(
    input_map: MapSource,
    rotation: srot | None = None,
    rotation_angles: EulerAngles | None = None,
    coord_space: str = "zxz",
    transpose_rotation: bool = False,
    degrees: bool = True,
    spline_order: int = 3,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """Rotate a 3D input map using a specified rotation matrix or rotation angles.

    Parameters
    ----------
    input_map : MapSource
        The input 3D map to be rotated, either as an ndarray or a path to a map
        file. Normalized via :func:`read`.

    rotation : scipy.spatial.transform.Rotation, optional
        A rotation object representing the rotation to be applied. If provided,
        `rotation_angles` (if provided) will not be considered. Default is None.

    rotation_angles : EulerAngles, optional
        Angles for rotation in the specified coordinate space (degrees by default,
        see ``degrees``). If provided, they will be considered only if ``rotation``
        is not specified. Default is None.

    coord_space : str, default='zxz'
        The coordinate space for the rotation angles. Default is 'zxz'.

    transpose_rotation : bool, default=False
        If True, the transpose of the rotation matrix will be used. Default is False.

    degrees : bool, default=True
        If True, the rotation angles are interpreted as degrees. Default is True.

    spline_order : int, default=3
        The order of the spline used for interpolation. Default is 3.

    output_path : PathOrStr, optional
        Path to the output file for the rotated structure. If not provided, the
        result is not saved. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    rot_struct : numpy.ndarray
        The rotated 3D map.

    Raises
    ------
    ValueError
        If neither `rotation` nor `rotation_angles` is specified.
    """

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)
    rot_struct = imageutils.rotate_volume(
        input_map,
        rotation=rotation,
        rotation_angles=rotation_angles,
        coord_space=coord_space,
        transpose_rotation=transpose_rotation,
        degrees=degrees,
        spline_order=spline_order,
    )

    if output_path is not None:
        write(rot_struct, output_path, **{"data_type": np.single, **output_kwargs})

    return rot_struct


def crop(
    input_map: MapSource,
    new_size: TripletLike,
    output_path: PathOrStr | None = None,
    crop_coord: TripletLike | None = None,
    **output_kwargs,
) -> np.ndarray:
    """
    This function crops a given input map to a new size. If no crop coordinates are provided, the function will crop from the center of the input map. If an output file is specified, the cropped volume will be written to this file.

    Parameters
    ----------
    input_map : MapSource
        The input map to be cropped, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    new_size : TripletLike
        The desired size of the cropped volume (single int broadcast to all axes,
        or a 3-element array-like).
    output_path : PathOrStr, optional
        Path to the output file where the cropped volume will be written. If not
        provided, the cropped volume will not be written to a file.
    crop_coord : TripletLike, optional
        The (integer) center coordinates of the crop. If not provided, the function
        will crop from the center of the input map.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    cropped_volume : np.array
        The cropped volume.

    Notes
    -----
    see also: trim
    """
    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)

    new_size = geom.as_triplet(new_size)

    if crop_coord is None:
        crop_coord = geom.as_triplet(input_map.shape) // 2
    else:
        crop_coord = geom.as_triplet(crop_coord)

    vs, ve, _, _ = get_start_end_indices(crop_coord, input_map.shape, new_size)

    # print(vs[0], ve[0], vs[1], ve[1], vs[2], ve[2])
    cropped_volume = input_map[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]

    if output_path is not None:
        write(cropped_volume, output_path, **{"data_type": np.single, **output_kwargs})

    return cropped_volume


def shift(
    input_map: MapSource,
    delta: ArrayLike,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """
    Shifts an input map by a specified delta.

    Parameters
    ----------
    input_map : MapSource
        The input map to be shifted, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    delta : ArrayLike
        The shift to apply to the input map (3-element array-like, one shift per
        axis). Normalized via :func:`numpy.asarray`.
    output_path : PathOrStr, optional
        Path to the output file for the shifted map. The data type of the output
        will be np.single. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    shifted_map : np.ndarray
        The input map, shifted by the specified delta.

    Notes
    -----
    The shift is applied using an affine transformation with a matrix that is the identity except for the last column, which is set to the negative of the delta. The mode of the affine transformation is "grid-wrap".
    """

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)
    shifted_map = imageutils.shift_array(input_map, np.asarray(delta))

    if output_path is not None:
        write(shifted_map, output_path, **{"data_type": np.single, **output_kwargs})

    return shifted_map


def recenter(input_map: MapSource, new_center: TripletLike) -> np.ndarray:
    """
    Recenter a given map around a new center.

    Parameters
    ----------
    input_map : MapSource
        The input map to be recentered, either as an ndarray or a path to a map
        file. Normalized via :func:`read`.
    new_center : TripletLike
        The new center coordinates for the map, with relation to the coordinate
        frame of the map (e.g. box size). Accepts a single int (broadcast to all
        three axes) or a 3-element array-like; normalized to a length-3 ndarray
        via :func:`cryocat.utils.geom.as_triplet`.

    Returns
    -------
    trans_struct : ndarray
        The recentered map.

    Notes
    -----
    The function creates a transformation matrix, calculates the shift required to move the center of the map to the new center, applies the transformation to the map, and returns the recentered map.
    """

    original_map = read(input_map)
    new_center = geom.as_triplet(new_center)
    return imageutils.recenter_volume(original_map, new_center)


def normalize_under_mask(input_map: MapSource, input_mask: MapSource) -> np.ndarray:
    """A function to take a reference volume and a mask, and normalize the area
    under the mask to 0-mean and standard deviation of 1.
    Based on stopgap code by W.Wan

    Parameters
    ----------
    input_map : MapSource
        The reference volume, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    input_mask : MapSource
        The mask volume, either as an ndarray or a path to a mask file.
        Normalized via :func:`read`.


    Returns
    -------
    norm_ref : np.ndarray
        The map with the area under the mask normalized.
    """

    return imageutils.normalize_under_mask(read(input_map), read(input_mask))


def get_start_end_indices(
    coord: TripletLike,
    volume_shape: TripletLike,
    subvolume_shape: TripletLike,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    This function calculates the start and end indices of a subvolume within a larger volume, given the center
    coordinate of the subvolume.

    Parameters
    ----------
    coord : TripletLike
        The (integer) center coordinate of the subvolume (3-element array-like).
    volume_shape : TripletLike
        The shape of the larger volume (3-element array-like).
    subvolume_shape : TripletLike
        The shape of the subvolume (3-element array-like).

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
    The function first calculates the start and end indices of the subvolume within the larger volume, without
    considering whether they are within the volume. It then clips these indices to ensure they are within the volume.
    Finally, it calculates the start and end indices of the subvolume, relative to its own shape.
    """

    subvolume_shape = np.asarray(subvolume_shape)
    subvolume_half = subvolume_shape / 2

    volume_start = np.floor(coord - subvolume_half).astype(int)
    volume_end = (volume_start + subvolume_shape).astype(int)

    volume_start_clip = np.minimum(np.maximum([0, 0, 0], volume_start), np.asarray(volume_shape))
    volume_end_clip = np.maximum(np.minimum(np.asarray(volume_shape), volume_end), [0, 0, 0])

    subvolume_start = volume_start_clip - volume_start
    subvolume_end = volume_end - volume_start
    subvolume_end = volume_end_clip - volume_end + subvolume_end

    subvolume_start = np.minimum(np.maximum([0, 0, 0], subvolume_start), subvolume_shape)
    subvolume_end = np.maximum(np.minimum(subvolume_shape, subvolume_end), [0, 0, 0])

    return volume_start_clip, volume_end_clip, subvolume_start, subvolume_end


def extract_subvolume(
    volume: np.ndarray,
    coordinates: TripletLike,
    subvolume_shape: TripletLike,
    enforce_shape: bool = False,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """
    Extracts a subvolume from a given volume.

    Parameters
    ----------
    volume : ndarray
        The 3D array from which to extract the subvolume.
    coordinates : TripletLike
        The (x, y, z) coordinates of the center of the subvolume to extract.
    subvolume_shape : TripletLike
        The (x, y, z) shape of the subvolume to extract.
    enforce_shape : bool, optional
        If True, the final volume will have the same shape as the original volume and the voxels outside the region of interest will be set to the mean value of the original volume. Default is False.
    output_path : PathOrStr, optional
        Path to the output file for the extracted subvolume. If not provided, the
        subvolume is not saved. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

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

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    vs, ve, ss, se = get_start_end_indices(coordinates, volume.shape, subvolume_shape)
    if enforce_shape is not False:
        subvolume = np.full(volume.shape, np.mean(volume))
        subvolume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]] = volume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]
    else:
        subvolume = np.full(subvolume_shape, np.mean(volume))
        subvolume[ss[0] : se[0], ss[1] : se[1], ss[2] : se[2]] = volume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]

    if output_path is not None:
        write(subvolume, output_path, **{"data_type": np.single, **output_kwargs})

    return subvolume


def get_cross_slices(
    input_map: MapSource,
    slice_half_dim: int | None = None,
    slice_numbers: ArrayLike | None = None,
    axis: ArrayLike | None = None,
) -> list[np.ndarray]:
    """
    This function generates cross slices across an axis from a given input map.

    Parameters
    ----------
    input_map : MapSource
        The input map from which to generate cross slices, either as an ndarray or
        a path to a map file. Normalized via :func:`read`.
    slice_half_dim : int, optional
        The half dimension of the slice. If None, the slice will cover the entire dimension of the input map.
    slice_numbers : ArrayLike, optional
        The slice numbers to use (scalar, list, tuple, or ndarray; normalized via
        :func:`numpy.asarray`). If None, the slice numbers will be calculated as
        the ceiling of half the shape of the input map.
    axis : ArrayLike, optional
        The axis (or axes) along which to generate the slices (scalar, list, tuple,
        or ndarray; normalized via :func:`numpy.asarray`). If None, slices will be
        generated along all axes.

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
    else:
        axis = np.atleast_1d(np.asarray(axis))

    if slice_numbers is None:
        cs = np.ceil(np.asarray(cmap.shape) / 2).astype(int)
    else:
        cs = np.atleast_1d(np.asarray(slice_numbers)).astype(int)

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


def pad(
    input_map: MapSource,
    new_size: TripletLike,
    fill_value: float | None = None,
) -> np.ndarray:
    """
    Pads an input volume to a new size.

    This function reads an input volume, calculates the mean of the volume if no fill value is provided, and creates a new volume of the specified size filled with the mean or provided fill value. The original volume is then placed in the center of the new volume.

    Parameters
    ----------
    input_map : MapSource
        The input volume to be padded, either as an ndarray or a path to a map
        file. Normalized via :func:`read`.
    new_size : TripletLike
        The desired size of the new volume (single int broadcast to all axes, or
        a 3-element array-like). Normalized to a length-3 ndarray via
        :func:`cryocat.utils.geom.as_triplet`.
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

    volume = read(input_map)
    new_size = geom.as_triplet(new_size)
    return imageutils.pad_volume(volume, new_size, fill_value=fill_value)


def place_object(
    input_object: MapSource | list[MapSource],
    motl: "cryomotl.Motl",
    volume_shape: TripletLike | None = None,
    volume: MapSource | None = None,
    feature_to_color: str = "object_id",
) -> np.ndarray:
    """
    Places an object or a list of objects into a volume based on the given motion list (motl).

    Parameters
    ----------
    input_object : MapSource or list of MapSource
        The object or list of objects to be placed, each either as an ndarray or
        a path to a map file. Normalized via :func:`read`.
    motl : cryomotl.Motl
        The motion list (Motl instance) containing the rotations and coordinates
        for placing the objects.
    volume_shape : TripletLike, optional
        The shape of the volume in which the objects are to be placed. If not provided, the volume parameter must be provided.
    volume : MapSource, optional
        The volume in which the objects are to be placed, as an ndarray or a path
        to a map file. If not provided, ``volume_shape`` must be provided.
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
    input_map: MapSource,
    pixel_size_a: float,
    defocus: float,
    snr_falloff: float,
    deconv_strength: float,
    highpass_nyquist: float,
    phase_flipped: bool = False,
    phaseshift: float = 0,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """Deconvolution adapted from MATLAB script tom_deconv_tomo by D. Tegunov (https://github.com/dtegunov/tom_deconv).
    Example for usage: deconvolve(my_map, 3.42, 6, 1.1, 1, 0.02, false, 0)

    Parameters
    ----------
    input_map : MapSource
        Tomogram volume, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
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
    phase_flipped : bool, default=False
        whether the data are already phase-flipped. Default is False.
    phaseshift : float, default=0
        CTF phase shift in degrees (e. g. from a phase plate). Default is 0.
    output_path : PathOrStr, optional
        Path to the output file for the deconvolved tomogram. Default is None
        (tomogram will not be written).
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    deconvolved_map : np.array
        deconvolved tomogram

    """
    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    input_map = read(input_map)
    interp_dim = np.maximum(2048, input_map.shape[0])
    wiener = imageutils.compute_wiener_1d(
        interp_dim,
        pixel_size_a,
        defocus,
        snr_falloff,
        deconv_strength,
        highpass_nyquist,
        phase_flipped,
        phaseshift,
    )
    deconvolved_map = imageutils.apply_wiener_radial(input_map, wiener, interp_dim)

    if output_path is not None:
        write(deconvolved_map, output_path, **{"data_type": np.single, "pixel_size": pixel_size_a, **output_kwargs})

    return deconvolved_map


def trim(
    input_map: MapSource,
    trim_start: TripletLike,
    trim_end: TripletLike,
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """
    Trims a 3D map to a specified range.

    Parameters
    ----------
    input_map : MapSource
        The 3D map to be trimmed, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    trim_start : TripletLike
        The starting coordinates for the trim (3-element array-like).
    trim_end : TripletLike
        The ending coordinates for the trim (3-element array-like).
    output_path : PathOrStr, optional
        Path to the output file for the trimmed map. The file will be written in
        single precision float format. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

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

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    output_map = read(input_map)

    trim_start = np.asarray(trim_start)
    trim_end = np.asarray(trim_end)

    ts = np.maximum(trim_start, np.zeros((3,))).astype(int)
    te = np.minimum(trim_end, np.asarray(output_map.shape)).astype(int)

    output_map = output_map[ts[0] : te[0], ts[1] : te[1], ts[2] : te[2]]

    if output_path is not None:
        write(output_map, output_path, **{"data_type": np.single, **output_kwargs})

    return output_map


def flip(
    input_map: MapSource,
    axis: str = "z",
    output_path: PathOrStr | None = None,
    **output_kwargs,
) -> np.ndarray:
    """
    Function to flip a given input map along specified axis.

    Parameters
    ----------
    input_map : MapSource
        The input map to be flipped, either as an ndarray or a path to a map file.
        Normalized via :func:`read`.
    axis : str, default="z"
        The axis (or axes) along which to flip the input map; any combination of
        ``"x"``, ``"y"``, ``"z"`` (case-insensitive). Default is ``"z"``.
    output_path : PathOrStr, optional
        Path to the output file. If not provided, the function will only return
        the flipped map. Default is None.
    **output_kwargs
        Forwarded to :func:`cryocat.core.cryomap.write` when ``output_path`` is provided.
        See that function for the available parameters (``pixel_size``, ``transpose``, ``data_type``, ``overwrite``).

    Returns
    -------
    output_map : array_like
        The flipped map.

    Notes
    -----
    The function reads the input map, flips it along the specified axis, and writes the output map to a file if an output name is provided.
    """

    if output_path is None and output_kwargs:
        raise ValueError(
            f"Got output kwargs {list(output_kwargs)} but no output_path. " f"These only apply when writing to disk."
        )

    output_map = imageutils.flip_array(read(input_map), axis=axis)

    if output_path is not None:
        write(output_map, output_path, **{"data_type": np.single, **output_kwargs})

    return output_map


def symmetrize_volume(input_map: MapSource, symmetry: Symmetry) -> np.ndarray:
    """
    Symmetrize the input volume based on the specified symmetry.

    Parameters:
    -----------
    input_map : MapSource
        The input volume to be symmetrized, either as an ndarray or a path to a
        map file. Normalized via :func:`read`.
    symmetry : Symmetry
        The point-group symmetry specifier. Accepts a string like ``"C5"`` or a
        bare number (interpreted as the order of the cyclic symmetry).

    Returns:
    --------
    ndarray: The symmetrized volume.

    Raises:
    -------
    ValueError
        If the symmetry is not specified correctly

    """
    return imageutils.symmetrize_volume(read(input_map), symmetry)


def calculate_masked_fsc(
    input_map_even: MapSource,
    input_map_odd: MapSource,
    pixel_size: float | None = None,
    input_mask: MapSource | None = None,
    n_repeats: int = 10,
    fourier_cutoff: int | None = None,
    output_path: PathOrStr | None = None,
) -> pd.DataFrame:
    """
        Calculate phase-randomisation corrected FSC between two half-maps.

        Implements the masked-corrected FSC procedure of Chen et al. (2013,
        Ultramicroscopy 142:18-25): phases of the *unmasked* half-maps are
        randomised beyond ``fourier_cutoff``, the mask is applied in real space
        after the inverse FFT, and the resulting noise-floor FSC is used to
        correct the masked FSC.

        Parameters
        ----------
        input_map_even : MapSource
            First half-map: file path or 3-D ndarray. Normalized via :func:`read`.
        input_map_odd : MapSource
            Second half-map: file path or 3-D ndarray. Normalized via :func:`read`.
        pixel_size : float, optional
            Pixel size in Angstroms. When given, the x-axis is expressed as
            spatial frequency in 1/Ã
    ; otherwise the Fourier shell index is used.
        input_mask : MapSource, optional
            Real-space mask, either as an ndarray or a path to a mask file. A
            box-filling mask of ones is used when None. Default is None.
        n_repeats : int, default=10
            Number of phase-randomisation repeats (default 10).  Set to 0 or
            None to skip phase-randomisation correction.
        fourier_cutoff : int, optional
            Shell index beyond which phases are randomised.  Determined from
            box size automatically when None (box//10 for box<100, box//15 for
            box<210, 15 otherwise).
        output_path : PathOrStr, optional
            File path for saving results.  Extension selects format:
            ``.csv`` â comma-separated table; ``.xml`` â ChimeraX-compatible XML.
            The best available FSC column (corrected or uncorrected) is written.

        Returns
        -------
        pandas.DataFrame
            Columns: ``x``, ``uncorrected_fsc`` and, when phase randomisation
            is performed, ``corrected_fsc`` and ``mean_phase_fsc``.

        Raises
        ------
        ValueError
            If half-map shapes do not match, volumes are not cubic 3-D arrays,
            the mask shape differs from the half-maps, or an unsupported output
            extension is provided.
    """
    map_a = read(input_map_even)
    map_b = read(input_map_odd)

    if map_a.shape != map_b.shape:
        raise ValueError("Half-maps must have the same shape.")
    if map_a.ndim != 3 or len(set(map_a.shape)) != 1:
        raise ValueError("Only cubic 3-D volumes are supported.")

    box = map_a.shape[0]
    max_shell = box // 2

    if input_mask is None:
        fsc_mask = np.ones(map_a.shape, dtype=np.float32)
    else:
        fsc_mask = read(input_mask)
        if fsc_mask.shape != map_a.shape:
            raise ValueError("Mask shape does not match half-map shape.")

    # Radial distance in Fourier pixels
    dist = imageutils.compute_frequency_array(map_a.shape, 1) * box
    shell_masks = [(dist >= r - 1) & (dist < r) for r in range(1, max_shell + 1)]

    def _fsc_compute(ft_a, ft_b):
        cross = ft_a * np.conj(ft_b)
        pwr_a = (ft_a * np.conj(ft_a)).real
        pwr_b = (ft_b * np.conj(ft_b)).real
        fsc = np.zeros(max_shell)
        for i, smask in enumerate(shell_masks):
            num = np.sum(cross[smask]).real
            denom = np.sqrt(np.sum(pwr_a[smask]) * np.sum(pwr_b[smask]))
            fsc[i] = num / denom if denom > 0 else np.nan
        return fsc

    mft_a = np.fft.fftshift(np.fft.fftn(map_a * fsc_mask))
    mft_b = np.fft.fftshift(np.fft.fftn(map_b * fsc_mask))
    full_fsc = _fsc_compute(mft_a, mft_b)

    shells = np.arange(1, max_shell + 1)
    x_vals = shells / (box * float(pixel_size)) if pixel_size is not None else shells.astype(float)

    result = {"x": x_vals, "uncorrected_fsc": full_fsc}

    if n_repeats:
        if fourier_cutoff is None:
            if box < 100:
                fourier_cutoff = int(np.floor(box / 10))
            elif box < 210:
                fourier_cutoff = int(np.floor(box / 15))
            else:
                fourier_cutoff = 15

        pr_fsc_runs = np.zeros((n_repeats, max_shell))
        mean_pr_fsc = np.zeros(max_shell)

        for n in range(n_repeats):
            pr_a = imageutils.randomize_phases(map_a, fourier_cutoff) * fsc_mask
            pr_b = imageutils.randomize_phases(map_b, fourier_cutoff) * fsc_mask
            rp_fsc = _fsc_compute(
                np.fft.fftshift(np.fft.fftn(pr_a)),
                np.fft.fftshift(np.fft.fftn(pr_b)),
            )
            mean_pr_fsc += rp_fsc
            denom = 1.0 - rp_fsc
            with np.errstate(invalid="ignore", divide="ignore"):
                pr_fsc_runs[n] = np.where(denom != 0, (full_fsc - rp_fsc) / denom, np.nan)

        mean_pr_fsc /= n_repeats
        corr_fsc = np.nanmean(pr_fsc_runs, axis=0)
        corr_fsc[:fourier_cutoff] = full_fsc[:fourier_cutoff]

        result["corrected_fsc"] = corr_fsc
        result["mean_phase_fsc"] = mean_pr_fsc

    df = pd.DataFrame(result)

    if output_path is not None:
        y_col = "corrected_fsc" if "corrected_fsc" in df.columns else "uncorrected_fsc"
        ioutils.fsc_write(output_path, df["x"].values, df[y_col].values, pixel_size)

    return df
