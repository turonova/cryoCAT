import emfile
import mrcfile
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation as srot
from scipy.interpolate import interp1d


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
        if (
            input_map.endswith(".mrc")
            or input_map.endswith(".rec")
            or input_map.endswith(".st")
            or input_map.endswith(".ali")
        ):
            data = mrcfile.open(input_map).data
        elif input_map.endswith(".em"):
            data = emfile.read(input_map)[1]
        else:
            raise ValueError("The input map file name", input_map, "is neither em or mrc file!")

        if transpose:
            data = data.transpose(2, 1, 0)
    else:
        data = np.array(input_map)

    if data_type is not None:
        data = data.astype(data_type)

    return data


def write(data_to_write, file_name, transpose=True, data_type=None, overwrite=True):
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


def em2mrc(map_name, invert=False, overwrite=True, output_name=None):
    data_to_write = read(map_name)

    if invert:
        data_to_write = data_to_write * (-1)

    if output_name is None:
        output_name = map_name[:-2] + "mrc"

    write(data_to_write, output_name, overwrite=overwrite)


def mrc2em(map_name, invert=False, overwrite=True, output_name=None):
    data_to_write = read(map_name)

    if invert:
        data_to_write = data_to_write * (-1)

    if output_name is None:
        output_name = map_name[:-3] + "em"

    write(data_to_write, output_name, overwrite=overwrite)


def normalize(map):
    norm_map = read(map)

    mean_v = np.mean(norm_map)
    std_v = np.std(norm_map)
    norm_map = (norm_map - mean_v) / std_v

    return norm_map


def rotate(
    map,
    rotation=None,
    rotation_angles=None,
    coord_space="zxz",
    transpose_rotation=False,
    degrees=True,
    spline_order=3,
):
    # create transaltion to the center of the box
    T = np.eye(4)
    structure_center = np.asarray(map.shape) // 2
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

    rot_struct = np.empty(map.shape)
    affine_transform(input=map, output=rot_struct, matrix=final_matrix, order=spline_order)

    return rot_struct


def shift(map, delta):
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


def shift2(map, delta):
    T = np.eye(4)
    T[:3, -1] = -delta

    shifted_map = np.empty(map.shape)
    affine_transform(input=map, output=shifted_map, matrix=T, mode="grid-wrap")

    return shifted_map


def recenter(map, new_center):
    T = np.eye(4)
    structure_center = np.asarray(map.shape) // 2
    shift = new_center - structure_center
    T[:3, -1] = -shift

    trans_struct = np.empty(map.shape)
    affine_transform(input=map, output=trans_struct, matrix=T)

    return trans_struct


def normalize_under_mask(ref, mask):
    """A function to take a reference volume and a mask, and normalize the area
    under the mask to 0-mean and standard deviation of 1.
    Based on stopgap code by W.Wan

    Parameters
    ----------
    ref :

    mask :


    Returns
    -------

    """

    # Calculate mask parameteres
    m_idx = mask > 0

    # Calcualte stats
    ref_mean = np.mean(ref[m_idx])
    ref_std = np.std(ref[m_idx])

    # Normalize reference
    norm_ref = ref - ref_mean
    norm_ref = norm_ref / ref_std

    return norm_ref


def get_start_end_indices(coord, volume_shape, subvolume_shape):
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


def extract_subvolume(volume, coordinates, subvolume_shape, output_file=None):
    vs, ve, ss, se = get_start_end_indices(coordinates, volume.shape, subvolume_shape)

    subvolume = np.full(subvolume_shape, np.mean(volume))

    subvolume[ss[0] : se[0], ss[1] : se[1], ss[2] : se[2]] = volume[vs[0] : ve[0], vs[1] : ve[1], vs[2] : ve[2]]

    if output_file is not None:
        write(subvolume, output_file, data_type=np.single)

    return subvolume


def get_cross_slices(input_map, slice_half_dim=None, slice_numbers=None, axis=None):
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
    # TODO add input_object_load

    if volume is not None:
        object_container = read(volume)
    elif volume_shape is not None:
        object_container = np.zeros(volume_shape)

    rotations = motl.get_rotations()
    coordinates = motl.get_coordinates() - 1.0
    colors = motl.df[feature_to_color]

    for i, coord in enumerate(coordinates):
        object_map = rotate(input_object, rotation=rotations[i], transpose_rotation=True)
        object_map = np.where(object_map > 0.1, 1.0, 0.0)

        ls, le, os, oe = get_start_end_indices(coord, object_container.shape, input_object.shape)

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
        whether the data are already phase-flipped. Defaults to False.
    phaseshift : int
        CTF phase shift in degrees (e. g. from a phase plate). Defaults to 0.
    output_name : str
        Name of the output file for the deconvolved tomogram. Defaults to None (tomogram will be not written).

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
    # get the size of the box and number of voxels contributing to the calculations
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
    cen = np.floor(box_size / 2).astype(int) + 1
    cc_map = np.flip(cc_map)
    cc_map = np.roll(cc_map, cen, (0, 1, 2))

    return np.clip(cc_map, 0.0, 1.0)
