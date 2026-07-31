import os
import warnings

import numpy as np
import pandas as pd
from cryocat.utils import ioutils
from cryocat.utils import starfileio
from cryocat.utils import geom
from cryocat.utils import imageutils
from cryocat.core import cryomask
from cryocat.core import cryomap
from cryocat._types import (
    CTFFileType,
    DataSource,
    EulerAngles,
    MapSource,
    PathOrStr,
    TomoDimensions,
    TomoList,
    TripletLike,
    WedgeMaskMethod,
)
from cryocat.utils.classutils import gui_exposed
import emfile
import math


def check_data_consistency(
    data1: np.ndarray,
    data2: np.ndarray,
    data_type1: str,
    data_type2: str,
) -> None:
    """Check the consistency of two sets of data.

    Parameters
    ----------
    data1 : numpy.ndarray
        The first set of data.
    data2 : numpy.ndarray
        The second set of data.
    data_type1 : str
        The type of data in data1.
    data_type2 : str
        The type of data in data2.

    Raises
    ------
    TypeError
        If data1 or data2 isn't numpy.ndarray type.
    ValueError
        If the number of entries in data1 is different from the number of entries in data2.

    Returns
    -------
    None
        This function does not return anything.
    """
    if not isinstance(data1, np.ndarray):
        raise TypeError(f"Expected np.ndarray but got {type(data1)} for {data_type1} file.")
    if not isinstance(data2, np.ndarray):
        raise TypeError(f"Expected np.ndarray but got {type(data2)} for {data_type2} file.")

    # Check entire shape of both nparrays: could have same number of rows but different number of other dimensions
    if data1.shape != data2.shape:
        raise ValueError(f"The {data_type1} file has different number of entries than the {data_type2} file!")


def create_wedge_list_sg(
    tomo_id: int,
    tomo_dim: TomoDimensions,
    pixel_size: float,
    tlt_file: DataSource,
    z_shift: float = 0.0,
    ctf_file: DataSource | None = None,
    ctf_file_type: CTFFileType = "gctf",
    dose_file: DataSource | None = None,
    voltage: float = 300.0,
    amp_contrast: float = 0.07,
    cs: float = 2.7000,
    output_path: PathOrStr | None = None,
    drop_nan_columns: bool = True,
) -> pd.DataFrame:
    """Create a wedge list dataframe for a single tomogram/tilt series in STOPGAP format.

    Parameters
    ----------
    tomo_id : int
        The ID of the tomogram.
    tomo_dim : str or array-like
        The path to the tomogram dimensions file or dimensions specified as array-like variable. See
        :meth:`cryocat.utils.ioutils.dimensions_load` for more information on formatting.
    pixel_size : float
        The pixel size of the tomogram/tilt series.
    tlt_file : str or array-like
        The path to the file containing information on tilts (tlt, mdoc, xml) or tilt angles specified as array-like
        variable. See :meth:`cryocat.utils.ioutils.tlt_load` for more information on formatting.
    z_shift : str or array-like or int or float, default=0.0
        The path to the file containing information on z-shift (txt, com) or z-shift specified as array-like, pandas
        DataFrame, int or float. See :meth:`cryocat.utils.ioutils.z_shift_load` for more information on formatting.
    ctf_file : str or pandas.DataFrame or array-like, optional
        Either the path to the file with defocus values - either in gctf (star), ctffind4 (txt) or warp (xml) format or
        array like structure of size Nx5 (N is number of tilts), or pandas.DataFrame. See
        :meth:`cryocat.utils.ioutils.defocus_load` for more information on formatting.
    ctf_file_type : str, {"gctf", "ctffind4", "warp"}
        The type of the CTF file with defocus values. It can be either "gctf", "ctffind4", "warp", defaults to "gctf".
    dose_file : str or array-like, optional
        The path to the file containing information on corrected dose (csv, mdoc, txt, xml) or the corrected
        dose specified as array-like variable. See :meth:`cryocat.utils.ioutils.total_dose_load` for more information on
        formatting.
    voltage : float, default=300.0
        The voltage of the microscope, defaults to 300.0.
    amp_contrast : float, default=0.07
        The amplitude contrast of the microscope, defaults to 0.07.
    cs : float, default=2.7
        The spherical aberration coefficient, defaults to 2.7.
    output_path : str, optional
        The path to the output file, by default None. If None, the output is not written out.
    drop_nan_columns : bool, default=True
        Whether to drop columns with NaN values, defaults to True.

    See also
    --------
    :meth:`cryocat.utils.ioutils.tlt_load`, :meth:`cryocat.utils.ioutils.z_shift_load`, :meth:`cryocat.utils.ioutils.defocus_load`,
    :meth:`cryocat.utils.ioutils.total_dose_load`

    Returns
    -------
    pandas.DataFrame
        The wedge list dataframe in STOPGAP format for single tomogram/tilt series.
    """

    wedge_list_df = pd.DataFrame(
        columns=[
            "tomo_num",
            "pixelsize",
            "tomo_x",
            "tomo_y",
            "tomo_z",
            "z_shift",
            "tilt_angle",
            "defocus",
            "exposure",
            "voltage",
            "amp_contrast",
            "cs",
        ]
    )

    tilts = ioutils.tlt_load(tlt_file)

    wedge_list_df["tilt_angle"] = tilts

    if ctf_file is not None:
        ctf_df = ioutils.defocus_load(ctf_file, ctf_file_type)
        defocus = ctf_df["defocus_mean"].values
        check_data_consistency(defocus, tilts, "ctf", tlt_file)
        wedge_list_df["defocus"] = defocus

    if dose_file is not None:
        dose = ioutils.total_dose_load(dose_file)
        check_data_consistency(dose, tilts, "dose", tlt_file)
        wedge_list_df["exposure"] = dose

    tomo_dimensions = ioutils.dimensions_load(tomo_dim)
    z_shift = ioutils.z_shift_load(z_shift)

    wedge_list_df["tomo_num"] = tomo_id
    wedge_list_df["pixelsize"] = pixel_size
    wedge_list_df[["tomo_x", "tomo_y", "tomo_z"]] = np.repeat(tomo_dimensions.values, tilts.shape[0], axis=0)
    wedge_list_df["z_shift"] = z_shift.values[0][0]
    wedge_list_df["voltage"] = voltage
    wedge_list_df["amp_contrast"] = amp_contrast
    wedge_list_df["cs"] = cs

    if drop_nan_columns:
        wedge_list_df = wedge_list_df.dropna(axis=1, how="all")

    if output_path is not None:
        starfileio.Starfile.write(
            [wedge_list_df], output_path, specifiers=["data_stopgap_wedgelist"], number_columns=False
        )
    return wedge_list_df


def create_wedge_list_sg_batch(
    tomo_list: TomoList,
    pixel_size: float,
    tlt_file_format: str,
    tomo_dim: TomoDimensions | None = None,
    tomo_dim_file_format: str | None = None,
    z_shift: float = 0.0,
    z_shift_file_format: str | None = None,
    ctf_file_format: str | None = None,
    ctf_file_type: CTFFileType = "gctf",
    dose_file_format: str | None = None,
    voltage: float = 300.0,
    amp_contrast: float = 0.07,
    cs: float = 2.7000,
    output_path: PathOrStr | None = None,
) -> pd.DataFrame:
    """Create a wedge list dataframe in STOPGAP format for all tomograms/tilt series specified in tomo_list.

    Parameters
    ----------
    tomo_list : str or array-like
        The path to the file containing list of tomograms (txt) or tomogram/tilt series numbers specified as array-like
        variable.
    pixel_size : float
        The pixel size of the tomogram/tilt series. The pixel size has to be same for all tomograms/tilt series
        otherwise STOPGAP will not accept it.
    tlt_file_format : str
        The format describing name of the input files (including the path) with tilt angles. See `Notes` below for more
        information. See :meth:`cryocat.utils.ioutils.tlt_load` for more information on allowed input files (tlt, mdoc,
        xml).
    tomo_dim : array-like, optional
        Tomogram dimensions specified as array-like variable. See :meth:`cryocat.utils.ioutils.dimensions_load` for more
        information on formatting. Defaults to None but either tomo_dim or tomo_dim_file_format has to be specified.
    tomo_dim_file_format : str, optional
        The format describing name of the input files (including the path) with tomogram dimensions. See `Notes` below
        for more information. See :meth:`cryocat.utils.ioutils.dimensions_load` for more information on allowed input files
        (txt, com). Defaults to None but either tomo_dim or tomo_dim_file_format has to be specified.
    z_shift : array-like or dataframe or int or float, default=0.0
        Z-shift specified as array-like, pandas DataFrame, int or float. See :meth:`cryocat.utils.ioutils.z_shift_load`
        for more information on input types. Defaults to 0.0.
    z_shift_file_format : str, optional
        The format describing name of the input files (including the path) with z-shift. See `Notes` below for more
        information. See :meth:`cryocat.utils.ioutils.z_shift_load` for more information on allowed input files (com, txt).
        Defaults to None.
    ctf_file_format : str, optional
        The format describing name of the input files (including the path) with defocus values. See `Notes` below for more
        information. Supported formats are gctf (star file), ctffind4 (txt file) and warp (xml file). Defaults to None.
        See :meth:`cryocat.utils.ioutils.defocus_load` on more information of file formats.
    ctf_file_type : str, {"gctf", "ctffind4", "warp"}
        The type of the CTF file with defocus values. It can be either "gctf", "ctffind4", or "warp", defaults to "gctf".
    dose_file_format : str or array-like, optional
        The format describing name of the input files (including the path) with corrected dose. See `Notes` below for more
        information. See :meth:`cryocat.utils.ioutils.total_dose_load` for more information on allowed input files
        (txt, mdoc, xml). Defaults to None.
    voltage : float, default=300.0
        The voltage of the microscope, defaults to 300.0.
    amp_contrast : float, default=0.07
        The amplitude contrast of the microscope, defaults to 0.07.
    cs : float, default=2.7
        The spherical aberration coefficient, defaults to 2.7.
    output_path : str, optional
        The path to the output file, by default None. If None, the output is not written out. Defaults to None.

    Returns
    -------
    pandas.DataFrame
        The wedge list dataframe in STOPGAP format for all tomograms/tilt series specified by tomo_list.

    Raises
    ------
    ValueError
        If neither tomo_dim or tomo_dim_file_format is specified.

    See also
    --------
    :meth:`cryocat.utils.ioutils.tlt_load`, :meth:`cryocat.utils.ioutils.z_shift_load`, :meth:`cryocat.utils.ioutils.defocus_load`,
    :meth:`cryocat.utils.ioutils.total_dose_load`

    Notes
    -----
    The variables with _file_format in name should contain pattern that will be replaced by tomogram/tilt series numbers
    specified in the tomo_list. The pattern should start with $ and should be followed by sequnece of x. The sequence of
    x will be replaced by tomogram/tilt series number from tomo_list and pad with zeros if necessary. For example,
    if tlt_file_format is specified as "TS_$xxx/$xxx.tlt" and the tomo_list contains numbers 79 and 155, the final
    file names will be TS_079/079.tlt and TS_155/155.tlt. The sequence of x can be of arbitrary length, even within one
    file format, e.g. "TS_$xxxx/$xxx.tlt". However, the minimal allowed length of the sequence is given by the number of
    digits of the largest tomogram/tilt series number. For instance, TS_$xx/$xxx.tlt will fail since tomogram 155 requires
    sequence of at least 3 x. It is expected that all files of the same type will follow same formatting. Different file
    types can follow different formatting. For example, the tlt_file_format can be TS_$xxx/$xxx.tlt but defocus files can
    be all in one folder specified as "ctf_files/$xxxx_ctffind4.txt".
    """

    wedge_list_df = pd.DataFrame()
    ctf_file = None
    dose_file = None

    tomograms = ioutils.tlt_load(tomo_list).astype(int)

    if tomo_dim_file_format is None:
        if tomo_dim is not None:
            tomo_dimensions = ioutils.dimensions_load(tomo_dim)
            if "tomo_id" not in tomo_dimensions.columns:
                repeated_values = np.repeat(tomo_dimensions[["x", "y", "z"]].values, len(tomograms), axis=0)
                tomo_dimensions = pd.DataFrame(repeated_values, columns=["x", "y", "z"])
                tomo_dimensions["tomo_id"] = tomograms
        else:
            raise ValueError("Either tomo_dim or tomo_dim_file_format has to be specified!")

    if z_shift_file_format is None:
        z_shift_df = ioutils.z_shift_load(z_shift)
        if "tomo_id" not in z_shift_df.columns:
            repeated_values = np.repeat(z_shift_df["z_shift"].values, len(tomograms), axis=0)
            z_shift_df = pd.DataFrame(repeated_values, columns=["z_shift"])
            z_shift_df["tomo_id"] = tomograms

    for t in tomograms:
        tlt_file = ioutils.fileformat_replace_pattern(tlt_file_format, t, "x", raise_error=False)

        if ctf_file_format is not None:
            ctf_file = ioutils.fileformat_replace_pattern(ctf_file_format, t, "x", raise_error=False)

        if dose_file_format is not None:
            dose_file = ioutils.fileformat_replace_pattern(dose_file_format, t, "x", raise_error=False)

        if tomo_dim_file_format is not None:
            t_dim = ioutils.fileformat_replace_pattern(tomo_dim_file_format, t, "x", raise_error=False)
        else:
            t_dim = tomo_dimensions.loc[tomo_dimensions["tomo_id"] == t, ["x", "y", "z"]].values[0]

        if z_shift_file_format is not None:
            z_shift_input = ioutils.fileformat_replace_pattern(z_shift_file_format, t, "x", raise_error=False)
        else:
            z_shift_input = z_shift_df.loc[z_shift_df["tomo_id"] == t, "z_shift"].values[0]

        wl_single_df = create_wedge_list_sg(
            t,
            tomo_dim=t_dim,
            pixel_size=pixel_size,
            tlt_file=tlt_file,
            z_shift=z_shift_input,
            ctf_file=ctf_file,
            ctf_file_type=ctf_file_type,
            dose_file=dose_file,
            voltage=voltage,
            amp_contrast=amp_contrast,
            cs=cs,
            output_path=None,
            drop_nan_columns=False,
        )

        wedge_list_df = pd.concat([wedge_list_df, wl_single_df])

    wedge_list_df = wedge_list_df.dropna(axis=1, how="all")
    wedge_list_df.reset_index(drop=True, inplace=True)
    if output_path is not None:
        starfileio.Starfile.write(
            [wedge_list_df], output_path, specifiers=["data_stopgap_wedgelist"], number_columns=False
        )
    return wedge_list_df


def create_wedge_list_em_batch(
    tomo_list: TomoList,
    tlt_file_format: str,
    output_path: PathOrStr | None = None,
) -> pd.DataFrame:
    """Create a wedge list dataframe in EM format for all tomograms/tilt series specified in tomo_list.

    Parameters
    ----------
    tomo_list : TomoList
        The path to the file containing list of tomograms (txt) or tomogram/tilt series numbers specified as array-like
        variable. See :meth:`cryocat.utils.ioutils.tlt_load` for more information on formatting.
    tlt_file_format : str
        The format describing name of the input files (including the path) with tilt angles. See `Notes` below for more
        information. See :meth:`cryocat.utils.ioutils.tlt_load` for more information on allowed input files (tlt, mdoc,
        xml).
    output_path : PathOrStr, optional
        The path to the output file, by default None. If None, the output is not written out.

    Returns
    -------
    pandas.DataFrame
        The wedge list dataframe in EM format for all tomograms/tilt series specified by tomo_list.

    See also
    --------
    :meth:`cryocat.utils.ioutils.tlt_load`

    Notes
    -----
    The variables with _file_format in name should contain pattern that will be replaced by tomogram/tilt series numbers
    specified in the tomo_list. The pattern should start with $ and should be followed by sequnece of x. The sequence of
    x will be replaced by tomogram/tilt series number from tomo_list and pad with zeros if necessary. For example,
    if tlt_file_format is specified as "TS_$xxx/$xxx.tlt" and the tomo_list contains numbers 79 and 155, the final
    file names will be TS_079/079.tlt and TS_155/155.tlt. The sequence of x can be of arbitrary length, even within one
    file format, e.g. "TS_$xxxx/$xxx.tlt". However, the minimal allowed length of the sequence is given by the number of
    digits of the largest tomogram/tilt series number. For instance, TS_$xx/$xxx.tlt will fail since tomogram 155 requires
    sequence of at least 3 x. It is expected that all files of the same type will follow same formatting. Different file
    types can follow different formatting. For example, the tlt_file_format can be TS_$xxx/$xxx.tlt but defocus files can
    be all in one folder specified as "ctf_files/$xxxx_ctffind4.txt".

    """

    wedge_list_df = pd.DataFrame(columns=["tomo_num", "min_angle", "max_angle"])

    tomograms = ioutils.tlt_load(tomo_list).astype(int)

    wedge_list_df["tomo_num"] = tomograms
    tilts_min = []
    tilts_max = []

    for t in tomograms:
        tlt_file = ioutils.fileformat_replace_pattern(tlt_file_format, t, "x", raise_error=False)
        tilts = ioutils.tlt_load(tlt_file).astype(np.single)
        tilts_min.append(np.min(tilts))
        tilts_max.append(np.max(tilts))

    wedge_list_df["min_angle"] = np.asarray(tilts_min)
    wedge_list_df["max_angle"] = np.asarray(tilts_max)

    if output_path is not None:
        wedge_array = wedge_list_df.to_numpy()
        wedge_array = wedge_array.reshape((1, wedge_array.shape[0], wedge_array.shape[1])).astype(np.single)
        emfile.write(output_path, wedge_array, {}, overwrite=True)

    return wedge_list_df


def load_wedge_list_sg(input_data: DataSource) -> pd.DataFrame:
    """Load a STOPGAP wedge list from a file or a pandas DataFrame.

    Parameters
    ----------
    input_data : DataSource
        The input data can either be a file path (string) to a star file or a pandas DataFrame containing the wedge list.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the wedge list extracted from the input data.

    Raises
    ------
    ValueError
        If the input_data is neither a string nor a pandas DataFrame.
    """

    if isinstance(input_data, str):
        if not os.path.exists(input_data):
            raise ValueError(f"Filepath '{input_data}' is not valid.")
        wedge_list_df, _, _ = starfileio.Starfile.read(input_data)
        wedge_list_df = wedge_list_df[0]
    elif isinstance(input_data, pd.DataFrame):
        wedge_list_df = input_data
    else:
        raise ValueError(f"Input must be either a valid pathfile either a dataframe")

    return wedge_list_df


def load_wedge_list_em(input_data: DataSource) -> pd.DataFrame:
    """Load an EM wedge list from various input formats.

    Parameters
    ----------
    input_data : DataSource
        The input data can be one of the following:
        - A path (``PathOrStr``) to a data source.
        - A 2D numpy array with a shape of (n, 3), where n is the number of wedges.
        - A pandas DataFrame containing wedge data.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the wedge list with columns:
        - 'tomo_id': Identifier for the tomograms.
        - 'min_tilt_angle': Minimum tilt angle.
        - 'max_tilt_angle': Maximum tilt angle.

    Raises
    ------
    ValueError
        If the input_data is not of a valid type or does not conform to the expected shape.
    """

    df_columns = ["tomo_id", "min_tilt_angle", "max_tilt_angle"]

    if isinstance(input_data, str):
        wedge_list_df = pd.DataFrame(columns=df_columns, data=np.squeeze(cryomap.read(input_data)).T)
    elif isinstance(input_data, np.ndarray):
        if input_data.ndim == 2 and input_data.shape[1] == 3:
            wedge_list_df = pd.DataFrame(columns=df_columns, data=input_data)
        else:
            raise ValueError(
                "Provided array does not have the correct shape - it has to be 2D with second dim of size 3!"
            )
    elif isinstance(input_data, pd.DataFrame):
        wedge_list_df = input_data
        if all(col in input_data.columns for col in df_columns):
            pass
        else:
            if len(input_data.columns) == len(df_columns):
                wedge_list_df.columns = df_columns
            else:
                raise ValueError("Provided data frame does not have the correct shape!")
    else:
        raise ValueError("Invalid input - only strings (file names), np.ndarrays or pandas data frames are supported.")

    return wedge_list_df


def wedge_list_sg_to_em(
    input_path: PathOrStr,
    output_path: PathOrStr,
    write_out: bool = True,
) -> pd.DataFrame:
    """Convert a STOPGAP star format wedge list into a em wedge list;
    only 3 columns [tomo_id, min_tilt_angle, max_tilt_angle] are collected

    Parameters
    ----------
    input_path : PathOrStr
        Path to a STOPGAP star wedge list.
    output_path : PathOrStr
        Path to save the new em format wedge list.
    write_out : bool, default=True
        Whether to save the output. Default is True.

    Returns
    -------
    wedge_list_em : pd.DataFrame
        Pandas Dataframe with the 3 columns mentioned above.
    """

    # read STOPGAP wedge list star file
    wedge_list_sg = load_wedge_list_sg(input_path)

    # get the min and max of tilt_angle column and create a new df
    wedge_list_em = wedge_list_sg.groupby("tomo_num").agg(
        min_tilt_angle=("tilt_angle", "min"), max_tilt_angle=("tilt_angle", "max")
    )
    wedge_list_em.reset_index(inplace=True)
    wedge_list_em.rename(columns={"tomo_num": "tomo_id"}, inplace=True)

    # write out to em format
    if write_out:
        wedge_array = wedge_list_em.to_numpy()
        wedge_array = wedge_array.reshape((1, wedge_array.shape[0], wedge_array.shape[1])).astype(np.single)
        emfile.write(output_path, wedge_array, {}, overwrite=True)

    return wedge_list_em


def create_wg_mask(
    wg_list_star_df: pd.DataFrame,
    tomo_list: TomoList,
    box_size: TripletLike,
    shape: str = "wedge",
    output_path: PathOrStr | None = None,
) -> np.ndarray:
    """Create a missing-wedge mask for each tomogram listed in ``tomo_list``.

    Parameters
    ----------
    wg_list_star_df : pandas.DataFrame
        Wedge-list star file loaded as a DataFrame; must contain columns
        ``tomo_num`` and ``tilt_angle``.
    tomo_list : TomoList
        Path to a ``.tlt`` file or an integer array of tomogram IDs.
    box_size : TripletLike
        Side length(s) of the cubic (or non-cubic) output mask volume.
    shape : str, default='wedge'
        Mask geometry.  Currently ``'wedge'`` and ``'sph_wedge'`` are
        supported.
    output_path : PathOrStr, optional
        If given, the mask is written to this file path.

    Returns
    -------
    numpy.ndarray
        The last generated wedge mask as a 3-D binary array (float32).

    Raises
    ------
    ValueError
        If ``wg_list_star_df`` is not a :class:`pandas.DataFrame`.
    """
    if not isinstance(wg_list_star_df, pd.DataFrame):
        raise ValueError("Provided wg_list_star_df is not a pandas DataFrame!")
    tomograms = ioutils.tlt_load(tomo_list).astype(int)
    for value in tomograms:
        sub_wg = wg_list_star_df.loc[wg_list_star_df["tomo_num"] == value].copy()
        angles = [i for i in sub_wg.loc[:, "tilt_angle"]]

        box_size = geom.as_triplet(box_size)
        mask = np.empty(box_size)

        if shape == "wedge" or shape == "sph_wedge":
            x = range(-box_size[0] // 2, box_size[0] // 2, 1)
            y = range(-box_size[1] // 2, box_size[1] // 2, 1)
            z = range(-box_size[2] // 2, box_size[2] // 2, 1)
            xx, yy, zz = np.mgrid[x, y, z]

            mask_xz1 = xx > (math.tan(np.deg2rad(min(angles))) * zz)
            mask_xz2 = xx < (math.tan(np.deg2rad(max(angles))) * zz)

            mask = ~np.logical_xor(mask_xz1, mask_xz2)
            mask[box_size[0] // 2, box_size[1] // 2, box_size[2] // 2] = 1

        mask = mask.transpose(2, 1, 0)

        if output_path is not None:
            cryomap.write(mask, output_path, transpose=True, data_type=np.single)

    return mask


def apply_wedge_mask(
    wedge_mask: MapSource,
    in_map: MapSource,
    rotation_zxz: EulerAngles | None = None,
    output_path: PathOrStr | None = None,
) -> np.ndarray:
    """Apply a wedge mask to a volume in Fourier space.

    Reads ``in_map``, optionally rotates it by ``rotation_zxz``, applies the
    wedge mask in Fourier space, and returns the filtered real-valued volume.

    Parameters
    ----------
    wedge_mask : MapSource
        Path or ndarray for the wedge mask volume.
    in_map : MapSource
        Path or ndarray for the input volume to be filtered.
    rotation_zxz : EulerAngles, optional
        zxz Euler angles (degrees) used to rotate the map before masking.
        When ``None`` no rotation is applied.
    output_path : PathOrStr, optional
        If given, the filtered volume is written to this path.

    Returns
    -------
    numpy.ndarray
        Filtered volume as a real-valued float32 array.
    """
    rot_map = cryomask.rotate(cryomap.read(in_map), rotation_zxz)

    ft_map = np.fft.fftshift(np.fft.fftn((rot_map)))
    ft_map = ft_map * cryomap.read(wedge_mask)
    out_map = np.fft.ifftn(np.fft.ifftshift(ft_map))

    # Convert complex array to real-valued array and force float32
    out_map = np.abs(out_map).astype(np.float32)

    if output_path is not None:
        cryomap.write(out_map, output_path)

    return out_map


# ── Wedge-mask generation from wedge lists ────────────────────────────────────


def _generate_exposure(
    wedgelist: pd.DataFrame, slice_idx: list, slice_weight: np.ndarray, binning: int | float
) -> np.ndarray:
    r"""Generate an exposure-based filter to account for frequency-dependent signal
    attenuation due to electron dose in cryo-electron tomography.

    Each tilt's accumulated dose is mapped through the Grant & Grigorieff
    attenuator :func:`cryocat.utils.imageutils.dose_attenuator` over the
    tilt's active Fourier slice.

    Parameters
    ----------
    wedgelist : pandas.DataFrame
        Metadata table containing at least ``"exposure"`` and ``"pixelsize"`` columns.
    slice_idx : list of tuple of ndarray
        Per-tilt active voxel indices in Fourier space.
    slice_weight : numpy.ndarray
        3D array of normalised frequency-domain weights.
    binning : int or float
        The binning factor of the tomogram.

    Returns
    -------
    exp_filt : numpy.ndarray
        3D exposure filter of the same shape as ``slice_weight``.
    """
    expo = wedgelist["exposure"].values
    pixelsize = wedgelist["pixelsize"].values[0] * binning

    freq_array = np.fft.ifftshift(imageutils.compute_frequency_array(slice_weight.shape, pixelsize))

    exp_filt = np.zeros_like(slice_weight)
    for expi, idx in zip(expo, slice_idx):
        # Per-tilt attenuator evaluated on this tilt's active frequencies.
        exp_filt[idx] += imageutils.dose_attenuator(expi, freq_array[idx])

    exp_filt *= slice_weight
    return exp_filt


def _geometric_wedgemask_slices(wedgelist: pd.DataFrame, map_filter: np.ndarray) -> tuple:
    """Geometric wedge-mask slab union via rotation-and-threshold of a 2D seed line.

    Each slab is the thresholded interpolation halo of the rotated line,
    extruded along the tilt axis. Note: slab content is bounded by the 2D
    image's extent, so the region near the volume corner may be uncovered
    for large boxes — use ``method="analytic"`` (see
    :func:`_analytical_wedgemask_slices`) to avoid this.

    Parameters
    ----------
    wedgelist : pandas.DataFrame
        Table with a ``"tilt_angle"`` column.
    map_filter : ndarray
        3D frequency-filtered volume; must be cubic.

    Returns
    -------
    active_slices_idx : list of tuple of ndarrays
        Per-tilt active voxel indices into ``map_filter`` (the support of the
        rotated tilt slice intersected with the band-pass support).
    wedge_slices_weights : ndarray
        Reciprocal of the per-voxel coverage count; zero where no tilt covered.
    wedge_slices : ndarray
        Binary union of all tilt slices (1 inside the wedge, 0 outside).
    """
    map_size = np.array(map_filter.shape)
    assert len(map_size) == 3, "The map is not 3D!"
    assert len(np.unique(map_size)) == 1, "The map is not cubic in shape!"

    mx = np.max(map_size[[2, 0]])
    img = np.zeros((mx, mx))
    img[:, mx // 2] = 1.0

    bpf_idx = map_filter > 0

    active_slices_idx = []
    wedge_slices_weights = np.zeros_like(map_filter)
    weight = np.zeros_like(map_filter)

    for alpha in wedgelist["tilt_angle"]:
        r_img = imageutils.rotate_2d(img, alpha)
        crop_r_img = r_img > np.exp(-2)
        slice_vol = np.fft.ifftshift(np.transpose(np.tile(crop_r_img, (mx, 1, 1)), (2, 0, 1)))
        slice_idx = slice_vol & bpf_idx
        weight += slice_idx
        active_slices_idx.append(np.nonzero(slice_idx))

    w_idx = np.nonzero(weight)
    wedge_slices_weights[w_idx] = 1.0 / weight[w_idx]

    wedge_slices = np.zeros_like(weight)
    wedge_slices[w_idx] = 1.0

    return active_slices_idx, wedge_slices_weights, wedge_slices


def _analytical_wedgemask_slices(
    wedgelist: pd.DataFrame,
    template_filter: np.ndarray,
    thickness: float | None = None,
) -> tuple:
    """Analytic wedge-mask slab union via perpendicular-distance thresholding.

    Each tilt's slab is the set of Fourier voxels within ``thickness`` pixels
    of the slab plane. Slabs extend across the full Fourier volume — no
    bound-by-2D-image-extent truncation.

    Slab plane convention: ``sin(α)·kx − cos(α)·kz = 0``, where α is the tilt
    angle. The sign matches :func:`cryocat.utils.imageutils.rotate_2d`:
    positive α rotates the slab clockwise in image-display coordinates. For
    symmetric tilt lists (paired ±α) the sign is invisible; for asymmetric
    lists it sets the bow-tie orientation and must match the rest of the
    pipeline.

    Output is in FFT layout (DC at index 0,0,0), matching the geometric
    version.

    Parameters
    ----------
    wedgelist : pandas.DataFrame
        Wedge list filtered to a single tomogram. Required column:
        ``tilt_angle``.
    template_filter : ndarray, shape (nx, ny, nz)
        Bandpass filter array; its shape defines the output mask shape, and
        its support (``template_filter > 0``) restricts which voxels each
        slab contributes to.
    thickness : float, optional
        Slab half-width in Fourier pixels. When ``None`` (default), chosen
        so adjacent slabs stay overlapping at the volume's Nyquist radius:
        ``max(1.0, (max(shape) / 2) * sin(max_gap_deg) / 2)``, where
        ``max_gap_deg`` is the largest angular gap between consecutive tilts.

    Returns
    -------
    active_slices_idx : list of tuple of ndarrays
        Per-tilt nonzero indices of the slab intersected with the bandpass
        support, one entry per row of ``wedgelist``. Same shape and meaning
        as the geometric version, so CTF / exposure weighting code consumes
        it identically.
    wedge_slices_weights : ndarray, float32
        ``1 / overlap_count`` where any slab is present; zero elsewhere.
    wedge_slices : ndarray, float32
        Binary union mask (1 where any slab is present).
    """
    shape = template_filter.shape
    nx, _ny, nz = shape
    bpf_idx = template_filter > 0

    # Auto-thickness: keep adjacent slabs overlapping at Nyquist.
    if thickness is None:
        tilts_sorted = np.sort(wedgelist["tilt_angle"].to_numpy(dtype=float))
        max_gap_deg = float(np.max(np.diff(tilts_sorted))) if len(tilts_sorted) > 1 else 1.0
        nyq = max(shape) / 2.0
        thickness = max(1.0, nyq * np.sin(np.deg2rad(max_gap_deg)) / 2.0)

    # FFT-layout pixel coordinates (DC at index 0).
    kx = np.fft.fftfreq(nx, d=1.0) * nx
    kz = np.fft.fftfreq(nz, d=1.0) * nz
    KX, KZ = np.meshgrid(kx, kz, indexing="ij")

    weight = np.zeros(shape, dtype=np.int32)
    active_slices_idx: list = []
    for alpha in wedgelist["tilt_angle"]:
        a = np.deg2rad(float(alpha))
        # Slab plane sin(α)·kx − cos(α)·kz = 0; sign matches rotate_2d.
        d = np.abs(np.cos(a) * KX - np.sin(a) * KZ)
        slab_2d = d < thickness  # (nx, nz)
        slab_3d = np.broadcast_to(slab_2d[:, None, :], shape) & bpf_idx
        weight += slab_3d
        active_slices_idx.append(np.nonzero(slab_3d))

    w_idx = np.nonzero(weight)
    wedge_slices_weights = np.zeros(shape, dtype=np.float32)
    wedge_slices_weights[w_idx] = 1.0 / weight[w_idx]
    wedge_slices = (weight > 0).astype(np.float32)

    return active_slices_idx, wedge_slices_weights, wedge_slices


@gui_exposed(
    label="Generate wedge mask",
    category="builder",
    standalone=True,
    output="map",
    hide=("output_path",),
)
def generate_wedge_mask(
    map_size: TripletLike | int,
    wedgelist: DataSource,
    tomo_number: int,
    *,
    method: WedgeMaskMethod = "geometric",
    thickness: float | None = None,
    binning: int = 1,
    low_pass_filter: int | None = None,
    high_pass_filter: int | None = None,
    ctf_weighting: bool = False,
    exposure_weighting: bool = False,
    output_path: PathOrStr | None = None,
) -> dict:
    """Generate a Fourier-space wedge mask for one volume of size ``map_size``.

    Builds the per-tomogram missing-wedge mask from a wedge-list row, optionally
    weighted by CTF and exposure terms.  The result is a single 3D array — no
    template/target pairing, no analysis-specific structure.  Callers that need
    both a template-side and a target-side mask call this function twice with
    different ``map_size`` values.

    Two construction methods are available, selected via ``method``:

    - ``"geometric"`` (default): build each slab by rotating a 2D seed line
      and thresholding the interpolation halo. Matches the historical
      cryoCAT behaviour exactly. Slab content is bounded by the 2D image
      extent, so high frequencies near the volume corner may be uncovered
      for large boxes.
    - ``"analytic"``: build each slab as the set of Fourier voxels within
      ``thickness`` pixels of the slab plane. Slabs extend across the full
      Fourier volume. For dense tilt lists, leave ``thickness=None``
      (default) so adjacent slabs stay overlapping at Nyquist.

    Both methods return masks in FFT layout (DC at index 0,0,0) and use the
    same slab orientation convention.

    Parameters
    ----------
    map_size : TripletLike or int
        Output volume shape.  A scalar is treated as cubic; a triplet is
        ``(x, y, z)``.  A ``TripletLike`` is a scalar or a 3-element sequence.
    wedgelist : DataSource
        Path to a STOPGAP wedge list (.star) or a preloaded DataFrame.
        A ``DataSource`` is a path to a file or an ndarray / DataFrame.
    tomo_number : int
        Tomogram number to select from the wedge list.
    method : WedgeMaskMethod, default "geometric"
        Slab-construction algorithm. ``"geometric"`` matches the historical
        cryoCAT output exactly; ``"analytic"`` extends slabs across the full
        Fourier volume so high-frequency corners stay covered. See the
        paragraph above for details.
    thickness : float, optional
        Slab half-width in Fourier pixels. Used only when
        ``method="analytic"``; passing it with ``method="geometric"`` emits a
        :class:`UserWarning` and the value is ignored. ``None`` (default)
        triggers auto-scaling based on the largest angular gap between
        consecutive tilts and the volume's Nyquist radius.
    binning : int, default 1
        Pixel-size binning factor used for exposure and CTF computations.
    low_pass_filter : int, optional
        Low-pass cutoff in Fourier pixels; ``None`` disables the filter.
    high_pass_filter : int, optional
        High-pass cutoff in Fourier pixels; ``None`` disables the filter.
    ctf_weighting : bool, default False
        Apply CTF weighting to the mask.
    exposure_weighting : bool, default False
        Apply exposure weighting to the mask.
    output_path : PathOrStr, optional
        Where to write the mask.  ``None`` → returned in memory only.
        A ``PathOrStr`` is a :class:`str` or :class:`pathlib.Path`.

    Returns
    -------
    dict
        ``mask`` : ndarray — the wedge mask, shaped ``map_size`` (transposed to
        xyz convention before return);
        ``output_path`` : str or None — path the mask was written to.

    Raises
    ------
    ValueError
        If ``method`` is neither ``"geometric"`` nor ``"analytic"``.
    """
    size_triplet = geom.as_triplet(map_size)
    filt = np.ones(size_triplet)

    wl = load_wedge_list_sg(wedgelist)
    wl = wl.loc[wl["tomo_num"] == tomo_number]

    if low_pass_filter is not None:
        filt *= np.fft.ifftshift(imageutils._spherical_mask_nd(size_triplet, low_pass_filter))
    if high_pass_filter is not None:
        filt *= np.fft.ifftshift(imageutils._spherical_mask_nd(size_triplet, high_pass_filter))

    if method == "geometric":
        if thickness is not None:
            warnings.warn(
                "thickness is only used when method='analytic'; ignoring.",
                stacklevel=2,
            )
        active_slices_idx, wedge_slices_weights, wedge_slices = _geometric_wedgemask_slices(wl, filt)
    elif method == "analytic":
        active_slices_idx, wedge_slices_weights, wedge_slices = _analytical_wedgemask_slices(
            wl, filt, thickness=thickness
        )
    else:
        raise ValueError(f"Unknown method {method!r}; expected 'geometric' or 'analytic'.")

    mask = wedge_slices * filt

    if exposure_weighting:
        mask *= _generate_exposure(wl, active_slices_idx, wedge_slices_weights, binning)

    if ctf_weighting:
        mask *= imageutils.generate_ctf_slice(wl, active_slices_idx, wedge_slices_weights, binning)

    if output_path is not None:
        cryomap.write(mask, output_path, transpose=False, data_type=np.single)

    return {"mask": mask.transpose(2, 1, 0), "output_path": output_path}
