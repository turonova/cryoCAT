import numpy as np
import pandas as pd
from cryocat import ioutils
from cryocat import tiltstack as ts
from cryocat import starfileio
import emfile


def check_data_consistency(data1, data2, data_type1, data_type2):
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
    ValueError
        If the number of entries in data1 is different from the number of entries in data2.

    Returns
    -------
    None
        This function does not return anything.
    """

    if data1.shape[0] != data2.shape[0]:
        raise ValueError(f"The {data_type1} file has different number of entries than the {data_type2} file!")


def create_wedge_list_sg(
    tomo_id,
    tomo_dim,
    pixel_size,
    tlt_file,
    z_shift=0.0,
    ctf_file=None,
    ctf_file_type="gctf",
    dose_file=None,
    voltage=300.0,
    amp_contrast=0.07,
    cs=2.7000,
    output_file=None,
    drop_nan_columns=True,
):
    """Create a wedge list dataframe for a single tomogram/tilt series in STOPGAP format.

    Parameters
    ----------
    tomo_id : int
        The ID of the tomogram.
    tomo_dim : str or array-like
        The path to the tomogram dimensions file or dimensions specified as array-like variable. See
        :meth:`cryocat.ioutils.dimensions_load` for more information on formatting.
    pixel_size : float
        The pixel size of the tomogram/tilt series.
    tlt_file : str or array-like
        The path to the file containing information on tilts (tlt, mdoc, xml) or tilt angles specified as array-like
        variable. See :meth:`cryocat.ioutils.tlt_load` for more information on formatting.
    z_shift : str or array-like or int or float, default=0.0
        The path to the file containing information on z-shift (txt, com) or z-shift specified as array-like, pandas
        DataFrame, int or float. See :meth:`cryocat.ioutils.z_shift_load` for more information on formatting.
    ctf_file : str, optional
        The path to the file with defocus values - either in gctf (star), ctffind4 (txt) or warp (xml) format. See
        :meth:`cryocat.ioutils.defocus_load` for more information on formatting.
    ctf_file_type : str, {"gctf", "ctffind4", "warp"}
        The type of the CTF file with defocus values. It can be either "gctf", "ctffind4", "warp", defaults to "gctf".
    dose_file : str or array-like, optional
        The path to the file containing information on corrected dose (csv, mdoc, txt, xml) or the corrected
        dose specified as array-like variable. See :meth:`cryocat.ioutils.total_dose_load` for more information on
        formatting.
    voltage : float, default=300.0
        The voltage of the microscope, defaults to 300.0.
    amp_contrast : float, default=0.07
        The amplitude contrast of the microscope, defaults to 0.07.
    cs : float, default=2.7
        The spherical aberration coefficient, defaults to 2.7.
    output_file : str, optional
        The path to the output file, by default None. If None, the output is not written out.
    drop_nan_columns : bool, default=True
        Whether to drop columns with NaN values, defaults to True.

    See also
    --------
    :meth:`cryocat.ioutils.tlt_load`, :meth:`cryocat.ioutils.z_shift_load`, :meth:`cryocat.ioutils.defocus_load`,
    :meth:`cryocat.ioutils.total_dose_load`

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
        dose = ts.load_corrected_dose(dose_file)
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

    if output_file is not None:
        starfileio.Starfile.write(
            [wedge_list_df], output_file, specifiers=["data_stopgap_wedgelist"], number_columns=False
        )
    return wedge_list_df


def create_wedge_list_sg_batch(
    tomo_list,
    pixel_size,
    tlt_file_format,
    tomo_dim=None,
    tomo_dim_file_format=None,
    z_shift=0.0,
    z_shift_file_format=None,
    ctf_file_format=None,
    ctf_file_type="gctf",
    dose_file_format=None,
    voltage=300.0,
    amp_contrast=0.07,
    cs=2.7000,
    output_file=None,
):
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
        information. See :meth:`cryocat.ioutils.tlt_load` for more information on allowed input files (tlt, mdoc,
        xml).
    tomo_dim : array-like, optional
        Tomogram dimensions specified as array-like variable. See :meth:`cryocat.ioutils.dimensions_load` for more
        information on formatting. Defaults to None but either tomo_dim or tomo_dim_file_format has to be specified.
    tomo_dim_file_format: str, optional
        The format describing name of the input files (including the path) with tomogram dimensions. See `Notes` below
        for more information. See :meth:`cryocat.ioutils.dimensions_load` for more information on allowed input files
        (txt, com). Defaults to None but either tomo_dim or tomo_dim_file_format has to be specified.
    z_shift : array-like or dataframe or int or float, default=0.0
        Z-shift specified as array-like, pandas DataFrame, int or float. See :meth:`cryocat.ioutils.z_shift_load`
        for more information on input types. Defaults to 0.0.
    z_shift_file_format: str, optional
        The format describing name of the input files (including the path) with z-shift. See `Notes` below for more
        information. See :meth:`cryocat.ioutils.z_shift_load` for more information on allowed input files (com, txt).
    ctf_file_format : str, optional
        The format describing name of the input files (including the path) with defocus values. See `Notes` below for more
        information. Supported formats are gctf (star file), ctffind4 (txt file) and warp (xml file). Defaults to None.
        See :meth:`cryocat.ioutils.defocus_load` on more information of file formats.
    ctf_file_type : str, {"gctf", "ctffind4", "warp"}
        The type of the CTF file with defocus values. It can be either "gctf", "ctffind4", or "warp", defaults to "gctf".
    dose_file_format : str or array-like, optional
        The format describing name of the input files (including the path) with corrected dose. See `Notes` below for more
        information. See :meth:`cryocat.ioutils.total_dose_load` for more information on allowed input files
        (txt, mdoc, xml).
    voltage : float, default=300.0
        The voltage of the microscope, defaults to 300.0.
    amp_contrast : float, default=0.07
        The amplitude contrast of the microscope, defaults to 0.07.
    cs : float, default=2.7
        The spherical aberration coefficient, defaults to 2.7.
    output_file : str, optional
        The path to the output file, by default None. If None, the output is not written out.

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
    :meth:`cryocat.ioutils.tlt_load`, :meth:`cryocat.ioutils.z_shift_load`, :meth:`cryocat.ioutils.defocus_load`,
    :meth:`cryocat.ioutils.total_dose_load`

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
            output_file=None,
            drop_nan_columns=False,
        )

        wedge_list_df = pd.concat([wedge_list_df, wl_single_df])

    wedge_list_df = wedge_list_df.dropna(axis=1, how="all")
    wedge_list_df.reset_index(drop=True, inplace=True)
    if output_file is not None:
        starfileio.Starfile.write(
            [wedge_list_df], output_file, specifiers=["data_stopgap_wedgelist"], number_columns=False
        )
    return wedge_list_df


def create_wedge_list_em_batch(
    tomo_list,
    tlt_file_format,
    output_file=None,
):
    """Create a wedge list dataframe in EM format for all tomograms/tilt series specified in tomo_list.

    Parameters
    ----------
    tomo_list : str or array-like
        The path to the file containing list of tomograms (txt) or tomogram/tilt series numbers specified as array-like
        variable. See :meth:`cryocat.ioutils.tlt_load` for more information on formatting.
    tlt_file_format : str
        The format describing name of the input files (including the path) with tilt angles. See `Notes` below for more
        information. See :meth:`cryocat.ioutils.tlt_load` for more information on allowed input files (tlt, mdoc,
        xml).
    output_file : str, optional
        The path to the output file, by default None. If None, the output is not written out.

    Returns
    -------
    pandas.DataFrame
        The wedge list dataframe in EM format for all tomograms/tilt series specified by tomo_list.

    See also
    --------
    :meth:`cryocat.ioutils.tlt_load`

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

    if output_file is not None:
        wedge_array = wedge_list_df.to_numpy()
        wedge_array = wedge_array.reshape((1, wedge_array.shape[0], wedge_array.shape[1])).astype(np.single)
        emfile.write(output_file, wedge_array, {}, overwrite=True)

    return wedge_list_df
