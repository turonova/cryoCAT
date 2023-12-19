import numpy as np
import pandas as pd
from cryocat import ioutils
from cryocat import tiltstack as ts
from cryocat import starfileio


def check_data_consistency(data1, data2, data_type1, data_type2):
    if data1.shape[0] != data2.shape[0]:
        raise ValueError(f"The {data_type1} file has different number of entries than the {data_type2} file!")


def create_wedge_list(
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
    """Create a wedge list dataframe for a single tomogram/tilt series.

    Parameters
    ----------
    tomo_id : int
        The ID of the tomogram.
    tomo_dim : str or array-like
        The path to the tomogram dimensions file or dimensions specified as array-like variable. See
        :meth:`cryocat.ioutils.dimensions_load` for more information on formatting.
    pixel_size : float
        The pixel size of the tomogram/tilt series.
    tlt_file : str
        The path to the tilt file.
    z_shift : float, optional
        The z shift value, by default 0.0.
    ctf_file : str, optional
        The path to the CTF file, by default None.
    ctf_file_type : str, optional
        The type of the CTF file, either "gctf" or "ctffind4", by default "gctf".
    dose_file : str, optional
        The path to the dose file, by default None.
    voltage : float, optional
        The voltage of the microscope, by default 300.0.
    amp_contrast : float, optional
        The amplitude contrast, by default 0.07.
    cs : float, optional
        The spherical aberration coefficient, by default 2.7000.
    output_file : str, optional
        The path to the output file, by default None.
    drop_nan_columns : bool, optional
        Whether to drop columns with NaN values, by default True.

    Returns
    -------
    pandas.DataFrame
        The wedge list dataframe.
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
        if ctf_file_type == "gctf":
            ctf_df = ioutils.gctf_read(ctf_file)
        elif ctf_file_type == "ctffind4":
            ctf_df = ioutils.ctffind4_read(ctf_file)
        else:
            raise ValueError(f"Provided ctf_file_type is not valid {ctf_file_type}. It should be gctf or ctffind4")

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


def create_wedge_list_batch(
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

        wl_single_df = create_wedge_list(
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
