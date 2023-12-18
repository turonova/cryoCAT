import numpy as np
import pandas as pd
import re
import os
from cryocat import starfileio as sf
from cryocat import mdoc


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def fileformat_replace_pattern(filename_format, input_number, test_letter, raise_error=True):
    pattern = f"\$(?:{test_letter})+"
    findings = re.findall(pattern, filename_format)
    if not findings:
        if raise_error:
            raise ValueError(
                f"The format {filename_format} does not contain any sequence of \$ followed by {test_letter}."
            )
        else:
            return filename_format
    else:
        for pattern in findings:
            # Pad the number to the length of the string
            padded_number = str(input_number).zfill(len(pattern) - 1)

            # Check if the number has more digits than the string
            if len(padded_number) >= len(pattern):
                raise ValueError(f"Number '{input_number}' has more digits than string '{pattern}'.")

            # Replace the string in the long string
            filename_format = filename_format.replace(pattern, padded_number)

        return filename_format


def gctf_read(file_path):
    gctf_df = sf.Starfile.read(file_path, data_id=0)[0]

    # extract columns number 2, 3, 4 to new df
    if "rlnPhaseShift" in gctf_df.columns:
        converted_gctf = gctf_df[["rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", "rlnPhaseShift"]].astype(float)
    else:
        converted_gctf = gctf_df[["rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle"]].astype(float)
        converted_gctf["rlnPhaseShift"] = 0.0

    converted_gctf.iloc[:, 0:2] = converted_gctf.iloc[:, 0:2] * 10e-5
    converted_gctf = converted_gctf.rename(
        columns={
            "rlnDefocusU": "defocus1",
            "rlnDefocusV": "defocus2",
            "rlnDefocusAngle": "astigmatism",
            "rlnPhaseShift": "phase_shift",
        }
    )

    converted_gctf["defocus_mean"] = (converted_gctf["defocus1"] + converted_gctf["defocus2"]).values / 2.0

    return converted_gctf


def ctffind4_read(file_path):
    ctf = pd.read_csv(file_path, skiprows=5, header=None, dtype=np.float64, delim_whitespace=True)
    converted_ctf = ctf.iloc[:, 1:5].copy()
    converted_ctf.loc[:, converted_ctf.columns[0:2]] *= 10e-5
    converted_ctf.columns = ["defocus1", "defocus2", "astigmatism", "phase_shift"]

    converted_ctf["defocus_mean"] = (converted_ctf["defocus1"] + converted_ctf["defocus2"]).values / 2.0
    return converted_ctf


def tlt_read(file_path):
    tlt_df = pd.read_csv(file_path, header=None, dtype=np.float64, delim_whitespace=True)
    return tlt_df


def tlt_load(input_tlt):
    if isinstance(input_tlt, np.ndarray):
        return input_tlt
    elif isinstance(input_tlt, str):
        if input_tlt.endswith(".mdoc"):
            tilt_data = mdoc.Mdoc(input_tlt)
            tilts = tilt_data.get_image_feature("TiltAngle").values
            return tilts

        else:
            tilts = tlt_read(input_tlt).iloc[:, 0].values
            return tilts
    else:
        ValueError("Error: the dose has to be either ndarray or path to .csv, .mdoc, or .txt file!")


def dimensions_load(input_dims):
    """Loads tomogram dimensions from a file or nd.array.

    Parameters
    ----------
    input_dims : str
        Either a path to a file with the dimensions or nd.array. The shape of the input should
        be 1x3 (x y z) in case of one tomogram or Nx4 for multiple tomograms (tomo_id x y z). In case of file the
        separator is a space.

    Returns
    -------
    nd.array
        Dimensions of a tomogram in x, y, z (shape 1x3) or tomogram idx and corresponding dimensions
        (shape Nx4 where N is the number of tomograms)

    Raises
    ------
    UserInputError
        Wrong size of the input.

    """

    if isinstance(input_dims, pd.DataFrame):
        dimensions = input_dims
    elif isinstance(input_dims, str):
        if input_dims.endswith(".com"):
            com_file_d = imod_com_read(input_dims)
            dimensions = np.zeros((1, 3))
            dimensions[0, 0:2] = com_file_d["FULLIMAGE"]
            dimensions[0, 2] = com_file_d["THICKNESS"][0]
            dimensions = pd.DataFrame(dimensions)
        else:
            if os.path.isfile(input_dims):
                dimensions = pd.read_csv(input_dims, sep="\s+", header=None, dtype=float)
    elif isinstance(input_dims, list):
        dimensions = pd.DataFrame(np.reshape(np.asarray(input_dims), (1, len(input_dims))))
    else:  # isinstance(input_dims, np.ndarray):
        if input_dims.ndim == 1:
            input_dims = np.reshape(input_dims, (1, input_dims.shape[0]))

        dimensions = pd.DataFrame(input_dims)

    if dimensions.shape == (1, 3):
        dimensions.columns = ["x", "y", "z"]
    elif dimensions.shape[1] == 4:
        dimensions.columns = ["tomo_id", "x", "y", "z"]
    else:
        raise ValueError(
            f"The dimensions should have shape of 1x3 or Nx4, where N is number of tomograms."
            f"Instead following shape was extracted from the prvoided files: {dimensions.shape}."
        )

    return dimensions


def z_shift_load(input_shift):
    """Loads tomogram dimensions from a file or nd.array.

    Parameters
    ----------
    input_dims : str
        Either a path to a file with the dimensions or nd.array. The shape of the input should
        be 1x3 (x y z) in case of one tomogram or Nx4 for multiple tomograms (tomo_id x y z). In case of file the
        separator is a space.

    Returns
    -------
    nd.array
        Dimensions of a tomogram in x, y, z (shape 1x3) or tomogram idx and corresponding dimensions
        (shape Nx4 where N is the number of tomograms)

    Raises
    ------
    UserInputError
        Wrong size of the input.

    """

    if isinstance(input_shift, pd.DataFrame):
        z_shift = input_shift
    elif isinstance(input_shift, str):
        if input_shift.endswith(".com"):
            com_file_d = imod_com_read(input_shift)
            z_shift = pd.DataFrame([com_file_d["SHIFT"][1]])
        else:
            if os.path.isfile(input_shift):
                z_shift = pd.read_csv(input_shift, sep="\s+", header=None, dtype=float)
    elif isinstance(input_shift, (float, int)):
        z_shift = pd.DataFrame([input_shift])
    else:
        z_shift = pd.DataFrame(input_shift)

    if len(z_shift.columns) == 1:
        z_shift.columns = ["z_shift"]
    elif len(z_shift.columns) == 2:
        z_shift.columns = ["tomo_id", "z_shift"]
    else:
        raise ValueError(
            f"The z_shift should be one number or Nx1, where N is number of tomograms."
            f"Instead following shape was extracted from the prvoided files: {z_shift.shape}."
        )

    return z_shift


def imod_com_read(filename):
    result_dict = {}

    with open(filename, "r") as file:
        for line in file:
            # Ignore lines starting with #
            if line.startswith("#") or line.startswith("$"):
                continue

            # Split the line into words
            words = line.split()

            # Use the first word as the key
            key = words[0]

            # Convert the remaining words to the correct type
            values = [int(word) if word.isdigit() else float(word) if is_float(word) else word for word in words[1:]]

            # Store in the dictionary
            result_dict[key] = values

    return result_dict
