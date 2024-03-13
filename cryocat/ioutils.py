import numpy as np
import pandas as pd
import re
import os
from cryocat import starfileio as sf
from cryocat import mdoc
import xml.etree.ElementTree as ET


def is_float(value):
    """Check if a value can be converted to a float.

    Parameters
    ----------
    value : any
        The value to be checked.

    Returns
    -------
    bool
        True if the value can be converted to a float, False otherwise.

    Examples
    --------
    >>> is_float(3.14)
    True

    >>> is_float("hello")
    False
    """

    try:
        float(value)
        return True
    except ValueError:
        return False


def fileformat_replace_pattern(filename_format, input_number, test_letter, raise_error=True):
    """Replace a pattern in a filename format string with a given number. If the pattern is longer than number
    of digits in the input number the pattern is pad with zeros.

    Parameters
    ----------
    filename_format : str
        The filename format string containing the pattern to be replaced. The pattern has to start with $ followed by
        arbitrary long sequence of test_letter. For instance some_text_$AAA_rest for test_letter "A" and input number
        79 results in some_text_$079_rest.
    input_number : int
        The number to be inserted into the pattern.
    test_letter : str
        The letter used in the pattern to identify the sequence to be replaced.
    raise_error : bool, default=True
        Whether to raise a ValueError if the pattern is not found in the filename format string.
        Default is True.

    Returns
    -------
    str
        The filename format string with the pattern replaced by the input number and padded with zeros if the input number
        has less digits than the pattern.

    Raises
    ------
    ValueError
        If the pattern is not found in the filename format string and `raise_error` is True.
        If the input number has more digits than the pattern.

    Examples
    --------
    >>> fileformat_replace_pattern("file_/$AAA/$B.txt", 123, "A")
    'file_123/$B.txt'

    >>> fileformat_replace_pattern("some_text_$AAA_rest", 79, "A")
    'some_text_079_rest'

    >>> fileformat_replace_pattern("file_/$A/$B.txt", 123, "C")
    ValueError: The format file_/$A/$B.txt does not contain any sequence of \$ followed by C.

    >>> fileformat_replace_pattern("file_/$A/$B.txt", 12345, "A")
    ValueError: Number '12345' has more digits than string '\$A'.
    """

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


def get_data_from_warp_xml(xml_file_path, node_name, node_level=1):
    """This function parses an XML file and extracts data based on the provided XPath expression. The function
    supports two levels of extraction: level 1 and level 2.

    Parameters
    ----------
    xml_file_path: str
        The path to the XML file.
    node_name: str
        The XPath expression to find elements in the XML file.
    node_level: int, default=1
        The level of extraction. Default is 1 which works for nodes that have values without further tags (i.e.,
        one value per line without a xml tag). The other allowed level is 2 which should be used for all tags that have
        values stored in Node tags (such as GridCTF).

    Returns
    -------
    ndarray or None
        The extracted data as a NumPy array if elements are found, otherwise None.

    Raises
    ------
    Exception
        If there is an error reading the XML file.

    Examples
    --------
    >>> data = get_data_from_warp_xml('path/to/xml/file.xml', 'GridCTF', node_level=2)
    """

    try:
        # Parse the XML file
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        # Find elements based on the provided XPath expression
        elements = root.findall(node_name)

        # Check if elements are found
        if elements:
            # Extract data from each element based on its type
            if node_level == 2:  # elements[0].tag == 'GridCTF':
                # Directly find all Node elements within GridCTF
                node_elements = elements[0].findall(".//Node")

                # Extract values from each Node element
                data = [node.get("Value") for node in node_elements]
            else:
                # Default behavior: extract text content from the elements
                data_text = elements[0].text.strip()
                data = [float(value) for value in data_text.split("\n") if value.strip()]

            data = np.asarray(data)
            return data
        else:
            print(f"No elements found based on the provided XPath expression: {node_name}")
            return None

    except Exception as e:
        print(f"Error reading XML file: {e}")
        return None


def warp_ctf_read(input_file):
    """Reads CTF parameters from a WARP XML file.

    Parameters
    ----------
    input_file: str
        Path to the input WARP XML file.

    Returns:
    pandas.DataFrame
        DataFrame containing the columns "defocus1", "defocus2", "astigmatism",
        "phase_shift", "defocus_mean". All defocii values are in micrometers. The phase shift is in radians.
    """

    df_columns = ["defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"]
    defocus_df = pd.DataFrame(columns=df_columns)
    defocus_df["defocus_mean"] = get_data_from_warp_xml(input_file, "GridCTF", node_level=2)
    defocus_df["phase_shift"] = get_data_from_warp_xml(input_file, "GridCTFPhase", node_level=2)
    defocus_df["defocus1"] = defocus_df["defocus_mean"]
    defocus_df["defocus2"] = defocus_df["defocus_mean"]
    defocus_df["astigmatism"] = get_data_from_warp_xml(input_file, "GridCTFDefocusAngle", node_level=2)
    return defocus_df


def defocus_load(input_data, file_type="gctf"):
    """Load defocus data from various file types or a pandas DataFrame.

    Parameters
    ----------
    input_data : pd.DataFrame or str or numpy.ndarray
        The input data to load. If a pandas DataFrame is provided, it is assumed to already contain the defocus data.
        If a string is provided, it is assumed to be the path to a file containing the defocus data. If a numpy ndarray
        is provided, it is assumed to be a 2D array of shape Nx5 where N is number of tilts.

    file_type : str, default="gctf"
        The type of file to load if `input_data` is a string. Supported file types are "gctf", "ctffind4", and "warp".
        Default is "gctf".

    Returns
    -------
    defocus_df : pd.DataFrame
        A pandas DataFrame containing the loaded defocus data with columns "defocus1", "defocus2", "astigmatism",
        "phase_shift", "defocus_mean". All defocii values are in micrometers. The phase shift is in radians.

    Raises
    ------
    ValueError
        If the provided `file_type` is not supported.
    """

    if isinstance(input_data, pd.DataFrame):
        defocus_df = input_data
    elif isinstance(input_data, str):
        if file_type == "gctf":
            defocus_df = gctf_read(input_data)
        elif file_type == "ctffind4":
            defocus_df = ctffind4_read(input_data)
        elif file_type == "warp":
            defocus_df = warp_ctf_read(input_data)
        else:
            raise ValueError(f"The file type {file_type} is not supported.")
    else:  # isinstance(input_data, np.ndarray):
        df_columns = ["defocus1", "defocus2", "astigmatism", "phase_shift", "defocus_mean"]
        defocus_df = pd.DataFrame(input_data, columns=df_columns)

    return defocus_df


def gctf_read(file_path):
    """This function reads in a gctf starfile and returns a pandas dataframe with the following columns:
    defocus1, defocus2, astigmatism, phase_shift, and defocus_mean. All defocii values are in micrometers.

    Parameters
    ----------
    file_path: str
        Path to the gctf star file that contains values for all tilts in the tilt series. The columns to be read in are
        "rlnDefocusU", "rlnDefocusV", "rlnDefocusAngle", and "rlnPhaseShift" (if present, otherwise the phase shift
        is set to 0.0). The defocii values are converted to micrometers.

    Returns
    -------
    pandas.DataFrame
        A dataframe with defocus1, defocus2, astigmatism, phase_shift, and defocus_mean columnes. All defocii values
        are in micrometers.

    """

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
    """This function reads in a ctffind4 file (typically .txt) and returns a pandas dataframe with the following columns:
    defocus1, defocus2, astigmatism, phase_shift, and defocus_mean. All defocii values are in micrometers.

    Parameters
    ----------
    file_path: str
        Path to the ctffind4 file (typically in .txt format) that contains values for all tilts in the tilt series.
        The defocii values are converted to micrometers.

    Returns
    -------
    pandas.DataFrame
        A dataframe with defocus1, defocus2, astigmatism, phase_shift, and defocus_mean columnes. All defocii values
        are in micrometers.

    """

    ctf = pd.read_csv(file_path, skiprows=5, header=None, dtype=np.float64, delim_whitespace=True)
    converted_ctf = ctf.iloc[:, 1:5].copy()
    converted_ctf.loc[:, converted_ctf.columns[0:2]] *= 10e-5
    converted_ctf.columns = ["defocus1", "defocus2", "astigmatism", "phase_shift"]

    converted_ctf["defocus_mean"] = (converted_ctf["defocus1"] + converted_ctf["defocus2"]).values / 2.0
    return converted_ctf


def one_value_per_line_read(file_path, data_type=np.float32):
    """This function reads in a file with one value per line and returns them as numpy ndarray. The values are expected
    to be in the format specified in data_type.

    Parameters
    ----------
    file_path: str
        Path to the file where on each line there is expected to be a one value of the type specified by data_type.
    data_type: dtype, default=np.float32
        A typde of the data to be read in.

    Returns
    -------
    numpy.ndarray
        A ndarray with values of the type data_type.
    """

    data_df = pd.read_csv(file_path, header=None, dtype=data_type, delim_whitespace=True)
    return data_df.iloc[:, 0].values


def total_dose_load(input_dose):
    """Load total dose for single tilt series that should be used for dose-filtering/weighting.

    Parameters
    ----------
    input_dose : str or array-like
        The input dose. If ndarray, it is returned as is. If str, it can be a path to a csv, xml (warp), mdoc
        or a file with one value per line for each tilt image in the tilt series (any extension, typically .txt).
        The values should correspond to the total dose applied to each tilt image (i.e., low values for tilts acquired
        as first, large values for the tilt images acqured as last). If mdoc file is used the total dose is corrected
        either as PriorRecordDose + ExposureDose for each image, or as ExposureDose * (ZValue + 1) (starting
        from 1). The latter will work only if the ZValue corresponds to the order of acquisition, i.e, for tilt series
        that are not sorted from min to max tilt angle.


    Returns
    -------
    numpy.ndarray
        The corrected dose.

    Raises
    ------
    ValueError
        If the input dose is neither ndarray or a valid path to a file with the total dose.
    """

    if isinstance(input_dose, np.ndarray):
        return input_dose
    elif isinstance(input_dose, str):
        if input_dose.endswith(".csv"):
            # load as panda frames
            return pd.read_csv(input_dose, usecols=["Corrected_dose"], squeeze=True).values
        elif input_dose.endswith(".mdoc"):
            # load as mdoc
            mdoc_file = mdoc.Mdoc(input_dose)
            # should always exist
            image_dose = mdoc_file.get_image_feature("ExposureDose").values

            # if PriorDose exists - it should be used
            if "PriorRecordDose" in mdoc_file.imgs:
                prior_dose = mdoc_file.get_image_feature("PriorRecordDose").values
                total_dose = image_dose + prior_dose
            else:
                z_values = (
                    mdoc_file.get_image_feature("ZValue").values.astype(int) + 1
                )  # This assumes that z value corresponds to the order of acquisition; correct?
                total_dose = image_dose * z_values

            return total_dose
        elif input_dose.endswith(".xml"):
            total_dose = get_data_from_warp_xml(input_dose, "Dose", node_level=1)
            return total_dose
        else:
            total_dose = one_value_per_line_read(input_dose)
            return total_dose
    else:
        ValueError("Error: the dose has to be either ndarray or str with valid path!")


def tlt_load(input_tlt):
    """This function loads in tilt angles in degrees and returns them as ndarray. The input can be either a path to the
    file or an ndarray of tilts. The function will check if the input is already an array, and if not it will read in
    the data from the specified file type.

    Parameters
    ----------
    input_tlt: str or array-like
        The input tilt data. If it is a numpy array, it will be returned as is. If it is a string, it can be a path to
        a mdoc file, a xml file (warp) or any file where the angles are stored one per line (e.g. tlt, rawtlt,
        csv, .txt file).

    Returns
    -------
    ndarray
        The tilt angles (in degrees) in the form of a numpy array.

    Notes
    -----
    In case of mdoc file, the entries are expected to be sorted from min to max tilt angle. No sorting is done within
    this function.

    Raises
    ------
    ValueError
        If the input_tlt is neither a numpy array nor a valid file path.

    """

    if isinstance(input_tlt, np.ndarray):
        return input_tlt
    elif isinstance(input_tlt, str):
        if input_tlt.endswith(".mdoc"):
            tilt_data = mdoc.Mdoc(input_tlt)
            tilts = tilt_data.get_image_feature("TiltAngle").values
        elif input_tlt.endswith(".xml"):
            tilts = get_data_from_warp_xml(input_tlt, "Angles", node_level=1)
        else:
            tilts = one_value_per_line_read(input_tlt)

        return tilts
    else:
        ValueError("Error: the dose has to be either ndarray or path to csv, mdoc, or tlt file!")


def dimensions_load(input_dims):
    """Loads tomogram dimensions from a file or numpy.ndarray.

    Parameters
    ----------
    input_dims : str or pandas.DataFrame or array-like
        Either a path to a file with the dimensions, array-like input or pandas.DataFrame. The shape of the input should
        be 1x3 (x y z) in case of one tomogram or Nx4 for multiple tomograms (tomo_id x y z). In case of file, the
        dimensions can be fetched from .com file (typically tilt.com file) from parameters FULLIMAGE (x,y) and
        THICKNESS (z) or from general file with either 1x3 values on a single line or Nx4 values on N lines (separator
        is a space(s)).

    Returns
    -------
    pandas.DataFrame
        Dimensions of a tomogram in x, y, z (shape 1x3, columns "x","y","z") or tomogram idx and corresponding dimensions
        (shape Nx4 where N is the number of tomograms, columns "tomo_id", "x", "y", "z").

    Raises
    ------
    ValueError
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
    """Loads tomogram z-shift from a file, number or numpy.ndarray.

    Parameters
    ----------
    input_dims : str or number or pandas.DataFrame or array-like
        Either a path to a file with z-shift, single number, pandas.DataFrame or numpy.ndarray. If the z-shift should
        be loaded for more than one tomogram and is different for each tomogram the shape of the input should be Nx2
        where N is number of tomograms. In the first column should be tomogram id, in the second one corresponding
        z-shift. In case the input is an array in the file (typically with .txt extension but it does not matter),
        the file should have two values per line - tomo_id and z-shift. The separator is space(s). In case the input
        is read from IMOD's .com files the second value from "SHIFT" parameter is used.

    Returns
    -------
    pandas.DataFrame
        Z-shift for a tomogram (with a single "z_shift" column) or for multiple tomograms (with columns "tomo_id",
        "z_shift").

    Raises
    ------
    ValueError
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
            f"The z_shift should be one number or Nx2, where N is number of tomograms."
            f"Instead following shape was extracted from the prvoided files: {z_shift.shape}."
        )

    return z_shift


def imod_com_read(filename):
    """Reads a file in IMOD's .com format and returns a dictionary containing the data.

    Parameters
    ----------
    filename : str
        The name of the IMOC .com file to be read. All lines starting with # or $ are ignored, the rest is read in as
        dictionary. The keys are the first words of each line, and the values are the remaining words converted to the
        correct type.

    Returns
    -------
    dict
        A dictionary containing the data read from the file.

    Notes
    -----
    - Lines starting with '#' or '$' are ignored.
    - Numeric values are converted to integers if they are digits, and to floats if they are floating-point numbers.
    - Non-numeric values are stored as strings.
    """

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
