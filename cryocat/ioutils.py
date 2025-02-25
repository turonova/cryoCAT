import numpy as np
import pandas as pd
import json
from copy import deepcopy
import re
import os
from cryocat import starfileio as sf
from cryocat import mdoc
import xml.etree.ElementTree as ET


def get_file_encoding(file_path):
    """Detects the encoding of a file by trying a list of common encodings.

    Parameters
    ----------
    file_path : str
        The path to the file for which the encoding needs to be determined.

    Returns
    -------
    str
        The name of the encoding if the file is successfully read.

    Raises
    ------
    UnicodeEncodeError
        If the file cannot be read with any of the tried encodings.

    Examples
    --------
    >>> get_file_encoding("example.txt")
    'utf-8'
    """

    encodings = ["utf-8", "iso-8859-1", "windows-1252"]
    for encoding in encodings:
        try:
            with open(file_path, "r", encoding=encoding) as file:
                _ = file.read()
            return encoding
        except UnicodeDecodeError:
            pass
    else:
        raise UnicodeEncodeError(f"Failed to read file {file_path} with any of the tried encodings.")


def get_all_files_matching_pattern(filename_pattern, numeric_wildcards_only=False, return_wildcards=True):
    """Get all files in a directory that match a specified filename pattern.

    Parameters
    ----------
    filename_pattern : str
        The pattern to match filenames against, which can include wildcards.
    numeric_wildcards_only : bool, default=False
        If True, only files with numeric wildcard parts will be included. Defaults to False.
    return_wildcards : bool, default=True
        If True, the function returns a tuple of (file_names, wildcards). If False, only file_names are returned.
        Defaults to True.

    Returns
    -------
    list
        A list of file paths that match the given pattern. If return_wildcards is True,
        a tuple of (file_names, wildcards) is returned, where wildcards are the parts of the filenames that
        matched the wildcard in the pattern.

    Raises
    ------
    FileNotFoundError
        If the specified directory does not exist.

    Notes
    -----
    The function uses regular expressions to match the filenames against the provided pattern. The '*' character
    in the pattern is treated as a wildcard that can match any sequence of characters.
    """

    # Split the pattern into directory and base name
    dir_name, base_pattern = os.path.split(filename_pattern)
    dir_name = dir_name or "."  # Use current directory if none provided

    # Escape the base pattern and replace '*' with a regex group
    pattern_regex = re.escape(base_pattern).replace(r"\*", r"(.*)")

    # List all files in the directory
    try:
        files_in_dir = os.listdir(dir_name)
    except FileNotFoundError:
        raise FileNotFoundError(f"Directory '{dir_name}' does not exist.")

    # Match files against the pattern regex
    wildcards = []
    file_names = []

    # No regex to be found
    if "*" not in filename_pattern:
        return [filename_pattern] if not return_wildcards else ([filename_pattern], [])

    for file_name in files_in_dir:
        full_path = os.path.join(dir_name, file_name)
        if os.path.isfile(full_path):
            match = re.match(pattern_regex, file_name)
            if match:
                wildcard_part = match.group(1)  # Extract the wildcard portion
                if numeric_wildcards_only:
                    try:
                        _ = int(wildcard_part)  # Try to convert the wildcard part to a number
                        file_names.append(full_path)
                        wildcards.append(wildcard_part)
                    except ValueError:
                        print(f"File {file_name} does not have numeric id and will be skipped.")
                else:
                    wildcards.append(wildcard_part)
                    file_names.append(full_path)

    if return_wildcards:
        return file_names, wildcards
    else:
        return file_names


def sort_files_by_idx(file_list, idx_list, order="ascending"):
    """Sorts a list of files based on corresponding indices.

    Parameters
    ----------
    file_list : list of str
        A list of file names to be sorted.
    idx_list : list of str
        A list of indices as strings corresponding to the file names.
    order : str, default='ascending'
        The order in which to sort the files. Can be 'ascending' or 'descending'.
        Defaults to 'ascending'.

    Returns
    -------
    numpy.ndarray
        An array of file names sorted according to the specified order of indices.

    Raises
    -------
    ValueError
        If idx_list and file_list aren't of list type.
        If idx_list doesn't contain only integers, or if file_list doesn't contain only strings.

    Examples
    --------
    >>> sort_files_by_idx(['file1.txt', 'file2.txt', 'file3.txt'], ['2', '1', '3'])
    array(['file2.txt', 'file1.txt', 'file3.txt'])

    >>> sort_files_by_idx(['file1.txt', 'file2.txt', 'file3.txt'], ['2', '1', '3'], order='descending')
    array(['file3.txt', 'file1.txt', 'file2.txt'])
    """

    if not isinstance(idx_list, list) or any(not isinstance(item, str) for item in idx_list) or len(idx_list) == 0:
        raise ValueError(f"idx_list must be a list of strings")
    if not isinstance(file_list, list) or any(not isinstance(item, str) for item in file_list) or len(file_list) == 0:
        raise ValueError(f"file_list must be a list of strings")

    if any(not x.isdigit() for x in idx_list):
        raise ValueError(f"idx_list can't contain elements that can't be converted to integer.")
    else:
        int_array = np.array([int(s) for s in idx_list])

    # indices can't be empty, can't be outside of file_list boundaries, indices can't be repeated
    if np.any(int_array > len(file_list)) or np.any(int_array < 1) or np.any(np.bincount(int_array) > 1):
        raise ValueError(f"idx_list contains invalid indices")

    if order == "ascending":
        sorted_indices = np.argsort(int_array)
    elif order == "descending":
        sorted_indices = np.argsort(int_array)[::-1]
    else:
        raise ValueError(f"Order must be ascending or descending")

    file_array = np.array(file_list)
    return file_array[sorted_indices]


def get_files_prefix_suffix(dir_path, prefix="", suffix=""):
    """Retrieve files from a specified directory that start with a given prefix and end with a given suffix.

    Parameters
    ----------
    dir_path : str
        The path to the directory from which to retrieve files.
    prefix : str, default=""
        The prefix that the files should start with. If ommited, no filtering based on prefix will be done. Defaults
        to an empty string.
    suffix : str, default=""
        The suffix that the files should end with. If ommited, no filtering based on suffix will be done. Defaults
        to an empty string.

    Returns
    -------
    list
        A list of filenames that match the given prefix and suffix criteria.

    Raises
    -------
    ValueError
        If file does not exist or if specified from file_path is not readable

    Examples
    --------
    >>> get_files_prefix_suffix('/path/to/dir', prefix='test', suffix='.txt')
    ['test_file1.txt', 'test_file2.txt']
    """
    if not os.path.exists(dir_path):
        raise ValueError(f"the directory '{dir_path}' does not exist.")

    matching_files = []
    for filename in os.listdir(dir_path):
        if filename.startswith(prefix) and filename.endswith(suffix):
            matching_files.append(filename)

    return sorted(matching_files)


def get_number_of_lines_with_character(filename, character):
    """Count the number of lines in a file that start with a specified character.

    Parameters
    ----------
    filename : str
        The path to the file to be read.
    character : str
        The character to check at the start of each line.

    Returns
    -------
    int
        The number of lines starting with the specified character.
    """

    # Open the file for reading
    with open(filename, "r") as file:
        # Count the number of lines starting with specified character
        count = sum(1 for line in file if line.startswith(character))

    return count


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
    if isinstance(value, bool):  # float(True) == 1.0, but we don't accept bool as valid
        return False

    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def get_filename_from_path(input_path, with_extension=True):
    """Get the filename from the given input path.

    Parameters
    ----------
    input_path: str
        The input path from which the filename is to be extracted.
    with_extension: bool, default=True
        Flag to indicate whether to include the file extension in the filename. Default is True.

    Returns:
        str
            The extracted filename from the input path.
    """

    # Get the filename without the path
    filename = os.path.basename(input_path)

    if not with_extension:
        # Remove the extension from the filename
        filename = os.path.splitext(filename)[0]

    return filename


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
    Value Error
        If node_level isn't 1 or 2

    Examples
    --------
    >>> data = get_data_from_warp_xml('path/to/xml/file.xml', 'GridCTF', node_level=2)
    """

    if node_level not in [1, 2]:
        raise ValueError(f"node_level can't be {node_level}, must be 1 or 2.")
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
                data = [float(node.get("Value")) for node in node_elements]
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
        if file_type.lower() == "gctf":
            defocus_df = gctf_read(input_data)
        elif file_type.lower() == "ctffind4":
            defocus_df = ctffind4_read(input_data)
        elif file_type.lower() == "warp":
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
    rows_to_skip = get_number_of_lines_with_character(file_path, "#")
    ctf = pd.read_csv(file_path, skiprows=rows_to_skip, header=None, dtype=np.float32, delim_whitespace=True)
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

    Raises
    -------
    ValueError
        If file does not exist or if specified from file_path is not readable
    """
    if not os.path.isfile(file_path):
        raise ValueError("The input file does not exist")

    try:
        data_df = pd.read_csv(file_path, header=None, dtype=data_type, delim_whitespace=True)
        if data_df.empty:
            raise ValueError("The input file is empty or contains no valid data.")
    except pd.errors.EmptyDataError:
        raise ValueError("The input file is empty or contains no valid data.")

    return data_df.iloc[:, 0].values


def total_dose_load(input_dose, sort_mdoc=True):
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
        that are not sorted from min to max tilt angle or are sorted with their ZValue unchanged.
    sort_mdoc : bool, default=True
        Whether the mdoc should be sorted by the tilt angles. This parameter is relevant only if the provided input
        is mdof file. If True mdoc will be sorted from min to max tilt angle however the ZValue will be kept as it was
        so the dose can still be computed correctly. Defaults to True.

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
    elif isinstance(input_dose, list):
        return np.asarray(input_dose)
    elif isinstance(input_dose, str):
        if input_dose.endswith(".csv"):
            # load as panda frames
            df = pd.read_csv(input_dose, index_col=0)
            if "CorrectedDose" in df.columns:
                if "Removed" in df.columns:
                    return df.loc[df["Removed"] == False, "CorrectedDose"].astype(np.single).to_numpy()
                else:
                    return df["CorrectedDose"].astype(np.single).to_numpy()
            else:
                raise ValueError(f"The file {input_dose} does not contain column with name CorrectedDose")
        elif input_dose.endswith(".mdoc"):
            # load as mdoc
            mdoc_file = mdoc.Mdoc(input_dose)

            # sort mdoc
            if sort_mdoc:
                mdoc_file.sort_by_tilt(reset_z_value=False)

            # should always exist
            image_dose = mdoc_file.get_image_feature("ExposureDose").values

            # if PriorDose exists - it should be used
            if "PriorRecordDose" in mdoc_file.imgs:
                prior_dose = mdoc_file.get_image_feature("PriorRecordDose").values
                total_dose = image_dose + prior_dose
                return total_dose
            else:
                mdoc_file.imgs["original_order"] = range(len(mdoc_file.imgs))
                mdoc_file.imgs["DateTime"] = pd.to_datetime(mdoc_file.imgs["DateTime"])  # Convert to datetime
                sorted_df = mdoc_file.imgs.sort_values("DateTime")
                sorted_df.reset_index(drop=True, inplace=True)
                sorted_df["total_dose"] = sorted_df["ExposureDose"] * (sorted_df.index + 1)
                result_df = sorted_df.sort_values("original_order").drop(columns=["original_order"])
                return result_df["total_dose"].values

        elif input_dose.endswith(".xml"):
            total_dose = get_data_from_warp_xml(input_dose, "Dose", node_level=1)
            return total_dose
        else:
            total_dose = one_value_per_line_read(input_dose)
            return total_dose
    else:
        raise ValueError("Error: the dose has to be either ndarray or str with valid path!")


def rot_angles_load(input_angles, angles_order="zxz"):
    """Load rotation angles from a file or numpy array and arrange them in a specified order.

    Parameters
    ----------
    input_angles : str or numpy.ndarray
        If a string, it should be the path to a CSV file containing the angles (three per line). If a numpy array, it
        should directly contain the angles.
    angles_order : str, default="zxz"
        The order of the angles in the output array. Default is "zxz" (phi, theta, psi). If "zzx", the order will be
        adjusted to phi, psi, theta.

    Returns
    -------
    angles : numpy.ndarray
        A numpy array of shape (N, 3) where n is the number of angle sets. Each row contains the angles phi, theta,
          and psi in the specified order.

    Raises
    ------
    ValueError
        If `input_angles` is neither a string path to a CSV file nor a numpy array.

    Examples
    --------
    >>> rot_angles_load("path/to/angles.csv")
    array([[phi1, theta1, psi1],
           [phi2, theta2, psi2],
           ...])

    >>> rot_angles_load(numpy.array([[0, 45, 90], [90, 45, 0]]), "zzx")
    array([[0, 90, 45],
           [90, 0, 45]])
    """

    if isinstance(input_angles, str):
        # Not all strings are valid: file can not exist
        if not os.path.exists(input_angles):
            raise ValueError(f"File '{input_angles}' does not exist.")

        angles = pd.read_csv(input_angles, header=None)
        # Check valid data
        if len(angles.columns) != 3:
            raise ValueError(f"File '{input_angles}' does not contain valid data.")

        if angles_order == "zzx":
            angles.columns = ["phi", "psi", "theta"]
        else:
            angles.columns = ["phi", "theta", "psi"]

        angles = angles.loc[:, ["phi", "theta", "psi"]].to_numpy()

    elif isinstance(input_angles, np.ndarray):
        angles = input_angles.copy()
    else:
        raise ValueError("The input_angles have to be either a valid path to a file or numpy array!!!")

    return angles


def tlt_load(input_tlt, sort_angles=True):
    """This function loads in tilt angles in degrees and returns them as ndarray. The input can be either a path to the
    file or an ndarray of tilts. The function will check if the input is already an array, and if not it will read in
    the data from the specified file type.

    Parameters
    ----------
    input_tlt : str or array-like
        The input tilt data. If it is a numpy array, it will be returned as is. If it is a string, it can be a path to
        a mdoc file, a xml file (warp) or any file where the angles are stored one per line (e.g. tlt, rawtlt,
        csv, .txt file).
    sort_angles : bool, default=True
        Whether the tilts should be sorted from min to max tilt angle. Defaults to True.

    Returns
    -------
    ndarray
        The tilt angles (in degrees) in the form of a numpy array.

    Raises
    ------
    ValueError
        If the input_tlt is neither a numpy array nor a valid file path.
        If the input_tlt is an empty numpy array or an empty list.

    """

    if isinstance(input_tlt, np.ndarray):
        if input_tlt.size == 0:
            raise ValueError(f"The input tilt data is empty!")
        else:
            return input_tlt
    elif isinstance(input_tlt, list):
        if len(input_tlt) == 0:
            raise ValueError(f"The input tilt data is empty")
        else:
            return np.asarray(input_tlt)
    elif isinstance(input_tlt, str):
        if input_tlt.endswith(".mdoc"):
            tilt_data = mdoc.Mdoc(input_tlt)
            tilts = tilt_data.get_image_feature("TiltAngle").values
        elif input_tlt.endswith(".xml"):
            tilts = get_data_from_warp_xml(input_tlt, "Angles", node_level=1)
        else:
            tilts = one_value_per_line_read(input_tlt)

        if sort_angles:
            tilts = np.sort(tilts)

        return tilts
    else:
        raise ValueError("Error: the dose has to be either ndarray or path to csv, mdoc, or tlt file!")


def dimensions_load(input_dims, tomo_idx=None):
    """Load and process tomogram dimensions from various input formats.

    Parameters
    ----------
    input_dims : pd.DataFrame, str, list, or np.ndarray
        Either a path to a file with the dimensions, array-like input or pandas.DataFrame. The shape of the input should
        be 1x3 (x y z) in case of one tomogram or Nx4 for multiple tomograms (tomo_id x y z). In case of file, the
        dimensions can be fetched from .com file (typically tilt.com file) from parameters FULLIMAGE (x,y) and
        THICKNESS (z) or from general file with either 1x3 values on a single line or Nx4 values on N lines (separator
        is a space(s)).
    tomo_idx : str or array-like, optional
        Path to a file containing tomogram indices or an 1D array with the indices. It is used only if the input_dims
        do not contain 4 columns (i.e., do not have tomo_id). If provided, the function will replicate the 1x3
        dimensions to the length of tomo_idx array and will add "tomo_id" column. Defaults to None.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the dimensions with columns adjusted based on the input shape.
        Columns will be named as ["x", "y", "z"] or ["tomo_id", "x", "y", "z"].

    Raises
    ------
    ValueError
        If the dimensions do not conform to the expected shapes of 1x3 or Nx4 or if file does not exist.

    Notes
    -----
    - If `input_dims` is a string ending with ".com", it is assumed to be a path to a .com file
      and will be processed accordingly.
    - If `input_dims` is a string not ending with ".com", it is treated as a path to a CSV file.
    - The function can handle reshaping of input dimensions if they are provided as a list or a
      one-dimensional numpy array.
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
            else:
                raise ValueError(f"The file at the path {input_dims} does not exist.")
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

    if tomo_idx is not None:
        tomos = tlt_load(tomo_idx).astype(int)
        if "tomo_id" not in dimensions.columns:
            repeated_values = np.repeat(dimensions[["x", "y", "z"]].values, len(tomos), axis=0)
            dimensions = pd.DataFrame(repeated_values, columns=["x", "y", "z"])
            dimensions["tomo_id"] = tomos

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
        Wrong size of the input, unsupported input type, or not existing filepath.

    """
    if not isinstance(input_shift, (str, pd.DataFrame, float, int, list, np.ndarray)):
        raise ValueError(
            f"Unsupported input type: {type(input_shift)}. Expected str, DataFrame, float, int, list, or np.ndarray."
        )
    if isinstance(input_shift, pd.DataFrame):
        z_shift = input_shift
    elif isinstance(input_shift, str):
        if input_shift.endswith(".com"):
            com_file_d = imod_com_read(input_shift)
            z_shift = pd.DataFrame([com_file_d["SHIFT"][1]])
        else:
            if os.path.isfile(input_shift):
                z_shift = pd.read_csv(input_shift, sep="\s+", header=None, dtype=float)
            else:
                raise ValueError(f"File {input_shift} does not exist.")
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


def remove_lines(filename, lines_to_remove, start_str_to_skip=None, number_start=0, output_file=None):
    """Reads a file, removes specified lines while skipping those that start with given strings and returns/writes out
    the rest.

    Parameters
    ----------
    filename : str
        The name of the file to remove the lines from.
    lines_to_remove: int or array-like
        Array/list (or single int) with numbers of lines to be removed. If start_str_to_skip is empty, the indices
        corresponds to the line numbers.
    start_str_to_skip: str or array-like
        Array/list of strings (or single string). The lines starting with any of those strings will be ignored. The indices
        from lines_to_remove will be applied to filter only the remaining lines. Dafaults to None.
    number_start: int. default=0
        Whether the line numbers provied start counting at 0 or 1. Defaults to 0.
    output_file: str
        Path to a file to write out the content into. Defaults to None (no file will be written out).

    Returns
    -------
    list
        A list of lines that were kept.

    """

    filtered_lines = []
    if isinstance(lines_to_remove, int):
        lines_to_remove = [lines_to_remove]
    if start_str_to_skip is None:
        start_str_to_skip = []
    elif not isinstance(start_str_to_skip, list):
        start_str_to_skip = [start_str_to_skip]

    kept_counter = number_start
    with open(filename, "r") as file:
        for line in file:
            if any(line.startswith(s) for s in start_str_to_skip):
                filtered_lines.append(line)
            else:
                if kept_counter not in lines_to_remove:
                    filtered_lines.append(line)

                kept_counter += 1

    if output_file is not None:
        with open(output_file, "w") as of:
            of.writelines(filtered_lines)

    return filtered_lines

    """ lines_to_write = []

    if start_str_to_skip is None:
        start_str_to_skip = []
    elif not isinstance(start_str_to_skip, list):
        start_str_to_skip = [start_str_to_skip]

    with open(filename, "r") as file:
        for line in file:
            if not any(line.startswith(s) for s in start_str_to_skip):
                lines_to_write.append(line)

    filtered_lines = [line for i, line in enumerate(lines_to_write) if i not in lines_to_remove]

    if output_file is not None:
        with open(output_file, "w") as of:
            of.writelines(filtered_lines)

    return filtered_lines """


def dict_write(dict_data, file_name):
    """Write the given dictionary to a file in JSON format.

    Parameters
    ----------
    dict_data : dict
        Dictionary containing the data to write to the file.
    file_name : str
        The name of the file where the dictionary will be written.

    Returns
    -------
    None
    """

    with open(file_name, "w") as json_file:
        json.dump(dict_data, json_file, indent=4)


def dict_load(input_data):
    """Load a dictionary from a JSON string or copy an existing dictionary.

    Parameters
    ----------
    input_data : str or dict
        The input data to load. This can be a JSON string or an existing dictionary.

    Returns
    -------
    dict
        A dictionary loaded from the JSON string or a deep copy of the provided dictionary.

    Raises
    ------
    ValueError
        If `input_data` is neither a string nor a dictionary.

    Notes
    -----
    If `input_data` is a JSON string and cannot be decoded, an empty dictionary is returned and an error message is printed.

    Examples
    --------
    >>> json_str = '{"key": "value"}'
    >>> dict_load(json_str)
    {'key': 'value'}

    >>> original_dict = {'key': 'value'}
    >>> new_dict = dict_load(original_dict)
    >>> new_dict is original_dict
    False
    """

    if isinstance(input_data, str):
        try:
            dict_data = json.loads(input_data)
        except json.JSONDecodeError:
            print("Invalid JSON string.")
            dict_data = {}
    elif isinstance(input_data, dict):
        dict_data = deepcopy(input_data)
    else:
        raise ValueError("The supported formats are dict or file in JSON format.")

    return dict_data


def indices_load(input_data, numbered_from_1=True):
    """Load indices from a specified input source.

    Parameters
    ----------
    input_data : str, list, or numpy.ndarray
        The input data can be a file path to a CSV file, a text file containing indices (one per line), or a list/array
        of indices. If a CSV file is provided, it is expected to have a column named "ToBeRemoved".
    numbered_from_1 : bool, default=True
        If True, the returned indices will be adjusted to be zero-based (i.e., subtracting 1 from each index).
        Defaults to True.

    Returns
    -------
    numpy.ndarray
        An array of indices, adjusted based on the input data and the numbered_from_1 flag.

    Raises
    -------
    ValueError
        If input data isn't either a path to valid file either a list/array
    """

    if isinstance(input_data, str):
        if input_data.endswith(".csv"):
            df = pd.read_csv(input_data)
            if "Removed" in df.columns:
                df = df[~df["Removed"]]
            indices = df.index[df["ToBeRemoved"]].to_numpy(dtype=int)
            numbered_from_1 = False  # Always from 0
        else:
            indices = np.loadtxt(input_data, dtype=int)

    elif isinstance(input_data, list) or isinstance(input_data, np.ndarray):
        indices = np.asarray(input_data)
        if len(indices) == 0:
            raise ValueError(f"Input indices can't be empty")
    else:
        raise ValueError(f"Input data must be either path to a valid file either list/array")

    if numbered_from_1:
        indices = indices - 1

    return indices


def indices_reset(input_data):
    """Reset the indices of a CSV file by modifying specific columns.

    Parameters
    ----------
    input_data : str
        The path to the CSV file that needs to be processed.

    Returns
    -------
    None

    Notes
    -----
    This function reads a CSV file into a DataFrame, checks for the presence of a "Removed" column,
    and updates it based on the "ToBeRemoved" column. It then resets the "ToBeRemoved" column to
    False and saves the modified DataFrame back to the original CSV file.
    """

    df = pd.read_csv(input_data)

    if "Removed" in df.columns:
        df.loc[df["ToBeRemoved"], "Removed"] = True

    df["ToBeRemoved"] = False
    df.to_csv(input_data, index=False)


def defocus_remove_file_entries(
    input_file, entries_to_remove, file_type="gctf", numbered_from_1=True, output_file=None
):
    """Remove specified entries from a file and optionally update a specification file.

    Parameters
    ----------
    input_file : str
        The path to the input file from which entries will be removed.
    entries_to_remove : str, list, or numpy.ndarray
        The entries to remove can be specified as a file path to a CSV file, a text file containing indices
        (one per line), or a list/array of indices. If a CSV file is provided, it is expected to have a column
        named "ToBeRemoved".
    file_type : str, default='gctf'
        The type of the input file. Can be 'gctf' or "ctffind4'. Defaults to 'gctf'.
    numbered_from_1 : bool=True
        Indicates whether the entries in `entries_to_remove` are numbered from 1. Defaults to True.
    output_file : str, optional
        The path to the output file where the modified content will be saved. If None, the input_file will be overwritten.
        Defaults to None.

    Returns
    -------
    None
        The function modifies the input file and/or creates an output file as specified.

    Notes
    -----
    The function handles two file types: 'gctf' and 'ctffind4', applying different methods for removing lines based
    on the file type. The `indices_load` and `indices_reset` functions are used to manage the indices of entries to be
    removed and to reset them if necessary.
    """

    lines_to_remove = indices_load(entries_to_remove, numbered_from_1=numbered_from_1)

    if output_file is None:
        output_file = input_file

    if file_type.lower() == "gctf":
        sf.Starfile.remove_lines(
            input_file,
            lines_to_remove,
            output_file=output_file,
            data_specifier="data_",
            number_columns=True,
        )
    elif file_type.lower() == "ctffind4":
        _ = remove_lines(input_file, lines_to_remove, start_str_to_skip=["#"], output_file=output_file)
    else:
        print(f"The defocus filetype {file_type} is not supported and thus will not be cleaned.")
