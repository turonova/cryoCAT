import argparse
import numpy as np
import re
from cryocat import wedgeutils
from cryocat import tmana
from numpydoc.docscrape import NumpyDocString


def parse_allowed_types(input_string):
    """Takes a string containing type descriptions (like from a docstring)
    and extracts the supported types while filtering out types that cannot be used
    on command line interfaces (like pandas dataframes).

    Parameters
    ----------
    input_string : str
        String containing type descriptions, typically from a function docstring.
        Can contain multiple types separated by commas, "and", or "or".

    Returns
    -------
    list of str
        Sorted list of supported type names with unsupported types filtered out.
    """
    # Define the words to filter out because they cannot be passed on command line
    unsupported_types = {"pandas", "dataframe", "pandas.dataframe"}

    # Split the input string by commas, "and", or "or"
    input_types = re.split(r",| and | or ", input_string)

    # Remove leading and trailing whitespace from each word
    input_types = [word.strip() for word in input_types]

    # Filter out empty strings and forbidden words
    input_types = [word for word in input_types if word and word.lower() not in unsupported_types]

    return sorted(input_types)


def parse_string_into_array(s):
    """Convert the input string to different data types
    in order: int, float, str. It returns a numpy array with the determined type.

    Parameters
    ----------
    s : str
        Comma-separated string of values to convert into an array.

    Returns
    -------
    numpy.ndarray
        Array containing the converted values with automatically detected dtype.
    """
    # Split the input string by commas
    values = s.split(",")

    # Try to convert the first value to different types until successful
    for dtype in (int, float, str):
        try:
            # Convert the values to the determined data type
            values = np.array([dtype(x) for x in values])
            break
        except ValueError:
            pass

    return values


def parse_choices(input_string):
    """Used to parse choice specifications from docstrings
    that are enclosed in curly braces. It automatically detects the type of choices
    (int, float, or str) and returns a list of converted values.

    Parameters
    ----------
    input_string : str
        String containing choice options, typically enclosed in curly braces.
        Example: "{1, 2, 3}" or "{'a', 'b', 'c'}" or "{1.0, 2.0, 3.0}"

    Returns
    -------
    list
        List of choice values converted to the appropriate Python type.

    """
    # Remove curly braces and split the string by comma
    s = input_string.strip("{}")
    elements = s.split(",")

    # Determine the type of the first element
    first_element = elements[0].strip()
    if first_element.isdigit():
        return [int(e.strip()) for e in elements]
    elif first_element.replace(".", "").isdigit():
        return [float(e.strip()) for e in elements]
    else:
        return [str(e.strip().strip('"')) for e in elements]


def parse_doc_param(doc_param):
    """Extracts parameter information from docstring format and converts it
    into components needed for creating argparse arguments.

    Parameters
    ----------
    doc_param : tuple
        A tuple from NumpyDocString containing parameter information.
        Expected format: (name, type_description, description_lines)

    Returns
    -------
    tuple
        A tuple containing:
        - param_name (str): Parameter name with "--" prefix
        - help_desc (str): Help description text
        - required_param (bool): Whether the parameter is required
        - param_types (list): List of allowed types for the parameter
        - default_value : Default value for optional parameters
        - choices (list): List of valid choices for the parameter

    Raises
    ------
    ValueError
        If the parameter description format is not recognized.
    """
    param_name = "--" + doc_param[0]
    help_desc = " ".join(doc_param[2])
    help_desc = replace_cross_references(help_desc)

    default_value = None
    required_param = False
    choices = []

    if "optional" in doc_param[1]:
        param_types = parse_allowed_types(doc_param[1].split("optional")[0].strip()[:-1])
    elif "default=" in doc_param[1]:
        param_types = parse_allowed_types(doc_param[1].split("default=")[0].strip()[:-1])
        default_value = eval(doc_param[1].split("default=")[1].strip())
    elif "{" in doc_param[1]:
        choices = parse_choices(doc_param[1].split("{")[1])
        param_types = type(choices[0])
        default_value = choices[0]
    else:
        required_param = True
        param_types = parse_allowed_types(doc_param[1])

    return param_name, help_desc, required_param, param_types, default_value, choices


def parse_input_types(input_value, allowed_types):
    """Handles both single values and comma-separated arrays, automatically
    detecting and converting to the appropriate data type. It supports int, float, str,
    bool, and array-like types.

    Parameters
    ----------
    input_value : str
        The input value string to parse. Can be a single value or comma-separated values.
    allowed_types : list
        List of allowed type names for this parameter.

    Returns
    -------
    various
        The converted value, which can be a single value or numpy array depending on input.

    Raises
    ------
    argparse.ArgumentTypeError
        If the input value cannot be converted to any of the allowed types.
    """
    def check_single_value(value, spec_types=None):

        if spec_types is None:
            basic_types = [int, float, str]
        else:
            basic_types = []
            for b in spec_types:
                basic_types.append(eval(b))

        for type_ in basic_types:
            try:
                return type_(value)
            except ValueError:
                continue

        # If conversion fails for all allowed types, raise an error
        raise argparse.ArgumentTypeError(f"Could not convert '{value}' to any of the allowed types: {basic_types}")

    is_array = True if len(input_value.split(",")) > 1 else False

    if is_array:
        if allowed_types[0] == "array-like":
            first_value = input_value.split(",")[0]
            conv_value = check_single_value(first_value)

            if isinstance(conv_value, (int, float)):
                conv_value = np.fromstring(input_value, dtype=type(conv_value), sep=",")
            elif isinstance(conv_value, bool):
                bool_str_list = input_value.split(",")
                # Convert each substring to a boolean value
                bool_map = {"True": True, "False": False, "0": False, "1": True}
                bool_list = [bool_map[s] for s in bool_str_list]
                # Create a NumPy array from the list of booleans
                conv_value = np.array(bool_list)
            else:  # string
                conv_value = np.array(input_value.split(","))
    else:
        if allowed_types[0] == "array-like":
            conv_value = check_single_value(input_value)
        else:
            conv_value = check_single_value(input_value, allowed_types)

    return conv_value


def replace_cross_references(input_string):
    """Cleans up docstring text by removing reStructuredText
    cross-reference markers that are not needed for command-line help.

    Parameters
    ----------
    input_string : str
        Input string potentially containing cross-reference markers.

    Returns
    -------
    str
        Cleaned string with cross-reference markers removed.

    Examples
    --------
    >>> replace_cross_references("See :meth:`other_function` for details")
    'See `other_function` for details'
    """
    if ":meth:" in input_string:
        input_string = input_string.replace(":meth:", "")

    return input_string


def add_params_from_docstring(subparsers, function_name, function_path):
    """Parses a function's numpy-style docstring and automatically
    creates argparse arguments for each parameter described in the docstring.
    It separates required and optional arguments into different groups.

    Parameters
    ----------
    subparsers : argparse._SubParsersAction
        The subparsers object to add the new parser to.
    function_name : str
        Name of the function/subcommand.
    function_path : function
        The actual function object whose docstring will be parsed.

    Returns
    -------
    None

    Notes
    -----
    Relies on the function having a properly formatted numpy-style
    docstring with a "Parameters" section. Each parameter should be documented
    with its type, description, and optionally default values or choices.
    """
    np_doc = NumpyDocString(function_path.__doc__)
    parser = subparsers.add_parser(
        function_name,
        description=" ".join(np_doc["Summary"]),
        help=" ".join(np_doc["Summary"]),
        add_help=False,
    )

    # parser = argparse.ArgumentParser(add_help=False, description=" ".join(np_doc["Summary"]))

    required = parser.add_argument_group("Required arguments")
    optional = parser.add_argument_group("Optional arguments")

    for p in range(len(np_doc["Parameters"])):
        param_name, help_desc, required_param, param_types, default_value, choices = parse_doc_param(
            np_doc["Parameters"][p]
        )
        if required_param:
            required.add_argument(
                param_name,
                help=help_desc,
                type=lambda x, param_types=param_types: parse_input_types(x, allowed_types=param_types),
                required=required_param,
                dest=param_name[2:],
                metavar=("value_" + param_name[2:]).upper(),
            )
        else:
            if len(choices) > 0:
                print(choices, param_types)
                optional.add_argument(
                    param_name,
                    help=help_desc,
                    type=param_types,  # lambda x, param_types=param_types: parse_input_types(x, allowed_types=param_types),
                    required=required_param,
                    dest=param_name[2:],
                    default=default_value,
                    metavar=("value_" + param_name[2:]).upper(),
                    choices=choices,
                )
            else:
                optional.add_argument(
                    param_name,
                    help=help_desc,
                    type=lambda x, param_types=param_types: parse_input_types(x, allowed_types=param_types),
                    required=required_param,
                    dest=param_name[2:],
                    default=default_value,
                    metavar=("value_" + param_name[2:]).upper(),
                )

    # Add back help
    optional.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit."
    )


def parse_arguments(f_dict, description):
    """Creates a main argument parser with subcommands based on the
    provided function dictionary. It automatically generates help and argument
    parsing based on the functions' docstrings.

    Parameters
    ----------
    f_dict : dict
        Dictionary mapping subcommand names to function objects.
    description : str
        Description for the main argument parser.

    Returns
    -------
    None
    """
    # Create the main parser
    parser = argparse.ArgumentParser(description=description)

    # Create subparsers
    subparsers = parser.add_subparsers(title="Options", dest="option")

    for key, value in f_dict.items():
        add_params_from_docstring(subparsers, key, value)

    # Parse the arguments
    args = parser.parse_args()

    # Call the appropriate function based on the subcommand
    for key, value in f_dict.items():
        if args.option == key:
            del args.option
            value(**vars(args))


def wedge_list():
    """Provides a command-line interface with subcommands for
    generating wedge lists using different methods and formats. It supports
    both Stopgap batch and regular Stopgap formats.

    Subcommands
    -----------
    stopgap_batch : function
        Create wedge lists in Stopgap batch format.
    stopgap : function
        Create wedge lists in regular Stopgap format.

    Examples
    --------
    >>> wedge_list()  # Called from command line
    # To get help: wedge_list --help
    # To get help on specific subcommand: wedge_list stopgap --help
    """
    f_dict = {"stopgap_batch": wedgeutils.create_wedge_list_sg_batch, "stopgap": wedgeutils.create_wedge_list_sg}
    description = (
        "Function to create wedge lists in different formats. For help on specific option run wedge_list option --help"
    )
    parse_arguments(f_dict, description=description)


def tm_ana():
    """Provides a command-line interface for particle extraction
    from template matching results.

    Subcommands
    -----------
    extract_particles : function
        Extract particles from template matching results.

    Examples
    --------
    >>> tm_ana()  # Called from command line
    # To get help: tm_ana --help
    # To get help on specific subcommand: tm_ana extract_particles --help
    """
    f_dict = {"extract_particles": tmana.scores_extract_particles}
    description = (
        "Function to extract particles from template matching. For help on specific option run wedge_list option --help"
    )
    parse_arguments(f_dict, description=description)
