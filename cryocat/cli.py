import argparse
import numpy as np
import re
from cryocat import wedgeutils
from cryocat import tmana
from numpydoc.docscrape import NumpyDocString


def parse_allowed_types(input_string):

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

    if ":meth:" in input_string:
        input_string = input_string.replace(":meth:", "")

    return input_string


def add_params_from_docstring(subparsers, function_name, function_path):

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

    f_dict = {"stopgap_batch": wedgeutils.create_wedge_list_sg_batch, "stopgap": wedgeutils.create_wedge_list_sg}
    description = (
        "Function to create wedge lists in different formats. For help on specific option run wedge_list option --help"
    )
    parse_arguments(f_dict, description=description)


def tm_ana():
    f_dict = {"extract_particles": tmana.scores_extract_particles}
    description = (
        "Function to extract particles from template matching. For help on specific option run wedge_list option --help"
    )
    parse_arguments(f_dict, description=description)
