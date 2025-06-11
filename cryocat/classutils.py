import importlib
import inspect
import sys
import types
import re
import numpy as np
from numpydoc.docscrape import NumpyDocString


def filter_strings(input_list, filter_contains=None, filter_exclude=None):

    # Normalize filters to lists
    if filter_contains is None:
        filter_contains = []
    elif isinstance(filter_contains, str):
        filter_contains = [filter_contains]

    if filter_exclude is None:
        filter_exclude = []
    elif isinstance(filter_exclude, str):
        filter_exclude = [filter_exclude]

    filtered = []
    for item in input_list:
        if filter_contains and not any(substr in item for substr in filter_contains):
            continue
        if any(substr in item for substr in filter_exclude):
            continue
        filtered.append(item)

    return filtered


def get_class_names_by_parent(parent_class_name: str, module_name: str, filter_contains=None, filter_exclude=None):
    """
    Get class names that are subclasses of a specified parent class in a given module.

    Parameters
    ----------
    parent_class_name : str
        Name of the parent class to check against for subclasses.
    module_name : str
        Name of the module to inspect (e.g., 'my_module.submodule').

    Returns
    -------
    list of str
        A list of class names that are subclasses of the specified parent class,
        excluding the parent class itself and only including classes defined in
        the specified module.
    """
    module = sys.modules.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)

    # Get the actual class object from the module using its name
    parent_class = getattr(module, parent_class_name, None)
    if parent_class is None or not inspect.isclass(parent_class):
        raise ValueError(f"'{parent_class_name}' is not a valid class in module '{module_name}'.")

    # Now find all subclasses of parent_class
    class_names = []
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if issubclass(cls, parent_class) and cls is not parent_class and cls.__module__ == module.__name__:
            class_names.append(name)

    class_names = filter_strings(class_names, filter_contains=filter_contains, filter_exclude=filter_exclude)
    return class_names


def get_classes_from_names(class_names, module_name):

    module = importlib.import_module(module_name)

    if not isinstance(class_names, list):
        if isinstance(class_names, str):
            return getattr(module, class_names)
        else:
            return class_names

    return [name if isinstance(name, type) else getattr(module, name) for name in class_names]


def get_class_names_by_prefix(prefix):
    """Get class names in the current module that start with a specified prefix.

    Parameters
    ----------
    prefix : str
        The prefix to filter class names.

    Returns
    -------
    list of str
        A list of class names that start with the given prefix and are defined in the current module.

    Examples
    --------
    >>> get_class_names_by_prefix('My')
    ['MyClass1', 'MyClass2']
    """

    current_module = sys.modules[__name__]
    class_names = [
        name
        for name, cls in inspect.getmembers(current_module, inspect.isclass)
        if name.startswith(prefix) and cls.__module__ == __name__
    ]
    return class_names


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


def parse_doc_param(doc_param, add_prefix=""):

    param_name = add_prefix + doc_param[0]
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

    # if not isinstance(list, param_types):
    #    param_types = [param_types]

    return param_name, help_desc, required_param, param_types, default_value, choices


def replace_cross_references(input_string):

    if ":meth:" in input_string:
        input_string = input_string.replace(":meth:", "")

    return input_string


def process_method_docstring(path_to_method, method_name, pretty_print=False):

    method_obj = inspect.getattr_static(path_to_method, method_name)
    docstring = inspect.getdoc(method_obj)
    np_doc = NumpyDocString(docstring)

    params_dict = {}
    for p in np_doc["Parameters"]:
        param_name, help_desc, required_param, param_types, default_value, choices = parse_doc_param(p)
        original_name = param_name
        if pretty_print:
            param_name = param_name.capitalize().replace("_", " ")

        params_dict[param_name] = {
            "desc": help_desc,
            "required": required_param,
            "types": param_types,
            "default": default_value,
            "options": choices,
            "name": original_name,
        }

    return params_dict
