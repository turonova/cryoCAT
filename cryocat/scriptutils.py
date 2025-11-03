import re
from cryocat import ioutils
import json


def parse_command(command_string, output_dict=None):
    """Splits command string into tokens and identifies command, flags, and parameters.
    Supports both -key value and -key=value formats.

    Parameters
    ----------
    command_string : str
        The command line string to parse
    output_dict : str, optional
        If provided, writes the parsed dictionary to this file

    Returns
    -------
    dict
        Dictionary with 'command' key and parameter key-value pairs

    Examples
    --------
    >>> parse_command("program --input file.txt -o output")
    {'command': 'program', '--input': 'file.txt', '-o': 'output'}

    >>> parse_command("cmd --key=value -flag=123")
    {'command': 'cmd', '--key': 'value', '-flag': '123'}
    """
    # Split the string by one or more spaces
    tokens = re.split(r"\s+", command_string.strip())
    #handle empty tokens?
    if not tokens or tokens == [""]:
        return {"command": ""}

    # Initialize the dictionary with the first word as "command"
    command_dict = {"command": tokens[0]}
    current_key = None #init
    # Iterate through the tokens
    for token in tokens[1:]:
        if "=" in token:
            # Handle cases like -key=value or --key=value
            key, value = token.split("=", 1)
            command_dict[key] = value
        elif token.startswith("-"):
            # Handle cases like -key value or --key value
            current_key = token
            command_dict[current_key] = ""  # Assign an empty string by default
        else:
            # Assign value to the most recent key
            if current_key is not None:
                command_dict[current_key] = token
                current_key = None  # Reset the current_key after assigning value
            else:
                # Misplaced value without preceding key (this part is less likely with the new format)
                print(f"Warning: Value '{token}' found without a preceding key. Ignoring it.")

    if output_dict is not None:
        ioutils.dict_write(command_dict, output_dict)

    return command_dict


def parse_command_for_gui(command_string, output_dict=None):
    """Similar to parse_command but formats parameter names for GUI display
    by removing dashes and capitalizing.

    Parameters
    ----------
    command_string : str
        The command line string to parse
    output_dict : str, optional
        If provided, writes the parsed dictionary as JSON to this file

    Returns
    -------
    dict
        Dictionary with 'Command' key and formatted parameter structures

    Examples
    --------
    >>> parse_command_for_gui("program --input file.txt")
    {'Command': 'program', 'Input': {'value': 'file.txt', 'param': '--input', 'tooltip': ''}}
    """
    # Split the string by one or more spaces
    tokens = re.split(r"\s+", command_string.strip())

    if not tokens or tokens == [""]:
        return {"Command": ""}

    # Initialize the dictionary with the first word as "command"
    command_dict = {"Command": tokens[0]}
    current_key = None #init

    # Iterate through the tokens
    for token in tokens[1:]:
        if "=" in token:
            # Handle cases like -key=value or --key=value
            key, value = token.split("=", 1)
            command_dict[key.lstrip("-").capitalize()] = {"value": value, "param": key, "tooltip": ""}
        elif token.startswith("-"):
            # Handle cases like -key value or --key value
            current_key = token
            command_dict[current_key.lstrip("-").capitalize()] = ""  # Assign an empty string by default
        else:
            # Assign value to the most recent key
            if current_key is not None:
                command_dict[current_key.lstrip("-").capitalize()] = {
                    "value": token,
                    "param": current_key,
                    "tooltip": "",
                }
                current_key = None  # Reset the current_key after assigning value
            else:
                # Misplaced value without preceding key (this part is less likely with the new format)
                print(f"Warning: Value '{token}' found without a preceding key. Ignoring it.")

    if output_dict is not None:
        with open(output_dict, "w") as f:
            json.dump(command_dict, f, indent=4)

    return command_dict


def process_cluster_params(script_file, cluster_params):
    """Similar to parse_command but formats parameter names for GUI display
    by removing dashes and capitalizing.

    Parameters
    ----------
    command_string : str
        The command line string to parse
    output_dict : str, optional
        If provided, writes the parsed dictionary as JSON to this file

    Returns
    -------
    dict
        Dictionary with 'Command' key and formatted parameter structures

    Examples
    --------
    >>> parse_command_for_gui("program --input file.txt")
    {'Command': 'program', 'Input': {'value': 'file.txt', 'param': '--input', 'tooltip': ''}}
    """
    # Check if cluster_params are specified
    if cluster_params is not None:
        cluster_params = ioutils.dict_load(cluster_params)

        for key, value in cluster_params.items():
            if key.startswith("--"):
                script_file.write(f"#SBATCH {key}={value}\n")
            elif key.startswith("-"):
                script_file.write(f"#SBATCH {key} {value}\n")


def parse_script_file(script_name):
    """Extracts interpreter, SBATCH parameters, module loads, variable assignments,
    and commands from a shell script file.

    Parameters
    ----------
    script_name : str
        Path to the shell script file to parse

    Returns
    -------
    dict
        Dictionary with keys: interpreter, sbatch_params, modules, values, coms_and_params

    Examples
    --------
    >>> result = parse_script_file('script.sh')
    >>> result['interpreter']
    '#!/bin/bash'
    >>> result['modules']
    ['python/3.9', 'cuda']
    """
    with open(script_name, "r") as file:
        lines = file.readlines()

    interpreter = lines[0].strip()  # Store the first line as interpreter
    sbatch_params = {}
    modules = []
    values = {}
    coms_and_params = []

    sbatch_pattern = re.compile(r"#SBATCH\s+(-{1,2}\S+)(\s+\S+)?")
    module_pattern = re.compile(r"^module load\s+(\S+)")
    value_pattern = re.compile(r"^(\S+)=(\S+)")
    command_pattern = re.compile(r"^(\S+)\s+(.*)")

    for line in lines[1:]:  # Skip the first line (interpreter)
        line = line.strip()

        if line.startswith("#"):
            if line.startswith("#SBATCH"):
                match = sbatch_pattern.findall(line)
                for key, value in match:
                    if key.startswith("--"):
                        # Split key=value if key starts with --
                        key, value = key.split("=", 1)
                    else:
                        key = key
                        value = value.strip() if value else ""
                    sbatch_params[key] = value
            # Skip any other lines starting with # that are not #SBATCH
            continue

        elif line.startswith("module load"):
            match = module_pattern.match(line)
            if match:
                modules.append(match.group(1))

        elif "=" in line:
            match = value_pattern.match(line)
            if match:
                key, value = match.groups()
                values[key] = value

        else:
            match = command_pattern.match(line)
            if match:
                coms_and_params.append(parse_command(line))

    return {
        "interpreter": interpreter,
        "sbatch_params": sbatch_params,
        "modules": modules,
        "values": values,
        "coms_and_params": coms_and_params,
    }


def generate_python_command(function_dict, import_dict=None):
    """Creates a python -c command string that calls a function with specified
    parameters and optional imports.

    Parameters
    ----------
    function_dict : dict
        Dictionary with 'function' key and parameter key-value pairs
    import_dict : dict, optional
        Dictionary mapping modules to imports

    Returns
    -------
    str
        Python command line string

    Examples
    --------
    >>> generate_python_command({'function': 'test', 'arg': 'value'})
    'python -c "test(arg='value')"\n'

    >>> generate_python_command({'function': 'func'}, {'numpy': 'array'})
    'python -c "from numpy import array; func()"\n'
    """
    command = 'python -c "'

    if import_dict:
        imports = "; ".join([f"from {k} import {v}" for k, v in import_dict.items()])
        imports = imports + "; "
    else:
        imports = ""

    function_name = function_dict.get("function", "") + "("

    function_variables = ", ".join(
        [f"{k}='{v}'" if isinstance(v, str) else f"{k}={v}" for k, v in function_dict.items() if k != "function"]
    )

    return f'{command}{imports}{function_name}{function_variables})"\n'


def generate_interactive_command(input_dict):
    """Creates a command string that uses heredoc syntax for interactive input.

    Parameters
    ----------
    input_dict : dict
        Dictionary with 'command' key and 'inputs' list

    Returns
    -------
    str
        Interactive command string with heredoc
    """
    command = input_dict.get("command", "") + " << foo"
    params = "\n".join([f"{k}" for k in input_dict["inputs"]])
    return f"{command}\n{params}\nfoo\n"


def generate_command_line(input_dict):
    """Converts a dictionary of command parameters into a executable command string.

    Parameters
    ----------
    input_dict : dict
        Dictionary with 'command' key and parameter key-value pairs

    Returns
    -------
    str
        Formatted command line string
    """
    command = input_dict.get("command", "")
    params = " ".join(
        [f"{k}={v}" if k.startswith("--") else f"{k} {v}" for k, v in input_dict.items() if k != "command"]
    )

    return f"{command} {params}\n"


def replace_command_in_script(script_file, input_dict):
    """Searches for a command in a script file and replaces it with a new command
    generated from the input dictionary.

    Parameters
    ----------
    script_file : str
        Path to the script file to modify
    input_dict : dict
        Dictionary with 'command' key and new parameters

    Returns
    -------
    bool
        True if command was found and replaced, False otherwise

    Raises
    ------
    ValueError
        If input dictionary doesn't have a 'command' key
    """
    command_to_replace = input_dict.get("command")
    if not command_to_replace:
        raise ValueError("Input dictionary must have a 'command' key.")

    with open(script_file, "r") as file:
        lines = file.readlines()

    new_lines = []
    command_replaced = False

    for line in lines:
        if line.startswith(command_to_replace) and not command_replaced:
            new_line = generate_command_line(input_dict) + "\n"
            new_lines.append(new_line)
            command_replaced = True  # Ensure only the first occurrence is replaced
        else:
            new_lines.append(line)

    with open(script_file, "w") as file:
        file.writelines(new_lines)

    return command_replaced


def generate_script(
    script_name,
    coms_and_params,
    script_header="#!/bin/bash",
    cluster_params=None,
    module_loads=None,
    template_script=None,
    parallelize=False,
    p_type="cluster",
    p_variable="tomo_list",
):
    """Creates a complete shell script with header, cluster parameters, module loads,
    and commands. Supports both list and dictionary inputs for modules.

    Parameters
    ----------
    script_name : str
        Output path for the generated script
    coms_and_params : list or str
        List of command dictionaries or path to command dictionary file
    script_header : str, optional
        Script header/shebang line
    cluster_params : str or dict, optional
        Cluster parameters for SBATCH directives
    module_loads : list or dict, optional
        Modules to load in the script
    template_script : str, optional
        Not currently implemented
    parallelize : bool, optional
        Not currently implemented
    p_type : str, optional
        Not currently implemented
    p_variable : str, optional
        Not currently implemented

    Raises
    ------
    ValueError
        If module_loads is not a list or dictionary
    """
    # Open the script_name file for writing
    with open(script_name, "w") as script_file:

        script_file.write(f"{script_header}\n")

        # Check if coms_and_params is a list, if not, turn it into a list
        if not isinstance(coms_and_params, list):
            coms_and_params = [coms_and_params]

        process_cluster_params(script_file=script_file, cluster_params=cluster_params)

        # Check if module_loads is not None
        if module_loads is not None:
            if isinstance(module_loads, list):
                for module in module_loads:
                    script_file.write(f"module load {module}\n")
            elif isinstance(module_loads, dict):
                modules = "\n".join([f"module load {v}" for v in module_loads.values()])
                script_file.write(f"{modules}\n")
            else:
                raise ValueError("module_loads should be a list of strings or dictionary.")

        # Go over the list of coms_and_params
        for item in coms_and_params:

            com_dict = ioutils.dict_load(item)
            command_line = generate_command_line(com_dict)
            script_file.write(f"{command_line}\n")
