import re
from cryocat import ioutils
import json


def parse_command(command_string, output_dict=None):

    # Split the string by one or more spaces
    tokens = re.split(r"\s+", command_string.strip())

    # Initialize the dictionary with the first word as "command"
    command_dict = {"command": tokens[0]}

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

    # Split the string by one or more spaces
    tokens = re.split(r"\s+", command_string.strip())

    # Initialize the dictionary with the first word as "command"
    command_dict = {"Command": tokens[0]}

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

    # Check if cluster_params are specified
    if cluster_params is not None:
        cluster_params = ioutils.dict_load(cluster_params)

        for key, value in cluster_params.items():
            if key.startswith("--"):
                script_file.write(f"#SBATCH {key}={value}\n")
            elif key.startswith("-"):
                script_file.write(f"#SBATCH {key} {value}\n")


def parse_script_file(script_name):
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
    command = input_dict.get("command", "") + " << foo"
    params = "\n".join([f"{k}" for k in input_dict["inputs"]])
    return f"{command}\n{params}\nfoo\n"


def generate_command_line(input_dict):
    command = input_dict.get("command", "")
    params = " ".join(
        [f"{k}={v}" if k.startswith("--") else f"{k} {v}" for k, v in input_dict.items() if k != "command"]
    )

    return f"{command} {params}\n"


def replace_command_in_script(script_file, input_dict):
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
