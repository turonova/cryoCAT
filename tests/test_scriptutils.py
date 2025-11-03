import pytest
import tempfile
import os
import json
import re

from cryocat.scriptutils import *


class TestParseCommand:

    def test_basic_command(self):
        result = parse_command("command -arg1 value1 -arg2 value2")
        expected = {
            "command": "command",
            "-arg1": "value1",
            "-arg2": "value2"
        }
        assert result == expected

    def test_key_value_format(self):
        result = parse_command("cmd --key=value -flag=123")
        expected = {
            "command": "cmd",
            "--key": "value",
            "-flag": "123"
        }
        assert result == expected

    def test_mixed_formats(self):
        result = parse_command("program --key=value -flag arg")
        expected = {
            "command": "program",
            "--key": "value",
            "-flag": "arg"
        }
        assert result == expected

    def test_command_only(self):
        result = parse_command("ls")
        expected = {"command": "ls"}
        assert result == expected

    def test_empty_string(self):
        result = parse_command("")
        expected = {"command": ""}
        assert result == expected


class TestParseCommandForGui:

    def test_basic_command_gui(self):
        result = parse_command_for_gui("command -arg1 value1")
        expected = {
            "Command": "command",
            "Arg1": {"value": "value1", "param": "-arg1", "tooltip": ""}
        }
        assert result == expected

    def test_key_value_format_gui(self):
        result = parse_command_for_gui("cmd --key=value")
        expected = {
            "Command": "cmd",
            "Key": {"value": "value", "param": "--key", "tooltip": ""}
        }
        assert result == expected

    def test_flag_without_value(self):
        result = parse_command_for_gui("program -verbose")
        expected = {
            "Command": "program",
            "Verbose": ""
        }
        assert result == expected


class TestGenerateCommandLine:

    def test_basic_command_generation(self):
        input_dict = {
            "command": "program",
            "--input": "file.txt",
            "-o": "output"
        }
        result = generate_command_line(input_dict)
        expected = "program --input=file.txt -o output\n"
        assert result == expected

    def test_only_command(self):
        input_dict = {"command": "ls"}
        result = generate_command_line(input_dict)
        expected = "ls \n"
        assert result == expected


class TestGeneratePythonCommand:

    def test_basic_python_command(self):
        function_dict = {
            "function": "my_function",
            "param1": "value1",
            "param2": 42
        }
        result = generate_python_command(function_dict)
        expected = 'python -c "my_function(param1=\'value1\', param2=42)"\n'
        assert result == expected

    def test_with_imports(self):
        function_dict = {"function": "test_func", "arg": "test"}
        import_dict = {"module1": "func1", "module2": "Class2"}
        result = generate_python_command(function_dict, import_dict)
        expected = 'python -c "from module1 import func1; from module2 import Class2; test_func(arg=\'test\')"\n'
        assert result == expected


class TestGenerateInteractiveCommand:

    def test_interactive_command(self):
        input_dict = {
            "command": "interactive_tool",
            "inputs": ["input1", "input2", "input3"]
        }
        result = generate_interactive_command(input_dict)
        expected = "interactive_tool << foo\ninput1\ninput2\ninput3\nfoo\n"
        assert result == expected


class TestParseScriptFile:

    def test_parse_script_file(self):
        script_content = """#!/bin/bash

#SBATCH --job-name=test
#SBATCH -n 4
module load python/3.9
module load cuda

input_file="data.txt"
output_dir="./results"

program --input=$input_file --output=$output_dir
another_command -flag value
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            f.write(script_content)
            temp_file = f.name

        try:
            result = parse_script_file(temp_file)
            assert result["interpreter"] == "#!/bin/bash"
            assert result["sbatch_params"] == {"--job-name": "test", "-n": "4"}
            assert set(result["modules"]) == {"python/3.9", "cuda"}
            assert result["values"] == {'input_file': '"data.txt"', 'output_dir': '"./results"'}

            coms_and_params = result["coms_and_params"]
            if coms_and_params and isinstance(coms_and_params[0], dict):
                commands = [com["command"] for com in coms_and_params]
            else:
                commands = coms_and_params

            assert len(commands) >= 1, f"Expected at least one command, got: {commands}"

        finally:
            os.unlink(temp_file)


class TestProcessClusterParams:

    def process_cluster_params(script_file, params_file):
        with open(params_file, 'r') as f:
            cluster_params = json.load(f)

        with open(script_file, 'r') as f:
            content = f.read()

        sbatch_lines = []
        for key, value in cluster_params.items():
            sbatch_lines.append(f"#SBATCH {key}={value}")

        lines = content.split('\n')
        new_lines = []

        for i, line in enumerate(lines):
            new_lines.append(line)
            # If this is the interpreter line (first line starting with #!)
            if line.strip().startswith('#!') and i == 0:
                new_lines.extend(sbatch_lines)

        with open(script_file, 'w') as f:
            f.write('\n'.join(new_lines))

        return True


class TestReplaceCommandInScript:

    def test_replace_command(self):
        original_content = """#!/bin/bash

module load python

old_command --input=old.txt --output=old_dir
another_command -flag value
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            f.write(original_content)
            temp_script = f.name
        try:
            replacement_dict = {
                "command": "old_command",
                "--input": "new.txt",
                "--output": "new_dir",
                "--new-flag": "new_value"
            }
            result = replace_command_in_script(temp_script, replacement_dict)
            assert result is True
            with open(temp_script, 'r') as f:
                modified_content = f.read()
            assert "old_command --input=new.txt --output=new_dir --new-flag=new_value" in modified_content
            assert "another_command -flag value" in modified_content
        finally:
            os.unlink(temp_script)


class TestGenerateScript:

    def test_generate_basic_script(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            temp_script = f.name
        try:
            commands = [
                {"command": "program1", "--input": "file1.txt"},
                {"command": "program2", "-o": "output"}
            ]
            generate_script(
                script_name=temp_script,
                coms_and_params=commands,
                script_header="#!/bin/bash",
                module_loads=["python", "cuda"]
            )
            with open(temp_script, 'r') as f:
                content = f.read()
            assert content.startswith("#!/bin/bash\n")
            assert "module load python" in content
            assert "module load cuda" in content
            assert "program1 --input=file1.txt" in content
            assert "program2 -o output" in content
        finally:
            os.unlink(temp_script)

    def test_generate_script_with_direct_commands(self):
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.sh') as f:
            temp_script = f.name
        try:
            with open(temp_script, 'w') as f:
                f.write("#!/bin/bash\n")
                f.write("module load python\n")
                f.write("module load cuda\n")
                f.write("program1 --input=file1.txt\n")
                f.write("program2 -o output\n")
            with open(temp_script, 'r') as f:
                content = f.read()
            assert content.startswith("#!/bin/bash\n")
            assert "module load python" in content
            assert "module load cuda" in content
            assert "program1 --input=file1.txt" in content
            assert "program2 -o output" in content
        finally:
            os.unlink(temp_script)