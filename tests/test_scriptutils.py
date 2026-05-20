"""Tests for cryocat.utils.scriptutils."""
import io
import json
import os

import pytest

from cryocat.utils.scriptutils import (
    generate_command_line,
    generate_interactive_command,
    generate_python_command,
    generate_script,
    parse_command,
    parse_command_for_gui,
    parse_script_file,
    process_cluster_params,
    replace_command_in_script,
)


# ---------------------------------------------------------------------------
# parse_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command_string, expected",
    [
        # basic -key value style
        (
            "command -arg1 value1 -arg2 value2",
            {"command": "command", "-arg1": "value1", "-arg2": "value2"},
        ),
        # --key=value style
        (
            "cmd --key=value -flag=123",
            {"command": "cmd", "--key": "value", "-flag": "123"},
        ),
        # mixed styles
        (
            "program --key=value -flag arg",
            {"command": "program", "--key": "value", "-flag": "arg"},
        ),
        # command only, no args
        ("ls", {"command": "ls"}),
        # empty string
        ("", {"command": ""}),
        # flag with no value (double-dash)
        (
            "prog --verbose",
            {"command": "prog", "--verbose": ""},
        ),
    ],
)
def test_parse_command(command_string, expected):
    assert parse_command(command_string) == expected


def test_parse_command_writes_dict(tmp_path):
    out_file = str(tmp_path / "out.json")
    parse_command("mycommand --key value", output_dict=out_file)
    assert os.path.isfile(out_file)


# ---------------------------------------------------------------------------
# parse_command_for_gui
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "command_string, expected",
    [
        # -key value style
        (
            "command -arg1 value1",
            {"Command": "command", "Arg1": {"value": "value1", "param": "-arg1", "tooltip": ""}},
        ),
        # --key=value style
        (
            "cmd --key=value",
            {"Command": "cmd", "Key": {"value": "value", "param": "--key", "tooltip": ""}},
        ),
        # flag without value
        (
            "program -verbose",
            {"Command": "program", "Verbose": ""},
        ),
        # empty string
        ("", {"Command": ""}),
    ],
)
def test_parse_command_for_gui(command_string, expected):
    assert parse_command_for_gui(command_string) == expected


def test_parse_command_for_gui_writes_json(tmp_path):
    out_file = str(tmp_path / "gui.json")
    parse_command_for_gui("prog --key value", output_dict=out_file)
    assert os.path.isfile(out_file)
    with open(out_file) as f:
        data = json.load(f)
    assert "Command" in data


# ---------------------------------------------------------------------------
# generate_command_line
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_dict, expected",
    [
        (
            {"command": "program", "--input": "file.txt", "-o": "output"},
            "program --input=file.txt -o output\n",
        ),
        (
            {"command": "ls"},
            "ls \n",
        ),
        (
            {"command": "run", "--alpha": "1", "--beta": "2"},
            "run --alpha=1 --beta=2\n",
        ),
    ],
)
def test_generate_command_line(input_dict, expected):
    assert generate_command_line(input_dict) == expected


# ---------------------------------------------------------------------------
# generate_python_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "function_dict, import_dict, expected",
    [
        (
            {"function": "my_function", "param1": "value1", "param2": 42},
            None,
            'python -c "my_function(param1=\'value1\', param2=42)"\n',
        ),
        (
            {"function": "test_func", "arg": "test"},
            {"module1": "func1", "module2": "Class2"},
            'python -c "from module1 import func1; from module2 import Class2; test_func(arg=\'test\')"\n',
        ),
        # no args at all
        (
            {"function": "bare"},
            None,
            'python -c "bare()"\n',
        ),
    ],
)
def test_generate_python_command(function_dict, import_dict, expected):
    assert generate_python_command(function_dict, import_dict) == expected


# ---------------------------------------------------------------------------
# generate_interactive_command
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_dict, expected",
    [
        (
            {"command": "interactive_tool", "inputs": ["input1", "input2", "input3"]},
            "interactive_tool << foo\ninput1\ninput2\ninput3\nfoo\n",
        ),
        (
            {"command": "imod", "inputs": ["1"]},
            "imod << foo\n1\nfoo\n",
        ),
        (
            {"command": "tool", "inputs": []},
            "tool << foo\n\nfoo\n",
        ),
    ],
)
def test_generate_interactive_command(input_dict, expected):
    assert generate_interactive_command(input_dict) == expected


# ---------------------------------------------------------------------------
# process_cluster_params
# ---------------------------------------------------------------------------

def test_process_cluster_params_none_is_noop():
    buf = io.StringIO()
    process_cluster_params(buf, None)
    assert buf.getvalue() == ""


def test_process_cluster_params_double_dash():
    buf = io.StringIO()
    process_cluster_params(buf, {"--job-name": "myjob", "--ntasks": "4"})
    content = buf.getvalue()
    assert "#SBATCH --job-name=myjob" in content
    assert "#SBATCH --ntasks=4" in content


def test_process_cluster_params_single_dash():
    buf = io.StringIO()
    process_cluster_params(buf, {"-n": "8"})
    assert "#SBATCH -n 8" in buf.getvalue()


# ---------------------------------------------------------------------------
# parse_script_file
# ---------------------------------------------------------------------------

_SCRIPT_CONTENT = """\
#!/bin/bash

#SBATCH --job-name=test
#SBATCH -n 4
module load python/3.9
module load cuda

input_file="data.txt"
output_dir="./results"

program --input=$input_file --output=$output_dir
another_command -flag value
"""


@pytest.fixture
def tmp_script(tmp_path):
    p = tmp_path / "test.sh"
    p.write_text(_SCRIPT_CONTENT)
    return str(p)


def test_parse_script_file_interpreter(tmp_script):
    result = parse_script_file(tmp_script)
    assert result["interpreter"] == "#!/bin/bash"


def test_parse_script_file_sbatch(tmp_script):
    result = parse_script_file(tmp_script)
    assert result["sbatch_params"] == {"--job-name": "test", "-n": "4"}


def test_parse_script_file_modules(tmp_script):
    result = parse_script_file(tmp_script)
    assert set(result["modules"]) == {"python/3.9", "cuda"}


def test_parse_script_file_values(tmp_script):
    result = parse_script_file(tmp_script)
    assert result["values"] == {"input_file": '"data.txt"', "output_dir": '"./results"'}


def test_parse_script_file_commands(tmp_script):
    # Lines with "=" are treated as value assignments; only plain commands
    # without "=" land in coms_and_params (another_command -flag value).
    result = parse_script_file(tmp_script)
    coms = result["coms_and_params"]
    assert len(coms) >= 1
    commands = [c["command"] for c in coms if isinstance(c, dict)]
    assert "another_command" in commands


# ---------------------------------------------------------------------------
# replace_command_in_script
# ---------------------------------------------------------------------------

_SCRIPT_WITH_CMD = """\
#!/bin/bash

module load python

old_command --input=old.txt --output=old_dir
another_command -flag value
"""


@pytest.fixture
def tmp_script_with_cmd(tmp_path):
    p = tmp_path / "replace.sh"
    p.write_text(_SCRIPT_WITH_CMD)
    return str(p)


def test_replace_command_returns_true(tmp_script_with_cmd):
    replacement = {"command": "old_command", "--input": "new.txt", "--output": "new_dir"}
    assert replace_command_in_script(tmp_script_with_cmd, replacement) is True


def test_replace_command_content_updated(tmp_script_with_cmd):
    replacement = {
        "command": "old_command",
        "--input": "new.txt",
        "--output": "new_dir",
        "--new-flag": "new_value",
    }
    replace_command_in_script(tmp_script_with_cmd, replacement)
    content = open(tmp_script_with_cmd).read()
    assert "old_command --input=new.txt --output=new_dir --new-flag=new_value" in content


def test_replace_command_only_first_occurrence(tmp_path):
    content = "#!/bin/bash\nprog --a=1\nprog --a=2\n"
    p = tmp_path / "dup.sh"
    p.write_text(content)
    replace_command_in_script(str(p), {"command": "prog", "--a": "99"})
    lines = p.read_text().splitlines()
    matching = [l for l in lines if "prog --a=99" in l]
    assert len(matching) == 1


def test_replace_command_other_commands_preserved(tmp_script_with_cmd):
    replacement = {"command": "old_command", "--input": "new.txt"}
    replace_command_in_script(tmp_script_with_cmd, replacement)
    content = open(tmp_script_with_cmd).read()
    assert "another_command -flag value" in content


def test_replace_command_not_found_returns_false(tmp_script_with_cmd):
    replacement = {"command": "nonexistent_cmd", "--x": "1"}
    assert replace_command_in_script(tmp_script_with_cmd, replacement) is False


def test_replace_command_missing_key_raises(tmp_script_with_cmd):
    with pytest.raises(ValueError, match="'command' key"):
        replace_command_in_script(tmp_script_with_cmd, {"--input": "x"})


# ---------------------------------------------------------------------------
# generate_script
# ---------------------------------------------------------------------------

def test_generate_script_header(tmp_path):
    out = str(tmp_path / "out.sh")
    commands = [{"command": "prog", "--input": "f.txt"}]
    generate_script(out, commands, script_header="#!/bin/bash")
    content = open(out).read()
    assert content.startswith("#!/bin/bash\n")


def test_generate_script_module_list(tmp_path):
    out = str(tmp_path / "out.sh")
    generate_script(out, [{"command": "prog"}], module_loads=["python", "cuda"])
    content = open(out).read()
    assert "module load python" in content
    assert "module load cuda" in content


def test_generate_script_module_dict(tmp_path):
    out = str(tmp_path / "out.sh")
    generate_script(out, [{"command": "prog"}], module_loads={"a": "python", "b": "cuda"})
    content = open(out).read()
    assert "module load python" in content


def test_generate_script_module_invalid_raises(tmp_path):
    out = str(tmp_path / "out.sh")
    with pytest.raises(ValueError):
        generate_script(out, [{"command": "prog"}], module_loads="bad_type")


def test_generate_script_commands_written(tmp_path):
    out = str(tmp_path / "out.sh")
    commands = [
        {"command": "program1", "--input": "file1.txt"},
        {"command": "program2", "-o": "output"},
    ]
    generate_script(out, commands)
    content = open(out).read()
    assert "program1 --input=file1.txt" in content
    assert "program2 -o output" in content
