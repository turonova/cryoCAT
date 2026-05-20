"""Tests for cryocat.cli."""
import argparse
import pytest
import numpy as np
from unittest.mock import patch

from cryocat.cli import (
    parse_allowed_types,
    parse_choices,
    parse_doc_param,
    parse_input_types,
    parse_string_into_array,
    replace_cross_references,
    add_params_from_docstring,
)


# ---------------------------------------------------------------------------
# parse_allowed_types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("str or int or float", ["float", "int", "str"]),
        ("str, pandas.DataFrame, array-like, or int", ["array-like", "int", "str"]),
        ("pandas.DataFrame or pandas", []),
        ("int", ["int"]),
        ("str or float", ["float", "str"]),
        ("array-like, int", ["array-like", "int"]),
        ("pandas.DataFrame, str", ["str"]),
        ("", []),
        ("int, Pandas, DataFrame, float", ["float", "int"]),
        ("int or bool", ["bool", "int"]),
    ],
)
def test_parse_allowed_types(input_str, expected):
    assert parse_allowed_types(input_str) == expected


# ---------------------------------------------------------------------------
# parse_string_into_array
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("1,2,3,4", np.array([1, 2, 3, 4])),
        ("1.1,2.2,3.3", np.array([1.1, 2.2, 3.3])),
        ("a,b,c", np.array(["a", "b", "c"])),
        ("42", np.array([42])),
    ],
)
def test_parse_string_into_array(input_str, expected):
    result = parse_string_into_array(input_str)
    np.testing.assert_array_equal(result, expected)


def test_parse_string_into_array_mixed_falls_back_to_str():
    result = parse_string_into_array("1,two,3.0")
    assert result.dtype.kind == "U"


def test_parse_string_into_array_returns_ndarray():
    result = parse_string_into_array("42")
    assert isinstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# parse_choices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("{1, 2, 3}", [1, 2, 3]),
        ("{0.1, 0.2, 0.3}", [0.1, 0.2, 0.3]),
        ('{"gctf", "ctffind4", "warp"}', ["gctf", "ctffind4", "warp"]),
        ("{x, y}", ["x", "y"]),
    ],
)
def test_parse_choices(input_str, expected):
    assert parse_choices(input_str) == expected


# ---------------------------------------------------------------------------
# replace_cross_references
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("See :meth:`cryocat.ioutils.tlt_load` for more info", "See `cryocat.ioutils.tlt_load` for more info"),
        ("Plain text unchanged", "Plain text unchanged"),
        (":meth:`a` and :meth:`b`", "`a` and `b`"),
        ("", ""),
    ],
)
def test_replace_cross_references(input_str, expected):
    assert replace_cross_references(input_str) == expected


# ---------------------------------------------------------------------------
# parse_doc_param  (cli version prepends "--" to the param name)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "doc_param, expected_name, expected_required, expected_default, expected_choices",
    [
        # required parameter
        (
            ("tomo_id", "int", ["The ID of the tomogram."]),
            "--tomo_id", True, None, [],
        ),
        # optional parameter
        (
            ("mode", "str, optional", ["Mode of operation."]),
            "--mode", False, None, [],
        ),
        # parameter with default value
        (
            ("voltage", "float, default=300.0", ["The voltage of the microscope."]),
            "--voltage", False, 300.0, [],
        ),
        # parameter with string choices
        (
            ("ctf_file_type", '{"gctf", "ctffind4", "warp"}', ["Type of CTF file."]),
            "--ctf_file_type", False, "gctf", ["gctf", "ctffind4", "warp"],
        ),
        # parameter with integer choices
        (
            ("level", "int, {1, 2, 3}", ["Level selection."]),
            "--level", False, 1, [1, 2, 3],
        ),
    ],
)
def test_parse_doc_param(doc_param, expected_name, expected_required, expected_default, expected_choices):
    name, help_desc, required, _types, default, choices = parse_doc_param(doc_param)
    assert name == expected_name
    assert required == expected_required
    assert default == expected_default
    assert choices == expected_choices
    assert isinstance(help_desc, str)
    assert len(help_desc) > 0


def test_parse_doc_param_help_contains_description():
    doc_param = ("param", "str", ["Detailed description text."])
    _, help_desc, *_ = parse_doc_param(doc_param)
    assert "Detailed description text." in help_desc


def test_parse_doc_param_types_for_required():
    doc_param = ("tomo_id", "int", ["ID."])
    _, _, _, types_, _, _ = parse_doc_param(doc_param)
    assert "int" in types_


# ---------------------------------------------------------------------------
# parse_input_types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_value, allowed_types, expected",
    [
        ("5", ["int"], 5),
        ("3.14", ["float"], 3.14),
        ("hello", ["str"], "hello"),
    ],
)
def test_parse_input_types_scalars(input_value, allowed_types, expected):
    result = parse_input_types(input_value, allowed_types)
    assert result == expected


@pytest.mark.parametrize(
    "input_value, expected",
    [
        ("1,2,3", np.array([1, 2, 3])),
        ("1.1,2.2", np.array([1.1, 2.2])),
        ("a,b,c", np.array(["a", "b", "c"])),
    ],
)
def test_parse_input_types_array_like(input_value, expected):
    result = parse_input_types(input_value, ["array-like"])
    np.testing.assert_array_equal(result, expected)


def test_parse_input_types_single_array_like_value():
    result = parse_input_types("7", ["array-like"])
    assert result == 7


def test_parse_input_types_invalid_raises():
    with pytest.raises(argparse.ArgumentTypeError):
        parse_input_types("not_a_number", ["int", "float"])


# ---------------------------------------------------------------------------
# add_params_from_docstring
# ---------------------------------------------------------------------------

def _sample_function(output_path: str, scale: float = 1.0):
    """Sample function for CLI testing.

    Parameters
    ----------
    output_path : str
        Path to the output file.
    scale : float, default=1.0
        Scale factor applied to the data.
    """
    pass


def test_add_params_from_docstring_creates_subcommand():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    add_params_from_docstring(subparsers, "sample", _sample_function)
    args = parser.parse_args(["sample", "--output_path", "out.mrc"])
    assert args.output_path == "out.mrc"


def test_add_params_from_docstring_optional_has_default():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    add_params_from_docstring(subparsers, "sample", _sample_function)
    args = parser.parse_args(["sample", "--output_path", "out.mrc"])
    assert args.scale == pytest.approx(1.0)


def test_add_params_from_docstring_optional_overridable():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd")
    add_params_from_docstring(subparsers, "sample", _sample_function)
    args = parser.parse_args(["sample", "--output_path", "out.mrc", "--scale", "2.5"])
    assert args.scale == pytest.approx(2.5)


# ---------------------------------------------------------------------------
# CLI entry points (smoke tests)
# ---------------------------------------------------------------------------

def test_wedge_list_exits_without_args():
    with pytest.raises(SystemExit):
        from cryocat.cli import wedge_list
        wedge_list()


def test_tm_ana_exits_without_args():
    with pytest.raises(SystemExit):
        from cryocat.cli import tm_ana
        tm_ana()