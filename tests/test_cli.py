import pytest
from cryocat.cli import *


class TestCLIParsingFunctions:
    def test_parse_allowed_types_basic(self):
        input_str = "str or int or float"
        result = parse_allowed_types(input_str)
        expected = ["float", "int", "str"]
        assert result == expected

    def test_parse_allowed_types_with_unsupported(self):
        input_str = "str, pandas.DataFrame, array-like, or int"
        result = parse_allowed_types(input_str)
        expected = ["array-like", "int", "str"]
        assert result == expected

    def test_parse_allowed_types_empty(self):
        input_str = "pandas.DataFrame or pandas"
        result = parse_allowed_types(input_str)
        assert result == []

    def test_parse_string_into_array_int(self):
        result = parse_string_into_array("1,2,3,4")
        expected = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(result, expected)

    def test_parse_string_into_array_float(self):
        result = parse_string_into_array("1.1,2.2,3.3")
        expected = np.array([1.1, 2.2, 3.3])
        np.testing.assert_array_equal(result, expected)

    def test_parse_string_into_array_str(self):
        result = parse_string_into_array("a,b,c")
        expected = np.array(['a', 'b', 'c'])
        np.testing.assert_array_equal(result, expected)

    def test_parse_choices_int(self):
        result = parse_choices("{1, 2, 3}")
        expected = [1, 2, 3]
        assert result == expected

    def test_parse_choices_float(self):
        result = parse_choices("{0.1, 0.2, 0.3}")
        expected = [0.1, 0.2, 0.3]
        assert result == expected

    def test_parse_choices_str(self):
        result = parse_choices('{"gctf", "ctffind4", "warp"}')
        expected = ["gctf", "ctffind4", "warp"]
        assert result == expected

    def test_replace_cross_references(self):
        input_str = "See :meth:`cryocat.ioutils.tlt_load` for more info"
        result = replace_cross_references(input_str)
        expected = "See `cryocat.ioutils.tlt_load` for more info"
        assert result == expected

    def test_parse_input_types_single_int(self):
        result = parse_input_types("5", ["int"])
        assert result == 5
        assert isinstance(result, int)

    def test_parse_input_types_single_float(self):
        result = parse_input_types("3.14", ["float"])
        assert result == 3.14
        assert isinstance(result, float)

    def test_parse_input_types_single_str(self):
        result = parse_input_types("hello", ["str"])
        assert result == "hello"
        assert isinstance(result, str)

    def test_parse_input_types_array_int(self):
        result = parse_input_types("1,2,3", ["array-like"])
        expected = np.array([1, 2, 3])
        np.testing.assert_array_equal(result, expected)

    def test_parse_input_types_array_float(self):
        result = parse_input_types("1.1,2.2", ["array-like"])
        expected = np.array([1.1, 2.2])
        np.testing.assert_array_equal(result, expected)


class TestDocParamParsing:
    def test_parse_doc_param_required(self):
        doc_param = [
            "tomo_id",
            "int",
            ["The ID of the tomogram."]
        ]
        name, help_desc, required, types, default, choices = parse_doc_param(doc_param)
        assert name == "--tomo_id"
        assert "ID of the tomogram" in help_desc
        assert required is True
        assert "int" in types

    def test_parse_doc_param_with_default(self):
        doc_param = [
            "voltage",
            "float, default=300.0",
            ["The voltage of the microscope."]
        ]
        name, help_desc, required, types, default, choices = parse_doc_param(doc_param)
        assert name == "--voltage"
        assert required is False
        assert default == 300.0
        assert "float" in types

    def test_parse_doc_param_with_choices(self):
        doc_param = [
            "ctf_file_type",
            '{"gctf", "ctffind4", "warp"}',
            ["The type of CTF file."]
        ]
        name, help_desc, required, types, default, choices = parse_doc_param(doc_param)
        assert name == "--ctf_file_type"
        assert required is False
        assert choices == ["gctf", "ctffind4", "warp"]
        assert default == "gctf"


class TestCLICommands:
    def test_wedge_list_help(self):
        try:
            wedge_list()
        except SystemExit:
            pass

    def test_tm_ana_help(self):
        try:
            tm_ana()
        except SystemExit:
            pass


class TestEdgeCases:
    def test_parse_input_types_invalid(self):
        with pytest.raises(argparse.ArgumentTypeError):
            parse_input_types("not_a_number", ["int", "float"])

    def test_parse_choices_empty(self):
        result = parse_choices("{}")
        assert result == [] or result == ['']

    def test_parse_string_into_array_single(self):
        result = parse_string_into_array("42")
        expected = np.array([42])
        np.testing.assert_array_equal(result, expected)
        assert isinstance(result, np.ndarray)


@pytest.mark.parametrize("input_str,expected", [
    ("int", ["int"]),
    ("str or float", ["float", "str"]),
    ("array-like, int", ["array-like", "int"]),
    ("pandas.DataFrame, str", ["str"]),
])
def test_parse_allowed_types_parametrized(input_str, expected):
    result = parse_allowed_types(input_str)
    assert result == expected


@pytest.mark.parametrize("input_str,expected", [
    ("1,2,3", np.array([1, 2, 3])),
    ("1.5,2.5", np.array([1.5, 2.5])),
    ("a,b,c", np.array(['a', 'b', 'c'])),
])
def test_parse_string_into_array_parametrized(input_str, expected):
    result = parse_string_into_array(input_str)
    np.testing.assert_array_equal(result, expected)