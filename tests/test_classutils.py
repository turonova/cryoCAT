import types
import pytest
import sys
from unittest.mock import patch, MagicMock
import numpy as np

from cryocat.utils.classutils import (
    filter_strings,
    get_class_names_by_parent,
    get_class_names_by_prefix,
    get_classes_from_names,
    parse_allowed_types,
    parse_choices,
    parse_doc_param,
    parse_string_into_array,
    process_method_docstring,
    replace_cross_references,
    as_list,
)


# ---------------------------------------------------------------------------
# filter_strings
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_list, filter_contains, filter_exclude, expected",
    [
        # include filter only
        (["apple", "banana", "cherry", "date"], ["a"], None, ["apple", "banana", "date"]),
        # exclude filter only
        (["apple", "banana", "cherry", "date"], None, ["a"], ["cherry"]),
        # both filters combined
        (["apple", "banana", "cherry", "date", "apricot"], ["a"], ["an"], ["apple", "date", "apricot"]),
        # string (not list) filters
        (["apple", "banana", "cherry"], "a", "n", ["apple"]),
        # empty input list
        ([], None, None, []),
        # no filters → identity
        (["apple", "banana"], None, None, ["apple", "banana"]),
        # filter_contains with no match → empty
        (["apple", "banana"], ["xyz"], None, []),
        # exclude removes all
        (["apple", "apricot"], None, ["a"], []),
    ],
)
def test_filter_strings(input_list, filter_contains, filter_exclude, expected):
    assert filter_strings(input_list, filter_contains=filter_contains, filter_exclude=filter_exclude) == expected


# ---------------------------------------------------------------------------
# parse_allowed_types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("int, float, str", ["float", "int", "str"]),
        ("int and float or str", ["float", "int", "str"]),
        ("int, pandas, dataframe, float", ["float", "int"]),
        ("", []),
        ("int, Pandas, DataFrame, float", ["float", "int"]),
        ("array-like", ["array-like"]),
        ("str", ["str"]),
    ],
)
def test_parse_allowed_types(input_str, expected):
    assert parse_allowed_types(input_str) == expected


# ---------------------------------------------------------------------------
# parse_string_into_array
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected_values, expected_dtype_check",
    [
        ("1,2,3,4", [1, 2, 3, 4], np.integer),
        ("1.1,2.2,3.3", [1.1, 2.2, 3.3], np.floating),
        ("a,b,c", ["a", "b", "c"], np.str_),
        ("1,2.2,hello", ["1", "2.2", "hello"], np.str_),
        ("42", [42], np.integer),
    ],
)
def test_parse_string_into_array(input_str, expected_values, expected_dtype_check):
    result = parse_string_into_array(input_str)
    assert np.array_equal(result, np.array(expected_values))
    assert np.issubdtype(result.dtype, expected_dtype_check)


# ---------------------------------------------------------------------------
# parse_choices
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_str, expected",
    [
        ("{1, 2, 3}", [1, 2, 3]),
        ("{1.1, 2.2, 3.3}", [1.1, 2.2, 3.3]),
        ('{"a", "b", "c"}', ["a", "b", "c"]),
        ("{a, b, c}", ["a", "b", "c"]),
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
        ("Use :meth:`some_method` for processing", "Use `some_method` for processing"),
        ("This is a normal string", "This is a normal string"),
        (":meth:`first` and :meth:`second`", "`first` and `second`"),
        ("No markers here", "No markers here"),
    ],
)
def test_replace_cross_references(input_str, expected):
    assert replace_cross_references(input_str) == expected


# ---------------------------------------------------------------------------
# parse_doc_param
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "doc_param, add_prefix, expected",
    [
        # optional parameter
        (
            ("param1", "int, optional", ["Some description"]),
            "",
            ("param1", "Some description", False, ["int"], None, []),
        ),
        # parameter with default
        (
            ("param2", "float, default=1.0", ["Description here"]),
            "",
            ("param2", "Description here", False, ["float"], 1.0, []),
        ),
        # required parameter
        (
            ("param3", "str", ["Required parameter"]),
            "",
            ("param3", "Required parameter", True, ["str"], None, []),
        ),
        # parameter with choices (int)
        (
            ("param4", "int, {1, 2, 3}", ["Choice parameter"]),
            "",
            ("param4", "Choice parameter", False, int, 1, [1, 2, 3]),
        ),
        # with prefix
        (
            ("param5", "str", ["Prefixed"]),
            "test_",
            ("test_param5", "Prefixed", True, ["str"], None, []),
        ),
    ],
)
def test_parse_doc_param(doc_param, add_prefix, expected):
    result = parse_doc_param(doc_param, add_prefix=add_prefix)
    assert result == expected


# ---------------------------------------------------------------------------
# get_classes_from_names
# ---------------------------------------------------------------------------

class TestGetClassesFromNames:

    def test_single_string(self):
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.TestClass = type("TestClass", (), {})
            mock_import.return_value = mock_module

            result = get_classes_from_names("TestClass", "test_module")
            assert result == mock_module.TestClass

    def test_list_of_strings(self):
        with patch("importlib.import_module") as mock_import:
            mock_module = MagicMock()
            mock_module.Class1 = type("Class1", (), {})
            mock_module.Class2 = type("Class2", (), {})
            mock_import.return_value = mock_module

            result = get_classes_from_names(["Class1", "Class2"], "test_module")
            assert result == [mock_module.Class1, mock_module.Class2]

    def test_already_class_object_passthrough(self):
        class SomeClass:
            pass

        result = get_classes_from_names(SomeClass, "test_module")
        assert result is SomeClass

    def test_list_of_class_objects_passthrough(self):
        class A:
            pass

        class B:
            pass

        result = get_classes_from_names([A, B], "test_module")
        assert result == [A, B]


# ---------------------------------------------------------------------------
# get_class_names_by_parent
# ---------------------------------------------------------------------------

def _make_test_module(name="test_module"):
    """Helper: build a synthetic module with Base, Child1, Child2, Unrelated."""
    mod = types.ModuleType(name)

    class BaseClass:
        pass

    class ChildClass1(BaseClass):
        pass

    class ChildClass2(BaseClass):
        pass

    class UnrelatedClass:
        pass

    for cls in (BaseClass, ChildClass1, ChildClass2, UnrelatedClass):
        cls.__module__ = name
        setattr(mod, cls.__name__, cls)

    return mod


class TestGetClassNamesByParent:

    def test_find_all_subclasses(self):
        mod = _make_test_module()
        with patch.dict("sys.modules", {mod.__name__: mod}):
            with patch("importlib.import_module", return_value=mod):
                result = get_class_names_by_parent("BaseClass", mod.__name__)
                assert set(result) == {"ChildClass1", "ChildClass2"}

    def test_filter_contains(self):
        mod = _make_test_module()
        with patch.dict("sys.modules", {mod.__name__: mod}):
            with patch("importlib.import_module", return_value=mod):
                result = get_class_names_by_parent("BaseClass", mod.__name__, filter_contains="1")
                assert result == ["ChildClass1"]

    def test_filter_exclude(self):
        mod = _make_test_module()
        with patch.dict("sys.modules", {mod.__name__: mod}):
            with patch("importlib.import_module", return_value=mod):
                result = get_class_names_by_parent("BaseClass", mod.__name__, filter_exclude="2")
                assert result == ["ChildClass1"]

    def test_filter_contains_and_exclude(self):
        mod = _make_test_module()
        with patch.dict("sys.modules", {mod.__name__: mod}):
            with patch("importlib.import_module", return_value=mod):
                result = get_class_names_by_parent(
                    "BaseClass", mod.__name__, filter_contains="Child", filter_exclude="2"
                )
                assert result == ["ChildClass1"]

    def test_invalid_parent_class_raises(self):
        mod = _make_test_module()
        with patch.dict("sys.modules", {mod.__name__: mod}):
            with patch("importlib.import_module", return_value=mod):
                with pytest.raises(ValueError, match="'NonExistentClass' is not a valid class"):
                    get_class_names_by_parent("NonExistentClass", mod.__name__)


# ---------------------------------------------------------------------------
# get_class_names_by_prefix
# The function inspects its *own* module (classutils.py), which defines no
# classes, so the result is always an empty list.  Tests verify behaviour
# rather than specific class names.
# ---------------------------------------------------------------------------

class TestGetClassNamesByPrefix:

    def test_returns_list(self):
        result = get_class_names_by_prefix("Prefix")
        assert isinstance(result, list)

    def test_all_results_start_with_prefix(self):
        result = get_class_names_by_prefix("Prefix")
        for name in result:
            assert name.startswith("Prefix")

    def test_no_match_returns_empty(self):
        result = get_class_names_by_prefix("ZZZNonexistentPrefix999")
        assert result == []

    @pytest.mark.parametrize("prefix", ["A", "B", "My", "Test"])
    def test_arbitrary_prefix_returns_list(self, prefix):
        result = get_class_names_by_prefix(prefix)
        assert isinstance(result, list)
        for name in result:
            assert name.startswith(prefix)


# ---------------------------------------------------------------------------
# process_method_docstring
# ---------------------------------------------------------------------------

class _SampleClass:
    def documented(self, param1: int, param2: str = "default"):
        """Sample method for testing.

        Parameters
        ----------
        param1 : int
            First required parameter.
        param2 : str, default="default"
            Second optional parameter.
        """
        pass

    def undocumented(self):
        pass


class TestProcessMethodDocstring:

    def test_required_param_detected(self):
        result = process_method_docstring(_SampleClass, "documented")
        assert "param1" in result
        assert result["param1"]["required"] is True
        assert result["param1"]["types"] == ["int"]

    def test_optional_param_with_default(self):
        result = process_method_docstring(_SampleClass, "documented")
        assert "param2" in result
        assert result["param2"]["required"] is False
        assert result["param2"]["default"] == "default"
        assert result["param2"]["types"] == ["str"]

    def test_empty_docstring_returns_empty_dict(self):
        result = process_method_docstring(_SampleClass, "undocumented")
        assert result == {}

    def test_pretty_print_capitalizes_name(self):
        class _Helper:
            def method(self, my_param: int):
                """Desc.

                Parameters
                ----------
                my_param : int
                    A parameter.
                """
                pass

        result = process_method_docstring(_Helper, "method", pretty_print=True)
        # capitalize() + replace("_", " ") → "My param"
        assert "My param" in result

    def test_result_keys_match_param_names(self):
        result = process_method_docstring(_SampleClass, "documented")
        assert set(result.keys()) == {"param1", "param2"}

    def test_metadata_keys_present(self):
        result = process_method_docstring(_SampleClass, "documented")
        for key in ("desc", "required", "types", "default", "options", "name"):
            assert key in result["param1"]


# ---------------------------------------------------------------------------
# as_list
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "input_val, expected",
    [
        (5, [5]),
        (3.14, [3.14]),
        ("hello", ["hello"]),
        (b"bytes", [b"bytes"]),
        ([1, 2, 3], [1, 2, 3]),
        ((1, 2, 3), [1, 2, 3]),
        (range(3), [0, 1, 2]),
        (np.array([1, 2, 3]), [1, 2, 3]),
        ([], []),
    ],
)
def test_as_list(input_val, expected):
    result = as_list(input_val)
    assert result == expected


def test_as_list_returns_same_list_object():
    original = [1, 2, 3]
    assert as_list(original) is original
