import types
import pytest
import sys
from unittest.mock import patch, MagicMock
import numpy as np

import inspect
import typing

from cryocat.utils.classutils import (
    filter_strings,
    get_class_names_by_parent,
    get_class_names_by_prefix,
    get_classes_from_names,
    gui_exposed,
    process_method_docstring,
    resolve_param_type,
    TYPE_HANDLERS,
    as_list,
)
from cryocat._types import MapSource, TripletLike, Symmetry, MotlType


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
# gui_exposed
# ---------------------------------------------------------------------------

def test_gui_exposed_bare():
    @gui_exposed
    def my_op(self):
        pass

    assert my_op._gui is not None
    assert my_op._gui["label"] == "My op"
    assert my_op._gui["hide"] == {"self", "cls"}
    assert my_op._gui["category"] is None
    assert my_op._gui["output"] is None


def test_gui_exposed_with_metadata():
    @gui_exposed(label="Custom", category="Cleaning", hide=("x",), output="motl")
    def my_op(self, x, y):
        pass

    assert my_op._gui["label"] == "Custom"
    assert my_op._gui["category"] == "Cleaning"
    assert my_op._gui["hide"] == {"self", "cls", "x"}
    assert my_op._gui["output"] == "motl"


def test_gui_exposed_undecorated_has_no_gui():
    def plain():
        pass

    assert getattr(plain, "_gui", None) is None


def test_gui_exposed_defaults_standalone_preview():
    """Bare @gui_exposed must carry standalone=False and preview=None."""
    @gui_exposed
    def my_op(self):
        pass

    assert my_op._gui["standalone"] is False
    assert my_op._gui["preview"] is None


def test_gui_exposed_standalone_and_preview_fields():
    """Custom standalone/preview values land on fn._gui."""
    @gui_exposed(category="builder", standalone=True, preview="orientational")
    def my_builder(x):
        pass

    assert my_builder._gui["standalone"] is True
    assert my_builder._gui["preview"] == "orientational"
    assert my_builder._gui["category"] == "builder"


def test_gui_exposed_builder_registered():
    """A builder with standalone=True is added to _GUI_BUILDER_REGISTRY."""
    from cryocat.utils.classutils import _GUI_BUILDER_REGISTRY

    @gui_exposed(category="builder", standalone=True, preview="test")
    def _test_registry_builder(x):
        pass

    ids = [e["id"] for e in _GUI_BUILDER_REGISTRY]
    assert "_test_registry_builder" in ids
    entry = next(e for e in _GUI_BUILDER_REGISTRY if e["id"] == "_test_registry_builder")
    assert entry["preview"] == "test"
    assert entry["fn"] is _test_registry_builder


def test_gui_exposed_non_standalone_not_registered():
    """A builder without standalone=True must NOT appear in the registry."""
    from cryocat.utils.classutils import _GUI_BUILDER_REGISTRY

    @gui_exposed(category="builder", standalone=False)
    def _test_non_standalone(x):
        pass

    ids = [e["id"] for e in _GUI_BUILDER_REGISTRY]
    assert "_test_non_standalone" not in ids


# ---------------------------------------------------------------------------
# resolve_param_type
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "annotation, expected_tag",
    [
        (inspect.Parameter.empty, "str"),
        (None, "str"),
        (int, "int"),
        (float, "float"),
        (bool, "bool"),
        (str, "str"),
        (int | None, "int"),
        (MapSource, "MapSource"),
        (TripletLike, "TripletLike"),
        (Symmetry, "Symmetry"),
        (MapSource | None, "MapSource"),
        (typing.Literal["a", "b"], "Literal"),
    ],
)
def test_resolve_param_type_tag(annotation, expected_tag):
    tag, _ = resolve_param_type(annotation)
    assert tag == expected_tag


def test_resolve_param_type_literal_alias_choices():
    tag, extra = resolve_param_type(MotlType)
    assert tag == "Literal"
    assert "emmotl" in extra["choices"]


# ---------------------------------------------------------------------------
# TYPE_HANDLERS
# ---------------------------------------------------------------------------

def test_type_handlers_shape():
    for tag, entry in TYPE_HANDLERS.items():
        assert set(entry) == {"widget", "parse", "argparse"}
        assert isinstance(entry["widget"], str)
        assert callable(entry["parse"])
        assert "type" in entry["argparse"]


def test_type_handlers_parse_roundtrip():
    assert TYPE_HANDLERS["bool"]["parse"]("True") is True
    assert TYPE_HANDLERS["bool"]["parse"]("False") is False
    assert TYPE_HANDLERS["int"]["parse"]("5") == 5
    assert TYPE_HANDLERS["TripletLike"]["parse"]("1,2,3") == [1, 2, 3]
    assert TYPE_HANDLERS["TripletLike"]["parse"]("64") == 64
    assert TYPE_HANDLERS["MapSource"]["parse"]("") is None
    assert TYPE_HANDLERS["Literal"]["parse"]("b", ["a", "b"]) == "b"


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
    # process_method_docstring now supplies parameter *descriptions* only —
    # types/required/default come from the signature, not the docstring.

    def test_returns_descriptions(self):
        result = process_method_docstring(_SampleClass, "documented")
        assert result["param1"] == "First required parameter."
        assert result["param2"] == "Second optional parameter."

    def test_empty_docstring_returns_empty_dict(self):
        result = process_method_docstring(_SampleClass, "undocumented")
        assert result == {}

    def test_result_keys_match_param_names(self):
        result = process_method_docstring(_SampleClass, "documented")
        assert set(result.keys()) == {"param1", "param2"}

    def test_accepts_callable_directly(self):
        # method_name=None -> the first argument is the callable itself.
        result = process_method_docstring(_SampleClass.documented)
        assert set(result.keys()) == {"param1", "param2"}


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
