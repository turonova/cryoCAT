import pytest
import sys
from unittest.mock import patch, MagicMock
import numpy as np

from cryocat.classutils import *


class TestFilterStrings:\

    def test_basic_filtering(self):
        input_list = ["apple", "banana", "cherry", "date"]
        result = filter_strings(input_list, filter_contains=["a"])
        assert result == ["apple", "banana", "date"]

    def test_exclude_filtering(self):
        input_list = ["apple", "banana", "cherry", "date"]
        result = filter_strings(input_list, filter_exclude=["a"])
        assert result == ["cherry"]

    def test_combined_filters(self):
        input_list = ["apple", "banana", "cherry", "date", "apricot"]
        result = filter_strings(input_list, filter_contains=["a"], filter_exclude=["an"])
        assert result == ["apple", "date", "apricot"]

    def test_string_filters(self):
        input_list = ["apple", "banana", "cherry"]
        result = filter_strings(input_list, filter_contains="a", filter_exclude="n")
        assert result == ["apple"]

    def test_empty_input(self):
        assert filter_strings([]) == []

    def test_none_filters(self):
        input_list = ["apple", "banana"]
        result = filter_strings(input_list)
        assert result == input_list


class TestParseAllowedTypes:

    def test_basic_types(self):
        input_str = "int, float, str"
        result = parse_allowed_types(input_str)
        assert result == ["float", "int", "str"]

    def test_with_and_or(self):
        input_str = "int and float or str"
        result = parse_allowed_types(input_str)
        assert result == ["float", "int", "str"]

    def test_filter_unsupported_types(self):
        input_str = "int, pandas, dataframe, float"
        result = parse_allowed_types(input_str)
        assert result == ["float", "int"]

    def test_empty_string(self):
        assert parse_allowed_types("") == []

    def test_mixed_case_unsupported(self):
        input_str = "int, Pandas, DataFrame, float"
        result = parse_allowed_types(input_str)
        assert result == ["float", "int"]


class TestParseStringIntoArray:

    def test_integer_array(self):
        result = parse_string_into_array("1,2,3,4")
        assert np.array_equal(result, np.array([1, 2, 3, 4]))

    def test_float_array(self):
        result = parse_string_into_array("1.1,2.2,3.3")
        assert np.array_equal(result, np.array([1.1, 2.2, 3.3]))

    def test_string_array(self):
        result = parse_string_into_array("a,b,c")
        assert np.array_equal(result, np.array(["a", "b", "c"]))

    def test_mixed_types_fallback_to_string(self):
        result = parse_string_into_array("1,2.2,hello")
        assert np.array_equal(result, np.array(["1", "2.2", "hello"]))

    def test_single_value(self):
        result = parse_string_into_array("42")
        assert np.array_equal(result, np.array([42]))


class TestParseChoices:

    def test_integer_choices(self):
        result = parse_choices("{1, 2, 3}")
        assert result == [1, 2, 3]

    def test_float_choices(self):
        result = parse_choices("{1.1, 2.2, 3.3}")
        assert result == [1.1, 2.2, 3.3]

    def test_string_choices(self):
        result = parse_choices('{"a", "b", "c"}')
        assert result == ["a", "b", "c"]

    def test_string_choices_no_quotes(self):
        result = parse_choices("{a, b, c}")
        assert result == ["a", "b", "c"]


class TestReplaceCrossReferences:

    def test_replace_meth(self):
        input_str = "Use :meth:`some_method` for processing"
        result = replace_cross_references(input_str)
        assert result == "Use `some_method` for processing"

    def test_no_replacement(self):
        input_str = "This is a normal string"
        result = replace_cross_references(input_str)
        assert result == input_str

    def test_multiple_replacements(self):
        input_str = ":meth:`first` and :meth:`second`"
        result = replace_cross_references(input_str)
        assert result == "`first` and `second`"


class TestParseDocParam:

    def test_optional_parameter(self):
        doc_param = ("param1", "int, optional", ["Some description"])
        result = parse_doc_param(doc_param)
        expected = ("param1", "Some description", False, ["int"], None, [])
        assert result == expected

    def test_parameter_with_default(self):
        doc_param = ("param2", "float, default=1.0", ["Description here"])
        result = parse_doc_param(doc_param)
        expected = ("param2", "Description here", False, ["float"], 1.0, [])
        assert result == expected

    def test_required_parameter(self):
        doc_param = ("param3", "str", ["Required parameter"])
        result = parse_doc_param(doc_param)
        expected = ("param3", "Required parameter", True, ["str"], None, [])
        assert result == expected

    def test_parameter_with_choices(self):
        doc_param = ("param4", "int, {1, 2, 3}", ["Choice parameter"])
        result = parse_doc_param(doc_param)
        expected = ("param4", "Choice parameter", False, int, 1, [1, 2, 3])
        assert result == expected

    def test_with_prefix(self):
        doc_param = ("param5", "str", ["Prefixed"])
        result = parse_doc_param(doc_param, add_prefix="test_")
        expected = ("test_param5", "Prefixed", True, ["str"], None, [])
        assert result == expected


class TestGetClassesFromNames:

    def test_single_string(self):
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.TestClass = type('TestClass', (), {})
            mock_import.return_value = mock_module

            result = get_classes_from_names("TestClass", "test_module")
            assert result == mock_module.TestClass

    def test_list_of_strings(self):
        with patch('importlib.import_module') as mock_import:
            mock_module = MagicMock()
            mock_module.Class1 = type('Class1', (), {})
            mock_module.Class2 = type('Class2', (), {})
            mock_import.return_value = mock_module

            result = get_classes_from_names(["Class1", "Class2"], "test_module")
            assert result == [mock_module.Class1, mock_module.Class2]

    def test_already_classes(self):
        class TestClass:
            pass

        result = get_classes_from_names(TestClass, "test_module")
        assert result == TestClass


class BaseClass:
    pass


class ChildClass1(BaseClass):
    pass


class ChildClass2(BaseClass):
    pass


class UnrelatedClass:
    pass


class TestGetClassNamesByParent:

    def test_find_subclasses(self):
        test_module = types.ModuleType('test_module')

        class BaseClass:
            pass

        class ChildClass1(BaseClass):
            pass

        class ChildClass2(BaseClass):
            pass

        class UnrelatedClass:
            pass

        BaseClass.__module__ = 'test_module'
        ChildClass1.__module__ = 'test_module'
        ChildClass2.__module__ = 'test_module'
        UnrelatedClass.__module__ = 'test_module'

        test_module.BaseClass = BaseClass
        test_module.ChildClass1 = ChildClass1
        test_module.ChildClass2 = ChildClass2
        test_module.UnrelatedClass = UnrelatedClass

        with patch.dict('sys.modules', {'test_module': test_module}):
            with patch('importlib.import_module') as mock_import:
                mock_import.return_value = test_module

                result = get_class_names_by_parent("BaseClass", "test_module")
                assert set(result) == {"ChildClass1", "ChildClass2"}

    def test_with_filters(self):
        test_module = types.ModuleType('test_module')

        class BaseClass:
            pass

        class ChildClass1(BaseClass):
            pass

        class ChildClass2(BaseClass):
            pass

        # Set module attributes
        BaseClass.__module__ = 'test_module'
        ChildClass1.__module__ = 'test_module'
        ChildClass2.__module__ = 'test_module'

        test_module.BaseClass = BaseClass
        test_module.ChildClass1 = ChildClass1
        test_module.ChildClass2 = ChildClass2

        with patch.dict('sys.modules', {'test_module': test_module}):
            with patch('importlib.import_module') as mock_import:
                mock_import.return_value = test_module

                result = get_class_names_by_parent(
                    "BaseClass", "test_module",
                    filter_contains="1",
                    filter_exclude="2"
                )
                assert result == ["ChildClass1"]

    def test_invalid_parent_class(self):
        test_module = types.ModuleType('test_module')

        class SomeOtherClass:
            pass

        SomeOtherClass.__module__ = 'test_module'
        test_module.SomeOtherClass = SomeOtherClass

        with patch.dict('sys.modules', {'test_module': test_module}):
            with patch('importlib.import_module') as mock_import:
                mock_import.return_value = test_module

                with pytest.raises(ValueError, match="'NonExistentClass' is not a valid class"):
                    get_class_names_by_parent("NonExistentClass", "test_module")



class TestClass1:
    pass

class TestClass2:
    pass

class OtherClass:
    pass

class TestGetClassNamesByPrefix:

    def test_find_by_prefix(self):
        result = get_class_names_by_prefix("Test")
        assert isinstance(result, list)
        for class_name in result:
            assert class_name.startswith("Test")

    def test_find_by_prefix_simple(self):
        for prefix in ["Test", "TestGet", "TestGetClass"]:
            result = get_class_names_by_prefix(prefix)
            print(f"Prefix '{prefix}': {result}")
            assert isinstance(result, list)

            if result:
                for name in result:
                    assert name.startswith(prefix)

class TestProcessMethodDocstring:

    def test_process_method_doc(self):
        class TestClass:
            def test_method(self, param1: int, param2: str = "default"):
                """
                Test method description.

                Parameters
                ----------
                param1 : int
                    First parameter description.
                param2 : str, default="default"
                    Second parameter description.
                """
                pass

        with patch('inspect.getattr_static') as mock_getattr:
            mock_getattr.return_value = TestClass.test_method

            result = process_method_docstring(TestClass, "test_method")

            assert "param1" in result
            assert result["param1"]["types"] == ["int"]
            assert result["param1"]["required"] == True

            assert "param2" in result
            assert result["param2"]["types"] == ["str"]
            assert result["param2"]["default"] == "default"
            assert result["param2"]["required"] == False

    def test_pretty_print(self):
        class TestClass:
            def method(self, test_param: int):
                pass

        with patch('inspect.getattr_static') as mock_getattr:
            mock_getattr.return_value = TestClass.method

            result = process_method_docstring(TestClass, "method", pretty_print=True)

            assert isinstance(result, dict)
            assert result == {}