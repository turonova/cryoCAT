"""Tests for customel component builders.

Covers:
- extract_style: merges style override, no-override, immutability.
- InlineLabeledDropdown: className, style kwarg, label className.
- InlineInputForm: className.
"""
from __future__ import annotations

import pytest

from cryocat.app.components.customel import (
    InlineLabeledDropdown,
    InlineInputForm,
    extract_style,
)


# ===========================================================================
# extract_style
# ===========================================================================

def test_extract_style_merges_override():
    default = {"flex": "1", "padding": "0"}
    kwargs = {"style": {"padding": "4px"}, "multi": True}
    style, cleaned = extract_style(default, kwargs)
    assert style["padding"] == "4px"
    assert style["flex"] == "1"
    assert "style" not in cleaned
    assert cleaned["multi"] is True


def test_extract_style_no_override():
    default = {"flex": "1"}
    kwargs = {"multi": True}
    style, cleaned = extract_style(default, kwargs)
    assert style == {"flex": "1"}
    assert cleaned == {"multi": True}


def test_extract_style_does_not_mutate_inputs():
    default = {"a": "1"}
    kwargs = {"style": {"a": "2"}}
    extract_style(default, kwargs)
    assert default == {"a": "1"}
    assert kwargs == {"style": {"a": "2"}}


# ===========================================================================
# InlineLabeledDropdown
# ===========================================================================

def test_inline_dropdown_className_not_in_dropdown_kwargs():
    """className must be popped, not left in **dropdown_kwargs passed to Dropdown."""
    div = InlineLabeledDropdown("test-dd", "Label", className="my-class")
    assert div.className == "my-class"
    dropdown = div.children[1]
    assert not getattr(dropdown, "className", None)


def test_inline_dropdown_default_className():
    div = InlineLabeledDropdown("test-dd", "Label")
    assert div.className == "flex"


def test_inline_dropdown_style_kwarg_does_not_raise():
    """style= must not raise TypeError 'multiple values for keyword argument'."""
    div = InlineLabeledDropdown("test-dd2", "Label", style={"color": "red"})
    dropdown = div.children[1]
    assert dropdown.style.get("color") == "red"


def test_inline_dropdown_style_merges_with_defaults():
    div = InlineLabeledDropdown("test-dd3", "Label", style={"flex": "2"})
    dropdown = div.children[1]
    assert dropdown.style["flex"] == "2"


def test_inline_dropdown_label_className_is_plain_string():
    """The label className is a plain string constant (no runtime f-string overhead)."""
    div = InlineLabeledDropdown("test-dd4", "Label")
    label = div.children[0]
    assert label.className == "label-dark mb-0 me-2"


# ===========================================================================
# InlineInputForm
# ===========================================================================

def test_inline_input_className_not_in_input_kwargs():
    """className must be popped from input_kwargs so it doesn't reach dbc.Input."""
    div = InlineInputForm("test-inp", "Label", className="inp-class")
    assert div.className == "inp-class"
    input_el = div.children[1]
    assert not getattr(input_el, "className", None)


def test_inline_input_default_className():
    div = InlineInputForm("test-inp2", "Label")
    assert div.className == "flex"


