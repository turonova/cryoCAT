"""Tests for motl I/O components and write_out.

Covers:
- get_motl_input component layout contract.
- _motlio_ops pure functions: motl_type_from_path, relion_version_to_type,
  validate_save, save_kwargs_from_store, load_kwargs_from_store.
- motlsource.picker_options pure function.
- relionopts.read_relion_kwargs pure function.
- Motl.write_out round-trips and error cases; annotation pinning.
"""
from __future__ import annotations

import inspect
import sys
import typing
import types

import numpy as np
import pandas as pd
import pytest

from dash import html, dcc
import dash_bootstrap_components as dbc

from cryocat.app.components.motlinput import get_motl_input
from cryocat.app.components._motlio_ops import (
    motl_type_from_path,
    relion_version_to_type,
    validate_save,
    save_kwargs_from_store,
    load_kwargs_from_store,
)
from cryocat.app.components.motlsource import picker_options
from cryocat.app.components.relionopts import read_relion_kwargs
from cryocat.core.cryomotl import Motl, StopgapMotl, RelionMotl
from cryocat.utils.exceptions import UserInputError


# ===========================================================================
# get_motl_input — layout helpers
# ===========================================================================

def _flatten_ids(component) -> set[str]:
    ids: set[str] = set()
    if hasattr(component, "id") and isinstance(component.id, str):
        ids.add(component.id)
    for child in _iter_children(component):
        ids.update(_flatten_ids(child))
    return ids


def _iter_children(component):
    children = getattr(component, "children", None)
    if children is None:
        return
    if isinstance(children, list):
        yield from children
    else:
        yield children


def _find_by_id(component, target_id: str):
    if hasattr(component, "id") and component.id == target_id:
        return component
    for child in _iter_children(component):
        result = _find_by_id(child, target_id)
        if result is not None:
            return result
    return None


# ===========================================================================
# get_motl_input layout contract
# ===========================================================================

class TestGetMotlInputLayout:
    def test_returns_div(self):
        c = get_motl_input("test")
        assert isinstance(c, html.Div)

    def test_value_store_present(self):
        c = get_motl_input("test")
        store = _find_by_id(c, "test-value")
        assert store is not None
        assert isinstance(store, dcc.Store)

    def test_source_mode_radio_present(self):
        c = get_motl_input("test")
        radio = _find_by_id(c, "test-source-mode")
        assert radio is not None
        opts = radio.options
        values = [o["value"] for o in opts]
        assert "pool" in values
        assert "group" in values

    def test_pool_section_present(self):
        c = get_motl_input("test")
        assert _find_by_id(c, "test-pool-section") is not None
        assert _find_by_id(c, "test-pool-select") is not None

    def test_group_section_hidden_by_default(self):
        c = get_motl_input("test")
        group_section = _find_by_id(c, "test-group-section")
        assert group_section is not None
        style = group_section.style or {}
        assert style.get("display") == "none"

    def test_group_header_row_above_checklist(self):
        group_section = _find_by_id(get_motl_input("test"), "test-group-section")
        children = group_section.children
        ids_in_order = [
            getattr(ch, "id", None)
            for ch in (children if isinstance(children, list) else [children])
        ]
        header_idx = next(
            (i for i, cid in enumerate(ids_in_order) if cid == "test-group-header-row"), None
        )
        checklist_idx = next(
            (i for i, cid in enumerate(ids_in_order) if cid == "test-group-checklist"), None
        )
        assert header_idx is not None, "test-group-header-row not found"
        assert checklist_idx is not None, "test-group-checklist not found"
        assert header_idx < checklist_idx

    def test_select_all_and_select_none_buttons_present(self):
        c = get_motl_input("test")
        sel_all = _find_by_id(c, "test-sel-all")
        sel_none = _find_by_id(c, "test-sel-none")
        assert sel_all is not None
        assert sel_none is not None
        assert isinstance(sel_all, dbc.Button)
        assert isinstance(sel_none, dbc.Button)

    def test_no_group_level_checkbox(self):
        c = get_motl_input("test")
        group_section = _find_by_id(c, "test-group-section")
        top_level_checklists = []
        for child in _iter_children(group_section):
            if isinstance(child, dcc.Checklist):
                top_level_checklists.append(child)
        assert len(top_level_checklists) == 1
        assert top_level_checklists[0].id == "test-group-checklist"

    def test_unique_ids_with_different_prefixes(self):
        c1 = get_motl_input("alpha")
        c2 = get_motl_input("beta")
        ids1 = _flatten_ids(c1)
        ids2 = _flatten_ids(c2)
        overlap = ids1 & ids2
        assert not overlap

    def test_deselect_hint_present(self):
        c = get_motl_input("test")
        group_section = _find_by_id(c, "test-group-section")

        def _all_text(comp) -> list[str]:
            texts = []
            if isinstance(getattr(comp, "children", None), str):
                texts.append(comp.children)
            for child in _iter_children(comp):
                texts.extend(_all_text(child))
            return texts

        texts = " ".join(_all_text(group_section)).lower()
        assert "does not change the pool group" in texts


# ===========================================================================
# _motlio_ops pure functions
# ===========================================================================

@pytest.mark.parametrize("path,expected", [
    ("foo.em", "emmotl"),
    ("bar.star", "relion"),
    ("baz.csv", "stopgap"),
    ("qux.tbl", "dynamo"),
    ("unknown.xyz", None),
    ("no_extension", None),
])
def test_motl_type_from_path(path, expected):
    assert motl_type_from_path(path) == expected


@pytest.mark.parametrize("version,expected", [
    (3.0, "relion"),
    (3.1, "relion"),
    (4.0, "relion"),
    (5.0, "relion5"),
    (5.1, "relion5_1"),
])
def test_relion_version_to_type(version, expected):
    assert relion_version_to_type(version) == expected


def test_validate_save_no_path():
    msgs = validate_save("emmotl", "", {})
    assert any("filename" in m.lower() for m in msgs)


def test_validate_save_no_type():
    msgs = validate_save("", "out.em", {})
    assert any("type" in m.lower() for m in msgs)


def test_validate_save_relion5_no_tomos():
    msgs = validate_save("relion", "out.star", {"rln_version": 5.0, "rln_tomos": None})
    assert any("tomogram" in m.lower() for m in msgs)


def test_validate_save_ok():
    msgs = validate_save("emmotl", "out.em", {})
    assert msgs == []


def test_save_kwargs_from_store_none_state():
    kwargs = save_kwargs_from_store(None, None)
    assert kwargs["rln_tomos"] is None
    assert kwargs["rln_binning"] is None


def test_save_kwargs_from_store_with_values():
    state = {"version": 3.1, "pixel_size": 2.5, "binning": 4.0, "tomos": None}
    kwargs = save_kwargs_from_store(state, "inherited_tomos")
    assert kwargs["rln_version"] == pytest.approx(3.1)
    assert kwargs["rln_pixel_size"] == pytest.approx(2.5)
    assert kwargs["rln_binning"] == pytest.approx(4.0)
    assert kwargs["rln_tomos"] == "inherited_tomos"


def test_save_kwargs_own_tomos_take_precedence():
    state = {"tomos": [{"a": 1}], "version": 3.0}
    kwargs = save_kwargs_from_store(state, "inherited")
    assert kwargs["rln_tomos"] == [{"a": 1}]


def test_load_kwargs_from_store_none():
    kwargs = load_kwargs_from_store(None)
    assert all(v is None for v in kwargs.values())


def test_load_kwargs_from_store_values():
    state = {"version": 3.1, "pixel_size": 1.5, "binning": 2.0, "tomo_format": "mrc"}
    kwargs = load_kwargs_from_store(state)
    assert kwargs["rln_version"] == pytest.approx(3.1)
    assert kwargs["rln_pixelsize"] == pytest.approx(1.5)
    assert kwargs["rln_binning"] == pytest.approx(2.0)
    assert kwargs["rln_tomoformat"] == "mrc"


# ===========================================================================
# motlsource.picker_options
# ===========================================================================

def _reg(*ids, active=True):
    return {mid: {"label": f"Label-{mid}", "active": active} for mid in ids}


def test_empty_registry_single():
    opts, val, msg = picker_options({}, None, multi=False)
    assert opts == []
    assert val is None
    assert "empty" in msg.lower()


def test_none_registry_multi():
    opts, val, msg = picker_options(None, None, multi=True)
    assert opts == []
    assert val == []
    assert "empty" in msg.lower()


def test_inactive_entries_excluded():
    reg = {"a": {"label": "A", "active": True}, "b": {"label": "B", "active": False}}
    opts, val, _ = picker_options(reg, None, multi=False)
    assert len(opts) == 1
    assert opts[0]["value"] == "a"


def test_missing_active_key_defaults_to_included():
    reg = {"x": {"label": "X"}}
    opts, _, _ = picker_options(reg, None, multi=False)
    assert len(opts) == 1


def test_single_preserves_valid_current():
    reg = _reg("a", "b", "c")
    _, val, _ = picker_options(reg, "b", multi=False)
    assert val == "b"


def test_single_falls_back_when_current_gone():
    reg = _reg("a", "b")
    _, val, _ = picker_options(reg, "x", multi=False)
    assert val == "a"


def test_single_falls_back_on_none_current():
    reg = _reg("a", "b")
    _, val, _ = picker_options(reg, None, multi=False)
    assert val == "a"


def test_multi_preserves_valid_subset():
    reg = _reg("a", "b", "c")
    _, val, _ = picker_options(reg, ["a", "c"], multi=True)
    assert val == ["a", "c"]


def test_multi_drops_stale_ids():
    reg = _reg("a", "b")
    _, val, _ = picker_options(reg, ["a", "gone"], multi=True)
    assert val == ["a"]


def test_multi_falls_back_to_all_when_nothing_kept():
    reg = _reg("a", "b")
    _, val, _ = picker_options(reg, ["gone"], multi=True)
    assert set(val) == {"a", "b"}


def test_multi_falls_back_on_none_current():
    reg = _reg("a", "b")
    _, val, _ = picker_options(reg, None, multi=True)
    assert set(val) == {"a", "b"}


def test_status_reports_count():
    reg = _reg("a", "b", "c")
    _, _, msg = picker_options(reg, None, multi=False)
    assert "3" in msg


def test_option_label_contains_mid():
    reg = {"my-id": {"label": "Nice label"}}
    opts, _, _ = picker_options(reg, None, multi=False)
    assert "my_id" in opts[0]["label"]
    assert "Nice label" in opts[0]["label"]


# ===========================================================================
# relionopts.read_relion_kwargs
# ===========================================================================

def test_read_relion_kwargs_empty_state():
    assert read_relion_kwargs({}) == {}


def test_read_relion_kwargs_none_state():
    assert read_relion_kwargs(None) == {}


def test_read_relion_kwargs_version_31():
    state = {"version": 3.1, "pixel_size": 3.42, "binning": 2.0}
    result = read_relion_kwargs(state)
    assert result["version"] == pytest.approx(3.1)
    assert result["pixel_size"] == pytest.approx(3.42)
    assert result["binning"] == pytest.approx(2.0)


def test_read_relion_kwargs_version_50():
    state = {"version": 5.0, "pixel_size": 1.5, "binning": None, "tomo_format": "mrc"}
    result = read_relion_kwargs(state)
    assert result["version"] == pytest.approx(5.0)
    assert result["tomo_format"] == "mrc"
    assert "binning" not in result


def test_read_relion_kwargs_version_51():
    state = {
        "version": 5.1,
        "pixel_size": 2.0,
        "binning": 4.0,
        "tomo_format": "mrc",
        "subtomo_format": "mrc",
    }
    result = read_relion_kwargs(state)
    assert result["version"] == pytest.approx(5.1)
    assert result["subtomo_format"] == "mrc"


def test_read_relion_kwargs_skips_empty_optional_strings():
    state = {"version": 3.1, "tomo_format": "", "subtomo_format": None}
    result = read_relion_kwargs(state)
    assert "tomo_format" not in result
    assert "subtomo_format" not in result


def test_read_relion_kwargs_none_version():
    state = {"version": None, "pixel_size": 2.0}
    result = read_relion_kwargs(state)
    assert "version" not in result
    assert result["pixel_size"] == pytest.approx(2.0)


# ===========================================================================
# Motl.write_out helpers
# ===========================================================================

def _emfile_available() -> bool:
    import importlib
    try:
        m = importlib.import_module("emfile")
        return callable(getattr(m, "read", None))
    except ImportError:
        return False


needs_emfile = pytest.mark.skipif(
    not _emfile_available(),
    reason="real emfile package not installed",
)


def _make_motl_df(n: int = 4) -> pd.DataFrame:
    data = {col: np.zeros(n) for col in Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    data["tomo_id"] = np.ones(n, dtype=float)
    return pd.DataFrame(data)


def _compare_common(expected: pd.DataFrame, actual: pd.DataFrame) -> None:
    shared = [c for c in expected.columns if c in actual.columns]
    assert shared, "no common columns to compare"
    significant = [c for c in shared if (expected[c].fillna(0) != 0).any()]
    assert len(expected) == len(actual)
    if not significant:
        return
    pd.testing.assert_frame_equal(
        expected[significant].reset_index(drop=True).fillna(0),
        actual[significant].reset_index(drop=True).fillna(0),
        check_dtype=False,
        rtol=1e-5,
    )


# ===========================================================================
# Motl.write_out round-trips
# ===========================================================================

@needs_emfile
def test_write_out_emmotl_roundtrip(tmp_path):
    from cryocat.core.cryomotl import EmMotl

    df = _make_motl_df()
    out = str(tmp_path / "test.em")
    Motl(df).write_out(out, "emmotl")
    _compare_common(df, EmMotl.load(out).df)


def test_write_out_stopgap_roundtrip(tmp_path):
    df = _make_motl_df()
    out = str(tmp_path / "test.star")
    Motl(df).write_out(out, "stopgap")
    loaded = StopgapMotl(out).df
    _compare_common(df, loaded)


@needs_emfile
def test_write_out_dynamo_roundtrip(tmp_path):
    from cryocat.core.cryomotl import DynamoMotl

    df = _make_motl_df()
    out = str(tmp_path / "test.tbl")
    Motl(df).write_out(out, "dynamo")
    _compare_common(df, DynamoMotl(out).df)


def test_write_out_relion_roundtrip(tmp_path):
    import warnings

    df = _make_motl_df()
    out = str(tmp_path / "test.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Motl(df).write_out(out, "relion")
    loaded = RelionMotl(out).df
    _compare_common(df, loaded)


def test_write_out_relion5_missing_tomos_raises(tmp_path):
    out = str(tmp_path / "test.star")
    with pytest.raises(UserInputError, match="requires tomogram data"):
        Motl(_make_motl_df()).write_out(out, "relion5")
    assert not (tmp_path / "test.star").exists()


def test_write_out_relion5_with_tomos(tmp_path):
    import warnings

    tomos = pd.DataFrame(
        {
            "rlnTomoName": ["TS_001"],
            "rlnTomoSizeX": [100],
            "rlnTomoSizeY": [100],
            "rlnTomoSizeZ": [50],
        }
    )
    out = str(tmp_path / "particles.star")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        Motl(_make_motl_df(2)).write_out(out, "relion5", input_tomograms=tomos, binning=1.0)
    assert (tmp_path / "particles.star").exists()
    assert (tmp_path / "particles.star").stat().st_size > 0


def test_write_out_relionv5_alias_raises_same(tmp_path):
    with pytest.raises(UserInputError, match="requires tomogram data"):
        Motl(_make_motl_df()).write_out(str(tmp_path / "x.star"), "relionv5")


def test_write_out_unknown_type_raises(tmp_path):
    out = str(tmp_path / "test.xyz")
    with pytest.raises(UserInputError):
        Motl(_make_motl_df()).write_out(out, "frobnitz")
    assert not (tmp_path / "test.xyz").exists()


def test_write_out_stopgap_kwargs_forwarded(tmp_path):
    df = _make_motl_df()
    out = str(tmp_path / "test.star")
    Motl(df).write_out(out, "stopgap", update_coord=False, reset_index=False)
    assert (tmp_path / "test.star").exists()


def test_writer_annotations_for_save_dialog():
    """Annotations the save dialog depends on to render correct widgets.

    build_form uses typing.get_type_hints to resolve annotations;
    bool -> bool dropdown, str -> text input. Removing these annotations
    would silently degrade the dynamic writer form to plain text inputs.
    """
    from cryocat.core.cryomotl import RelionMotlv5

    sg_hints = typing.get_type_hints(StopgapMotl.write_out)
    assert sg_hints.get("update_coord") is bool
    assert sg_hints.get("reset_index") is bool

    rln_hints = typing.get_type_hints(RelionMotl.write_out)
    assert rln_hints.get("keep_all_entries") is bool
    assert rln_hints.get("add_object_id") is bool
    assert rln_hints.get("add_subunit_id") is bool

    v5_hints = typing.get_type_hints(RelionMotlv5.write_out)
    assert v5_hints.get("keep_all_entries") is bool
    assert v5_hints.get("add_object_id") is bool
    assert v5_hints.get("add_subunit_id") is bool
