"""Tests for the unified entry picker.

Covers pure function tests and structural / integration tests.
"""
from __future__ import annotations

import pandas as pd
import pytest

from cryocat.app.components.entrypicker import _build_options, decode_value, ref_label


# ── Helpers ────────────────────────────────────────────────────────────────────

def _pool_reg(n=2):
    """Minimal motl pool registry with *n* entries."""
    return {
        f"motl_{i+1}": {"label": f"Run{i+1}", "n_rows": 100 * (i + 1)}
        for i in range(n)
    }

def _dp_reg_df(n=1):
    """Data pool registry with *n* DataFrame entries."""
    return {
        f"data_{i+1}": {"label": f"Table{i+1}", "kind": "dataframe", "n_rows": 50 * (i + 1)}
        for i in range(n)
    }

def _dp_reg_mixed():
    """Data pool registry with a DataFrame, a volume, and an array entry."""
    return {
        "data_1": {"label": "MyTable",  "kind": "dataframe", "n_rows": 40},
        "data_2": {"label": "MyVolume", "kind": "volume"},
        "data_3": {"label": "MyArray",  "kind": "array"},
        "data_4": {"label": "MyDict",   "kind": "dict"},
    }


# ── Pure function tests ────────────────────────────────────────────────────────

class TestBuildOptions:

    def test_both_pools_grouped_and_labelled(self):
        """Options span motl pool + DF-kind data pool with section headers."""
        opts = _build_options(_pool_reg(2), _dp_reg_df(1))
        values = [o["value"] for o in opts]
        # Separator labels present
        assert "_sep_motl"   in values
        assert "_sep_tables" in values
        # Motl entries
        assert "motl:motl_1" in values
        assert "motl:motl_2" in values
        # Table entry
        assert "data:data_1" in values

    def test_only_motl_pool(self):
        """Options only contain motl group when data pool is empty."""
        opts = _build_options(_pool_reg(1), {})
        values = [o["value"] for o in opts]
        assert "_sep_motl"    in values
        assert "motl:motl_1"  in values
        assert "_sep_tables" not in values

    def test_only_data_pool(self):
        """Options only contain tables group when motl pool is empty."""
        opts = _build_options({}, _dp_reg_df(1))
        values = [o["value"] for o in opts]
        assert "_sep_tables"  in values
        assert "data:data_1"  in values
        assert "_sep_motl"   not in values

    def test_empty_both_pools(self):
        """No options when both pools are empty."""
        assert _build_options({}, {}) == []

    def test_none_registries_treated_as_empty(self):
        """None registries do not raise."""
        assert _build_options(None, None) == []

    def test_separators_are_disabled(self):
        """Section separator pseudo-options are marked disabled (never selectable)."""
        opts = _build_options(_pool_reg(1), _dp_reg_df(1))
        seps = [o for o in opts if o["value"].startswith("_sep_")]
        assert seps, "No separator options found"
        for sep in seps:
            assert sep.get("disabled") is True

    def test_motl_rows_in_label(self):
        """Row count appears in the motl option label."""
        opts = _build_options(_pool_reg(1), {})
        motl_opt = next(o for o in opts if o["value"] == "motl:motl_1")
        assert "100" in motl_opt["label"]

    def test_df_rows_in_label(self):
        """Row count appears in the data table option label."""
        opts = _build_options({}, _dp_reg_df(1))
        df_opt = next(o for o in opts if o["value"] == "data:data_1")
        assert "50" in df_opt["label"]


class TestNonDataFrameEntriesAbsent:
    """Volume, array, dict entries must be absent from the picker."""

    def test_volume_absent(self):
        opts = _build_options({}, _dp_reg_mixed())
        values = [o["value"] for o in opts]
        assert "data:data_2" not in values   # volume

    def test_array_absent(self):
        opts = _build_options({}, _dp_reg_mixed())
        values = [o["value"] for o in opts]
        assert "data:data_3" not in values   # array

    def test_dict_absent(self):
        opts = _build_options({}, _dp_reg_mixed())
        values = [o["value"] for o in opts]
        assert "data:data_4" not in values   # dict

    def test_dataframe_present(self):
        opts = _build_options({}, _dp_reg_mixed())
        values = [o["value"] for o in opts]
        assert "data:data_1" in values       # dataframe — must appear


class TestDecodeValue:

    def test_motl_prefix(self):
        assert decode_value("motl:motl_1") == {"motl_id": "motl_1"}

    def test_data_prefix(self):
        assert decode_value("data:data_3") == {"data_id": "data_3"}

    def test_none_returns_none(self):
        assert decode_value(None) is None

    def test_separator_returns_none(self):
        assert decode_value("_sep_motl")   is None
        assert decode_value("_sep_tables") is None

    def test_empty_string_returns_none(self):
        assert decode_value("") is None


class TestRefLabel:

    def test_motl_ref(self):
        label = ref_label({"motl_id": "motl_1"}, _pool_reg(1), {})
        assert label == "Run1"

    def test_data_ref(self):
        label = ref_label({"data_id": "data_1"}, {}, _dp_reg_df(1))
        assert label == "Table1"

    def test_none_ref(self):
        assert ref_label(None, _pool_reg(1), _dp_reg_df(1)) == ""

    def test_missing_id_falls_back_to_id(self):
        label = ref_label({"motl_id": "motl_99"}, _pool_reg(1), {})
        assert label == "motl_99"


# ── Structural tests ───────────────────────────────────────────────────────────

class TestPickerStructure:

    def test_get_entry_picker_is_single_definition(self):
        """One shared definition; both tableeditor and tableview import it."""
        from cryocat.app.components import entrypicker as ep
        from cryocat.app.components import tableeditor as te
        import inspect
        # tableeditor.get_table_editor embeds entrypicker.get_entry_picker by calling it
        src = inspect.getsource(te.get_table_editor)
        assert "entrypicker.get_entry_picker" in src, (
            "get_table_editor must embed the shared entry picker via entrypicker.get_entry_picker"
        )

    def test_get_entry_picker_returns_html_div(self):
        from dash import html
        from cryocat.app.components.entrypicker import get_entry_picker
        result = get_entry_picker("test-pfx")
        assert isinstance(result, html.Div)

    def test_picker_store_id_uses_prefix(self):
        """The picker ref store ID is correctly scoped to the prefix."""
        from dash import dcc
        from cryocat.app.components.entrypicker import get_entry_picker

        pfx = "pfx-test"
        picker = get_entry_picker(pfx)

        found_ids: list[str] = []

        def _walk(comp):
            cid = getattr(comp, "id", None)
            if cid:
                found_ids.append(cid)
            children = getattr(comp, "children", None) or []
            if not isinstance(children, (list, tuple)):
                children = [children]
            for child in children:
                if child is not None:
                    _walk(child)

        _walk(picker)
        assert f"{pfx}-ref" in found_ids, f"Store {pfx}-ref not found; ids={found_ids}"

    def test_decode_round_trip_motl(self):
        """Encoding produced by _build_options decodes back via decode_value."""
        opts = _build_options(_pool_reg(1), {})
        motl_opt = next(o for o in opts if o["value"].startswith("motl:"))
        ref = decode_value(motl_opt["value"])
        assert ref is not None
        assert "motl_id" in ref

    def test_decode_round_trip_data(self):
        """Encoding produced by _build_options decodes back via decode_value."""
        opts = _build_options({}, _dp_reg_df(1))
        data_opt = next(o for o in opts if o["value"].startswith("data:"))
        ref = decode_value(data_opt["value"])
        assert ref is not None
        assert "data_id" in ref
