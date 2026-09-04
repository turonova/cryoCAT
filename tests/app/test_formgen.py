"""Tests for cryocat.app.formgen — T5 acceptance suite."""

import pytest
from typing import Literal

from dash import html

from cryocat.app import formgen
from cryocat.app.formgen import build_form, form_row, WIDGET_FACTORIES
from cryocat.app.apputils import generate_kwargs
from cryocat.app.pool import PoolState
from cryocat.utils.classutils import TYPE_HANDLERS


# ── Synthetic test fixture ────────────────────────────────────────────────────

def _sample_fn(
    name: str,
    count: int,
    scale: float = 1.0,
    mode: Literal["a", "b"] = "a",
) -> None:
    pass


# ── T5a: form_row is public ───────────────────────────────────────────────────

class TestFormRowPublic:
    def test_form_row_returns_dash_div(self):
        row = form_row("myfield", html.Div("w"), "A description")
        assert isinstance(row, html.Div)

    def test_form_row_optional_marker_in_label(self):
        row = form_row("myfield", html.Div("w"), "", truly_optional=True)
        label_text = row.children[0].children[0].children
        assert "(opt.)" in label_text

    def test_form_row_no_optional_marker_by_default(self):
        row = form_row("myfield", html.Div("w"), "")
        label_text = row.children[0].children[0].children
        assert "(opt.)" not in label_text

    def test_form_row_label_capitalises_name(self):
        row = form_row("my_field", html.Div("w"), "")
        label_text = row.children[0].children[0].children
        assert label_text.startswith("My field")

    def test_form_row_custom_label_id(self):
        row = form_row("x", html.Div("w"), "", label_id="custom-lbl")
        label_el = row.children[0].children[0]
        assert label_el.id == "custom-lbl"

    def test_form_row_tooltip_when_description(self):
        """form_row tooltip respects styles.TOOLTIPS_ENABLED.

        Off (default): the second label child is None — no Tooltip created.
        On: the second label child is a dbc.Tooltip instance.
        """
        import dash_bootstrap_components as dbc
        from cryocat.app import styles

        # Default flag value is False — no tooltip created.
        assert not styles.TOOLTIPS_ENABLED, (
            "TOOLTIPS_ENABLED default changed; update this test to match"
        )
        row_off = form_row("x", html.Div("w"), "some tooltip text")
        assert row_off.children[0].children[1] is None, (
            "Expected None when TOOLTIPS_ENABLED is False"
        )

        # With flag on, a Tooltip is created.
        old = styles.TOOLTIPS_ENABLED
        try:
            styles.TOOLTIPS_ENABLED = True
            row_on = form_row("x", html.Div("w"), "some tooltip text")
            assert isinstance(row_on.children[0].children[1], dbc.Tooltip), (
                "Expected dbc.Tooltip when TOOLTIPS_ENABLED is True"
            )
        finally:
            styles.TOOLTIPS_ENABLED = old

    def test_form_row_no_tooltip_when_empty_description(self):
        row = form_row("x", html.Div("w"), "")
        assert row.children[0].children[1] is None


# ── T5b: WIDGET_FACTORIES is public and complete ──────────────────────────────

class TestWidgetFactoriesPublic:
    def test_widget_factories_is_dict(self):
        assert isinstance(WIDGET_FACTORIES, dict)

    def test_all_type_handler_widgets_in_factories(self):
        for tag, handler in TYPE_HANDLERS.items():
            widget_key = handler["widget"]
            assert widget_key in WIDGET_FACTORIES, (
                f"TYPE_HANDLERS[{tag!r}]['widget'] = {widget_key!r} "
                f"not found in WIDGET_FACTORIES"
            )

    def test_every_factory_callable(self):
        for key, fn in WIDGET_FACTORIES.items():
            assert callable(fn), f"WIDGET_FACTORIES[{key!r}] is not callable"

    def test_widget_factory_type_annotation(self):
        assert WIDGET_FACTORIES.__class__ is dict


# ── T5c: build_form produces correct rows ────────────────────────────────────

class TestBuildForm:
    def test_returns_one_row_per_param(self):
        rows = build_form(_sample_fn)
        assert len(rows) == 4

    def test_rows_are_dash_divs(self):
        rows = build_form(_sample_fn, id_type="test-param")
        for row in rows:
            assert isinstance(row, html.Div)

    def test_exclude_removes_param(self):
        rows = build_form(_sample_fn, exclude=["name"])
        assert len(rows) == 3

    def test_id_extra_propagated(self):
        rows = build_form(_sample_fn, id_type="t", id_extra={"scope": "demo"})
        assert len(rows) == 4

    def test_no_params_returns_hint(self):
        def _empty() -> None:
            pass

        rows = build_form(_empty)
        assert len(rows) == 1
        assert isinstance(rows[0], html.Div)

    def test_type_error_from_factory_propagates(self):
        """build_form must NOT swallow TypeError from widget factories."""
        orig = WIDGET_FACTORIES.get("text")
        try:
            def _bad(*args, **kwargs):
                raise TypeError("simulated bad factory")

            WIDGET_FACTORIES["text"] = _bad
            with pytest.raises(TypeError, match="simulated bad factory"):
                build_form(_sample_fn)
        finally:
            if orig is not None:
                WIDGET_FACTORIES["text"] = orig


# ── T5d: generate_kwargs round-trip ──────────────────────────────────────────

_PS = PoolState.empty()


class TestGenerateKwargsRoundTrip:
    def test_str_roundtrip(self):
        ids = [{"type": "op-param", "param": "name", "tag": "str"}]
        assert generate_kwargs(ids, ["hello"], _PS) == {"name": "hello"}

    def test_int_roundtrip(self):
        ids = [{"type": "op-param", "param": "count", "tag": "int"}]
        assert generate_kwargs(ids, [42], _PS) == {"count": 42}

    def test_float_roundtrip(self):
        ids = [{"type": "op-param", "param": "scale", "tag": "float"}]
        assert generate_kwargs(ids, [3.14], _PS) == {"scale": pytest.approx(3.14)}

    def test_bool_roundtrip(self):
        ids = [{"type": "op-param", "param": "flag", "tag": "bool"}]
        assert generate_kwargs(ids, ["True"], _PS) == {"flag": True}

    def test_literal_roundtrip(self):
        ids = [{"type": "op-param", "param": "mode", "tag": "Literal"}]
        assert generate_kwargs(ids, ["a"], _PS) == {"mode": "a"}

    def test_tuple_roundtrip(self):
        ids = [
            {"type": "op-param", "param": "size", "tag": "Tuple", "slot": 0, "elem": "float"},
            {"type": "op-param", "param": "size", "tag": "Tuple", "slot": 1, "elem": "float"},
        ]
        result = generate_kwargs(ids, [1.0, 2.0], _PS)
        assert result == {"size": (1.0, 2.0)}

    def test_multiple_params_roundtrip(self):
        ids = [
            {"type": "op-param", "param": "name", "tag": "str"},
            {"type": "op-param", "param": "count", "tag": "int"},
        ]
        result = generate_kwargs(ids, ["test", 5], _PS)
        assert result == {"name": "test", "count": 5}
