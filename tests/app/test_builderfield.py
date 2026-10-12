"""Tests for builderfield, anglesbuilder public API, and anglesfield."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from dash import html
import dash_bootstrap_components as dbc

from cryocat.app.components.builderfield import get_builder_field
from cryocat.app.components.anglesbuilder import (
    inplane_figure,
    angles_preview_figures,
    get_angles_builder_panel,
    get_angles_builder_sidebar_content,
    _ID_TYPE,
)
from cryocat.app.components.anglesfield import get_angles_field
from cryocat.app.components.rotationbuilder import get_rotation_builder_panel


# ── builderfield layout ───────────────────────────────────────────────────────

class TestGetBuilderField:
    def _get(self, prefix="test-bf", target_id="test-input", **kwargs):
        panel = html.Div("body")
        return get_builder_field(
            prefix, panel=panel, title="Test", target_id=target_id, **kwargs
        )

    def _flatten(self, component, result=None):
        if result is None:
            result = []
        result.append(component)
        children = getattr(component, "children", None)
        if children is None:
            return result
        if isinstance(children, list):
            for c in children:
                self._flatten(c, result)
        elif hasattr(children, "children"):
            self._flatten(children, result)
        return result

    def _find(self, component, predicate):
        return [c for c in self._flatten(component) if predicate(c)]

    def test_returns_div(self):
        result = self._get()
        assert isinstance(result, html.Div)

    def test_build_btn_present(self):
        result = self._get(prefix="bf1")
        btns = self._find(result, lambda c: isinstance(c, dbc.Button) and getattr(c, "id", None) == "bf1-build-btn")
        assert len(btns) == 1

    def test_modal_present(self):
        result = self._get(prefix="bf2")
        modals = self._find(result, lambda c: isinstance(c, dbc.Modal) and getattr(c, "id", None) == "bf2-modal")
        assert len(modals) == 1

    def test_modal_default_size_lg(self):
        result = self._get(prefix="bf3")
        modals = self._find(result, lambda c: isinstance(c, dbc.Modal))
        assert modals[0].size == "lg"

    def test_modal_size_xl(self):
        result = self._get(prefix="bf4", modal_size="xl")
        modals = self._find(result, lambda c: isinstance(c, dbc.Modal))
        assert modals[0].size == "xl"

    def test_string_target_id_on_input(self):
        result = self._get(prefix="bf5", target_id="my-input")
        inputs = self._find(result, lambda c: isinstance(c, dbc.Input) and getattr(c, "id", None) == "my-input")
        assert len(inputs) == 1

    def test_dict_target_id_on_input(self):
        tid = {"type": "some-type", "param": "p"}
        result = self._get(prefix="bf6", target_id=tid)
        inputs = self._find(result, lambda c: isinstance(c, dbc.Input) and getattr(c, "id", None) == tid)
        assert len(inputs) == 1


# ── builderfield duplicate registration guard ─────────────────────────────────

class TestRegisterBuilderFieldCallbacksDuplicateGuard:
    def test_duplicate_raises(self, tmp_path):
        import importlib
        import sys
        # Use a fresh import of the module to get a clean _REGISTERED set.
        # We mock the dash app to avoid needing a real Dash instance.
        from unittest.mock import MagicMock
        import cryocat.app.components.builderfield as bf_mod

        saved = set(bf_mod._REGISTERED)
        unique_prefix = "__test_dup_bf__"
        bf_mod._REGISTERED.discard(unique_prefix)

        app_mock = MagicMock()
        app_mock.callback.return_value = lambda fn: fn

        bf_mod.register_builder_field_callbacks(
            app_mock, unique_prefix,
            value_store_id="x-value",
            target_id="x-input",
        )
        with pytest.raises(RuntimeError, match="already registered"):
            bf_mod.register_builder_field_callbacks(
                app_mock, unique_prefix,
                value_store_id="x-value",
                target_id="x-input",
            )

        # Restore state so other tests are unaffected.
        bf_mod._REGISTERED.discard(unique_prefix)
        bf_mod._REGISTERED.update(saved)


# ── anglesbuilder public API ──────────────────────────────────────────────────

class TestInplaneFigure:
    def _angles(self, n=10):
        phi = np.linspace(0, 360, n, endpoint=False)
        theta = np.full(n, 45.0)
        psi = np.zeros(n)
        return np.column_stack([phi, theta, psi])

    def test_returns_go_figure(self):
        import plotly.graph_objects as go
        fig = inplane_figure(self._angles(), None)
        assert isinstance(fig, go.Figure)

    def test_polar_trace_present(self):
        import plotly.graph_objects as go
        fig = inplane_figure(self._angles(12), None)
        assert any(isinstance(t, go.Scatterpolar) for t in fig.data)

    def test_unique_phi_values(self):
        angles = self._angles(8)
        import plotly.graph_objects as go
        fig = inplane_figure(angles, None)
        polar_trace = next(t for t in fig.data if isinstance(t, go.Scatterpolar))
        expected_n = len(np.unique(np.round(angles[:, 0], 8)))
        assert len(polar_trace.r) == expected_n


class TestAnglesPreviewFigures:
    def _angles(self):
        phi = np.linspace(0, 360, 8, endpoint=False)
        theta = np.full(8, 45.0)
        psi = np.zeros(8)
        return np.column_stack([phi, theta, psi])

    def test_returns_tuple_of_two(self):
        angles = self._angles()
        result = angles_preview_figures(angles, None)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_both_are_figures(self):
        import plotly.graph_objects as go
        angles = self._angles()
        sphere_fig, inplane_fig = angles_preview_figures(angles, None)
        assert isinstance(sphere_fig, go.Figure)
        assert isinstance(inplane_fig, go.Figure)


# ── anglesbuilder layout ─────────────────────────────────────────────────────

class TestAnglesBuilderLayout:
    def _flatten(self, component, result=None):
        if result is None:
            result = []
        result.append(component)
        children = getattr(component, "children", None)
        if children is None:
            return result
        if isinstance(children, list):
            for c in children:
                self._flatten(c, result)
        elif hasattr(children, "children"):
            self._flatten(children, result)
        return result

    def _stores(self, component):
        from dash import dcc
        return [c for c in self._flatten(component) if isinstance(c, dcc.Store)]

    def test_panel_has_value_store(self):
        panel = get_angles_builder_panel("p1")
        store_ids = [s.id for s in self._stores(panel)]
        assert "p1-value" in store_ids

    def test_panel_has_no_created_path_store(self):
        panel = get_angles_builder_panel("p2")
        store_ids = [s.id for s in self._stores(panel)]
        assert "p2-created-path" not in store_ids

    def test_sidebar_has_value_store(self):
        sidebar = get_angles_builder_sidebar_content("p3")
        store_ids = [s.id for s in self._stores(sidebar)]
        assert "p3-value" in store_ids


# ── anglesfield layout ────────────────────────────────────────────────────────

class TestAnglesFieldLayout:
    def _flatten(self, component, result=None):
        if result is None:
            result = []
        result.append(component)
        children = getattr(component, "children", None)
        if children is None:
            return result
        if isinstance(children, list):
            for c in children:
                self._flatten(c, result)
        elif hasattr(children, "children"):
            self._flatten(children, result)
        return result

    def test_returns_div(self):
        result = get_angles_field("af1")
        assert isinstance(result, html.Div)

    def test_has_build_btn(self):
        result = get_angles_field("af2")
        btns = [c for c in self._flatten(result) if isinstance(c, dbc.Button) and getattr(c, "id", None) == "af2-build-btn"]
        assert len(btns) == 1

    def test_has_modal_xl(self):
        result = get_angles_field("af3")
        modals = [c for c in self._flatten(result) if isinstance(c, dbc.Modal)]
        assert len(modals) == 1
        assert modals[0].size == "xl"

    def test_path_input_id(self):
        result = get_angles_field("af4")
        inputs = [c for c in self._flatten(result) if isinstance(c, dbc.Input) and getattr(c, "id", None) == "af4-path"]
        assert len(inputs) == 1


# ── rotationbuilder store rename ──────────────────────────────────────────────

class TestRotationBuilderValueStore:
    def _flatten(self, component, result=None):
        if result is None:
            result = []
        result.append(component)
        children = getattr(component, "children", None)
        if children is None:
            return result
        if isinstance(children, list):
            for c in children:
                self._flatten(c, result)
        elif hasattr(children, "children"):
            self._flatten(children, result)
        return result

    def test_has_value_store(self):
        from dash import dcc
        panel = get_rotation_builder_panel("rb1")
        stores = [c for c in self._flatten(panel) if isinstance(c, dcc.Store)]
        store_ids = [s.id for s in stores]
        assert "rb1-inner-value" in store_ids or "rb1-value" in store_ids

    def test_no_rot_euler_store(self):
        from dash import dcc
        panel = get_rotation_builder_panel("rb2")
        stores = [c for c in self._flatten(panel) if isinstance(c, dcc.Store)]
        store_ids = [s.id for s in stores]
        assert not any("rot-euler-store" in str(sid) for sid in store_ids)
