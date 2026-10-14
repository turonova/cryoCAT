"""Tests for cryocat.app.components.surfaceview pure functions.

GRID_INITIAL_LOAD_AND_LEFTOVERS T3 — _build_figure with empty or all-invisible
handles returns an empty figure without reading the registry, building traces,
or calling styled_figure (palette resolution).

test_build_figure_*_skips_styled_figure are RED with current code (styled_figure
is always called).  The registry-access tests are already GREEN (the loop body
is skipped) and serve as regression guards.
"""
from __future__ import annotations

import plotly.graph_objects as go


def test_build_figure_empty_handles_does_not_access_registry(monkeypatch):
    """Empty handles: _surface_registry.get() is never called."""
    from cryocat.app.components.surfaceview import _build_figure
    import cryocat.app.components.surfaceview as sv_mod

    calls = []
    monkeypatch.setattr(sv_mod._surface_registry, "get",
                        lambda sid: calls.append(sid) or None)
    _build_figure({}, None, None)
    assert calls == [], f"registry.get() called for empty handles: {calls}"


def test_build_figure_all_invisible_does_not_access_registry(monkeypatch):
    """All-invisible handles: _surface_registry.get() is never called."""
    from cryocat.app.components.surfaceview import _build_figure
    import cryocat.app.components.surfaceview as sv_mod

    calls = []
    monkeypatch.setattr(sv_mod._surface_registry, "get",
                        lambda sid: calls.append(sid) or None)
    handles = {
        "s1": {"visible": False, "representation": "mesh"},
        "s2": {"visible": False, "representation": "point_cloud"},
    }
    _build_figure(handles, None, None)
    assert calls == [], f"registry.get() called for invisible handles: {calls}"


def test_build_figure_empty_handles_skips_styled_figure(monkeypatch):
    """Empty handles: styled_figure is not called (palette resolution skipped)."""
    from cryocat.app.components.surfaceview import _build_figure
    import cryocat.app.components.surfaceview as sv_mod

    styled_calls = []

    def _mock_styled(fig, *args, **kwargs):
        styled_calls.append(1)
        return go.Figure()

    monkeypatch.setattr(sv_mod, "styled_figure", _mock_styled)
    result = _build_figure({}, None, None)
    assert styled_calls == [], "styled_figure must not be called for empty handles"
    assert len(result.data) == 0


def test_build_figure_all_invisible_skips_styled_figure(monkeypatch):
    """All-invisible handles: styled_figure is not called."""
    from cryocat.app.components.surfaceview import _build_figure
    import cryocat.app.components.surfaceview as sv_mod

    styled_calls = []

    def _mock_styled(fig, *args, **kwargs):
        styled_calls.append(1)
        return go.Figure()

    monkeypatch.setattr(sv_mod, "styled_figure", _mock_styled)
    handles = {
        "s1": {"visible": False, "representation": "mesh"},
        "s2": {"visible": False, "representation": "point_cloud"},
    }
    result = _build_figure(handles, None, None)
    assert styled_calls == [], "styled_figure must not be called for all-invisible handles"
    assert len(result.data) == 0
