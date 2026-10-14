"""Orientation-picker page — retained for import compatibility only.

The canonical implementation moved to cryocat.app.components.orientpicker.
This page is no longer in the TOOLS registry; the picker is accessible from
the Utilities tab (Utilities → Orientation Picker accordion) and via the
app-level orientation modal (orient-modal-*).
"""
from __future__ import annotations

# Re-export pure helpers so existing imports keep working.
from cryocat.app.components.orientpicker import (
    _normalize,
    _compute_angles,
    _nearest_sphere_point,
    _rotate_mesh_verts,
    _SPHERE_PTS,
    get_orientation_picker_controls,
    get_orientation_picker_graph,
    register_orientation_picker_callbacks,
)

from dash import html

_PREFIX = "op"

layout = html.Div([
    get_orientation_picker_controls(_PREFIX, mode=None, show_structure=True),
    get_orientation_picker_graph(_PREFIX),
])


def register_callbacks(app) -> None:
    register_orientation_picker_callbacks(app, _PREFIX, mode=None, show_structure=True)
