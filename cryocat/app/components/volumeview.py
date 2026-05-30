"""Shared 3D volume viewer component.

get_volume_view(prefix)                    -- layout: per-prefix stores + 3D graph.
register_volume_view_callbacks(app, prefix) -- extraction + draw callbacks.

Stores owned by this component (all under the given prefix):
  {prefix}-map-store  : {"data": list, "shape": [n0, n1, n2]}  binned volume
  {prefix}-map-mesh   : marching-cubes dict for the map isosurface (or None)
  {prefix}-mask-mesh  : single layer  {"core": mesh|None, "outer": mesh|None}
                        OR multi-layer [{"core", "outer", "color", "name"}, ...]
  {prefix}-mask-store : {"center": [c0,c1,c2], "radius": r, "gaussian": g}
                        all values in **binned** voxels

The embedding page must additionally render:
  {prefix}-iso-slider          dcc.Slider (range set by the load callback)
  {prefix}-mask-opacity-slider dcc.Slider

Public helpers
--------------
mesh_at(volume, level) -- marching-cubes on a volume array; returns go.Mesh3d kwarg dict or None.
"""

import numpy as np
import plotly.graph_objects as go
from dash import html, dcc, Input, Output, State

from cryocat.app.components.graphsettings import apply_settings_to_figure


# ── Private helpers ────────────────────────────────────────────────────────────

def mesh_at(volume, level):
    """Marching-cubes on *volume* at *level*. Returns a go.Mesh3d kwarg dict or None."""
    try:
        from skimage.measure import marching_cubes
        verts, faces, _, _ = marching_cubes(volume, level=level)
        if len(faces) == 0:
            return None
        return {
            "x": verts[:, 0].tolist(),
            "y": verts[:, 1].tolist(),
            "z": verts[:, 2].tolist(),
            "i": faces[:, 0].tolist(),
            "j": faces[:, 1].tolist(),
            "k": faces[:, 2].tolist(),
        }
    except Exception:
        return None


# Keep the private alias for backward compat within this module.
_mesh_at = mesh_at


def _scene_layout(shape):
    """Fixed-box scene dict that pins axes to the binned box dimensions."""
    n0, n1, n2 = shape
    m = max(n0, n1, n2)
    return {
        "xaxis": {"range": [0, n0], "title": "x"},
        "yaxis": {"range": [0, n1], "title": "y"},
        "zaxis": {"range": [0, n2], "title": "z"},
        "aspectmode": "manual",
        "aspectratio": {"x": n0 / m, "y": n1 / m, "z": n2 / m},
    }


def _finalize(fig, gs, shape):
    """Apply graph settings, then stamp fixed scene + uirevision on the dict."""
    fig_dict = apply_settings_to_figure(fig.to_plotly_json(), gs)
    layout = fig_dict.setdefault("layout", {})
    layout["uirevision"] = "volume-view"
    layout["height"] = 620
    layout["margin"] = {"t": 0, "b": 0, "l": 0, "r": 0}
    if shape:
        layout["scene"] = _scene_layout(shape)
    return go.Figure(fig_dict)


# ── Public API ─────────────────────────────────────────────────────────────────

def get_volume_view(prefix: str):
    """Layout fragment: per-prefix stores + 3D graph. Embed in main content area."""
    return html.Div(
        [
            dcc.Store(id=f"{prefix}-map-store"),
            dcc.Store(id=f"{prefix}-map-mesh"),
            dcc.Store(id=f"{prefix}-mask-mesh"),
            dcc.Store(id=f"{prefix}-mask-store"),
            dcc.Graph(
                id=f"{prefix}-3d-graph",
                style={"height": "620px"},
                config={"scrollZoom": True},
            ),
        ]
    )


def register_volume_view_callbacks(app, prefix: str, register_mask: bool = True):
    """Register callbacks for the volume viewer.

    Parameters
    ----------
    register_mask : bool
        If False, skip the ``{prefix}-mask-mesh`` extraction callback so the
        embedding page can supply its own multi-layer version.
    """

    # ── Map isosurface (recomputes only on map-store or iso-slider change) ─────
    @app.callback(
        Output(f"{prefix}-map-mesh", "data"),
        Input(f"{prefix}-map-store", "data"),
        Input(f"{prefix}-iso-slider", "value"),
        prevent_initial_call=True,
    )
    def _extract_map(map_data, iso_level):
        if not map_data:
            return None
        vol = np.asarray(map_data["data"], dtype=np.float32)
        dmin, dmax = float(vol.min()), float(vol.max())
        level = float(np.clip(
            iso_level if iso_level is not None else dmin + 0.3 * (dmax - dmin),
            dmin + 1e-6, dmax - 1e-6,
        ))
        return mesh_at(vol, level)

    # ── Mask mesh (recomputes only on mask-store change) ──────────────────────
    # Only registered when the caller doesn't supply its own mask callback.
    if register_mask:
        @app.callback(
            Output(f"{prefix}-mask-mesh", "data"),
            Input(f"{prefix}-mask-store", "data"),
            State(f"{prefix}-map-store", "data"),
            prevent_initial_call=True,
        )
        def _extract_mask(mask_params, map_data):
            if not mask_params or not map_data:
                return None
            shape = map_data.get("shape")
            if not shape:
                return None
            center = mask_params.get("center", [s // 2 for s in shape])
            radius = float(mask_params.get("radius", min(shape) // 4))
            gaussian = float(mask_params.get("gaussian", 0.0))
            if radius < 1:
                return None
            try:
                from cryocat.core import cryomask
                m = cryomask.spherical_mask(
                    mask_size=tuple(shape),
                    radius=radius,
                    center=center,
                    gaussian=gaussian,
                )
            except Exception:
                return None
            return {
                "core": mesh_at(m, 0.99),
                "outer": mesh_at(m, 0.10),
            }

    # ── Figure assembly ────────────────────────────────────────────────────────
    @app.callback(
        Output(f"{prefix}-3d-graph", "figure"),
        Input(f"{prefix}-map-mesh", "data"),
        Input(f"{prefix}-mask-mesh", "data"),
        Input(f"{prefix}-mask-opacity-slider", "value"),
        State(f"{prefix}-map-store", "data"),
        State("graph-settings-store", "data"),
        prevent_initial_call=True,
    )
    def _draw(map_mesh, mask_mesh, mask_opacity, map_data, gs):
        shape = (map_data or {}).get("shape")
        opacity = float(mask_opacity or 0.3)
        traces = []

        if map_mesh:
            traces.append(go.Mesh3d(
                **map_mesh,
                color="lightblue", opacity=0.6,
                name="Map isosurface", hoverinfo="skip",
            ))

        if mask_mesh:
            if isinstance(mask_mesh, list):
                # Multi-layer format: [{core, outer, color, name}, ...]
                for layer_mesh in mask_mesh:
                    color = layer_mesh.get("color", "#865B96")
                    name = layer_mesh.get("name", "Mask")
                    if layer_mesh.get("outer"):
                        traces.append(go.Mesh3d(
                            **layer_mesh["outer"],
                            color=color, opacity=opacity * 0.4,
                            name=f"{name}: extent", hoverinfo="skip",
                        ))
                    if layer_mesh.get("core"):
                        traces.append(go.Mesh3d(
                            **layer_mesh["core"],
                            color=color, opacity=opacity,
                            name=f"{name}: core", hoverinfo="skip",
                        ))
            else:
                # Single-layer format: {core, outer}
                outer = mask_mesh.get("outer")
                core = mask_mesh.get("core")
                if outer:
                    traces.append(go.Mesh3d(
                        **outer,
                        color="#865B96", opacity=opacity * 0.4,
                        name="Mask: smoothed extent", hoverinfo="skip",
                    ))
                if core:
                    traces.append(go.Mesh3d(
                        **core,
                        color="#865B96", opacity=opacity,
                        name="Mask: core (=1.0)", hoverinfo="skip",
                    ))

        return _finalize(go.Figure(data=traces), gs, shape)
