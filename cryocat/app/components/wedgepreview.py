"""Shared wedge-mask preview rendering.

Both ``cryocat.app.suite.pages.putilities`` (standalone builder) and
``cryocat.app.suite.pages.ppana`` (peak-analysis modal) call into
:func:`generate_wedge_mask` and display the result in-page. The actual
rendering is identical across both call sites, so it lives here.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go


def wedge_xz_figure(mask: np.ndarray) -> go.Figure:
    """Render the middle XZ slice of a 3D wedge mask as a square heatmap.

    The mask is returned by
    :func:`cryocat.core.cryowedge.generate_wedge_mask` in xyz convention
    (axis 0 = x, axis 1 = y, axis 2 = z) and FFT layout (DC at corner). We
    ``fftshift`` first so DC lands at the centre, then slice at ``y = ny // 2``
    to get the XZ plane. ``scaleanchor`` locks the y axis to the x axis so the
    plot stays pixel-square regardless of container size, and the Greys
    colorscale is reversed so 1 renders white and 0 renders black.
    """
    arr = np.asarray(mask)
    if arr.ndim != 3:
        raise ValueError(f"Wedge mask must be 3D, got shape {arr.shape}.")
    arr = np.fft.fftshift(arr)
    nx, ny, nz = arr.shape
    mid_y = ny // 2
    sl = arr[:, mid_y, :]  # (nx, nz)

    fig = go.Figure(
        data=go.Heatmap(
            z=sl.T,  # rows -> z, cols -> x  (image-like orientation)
            colorscale="Greys",
            reversescale=True,
            colorbar=dict(title="weight"),
            zsmooth=False,
        )
    )
    fig.update_xaxes(title="x", range=[0, nx - 1], constrain="domain")
    fig.update_yaxes(
        title="z", range=[0, nz - 1],
        scaleanchor="x", scaleratio=1, constrain="domain",
    )
    fig.update_layout(
        title=f"Wedge mask — middle XZ slice (y = {mid_y})",
        margin={"l": 40, "r": 20, "t": 40, "b": 40},
        uirevision="wedge-preview",
    )
    return fig
