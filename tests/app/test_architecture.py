"""Behaviour checks left over from the GUI-conventions suite.

Metadata helpers take pool handles, not row data; pool stores stay small;
registering the same palette loader twice should raise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── PERF_FINISH T1 — metadata pure functions accept handles, reject row data ──

def test_color_options_from_handle_accepts_handle():
    """T1 — color_options_from_handle returns dropdown options from a PoolEntry dict."""
    from cryocat.app.components.tomoview import color_options_from_handle

    handle = {"numeric_columns": ["x", "y", "score", "tomo_id"], "tomo_ids": [1]}
    opts = color_options_from_handle(handle)
    assert {"label": "x", "value": "x"} in opts
    assert not any(o["value"] == "tomo_id" for o in opts), "tomo_id must be excluded from color opts"


def test_color_options_from_handle_rejects_row_data():
    """T1 — color_options_from_handle raises TypeError when passed list[dict] store data."""
    from cryocat.app.components.tomoview import color_options_from_handle

    with pytest.raises(TypeError):
        color_options_from_handle([{"x": 1.0, "tomo_id": 1}])


def test_tomo_items_from_handle_accepts_handle():
    """T1 — tomo_items_from_handle returns menu items from a PoolEntry dict."""
    from cryocat.app.components.tomoview import tomo_items_from_handle

    handle = {"tomo_ids": [1, 2, 3], "numeric_columns": ["x"]}
    items = tomo_items_from_handle(handle, "test")
    assert len(items) == 3


def test_tomo_items_from_handle_rejects_row_data():
    """T1 — tomo_items_from_handle raises TypeError when passed list[dict] store data."""
    from cryocat.app.components.tomoview import tomo_items_from_handle

    with pytest.raises(TypeError):
        tomo_items_from_handle([{"tomo_id": 1}], "test")


def test_slider_specs_rejects_non_dict():
    """T1 — slider_specs raises TypeError when passed list[dict] instead of column_ranges dict."""
    from cryocat.app.components.tablefilter import slider_specs

    with pytest.raises(TypeError):
        slider_specs([{"x": 1}])


# ── PERF_FINISH T4 — stores carry no row data ──────────

def test_pool_stores_carry_no_row_data():
    """T4a — PoolState.to_stores() for a large motl fits under 256 KB with no list[dict]."""
    import json
    from cryocat.app.pool import insert_motl, PoolState, clear_payloads

    clear_payloads()
    rng = np.random.default_rng(0)
    n = 100_000
    df = pd.DataFrame({
        "tomo_id": rng.integers(1, 50, size=n).astype(float),
        "x": rng.random(n),
        "y": rng.random(n),
        "z": rng.random(n),
        "score": rng.random(n),
    })
    state = PoolState(registry={}, meta={}, next_id=0)
    new_state, _mid = insert_motl(state, df)
    registry, meta, _next = new_state.to_stores()

    total_bytes = len(json.dumps(registry).encode()) + len(json.dumps(meta).encode())
    assert total_bytes < 256_000, (
        f"Pool stores for 100k-row motl exceed 256 KB: {total_bytes:,} bytes"
    )
    for mid, entry in registry.items():
        assert not isinstance(entry, list), (
            f"registry[{mid!r}] is a list — stores must carry handles, not rows"
        )
    clear_payloads()



# ── Registration guard (§10) ──────────────────────────────────────────────────

@pytest.mark.xfail(
    reason=(
        "§10 registration guard not yet implemented in components — "
        "doc 8 will add a per-component seen-prefix set that raises on duplicate. "
        "Until then, Dash silently overwrites callbacks on re-registration."
    ),
    strict=True,
)
def test_paletteloader_double_registration_raises():
    """§10 — registering palette-loader callbacks twice must raise."""
    import dash
    from cryocat.app.components.paletteloader import register_palette_loader_callbacks

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    prefix = "test-pal-reg"
    register_palette_loader_callbacks(app, prefix, mode="discrete")
    with pytest.raises(Exception, match=prefix):
        register_palette_loader_callbacks(app, prefix, mode="discrete")
