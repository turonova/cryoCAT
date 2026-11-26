"""GUI-logic tests for cryocat/app/suite/pages/psta.py.

Tests pure helper functions that are used by callbacks — no running Dash app required.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.app.assertions import assert_grid_cells_have_text, assert_grid_shows
from cryocat.app.suite.pages.psta import (
    _build_sta_run_from_form,
    _validate_sep_ranges,
    _expand_sep_ranges,
)


# ── _build_sta_run_from_form ─────────────────────────────────────────────────


def _form_kwargs(tmp_path: Path) -> dict:
    sub = tmp_path / "subtomos"
    sub.mkdir()
    return {
        "output_base": str(tmp_path / "out"),
        "folder_name": "run01",
        "n_runs": 1,
        "subtomo_path": str(sub),
        "run_mode": "singleref",
        "motl_paths_text": None,
        "cls_paths": None,
        "iter_n": 10,
        "mask": None,
        "ccmask": None,
        "wedgelist": None,
        "ref_files_text": None,
    }


def test_build_sta_run_success(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    sta_run, motl_paths, errors = _build_sta_run_from_form(**kwargs)
    assert errors == []
    assert sta_run is not None
    assert str(sta_run.folder_name) == "run01"
    assert sta_run.run_mode == "singleref"
    assert len(sta_run.schedule) == 1


def test_build_sta_run_missing_output_base(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["output_base"] = ""
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert any("output base" in e.lower() for e in errors)


def test_build_sta_run_missing_folder_name(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["folder_name"] = None
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert any("folder name" in e.lower() for e in errors)


def test_build_sta_run_missing_subtomo_path(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["subtomo_path"] = None
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert any("subtomogram" in e.lower() or "subtomo" in e.lower() for e in errors)


def test_build_sta_run_motl_paths_from_cls(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["cls_paths"] = ["/tmp/motl_mr1.star", "/tmp/motl_mr2.star"]
    sta_run, motl_paths, errors = _build_sta_run_from_form(**kwargs)
    assert errors == []
    assert len(motl_paths) == 2


def test_build_sta_run_motl_paths_from_text(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["motl_paths_text"] = "/tmp/a.star, /tmp/b.star"
    sta_run, motl_paths, errors = _build_sta_run_from_form(**kwargs)
    assert errors == []
    assert len(motl_paths) == 2


def test_build_sta_run_cls_paths_take_priority_over_text(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["cls_paths"] = ["/cls/motl.star"]
    kwargs["motl_paths_text"] = "/text/motl.star"
    _, motl_paths, _ = _build_sta_run_from_form(**kwargs)
    assert len(motl_paths) == 1
    assert "cls" in str(motl_paths[0])


def test_build_sta_run_base_params_from_paths(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["mask"] = "/masks/mask.em"
    kwargs["ccmask"] = "/masks/ccmask.em"
    kwargs["wedgelist"] = "/lists/wedge.star"
    sta_run, _, _ = _build_sta_run_from_form(**kwargs)
    bp = sta_run.base_params
    assert bp.get("mask name") == "/masks/mask.em"
    assert bp.get("cc mask name") == "/masks/ccmask.em"
    assert bp.get("wedge list") == "/lists/wedge.star"


def test_build_sta_run_n_runs(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["n_runs"] = 4
    sta_run, _, _ = _build_sta_run_from_form(**kwargs)
    assert sta_run.n_runs == 4


def test_build_sta_run_references_from_text(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["ref_files_text"] = "/ref/ref_1.em, /ref/ref_2.em"
    sta_run, _, _ = _build_sta_run_from_form(**kwargs)
    assert len(sta_run.references) == 2


def test_build_sta_run_multiple_errors_reported(tmp_path):
    kwargs = _form_kwargs(tmp_path)
    kwargs["output_base"] = ""
    kwargs["folder_name"] = ""
    kwargs["subtomo_path"] = ""
    _, _, errors = _build_sta_run_from_form(**kwargs)
    assert len(errors) >= 3


# ── _validate_sep_ranges ─────────────────────────────────────────────────────


def test_validate_sep_ranges_valid_contiguous():
    rows = [
        {"from_iter": 1, "to_iter": 5},
        {"from_iter": 6, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert errors == []


def test_validate_sep_ranges_gap():
    rows = [
        {"from_iter": 1, "to_iter": 4},
        {"from_iter": 6, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("gap" in e.lower() for e in errors)


def test_validate_sep_ranges_overlap():
    rows = [
        {"from_iter": 1, "to_iter": 6},
        {"from_iter": 5, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("overlap" in e.lower() for e in errors)


def test_validate_sep_ranges_bad_start():
    rows = [
        {"from_iter": 2, "to_iter": 10},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("start at 1" in e for e in errors)


def test_validate_sep_ranges_bad_end():
    rows = [
        {"from_iter": 1, "to_iter": 8},
    ]
    errors = _validate_sep_ranges(rows, 10)
    assert any("end at 10" in e for e in errors)


# ── _expand_sep_ranges ───────────────────────────────────────────────────────


def test_expand_sep_ranges_constant():
    rows = [{"from_iter": 1, "to_iter": 5, "cone_angle": "30", "cone_sampling": "5",
             "inplane_angle": "360", "inplane_sampling": "5"}]
    fields = ["cone_angle", "cone_sampling", "inplane_angle", "inplane_sampling"]
    result = _expand_sep_ranges(rows, fields, iter_n=5, start_idx=1)
    assert result["cone_angle"] == "30 30 30 30 30"
    assert result["cone_sampling"] == "5 5 5 5 5"


def test_expand_sep_ranges_two_ranges():
    rows = [
        {"from_iter": 1, "to_iter": 3, "cone_angle": "30", "cone_sampling": "5",
         "inplane_angle": "360", "inplane_sampling": "5"},
        {"from_iter": 4, "to_iter": 5, "cone_angle": "10", "cone_sampling": "2",
         "inplane_angle": "360", "inplane_sampling": "2"},
    ]
    fields = ["cone_angle", "cone_sampling", "inplane_angle", "inplane_sampling"]
    result = _expand_sep_ranges(rows, fields, iter_n=5, start_idx=1)
    assert result["cone_angle"] == "30 30 30 10 10"
    assert result["cone_sampling"] == "5 5 5 2 2"


# ── Table content: the STA grids show their rows ──────────────────────────────

# Setup-form values that, on top of the layout defaults, give a complete
# parameter table (every mandatory column filled).
_SETUP_FORM = {
    "sta-setup-motl": "motl_1.em",
    "sta-setup-ref": "ref",
    "sta-setup-mask": "mask.em",
    "sta-setup-cc-mask": "ccmask.em",
    "sta-setup-fsc-mask": "fsc.em",
    "sta-setup-wedge-list": "wedge.em",
    "sta-setup-subtomo-path": "subtomos/subtomo",
    "sta-setup-cone-angle": 10,
    "sta-setup-cone-sampling": 2,
    "sta-setup-inplane-angle": 10,
    "sta-setup-inplane-sampling": 2,
    "sta-setup-low-pass": 20,
    "sta-setup-pixel-size": 1.0,
}


def _page_app():
    import dash
    from cryocat.app.suite.pages import psta

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    psta.register_callbacks(app)
    return app


def _callback_entry(app, name: str, output: str) -> tuple:
    """Return (undecorated callback, callback_map entry) for *name* writing *output*."""
    for key, entry in app.callback_map.items():
        fn = getattr(entry.get("callback"), "__wrapped__", None)
        if fn is not None and fn.__name__ == name and output in key:
            return fn, entry
    raise LookupError(f"no callback {name!r} writing {output!r}")


def _call(fn, *args, triggered: str = ""):
    """Run *fn* inside a Dash callback context (for ``ctx.triggered_id``)."""
    from contextvars import copy_context
    from dash._callback_context import context_value
    from dash._utils import AttributeDict

    def _run():
        context_value.set(AttributeDict(triggered_inputs=[{"prop_id": triggered, "value": None}]))
        return fn(*args)

    return copy_context().run(_run)


def _layout_components() -> dict:
    """Map every string id in the psta layout to its component."""
    from cryocat.app.suite.pages import psta

    found: dict = {}

    def _walk(node):
        if node is None or isinstance(node, (str, int, float, bool)):
            return
        if isinstance(node, (list, tuple)):
            for child in node:
                _walk(child)
            return
        if isinstance(getattr(node, "id", None), str):
            found[node.id] = node
        _walk(getattr(node, "children", None))

    _walk(psta.layout)
    return found


def _wire(col_defs, rows) -> tuple[list[dict], list[dict]]:
    import json
    from dash._utils import to_json

    wire = json.loads(to_json({"columnDefs": col_defs, "rowData": rows}))
    return wire["columnDefs"], wire["rowData"]


def _setup_store(app) -> dict:
    """Run Create on the setup form (layout defaults plus _SETUP_FORM); return the store data."""
    import dash

    fn, entry = _callback_entry(app, "create_setup_df", "sta-setup-df-store.data")
    comps = _layout_components()
    values = [
        _SETUP_FORM.get(s["id"], getattr(comps.get(s["id"]), s["property"], None))
        for s in entry["state"]
    ]
    data, status, _ = fn(1, *values)
    assert data is not dash.no_update, f"setup dataframe not created: {status}"
    return data


def _params_store(app, tmp_path) -> dict:
    """Save the created setup table, load it as a parameter file; return the params store data."""
    import dash

    setup = _setup_store(app)
    path = str(tmp_path / "params.txt")
    saved = _callback_entry(app, "save_setup", "sta-setup-save-status.children")[0](
        1, setup["records"], setup, path,
    )
    assert "failed" not in saved, saved
    _, params, status = _callback_entry(app, "load_from_params", "sta-param-status.children")[0](
        1, path, "auto", "_", None,
    )
    assert params is not dash.no_update, f"parameter file not loaded: {status}"
    return params


def _manifest_rows(app, tmp_path) -> list[dict]:
    """Run Create folders for a single run in *tmp_path*; return the manifest grid rows."""
    import dash

    subtomos = tmp_path / "subtomos"
    subtomos.mkdir()
    status, rows = _callback_entry(app, "_create_run_folders", "sta-rf-manifest-grid.rowData")[0](
        1, str(tmp_path / "out"), "run01", [], 1, str(subtomos), None, None, 10, 1,
        None, None, None, None, [],
    )
    assert rows is not dash.no_update, f"run folders not created: {status}"
    return rows


def _sep_rows(app, group: str) -> list[dict]:
    """Open the *group* ('ang' or 'bp') set-separately modal; return the rows it puts in the grid."""
    form = {"ang": (10, 10, 2, 10, 2), "bp": (10, 2, 2.0, 20, 3.0)}[group]
    is_open, rows = _call(
        _callback_entry(app, f"_toggle_{group}_sep_modal", f"sta-{group}-sep-modal.is_open")[0],
        1, None, False, *form, None,
        triggered=f"sta-{group}-sep-btn.n_clicks",
    )
    assert is_open, f"sta-{group}-sep-modal did not open"
    return rows


def test_setup_grid_rows_carry_every_column():
    app = _page_app()
    setup = _setup_store(app)
    rows, col_defs, _ = _callback_entry(app, "populate_setup_grid", "sta-setup-grid.rowData")[0](setup)
    assert_grid_shows(*_wire(col_defs, rows), setup["records"])


def test_params_grid_rows_carry_every_column(tmp_path):
    app = _page_app()
    params = _params_store(app, tmp_path)
    rows, col_defs = _callback_entry(app, "update_params_table", "sta-params-grid.rowData")[0](params)
    assert_grid_shows(*_wire(col_defs, rows), params["records"])


def test_run_folder_manifest_grid_rows_carry_every_column(tmp_path):
    app = _page_app()
    rows = _manifest_rows(app, tmp_path)
    col_defs = _layout_components()["sta-rf-manifest-grid"].columnDefs
    assert_grid_shows(*_wire(col_defs, rows), rows)


def test_angular_sep_grid_rows_carry_every_column():
    app = _page_app()
    rows = _sep_rows(app, "ang")
    col_defs = _layout_components()["sta-ang-sep-grid"].columnDefs
    assert_grid_shows(*_wire(col_defs, rows), rows)


def test_bandpass_sep_grid_rows_carry_every_column():
    app = _page_app()
    rows = _sep_rows(app, "bp")
    col_defs = _layout_components()["sta-bp-sep-grid"].columnDefs
    assert_grid_shows(*_wire(col_defs, rows), rows)


# ── Browser: the STA grids' cells contain text ──────────────────────────────


def _browser_app(outputs: list[tuple[str, str, object]]):
    """The STA page with the suite stores and a button that writes *outputs*."""
    import dash
    from dash import Input, Output, html
    from cryocat.app.suite.app import POOL_STORES
    from cryocat.app.components.graphsettings import get_graph_settings_components
    from cryocat.app.suite.pages import psta

    app = dash.Dash(__name__, suppress_callback_exceptions=True)
    app.layout = html.Div(
        [
            html.Button("Load", id="grid-test-load-btn"),
            *POOL_STORES,
            *get_graph_settings_components(),
            psta.layout,
        ]
    )
    psta.register_callbacks(app)

    @app.callback(
        [Output(cid, prop, allow_duplicate=True) for cid, prop, _ in outputs],
        Input("grid-test-load-btn", "n_clicks"),
        prevent_initial_call=True,
    )
    def _load(_):
        return [value for _, _, value in outputs]

    return app


def _show_in_browser(dash_duo, outputs: list[tuple[str, str, object]], grid_id: str) -> None:
    dash_duo.start_server(_browser_app(outputs))
    dash_duo.wait_for_element("#grid-test-load-btn")
    dash_duo.driver.execute_script("document.getElementById('grid-test-load-btn').click()")
    assert_grid_cells_have_text(dash_duo, grid_id)


@pytest.mark.slow
def test_setup_grid_cells_show_text_in_browser(dash_duo):
    setup = _setup_store(_page_app())
    _show_in_browser(dash_duo, [
        ("sta-setup-df-store", "data", setup),
        ("sta-main-tabs", "active_tab", "tab-setup"),
    ], "sta-setup-grid")


@pytest.mark.slow
def test_params_grid_cells_show_text_in_browser(dash_duo, tmp_path):
    params = _params_store(_page_app(), tmp_path)
    _show_in_browser(dash_duo, [
        ("sta-params-store", "data", params),
        ("sta-main-tabs", "active_tab", "tab-params"),
    ], "sta-params-grid")


@pytest.mark.slow
def test_run_folder_manifest_grid_cells_show_text_in_browser(dash_duo, tmp_path):
    rows = _manifest_rows(_page_app(), tmp_path)
    _show_in_browser(dash_duo, [
        ("sta-rf-manifest-grid", "rowData", rows),
        ("sta-main-tabs", "active_tab", "tab-rf-manifest"),
    ], "sta-rf-manifest-grid")


@pytest.mark.slow
def test_angular_sep_grid_cells_show_text_in_browser(dash_duo):
    rows = _sep_rows(_page_app(), "ang")
    _show_in_browser(dash_duo, [
        ("sta-ang-sep-grid", "rowData", rows),
        ("sta-ang-sep-modal", "is_open", True),
    ], "sta-ang-sep-grid")


@pytest.mark.slow
def test_bandpass_sep_grid_cells_show_text_in_browser(dash_duo):
    rows = _sep_rows(_page_app(), "bp")
    _show_in_browser(dash_duo, [
        ("sta-bp-sep-grid", "rowData", rows),
        ("sta-bp-sep-modal", "is_open", True),
    ], "sta-bp-sep-grid")


@pytest.mark.slow
def test_psta_page_loads_in_browser(dash_duo, suite_app, monkeypatch):
    """The STA page loads, its main components render, and the console shows no errors."""
    from cryocat.app.suite.pages import psta
    from tests.app.conftest import main_component_ids, open_suite_page

    errors = open_suite_page(dash_duo, suite_app, "sta", monkeypatch)
    missing = [cid for cid in main_component_ids(psta.layout) if not dash_duo.find_elements(cid, attribute="ID")]
    assert not missing, f"components not rendered: {missing}"
    assert not errors, f"browser errors: {errors}"
