"""Tests for cryocat.app.apputils pure functions and run_operation_to_pool.

Covers:
- format_columns, get_relevant_features, _scalar, _format_relion_params,
  get_class_by_name, make_axis_trace.
- run_operation_to_pool: success path (event, pool entry, provenance),
  error path (event, pool unchanged, no provenance),
  replaces= semantics, receiver resolution.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import numpy as np
import pytest

import cryocat.app.provenance as prov
from cryocat.app import session
from cryocat.app.apputils import (
    format_columns,
    get_relevant_features,
    _scalar,
    _format_relion_params,
    get_class_by_name,
    make_axis_trace,
    run_operation_to_pool,
)
from cryocat.app.pool import PoolState


# ===========================================================================
# format_columns
# ===========================================================================

def test_format_columns_float():
    df = pd.DataFrame({"x": [1.5, 2.5], "n": [1, 2]})
    cols = format_columns(df)
    assert len(cols) == 2
    float_col = next(c for c in cols if c["name"] == "x")
    assert float_col["type"] == "numeric"
    int_col = next(c for c in cols if c["name"] == "n")
    assert "type" not in int_col


def test_format_columns_not_editable():
    df = pd.DataFrame({"a": [1]})
    cols = format_columns(df)
    assert all(not c["editable"] for c in cols)


def test_format_columns_empty():
    df = pd.DataFrame()
    assert format_columns(df) == []


# ===========================================================================
# get_relevant_features
# ===========================================================================

def test_get_relevant_features_basic():
    features = ["feat_a", "feat_b", "feat_c_desc"]
    result = get_relevant_features("desc", features)
    assert "feat_c_desc" not in result
    assert "feat_a" in result
    assert "feat_b" in result


def test_get_relevant_features_none_removed():
    features = ["a", "b_desc", "c_desc"]
    result = get_relevant_features("desc", features)
    assert result == ["a"]


# ===========================================================================
# _scalar
# ===========================================================================

class TestScalar:
    def test_none(self):
        assert _scalar(None) is None

    def test_float(self):
        assert _scalar(3.14) == pytest.approx(3.14)

    def test_list(self):
        assert _scalar([5.0]) == pytest.approx(5.0)

    def test_empty_list(self):
        assert _scalar([]) is None

    def test_non_numeric_string(self):
        assert _scalar("abc") is None

    def test_numeric_string(self):
        result = _scalar("2.5")
        assert result == pytest.approx(2.0)


# ===========================================================================
# _format_relion_params
# ===========================================================================

def test_format_relion_params_empty():
    assert _format_relion_params({}) == ""
    assert _format_relion_params(None) == ""


def test_format_relion_params_version():
    result = _format_relion_params({"version": 3.1, "pixel_size": 2.0})
    assert "Relion 3.1" in result
    assert "pixel size" in result
    assert "2" in result


def test_format_relion_params_binning():
    result = _format_relion_params({"version": 5.0, "binning": 4})
    assert "binning: 4" in result


def test_format_relion_params_formats():
    result = _format_relion_params({
        "version": 3.1,
        "tomo_format": ".mrc",
        "subtomo_format": ".hdf",
    })
    assert ".mrc" in result
    assert ".hdf" in result


# ===========================================================================
# get_class_by_name
# ===========================================================================

def test_get_class_by_name_found():
    from cryocat.analysis.tango import TwistDescriptor
    cls = get_class_by_name("TwistDescriptor")
    assert cls is TwistDescriptor


def test_get_class_by_name_not_found():
    with pytest.raises(ValueError, match="not found"):
        get_class_by_name("NonExistentClass")


# ===========================================================================
# make_axis_trace
# ===========================================================================

def test_make_axis_trace_adds_three_traces():
    import plotly.graph_objects as go
    fig = go.Figure()
    initial = len(fig.data)
    make_axis_trace(fig)
    assert len(fig.data) == initial + 3


def test_make_axis_trace_custom_colors():
    import plotly.graph_objects as go
    fig = go.Figure()
    colors = ["red", "green", "blue"]
    make_axis_trace(fig, colors=colors)
    for trace, color in zip(fig.data, colors):
        assert trace.line.color == color


# ===========================================================================
# run_operation_to_pool — shared fixtures
# ===========================================================================

@pytest.fixture(autouse=True)
def _clean(tmp_path):
    session.close_session()
    prov.clear()
    session.start_session(log_dir=tmp_path)
    yield
    session.close_session()
    prov.clear()


@pytest.fixture
def empty_state():
    return PoolState.from_stores({}, {}, {}, {}, 0)


class FakeMotl:
    def __init__(self, n: int = 5):
        self.df = pd.DataFrame({"x": range(n)})

    def get_unique_values(self, col: str):
        return self.df[col].unique()

    def shrink(self, *, by: int = 1) -> "FakeMotl":
        return FakeMotl(max(0, len(self.df) - by))


def _make_motl(n: int = 5) -> FakeMotl:
    return FakeMotl(n)


def _load(*, n: int = 5) -> FakeMotl:
    return FakeMotl(n)


def _fail(**_kw):
    raise ValueError("simulated failure")


def _call_events():
    return [e for e in session.events() if e.get("kind") == "call"]


# ===========================================================================
# Success path
# ===========================================================================

class TestSuccess:
    def test_writes_exactly_one_call_event(self, empty_state):
        before = len(_call_events())
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert len(_call_events()) - before == 1

    def test_call_event_status_ok(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert ev["status"] == "ok"

    def test_creates_one_pool_entry(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert motl_id in new_state.registry
        assert len(new_state.registry) == 1

    def test_pool_entry_has_correct_n_rows(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 7}, empty_state)
        assert new_state.registry[motl_id]["n_rows"] == 7

    def test_assign_to_in_event(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert ev.get("assign_to") == "motl_1"

    def test_motl_id_is_motl_zero(self, empty_state):
        _, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert motl_id == "motl_1"

    def test_result_stamped_with_pool_motl_id(self, empty_state):
        _, motl_id, result = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert getattr(result, "_pool_motl_id", None) == motl_id

    def test_provenance_recorded(self, empty_state):
        _, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        assert prov.var_for(motl_id) == prov.bind(motl_id)

    def test_provenance_seq_matches_last_event(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert prov.producer("motl_1") == ev["seq"]

    def test_pool_rows_match_result(self, empty_state):
        from cryocat.app.pool import get_rows
        new_state, motl_id, result = run_operation_to_pool(_load, {"n": 4}, empty_state)
        stored_df = get_rows(motl_id)
        assert len(stored_df) == len(result.df)

    def test_next_id_incremented(self, empty_state):
        new_state, _, _ = run_operation_to_pool(_load, {"n": 3}, empty_state)
        assert new_state.next_id == 1

    def test_custom_label(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 3}, empty_state, label="My Motl")
        assert new_state.registry[motl_id]["label"] == "My Motl"


# ===========================================================================
# Error path
# ===========================================================================

class TestError:
    def test_writes_exactly_one_call_event(self, empty_state):
        before = len(_call_events())
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        assert len(_call_events()) - before == 1

    def test_call_event_status_error(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        ev = _call_events()[-1]
        assert ev["status"] == "error"

    def test_pool_unchanged_on_error(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        assert len(empty_state.registry) == 0
        assert empty_state.next_id == 0

    def test_no_new_pool_entry_on_error(self, empty_state):
        try:
            run_operation_to_pool(_fail, {}, empty_state)
        except ValueError:
            pass
        assert len(empty_state.registry) == 0

    def test_provenance_not_recorded_on_error(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        assert prov.producer("motl_1") is None

    def test_error_event_has_assign_to(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        ev = _call_events()[-1]
        assert ev.get("assign_to") == "motl_1"

    def test_error_event_has_error_dict(self, empty_state):
        with pytest.raises(ValueError):
            run_operation_to_pool(_fail, {}, empty_state)
        ev = _call_events()[-1]
        assert "error" in ev
        assert ev["error"]["type"] == "ValueError"

    def test_exception_is_reraised(self, empty_state):
        with pytest.raises(ValueError, match="simulated failure"):
            run_operation_to_pool(_fail, {}, empty_state)


# ===========================================================================
# replaces= semantics
# ===========================================================================

class TestReplaces:
    def _setup(self, empty_state):
        new_state, motl_id, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        return new_state, motl_id

    def test_replaces_keeps_same_motl_id(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid
        prov.record(mid, 0)

        new_state2, returned_id, _ = run_operation_to_pool(
            motl.shrink, {"by": 2}, state1, replaces=mid
        )
        assert returned_id == mid

    def test_replaces_does_not_create_new_entry(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid

        new_state2, _, _ = run_operation_to_pool(
            motl.shrink, {"by": 2}, state1, replaces=mid
        )
        assert len(new_state2.registry) == 1
        assert mid in new_state2.registry

    def test_replaces_updates_n_rows(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid

        new_state2, _, _ = run_operation_to_pool(
            motl.shrink, {"by": 2}, state1, replaces=mid
        )
        assert new_state2.registry[mid]["n_rows"] == 3

    def test_replaces_next_id_unchanged(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)

        new_state2, _, _ = run_operation_to_pool(
            motl.shrink, {"by": 1}, state1, replaces=mid
        )
        assert new_state2.next_id == state1.next_id

    def test_replaces_assign_to_uses_same_var(self, empty_state):
        state1, mid = self._setup(empty_state)
        motl = _make_motl(5)

        run_operation_to_pool(motl.shrink, {"by": 1}, state1, replaces=mid)
        ev = _call_events()[-1]
        assert ev.get("assign_to") == prov.bind(mid)

    def test_replaces_updates_provenance_seq(self, empty_state):
        state1, mid = self._setup(empty_state)
        seq_after_first = prov.producer(mid)
        motl = _make_motl(5)

        run_operation_to_pool(motl.shrink, {"by": 1}, state1, replaces=mid)
        seq_after_second = prov.producer(mid)
        assert seq_after_second > seq_after_first


# ===========================================================================
# Receiver resolution
# ===========================================================================

class TestReceiverResolution:
    def test_receiver_field_set_when_pool_motl_id_stamped(self, empty_state):
        state1, mid, _ = run_operation_to_pool(_load, {"n": 5}, empty_state)
        motl = _make_motl(5)
        motl._pool_motl_id = mid

        run_operation_to_pool(motl.shrink, {"by": 1}, state1, replaces=mid)
        ev = _call_events()[-1]
        assert ev.get("receiver") == prov.bind(mid)

    def test_receiver_none_for_free_function(self, empty_state):
        run_operation_to_pool(_load, {"n": 5}, empty_state)
        ev = _call_events()[-1]
        assert ev.get("receiver") is None


# ===========================================================================
# Provenance per operation kind: record content and replay
# ===========================================================================
#
# Each test runs one operation the way the page commit callback runs it,
# reads the call events back from the session stream, renders them with the
# script projection and executes that script in a fresh namespace.

_TEST_DATA = Path(__file__).resolve().parents[1] / "test_data"


def _replay_script() -> tuple[str, dict, object]:
    """Render the session stream to a script and execute it.

    Returns ``(script, namespace, last_value)`` where *last_value* is the value
    of the final statement when it is a bare expression (a call whose result
    the record does not bind to a name), else ``None``.
    """
    import ast
    from cryocat.app.record.script import render_script

    script = render_script(session.events())
    tree = ast.parse(script)
    ns: dict = {}
    last_value = None
    body = tree.body
    if body and isinstance(body[-1], ast.Expr):
        exec(compile(ast.Module(body=body[:-1], type_ignores=[]), "<replay>", "exec"), ns)
        last_value = eval(compile(ast.Expression(body=body[-1].value), "<replay>", "eval"), ns)
    else:
        exec(compile(tree, "<replay>", "exec"), ns)
    return script, ns, last_value


def _load_motl_to_pool(path: Path, state):
    """Load a motl file into the pool the way the motl sidebar load callback does."""
    from cryocat.core.cryomotl import Motl
    from cryocat.app.apputils import record_load_to_pool

    motl_data = Motl.load(str(path), "emmotl").df.to_dict("records")
    return record_load_to_pool(motl_data, "emmotl", str(path), None, state, label=path.name)


class TestMotlOperationProvenance:
    """In-place motl operation from the motl sidebar: load, then apply on the active slot."""

    def _run(self, empty_state):
        from cryocat.core.cryomotl import Motl
        from cryocat.app.logger import invoke_operation
        from cryocat.app.pool import get_rows, replace_motl_rows

        state, mid, _ = _load_motl_to_pool(_TEST_DATA / "au_1.em", empty_state)
        motl = Motl(get_rows(mid))
        motl._pool_motl_id = mid
        if prov.var_for(mid) is None:
            prov.record(mid, session.last_seq())
        result = invoke_operation(getattr(motl, "scale_coordinates"), {"scaling_factor": 2.0})
        result_df = motl.df if result is None else result.df
        state = replace_motl_rows(state, mid, result_df)
        return state, mid

    def test_load_record_names_function_inputs_and_output(self, empty_state):
        path = _TEST_DATA / "au_1.em"
        state, mid = self._run(empty_state)
        load_ev = _call_events()[0]
        assert load_ev["status"] == "ok"
        assert load_ev["fn"] == "cryocat.core.cryomotl.Motl.load"
        assert load_ev["kwargs_src"] == {"input_motl": repr(str(path)), "motl_type": repr("emmotl")}
        assert load_ev["assign_to"] == prov.bind(mid)
        assert load_ev["result"]["pool_id"] == mid
        assert load_ev["result"]["n_rows"] == state.registry[mid]["n_rows"]
        assert prov.producer(mid) == load_ev["seq"]

    def test_operation_record_names_function_receiver_and_arguments(self, empty_state):
        _, mid = self._run(empty_state)
        op_ev = _call_events()[-1]
        assert op_ev["status"] == "ok"
        assert op_ev["fn"] == "cryocat.core.cryomotl.Motl.scale_coordinates"
        assert op_ev["receiver"] == prov.bind(mid)
        assert op_ev["kwargs_src"] == {"scaling_factor": "2.0"}

    def test_replay_reproduces_pool_rows(self, empty_state):
        from cryocat.app.pool import get_rows

        _, mid = self._run(empty_state)
        _, ns, _ = _replay_script()
        replayed = ns[prov.bind(mid)]
        pd.testing.assert_frame_equal(
            replayed.df.reset_index(drop=True),
            get_rows(mid).reset_index(drop=True),
            check_dtype=False,
        )


class TestMapOperationProvenance:
    """Create-mask commit on the volume page: build a sphere layer, write it as a map."""

    _SHAPE = (16, 16, 16)

    def _run(self, out_path: Path) -> None:
        from cryocat.core import cryomap, cryomask
        from cryocat.app.apputils import run_operation

        mask = cryomask.spherical_mask(
            mask_size=self._SHAPE, radius=5, gaussian=0.0, gaussian_outwards=True,
        )
        run_operation(cryomap.write, {
            "data_to_write": mask.astype(np.float32),
            "output_path": str(out_path),
            "pixel_size": 1.0,
        })

    def test_record_names_function_arguments_and_output(self, tmp_path):
        out = tmp_path / "mask.em"
        self._run(out)
        ev = _call_events()[-1]
        assert ev["status"] == "ok"
        assert ev["fn"] == "cryocat.core.cryomap.write"
        assert ev["kwargs_src"]["output_path"] == repr(str(out))
        assert ev["kwargs_src"]["pixel_size"] == "1.0"

    def test_record_names_the_map_input(self, tmp_path):
        self._run(tmp_path / "mask.em")
        ev = _call_events()[-1]
        src = ev["kwargs_src"]["data_to_write"]
        assert src is not None and not src.startswith("..."), (
            f"data_to_write is recorded as {src!r}, not as a replayable expression"
        )

    def test_replay_reproduces_written_map(self, tmp_path):
        from cryocat.core import cryomap

        out = tmp_path / "mask.em"
        self._run(out)
        gui_map = np.array(cryomap.read(str(out)))
        out.unlink()
        _replay_script()
        assert out.exists()
        np.testing.assert_array_equal(np.array(cryomap.read(str(out))), gui_map)


def _ellipsoid_motl_file(path: Path) -> None:
    """Write a small motl with particles on one ellipsoid (one object, one tomogram)."""
    from cryocat.core.cryomotl import Motl

    rng = np.random.default_rng(0)
    n = 60
    u = rng.uniform(0, np.pi, n)
    v = rng.uniform(0, 2 * np.pi, n)
    data = {c: np.zeros(n) for c in Motl.motl_columns}
    data["subtomo_id"] = np.arange(1, n + 1, dtype=float)
    data["tomo_id"] = np.ones(n)
    data["object_id"] = np.ones(n)
    data["class"] = np.ones(n)
    data["x"] = 50 + 30.0 * np.sin(u) * np.cos(v)
    data["y"] = 50 + 20.0 * np.sin(u) * np.sin(v)
    data["z"] = 50 + 15.0 * np.cos(u)
    Motl(pd.DataFrame(data)).write_out(str(path))


class TestAssemblyOperationProvenance:
    """Parametric assembly on the assembly page: fit from a pool motl, then distance to surface."""

    @pytest.fixture(autouse=True)
    def _clear_fit(self):
        from cryocat.app.components import parametric_registry as pr

        yield
        pr.registry.clear()

    def _run(self, empty_state, tmp_path: Path):
        from cryocat.analysis.assembly import ParametricAssembly
        from cryocat.core.cryomotl import Motl
        from cryocat.app.apputils import run_operation
        from cryocat.app.logger import invoke_operation
        from cryocat.app.pool import get_rows
        from cryocat.app.components import parametric_registry as pr
        from cryocat.app.components import surface_registry as sr

        path = tmp_path / "ellipsoid.em"
        _ellipsoid_motl_file(path)
        _, mid, _ = _load_motl_to_pool(path, empty_state)

        # Loader: the pool motl is injected under the loader's motl kwarg.
        motl = Motl(get_rows(mid))
        motl._pool_motl_id = mid
        load_var = "_" + prov.bind(sr.registry.peek_next_key())
        fit = invoke_operation(ParametricAssembly.from_motl, {"input_motl": motl}, assign_to=load_var)
        pr.registry.add(fit)
        pr.make_handle(fit, source=f"motl:{mid}")

        # Operation on the active fit with an input motl picked from the pool.
        in_motl = Motl(get_rows(mid))
        in_motl._pool_motl_id = mid
        psurf = pr.registry.get(pr.registry.keys()[0])
        result = run_operation(
            getattr(psurf, "compute_point_surface_distance"), {"input_motl": in_motl},
        )
        return mid, load_var, result

    def test_fit_record_names_function_input_and_output(self, empty_state, tmp_path):
        mid, load_var, _ = self._run(empty_state, tmp_path)
        fit_ev = _call_events()[-2]
        assert fit_ev["status"] == "ok"
        assert fit_ev["fn"] == "cryocat.analysis.assembly.ParametricAssembly.from_motl"
        assert fit_ev["kwargs_src"] == {"input_motl": prov.bind(mid)}
        assert fit_ev["assign_to"] == load_var
        assert fit_ev["result"]["type"] == "ParametricAssembly"

    def test_operation_record_names_function_receiver_input_and_output(self, empty_state, tmp_path):
        mid, load_var, result = self._run(empty_state, tmp_path)
        op_ev = _call_events()[-1]
        assert op_ev["status"] == "ok"
        assert op_ev["fn"] == (
            "cryocat.analysis.assembly.ParametricAssembly.compute_point_surface_distance"
        )
        assert op_ev["receiver"] == load_var
        assert op_ev["kwargs_src"] == {"input_motl": prov.bind(mid)}
        assert op_ev["result"]["n_rows"] == len(result.df)

    def test_replay_reproduces_operation_result(self, empty_state, tmp_path):
        _, _, result = self._run(empty_state, tmp_path)
        _, _, replayed = _replay_script()
        assert replayed is not None, "the operation result is not bound in the replayed script"
        pd.testing.assert_frame_equal(
            replayed.df.reset_index(drop=True),
            result.df.reset_index(drop=True),
            check_dtype=False,
        )
