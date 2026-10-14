"""Temporary diagnostic: count and time every callback, per instance.

Set the environment variable ``CRYOCAT_TRACE=1`` to enable continuous
ENTER/EXIT tracing from the very first callback, regardless of whether the
app is running with ``debug=True`` or ``debug=False``.  When the variable is
not set the trace is controlled programmatically via :func:`start_trace` and
:func:`reset` (the existing per-operation windows in motlio / ptango).
"""

import atexit, collections, datetime, functools, os, time
import dash
from dash import ctx

COUNTS = collections.Counter()
TIMES = collections.Counter()
WRAPPED = []
TRACE: list = []        # (timestamp_float, event, key) when _tracing is True
_tracing: bool = False
_snap_t0: float = 0.0

# True when CRYOCAT_TRACE is set in the environment.  Per-operation callbacks
# (load_motl, NN compute) check this flag and skip their reset()/start_trace()
# so they do not wipe the continuous trace mid-run.
env_tracing: bool = bool(os.environ.get("CRYOCAT_TRACE"))


def reset() -> None:
    """Zero COUNTS, TIMES, TRACE and record the snapshot start time."""
    global _snap_t0, _tracing
    COUNTS.clear()
    TIMES.clear()
    TRACE.clear()
    _tracing = False
    _snap_t0 = time.perf_counter()


def start_trace() -> None:
    """Enable entry/exit wall-clock tracing from this point.  reset() stops it."""
    global _tracing
    TRACE.clear()
    _tracing = True


# Activate immediately when CRYOCAT_TRACE is in the environment so the trace
# covers every callback from the first request, not just after a user action.
if env_tracing:
    start_trace()


def _ts(t: float) -> str:
    """Format a time.time() float as HH:MM:SS.mmm."""
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.") + f"{datetime.datetime.fromtimestamp(t).microsecond // 1000:03d}"


def print_trace(label: str = "") -> None:
    """Print the wall-clock entry/exit log recorded since start_trace()."""
    if not TRACE:
        return
    tag = f" [{label}]" if label else ""
    print(f"\n=== trace{tag} ===")
    for ts, event, key in TRACE:
        # Strip module prefix: keep only last two segments for readability.
        parts = key.split(".")
        short = ".".join(parts[-2:]) if len(parts) >= 2 else key
        print(f"  {_ts(ts)}  {event:5s}  {short}")


def snapshot(label: str = "") -> None:
    """Print the trace (if any), then per-callback counts/times, then reset.

    Rows are sorted by total time descending.  Wall time is elapsed since reset().
    """
    print_trace(label)
    elapsed = time.perf_counter() - _snap_t0
    tag = f" [{label}]" if label else ""
    print(f"\n=== snapshot{tag}  wall {elapsed * 1000:.0f} ms ===")
    items = [(k, TIMES[k]) for k in COUNTS if TIMES[k] > 0]
    for key, ms in sorted(items, key=lambda kv: -kv[1]):
        print(f"  {COUNTS[key]:4d}x  {ms * 1000:8.1f} ms  {key}")
    reset()


def _instance_key(default):
    """Append the callback's first Output id so instances are distinguishable."""
    try:
        outs = ctx.outputs_list
    except Exception:
        return default
    ids = []

    def walk(o):
        if isinstance(o, list):
            for x in o:
                walk(x)
        elif isinstance(o, dict) and "id" in o:
            ids.append(o["id"])

    walk(outs)
    if not ids:
        return default
    first = ids[0]
    if isinstance(first, dict):
        first = ",".join(f"{k}={v}" for k, v in sorted(first.items()))
    return f"{default}  ->  {first}"


def _emit(event: str, key: str, ts: float) -> None:
    """Write one ENTER/EXIT line to stdout immediately (env_tracing mode)."""
    parts = key.split(".")
    short = ".".join(parts[-2:]) if len(parts) >= 2 else key
    print(f"  {_ts(ts)}  {event:5s}  {short}", flush=True)


def _set_flask_g(attr: str, value) -> None:
    """Write *value* onto Flask's per-request ``g`` object under *attr*.

    Silently no-ops when called outside a Flask request context (tests,
    background threads, startup).
    """
    try:
        from flask import g as _g, has_request_context as _hrc
        if _hrc():
            setattr(_g, attr, value)
    except Exception:
        pass


def _wrap(fn):
    name = f"{fn.__module__}.{fn.__name__}"
    WRAPPED.append(name)

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        t0 = time.perf_counter()
        _set_flask_g("_t_fn_start", t0)   # arrival → this = Dash dispatch time
        if _tracing:
            ts = time.time()
            key_enter = _instance_key(name)
            TRACE.append((ts, "ENTER", key_enter))
            if env_tracing:
                _emit("ENTER", key_enter, ts)
        try:
            return fn(*args, **kwargs)
        finally:
            t1 = time.perf_counter()
            key = _instance_key(name)
            _set_flask_g("_t_fn_end", t1)  # this → after_request = serialisation time
            _set_flask_g("_cb_name",  key)
            COUNTS[key] += 1
            TIMES[key] += t1 - t0
            if _tracing:
                ts = time.time()
                TRACE.append((ts, "EXIT ", key))
                if env_tracing:
                    _emit("EXIT ", key, ts)

    return inner


def hook_flask(server) -> None:
    """Register before/after request hooks for per-request pipeline timing.

    Each ``/_dash-update-component`` request is decomposed into four wall-clock
    durations and printed to stdout whenever :data:`_tracing` is active:

    ======  ============================================================
    gap     previous response-sent → this arrival  (client + queueing)
    disp    arrival → callback-start               (Dash routing/JSON)
    fn      callback-start → callback-return       (your code)
    ser     callback-return → response-sent        (serialisation)
    ======  ============================================================

    Call once after :func:`instrument`::

        from cryocat.app.instrument import instrument, hook_flask
        instrument(app)
        hook_flask(app.server)
    """
    from flask import g as _fg, request as _freq

    _last_sent: list[float | None] = [None]  # mutable cell — survives closures

    @server.before_request
    def _t_arrive() -> None:
        if _freq.path != "/_dash-update-component":
            return
        now = time.perf_counter()
        _fg._t_arrival = now
        _fg._t_gap = (
            (now - _last_sent[0]) * 1000.0 if _last_sent[0] is not None else None
        )

    @server.after_request
    def _t_send(response):
        if _freq.path != "/_dash-update-component":
            return response
        if not hasattr(_fg, "_t_arrival"):
            return response

        t_sent = time.perf_counter()
        _last_sent[0] = t_sent   # always update so gap is correct next request

        if not _tracing:
            return response

        t_arr  = _fg._t_arrival
        t_fn0  = getattr(_fg, "_t_fn_start", None)
        t_fn1  = getattr(_fg, "_t_fn_end",   None)
        gap    = getattr(_fg, "_t_gap",       None)
        label  = getattr(_fg, "_cb_name",     "?")

        disp_ms = (t_fn0 - t_arr)  * 1000.0 if t_fn0 is not None else float("nan")
        fn_ms   = (t_fn1 - t_fn0)  * 1000.0 if (t_fn0 is not None and t_fn1 is not None) else float("nan")
        ser_ms  = (t_sent - (t_fn1 if t_fn1 is not None else t_arr)) * 1000.0
        gap_s   = f"{gap:7.1f}" if gap is not None else "    ---"

        # Shorten label: keep last two module segments + first Output id
        parts = label.split("  ->  ", 1)
        short_fn  = ".".join(parts[0].split(".")[-2:]) if parts[0] else "?"
        short_out = parts[1][:36] if len(parts) > 1 else ""
        short = f"{short_fn}  ->  {short_out}" if short_out else short_fn

        print(
            f"  REQ  {short:<60s}"
            f"  gap={gap_s} ms"
            f"  disp={disp_ms:6.1f} ms"
            f"  fn={fn_ms:6.1f} ms"
            f"  ser={ser_ms:6.1f} ms",
            flush=True,
        )
        return response


def instrument(app=None):
    """Patch both decorator styles. The `app` argument is accepted and unused."""
    orig_method = dash.Dash.callback

    def method(self, *a, **kw):
        deco = orig_method(self, *a, **kw)
        return lambda fn: deco(_wrap(fn))

    dash.Dash.callback = method

    orig_func = dash.callback

    def func(*a, **kw):
        deco = orig_func(*a, **kw)
        return lambda fn: deco(_wrap(fn))

    dash.callback = func


@atexit.register
def report():
    if env_tracing:
        print_trace("env-final")
    print(f"\n=== callback report ({len(WRAPPED)} wrapped) ===")
    for key, ms in TIMES.most_common(40):
        print(f"{COUNTS[key]:7d} calls {ms * 1000:10.1f} ms total  {key}")
