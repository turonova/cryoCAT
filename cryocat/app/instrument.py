"""Temporary diagnostic: count and time every callback, per instance."""

import atexit, collections, datetime, functools, time
import dash
from dash import ctx

COUNTS = collections.Counter()
TIMES = collections.Counter()
WRAPPED = []
TRACE: list = []        # (timestamp_float, event, key) when _tracing is True
_tracing: bool = False
_snap_t0: float = 0.0


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


def _wrap(fn):
    name = f"{fn.__module__}.{fn.__name__}"
    WRAPPED.append(name)

    @functools.wraps(fn)
    def inner(*args, **kwargs):
        t0 = time.perf_counter()
        if _tracing:
            TRACE.append((time.time(), "ENTER", _instance_key(name)))
        try:
            return fn(*args, **kwargs)
        finally:
            key = _instance_key(name)
            COUNTS[key] += 1
            TIMES[key] += time.perf_counter() - t0
            if _tracing:
                TRACE.append((time.time(), "EXIT ", key))

    return inner


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
    print(f"\n=== callback report ({len(WRAPPED)} wrapped) ===")
    for key, ms in TIMES.most_common(40):
        print(f"{COUNTS[key]:7d} calls {ms * 1000:10.1f} ms total  {key}")
