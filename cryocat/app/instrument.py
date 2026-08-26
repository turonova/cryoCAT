"""Temporary diagnostic: count and time every callback, per instance."""

import atexit, collections, functools, time
import dash
from dash import ctx

COUNTS = collections.Counter()
TIMES = collections.Counter()
WRAPPED = []


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
        try:
            return fn(*args, **kwargs)
        finally:
            key = _instance_key(name)
            COUNTS[key] += 1
            TIMES[key] += time.perf_counter() - t0

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
