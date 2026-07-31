import inspect
import time as _time
import traceback as _tb
import datetime as _dt

class StreamToList:
    def __init__(self):
        self.buffer = []        # list of (msg, source)

    def write(self, msg, source="dash"):
        if msg.strip():
            self.buffer.append((msg.strip(), source))

    def flush(self):
        pass

    def get_logs(self, last_index=0):
        """Return new log entries since last_index plus summary flags.

        Returns
        -------
        entries : list of (msg, source) tuples
        new_index : int
        new_dash : bool  — any "dash" source in the new entries
        new_error : bool — any "error" source in the new entries
        """
        sliced = self.buffer[last_index:]
        new_dash = any(s == "dash" for _, s in sliced)
        new_error = any(s == "error" for _, s in sliced)
        return list(sliced), len(self.buffer), new_dash, new_error

    def get_all_logs(self):
        return "\n".join(
            msg.decode("utf-8") if isinstance(msg, bytes) else msg
            for msg, _ in self.buffer
        )

    def clear(self):
        self.buffer.clear()



def print_dash(*args):
    dash_logger.write(" ".join(map(str, args)), source="dash")


# Global singleton instance
dash_logger = StreamToList()


# ── Compact argument formatter (pane display) ─────────────────────────────────
# format_arg is for one-line call labels only. Do not call it from other modules.

def format_arg(value):
    """Compact repr for a single argument — for one-line call labels only."""
    if hasattr(value, "df") and hasattr(value, "get_unique_values"):
        try:
            return f"<Motl({len(value.df)} rows)>"
        except Exception:
            return "<Motl>"
    if isinstance(value, (list, dict)) and len(value) > 5:
        return f"<{type(value).__name__}[{len(value)}]>"
    try:
        r = repr(value)
    except Exception:
        r = f"<{type(value).__name__}>"
    return r if len(r) <= 100 else r[:97] + "..."


def _render_call(fn, kwargs: dict) -> str:
    """Readable pane label — uses ``format_arg``, named kwargs, not truncated."""
    parts = ", ".join(f"{k}={format_arg(v)}" for k, v in kwargs.items())
    return f"{fn.__qualname__}({parts})"


# ── Script-line renderer ──────────────────────────────────────────────────────

def _module_short(fn) -> tuple[str, str, str]:
    """Return ``(full_module, short_name, import_statement)`` for *fn*."""
    mod = fn.__module__
    parts = mod.rsplit(".", 1)
    if len(parts) == 2:
        pkg, short = parts
        return mod, short, f"from {pkg} import {short}"
    return mod, mod, f"import {mod}"


def _render_value(
    v, *, obj_to_var: dict[int, str] | None = None
) -> tuple[str | None, list]:
    """Render a kwarg value as a Python expression.

    Returns ``(expr, imports)`` where *expr* is a Python source string, or
    ``None`` for an unresolvable Motl argument (caller emits a message event).

    *obj_to_var* is a short-lived ``{id(obj): var_name}`` map built for a
    single ``invoke_operation`` call from the kwargs it received.  Never a
    process-lifetime table.
    """
    _ovmap = obj_to_var or {}
    # Any object with an explicit variable mapping takes priority
    if _ovmap and id(v) in _ovmap:
        return _ovmap[id(v)], []
    # Primitives
    if isinstance(v, (str, int, float, bool)) or v is None:
        return repr(v), []
    # Small list/tuple of scalars — verbatim
    if isinstance(v, (list, tuple)) and len(v) <= 32 and all(
        isinstance(x, (str, int, float, bool)) for x in v
    ):
        return repr(list(v)), []
    # Motl-like: resolve via the per-call variable map (provenance.bind)
    if hasattr(v, "df") and hasattr(v, "get_unique_values"):
        var = _ovmap.get(id(v))
        if var:
            return var, [("cryomotl", "from cryocat.core import cryomotl")]
        return None, []  # unresolvable — caller emits message event
    # ndarray / large array → placeholder
    if hasattr(v, "shape"):
        shape = tuple(getattr(v, "shape", ()))
        return f"None  # <array shape {shape}: supply input>", []
    # dict / other → compact repr fallback
    try:
        r = repr(v)
    except Exception:
        r = f"<{type(v).__name__}>"
    if len(r) <= 120:
        return r, []
    return f"None  # {type(v).__name__}: {r[:80]}...", []


def _render_python_line(
    fn, kwargs: dict, *, obj_to_var: dict[int, str] | None = None
) -> tuple[str, list, dict[str, str | None]]:
    """Return ``(runnable_line, imports_needed, kwargs_src)`` for the given call.

    *kwargs_src* maps each kwarg name to its rendered Python source string, or
    ``None`` for an unresolvable Motl argument.  ``None`` entries render as
    the Python literal ``None`` in the script line so it remains syntactically
    valid.
    """
    imports: list = []
    args: list[str] = []
    kwargs_src: dict[str, str | None] = {}

    for k, v in kwargs.items():
        expr, more = _render_value(v, obj_to_var=obj_to_var)
        imports.extend(more)
        args.append(f"{k}=None" if expr is None else f"{k}={expr}")
        kwargs_src[k] = expr

    arg_str = ", ".join(args)

    if hasattr(fn, "__self__") and fn.__self__ is not None:
        if inspect.isclass(fn.__self__):
            mod, short, stmt = _module_short(fn)
            imports = [(short, stmt)] + imports
            return f"{short}.{fn.__qualname__}({arg_str})", imports, kwargs_src
        recv_expr, recv_imps = _render_value(fn.__self__, obj_to_var=obj_to_var)
        recv_src = recv_expr or "None"
        imports = recv_imps + imports
        ctor_kwargs = getattr(fn.__self__, "_ctor_kwargs", None)
        if ctor_kwargs:
            cls_name = type(fn.__self__).__name__
            data_src = f"{recv_src}.df" if recv_src != "None" else "None  # supply data"
            ctor_parts = [data_src] + [f"{k}={v!r}" for k, v in ctor_kwargs.items()]
            cryomotl_imp = ("cryomotl", "from cryocat.core import cryomotl")
            if cryomotl_imp not in imports:
                imports = [cryomotl_imp] + imports
            return (
                f"cryomotl.{cls_name}({', '.join(ctor_parts)}).{fn.__name__}({arg_str})",
                imports,
                kwargs_src,
            )
        return f"{recv_src}.{fn.__name__}({arg_str})", imports, kwargs_src

    mod, short, stmt = _module_short(fn)
    imports = [(short, stmt)] + imports
    line = f"{short}.{fn.__qualname__}({arg_str})"
    return line, imports, kwargs_src


# ── Dispatch wrapper ──────────────────────────────────────────────────────────

def invoke_operation(
    fn,
    kwargs: dict,
    *,
    assign_to: str | None = None,
    pool_id: str | None = None,
    label: str | None = None,
    source: str | None = None,
):
    """Invoke a ``@gui_exposed`` function, logging to the pane, script, and stream.

    Parameters
    ----------
    fn:
        The callable to invoke (bound method or free function).
    kwargs:
        The keyword arguments to pass to *fn*.
    assign_to:
        Script variable the result will be bound to (e.g. ``"motl_1"``).
        When ``None``, the call event carries no ``assign_to``.  Set by
        :func:`~cryocat.app.apputils.run_operation_to_pool`.
    pool_id:
        Pool id of the entry this result will occupy (e.g. ``"motl-1"``).
        Folded into the result summary so the record shows pool identity.
    label:
        Human-readable label of the pool entry (folded into result summary).
    source:
        Input file path for load operations (folded into result summary).

    * Success → pane gets ``▶``/``✓``; script gets a runnable line;
      stream gets a ``call`` event with ``status="ok"``.
    * Failure → pane gets ``✗`` + traceback (``source="error"``); script gets a
      comment; stream gets a ``call`` event with ``status="error"``;
      exception is re-raised so the callback can surface a message.
    """
    from cryocat.app import session as _session
    from cryocat.app import provenance as _prov
    from cryocat.app.event import call_event, describe, message_event, validate_result

    pane_name = _render_call(fn, kwargs)
    dash_logger.write(f"▶ {pane_name}", source="cryocat")

    # Build a short-lived {id(obj): var_name} map so that Motl arguments and
    # the bound receiver can be rendered as their pool variable names.
    obj_to_var: dict[int, str] = {}

    # Check the receiver (fn.__self__ for bound instance methods).
    self_obj = getattr(fn, "__self__", None)
    if self_obj is not None and not inspect.isclass(self_obj):
        mid = getattr(self_obj, "_pool_motl_id", None)
        if mid is not None:
            var = _prov.var_for(mid)
            if var:
                obj_to_var[id(self_obj)] = var

    # Check kwargs for Motl arguments stamped with _pool_motl_id.
    for v in kwargs.values():
        mid = getattr(v, "_pool_motl_id", None)
        if mid is not None:
            var = _prov.var_for(mid)
            if var:
                obj_to_var[id(v)] = var

    # Derive the receiver variable name for the call event.
    derived_receiver: str | None = (
        obj_to_var.get(id(self_obj))
        if self_obj is not None and not inspect.isclass(self_obj)
        else None
    )

    # Pre-compute script rendering (used in both success and error paths).
    fn_name = f"{fn.__module__}.{fn.__qualname__}"
    line, imports_needed, kwargs_src = _render_python_line(fn, kwargs, obj_to_var=obj_to_var)
    imports_json = [[s, stmt] for s, stmt in imports_needed]

    # Warn once per unresolvable Motl argument.
    for k, expr in kwargs_src.items():
        if expr is None:
            _session.emit(message_event(
                f"Argument {k!r} of {fn_name} could not be resolved to a variable "
                "— pool provenance missing. Load the motl through the pool first.",
                level="error",
            ))

    # Capture receiver's row count before the call for delta computation.
    before: dict | None = None
    if self_obj is not None and not inspect.isclass(self_obj) and hasattr(self_obj, "df"):
        try:
            before = {"n_rows": int(len(self_obj.df))}
        except Exception:
            pass

    t0 = _time.monotonic()
    try:
        result = fn(**kwargs)
    except Exception as exc:
        duration = _time.monotonic() - t0
        tb_str = _tb.format_exc()
        dash_logger.write(f"✗ {pane_name} — {type(exc).__name__}: {exc}", source="error")
        dash_logger.write(tb_str, source="error")
        _session.emit(call_event(
            fn_name, kwargs_src,
            status="error",
            imports=imports_json,
            receiver=derived_receiver,
            assign_to=assign_to,
            duration_s=duration,
            error={
                "type": type(exc).__name__,
                "msg": str(exc),
                "traceback": tb_str,
            },
        ))
        raise

    duration = _time.monotonic() - t0
    dash_logger.write(f"✓ {pane_name}", source="cryocat")

    result_summary = describe(result, pool_id=pool_id, label=label, source=source, before=before)
    try:
        validate_result(result_summary)
    except ValueError:
        result_summary = {"type": type(result).__name__}
    _session.emit(call_event(
        fn_name, kwargs_src,
        status="ok",
        imports=imports_json,
        receiver=derived_receiver,
        assign_to=assign_to,
        duration_s=duration,
        result=result_summary,
    ))
    return result

