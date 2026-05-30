import inspect
import functools
import threading
import traceback as _tb
import datetime as _dt
import os

_reentry = threading.local()


_MAX_MOTL_DEPTH = 3  # max nesting depth for inline motl expression rendering


class StreamToList:
    def __init__(self):
        self.buffer = []        # list of (msg, source)
        self.log_path = None    # path to the session script
        self._imports = set()   # module short-names already written to the file
        self._log_fh = None     # open append handle for the script file
        self._motl_sources: dict = {}   # id(motl_obj) -> Python expression string
        self.last_script_line: str | None = None  # last line written by invoke_operation

    def record_motl_source(self, motl_obj, expr: str) -> None:
        """Associate a Python expression with a Motl object for script rendering."""
        self._motl_sources[id(motl_obj)] = expr

    def get_motl_source(self, motl_obj) -> "str | None":
        """Retrieve the Python expression recorded for a Motl object, or None."""
        return self._motl_sources.get(id(motl_obj))

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
        # Does NOT delete the session script — that's a persistent artifact.

    # ── Session script management ─────────────────────────────────────────────

    def start_session(self, log_dir: str) -> None:
        """Open a per-session Python script at ``log_dir/cryocat_session_<ts>.py``."""
        os.makedirs(log_dir, exist_ok=True)
        ts = _dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.log_path = os.path.join(log_dir, f"cryocat_session_{ts}.py")
        self._imports = set()
        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        self._log_fh.write(f"# cryocat session {ts}\n\n")
        self._log_fh.flush()

    def append_script_line(self, line: str, imports_needed=()) -> None:
        """Append *line* to the session script, emitting missing imports first."""
        if self._log_fh is None:
            return
        for short, statement in imports_needed:
            if short not in self._imports:
                self._log_fh.write(statement + "\n")
                self._imports.add(short)
        self._log_fh.write(line + "\n")
        self._log_fh.flush()

    def append_script_comment(self, text: str) -> None:
        """Append a ``# …`` comment block (used for failures)."""
        if self._log_fh is None:
            return
        for ln in (text.splitlines() or [""]):
            self._log_fh.write(f"# {ln}\n")
        self._log_fh.flush()


def print_dash(*args):
    dash_logger.write(" ".join(map(str, args)), source="dash")


# Global singleton instance
dash_logger = StreamToList()


# ── Compact argument formatter (pane display) ─────────────────────────────────

def _fmt(value):
    """Compact repr for a single argument — truncates large containers."""
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
    """Readable pane label — uses ``_fmt``, named kwargs, not truncated."""
    parts = ", ".join(f"{k}={_fmt(v)}" for k, v in kwargs.items())
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


def _render_value(v, depth: int = 0) -> tuple[str, list]:
    """Render a kwarg value as a Python expression. Returns ``(expr, imports)``."""
    # Primitives
    if isinstance(v, (str, int, float, bool)) or v is None:
        return repr(v), []
    # Small numeric/string list/tuple — verbatim
    if isinstance(v, (list, tuple)) and len(v) <= 32 and all(
        isinstance(x, (str, int, float, bool)) for x in v
    ):
        return repr(list(v)), []
    # List of Motl objects -> list of inline expressions
    if isinstance(v, (list, tuple)) and v and all(
        hasattr(x, "df") and hasattr(x, "get_unique_values") for x in v
    ):
        exprs, all_imps = [], []
        for m in v:
            e, imps = _render_value(m, depth + 1)
            exprs.append(e)
            all_imps.extend(imps)
        return f"[{', '.join(exprs)}]", all_imps
    # Motl object — check the side-table first, then fall back to path / placeholder
    if hasattr(v, "df") and hasattr(v, "get_unique_values"):
        imp = ("cryomotl", "from cryocat.core import cryomotl")
        if depth < _MAX_MOTL_DEPTH:
            expr = dash_logger.get_motl_source(v)
            if expr:
                return expr, [imp]
        path = getattr(v, "path", None) or getattr(v, "input_path", None)
        if path:
            return f"cryomotl.Motl.load({path!r})", [imp]
        # Source not tracked — emit a WARN comment so the gap is visible in the script
        dash_logger.append_script_comment(
            "WARN: Motl source not tracked at this call site. "
            "A pool insertion is missing a record_motl_source() call."
        )
        return "None  # <Motl source not tracked — see WARN above>", [imp]
    # ndarray / large array -> placeholder
    if hasattr(v, "shape"):
        shape = tuple(getattr(v, "shape", ()))
        return f"None  # <array shape {shape}: supply input>", []
    # dict / other -> compact repr fallback
    try:
        r = repr(v)
    except Exception:
        r = f"<{type(v).__name__}>"
    if len(r) <= 120:
        return r, []
    return f"None  # {type(v).__name__}: {r[:80]}...", []


def _render_python_line(fn, kwargs: dict) -> tuple[str, list]:
    """Return ``(runnable_line, imports_needed)`` for the given call."""
    imports: list = []
    args: list[str] = []

    for k, v in kwargs.items():
        expr, more = _render_value(v)
        imports.extend(more)
        args.append(f"{k}={expr}")

    arg_str = ", ".join(args)

    if hasattr(fn, "__self__") and fn.__self__ is not None:
        # Classmethod bound to a class → render as module_short.ClassName.method(...)
        if inspect.isclass(fn.__self__):
            mod, short, stmt = _module_short(fn)
            imports = [(short, stmt)] + imports
            return f"{short}.{fn.__qualname__}({arg_str})", imports
        # Bound instance method → render as receiver.method(kwargs)
        recv_expr, recv_imps = _render_value(fn.__self__)
        imports = recv_imps + imports
        return f"{recv_expr}.{fn.__name__}({arg_str})", imports

    # Regular function or unbound class method
    mod, short, stmt = _module_short(fn)
    imports = [(short, stmt)] + imports
    line = f"{short}.{fn.__qualname__}({arg_str})"
    return line, imports


# ── Dispatch wrapper ──────────────────────────────────────────────────────────

def invoke_operation(fn, kwargs: dict):
    """Invoke a ``@gui_exposed`` function, logging to the pane and session script.

    * Success → pane gets ``▶``/``✓``; session script gets a runnable line.
    * Failure → pane gets ``✗`` + traceback (``source="error"``); script gets a
      comment; exception is re-raised so the callback can surface a message.
    """
    pane_name = _render_call(fn, kwargs)
    dash_logger.write(f"▶ {pane_name}", source="cryocat")
    try:
        result = fn(**kwargs)
    except Exception as exc:
        ts = _dt.datetime.now().strftime("%H:%M:%S")
        dash_logger.write(f"✗ {pane_name} — {type(exc).__name__}: {exc}", source="error")
        dash_logger.write(_tb.format_exc(), source="error")
        dash_logger.append_script_comment(
            f"✗ {ts}  {pane_name} — {type(exc).__name__}: {exc}"
        )
        raise
    line, imports_needed = _render_python_line(fn, kwargs)
    dash_logger.append_script_line(line, imports_needed=imports_needed)
    dash_logger.last_script_line = line
    # Register result Motl so nested callers can inline-expand it
    if hasattr(result, "df") and hasattr(result, "get_unique_values"):
        dash_logger.record_motl_source(result, line)
    dash_logger.write(f"✓ {pane_name}", source="cryocat")
    return result


# ── Automatic cryoCAT call logging (pane-only, legacy) ───────────────────────

def _make_logged(func, display_name, skip_first=False):
    """Return a wrapper that logs the call once (reentrancy-safe)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if getattr(_reentry, "active", False):
            return func(*args, **kwargs)
        try:
            call_args = args[1:] if skip_first else args
            parts = [_fmt(a) for a in call_args]
            parts += [f"{k}={_fmt(v)}" for k, v in kwargs.items()]
            dash_logger.write(f"{display_name}({', '.join(parts)})", source="cryocat")
        except Exception:
            pass
        _reentry.active = True
        try:
            return func(*args, **kwargs)
        finally:
            _reentry.active = False
    return wrapper


def patch_class(cls, exclude=()):
    """Wrap every public method on *cls* with pane-only call logging."""
    exclude = set(exclude)
    for name, raw in list(vars(cls).items()):
        if name.startswith("_") or name in exclude:
            continue
        if isinstance(raw, staticmethod):
            setattr(cls, name, staticmethod(
                _make_logged(raw.__func__, f"{cls.__name__}.{name}")))
        elif isinstance(raw, classmethod):
            setattr(cls, name, classmethod(
                _make_logged(raw.__func__, f"{cls.__name__}.{name}", skip_first=True)))
        elif inspect.isfunction(raw):
            setattr(cls, name, _make_logged(raw, f"{cls.__name__}.{name}", skip_first=True))


def patch_function(module_or_obj, func_name):
    """Replace a public function on a module with a logged version."""
    func = getattr(module_or_obj, func_name)
    setattr(module_or_obj, func_name, _make_logged(func, func_name))
