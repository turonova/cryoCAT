import inspect
import functools
import threading

_reentry = threading.local()


class StreamToList:
    def __init__(self):
        self.buffer = []  # list of (msg, source)

    def write(self, msg, source="dash"):
        if msg.strip():
            self.buffer.append((msg.strip(), source))

    def flush(self):
        pass

    def get_logs(self, last_index=0):
        """Return new logs since last_index, and if any are dash-triggered."""
        sliced = self.buffer[last_index:]
        new_messages = [msg for msg, _ in sliced]
        new_dash = any(source == "dash" for _, source in sliced)
        return new_messages, len(self.buffer), new_dash

    def get_all_logs(self):
        return "\n".join(
            msg.decode('utf-8') if isinstance(msg, bytes) else msg
            for msg, _ in self.buffer
        )

    def clear(self):
        self.buffer.clear()


def print_dash(*args):
    dash_logger.write(" ".join(map(str, args)), source="dash")


# Global singleton instance
dash_logger = StreamToList()


# ── Automatic cryoCAT call logging ────────────────────────────────────────────

def _fmt(value):
    """Compact repr for a single argument — truncates large containers."""
    if hasattr(value, 'df') and hasattr(value, 'get_unique_values'):
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


def _make_logged(func, display_name, skip_first=False):
    """Return a wrapper that logs the call once (reentrancy-safe)."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if getattr(_reentry, 'active', False):
            # Inner call from within a cryoCAT function — skip logging.
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
    """
    Wrap every public instance method, classmethod and staticmethod on cls
    with call logging.  Call this once at app startup.
    """
    exclude = set(exclude)
    for name, raw in list(vars(cls).items()):
        if name.startswith('_') or name in exclude:
            continue
        if isinstance(raw, staticmethod):
            setattr(cls, name,
                    staticmethod(_make_logged(raw.__func__, f"{cls.__name__}.{name}")))
        elif isinstance(raw, classmethod):
            setattr(cls, name,
                    classmethod(_make_logged(raw.__func__, f"{cls.__name__}.{name}",
                                             skip_first=True)))
        elif inspect.isfunction(raw):
            setattr(cls, name,
                    _make_logged(raw, f"{cls.__name__}.{name}", skip_first=True))


def patch_function(module_or_obj, func_name):
    """Replace a public function on a module with a logged version."""
    func = getattr(module_or_obj, func_name)
    setattr(module_or_obj, func_name, _make_logged(func, func_name))
