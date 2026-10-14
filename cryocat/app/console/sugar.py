"""Pool-reference sugar for the cryoCAT console.

``#n`` is shorthand for the pool variable ``motl_n``.  All transformations are
token-level rewrites that preserve string-literal contents.

Public API
----------
* :func:`desugar` — ``#n`` → ``motl_n`` (used before Python parsing).
* :func:`resugar` — ``motl_n`` → ``#n`` (used for compact display).
"""
from __future__ import annotations

import re

_MOTL_VAR_RE = re.compile(r"\bmotl_(\d+)\b")


def desugar(text: str) -> str:
    """Replace ``#n`` with ``motl_n``, leaving ``#`` inside string literals untouched.

    Operates character-by-character to track string-literal state correctly,
    including triple-quoted strings and escape sequences.
    ``#`` not followed by digits (i.e. a real comment) causes the rest of the
    line to be passed through verbatim.
    """
    result: list[str] = []
    i = 0
    n = len(text)

    while i < n:
        c = text[i]

        # Triple-quoted strings  (must check before single-char)
        if c in ('"', "'") and text[i : i + 3] in ('"""', "'''"):
            quote = text[i : i + 3]
            result.append(quote)
            i += 3
            while i < n:
                if text[i : i + 3] == quote:
                    result.append(quote)
                    i += 3
                    break
                if text[i] == "\\":
                    result.append(text[i : i + 2])
                    i += 2
                else:
                    result.append(text[i])
                    i += 1
            continue

        # Single-quoted strings
        if c in ('"', "'"):
            quote = c
            result.append(c)
            i += 1
            while i < n:
                if text[i] == "\\":
                    result.append(text[i : i + 2])
                    i += 2
                elif text[i] == quote:
                    result.append(text[i])
                    i += 1
                    break
                else:
                    result.append(text[i])
                    i += 1
            continue

        # Pool reference: #<digits>
        if c == "#":
            m = re.match(r"#(\d+)", text[i:])
            if m:
                result.append(f"motl_{m.group(1)}")
                i += len(m.group(0))
                continue
            # Real comment — pass through to end of logical line
            result.append(text[i:])
            break

        result.append(c)
        i += 1

    return "".join(result)


def resugar(src: str) -> str:
    """Replace ``motl_n`` with ``#n`` for compact display.

    Operates on whole-word boundaries so ``motl_10`` → ``#10`` and
    ``my_motl_3_copy`` is left untouched.
    """
    return _MOTL_VAR_RE.sub(lambda m: f"#{m.group(1)}", src)
