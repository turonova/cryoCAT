# GUI Phase 2 — Provenance and the Session Script

Read `GUI_CONVENTIONS.md` first (§6 is the governing section). Phase 1 must be
complete: this phase builds on `app/pool.py` (T6) and consumes the
`# TODO(doc-2):` markers it left behind.

## Why this phase exists

The session script — a runnable `.py` reproducing what the user did — is the
answer to the reproducibility objection every click-driven scientific tool faces.
It is a load-bearing feature, not a debug aid. **Treat a gap in it as a defect of
the same severity as a wrong numerical result.**

It is also the foundation for the console (doc 5): the console executes the same
calls through the same chokepoint, so a script line must be something a user can
paste back into the console and vice versa.

Today the feature is roughly half-built. `logger.py` implements the hard parts
well; the wiring is incomplete, and two design flaws mean that even where it is
wired, the output is not a runnable script.

---

## Audit: what is actually broken

Verify each of these before changing anything, so the fix is measured against
reality rather than this document.

### A1 — Most commit paths bypass the chokepoint
`run_operation` appears in 7 modules (`anglesbuilder`, `apputils`, `pcomplexes`,
`pmemthick`, `ppana`, `pstructure`, `putilities`) and in **none** of `motlio`,
`motlsidebar`, `pmotl`, `pnn`, `psta`, `tableview`, `tablecluster`, `tomoview`, or
anything under `app/tango/`. `append_script_line` has exactly one caller:
`invoke_operation` (`logger.py:241`).

### A2 — `run_operation_to_pool` does not exist
`apputils.py`'s header comment and `run_operation`'s docstring both reference it.
It was never written, which is why pool insertion and provenance recording are
separate, manual, and inconsistent.

### A3 — Loads hand-write a fake script line
`motlio.load_motl` composes a script-looking string and passes it to
`dash_logger.write(..., source="cryocat")`. That reaches the **log pane only** —
not the script — and never calls `record_motl_source`. Every later operation on
that motl therefore takes the `WARN: Motl source not tracked at this call site`
branch (`logger.py:168`). The presence of that branch is the tell: the gap was
known and papered over.

### A4 — Saves are not logged at all
`apputils.save_motl` and `apputils.save_output` write files and return a status
string, with no script line. Call sites: `motlio.py:463`, `motlio.py:694`,
`tableview.py:784`, `tableview.py:854`, plus `save_output` in `tableview` and
`tableplot`, and `pmemthick.py:1236`. **A generated script can load and transform
but never saves its results** — so it reproduces nothing a user can inspect.

### A5 — Provenance is keyed by `id()`, which is unsound
`StreamToList._motl_sources` maps `id(motl_obj) -> expression`. CPython reuses
`id()` values after garbage collection, so a freed motl's address can be inherited
by a new one, which then reports the wrong provenance. The dict never evicts, so
the risk grows monotonically over a session.

Worse, it cannot work across callbacks at all: a motl is serialised to store rows
between callbacks and reconstructed as a **different object** on the next one.
Object identity is only meaningful *within* one callback. The two
`record_motl_source` calls in `motlsidebar` (`:681`, `:822`) therefore have no
effect on any later operation.

### A6 — The script is a call log, not a program
`_render_python_line` emits bare expression statements —
`cryomotl.Motl.load('x.em')`, `motl.clean_by_distance(distance=20)` — with no
variable binding. Nothing is named, so nothing can be referenced; the script
computes values and discards them. Running it top to bottom produces no result.
This is the single largest gap and it is why A5's inline-expansion machinery
(`_MAX_MOTL_DEPTH`, nested expression rendering) exists: it is compensating for
the absence of variables.

### A7 — Tango never starts a session
`suite/app.py` calls `dash_logger.start_session(...)`; `tango/app.py` does not.
`append_script_line` opens with `if self._log_fh is None: return`, so every tango
operation is silently discarded. Silent, because there is no session to warn into.

### A8 — Dead logging machinery
`logger.patch_class`, `logger.patch_function`, `logger._make_logged` and the
`_reentry` thread-local are defined and **never called**. Per §14, delete them.
`apputils.get_print_out` returns `""` with its whole body commented out and is
still imported by `motlio` and `tableplot`; delete it and its call sites.

---

## Carried over from phase 1 (T0)

Three findings from the harness run change this phase.

### C1 — `motlio`'s simple-save component was writing into the void
T0 found `get_motl_simple_save_component` missing the two `dcc.Store`s its own
callbacks write to. That fix landed in phase 1, but it means **the save paths P6
touches were partially non-functional**: the write itself presumably happened, and
the status/path feedback went nowhere.

So P6 cannot assume the pre-refactor behaviour was correct and preserve it.
Before routing a save through the chokepoint, **establish what it actually does
today** — does the file appear, does the status update, does the path store
populate — and record that in the commit message alongside the new behaviour. This
is the one place in phase 2 where "behaviour-preserving" is the wrong instinct.

### C2 — Dynamic controls change how P5 can be verified
The motl editor renders its operation form into a placeholder div when the user
selects an operation, so those control ids do not exist at boot (§11.3). Two
consequences:

- P5's audit **must** be AST-based, not id-based. An id-based check cannot see a
  control that is rendered on demand, so it cannot tell a routed commit path from
  an unrouted one.
- When routing the editor's single- and multi-motl operations, the kwargs come
  from a dynamically-rendered form. Read them with the standard `ALL`-state
  pattern; do not add a parallel path that only works because the form happens to
  be mounted.

### C3 — The rotation-field defects need triage before T9, not after
T0 left 30 unresolved rotation ids (`rotfld-*` and
`me-op-param input_rotation`). Those are **two different problems wearing one
symptom**, and T9 should separate them first:

- **Genuine orphans** — `formgen._rotation_field` builds the modal prefix as
  `rotfld-{builder}-{type}-{param}` when `id_extra` carries a `builder` key, while
  `motlsidebar._register_rotation_fields_for_form` registers against
  `rotfld-{type}-{param}`. Where those disagree, the Build… button is wired to
  nothing. Test: compare the two formulas for each `RotationLike` parameter.
- **Legitimately dynamic** — `me-op-param input_rotation` is a control inside the
  editor's on-demand operation form. It is unresolved at boot **by design** and
  belongs in `DYNAMIC_IDS`, not in the defect list.

Triage matters for this phase because a rotation value that never reaches the form
control is a rotation that never reaches `run_operation`, which means the script
records an operation with a wrong or absent rotation. **Fix or declare all 30
before P5 routes the editor's operations**, or provenance will faithfully record
incorrect calls.

---



Four ideas, in dependency order.

### D1 — A pool entry is a variable

The unit of provenance is the **pool entry**, not the Python object. Each pool
entry owns a script variable name derived from its `motl_id`:

```
motl-3  ->  motl_3
```

Ids are never reused (phase 1, §5), so variable names are stable and unique for
the process lifetime. Provenance lives with the entry, survives serialisation,
and is exactly what the console needs to resolve `#3`.

### D2 — Every logged call is an assignment

```python
motl_0 = cryomotl.Motl.load('/data/run1.em')
motl_0 = motl_0.clean_by_distance(distance=20)          # applied in place
motl_4 = motl_0.split_by_feature(column_names='tomo_id')
emmotl.EmMotl(motl_4).write_out('/data/split.em')
```

- An operation that **replaces** a pool entry reassigns its variable.
- An operation that **creates** a new entry binds a new variable.
- An operation with no pool output (a save, a mask write) is a bare statement.
- Inputs are referenced **by variable name**, which removes the need for nested
  inline expansion — `_MAX_MOTL_DEPTH` and the recursive motl branch of
  `_render_value` go away.

### D3 — Log the user's arguments, not normalised internals

`run_operation` receives the kwargs the callback assembled. **Pass the raw form
values** — paths, numbers, strings — not values already normalised into arrays or
objects. A logged `map_path='/data/mask.em'` is reproducible; a logged
`None  # <array shape (64,64,64): supply input>` is not.

Where a callback must normalise before calling (e.g. it needs the array for a
preview as well), it logs the un-normalised call and lets the library normalise
again inside. Where that is genuinely impossible, flag it rather than emitting a
placeholder.

### D4 — One chokepoint, three entry points

```
GUI commit callback ──┐
console command ──────┼──► invoke_operation ──► executes, appends script line
scripted replay ──────┘
```

`invoke_operation` stays the only writer to the script. `run_operation` and
`run_operation_to_pool` are thin wrappers over it; the console (doc 5) is a third
caller. Nothing else may call `append_script_line`.

---

## Tasks

### P1 — Session lifecycle

**Create** `app/session.py` (or extend `logger.py` — one place, not both):

```python
def start_session(log_dir: str | Path | None = None) -> Path:
    """Open the session script. Idempotent: a second call is a no-op returning
    the existing path. Default dir: ~/.cryocat/sessions."""

def session_path() -> Path | None: ...
def close_session() -> None: ...
```

- **Called once from `app/server.py`**, before either app is imported, so both
  apps share one session. Remove the call from `suite/app.py`; do not add one to
  `tango/app.py` (A7).
- Script header: shebang, generation timestamp, cryocat version, and a comment
  noting it was generated from a GUI session. It is a paper artifact; make it
  self-describing.
- `append_script_line` with no open session **warns once** to the pane with
  `source="error"` instead of silently returning.
- Register `close_session` with `atexit`.

**Delete** (§14): `patch_class`, `patch_function`, `_make_logged`, `_reentry`,
`apputils.get_print_out` and its imports in `motlio` and `tableplot` (A8).

**Tests** `tests/app/test_session.py`: header is present and parses; second
`start_session` is a no-op; `append_script_line` with no session emits exactly one
error entry, not N.

---

### P2 — Provenance keyed by pool id

**Replace** `StreamToList._motl_sources` (A5).

```python
# app/provenance.py
def bind(motl_id: str) -> str:
    """Return the script variable name for a pool entry ('motl-3' -> 'motl_3')."""

def record(motl_id: str, expr: str) -> None:
    """Record the expression that produced this entry."""

def expr_for(motl_id: str) -> str | None: ...
def var_for(motl_id: str) -> str | None:
    """Variable name if this entry has been bound in the script, else None."""

def forget(motl_id: str) -> None: ...
def clear() -> None: ...
```

Within a single call, `invoke_operation` still needs to recognise a motl object it
was handed in order to render it as a variable. Keep a **short-lived**
object→variable map scoped to one `invoke_operation` invocation, populated from
the kwargs it was given — not a process-lifetime `id()` table.

Where a motl argument cannot be resolved to a bound variable, emit
`# WARN: unbound input <param>` **and** an error-source pane entry. Do not emit a
plausible-looking expression for an input whose origin is unknown.

**Tests** `tests/app/test_provenance.py`: `bind` is deterministic and
round-trips; `expr_for` on an unknown id is `None`; ids removed from the pool are
forgotten; two entries never share a variable name.

---

### P3 — Assignment-emitting renderer

**Rewrite** `logger._render_python_line` (A6).

```python
def render_call(fn, kwargs: dict, *, assign_to: str | None = None
                ) -> tuple[str, list[tuple[str, str]]]:
    """Return (source_line, imports_needed).

    assign_to: variable name to bind the result to, or None for a bare statement.
    """
```

- Motl-valued kwargs render as their bound variable name (P2), never as a nested
  constructor call. Delete `_MAX_MOTL_DEPTH` and the recursive motl branch of
  `_render_value`.
- Keep the existing correct handling of bound methods, classmethods and plain
  functions in `_module_short` / receiver rendering.
- Non-reproducible values (arrays with no originating path) produce a
  `# WARN:` comment and an error pane entry — not a silent `None` placeholder that
  makes the script look runnable when it is not.

**Tests** `tests/app/test_script_render.py`:
- `assign_to="motl_4"` produces `motl_4 = <expr>`; `None` produces a bare
  statement.
- A motl kwarg bound to `motl_0` renders as the bare name `motl_0`.
- Import statements are emitted once per module across a sequence of calls.
- **Names-bound-before-use check** (the structural proof that the script is a
  program): render a synthetic session, `ast.parse` it, walk it, and assert every
  `Name` in load context is bound by a prior assignment, an import, or a builtin.
  This is the test that A6 cannot regress past.
- **Replay check**: render a session using a stub module, `exec` it in a fresh
  namespace with the stub injected, and assert it completes and the final variable
  holds the expected value.

---

### P4 — `run_operation_to_pool`

**Create** in `apputils.py` (A2), built on phase 1's `pool.insert_motl`:

```python
def run_operation_to_pool(
    fn, kwargs: dict, state: PoolState, *,
    label: str | None = None,
    replaces: str | None = None,   # motl_id to overwrite, or None to create
) -> tuple[PoolState, str, Any]:
    """Invoke fn through the chokepoint and land the result in the pool.

    Returns (new_state, motl_id, result). Atomic: either the pool gained/updated
    the entry and the script gained its line, or neither happened.
    """
```

Sequence: resolve the target variable name → `invoke_operation(fn, kwargs, assign_to=var)`
→ `pool.insert_motl` (or replace) → `provenance.record`. On exception, nothing is
written to the pool and `invoke_operation`'s failure comment is the only script
output.

**Tests** `tests/app/test_run_operation.py`: success writes exactly one script
line and one pool entry; a raising `fn` writes a comment and leaves the pool
byte-identical; `replaces=` reassigns the existing variable rather than binding a
new one.

---

### P5 — Commit-path audit

Route every commit action through the chokepoint (A1). Work the list; each item is
"find the commit callback, move its cryocat call behind `run_operation` /
`run_operation_to_pool`, leave preview callbacks alone".

| Module | Commit paths |
|---|---|
| `motlio` | load (P7), 3× save (P6) |
| `motlsidebar` | load→pool, single-motl ops, multi-motl ops (the 3 `# TODO(doc-2):` sites) |
| `pmotl` | pool insert, apply-operation |
| `pnn` | send-to-editor, NN run |
| `psta` | commit actions |
| `tableview` | 2× save |
| `tablecluster` | cluster commit |
| `motlsink` | send-to-editor |
| `tango/callbacks` | k-means (P8) |

**Do not** route preview / live-recompute callbacks (§6). They fire per keystroke;
logging them floods the script. If a callback is both — recomputes a preview *and*
commits — split it.

Add a guard test rather than trusting the convention: `tests/app/test_dispatch.py`
asserts that no module under `app/suite/**` or `app/tango/**` calls a known
mutating library entry point (`write_out`, `to_csv`, `pickle.dump`,
`save_motl`, `save_output`) directly. AST-based, not grep — a page importing
`save_output` and calling it is the exact failure mode this catches.

---

### P6 — Saves become logged operations

`save_motl` and `save_output` (A4) are commit actions. Options, in order of
preference:

1. Have callers invoke them through `run_operation`, so the script gets
   `apputils.save_motl(...)` — **rejected**: the script would depend on
   `cryocat.app`, which §6 forbids in generated code.
2. **Preferred:** make the script line the *library* call the save resolves to —
   `emmotl.EmMotl(motl_3).write_out('/data/out.em')`,
   `motl_3.df.to_csv('/data/out.csv', index=False)`. `save_motl` becomes a thin
   dispatcher that computes the library call, logs it via `invoke_operation`, and
   executes it.

This means `save_motl`'s Relion branch — which currently constructs
`RelionMotl`/`RelionMotlv5` with six keyword arguments — must render that
construction into the script too. That is the messiest renderer case; get it right
here rather than in doc 6.

**Tests**: for each of the five save formats (em, stopgap, dynamo, relion 3.1,
relion 5.0), the emitted line parses, imports only `cryocat.*`, and contains no
reference to `cryocat.app`.

**Before starting, read C1.** The simple-save component's stores were missing
until phase 1 fixed them, so its status and path feedback previously went nowhere.
Characterise the current behaviour before changing it.

---

### P7 — Loads become logged operations

Delete `motlio.load_motl`'s hand-composed script string (A3). A load is an
operation like any other: `run_operation_to_pool(Motl.load, {...})` binds
`motl_N` and records provenance, so downstream ops reference the variable and the
`WARN: Motl source not tracked` branch stops firing.

`motlio` is otherwise **out of scope** — its 1013 lines and 3× Relion block are
doc 6. Touch only the load and save seams.

**Tests**: after a simulated load-then-operate sequence, the script contains an
assignment for the load, the operation references that variable, and no
`WARN: Motl source not tracked` comment appears anywhere.

---

### P8 — Tango provenance

With P1 done, tango shares the session. Route `tango/callbacks.compute_k_means`
(`CustomDescriptor.load` + `k_means_clustering`) through the chokepoint, and audit
`tango/sidebar.py` and `tango/table.py` for load/save commit paths — they use the
shared `motlio` and `tableview` components, so P6 and P7 cover most of it.

Tango's descriptors are not pool entries, so they need their own variable naming
(`desc_0`, …). Use the same `provenance.bind` mechanism with a different key
space rather than a second implementation.

---

### P9 — Errors reach the log

`invoke_operation` already writes `✗` plus a traceback with `source="error"` and a
script comment. The remaining problem is handlers that swallow **before** it is
reached — 92 `except Exception` sites, many bare (§9).

In this phase, fix only those on commit paths touched by P5–P8: each must
re-raise, or write with `source="error"`, or return a user-visible status string.
An empty graph or an unchanged table is not an error message. The rest are doc 7.

---

## Exit criteria

1. `grep -rn "append_script_line" cryocat/app/` → definition plus
   `invoke_operation` only.
2. `grep -rn "dash_logger.write" cryocat/app/` → `logger.py` only (no module
   hand-writes script-shaped lines).
3. No `WARN: Motl source not tracked` comment is produced by any scripted test
   sequence.
4. The names-bound-before-use check and the replay check (P3) pass.
5. A manual end-to-end pass: load a motl, apply two operations, save it, and
   confirm the resulting script runs from a clean interpreter and produces a file
   identical to the one the GUI wrote. **This is the phase's real acceptance
   test** — run it and record the result in the commit message.
6. `patch_class`, `patch_function`, `_make_logged`, `_reentry`, `get_print_out`
   are gone.
7. Every `# TODO(doc-2):` marker from phase 1 is resolved and removed.

## Two decisions to confirm before starting

- **Script granularity.** One script per app session (current behaviour, appended
  to) versus one per tool or per user-triggered "start recording". Appending
  everything is simplest and matches how a lab notebook works, but a long session
  produces a long file with abandoned experiments in it. My recommendation: keep
  one appended file, and add a "start a new script" button later if it becomes
  annoying — the file is cheap and losing provenance is not.
- **Failed operations in the script.** Currently a failure appends a `#`-commented
  line. Keep them (they are part of what happened, and a script that silently
  omits a failed step is misleading) — confirm you agree, because it means the
  generated script is a record rather than a clean recipe.
