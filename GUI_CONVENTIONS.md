# cryoCAT GUI — Standing Conventions

Companion to `REFACTOR_GUIDELINES.md`. That document governs the **library**;
this one governs `cryocat/app/**`. Both apply to every GUI task — where they
overlap, `REFACTOR_GUIDELINES.md` wins on typing, docstrings and exception
style, and this document wins on Dash structure, ids, dispatch and testing.

These rules apply to **every** task in docs 1–7. Task documents assume them and
will not repeat them. They override convenience: **if a quick fix would break a
rule, stop and flag it rather than working around it.**

Scope: `app/apputils.py`, `app/formgen.py`, `app/logger.py`, `app/server.py`,
`app/components/**`, `app/suite/**`, `app/tango/**`.

**Posture: this is a restart refactor.** There is no backward compatibility to
maintain, no external consumer, and no legacy carve-out for any module including
tango. Rename, delete and restructure freely — see §14.

---

## 1. The component contract

- A reusable component in `app/components/` exposes:
  - `get_<name>(prefix, ...) -> Component` — layout only, no side effects, no
    callback registration, no IO.
  - `register_<name>_callbacks(app, prefix, ...) -> None` — callbacks only, no
    layout construction.
  - `read_<name>(state) -> <python value>` — optional; a **pure** function
    converting the widget's `State` payload into the value the library accepts.
- **Every id the component creates is derived from `prefix`.** No exceptions.
- **A component never reads a global id it was not given.** Store ids it must
  observe are passed as arguments (`surfaceview.register_surface_view_callbacks(
  app, prefix, pool_store_id, *, selected_store_id=None)` is the reference
  shape). The only ids a component may reference without being handed them are
  those in §2.4.
- **A component never encodes knowledge of a specific page.** A parameter like
  `tabs_id="table-tabs"` with `visible_on_tabs=("motl-tab", "twist-tab", ...)`
  is a contract violation: the page owns that callback, or passes a predicate.
- The module docstring lists, under a `Contract` or `Public API` heading:
  1. every store the component **owns**;
  2. every id the embedding page must **supply**;
  3. the id at which the component **publishes its result** (§2.3).
- Templates to copy: `volumeview.py`, `motlsource.py`, `paletteloader.py`.

## 2. ID grammar

### 2.1 String ids

`f"{prefix}-<role>"`, lowercase, hyphen-separated, no underscores. `role` is
stable and descriptive (`-graph`, `-status`, `-params`, `-value`, `-modal`).

### 2.2 Pattern-matching (dict) ids

**One scheme only:**

```python
{"type": "<static-kind>", "owner": prefix, ...discriminators}
```

- `type` is a **static string constant** describing the kind of control
  (`"angles-param"`, `"label-row"`, `"styled-graph"`). It is never built from
  `prefix`.
- `owner` carries the `prefix`. This is what scopes an `ALL` match to one
  component instance.
- Additional keys discriminate within the instance (`param`, `tag`, `slot`,
  `row`, `label`, `name`).

**Never** `{"type": f"{prefix}-row", ...}`. Baking the prefix into `type` makes
app-wide `ALL` matching impossible, which is what the live-restyle mechanism
(§4.3) depends on. Existing offenders to migrate: `memthick_widgets`
(`f"{prefix}-row"`, `f"{prefix}-del-btn"`, `f"{prefix}-per-label-mode"`),
`memthick_widgets.get_analyzer_subform` (`id_type=prefix`), `tomoview`
(`f"{prefix}-menu-item"`), `graphsettings` (`{"index": prefix}` → `owner`),
`anglesbuilder`/`putilities` (`{"builder": prefix}` → `owner`).

Form-control ids produced by `formgen._mk_id` are the one place the grammar is
generated rather than hand-written — **hand-written ids that imitate it are
forbidden** (`pnn.py` currently hand-builds
`{"type": "nn-forms-params", "param": "nn_type", "tag": "Literal", "cls_name": ...}`;
call `formgen` or a documented helper instead, so render and parse cannot drift).

### 2.3 Reserved role suffixes

| Suffix | Meaning |
|---|---|
| `-value` | The component's **published result**. Consumers read this and nothing else. |
| `-params` | Collected form kwargs (`dict`), produced by `generate_kwargs`. |
| `-status` | User-facing status text for this component. |
| `-graph` | A `dcc.Graph`; must also carry the dict id from §4.3. |
| `-modal`, `-panel`, `-container` | Structural wrappers. |

`paletteloader` already follows this. Components that publish under some other
name must be migrated: `anglesbuilder` (`-created-path`), `rotationbuilder`
(`-rot-euler-store`), `anglesfield`/`rotationfield` (`-path`, which in
`rotationfield` does not hold a path at all).

### 2.4 Reserved global ids

The **only** unprefixed ids any module may reference. Defined as constants in
`app/ids.py`; referenced by name, never as a literal.

```
graph-settings-store
pool-registry   pool-motls   pool-extra   pool-meta   pool-next-id
suite-url   suite-tool-selector   suite-page-content   page-wrap-<tool_id>
suite-log-*
```

- Pool ids are referenced **only** through `app/pool.py` (§5). No module
  outside it contains the string `"pool-registry"`.
- `app/tango/layout.py` declares ~40 top-level stores. **There is no carve-out
  for them.** They are migrated to prefix-scoped state in doc 7 like everything
  else; until that task lands, the tango arm of the §11.3 test is a single
  `xfail` pointing at doc 7. The list above is the complete and final set of
  permitted global ids for both apps — it does not grow.
- The test in §11.3 asserts that every callback id in either app resolves to a
  layout id or an entry in that set.

## 3. Callbacks are thin

**A `@app.callback` body contains no logic.** It unpacks inputs, calls one
module-level function, and returns. Target ≤ 5 statements; a `PreventUpdate`
guard and a `try/except` that converts an exception to a status string are the
only control flow permitted inline.

```python
@app.callback(Output(f"{prefix}-value", "data"), Input(f"{prefix}-params", "data"))
def _compute(params):
    if not params:
        raise PreventUpdate
    return compute_thing(params)          # ← all decisions live here
```

- The called function is **module-level, importable, and Dash-free** — no
  `dash.no_update`, no `Input`/`State`, no component construction inside it. It
  takes plain Python and returns plain Python.
- Reference implementations already in the tree — read these before writing new
  helpers: `suite/pages/_memthick_analysis.py`, `_memthick_codegen.py`,
  `_pana_codegen.py`, `_pstructure_intersect.py`. New pure helpers for page
  `pfoo.py` go in `suite/pages/_foo_<topic>.py` following the same pattern.
- Callbacks that currently violate this and are named in later task docs:
  `motlio.load_motl`, `motlio.save_data` (×3), `tomoview.update_plot`,
  `anglesbuilder._preview`, `rotationbuilder._update`, `putilities._preview`.
- **`allow_duplicate=True` requires a comment** naming the other writers of
  that output and why the race is safe. There are ~180 uses today; each one
  touched must either gain that comment or be restructured.
- Prefer `PreventUpdate` for "nothing to do" and `no_update` for "this one
  output is unchanged". Do not mix idioms within a callback.

## 4. Figures and graph settings

### 4.1 One helper

All figure finalisation goes through `graphsettings.styled_figure`:

```python
styled_figure(fig, gs, *, uirevision, title=None, margin=None, scene=None) -> go.Figure
```

- `uirevision` is a **required keyword**. Omitting it is what resets 3D cameras
  on every redraw; the signature makes that impossible.
- The `fig.to_plotly_json()` → `apply_settings_to_figure` → `go.Figure(...)`
  dance appears at 24 sites today. After this, it appears once. `volumeview._finalize`
  is the prototype to promote.
- Error placeholders go through `graphsettings.error_figure(msg)`. Never return
  a bare `go.Figure()` on failure (`rotationbuilder` currently does), and never
  define a local `_err`.

### 4.2 Fill-only semantics

**Global graph settings supply defaults; they never override styling a plot set
deliberately.** `apply_settings_to_figure` must not overwrite a value already
present on a trace. Today it clobbers any scalar `marker.size` and forces
`line.dash` onto every scatter, which destroys deliberately dashed reference
lines and tuned point-cloud sizing.

Where an explicit override is genuinely wanted, it is opt-in:
`apply_settings_to_figure(fig_dict, gs, override=True)`.

`_CONTINUOUS_TRACE_TYPES` must include the types this app actually emits:
`mesh3d`, `isosurface`, `volume`, `histogram2d` in addition to the current set.

### 4.3 Deliberately-coloured traces must be re-pinned after styling

`apply_settings_to_figure` (called inside `style_figure` and `styled_figure`) clears existing
scalar marker colours when `palette_is_user_set=True`, then overwrites them from the chosen
palette.  A colour set on a trace **before** the styling call — whether via `color_discrete_map`
or directly on the trace — is not preserved.

**Rule:** Any trace that must keep a fixed colour (e.g. a "noise" cluster rendered in neutral grey)
must have that colour applied **after** `style_figure` / `styled_figure`, not before.  The earlier
`color_discrete_map` exists only to drive Plotly Express's initial rendering; it is not sufficient
on its own.

**Current instance:** `ploteditor._build_figure` re-pins the noise trace colour after `style_figure`.
Moving that re-pin above `style_figure` would silently restore noise to the palette colour when the
user has set a custom palette.

### 4.3 Live restyle

Every `dcc.Graph` carries a dict id **in addition to** its string id role:

```python
dcc.Graph(id={"type": "styled-graph", "owner": prefix, "name": "preview"})
```

One app-level callback maps `Input("graph-settings-store", "data")` to
`Output({"type": "styled-graph", "owner": ALL}, "figure")` and restyles in
place. Consequences:

- Component callbacks read `graph-settings-store` as **`State`**, never `Input`
  — the app-level callback owns reacting to changes. Today six modules use
  `State` and three use `Input`, so restyling works in about a third of the app
  at random. Migrate the three (`pmemthick`, `surfaceview`, `tableplot`) to
  `State`.
- The modal's claim that settings "apply immediately to all existing graphs"
  becomes true. Do not weaken the copy instead.
- Background colour must derive readable font and grid colours from its
  luminance. The current `"Dark"` option produces near-black text on `#1e1e1e`.

## 5. The motl pool

`app/pool.py` is the **sole owner** of pool state. It exports:

- the store id constants;
- the entry schema as a dataclass (`PoolEntry`: `label`, `type`, `n_rows`,
  `active`, …) — handles are built with `asdict()`, never hand-written dicts;
- pure reducers: `insert_motl(state, rows, label, *, extra=None, meta=None) -> PoolState`,
  `remove_motl`, `set_active`, `next_label`.

Rules:

- **No module outside `pool.py` mutates pool stores directly.** The insert
  reducer is currently copy-pasted in six places (`motlsidebar` ×3, `pmotl`,
  `pnn`, `pcomplexes`, `motlsink`) and the copies have diverged: `motlsink`
  writes only `pool-registry` and `pool-motls`, so a motl sent from a tool has
  no `pool-extra`/`pool-meta` and a later save-as-STOPGAP or save-as-Relion
  from the editor silently loses the extra dataframe. Fixing that divergence is
  the point of centralising.
- `motl_id` is `motl-<n>` from the `pool-next-id` counter. **Ids are never
  reused or renumbered**, including after removal — the console (§7) and the
  session script both reference them.
- Every insert path also calls `record_motl_source` (§6). Use
  `run_operation_to_pool`, which does both atomically.
- Pool ids are **user-visible**. The registry UI shows `motl-3` alongside the
  label, because the console addresses entries by id.

## 6. Dispatch and provenance

The session script — a runnable `.py` reproducing the user's session — is a
load-bearing feature, not a debug aid. Treat a gap in it as a bug of the same
severity as a wrong result.

- **Preview / live-recompute callbacks call the cryocat function directly.**
  They fire on every keystroke; logging them floods the script.
- **Commit actions** (Create, Save, Apply, Add to pool, Send to editor, and
  every console command) route through `run_operation` — or
  `run_operation_to_pool` when the result enters the pool. This is the only
  path that may write to the script.
- **Never hand-write a script line.** `motlio.load_motl` currently composes
  `f'motl = Motl.load("{filename}", "{motl_type}")'` and passes it to
  `dash_logger.write(..., source="cryocat")`, which reaches the log pane but not
  the script, and never registers the motl's source — so every later operation
  on that motl emits the `WARN: Motl source not tracked` comment that
  `logger.py` already has a branch for.
- **Every object that enters the pool gets `record_motl_source`.** Two call
  sites exist today, both in `motlsidebar`.
- **Saving is a commit action.** `save_motl` and `save_output` currently write
  files with no script line at all, so a generated script can load and
  transform but never save its results.
- Modules with commit paths and no `run_operation` today, to be audited:
  `motlio`, `motlsidebar`, `pmotl`, `pnn`, `psta`, `tableview`, `tablecluster`.
- The same rule applies to `app/tango/**`.

## 7. Console compatibility

A command console (doc 5) executes restricted Python against the pool
namespace. Two rules bind earlier work:

- Console commands go through the same `invoke_operation` chokepoint, so the
  script records GUI actions and console commands identically and a script line
  can be pasted back into the console.
- Anything reachable from the console must be addressable: stable pool ids (§5),
  a single `@gui_exposed` registry (doc 4), and no page-local shadow state that
  the console cannot see.

## 8. Styles and layout

- **`app/styles.py` is the only place style dicts are defined.** `_HINT_STYLE`
  / `_HINT` is currently redefined in 5 modules, `_ROW_STYLE` and `_LABEL_STYLE`
  in 3 each. Import them; do not re-type them.
- **`app/page_shell.py`** provides `page_shell(sidebar_children, main_children)`.
  The `dbc.Col(html.Div([...], className="sidebar", style={...100vh, sticky...}),
  width=3)` skeleton is duplicated byte-for-byte in 10 modules.
- **`formgen.form_row(name, widget, description, ...)` becomes public.** Pages
  needing a manual control use it instead of re-typing the 45 % / 55 % label /
  input split by hand (`pnn.py`).
- Colours come from CSS custom properties (`var(--color9)` etc.) or the palette
  helpers. Hard-coded hex lists in component modules (`surfaceview._PALETTE`,
  `tomoview`'s six-name colourscale dropdown) are migrated to
  `paletteloader` / `visplot.resolve_palette`.

### 8.1 Checkbox and radio-button vertical alignment

Browser defaults place the checkbox or radio input element at the text
**baseline**, which puts it 2–3 px above the label text.  Always set:

```python
# dbc.Checklist
inputStyle={"verticalAlign": "middle", "marginTop": "-2px"}
labelStyle={"verticalAlign": "middle"}

# dcc.RadioItems (stacked — one option per line)
inputStyle={"verticalAlign": "middle", "marginTop": "-2px", "marginRight": "0.4rem"}
labelStyle={"display": "block", "marginBottom": "0.3rem", "verticalAlign": "middle"}

# dcc.RadioItems (inline — options side-by-side)
# labelStyle alone is not enough; the container must also be flex because
# Dash wraps each option in a block-level element.
inputStyle={"verticalAlign": "middle", "marginTop": "-2px", "marginRight": "0.4rem"}
labelStyle={"verticalAlign": "middle", "marginRight": "1.4rem"}
style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"}
```

The `marginTop: -2px` corrects a systematic upward bias present in most
browsers.  Never omit it and never rely on `marginBottom` alone.

**Use `dbc.Checklist` (via `_field_check`) instead of `dbc.Checkbox`.**
`dbc.Checkbox` exposes no `inputStyle`/`labelStyle` and cannot be aligned
without global CSS overrides.  `_field_check` already has the correct styles
baked in; use it everywhere a single checkbox is needed.

### 8.2 Two-option radio groups must be side by side

### 8.3 Status-text colours

| State | Colour |
|---|---|
| Positive / success | `#EAAE47` (amber) |
| Uncertain / warn | `var(--bs-warning)` |
| Error | `var(--bs-danger)` |
| Muted / hint | `var(--color9)` |

**Never use `var(--bs-success)` (Bootstrap green).** The amber `#EAAE47` reads well
in both light and dark themes, avoids red–green colour-blindness conflict, and
matches the application palette. Apply to all status spans, verdict text, and
any inline indicator of successful completion.

When a `RadioItems` (dbc or dcc) has **exactly two options**, always render
them in a single row — never stacked.

```python
# dbc.RadioItems — two options
dbc.RadioItems(
    id="...",
    options=[{"label": "A", "value": "a"}, {"label": "B", "value": "b"}],
    value="a",
    inline=True,
    style={"display": "flex", "gap": "1.5rem"},
)

# dcc.RadioItems — two options (follow §8.1 inline style)
dcc.RadioItems(
    id="...",
    options=[{"label": "A", "value": "a"}, {"label": "B", "value": "b"}],
    value="a",
    inputStyle={"verticalAlign": "middle", "marginTop": "-2px", "marginRight": "0.4rem"},
    labelStyle={"verticalAlign": "middle", "marginRight": "1.4rem"},
    style={"display": "flex", "flexWrap": "wrap", "alignItems": "center"},
)
```

`inline=True` alone can be overridden by Bootstrap grid CSS; the explicit
`"display": "flex"` on the container is the authoritative override.

## 9. Errors

- Raise from the `exceptions.py` hierarchy (`UserInputError`, `ProcessError`,
  `MotlException`) per `REFACTOR_GUIDELINES.md` §1.
- **No silent `except Exception`.** There are 92 today, many bare. Every handler
  either re-raises, or writes to the log pane with `source="error"`, or returns
  a user-visible status string. An empty graph is not an error message.
- Callbacks convert exceptions to status text; they do not let a traceback reach
  the browser as a blank component.

## 10. Imports and registration

- **Heavy scientific imports are lazy** (inside the function): `open3d`,
  `skimage`, `visplot`, `cryomask`, anything pulling large arrays. Light Dash /
  numpy / pandas imports stay at module level. Today `rotationbuilder` imports
  `visplot` at module level while `anglesbuilder` imports it inside a callback —
  pick lazy and be consistent.
- **No registry population by import side effect.** `iter_standalone_builders`
  currently relies on `import cryocat.utils.geom` firing decorators. Modules to
  scan are named in one explicit list (doc 4).
- **Registering the same prefix twice must fail loudly**, with the prefix in the
  message — not with a Dash duplicate-output error 200 frames away.
- Never import a private name across modules. `putilities` imports
  `_inplane_figure` and `_ID_TYPE` from `anglesbuilder`; either promote them or
  extract the shared preview.

## 11. Testing obligations

`REFACTOR_GUIDELINES.md` §2 stands: new functions ship with tests; existing
assertions are never weakened without consent. GUI-specific tiers:

### 11.1 Pure functions (the bulk)
Every `read_*`, reducer, codegen, handle builder and `_`-module helper gets
direct unit tests. No Dash, no browser, no mocks.

### 11.2 Layout invariants
Parametrised over every `get_*` factory: no duplicate ids in the returned tree;
every id derived from `prefix`; the id set snapshotted so renames surface in
review.

### 11.3 Whole-app coupling
`suite/app.py` mounts every page and registers every callback at import, so one
test covers the whole app: every id in `app.callback_map` resolves to a mounted
layout id, a **declared dynamic id** (below), or an entry from §2.4; no two
callbacks write the same output without `allow_duplicate`; registering a prefix
twice raises. Same test for `tango/app.py`.

**Dynamic ids are a legitimate third category.** Pages mount every *page*, but not
every *control*: the motl editor renders its operation form into a placeholder div
when the user picks an operation, and the Structure page injects controls the same
way. Such ids cannot exist at boot, which is what `suppress_callback_exceptions`
is actually for. They are **declared, not allowlisted** — a module that renders
controls dynamically exports a manifest of the id shapes it will produce and the
container it renders them into:

```python
DYNAMIC_IDS = [
    ("me-op-form", {"type": "me-op-param", "param": ANY, "tag": ANY}),
    ...
]
```

The test accepts an unresolved id only if it matches a declared shape. An id
matching nothing is a defect. Adding an entry to `DYNAMIC_IDS` to silence a
failure without rendering that control is a defect too.

**Harness assumption:** Dash 4.1.0 stores dict-pattern ids in `_callback_list` as
**JSON-encoded strings**, not Python dicts. The walker must parse JSON before
comparing, or every wildcard callback is misreported as unresolved. Assert the
Dash version in the harness so a representation change fails loudly rather than
silently passing everything.

### 11.4 Provenance
For each commit-path operation: the session script gains a line; the line
parses with `ast.parse`; its imports were emitted; no `WARN: Motl source not
tracked` comment appears.

### 11.5 `@gui_exposed` pipeline
Parametrised over every registered callable — the automated form of
`REFACTOR_GUIDELINES.md` §3: annotation resolves via `resolve_param_type` → tag
exists in `TYPE_HANDLERS` → `formgen._WIDGET_FACTORIES` has a factory → a
round-trip through `generate_kwargs` returns the declared type. **This is what
makes "just add `@gui_exposed`" safe**; without it, a missing link is only
discovered by clicking.

### 11.6 Figures
Assert invariants — trace count and types, `uirevision` present,
`layout.font.family` matches settings, `colorway == resolve_palette(...)`.
**Never** full-JSON golden files.

### 11.7 End to end
`dash.testing`, three or four journeys only. Slow and chromedriver-dependent;
not where coverage comes from.

## 12. Deployment

- **Single process, single worker.** Each user runs their own instance,
  including on a cluster login node. `surface_registry`, `parametric_registry`,
  `memthick_registry` and `dash_logger.buffer` are module-level state and are
  correct only under that assumption — so assert it at startup and refuse to
  run multi-worker rather than corrupting state silently. Remove the
  `gunicorn` invocation from `server.py`'s docstring.
- **Bind `127.0.0.1`, not `0.0.0.0`.** On a shared login node `0.0.0.0:8050`
  exposes one user's motls to everyone on the node, and once the console exists
  it is remote code execution as that user. Document SSH port forwarding
  (`ssh -L 8050:localhost:8050 <login-node>`) as the access path.
- Port is configurable (`--port`) and auto-increments on collision, since
  several users share a login node.
- Server-side registries never evict on their own; removal is explicit and
  driven by the UI (closing a tab, removing a surface).

## 13. Tango parity

Tango is in scope and changes with the suite.

- Changes to `app/*.py` root modules or `app/components/**` must be verified
  against **both** apps: `import cryocat.app.tango.app` and
  `import cryocat.app.suite.app` both succeed, and §11.3 runs for both.
- Tango gets the same provenance treatment (§6) — its operations currently
  produce no session script at all.
- Tango's ~40 global stores are migrated to prefix-scoped state (doc 7), not
  preserved. Nothing in tango is exempt on grounds of age.

## 14. No compatibility layers

This is a **restart refactor**. Nothing outside `cryocat/app/**` depends on these
modules, there are no external consumers, and no released GUI API to honour.

- **Never add a shim, alias, or deprecation path.** Rename the thing and update
  every call site in the same edit. A wrapper that exists to keep an old name
  working is a defect.
- **Delete the shims that exist.** On sight, in whichever task first touches the
  file:
  - `_memthick_codegen.py` — `render_py`, `render_ipynb`, `render_ipynb_json`,
    `wrap_slurm` (four "backwards-compatible" wrappers around the real
    renderers, plus the older `(kwargs, analyzer_kwargs)` argument split).
  - `volumeview.py` — `_mesh_at = mesh_at`, kept "for backward compat within
    this module".
  - `apputils.py` — `get_motl_operation_methods`, a "backwards-compatible alias"
    for `get_single_motl_methods`.
- **Delete dead code rather than commenting it out.** `apputils.get_print_out`
  returns `""` with its entire body commented out and is still called from
  `motlio`; `apputils`, `motlio`, `tomoview`, `tableplot` and `formgen` all carry
  commented-out blocks. Remove the function and its call sites, not just the
  comment.
- **Do not preserve a signature you disagree with.** If a parameter exists only
  to paper over a design problem (the boolean layout flags in §Anti-patterns),
  remove it and restructure the caller.
- Where a rename is wide-reaching, do it as one mechanical commit with the call
  sites updated, and rely on §11.3 to prove nothing was missed.

## 15. Scope exclusions

The **Structure and Surfaces tabs are unfinished and under redesign.**
Affected: `suite/pages/pstructure.py`, `suite/pages/pcomplexes.py`,
`components/surfaceview.py`, `components/surface_registry.py`,
`components/parametric_registry.py`, `suite/pages/_pstructure_intersect.py`.

- **Include** them in mechanical, codebase-wide migrations — pool constants,
  id grammar, `styled_figure`, `page_shell`, styles, error handling — so they do
  not drift out of the conventions.
- **Exclude** them from behavioural refactoring, API redesign, and test-coverage
  targets. Do not restructure their callbacks or invent new abstractions there.
- If a mechanical migration in one of these files looks like it needs a design
  decision, **stop and flag it** instead of deciding.

## 16. Definition of done

Before reporting a task complete, verify:

1. Both apps import and start (§13).
2. §11.3 passes for both apps.
3. New pure functions have tests (§11.1); touched commit paths have §11.4 tests.
4. No new global id; no new hand-written pool mutation; no new local style dict;
   no new `to_plotly_json()` call site.
5. No shim, alias or deprecation path was added, and any encountered in a touched
   file was deleted (§14).
6. Every touched `allow_duplicate=True` has its justifying comment (§3).
7. Docstrings match signatures, including after a retype
   (`REFACTOR_GUIDELINES.md` §1).
8. Consumers of every edited shared component were enumerated and checked
   (`REFACTOR_GUIDELINES.md` §4).

## Anti-patterns

Each of these is present in the codebase today and is a defect on sight:

- A callback whose `Output` id does not exist in any mounted layout —
  `suppress_callback_exceptions=True` makes it silent forever.
  (`anglesbuilder`'s `skip_preview` flag is the archetype: the sidebar layout
  has no `-preview` graph, so forgetting the flag writes into the void.)
- Two layout variants of one component kept in sync by a boolean flag on the
  registrar. Split the layout into composable pieces instead.
- `return not is_open` as a modal toggle driven by three different buttons
  (`anglesfield`, `rotationfield`) — correct only by luck of call order.
- Re-rendering an entire list from index 0 on every poll tick
  (`logpanel.update_log`), and polling while the panel is closed.
- Unpacking callback arguments positionally (`cb_args[-2]`, `cb_args[-1]`).
- A `**kwargs` passthrough that reads `className` without popping it, applying
  it to both wrapper and inner control (`customel`, both factories).
- Hardcoding `style=` in a factory that also accepts `**kwargs`
  (`customel.InlineLabeledDropdown(id_, label, style={...})` raises
  `TypeError: multiple values for keyword 'style'`).
- Resetting a user's selection because an unrelated store changed
  (`motlsource._populate` returns `active_ids[0]` on every `pool-registry`
  change).
- A handle dict whose docstring and returned keys disagree
  (`memthick_registry.make_handle` omits `n_finite_inflection_thickness_nm`).
  Use a dataclass.
- Referencing a class before its definition (`_pana_codegen._format_value` uses
  `_Verbatim`, defined below it).
- The same helper defined twice in two modules (`_Verbatim`, `_format_value`,
  `_format_kwargs`, `render_slurm_wrapper` in both codegen modules).


## Standing note — tests outside `cryocat/app/`

This applies to **every** phase, without exception, and it is deliberately stricter
than `REFACTOR_GUIDELINES.md` §2.

**Tests that are not under `tests/app/` — `cryomotl`, `geom`, `classutils`, every
library test — must not be modified during GUI work.** Not the assertions, not the
fixtures, not the parametrisation, not the tolerances. Do not add `skip`, `xfail`, or
a marker to make one pass.

§2 of `REFACTOR_GUIDELINES.md` permits editing an existing test to track a changed
call signature. **That permission does not extend to GUI tasks**, because a GUI task
should not be changing a library signature in the first place. If one appears to need
changing, that is itself the thing to report.

### Protocol when a library test fails

1. **Stop.** Do not edit the test, the assertion, or the fixture.
2. **Report**, with:
   - the test's full node id;
   - the failing assertion, expected vs actual;
   - the GUI-side change that caused it;
   - the smallest reproduction that does not involve the app.
3. **Wait for explicit consent** before touching anything outside `tests/app/`.
4. If the failure looks like a genuine library bug that the GUI has exposed, **still
   report it**. Do not fix the library from a GUI task, and do not work around it in
   the app without saying so.

### Baseline

**Run the full test suite before starting each phase and record the pass/fail counts
in the first commit message.** Without a baseline, a pre-existing failure is
indistinguishable from a new one — which is exactly when someone "fixes" a test that
was already red.

### How to read a library test failure

GUI work should change **no library behaviour at all**. A failing library test is
therefore not a problem to be resolved; it is evidence that either an unintended
library edit crept in, or GUI code was relying on library behaviour it should not
have been. Both are reasons to stop and report rather than to proceed.

New tests for GUI behaviour go in `tests/app/`, always — never as an extension to a
library test module.