"""Global id constants for the cryoCAT GUI.

§2.4 of GUI_CONVENTIONS.md — the only unprefixed ids any module may reference.
Every id here is unique in the app; no two components share one.
Reference these by name; never use the string literals in other modules.

Pool ids are imported by ``app/pool.py`` only.  No other module may contain the
string ``"pool-registry"`` etc.; the §11.3 test enforces this.
"""

# ── Graph settings ──────────────────────────────────────────────────────────────
# Owned by app/components/graphsettings.py.  Every dcc.Graph in the app reads
# this store as State; one app-level callback (T4c) rewrites graphs on change.

GRAPH_SETTINGS_STORE: str = "graph-settings-store"

# ── Motl pool ───────────────────────────────────────────────────────────────────
# Managed exclusively through app/pool.py (§5).  No other module may mutate
# these stores directly.

POOL_REGISTRY: str = "pool-registry"   # { motl_id: {label, type, n_rows, active, …} }
POOL_MOTLS:    str = "pool-motls"       # { motl_id: <serialised motl rows> }
POOL_EXTRA:    str = "pool-extra"       # { motl_id: <stopgap/relion/dynamo extra df> }
POOL_META:     str = "pool-meta"        # { motl_id: <relion params, data_type, …> }
POOL_NEXT_ID:  str = "pool-next-id"    # incrementing counter for stable motl_id

# ── Suite navigation / chrome ────────────────────────────────────────────────────
# Owned by app/suite/app.py.  Navigation components and the page-content container.

SUITE_URL:           str = "suite-url"
SUITE_TOOL_SELECTOR: str = "suite-tool-selector"
SUITE_PAGE_CONTENT:  str = "suite-page-content"

# Prefixes for the stable page-wrapper divs and log-panel components.
PAGE_WRAP_PREFIX: str = "page-wrap-"
SUITE_LOG_PREFIX: str = "suite-log-"


def page_wrap_id(tool_id: str) -> str:
    """Return the stable wrapper-div id for a tool page."""
    return f"{PAGE_WRAP_PREFIX}{tool_id}"


def suite_log_id(suffix: str) -> str:
    """Return the log-panel component id for the given suffix."""
    return f"{SUITE_LOG_PREFIX}{suffix}"


# ── File browser (Phase 10) ──────────────────────────────────────────────────────
# Owned exclusively by app/components/filebrowser.py.  Exactly one browser modal
# is mounted at app level in each app; these stores are shared across all path
# fields (D1 — one modal, not one per field).

BROWSER_REQUEST: str = "browser-request"
# { "owner": prefix, "mode": open|directory|save, "kind": str, "extensions": [...] }

BROWSER_CWD: str = "browser-cwd"
# absolute path string of the directory currently shown in the modal

BROWSER_LAST_DIR: str = "browser-last-dir"
# { kind: last_absolute_dir } — persisted per-kind across opens (D4)

BROWSER_RESULT: str = "browser-result"
# { "owner": prefix, "value": absolute_path } — written on Confirm for formgen
# write-back (path-input fields with a different id_type read this store)

# ── Rotation-builder modal (Phase 11) ───────────────────────────────────────────
# Owned exclusively by app/components/rotationmodal.py.  One modal per app (D1).
# Build buttons write to this store; write-back callback reads it to route output.

ROTATION_REQUEST: str = "rotation-request"
# { "target": <rotation-build-btn id dict> } — written when any Build button fires
