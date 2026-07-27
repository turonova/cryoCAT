"""Shared style dictionaries for the cryoCAT GUI.

§8 of GUI_CONVENTIONS.md — the only place inline style dicts are defined.
Import from here; never re-type a style dict in a component or page module.
"""

# ── Hint / caption text ───────────────────────────────────────────────────────
HINT: dict = {"fontSize": "0.85rem", "color": "var(--color9)"}
HINT_SM: dict = {"fontSize": "0.75rem", "color": "var(--color9)", "marginTop": "2px"}

# ── formgen form rows (45 % label / 55 % input fixed split) ───────────────────
FORM_HINT: dict = {"fontSize": "0.85rem", "color": "var(--color9)", "padding": "2px 0"}
FORM_LABEL: dict = {
    "width": "45%", "display": "flex", "alignItems": "center",
    "boxSizing": "border-box", "paddingRight": "4px",
}
FORM_INPUT: dict = {"width": "55%"}
FORM_ROW: dict = {
    "display": "flex", "flexDirection": "row", "marginBottom": "0.25rem",
    "width": "100%", "alignItems": "center",
}
FORM_COMPACT_INPUT: dict = {
    "width": "100%", "height": "22px", "minHeight": "22px",
    "padding": "0 6px", "fontSize": "11px", "lineHeight": "20px",
    "boxSizing": "border-box", "borderRadius": "3px",
}

# ── Manual control rows (flex-gap style for hand-written panel rows) ──────────
CTRL_ROW: dict = {
    "display": "flex", "alignItems": "center",
    "gap": "0.5rem", "marginBottom": "0.35rem",
}
CTRL_LABEL: dict = {"fontSize": "0.85rem", "flex": "0 0 45%", "marginBottom": "0"}
CTRL_INPUT: dict = {"flex": "1 1 auto", "minWidth": "0"}

# ── Section headers ───────────────────────────────────────────────────────────
SECTION_HEADER: dict = {
    "fontSize": "0.9rem", "fontWeight": 600, "margin": "0.5rem 0 0.2rem",
}
