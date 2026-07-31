"""Shared style dictionaries for the cryoCAT GUI.

§8 of GUI_CONVENTIONS.md — the only place inline style dicts are defined.
Import from here; never re-type a style dict in a component or page module.
"""

# ── Scalar tokens ─────────────────────────────────────────────────────────────
# Label column: fixed width so labels align across the whole form regardless of
# text length.  130 px ≈ 45 % of a ~290 px sidebar (the previous percentage
# approach).  Verify at the narrow end of the sidebar before changing.
FORM_LABEL_COL_WIDTH = "130px"
FORM_ROW_GAP = "0.25rem"        # vertical gap between consecutive form rows
FORM_LABEL_GAP = "4px"          # horizontal gap between label and control
SECTION_GAP = "0.5rem"          # gap above a section header inside a panel
CONTROL_HEIGHT = "22px"         # mirrors .dash-dropdown-trigger height in CSS

# Button colour tokens — values are dbc.Button color= parameter strings.
# Use these instead of typing "primary"/"secondary"/"success" per call site.
BTN_PRIMARY = "primary"         # main action (strongest blue)
BTN_SECONDARY = "secondary"     # secondary / muted action
BTN_NEUTRAL = "outline-secondary"  # cancel / neutral outline variant

# ── Hint / caption text ───────────────────────────────────────────────────────
HINT: dict = {"fontSize": "0.85rem", "color": "var(--color9)"}
HINT_SM: dict = {"fontSize": "0.75rem", "color": "var(--color9)", "marginTop": "2px"}

# ── formgen form rows ─────────────────────────────────────────────────────────
FORM_HINT: dict = {"fontSize": "0.85rem", "color": "var(--color9)", "padding": "2px 0"}
FORM_LABEL: dict = {
    "width": FORM_LABEL_COL_WIDTH,
    "minWidth": FORM_LABEL_COL_WIDTH,
    "display": "flex",
    "alignItems": "center",
    "boxSizing": "border-box",
    "paddingRight": FORM_LABEL_GAP,
}
FORM_INPUT: dict = {"flex": "1 1 auto", "minWidth": "0"}
FORM_ROW: dict = {
    "display": "flex", "flexDirection": "row", "marginBottom": FORM_ROW_GAP,
    "width": "100%", "alignItems": "center",
}
FORM_COMPACT_INPUT: dict = {
    "width": "100%", "height": CONTROL_HEIGHT, "minHeight": CONTROL_HEIGHT,
    "padding": "0 6px", "fontSize": "11px", "lineHeight": "20px",
    "boxSizing": "border-box", "borderRadius": "3px",
}

# ── Manual control rows (flex-gap style for hand-written panel rows) ──────────
CTRL_ROW: dict = {
    "display": "flex", "alignItems": "center",
    "gap": "0.5rem", "marginBottom": "0.35rem",
}
CTRL_LABEL: dict = {"flex": f"0 0 {FORM_LABEL_COL_WIDTH}", "marginBottom": "0"}
CTRL_INPUT: dict = {"flex": "1 1 auto", "minWidth": "0"}

# ── Section headers ───────────────────────────────────────────────────────────
SECTION_HEADER: dict = {
    "fontSize": "0.9rem", "fontWeight": 600,
    "margin": f"{SECTION_GAP} 0 0.2rem",
}
