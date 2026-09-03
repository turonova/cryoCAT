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

# Font-size tokens — always reference these; never write "0.85rem" inline (§1).
FONT_MED   = "0.9rem"           # medium-reduced text (table headers, compact buttons)
FONT_SM    = "0.85rem"          # reduced text (hint lines, heatmap skip notice …)
FONT_TIGHT = "0.8rem"           # tighter hint text (range/modal display overrides)
FONT_XS    = "0.75rem"          # very small captions (HINT_SM)

# Button colour tokens — values are dbc.Button color= parameter strings.
# Use these instead of typing "primary"/"secondary"/"success" per call site.
BTN_PRIMARY = "primary"         # main action (strongest blue)
BTN_SECONDARY = "secondary"     # secondary / muted action
BTN_NEUTRAL = "outline-secondary"  # cancel / neutral outline variant

# Status-text colours (§8.3 of GUI_CONVENTIONS.md).
# Never use "var(--bs-success)" (Bootstrap green).
COLOR_POSITIVE = "#EAAE47"      # amber — positive / success state
COLOR_MUTED    = "var(--color9)"  # hint / secondary

# ── Hint / caption text ───────────────────────────────────────────────────────
HINT: dict = {"fontSize": FONT_SM, "color": COLOR_MUTED}
HINT_SM: dict = {"fontSize": FONT_XS, "color": COLOR_MUTED, "marginTop": "2px"}

# ── Inline control rows (flex, no margin — margin belongs on a wrapper div) ───
# §3: never put margin* inside a flex-row dict; place marginBottom on an outer
# non-flex wrapper div if vertical spacing between rows is needed.
INLINE_CTRL_ROW: dict = {
    "display": "flex", "alignItems": "center", "flexWrap": "nowrap", "gap": "0.5rem",
}
# Prescribed RadioItems prop dicts for inline (horizontal) radio groups (§8.2).
RADIO_INLINE_INPUT: dict = {
    "verticalAlign": "middle", "marginTop": "-2px",
    "marginRight": "0.4rem", "cursor": "pointer",
}
RADIO_INLINE_LABEL: dict = {
    "verticalAlign": "middle", "marginRight": "1.4rem", "cursor": "pointer",
}

# ── formgen form rows ─────────────────────────────────────────────────────────
FORM_HINT: dict = {"fontSize": FONT_SM, "color": COLOR_MUTED, "padding": "2px 0"}
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

# ── Tab strip sizing ──────────────────────────────────────────────────────────
# Applied globally via .nav-tabs .nav-link in assets/styles.css.
# All dbc.Tabs strips in the app inherit these; never set per-tab literally.
TAB_NAV_PADDING     = "4px 12px"   # padding on each nav-link button
TAB_NAV_LINE_HEIGHT = "1.4"        # line-height on each nav-link button

# ── Section headers ───────────────────────────────────────────────────────────
SECTION_HEADER: dict = {
    "fontSize": "0.9rem", "fontWeight": 600,
    "margin": f"{SECTION_GAP} 0 0.2rem",
}
