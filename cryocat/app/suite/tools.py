"""Tool registry for the Suite app.

Single source of truth that drives both the router and the top-nav tool
selector in :mod:`cryocat.app.suite.app`. Adding a tool = one entry here +
one page module exposing ``layout`` (attribute) and ``register_callbacks(app)``.

Each entry:
    id      -- short stable identifier
    label   -- text shown in the tool selector
    path    -- route the tool is served at (also the NavLink href)
    module  -- dotted path to the page module (imported lazily by the router)
"""

TOOLS = [
    {"id": "editor",    "label": "Motl Editor",       "path": "/motl",      "module": "cryocat.app.suite.pages.pmotl"},
    {"id": "nn",        "label": "Nearest Neighbor",  "path": "/nn",        "module": "cryocat.app.suite.pages.pnn"},
    {"id": "sta",       "label": "STA",               "path": "/sta",       "module": "cryocat.app.suite.pages.psta"},
    {"id": "pana",      "label": "Peak Analysis",     "path": "/pana",      "module": "cryocat.app.suite.pages.ppana"},
    {"id": "volume",    "label": "Mask generation",   "path": "/volume",    "module": "cryocat.app.suite.pages.pvolume"},
    {"id": "utilities", "label": "Utilities",         "path": "/utilities", "module": "cryocat.app.suite.pages.putilities"},
    {"id": "surfaces",  "label": "Surfaces",          "path": "/surfaces",  "module": "cryocat.app.suite.pages.pstructure"},
    {"id": "memthick",  "label": "Membrane thickness", "path": "/memthick",  "module": "cryocat.app.suite.pages.pmemthick"},
    {"id": "complexes",   "label": "Complexes",           "path": "/complexes",   "module": "cryocat.app.suite.pages.pcomplexes"},
    {"id": "tango",       "label": "Tango",               "path": "/tango",       "module": "cryocat.app.suite.pages.ptango"},
    {"id": "datapool",     "label": "Tables",             "path": "/datapool",     "module": "cryocat.app.suite.pages.pdatapool"},
    {"id": "graphs",       "label": "Graphs",             "path": "/graphs",       "module": "cryocat.app.suite.pages.pgraphs"},
]

# Route used when the URL is empty / root / unknown.
DEFAULT_PATH = "/motl"
