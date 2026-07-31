"""
Entry point for the combined cryoCAT Dash application.

Two independent Dash apps are mounted via Werkzeug's DispatcherMiddleware:
  - Tango (twist analysis): served at /tango/
  - Suite (motl editor):    served at /

Usage:
    python -m cryocat.app.server

The server binds to 127.0.0.1 (localhost) only.  Do NOT expose it on 0.0.0.0
or through a multi-worker process manager: the GUI console allows arbitrary
Python execution as the current user (§12 of GUI_CONVENTIONS.md).
"""

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from cryocat.app import session as _session
_session.start_session()

# Populate GUI_REGISTRY before importing apps that read it at module level.
from cryocat.app import discovery as _discovery
_discovery.load_registry()

from cryocat.app.tango.app import app as tango_app
from cryocat.app.suite.app import app as suite_app

# Mount tango under /tango, suite at root.
application = DispatcherMiddleware(
    suite_app.server,
    {"/tango": tango_app.server},
)


def main():
    run_simple(
        "127.0.0.1",
        8050,
        application,
        use_reloader=False,
        use_debugger=False,
    )


if __name__ == "__main__":
    main()
