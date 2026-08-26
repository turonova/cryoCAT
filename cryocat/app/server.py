"""
Entry point for the cryoCAT Dash application.

Tango twist analysis is now served as a tab inside the suite (at /tango).
The separate tango Dash app and DispatcherMiddleware have been removed.

Usage:
    python -m cryocat.app.server

The server binds to 127.0.0.1 (localhost) only.  Do NOT expose it on 0.0.0.0
or through a multi-worker process manager: the GUI console allows arbitrary
Python execution as the current user (§12 of GUI_CONVENTIONS.md).
"""

from werkzeug.serving import run_simple

from cryocat.app import session as _session

_session.start_session()

from cryocat.app import discovery as _discovery

_discovery.load_registry()

from cryocat.app.suite.app import app as suite_app

application = suite_app.server


def main():
    run_simple(
        "127.0.0.1",
        8050,
        application,
        use_reloader=False,
        use_debugger=False,
        threaded=True,
    )


if __name__ == "__main__":
    main()
