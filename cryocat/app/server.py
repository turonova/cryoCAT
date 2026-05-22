"""
Entry point for the combined cryoCAT Dash application.

Two independent Dash apps are mounted via Werkzeug's DispatcherMiddleware:
  - Tango (twist analysis): served at /tango/
  - Suite (motl editor):    served at /

Usage:
    python -m cryocat.app.server

Or with gunicorn:
    gunicorn "cryocat.app.server:application"
"""

from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

from cryocat.app.tango.app import app as tango_app
from cryocat.app.suite.app import app as suite_app

# Mount tango under /tango, suite at root.
application = DispatcherMiddleware(
    suite_app.server,
    {"/tango": tango_app.server},
)


def main():
    run_simple(
        "0.0.0.0",
        8050,
        application,
        use_reloader=False,
        use_debugger=False,
    )


if __name__ == "__main__":
    main()
