"""Q8 — End-to-end journeys (browser-driven, marked slow).

Four journeys proving the wiring rather than finding logic bugs.  These are
excluded from the default CI run; invoke with ``pytest -m slow``.

Each journey must clean up after itself (tmp_path, no writes to ~).  If a
journey needs real tomogram data it is marked ``needs_data`` too.

Prerequisites
-------------
- ``pip install pytest-dash`` or similar Dash testing harness
- A local Chrome/Chromium with matching chromedriver on PATH

Current state: journeys are defined but skipped until the Dash testing
infrastructure is set up (doc 6).  The definitions here record the intended
contract so it cannot be forgotten.
"""
from __future__ import annotations

import pytest


# ── 1. Load → pool → tool ─────────────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.needs_data
def test_journey_load_pool_tool(tmp_path):
    """Load a motl in the editor; it appears in the pool; a tool's motlsource
    picker offers it; selecting it populates the tool."""
    pytest.skip("Requires browser + motl data file — run manually or in slow CI")


# ── 2. Builder → field ────────────────────────────────────────────────────────

@pytest.mark.slow
def test_journey_builder_field(tmp_path):
    """Open the angles Build… modal, set parameters, 'Use this file'; the path
    lands in the outer input and the modal closes.

    This is the journey that would have caught the orphaned rotation modal.
    """
    pytest.skip("Requires browser — run manually or in slow CI")


# ── 3. Settings → restyle ─────────────────────────────────────────────────────

@pytest.mark.slow
def test_journey_settings_restyle(tmp_path):
    """Render a graph, change the font and palette in Graph Settings; confirm
    the existing graph restyles without re-triggering its own callback (§4.3).
    """
    pytest.skip("Requires browser — run manually or in slow CI")


# ── 4. Operate → record → export ─────────────────────────────────────────────

@pytest.mark.slow
@pytest.mark.needs_data
def test_journey_operate_record_export(tmp_path):
    """Apply an operation, trigger a failure, save; hit 'Generate script';
    assert the downloaded file matches the projection called directly and that
    the session record shows the failure.
    """
    pytest.skip("Requires browser + motl data file — run manually or in slow CI")
