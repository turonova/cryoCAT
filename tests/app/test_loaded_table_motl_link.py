"""Tests for the loaded-table / motl-link subsystem.

Covers check_motl_overlap (full match, partial match, no overlap) and
has_source_motl gate logic (tables without a source motl disable the gate;
setting one re-enables it).
"""
from __future__ import annotations

import pandas as pd
import pytest


# ── check_motl_overlap ────────────────────────────────────────────────────────

class TestCheckMotlOverlap:
    """Pure overlap check; uses a mock pool so no Dash app is needed."""

    def _make_df(self, ids: list[int]) -> pd.DataFrame:
        return pd.DataFrame({"qp_subtomo_id": ids, "nn_dist": [1.0] * len(ids)})

    def _mock_pool(self, monkeypatch, motl_ids: list[int]):
        """Patch pool.get_rows to return a motl with the given subtomo_ids."""
        motl_rows = [{"subtomo_id": i} for i in motl_ids]

        import cryocat.app.suite.pages._motl_link as _ml
        original_get_rows = None

        def fake_get_rows(motl_id):
            if motl_id == "motl_mock":
                return motl_rows
            from cryocat.app.pool import PoolPayloadMissing
            raise PoolPayloadMissing(motl_id)

        monkeypatch.setattr("cryocat.app.suite.pages._motl_link.check_motl_overlap.__globals__['__builtins__']", None, raising=False)

        import cryocat.app.pool as _pool
        monkeypatch.setattr(_pool, "get_rows", fake_get_rows)

    def test_full_overlap(self, monkeypatch):
        """All table ids present in motl → matched == total, ✓ in message."""
        from cryocat.app.suite.pages._motl_link import check_motl_overlap
        from cryocat.app.pool import PoolPayloadMissing

        motl_rows = [{"subtomo_id": i} for i in [1, 2, 3, 4, 5]]

        def fake_get_rows(mid):
            if mid == "motl_mock":
                return motl_rows
            raise PoolPayloadMissing(mid)

        import cryocat.app.pool as _pool
        monkeypatch.setattr(_pool, "get_rows", fake_get_rows)

        df = self._make_df([1, 2, 3, 4, 5])
        matched, total, msg = check_motl_overlap(df, "qp_subtomo_id", "motl_mock")

        assert matched == 5
        assert total == 5
        assert "✓" in msg

    def test_partial_overlap(self, monkeypatch):
        """Only some table ids present in motl → ⚠ in message."""
        from cryocat.app.suite.pages._motl_link import check_motl_overlap
        from cryocat.app.pool import PoolPayloadMissing

        motl_rows = [{"subtomo_id": i} for i in [1, 2, 3, 4, 5]]

        def fake_get_rows(mid):
            if mid == "motl_mock":
                return motl_rows
            raise PoolPayloadMissing(mid)

        import cryocat.app.pool as _pool
        monkeypatch.setattr(_pool, "get_rows", fake_get_rows)

        df = self._make_df([1, 2, 3, 99, 100])
        matched, total, msg = check_motl_overlap(df, "qp_subtomo_id", "motl_mock")

        assert matched == 3
        assert total == 5
        assert "⚠" in msg

    def test_no_overlap(self, monkeypatch):
        """No table ids in motl → matched == 0, ✗ in message."""
        from cryocat.app.suite.pages._motl_link import check_motl_overlap
        from cryocat.app.pool import PoolPayloadMissing

        motl_rows = [{"subtomo_id": i} for i in [10, 20, 30]]

        def fake_get_rows(mid):
            if mid == "motl_mock":
                return motl_rows
            raise PoolPayloadMissing(mid)

        import cryocat.app.pool as _pool
        monkeypatch.setattr(_pool, "get_rows", fake_get_rows)

        df = self._make_df([1, 2, 3])
        matched, total, msg = check_motl_overlap(df, "qp_subtomo_id", "motl_mock")

        assert matched == 0
        assert total == 3
        assert "✗" in msg

    def test_missing_motl(self, monkeypatch):
        """PoolPayloadMissing → (0, 0, error message); does not raise."""
        from cryocat.app.suite.pages._motl_link import check_motl_overlap
        from cryocat.app.pool import PoolPayloadMissing

        def fake_get_rows(mid):
            raise PoolPayloadMissing(mid)

        import cryocat.app.pool as _pool
        monkeypatch.setattr(_pool, "get_rows", fake_get_rows)

        df = self._make_df([1, 2, 3])
        matched, total, msg = check_motl_overlap(df, "qp_subtomo_id", "missing_motl")

        assert matched == 0
        assert total == 0
        assert "not found" in msg.lower() or "missing" in msg.lower()


# ── has_source_motl gate ──────────────────────────────────────────────────────

class TestHasSourceMotl:
    """Gate logic for loaded tables: setting motl_links enables operations.

    Uses ref dicts constructed directly (no Dash app needed).
    The test verifies that:
    * A loaded ref without motl_links set (=None) → disabled (False).
    * Setting motl_links to a role→id mapping → enabled (True).
    * A computed ref (motl_links set at compute time) → always enabled (True).
    """

    def test_loaded_no_source_is_gated(self):
        """Loaded table with motl_links=None → has_source_motl returns False."""
        from cryocat.app.suite.pages._motl_link import has_source_motl
        ref = {"table_id": "table_1", "n_rows": 100, "id_column": "qp_subtomo_id",
               "motl_links": None}
        assert has_source_motl(ref) is False

    def test_loaded_with_source_is_enabled(self):
        """Loaded table with motl_links set → has_source_motl returns True."""
        from cryocat.app.suite.pages._motl_link import has_source_motl
        ref = {"table_id": "table_1", "n_rows": 100, "id_column": "qp_subtomo_id",
               "motl_links": {"source": ["motl_3"]}}
        assert has_source_motl(ref) is True

    def test_computed_table_not_gated(self):
        """Computed table (motl_links set at compute time) → has_source_motl returns True."""
        from cryocat.app.suite.pages._motl_link import has_source_motl
        ref = {"table_id": "table_2", "n_rows": 50, "id_column": "qp_subtomo_id",
               "motl_links": {"source": ["motl_7"]}}
        assert has_source_motl(ref) is True

    def test_none_ref_not_gated(self):
        """None ref → has_source_motl returns False (no table, no table-to-motl)."""
        from cryocat.app.suite.pages._motl_link import has_source_motl
        assert has_source_motl(None) is False

    def test_setting_source_motl_afterwards_enables(self):
        """Updating motl_links on a loaded table enables the gate."""
        from cryocat.app.suite.pages._motl_link import has_source_motl

        ref_loaded = {"table_id": "table_3", "n_rows": 200, "id_column": "qp_subtomo_id",
                      "motl_links": None}
        assert has_source_motl(ref_loaded) is False, "should be gated before setting source"

        ref_updated = {**ref_loaded, "motl_links": {"source": ["motl_1"]}}
        assert has_source_motl(ref_updated) is True, "should be enabled after setting source"
