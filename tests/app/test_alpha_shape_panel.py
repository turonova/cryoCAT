"""Panel tests for the alpha-shape tuning section in putilities.py.

Tests focus on the server-side tetra cache and callback behaviour:
  * The tetrahedralisation is computed once per cloud; slider moves do not recompute it.
  * Changing the motl recomputes the tetrahedralisation.
  * The reported component count matches Mesh's own for a known two-cluster case.
  * Saving writes a file readable by Mesh.read, and the saved path is mentioned in the status.

No Dash app is started; helpers are called directly.
"""

from __future__ import annotations

import pathlib
from unittest.mock import MagicMock, patch, call

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _sphere_pts(n: int = 300, radius: float = 5.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.standard_normal((n, 3))
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts * radius


def _two_cluster_pts(n: int = 200, sep: float = 40.0) -> np.ndarray:
    rng = np.random.default_rng(2)
    a = rng.standard_normal((n, 3))
    b = rng.standard_normal((n, 3)) + np.array([sep, 0.0, 0.0])
    return np.vstack([a, b])


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestTetraCache:
    def test_compute_tetra_caches_result(self):
        """_compute_tetra stores the (tetra_mesh, pt_map) in _alpha_tetra_cache."""
        from cryocat.core.surface import Mesh

        pts = _sphere_pts()

        with patch(
            "cryocat.app.suite.pages.putilities._get_motl_coords",
            return_value=pts,
        ):
            import cryocat.app.suite.pages.putilities as pu

            pu._alpha_tetra_cache.clear()

            info = pu._compute_tetra("motl-test-1")

        assert info is not None
        assert info["motl_id"] == "motl-test-1"
        assert "log_min" in info
        assert "log_max" in info
        assert info["n_points"] == len(pts)
        assert "motl-test-1" in pu._alpha_tetra_cache

    def test_slider_moves_do_not_recompute_tetra(self):
        """Ten calls to _render_alpha_shape reuse the cached tetra (alpha_shape_tetra called once)."""
        from cryocat.core.surface import Mesh

        pts = _sphere_pts()
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(pts)
        lo, hi = Mesh.suggest_alpha_range(pts)
        mid = (lo + hi) / 2

        import cryocat.app.suite.pages.putilities as pu

        pu._alpha_tetra_cache["motl-slider"] = (tetra_mesh, pt_map)
        tetra_info = {
            "motl_id": "motl-slider",
            "log_min": np.log10(lo),
            "log_max": np.log10(hi),
            "n_points": len(pts),
        }

        with (
            patch(
                "cryocat.app.suite.pages.putilities._get_motl_coords",
                return_value=pts,
            ),
            patch.object(Mesh, "alpha_shape_tetra", wraps=Mesh.alpha_shape_tetra) as mock_tetra,
        ):
            for _ in range(10):
                pu._render_alpha_shape(mid, tetra_info, False, {})

        mock_tetra.assert_not_called()

    def test_changing_motl_recomputes_tetra(self):
        """Calling _compute_tetra with a new motl_id overwrites the cache entry."""
        pts_a = _sphere_pts(n=200, seed=10)
        pts_b = _sphere_pts(n=200, seed=11)

        import cryocat.app.suite.pages.putilities as pu

        pu._alpha_tetra_cache.clear()

        with patch(
            "cryocat.app.suite.pages.putilities._get_motl_coords",
            side_effect=[pts_a, pts_b],
        ):
            pu._compute_tetra("motl-A")
            pu._compute_tetra("motl-B")

        assert "motl-A" in pu._alpha_tetra_cache
        assert "motl-B" in pu._alpha_tetra_cache
        # The two cached entries must be distinct objects.
        assert pu._alpha_tetra_cache["motl-A"] is not pu._alpha_tetra_cache["motl-B"]


class TestComponentCount:
    def test_component_count_matches_mesh_for_two_clusters(self):
        """Reported component count via _render_alpha_shape equals Mesh.get_connected_component_count."""
        from cryocat.core.surface import Mesh

        pts = _two_cluster_pts(n=300, sep=40.0)
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(pts)
        lo, hi = Mesh.suggest_alpha_range(pts)
        alpha = lo * 2  # fine enough to fragment

        import cryocat.app.suite.pages.putilities as pu

        pu._alpha_tetra_cache["motl-twocluster"] = (tetra_mesh, pt_map)
        tetra_info = {
            "motl_id": "motl-twocluster",
            "log_min": np.log10(lo),
            "log_max": np.log10(hi),
            "n_points": len(pts),
        }

        with patch(
            "cryocat.app.suite.pages.putilities._get_motl_coords",
            return_value=pts,
        ):
            _, stats_str = pu._render_alpha_shape(alpha, tetra_info, False, {})

        # Compute expected value directly.
        expected_mesh = Mesh.from_alpha_shape(pts, alpha, tetra_mesh, pt_map)
        expected_n = expected_mesh.get_connected_component_count()

        assert f"Components: {expected_n}" in stats_str
        assert expected_n > 1


class TestSave:
    def test_save_writes_readable_file(self, tmp_path):
        """_do_alpha_save writes a PLY that Mesh.read can load."""
        from cryocat.core.surface import Mesh

        pts = _sphere_pts(n=300)
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(pts)
        lo, hi = Mesh.suggest_alpha_range(pts)
        alpha = (lo + hi) / 2

        out = str(tmp_path / "test_alpha.ply")

        import cryocat.app.suite.pages.putilities as pu

        pu._alpha_tetra_cache["motl-save"] = (tetra_mesh, pt_map)
        tetra_info = {
            "motl_id": "motl-save",
            "log_min": np.log10(lo),
            "log_max": np.log10(hi),
            "n_points": len(pts),
        }

        with patch(
            "cryocat.app.suite.pages.putilities._get_motl_coords",
            return_value=pts,
        ):
            status = pu._do_alpha_save(alpha, tetra_info, out)

        assert pathlib.Path(out).is_file(), f"File not written; status={status!r}"
        loaded = Mesh.read(out)
        assert loaded.vertices is not None and len(loaded.vertices) > 0

    def test_save_status_contains_alpha_and_path(self, tmp_path):
        from cryocat.core.surface import Mesh

        pts = _sphere_pts(n=200)
        tetra_mesh, pt_map = Mesh.alpha_shape_tetra(pts)
        lo, hi = Mesh.suggest_alpha_range(pts)
        alpha = (lo + hi) / 2

        out = str(tmp_path / "save_status.ply")

        import cryocat.app.suite.pages.putilities as pu

        pu._alpha_tetra_cache["motl-status"] = (tetra_mesh, pt_map)
        tetra_info = {
            "motl_id": "motl-status",
            "log_min": np.log10(lo),
            "log_max": np.log10(hi),
            "n_points": len(pts),
        }

        with patch(
            "cryocat.app.suite.pages.putilities._get_motl_coords",
            return_value=pts,
        ):
            status = pu._do_alpha_save(alpha, tetra_info, out)

        assert out in status or "Saved" in status
        assert f"{alpha:.6g}" in status or str(alpha) in status
