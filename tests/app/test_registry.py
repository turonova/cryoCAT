"""Tests for cryocat.app.components.registry (T7)."""
from dataclasses import fields

import pytest

from cryocat.app.components.registry import Registry


# ── Registry core behaviour ───────────────────────────────────────────────────

class TestRegistryAdd:
    def test_returns_distinct_keys(self):
        reg = Registry("x")
        k1 = reg.add(object())
        k2 = reg.add(object())
        assert k1 != k2

    def test_keys_not_reused_after_remove(self):
        reg = Registry("x")
        k1 = reg.add(object())
        reg.remove(k1)
        k2 = reg.add(object())
        assert k2 != k1

    def test_key_uses_prefix(self):
        reg = Registry("surf")
        k = reg.add(object())
        assert k.startswith("surf-")

    def test_len_increments(self):
        reg = Registry("x")
        reg.add(object())
        reg.add(object())
        assert len(reg) == 2


class TestRegistryGet:
    def test_get_returns_stored_object(self):
        reg = Registry("x")
        obj = object()
        k = reg.add(obj)
        assert reg.get(k) is obj

    def test_get_removed_returns_none(self):
        reg = Registry("x")
        k = reg.add(object())
        reg.remove(k)
        assert reg.get(k) is None

    def test_get_never_present_returns_none(self):
        reg = Registry("x")
        assert reg.get("x-99999") is None


class TestRegistryReplace:
    def test_replace_swaps_object(self):
        reg = Registry("x")
        k = reg.add("old")
        reg.replace(k, "new")
        assert reg.get(k) == "new"

    def test_replace_absent_raises_key_error(self):
        reg = Registry("x")
        with pytest.raises(KeyError):
            reg.replace("x-99999", object())


class TestRegistryRemove:
    def test_remove_drops_key(self):
        reg = Registry("x")
        k = reg.add(object())
        reg.remove(k)
        assert reg.get(k) is None

    def test_remove_absent_is_noop(self):
        reg = Registry("x")
        reg.remove("x-99999")  # must not raise


class TestRegistryClear:
    def test_clear_empties_store(self):
        reg = Registry("x")
        reg.add(object())
        reg.add(object())
        reg.clear()
        assert len(reg) == 0
        assert reg.keys() == []


class TestRegistryMaxItems:
    def test_max_items_1_evicts_on_add(self):
        reg = Registry("x", max_items=1)
        k1 = reg.add(object())
        k2 = reg.add(object())
        assert reg.get(k1) is None
        assert reg.get(k2) is not None
        assert len(reg) == 1

    def test_max_items_2_evicts_oldest(self):
        reg = Registry("x", max_items=2)
        k1 = reg.add(object())
        k2 = reg.add(object())
        k3 = reg.add(object())
        assert reg.get(k1) is None
        assert reg.get(k2) is not None
        assert reg.get(k3) is not None

    def test_max_items_1_key_not_reused(self):
        reg = Registry("x", max_items=1)
        k1 = reg.add(object())
        k2 = reg.add(object())
        assert k1 != k2


# ── Handle dataclass schemas ──────────────────────────────────────────────────

class TestSurfaceHandleSchema:
    def test_field_names(self):
        from cryocat.app.components.surface_registry import SurfaceHandle
        assert {f.name for f in fields(SurfaceHandle)} == {
            "label", "representation", "n_elements",
            "parent_id", "visible", "has_curvatures",
        }


class TestParametricHandleSchema:
    def test_field_names(self):
        from cryocat.app.components.parametric_registry import ParametricHandle
        assert {f.name for f in fields(ParametricHandle)} == {
            "column_name", "surface_type", "n_quadrics", "source",
        }


class TestMembraneHandleSchema:
    def test_field_names(self):
        from cryocat.app.components.memthick_registry import MembraneHandle
        assert {f.name for f in fields(MembraneHandle)} == {
            "membrane", "n_rows", "n_resolved", "n_unresolved",
            "n_finite_inflection_thickness_nm", "by_detection_mode",
            "pixel_size_nm", "thickness_csv",
        }
