"""Graph pool — server-side figure/spec store and pure state reducers (C2)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

_graph_payloads: dict[str, dict] = {}


class GraphPayloadMissing(KeyError):
    pass


@dataclass
class GraphPoolState:
    registry: dict = field(default_factory=dict)
    next_id: int = 1

    @classmethod
    def from_stores(cls, registry, next_id) -> "GraphPoolState":
        return cls(registry=dict(registry or {}), next_id=int(next_id or 1))

    def to_stores(self) -> tuple[dict, int]:
        return dict(self.registry), self.next_id


def insert_graph_entry(
    state: GraphPoolState,
    payload: dict,
    *,
    label: str,
    kind: str,
) -> tuple[GraphPoolState, str]:
    """Insert a new entry; return (new_state, graph_id)."""
    graph_id = f"graph_{state.next_id}"
    _graph_payloads[graph_id] = copy.deepcopy(payload)
    entry = {"graph_id": graph_id, "label": label, "kind": kind}
    new_registry = {**state.registry, graph_id: entry}
    return GraphPoolState(registry=new_registry, next_id=state.next_id + 1), graph_id


def get_graph_payload(graph_id: str) -> dict:
    try:
        return _graph_payloads[graph_id]
    except KeyError:
        raise GraphPayloadMissing(graph_id)


def remove_graph_entry(state: GraphPoolState, graph_id: str) -> GraphPoolState:
    _graph_payloads.pop(graph_id, None)
    new_registry = {k: v for k, v in state.registry.items() if k != graph_id}
    return GraphPoolState(registry=new_registry, next_id=state.next_id)


def duplicate_graph_entry(state: GraphPoolState, graph_id: str) -> tuple[GraphPoolState, str]:
    payload = get_graph_payload(graph_id)
    old_entry = state.registry.get(graph_id, {})
    label = f"{old_entry.get('label', graph_id)} (copy)"
    kind = old_entry.get("kind", "spec")
    return insert_graph_entry(state, payload, label=label, kind=kind)


def rename_graph_entry(state: GraphPoolState, graph_id: str, new_label: str) -> GraphPoolState:
    if graph_id not in state.registry:
        return state
    new_registry = {
        k: ({**v, "label": new_label} if k == graph_id else v)
        for k, v in state.registry.items()
    }
    return GraphPoolState(registry=new_registry, next_id=state.next_id)
