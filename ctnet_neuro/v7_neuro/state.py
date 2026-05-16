from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class LocalMetrics:
    C_local: float = 0.0
    C_neighbors: float = 0.0
    C_body: float = 0.0
    C_tissue: float = 0.0
    C_context: float = 0.0
    C_memory: float = 0.0
    C_offline: float = 0.0
    C_total: float = 0.0
    mass: float = 0.0


@dataclass
class HolonicNodeState:
    node_id: int
    dynamic: float = 0.0
    symbolic: dict[str, float] = field(default_factory=dict)
    body: dict[str, float] = field(default_factory=dict)
    tissue: dict[str, float] = field(default_factory=dict)
    context: dict[str, Any] = field(default_factory=dict)
    executive: dict[str, Any] = field(default_factory=dict)
    memory: dict[str, Any] = field(default_factory=dict)
    offline: dict[str, Any] = field(default_factory=dict)
    regime: dict[str, Any] = field(default_factory=dict)
    verifier: dict[str, Any] = field(default_factory=dict)
    semantic_mass: float = 0.0
    local_babel: dict[str, Any] = field(default_factory=dict)
    local_metrics: LocalMetrics = field(default_factory=LocalMetrics)


@dataclass
class V7NeuroGlobalState:
    nodes: list[HolonicNodeState]
    topology: list[list[float]]
    step_count: int = 0
    global_readout: str = ""
    global_coherence: float = 0.0
    global_closure_error: float = 0.0
