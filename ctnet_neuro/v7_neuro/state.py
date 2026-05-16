from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass
class HolonicNodeState:
    node_id: int
    dynamic: list[float]
    symbolic: dict[str, float]
    body: dict[str, float]
    tissue: dict[str, float]
    context: dict[str, Any]
    executive: dict[str, Any]
    memory: dict[str, Any]
    offline: dict[str, Any]
    regime: dict[str, Any]
    verifier: dict[str, Any]
    semantic_mass: float
    local_babel: dict[str, Any]
    local_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class V7NeuroConfig:
    n_nodes: int = 4
    vector_dim: int = 12
    memory_capacity: int = 32
    recirculation_gain: float = 0.35
    closure_tolerance: float = 0.35
    offline_threshold: float = 0.65
