from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class HolonicNodeState:
    node_id: int
    dynamic: list[float] = field(default_factory=lambda: [0.0] * 6)
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
    local_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class V7NeuroGlobalState:
    nodes: list[HolonicNodeState]
    topology: list[list[float]]
    transforms: list[list[list[float]]]
    step_count: int = 0
