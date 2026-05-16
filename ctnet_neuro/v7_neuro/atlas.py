from __future__ import annotations

import math

from .node import V7NeuroNode
from .state import HolonicNodeState


class NDimensionalAtlas:
    def __init__(self, num_nodes: int = 4):
        self.num_nodes = num_nodes
        self.nodes = [V7NeuroNode(i) for i in range(num_nodes)]
        self.states = [n.init_state() for n in self.nodes]
        self.A = [[1.0 if i != j else 0.0 for j in range(num_nodes)] for i in range(num_nodes)]
        self.Q = [[1.0 if i == j else 0.5 for j in range(num_nodes)] for i in range(num_nodes)]

    def step(self, up_signals: list[float], world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool):
        old = [s.dynamic.get("activation", 0.0) for s in self.states]
        new_states = []
        for i, node in enumerate(self.nodes):
            neigh = 0.0
            for j in range(self.num_nodes):
                neigh += self.A[i][j] * self.Q[i][j] * old[j]
            neigh /= max(1, self.num_nodes - 1)
            new_states.append(node.step(self.states[i], neigh, up_signals[i], world_drive, body, tissue, context, offline_trigger))
        self.states = new_states

    def reconstruct_global(self) -> list[float]:
        act = [s.dynamic.get("activation", 0.0) for s in self.states]
        mean = sum(act) / max(1, len(act))
        return [mean for _ in range(self.num_nodes)]

    def local_closure_errors(self) -> list[float]:
        xhat = self.reconstruct_global()
        errs = []
        for i,s in enumerate(self.states):
            local = s.dynamic.get("activation", 0.0)
            errs.append(abs(local - self.Q[i][i] * xhat[i]))
        return errs

    def total_coherence(self) -> float:
        vals = [s.local_metrics.get("C_total", 0.0) for s in self.states]
        return sum(vals)/max(1, len(vals))
