from __future__ import annotations

import math

from .node import V7NeuroNode


class NDimensionalAtlas:
    def __init__(self, node_count: int):
        self.nodes = [V7NeuroNode(i) for i in range(node_count)]
        self.A = [[0.0 for _ in range(node_count)] for _ in range(node_count)]
        for i in range(node_count):
            self.A[i][i] = 1.0
            self.A[i][(i - 1) % node_count] = 0.5
            self.A[i][(i + 1) % node_count] = 0.5

    def neighbor_projection(self, i: int) -> float:
        return sum(self.A[i][j] * self.nodes[j].state.dynamic for j in range(len(self.nodes))) / max(1e-6, sum(self.A[i]))

    def step(self, up_signals: list[float], world_signal: float, body: dict, tissue: dict, context: dict) -> None:
        for i, node in enumerate(self.nodes):
            node.step(self.neighbor_projection(i), up_signals[i], world_signal, body, tissue, context)

    def reconstruct_global(self) -> list[float]:
        return [n.state.dynamic for n in self.nodes]

    def local_closure_errors(self) -> list[float]:
        xhat = sum(self.reconstruct_global()) / len(self.nodes)
        return [abs(n.state.dynamic - xhat) for n in self.nodes]

    def total_coherence(self) -> float:
        vals = [n.state.local_metrics.C_total for n in self.nodes]
        return sum(vals) / len(vals) if vals else 0.0

    def closure_error(self) -> float:
        errs = self.local_closure_errors()
        return math.sqrt(sum(e * e for e in errs) / len(errs)) if errs else 0.0
