from __future__ import annotations
import math
from .node import V7NeuroNode


class NDimensionalAtlas:
    def __init__(self, nodes: list[V7NeuroNode], recirculation_gain: float = 0.35):
        self.nodes = nodes
        n = len(nodes)
        self.A = [[0.0 if i == j else 1.0 / max(1, n - 1) for j in range(n)] for i in range(n)]
        self.Q = [[1.0 if i == j else 0.9 for j in range(n)] for i in range(n)]
        self.recirculation_gain = recirculation_gain

    def step(self) -> list[list[float]]:
        projections = [n.project_vector() for n in self.nodes]
        recirc = []
        for i in range(len(self.nodes)):
            acc = [0.0] * len(projections[i])
            for j in range(len(self.nodes)):
                if i == j: continue
                for k, v in enumerate(projections[j]):
                    acc[k] += self.A[i][j] * self.Q[i][j] * v
            recirc.append(acc)
        return recirc

    def reconstruct_global(self) -> list[float]:
        vecs = [n.project_vector() for n in self.nodes]
        d = len(vecs[0])
        return [sum(v[k] for v in vecs) / len(vecs) for k in range(d)]

    def local_closure_errors(self) -> list[float]:
        xh = self.reconstruct_global()
        out = []
        for n in self.nodes:
            err = sum(abs(a - b) for a, b in zip(n.project_vector(), xh)) / max(1, len(xh))
            out.append(err)
        return out

    def total_coherence(self) -> float:
        vals = [n.state.local_metrics.get("C_total", 0.0) for n in self.nodes]
        return sum(vals) / max(1, len(vals))
