from __future__ import annotations

from .node import V7NeuroNode


def _mat_vec(m: list[list[float]], v: list[float]) -> list[float]:
    return [sum(m[i][j] * v[j] for j in range(len(v))) for i in range(len(m))]


def _vec_add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_scale(a: list[float], s: float) -> list[float]:
    return [x * s for x in a]


class NDimensionalAtlas:
    def __init__(self, num_nodes: int = 4, dim: int = 6):
        self.num_nodes = num_nodes
        self.dim = dim
        self.nodes = [V7NeuroNode(i, dim=dim) for i in range(num_nodes)]
        self.states = [n.init_state() for n in self.nodes]
        self.A = [[1.0 if i != j else 0.0 for j in range(num_nodes)] for i in range(num_nodes)]
        self.Q = [[self._qij(i, j) for j in range(num_nodes)] for i in range(num_nodes)]

    def _qij(self, i: int, j: int) -> list[list[float]]:
        m = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
        shift = (j - i) % self.dim
        scale = 1.0 if i == j else 0.85
        for r in range(self.dim):
            c = (r + shift) % self.dim
            m[r][c] = scale
        return m

    def step(self, up_signals: list[float], world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool):
        old = [s.dynamic[:] for s in self.states]
        new_states = []
        for i, node in enumerate(self.nodes):
            neigh = [0.0] * self.dim
            for j in range(self.num_nodes):
                if i == j:
                    continue
                proj = _mat_vec(self.Q[i][j], old[j])
                neigh = _vec_add(neigh, _vec_scale(proj, self.A[i][j]))
            if self.num_nodes > 1:
                neigh = _vec_scale(neigh, 1.0 / (self.num_nodes - 1))
            new_states.append(node.step(self.states[i], neigh, up_signals[i], world_drive, body, tissue, context, offline_trigger))
        self.states = new_states

    def reconstruct_global(self) -> list[float]:
        xhat = [0.0] * self.dim
        for i, s in enumerate(self.states):
            xhat = _vec_add(xhat, _mat_vec(self.Q[i][i], s.dynamic))
        return _vec_scale(xhat, 1.0 / max(1, self.num_nodes))

    def local_closure_errors(self) -> list[float]:
        xhat = self.reconstruct_global()
        errs = []
        for i, s in enumerate(self.states):
            target = _mat_vec(self.Q[i][i], xhat)
            errs.append(sum(abs(a - b) for a, b in zip(s.dynamic, target)) / self.dim)
        return errs

    def total_coherence(self) -> float:
        vals = [s.local_metrics.get("C_total", 0.0) for s in self.states]
        return sum(vals) / max(1, len(vals))
