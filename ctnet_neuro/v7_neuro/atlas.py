from __future__ import annotations

from .node import V7NeuroNode


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(row[j] * vector[j] for j in range(len(vector))) for row in matrix]


def _vadd(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def _vscale(a: list[float], s: float) -> list[float]:
    return [s * x for x in a]


def _vmean(vectors: list[list[float]]) -> list[float]:
    n = len(vectors)
    out = [0.0 for _ in vectors[0]]
    for v in vectors:
        out = _vadd(out, v)
    return _vscale(out, 1.0 / max(1, n))


class NDimensionalAtlas:
    def __init__(self, num_nodes: int = 4, dim: int = 6):
        self.num_nodes = num_nodes
        self.dim = dim
        self.nodes = [V7NeuroNode(i, dim=dim) for i in range(num_nodes)]
        self.states = [n.init_state() for n in self.nodes]
        self.A = [[1.0 if i != j else 0.0 for j in range(num_nodes)] for i in range(num_nodes)]
        self.Q = [
            [self._build_qij(i, j, dim) for j in range(num_nodes)]
            for i in range(num_nodes)
        ]

    def _build_qij(self, i: int, j: int, dim: int) -> list[list[float]]:
        # diagonal + permutación cíclica determinista
        shift = (i + j) % dim
        scale = 1.0 if i == j else (0.35 + 0.05 * ((i + j) % 5))
        mat = [[0.0 for _ in range(dim)] for _ in range(dim)]
        for r in range(dim):
            c = (r + shift) % dim
            mat[r][c] = scale
        return mat

    def step(self, up_signals: list[float], world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool):
        old = [s.dynamic[:] for s in self.states]
        new_states = []
        for i, node in enumerate(self.nodes):
            neigh_vec = [0.0 for _ in range(self.dim)]
            for j in range(self.num_nodes):
                projected = _matvec(self.Q[i][j], old[j])
                neigh_vec = _vadd(neigh_vec, _vscale(projected, self.A[i][j]))
            neigh_vec = _vscale(neigh_vec, 1.0 / max(1, self.num_nodes - 1))
            new_states.append(node.step(self.states[i], neigh_vec, up_signals[i], world_drive, body, tissue, context, offline_trigger))
        self.states = new_states

    def reconstruct_global(self) -> list[float]:
        accum: list[list[float]] = []
        for i, s in enumerate(self.states):
            qii = self.Q[i][i]
            inv = [[0.0 for _ in range(self.dim)] for _ in range(self.dim)]
            for r in range(self.dim):
                c = max(range(self.dim), key=lambda cc: abs(qii[r][cc]))
                val = qii[r][c] if abs(qii[r][c]) > 1e-9 else 1.0
                inv[c][r] = 1.0 / val
            accum.append(_matvec(inv, s.dynamic))
        return _vmean(accum)

    def local_closure_errors(self) -> list[float]:
        xhat = self.reconstruct_global()
        errs = []
        for i, s in enumerate(self.states):
            projected = _matvec(self.Q[i][i], xhat)
            err = sum(abs(a - b) for a, b in zip(s.dynamic, projected)) / len(projected)
            errs.append(err)
        return errs

    def total_coherence(self) -> float:
        vals = [s.local_metrics.get("C_total", 0.0) for s in self.states]
        return sum(vals) / max(1, len(vals))
