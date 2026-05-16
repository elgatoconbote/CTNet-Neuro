from __future__ import annotations

from .node import V7NeuroNode


def matvec(M: list[list[float]], v: list[float]) -> list[float]:
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


class NDimensionalAtlas:
    def __init__(self, num_nodes: int = 4):
        self.num_nodes = num_nodes
        self.nodes = [V7NeuroNode(i) for i in range(num_nodes)]
        self.states = [n.init_state() for n in self.nodes]
        dim = self.nodes[0].DYNAMIC_DIM
        self.A = [[1.0 if i != j else 0.0 for j in range(num_nodes)] for i in range(num_nodes)]
        self.Q = [[identity(dim) for _ in range(num_nodes)] for _ in range(num_nodes)]
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    shift = (i + j) % dim
                    self.Q[i][j] = [[1.0 if c == (r + shift) % dim else 0.0 for c in range(dim)] for r in range(dim)]

    def step(self, up_signals: list[float], world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool):
        old = [s.dynamic[:] for s in self.states]
        new_states = []
        dim = len(old[0])
        for i, node in enumerate(self.nodes):
            neigh = [0.0] * dim
            denom = 0.0
            for j in range(self.num_nodes):
                if i == j:
                    continue
                w = self.A[i][j]
                proj = matvec(self.Q[i][j], old[j])
                neigh = [neigh[k] + w * proj[k] for k in range(dim)]
                denom += w
            if denom > 0:
                neigh = [x / denom for x in neigh]
            new_states.append(node.step(self.states[i], neigh, up_signals[i], world_drive, body, tissue, context, offline_trigger))
        self.states = new_states

    def reconstruct_global(self) -> list[float]:
        dim = len(self.states[0].dynamic)
        acc = [0.0] * dim
        for i, s in enumerate(self.states):
            # inverse of permutation == transpose
            Qi = self.Q[i][i]
            inv = list(map(list, zip(*Qi)))
            proj = matvec(inv, s.dynamic)
            acc = [acc[k] + proj[k] for k in range(dim)]
        return [x / self.num_nodes for x in acc]

    def local_closure_errors(self) -> list[float]:
        xhat = self.reconstruct_global()
        errs = []
        for i, s in enumerate(self.states):
            qx = matvec(self.Q[i][i], xhat)
            diff = [(s.dynamic[k] - qx[k]) for k in range(len(qx))]
            errs.append(sum(d*d for d in diff) ** 0.5)
        return errs

    def total_coherence(self) -> float:
        vals = [s.local_metrics.get("C_total", 0.0) for s in self.states]
        return sum(vals)/max(1, len(vals))
