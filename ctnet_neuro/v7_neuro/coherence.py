from __future__ import annotations


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


class LocalCoherenceTensor:
    def evaluate(self, state, memory_state, neighbor_proj: list[float], world_drive: float) -> dict[str, float]:
        body = state.body
        tissue = state.tissue
        ctx = state.context
        dyn = state.dynamic
        dyn_mean = sum(dyn) / max(1, len(dyn))
        neigh_mean = sum(neighbor_proj) / max(1, len(neighbor_proj))
        C_body = _clip01(1.0 - abs(body["stress"] - body["energy"]))
        C_tissue = _clip01(1.0 - abs(tissue["excitation"] - tissue["inhibition"]))
        C_context = _clip01(0.5 + 0.5 * (ctx["compatibility"] - ctx["incompatibility"]))
        C_memory = _clip01(0.5 + 0.5 * memory_state.invariant_mass)
        C_offline = _clip01(1.0 - state.offline.get("pruning", 0.0))
        C_neighbors = _clip01(1.0 - abs(dyn_mean - neigh_mean))
        C_local = _clip01((C_body + C_tissue + C_context) / 3.0)
        C_total = _clip01(0.18*C_local + 0.16*C_neighbors + 0.12*C_body + 0.12*C_tissue + 0.14*C_context + 0.14*C_memory + 0.14*C_offline)
        mass = _clip01(0.5 * C_total + 0.5 * _clip01(abs(world_drive)))
        return dict(C_local=C_local, C_neighbors=C_neighbors, C_body=C_body, C_tissue=C_tissue, C_context=C_context, C_memory=C_memory, C_offline=C_offline, C_total=C_total, mass=mass)
