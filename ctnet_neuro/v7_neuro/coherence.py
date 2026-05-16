from __future__ import annotations

from .state import LocalMetrics


def _clip01(v: float) -> float:
    return max(0.0, min(1.0, v))


class LocalCoherenceTensor:
    def evaluate(self, node, neighbor_projection: float, world_signal: float) -> LocalMetrics:
        body = node.state.body
        tissue = node.state.tissue
        context = node.state.context
        memory = node.state.memory
        offline = node.state.offline
        verifier = node.state.verifier

        c_local = _clip01(1.0 - abs(node.state.dynamic - node.state.symbolic.get("intent", 0.0)))
        c_neighbors = _clip01(1.0 - abs(node.state.dynamic - neighbor_projection))
        c_body = _clip01(1.0 - abs(body.get("stress", 0.0) - body.get("energy", 0.0)))
        c_tissue = _clip01(1.0 - abs(tissue.get("excitation", 0.0) - tissue.get("inhibition", 0.0)))
        c_context = _clip01(context.get("compatibility", 0.0) - 0.5 * context.get("incompatibility", 0.0))
        c_memory = _clip01(1.0 - abs(memory.get("reconstruction", 0.0) - node.state.dynamic))
        c_offline = _clip01(1.0 - offline.get("pruning", 0.0))
        c_verifier = _clip01(verifier.get("consistency", 0.0))
        c_total = _clip01((c_local + c_neighbors + c_body + c_tissue + c_context + c_memory + c_offline + c_verifier) / 8.0)
        mass = _clip01(0.5 * abs(node.state.dynamic) + 0.5 * abs(world_signal))
        return LocalMetrics(
            C_local=c_local,
            C_neighbors=c_neighbors,
            C_body=c_body,
            C_tissue=c_tissue,
            C_context=c_context,
            C_memory=c_memory,
            C_offline=c_offline,
            C_total=c_total,
            mass=mass,
        )
