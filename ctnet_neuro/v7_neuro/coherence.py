from __future__ import annotations
import math


class LocalCoherenceTensor:
    def evaluate(self, node, neighbors: list[list[float]], world: dict, body: dict) -> dict:
        d = node.state.dynamic
        d_mean = sum(d) / max(1, len(d))
        neigh_mean = sum(sum(v)/max(1,len(v)) for v in neighbors) / max(1, len(neighbors)) if neighbors else d_mean
        c_local = max(0.0, min(1.0, 1.0 - abs(d_mean - node.state.symbolic.get("intent", 0.0))))
        c_neighbors = max(0.0, min(1.0, 1.0 - abs(d_mean - neigh_mean)))
        c_body = max(0.0, min(1.0, 1.0 - abs(body["stress"] - (1.0 - body["energy"]))))
        c_tissue = max(0.0, min(1.0, 1.0 - abs(node.state.tissue["excitation"] - node.state.tissue["inhibition"])))
        c_context = max(0.0, min(1.0, node.state.context["compatibility"] - 0.2 * node.state.context["incompatibility"]))
        c_memory = max(0.0, min(1.0, node.state.memory.get("invariants_mass", 0.0)))
        c_offline = max(0.0, min(1.0, 1.0 - node.state.offline.get("pruning", 0.0)))
        c_total = max(0.0, min(1.0, (c_local+c_neighbors+c_body+c_tissue+c_context+c_memory+c_offline)/7.0))
        mass = max(0.0, min(2.0, node.state.semantic_mass + 0.1 * c_total - 0.03 * abs(world.get("threat_drive",0.0))))
        return dict(C_local=c_local, C_neighbors=c_neighbors, C_body=c_body, C_tissue=c_tissue, C_context=c_context, C_memory=c_memory, C_offline=c_offline, C_total=c_total, mass=mass)
