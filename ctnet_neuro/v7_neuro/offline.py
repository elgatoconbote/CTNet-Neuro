from __future__ import annotations


class OfflineReplay:
    def step(self, node, trigger: bool, intensity: float) -> dict:
        status = {"replay": 0.0, "restoration": 0.0, "cleanup": 0.0, "consolidation": 0.0, "pruning": 0.0, "active": False}
        if trigger:
            status.update({"replay": 0.6*intensity, "restoration": 0.4*intensity, "cleanup": 0.5*intensity, "consolidation": 0.7*intensity, "pruning": 0.3*intensity, "active": True})
            node.state.semantic_mass = min(2.0, node.state.semantic_mass + 0.08 * intensity)
            node.memory.consolidate(status["consolidation"])
            node.memory.prune(status["pruning"])
        return status
