from __future__ import annotations


class OfflineReplay:
    def apply_local(self, node, trigger: bool, bias: float) -> None:
        if not trigger:
            node.state.offline = {"replay": 0.0, "restoration": 0.0, "cleanup": 0.0, "consolidation": 0.0, "pruning": 0.0}
            return
        replay = min(1.0, 0.5 + 0.5 * bias)
        restoration = min(1.0, 0.3 + 0.4 * bias)
        cleanup = min(1.0, 0.2 + 0.6 * (1.0 - bias))
        consolidation = min(1.0, 0.4 + 0.5 * bias)
        pruning = min(0.8, 0.1 + 0.3 * (1.0 - bias))
        node.memory.consolidate(consolidation)
        node.memory.prune(pruning)
        node.state.semantic_mass *= (1.0 - 0.1 * pruning) * (1.0 + 0.1 * consolidation)
        node.state.offline = {
            "replay": replay,
            "restoration": restoration,
            "cleanup": cleanup,
            "consolidation": consolidation,
            "pruning": pruning,
        }
