from __future__ import annotations


class OfflineReplay:
    def __init__(self):
        self.last_action = "idle"

    def apply(self, node_state: dict, trigger: bool, push: float) -> dict:
        if not trigger:
            self.last_action = "idle"
            return {"status": "idle", "consolidation": 0.0, "pruning": 0.0}
        consolidation = min(1.0, 0.2 + 0.6 * push)
        pruning = min(1.0, 0.1 + 0.4 * push)
        node_state["semantic_mass"] = max(0.0, node_state.get("semantic_mass", 0.0) * (1.0 - 0.05 * pruning) + 0.08 * consolidation)
        self.last_action = "replay"
        return {"status": "replay", "consolidation": consolidation, "pruning": pruning}
