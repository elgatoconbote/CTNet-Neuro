from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def test_memory_capacity_fixed():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=8), num_nodes=3)
    for _ in range(40):
        orch.step()
    assert all(s.memory["size"] <= s.memory["capacity"] for s in orch.atlas.states)


def test_offline_changes_state():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=2), num_nodes=2)
    changed = False
    for _ in range(6):
        orch.step()
        after = sum(s.offline.get("consolidation", 0.0) + s.offline.get("pruning", 0.0) for s in orch.atlas.states)
        if after > 0:
            changed = True
    assert changed
