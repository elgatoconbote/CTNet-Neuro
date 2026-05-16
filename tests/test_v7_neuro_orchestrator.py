import math

from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def test_memory_capacity_fixed():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=8), num_nodes=3)
    for _ in range(40):
        orch.step()
    assert all(s.memory["size"] <= s.memory["capacity"] for s in orch.atlas.states)


def test_offline_changes_state():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=999), num_nodes=2)
    orch.atlas.step([0.1,0.1], 0.2, {"energy":0.8,"stress":0.1,"pain":0.0,"interoception":0.2,"proprioception":0.1,"autonomic_load":0.1}, {"excitation":0.2,"inhibition":0.1,"glial_modulation":0.2,"local_plasticity":0.1,"energy":0.8,"structural_signal":0.2,"patch_A":0.1,"patch_B":0.1}, {"hypotheses":4,"compatibility":0.5,"incompatibility":0.1,"grammar":0.2,"return_signal":0.2,"active_regime":"orient"}, True)
    assert any(s.offline.get("consolidation",0)>0 and s.offline.get("pruning",0)>0 for s in orch.atlas.states)


def test_stability_200_steps():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=8), num_nodes=4)
    for _ in range(200):
        info = orch.step()
        assert 0.0 <= info["global_coherence"] <= 1.0
        assert math.isfinite(info["closure_error"])
        assert all(s.memory["size"] <= s.memory["capacity"] for s in orch.atlas.states)
        assert all(all(math.isfinite(v) for v in s.dynamic) for s in orch.atlas.states)
