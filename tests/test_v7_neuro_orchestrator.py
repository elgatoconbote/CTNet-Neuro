import math

from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def test_memory_capacity_fixed():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=8), num_nodes=3)
    for _ in range(40):
        orch.step()
    assert all(s.memory["size"] <= s.memory["capacity"] for s in orch.atlas.states)


def test_offline_changes_state():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=1000), num_nodes=2)
    before = [dict(s.offline) for s in orch.atlas.states]
    body = {"energy": 0.5, "stress": 0.4, "pain": 0.2, "interoception": 0.2, "proprioception": 0.1, "autonomic_load": 0.1}
    tissue = {"excitation": 0.2, "inhibition": 0.1, "glial_modulation": 0.2, "local_plasticity": 0.3, "energy": 0.8, "structural_signal": 0.1, "patch_A": 0.1, "patch_B": 0.1}
    context = {"hypotheses": 4, "compatibility": 0.4, "incompatibility": 0.1, "grammar": 0.2, "return_signal": 0.2, "active_regime": "orient"}
    orch.atlas.step([0.1, 0.2], 0.2, body, tissue, context, True)
    after = [dict(s.offline) for s in orch.atlas.states]
    assert any(a != b for a, b in zip(before, after))
    assert all(s.offline.get("consolidation", 0.0) > 0.0 for s in orch.atlas.states)


def test_stability_200_steps():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(sleep_every=7), num_nodes=4)
    for _ in range(200):
        info = orch.step()
        assert 0.0 <= info["global_coherence"] <= 1.0
        assert math.isfinite(info["closure_error"])
        assert all(math.isfinite(x) for x in info["x_hat"])
        assert all(s.memory["size"] <= s.memory["capacity"] for s in orch.atlas.states)
