import math
from ctnet_neuro.v7_neuro.node import V7NeuroNode


def test_coherence_bounds():
    node = V7NeuroNode(0)
    node.step(0.1, 0.2, 0.3, {"energy":0.8,"stress":0.2,"pain":0.1,"interoception":0.4,"proprioception":0.4,"autonomic_load":0.2}, {"excitation":0.4,"inhibition":0.2,"glial_modulation":0.1,"local_plasticity":0.1,"energy":0.8,"structural_signal":0.2,"patch_A":0.0,"patch_B":0.0}, {"hypotheses":[],"compatibility":0.6,"incompatibility":0.2,"grammar":0.5,"return_signal":0.4,"active_regime":0})
    vals = node.state.local_metrics
    for key in ["C_local","C_neighbors","C_body","C_tissue","C_context","C_memory","C_offline","C_total","mass"]:
        v = getattr(vals, key)
        assert 0.0 <= v <= 1.0


def test_memory_capacity_fixed():
    node = V7NeuroNode(0)
    for _ in range(200):
        node.step(0.1,0.2,0.3,{"energy":0.8,"stress":0.2,"pain":0.1,"interoception":0.4,"proprioception":0.4,"autonomic_load":0.2},{"excitation":0.4,"inhibition":0.2,"glial_modulation":0.1,"local_plasticity":0.1,"energy":0.8,"structural_signal":0.2,"patch_A":0.0,"patch_B":0.0},{"hypotheses":[],"compatibility":0.6,"incompatibility":0.2,"grammar":0.5,"return_signal":0.4,"active_regime":0})
    assert node.state.memory["size"] <= node.state.memory["capacity"]
    assert math.isfinite(node.state.memory["reconstruction"])
