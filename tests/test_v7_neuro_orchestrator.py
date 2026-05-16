import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from state import CerebroVirtualConfig, make_initial_state
from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator, V7NeuroConfig

def test_global_reconstruction_shape():
    s = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig(vector_dim=9))
    out = orch.step(s.world_state, s.body_state, s.tissue_state, s.context_state)
    assert len(out["reconstruction"]) == 9

def test_coherence_bounds():
    s = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig())
    orch.step(s.world_state, s.body_state, s.tissue_state, s.context_state)
    for n in orch.nodes:
        assert 0.0 <= n.state.local_metrics["C_total"] <= 1.0

def test_offline_changes_state():
    s = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig(offline_threshold=0.4))
    before = orch.nodes[0].state.semantic_mass
    for n in orch.nodes: n.state.executive={"offline":1.0}
    orch.step(s.world_state, s.body_state, s.tissue_state, s.context_state)
    assert orch.nodes[0].state.semantic_mass >= before
