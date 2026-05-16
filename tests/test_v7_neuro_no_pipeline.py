from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def test_no_sequential_pipeline():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=5)
    pre = [tuple(s.dynamic) for s in orch.atlas.states]
    orch.step()
    post = orch.atlas.states
    assert len(orch.atlas.nodes) == 5
    required = ["dynamic", "symbolic", "body", "tissue", "context", "executive", "memory", "offline", "regime", "verifier", "semantic_mass", "local_babel", "local_metrics"]
    for idx, s in enumerate(post):
        for key in required:
            assert hasattr(s, key)
        assert tuple(s.dynamic) != pre[idx]
        assert s.body and s.tissue and s.context and s.memory and s.local_babel and s.verifier


def test_demo_runs():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=2)
    assert orch.step()["step"] == 1
