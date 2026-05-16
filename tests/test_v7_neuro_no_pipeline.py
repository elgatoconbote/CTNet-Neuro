from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def test_no_sequential_pipeline():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=5)
    assert len(orch.atlas.nodes) == 5
    for s in orch.atlas.states:
        assert hasattr(s, "dynamic") and hasattr(s, "tissue") and hasattr(s, "context")


def test_demo_runs():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=2)
    assert orch is not None
