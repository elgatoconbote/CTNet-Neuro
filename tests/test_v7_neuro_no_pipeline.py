from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def test_no_sequential_pipeline():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=5)
    before = [tuple(s.dynamic) for s in orch.atlas.states]
    orch.step()
    after = [tuple(s.dynamic) for s in orch.atlas.states]
    assert len(orch.atlas.nodes) == 5
    for idx,s in enumerate(orch.atlas.states):
        for key in ["dynamic","symbolic","body","tissue","context","executive","memory","offline","regime","verifier","local_babel","local_metrics"]:
            assert getattr(s, key) is not None
        assert after[idx] != before[idx]


def test_demo_runs():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=2)
    assert orch.step()["step"] == 1
