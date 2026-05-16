from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


FULL_KEYS = {"dynamic", "symbolic", "body", "tissue", "context", "executive", "memory", "offline", "regime", "verifier", "semantic_mass", "local_babel", "local_metrics"}


def test_no_sequential_pipeline():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=5)
    assert len(orch.atlas.nodes) == 5
    before = [s.dynamic[:] for s in orch.atlas.states]
    out = orch.step()
    assert isinstance(out["x_hat"], list) and len(out["x_hat"]) == orch.atlas.dim
    for idx, s in enumerate(orch.atlas.states):
        assert FULL_KEYS.issubset(set(s.__dict__.keys()))
        assert len(s.dynamic) == orch.atlas.dim
        assert s.dynamic != before[idx]
        assert s.body and s.tissue and s.context and s.executive and s.memory and s.verifier


def test_demo_runs():
    orch = CTNetV7NeuroOrchestrator(CerebroVirtualConfig(), num_nodes=2)
    assert orch.step()["readout"].startswith("consensus:")

def test_root_compat_export_orchestrator():
    from cerebro_virtual import CerebroVirtualOrchestrator

    assert CerebroVirtualOrchestrator is not None
