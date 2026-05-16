import math
from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator
from state import CerebroVirtualConfig, make_initial_state


def test_global_reconstruction_shape():
    orch = CTNetV7NeuroOrchestrator(node_count=4)
    st = make_initial_state(CerebroVirtualConfig())
    orch.step(st,[0.2]*4,0.3)
    xhat = orch.atlas.reconstruct_global()
    assert len(xhat) == 4


def test_local_closure_errors_finite():
    orch = CTNetV7NeuroOrchestrator(node_count=4)
    st = make_initial_state(CerebroVirtualConfig())
    orch.step(st,[0.2]*4,0.3)
    errs = orch.atlas.local_closure_errors()
    assert all(math.isfinite(e) for e in errs)


def test_offline_changes_state():
    orch = CTNetV7NeuroOrchestrator(node_count=3)
    st = make_initial_state(CerebroVirtualConfig())
    orch.step(st,[0.2]*3,0.3,offline_trigger=False)
    before = orch.atlas.nodes[0].state.offline
    orch.step(st,[0.2]*3,0.3,offline_trigger=True)
    after = orch.atlas.nodes[0].state.offline
    assert after.get("consolidation",0) != before.get("consolidation",0) or after.get("pruning",0) != before.get("pruning",0)
