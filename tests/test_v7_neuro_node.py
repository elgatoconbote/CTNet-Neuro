import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import math
from state import CerebroVirtualConfig, make_initial_state
from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator, V7NeuroConfig

def test_local_closure_errors_finite():
    s = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig())
    out = orch.step(s.world_state, s.body_state, s.tissue_state, s.context_state, [0.1,0.2,0.0,-0.1])
    assert all(math.isfinite(x) for x in out["closure_errors"])

def test_memory_capacity_fixed():
    s = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig(memory_capacity=8))
    for _ in range(40):
        orch.step(s.world_state, s.body_state, s.tissue_state, s.context_state)
    assert all(n.state.memory["size"] <= 8 for n in orch.nodes)
