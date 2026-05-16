import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from state import CerebroVirtualConfig, make_initial_state
from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator, V7NeuroConfig
from examples.run_v7_neuro_demo import main

def test_no_sequential_pipeline():
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig(n_nodes=5))
    assert len(orch.nodes) == 5
    assert all(hasattr(n.state, "dynamic") and hasattr(n.state, "tissue") for n in orch.nodes)

def test_demo_runs():
    main()
