import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator, V7NeuroConfig

def test_node_contains_full_signature():
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig())
    n = orch.nodes[0].state
    for attr in ["dynamic","symbolic","body","tissue","context","executive","memory","offline","regime","verifier","semantic_mass","local_babel","local_metrics"]:
        assert hasattr(n, attr)
