from ctnet_neuro.v7_neuro.node import V7NeuroNode


def test_node_contains_full_signature():
    s = V7NeuroNode(0).init_state()
    for key in ["dynamic", "symbolic", "body", "tissue", "context", "executive", "memory", "offline", "regime", "verifier", "semantic_mass", "local_babel", "local_metrics"]:
        assert hasattr(s, key)
    assert isinstance(s.dynamic, list)
