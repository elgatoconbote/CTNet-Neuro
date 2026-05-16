from ctnet_neuro.v7_neuro.node import V7NeuroNode


def test_node_contains_full_signature():
    node = V7NeuroNode(0)
    fields = {"dynamic","symbolic","body","tissue","context","executive","memory","offline","regime","verifier","semantic_mass","local_babel","local_metrics"}
    assert fields.issubset(set(node.state.__dict__.keys()))
