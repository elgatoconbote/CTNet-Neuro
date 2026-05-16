from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator
from state import CerebroVirtualConfig, make_initial_state


def test_no_sequential_pipeline():
    orch = CTNetV7NeuroOrchestrator(node_count=5)
    st = make_initial_state(CerebroVirtualConfig())
    orch.step(st,[0.2]*5,0.3)
    assert len(orch.atlas.nodes) > 1
    for n in orch.atlas.nodes:
        assert n.state.symbolic is not None and n.state.tissue is not None and n.state.body is not None


def test_demo_runs():
    from examples.run_v7_neuro_demo import main
    main()
