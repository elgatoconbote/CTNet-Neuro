from __future__ import annotations
from state import CerebroVirtualConfig, make_initial_state
from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator, V7NeuroConfig


def apply_synthetic_world(state, step: int):
    t = step % 20
    if t < 5:
        state.world_state.regime_label, state.world_state.regime_index = "orient", 0
        state.world_state.threat_drive = 0.1
    elif t < 10:
        state.world_state.regime_label, state.world_state.regime_index = "threat", 1
        state.world_state.threat_drive = 0.9
    elif t < 15:
        state.world_state.regime_label, state.world_state.regime_index = "recover", 2
        state.world_state.threat_drive = 0.05
    else:
        state.world_state.regime_label, state.world_state.regime_index = "goal_recontext", 3
        state.world_state.threat_drive = 0.2


def main() -> None:
    state = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(V7NeuroConfig(n_nodes=4, vector_dim=10, memory_capacity=24))
    signals = [0.2, -0.1, 0.05, 0.0]
    for step in range(20):
        apply_synthetic_world(state, step)
        state.body_state.stress = min(1.0, 0.6 * state.world_state.threat_drive)
        state.body_state.energy = max(0.0, 1.0 - 0.3 * state.world_state.threat_drive)
        out = orch.step(state.world_state, state.body_state, state.tissue_state, state.context_state, signals)
        print(f"step={step+1:02d} global={out['metrics']['global_coherence']:.3f} mean={out['metrics']['mean_local_coherence']:.3f} closure={out['metrics']['closure_error']:.3f} body(E={state.body_state.energy:.2f},S={state.body_state.stress:.2f}) tissue(X={state.tissue_state.mean_excitation:.2f},I={state.tissue_state.mean_inhibition:.2f}) modes={[n.state.executive.get('mode') for n in orch.nodes]} offline={out['metrics']['offline_active']} readout={out['readout']}")

if __name__ == '__main__':
    main()
