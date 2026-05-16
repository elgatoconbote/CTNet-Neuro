from __future__ import annotations

from ctnet_neuro.v7_neuro import CTNetV7NeuroOrchestrator
from state import CerebroVirtualConfig, make_initial_state


def main() -> None:
    state = make_initial_state(CerebroVirtualConfig())
    orch = CTNetV7NeuroOrchestrator(node_count=6)
    regimes = ["orient"] * 5 + ["threat"] * 5 + ["recover"] * 5 + ["goal_recontext"] * 5
    print("step | coh | local_coh | closure | energy/stress | exc/inh | modes | offline | readout")
    for step, regime in enumerate(regimes, start=1):
        if regime == "orient":
            world_signal, ups = 0.25, [0.3] * 6
        elif regime == "threat":
            state.body_state.stress = min(1.0, state.body_state.stress + 0.1)
            world_signal, ups = 0.8, [0.7, 0.8, 0.6, 0.75, 0.7, 0.65]
        elif regime == "recover":
            state.body_state.stress = max(0.1, state.body_state.stress - 0.12)
            world_signal, ups = 0.35, [0.2] * 6
        else:
            world_signal, ups = 0.55, [0.5, 0.55, 0.45, 0.5, 0.6, 0.52]
        offline_trigger = step % 7 == 0
        m = orch.step(state, up_signals=ups, world_signal=world_signal, offline_trigger=offline_trigger)
        modes = ",".join(n.state.executive["mode"] for n in orch.atlas.nodes)
        print(f"{step:02d} | {m.global_coherence:.3f} | {m.mean_local_coherence:.3f} | {m.closure_error:.3f} | "
              f"{state.body_state.energy:.2f}/{state.body_state.stress:.2f} | "
              f"{state.tissue_state.mean_excitation:.2f}/{state.tissue_state.mean_inhibition:.2f} | "
              f"{modes} | {m.offline_active} | {m.readout}")


if __name__ == "__main__":
    main()
