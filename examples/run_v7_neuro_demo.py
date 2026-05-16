from __future__ import annotations

from state import CerebroVirtualConfig
from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator


def main() -> None:
    config = CerebroVirtualConfig(sleep_every=8)
    orch = CTNetV7NeuroOrchestrator(config=config, num_nodes=4)
    for _ in range(20):
        info = orch.step()
        print(f"step={info['step']} coherence={info['global_coherence']:.3f} local={info['mean_local_coherence']:.3f} closure={info['closure_error']:.3f} body=({info['body']['energy']:.3f},{info['body']['stress']:.3f}) tissue=({info['tissue']['excitation']:.3f},{info['tissue']['inhibition']:.3f}) modes={info['executive_modes']} offline={info['offline']} readout={info['readout']}")


if __name__ == "__main__":
    main()
