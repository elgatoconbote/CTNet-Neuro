from __future__ import annotations

from body.body_system import BodySystem
from context.context_bank import ContextBank
from offline.sleep_cycle import SleepCycle
from world.world_loop import WorldLoop

from state import CerebroVirtualConfig, make_initial_state
from .atlas import NDimensionalAtlas
from .babel import consensus
from .body_bridge import project_body
from .context_bridge import project_context
from .tissue_bridge import project_tissue


class CTNetV7NeuroOrchestrator:
    def __init__(self, config: CerebroVirtualConfig, num_nodes: int = 4):
        self.config = config
        self.state = make_initial_state(config)
        self.world = WorldLoop()
        self.body = BodySystem()
        self.tissue = None
        self.context = ContextBank()
        self.sleep = SleepCycle()
        self.atlas = NDimensionalAtlas(num_nodes)

    def step(self) -> dict:
        self.world.step(self.state)
        self.body.step(self.state)
        if self.tissue is not None:
            self.tissue.step(self.state)
        else:
            self.state.tissue_state.mean_excitation = 0.85*self.state.tissue_state.mean_excitation + 0.15*abs(self.state.world_state.sensory_drive)
            self.state.tissue_state.mean_inhibition = 0.9*self.state.tissue_state.mean_inhibition + 0.1*self.state.body_state.stress
            self.state.tissue_state.glia_modulation = 0.9*self.state.tissue_state.glia_modulation + 0.1*self.state.body_state.autonomic_load
            self.state.tissue_state.local_plasticity = 0.9*self.state.tissue_state.local_plasticity + 0.1*self.state.context_state.last_alignment
            self.state.tissue_state.structural_signal = self.state.tissue_state.mean_excitation - self.state.tissue_state.mean_inhibition
        self.context.step(self.state)
        self.body.apply_action(self.state)
        self.sleep.step(self.state, self.config)

        body = project_body(self.state.body_state)
        tissue = project_tissue(self.state.tissue_state)
        context = project_context(self.state.context_state, self.state.world_state)
        world_drive = float(self.state.world_state.sensory_drive)
        up = [body["interoception"] * (1.0 + 0.05 * i) for i in range(self.atlas.num_nodes)]

        self.atlas.step(up, world_drive, body, tissue, context, self.state.offline_state.asleep)
        closure = self.atlas.local_closure_errors()
        xhat = self.atlas.reconstruct_global()
        local_reads = [s.local_babel for s in self.atlas.states]
        weights = [s.local_metrics.get("C_total", 0.0) for s in self.atlas.states]
        readout = consensus(local_reads, weights)
        return {
            "step": self.state.world_state.step_count,
            "global_coherence": self.atlas.total_coherence(),
            "mean_local_coherence": sum(weights) / max(1, len(weights)),
            "closure_error": sum(closure) / max(1, len(closure)),
            "body": {"energy": body["energy"], "stress": body["stress"]},
            "tissue": {"excitation": tissue["excitation"], "inhibition": tissue["inhibition"]},
            "executive_modes": [s.executive.get("mode", "reflect") for s in self.atlas.states],
            "offline": self.state.offline_state.asleep,
            "readout": readout,
            "x_hat": xhat,
        }
