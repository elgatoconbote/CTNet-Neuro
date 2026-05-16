from __future__ import annotations

from dataclasses import dataclass

from .atlas import NDimensionalAtlas
from .babel import global_consensus
from .body_bridge import body_snapshot
from .context_bridge import context_snapshot
from .offline import OfflineReplay
from .tissue_bridge import tissue_snapshot


@dataclass
class V7NeuroStepMetrics:
    step: int
    global_coherence: float
    closure_error: float
    mean_local_coherence: float
    readout: str
    offline_active: bool


class CTNetV7NeuroOrchestrator:
    def __init__(self, node_count: int = 5):
        self.atlas = NDimensionalAtlas(node_count=node_count)
        self.offline = OfflineReplay()
        self.step_idx = 0

    def step(self, global_state, up_signals: list[float], world_signal: float, offline_trigger: bool = False) -> V7NeuroStepMetrics:
        body = body_snapshot(global_state)
        tissue = tissue_snapshot(global_state)
        context = context_snapshot(global_state)
        self.atlas.step(up_signals=up_signals, world_signal=world_signal, body=body, tissue=tissue, context=context)
        for node in self.atlas.nodes:
            bias = node.state.executive.get("offline_bias", 0.0)
            self.offline.apply_local(node, offline_trigger, bias)
        coherences = [n.state.local_metrics.C_total for n in self.atlas.nodes]
        readouts = [n.state.verifier for n in self.atlas.nodes]
        consensus = global_consensus(readouts, coherences)
        self.step_idx += 1
        return V7NeuroStepMetrics(
            step=self.step_idx,
            global_coherence=self.atlas.total_coherence(),
            closure_error=self.atlas.closure_error(),
            mean_local_coherence=self.atlas.total_coherence(),
            readout=consensus,
            offline_active=offline_trigger,
        )
