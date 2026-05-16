from __future__ import annotations

from .babel import local_read, verify_local
from .coherence import LocalCoherenceTensor
from .executive import ExecutiveLocal
from .memory import LocalMemory
from .offline import OfflineReplay
from .state import HolonicNodeState


class V7NeuroNode:
    def __init__(self, node_id: int, capacity: int = 32):
        self.node_id = node_id
        self.memory_state = LocalMemory(capacity)
        self.coherence = LocalCoherenceTensor()
        self.exec = ExecutiveLocal()
        self.offline = OfflineReplay()

    def init_state(self) -> HolonicNodeState:
        return HolonicNodeState(node_id=self.node_id, dynamic={"activation": 0.0}, symbolic={"intent": 0.0})

    def step(self, state: HolonicNodeState, neighbor_drive: float, up: float, world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool) -> HolonicNodeState:
        sig = self.exec.select(body, context, offline_trigger)
        activation = state.dynamic.get("activation", 0.0)
        activation = (activation * 0.7 + sig.update_rate * (0.2 * up + 0.25 * world_drive + 0.25 * neighbor_drive + 0.15 * body["interoception"] + 0.15 * tissue["excitation"]))
        state.dynamic = {"activation": activation}
        state.symbolic = {"intent": sig.babel_opening * activation}
        state.body = body
        state.tissue = tissue
        state.context = context
        state.executive = {"mode": sig.mode, "rate": sig.update_rate}
        vec = [activation, state.symbolic["intent"], body["energy"], tissue["excitation"], context["compatibility"]]
        mem = self.memory_state.update(vec, [neighbor_drive]*5, sig.recirculation_strength, sig.memory_compression)
        state.memory = mem
        state.regime = {"label": context["active_regime"]}
        state.semantic_mass = 0.9 * state.semantic_mass + 0.1 * (abs(activation) + mem["invariant_mass"]) * 0.5
        state.offline = self.offline.apply(state.__dict__, offline_trigger, sig.offline_push)
        self.offline_state = state.offline
        coh = self.coherence.evaluate(state, self.memory_state, neighbor_drive, world_drive)
        state.local_metrics = coh
        read = local_read(self.node_id, context["active_regime"], activation, coh["C_total"])
        state.local_babel = read
        state.verifier = verify_local(read, sig.verifier_priority)
        return state
