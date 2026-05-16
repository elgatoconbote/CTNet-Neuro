from __future__ import annotations

from .babel import local_read, verify_local
from .coherence import LocalCoherenceTensor
from .executive import ExecutiveLocal
from .memory import LocalMemory
from .offline import OfflineReplay
from .state import HolonicNodeState


class V7NeuroNode:
    DYNAMIC_DIM = 5

    def __init__(self, node_id: int, capacity: int = 32):
        self.node_id = node_id
        self.memory_state = LocalMemory(capacity)
        self.coherence = LocalCoherenceTensor()
        self.exec = ExecutiveLocal()
        self.offline = OfflineReplay()

    def init_state(self) -> HolonicNodeState:
        return HolonicNodeState(node_id=self.node_id, dynamic=[0.0] * self.DYNAMIC_DIM, symbolic={"intent": 0.0})

    def step(self, state: HolonicNodeState, neighbor_drive: list[float], up: float, world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool) -> HolonicNodeState:
        sig = self.exec.select(body, context, offline_trigger)
        prev = state.dynamic if state.dynamic else [0.0] * self.DYNAMIC_DIM
        drive = [up, world_drive, body["interoception"], tissue["excitation"], context["compatibility"]]
        state.dynamic = [0.7 * prev[k] + sig.update_rate * (0.55 * drive[k] + 0.45 * neighbor_drive[k]) for k in range(self.DYNAMIC_DIM)]
        state.symbolic = {"intent": sig.babel_opening * sum(state.dynamic) / self.DYNAMIC_DIM}
        state.body = body
        state.tissue = tissue
        state.context = context
        state.executive = {"mode": sig.mode, "rate": sig.update_rate}
        mem = self.memory_state.update(state.dynamic, neighbor_drive, sig.recirculation_strength, sig.memory_compression)
        state.memory = mem
        state.regime = {"label": context["active_regime"]}
        state.semantic_mass = 0.9 * state.semantic_mass + 0.1 * (sum(abs(x) for x in state.dynamic) / self.DYNAMIC_DIM + mem["invariant_mass"]) * 0.5
        state.offline = self.offline.apply(state.__dict__, offline_trigger, sig.offline_push)
        coh = self.coherence.evaluate(state, self.memory_state, neighbor_drive, world_drive)
        state.local_metrics = coh
        read = local_read(self.node_id, context["active_regime"], sum(state.dynamic) / self.DYNAMIC_DIM, coh["C_total"])
        state.local_babel = read
        state.verifier = verify_local(read, sig.verifier_priority)
        return state
