from __future__ import annotations

from .babel import local_read, verify_local
from .coherence import LocalCoherenceTensor
from .executive import ExecutiveLocal
from .memory import LocalMemory
from .offline import OfflineReplay
from .state import HolonicNodeState


class V7NeuroNode:
    def __init__(self, node_id: int, capacity: int = 32, dim: int = 6):
        self.node_id = node_id
        self.dim = dim
        self.memory_state = LocalMemory(capacity)
        self.coherence = LocalCoherenceTensor()
        self.exec = ExecutiveLocal()
        self.offline = OfflineReplay()

    def init_state(self) -> HolonicNodeState:
        return HolonicNodeState(node_id=self.node_id, dynamic=[0.0] * self.dim, symbolic={"intent": 0.0})

    def step(self, state: HolonicNodeState, neighbor_drive: list[float], up: float, world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool) -> HolonicNodeState:
        sig = self.exec.select(body, context, offline_trigger)
        prev = state.dynamic
        local_in = [up, world_drive, body["interoception"], tissue["excitation"], context["compatibility"], body["energy"]]
        updated = []
        for k in range(self.dim):
            v = 0.55 * prev[k] + sig.update_rate * (0.20 * local_in[k] + 0.25 * neighbor_drive[k])
            updated.append(v)
        state.dynamic = updated
        activation = sum(updated) / self.dim
        state.symbolic = {"intent": sig.babel_opening * activation, "vector_norm": sum(abs(v) for v in updated) / self.dim}
        state.body = body
        state.tissue = tissue
        state.context = context
        state.executive = {"mode": sig.mode, "rate": sig.update_rate}
        mem = self.memory_state.update(updated, neighbor_drive, sig.recirculation_strength, sig.memory_compression)
        state.memory = mem
        state.regime = {"label": context["active_regime"]}
        state.semantic_mass = 0.9 * state.semantic_mass + 0.1 * (state.symbolic["vector_norm"] + mem["invariant_mass"]) * 0.5
        state.offline = self.offline.apply(state.__dict__, offline_trigger, sig.offline_push)
        coh = self.coherence.evaluate(state, self.memory_state, neighbor_drive, world_drive)
        state.local_metrics = coh
        read = local_read(self.node_id, context["active_regime"], activation, coh["C_total"])
        state.local_babel = read
        state.verifier = verify_local(read, sig.verifier_priority)
        return state
