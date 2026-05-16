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
        return HolonicNodeState(node_id=self.node_id, dynamic=[0.0 for _ in range(self.dim)], symbolic={"intent": 0.0})

    def step(self, state: HolonicNodeState, neighbor_drive: list[float], up: float, world_drive: float, body: dict, tissue: dict, context: dict, offline_trigger: bool) -> HolonicNodeState:
        sig = self.exec.select(body, context, offline_trigger)
        base = 0.2 * up + 0.25 * world_drive + 0.15 * body["interoception"] + 0.15 * tissue["excitation"]
        next_dynamic = []
        for k in range(self.dim):
            local = state.dynamic[k] if k < len(state.dynamic) else 0.0
            neigh = neighbor_drive[k] if k < len(neighbor_drive) else 0.0
            phase = 1.0 + 0.05 * ((k + self.node_id) % 4)
            next_dynamic.append(local * 0.7 + sig.update_rate * phase * (base + 0.25 * neigh))
        state.dynamic = next_dynamic
        activation = sum(abs(x) for x in state.dynamic) / self.dim
        state.symbolic = {"intent": sig.babel_opening * activation}
        state.body = body
        state.tissue = tissue
        state.context = context
        state.executive = {"mode": sig.mode, "rate": sig.update_rate}
        vec = [state.dynamic[0], state.dynamic[1], state.dynamic[2], state.symbolic["intent"], body["energy"], tissue["excitation"]]
        mem = self.memory_state.update(vec, neighbor_drive[:6], sig.recirculation_strength, sig.memory_compression)
        state.memory = mem
        state.regime = {"label": context["active_regime"]}
        state.semantic_mass = 0.9 * state.semantic_mass + 0.1 * (activation + mem["invariant_mass"]) * 0.5
        state.offline = self.offline.apply(state.__dict__, offline_trigger, sig.offline_push)
        coh = self.coherence.evaluate(state, self.memory_state, neighbor_drive, world_drive)
        state.local_metrics = coh
        read = local_read(self.node_id, context["active_regime"], activation, coh["C_total"])
        state.local_babel = read
        state.verifier = verify_local(read, sig.verifier_priority)
        return state
