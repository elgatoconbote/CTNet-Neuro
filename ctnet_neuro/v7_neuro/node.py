from __future__ import annotations

from dataclasses import asdict

from .babel import LocalBabel, LocalVerifier
from .coherence import LocalCoherenceTensor
from .executive import ExecutiveLocal
from .memory import LocalMemory
from .state import HolonicNodeState


class V7NeuroNode:
    def __init__(self, node_id: int):
        self.node_id = node_id
        self.memory = LocalMemory()
        self.exec = ExecutiveLocal()
        self.babel = LocalBabel(node_id)
        self.verifier = LocalVerifier()
        self.coh = LocalCoherenceTensor()
        self.state = HolonicNodeState(node_id=node_id)

    def step(self, neighbor_projection: float, up_signal: float, world_signal: float, body: dict, tissue: dict, context: dict) -> dict:
        mode = self.exec.update(body["stress"], body.get("pain", 0.0), body["energy"])
        profile = self.exec.profile()
        dynamic = self.state.dynamic
        dynamic = dynamic + profile["update_rate"] * (0.25 * up_signal + 0.25 * world_signal + 0.25 * neighbor_projection - 0.2 * body["stress"] + 0.1 * tissue["excitation"])
        dynamic *= (1.0 - 0.05 * tissue["inhibition"])
        self.state.dynamic = max(-1.0, min(1.0, dynamic))
        self.state.symbolic = {"intent": up_signal, "mode": mode, "grammar": context["grammar"]}
        self.state.body = dict(body)
        self.state.tissue = dict(tissue)
        self.state.context = dict(context)
        self.state.executive = {"mode": mode, **profile}
        self.state.memory = self.memory.recirculate(self.state.dynamic, neighbor_projection, profile["memory_compression"])
        self.state.regime = {"active_regime": context["active_regime"]}
        readout = self.babel.read(self.state.dynamic, self.state.symbolic, up_signal, profile["babel_open"])
        readout = self.verifier.verify(readout, self.state.dynamic, neighbor_projection, profile["verify_priority"])
        self.state.verifier = readout
        self.state.local_babel = {"open": profile["babel_open"], "frontier": readout["frontier_unverified"]}
        self.state.local_metrics = self.coh.evaluate(self, neighbor_projection, world_signal)
        self.state.semantic_mass = self.state.local_metrics.mass + self.memory.invariant_mass
        return asdict(self.state.local_metrics)
