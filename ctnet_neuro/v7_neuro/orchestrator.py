from __future__ import annotations
from .state import V7NeuroConfig
from .node import V7NeuroNode
from .atlas import NDimensionalAtlas
from .coherence import LocalCoherenceTensor
from .offline import OfflineReplay
from .babel import global_consensus
from .body_bridge import project_body_state
from .tissue_bridge import project_tissue_state
from .context_bridge import project_context_state
from .metrics import collect_v7_neuro_metrics


class CTNetV7NeuroOrchestrator:
    def __init__(self, config: V7NeuroConfig):
        self.config = config
        self.nodes = [V7NeuroNode(i, config.vector_dim, config.memory_capacity) for i in range(config.n_nodes)]
        self.atlas = NDimensionalAtlas(self.nodes, config.recirculation_gain)
        self.coh = LocalCoherenceTensor()
        self.offline = OfflineReplay()

    def step(self, world_state, body_state, tissue_state, context_state, up_signals: list[float] | None = None):
        up_signals = up_signals or [0.0] * len(self.nodes)
        recirc = self.atlas.step()
        readouts = []
        for i, node in enumerate(self.nodes):
            body = project_body_state(body_state)
            tissue = project_tissue_state(tissue_state)
            context = project_context_state(context_state, world_state)
            node.state.body = body; node.state.tissue = tissue; node.state.context = context
            node.state.regime = {"label": world_state.regime_label, "index": world_state.regime_index}
            ex = node.executive.step(world_state.regime_label, body["stress"], node.state.local_metrics.get("C_total", 0.5))
            node.state.executive = ex
            rate = ex["update_rate"]
            node.state.dynamic = [0.82 * x + rate * 0.12 * recirc[i][k] + 0.08 * up_signals[i] for k, x in enumerate(node.state.dynamic)]
            node.state.symbolic["intent"] = 0.85 * node.state.symbolic.get("intent", 0.0) + 0.15 * up_signals[i]
            xh = self.atlas.reconstruct_global()
            node.state.memory = node.memory.update(node.state.dynamic, xh, recirc[i], ex["compress"])
            node.state.local_babel = node.babel.read(i, node.state.symbolic, node.state.dynamic, up_signals[i], ex["babel_open"])
            m = self.coh.evaluate(node, recirc, world_state.__dict__, body)
            node.state.semantic_mass = m["mass"]
            node.state.local_metrics = m
            node.state.verifier = node.verifier.verify(node.state.local_babel, m["C_total"], ex["verify"])
            off = self.offline.step(node, ex["offline"] > self.config.offline_threshold, ex["offline"])
            node.state.offline = off
            readouts.append(node.state.local_babel)
        closure = self.atlas.local_closure_errors()
        readout = global_consensus(readouts, [n.state.local_metrics["C_total"] for n in self.nodes])
        metrics = collect_v7_neuro_metrics(self, closure, readout, any(n.state.offline.get("active", False) for n in self.nodes))
        return {"reconstruction": self.atlas.reconstruct_global(), "closure_errors": closure, "readout": readout, "metrics": metrics}
