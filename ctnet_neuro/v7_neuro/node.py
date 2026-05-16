from __future__ import annotations
from .state import HolonicNodeState
from .memory import LocalMemory
from .executive import ExecutiveLocal
from .babel import LocalBabel, LocalVerifier


class V7NeuroNode:
    def __init__(self, node_id: int, vector_dim: int, memory_capacity: int):
        self.memory = LocalMemory(memory_capacity)
        self.executive = ExecutiveLocal()
        self.babel = LocalBabel()
        self.verifier = LocalVerifier()
        self.state = HolonicNodeState(
            node_id=node_id,
            dynamic=[0.0] * vector_dim,
            symbolic={"intent": 0.0},
            body={}, tissue={}, context={}, executive={}, memory={}, offline={}, regime={}, verifier={}, semantic_mass=0.3,
            local_babel={}, local_metrics={},
        )

    def project_vector(self) -> list[float]:
        return list(self.state.dynamic)
