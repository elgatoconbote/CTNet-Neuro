from __future__ import annotations
from collections import deque


class LocalMemory:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.local_chart = deque(maxlen=capacity)
        self.reconstructions = deque(maxlen=capacity)
        self.recirculation = deque(maxlen=capacity)
        self.invariants_mass = 0.0

    def update(self, local_vec: list[float], reconstruction: list[float], recirc: list[float], compress: float) -> dict:
        self.local_chart.append(local_vec)
        self.reconstructions.append(reconstruction)
        self.recirculation.append(recirc)
        novelty = sum(abs(a - b) for a, b in zip(local_vec, reconstruction)) / max(1, len(local_vec))
        self.invariants_mass = max(0.0, 0.92 * self.invariants_mass + (1.0 - compress) * (1.0 - novelty))
        return {
            "size": len(self.local_chart),
            "capacity": self.capacity,
            "invariants_mass": self.invariants_mass,
        }

    def consolidate(self, strength: float) -> float:
        self.invariants_mass = min(1.5, self.invariants_mass + 0.2 * strength)
        return self.invariants_mass

    def prune(self, ratio: float) -> int:
        target = max(1, int(self.capacity * (1.0 - ratio)))
        while len(self.local_chart) > target:
            self.local_chart.popleft(); self.reconstructions.popleft(); self.recirculation.popleft()
        return len(self.local_chart)
