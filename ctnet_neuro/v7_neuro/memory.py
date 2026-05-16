from __future__ import annotations

from collections import deque


class LocalMemory:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self.topological_buffer: deque[float] = deque(maxlen=capacity)
        self.invariant_mass: float = 0.0
        self.last_reconstruction: float = 0.0

    def recirculate(self, local_value: float, neighbor_value: float, compression: float) -> dict:
        projected = 0.5 * local_value + 0.5 * neighbor_value
        self.topological_buffer.append(projected)
        if self.topological_buffer:
            self.last_reconstruction = sum(self.topological_buffer) / len(self.topological_buffer)
        self.invariant_mass = (1.0 - 0.1 * compression) * self.invariant_mass + 0.1 * abs(projected)
        return self.snapshot()

    def consolidate(self, gain: float) -> None:
        self.invariant_mass = min(10.0, self.invariant_mass * (1.0 + 0.1 * gain))

    def prune(self, amount: float) -> None:
        keep = max(1, int(round(self.capacity * (1.0 - min(0.9, amount)))))
        while len(self.topological_buffer) > keep:
            self.topological_buffer.popleft()

    def snapshot(self) -> dict:
        return {
            "capacity": self.capacity,
            "size": len(self.topological_buffer),
            "invariant_mass": self.invariant_mass,
            "reconstruction": self.last_reconstruction,
        }
