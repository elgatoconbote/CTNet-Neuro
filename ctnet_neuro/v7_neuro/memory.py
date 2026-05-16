from __future__ import annotations

from collections import deque


class LocalMemory:
    def __init__(self, capacity: int = 32):
        self.capacity = capacity
        self._atlas = deque(maxlen=capacity)
        self.invariant_mass = 0.0

    def update(self, local_vec: list[float], reconstructed_vec: list[float], recirculation: float, compression: float) -> dict:
        compressed = [(1.0 - compression) * v + compression * r for v, r in zip(local_vec, reconstructed_vec)]
        self._atlas.append(compressed)
        mean_abs = sum(abs(x) for x in compressed) / max(1, len(compressed))
        self.invariant_mass = 0.92 * self.invariant_mass + 0.08 * mean_abs + 0.02 * recirculation
        return {
            "capacity": self.capacity,
            "size": len(self._atlas),
            "invariant_mass": self.invariant_mass,
            "last": compressed,
        }

    def consolidate(self, factor: float) -> None:
        self.invariant_mass *= (1.0 + 0.1 * factor)
