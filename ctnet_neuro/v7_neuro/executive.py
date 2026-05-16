from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExecutiveSignal:
    mode: str
    update_rate: float
    babel_opening: float
    memory_compression: float
    recirculation_strength: float
    verifier_priority: float
    offline_push: float


class ExecutiveLocal:
    MODES = {
        "intent": ExecutiveSignal("intent", 1.00, 0.70, 0.30, 0.60, 0.60, 0.20),
        "explore": ExecutiveSignal("explore", 1.12, 0.92, 0.20, 0.65, 0.35, 0.10),
        "decide": ExecutiveSignal("decide", 0.96, 0.62, 0.45, 0.70, 0.88, 0.15),
        "reflect": ExecutiveSignal("reflect", 0.82, 0.55, 0.55, 0.55, 0.82, 0.25),
        "stabilize": ExecutiveSignal("stabilize", 0.75, 0.48, 0.68, 0.82, 0.92, 0.35),
        "sleep": ExecutiveSignal("sleep", 0.58, 0.35, 0.80, 0.40, 0.95, 0.95),
        "repair": ExecutiveSignal("repair", 0.72, 0.42, 0.74, 0.85, 0.98, 0.55),
    }

    def select(self, body: dict[str, float], context: dict[str, float], offline: bool) -> ExecutiveSignal:
        if offline:
            return self.MODES["sleep"]
        if body.get("pain", 0.0) > 0.6 or body.get("stress", 0.0) > 0.72:
            return self.MODES["repair"]
        if context.get("compatibility", 0.0) > 0.68:
            return self.MODES["decide"]
        if body.get("energy", 0.0) > 0.7 and context.get("incompatibility", 0.0) < 0.35:
            return self.MODES["explore"]
        return self.MODES["reflect"]
