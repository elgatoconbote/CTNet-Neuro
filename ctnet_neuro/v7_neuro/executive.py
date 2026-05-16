from __future__ import annotations

from dataclasses import dataclass


MODES = ("intent", "explore", "decide", "reflect", "stabilize", "sleep", "repair")


@dataclass
class ExecutiveLocal:
    mode: str = "intent"

    def update(self, stress: float, threat: float, energy: float) -> str:
        if energy < 0.25:
            self.mode = "sleep"
        elif stress > 0.8:
            self.mode = "repair"
        elif threat > 0.65:
            self.mode = "decide"
        elif stress > 0.55:
            self.mode = "stabilize"
        elif threat < 0.2 and energy > 0.7:
            self.mode = "explore"
        elif stress < 0.3:
            self.mode = "reflect"
        else:
            self.mode = "intent"
        return self.mode

    def profile(self) -> dict[str, float]:
        table = {
            "intent": (1.0, 0.6, 0.5, 0.6, 0.5, 0.2),
            "explore": (1.2, 0.9, 0.4, 0.9, 0.4, 0.1),
            "decide": (1.1, 0.7, 0.6, 0.8, 0.6, 0.1),
            "reflect": (0.8, 0.5, 0.8, 0.5, 0.8, 0.3),
            "stabilize": (0.7, 0.4, 0.9, 0.5, 0.9, 0.4),
            "sleep": (0.4, 0.2, 1.0, 0.2, 0.9, 1.0),
            "repair": (0.6, 0.3, 0.9, 0.4, 1.0, 0.7),
        }
        upd, babel, mem_comp, recirc, verify, offline = table[self.mode]
        return {
            "update_rate": upd,
            "babel_open": babel,
            "memory_compression": mem_comp,
            "recirculation": recirc,
            "verify_priority": verify,
            "offline_bias": offline,
        }
