from __future__ import annotations

MODES = ("intent", "explore", "decide", "reflect", "stabilize", "sleep", "repair")


class ExecutiveLocal:
    def step(self, regime: str, body_stress: float, coherence: float) -> dict:
        if body_stress > 0.75:
            mode = "repair"
        elif coherence < 0.35:
            mode = "reflect"
        elif regime == "threat":
            mode = "decide"
        elif regime == "recover":
            mode = "stabilize"
        elif regime == "goal_recontext":
            mode = "explore"
        else:
            mode = "intent"
        if regime == "recover" and body_stress < 0.2:
            mode = "sleep"
        gains = {
            "intent": dict(update_rate=1.0, babel_open=0.7, compress=0.35, recirculation=0.4, verify=0.6, offline=0.2),
            "explore": dict(update_rate=1.1, babel_open=0.9, compress=0.25, recirculation=0.5, verify=0.45, offline=0.15),
            "decide": dict(update_rate=1.2, babel_open=0.55, compress=0.45, recirculation=0.45, verify=0.7, offline=0.1),
            "reflect": dict(update_rate=0.9, babel_open=0.6, compress=0.5, recirculation=0.3, verify=0.9, offline=0.4),
            "stabilize": dict(update_rate=0.8, babel_open=0.5, compress=0.55, recirculation=0.25, verify=0.85, offline=0.5),
            "sleep": dict(update_rate=0.45, babel_open=0.2, compress=0.75, recirculation=0.15, verify=0.95, offline=0.95),
            "repair": dict(update_rate=0.7, babel_open=0.35, compress=0.65, recirculation=0.2, verify=0.95, offline=0.8),
        }
        out = gains[mode].copy(); out["mode"] = mode
        return out
