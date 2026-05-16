from __future__ import annotations


def body_snapshot(global_state) -> dict:
    b = getattr(global_state, "body_state", None)
    if b is None:
        return {"energy": 0.8, "stress": 0.2, "pain": 0.1, "interoception": 0.5, "proprioception": 0.5, "autonomic_load": 0.2}
    return {
        "energy": float(getattr(b, "energy", 0.8)),
        "stress": float(getattr(b, "stress", 0.2)),
        "pain": float(getattr(b, "pain", 0.1)),
        "interoception": float(getattr(b, "interoception", 0.5)),
        "proprioception": float(getattr(b, "proprioception", 0.5)),
        "autonomic_load": float(getattr(b, "autonomic_load", 0.2)),
    }
