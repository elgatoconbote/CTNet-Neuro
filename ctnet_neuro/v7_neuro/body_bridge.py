from __future__ import annotations


def project_body_state(body_state) -> dict:
    return {
        "energy": float(getattr(body_state, "energy", 0.6)),
        "stress": float(getattr(body_state, "stress", 0.4)),
        "pain": float(getattr(body_state, "pain", 0.0)),
        "interoception": float(getattr(body_state, "interoception", 0.0)),
        "proprioception": float(getattr(body_state, "proprioception", 0.0)),
        "autonomic_load": float(getattr(body_state, "autonomic_load", 0.0)),
    }
