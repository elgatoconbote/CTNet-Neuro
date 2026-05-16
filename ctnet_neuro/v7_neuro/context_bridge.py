from __future__ import annotations


def project_context_state(context_state, world_state) -> dict:
    return {
        "hypotheses": len(getattr(context_state, "hypotheses", [])),
        "compatibility": float(getattr(context_state, "compatibility_trace", 0.3)),
        "incompatibility": float(getattr(context_state, "incompatibility_trace", 0.1)),
        "grammar": float(getattr(context_state, "grammar_trace", 0.0)),
        "return_signal": float(getattr(context_state, "return_trace", 0.0)),
        "active_regime": getattr(world_state, "regime_label", "orient"),
    }
