from __future__ import annotations


def project_context(context_state, world_state) -> dict:
    hyp = len(getattr(context_state, "hypotheses", []))
    return {
        "hypotheses": hyp,
        "compatibility": float(getattr(context_state, "compatibility_trace", 0.0)),
        "incompatibility": float(getattr(context_state, "incompatibility_trace", 0.0)),
        "grammar": float(getattr(context_state, "grammar_trace", 0.0)),
        "return_signal": float(getattr(context_state, "return_trace", 0.0)),
        "active_regime": str(getattr(world_state, "regime_label", "orient")),
    }
