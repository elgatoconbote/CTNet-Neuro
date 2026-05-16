from __future__ import annotations


def context_snapshot(global_state) -> dict:
    c = getattr(global_state, "context_state", None)
    if c is None:
        return {"hypotheses": [], "compatibility": 0.6, "incompatibility": 0.2, "grammar": 0.5, "return_signal": 0.4, "active_regime": "orient"}
    hypotheses = [getattr(h, "mean", 0.0) for h in getattr(c, "hypotheses", [])]
    return {
        "hypotheses": hypotheses,
        "compatibility": float(getattr(c, "compatibility_trace", 0.0)),
        "incompatibility": float(getattr(c, "incompatibility_trace", 0.0)),
        "grammar": float(getattr(c, "grammar_trace", 0.0)),
        "return_signal": float(getattr(c, "return_trace", 0.0)),
        "active_regime": int(getattr(c, "active_index", 0)),
    }
