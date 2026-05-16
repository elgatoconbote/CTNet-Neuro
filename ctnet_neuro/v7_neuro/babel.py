from __future__ import annotations


def local_read(node_id: int, regime: str, activation: float, coherence: float) -> dict:
    return {"node": node_id, "regime": regime, "structural_response": f"node_{node_id}:{regime}:{activation:.3f}", "unverified_frontier": max(0.0, 1.0 - coherence)}


def verify_local(local_readout: dict, threshold: float) -> dict:
    c = 1.0 - local_readout["unverified_frontier"]
    return {"consistent": c >= threshold, "score": c}


def consensus(readouts: list[dict], weights: list[float]) -> str:
    if not readouts:
        return "no_readout"
    idx = max(range(len(readouts)), key=lambda i: weights[i])
    return readouts[idx]["structural_response"]
