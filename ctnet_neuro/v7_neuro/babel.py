from __future__ import annotations


class LocalBabel:
    def __init__(self, node_id: int):
        self.node_id = node_id

    def read(self, dynamic: float, symbolic: dict[str, float], up_signal: float, openness: float) -> dict:
        val = 0.5 * dynamic + 0.3 * symbolic.get("intent", 0.0) + 0.2 * up_signal
        unverified = max(0.0, abs(dynamic - up_signal) * (1.0 - openness))
        return {
            "node": self.node_id,
            "structural_response": f"node-{self.node_id}:mode={symbolic.get('mode','intent')}:val={val:.3f}",
            "frontier_unverified": unverified,
            "value": val,
        }


class LocalVerifier:
    def verify(self, readout: dict, local_dynamic: float, neighbor_projection: float, priority: float) -> dict:
        mismatch = abs(local_dynamic - neighbor_projection)
        consistency = max(0.0, min(1.0, 1.0 - mismatch * priority))
        readout["consistency"] = consistency
        readout["verified"] = consistency > 0.35
        return readout


def global_consensus(readouts: list[dict], coherences: list[float]) -> str:
    if not readouts:
        return ""
    weighted = sum(r["value"] * c for r, c in zip(readouts, coherences)) / max(1e-6, sum(coherences))
    verified_frac = sum(1 for r in readouts if r.get("verified", False)) / len(readouts)
    return f"consensus:value={weighted:.3f};verified={verified_frac:.2f};nodes={len(readouts)}"
