from __future__ import annotations


class LocalBabel:
    def read(self, node_id: int, symbolic: dict, dynamic: list[float], up: float, open_factor: float) -> dict:
        struct = f"node={node_id}|intent={symbolic.get('intent',0):.2f}|signal={sum(dynamic)/max(1,len(dynamic)):.3f}"
        boundary = max(0.0, 1.0 - open_factor) * (1.0 - min(1.0, abs(up)))
        return {"structural_response": struct, "unverified_frontier": boundary}


class LocalVerifier:
    def verify(self, babel_read: dict, coherence_total: float, priority: float) -> dict:
        consistency = max(0.0, min(1.0, 0.7 * coherence_total + 0.3 * priority))
        return {
            "consistency": consistency,
            "verified": consistency >= 0.45 and babel_read["unverified_frontier"] <= 0.7,
        }


def global_consensus(readouts: list[dict], local_coherences: list[float]) -> str:
    weights = [max(1e-6, c) for c in local_coherences]
    best = max(range(len(readouts)), key=lambda i: weights[i])
    return f"consensus[{best}]::{readouts[best]['structural_response']}"
