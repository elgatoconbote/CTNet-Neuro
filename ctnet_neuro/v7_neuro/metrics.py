from __future__ import annotations


def collect_v7_neuro_metrics(orch, closure_errors: list[float], readout: str, offline_active: bool) -> dict:
    return {
        "global_coherence": orch.atlas.total_coherence(),
        "mean_local_coherence": sum(n.state.local_metrics.get("C_local", 0.0) for n in orch.nodes) / max(1, len(orch.nodes)),
        "closure_error": sum(closure_errors) / max(1, len(closure_errors)),
        "offline_active": offline_active,
        "readout": readout,
    }
