from __future__ import annotations


def collect_v7_neuro_metrics(orchestrator) -> dict:
    atlas = orchestrator.atlas
    return {
        "global_coherence": atlas.total_coherence(),
        "closure_error": atlas.closure_error(),
        "local_coherences": [n.state.local_metrics.C_total for n in atlas.nodes],
        "modes": [n.state.executive.get("mode", "intent") for n in atlas.nodes],
    }
