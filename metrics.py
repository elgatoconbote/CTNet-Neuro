from __future__ import annotations

from dataclasses import asdict
import math

from state import CerebroVirtualState


def _norm5(v0, v1, v2, v3, v4):
    n = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4)
    if n <= 1e-9:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (v0 / n, v1 / n, v2 / n, v3 / n, v4 / n)


def _dot5(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4]


def _norm_row(row):
    n = math.sqrt(sum(x * x for x in row))
    if n <= 1e-9:
        return [0.0 for _ in row]
    return [x / n for x in row]


def _pairwise_signature_stats(hypotheses):
    vecs = [
        _norm5(
            h.signature_projection,
            h.signature_tissue,
            h.signature_structure,
            h.signature_salience,
            h.signature_body,
        )
        for h in hypotheses
    ]
    sims = []
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            sims.append(_dot5(vecs[i], vecs[j]))
    if not sims:
        return 0.0, 0.0, 0.0
    return sum(sims) / len(sims), min(sims), max(sims)


def _regime_overlap_mean(binding):
    rows = [_norm_row(r) for r in binding]
    vals = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vals.append(sum(a * b for a, b in zip(rows[i], rows[j])))
    return sum(vals) / len(vals) if vals else 0.0


def _lineage_diversity_mean(hypotheses):
    vals = []
    for h in hypotheses:
        v = [
            abs(h.signature_projection),
            abs(h.signature_tissue),
            abs(h.signature_structure),
            abs(h.signature_salience),
            abs(h.signature_body),
        ]
        total = sum(v) or 1.0
        probs = [x / total for x in v]
        ent = 0.0
        for p in probs:
            if p > 0.0:
                ent -= p * math.log(p)
        vals.append(1.0 - ent / math.log(5.0))
    return sum(vals) / len(vals) if vals else 0.0


def _matrix_mean(m):
    vals = []
    for i in range(len(m)):
        for j in range(len(m[i])):
            if i == j:
                continue
            vals.append(m[i][j])
    return sum(vals) / len(vals) if vals else 0.0


def _binding_row_stats(binding):
    rows = [sum(r) for r in binding]
    if not rows:
        return 0.0, 0.0, 0.0
    return min(rows), max(rows), sum(rows) / len(rows)


def _binding_col_stats(binding):
    if not binding:
        return 0.0, 0.0, 0.0
    ncols = len(binding[0])
    cols = [sum(binding[i][j] for i in range(len(binding))) for j in range(ncols)]
    return min(cols), max(cols), sum(cols) / len(cols)


def _dead_slot_fraction(binding, threshold=0.72):
    if not binding:
        return 0.0
    rows = [sum(r) for r in binding]
    dead = sum(1 for x in rows if x < threshold)
    return dead / len(rows)


def _hub_pressure_mean(binding, target=1.52):
    if not binding:
        return 0.0
    rows = [sum(r) for r in binding]
    vals = [max(0.0, x - target) / max(target, 1e-6) for x in rows]
    return sum(vals) / len(vals)


def _binding_specificity_mean(binding):
    vals = []
    for row in binding:
        if not row:
            continue
        best = max(row)
        others = sorted(row, reverse=True)
        second = others[1] if len(others) > 1 else 0.0
        vals.append(max(0.0, best - second))
    return sum(vals) / len(vals) if vals else 0.0


def collect_metrics(state: CerebroVirtualState) -> dict:
    active = state.context_state.hypotheses[state.context_state.active_index]
    material = state.tissue_state.material_ref
    pair_mean, pair_min, pair_max = _pairwise_signature_stats(state.context_state.hypotheses)
    row_min, row_max, row_mean = _binding_row_stats(state.context_state.slot_regime_binding)
    col_min, col_max, col_mean = _binding_col_stats(state.context_state.slot_regime_binding)

    out = {
        "step_count": state.world_state.step_count,
        "world_regime_label": state.world_state.regime_label,
        "world_regime_index": state.world_state.regime_index,
        "world_regime_transition_signal": state.world_state.regime_transition_signal,
        "world_homeostatic_drive": state.world_state.homeostatic_drive,
        "world_threat_drive": state.world_state.threat_drive,
        "world_novelty_drive": state.world_state.novelty_drive,
        "mean_excitation": state.tissue_state.mean_excitation,
        "energy": state.tissue_state.energy,
        "working_memory": state.meso_state.working_memory,
        "projection_readout": state.meso_state.projection_readout,
        "active_context_slot": active.slot,
        "active_context_mean": active.mean,
        "active_context_confidence": active.confidence,
        "active_context_signature_projection": active.signature_projection,
        "active_context_signature_tissue": active.signature_tissue,
        "active_context_signature_structure": active.signature_structure,
        "active_context_signature_salience": active.signature_salience,
        "active_context_signature_body": active.signature_body,
        "active_context_return_bias": active.return_bias,
        "active_context_transition_credit": active.transition_credit,
        "context_alignment": state.context_state.last_alignment,
        "context_entropy": state.context_state.competition_entropy,
        "context_pairwise_mean_similarity": pair_mean,
        "context_pairwise_min_similarity": pair_min,
        "context_pairwise_max_similarity": pair_max,
        "context_regime_overlap_mean": _regime_overlap_mean(state.context_state.slot_regime_binding),
        "context_lineage_diversity_mean": _lineage_diversity_mean(state.context_state.hypotheses),
        "context_binding_specificity_mean": _binding_specificity_mean(state.context_state.slot_regime_binding),
        "context_binding_row_min": row_min,
        "context_binding_row_max": row_max,
        "context_binding_row_mean": row_mean,
        "context_binding_col_min": col_min,
        "context_binding_col_max": col_max,
        "context_binding_col_mean": col_mean,
        "context_dead_slot_fraction": _dead_slot_fraction(state.context_state.slot_regime_binding),
        "context_hub_pressure_mean": _hub_pressure_mean(state.context_state.slot_regime_binding),
        "context_active_duration": state.context_state.active_duration,
        "context_transition_trace": state.context_state.transition_trace,
        "context_compatibility_trace": state.context_state.compatibility_trace,
        "context_incompatibility_trace": state.context_state.incompatibility_trace,
        "context_transition_matrix_mean": _matrix_mean(state.context_state.transition_matrix),
        "context_compatibility_matrix_mean": _matrix_mean(state.context_state.compatibility_matrix),
        "context_incompatibility_matrix_mean": _matrix_mean(state.context_state.incompatibility_matrix),
        "context_persistence_illegitimacy": state.context_state.persistence_illegitimacy,
        "context_best_challenger_margin": state.context_state.best_challenger_margin,
        "context_challenger_index": state.context_state.challenger_index,
        "context_regime_transition_matrix_mean": _matrix_mean(state.context_state.regime_transition_matrix),
        "context_slot_regime_binding_mean": sum(sum(r for r in row) for row in state.context_state.slot_regime_binding) / max(1, len(state.context_state.slot_regime_binding) * 4),
        "context_grammar_trace": state.context_state.grammar_trace,
        "context_return_trace": state.context_state.return_trace,
        "context_transition_grammar_mean": _matrix_mean(state.context_state.transition_grammar_matrix),
        "context_return_path_mean": _matrix_mean(state.context_state.return_path_matrix),
        "development_progress": state.development_state.differentiation_progress,
        "critical_period_open": state.development_state.critical_period_open,
        "development_graph_maturation": state.development_state.graph_maturation,
        "development_graph_separation": state.development_state.graph_separation,
        "development_grammar_stability": state.development_state.grammar_stability,
        "development_return_stability": state.development_state.return_stability,
        "development_closure_tension": state.development_state.closure_tension,
        "body_energy": state.body_state.energy,
        "body_stress": state.body_state.stress,
        "body_pain": state.body_state.pain,
        "body_autonomic_load": state.body_state.autonomic_load,
        "offline_asleep": state.offline_state.asleep,
        "offline_graph_replay_strength": state.offline_state.graph_replay_strength,
        "offline_graph_consolidation_gain": state.offline_state.graph_consolidation_gain,
        "offline_graph_pruning_signal": state.offline_state.graph_pruning_signal,
        "tissue_patch_delta": state.tissue_state.patch_delta,
        "tissue_readout_delta": state.tissue_state.readout_b - state.tissue_state.readout_a,
        "tissue_structural_signal": state.tissue_state.structural_signal,
        "tissue_local_plasticity": state.tissue_state.local_plasticity,
    }

    if material is not None and getattr(material, "runtime_summary", None):
        rs = material.runtime_summary
        out["runtime_active_frac"] = rs.get("active_frac")
        out["runtime_tag_mean"] = rs.get("tag_mean")
        out["runtime_mean_abs_a"] = rs.get("mean_abs_a")

    return out


def flatten_state(state: CerebroVirtualState) -> dict:
    return asdict(state)
