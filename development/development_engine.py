from __future__ import annotations

import math

from cerebro_virtual.state import CerebroVirtualState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm5(v0, v1, v2, v3, v4):
    n = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4)
    if n <= 1e-9:
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    return (v0 / n, v1 / n, v2 / n, v3 / n, v4 / n)


def _dot5(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2] + a[3]*b[3] + a[4]*b[4]


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


def _norm_row(row):
    n = math.sqrt(sum(x * x for x in row))
    if n <= 1e-9:
        return [0.0 for _ in row]
    return [x / n for x in row]


def _regime_overlap_stats(binding):
    rows = [_norm_row(r) for r in binding]
    vals = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vals.append(sum(a * b for a, b in zip(rows[i], rows[j])))
    return sum(vals) / len(vals) if vals else 0.0


def _row_stats(binding):
    rows = [sum(r) for r in binding]
    if not rows:
        return 0.0, 0.0, 0.0
    return min(rows), max(rows), sum(rows) / len(rows)


class DevelopmentEngine:
    def step(self, state: CerebroVirtualState) -> None:
        dev = state.development_state
        tissue = state.tissue_state
        meso = state.meso_state
        context = state.context_state
        body = state.body_state
        offline = state.offline_state

        pair_mean, pair_min, pair_max = _pairwise_signature_stats(context.hypotheses)
        regime_overlap_mean = _regime_overlap_stats(context.slot_regime_binding)
        row_min, row_max, row_mean = _row_stats(context.slot_regime_binding)

        grammar_strength = _clip(
            0.14 * context.grammar_trace
            + 0.16 * context.return_trace
            + 0.10 * context.transition_trace
            + 0.08 * context.compatibility_trace,
            0.0,
            1.0,
        )

        maturation_drive = _clip(
            0.10 * grammar_strength
            + 0.14 * offline.graph_consolidation_gain
            + 0.10 * tissue.structural_signal
            + 0.08 * max(0.0, 1.0 - body.stress),
            0.0,
            1.0,
        )

        dev.graph_maturation = _clip(
            0.988 * dev.graph_maturation + 0.018 * maturation_drive,
            0.0,
            1.0,
        )

        sig_corridor = 1.0 - min(1.0, abs(pair_mean - 0.52) / 0.24)
        bind_corridor = 1.0 - min(1.0, abs(regime_overlap_mean - 0.58) / 0.24)
        row_corridor = 1.0 - min(1.0, abs(row_mean - 0.92) / 0.30)
        comp_corridor = 1.0 - min(1.0, abs(context.compatibility_trace - 0.40) / 0.22)
        incomp_corridor = 1.0 - min(1.0, abs(context.incompatibility_trace - 0.42) / 0.22)
        grammar_corridor = min(1.0, context.grammar_trace / 0.10)
        return_corridor = min(1.0, context.return_trace / 0.05)

        separation_target = _clip(
            0.18 * sig_corridor
            + 0.18 * bind_corridor
            + 0.14 * row_corridor
            + 0.14 * comp_corridor
            + 0.14 * incomp_corridor
            + 0.10 * grammar_corridor
            + 0.12 * return_corridor,
            0.0,
            1.0,
        )

        dev.graph_separation = _clip(
            0.975 * dev.graph_separation + 0.03 * separation_target,
            0.0,
            1.0,
        )

        activity = abs(tissue.mean_excitation) + meso.salience_map + 0.25 * context.last_alignment

        dev.differentiation_progress = _clip(
            dev.differentiation_progress
            + 0.003 * activity
            + 0.003 * dev.graph_maturation,
            0.0,
            1.0,
        )

        dev.pruning_pressure = _clip(
            0.97 * dev.pruning_pressure
            + 0.02 * max(0.0, regime_overlap_mean - 0.72) / 0.28
            + 0.01 * max(0.0, body.stress),
            0.0,
            1.0,
        )

        dev.developmental_memory = _clip(
            0.990 * dev.developmental_memory
            + 0.006 * context.last_alignment
            + 0.006 * dev.return_stability,
            0.0,
            1.0,
        )

        active = context.hypotheses[context.active_index]
        confidence_excess = max(0.0, active.confidence - 0.88) / 0.12

        dev.closure_tension = _clip(
            0.20 * dev.differentiation_progress
            + 0.10 * dev.graph_maturation
            - 0.10 * dev.graph_separation
            - 0.04 * context.return_trace
            - 0.04 * context.grammar_trace
            + 0.08 * confidence_excess
            + 0.06 * max(0.0, regime_overlap_mean - 0.72) / 0.28,
            0.0,
            1.0,
        )

        dev.critical_period_open = not (
            dev.differentiation_progress > 0.92
            and dev.graph_maturation > 0.35
            and 0.28 < pair_mean < 0.70
            and 0.40 < regime_overlap_mean < 0.76
            and row_min > 0.70
            and row_max < 1.60
            and 0.20 < context.compatibility_trace < 0.60
            and 0.20 < context.incompatibility_trace < 0.60
            and 0.02 < context.grammar_trace < 0.16
            and 0.01 < context.return_trace < 0.08
            and active.confidence < 0.88
        )

        if dev.progenitors > 0 and dev.critical_period_open and activity > 0.45:
            dev.progenitors -= 1

        dev.phase = "plastic" if dev.critical_period_open else "stabilizing"
