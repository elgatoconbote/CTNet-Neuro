from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _row_sum(row: list[float]) -> float:
    return float(sum(row)) if row else 0.0


def _pairwise(values: list[list[float]]) -> tuple[float, float, float]:
    sims: list[float] = []
    n = len(values)
    for i in range(n):
        a = values[i]
        na = sum(x * x for x in a) ** 0.5
        for j in range(i + 1, n):
            b = values[j]
            nb = sum(x * x for x in b) ** 0.5
            if na <= 1e-9 or nb <= 1e-9:
                sims.append(0.0)
            else:
                sims.append(sum(x * y for x, y in zip(a, b)) / (na * nb))
    if not sims:
        return 0.0, 0.0, 0.0
    return sum(sims) / len(sims), min(sims), max(sims)


def _binding_specificity(rows: list[list[float]]) -> float:
    vals = []
    for row in rows:
        if not row:
            continue
        xs = sorted(row, reverse=True)
        if len(xs) < 2:
            vals.append(xs[0])
        else:
            vals.append(max(0.0, xs[0] - xs[1]))
    return sum(vals) / len(vals) if vals else 0.0


def _graph_stats(state) -> dict[str, float]:
    ctx = state.context_state
    rows = ctx.slot_regime_binding

    row_sums = [_row_sum(r) for r in rows]
    col_sums = [sum(rows[i][j] for i in range(len(rows))) for j in range(4)]
    pair_mean, pair_min, pair_max = _pairwise(rows)

    return {
        "row_min": min(row_sums) if row_sums else 0.0,
        "row_max": max(row_sums) if row_sums else 0.0,
        "row_mean": sum(row_sums) / len(row_sums) if row_sums else 0.0,
        "col_mean": sum(col_sums) / len(col_sums) if col_sums else 0.0,
        "pair_mean": pair_mean,
        "pair_min": pair_min,
        "pair_max": pair_max,
        "specificity": _binding_specificity(rows),
        "grammar": float(ctx.grammar_trace),
        "return_": float(ctx.return_trace),
        "transition": float(ctx.transition_trace),
        "compat": float(ctx.compatibility_trace),
        "incompat": float(ctx.incompatibility_trace),
        "alignment": float(ctx.last_alignment),
        "challenger_margin": float(ctx.best_challenger_margin),
    }



def _get_closure_tension(dev) -> float:
    if hasattr(dev, "graph_closure_tension"):
        return float(dev.graph_closure_tension)
    if hasattr(dev, "closure_tension"):
        return float(dev.closure_tension)
    return 0.0


def _set_closure_tension(dev, value: float) -> None:
    value = float(value)
    if hasattr(dev, "graph_closure_tension"):
        dev.graph_closure_tension = value
    if hasattr(dev, "closure_tension"):
        dev.closure_tension = value
    if not hasattr(dev, "graph_closure_tension") and not hasattr(dev, "closure_tension"):
        setattr(dev, "closure_tension", value)


def _phase_from_graph(state) -> str:
    g = _graph_stats(state)
    dev = state.development_state
    body = state.body_state

    if g["specificity"] < 0.22 or g["row_min"] < 0.72:
        return "scaffold"
    if g["grammar"] < 0.18 or g["return_"] < 0.035:
        return "refinement"
    if g["alignment"] > 0.96 and g["specificity"] > 0.28 and g["return_"] > 0.05 and body.stress < 0.40:
        return "sensorimotor_binding"
    if dev.graph_separation > 0.75 and g["grammar"] > 0.22 and g["return_"] > 0.055:
        return "recontextualization"
    return "refinement"


@dataclass
class CV1Log:
    phase_counts: dict[str, int] = field(default_factory=lambda: {
        "scaffold": 0,
        "refinement": 0,
        "sensorimotor_binding": 0,
        "recontextualization": 0,
    })
    phase_trace: list[str] = field(default_factory=list)


def apply_cv1_supervisor(state, log: CV1Log) -> None:
    g = _graph_stats(state)
    dev = state.development_state
    body = state.body_state
    world = state.world_state
    offline = state.offline_state
    tissue = state.tissue_state

    phase = _phase_from_graph(state)
    log.phase_counts[phase] += 1
    log.phase_trace.append(phase)

    # 1) Ontogenia como memoria estructural guiada por el grafo
    coherence_gain = 0.35 * g["alignment"] + 0.25 * g["grammar"] + 0.20 * g["return_"] + 0.20 * g["specificity"]
    dev.developmental_memory = _clip(
        0.995 * dev.developmental_memory + 0.010 * coherence_gain,
        0.0,
        1.0,
    )

    dev.graph_maturation = _clip(
        0.992 * dev.graph_maturation
        + 0.010 * g["grammar"]
        + 0.010 * g["return_"]
        + 0.006 * g["alignment"],
        0.0,
        1.0,
    )

    dev.graph_separation = _clip(
        0.992 * dev.graph_separation
        + 0.012 * g["specificity"]
        + 0.008 * max(0.0, 1.0 - g["pair_mean"]),
        0.0,
        1.0,
    )

    dev.graph_closure_tension = _clip(
        0.992 * dev.graph_closure_tension
        + 0.008 * max(0.0, g["compat"] - g["return_"])
        + 0.006 * max(0.0, g["transition"] - g["grammar"]),
        0.0,
        1.0,
    )

    # 2) Períodos críticos / cierres relativos
    close_signal = 0.40 * dev.graph_maturation + 0.30 * dev.graph_separation + 0.20 * g["return_"] + 0.10 * g["grammar"]
    reopen_signal = 0.35 * max(0.0, 0.20 - g["specificity"]) + 0.25 * max(0.0, 0.72 - g["row_min"]) + 0.20 * body.stress + 0.20 * _get_closure_tension(dev)

    if close_signal > 0.42 and reopen_signal < 0.18:
        dev.critical_period_open = False
    elif reopen_signal > 0.20:
        dev.critical_period_open = True

    # 3) Modulación por época ontogenética
    if phase == "scaffold":
        dev.pruning_pressure = _clip(0.985 * dev.pruning_pressure + 0.004, 0.0, 1.0)
        tissue.local_plasticity = _clip(tissue.local_plasticity + 0.010, 0.0, 1.0)
        body.stress = _clip(body.stress * 0.94, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load * 0.95, 0.0, 1.0)

    elif phase == "refinement":
        dev.pruning_pressure = _clip(
            0.985 * dev.pruning_pressure
            + 0.006 * g["grammar"]
            + 0.004 * g["specificity"],
            0.0,
            1.0,
        )
        tissue.local_plasticity = _clip(
            0.996 * tissue.local_plasticity
            + 0.006 * g["transition"]
            + 0.004 * g["grammar"],
            0.0,
            1.0,
        )
        body.autonomic_load = _clip(
            0.990 * body.autonomic_load
            + 0.010 * max(0.0, 0.20 - g["return_"]),
            0.0,
            1.0,
        )

    elif phase == "sensorimotor_binding":
        body.energy = _clip(body.energy + 0.006 * g["alignment"] - 0.002 * body.stress, 0.08, 1.0)
        body.stress = _clip(body.stress - 0.010 * g["return_"], 0.0, 1.0)
        tissue.structural_signal = _clip(
            tissue.structural_signal
            + 0.008 * g["return_"]
            + 0.006 * g["grammar"],
            0.0,
            1.0,
        )
        dev.pruning_pressure = _clip(
            0.992 * dev.pruning_pressure
            + 0.004 * max(0.0, g["specificity"] - 0.25),
            0.0,
            1.0,
        )

    elif phase == "recontextualization":
        body.autonomic_load = _clip(
            0.992 * body.autonomic_load
            + 0.006 * world.regime_transition_signal
            - 0.006 * g["return_"],
            0.0,
            1.0,
        )
        dev.graph_closure_tension = _clip(
            0.992 * dev.graph_closure_tension
            + 0.010 * world.regime_transition_signal
            - 0.006 * g["return_"],
            0.0,
            1.0,
        )

    # 4) Sueño / replay ontogenético
    if offline.asleep:
        offline.graph_replay_strength = _clip(
            0.992 * offline.graph_replay_strength
            + 0.012 * g["grammar"]
            + 0.010 * g["return_"],
            0.0,
            1.0,
        )
        offline.graph_consolidation_gain = _clip(
            0.992 * offline.graph_consolidation_gain
            + 0.010 * dev.graph_maturation
            + 0.008 * dev.graph_separation,
            0.0,
            1.0,
        )
        offline.graph_pruning_signal = _clip(
            0.992 * offline.graph_pruning_signal
            + 0.008 * dev.pruning_pressure
            - 0.004 * g["return_"],
            0.0,
            1.0,
        )

    # 5) Progreso ontogenético total
    epoch_gain = (
        0.20 * dev.developmental_memory
        + 0.20 * dev.graph_maturation
        + 0.20 * dev.graph_separation
        + 0.15 * g["grammar"]
        + 0.15 * g["return_"]
        + 0.10 * g["alignment"]
    )
    dev.differentiation_progress = _clip(
        dev.differentiation_progress + 0.0025 * epoch_gain,
        0.0,
        1.0,
    )


def collect_cv1_summary(state, log: CV1Log) -> dict[str, Any]:
    g = _graph_stats(state)
    dev = state.development_state
    body = state.body_state
    world = state.world_state
    offline = state.offline_state

    return {
        "phase_counts": dict(log.phase_counts),
        "last_phase": log.phase_trace[-1] if log.phase_trace else None,
        "graph_row_min": g["row_min"],
        "graph_row_max": g["row_max"],
        "graph_pair_mean": g["pair_mean"],
        "graph_specificity": g["specificity"],
        "graph_alignment": g["alignment"],
        "graph_transition": g["transition"],
        "graph_grammar": g["grammar"],
        "graph_return": g["return_"],
        "developmental_memory": float(dev.developmental_memory),
        "graph_maturation": float(dev.graph_maturation),
        "graph_separation": float(dev.graph_separation),
        "graph_closure_tension": float(_get_closure_tension(dev)),
        "critical_period_open": bool(dev.critical_period_open),
        "body_energy": float(body.energy),
        "body_stress": float(body.stress),
        "body_autonomic_load": float(body.autonomic_load),
        "world_regime_label": getattr(world, "regime_label", None),
        "world_regime_index": getattr(world, "regime_index", None),
        "offline_graph_replay_strength": float(offline.graph_replay_strength),
        "offline_graph_consolidation_gain": float(offline.graph_consolidation_gain),
        "offline_graph_pruning_signal": float(offline.graph_pruning_signal),
    }
