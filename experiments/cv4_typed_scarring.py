from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _softmax(vals: list[float], temp: float = 1.0) -> list[float]:
    import math
    if not vals:
        return []
    scaled = [v / max(1e-9, temp) for v in vals]
    m = max(scaled)
    exps = [math.exp(v - m) for v in scaled]
    s = sum(exps) or 1.0
    return [e / s for e in exps]


def _family_of_slot(slot: int) -> int:
    # dos subfamilias legítimas y estables
    return 0 if slot in (0, 3) else 1


def _ensure_typed_buffers(body, n_slots: int = 4, n_regimes: int = 4) -> None:
    if not hasattr(body, "debt_by_regime"):
        setattr(body, "debt_by_regime", [0.0] * n_regimes)
    if not hasattr(body, "scar_by_regime"):
        setattr(body, "scar_by_regime", [0.0] * n_regimes)
    if not hasattr(body, "debt_by_slot"):
        setattr(body, "debt_by_slot", [0.0] * n_slots)
    if not hasattr(body, "scar_by_slot"):
        setattr(body, "scar_by_slot", [0.0] * n_slots)
    if not hasattr(body, "residual_by_family"):
        setattr(body, "residual_by_family", [0.0, 0.0])
    if not hasattr(body, "typed_reopen_count"):
        setattr(body, "typed_reopen_count", 0)


@dataclass
class CV4Log:
    typed_reopen_events: int = 0
    typed_debt_peak: float = 0.0
    typed_scar_peak: float = 0.0
    typed_residual_peak: float = 0.0
    final_regime_debt_entropy: float = 0.0
    final_slot_debt_entropy: float = 0.0
    family_divergence: float = 0.0
    regime_debt_trace_peak: float = 0.0
    slot_debt_trace_peak: float = 0.0


def _entropy(xs: list[float]) -> float:
    import math
    s = sum(xs)
    if s <= 1e-12:
        return 0.0
    ps = [x / s for x in xs if x > 1e-12]
    return -sum(p * math.log(p) for p in ps)


def _current_local_mismatch(state) -> tuple[int, int, int, float, float]:
    ctx = state.context_state
    world = state.world_state
    body = state.body_state

    active_slot = int(ctx.active_index)
    active_regime = int(getattr(world, "regime_index", 0))
    family = _family_of_slot(active_slot)

    alignment = float(ctx.last_alignment)
    pain = float(getattr(body, "pain", 0.0))
    stress = float(body.stress)
    energy = float(body.energy)

    local_strain = _clip(
        0.40 * max(0.0, 1.0 - alignment)
        + 0.25 * pain
        + 0.20 * stress
        + 0.15 * max(0.0, 0.18 - energy),
        0.0,
        1.0,
    )

    spill_strength = _clip(
        0.55 * max(0.0, 1.0 - alignment)
        + 0.25 * float(getattr(world, "regime_transition_signal", 0.0))
        + 0.20 * pain,
        0.0,
        1.0,
    )
    return active_slot, active_regime, family, local_strain, spill_strength


def apply_cv4_typed_scarring(state, log: CV4Log) -> None:
    body = state.body_state
    world = state.world_state
    ctx = state.context_state
    dev = state.development_state
    offline = state.offline_state
    tissue = state.tissue_state

    _ensure_typed_buffers(body)

    debt_by_regime = list(getattr(body, "debt_by_regime"))
    scar_by_regime = list(getattr(body, "scar_by_regime"))
    debt_by_slot = list(getattr(body, "debt_by_slot"))
    scar_by_slot = list(getattr(body, "scar_by_slot"))
    residual_by_family = list(getattr(body, "residual_by_family"))

    active_slot, active_regime, family, local_strain, spill_strength = _current_local_mismatch(state)

    policy_q = float(getattr(world, "policy_quality", 0.5))
    policy_b = float(getattr(world, "policy_badness", 0.5))
    survival_p = float(getattr(world, "survival_pressure", 0.5))
    reward = float(getattr(world, "reward", 0.0))

    # 1) deuda local primaria por régimen activo
    debt_push = _clip(
        0.60 * policy_b
        + 0.20 * survival_p
        + 0.20 * local_strain,
        0.0,
        1.0,
    )
    debt_relief = _clip(
        0.55 * policy_q + 0.15 * max(0.0, reward),
        0.0,
        1.0,
    )

    debt_by_regime[active_regime] = _clip(
        0.988 * debt_by_regime[active_regime]
        + 0.055 * debt_push
        - 0.030 * debt_relief,
        0.0,
        1.0,
    )

    debt_by_slot[active_slot] = _clip(
        0.988 * debt_by_slot[active_slot]
        + 0.052 * debt_push
        - 0.028 * debt_relief,
        0.0,
        1.0,
    )

    # 2) spillover parcial y tipado
    for r in range(4):
        if r == active_regime:
            continue
        debt_by_regime[r] = _clip(
            0.992 * debt_by_regime[r]
            + 0.010 * spill_strength * float(getattr(world, "regime_transition_matrix", [[0]*4 for _ in range(4)])[active_regime][r] if hasattr(world, "regime_transition_matrix") else 0.0),
            0.0,
            1.0,
        )

    for s in range(4):
        if s == active_slot:
            continue
        family_link = 1.0 if _family_of_slot(s) == family else 0.35
        compat = 0.0
        try:
            compat = float(ctx.compatibility_matrix[active_slot][s])
        except Exception:
            compat = 0.0
        debt_by_slot[s] = _clip(
            0.992 * debt_by_slot[s]
            + 0.014 * spill_strength * family_link * compat,
            0.0,
            1.0,
        )

    # 3) cicatriz lenta tipada
    scar_by_regime[active_regime] = _clip(
        0.996 * scar_by_regime[active_regime]
        + 0.018 * max(0.0, debt_by_regime[active_regime] - 0.42)
        + 0.010 * max(0.0, float(getattr(body, "pain", 0.0)) - 0.30),
        0.0,
        1.0,
    )

    scar_by_slot[active_slot] = _clip(
        0.996 * scar_by_slot[active_slot]
        + 0.016 * max(0.0, debt_by_slot[active_slot] - 0.40)
        + 0.010 * max(0.0, local_strain - 0.25),
        0.0,
        1.0,
    )

    residual_by_family[family] = _clip(
        0.994 * residual_by_family[family]
        + 0.020 * debt_by_slot[active_slot]
        + 0.012 * scar_by_slot[active_slot]
        - 0.008 * policy_q,
        0.0,
        1.0,
    )

    # 4) recuperación diferencial: nunca borra todo, y solo baja más lo activo/actual
    epoch = getattr(world, "curriculum_epoch", "")
    if epoch == "recovery_sleep":
        debt_by_regime[active_regime] = _clip(debt_by_regime[active_regime] * 0.965, 0.0, 1.0)
        debt_by_slot[active_slot] = _clip(debt_by_slot[active_slot] * 0.965, 0.0, 1.0)
        residual_by_family[family] = _clip(residual_by_family[family] * 0.988, 0.0, 1.0)

        # el resto apenas se drena
        for r in range(4):
            if r != active_regime:
                debt_by_regime[r] = _clip(debt_by_regime[r] * 0.993, 0.0, 1.0)
                scar_by_regime[r] = _clip(scar_by_regime[r] * 0.9985, 0.0, 1.0)
        for s in range(4):
            if s != active_slot:
                debt_by_slot[s] = _clip(debt_by_slot[s] * 0.993, 0.0, 1.0)
                scar_by_slot[s] = _clip(scar_by_slot[s] * 0.9985, 0.0, 1.0)

        scar_by_regime[active_regime] = _clip(scar_by_regime[active_regime] * 0.998, 0.0, 1.0)
        scar_by_slot[active_slot] = _clip(scar_by_slot[active_slot] * 0.998, 0.0, 1.0)

    # 5) señales agregadas pero no colapsadas
    typed_debt_total = sum(debt_by_regime) + 0.6 * sum(debt_by_slot)
    typed_scar_total = sum(scar_by_regime) + 0.6 * sum(scar_by_slot)
    typed_residual_total = sum(residual_by_family)

    body.energy = _clip(
        body.energy
        - 0.006 * typed_debt_total
        - 0.004 * typed_residual_total
        + 0.005 * policy_q,
        0.05,
        1.0,
    )
    body.stress = _clip(
        body.stress
        + 0.018 * max(0.0, debt_by_regime[active_regime] - 0.25)
        + 0.010 * residual_by_family[family]
        - 0.010 * policy_q,
        0.0,
        1.0,
    )
    body.pain = _clip(
        getattr(body, "pain", 0.0)
        + 0.014 * max(0.0, scar_by_slot[active_slot] - 0.18)
        + 0.010 * residual_by_family[family]
        - 0.008 * policy_q,
        0.0,
        1.0,
    )
    body.autonomic_load = _clip(
        body.autonomic_load
        + 0.016 * max(0.0, debt_by_regime[active_regime] - 0.22)
        + 0.012 * scar_by_regime[active_regime]
        - 0.008 * policy_q,
        0.0,
        1.0,
    )

    # 6) mundo y consolidación sienten tipado, no castigo global
    world.reward = _clip(
        world.reward
        + 0.030 * policy_q
        - 0.028 * debt_by_regime[active_regime]
        - 0.016 * scar_by_slot[active_slot],
        -1.0,
        1.0,
    )

    tissue.structural_signal = _clip(
        tissue.structural_signal
        + 0.004 * policy_q
        - 0.004 * max(0.0, debt_by_regime[active_regime] - 0.40),
        0.0,
        1.0,
    )

    if offline.asleep:
        offline.graph_replay_strength = _clip(
            offline.graph_replay_strength
            + 0.006 * policy_q
            - 0.004 * scar_by_slot[active_slot],
            0.0,
            1.0,
        )
        offline.graph_consolidation_gain = _clip(
            offline.graph_consolidation_gain
            + 0.006 * max(0.0, policy_q - policy_b)
            - 0.004 * residual_by_family[family],
            0.0,
            1.0,
        )
        offline.graph_pruning_signal = _clip(
            offline.graph_pruning_signal
            + 0.006 * max(0.0, debt_by_slot[active_slot] - 0.28)
            + 0.004 * scar_by_regime[active_regime],
            0.0,
            1.0,
        )

    # 7) reapertura local legítima
    local_illegitimacy = _clip(
        0.40 * debt_by_regime[active_regime]
        + 0.25 * debt_by_slot[active_slot]
        + 0.20 * residual_by_family[family]
        + 0.15 * max(0.0, 0.94 - float(ctx.last_alignment)),
        0.0,
        1.0,
    )

    if (not dev.critical_period_open) and local_illegitimacy > 0.56:
        dev.critical_period_open = True
        setattr(body, "typed_reopen_count", int(getattr(body, "typed_reopen_count", 0)) + 1)

    # guardar buffers
    setattr(body, "debt_by_regime", debt_by_regime)
    setattr(body, "scar_by_regime", scar_by_regime)
    setattr(body, "debt_by_slot", debt_by_slot)
    setattr(body, "scar_by_slot", scar_by_slot)
    setattr(body, "residual_by_family", residual_by_family)

    setattr(body, "typed_debt_total", float(typed_debt_total))
    setattr(body, "typed_scar_total", float(typed_scar_total))
    setattr(body, "typed_residual_total", float(typed_residual_total))
    setattr(world, "local_illegitimacy", float(local_illegitimacy))

    # logs
    log.typed_debt_peak = max(log.typed_debt_peak, max(debt_by_regime + debt_by_slot))
    log.typed_scar_peak = max(log.typed_scar_peak, max(scar_by_regime + scar_by_slot))
    log.typed_residual_peak = max(log.typed_residual_peak, max(residual_by_family))
    log.regime_debt_trace_peak = max(log.regime_debt_trace_peak, sum(debt_by_regime))
    log.slot_debt_trace_peak = max(log.slot_debt_trace_peak, sum(debt_by_slot))
    log.family_divergence = abs(residual_by_family[0] - residual_by_family[1])

    current_reopens = int(getattr(body, "typed_reopen_count", 0))
    log.typed_reopen_events = max(log.typed_reopen_events, current_reopens)

    log.final_regime_debt_entropy = _entropy(debt_by_regime)
    log.final_slot_debt_entropy = _entropy(debt_by_slot)


def collect_cv4_summary(state, cv3_summary: dict[str, Any], log: CV4Log) -> dict[str, Any]:
    body = state.body_state
    world = state.world_state

    out = dict(cv3_summary)
    out.update({
        "cv4_typed_reopen_events": int(log.typed_reopen_events),
        "cv4_typed_debt_peak": float(log.typed_debt_peak),
        "cv4_typed_scar_peak": float(log.typed_scar_peak),
        "cv4_typed_residual_peak": float(log.typed_residual_peak),
        "cv4_regime_debt_trace_peak": float(log.regime_debt_trace_peak),
        "cv4_slot_debt_trace_peak": float(log.slot_debt_trace_peak),
        "cv4_final_regime_debt_entropy": float(log.final_regime_debt_entropy),
        "cv4_final_slot_debt_entropy": float(log.final_slot_debt_entropy),
        "cv4_family_divergence": float(log.family_divergence),
        "cv4_final_debt_by_regime": list(getattr(body, "debt_by_regime", [])),
        "cv4_final_scar_by_regime": list(getattr(body, "scar_by_regime", [])),
        "cv4_final_debt_by_slot": list(getattr(body, "debt_by_slot", [])),
        "cv4_final_scar_by_slot": list(getattr(body, "scar_by_slot", [])),
        "cv4_final_residual_by_family": list(getattr(body, "residual_by_family", [])),
        "cv4_final_typed_debt_total": float(getattr(body, "typed_debt_total", 0.0)),
        "cv4_final_typed_scar_total": float(getattr(body, "typed_scar_total", 0.0)),
        "cv4_final_typed_residual_total": float(getattr(body, "typed_residual_total", 0.0)),
        "cv4_local_illegitimacy": float(getattr(world, "local_illegitimacy", 0.0)),
    })
    return out
