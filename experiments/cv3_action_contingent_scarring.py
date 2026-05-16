from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _sign(x: float) -> float:
    if x > 0.0:
        return 1.0
    if x < 0.0:
        return -1.0
    return 0.0


@dataclass
class CV3Log:
    policy_good_steps: int = 0
    policy_bad_steps: int = 0
    debt_peak: float = 0.0
    scar_peak: float = 0.0
    residual_peak: float = 0.0
    reward_peak: float = 0.0
    reopen_from_debt_events: int = 0
    last_critical_open: bool | None = None
    epoch_action_stats: dict[str, dict[str, float]] = field(default_factory=dict)


def _epoch_signature(epoch: str) -> tuple[float, float, float]:
    # target_motor_sign, exploration_target, threat_weight
    if epoch == "baseline_scaffold":
        return (+1.0, 0.45, 0.20)
    if epoch == "scarcity":
        return (+1.0, 0.15, 0.35)
    if epoch == "threat_avoidance":
        return (-1.0, 0.10, 1.00)
    if epoch == "exploration_conflict":
        return (+1.0, 1.00, 0.45)
    if epoch == "pain_recontext":
        return (-1.0, 0.70, 0.75)
    if epoch == "multi_demand":
        return (0.0, 0.85, 0.85)
    if epoch == "recovery_sleep":
        return (+1.0, 0.30, 0.10)
    return (0.0, 0.40, 0.40)


def _policy_quality(state) -> tuple[float, float, float]:
    body = state.body_state
    world = state.world_state
    ctx = state.context_state
    meso = state.meso_state
    epoch = getattr(world, "curriculum_epoch", "baseline_scaffold")

    target_sign, exploration_target, threat_weight = _epoch_signature(epoch)

    motor = float(body.motor_command)
    affordance = float(getattr(world, "affordance", 0.0))
    reward = float(getattr(world, "reward", 0.0))
    align = float(ctx.last_alignment)
    novelty = float(getattr(world, "novelty_drive", 0.0))
    threat = float(getattr(world, "threat_drive", 0.0))
    homeo = float(getattr(world, "homeostatic_drive", 0.0))
    pain = float(getattr(body, "pain", 0.0))
    stress = float(body.stress)
    energy = float(body.energy)

    if target_sign == 0.0:
        direction_fit = 1.0 - abs(motor)
    else:
        direction_fit = 0.5 * (1.0 + target_sign * _sign(motor))

    exploration_fit = 1.0 - abs(abs(motor) - exploration_target)
    exploration_fit = _clip(exploration_fit, 0.0, 1.0)

    contextual_fit = _clip(
        0.45 * align
        + 0.20 * max(0.0, reward)
        + 0.15 * (1.0 - pain)
        + 0.10 * (1.0 - stress)
        + 0.10 * affordance,
        0.0,
        1.0,
    )

    survival_pressure = _clip(
        0.45 * homeo + 0.35 * threat_weight * threat + 0.20 * pain,
        0.0,
        1.0,
    )

    quality = _clip(
        0.35 * direction_fit
        + 0.20 * exploration_fit
        + 0.25 * contextual_fit
        + 0.10 * max(0.0, reward)
        + 0.10 * (1.0 - stress),
        0.0,
        1.0,
    )

    badness = _clip(
        survival_pressure * (1.0 - quality)
        + 0.20 * max(0.0, 0.18 - energy)
        + 0.10 * max(0.0, novelty - abs(motor)),
        0.0,
        1.0,
    )

    return quality, badness, survival_pressure


def apply_cv3_action_contingency(state, log: CV3Log) -> None:
    body = state.body_state
    world = state.world_state
    dev = state.development_state
    offline = state.offline_state
    tissue = state.tissue_state
    ctx = state.context_state

    epoch = getattr(world, "curriculum_epoch", "baseline_scaffold")
    quality, badness, survival_pressure = _policy_quality(state)

    debt = float(getattr(body, "debt_load", 0.0))
    scar = float(getattr(body, "scar_load", 0.0))
    residual = float(getattr(body, "residual_strain", 0.0))

    if quality >= 0.58:
        log.policy_good_steps += 1
    else:
        log.policy_bad_steps += 1

    # deuda inmediata dependiente de mala política
    debt = _clip(
        0.985 * debt
        + 0.060 * badness
        + 0.020 * survival_pressure * max(0.0, 0.22 - body.energy)
        - 0.030 * quality,
        0.0,
        1.0,
    )

    # cicatriz lenta: no se borra rápido y depende de episodios malos repetidos
    scar = _clip(
        0.996 * scar
        + 0.018 * max(0.0, debt - 0.45)
        + 0.010 * max(0.0, getattr(body, "pain", 0.0) - 0.35),
        0.0,
        1.0,
    )

    # strain residual: remanente corporal subagudo
    residual = _clip(
        0.992 * residual
        + 0.025 * debt
        + 0.015 * scar
        - 0.006 * quality,
        0.0,
        1.0,
    )

    setattr(body, "debt_load", debt)
    setattr(body, "scar_load", scar)
    setattr(body, "residual_strain", residual)

    # coste corporal contingente
    body.energy = _clip(
        body.energy
        - 0.012 * debt
        - 0.006 * residual
        + 0.008 * quality,
        0.05,
        1.0,
    )

    body.stress = _clip(
        body.stress
        + 0.035 * badness
        + 0.018 * residual
        - 0.018 * quality,
        0.0,
        1.0,
    )

    body.pain = _clip(
        getattr(body, "pain", 0.0)
        + 0.024 * max(0.0, debt - 0.25)
        + 0.012 * residual
        - 0.010 * quality,
        0.0,
        1.0,
    )

    body.autonomic_load = _clip(
        body.autonomic_load
        + 0.030 * badness
        + 0.020 * scar
        - 0.012 * quality,
        0.0,
        1.0,
    )

    # el mundo responde a la calidad de política
    world.reward = _clip(
        float(getattr(world, "reward", 0.0))
        + 0.045 * quality
        - 0.060 * badness
        - 0.020 * scar,
        -1.0,
        1.0,
    )

    setattr(world, "policy_quality", quality)
    setattr(world, "policy_badness", badness)
    setattr(world, "survival_pressure", survival_pressure)

    # penalización/bonificación estructural
    tissue.structural_signal = _clip(
        tissue.structural_signal
        + 0.006 * quality
        - 0.008 * max(0.0, debt - 0.40),
        0.0,
        1.0,
    )

    tissue.local_plasticity = _clip(
        tissue.local_plasticity
        + 0.006 * max(0.0, badness - 0.45)
        + 0.004 * max(0.0, quality - 0.60),
        0.0,
        1.0,
    )

    # reapertura si el cierre es ilegítimo bajo deuda persistente
    if (
        not dev.critical_period_open
        and debt > 0.58
        and residual > 0.30
        and float(ctx.last_alignment) < 0.93
    ):
        dev.critical_period_open = True

    # consolidación offline según supervivencia legítima, no recuperación trivial
    if offline.asleep:
        offline.graph_replay_strength = _clip(
            offline.graph_replay_strength
            + 0.008 * quality
            - 0.006 * scar,
            0.0,
            1.0,
        )
        offline.graph_consolidation_gain = _clip(
            offline.graph_consolidation_gain
            + 0.010 * max(0.0, quality - badness)
            - 0.006 * residual,
            0.0,
            1.0,
        )
        offline.graph_pruning_signal = _clip(
            offline.graph_pruning_signal
            + 0.008 * max(0.0, debt - 0.35)
            + 0.004 * scar,
            0.0,
            1.0,
        )

    # recuperación nunca borra del todo la historia
    if epoch == "recovery_sleep":
        setattr(body, "debt_load", _clip(getattr(body, "debt_load", 0.0) * 0.96, 0.0, 1.0))
        setattr(body, "residual_strain", _clip(getattr(body, "residual_strain", 0.0) * 0.985, 0.0, 1.0))
        setattr(body, "scar_load", _clip(getattr(body, "scar_load", 0.0) * 0.997, 0.0, 1.0))

    # log
    log.debt_peak = max(log.debt_peak, getattr(body, "debt_load", 0.0))
    log.scar_peak = max(log.scar_peak, getattr(body, "scar_load", 0.0))
    log.residual_peak = max(log.residual_peak, getattr(body, "residual_strain", 0.0))
    log.reward_peak = max(log.reward_peak, float(getattr(world, "reward", 0.0)))

    if log.last_critical_open is False and dev.critical_period_open is True:
        log.reopen_from_debt_events += 1
    log.last_critical_open = bool(dev.critical_period_open)

    stats = log.epoch_action_stats.setdefault(epoch, {
        "quality_sum": 0.0,
        "badness_sum": 0.0,
        "count": 0.0,
    })
    stats["quality_sum"] += quality
    stats["badness_sum"] += badness
    stats["count"] += 1.0


def collect_cv3_summary(state, cv2_summary: dict[str, Any], log: CV3Log) -> dict[str, Any]:
    body = state.body_state
    world = state.world_state

    epoch_means = {}
    for k, v in log.epoch_action_stats.items():
        c = max(1.0, v["count"])
        epoch_means[k] = {
            "quality_mean": v["quality_sum"] / c,
            "badness_mean": v["badness_sum"] / c,
        }

    out = dict(cv2_summary)
    out.update({
        "cv3_policy_good_steps": int(log.policy_good_steps),
        "cv3_policy_bad_steps": int(log.policy_bad_steps),
        "cv3_debt_peak": float(log.debt_peak),
        "cv3_scar_peak": float(log.scar_peak),
        "cv3_residual_peak": float(log.residual_peak),
        "cv3_reward_peak": float(log.reward_peak),
        "cv3_reopen_from_debt_events": int(log.reopen_from_debt_events),
        "cv3_final_debt_load": float(getattr(body, "debt_load", 0.0)),
        "cv3_final_scar_load": float(getattr(body, "scar_load", 0.0)),
        "cv3_final_residual_strain": float(getattr(body, "residual_strain", 0.0)),
        "cv3_final_policy_quality": float(getattr(world, "policy_quality", 0.0)),
        "cv3_final_policy_badness": float(getattr(world, "policy_badness", 0.0)),
        "cv3_final_survival_pressure": float(getattr(world, "survival_pressure", 0.0)),
        "cv3_epoch_policy_means": epoch_means,
    })
    return out
