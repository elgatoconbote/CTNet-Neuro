from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass
class CV2Log:
    epoch_counts: dict[str, int] = field(default_factory=lambda: {
        "baseline_scaffold": 0,
        "scarcity": 0,
        "threat_avoidance": 0,
        "exploration_conflict": 0,
        "pain_recontext": 0,
        "multi_demand": 0,
        "recovery_sleep": 0,
    })
    epoch_trace: list[str] = field(default_factory=list)
    epoch_transition_count: int = 0
    critical_open_steps: int = 0
    critical_closed_steps: int = 0
    closure_reopen_events: int = 0
    last_critical_open: bool | None = None
    body_energy_min: float = 1.0
    body_stress_max: float = 0.0
    body_pain_max: float = 0.0
    body_autonomic_max: float = 0.0


def _epoch_for_step(step: int) -> str:
    if step < 64:
        return "baseline_scaffold"
    if step < 144:
        return "scarcity"
    if step < 224:
        return "threat_avoidance"
    if step < 304:
        return "exploration_conflict"
    if step < 384:
        return "pain_recontext"
    if step < 464:
        return "multi_demand"
    return "recovery_sleep"


def apply_cv2_curriculum(state, step: int, log: CV2Log) -> None:
    body = state.body_state
    world = state.world_state
    offline = state.offline_state
    tissue = state.tissue_state

    epoch = _epoch_for_step(step)

    if not log.epoch_trace or log.epoch_trace[-1] != epoch:
        if log.epoch_trace:
            log.epoch_transition_count += 1
        log.epoch_trace.append(epoch)

    log.epoch_counts[epoch] += 1

    # baseline scaffold
    if epoch == "baseline_scaffold":
        world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.35)
        world.threat_drive = min(getattr(world, "threat_drive", 0.0), 0.20)
        world.novelty_drive = max(getattr(world, "novelty_drive", 0.0), 0.40)
        world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.015)
        body.energy = _clip(body.energy + 0.004, 0.08, 1.0)
        body.stress = _clip(body.stress - 0.004, 0.0, 1.0)
        body.pain = _clip(body.pain * 0.96, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load * 0.98, 0.0, 1.0)

    # scarcity
    elif epoch == "scarcity":
        world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.92)
        world.threat_drive = max(getattr(world, "threat_drive", 0.0), 0.25)
        world.novelty_drive = min(getattr(world, "novelty_drive", 1.0), 0.30)
        world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.035)
        body.energy = _clip(body.energy - 0.016, 0.05, 1.0)
        body.interoception = _clip(body.interoception + 0.035, -1.0, 1.0)
        body.stress = _clip(body.stress + 0.010, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load + 0.006, 0.0, 1.0)
        world.reward = _clip(getattr(world, "reward", 0.0) - 0.035, -1.0, 1.0)

    # threat
    elif epoch == "threat_avoidance":
        world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.45)
        world.threat_drive = max(getattr(world, "threat_drive", 0.0), 0.95)
        world.novelty_drive = min(getattr(world, "novelty_drive", 1.0), 0.25)
        world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.085)
        body.energy = _clip(body.energy - 0.008, 0.05, 1.0)
        body.stress = _clip(body.stress + 0.040, 0.0, 1.0)
        body.pain = _clip(body.pain + 0.014, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load + 0.045, 0.0, 1.0)
        tissue.energy = _clip(tissue.energy - 0.004, 0.0, 1.0)

    # exploration conflict
    elif epoch == "exploration_conflict":
        world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.55)
        world.threat_drive = max(getattr(world, "threat_drive", 0.0), 0.42)
        world.novelty_drive = max(getattr(world, "novelty_drive", 0.0), 0.95)
        world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.070)
        body.energy = _clip(body.energy - 0.011, 0.05, 1.0)
        body.stress = _clip(body.stress + 0.018, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load + 0.015, 0.0, 1.0)
        world.reward = _clip(getattr(world, "reward", 0.0) + 0.015 - 0.020 * body.stress, -1.0, 1.0)

    # pain + recontext
    elif epoch == "pain_recontext":
        local = step - 304
        if local < 36:
            world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.58)
            world.threat_drive = max(getattr(world, "threat_drive", 0.0), 0.60)
            world.novelty_drive = max(getattr(world, "novelty_drive", 0.0), 0.35)
            world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.085)
            body.energy = _clip(body.energy - 0.012, 0.05, 1.0)
            body.stress = _clip(body.stress + 0.020, 0.0, 1.0)
            body.pain = _clip(body.pain + 0.028, 0.0, 1.0)
            body.autonomic_load = _clip(body.autonomic_load + 0.030, 0.0, 1.0)
        else:
            world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.46)
            world.threat_drive = min(max(getattr(world, "threat_drive", 0.0), 0.24), 0.40)
            world.novelty_drive = max(getattr(world, "novelty_drive", 0.0), 0.78)
            world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.095)
            body.energy = _clip(body.energy + 0.006, 0.05, 1.0)
            body.stress = _clip(body.stress - 0.012, 0.0, 1.0)
            body.pain = _clip(body.pain - 0.022, 0.0, 1.0)
            body.autonomic_load = _clip(body.autonomic_load - 0.010, 0.0, 1.0)

    # multi demand
    elif epoch == "multi_demand":
        world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.86)
        world.threat_drive = max(getattr(world, "threat_drive", 0.0), 0.66)
        world.novelty_drive = max(getattr(world, "novelty_drive", 0.0), 0.76)
        world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.090)
        body.energy = _clip(body.energy - 0.015, 0.05, 1.0)
        body.stress = _clip(body.stress + 0.022, 0.0, 1.0)
        body.pain = _clip(body.pain + 0.010, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load + 0.026, 0.0, 1.0)
        tissue.local_plasticity = _clip(tissue.local_plasticity + 0.004, 0.0, 1.0)

    # recovery + sleep-biased consolidation
    elif epoch == "recovery_sleep":
        world.homeostatic_drive = max(getattr(world, "homeostatic_drive", 0.0), 0.45)
        world.threat_drive = min(getattr(world, "threat_drive", 1.0), 0.15)
        world.novelty_drive = max(getattr(world, "novelty_drive", 0.0), 0.42)
        world.regime_transition_signal = max(getattr(world, "regime_transition_signal", 0.0), 0.025)
        body.energy = _clip(body.energy + 0.020, 0.05, 1.0)
        body.stress = _clip(body.stress - 0.020, 0.0, 1.0)
        body.pain = _clip(body.pain - 0.020, 0.0, 1.0)
        body.autonomic_load = _clip(body.autonomic_load - 0.016, 0.0, 1.0)

        if offline.asleep:
            offline.graph_replay_strength = _clip(offline.graph_replay_strength + 0.010, 0.0, 1.0)
            offline.graph_consolidation_gain = _clip(offline.graph_consolidation_gain + 0.010, 0.0, 1.0)

    setattr(world, "curriculum_epoch", epoch)


def update_cv2_log_after_supervision(state, log: CV2Log) -> None:
    body = state.body_state
    dev = state.development_state

    log.body_energy_min = min(log.body_energy_min, float(body.energy))
    log.body_stress_max = max(log.body_stress_max, float(body.stress))
    log.body_pain_max = max(log.body_pain_max, float(body.pain))
    log.body_autonomic_max = max(log.body_autonomic_max, float(body.autonomic_load))

    if dev.critical_period_open:
        log.critical_open_steps += 1
    else:
        log.critical_closed_steps += 1

    if log.last_critical_open is False and dev.critical_period_open is True:
        log.closure_reopen_events += 1

    log.last_critical_open = bool(dev.critical_period_open)


def collect_cv2_summary(state, cv1_summary: dict[str, Any], log: CV2Log) -> dict[str, Any]:
    world = state.world_state
    body = state.body_state

    out = dict(cv1_summary)
    out.update({
        "cv2_epoch_counts": dict(log.epoch_counts),
        "cv2_last_epoch": log.epoch_trace[-1] if log.epoch_trace else None,
        "cv2_epoch_transition_count": int(log.epoch_transition_count),
        "cv2_critical_open_steps": int(log.critical_open_steps),
        "cv2_critical_closed_steps": int(log.critical_closed_steps),
        "cv2_closure_reopen_events": int(log.closure_reopen_events),
        "cv2_body_energy_min": float(log.body_energy_min),
        "cv2_body_stress_max": float(log.body_stress_max),
        "cv2_body_pain_max": float(log.body_pain_max),
        "cv2_body_autonomic_max": float(log.body_autonomic_max),
        "cv2_final_body_energy": float(body.energy),
        "cv2_final_body_stress": float(body.stress),
        "cv2_final_body_pain": float(body.pain),
        "cv2_final_body_autonomic_load": float(body.autonomic_load),
        "cv2_final_curriculum_epoch": getattr(world, "curriculum_epoch", None),
    })
    return out
