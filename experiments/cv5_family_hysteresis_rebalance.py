from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _family_of_slot(slot: int) -> int:
    return 0 if slot in (0, 3) else 1


def _ensure_cv5_buffers(body) -> None:
    if not hasattr(body, "family_hysteresis"):
        setattr(body, "family_hysteresis", [0.0, 0.0])
    if not hasattr(body, "cv5_reopen_count"):
        setattr(body, "cv5_reopen_count", 0)
    if not hasattr(body, "cv5_reopen_cooldown"):
        setattr(body, "cv5_reopen_cooldown", 0)


def _row_sum(row: list[float]) -> float:
    return float(sum(row)) if row else 0.0


def _argmax(xs: list[float]) -> int:
    best_i = 0
    best_v = xs[0]
    for i, v in enumerate(xs):
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def _ipf_project(mat: list[list[float]], row_targets: list[float], col_targets: list[float], iters: int = 8) -> list[list[float]]:
    n = len(mat)
    m = len(mat[0]) if n else 0
    x = [[max(1e-9, float(v)) for v in row] for row in mat]

    for _ in range(iters):
        # rows
        for i in range(n):
            s = sum(x[i]) or 1.0
            scale = row_targets[i] / s
            for j in range(m):
                x[i][j] *= scale

        # cols
        for j in range(m):
            s = sum(x[i][j] for i in range(n)) or 1.0
            scale = col_targets[j] / s
            for i in range(n):
                x[i][j] *= scale

    return x


def _apply_conservative_local_rebalance(ctx, body, world) -> float:
    rows = [[float(v) for v in row] for row in ctx.slot_regime_binding]
    if not rows:
        return 0.0

    n = len(rows)
    active_slot = int(ctx.active_index)
    active_regime = int(getattr(world, "regime_index", 0))
    family = _family_of_slot(active_slot)

    debt_by_regime = list(getattr(body, "debt_by_regime", [0.0] * 4))
    debt_by_slot = list(getattr(body, "debt_by_slot", [0.0] * n))
    scar_by_regime = list(getattr(body, "scar_by_regime", [0.0] * 4))
    scar_by_slot = list(getattr(body, "scar_by_slot", [0.0] * n))
    family_hys = list(getattr(body, "family_hysteresis", [0.0, 0.0]))

    row_targets = [_row_sum(r) for r in rows]
    col_targets = [0.78, 0.78, 0.78, 0.94]

    local_pressure = _clip(
        0.45 * debt_by_regime[active_regime]
        + 0.20 * debt_by_slot[active_slot]
        + 0.20 * scar_by_regime[active_regime]
        + 0.15 * family_hys[family],
        0.0,
        1.0,
    )

    # 1) sharpen rows away from cheap co-occupation of active regime
    for s in range(n):
        row = rows[s]
        dom = _argmax(row)
        fam = _family_of_slot(s)
        same_family = 1.0 if fam == family else 0.35

        if s != active_slot and dom != active_regime:
            take = min(
                max(0.0, row[active_regime] - 0.02),
                0.020 * local_pressure * same_family,
            )
            if take > 0.0:
                row[active_regime] -= take
                row[dom] += take

        # active slot: if local pressure is high, punish off-regime leakage
        if s == active_slot:
            leak = 0.0
            for r in range(4):
                if r == active_regime:
                    continue
                t = min(max(0.0, row[r] - 0.015), 0.010 * local_pressure)
                row[r] -= t
                leak += t
            row[active_regime] += leak

    # 2) small family-aware redistribution to keep family hysteresis meaningful
    for s in range(n):
        fam = _family_of_slot(s)
        row = rows[s]
        if fam == family:
            dom = _argmax(row)
            bonus = 0.008 * family_hys[fam]
            if dom != active_regime:
                take = min(max(0.0, row[active_regime] - 0.015), bonus)
                row[active_regime] -= take
                row[dom] += take

    # 3) conservative reprojection: preserve row mass and target columns
    rows = _ipf_project(rows, row_targets=row_targets, col_targets=col_targets, iters=10)

    # write back
    for i in range(n):
        ctx.slot_regime_binding[i] = [float(v) for v in rows[i]]

    return local_pressure


@dataclass
class CV5Log:
    family_hysteresis_peak: float = 0.0
    family_divergence_peak: float = 0.0
    reopen_events: int = 0
    local_illegitimacy_peak: float = 0.0
    rebalance_pressure_peak: float = 0.0


def apply_cv5_family_hysteresis(state, log: CV5Log) -> None:
    body = state.body_state
    world = state.world_state
    ctx = state.context_state
    dev = state.development_state

    _ensure_cv5_buffers(body)

    active_slot = int(ctx.active_index)
    active_regime = int(getattr(world, "regime_index", 0))
    family = _family_of_slot(active_slot)

    debt_by_regime = list(getattr(body, "debt_by_regime", [0.0] * 4))
    scar_by_regime = list(getattr(body, "scar_by_regime", [0.0] * 4))
    debt_by_slot = list(getattr(body, "debt_by_slot", [0.0] * 4))
    scar_by_slot = list(getattr(body, "scar_by_slot", [0.0] * 4))
    fam_h = list(getattr(body, "family_hysteresis", [0.0, 0.0]))

    align = float(ctx.last_alignment)
    pain = float(getattr(body, "pain", 0.0))
    stress = float(body.stress)
    energy = float(body.energy)
    policy_q = float(getattr(world, "policy_quality", 0.5))
    policy_b = float(getattr(world, "policy_badness", 0.5))

    family_slots = [i for i in range(4) if _family_of_slot(i) == family]
    other_family = 1 - family

    family_scar_mean = sum(scar_by_slot[i] for i in family_slots) / max(1, len(family_slots))
    family_debt_mean = sum(debt_by_slot[i] for i in family_slots) / max(1, len(family_slots))

    fam_h[family] = _clip(
        0.998 * fam_h[family]
        + 0.015 * family_scar_mean
        + 0.012 * family_debt_mean
        + 0.010 * max(0.0, 0.96 - align)
        + 0.008 * max(0.0, pain - 0.15)
        - 0.004 * policy_q,
        0.0,
        1.0,
    )

    # spillover parcial real, no cero absoluto
    fam_h[other_family] = _clip(
        0.999 * fam_h[other_family]
        + 0.0035 * family_scar_mean
        + 0.0025 * family_debt_mean
        - 0.0020 * policy_q,
        0.0,
        1.0,
    )

    # recovery no borra familias; solo las afloja
    if getattr(world, "curriculum_epoch", "") == "recovery_sleep":
        fam_h[family] = _clip(fam_h[family] * 0.995, 0.0, 1.0)
        fam_h[other_family] = _clip(fam_h[other_family] * 0.9975, 0.0, 1.0)

    setattr(body, "family_hysteresis", fam_h)

    local_illegitimacy = _clip(
        0.30 * debt_by_regime[active_regime]
        + 0.20 * debt_by_slot[active_slot]
        + 0.18 * scar_by_regime[active_regime]
        + 0.14 * scar_by_slot[active_slot]
        + 0.12 * fam_h[family]
        + 0.06 * max(0.0, 0.965 - align),
        0.0,
        1.0,
    )
    setattr(world, "cv5_local_illegitimacy", float(local_illegitimacy))

    cooldown = int(getattr(body, "cv5_reopen_cooldown", 0))
    if cooldown > 0:
        cooldown -= 1

    # reapertura diferencial: menos umbral que CV4, pero local y con cooldown
    if (not dev.critical_period_open) and cooldown == 0 and local_illegitimacy > 0.305:
        dev.critical_period_open = True
        log.reopen_events += 1
        cooldown = 24

    setattr(body, "cv5_reopen_cooldown", cooldown)
    setattr(body, "cv5_reopen_count", int(getattr(body, "cv5_reopen_count", 0)) + (1 if cooldown == 24 else 0))

    rebalance_pressure = _apply_conservative_local_rebalance(ctx, body, world)

    # logs
    log.family_hysteresis_peak = max(log.family_hysteresis_peak, max(fam_h))
    log.family_divergence_peak = max(log.family_divergence_peak, abs(fam_h[0] - fam_h[1]))
    log.local_illegitimacy_peak = max(log.local_illegitimacy_peak, local_illegitimacy)
    log.rebalance_pressure_peak = max(log.rebalance_pressure_peak, rebalance_pressure)


def collect_cv5_summary(state, cv4_summary: dict, log: CV5Log) -> dict:
    body = state.body_state
    world = state.world_state

    out = dict(cv4_summary)
    out.update({
        "cv5_family_hysteresis_peak": float(log.family_hysteresis_peak),
        "cv5_family_divergence_peak": float(log.family_divergence_peak),
        "cv5_reopen_events": int(log.reopen_events),
        "cv5_local_illegitimacy_peak": float(log.local_illegitimacy_peak),
        "cv5_rebalance_pressure_peak": float(log.rebalance_pressure_peak),
        "cv5_final_family_hysteresis": list(getattr(body, "family_hysteresis", [])),
        "cv5_final_local_illegitimacy": float(getattr(world, "cv5_local_illegitimacy", 0.0)),
        "cv5_final_reopen_cooldown": int(getattr(body, "cv5_reopen_cooldown", 0)),
        "cv5_final_reopen_count": int(getattr(body, "cv5_reopen_count", 0)),
    })
    return out
