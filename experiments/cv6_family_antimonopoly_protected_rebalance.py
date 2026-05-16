from __future__ import annotations

from dataclasses import dataclass
import math


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _family_of_slot(slot: int) -> int:
    return 0 if slot in (0, 3) else 1


def _ensure_cv6_buffers(body) -> None:
    if not hasattr(body, "family_hysteresis"):
        setattr(body, "family_hysteresis", [0.0, 0.0])
    if not hasattr(body, "cv6_reopen_count"):
        setattr(body, "cv6_reopen_count", 0)
    if not hasattr(body, "cv6_reopen_cooldown"):
        setattr(body, "cv6_reopen_cooldown", 0)


def _safe_get_mat(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


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


def _argmax_excluding(xs: list[float], banned: int) -> int:
    best_i = 0
    best_v = -1e18
    for i, v in enumerate(xs):
        if i == banned:
            continue
        if v > best_v:
            best_i = i
            best_v = v
    return best_i


def _ipf_project(mat: list[list[float]], row_targets: list[float], col_targets: list[float], iters: int = 10) -> list[list[float]]:
    n = len(mat)
    m = len(mat[0]) if n else 0
    x = [[max(1e-9, float(v)) for v in row] for row in mat]

    for _ in range(iters):
        for i in range(n):
            s = sum(x[i]) or 1.0
            scale = row_targets[i] / s
            for j in range(m):
                x[i][j] *= scale

        for j in range(m):
            s = sum(x[i][j] for i in range(n)) or 1.0
            scale = col_targets[j] / s
            for i in range(n):
                x[i][j] *= scale

    return x


def _slot_flow_protection(ctx, slot: int) -> float:
    n = len(getattr(ctx, "slot_regime_binding", []))
    if n <= 1:
        return 0.0

    tg = getattr(ctx, "transition_grammar_matrix", [])
    rp = getattr(ctx, "return_path_matrix", [])

    vals = []
    for j in range(n):
        if j == slot:
            continue
        vals.append(0.55 * _safe_get_mat(tg, slot, j) + 0.45 * _safe_get_mat(rp, slot, j))
        vals.append(0.35 * _safe_get_mat(tg, j, slot) + 0.65 * _safe_get_mat(rp, j, slot))

    if not vals:
        return 0.0
    return _clip(sum(vals) / len(vals), 0.0, 1.0)


def _family_viability(body, ctx, fam: int) -> float:
    slots = [i for i in range(4) if _family_of_slot(i) == fam]
    debt_by_slot = list(getattr(body, "debt_by_slot", [0.0] * 4))
    scar_by_slot = list(getattr(body, "scar_by_slot", [0.0] * 4))
    fam_h = list(getattr(body, "family_hysteresis", [0.0, 0.0]))

    debt_mean = sum(debt_by_slot[i] for i in slots) / max(1, len(slots))
    scar_mean = sum(scar_by_slot[i] for i in slots) / max(1, len(slots))

    compat = getattr(ctx, "compatibility_matrix", [])
    pair = 0.0
    if len(slots) == 2:
        a, b = slots
        pair = 0.5 * (_safe_get_mat(compat, a, b) + _safe_get_mat(compat, b, a))

    return _clip(
        0.45 * fam_h[fam]
        + 0.25 * scar_mean
        + 0.20 * debt_mean
        + 0.10 * pair,
        0.0,
        1.0,
    )


def _update_binding_summary(ctx) -> None:
    rows = [[float(v) for v in row] for row in getattr(ctx, "slot_regime_binding", [])]
    if not rows:
        return

    n = len(rows)
    m = len(rows[0])

    row_sums = [sum(r) for r in rows]
    col_sums = [sum(rows[i][j] for i in range(n)) for j in range(m)]

    specs = []
    for row in rows:
        srt = sorted(row, reverse=True)
        specs.append((srt[0] - srt[1]) if len(srt) >= 2 else srt[0])

    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            ai = rows[i]
            bj = rows[j]
            da = math.sqrt(sum(x * x for x in ai)) or 1.0
            db = math.sqrt(sum(x * x for x in bj)) or 1.0
            dot = sum(ai[k] * bj[k] for k in range(m))
            pairwise.append(dot / (da * db))

    setattr(ctx, "binding_row_min", min(row_sums))
    setattr(ctx, "binding_row_max", max(row_sums))
    setattr(ctx, "binding_row_mean", sum(row_sums) / len(row_sums))
    setattr(ctx, "binding_col_min", min(col_sums))
    setattr(ctx, "binding_col_max", max(col_sums))
    setattr(ctx, "binding_col_mean", sum(col_sums) / len(col_sums))
    setattr(ctx, "slot_regime_binding_mean", sum(col_sums) / (len(col_sums) * len(row_sums)))
    setattr(ctx, "dead_slot_fraction", sum(1 for x in row_sums if x < 0.72) / len(row_sums))
    setattr(ctx, "binding_specificity_mean", sum(specs) / len(specs))
    setattr(ctx, "regime_overlap_mean", (sum(pairwise) / len(pairwise)) if pairwise else 0.0)
    setattr(ctx, "hub_pressure_mean", 0.0)


def _protected_conservative_rebalance(ctx, body, world, family_viabilities: list[float]) -> float:
    rows = [[float(v) for v in row] for row in ctx.slot_regime_binding]
    if not rows:
        return 0.0

    n = len(rows)
    active_slot = int(ctx.active_index)
    active_regime = int(getattr(world, "regime_index", 0))
    active_family = _family_of_slot(active_slot)

    debt_by_regime = list(getattr(body, "debt_by_regime", [0.0] * 4))
    debt_by_slot = list(getattr(body, "debt_by_slot", [0.0] * n))
    scar_by_regime = list(getattr(body, "scar_by_regime", [0.0] * 4))
    scar_by_slot = list(getattr(body, "scar_by_slot", [0.0] * n))
    fam_h = list(getattr(body, "family_hysteresis", [0.0, 0.0]))

    row_targets = [_row_sum(r) for r in rows]
    col_targets = [0.78, 0.78, 0.78, 0.94]

    local_pressure = _clip(
        0.30 * debt_by_regime[active_regime]
        + 0.18 * debt_by_slot[active_slot]
        + 0.18 * scar_by_regime[active_regime]
        + 0.12 * scar_by_slot[active_slot]
        + 0.12 * fam_h[active_family]
        + 0.10 * max(0.0, family_viabilities[1 - active_family] - family_viabilities[active_family] + 0.10),
        0.0,
        1.0,
    )

    for s in range(n):
        row = rows[s]
        protection = _slot_flow_protection(ctx, s)
        fam = _family_of_slot(s)
        family_link = 1.0 if fam == active_family else 0.45

        if s != active_slot:
            dom_non_active = _argmax_excluding(row, active_regime)
            floor = 0.020 + 0.020 * protection + 0.010 * family_link
            excess = max(0.0, row[active_regime] - floor)

            # si hay mucha protección grammar/return, casi no tocar
            move = min(
                excess,
                0.024 * local_pressure * family_link * (1.0 - 0.80 * protection)
            )
            if move > 0.0:
                row[active_regime] -= move
                row[dom_non_active] += move

            # sostén mínimo para familia débil si no está monopolizando
            fam_gap = fam_h[active_family] - fam_h[1 - active_family]
            if fam == (1 - active_family) and fam_gap > 0.12:
                dom_non_active = _argmax_excluding(row, active_regime)
                seed = min(
                    max(0.0, row[active_regime] - (0.018 + 0.022 * protection)),
                    0.010 * fam_gap * (1.0 - 0.75 * protection)
                )
                if seed > 0.0:
                    row[active_regime] -= seed
                    row[dom_non_active] += seed

        else:
            # fila activa: reducir fuga barata, pero proteger rutas vivas
            recovered = 0.0
            for r in range(4):
                if r == active_regime:
                    continue
                keep = 0.012 + 0.040 * protection
                excess = max(0.0, row[r] - keep)
                move = min(excess, 0.018 * local_pressure * (1.0 - 0.65 * protection))
                row[r] -= move
                recovered += move
            row[active_regime] += recovered

    rows = _ipf_project(rows, row_targets=row_targets, col_targets=col_targets, iters=10)

    for i in range(n):
        ctx.slot_regime_binding[i] = [float(v) for v in rows[i]]

    return local_pressure


@dataclass
class CV6Log:
    family_hysteresis_peak: float = 0.0
    family_divergence_peak: float = 0.0
    reopen_events: int = 0
    local_illegitimacy_peak: float = 0.0
    rebalance_pressure_peak: float = 0.0
    monopoly_penalty_peak: float = 0.0
    rival_viability_peak: float = 0.0


def apply_cv6_family_antimonopoly(state, log: CV6Log) -> None:
    body = state.body_state
    world = state.world_state
    ctx = state.context_state
    dev = state.development_state

    _ensure_cv6_buffers(body)

    active_slot = int(ctx.active_index)
    active_regime = int(getattr(world, "regime_index", 0))
    active_family = _family_of_slot(active_slot)
    rival_family = 1 - active_family

    fam_h = list(getattr(body, "family_hysteresis", [0.0, 0.0]))
    debt_by_regime = list(getattr(body, "debt_by_regime", [0.0] * 4))
    debt_by_slot = list(getattr(body, "debt_by_slot", [0.0] * 4))
    scar_by_regime = list(getattr(body, "scar_by_regime", [0.0] * 4))
    scar_by_slot = list(getattr(body, "scar_by_slot", [0.0] * 4))

    align = float(ctx.last_alignment)
    policy_q = float(getattr(world, "policy_quality", 0.5))
    pain = float(getattr(body, "pain", 0.0))
    stress = float(body.stress)

    family_viabilities = [
        _family_viability(body, ctx, 0),
        _family_viability(body, ctx, 1),
    ]

    # suelo bilateral: la familia rival no puede evaporarse
    if fam_h[rival_family] < 0.035:
        fam_h[rival_family] = max(
            fam_h[rival_family],
            0.018 + 0.060 * family_viabilities[rival_family]
        )

    # penalización de monopolio familiar
    monopoly_gap = max(0.0, fam_h[active_family] - fam_h[rival_family])
    monopoly_penalty = _clip(
        monopoly_gap - 0.18,
        0.0,
        1.0,
    )
    if monopoly_penalty > 0.0:
        transfer = min(
            0.020 * monopoly_penalty,
            max(0.0, fam_h[active_family] - 0.08)
        )
        fam_h[active_family] = _clip(fam_h[active_family] - transfer, 0.0, 1.0)
        fam_h[rival_family] = _clip(fam_h[rival_family] + 0.85 * transfer, 0.0, 1.0)

    # mantenimiento suave de ambas familias
    fam_h[active_family] = _clip(
        0.999 * fam_h[active_family]
        + 0.004 * max(0.0, scar_by_slot[active_slot])
        + 0.003 * max(0.0, debt_by_slot[active_slot])
        - 0.002 * policy_q,
        0.0,
        1.0,
    )
    fam_h[rival_family] = _clip(
        0.9995 * fam_h[rival_family]
        + 0.002 * family_viabilities[rival_family]
        - 0.001 * policy_q,
        0.0,
        1.0,
    )

    setattr(body, "family_hysteresis", fam_h)

    rival_viability = family_viabilities[rival_family]
    local_illegitimacy = _clip(
        0.26 * debt_by_regime[active_regime]
        + 0.18 * debt_by_slot[active_slot]
        + 0.16 * scar_by_regime[active_regime]
        + 0.12 * scar_by_slot[active_slot]
        + 0.10 * fam_h[active_family]
        + 0.08 * max(0.0, 0.975 - align)
        + 0.06 * pain
        + 0.04 * stress,
        0.0,
        1.0,
    )
    setattr(world, "cv6_local_illegitimacy", float(local_illegitimacy))
    setattr(world, "cv6_rival_family_viability", float(rival_viability))

    cooldown = int(getattr(body, "cv6_reopen_cooldown", 0))
    if cooldown > 0:
        cooldown -= 1

    if (not dev.critical_period_open) and cooldown == 0 and local_illegitimacy > 0.29 and rival_viability > 0.07:
        dev.critical_period_open = True
        cooldown = 24
        log.reopen_events += 1

    setattr(body, "cv6_reopen_cooldown", cooldown)
    setattr(body, "cv6_reopen_count", int(getattr(body, "cv6_reopen_count", 0)) + (1 if cooldown == 24 else 0))

    rebalance_pressure = _protected_conservative_rebalance(ctx, body, world, family_viabilities)
    _update_binding_summary(ctx)

    log.family_hysteresis_peak = max(log.family_hysteresis_peak, max(fam_h))
    log.family_divergence_peak = max(log.family_divergence_peak, abs(fam_h[0] - fam_h[1]))
    log.local_illegitimacy_peak = max(log.local_illegitimacy_peak, local_illegitimacy)
    log.rebalance_pressure_peak = max(log.rebalance_pressure_peak, rebalance_pressure)
    log.monopoly_penalty_peak = max(log.monopoly_penalty_peak, monopoly_penalty)
    log.rival_viability_peak = max(log.rival_viability_peak, rival_viability)


def collect_cv6_summary(state, cv5_summary: dict, log: CV6Log) -> dict:
    body = state.body_state
    world = state.world_state

    out = dict(cv5_summary)
    out.update({
        "cv6_family_hysteresis_peak": float(log.family_hysteresis_peak),
        "cv6_family_divergence_peak": float(log.family_divergence_peak),
        "cv6_reopen_events": int(log.reopen_events),
        "cv6_local_illegitimacy_peak": float(log.local_illegitimacy_peak),
        "cv6_rebalance_pressure_peak": float(log.rebalance_pressure_peak),
        "cv6_monopoly_penalty_peak": float(log.monopoly_penalty_peak),
        "cv6_rival_viability_peak": float(log.rival_viability_peak),
        "cv6_final_family_hysteresis": list(getattr(body, "family_hysteresis", [])),
        "cv6_final_local_illegitimacy": float(getattr(world, "cv6_local_illegitimacy", 0.0)),
        "cv6_final_rival_family_viability": float(getattr(world, "cv6_rival_family_viability", 0.0)),
        "cv6_final_reopen_cooldown": int(getattr(body, "cv6_reopen_cooldown", 0)),
        "cv6_final_reopen_count": int(getattr(body, "cv6_reopen_count", 0)),
    })
    return out
