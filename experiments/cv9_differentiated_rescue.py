from __future__ import annotations

from dataclasses import dataclass
import math


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_get(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


def _safe_set(mat, i: int, j: int, value: float) -> None:
    mat[i][j] = float(value)


def _row_sum(row: list[float]) -> float:
    return float(sum(row)) if row else 0.0


def _family_of_slot(slot: int) -> int:
    return 0 if slot in (0, 3) else 1


def _row_argmax(row: list[float]) -> int:
    return max(range(len(row)), key=lambda j: row[j])


def _row_argsecond(row: list[float]) -> int:
    idx = sorted(range(len(row)), key=lambda j: row[j], reverse=True)
    return idx[1] if len(idx) >= 2 else idx[0]


def _cos(a: list[float], b: list[float]) -> float:
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return sum(a[k] * b[k] for k in range(len(a))) / (da * db)


def _pairwise_mean(rows: list[list[float]]) -> float:
    vals = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            vals.append(_cos(rows[i], rows[j]))
    return sum(vals) / len(vals) if vals else 0.0


def _binding_specificity(row: list[float]) -> float:
    srt = sorted(row, reverse=True)
    if len(srt) < 2:
        return srt[0] if srt else 0.0
    return srt[0] - srt[1]


def _update_binding_summary(ctx) -> None:
    rows = [[float(v) for v in row] for row in getattr(ctx, "slot_regime_binding", [])]
    if not rows:
        return

    n = len(rows)
    m = len(rows[0])

    row_sums = [sum(r) for r in rows]
    col_sums = [sum(rows[i][j] for i in range(n)) for j in range(m)]
    specs = [_binding_specificity(r) for r in rows]

    pairwise = []
    for i in range(n):
        for j in range(i + 1, n):
            pairwise.append(_cos(rows[i], rows[j]))

    ctx.binding_row_min = min(row_sums)
    ctx.binding_row_max = max(row_sums)
    ctx.binding_row_mean = sum(row_sums) / len(row_sums)
    ctx.binding_col_min = min(col_sums)
    ctx.binding_col_max = max(col_sums)
    ctx.binding_col_mean = sum(col_sums) / len(col_sums)
    ctx.slot_regime_binding_mean = sum(col_sums) / (len(col_sums) * len(row_sums))
    ctx.dead_slot_fraction = sum(1 for x in row_sums if x < 0.72) / len(row_sums)
    ctx.binding_specificity_mean = sum(specs) / len(specs)
    ctx.regime_overlap_mean = (sum(pairwise) / len(pairwise)) if pairwise else 0.0
    ctx.hub_pressure_mean = 0.0
    ctx.lineage_diversity_mean = sum(abs(r[0] + r[3] - (r[1] + r[2])) for r in rows) / len(rows)


def _weighted_ipf(
    weights: list[list[float]],
    row_targets: list[float],
    col_targets: list[float],
    iters: int = 14,
) -> list[list[float]]:
    n = len(weights)
    m = len(weights[0]) if n else 0
    x = [[max(1e-9, float(weights[i][j])) for j in range(m)] for i in range(n)]

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


def _flow(ctx, i: int, j: int) -> float:
    tm = getattr(ctx, "transition_matrix", [])
    tg = getattr(ctx, "transition_grammar_matrix", [])
    rp = getattr(ctx, "return_path_matrix", [])
    return _clip(
        0.42 * _safe_get(tm, i, j)
        + 0.30 * _safe_get(tg, i, j)
        + 0.28 * _safe_get(rp, i, j),
        0.0,
        1.0,
    )


def _best_partner(ctx, rows: list[list[float]], weak: int) -> int:
    compat = getattr(ctx, "compatibility_matrix", [])
    incomp = getattr(ctx, "incompatibility_matrix", [])

    best_score = -1.0
    best_j = 0
    for j in range(len(rows)):
        if j == weak:
            continue
        sim = _cos(rows[weak], rows[j])
        reciprocity = min(_flow(ctx, weak, j), _flow(ctx, j, weak))
        comp = 0.5 * (_safe_get(compat, weak, j) + _safe_get(compat, j, weak))
        inc = 0.5 * (_safe_get(incomp, weak, j) + _safe_get(incomp, j, weak))

        # preferimos socio compatible pero no demasiado parecido
        score = (
            0.42 * comp
            + 0.28 * reciprocity
            + 0.18 * (1.0 - inc)
            + 0.12 * (1.0 - sim)
        )
        if score > best_score:
            best_score = score
            best_j = j
    return best_j


@dataclass
class CV9Log:
    differentiated_rescue_peak: float = 0.0
    escort_typed_gain_peak: float = 0.0
    overlap_guard_peak: float = 0.0
    weak_slot_last: int = -1
    partner_last: int = -1
    dominant_regime_last: int = -1


def apply_cv9_differentiated_rescue(state, log: CV9Log) -> None:
    ctx = state.context_state
    offline = state.offline_state

    rows = [[float(v) for v in row] for row in getattr(ctx, "slot_regime_binding", [])]
    if not rows:
        return

    n = len(rows)
    row_sums = [_row_sum(r) for r in rows]
    weak = min(range(n), key=lambda i: row_sums[i])
    weak_sum = row_sums[weak]
    partner = _best_partner(ctx, rows, weak)

    weak_dom = _row_argmax(rows[weak])
    weak_second = _row_argsecond(rows[weak])

    log.weak_slot_last = weak
    log.partner_last = partner
    log.dominant_regime_last = weak_dom

    current_overlap = _pairwise_mean(rows)
    replay = 1.0 + 0.16 * float(getattr(offline, "graph_replay_strength", 0.0))

    # objetivo de fila débil: rescate mínimo y conservativo
    rescue_need = max(0.0, 0.79 - weak_sum)

    # pesos: rescate por diferenciación, no por mezcla
    weights = [[max(1e-9, rows[i][j]) for j in range(4)] for i in range(n)]

    # slot débil: concentrar el rescate en sus regímenes propios, no difundirlo
    for j in range(4):
        if j == weak_dom:
            weights[weak][j] *= 1.95
        elif j == weak_second:
            weights[weak][j] *= 1.28
        else:
            weights[weak][j] *= 0.58

    # otros slots: desaturar columnas compartidas con la carta débil para evitar mezcla
    for i in range(n):
        if i == weak:
            continue
        sim = _cos(rows[weak], rows[i])
        donor_dom = _row_argmax(rows[i])
        for j in range(4):
            if j == weak_dom:
                weights[i][j] *= (0.88 - 0.22 * sim)
            if j == donor_dom:
                weights[i][j] *= 1.06
            if j not in (weak_dom, donor_dom):
                weights[i][j] *= 0.96

    # rescate conservativo por filas
    row_targets = row_sums[:]
    if rescue_need > 1e-12:
        row_targets[weak] += rescue_need
        donor_caps = [max(0.0, row_sums[i] - 0.80) if i != weak else 0.0 for i in range(n)]
        cap_sum = sum(donor_caps)
        if cap_sum > 1e-12:
            for i in range(n):
                if i == weak or donor_caps[i] <= 0.0:
                    continue
                row_targets[i] -= rescue_need * donor_caps[i] / cap_sum

    # columnas nominales cerradas
    col_targets = [0.78, 0.78, 0.78, 0.94]

    projected = _weighted_ipf(weights, row_targets=row_targets, col_targets=col_targets, iters=16)

    # guardia anti-mezcla: si el overlap proyectado sube demasiado, reforzar diferenciación local
    new_overlap = _pairwise_mean(projected)
    overlap_guard = 0.0
    if new_overlap > 0.365:
        excess = new_overlap - 0.365
        overlap_guard = excess

        for i in range(n):
            dom = _row_argmax(projected[i])
            sec = _row_argsecond(projected[i])
            for j in range(4):
                if j == dom:
                    projected[i][j] *= 1.0 + 0.08 * excess
                elif j == sec:
                    projected[i][j] *= 1.0 + 0.02 * excess
                else:
                    projected[i][j] *= 1.0 - 0.06 * excess

        projected = _weighted_ipf(projected, row_targets=row_targets, col_targets=col_targets, iters=10)

    # injerto escort tipado: solo si añade retorno sin homogeneizar
    typed_gain = 0.0
    sim_wp = _cos(projected[weak], projected[partner])
    if sim_wp < 0.78:
        tg = getattr(ctx, "transition_grammar_matrix", [])
        rp = getattr(ctx, "return_path_matrix", [])
        tm = getattr(ctx, "transition_matrix", [])

        fam_bonus = 1.0 if _family_of_slot(weak) != _family_of_slot(partner) else 0.92
        gain_base = (0.010 + 0.050 * max(0.0, 0.79 - weak_sum)) * replay * fam_bonus

        for a, b in ((weak, partner), (partner, weak)):
            tg_old = _safe_get(tg, a, b)
            rp_old = _safe_get(rp, a, b)
            tm_old = _safe_get(tm, a, b)

            tg_gain = gain_base * 0.80 * (1.0 - tg_old)
            rp_gain = gain_base * 0.95 * (1.0 - rp_old)
            tm_gain = gain_base * 0.28 * (1.0 - tm_old)

            _safe_set(tg, a, b, _clip(tg_old + tg_gain, 0.0, 1.0))
            _safe_set(rp, a, b, _clip(rp_old + rp_gain, 0.0, 1.0))
            _safe_set(tm, a, b, _clip(tm_old + tm_gain, 0.0, 1.0))

            typed_gain += tg_gain + rp_gain + tm_gain

    # escribir binding final
    for i in range(n):
        ctx.slot_regime_binding[i] = [float(v) for v in projected[i]]

    _update_binding_summary(ctx)

    log.differentiated_rescue_peak = max(log.differentiated_rescue_peak, rescue_need)
    log.escort_typed_gain_peak = max(log.escort_typed_gain_peak, typed_gain)
    log.overlap_guard_peak = max(log.overlap_guard_peak, overlap_guard)


def collect_cv9_summary(state, cv8_summary: dict, log: CV9Log) -> dict:
    out = dict(cv8_summary)
    out.update({
        "cv9_differentiated_rescue_peak": float(log.differentiated_rescue_peak),
        "cv9_escort_typed_gain_peak": float(log.escort_typed_gain_peak),
        "cv9_overlap_guard_peak": float(log.overlap_guard_peak),
        "cv9_weak_slot_last": int(log.weak_slot_last),
        "cv9_partner_last": int(log.partner_last),
        "cv9_dominant_regime_last": int(log.dominant_regime_last),
    })
    return out
