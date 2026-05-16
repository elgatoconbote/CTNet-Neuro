from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _family_of_slot(slot: int) -> int:
    return 0 if slot in (0, 3) else 1


def _safe_get(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


def _safe_set(mat, i: int, j: int, value: float) -> None:
    mat[i][j] = float(value)


def _row_sum(row: list[float]) -> float:
    return float(sum(row)) if row else 0.0


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
            da = (sum(x * x for x in ai) ** 0.5) or 1.0
            db = (sum(x * x for x in bj) ** 0.5) or 1.0
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


def _flow(ctx, i: int, j: int) -> float:
    tm = getattr(ctx, "transition_matrix", [])
    tg = getattr(ctx, "transition_grammar_matrix", [])
    rp = getattr(ctx, "return_path_matrix", [])
    return _clip(
        0.45 * _safe_get(tm, i, j)
        + 0.30 * _safe_get(tg, i, j)
        + 0.25 * _safe_get(rp, i, j),
        0.0,
        1.0,
    )


def _pair_score(ctx, body, i: int, j: int) -> tuple[float, float]:
    compat = getattr(ctx, "compatibility_matrix", [])
    incomp = getattr(ctx, "incompatibility_matrix", [])

    fam_h = list(getattr(body, "family_hysteresis", [0.0, 0.0]))

    compat_ij = 0.5 * (_safe_get(compat, i, j) + _safe_get(compat, j, i))
    incomp_ij = 0.5 * (_safe_get(incomp, i, j) + _safe_get(incomp, j, i))
    recip = min(_flow(ctx, i, j), _flow(ctx, j, i))

    fi = _family_of_slot(i)
    fj = _family_of_slot(j)
    fam_balance = 1.0 - abs(fam_h[fi] - fam_h[fj])
    fam_balance = _clip(fam_balance, 0.0, 1.0)

    score = _clip(
        0.40 * compat_ij
        + 0.34 * recip
        + 0.16 * (1.0 - incomp_ij)
        + 0.10 * fam_balance,
        0.0,
        1.0,
    )
    return score, recip


def _best_pair(ctx, body) -> tuple[int, int, float]:
    best = (0, 1, -1.0)
    for i in range(4):
        for j in range(i + 1, 4):
            s, r = _pair_score(ctx, body, i, j)
            total = 0.65 * s + 0.35 * r
            if total > best[2]:
                best = (i, j, total)
    return best


def _choose_escort_partner(ctx, body, weak_slot: int, best_pair: tuple[int, int, float]) -> int:
    compat = getattr(ctx, "compatibility_matrix", [])
    incomp = getattr(ctx, "incompatibility_matrix", [])
    a, b, _ = best_pair

    candidates = []
    for j in range(4):
        if j == weak_slot:
            continue
        score = (
            0.45 * _safe_get(compat, weak_slot, j)
            + 0.20 * _safe_get(compat, j, weak_slot)
            + 0.20 * _flow(ctx, weak_slot, j)
            + 0.10 * (1.0 - _safe_get(incomp, weak_slot, j))
            + 0.05 * (1.0 if j in (a, b) else 0.0)
        )
        candidates.append((score, j))
    candidates.sort(reverse=True)
    return candidates[0][1]


def _apply_floor_rescue(ctx, protected_slots: set[int], floor_value: float = 0.76) -> tuple[float, int]:
    rows = [[float(v) for v in row] for row in ctx.slot_regime_binding]
    if not rows:
        return 0.0, -1

    current = [_row_sum(r) for r in rows]
    weakest = min(range(len(current)), key=lambda i: current[i])

    needs = [max(0.0, floor_value - s) for s in current]
    total_need = sum(needs)
    if total_need <= 1e-12:
        return 0.0, weakest

    caps = []
    for i, s in enumerate(current):
        headroom = max(0.0, s - (floor_value + 0.010))
        if i in protected_slots:
            headroom *= 0.45
        caps.append(headroom)

    total_cap = sum(caps)
    if total_cap <= 1e-12:
        return 0.0, weakest

    transfer = min(total_need, total_cap)
    row_targets = current[:]

    if total_need > 0.0:
        for i in range(len(row_targets)):
            if needs[i] > 0.0:
                row_targets[i] += transfer * (needs[i] / total_need)

    if total_cap > 0.0:
        for i in range(len(row_targets)):
            if caps[i] > 0.0:
                row_targets[i] -= transfer * (caps[i] / total_cap)

    col_targets = [0.78, 0.78, 0.78, 0.94]
    projected = _ipf_project(rows, row_targets=row_targets, col_targets=col_targets, iters=12)

    for i in range(len(projected)):
        ctx.slot_regime_binding[i] = [float(v) for v in projected[i]]

    return transfer, weakest


@dataclass
class CV8Log:
    floor_rescue_peak: float = 0.0
    escort_gain_peak: float = 0.0
    selected_pair_guard_peak: float = 0.0
    weak_slot_last: int = -1
    best_pair_last: str = ""


def apply_cv8_local_reciprocity_guard(state, log: CV8Log) -> None:
    ctx = state.context_state
    body = state.body_state
    offline = state.offline_state

    rows = getattr(ctx, "slot_regime_binding", [])
    if not rows:
        return

    row_sums = [_row_sum(r) for r in rows]
    weak_slots = [i for i, s in enumerate(row_sums) if s < 0.74]

    best_pair = _best_pair(ctx, body)
    a, b, pair_score = best_pair
    log.best_pair_last = f"{a}-{b}"

    replay = 1.0 + 0.20 * float(getattr(offline, "graph_replay_strength", 0.0))
    escort_gain = 0.0
    selected_guard = 0.0

    # si hay fila débil, no seguir inflando el par fuerte: derivar parte de la legitimidad hacia el slot periférico
    if weak_slots:
        weak = min(weak_slots, key=lambda i: row_sums[i])
        partner = _choose_escort_partner(ctx, body, weak, best_pair)
        gap = _clip(0.76 - row_sums[weak], 0.0, 0.20)

        tg = getattr(ctx, "transition_grammar_matrix", [])
        rp = getattr(ctx, "return_path_matrix", [])
        tm = getattr(ctx, "transition_matrix", [])

        # poda muy local del par dominante si no incluye la fila débil
        if weak not in (a, b):
            for x, y in ((a, b), (b, a)):
                tg_old = _safe_get(tg, x, y)
                rp_old = _safe_get(rp, x, y)
                tm_old = _safe_get(tm, x, y)

                tg_new = tg_old * (1.0 - 0.020 * gap)
                rp_new = rp_old * (1.0 - 0.018 * gap)
                tm_new = tm_old * (1.0 - 0.010 * gap)

                _safe_set(tg, x, y, _clip(tg_new, 0.0, 1.0))
                _safe_set(rp, x, y, _clip(rp_new, 0.0, 1.0))
                _safe_set(tm, x, y, _clip(tm_new, 0.0, 1.0))

                selected_guard += max(0.0, tg_old - tg_new) + max(0.0, rp_old - rp_new) + max(0.0, tm_old - tm_new)

        # injerto recíproco local compatible con el slot débil
        for x, y in ((weak, partner), (partner, weak)):
            tg_old = _safe_get(tg, x, y)
            rp_old = _safe_get(rp, x, y)
            tm_old = _safe_get(tm, x, y)

            tg_gain = 0.014 * (0.40 + pair_score) * replay * (1.0 - tg_old) * (0.40 + 2.0 * gap)
            rp_gain = 0.016 * (0.35 + pair_score) * replay * (1.0 - rp_old) * (0.35 + 2.4 * gap)
            tm_gain = 0.006 * (0.30 + pair_score) * replay * (1.0 - tm_old) * (0.25 + 1.5 * gap)

            _safe_set(tg, x, y, _clip(tg_old + tg_gain, 0.0, 1.0))
            _safe_set(rp, x, y, _clip(rp_old + rp_gain, 0.0, 1.0))
            _safe_set(tm, x, y, _clip(tm_old + tm_gain, 0.0, 1.0))

            escort_gain += tg_gain + rp_gain + tm_gain

        transfer, weakest = _apply_floor_rescue(ctx, protected_slots={a, b, partner}, floor_value=0.76)
        log.floor_rescue_peak = max(log.floor_rescue_peak, transfer)
        log.weak_slot_last = weakest

    else:
        # si no hay fila débil, no abrir otra deriva: solo rescate homeostático muy leve si row_min se acerca demasiado al suelo
        transfer, weakest = _apply_floor_rescue(ctx, protected_slots={a, b}, floor_value=0.745)
        log.floor_rescue_peak = max(log.floor_rescue_peak, transfer)
        log.weak_slot_last = weakest

    _update_binding_summary(ctx)

    log.escort_gain_peak = max(log.escort_gain_peak, escort_gain)
    log.selected_pair_guard_peak = max(log.selected_pair_guard_peak, selected_guard)


def collect_cv8_summary(state, cv7_summary: dict, log: CV8Log) -> dict:
    out = dict(cv7_summary)
    out.update({
        "cv8_floor_rescue_peak": float(log.floor_rescue_peak),
        "cv8_escort_gain_peak": float(log.escort_gain_peak),
        "cv8_selected_pair_guard_peak": float(log.selected_pair_guard_peak),
        "cv8_weak_slot_last": int(log.weak_slot_last),
        "cv8_best_pair_last": log.best_pair_last,
    })
    return out
