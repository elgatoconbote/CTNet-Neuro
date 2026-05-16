from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _family_of_slot(slot: int) -> int:
    return 0 if slot in (0, 3) else 1


def _ensure_matrix(ctx, name: str, n: int = 4) -> None:
    if not hasattr(ctx, name):
        setattr(ctx, name, [[0.0 for _ in range(n)] for _ in range(n)])


def _safe_get(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


def _safe_set(mat, i: int, j: int, value: float) -> None:
    mat[i][j] = float(value)


def _row_cap_keep_top_k(mat, row_cap: float, k: int = 2) -> float:
    n = len(mat)
    pruned = 0.0
    for i in range(n):
        row = mat[i]
        total = sum(row)
        if total <= row_cap + 1e-12:
            continue

        idx = sorted(range(len(row)), key=lambda j: row[j], reverse=True)
        keep = set(idx[:k])
        keep_sum = sum(row[j] for j in keep)
        rest_sum = sum(row[j] for j in range(len(row)) if j not in keep)

        target_rest = max(0.0, row_cap - keep_sum)
        if rest_sum <= 1e-12:
            continue

        scale = target_rest / rest_sum
        for j in range(len(row)):
            if j in keep:
                continue
            old = row[j]
            row[j] *= scale
            pruned += max(0.0, old - row[j])

    return pruned


def _flow(ctx, i: int, j: int) -> float:
    tm = getattr(ctx, "transition_matrix", [])
    tg = getattr(ctx, "transition_grammar_matrix", [])
    rp = getattr(ctx, "return_path_matrix", [])
    return _clip(
        0.50 * _safe_get(tm, i, j)
        + 0.30 * _safe_get(tg, i, j)
        + 0.20 * _safe_get(rp, i, j),
        0.0,
        1.0,
    )


def _pair_score(ctx, body, i: int, j: int) -> tuple[float, float]:
    compat = getattr(ctx, "compatibility_matrix", [])
    incomp = getattr(ctx, "incompatibility_matrix", [])

    fam_h = list(getattr(body, "family_hysteresis", [0.0, 0.0]))
    debt_by_slot = list(getattr(body, "debt_by_slot", [0.0] * 4))
    scar_by_slot = list(getattr(body, "scar_by_slot", [0.0] * 4))

    compat_ij = 0.5 * (_safe_get(compat, i, j) + _safe_get(compat, j, i))
    incomp_ij = 0.5 * (_safe_get(incomp, i, j) + _safe_get(incomp, j, i))

    fwd = _flow(ctx, i, j)
    bwd = _flow(ctx, j, i)
    reciprocal = min(fwd, bwd)

    fi = _family_of_slot(i)
    fj = _family_of_slot(j)
    family_balance = 1.0 - abs(fam_h[fi] - fam_h[fj])
    family_balance = _clip(family_balance, 0.0, 1.0)

    debt_pen = 0.5 * (debt_by_slot[i] + debt_by_slot[j])
    scar_pen = 0.5 * (scar_by_slot[i] + scar_by_slot[j])

    score = _clip(
        0.36 * compat_ij
        + 0.30 * reciprocal
        + 0.18 * (1.0 - incomp_ij)
        + 0.10 * family_balance
        - 0.04 * debt_pen
        - 0.02 * scar_pen,
        0.0,
        1.0,
    )
    return score, reciprocal


@dataclass
class CV7Log:
    selected_pair_count_peak: int = 0
    reciprocal_gain_peak: float = 0.0
    spurious_prune_peak: float = 0.0
    best_pair_score_last: float = 0.0
    best_pair_last: str = ""


def apply_cv7_selective_reciprocal_routes(state, log: CV7Log) -> None:
    ctx = state.context_state
    body = state.body_state
    world = state.world_state
    offline = state.offline_state

    _ensure_matrix(ctx, "transition_matrix", 4)
    _ensure_matrix(ctx, "transition_grammar_matrix", 4)
    _ensure_matrix(ctx, "return_path_matrix", 4)

    tm = ctx.transition_matrix
    tg = ctx.transition_grammar_matrix
    rp = ctx.return_path_matrix

    active = int(ctx.active_index)
    challenger = int(getattr(ctx, "challenger_index", active))

    pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            score, reciprocal = _pair_score(ctx, body, i, j)
            active_bonus = 0.0
            if active in (i, j):
                active_bonus += 0.08
            if challenger in (i, j):
                active_bonus += 0.05
            total = _clip(score + active_bonus, 0.0, 1.0)
            pairs.append((total, reciprocal, i, j))

    pairs.sort(reverse=True, key=lambda x: (x[0], x[1]))

    # solo pares legítimos y ya parcialmente recíprocos
    selected = []
    for total, reciprocal, i, j in pairs:
        if total >= 0.26 and reciprocal >= 0.045:
            selected.append((total, reciprocal, i, j))
        if len(selected) >= 2:
            break

    log.selected_pair_count_peak = max(log.selected_pair_count_peak, len(selected))
    if selected:
        log.best_pair_score_last = float(selected[0][0])
        log.best_pair_last = f"{selected[0][2]}-{selected[0][3]}"

    replay_bonus = 1.0 + 0.25 * float(getattr(offline, "graph_replay_strength", 0.0))
    gain_acc = 0.0

    for total, reciprocal, i, j in selected:
        same_family = (_family_of_slot(i) == _family_of_slot(j))
        grammar_bias = 1.05 if same_family else 0.92
        return_bias = 1.10 if not same_family else 0.88

        for a, b in ((i, j), (j, i)):
            tg_old = _safe_get(tg, a, b)
            rp_old = _safe_get(rp, a, b)
            tm_old = _safe_get(tm, a, b)

            tg_gain = 0.020 * total * replay_bonus * grammar_bias * (1.0 - tg_old)
            rp_gain = 0.016 * total * replay_bonus * return_bias * (1.0 - rp_old)
            tm_gain = 0.006 * total * replay_bonus * (1.0 - tm_old)

            _safe_set(tg, a, b, _clip(tg_old + tg_gain, 0.0, 1.0))
            _safe_set(rp, a, b, _clip(rp_old + rp_gain, 0.0, 1.0))
            _safe_set(tm, a, b, _clip(tm_old + tm_gain, 0.0, 1.0))

            gain_acc += tg_gain + rp_gain + tm_gain

    # poda espuria: transición alta sin apoyo recíproco/compatible
    pruned = 0.0
    for i in range(4):
        for j in range(4):
            if i == j:
                continue

            selected_edge = any((i, j) == (a, b) or (i, j) == (b, a) for _, _, a, b in selected)
            if selected_edge:
                continue

            compat = getattr(ctx, "compatibility_matrix", [])
            incomp = getattr(ctx, "incompatibility_matrix", [])
            compat_ij = _safe_get(compat, i, j)
            incomp_ij = _safe_get(incomp, i, j)
            recip = min(_flow(ctx, i, j), _flow(ctx, j, i))

            spur = max(
                0.0,
                _safe_get(tm, i, j)
                - (0.55 * compat_ij + 0.30 * recip + 0.10 * (1.0 - incomp_ij))
            )

            if spur > 0.08:
                tm_old = _safe_get(tm, i, j)
                tg_old = _safe_get(tg, i, j)
                rp_old = _safe_get(rp, i, j)

                tm_new = tm_old * (1.0 - 0.050 * spur)
                tg_new = tg_old * (1.0 - 0.070 * spur)
                rp_new = rp_old * (1.0 - 0.085 * spur)

                _safe_set(tm, i, j, _clip(tm_new, 0.0, 1.0))
                _safe_set(tg, i, j, _clip(tg_new, 0.0, 1.0))
                _safe_set(rp, i, j, _clip(rp_new, 0.0, 1.0))

                pruned += max(0.0, tm_old - tm_new) + max(0.0, tg_old - tg_new) + max(0.0, rp_old - rp_new)

    pruned += _row_cap_keep_top_k(tm, row_cap=2.20, k=2)
    pruned += _row_cap_keep_top_k(tg, row_cap=0.95, k=2)
    pruned += _row_cap_keep_top_k(rp, row_cap=0.48, k=2)

    log.reciprocal_gain_peak = max(log.reciprocal_gain_peak, gain_acc)
    log.spurious_prune_peak = max(log.spurious_prune_peak, pruned)


def collect_cv7_summary(state, cv6_summary: dict, log: CV7Log) -> dict:
    out = dict(cv6_summary)
    out.update({
        "cv7_selected_pair_count_peak": int(log.selected_pair_count_peak),
        "cv7_reciprocal_gain_peak": float(log.reciprocal_gain_peak),
        "cv7_spurious_prune_peak": float(log.spurious_prune_peak),
        "cv7_best_pair_last": log.best_pair_last,
        "cv7_best_pair_score_last": float(log.best_pair_score_last),
    })
    return out
