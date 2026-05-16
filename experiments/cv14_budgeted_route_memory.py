from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_get(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


def _ensure_square_attr(obj, name: str, n: int, fill: float = 0.0):
    mat = getattr(obj, name, None)
    ok = isinstance(mat, list) and len(mat) == n and all(isinstance(row, list) and len(row) == n for row in mat)
    if not ok:
        mat = [[float(fill) for _ in range(n)] for _ in range(n)]
        setattr(obj, name, mat)
    return mat


@dataclass
class CV14Log:
    route_memory_peak: float = 0.0
    protected_gain_peak: float = 0.0
    unsupported_prune_peak: float = 0.0
    geometry_budget_peak: float = 0.0
    selected_pair_count_peak: float = 0.0
    best_pair_last: str = ""
    best_pair_score_last: float = 0.0


def _pair_score(ctx, i: int, j: int) -> float:
    compat = getattr(ctx, "compatibility_matrix", [])
    incomp = getattr(ctx, "incompatibility_matrix", [])
    trans = getattr(ctx, "transition_matrix", [])
    gram = getattr(ctx, "transition_grammar_matrix", [])
    ret = getattr(ctx, "return_path_matrix", [])

    comp = 0.5 * (_safe_get(compat, i, j) + _safe_get(compat, j, i))
    inc = 0.5 * (_safe_get(incomp, i, j) + _safe_get(incomp, j, i))
    trans_rec = min(_safe_get(trans, i, j), _safe_get(trans, j, i))
    gram_rec = min(_safe_get(gram, i, j), _safe_get(gram, j, i))
    ret_rec = min(_safe_get(ret, i, j), _safe_get(ret, j, i))

    return _clip(
        0.16 * comp
        + 0.20 * trans_rec
        + 0.32 * gram_rec
        + 0.32 * ret_rec
        - 0.14 * inc,
        0.0,
        1.0,
    )


def apply_cv14_budgeted_route_memory(state, log: CV14Log) -> None:
    ctx = state.context_state
    hyps = getattr(ctx, "hypotheses", [])
    n = len(hyps)
    if n <= 1:
        return

    trans = getattr(ctx, "transition_matrix", None)
    gram = getattr(ctx, "transition_grammar_matrix", None)
    ret = getattr(ctx, "return_path_matrix", None)
    if trans is None or gram is None or ret is None:
        return

    mem = _ensure_square_attr(ctx, "cv14_route_memory", n, 0.0)

    overlap = float(getattr(ctx, "regime_overlap_mean", 1.0))
    specificity = float(getattr(ctx, "binding_specificity_mean", 0.0))
    pair_mean = float(getattr(ctx, "pairwise_mean_similarity", 1.0))
    row_min = float(getattr(ctx, "binding_row_min", 0.0))
    dead_frac = float(getattr(ctx, "dead_slot_fraction", 1.0))
    alignment = float(getattr(ctx, "last_alignment", 0.0))
    grammar_trace = float(getattr(ctx, "grammar_trace", 0.0))
    return_trace = float(getattr(ctx, "return_trace", 0.0))

    # Presupuesto geométrico: si el manifold empieza a densificarse demasiado, se cierra el grifo.
    geometry_budget = _clip(
        1.40 * (0.17 - overlap)
        + 1.10 * (specificity - 0.60)
        + 0.95 * (0.64 - pair_mean)
        + 0.55 * (row_min - 0.76)
        + 0.25 * (alignment - 0.97)
        + 0.10 * (1.0 - dead_frac),
        0.0,
        1.0,
    )

    # Decaimiento basal de memoria
    for i in range(n):
        for j in range(n):
            if i != j:
                mem[i][j] *= 0.991
                if mem[i][j] < 1e-6:
                    mem[i][j] = 0.0

    candidates = []
    best_pair = (0, 0)
    best_score = 0.0

    for i in range(n):
        for j in range(i + 1, n):
            score = _pair_score(ctx, i, j)
            if score > best_score:
                best_pair = (i, j)
                best_score = score

            if score >= 0.36:
                candidates.append((score, i, j))

    candidates.sort(reverse=True)

    # Cupo competitivo: no más de 2 pares, y a veces solo 1 si el presupuesto es estrecho.
    max_pairs = 0
    if geometry_budget > 0.10:
        max_pairs = 1
    if geometry_budget > 0.40 and len(candidates) >= 2:
        max_pairs = 2

    selected = candidates[:max_pairs]

    for score, i, j in selected:
        mem_gain = _clip(
            geometry_budget * (0.003 + 0.012 * (score - 0.36) / 0.64),
            0.0,
            0.012,
        )
        mem[i][j] = _clip(mem[i][j] + mem_gain, 0.0, 1.0)
        mem[j][i] = _clip(mem[j][i] + mem_gain, 0.0, 1.0)

        trans_rec = min(_safe_get(trans, i, j), _safe_get(trans, j, i))

        protected_gain = _clip(
            0.004 * geometry_budget
            + 0.015 * mem[i][j]
            + 0.006 * trans_rec,
            0.0,
            0.016,
        )

        # Refuerzo protegido y presupuestado
        gram[i][j] = _clip(gram[i][j] + 0.90 * protected_gain, 0.0, 1.0)
        gram[j][i] = _clip(gram[j][i] + 0.90 * protected_gain, 0.0, 1.0)

        ret[i][j] = _clip(ret[i][j] + 0.78 * protected_gain, 0.0, 1.0)
        ret[j][i] = _clip(ret[j][i] + 0.78 * protected_gain, 0.0, 1.0)

        if trans_rec > 0.22:
            bump = 0.18 * protected_gain
            trans[i][j] = _clip(trans[i][j] + bump, 0.0, 1.0)
            trans[j][i] = _clip(trans[j][i] + bump, 0.0, 1.0)

        log.protected_gain_peak = max(log.protected_gain_peak, protected_gain)

    # Poda más fuerte para transición sin respaldo real
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            score = _pair_score(ctx, i, j)
            t = _safe_get(trans, i, j)

            if t > 0.52 and score < 0.24:
                prune = _clip(
                    0.050 * (1.0 - geometry_budget) * (t - 0.52)
                    + 0.016 * max(0.0, 0.24 - score),
                    0.0,
                    0.028,
                )
                trans[i][j] = _clip(trans[i][j] - prune, 0.0, 1.0)
                gram[i][j] = _clip(gram[i][j] - 0.70 * prune, 0.0, 1.0)
                ret[i][j] = _clip(ret[i][j] - 0.55 * prune, 0.0, 1.0)
                log.unsupported_prune_peak = max(log.unsupported_prune_peak, prune)

    route_memory_peak = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                route_memory_peak = max(route_memory_peak, float(mem[i][j]))

    log.route_memory_peak = max(log.route_memory_peak, route_memory_peak)
    log.geometry_budget_peak = max(log.geometry_budget_peak, geometry_budget)
    log.selected_pair_count_peak = max(log.selected_pair_count_peak, float(len(selected)))
    log.best_pair_last = f"{best_pair[0]}-{best_pair[1]}"
    log.best_pair_score_last = float(best_score)


def collect_cv14_summary(state, cv12_summary: dict, log: CV14Log) -> dict:
    out = dict(cv12_summary)
    out.update({
        "cv14_route_memory_peak": float(log.route_memory_peak),
        "cv14_protected_gain_peak": float(log.protected_gain_peak),
        "cv14_unsupported_prune_peak": float(log.unsupported_prune_peak),
        "cv14_geometry_budget_peak": float(log.geometry_budget_peak),
        "cv14_selected_pair_count_peak": float(log.selected_pair_count_peak),
        "cv14_best_pair_last": log.best_pair_last,
        "cv14_best_pair_score_last": float(log.best_pair_score_last),
    })
    return out
