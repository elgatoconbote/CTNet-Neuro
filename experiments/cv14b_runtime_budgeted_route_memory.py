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


def _ensure_square_attr(obj, name: str, n: int, fill: float = 0.0):
    mat = getattr(obj, name, None)
    ok = isinstance(mat, list) and len(mat) == n and all(isinstance(row, list) and len(row) == n for row in mat)
    if not ok:
        mat = [[float(fill) for _ in range(n)] for _ in range(n)]
        setattr(obj, name, mat)
    return mat


def _sig_vec(hyp) -> tuple[float, float, float, float, float]:
    return (
        float(getattr(hyp, "signature_projection", 0.0)),
        float(getattr(hyp, "signature_tissue", 0.0)),
        float(getattr(hyp, "signature_structure", 0.0)),
        float(getattr(hyp, "signature_salience", 0.0)),
        float(getattr(hyp, "signature_body", 0.0)),
    )


def _cos_sim(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return dot / (na * nb)


def _runtime_pair_mean(hyps) -> float:
    n = len(hyps)
    if n <= 1:
        return 0.0
    vals = []
    vecs = [_sig_vec(h) for h in hyps]
    for i in range(n):
        for j in range(i + 1, n):
            vals.append(_cos_sim(vecs[i], vecs[j]))
    return float(sum(vals) / max(1, len(vals)))


def _row_sum(row) -> float:
    return float(sum(float(x) for x in row))


def _runtime_specificity(binding) -> float:
    vals = []
    for row in binding:
        total = _row_sum(row)
        if total <= 1e-12:
            vals.append(0.0)
        else:
            vals.append(max(float(x) for x in row) / total)
    return float(sum(vals) / max(1, len(vals)))


def _runtime_overlap(binding) -> float:
    n = len(binding)
    if n <= 1:
        return 0.0
    cols = len(binding[0]) if binding else 0
    vals = []
    for r in range(cols):
        local = []
        for i in range(n):
            for j in range(i + 1, n):
                local.append(min(float(binding[i][r]), float(binding[j][r])))
        if local:
            vals.append(sum(local) / len(local))
    return float(sum(vals) / max(1, len(vals)))


def _dead_slot_fraction(binding, floor: float = 0.74) -> float:
    if not binding:
        return 1.0
    bad = 0
    for row in binding:
        if _row_sum(row) < floor:
            bad += 1
    return float(bad) / float(len(binding))


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
        0.17 * comp
        + 0.19 * trans_rec
        + 0.31 * gram_rec
        + 0.33 * ret_rec
        - 0.12 * inc,
        0.0,
        1.0,
    )


@dataclass
class CV14bLog:
    route_memory_peak: float = 0.0
    protected_gain_peak: float = 0.0
    unsupported_prune_peak: float = 0.0
    geometry_budget_peak: float = 0.0
    candidate_count_peak: float = 0.0
    selected_pair_count_peak: float = 0.0
    budget_last: float = 0.0
    candidate_count_last: int = 0
    selected_pair_count_last: int = 0
    best_pair_last: str = ""
    best_pair_score_last: float = 0.0


def apply_cv14b_runtime_budgeted_route_memory(state, log: CV14bLog) -> None:
    ctx = state.context_state
    hyps = getattr(ctx, "hypotheses", [])
    n = len(hyps)
    if n <= 1:
        return

    binding = getattr(ctx, "slot_regime_binding", None)
    trans = getattr(ctx, "transition_matrix", None)
    gram = getattr(ctx, "transition_grammar_matrix", None)
    ret = getattr(ctx, "return_path_matrix", None)
    compat = getattr(ctx, "compatibility_matrix", None)
    incomp = getattr(ctx, "incompatibility_matrix", None)

    if binding is None or trans is None or gram is None or ret is None or compat is None or incomp is None:
        return

    mem = _ensure_square_attr(ctx, "cv14b_route_memory", n, 0.0)

    pair_mean = _runtime_pair_mean(hyps)
    overlap = _runtime_overlap(binding)
    specificity = _runtime_specificity(binding)
    row_min = min(_row_sum(row) for row in binding)
    dead_frac = _dead_slot_fraction(binding)
    alignment = float(getattr(ctx, "last_alignment", 0.0))
    grammar_trace = float(getattr(ctx, "grammar_trace", 0.0))
    return_trace = float(getattr(ctx, "return_trace", 0.0))

    geometry_budget = _clip(
        1.55 * (0.19 - overlap)
        + 1.10 * (specificity - 0.56)
        + 0.75 * (0.67 - pair_mean)
        + 0.45 * (row_min - 0.76)
        + 0.25 * (alignment - 0.96)
        + 0.15 * grammar_trace
        + 0.15 * return_trace
        + 0.05 * (1.0 - dead_frac),
        0.0,
        1.0,
    )

    # decaimiento basal
    for i in range(n):
        for j in range(n):
            if i != j:
                mem[i][j] *= 0.992
                if mem[i][j] < 1e-6:
                    mem[i][j] = 0.0

    candidates = []
    best_i, best_j, best_score = 0, 0, 0.0

    for i in range(n):
        for j in range(i + 1, n):
            score = _pair_score(ctx, i, j)
            if score > best_score:
                best_i, best_j, best_score = i, j, score
            if score >= 0.34:
                candidates.append((score, i, j))

    candidates.sort(reverse=True)

    max_pairs = 0
    if geometry_budget > 0.07:
        max_pairs = 1
    if geometry_budget > 0.22 and len(candidates) >= 2:
        max_pairs = 2

    selected = candidates[:max_pairs]

    for score, i, j in selected:
        mem_gain = _clip(
            0.003 + 0.012 * geometry_budget * score,
            0.0,
            0.014,
        )
        mem[i][j] = _clip(mem[i][j] + mem_gain, 0.0, 1.0)
        mem[j][i] = _clip(mem[j][i] + mem_gain, 0.0, 1.0)

        trans_rec = min(_safe_get(trans, i, j), _safe_get(trans, j, i))

        protected_gain = _clip(
            0.0025
            + 0.012 * mem[i][j]
            + 0.008 * geometry_budget
            + 0.004 * trans_rec,
            0.0,
            0.018,
        )

        gram[i][j] = _clip(gram[i][j] + 0.90 * protected_gain, 0.0, 1.0)
        gram[j][i] = _clip(gram[j][i] + 0.90 * protected_gain, 0.0, 1.0)

        ret[i][j] = _clip(ret[i][j] + 0.78 * protected_gain, 0.0, 1.0)
        ret[j][i] = _clip(ret[j][i] + 0.78 * protected_gain, 0.0, 1.0)

        if trans_rec > 0.22:
            bump = 0.18 * protected_gain
            trans[i][j] = _clip(trans[i][j] + bump, 0.0, 1.0)
            trans[j][i] = _clip(trans[j][i] + bump, 0.0, 1.0)

        log.protected_gain_peak = max(log.protected_gain_peak, protected_gain)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            score = _pair_score(ctx, i, j)
            t = _safe_get(trans, i, j)

            if t > 0.52 and score < 0.24:
                prune = _clip(
                    (
                        0.050 * (1.0 - geometry_budget) * (t - 0.52)
                        + 0.016 * max(0.0, 0.24 - score)
                    ),
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
    log.candidate_count_peak = max(log.candidate_count_peak, float(len(candidates)))
    log.selected_pair_count_peak = max(log.selected_pair_count_peak, float(len(selected)))
    log.budget_last = float(geometry_budget)
    log.candidate_count_last = int(len(candidates))
    log.selected_pair_count_last = int(len(selected))
    log.best_pair_last = f"{best_i}-{best_j}"
    log.best_pair_score_last = float(best_score)


def collect_cv14b_summary(state, cv12_summary: dict, log: CV14bLog) -> dict:
    out = dict(cv12_summary)
    out.update({
        "cv14b_route_memory_peak": float(log.route_memory_peak),
        "cv14b_protected_gain_peak": float(log.protected_gain_peak),
        "cv14b_unsupported_prune_peak": float(log.unsupported_prune_peak),
        "cv14b_geometry_budget_peak": float(log.geometry_budget_peak),
        "cv14b_candidate_count_peak": float(log.candidate_count_peak),
        "cv14b_selected_pair_count_peak": float(log.selected_pair_count_peak),
        "cv14b_budget_last": float(log.budget_last),
        "cv14b_candidate_count_last": int(log.candidate_count_last),
        "cv14b_selected_pair_count_last": int(log.selected_pair_count_last),
        "cv14b_best_pair_last": log.best_pair_last,
        "cv14b_best_pair_score_last": float(log.best_pair_score_last),
    })
    return out
