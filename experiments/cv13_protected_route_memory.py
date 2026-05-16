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
class CV13Log:
    protected_pair_count_peak: float = 0.0
    route_memory_peak: float = 0.0
    protected_gain_peak: float = 0.0
    unsupported_prune_peak: float = 0.0
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

    score = _clip(
        0.18 * comp
        + 0.22 * trans_rec
        + 0.32 * gram_rec
        + 0.28 * ret_rec
        - 0.12 * inc,
        0.0,
        1.0,
    )
    return score


def _best_pair(ctx, n: int) -> tuple[int, int, float]:
    best_i, best_j, best_score = 0, 0, 0.0
    for i in range(n):
        for j in range(i + 1, n):
            score = _pair_score(ctx, i, j)
            if score > best_score:
                best_i, best_j, best_score = i, j, score
    return best_i, best_j, best_score


def apply_cv13_protected_route_memory(state, log: CV13Log) -> None:
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

    mem = _ensure_square_attr(ctx, "cv13_route_memory", n, 0.0)

    overlap = float(getattr(ctx, "regime_overlap_mean", 1.0))
    specificity = float(getattr(ctx, "binding_specificity_mean", 0.0))
    row_min = float(getattr(ctx, "binding_row_min", 0.0))
    dead_frac = float(getattr(ctx, "dead_slot_fraction", 1.0))
    alignment = float(getattr(ctx, "last_alignment", 0.0))
    grammar_trace = float(getattr(ctx, "grammar_trace", 0.0))
    return_trace = float(getattr(ctx, "return_trace", 0.0))

    geometry_guard = _clip(
        1.25 * (0.19 - overlap)
        + 1.10 * (specificity - 0.59)
        + 0.75 * (row_min - 0.76)
        + 0.35 * (alignment - 0.97)
        + 0.20 * grammar_trace
        + 0.20 * return_trace
        + 0.10 * (1.0 - dead_frac),
        0.0,
        1.0,
    )

    # Decaimiento suave de memoria de ruta
    for i in range(n):
        for j in range(n):
            if i != j:
                mem[i][j] *= 0.9925
                if mem[i][j] < 1e-6:
                    mem[i][j] = 0.0

    best_i, best_j, best_score = _best_pair(ctx, n)

    protected_pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            score = _pair_score(ctx, i, j)
            if score >= 0.34:
                protected_pairs += 1

                # si la ruta ya es legítima, gana memoria
                mem_gain = _clip(
                    geometry_guard * (0.004 + 0.014 * (score - 0.34) / 0.66),
                    0.0,
                    0.014,
                )
                mem[i][j] = _clip(mem[i][j] + mem_gain, 0.0, 1.0)
                mem[j][i] = _clip(mem[j][i] + mem_gain, 0.0, 1.0)

                trans_rec = min(_safe_get(trans, i, j), _safe_get(trans, j, i))
                protected_gain = _clip(
                    0.006 * geometry_guard
                    + 0.022 * mem[i][j]
                    + 0.008 * trans_rec,
                    0.0,
                    0.025,
                )

                # Refuerzo protegido: grammar y return solo sobre pares ya legítimos
                gram[i][j] = _clip(gram[i][j] + 0.95 * protected_gain, 0.0, 1.0)
                gram[j][i] = _clip(gram[j][i] + 0.95 * protected_gain, 0.0, 1.0)

                ret[i][j] = _clip(ret[i][j] + 0.80 * protected_gain, 0.0, 1.0)
                ret[j][i] = _clip(ret[j][i] + 0.80 * protected_gain, 0.0, 1.0)

                # soporte muy pequeño de transición solo si ya había reciprocidad real
                if trans_rec > 0.18:
                    bump = 0.30 * protected_gain
                    trans[i][j] = _clip(trans[i][j] + bump, 0.0, 1.0)
                    trans[j][i] = _clip(trans[j][i] + bump, 0.0, 1.0)

                log.protected_gain_peak = max(log.protected_gain_peak, protected_gain)

    # poda de transición espuria: alta pero sin respaldo recíproco suficiente
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            local_score = _pair_score(ctx, i, j)
            t = _safe_get(trans, i, j)
            g = _safe_get(gram, i, j)
            r = _safe_get(ret, i, j)

            if t > 0.55 and local_score < 0.22:
                prune = _clip(0.030 * geometry_guard * (t - 0.55), 0.0, 0.018)
                trans[i][j] = _clip(trans[i][j] - prune, 0.0, 1.0)
                gram[i][j] = _clip(gram[i][j] - 0.70 * prune, 0.0, 1.0)
                ret[i][j] = _clip(ret[i][j] - 0.55 * prune, 0.0, 1.0)
                log.unsupported_prune_peak = max(log.unsupported_prune_peak, prune)

    route_memory_peak = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                route_memory_peak = max(route_memory_peak, float(mem[i][j]))

    log.protected_pair_count_peak = max(log.protected_pair_count_peak, float(protected_pairs))
    log.route_memory_peak = max(log.route_memory_peak, route_memory_peak)
    log.best_pair_last = f"{best_i}-{best_j}"
    log.best_pair_score_last = float(best_score)


def collect_cv13_summary(state, cv12_summary: dict, log: CV13Log) -> dict:
    out = dict(cv12_summary)
    out.update({
        "cv13_protected_pair_count_peak": float(log.protected_pair_count_peak),
        "cv13_route_memory_peak": float(log.route_memory_peak),
        "cv13_protected_gain_peak": float(log.protected_gain_peak),
        "cv13_unsupported_prune_peak": float(log.unsupported_prune_peak),
        "cv13_best_pair_last": log.best_pair_last,
        "cv13_best_pair_score_last": float(log.best_pair_score_last),
    })
    return out
