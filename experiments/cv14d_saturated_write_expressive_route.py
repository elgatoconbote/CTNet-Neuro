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
        0.15 * comp
        + 0.14 * trans_rec
        + 0.33 * gram_rec
        + 0.38 * ret_rec
        - 0.10 * inc,
        0.0,
        1.0,
    )


@dataclass
class CV14dLog:
    route_memory_peak: float = 0.0
    write_gain_peak: float = 0.0
    express_gain_peak: float = 0.0
    shadow_gain_peak: float = 0.0
    route_recall_peak: float = 0.0
    unsupported_prune_peak: float = 0.0
    geometry_budget_peak: float = 0.0
    geom_headroom_peak: float = 0.0
    candidate_count_peak: float = 0.0
    selected_pair_count_peak: float = 0.0
    shadow_pair_count_peak: float = 0.0
    budget_last: float = 0.0
    geom_headroom_last: float = 0.0
    candidate_count_last: int = 0
    selected_pair_count_last: int = 0
    shadow_pair_count_last: int = 0
    best_pair_last: str = ""
    best_pair_score_last: float = 0.0


def apply_cv14d_saturated_write_expressive_route(state, log: CV14dLog) -> None:
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

    MEM_CAP = 0.34
    mem = _ensure_square_attr(ctx, "cv14d_route_memory", n, 0.0)

    pair_mean = _runtime_pair_mean(hyps)
    overlap = _runtime_overlap(binding)
    specificity = _runtime_specificity(binding)
    row_min = min(_row_sum(row) for row in binding)
    dead_frac = _dead_slot_fraction(binding)
    alignment = float(getattr(ctx, "last_alignment", 0.0))
    grammar_trace = float(getattr(ctx, "grammar_trace", 0.0))
    return_trace = float(getattr(ctx, "return_trace", 0.0))

    overlap_head = _clip((0.19 - overlap) / 0.19, 0.0, 1.0)
    spec_head = _clip((specificity - 0.57) / 0.11, 0.0, 1.0)
    sim_head = _clip((0.70 - pair_mean) / 0.22, 0.0, 1.0)
    row_head = _clip((row_min - 0.76) / 0.09, 0.0, 1.0)
    dead_head = _clip(1.0 - dead_frac / 0.25, 0.0, 1.0)

    geom_headroom = _clip(
        0.28 * overlap_head
        + 0.25 * spec_head
        + 0.15 * sim_head
        + 0.18 * row_head
        + 0.14 * dead_head,
        0.0,
        1.0,
    )

    grammar_need = _clip((0.35 - grammar_trace) / 0.35, 0.0, 1.0)
    return_need = _clip((0.23 - return_trace) / 0.23, 0.0, 1.0)
    alignment_need = _clip((0.988 - alignment) / 0.045, 0.0, 1.0)

    rel_need = _clip(
        0.52 * grammar_need
        + 0.36 * return_need
        + 0.12 * alignment_need,
        0.0,
        1.0,
    )

    geometry_budget = _clip(
        rel_need * (0.28 + 0.72 * geom_headroom),
        0.0,
        1.0,
    )

    for i in range(n):
        for j in range(n):
            if i != j:
                mem[i][j] *= 0.995
                if mem[i][j] < 1e-6:
                    mem[i][j] = 0.0

    candidates = []
    best_i, best_j, best_score = 0, 0, 0.0

    for i in range(n):
        for j in range(i + 1, n):
            score = _pair_score(ctx, i, j)
            if score > best_score:
                best_i, best_j, best_score = i, j, score
            if score >= 0.36:
                candidates.append((score, i, j))

    candidates.sort(reverse=True)

    selected = []
    shadow = []

    if geometry_budget > 0.05 and candidates:
        selected.append(candidates[0])

    if (
        geometry_budget > 0.74
        and geom_headroom > 0.82
        and len(candidates) >= 2
        and candidates[1][0] > 0.92 * candidates[0][0]
    ):
        shadow.append(candidates[1])

    overlap_tax = _clip((overlap - 0.18) / 0.08, 0.0, 1.0)
    spec_tax = _clip((0.58 - specificity) / 0.10, 0.0, 1.0)
    geom_tax = _clip(0.58 * overlap_tax + 0.42 * spec_tax, 0.0, 1.0)

    selected_pairs = {(i, j) for _, i, j in selected}
    shadow_pairs = {(i, j) for _, i, j in shadow}

    for score, i, j in selected:
        mem_headroom = _clip(1.0 - (mem[i][j] / MEM_CAP), 0.0, 1.0)

        # escritura pequeña, estrictamente saturada
        write_gain = _clip(
            (0.0009 + 0.0034 * geometry_budget * score) * mem_headroom,
            0.0,
            0.0042,
        )
        mem[i][j] = _clip(mem[i][j] + write_gain, 0.0, MEM_CAP)
        mem[j][i] = _clip(mem[j][i] + write_gain, 0.0, MEM_CAP)

        mem_norm = _clip(mem[i][j] / MEM_CAP, 0.0, 1.0)
        route_recall = _clip(
            (0.18 + 0.82 * mem_norm) * (0.35 + 0.65 * score) * (0.25 + 0.75 * rel_need),
            0.0,
            1.0,
        )

        trans_rec = min(_safe_get(trans, i, j), _safe_get(trans, j, i))

        # expresion desacoplada de la escritura: puede subir aunque la memoria ya esté cerca del tope
        express_gain = _clip(
            (
                0.0008
                + 0.0058 * route_recall
                + 0.0024 * geometry_budget
                + 0.0018 * trans_rec
            )
            * (1.0 - 0.45 * geom_tax),
            0.0,
            0.0088,
        )

        gram[i][j] = _clip(gram[i][j] + 1.00 * express_gain, 0.0, 1.0)
        gram[j][i] = _clip(gram[j][i] + 1.00 * express_gain, 0.0, 1.0)

        ret[i][j] = _clip(ret[i][j] + 0.86 * express_gain, 0.0, 1.0)
        ret[j][i] = _clip(ret[j][i] + 0.86 * express_gain, 0.0, 1.0)

        if trans_rec > 0.22:
            bump = 0.07 * express_gain
            trans[i][j] = _clip(trans[i][j] + bump, 0.0, 1.0)
            trans[j][i] = _clip(trans[j][i] + bump, 0.0, 1.0)

        log.write_gain_peak = max(log.write_gain_peak, write_gain)
        log.express_gain_peak = max(log.express_gain_peak, express_gain)
        log.route_recall_peak = max(log.route_recall_peak, route_recall)

    for score, i, j in shadow:
        mem_headroom = _clip(1.0 - (mem[i][j] / MEM_CAP), 0.0, 1.0)
        write_gain = _clip(
            (0.0004 + 0.0014 * geometry_budget * score) * mem_headroom,
            0.0,
            0.0018,
        )
        mem[i][j] = _clip(mem[i][j] + write_gain, 0.0, MEM_CAP)
        mem[j][i] = _clip(mem[j][i] + write_gain, 0.0, MEM_CAP)

        mem_norm = _clip(mem[i][j] / MEM_CAP, 0.0, 1.0)
        shadow_gain = _clip(
            (0.0003 + 0.0012 * mem_norm + 0.0010 * geometry_budget) * (1.0 - 0.70 * geom_tax),
            0.0,
            0.0018,
        )

        gram[i][j] = _clip(gram[i][j] + 0.78 * shadow_gain, 0.0, 1.0)
        gram[j][i] = _clip(gram[j][i] + 0.78 * shadow_gain, 0.0, 1.0)

        ret[i][j] = _clip(ret[i][j] + 0.50 * shadow_gain, 0.0, 1.0)
        ret[j][i] = _clip(ret[j][i] + 0.50 * shadow_gain, 0.0, 1.0)

        log.shadow_gain_peak = max(log.shadow_gain_peak, shadow_gain)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            pair_ij = (min(i, j), max(i, j))
            protected = pair_ij in selected_pairs or pair_ij in shadow_pairs

            score = _pair_score(ctx, i, j)
            t = _safe_get(trans, i, j)
            g = _safe_get(gram, i, j)
            r = _safe_get(ret, i, j)

            if protected:
                threshold = 0.50
                score_floor = 0.24
                scale = 0.55
            else:
                threshold = 0.44
                score_floor = 0.30
                scale = 1.00

            if t > threshold and score < score_floor:
                prune = _clip(
                    (
                        0.020 * (t - threshold)
                        + 0.020 * max(0.0, score_floor - score)
                        + 0.018 * overlap_tax
                        + 0.014 * spec_tax
                    ) * scale,
                    0.0,
                    0.026,
                )
                trans[i][j] = _clip(t - prune, 0.0, 1.0)
                gram[i][j] = _clip(g - 0.58 * prune, 0.0, 1.0)
                ret[i][j] = _clip(r - 0.46 * prune, 0.0, 1.0)
                log.unsupported_prune_peak = max(log.unsupported_prune_peak, prune)

    route_memory_peak = 0.0
    for i in range(n):
        for j in range(n):
            if i != j:
                route_memory_peak = max(route_memory_peak, float(mem[i][j]))

    log.route_memory_peak = max(log.route_memory_peak, route_memory_peak)
    log.geometry_budget_peak = max(log.geometry_budget_peak, geometry_budget)
    log.geom_headroom_peak = max(log.geom_headroom_peak, geom_headroom)
    log.candidate_count_peak = max(log.candidate_count_peak, float(len(candidates)))
    log.selected_pair_count_peak = max(log.selected_pair_count_peak, float(len(selected)))
    log.shadow_pair_count_peak = max(log.shadow_pair_count_peak, float(len(shadow)))
    log.budget_last = float(geometry_budget)
    log.geom_headroom_last = float(geom_headroom)
    log.candidate_count_last = int(len(candidates))
    log.selected_pair_count_last = int(len(selected))
    log.shadow_pair_count_last = int(len(shadow))
    log.best_pair_last = f"{best_i}-{best_j}"
    log.best_pair_score_last = float(best_score)


def collect_cv14d_summary(state, cv12_summary: dict, log: CV14dLog) -> dict:
    out = dict(cv12_summary)
    out.update({
        "cv14d_route_memory_peak": float(log.route_memory_peak),
        "cv14d_write_gain_peak": float(log.write_gain_peak),
        "cv14d_express_gain_peak": float(log.express_gain_peak),
        "cv14d_shadow_gain_peak": float(log.shadow_gain_peak),
        "cv14d_route_recall_peak": float(log.route_recall_peak),
        "cv14d_unsupported_prune_peak": float(log.unsupported_prune_peak),
        "cv14d_geometry_budget_peak": float(log.geometry_budget_peak),
        "cv14d_geom_headroom_peak": float(log.geom_headroom_peak),
        "cv14d_candidate_count_peak": float(log.candidate_count_peak),
        "cv14d_selected_pair_count_peak": float(log.selected_pair_count_peak),
        "cv14d_shadow_pair_count_peak": float(log.shadow_pair_count_peak),
        "cv14d_budget_last": float(log.budget_last),
        "cv14d_geom_headroom_last": float(log.geom_headroom_last),
        "cv14d_candidate_count_last": int(log.candidate_count_last),
        "cv14d_selected_pair_count_last": int(log.selected_pair_count_last),
        "cv14d_shadow_pair_count_last": int(log.shadow_pair_count_last),
        "cv14d_best_pair_last": log.best_pair_last,
        "cv14d_best_pair_score_last": float(log.best_pair_score_last),
    })
    return out
