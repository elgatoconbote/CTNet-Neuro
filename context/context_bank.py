from __future__ import annotations

import math

from context.context_competition import softmax_scores
from context.structural_confirmation import is_structurally_confirmed
from state import CerebroVirtualState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _norm5(v0: float, v1: float, v2: float, v3: float, v4: float) -> tuple[float, float, float, float, float]:
    n = math.sqrt(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3 + v4 * v4)
    if n <= 1e-9:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    return v0 / n, v1 / n, v2 / n, v3 / n, v4 / n


def _dot5(a: tuple[float, float, float, float, float], b: tuple[float, float, float, float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3] + a[4] * b[4]


def _softmax_sharpen(scores: list[float], power: float = 1.7) -> list[float]:
    probs = softmax_scores(scores)
    shaped = [max(0.0, p) ** power for p in probs]
    total = sum(shaped) or 1.0
    return [x / total for x in shaped]


def _slot_anchor(slot: int) -> tuple[float, float, float, float, float]:
    anchors = [
        (1.00, 0.72, 0.55, -0.15, 0.18),
        (0.22, 0.88, 0.42, 1.00, -0.12),
        (0.10, 0.48, 0.82, 0.08, 1.00),
        (0.86, 0.56, 0.16, -0.78, 0.62),
    ]
    return _norm5(*anchors[slot % len(anchors)])


def _regime_mask(regime_idx: int) -> tuple[float, float, float, float, float]:
    masks = [
        (0.82, 0.76, 0.52, -0.08, 0.28),
        (0.34, 0.90, 0.66, 1.00, -0.22),
        (0.68, 0.62, 0.50, 0.24, 1.00),
        (1.00, 0.74, 0.52, -0.42, 0.40),
    ]
    return _norm5(*masks[regime_idx % len(masks)])


def _slot_prior(i: int, j: int) -> float:
    if i == j:
        return 1.0
    priors = [
        [1.00, 0.38, 0.16, 0.72],
        [0.38, 1.00, 0.72, 0.18],
        [0.16, 0.72, 1.00, 0.42],
        [0.72, 0.18, 0.42, 1.00],
    ]
    return priors[i % 4][j % 4]


def _slot_regime_prior(slot: int, regime_idx: int) -> float:
    priors = [
        [0.42, 0.32, 0.26, 0.44],
        [0.18, 0.44, 0.20, 0.16],
        [0.16, 0.20, 0.42, 0.22],
        [0.26, 0.16, 0.22, 0.46],
    ]
    return priors[slot % 4][regime_idx % 4]


def _binding_overlap(row_a: list[float], row_b: list[float]) -> float:
    if not row_a or not row_b:
        return 0.0
    return sum(a * b for a, b in zip(row_a, row_b)) / max(1, len(row_a))


def _norm_row(row: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in row))
    if n <= 1e-9:
        return [0.0 for _ in row]
    return [x / n for x in row]


def _pair_regime_overlap(context, i: int, j: int) -> float:
    ri = _norm_row(context.slot_regime_binding[i])
    rj = _norm_row(context.slot_regime_binding[j])
    return _clip(sum(a * b for a, b in zip(ri, rj)), 0.0, 1.0)


def _row_sum(row: list[float]) -> float:
    return sum(row) if row else 0.0


def _binding_specificity(row: list[float], r: int) -> float:
    if not row:
        return 0.0
    target = row[r]
    others = [row[k] for k in range(len(row)) if k != r]
    second = max(others) if others else 0.0
    return _clip(target - second, 0.0, 1.0)


def _row_best_gap(row: list[float]) -> float:
    if not row:
        return 0.0
    xs = sorted(row, reverse=True)
    if len(xs) < 2:
        return xs[0]
    return max(0.0, xs[0] - xs[1])


def _signature_regime_fit(sig: tuple[float, float, float, float, float], regime_idx: int) -> float:
    return _clip(0.5 * (1.0 + _dot5(sig, _regime_mask(regime_idx))), 0.0, 1.0)


def _vec_from_state(state: CerebroVirtualState) -> tuple[float, float, float, float, float]:
    tissue = state.tissue_state
    meso = state.meso_state
    body = state.body_state
    world = state.world_state

    projection = meso.projection_readout
    tissue_axis = 0.58 * tissue.mean_excitation + 0.42 * tissue.mean_inhibition
    structure_axis = 0.55 * tissue.structural_signal + 0.45 * tissue.local_plasticity
    salience_axis = 0.55 * meso.salience_map + 0.25 * body.pain + 0.20 * body.stress
    body_axis = (
        0.28 * body.interoception
        + 0.28 * body.proprioception
        + 0.22 * (2.0 * body.energy - 1.0)
        + 0.12 * world.homeostatic_drive
        - 0.10 * body.autonomic_load
    )
    return _norm5(projection, tissue_axis, structure_axis, salience_axis, body_axis)


def _scalar_from_signature(v: tuple[float, float, float, float, float]) -> float:
    return _clip(
        0.30 * v[0] + 0.24 * v[1] + 0.20 * v[2] + 0.14 * v[3] + 0.12 * v[4],
        -1.0,
        1.0,
    )


def _blended_target(
    base: tuple[float, float, float, float, float],
    slot: int,
    regime_idx: int,
    regime_affinity: list[float],
    return_bias: float,
    transition_credit: float,
) -> tuple[float, float, float, float, float]:
    anchor = _slot_anchor(slot)
    regime = _regime_mask(regime_idx)
    aff = regime_affinity[regime_idx] if regime_affinity else 0.25

    v = [0.0] * 5
    for k in range(5):
        v[k] = (
            0.44 * anchor[k]
            + (0.18 + 0.10 * aff) * regime[k]
            + 0.28 * base[k]
        )

    v[0] += 0.04 * transition_credit
    v[2] += 0.04 * transition_credit
    v[3] -= 0.03 * return_bias
    v[4] += 0.04 * return_bias

    return _norm5(v[0], v[1], v[2], v[3], v[4])


def _ensure_shapes(context) -> None:
    n = len(context.hypotheses)

    if len(context.transition_matrix) != n:
        context.transition_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    if len(context.compatibility_matrix) != n:
        context.compatibility_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    if len(context.incompatibility_matrix) != n:
        context.incompatibility_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    if len(context.transition_grammar_matrix) != n:
        context.transition_grammar_matrix = [[0.0 for _ in range(n)] for _ in range(n)]
    if len(context.return_path_matrix) != n:
        context.return_path_matrix = [[0.0 for _ in range(n)] for _ in range(n)]

    for hyp in context.hypotheses:
        if len(hyp.regime_affinity) == 0:
            hyp.regime_affinity = [0.25, 0.25, 0.25, 0.25]
        if len(hyp.predecessor_bias) != n:
            hyp.predecessor_bias = [0.0 for _ in range(n)]
        if len(hyp.successor_bias) != n:
            hyp.successor_bias = [0.0 for _ in range(n)]


def _family_neighborhood(context, i: int, j: int) -> float:
    hi = context.hypotheses[i]
    hj = context.hypotheses[j]

    si = _norm5(
        hi.signature_projection, hi.signature_tissue, hi.signature_structure, hi.signature_salience, hi.signature_body
    )
    sj = _norm5(
        hj.signature_projection, hj.signature_tissue, hj.signature_structure, hj.signature_salience, hj.signature_body
    )

    sig_sim = 0.5 * (1.0 + _dot5(si, sj))
    bind_ov = _binding_overlap(context.slot_regime_binding[i], context.slot_regime_binding[j])
    slot_p = _slot_prior(i, j)
    trans_hist = min(1.0, 0.5 * (context.transition_matrix[i][j] + context.transition_matrix[j][i]))
    ret_hist = min(1.0, 0.5 * (context.return_path_matrix[i][j] + context.return_path_matrix[j][i]))

    return _clip(
        0.34 * slot_p
        + 0.22 * sig_sim
        + 0.12 * bind_ov
        + 0.18 * trans_hist
        + 0.14 * ret_hist,
        0.0,
        1.0,
    )


def _typed_support(context, i: int, j: int, regime_idx: int) -> float:
    row_i = context.slot_regime_binding[i]
    row_j = context.slot_regime_binding[j]

    shared = min(row_i[regime_idx], row_j[regime_idx])
    spec_i = _binding_specificity(row_i, regime_idx)
    spec_j = _binding_specificity(row_j, regime_idx)

    return _clip(
        0.48 * shared + 0.26 * spec_i + 0.26 * spec_j,
        0.0,
        1.0,
    )


def _legitimacy(context, i: int, j: int, regime_idx: int) -> float:
    fam = _family_neighborhood(context, i, j)
    typ = _typed_support(context, i, j, regime_idx)
    predsucc = min(
        context.hypotheses[i].successor_bias[j],
        context.hypotheses[j].predecessor_bias[i],
    )
    grammar_pair = min(
        context.transition_grammar_matrix[i][j],
        context.transition_grammar_matrix[j][i],
    )
    return_pair = max(
        context.return_path_matrix[i][j],
        context.return_path_matrix[j][i],
    )

    overlap_pen = _clip(max(0.0, _pair_regime_overlap(context, i, j) - 0.78) / 0.22, 0.0, 1.0)
    low_spec_pen = _clip(max(0.0, 0.10 - min(_row_best_gap(context.slot_regime_binding[i]), _row_best_gap(context.slot_regime_binding[j]))) / 0.10, 0.0, 1.0)

    return _clip(
        0.28 * fam
        + 0.24 * typ
        + 0.14 * predsucc
        + 0.18 * grammar_pair
        + 0.10 * return_pair
        - 0.16 * overlap_pen
        - 0.12 * low_spec_pen,
        0.0,
        1.0,
    )


def _renormalize_bindings(context, active_idx: int, regime_idx: int) -> None:
    n = len(context.hypotheses)
    recent = set(context.recent_active_slots[-12:])

    # competencia por columna de régimen
    for r in range(4):
        floors = [0.022 + 0.070 * _slot_regime_prior(i, r) for i in range(n)]
        floor_sum = sum(floors)
        target_total = 0.94 if r == regime_idx else 0.78
        free_budget = max(0.0, target_total - floor_sum)

        logits = []
        for i, hyp in enumerate(context.hypotheses):
            sig = _norm5(
                hyp.signature_projection,
                hyp.signature_tissue,
                hyp.signature_structure,
                hyp.signature_salience,
                hyp.signature_body,
            )
            row_total = _row_sum(context.slot_regime_binding[i])
            hub_pen = max(0.0, row_total - 1.42)
            spec = _binding_specificity(context.slot_regime_binding[i], r)
            prior = _slot_regime_prior(i, r)
            sig_fit = _signature_regime_fit(sig, r)

            score = (
                1.10 * context.slot_regime_binding[i][r]
                + 1.05 * hyp.regime_affinity[r]
                + 0.90 * prior
                + 0.95 * sig_fit
                + 0.65 * spec
                + 0.12 * hyp.return_bias
                - 0.35 * hub_pen
            )

            if i == active_idx and r == regime_idx:
                score += 0.18
            elif i == active_idx and r != regime_idx:
                score -= 0.22

            if i not in recent and row_total < 0.82:
                score += 0.14 * prior

            logits.append(score)

        probs = _softmax_sharpen(logits, power=1.9)
        col = [floors[i] + free_budget * probs[i] for i in range(n)]

        for i in range(n):
            context.slot_regime_binding[i][r] = _clip(col[i], floors[i], 1.0)

    # anti-co-ocupación por fila
    for i in range(n):
        row = context.slot_regime_binding[i]
        sorted_r = sorted(range(4), key=lambda r: row[r], reverse=True)
        best_r, second_r = sorted_r[0], sorted_r[1]
        gap = row[best_r] - row[second_r]

        if gap < 0.12:
            shift = 0.025 + 0.18 * (0.12 - gap)
            floor_best = 0.022 + 0.070 * _slot_regime_prior(i, best_r)
            floor_second = 0.022 + 0.070 * _slot_regime_prior(i, second_r)
            take = min(shift, max(0.0, row[second_r] - floor_second))
            row[second_r] = _clip(row[second_r] - take, floor_second, 1.0)
            row[best_r] = _clip(row[best_r] + take, floor_best, 1.0)

        for r in range(4):
            if r == best_r:
                continue
            floor = 0.022 + 0.070 * _slot_regime_prior(i, r)
            if row[r] > 0.22 and row[best_r] > 0.24:
                damp = 0.10 * (row[r] - 0.22)
                row[r] = _clip(row[r] - damp, floor, 1.0)
                row[best_r] = _clip(
                    row[best_r] + 0.6 * damp,
                    0.022 + 0.070 * _slot_regime_prior(i, best_r),
                    1.0,
                )

    # competencia explícita dentro del régimen activo
    active_claim = context.slot_regime_binding[active_idx][regime_idx]
    for i in range(n):
        if i == active_idx:
            continue
        fam = _family_neighborhood(context, active_idx, i)
        legit = _legitimacy(context, active_idx, i, regime_idx)
        floor = 0.022 + 0.070 * _slot_regime_prior(i, regime_idx)
        penalty = 0.055 * max(0.0, active_claim - 0.26) * max(0.0, 1.0 - fam) * max(0.0, 1.0 - legit)
        if penalty > 0.0:
            context.slot_regime_binding[i][regime_idx] = _clip(
                context.slot_regime_binding[i][regime_idx] - penalty,
                floor,
                1.0,
            )

    # homeostasis de masa por fila con transferencia desde filas ricas
    row_totals = [_row_sum(row) for row in context.slot_regime_binding]
    poor = [i for i, s in enumerate(row_totals) if s < 0.72]
    rich = [i for i, s in enumerate(row_totals) if s > 0.86]

    for i in poor:
        row = context.slot_regime_binding[i]
        sorted_r = sorted(range(4), key=lambda r: row[r], reverse=True)
        primary_r, secondary_r = sorted_r[0], sorted_r[1]

        target_row = 0.82
        deficit = min(0.20, max(0.0, target_row - _row_sum(row)))
        if deficit <= 0.0:
            continue

        plans = [(primary_r, 0.68 * deficit), (secondary_r, 0.32 * deficit)]

        for r, need0 in plans:
            need = need0
            floor_i = 0.022 + 0.070 * _slot_regime_prior(i, r)

            donors = sorted(
                rich,
                key=lambda j: context.slot_regime_binding[j][r],
                reverse=True,
            )

            for j in donors:
                if j == i:
                    continue
                donor_floor = 0.022 + 0.070 * _slot_regime_prior(j, r)
                donor_keep = donor_floor + 0.12
                available = max(0.0, context.slot_regime_binding[j][r] - donor_keep)
                give = min(need, 0.60 * available)
                if give <= 0.0:
                    continue

                    context.slot_regime_binding[j][r] = _clip(
                        context.slot_regime_binding[j][r] - give,
                        donor_floor,
                        1.0,
                    )
                    row[r] = _clip(row[r] + give, floor_i, 1.0)
                    need -= give
                    if need <= 1e-9:
                        break

            if need > 1e-9:
                row[r] = _clip(row[r] + need, floor_i, 1.0)

    # drenaje suave de filas demasiado altas
    for i in range(n):
        row = context.slot_regime_binding[i]
        total = _row_sum(row)
        if total <= 0.98:
            continue

        excess = min(0.18, total - 0.88)
        sorted_r = sorted(range(4), key=lambda r: row[r], reverse=True)

        for r in sorted_r[1:]:
            floor = 0.022 + 0.070 * _slot_regime_prior(i, r)
            available = max(0.0, row[r] - floor)
            take = min(excess, 0.70 * available)
            row[r] = _clip(row[r] - take, floor, 1.0)
            excess -= take
            if excess <= 1e-9:
                break

        if excess > 1e-9:
            best_r = sorted_r[0]
            floor = 0.022 + 0.070 * _slot_regime_prior(i, best_r)
            row[best_r] = _clip(row[best_r] - excess, floor, 1.0)


def _reproject_columns_after_row_homeostasis(context, active_idx: int, regime_idx: int) -> None:
    n = len(context.hypotheses)

    target_cols = [0.78, 0.78, 0.78, 0.78]
    target_cols[regime_idx] = 0.94

    def cell_floor(i: int, r: int) -> float:
        return 0.022 + 0.070 * _slot_regime_prior(i, r)

    row_floor_totals = []
    for i in range(n):
        base = 0.80
        if i == active_idx:
            base = 0.84
        row_floor_totals.append(base)

    # 1) quitar exceso de columna sin romper el mínimo de fila
    for r in range(4):
        col_sum = sum(context.slot_regime_binding[i][r] for i in range(n))
        target = target_cols[r]
        excess = max(0.0, col_sum - target)
        if excess <= 1e-12:
            continue

        donors = list(range(n))
        donors.sort(
            key=lambda i: (
                context.slot_regime_binding[i][r] - cell_floor(i, r),
                _row_sum(context.slot_regime_binding[i]) - row_floor_totals[i],
                0.0 if (i == active_idx and r == regime_idx) else 1.0,
            ),
            reverse=True,
        )

        for i in donors:
            row = context.slot_regime_binding[i]
            floor_r = cell_floor(i, r)
            avail_cell = max(0.0, row[r] - floor_r)
            avail_row = max(0.0, _row_sum(row) - row_floor_totals[i])

            protect = 0.40 if (i == active_idx and r == regime_idx) else 1.0
            take = min(excess, 0.85 * avail_cell, protect * avail_row)
            if take <= 1e-12:
                continue

            row[r] = _clip(row[r] - take, floor_r, 1.0)
            excess -= take
            if excess <= 1e-12:
                break

    # 2) rellenar déficit de columna sin volver a sopa
    for r in range(4):
        col_sum = sum(context.slot_regime_binding[i][r] for i in range(n))
        target = target_cols[r]
        deficit = max(0.0, target - col_sum)
        if deficit <= 1e-12:
            continue

        logits = []
        for i, hyp in enumerate(context.hypotheses):
            sig = _norm5(
                hyp.signature_projection,
                hyp.signature_tissue,
                hyp.signature_structure,
                hyp.signature_salience,
                hyp.signature_body,
            )
            spec = _binding_specificity(context.slot_regime_binding[i], r)
            prior = _slot_regime_prior(i, r)
            sig_fit = _signature_regime_fit(sig, r)
            row_total = _row_sum(context.slot_regime_binding[i])
            room = max(0.0, 0.96 - row_total)
            score = (
                1.05 * prior
                + 0.95 * hyp.regime_affinity[r]
                + 0.80 * sig_fit
                + 0.60 * spec
                + 0.30 * room
            )
            if i == active_idx and r == regime_idx:
                score += 0.12
            logits.append(score)

        probs = _softmax_sharpen(logits, power=1.5)

        for i in range(n):
            if deficit <= 1e-12:
                break
            row = context.slot_regime_binding[i]
            floor_r = cell_floor(i, r)
            row_total = _row_sum(row)
            room = max(0.0, 0.96 - row_total)
            give = min(deficit, probs[i] * target, room)
            if give <= 1e-12:
                continue
            row[r] = _clip(row[r] + give, floor_r, 1.0)
            deficit -= give

def _prune_dominant_loops(context) -> None:
    n = len(context.hypotheses)
    pair_strengths = []

    def loop_strength(i: int, j: int) -> float:
        t = min(context.transition_matrix[i][j], context.transition_matrix[j][i])
        g = min(context.transition_grammar_matrix[i][j], context.transition_grammar_matrix[j][i])
        r = min(context.return_path_matrix[i][j], context.return_path_matrix[j][i])
        return 0.45 * t + 0.30 * g + 0.25 * r

    for i in range(n):
        for j in range(i + 1, n):
            pair_strengths.append(loop_strength(i, j))

    mean_loop = sum(pair_strengths) / len(pair_strengths) if pair_strengths else 0.0
    threshold = max(0.22, mean_loop + 0.10)

    for i in range(n):
        for j in range(i + 1, n):
            strength = loop_strength(i, j)
            if strength <= threshold:
                continue

            diversification = 0.0
            count = 0
            for k in range(n):
                if k in (i, j):
                    continue
                diversification += context.transition_matrix[i][k]
                diversification += context.transition_matrix[j][k]
                diversification += context.transition_matrix[k][i]
                diversification += context.transition_matrix[k][j]
                count += 4
            diversification = diversification / count if count else 0.0

            if diversification >= 0.18:
                continue

            damp = _clip(0.04 + 0.12 * (strength - threshold), 0.0, 0.16)

            for a, b in ((i, j), (j, i)):
                context.transition_matrix[a][b] = _clip(
                    context.transition_matrix[a][b] * (1.0 - damp),
                    0.0,
                    1.0,
                )
                context.transition_grammar_matrix[a][b] = _clip(
                    context.transition_grammar_matrix[a][b] * (1.0 - 0.85 * damp),
                    0.0,
                    1.0,
                )
                context.return_path_matrix[a][b] = _clip(
                    context.return_path_matrix[a][b] * (1.0 - 0.75 * damp),
                    0.0,
                    1.0,
                )


class ContextBank:
    def step(self, state: CerebroVirtualState) -> None:
        context = state.context_state
        _ensure_shapes(context)

        hypotheses = context.hypotheses
        n = len(hypotheses)
        world = state.world_state
        base = _vec_from_state(state)

        regime_idx = world.regime_index
        prev_active = context.active_index
        prev_regime = context.prev_regime_index

        for i in range(n):
            for r in range(4):
                floor = 0.022 + 0.070 * _slot_regime_prior(i, r)
                context.slot_regime_binding[i][r] = max(floor, context.slot_regime_binding[i][r])

        targets = []
        aligns = []
        scores = []

        for idx, hyp in enumerate(hypotheses):
            targets.append(
                _blended_target(
                    base=base,
                    slot=idx,
                    regime_idx=regime_idx,
                    regime_affinity=hyp.regime_affinity,
                    return_bias=hyp.return_bias,
                    transition_credit=hyp.transition_credit,
                )
            )

        regime_changed = prev_regime != regime_idx
        stagnation = _clip(max(0.0, context.active_duration - 10) / 14.0, 0.0, 1.0)

        row_sums_now = [_row_sum(row) for row in context.slot_regime_binding]
        seed_pulses = [0.0 for _ in range(n)]

        for idx in range(n):
            if idx == prev_active:
                continue

            fam = _family_neighborhood(context, prev_active, idx)
            legit = _legitimacy(context, prev_active, idx, regime_idx)
            regime_adv = context.slot_regime_binding[idx][regime_idx] - context.slot_regime_binding[prev_active][regime_idx]
            slot_low = max(0.0, 0.84 - row_sums_now[idx]) / 0.84

            seed = 0.0
            seed += 0.04 * max(0.0, fam - 0.16)
            seed += 0.05 * max(0.0, legit - 0.14)
            seed += 0.05 * slot_low
            seed += 0.04 * max(0.0, regime_adv)

            if regime_changed:
                seed += 0.05 * _slot_regime_prior(idx, regime_idx)

            if stagnation > 0.0:
                seed += 0.04 * stagnation * max(0.0, fam - 0.14)

            if idx in context.recent_active_slots[-24:-1]:
                seed += 0.04 * max(0.0, legit - 0.12)

            seed_pulses[idx] = _clip(seed, 0.0, 0.16)

        for idx, hyp in enumerate(hypotheses):
            sig = _norm5(
                hyp.signature_projection,
                hyp.signature_tissue,
                hyp.signature_structure,
                hyp.signature_salience,
                hyp.signature_body,
            )
            target = targets[idx]
            align = 0.5 * (1.0 + _dot5(sig, target))
            aligns.append(align)

            regime_fit = context.slot_regime_binding[idx][regime_idx]
            legit_fit = 0.0 if idx == prev_active else _legitimacy(context, prev_active, idx, regime_idx)
            family_fit = 0.0 if idx == prev_active else _family_neighborhood(context, prev_active, idx)
            row_total = _row_sum(context.slot_regime_binding[idx])
            hub_penalty = max(0.0, row_total - 1.42)
            starvation_bonus = max(0.0, 0.84 - row_total) / 0.84 if idx != prev_active else 0.0

            if idx == prev_active:
                spec_bonus = _row_best_gap(context.slot_regime_binding[idx])
                permanence_penalty = 0.08 * stagnation
                overlap_pen = 0.08 * sum(
                    max(0.0, _pair_regime_overlap(context, idx, j) - 0.82) / 0.18
                    for j in range(n) if j != idx
                ) / max(1, n - 1)
            else:
                spec_bonus = 0.0
                permanence_penalty = 0.0
                overlap_pen = 0.0

            score = (
                0.34 * align
                + 0.12 * hyp.confidence
                + 0.10 * hyp.structural_weight
                + 0.12 * regime_fit
                + 0.10 * legit_fit
                + 0.08 * family_fit
                + 0.05 * seed_pulses[idx]
                + 0.05 * starvation_bonus
                + 0.04 * spec_bonus
                - 0.05 * hyp.history_trace
                - 0.07 * hub_penalty
                - permanence_penalty
                - overlap_pen
            )
            scores.append(score)

        probs = softmax_scores(scores)
        entropy = 0.0
        for p in probs:
            if p > 0.0:
                entropy -= p * math.log(p)
        context.competition_entropy = entropy

        ranked = sorted(range(n), key=lambda i: probs[i], reverse=True)
        best_idx = ranked[0]
        challenger_idx = ranked[1] if len(ranked) > 1 else ranked[0]

        current_align = aligns[prev_active]
        current_regime_fit = context.slot_regime_binding[prev_active][regime_idx]
        challenger_legit = 0.0 if challenger_idx == prev_active else _legitimacy(context, prev_active, challenger_idx, regime_idx)
        challenger_family = 0.0 if challenger_idx == prev_active else _family_neighborhood(context, prev_active, challenger_idx)
        challenger_overlap = 0.0 if challenger_idx == prev_active else _pair_regime_overlap(context, prev_active, challenger_idx)

        illeg_vals = []
        for j in range(n):
            if j == prev_active:
                continue
            hi = hypotheses[prev_active]
            hj = hypotheses[j]

            si = _norm5(
                hi.signature_projection, hi.signature_tissue, hi.signature_structure, hi.signature_salience, hi.signature_body
            )
            sj = _norm5(
                hj.signature_projection, hj.signature_tissue, hj.signature_structure, hj.signature_salience, hj.signature_body
            )

            sim = 0.5 * (1.0 + _dot5(si, sj))
            ov = _binding_overlap(context.slot_regime_binding[prev_active], context.slot_regime_binding[j])
            comp = context.compatibility_matrix[prev_active][j]
            incomp = context.incompatibility_matrix[prev_active][j]
            fam = _family_neighborhood(context, prev_active, j)
            legit = _legitimacy(context, prev_active, j, regime_idx)
            pair_overlap = _pair_regime_overlap(context, prev_active, j)

            illeg = _clip(
                0.16 * max(0.0, sim - 0.88) / 0.12
                + 0.16 * max(0.0, ov - 0.62) / 0.38
                + 0.18 * max(0.0, pair_overlap - 0.80) / 0.20
                + 0.08 * max(0.0, comp - 0.62) / 0.38
                + 0.08 * stagnation
                - 0.16 * fam
                - 0.10 * incomp
                - 0.10 * current_align
                - 0.08 * current_regime_fit
                - 0.10 * legit,
                0.0,
                1.0,
            )
            illeg_vals.append(illeg)

        context.persistence_illegitimacy = sum(illeg_vals) / len(illeg_vals) if illeg_vals else 0.0
        context.best_challenger_margin = probs[challenger_idx] - probs[prev_active]
        context.challenger_index = challenger_idx

        should_switch = False
        if challenger_idx != prev_active:
            if challenger_legit > 0.16 and challenger_overlap < 0.88 and context.best_challenger_margin > 0.002:
                should_switch = True
            elif regime_changed and challenger_family > 0.16 and context.best_challenger_margin > -0.01:
                should_switch = True
            elif stagnation > 0.45 and challenger_legit > 0.14 and challenger_overlap < 0.90:
                should_switch = True
            elif context.persistence_illegitimacy > 0.10 and challenger_family > 0.14:
                should_switch = True

        if should_switch:
            context.active_index = challenger_idx
            context.active_duration = 1
        else:
            context.active_index = prev_active
            context.active_duration += 1

        active_idx = context.active_index
        active = hypotheses[active_idx]
        context.prev_active_index = prev_active
        context.last_alignment = aligns[active_idx]
        switched = active_idx != prev_active

        if switched:
            legit = _legitimacy(context, prev_active, active_idx, regime_idx)
            gain = _clip(
                0.09
                + 0.10 * max(0.0, context.best_challenger_margin)
                + 0.14 * challenger_family
                + 0.14 * legit
                + 0.08 * seed_pulses[active_idx]
                + 0.03 * world.regime_transition_signal,
                0.0,
                1.0,
            )

            context.transition_matrix[prev_active][active_idx] = _clip(
                0.984 * context.transition_matrix[prev_active][active_idx] + gain,
                0.0,
                1.0,
            )
            context.transition_trace = _clip(0.986 * context.transition_trace + 0.045 * gain, 0.0, 1.0)

            hypotheses[prev_active].successor_bias[active_idx] = _clip(
                0.984 * hypotheses[prev_active].successor_bias[active_idx] + 0.035 * gain,
                0.0,
                1.0,
            )
            active.predecessor_bias[prev_active] = _clip(
                0.984 * active.predecessor_bias[prev_active] + 0.035 * gain,
                0.0,
                1.0,
            )

            grammar_gain = _clip(
                0.18 * context.transition_matrix[prev_active][active_idx]
                + 0.16 * legit
                + 0.10 * min(
                    hypotheses[prev_active].successor_bias[active_idx],
                    active.predecessor_bias[prev_active],
                ),
                0.0,
                1.0,
            )
            context.transition_grammar_matrix[prev_active][active_idx] = _clip(
                0.992 * context.transition_grammar_matrix[prev_active][active_idx] + 0.028 * grammar_gain,
                0.0,
                1.0,
            )
            context.grammar_trace = _clip(0.989 * context.grammar_trace + 0.025 * grammar_gain, 0.0, 1.0)

            pair_overlap = _pair_regime_overlap(context, prev_active, active_idx)
            spec_pair = min(_row_best_gap(context.slot_regime_binding[prev_active]), _row_best_gap(context.slot_regime_binding[active_idx]))

            # retorno solo sobre pares recíprocos y no demasiado solapados
            if (
                active_idx in context.recent_active_slots[-28:-1]
                and legit > 0.16
                and pair_overlap < 0.86
                and spec_pair > 0.06
                and (
                    context.transition_matrix[prev_active][active_idx] > 0.12
                    or context.transition_matrix[active_idx][prev_active] > 0.12
                )
            ):
                ret_gain = _clip(
                    0.03
                    + 0.14 * legit
                    + 0.10 * min(
                        context.transition_grammar_matrix[prev_active][active_idx],
                        context.transition_grammar_matrix[active_idx][prev_active],
                    )
                    + 0.06 * spec_pair,
                    0.0,
                    1.0,
                )
                context.return_path_matrix[prev_active][active_idx] = _clip(
                    0.994 * context.return_path_matrix[prev_active][active_idx] + 0.015 * ret_gain,
                    0.0,
                    1.0,
                )
                context.return_path_matrix[active_idx][prev_active] = _clip(
                    0.997 * context.return_path_matrix[active_idx][prev_active] + 0.006 * ret_gain,
                    0.0,
                    1.0,
                )
                context.return_trace = _clip(0.992 * context.return_trace + 0.014 * ret_gain, 0.0, 1.0)
                active.return_bias = _clip(0.994 * active.return_bias + 0.008 * ret_gain, 0.0, 1.0)
            else:
                context.return_trace = _clip(0.996 * context.return_trace, 0.0, 1.0)
        else:
            provisional_idx = challenger_idx
            provisional_legit = 0.0 if provisional_idx == active_idx else _legitimacy(context, active_idx, provisional_idx, regime_idx)
            provisional_overlap = 0.0 if provisional_idx == active_idx else _pair_regime_overlap(context, active_idx, provisional_idx)

            # siembra fantasma, pero sin co-ocupación demasiado barata
            if provisional_idx != active_idx and provisional_legit > 0.12 and provisional_overlap < 0.88:
                ghost = _clip(
                    0.008 + 0.07 * provisional_legit + 0.05 * seed_pulses[provisional_idx],
                    0.0,
                    0.07,
                )
                context.transition_matrix[active_idx][provisional_idx] = _clip(
                    0.996 * context.transition_matrix[active_idx][provisional_idx] + ghost,
                    0.0,
                    1.0,
                )
                hypotheses[active_idx].successor_bias[provisional_idx] = _clip(
                    0.996 * hypotheses[active_idx].successor_bias[provisional_idx] + 0.018 * ghost,
                    0.0,
                    1.0,
                )
                hypotheses[provisional_idx].predecessor_bias[active_idx] = _clip(
                    0.996 * hypotheses[provisional_idx].predecessor_bias[active_idx] + 0.018 * ghost,
                    0.0,
                    1.0,
                )
                context.transition_trace = _clip(0.993 * context.transition_trace + 0.016 * ghost, 0.0, 1.0)

                if provisional_legit > 0.16:
                    gram_seed = _clip(0.005 + 0.04 * provisional_legit, 0.0, 0.03)
                    context.transition_grammar_matrix[active_idx][provisional_idx] = _clip(
                        0.997 * context.transition_grammar_matrix[active_idx][provisional_idx] + gram_seed,
                        0.0,
                        1.0,
                    )
                    context.grammar_trace = _clip(0.995 * context.grammar_trace + 0.010 * gram_seed, 0.0, 1.0)

                    if provisional_idx in context.recent_active_slots[-28:-1] and provisional_overlap < 0.84:
                        ret_seed = _clip(0.003 + 0.02 * provisional_legit, 0.0, 0.012)
                        context.return_path_matrix[active_idx][provisional_idx] = _clip(
                            0.998 * context.return_path_matrix[active_idx][provisional_idx] + ret_seed,
                            0.0,
                            1.0,
                        )
                        context.return_trace = _clip(0.996 * context.return_trace + 0.006 * ret_seed, 0.0, 1.0)
            else:
                context.transition_trace = _clip(0.996 * context.transition_trace, 0.0, 1.0)
                context.grammar_trace = _clip(0.996 * context.grammar_trace, 0.0, 1.0)
                context.return_trace = _clip(0.997 * context.return_trace, 0.0, 1.0)

        if prev_regime != regime_idx:
            context.regime_transition_matrix[prev_regime][regime_idx] = _clip(
                0.986 * context.regime_transition_matrix[prev_regime][regime_idx] + 0.035,
                0.0,
                1.0,
            )
        context.prev_regime_index = regime_idx

        for idx in range(n):
            for r in range(4):
                floor = 0.022 + 0.070 * _slot_regime_prior(idx, r)

                if idx == active_idx and r == regime_idx:
                    context.slot_regime_binding[idx][r] = _clip(
                        0.995 * context.slot_regime_binding[idx][r] + 0.012,
                        floor,
                        1.0,
                    )
                elif idx != active_idx and r == regime_idx:
                    context.slot_regime_binding[idx][r] = _clip(
                        0.998 * context.slot_regime_binding[idx][r] + 0.05 * seed_pulses[idx],
                        floor,
                        1.0,
                    )
                else:
                    context.slot_regime_binding[idx][r] = _clip(
                        0.999 * context.slot_regime_binding[idx][r],
                        floor,
                        1.0,
                    )

        context.recent_active_slots.append(active_idx)
        context.recent_active_slots = context.recent_active_slots[-28:]
        context.recent_regimes.append(regime_idx)
        context.recent_regimes = context.recent_regimes[-28:]

        _renormalize_bindings(context, active_idx, regime_idx)
        _reproject_columns_after_row_homeostasis(context, active_idx, regime_idx)
        _prune_dominant_loops(context)
        active_target = targets[active_idx]
        active_anchor = _slot_anchor(active_idx)

        for idx, hyp in enumerate(hypotheses):
            sig = [
                hyp.signature_projection,
                hyp.signature_tissue,
                hyp.signature_structure,
                hyp.signature_salience,
                hyp.signature_body,
            ]
            target = list(targets[idx])
            anchor = _slot_anchor(idx)

            row_total = _row_sum(context.slot_regime_binding[idx])
            hub_penalty = max(0.0, row_total - 1.42)

            if idx == active_idx:
                learn = 0.030 + 0.010 * context.slot_regime_binding[idx][regime_idx]
                for k in range(5):
                    desired = 0.62 * target[k] + 0.38 * anchor[k]
                    sig[k] = (1.0 - learn) * sig[k] + learn * desired

                normed = _norm5(sig[0], sig[1], sig[2], sig[3], sig[4])
                (
                    hyp.signature_projection,
                    hyp.signature_tissue,
                    hyp.signature_structure,
                    hyp.signature_salience,
                    hyp.signature_body,
                ) = normed

                scalar = _scalar_from_signature(normed)
                hyp.mean = _clip(0.960 * hyp.mean + 0.040 * scalar, -1.0, 1.0)

                align = aligns[idx]
                support_cap = _clip(
                    0.72
                    + 0.03 * min(1.0, context.grammar_trace / 0.16)
                    + 0.02 * min(1.0, context.return_trace / 0.06)
                    - 0.03 * hub_penalty
                    - 0.03 * stagnation,
                    0.70,
                    0.86,
                )
                hyp.confidence = _clip(
                    0.995 * hyp.confidence + 0.008 * align - 0.003 * context.persistence_illegitimacy,
                    0.05,
                    support_cap,
                )
                hyp.structural_weight = _clip(
                    0.995 * hyp.structural_weight + (0.005 if is_structurally_confirmed(align, hyp.confidence, threshold=0.82) else 0.002),
                    0.0,
                    1.0,
                )
                hyp.history_trace = _clip(
                    0.965 * hyp.history_trace + 0.010 * max(0.0, 1.0 - align),
                    0.0,
                    1.0,
                )
            else:
                ai = _norm5(sig[0], sig[1], sig[2], sig[3], sig[4])
                sim_to_active = 0.5 * (1.0 + _dot5(ai, active_target))
                fam = _family_neighborhood(context, idx, active_idx)
                overlap_pen = _clip(max(0.0, _pair_regime_overlap(context, idx, active_idx) - 0.80) / 0.20, 0.0, 1.0)
                repel = _clip(max(0.0, sim_to_active - 0.90) / 0.10, 0.0, 1.0)

                learn = 0.006 + 0.007 * context.slot_regime_binding[idx][regime_idx] + 0.010 * seed_pulses[idx]
                rest_target = _norm5(
                    0.82 * anchor[0] + 0.18 * target[0],
                    0.82 * anchor[1] + 0.18 * target[1],
                    0.82 * anchor[2] + 0.18 * target[2],
                    0.82 * anchor[3] + 0.18 * target[3],
                    0.82 * anchor[4] + 0.18 * target[4],
                )

                for k in range(5):
                    repulse = 0.008 * (repel + overlap_pen) * max(0.0, 1.0 - fam) * (anchor[k] - active_anchor[k])
                    sig[k] = 0.998 * sig[k] + learn * rest_target[k] + repulse

                normed = _norm5(sig[0], sig[1], sig[2], sig[3], sig[4])
                (
                    hyp.signature_projection,
                    hyp.signature_tissue,
                    hyp.signature_structure,
                    hyp.signature_salience,
                    hyp.signature_body,
                ) = normed

                scalar = _scalar_from_signature(normed)
                hyp.mean = _clip(0.995 * hyp.mean + 0.005 * scalar, -1.0, 1.0)
                hyp.confidence = _clip(
                    0.999 * hyp.confidence + 0.005 * seed_pulses[idx] + 0.003 * hyp.return_bias - 0.0012 * (repel + overlap_pen),
                    0.05,
                    0.82,
                )
                hyp.history_trace = _clip(
                    0.976 * hyp.history_trace + 0.007 * (repel + overlap_pen) * max(0.0, 1.0 - fam),
                    0.0,
                    1.0,
                )

            for r in range(4):
                base_aff = _slot_regime_prior(idx, r)
                if idx == active_idx and r == regime_idx:
                    hyp.regime_affinity[r] = _clip(0.995 * hyp.regime_affinity[r] + 0.010, base_aff, 1.0)
                elif idx != active_idx and r == regime_idx:
                    hyp.regime_affinity[r] = _clip(0.998 * hyp.regime_affinity[r] + 0.005 * seed_pulses[idx], base_aff, 1.0)
                else:
                    hyp.regime_affinity[r] = _clip(0.999 * hyp.regime_affinity[r], base_aff, 1.0)

        comp_vals = []
        incomp_vals = []

        for i in range(n):
            for j in range(i + 1, n):
                hi = hypotheses[i]
                hj = hypotheses[j]

                si = _norm5(
                    hi.signature_projection, hi.signature_tissue, hi.signature_structure, hi.signature_salience, hi.signature_body
                )
                sj = _norm5(
                    hj.signature_projection, hj.signature_tissue, hj.signature_structure, hj.signature_salience, hj.signature_body
                )
                sig_sim = 0.5 * (1.0 + _dot5(si, sj))
                bind_ov = _binding_overlap(context.slot_regime_binding[i], context.slot_regime_binding[j])
                slot_p = _slot_prior(i, j)
                fam = _family_neighborhood(context, i, j)
                legit = _legitimacy(context, i, j, regime_idx)
                pair_overlap = _pair_regime_overlap(context, i, j)

                comp = context.compatibility_matrix[i][j]
                incomp = context.incompatibility_matrix[i][j]

                target_comp = _clip(
                    0.12 + 0.14 * slot_p + 0.10 * fam + 0.08 * legit - 0.06 * max(0.0, pair_overlap - 0.80) / 0.20,
                    0.10,
                    0.46,
                )
                target_incomp = _clip(
                    0.16 + 0.08 * max(0.0, 1.0 - fam) + 0.10 * max(0.0, bind_ov - 0.58) / 0.42 + 0.08 * max(0.0, pair_overlap - 0.78) / 0.22,
                    0.16,
                    0.56,
                )

                new_comp = _clip(0.986 * comp + 0.018 * target_comp, 0.0, 1.0)
                new_incomp = _clip(0.986 * incomp + 0.018 * target_incomp, 0.0, 1.0)

                context.compatibility_matrix[i][j] = context.compatibility_matrix[j][i] = new_comp
                context.incompatibility_matrix[i][j] = context.incompatibility_matrix[j][i] = new_incomp

                comp_vals.append(new_comp)
                incomp_vals.append(new_incomp)

        comp_mean = sum(comp_vals) / len(comp_vals) if comp_vals else 0.0
        incomp_mean = sum(incomp_vals) / len(incomp_vals) if incomp_vals else 0.0

        context.compatibility_trace = _clip(0.982 * context.compatibility_trace + 0.025 * comp_mean, 0.0, 1.0)
        context.incompatibility_trace = _clip(0.982 * context.incompatibility_trace + 0.025 * incomp_mean, 0.0, 1.0)
