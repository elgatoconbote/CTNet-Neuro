from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_get(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


@dataclass
class CV12Log:
    gate_peak: float = 0.0
    reciprocal_score_peak: float = 0.0
    active_lift_peak: float = 0.0
    pair_lift_peak: float = 0.0
    active_correction_peak: float = 0.0
    pair_correction_peak: float = 0.0
    active_slot_last: int = -1
    partner_last: int = -1
    alignment_before_last: float = 0.0
    alignment_after_last: float = 0.0


def _best_partner(ctx, active_idx: int) -> tuple[int, float]:
    n = len(ctx.hypotheses)
    compat = getattr(ctx, "compatibility_matrix", [])
    incomp = getattr(ctx, "incompatibility_matrix", [])
    trans = getattr(ctx, "transition_matrix", [])
    gram = getattr(ctx, "transition_grammar_matrix", [])
    ret = getattr(ctx, "return_path_matrix", [])

    best_j = active_idx
    best_score = 0.0

    for j in range(n):
        if j == active_idx:
            continue

        comp = 0.5 * (_safe_get(compat, active_idx, j) + _safe_get(compat, j, active_idx))
        inc = 0.5 * (_safe_get(incomp, active_idx, j) + _safe_get(incomp, j, active_idx))
        trans_rec = min(_safe_get(trans, active_idx, j), _safe_get(trans, j, active_idx))
        gram_rec = min(_safe_get(gram, active_idx, j), _safe_get(gram, j, active_idx))
        ret_rec = min(_safe_get(ret, active_idx, j), _safe_get(ret, j, active_idx))

        score = _clip(
            0.24 * comp
            + 0.22 * trans_rec
            + 0.28 * gram_rec
            + 0.26 * ret_rec
            - 0.14 * inc,
            0.0,
            1.0,
        )
        if score > best_score:
            best_score = score
            best_j = j

    return best_j, best_score


def _align(target: float, mean_value: float) -> float:
    return _clip(1.0 - abs(target - mean_value), 0.0, 1.0)


def apply_cv12_bidirectional_alignment(state, log: CV12Log) -> None:
    ctx = state.context_state
    meso = state.meso_state
    body = state.body_state

    hyps = getattr(ctx, "hypotheses", [])
    if not hyps:
        return

    active_idx = int(getattr(ctx, "active_index", 0))
    active_idx = max(0, min(active_idx, len(hyps) - 1))
    partner_idx, reciprocal_score = _best_partner(ctx, active_idx)

    active = hyps[active_idx]
    partner = hyps[partner_idx]

    target = float(getattr(meso, "projection_readout", 0.0))

    active_mean_before = float(getattr(active, "mean", 0.0))
    partner_mean_before = float(getattr(partner, "mean", 0.0))

    active_align_before = _align(target, active_mean_before)
    partner_align_before = _align(target, partner_mean_before)

    overlap = float(getattr(ctx, "regime_overlap_mean", 0.5))
    specificity = float(getattr(ctx, "binding_specificity_mean", 0.0))
    row_min = float(getattr(ctx, "binding_row_min", 0.0))
    dead_frac = float(getattr(ctx, "dead_slot_fraction", 1.0))
    grammar = float(getattr(ctx, "grammar_trace", 0.0))
    ret = float(getattr(ctx, "return_trace", 0.0))

    geometry_gate = _clip(
        1.30 * (0.25 - overlap)
        + 1.10 * (specificity - 0.52)
        + 0.90 * (row_min - 0.77)
        + 0.25 * grammar
        + 0.25 * ret
        + 0.08 * (1.0 - dead_frac),
        0.0,
        1.0,
    )

    reciprocity_gate = _clip(
        2.5 * (reciprocal_score - 0.32)
        + 0.75 * max(0.0, grammar - 0.30)
        + 0.75 * max(0.0, ret - 0.19),
        0.0,
        1.0,
    )

    gate = geometry_gate * reciprocity_gate

    active_error = target - active_mean_before
    partner_error = target - partner_mean_before

    # Corrección asimétrica: la mayor parte la absorbe el activo.
    active_gain = _clip(
        0.030 * gate
        + 0.030 * gate * reciprocal_score
        + 0.018 * gate * grammar
        + 0.018 * gate * ret,
        0.0,
        0.085,
    )
    partner_gain = _clip(
        0.012 * gate
        + 0.015 * gate * reciprocal_score
        + 0.010 * gate * grammar
        + 0.010 * gate * ret,
        0.0,
        0.040,
    )

    active_correction = _clip(active_gain * active_error, -0.026, 0.026)
    partner_correction = _clip(partner_gain * partner_error, -0.012, 0.012)

    active.mean = _clip(active_mean_before + active_correction, -1.0, 1.0)
    if partner_idx != active_idx:
        partner.mean = _clip(partner_mean_before + partner_correction, -1.0, 1.0)

    # Ajuste proyectivo local y pequeño, también asimétrico.
    active_proj_before = float(getattr(active, "signature_projection", 0.0))
    partner_proj_before = float(getattr(partner, "signature_projection", 0.0))

    active_proj_shift = _clip(0.34 * active_correction, -0.012, 0.012)
    partner_proj_shift = _clip(0.22 * partner_correction, -0.006, 0.006)

    active.signature_projection = _clip(active_proj_before + active_proj_shift, -1.0, 1.0)
    if partner_idx != active_idx:
        partner.signature_projection = _clip(partner_proj_before + partner_proj_shift, -1.0, 1.0)

    # Microacople corporal mínimo, sin tocar bindings ni escorts.
    body_target = _clip(
        0.62 * float(getattr(body, "energy", 0.0))
        - 0.28 * float(getattr(body, "stress", 0.0))
        - 0.20 * float(getattr(body, "autonomic_load", 0.0)),
        -1.0,
        1.0,
    )

    active_body_before = float(getattr(active, "signature_body", 0.0))
    active.signature_body = _clip(
        active_body_before + _clip(0.006 * gate * (body_target - active_body_before), -0.004, 0.004),
        -1.0,
        1.0,
    )

    if partner_idx != active_idx:
        partner_body_before = float(getattr(partner, "signature_body", 0.0))
        partner.signature_body = _clip(
            partner_body_before + _clip(0.003 * gate * (body_target - partner_body_before), -0.002, 0.002),
            -1.0,
            1.0,
        )

    active_align_after = _align(target, float(active.mean))
    partner_align_after = _align(target, float(partner.mean))

    ctx.last_alignment = active_align_after

    active_conf_before = float(getattr(active, "confidence", 0.35))
    active_conf_delta = _clip(0.018 * max(0.0, active_align_after - active_align_before), 0.0, 0.006)
    active.confidence = _clip(active_conf_before + active_conf_delta, 0.05, 0.79)

    if partner_idx != active_idx:
        partner_conf_before = float(getattr(partner, "confidence", 0.35))
        partner_conf_delta = _clip(0.009 * max(0.0, partner_align_after - partner_align_before), 0.0, 0.003)
        partner.confidence = _clip(partner_conf_before + partner_conf_delta, 0.05, 0.79)

    log.gate_peak = max(log.gate_peak, gate)
    log.reciprocal_score_peak = max(log.reciprocal_score_peak, reciprocal_score)
    log.active_lift_peak = max(log.active_lift_peak, max(0.0, active_align_after - active_align_before))
    log.pair_lift_peak = max(log.pair_lift_peak, max(0.0, partner_align_after - partner_align_before))
    log.active_correction_peak = max(log.active_correction_peak, abs(active_correction))
    log.pair_correction_peak = max(log.pair_correction_peak, abs(partner_correction))
    log.active_slot_last = active_idx
    log.partner_last = partner_idx
    log.alignment_before_last = active_align_before
    log.alignment_after_last = active_align_after


def collect_cv12_summary(state, cv11_summary: dict, log: CV12Log) -> dict:
    out = dict(cv11_summary)
    out.update({
        "cv12_gate_peak": float(log.gate_peak),
        "cv12_reciprocal_score_peak": float(log.reciprocal_score_peak),
        "cv12_active_lift_peak": float(log.active_lift_peak),
        "cv12_pair_lift_peak": float(log.pair_lift_peak),
        "cv12_active_correction_peak": float(log.active_correction_peak),
        "cv12_pair_correction_peak": float(log.pair_correction_peak),
        "cv12_active_slot_last": int(log.active_slot_last),
        "cv12_partner_last": int(log.partner_last),
        "cv12_alignment_before_last": float(log.alignment_before_last),
        "cv12_alignment_after_last": float(log.alignment_after_last),
    })
    return out
