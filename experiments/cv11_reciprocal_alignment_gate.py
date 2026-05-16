from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_get(mat, i: int, j: int) -> float:
    try:
        return float(mat[i][j])
    except Exception:
        return 0.0


def _row_sum(row: list[float]) -> float:
    return float(sum(float(x) for x in row))


@dataclass
class CV11Log:
    reciprocal_gate_peak: float = 0.0
    reciprocal_score_peak: float = 0.0
    alignment_lift_peak: float = 0.0
    mean_correction_peak: float = 0.0
    proj_shift_peak: float = 0.0
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
            0.28 * comp
            + 0.22 * trans_rec
            + 0.28 * gram_rec
            + 0.22 * ret_rec
            - 0.18 * inc,
            0.0,
            1.0,
        )
        if score > best_score:
            best_score = score
            best_j = j

    return best_j, best_score


def apply_cv11_reciprocal_alignment(state, log: CV11Log) -> None:
    ctx = state.context_state
    meso = state.meso_state
    body = state.body_state

    hyps = getattr(ctx, "hypotheses", [])
    if not hyps:
        return

    active_idx = int(getattr(ctx, "active_index", 0))
    active_idx = max(0, min(active_idx, len(hyps) - 1))
    hyp = hyps[active_idx]

    partner_idx, reciprocal_score = _best_partner(ctx, active_idx)

    overlap = float(getattr(ctx, "regime_overlap_mean", 0.5))
    specificity = float(getattr(ctx, "binding_specificity_mean", 0.0))
    row_min = float(getattr(ctx, "binding_row_min", 0.0))
    grammar = float(getattr(ctx, "grammar_trace", 0.0))
    ret = float(getattr(ctx, "return_trace", 0.0))
    dead_frac = float(getattr(ctx, "dead_slot_fraction", 1.0))

    target = float(getattr(meso, "projection_readout", 0.0))
    mean_before = float(getattr(hyp, "mean", 0.0))
    align_before = _clip(1.0 - abs(target - mean_before), 0.0, 1.0)

    # Gate de legitimidad: solo corregimos si la geometría sigue sana Y si existe soporte recíproco real.
    geometry_gate = _clip(
        1.25 * (0.30 - overlap)
        + 1.00 * (specificity - 0.46)
        + 0.80 * (row_min - 0.76)
        + 0.25 * grammar
        + 0.25 * ret
        + 0.10 * (1.0 - dead_frac),
        0.0,
        1.0,
    )

    reciprocity_gate = _clip(
        2.2 * (reciprocal_score - 0.26)
        + 0.70 * max(0.0, grammar - 0.26)
        + 0.70 * max(0.0, ret - 0.07),
        0.0,
        1.0,
    )

    gate = geometry_gate * reciprocity_gate

    local_error = target - mean_before

    # Compensación pequeña y estrictamente local, soportada por par recíproco.
    gain = _clip(
        0.035 * gate
        + 0.045 * gate * reciprocal_score
        + 0.025 * gate * grammar
        + 0.020 * gate * ret,
        0.0,
        0.11,
    )

    mean_correction = _clip(gain * local_error, -0.030, 0.030)
    hyp.mean = _clip(mean_before + mean_correction, -1.0, 1.0)

    proj_before = float(getattr(hyp, "signature_projection", 0.0))
    proj_shift = _clip(
        0.40 * mean_correction + 0.015 * gate * local_error,
        -0.022,
        0.022,
    )
    hyp.signature_projection = _clip(proj_before + proj_shift, -1.0, 1.0)

    # Microacople corporal, pero muy contenido y solo si el gate está abierto.
    body_before = float(getattr(hyp, "signature_body", 0.0))
    body_target = _clip(
        0.60 * float(getattr(body, "energy", 0.0))
        - 0.30 * float(getattr(body, "stress", 0.0))
        - 0.20 * float(getattr(body, "autonomic_load", 0.0)),
        -1.0,
        1.0,
    )
    body_shift = _clip(0.010 * gate * (body_target - body_before), -0.008, 0.008)
    hyp.signature_body = _clip(body_before + body_shift, -1.0, 1.0)

    align_after = _clip(1.0 - abs(target - float(hyp.mean)), 0.0, 1.0)
    ctx.last_alignment = align_after

    conf_before = float(getattr(hyp, "confidence", 0.35))
    conf_delta = _clip(0.020 * max(0.0, align_after - align_before), 0.0, 0.008)
    hyp.confidence = _clip(conf_before + conf_delta, 0.05, 0.79)

    log.reciprocal_gate_peak = max(log.reciprocal_gate_peak, gate)
    log.reciprocal_score_peak = max(log.reciprocal_score_peak, reciprocal_score)
    log.alignment_lift_peak = max(log.alignment_lift_peak, max(0.0, align_after - align_before))
    log.mean_correction_peak = max(log.mean_correction_peak, abs(mean_correction))
    log.proj_shift_peak = max(log.proj_shift_peak, abs(proj_shift))
    log.active_slot_last = active_idx
    log.partner_last = partner_idx
    log.alignment_before_last = align_before
    log.alignment_after_last = align_after


def collect_cv11_summary(state, cv9_summary: dict, log: CV11Log) -> dict:
    out = dict(cv9_summary)
    out.update({
        "cv11_reciprocal_gate_peak": float(log.reciprocal_gate_peak),
        "cv11_reciprocal_score_peak": float(log.reciprocal_score_peak),
        "cv11_alignment_lift_peak": float(log.alignment_lift_peak),
        "cv11_mean_correction_peak": float(log.mean_correction_peak),
        "cv11_proj_shift_peak": float(log.proj_shift_peak),
        "cv11_active_slot_last": int(log.active_slot_last),
        "cv11_partner_last": int(log.partner_last),
        "cv11_alignment_before_last": float(log.alignment_before_last),
        "cv11_alignment_after_last": float(log.alignment_after_last),
    })
    return out
