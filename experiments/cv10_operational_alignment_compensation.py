from __future__ import annotations

from dataclasses import dataclass


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _safe_attr(obj, name: str, default):
    return getattr(obj, name, default)


@dataclass
class CV10Log:
    alignment_lift_peak: float = 0.0
    active_mean_correction_peak: float = 0.0
    active_signature_projection_shift_peak: float = 0.0
    geometry_guard_peak: float = 0.0
    active_slot_last: int = -1
    active_alignment_before_last: float = 0.0
    active_alignment_after_last: float = 0.0


def apply_cv10_operational_alignment(state, log: CV10Log) -> None:
    ctx = state.context_state
    meso = state.meso_state
    body = state.body_state
    dev = state.development_state

    hyps = getattr(ctx, "hypotheses", [])
    if not hyps:
        return

    active_idx = int(getattr(ctx, "active_index", 0))
    active_idx = max(0, min(active_idx, len(hyps) - 1))
    hyp = hyps[active_idx]

    target = float(getattr(meso, "projection_readout", 0.0))
    mean_before = float(getattr(hyp, "mean", 0.0))
    align_before = _clip(1.0 - abs(target - mean_before), 0.0, 1.0)

    overlap = float(_safe_attr(ctx, "regime_overlap_mean", 0.5))
    specificity = float(_safe_attr(ctx, "binding_specificity_mean", 0.0))
    row_min = float(_safe_attr(ctx, "binding_row_min", 0.0))
    grammar = float(_safe_attr(ctx, "grammar_trace", 0.0))
    ret = float(_safe_attr(ctx, "return_trace", 0.0))
    closure_tension = float(_safe_attr(dev, "closure_tension", _safe_attr(dev, "graph_closure_tension", 0.0)))
    autonomic = float(getattr(body, "autonomic_load", 0.0))
    stress = float(getattr(body, "stress", 0.0))
    body_energy = float(getattr(body, "energy", 0.0))

    # Guardia geométrica: solo permitimos compensación fuerte si la geometría sigue sana.
    geometry_guard = _clip(
        1.6 * (0.34 - overlap)
        + 1.4 * (specificity - 0.38)
        + 0.8 * (row_min - 0.75)
        + 0.35 * grammar
        + 0.25 * ret
        - 0.35 * closure_tension,
        0.0,
        1.0,
    )

    # Señal operativa: si el cuerpo está tenso o bajo en energía, el acople debe ser más fino.
    body_drive = _clip(
        0.45 * (1.0 - body_energy)
        + 0.35 * stress
        + 0.20 * autonomic,
        0.0,
        1.0,
    )

    local_error = target - mean_before

    # Compensación puramente local del slot activo.
    gain = (
        0.06
        + 0.10 * geometry_guard
        + 0.05 * grammar
        + 0.05 * ret
        + 0.04 * body_drive
    )
    gain = _clip(gain, 0.0, 0.22)

    mean_correction = _clip(gain * local_error, -0.085, 0.085)
    hyp.mean = _clip(mean_before + mean_correction, -1.0, 1.0)

    # Ajuste proyectivo inmediato de la firma activa, sin tocar bindings.
    sig_proj_before = float(getattr(hyp, "signature_projection", 0.0))
    proj_shift = _clip(
        (0.55 * mean_correction + 0.18 * local_error * geometry_guard),
        -0.060,
        0.060,
    )
    hyp.signature_projection = _clip(sig_proj_before + proj_shift, -1.0, 1.0)

    # Ligera corrección corporal de la firma body, solo para evitar desacople activo-cuerpo.
    sig_body_before = float(getattr(hyp, "signature_body", 0.0))
    body_target = _clip(
        0.55 * body_energy
        - 0.35 * stress
        - 0.25 * autonomic,
        -1.0,
        1.0,
    )
    body_shift = _clip(
        0.03 * geometry_guard * (body_target - sig_body_before),
        -0.020,
        0.020,
    )
    hyp.signature_body = _clip(sig_body_before + body_shift, -1.0, 1.0)

    # Recomputar alignment local tras la compensación.
    align_after = _clip(1.0 - abs(target - float(hyp.mean)), 0.0, 1.0)
    ctx.last_alignment = align_after

    # Confianza: subir solo si mejora la alineación, sin monopolio.
    conf_before = float(getattr(hyp, "confidence", 0.35))
    conf_delta = _clip(0.06 * max(0.0, align_after - align_before), 0.0, 0.02)
    hyp.confidence = _clip(conf_before + conf_delta, 0.05, 0.79)

    log.alignment_lift_peak = max(log.alignment_lift_peak, max(0.0, align_after - align_before))
    log.active_mean_correction_peak = max(log.active_mean_correction_peak, abs(mean_correction))
    log.active_signature_projection_shift_peak = max(log.active_signature_projection_shift_peak, abs(proj_shift))
    log.geometry_guard_peak = max(log.geometry_guard_peak, geometry_guard)
    log.active_slot_last = active_idx
    log.active_alignment_before_last = align_before
    log.active_alignment_after_last = align_after


def collect_cv10_summary(state, cv9_summary: dict, log: CV10Log) -> dict:
    out = dict(cv9_summary)
    out.update({
        "cv10_alignment_lift_peak": float(log.alignment_lift_peak),
        "cv10_active_mean_correction_peak": float(log.active_mean_correction_peak),
        "cv10_active_signature_projection_shift_peak": float(log.active_signature_projection_shift_peak),
        "cv10_geometry_guard_peak": float(log.geometry_guard_peak),
        "cv10_active_slot_last": int(log.active_slot_last),
        "cv10_active_alignment_before_last": float(log.active_alignment_before_last),
        "cv10_active_alignment_after_last": float(log.active_alignment_after_last),
    })
    return out
