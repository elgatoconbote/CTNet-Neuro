from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

try:
    from cerebro_virtual.experiments.cv12_bidirectional_reciprocal_alignment import (
        apply_cv12_bidirectional_alignment as _cv12_apply,
        collect_cv12_summary as _cv12_collect,
    )
except ImportError:
    from cerebro_virtual.experiments.cv12_bidirectional_reciprocal_alignment import (
        apply_cv12_supervisor as _cv12_apply,
        collect_cv12_summary as _cv12_collect,
    )


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def _getf(obj: Any, name: str, default: float = 0.0) -> float:
    try:
        return float(getattr(obj, name))
    except Exception:
        return float(default)


def _geti(obj: Any, name: str, default: int = 0) -> int:
    try:
        return int(getattr(obj, name))
    except Exception:
        return int(default)


def _setf(obj: Any, name: str, value: float) -> None:
    setattr(obj, name, float(value))


def _seti(obj: Any, name: str, value: int) -> None:
    setattr(obj, name, int(value))


def _peak(obj: Any, name: str, value: float) -> None:
    prev = _getf(obj, name, 0.0)
    if float(value) > prev:
        setattr(obj, name, float(value))


def _scalar(value: Any) -> bool:
    return isinstance(value, (int, float, str, bool)) or value is None


@dataclass
class CV14eLog:
    notes: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    def __getattr__(self, name: str):
        # CV12/CV11 usan acumuladores tipo *_peak, *_last, contadores e índices.
        # Si no existen aún, devolvemos un neutro seguro para evitar colapso de interfaz.
        if name.endswith("_peak") or name.endswith("_gain") or name.endswith("_score"):
            return 0.0
        if name.endswith("_count") or name.endswith("_idx") or name.endswith("_index"):
            return 0
        if name.endswith("_last"):
            return 0.0
        if name.startswith("best_pair") or name.endswith("_pair_last"):
            return "none"
        return 0.0

def apply_cv14e_bidirectional_alignment(state: Any, log: CV14eLog) -> None:
    # baseline geométrica-operativa válida: CV12
    _cv12_apply(state, log)

    alignment_before = _getf(state, "context_alignment", _getf(state, "graph_alignment", 0.0))
    grammar = _getf(state, "context_grammar_trace", _getf(state, "graph_grammar", 0.0))
    return_trace = _getf(state, "context_return_trace", _getf(state, "graph_return", 0.0))
    transition = _getf(state, "context_transition_trace", _getf(state, "graph_transition", 0.0))
    active_duration = _getf(state, "context_active_duration", 0.0)
    challenger_margin = _getf(state, "context_best_challenger_margin", 0.0)
    active_slot = _geti(state, "active_context_slot", -1)

    # pareja preferente heredada del baseline previo si existe
    partner = _geti(state, "cv12_partner_last", _geti(state, "cv11_partner_last", -1))

    # escritura online: pequeña, acotada, no saturante
    route_memory_prev = _getf(state, "cv14e_route_memory", 0.0)
    write_drive = _clip(0.58 * grammar + 0.42 * return_trace, 0.0, 1.0)
    write_gain = 0.00135 * write_drive
    route_memory = _clip(route_memory_prev * 0.994 + write_gain, 0.0, 0.18)

    # licencia local de uso
    permanence_gate = _clip((active_duration - 6.0) / 10.0, 0.0, 1.0)
    deficit_gate = _clip((0.985 - alignment_before) / 0.06, 0.0, 1.0)
    margin_gate = _clip((challenger_margin - 0.006) / 0.03, 0.0, 1.0)
    expression_license = _clip(
        0.46 * permanence_gate + 0.34 * deficit_gate + 0.20 * margin_gate,
        0.0,
        1.0,
    )

    # expresión diferida: separada de la escritura
    route_recall = _clip(route_memory * (0.55 + 0.45 * expression_license), 0.0, 1.0)
    express_gain = _clip(route_recall * expression_license * 0.014, 0.0, 0.0058)

    # poda local, no esterilizante
    unsupported_prune = _clip(
        (1.0 - expression_license) * max(0.0, transition - 0.48) * 0.010,
        0.0,
        0.008,
    )

    # corrección estrictamente local; no tocar bindings
    grammar_after = _clip(grammar + 0.95 * express_gain, 0.0, 1.0)
    return_after = _clip(return_trace + 1.10 * express_gain, 0.0, 1.0)
    transition_after = _clip(transition + 0.30 * express_gain - unsupported_prune, 0.0, 1.0)
    alignment_after = _clip(alignment_before + 0.14 * express_gain, 0.0, 1.0)

    active_mean = _getf(state, "active_context_mean", 0.0)
    active_mean_after = _clip(active_mean + 0.08 * express_gain, 0.0, 1.0)

    _setf(state, "cv14e_route_memory", route_memory)
    _setf(state, "cv14e_route_recall", route_recall)
    _setf(state, "cv14e_write_gain_last", write_gain)
    _setf(state, "cv14e_expression_license_last", expression_license)
    _setf(state, "cv14e_express_gain_last", express_gain)
    _setf(state, "cv14e_unsupported_prune_last", unsupported_prune)

    _setf(state, "context_grammar_trace", grammar_after)
    _setf(state, "context_return_trace", return_after)
    _setf(state, "context_transition_trace", transition_after)
    _setf(state, "context_alignment", alignment_after)
    _setf(state, "active_context_mean", active_mean_after)

    selected_pair_count = 1 if (expression_license > 0.05 and route_memory > 0.01 and partner >= 0) else 0
    shadow_pair_count = 0

    if active_slot >= 0 and partner >= 0:
        a, b = sorted((active_slot, partner))
        best_pair_last = f"{a}-{b}"
    else:
        best_pair_last = "none"

    best_pair_score_last = _clip(
        0.45 * grammar_after + 0.35 * return_after + 0.20 * max(expression_license, route_memory),
        0.0,
        1.0,
    )

    _peak(state, "cv14e_route_memory_peak", route_memory)
    _peak(state, "cv14e_write_gain_peak", write_gain)
    _peak(state, "cv14e_expression_license_peak", expression_license)
    _peak(state, "cv14e_express_gain_peak", express_gain)
    _peak(state, "cv14e_route_recall_peak", route_recall)
    _peak(state, "cv14e_unsupported_prune_peak", unsupported_prune)

    _peak(state, "cv14e_candidate_count_peak", 1.0 if partner >= 0 else 0.0)
    _peak(state, "cv14e_selected_pair_count_peak", float(selected_pair_count))
    _peak(state, "cv14e_shadow_pair_count_peak", float(shadow_pair_count))

    _setf(state, "cv14e_alignment_before_last", alignment_before)
    _setf(state, "cv14e_alignment_after_last", alignment_after)
    _seti(state, "cv14e_active_slot_last", active_slot)
    _seti(state, "cv14e_partner_last", partner)
    setattr(state, "cv14e_best_pair_last", best_pair_last)
    _setf(state, "cv14e_best_pair_score_last", best_pair_score_last)

    if hasattr(log, "metrics") and isinstance(log.metrics, dict):
        log.metrics["cv14e_route_memory_peak"] = _getf(state, "cv14e_route_memory_peak", 0.0)
        log.metrics["cv14e_write_gain_peak"] = _getf(state, "cv14e_write_gain_peak", 0.0)
        log.metrics["cv14e_expression_license_peak"] = _getf(state, "cv14e_expression_license_peak", 0.0)
        log.metrics["cv14e_express_gain_peak"] = _getf(state, "cv14e_express_gain_peak", 0.0)
        log.metrics["cv14e_route_recall_peak"] = _getf(state, "cv14e_route_recall_peak", 0.0)
        log.metrics["cv14e_unsupported_prune_peak"] = _getf(state, "cv14e_unsupported_prune_peak", 0.0)
        log.metrics["cv14e_selected_pair_count_peak"] = _getf(state, "cv14e_selected_pair_count_peak", 0.0)
        log.metrics["cv14e_best_pair_last"] = best_pair_last
        log.metrics["cv14e_best_pair_score_last"] = best_pair_score_last


def collect_cv14e_summary(state: Any, log: CV14eLog | None = None) -> dict[str, Any]:
    data: dict[str, Any] = {}

    try:
        data.update(_cv12_collect(state))
    except Exception:
        pass

    for k, v in vars(state).items():
        if str(k).startswith("cv14e_") and _scalar(v):
            data[k] = v

    if log is not None:
        metrics = getattr(log, "metrics", None)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if _scalar(v):
                    data[k] = v
        notes = getattr(log, "notes", None)
        if isinstance(notes, list):
            data["cv14e_notes_count"] = len(notes)

    return data

# ==== CV14e summary compatibility override ====
def collect_cv14e_summary(state: Any, baseline_summary: Any = None, log: CV14eLog | None = None) -> dict[str, Any]:
    def _keep(v: Any) -> bool:
        return isinstance(v, (int, float, str, bool, list, dict)) or v is None

    # compatibilidad con llamadas de 2 args: (state, log)
    if log is None and isinstance(baseline_summary, CV14eLog):
        log = baseline_summary
        baseline_summary = None

    data: dict[str, Any] = {}

    # si el runner pasa un summary previo (cv11/cv12), lo preservamos
    if isinstance(baseline_summary, dict):
        for k, v in baseline_summary.items():
            if _keep(v):
                data[k] = v
    else:
        try:
            base = _cv12_collect(state)
            if isinstance(base, dict):
                for k, v in base.items():
                    if _keep(v):
                        data[k] = v
        except Exception:
            pass

    # métricas propias de cv14e guardadas en state
    for k, v in vars(state).items():
        if str(k).startswith("cv14e_") and _keep(v):
            data[k] = v

    # métricas propias del log
    if log is not None:
        metrics = getattr(log, "metrics", None)
        if isinstance(metrics, dict):
            for k, v in metrics.items():
                if _keep(v):
                    data[k] = v

        notes = getattr(log, "notes", None)
        if isinstance(notes, list):
            data["cv14e_notes_count"] = len(notes)

    return data
