from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TissueState:
    neurons: int
    glia: int
    material_ref: Any | None = None
    mean_excitation: float = 0.0
    mean_inhibition: float = 0.0
    energy: float = 1.0
    glia_modulation: float = 0.0
    local_plasticity: float = 0.0
    patch_a_mean: float = 0.0
    patch_b_mean: float = 0.0
    outside_mean: float = 0.0
    patch_delta: float = 0.0
    readout_a: float = 0.0
    readout_b: float = 0.0
    structural_signal: float = 0.0


@dataclass
class MesoState:
    sensory_map: float = 0.0
    motor_map: float = 0.0
    salience_map: float = 0.0
    working_memory: float = 0.0
    projection_readout: float = 0.0


@dataclass
class ContextHypothesis:
    slot: int
    mean: float = 0.0
    confidence: float = 0.35
    structural_weight: float = 0.25
    history_trace: float = 0.0
    age: int = 0
    signature_projection: float = 0.0
    signature_tissue: float = 0.0
    signature_structure: float = 0.0
    signature_salience: float = 0.0
    signature_body: float = 0.0
    regime_affinity: list[float] = field(default_factory=list)
    predecessor_bias: list[float] = field(default_factory=list)
    successor_bias: list[float] = field(default_factory=list)
    return_bias: float = 0.0
    transition_credit: float = 0.0


@dataclass
class ContextState:
    hypotheses: list[ContextHypothesis] = field(default_factory=list)
    active_index: int = 0
    last_alignment: float = 0.0
    competition_entropy: float = 0.0

    prev_active_index: int = 0
    active_duration: int = 0
    transition_matrix: list[list[float]] = field(default_factory=list)
    compatibility_matrix: list[list[float]] = field(default_factory=list)
    incompatibility_matrix: list[list[float]] = field(default_factory=list)
    transition_trace: float = 0.0
    compatibility_trace: float = 0.0
    incompatibility_trace: float = 0.0

    persistence_illegitimacy: float = 0.0
    best_challenger_margin: float = 0.0
    challenger_index: int = 0

    prev_regime_index: int = 0
    regime_transition_matrix: list[list[float]] = field(default_factory=list)
    slot_regime_binding: list[list[float]] = field(default_factory=list)
    transition_grammar_matrix: list[list[float]] = field(default_factory=list)
    return_path_matrix: list[list[float]] = field(default_factory=list)
    recent_active_slots: list[int] = field(default_factory=list)
    recent_regimes: list[int] = field(default_factory=list)
    regime_trace: list[float] = field(default_factory=list)
    grammar_trace: float = 0.0
    return_trace: float = 0.0


@dataclass
class DevelopmentState:
    phase: str = "boot"
    progenitors: int = 0
    differentiation_progress: float = 0.0
    pruning_pressure: float = 0.0
    critical_period_open: bool = True
    developmental_memory: float = 0.0
    graph_maturation: float = 0.0
    graph_separation: float = 0.0
    grammar_stability: float = 0.0
    return_stability: float = 0.0
    closure_tension: float = 0.0



    @property
    def graph_closure_tension(self) -> float:
        return float(self.closure_tension)

    @graph_closure_tension.setter
    def graph_closure_tension(self, value: float) -> None:
        self.closure_tension = float(value)


@dataclass
class BodyState:
    interoception: float = 0.0
    proprioception: float = 0.0
    energy: float = 1.0
    stress: float = 0.0
    circadian_phase: float = 0.0
    motor_command: float = 0.0
    pain: float = 0.0
    autonomic_load: float = 0.0


@dataclass
class WorldState:
    step_count: int = 0
    sensory_drive: float = 0.0
    affordance: float = 0.0
    reward: float = 0.0
    circadian_drive: float = 0.5
    regime_label: str = "orient"
    regime_index: int = 0
    regime_transition_signal: float = 0.0
    homeostatic_drive: float = 0.0
    threat_drive: float = 0.0
    novelty_drive: float = 0.0


@dataclass
class OfflineState:
    asleep: bool = False
    replay_buffer: list[float] = field(default_factory=list)
    restoration: float = 0.0
    cleanup_load: float = 0.0
    graph_replay_strength: float = 0.0
    graph_consolidation_gain: float = 0.0
    graph_pruning_signal: float = 0.0


@dataclass
class CerebroVirtualState:
    tissue_state: TissueState
    meso_state: MesoState
    context_state: ContextState
    development_state: DevelopmentState
    body_state: BodyState
    world_state: WorldState
    offline_state: OfflineState


@dataclass
class CerebroVirtualConfig:
    dt: float = 1.0
    context_slots: int = 4
    regime_count: int = 4
    sleep_every: int = 16
    initial_neurons: int = 1422
    initial_glia: int = 1350
    initial_progenitors: int = 512


def _zeros_matrix(n: int) -> list[list[float]]:
    return [[0.0 for _ in range(n)] for _ in range(n)]


def _zeros_rect(r: int, c: int) -> list[list[float]]:
    return [[0.0 for _ in range(c)] for _ in range(r)]


def make_initial_state(config: CerebroVirtualConfig) -> CerebroVirtualState:
    context_slots = []
    for i in range(config.context_slots):
        context_slots.append(
            ContextHypothesis(
                slot=i,
                regime_affinity=[0.25 for _ in range(config.regime_count)],
                predecessor_bias=[0.0 for _ in range(config.context_slots)],
                successor_bias=[0.0 for _ in range(config.context_slots)],
            )
        )

    return CerebroVirtualState(
        tissue_state=TissueState(
            neurons=config.initial_neurons,
            glia=config.initial_glia,
        ),
        meso_state=MesoState(),
        context_state=ContextState(
            hypotheses=context_slots,
            active_index=0,
            prev_active_index=0,
            active_duration=0,
            prev_regime_index=0,
            transition_matrix=_zeros_matrix(config.context_slots),
            compatibility_matrix=_zeros_matrix(config.context_slots),
            incompatibility_matrix=_zeros_matrix(config.context_slots),
            regime_transition_matrix=_zeros_matrix(config.regime_count),
            slot_regime_binding=_zeros_rect(config.context_slots, config.regime_count),
            transition_grammar_matrix=_zeros_matrix(config.context_slots),
            return_path_matrix=_zeros_matrix(config.context_slots),
            recent_active_slots=[],
            recent_regimes=[],
            regime_trace=[0.0 for _ in range(config.regime_count)],
            challenger_index=0,
        ),
        development_state=DevelopmentState(
            progenitors=config.initial_progenitors,
        ),
        body_state=BodyState(),
        world_state=WorldState(),
        offline_state=OfflineState(),
    )
