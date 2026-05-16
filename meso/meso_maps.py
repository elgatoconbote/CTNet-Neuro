from __future__ import annotations

from state import CerebroVirtualState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class MesoMaps:
    def step(self, state: CerebroVirtualState) -> None:
        tissue = state.tissue_state
        body = state.body_state
        world = state.world_state
        meso = state.meso_state

        signed_salience_seed = (
            0.40 * world.threat_drive
            - 0.25 * world.homeostatic_drive
            + 0.20 * world.regime_transition_signal
        )

        meso.sensory_map = _clip(
            0.68 * meso.sensory_map
            + 0.38 * tissue.mean_excitation
            + 0.22 * world.sensory_drive
            + 0.12 * tissue.patch_delta
            - 0.10 * body.stress
            + 0.06 * world.novelty_drive,
            -1.0,
            1.0,
        )

        meso.salience_map = _clip(
            0.35 * abs(meso.sensory_map)
            + 0.22 * abs(world.reward)
            + 0.18 * body.stress
            + 0.15 * world.threat_drive
            + 0.10 * abs(tissue.patch_delta)
            + 0.08 * body.pain
            + 0.05 * signed_salience_seed,
            0.0,
            1.0,
        )

        meso.working_memory = _clip(
            0.85 * meso.working_memory
            + 0.16 * meso.sensory_map
            + 0.10 * tissue.structural_signal
            - 0.08 * body.stress
            + 0.06 * world.homeostatic_drive,
            -1.0,
            1.0,
        )

        meso.motor_map = _clip(
            0.40 * meso.working_memory
            + 0.25 * world.affordance
            - 0.20 * body.stress
            - 0.10 * world.threat_drive
            + 0.15 * body.proprioception
            + 0.10 * world.homeostatic_drive,
            -1.0,
            1.0,
        )

        meso.projection_readout = _clip(
            0.28 * meso.sensory_map
            + 0.20 * meso.working_memory
            + 0.10 * meso.motor_map
            + 0.18 * ((2.0 * meso.salience_map) - 1.0)
            + 0.18 * (tissue.readout_b - tissue.readout_a)
            + 0.12 * (world.homeostatic_drive - world.threat_drive),
            -1.0,
            1.0,
        )
