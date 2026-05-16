from __future__ import annotations

from cerebro_virtual.state import CerebroVirtualState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class BodySystem:
    def step(self, state: CerebroVirtualState) -> None:
        body = state.body_state
        world = state.world_state

        body.circadian_phase = world.circadian_drive

        body.interoception = _clip(
            0.82 * body.interoception
            + 0.18 * (
                0.60 * world.homeostatic_drive
                - 0.35 * world.threat_drive
                + 0.20 * world.reward
            ),
            -1.0,
            1.0,
        )

        body.proprioception = _clip(
            0.78 * body.proprioception
            + 0.14 * body.motor_command
            + 0.08 * world.affordance,
            -1.0,
            1.0,
        )

        body.energy = _clip(
            0.97 * body.energy
            + 0.035 * world.homeostatic_drive
            - 0.050 * abs(body.motor_command)
            - 0.030 * world.threat_drive,
            0.0,
            1.0,
        )

        body.stress = _clip(
            0.90 * body.stress
            + 0.16 * world.threat_drive
            + 0.08 * world.regime_transition_signal
            - 0.08 * world.homeostatic_drive
            - 0.04 * world.reward,
            0.0,
            1.0,
        )

        body.pain = _clip(
            0.88 * body.pain
            + 0.22 * world.threat_drive
            - 0.10 * world.homeostatic_drive,
            0.0,
            1.0,
        )

        body.autonomic_load = _clip(
            0.85 * body.autonomic_load
            + 0.20 * world.threat_drive
            + 0.10 * world.regime_transition_signal
            - 0.08 * world.homeostatic_drive,
            0.0,
            1.0,
        )

    def apply_action(self, state: CerebroVirtualState) -> None:
        body = state.body_state
        meso = state.meso_state
        context = state.context_state
        world = state.world_state

        gated_alignment = (2.0 * context.last_alignment) - 1.0
        body.motor_command = _clip(
            meso.motor_map
            + 0.20 * gated_alignment
            - 0.18 * world.threat_drive
            + 0.10 * world.homeostatic_drive,
            -1.0,
            1.0,
        )
