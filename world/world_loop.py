from __future__ import annotations

import math

from cerebro_virtual.state import CerebroVirtualState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class WorldLoop:
    def step(self, state: CerebroVirtualState) -> None:
        world = state.world_state
        body = state.body_state

        world.step_count += 1
        t = (world.step_count - 1) % 64

        if t < 16:
            regime_index = 0
            regime_label = "orient"
            novelty = 0.90
            threat = 0.10
            homeo = 0.20
            sensory = 0.25 + 0.35 * math.sin(world.step_count / 3.0) + 0.10 * body.proprioception
            affordance = 0.45 + 0.15 * math.cos(world.step_count / 5.0) + 0.05 * body.energy
            reward = 0.08 + 0.12 * affordance * max(0.0, body.motor_command) - 0.04 * body.stress
        elif t < 32:
            regime_index = 1
            regime_label = "threat"
            novelty = 0.35
            threat = 0.95
            homeo = 0.15
            sensory = 0.80 + 0.15 * math.sin(world.step_count / 2.0) - 0.05 * body.energy
            affordance = -0.20 + 0.10 * math.cos(world.step_count / 4.0) - 0.05 * body.stress
            reward = -0.28 * threat - 0.10 * abs(body.motor_command) + 0.03 * body.proprioception
        elif t < 48:
            regime_index = 2
            regime_label = "recover"
            novelty = 0.20
            threat = 0.05
            homeo = 0.95
            sensory = -0.10 + 0.15 * math.sin(world.step_count / 6.0) - 0.05 * body.stress
            affordance = 0.10 + 0.10 * body.energy - 0.08 * abs(body.motor_command)
            reward = 0.18 * homeo * (1.0 - abs(body.motor_command)) - 0.04 * body.stress
        else:
            regime_index = 3
            regime_label = "goal_recontext"
            novelty = 0.65
            threat = 0.20
            homeo = 0.45
            sensory = 0.35 + 0.30 * math.cos(world.step_count / 4.0) + 0.10 * body.proprioception
            affordance = 0.65 + 0.10 * math.sin(world.step_count / 5.0) + 0.05 * body.energy
            reward = 0.22 * affordance * max(0.0, body.motor_command) + 0.06 * novelty - 0.05 * body.stress

        transitioned = 1.0 if regime_index != world.regime_index else 0.0
        world.regime_transition_signal = _clip(
            0.75 * world.regime_transition_signal + 0.45 * transitioned,
            0.0,
            1.0,
        )

        world.regime_index = regime_index
        world.regime_label = regime_label
        world.novelty_drive = _clip(novelty, 0.0, 1.0)
        world.threat_drive = _clip(threat, 0.0, 1.0)
        world.homeostatic_drive = _clip(homeo, 0.0, 1.0)

        world.circadian_drive = 0.5 + 0.5 * math.sin(world.step_count / 8.0)
        world.sensory_drive = _clip(sensory, -1.0, 1.0)
        world.affordance = _clip(affordance, -1.0, 1.0)
        world.reward = _clip(reward, -1.0, 1.0)
