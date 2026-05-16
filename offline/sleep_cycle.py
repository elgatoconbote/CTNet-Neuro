from __future__ import annotations

import math

from state import CerebroVirtualConfig, CerebroVirtualState


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _matrix_mean(m):
    vals = []
    for i in range(len(m)):
        for j in range(len(m[i])):
            if i == j:
                continue
            vals.append(m[i][j])
    return sum(vals) / len(vals) if vals else 0.0


class SleepCycle:
    def step(self, state: CerebroVirtualState, config: CerebroVirtualConfig) -> None:
        offline = state.offline_state
        meso = state.meso_state
        tissue = state.tissue_state
        body = state.body_state
        world = state.world_state
        ctx = state.context_state
        dev = state.development_state

        should_sleep = world.step_count % config.sleep_every == 0 and world.step_count > 0
        offline.asleep = should_sleep

        if should_sleep:
            offline.replay_buffer.append(meso.working_memory)
            offline.replay_buffer = offline.replay_buffer[-24:]

            t_mean = _matrix_mean(ctx.transition_matrix)
            g_mean = _matrix_mean(ctx.transition_grammar_matrix)
            r_mean = _matrix_mean(ctx.return_path_matrix)

            replay_strength = _clip(
                0.16 * ctx.transition_trace
                + 0.16 * ctx.grammar_trace
                + 0.18 * ctx.return_trace
                + 0.10 * max(0.0, tissue.structural_signal)
                + 0.08 * meso.working_memory,
                0.0,
                1.0,
            )

            consolidation_gain = _clip(
                0.14 * ctx.transition_trace
                + 0.16 * ctx.grammar_trace
                + 0.18 * ctx.return_trace
                + 0.10 * replay_strength,
                0.0,
                1.0,
            )

            # consolidación suave, no esterilizante
            ctx.transition_trace = _clip(0.985 * ctx.transition_trace + 0.03 * t_mean, 0.0, 1.0)
            ctx.grammar_trace = _clip(0.986 * ctx.grammar_trace + 0.03 * g_mean, 0.0, 1.0)
            ctx.return_trace = _clip(0.988 * ctx.return_trace + 0.025 * r_mean, 0.0, 1.0)

            offline.graph_replay_strength = replay_strength
            offline.graph_consolidation_gain = consolidation_gain
            offline.graph_pruning_signal = _clip(
                0.12 * max(0.0, t_mean - 0.55)
                + 0.10 * max(0.0, g_mean - 0.30)
                + 0.08 * max(0.0, r_mean - 0.16),
                0.0,
                1.0,
            )

            offline.restoration = _clip(offline.restoration + 0.10, 0.0, 1.0)
            offline.cleanup_load = _clip(offline.cleanup_load + 0.08, 0.0, 1.0)
            tissue.energy = _clip(tissue.energy + 0.12, 0.0, 1.0)
            body.energy = _clip(body.energy + 0.08, 0.0, 1.0)
            body.stress = _clip(body.stress - 0.05, 0.0, 1.0)
            meso.working_memory *= 0.92

            dev.grammar_stability = _clip(0.975 * dev.grammar_stability + 0.03 * ctx.grammar_trace, 0.0, 1.0)
            dev.return_stability = _clip(0.975 * dev.return_stability + 0.03 * ctx.return_trace, 0.0, 1.0)
        else:
            offline.restoration = _clip(offline.restoration * 0.95, 0.0, 1.0)
            offline.cleanup_load = _clip(offline.cleanup_load * 0.97, 0.0, 1.0)
            offline.graph_replay_strength = _clip(0.97 * offline.graph_replay_strength, 0.0, 1.0)
            offline.graph_consolidation_gain = _clip(0.97 * offline.graph_consolidation_gain, 0.0, 1.0)
            offline.graph_pruning_signal = _clip(0.97 * offline.graph_pruning_signal, 0.0, 1.0)
