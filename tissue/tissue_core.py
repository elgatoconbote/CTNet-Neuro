from __future__ import annotations

from cerebro_virtual.state import CerebroVirtualState
from cerebro_virtual.tissue.ctnet_adapter import step_material, summarize_material


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class TissueCore:
    def step(self, state: CerebroVirtualState) -> None:
        tissue = state.tissue_state
        world = state.world_state
        body = state.body_state

        material = tissue.material_ref
        if material is None:
            raise RuntimeError("TissueState.material_ref es None. Falta injertar el tejido CTNet real.")

        summary = summarize_material(material)

        signed_drive = (
            0.72 * world.sensory_drive
            + 0.18 * body.proprioception
            + 0.10 * body.interoception
        )
        pulse_a = max(0.0, -signed_drive) + 0.10 * max(0.0, -body.motor_command)
        pulse_b = max(0.0, signed_drive) + 0.10 * max(0.0, body.motor_command)

        runtime = step_material(
            material,
            pulse_a=pulse_a,
            pulse_b=pulse_b,
            stress=body.stress,
            substeps=6,
        )

        inh_ratio = 0.0
        if summary["num_neurons"] > 0:
            inh_ratio = summary["num_inh"] / summary["num_neurons"]

        tissue.patch_a_mean = runtime["patch_a_mean"]
        tissue.patch_b_mean = runtime["patch_b_mean"]
        tissue.outside_mean = runtime["outside_mean"]
        tissue.patch_delta = runtime["patch_delta"]
        tissue.readout_a = runtime["readout_a"]
        tissue.readout_b = runtime["readout_b"]
        tissue.structural_signal = _clip(
            0.60 * runtime["patch_delta"] + 0.40 * runtime["tag_delta"],
            -1.0,
            1.0,
        )

        tissue.mean_excitation = _clip(
            0.70 * tissue.mean_excitation
            + 0.18 * runtime["mean_abs_a"]
            + 0.12 * abs(runtime["patch_delta"]),
            -1.0,
            1.0,
        )

        tissue.mean_inhibition = _clip(
            0.74 * tissue.mean_inhibition
            + 0.16 * inh_ratio
            + 0.16 * runtime["inh_activity"],
            0.0,
            1.0,
        )

        tissue.glia_modulation = _clip(
            0.78 * tissue.glia_modulation
            + 0.22 * runtime["glia_mean"],
            -1.0,
            1.0,
        )

        tissue.local_plasticity = _clip(
            0.84 * tissue.local_plasticity
            + 0.16 * runtime["tag_mean"],
            0.0,
            1.0,
        )

        tissue.energy = _clip(
            tissue.energy
            - 0.010 * runtime["mean_abs_a"]
            + 0.016 * runtime["glia_mean"]
            + 0.010 * max(0.0, 1.0 - body.stress),
            0.0,
            1.0,
        )
