from __future__ import annotations


def tissue_snapshot(global_state) -> dict:
    t = getattr(global_state, "tissue_state", None)
    if t is None:
        return {"excitation": 0.5, "inhibition": 0.4, "glial_modulation": 0.2, "local_plasticity": 0.2, "energy": 0.8, "structural_signal": 0.2, "patch_A": 0.0, "patch_B": 0.0}
    return {
        "excitation": float(getattr(t, "mean_excitation", 0.5)),
        "inhibition": float(getattr(t, "mean_inhibition", 0.4)),
        "glial_modulation": float(getattr(t, "glia_modulation", 0.2)),
        "local_plasticity": float(getattr(t, "local_plasticity", 0.2)),
        "energy": float(getattr(t, "energy", 0.8)),
        "structural_signal": float(getattr(t, "structural_signal", 0.2)),
        "patch_A": float(getattr(t, "patch_a_mean", 0.0)),
        "patch_B": float(getattr(t, "patch_b_mean", 0.0)),
    }
