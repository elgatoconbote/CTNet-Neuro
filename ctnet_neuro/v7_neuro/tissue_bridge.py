from __future__ import annotations


def project_tissue(tissue_state) -> dict[str, float]:
    return {
        "excitation": float(getattr(tissue_state, "mean_excitation", 0.0)),
        "inhibition": float(getattr(tissue_state, "mean_inhibition", 0.0)),
        "glial_modulation": float(getattr(tissue_state, "glia_modulation", 0.0)),
        "local_plasticity": float(getattr(tissue_state, "local_plasticity", 0.0)),
        "energy": float(getattr(tissue_state, "energy", 0.5)),
        "structural_signal": float(getattr(tissue_state, "structural_signal", 0.0)),
        "patch_A": float(getattr(tissue_state, "patch_a_mean", 0.0)),
        "patch_B": float(getattr(tissue_state, "patch_b_mean", 0.0)),
    }
