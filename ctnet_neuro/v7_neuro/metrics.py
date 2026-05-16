from __future__ import annotations


def summarize_step(step_info: dict) -> dict:
    return {
        "step": step_info["step"],
        "global_coherence": step_info["global_coherence"],
        "closure_error": step_info["closure_error"],
        "offline": step_info["offline"],
    }
