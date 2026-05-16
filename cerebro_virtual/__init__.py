from state import CerebroVirtualConfig, CerebroVirtualState, make_initial_state

__all__ = [
    "CerebroVirtualConfig",
    "CerebroVirtualState",
    "CerebroVirtualOrchestrator",
    "make_initial_state",
]


def __getattr__(name: str):
    if name == "CerebroVirtualOrchestrator":
        from ctnet_neuro.v7_neuro.orchestrator import CTNetV7NeuroOrchestrator as CerebroVirtualOrchestrator
        return CerebroVirtualOrchestrator
    raise AttributeError(name)
