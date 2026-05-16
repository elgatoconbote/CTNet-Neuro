try:
    from cerebro_virtual.state import CerebroVirtualConfig, CerebroVirtualState, make_initial_state
except Exception:  # pragma: no cover
    from state import CerebroVirtualConfig, CerebroVirtualState, make_initial_state

try:
    from cerebro_virtual.orchestrator import CerebroVirtualOrchestrator
except Exception:  # pragma: no cover
    CerebroVirtualOrchestrator = None
