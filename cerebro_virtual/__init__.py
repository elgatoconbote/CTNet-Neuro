from pathlib import Path

from .state import CerebroVirtualConfig, CerebroVirtualState, make_initial_state

# Compat namespace path: permite resolver `cerebro_virtual.<subpkg>` sobre layout histórico.
_pkg_root = Path(__file__).resolve().parent
_repo_root = _pkg_root.parent
__path__ = [str(_pkg_root), str(_repo_root)]

__all__ = [
    "CerebroVirtualConfig",
    "CerebroVirtualState",
    "CerebroVirtualOrchestrator",
    "make_initial_state",
]


def __getattr__(name: str):
    if name == "CerebroVirtualOrchestrator":
        from .orchestrator import CerebroVirtualOrchestrator
        return CerebroVirtualOrchestrator
    raise AttributeError(name)
