"""Compatibility package for historical `cerebro_virtual.*` imports."""

from __future__ import annotations

import importlib
import sys


def _alias(name: str, target: str) -> None:
    mod = importlib.import_module(target)
    sys.modules[name] = mod


for pkg in ("body", "context", "development", "meso", "offline", "tissue", "world"):
    _alias(f"cerebro_virtual.{pkg}", pkg)

_alias("cerebro_virtual.state", "state")
_alias("cerebro_virtual.metrics", "metrics")
_alias("cerebro_virtual.serialization", "serialization")

from state import CerebroVirtualConfig, CerebroVirtualState, make_initial_state


class CerebroVirtualOrchestrator:  # lazy compatibility proxy
    def __new__(cls, *args, **kwargs):
        mod = importlib.import_module("orchestrator")
        impl = getattr(mod, "CerebroVirtualOrchestrator")
        return impl(*args, **kwargs)


__all__ = [
    "CerebroVirtualConfig",
    "CerebroVirtualState",
    "CerebroVirtualOrchestrator",
    "make_initial_state",
]
