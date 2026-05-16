from __future__ import annotations

from body.body_system import BodySystem
from context.context_bank import ContextBank
from development.development_engine import DevelopmentEngine
from meso.meso_maps import MesoMaps
from offline.sleep_cycle import SleepCycle
from state import CerebroVirtualConfig, CerebroVirtualState
from world.world_loop import WorldLoop


class CerebroVirtualOrchestrator:
    def __init__(self, config: CerebroVirtualConfig):
        self.config = config
        self.world = WorldLoop()
        self.body = BodySystem()
        from tissue.tissue_core import TissueCore
        self.tissue = TissueCore()
        self.meso = MesoMaps()
        self.context = ContextBank()
        self.development = DevelopmentEngine()
        self.offline = SleepCycle()

    def step(self, state: CerebroVirtualState) -> CerebroVirtualState:
        self.world.step(state)
        self.body.step(state)
        self.tissue.step(state)
        self.meso.step(state)
        self.context.step(state)
        self.body.apply_action(state)
        self.development.step(state)
        self.offline.step(state, self.config)
        return state

    def run(self, state: CerebroVirtualState, steps: int) -> CerebroVirtualState:
        for _ in range(steps):
            self.step(state)
        return state
