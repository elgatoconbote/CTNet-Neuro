from __future__ import annotations

from cerebro_virtual.body.body_system import BodySystem
from cerebro_virtual.context.context_bank import ContextBank
from cerebro_virtual.development.development_engine import DevelopmentEngine
from cerebro_virtual.meso.meso_maps import MesoMaps
from cerebro_virtual.offline.sleep_cycle import SleepCycle
from cerebro_virtual.state import CerebroVirtualConfig, CerebroVirtualState
from cerebro_virtual.tissue.tissue_core import TissueCore
from cerebro_virtual.world.world_loop import WorldLoop


class CerebroVirtualOrchestrator:
    def __init__(self, config: CerebroVirtualConfig):
        self.config = config
        self.world = WorldLoop()
        self.body = BodySystem()
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
