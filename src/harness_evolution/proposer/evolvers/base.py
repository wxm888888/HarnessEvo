from __future__ import annotations

from abc import ABC, abstractmethod

from harness_evolution.proposer.evolvers.protocol import EvolverRequest, EvolverResult


class BaseEvolver(ABC):
    evolver_name: str

    @abstractmethod
    def evolve(self, request: EvolverRequest) -> EvolverResult:
        """Apply one evolve method and return standardized result."""
