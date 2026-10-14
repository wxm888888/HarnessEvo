from __future__ import annotations

from harness_evolution.proposer.evolvers.base import BaseEvolver


class EvolverRegistry:
    def __init__(self) -> None:
        self._evolver_classes: dict[str, type[BaseEvolver]] = {}

    def register(self, evolver_name: str):
        """Decorator style registration: @registry.register('name')."""
        key = evolver_name.strip()
        if not key:
            raise ValueError("evolver_name must not be empty")

        def decorator(evolver_cls: type[BaseEvolver]) -> type[BaseEvolver]:
            if key in self._evolver_classes:
                raise ValueError(f"duplicate evolver registration: {key}")
            evolver_cls.evolver_name = key
            self._evolver_classes[key] = evolver_cls
            return evolver_cls

        return decorator

    def build(self, evolver_name: str, init_kwargs: dict | None = None) -> BaseEvolver:
        key = evolver_name.strip()
        if key not in self._evolver_classes:
            available = ", ".join(sorted(self._evolver_classes))
            raise ValueError(f"unknown evolver '{evolver_name}', available: {available}")
        kwargs = init_kwargs or {}
        try:
            return self._evolver_classes[key](**kwargs)
        except TypeError as exc:
            raise ValueError(f"failed to initialize evolver '{evolver_name}' with kwargs={kwargs}: {exc}") from exc

    def available_evolvers(self) -> list[str]:
        return sorted(self._evolver_classes.keys())


EVOLVER_REGISTRY = EvolverRegistry()

