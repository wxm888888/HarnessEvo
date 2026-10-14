from __future__ import annotations

from harness_evolution.agents.base import BaseClient


class ClientRegistry:
    def __init__(self) -> None:
        self._client_classes: dict[str, type[BaseClient]] = {}

    def register(self, client_name: str):
        """Decorator style registration: @registry.register('name')."""
        key = client_name.strip()
        if not key:
            raise ValueError("client_name must not be empty")

        def decorator(client_cls: type[BaseClient]) -> type[BaseClient]:
            if key in self._client_classes:
                raise ValueError(f"duplicate client registration: {key}")
            client_cls.client_name = key
            self._client_classes[key] = client_cls
            return client_cls

        return decorator

    def build(self, client_name: str, init_kwargs: dict | None = None) -> BaseClient:
        key = client_name.strip()
        if key not in self._client_classes:
            available = ", ".join(sorted(self._client_classes))
            raise ValueError(f"unknown client '{client_name}', available: {available}")
        kwargs = init_kwargs or {}
        try:
            return self._client_classes[key](**kwargs)
        except TypeError as exc:
            raise ValueError(f"failed to initialize client '{client_name}' with kwargs={kwargs}: {exc}") from exc

    def available_clients(self) -> list[str]:
        return sorted(self._client_classes.keys())


CLIENT_REGISTRY = ClientRegistry()

