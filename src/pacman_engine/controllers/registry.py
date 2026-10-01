"""Controller registry and specification dataclass."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pacman_engine.controllers.base import Controller

_REGISTRY: dict[str, Callable[..., Controller]] = {}


@dataclass(frozen=True)
class ControllerSpec:
    """Specification for creating a Controller instance with parameters.

    Attributes:
        name: Registered name of the controller.
        params: Keyword parameters forwarded to the controller constructor.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(f"ControllerSpec name must be a non-empty string, got {self.name!r}")
        if self.params is None:
            object.__setattr__(self, "params", {})
        elif not isinstance(self.params, dict):
            raise ValueError(
                f"ControllerSpec params must be a dict, got {type(self.params).__name__}"
            )
        else:
            object.__setattr__(self, "params", dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to a dictionary."""
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str) -> ControllerSpec:
        """Create a ControllerSpec from a dictionary or plain controller name string."""
        if isinstance(data, str):
            return cls(name=data)
        if isinstance(data, ControllerSpec):
            return data
        if not isinstance(data, dict):
            raise ValueError(f"Expected dict or str for ControllerSpec, got {type(data).__name__}")
        name = data.get("name")
        if name is None:
            raise ValueError(
                f"ControllerSpec dict must contain a 'name' key, got keys: {list(data.keys())}"
            )
        params = data.get("params", {})
        return cls(name=str(name), params=params)


def register_controller(
    name: str,
) -> Callable[[Callable[..., Controller]], Callable[..., Controller]]:
    """Decorator to register a controller class or factory under a given name.

    Args:
        name: Unique identifier for the controller.

    Returns:
        Decorator function.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Controller registration name must be a non-empty string, got {name!r}")

    normalized_name = name.strip()

    def decorator(cls_or_factory: Callable[..., Controller]) -> Callable[..., Controller]:
        _REGISTRY[normalized_name] = cls_or_factory
        return cls_or_factory

    return decorator


def list_controllers() -> list[str]:
    """Return a sorted list of registered controller names."""
    return sorted(_REGISTRY.keys())


def create_controller(name: str, **params: Any) -> Controller:
    """Instantiate a controller by registered name with the given parameters.

    Args:
        name: Registered name of the controller.
        **params: Parameters passed to the controller constructor.

    Returns:
        The instantiated Controller.

    Raises:
        ValueError: If the controller name is not registered.
        TypeError: If invalid parameters are passed to the controller constructor.
    """
    if not isinstance(name, str):
        raise ValueError(f"Controller name must be a string, got {type(name).__name__}")

    normalized_name = name.strip()
    if normalized_name not in _REGISTRY:
        available = ", ".join(list_controllers())
        raise ValueError(
            f"Unknown controller '{normalized_name}'. Registered controllers: [{available}]"
        )

    factory = _REGISTRY[normalized_name]

    # Validate parameters against constructor signature if inspectable
    try:
        sig = inspect.signature(factory)
        has_var_keyword = any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if not has_var_keyword:
            accepted_params = set(sig.parameters.keys())
            unexpected = set(params.keys()) - accepted_params
            if unexpected:
                raise TypeError(
                    f"Unexpected parameter(s) for controller '{normalized_name}': "
                    f"{sorted(unexpected)}. Accepted parameters: {sorted(accepted_params)}"
                )
    except (ValueError, TypeError) as e:
        if "Unexpected parameter" in str(e):
            raise
        # Signature inspection may not be supported for some callables; proceed to invocation

    try:
        return factory(**params)
    except TypeError as exc:
        raise TypeError(f"Failed to instantiate controller '{normalized_name}': {exc}") from exc
