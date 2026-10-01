"""Base definitions, ObservationSpec, ObservationBuilder protocol, and registry."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from pacman_engine.config import RulesConfig
    from pacman_engine.maps.grid_map import GridMap
    from pacman_engine.state import GameView


@dataclass(frozen=True)
class ObservationSpec:
    """Specification describing the shape, data type, bounds, and features of an observation.

    Designed to be cleanly convertible to Gymnasium spaces (e.g., Box) in an external wrapper
    without introducing a direct dependency on Gymnasium.

    Attributes:
        shape: Dimensions of the observation tensor or vector.
        dtype: Data type of array elements (e.g. np.float32, np.uint8).
        low: Minimum element bounds (scalar or matching ndarray).
        high: Maximum element bounds (scalar or matching ndarray).
        feature_names: Optional human-readable labels corresponding to feature indices or channels.
    """

    shape: tuple[int, ...]
    dtype: np.dtype
    low: float | np.ndarray
    high: float | np.ndarray
    feature_names: list[str] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if not isinstance(self.dtype, np.dtype):
            object.__setattr__(self, "dtype", np.dtype(self.dtype))
        if self.feature_names is not None and not isinstance(self.feature_names, list):
            object.__setattr__(self, "feature_names", list(self.feature_names))


@runtime_checkable
class ObservationBuilder(Protocol):
    """Protocol defining the interface for state observation builders."""

    def spec(self, game_map: GridMap, rules: RulesConfig) -> ObservationSpec:
        """Derive the static ObservationSpec from map layout and game rules."""
        ...

    def build(self, view: GameView) -> np.ndarray:
        """Construct the observation array from a read-only GameView snapshot."""
        ...


@dataclass(frozen=True)
class ObservationConfig:
    """Configuration specification for an ObservationBuilder.

    Attributes:
        name: Registered name of the observation builder (e.g. 'vector', 'grid', 'rgb').
        params: Keyword parameters passed to the builder constructor.
    """

    name: str = "vector"
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError(
                f"ObservationConfig name must be a non-empty string, got {self.name!r}"
            )
        if self.params is None:
            object.__setattr__(self, "params", {})
        elif not isinstance(self.params, dict):
            raise ValueError(
                f"ObservationConfig params must be a dict, got {type(self.params).__name__}"
            )
        else:
            object.__setattr__(self, "params", dict(self.params))

    def to_dict(self) -> dict[str, Any]:
        """Convert specification to a dictionary."""
        return {"name": self.name, "params": dict(self.params)}

    @classmethod
    def from_dict(cls, data: dict[str, Any] | str | ObservationConfig) -> ObservationConfig:
        """Create an ObservationConfig from a dictionary, string, or existing config."""
        if isinstance(data, ObservationConfig):
            return data
        if isinstance(data, str):
            return cls(name=data)
        if not isinstance(data, dict):
            raise ValueError(
                f"Expected dict or str for ObservationConfig, got {type(data).__name__}"
            )
        name = data.get("name", "vector")
        params = data.get("params", {})
        return cls(name=str(name), params=params)


_OBSERVATION_REGISTRY: dict[str, Callable[..., ObservationBuilder]] = {}


def register_observation(
    name: str,
) -> Callable[[Callable[..., ObservationBuilder]], Callable[..., ObservationBuilder]]:
    """Decorator to register an observation builder class or factory under a given name.

    Args:
        name: Unique identifier for the observation builder.

    Returns:
        Decorator function.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"Observation registration name must be a non-empty string, got {name!r}")

    normalized_name = name.strip()

    def decorator(
        cls_or_factory: Callable[..., ObservationBuilder],
    ) -> Callable[..., ObservationBuilder]:
        _OBSERVATION_REGISTRY[normalized_name] = cls_or_factory
        return cls_or_factory

    return decorator


def list_observations() -> list[str]:
    """Return a sorted list of registered observation builder names."""
    return sorted(_OBSERVATION_REGISTRY.keys())


def create_observation(name: str, **params: Any) -> ObservationBuilder:
    """Instantiate an observation builder by registered name with the given parameters.

    Args:
        name: Registered name of the builder (e.g. 'vector', 'grid', 'rgb').
        **params: Parameters passed to the builder constructor.

    Returns:
        The instantiated ObservationBuilder.

    Raises:
        ValueError: If the observation builder name is not registered.
        TypeError: If invalid parameters are passed to the builder constructor.
    """
    if not isinstance(name, str):
        raise ValueError(f"Observation builder name must be a string, got {type(name).__name__}")

    normalized_name = name.strip()
    if normalized_name not in _OBSERVATION_REGISTRY:
        available = ", ".join(list_observations())
        raise ValueError(
            f"Unknown observation builder '{normalized_name}'. Registered builders: [{available}]"
        )

    factory = _OBSERVATION_REGISTRY[normalized_name]

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
                    f"Unexpected parameter(s) for observation builder '{normalized_name}': "
                    f"{sorted(unexpected)}. Accepted parameters: {sorted(accepted_params)}"
                )
    except (ValueError, TypeError) as e:
        if "Unexpected parameter" in str(e):
            raise

    try:
        return factory(**params)
    except TypeError as exc:
        raise TypeError(
            f"Failed to instantiate observation builder '{normalized_name}': {exc}"
        ) from exc
