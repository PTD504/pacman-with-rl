"""Base controller protocol and interfaces."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from pacman_engine.types import Action

if TYPE_CHECKING:
    from pacman_engine.state import GameView


@runtime_checkable
class Controller(Protocol):
    """Protocol for entity decision-making controllers (Pacman and Ghosts).

    A controller must implement `act()`. It may optionally implement `reset()`
    which is invoked by the engine on `game.reset()`.
    """

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        """Select an action given the game view, entity ID, and seeded PRNG.

        Args:
            view: Read-only snapshot of current tick game state.
            entity_id: None for Pacman, or ghost ID (0-9).
            rng: Independent PRNG generator partitioned for this entity.

        Returns:
            The chosen Action.
        """
        ...


class BaseController:
    """Convenience base class providing default no-op reset."""

    def reset(self) -> None:
        """Reset internal controller state between episodes."""
        pass
