"""Core types, coordinate definitions, actions, and map validation exceptions."""

from __future__ import annotations

from enum import IntEnum
from typing import NamedTuple


class Position(NamedTuple):
    """Immutable coordinate in (row, col) grid format, origin at top-left."""

    row: int
    col: int

    def __add__(self, other: tuple[int, int] | Position) -> Position:  # type: ignore[override]
        if isinstance(other, (Position, tuple)) and len(other) == 2:
            return Position(self.row + other[0], self.col + other[1])
        return NotImplemented

    def __sub__(self, other: tuple[int, int] | Position) -> Position:
        if isinstance(other, (Position, tuple)) and len(other) == 2:
            return Position(self.row - other[0], self.col - other[1])
        return NotImplemented


class Action(IntEnum):
    """Discrete entity actions and their corresponding (row, col) movement deltas."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3
    STAY = 4

    @property
    def delta(self) -> tuple[int, int]:
        """Return the (row_delta, col_delta) vector for this action."""
        match self:
            case Action.UP:
                return (-1, 0)
            case Action.DOWN:
                return (1, 0)
            case Action.LEFT:
                return (0, -1)
            case Action.RIGHT:
                return (0, 1)
            case Action.STAY:
                return (0, 0)

    @property
    def opposite(self) -> Action:
        """Return the reversing action, or STAY if STAY."""
        match self:
            case Action.UP:
                return Action.DOWN
            case Action.DOWN:
                return Action.UP
            case Action.LEFT:
                return Action.RIGHT
            case Action.RIGHT:
                return Action.LEFT
            case Action.STAY:
                return Action.STAY

    @classmethod
    def from_delta(cls, drow: int, dcol: int) -> Action:
        """Convert a (drow, dcol) step delta to an Action."""
        match (drow, dcol):
            case (-1, 0):
                return Action.UP
            case (1, 0):
                return Action.DOWN
            case (0, -1):
                return Action.LEFT
            case (0, 1):
                return Action.RIGHT
            case (0, 0):
                return Action.STAY
            case _:
                raise ValueError(f"Invalid movement delta: ({drow}, {dcol})")


class MapValidationError(ValueError):
    """Raised when a map layout violates topological or game integrity constraints."""

    pass


class GhostMode(IntEnum):
    """Behavioral and lifecycle modes of a ghost entity."""

    SCATTER = 0
    CHASE = 1
    FRIGHTENED = 2
    DEAD = 3


class Outcome(IntEnum):
    """Game termination status."""

    RUNNING = 0
    WIN = 1
    LOSS = 2
    TIMEOUT = 3


class GameOverError(RuntimeError):
    """Raised when step() is called on an already terminated or truncated game."""

    pass


class InvalidActionError(ValueError):
    """Raised when an invalid action is taken and invalid_action_policy is 'raise'."""

    pass
