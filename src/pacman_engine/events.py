"""Immutable typed game events emitted during simulation ticks."""

from __future__ import annotations

from dataclasses import dataclass

from pacman_engine.types import Position


@dataclass(frozen=True)
class GameEvent:
    """Base class for all tick-bound game events."""

    tick: int


@dataclass(frozen=True)
class PelletEaten(GameEvent):
    """Emitted when Pacman consumes a standard pellet."""

    pos: Position


@dataclass(frozen=True)
class PowerPelletEaten(GameEvent):
    """Emitted when Pacman consumes an energizer / power pellet."""

    pos: Position


@dataclass(frozen=True)
class GhostEaten(GameEvent):
    """Emitted when Pacman consumes a frightened ghost."""

    ghost_id: int
    pos: Position
    score: int
    combo_index: int


@dataclass(frozen=True)
class PacmanDied(GameEvent):
    """Emitted when a non-frightened ghost collides with Pacman."""

    ghost_id: int
    pos: Position
    lives_left: int


@dataclass(frozen=True)
class FrightenedStarted(GameEvent):
    """Emitted when ghosts become frightened (or timer resets)."""

    pass


@dataclass(frozen=True)
class FrightenedEnded(GameEvent):
    """Emitted when the frightened countdown reaches zero."""

    pass


@dataclass(frozen=True)
class GhostRespawned(GameEvent):
    """Emitted when an eaten ghost finishes its respawn delay."""

    ghost_id: int
    pos: Position


@dataclass(frozen=True)
class LevelCleared(GameEvent):
    """Emitted when all pellets and power pellets have been consumed."""

    pass


@dataclass(frozen=True)
class TimeLimitReached(GameEvent):
    """Emitted when the maximum step limit is reached."""

    pass


Event = (
    PelletEaten
    | PowerPelletEaten
    | GhostEaten
    | PacmanDied
    | FrightenedStarted
    | FrightenedEnded
    | GhostRespawned
    | LevelCleared
    | TimeLimitReached
)
