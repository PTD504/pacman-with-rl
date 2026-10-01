"""Internal mutable simulation state and read-only GameView for agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pacman_engine.types import Action, GhostMode, Position

if TYPE_CHECKING:
    from pacman_engine.maps.graph import MapGraph


@dataclass
class PacmanState:
    """Mutable runtime state of Pacman."""

    pos: Position
    direction: Action = Action.STAY


@dataclass
class GhostState:
    """Mutable runtime state of an individual ghost."""

    id: int
    pos: Position
    direction: Action = Action.STAY
    mode: GhostMode = GhostMode.SCATTER
    respawn_timer: int = 0
    start_delay_remaining: int = 0
    move_tick_counter: int = 0
    speed_accumulator: float = 0.0


class GameView:
    """Read-only snapshot of the game state for controllers and observation builders.

    Attributes:
        pacman_pos: Current position of Pacman.
        ghost_positions: Mapping of ghost ID to position.
        pacman_direction: Current direction of Pacman.
        ghost_directions: Mapping of ghost ID to direction.
        ghost_modes: Mapping of ghost ID to GhostMode.
        ghost_timers: Mapping of ghost ID to active countdown timer (respawn/frightened).
        frightened_timer: Global ticks remaining in frightened mode (0 if inactive).
        pellets: 2D boolean array of remaining pellets (write-protected).
        power_pellets: 2D boolean array of remaining power pellets (write-protected).
        lives: Remaining Pacman lives.
        score: Cumulative game score.
        tick: Elapsed game ticks (0-indexed).
        remaining_pellets: Total count of remaining pellets and power pellets.
        graph: Reference to the static MapGraph topology.
    """

    __slots__ = (
        "pacman_pos",
        "ghost_positions",
        "pacman_direction",
        "ghost_directions",
        "ghost_modes",
        "ghost_timers",
        "frightened_timer",
        "pellets",
        "power_pellets",
        "lives",
        "score",
        "tick",
        "remaining_pellets",
        "graph",
        "_positions",
        "_directions",
    )

    def __init__(
        self,
        pacman_pos: Position,
        ghost_positions: dict[int, Position],
        pacman_direction: Action,
        ghost_directions: dict[int, Action],
        ghost_modes: dict[int, GhostMode],
        ghost_timers: dict[int, int],
        frightened_timer: int,
        pellets: np.ndarray,
        power_pellets: np.ndarray,
        lives: int,
        score: int,
        tick: int,
        remaining_pellets: int,
        graph: MapGraph,
    ) -> None:
        self.pacman_pos = pacman_pos
        self.ghost_positions = ghost_positions
        self.pacman_direction = pacman_direction
        self.ghost_directions = ghost_directions
        self.ghost_modes = ghost_modes
        self.ghost_timers = ghost_timers
        self.frightened_timer = frightened_timer

        # Fast read-only numpy view
        p_view = pellets.view()
        p_view.flags.writeable = False
        self.pellets = p_view

        pp_view = power_pellets.view()
        pp_view.flags.writeable = False
        self.power_pellets = pp_view

        self.lives = lives
        self.score = score
        self.tick = tick
        self.remaining_pellets = remaining_pellets
        self.graph = graph

        self._positions: dict[int | None, Position] | None = None
        self._directions: dict[int | None, Action] | None = None

    @property
    def positions(self) -> dict[int | None, Position]:
        """Unified mapping of entity IDs to positions (None for Pacman, int for ghosts)."""
        if self._positions is None:
            res: dict[int | None, Position] = {None: self.pacman_pos}
            res.update(self.ghost_positions)
            self._positions = res
        return self._positions

    @property
    def directions(self) -> dict[int | None, Action]:
        """Unified mapping of entity IDs to directions (None for Pacman, int for ghosts)."""
        if self._directions is None:
            res: dict[int | None, Action] = {None: self.pacman_direction}
            res.update(self.ghost_directions)
            self._directions = res
        return self._directions

    def to_dict(self) -> dict[str, Any]:
        """Return plain dictionary with named entity coordinates and state features.

        Suitable for JSON serialization, debugging, and logging.
        """
        return {
            "pacman": {
                "row": self.pacman_pos.row,
                "col": self.pacman_pos.col,
                "direction": self.pacman_direction.name,
            },
            "ghosts": {
                gid: {
                    "row": pos.row,
                    "col": pos.col,
                    "direction": self.ghost_directions[gid].name,
                    "mode": self.ghost_modes[gid].name,
                    "timer": self.ghost_timers[gid],
                }
                for gid, pos in sorted(self.ghost_positions.items())
            },
            "frightened_timer": self.frightened_timer,
            "lives": self.lives,
            "score": self.score,
            "tick": self.tick,
            "remaining_pellets": self.remaining_pellets,
        }
