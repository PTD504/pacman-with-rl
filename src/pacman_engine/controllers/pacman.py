"""Pacman behavior controllers: external, random, scripted, manual, and greedy pellet."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pacman_engine.controllers.base import BaseController
from pacman_engine.controllers.registry import register_controller
from pacman_engine.types import Action, GhostMode, Position

if TYPE_CHECKING:
    from pacman_engine.state import GameView


@register_controller("external")
class ExternalController(BaseController):
    """Marker controller indicating actions are provided externally to game.step().

    If act() is called directly, raises RuntimeError.
    """

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        raise RuntimeError(
            "ExternalController does not generate autonomous actions. "
            "Pass the action directly to PacmanGame.step(pacman_action)."
        )


@register_controller("pacman_random")
class RandomPacmanController(BaseController):
    """Controller that selects uniform random legal moves for Pacman.

    Never chooses actions leading into walls or door-blocked cells when legal moves exist.
    """

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        pos = view.pacman_pos
        neighbors = view.graph.neighbors(pos, for_ghost=False)
        if not neighbors:
            return Action.STAY

        legal_actions = [view.graph.action_between(pos, nbr) for nbr in neighbors]
        choice_idx = int(rng.integers(0, len(legal_actions)))
        return legal_actions[choice_idx]


@register_controller("scripted")
class ScriptedController(BaseController):
    """Controller executing a predefined sequence of actions with optional looping.

    Attributes:
        actions: Sequence of actions to execute sequentially.
        loop: Whether to loop back to the start when the sequence is exhausted.
    """

    def __init__(
        self,
        actions: list[Action | str] | tuple[Action | str, ...],
        loop: bool = False,
    ) -> None:
        parsed: list[Action] = []
        for a in actions:
            if isinstance(a, Action):
                parsed.append(a)
            elif isinstance(a, str):
                parsed.append(Action[a.strip().upper()])
            elif isinstance(a, int):
                parsed.append(Action(a))
            else:
                raise ValueError(f"Invalid action item in script: {a!r}")
        self.actions: tuple[Action, ...] = tuple(parsed)
        self.loop = bool(loop)
        self._index: int = 0

    def reset(self) -> None:
        """Reset sequence index back to the beginning."""
        self._index = 0

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        if not self.actions:
            return Action.STAY

        if self._index < len(self.actions):
            action = self.actions[self._index]
            self._index += 1
            return action

        if self.loop:
            self._index = 0
            action = self.actions[self._index]
            self._index += 1
            return action

        return Action.STAY


_REVERSE_ACTIONS = {
    Action.UP: Action.DOWN,
    Action.DOWN: Action.UP,
    Action.LEFT: Action.RIGHT,
    Action.RIGHT: Action.LEFT,
}


@register_controller("manual")
class ManualController(BaseController):
    """Classic arcade-style keyboard controller for Pacman with turn queueing and latching.

    Features:
    - Maintains current heading until a new legal direction is taken or blocked by wall.
    - Latches key-down events between ticks (most recent key wins).
    - Reversing direction takes effect immediately when legal.
    - Turning into a perpendicular corridor is queued and applied on the first tick where legal.
    """

    def __init__(self, initial_action: Action | str = Action.STAY) -> None:
        init_act = (
            Action[initial_action.strip().upper()]
            if isinstance(initial_action, str)
            else Action(initial_action)
        )
        self._current_heading: Action = init_act
        self._latched_action: Action | None = None
        self._queued_action: Action | None = None if init_act == Action.STAY else init_act

    def set_action(self, action: Action | str) -> None:
        """Latch the most recent key press arriving between or during ticks."""
        if isinstance(action, str):
            act = Action[action.strip().upper()]
        elif isinstance(action, (Action, int)):
            act = Action(action)
        else:
            raise ValueError(f"Invalid action for ManualController: {action!r}")
        self._latched_action = act

    def reset(self) -> None:
        """Reset held actions between episodes."""
        self._current_heading = Action.STAY
        self._latched_action = None
        self._queued_action = None

    def _is_legal(self, action: Action, view: GameView) -> bool:
        if action == Action.STAY:
            return True
        curr = view.pacman_pos
        for neighbor in view.graph.neighbors(curr, for_ghost=False):
            if view.graph.action_between(curr, neighbor) == action:
                return True
        return False

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        # 1. Process latched input from keys pressed since last tick
        if self._latched_action is not None:
            new_action = self._latched_action
            self._latched_action = None

            if new_action == Action.STAY:
                self._current_heading = Action.STAY
                self._queued_action = None
                return Action.STAY

            # Immediate reverse check
            is_reverse = (
                self._current_heading in _REVERSE_ACTIONS
                and new_action == _REVERSE_ACTIONS[self._current_heading]
            )

            if is_reverse and self._is_legal(new_action, view):
                self._current_heading = new_action
                self._queued_action = None
                return self._current_heading

            if self._is_legal(new_action, view):
                self._current_heading = new_action
                self._queued_action = None
                return self._current_heading
            else:
                self._queued_action = new_action

        # 2. Check if queued turn is now legal
        if self._queued_action is not None:
            if self._is_legal(self._queued_action, view):
                self._current_heading = self._queued_action
                self._queued_action = None
                return self._current_heading

        # 3. Maintain current heading if legal
        if self._current_heading != Action.STAY and self._is_legal(self._current_heading, view):
            return self._current_heading

        return Action.STAY


@register_controller("greedy_pellet")
class GreedyPelletController(BaseController):
    """Heuristic BFS controller navigating toward the nearest pellet or power pellet.

    Optionally evades dangerous ghosts within avoid_ghost_radius and hunts frightened ghosts.

    Attributes:
        avoid_ghost_radius: Distance within which dangerous ghosts are avoided.
        hunt_frightened: If True, prioritizes hunting frightened ghosts over pellets.
    """

    def __init__(
        self,
        avoid_ghost_radius: int = 0,
        hunt_frightened: bool = False,
    ) -> None:
        if avoid_ghost_radius < 0:
            raise ValueError(f"avoid_ghost_radius must be >= 0, got {avoid_ghost_radius}")
        self.avoid_ghost_radius = int(avoid_ghost_radius)
        self.hunt_frightened = bool(hunt_frightened)

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        curr = view.pacman_pos
        legal_neighbors = view.graph.neighbors(curr, for_ghost=False)
        if not legal_neighbors:
            return Action.STAY

        # 1. Identify dangerous ghosts (SCATTER or CHASE mode)
        dangerous_ghost_positions: list[Position] = [
            view.ghost_positions[gid]
            for gid, mode in view.ghost_modes.items()
            if mode in (GhostMode.SCATTER, GhostMode.CHASE) and gid in view.ghost_positions
        ]

        # 2. Check for huntable frightened ghosts if hunt_frightened is active
        target_pos: Position | None = None
        if self.hunt_frightened:
            frightened_targets = [
                view.ghost_positions[gid]
                for gid, mode in view.ghost_modes.items()
                if mode == GhostMode.FRIGHTENED and gid in view.ghost_positions
            ]
            if frightened_targets:
                best_d = float("inf")
                for fg in frightened_targets:
                    d = view.graph.distance(curr, fg, for_ghost=False)
                    if 0 <= d < best_d:
                        best_d = d
                        target_pos = fg

        # 3. If no frightened target, find nearest pellet / power pellet
        if target_pos is None:
            best_d = float("inf")
            # Collect pellet coordinates
            pellet_indices = np.argwhere(view.pellets | view.power_pellets)
            for r, c in pellet_indices:
                pos = Position(int(r), int(c))
                d = view.graph.distance(curr, pos, for_ghost=False)
                if 0 <= d < best_d:
                    best_d = d
                    target_pos = pos

        # If no target exists (e.g. all pellets cleared), default to staying or safe move
        if target_pos is None:
            return Action.STAY

        # 4. Evaluate legal moves, filtering or scoring based on ghost avoidance
        safe_moves: list[tuple[Action, Position, int]] = []
        # (action, pos, dist_to_target, min_ghost_dist)
        all_moves: list[tuple[Action, Position, int, int]] = []

        for nbr in legal_neighbors:
            action = view.graph.action_between(curr, nbr)
            d_target = view.graph.distance(nbr, target_pos, for_ghost=False)
            if d_target < 0:
                d_target = 100000

            min_ghost_dist = 100000
            for gp in dangerous_ghost_positions:
                gd = view.graph.distance(nbr, gp, for_ghost=False)
                if gd >= 0 and gd < min_ghost_dist:
                    min_ghost_dist = gd

            all_moves.append((action, nbr, d_target, min_ghost_dist))

            if min_ghost_dist > self.avoid_ghost_radius:
                safe_moves.append((action, nbr, d_target))

        # Tie-breaking priority order
        order = {Action.UP: 0, Action.LEFT: 1, Action.DOWN: 2, Action.RIGHT: 3, Action.STAY: 4}

        # If safe moves exist, choose safe move minimizing distance to target
        if safe_moves:
            safe_moves.sort(key=lambda item: (item[2], order.get(item[0], 5)))
            return safe_moves[0][0]

        # Pacman is trapped / in danger: maximize distance to nearest dangerous ghost
        all_moves.sort(key=lambda item: (-item[3], item[2], order.get(item[0], 5)))
        return all_moves[0][0]
