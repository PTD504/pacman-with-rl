"""Ghost behavior controllers: GhostController base, chase, ambush, flank, patrol, and random."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pacman_engine.controllers.base import BaseController
from pacman_engine.controllers.registry import register_controller
from pacman_engine.types import Action, GhostMode, Position

if TYPE_CHECKING:
    from pacman_engine.state import GameView


class GhostController(BaseController):
    """Base class for ghost entity controllers handling behavioral modes.

    Subclasses define chase_target(view) to specify targeting logic during CHASE mode.
    Modes handled automatically:
    - SCATTER: Navigates toward scatter_target (or default corner for this ghost ID).
    - FRIGHTENED: Flees from Pacman (maximizing distance) or moves randomly.
    - CHASE: Navigates toward chase_target(view).

    Attributes:
        scatter_target: Optional target coordinate during SCATTER mode.
        frightened_behavior: 'flee' (maximizes distance from Pacman) or 'random'.
        allow_reverse: Whether reversing facing direction is permitted (except in dead ends).
        epsilon: Probability of choosing a random legal move instead of the optimal move.
        last_target: Last computed target position (exposed for visualization/debug).
    """

    def __init__(
        self,
        scatter_target: Position | tuple[int, int] | list[int] | None = None,
        frightened_behavior: Literal["flee", "random"] = "flee",
        allow_reverse: bool = False,
        epsilon: float = 0.0,
    ) -> None:
        if scatter_target is not None and not isinstance(scatter_target, Position):
            if isinstance(scatter_target, (tuple, list)) and len(scatter_target) == 2:
                scatter_target = Position(int(scatter_target[0]), int(scatter_target[1]))
            else:
                raise ValueError(f"Invalid scatter_target: {scatter_target!r}")

        if frightened_behavior not in ("flee", "random"):
            raise ValueError(
                f"frightened_behavior must be 'flee' or 'random', got {frightened_behavior!r}"
            )
        if not (0.0 <= epsilon <= 1.0):
            raise ValueError(f"epsilon must be between 0.0 and 1.0, got {epsilon}")

        self.scatter_target: Position | None = scatter_target
        self.frightened_behavior: Literal["flee", "random"] = frightened_behavior
        self.allow_reverse: bool = bool(allow_reverse)
        self.epsilon: float = float(epsilon)
        self.last_target: Position | None = None
        self._current_entity_id: int | None = None

    def reset(self) -> None:
        """Reset internal controller state."""
        self.last_target = None
        self._current_entity_id = None

    def chase_target(self, view: GameView) -> Position:
        """Determine target coordinate during CHASE mode.

        Must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement chase_target(view)")

    def _default_corner_target(self, view: GameView, entity_id: int | None) -> Position:
        """Determine default home corner for ghost ID (classic Pacman corners)."""
        rows = view.graph.grid_map.rows
        cols = view.graph.grid_map.cols
        gid = entity_id if entity_id is not None else 0
        match gid % 4:
            case 0:
                # Top-Right (Blinky)
                return Position(0, cols - 1)
            case 1:
                # Top-Left (Pinky)
                return Position(0, 0)
            case 2:
                # Bottom-Right (Inky)
                return Position(rows - 1, cols - 1)
            case 3:
                # Bottom-Left (Clyde)
                return Position(rows - 1, 0)
            case _:
                return Position(0, cols - 1)

    def _get_scatter_target(self, view: GameView, entity_id: int | None) -> Position:
        if self.scatter_target is not None:
            return self.scatter_target
        return self._default_corner_target(view, entity_id)

    def _filter_legal_moves(
        self,
        curr_pos: Position,
        curr_dir: Action,
        legal_neighbors: list[Position],
        view: GameView,
    ) -> list[tuple[Action, Position]]:
        """Compute legal moves, respecting allow_reverse unless in a dead end."""
        candidates: list[tuple[Action, Position]] = []
        for nbr in legal_neighbors:
            action = view.graph.action_between(curr_pos, nbr)
            candidates.append((action, nbr))

        if not candidates:
            return []

        if not self.allow_reverse and curr_dir != Action.STAY:
            rev_action = curr_dir.opposite
            non_reverse = [c for c in candidates if c[0] != rev_action]
            if non_reverse:
                return non_reverse

        return candidates

    def _navigate_towards_target(
        self,
        curr_pos: Position,
        target: Position,
        candidate_moves: list[tuple[Action, Position]],
        view: GameView,
        rng: np.random.Generator,
    ) -> Action:
        """Select action advancing toward target, or random if epsilon triggers."""
        if not candidate_moves:
            return Action.STAY

        # Epsilon exploration
        if self.epsilon > 0.0 and rng.random() < self.epsilon:
            idx = int(rng.integers(0, len(candidate_moves)))
            return candidate_moves[idx][0]

        # Calculate costs for candidates
        # If target is reachable on the graph, use BFS shortest path distance.
        # If target is in a wall or unreachable, fallback to Euclidean distance squared.
        bfs_dists = [view.graph.distance(nbr, target, for_ghost=True) for _, nbr in candidate_moves]
        has_reachable_bfs = any(d >= 0 for d in bfs_dists)

        order = {Action.UP: 0, Action.LEFT: 1, Action.DOWN: 2, Action.RIGHT: 3, Action.STAY: 4}

        scored_moves: list[tuple[float, int, Action]] = []
        for i, (action, nbr) in enumerate(candidate_moves):
            if has_reachable_bfs:
                cost = float(bfs_dists[i]) if bfs_dists[i] >= 0 else 100000.0
            else:
                cost = float((nbr.row - target.row) ** 2 + (nbr.col - target.col) ** 2)
            scored_moves.append((cost, order.get(action, 5), action))

        scored_moves.sort()
        return scored_moves[0][2]

    def _act_frightened(
        self,
        curr_pos: Position,
        candidate_moves: list[tuple[Action, Position]],
        view: GameView,
        rng: np.random.Generator,
    ) -> Action:
        """Handle movement during FRIGHTENED mode."""
        if not candidate_moves:
            return Action.STAY

        if self.frightened_behavior == "random":
            idx = int(rng.integers(0, len(candidate_moves)))
            return candidate_moves[idx][0]

        # "flee": maximize distance from Pacman
        p_pos = view.pacman_pos
        order = {Action.UP: 0, Action.LEFT: 1, Action.DOWN: 2, Action.RIGHT: 3, Action.STAY: 4}

        scored_moves: list[tuple[float, int, Action]] = []
        for action, nbr in candidate_moves:
            d = view.graph.distance(nbr, p_pos, for_ghost=True)
            if d < 0:
                d = (nbr.row - p_pos.row) ** 2 + (nbr.col - p_pos.col) ** 2
            # Negative d so sorting puts highest distance first
            scored_moves.append((-float(d), order.get(action, 5), action))

        scored_moves.sort()
        return scored_moves[0][2]

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        self._current_entity_id = entity_id

        if entity_id is not None:
            curr_pos = view.ghost_positions[entity_id]
            curr_dir = view.ghost_directions.get(entity_id, Action.STAY)
            curr_mode = view.ghost_modes.get(entity_id, GhostMode.CHASE)
        else:
            curr_pos = view.pacman_pos
            curr_dir = view.pacman_direction
            curr_mode = GhostMode.CHASE

        if curr_mode == GhostMode.DEAD:
            self.last_target = None
            return Action.STAY

        legal_neighbors = view.graph.neighbors(curr_pos, for_ghost=True)
        if not legal_neighbors:
            return Action.STAY

        candidate_moves = self._filter_legal_moves(curr_pos, curr_dir, legal_neighbors, view)
        if not candidate_moves:
            return Action.STAY

        if curr_mode == GhostMode.FRIGHTENED:
            self.last_target = None
            return self._act_frightened(curr_pos, candidate_moves, view, rng)

        if curr_mode == GhostMode.SCATTER:
            target = self._get_scatter_target(view, entity_id)
            self.last_target = target
            return self._navigate_towards_target(curr_pos, target, candidate_moves, view, rng)

        # CHASE mode
        target = self.chase_target(view)
        self.last_target = target
        return self._navigate_towards_target(curr_pos, target, candidate_moves, view, rng)


@register_controller("chase")
class ChaseGhostController(GhostController):
    """Ghost controller targeting Pacman's current position via shortest path."""

    def chase_target(self, view: GameView) -> Position:
        return view.pacman_pos


def _clamp_to_walkable(target: Position, origin: Position, view: GameView) -> Position:
    """Clamp coordinate to grid boundaries and find the closest walkable cell."""
    grid = view.graph.grid_map
    r = max(0, min(grid.rows - 1, target.row))
    c = max(0, min(grid.cols - 1, target.col))
    pos = Position(r, c)
    if view.graph._is_cell_walkable(pos, for_ghost=True):
        return pos

    # Step back along vector toward origin
    dr = 0 if r == origin.row else (1 if origin.row > r else -1)
    dc = 0 if c == origin.col else (1 if origin.col > c else -1)
    curr = pos
    while curr != origin:
        nr = curr.row + dr
        nc = curr.col + dc
        curr = Position(nr, nc)
        if view.graph._is_cell_walkable(curr, for_ghost=True):
            return curr

    # Fallback to closest walkable by Manhattan distance
    best_dist = float("inf")
    best_cell = origin
    for row in range(grid.rows):
        for col in range(grid.cols):
            cell = Position(row, col)
            if view.graph._is_cell_walkable(cell, for_ghost=True):
                dist = abs(cell.row - target.row) + abs(cell.col - target.col)
                if dist < best_dist:
                    best_dist = dist
                    best_cell = cell

    return best_cell


@register_controller("ambush")
class AmbushGhostController(GhostController):
    """Ghost controller targeting cells ahead of Pacman's current facing direction.

    Attributes:
        lookahead: Number of cells ahead of Pacman to target.
    """

    def __init__(
        self,
        lookahead: int = 4,
        scatter_target: Position | tuple[int, int] | list[int] | None = None,
        frightened_behavior: Literal["flee", "random"] = "flee",
        allow_reverse: bool = False,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            scatter_target=scatter_target,
            frightened_behavior=frightened_behavior,
            allow_reverse=allow_reverse,
            epsilon=epsilon,
        )
        if lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {lookahead}")
        self.lookahead: int = int(lookahead)

    def chase_target(self, view: GameView) -> Position:
        p_pos = view.pacman_pos
        p_dir = view.pacman_direction
        if p_dir == Action.STAY:
            return p_pos

        dr, dc = p_dir.delta
        raw_target = Position(p_pos.row + dr * self.lookahead, p_pos.col + dc * self.lookahead)
        return _clamp_to_walkable(raw_target, p_pos, view)


@register_controller("flank")
class FlankGhostController(GhostController):
    """Ghost controller targeting a cell mirrored through a partner ghost (Inky mechanic).

    Attributes:
        partner_id: ID of the partner ghost to mirror through.
    """

    def __init__(
        self,
        partner_id: int = 0,
        scatter_target: Position | tuple[int, int] | list[int] | None = None,
        frightened_behavior: Literal["flee", "random"] = "flee",
        allow_reverse: bool = False,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            scatter_target=scatter_target,
            frightened_behavior=frightened_behavior,
            allow_reverse=allow_reverse,
            epsilon=epsilon,
        )
        if not (0 <= partner_id <= 9):
            raise ValueError(f"partner_id must be between 0 and 9, got {partner_id}")
        self.partner_id: int = int(partner_id)

    def chase_target(self, view: GameView) -> Position:
        p_pos = view.pacman_pos
        partner_pos = view.ghost_positions.get(self.partner_id)
        if partner_pos is None:
            return p_pos

        # Mirrored target: vector from partner to Pacman doubled beyond Pacman
        # Target = Pacman + (Pacman - Partner) = 2 * Pacman - Partner
        target_row = 2 * p_pos.row - partner_pos.row
        target_col = 2 * p_pos.col - partner_pos.col
        raw_target = Position(target_row, target_col)
        return _clamp_to_walkable(raw_target, p_pos, view)


@register_controller("patrol")
class PatrolGhostController(GhostController):
    """Ghost controller sequentially visiting looping waypoints.

    Attributes:
        waypoints: List of coordinate waypoints to patrol in order.
    """

    def __init__(
        self,
        waypoints: list[Position | tuple[int, int] | list[int]],
        scatter_target: Position | tuple[int, int] | list[int] | None = None,
        frightened_behavior: Literal["flee", "random"] = "flee",
        allow_reverse: bool = False,
        epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            scatter_target=scatter_target,
            frightened_behavior=frightened_behavior,
            allow_reverse=allow_reverse,
            epsilon=epsilon,
        )
        if not waypoints:
            raise ValueError("waypoints list cannot be empty for PatrolGhostController")

        parsed: list[Position] = []
        for wp in waypoints:
            if isinstance(wp, Position):
                parsed.append(wp)
            elif isinstance(wp, (tuple, list)) and len(wp) == 2:
                parsed.append(Position(int(wp[0]), int(wp[1])))
            else:
                raise ValueError(f"Invalid waypoint: {wp!r}")

        self.waypoints: tuple[Position, ...] = tuple(parsed)
        self._waypoint_index: int = 0

    def reset(self) -> None:
        super().reset()
        self._waypoint_index = 0

    def chase_target(self, view: GameView) -> Position:
        if self._current_entity_id is not None:
            curr_pos = view.ghost_positions.get(self._current_entity_id)
            if curr_pos == self.waypoints[self._waypoint_index]:
                self._waypoint_index = (self._waypoint_index + 1) % len(self.waypoints)

        return self.waypoints[self._waypoint_index]


@register_controller("ghost_random")
class RandomGhostController(GhostController):
    """Ghost controller that makes random legal choices while respecting legal moves."""

    def __init__(
        self,
        scatter_target: Position | tuple[int, int] | list[int] | None = None,
        frightened_behavior: Literal["flee", "random"] = "random",
        allow_reverse: bool = False,
        epsilon: float = 1.0,
    ) -> None:
        super().__init__(
            scatter_target=scatter_target,
            frightened_behavior=frightened_behavior,
            allow_reverse=allow_reverse,
            epsilon=epsilon,
        )

    def chase_target(self, view: GameView) -> Position:
        return view.pacman_pos


@register_controller("random")
class RandomController(BaseController):
    """Unified random controller operating for both Pacman and Ghost entities."""

    def __init__(
        self,
        allow_reverse: bool = False,
        epsilon: float = 1.0,
        scatter_target: Position | tuple[int, int] | list[int] | None = None,
        frightened_behavior: Literal["flee", "random"] = "random",
    ) -> None:
        self.allow_reverse = allow_reverse
        self.epsilon = epsilon
        self.scatter_target = scatter_target
        self.frightened_behavior = frightened_behavior
        self.last_target: Position | None = None

        self._ghost_ctrl = RandomGhostController(
            scatter_target=scatter_target,
            frightened_behavior=frightened_behavior,
            allow_reverse=allow_reverse,
            epsilon=epsilon,
        )

    def reset(self) -> None:
        self.last_target = None
        self._ghost_ctrl.reset()

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        if entity_id is None:
            # Pacman random move
            pos = view.pacman_pos
            neighbors = view.graph.neighbors(pos, for_ghost=False)
            if not neighbors:
                return Action.STAY
            legal_actions = [view.graph.action_between(pos, nbr) for nbr in neighbors]
            idx = int(rng.integers(0, len(legal_actions)))
            return legal_actions[idx]

        # Ghost random move
        action = self._ghost_ctrl.act(view, entity_id, rng)
        self.last_target = self._ghost_ctrl.last_target
        return action
