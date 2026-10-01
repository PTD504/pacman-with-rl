"""Graph representation of a GridMap with shortest paths, distances, and adjacency."""

from collections import deque

from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import Action, Position


class MapGraph:
    """Graph model of a GridMap supporting 4-neighbor adjacency, wrap-around, and cached BFS.

    Attributes:
        grid_map: The underlying immutable GridMap layout.
    """

    def __init__(self, grid_map: GridMap) -> None:
        self.grid_map = grid_map
        self._dist_cache: dict[tuple[Position, bool], dict[Position, int]] = {}
        self._parent_cache: dict[tuple[Position, bool], dict[Position, Position | None]] = {}

    def _is_cell_walkable(self, pos: Position, for_ghost: bool) -> bool:
        if not (0 <= pos.row < self.grid_map.rows and 0 <= pos.col < self.grid_map.cols):
            return False
        if self.grid_map.walls[pos.row, pos.col]:
            return False
        if self.grid_map.doors[pos.row, pos.col] and not for_ghost:
            return False
        return True

    def neighbors(self, pos: Position, for_ghost: bool = False) -> list[Position]:
        """Return accessible 4-neighbor coordinates honoring walls, doors, and edge wrap."""
        if not self._is_cell_walkable(pos, for_ghost):
            return []

        result: list[Position] = []
        # Consistent exploration order: UP, DOWN, LEFT, RIGHT
        for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT):
            dr, dc = action.delta
            nr = pos.row + dr
            nc = pos.col + dc

            # Horizontal wrap handling
            if self.grid_map.wrap_horizontal and dr == 0:
                if nc < 0:
                    nc = self.grid_map.cols - 1
                elif nc >= self.grid_map.cols:
                    nc = 0
            elif nc < 0 or nc >= self.grid_map.cols:
                continue

            # Vertical wrap handling
            if self.grid_map.wrap_vertical and dc == 0:
                if nr < 0:
                    nr = self.grid_map.rows - 1
                elif nr >= self.grid_map.rows:
                    nr = 0
            elif nr < 0 or nr >= self.grid_map.rows:
                continue

            target = Position(nr, nc)
            if self._is_cell_walkable(target, for_ghost):
                result.append(target)

        return result

    def _compute_bfs(self, src: Position, for_ghost: bool) -> None:
        dist: dict[Position, int] = {src: 0}
        parent: dict[Position, Position | None] = {src: None}

        if not self._is_cell_walkable(src, for_ghost):
            self._dist_cache[(src, for_ghost)] = dist
            self._parent_cache[(src, for_ghost)] = parent
            return

        queue: deque[Position] = deque([src])
        while queue:
            curr = queue.popleft()
            curr_d = dist[curr]
            for nxt in self.neighbors(curr, for_ghost=for_ghost):
                if nxt not in dist:
                    dist[nxt] = curr_d + 1
                    parent[nxt] = curr
                    queue.append(nxt)

        self._dist_cache[(src, for_ghost)] = dist
        self._parent_cache[(src, for_ghost)] = parent

    def distance(self, a: Position, b: Position, for_ghost: bool = False) -> int:
        """Calculate shortest distance in steps between two positions, or -1 if unreachable."""
        if a == b:
            return 0
        key = (a, for_ghost)
        if key not in self._dist_cache:
            self._compute_bfs(a, for_ghost)
        return self._dist_cache[key].get(b, -1)

    def shortest_path(
        self, a: Position, b: Position, for_ghost: bool = False
    ) -> list[Position] | None:
        """Find the shortest path from a to b (inclusive), or None if unreachable."""
        if a == b:
            return [a]
        key = (a, for_ghost)
        if key not in self._dist_cache:
            self._compute_bfs(a, for_ghost)
        if b not in self._dist_cache[key]:
            return None

        path: list[Position] = []
        curr: Position | None = b
        parent_map = self._parent_cache[key]
        while curr is not None:
            path.append(curr)
            if curr == a:
                break
            curr = parent_map[curr]

        path.reverse()
        return path

    def action_between(self, src: Position, dest: Position) -> Action:
        """Return the Action that transitions directly from adjacent src to dest."""
        if src == dest:
            return Action.STAY

        drow = dest.row - src.row
        dcol = dest.col - src.col

        # Horizontal wrap step
        if self.grid_map.wrap_horizontal and drow == 0:
            if src.col == 0 and dest.col == self.grid_map.cols - 1:
                return Action.LEFT
            if src.col == self.grid_map.cols - 1 and dest.col == 0:
                return Action.RIGHT

        # Vertical wrap step
        if self.grid_map.wrap_vertical and dcol == 0:
            if src.row == 0 and dest.row == self.grid_map.rows - 1:
                return Action.UP
            if src.row == self.grid_map.rows - 1 and dest.row == 0:
                return Action.DOWN

        return Action.from_delta(drow, dcol)

    def next_step_towards(self, a: Position, b: Position, for_ghost: bool = False) -> Action | None:
        """Determine the next Action to advance along the shortest path from a to b."""
        if a == b:
            return Action.STAY
        path = self.shortest_path(a, b, for_ghost=for_ghost)
        if path is None or len(path) < 2:
            return None
        return self.action_between(a, path[1])
