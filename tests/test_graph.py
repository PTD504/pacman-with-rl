"""Tests for MapGraph comparing distances with brute-force BFS, including wrap and doors."""

from collections import deque

import pytest

from pacman_engine.maps.builder import MapBuilder
from pacman_engine.maps.graph import MapGraph
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import Action, Position


def brute_force_bfs(gm: GridMap, start: Position, goal: Position, for_ghost: bool) -> int:
    """Independent reference BFS implementation for testing."""
    if start == goal:
        return 0

    def is_walkable(p: Position) -> bool:
        if not (0 <= p.row < gm.rows and 0 <= p.col < gm.cols):
            return False
        if gm.walls[p.row, p.col]:
            return False
        if gm.doors[p.row, p.col] and not for_ghost:
            return False
        return True

    if not is_walkable(start) or not is_walkable(goal):
        return -1

    visited: dict[Position, int] = {start: 0}
    q: deque[Position] = deque([start])

    while q:
        curr = q.popleft()
        d = visited[curr]
        if curr == goal:
            return d

        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = curr.row + dr, curr.col + dc
            if gm.wrap_horizontal and dr == 0:
                if nc < 0:
                    nc = gm.cols - 1
                elif nc >= gm.cols:
                    nc = 0
            elif nc < 0 or nc >= gm.cols:
                continue

            if gm.wrap_vertical and dc == 0:
                if nr < 0:
                    nr = gm.rows - 1
                elif nr >= gm.rows:
                    nr = 0
            elif nr < 0 or nr >= gm.rows:
                continue

            nxt = Position(nr, nc)
            if is_walkable(nxt) and nxt not in visited:
                visited[nxt] = d + 1
                q.append(nxt)

    return -1


def test_map_graph_distances_match_brute_force_bfs() -> None:
    """Compare all-pairs distances on a map containing walls, doors, and horizontal wrap."""
    ascii_map = (
        "; wrap_horizontal: true\n#######\n#P...o#\n#.###.#\n .0=1. \n#.###.#\n#o...o#\n#######\n"
    )
    gm = GridMap.from_ascii(ascii_map)
    graph = MapGraph(gm)

    # Check all pairs for both pacman and ghost
    for for_ghost in (False, True):
        for r1 in range(gm.rows):
            for c1 in range(gm.cols):
                p1 = Position(r1, c1)
                for r2 in range(gm.rows):
                    for c2 in range(gm.cols):
                        p2 = Position(r2, c2)
                        expected_dist = brute_force_bfs(gm, p1, p2, for_ghost=for_ghost)
                        actual_dist = graph.distance(p1, p2, for_ghost=for_ghost)
                        assert actual_dist == expected_dist, (
                            f"Mismatch between {p1} and {p2} (for_ghost={for_ghost}): "
                            f"expected {expected_dist}, got {actual_dist}"
                        )


def test_wrap_tunnel_distance_and_next_step() -> None:
    """Horizontal wrap makes opposite walkable edges 1 step away."""
    ascii_map = "; wrap_horizontal: true\n#####\n.P.o.\n#####\n"
    gm = GridMap.from_ascii(ascii_map)
    graph = MapGraph(gm)

    left_edge = Position(1, 0)
    right_edge = Position(1, 4)

    assert graph.distance(left_edge, right_edge) == 1
    assert graph.distance(right_edge, left_edge) == 1

    # Moving left from left_edge wraps directly to right_edge
    assert graph.next_step_towards(left_edge, right_edge) == Action.LEFT
    assert graph.next_step_towards(right_edge, left_edge) == Action.RIGHT


def test_vertical_wrap_distance_and_next_step() -> None:
    """Vertical wrap makes opposite walkable edges 1 step away."""
    ascii_map = "; wrap_vertical: true\n#.###\n#P..#\n#.0.#\n#o..#\n#.###\n"
    gm = GridMap.from_ascii(ascii_map)
    graph = MapGraph(gm)

    top_edge = Position(0, 1)
    bottom_edge = Position(4, 1)

    assert graph.distance(top_edge, bottom_edge) == 1
    assert graph.distance(bottom_edge, top_edge) == 1

    assert graph.next_step_towards(top_edge, bottom_edge) == Action.UP
    assert graph.next_step_towards(bottom_edge, top_edge) == Action.DOWN


def test_ghost_door_passage_differences() -> None:
    """Ghost can pass door while Pacman is blocked by door."""
    ascii_map = "#####\n#P#0#\n#.#=#\n#...#\n#####\n"
    gm = GridMap.from_ascii(ascii_map)
    graph = MapGraph(gm)

    pacman_spawn = Position(1, 1)
    ghost_spawn = Position(1, 3)
    door_pos = Position(2, 3)

    # Ghost can traverse door
    assert graph.distance(ghost_spawn, door_pos, for_ghost=True) == 1
    assert graph.shortest_path(ghost_spawn, door_pos, for_ghost=True) == [ghost_spawn, door_pos]

    # Pacman cannot enter door
    assert graph.distance(pacman_spawn, door_pos, for_ghost=False) == -1
    assert graph.shortest_path(pacman_spawn, door_pos, for_ghost=False) is None


def test_shortest_path_and_next_step_towards() -> None:
    ascii_map = "#####\n#P..#\n###.#\n#o0.#\n#####\n"
    gm = GridMap.from_ascii(ascii_map)
    graph = MapGraph(gm)

    start = Position(1, 1)
    goal = Position(3, 1)

    path = graph.shortest_path(start, goal)
    assert path == [
        Position(1, 1),
        Position(1, 2),
        Position(1, 3),
        Position(2, 3),
        Position(3, 3),
        Position(3, 2),
        Position(3, 1),
    ]

    assert graph.next_step_towards(start, goal) == Action.RIGHT
    assert graph.next_step_towards(start, start) == Action.STAY


def test_grid_map_immutability() -> None:
    """GridMap numpy arrays must be read-only."""
    gm = MapBuilder(5, 5).draw_border().set_pacman_spawn(1, 1).add_pellet(1, 2).build()
    with pytest.raises(ValueError, match="read-only"):
        gm.walls[0, 0] = False
    with pytest.raises(ValueError, match="read-only"):
        gm.doors[0, 0] = True
    with pytest.raises(ValueError, match="read-only"):
        gm.pellets[0, 0] = True
    with pytest.raises(ValueError, match="read-only"):
        gm.power_pellets[0, 0] = True
