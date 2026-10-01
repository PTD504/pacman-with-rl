"""Procedural generator for seeded, valid Pacman maps with optional symmetry and loops."""

import numpy as np

from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import Position


class _DSU:
    def __init__(self, elements: list[tuple[int, int]]) -> None:
        self.parent = {elem: elem for elem in elements}

    def find(self, x: tuple[int, int]) -> tuple[int, int]:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: tuple[int, int], y: tuple[int, int]) -> bool:
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x == root_y:
            return False
        self.parent[root_x] = root_y
        return True


def generate_random_map(
    rows: int,
    cols: int,
    num_ghosts: int = 4,
    num_power_pellets: int = 4,
    loop_fraction: float = 0.2,
    symmetric: bool = True,
    seed: int | None = None,
) -> GridMap:
    """Generate a procedurally generated, fully valid Pacman map.

    Args:
        rows: Map height (must be odd and >= 7).
        cols: Map width (must be odd and >= 7).
        num_ghosts: Number of ghost spawns (0 to 10).
        num_power_pellets: Number of energizers/power pellets.
        loop_fraction: Fraction of internal walls to remove to create cycles.
        symmetric: If True, enforce left-right mirror symmetry.
        seed: Random seed for reproducibility.

    Returns:
        A validated immutable GridMap.
    """
    if rows < 7 or cols < 7 or rows % 2 == 0 or cols % 2 == 0:
        raise ValueError(f"Dimensions must be odd integers >= 7, got rows={rows}, cols={cols}")
    if not (0 <= num_ghosts <= 10):
        raise ValueError(f"num_ghosts must be between 0 and 10, got {num_ghosts}")
    if num_power_pellets < 0:
        raise ValueError(f"num_power_pellets must be non-negative, got {num_power_pellets}")
    if not (0.0 <= loop_fraction <= 1.0):
        raise ValueError(f"loop_fraction must be between 0.0 and 1.0, got {loop_fraction}")

    rng = np.random.default_rng(seed)

    # All cells start as walls
    walls = np.ones((rows, cols), dtype=bool)

    # Passage nodes are odd coordinates
    nodes: list[tuple[int, int]] = [
        (r, c) for r in range(1, rows - 1, 2) for c in range(1, cols - 1, 2)
    ]
    for r, c in nodes:
        walls[r, c] = False

    def mirror_coord(pos: tuple[int, int]) -> tuple[int, int]:
        return (pos[0], cols - 1 - pos[1])

    # Find candidate edges between passage nodes
    candidate_edge_classes: set[tuple[tuple[int, int], ...]] = set()
    node_set = set(nodes)

    for r, c in nodes:
        for dr, dc in ((2, 0), (0, 2)):
            nr, nc = r + dr, c + dc
            if (nr, nc) in node_set:
                wall = ((r + nr) // 2, (c + nc) // 2)
                if symmetric:
                    m_wall = mirror_coord(wall)
                    edge_class = tuple(sorted({wall, m_wall}))
                else:
                    edge_class = (wall,)
                candidate_edge_classes.add(edge_class)

    # Sort deterministically before shuffling
    sorted_edge_classes = sorted(candidate_edge_classes)
    perm = rng.permutation(len(sorted_edge_classes))
    shuffled_edges = [sorted_edge_classes[i] for i in perm]

    dsu = _DSU(nodes)

    for edge_class in shuffled_edges:
        pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
        connects_new = False
        for wr, wc in edge_class:
            if wr % 2 == 0:
                u = (wr - 1, wc)
                v = (wr + 1, wc)
            else:
                u = (wr, wc - 1)
                v = (wr, wc + 1)
            pairs.append((u, v))
            if dsu.find(u) != dsu.find(v):
                connects_new = True

        if connects_new:
            for wr, wc in edge_class:
                walls[wr, wc] = False
            for u, v in pairs:
                dsu.union(u, v)

    # Loop carving: identify remaining internal walls separating walkable cells
    loop_wall_classes: set[tuple[tuple[int, int], ...]] = set()
    for wr in range(1, rows - 1):
        for wc in range(1, cols - 1):
            if not walls[wr, wc]:
                continue
            # Wall between vertical cells or horizontal cells
            is_valid_wall = False
            if wr % 2 == 0 and wc % 2 == 1:
                # Vertical wall between (wr-1, wc) and (wr+1, wc)
                if not walls[wr - 1, wc] and not walls[wr + 1, wc]:
                    is_valid_wall = True
            elif wr % 2 == 1 and wc % 2 == 0:
                # Horizontal wall between (wr, wc-1) and (wr, wc+1)
                if not walls[wr, wc - 1] and not walls[wr, wc + 1]:
                    is_valid_wall = True

            if is_valid_wall:
                w_pos = (wr, wc)
                if symmetric:
                    m_w_pos = mirror_coord(w_pos)
                    w_class = tuple(sorted({w_pos, m_w_pos}))
                else:
                    w_class = (w_pos,)
                loop_wall_classes.add(w_class)

    if loop_wall_classes and loop_fraction > 0.0:
        sorted_loops = sorted(loop_wall_classes)
        perm_loops = rng.permutation(len(sorted_loops))
        num_to_carve = int(round(len(sorted_loops) * loop_fraction))
        for i in range(num_to_carve):
            for wr, wc in sorted_loops[perm_loops[i]]:
                walls[wr, wc] = False

    # Collect all walkable cells
    walkable: list[Position] = [
        Position(r, c) for r in range(rows) for c in range(cols) if not walls[r, c]
    ]

    mid_c = cols // 2
    # Place Pacman spawn: prefer bottom center
    center_walkable = [p for p in walkable if p.col == mid_c]
    if center_walkable and symmetric:
        pacman_spawn = max(center_walkable, key=lambda p: p.row)
    else:
        pacman_spawn = max(
            walkable,
            key=lambda p: (p.row, -abs(p.col - mid_c), -p.col),
        )

    # Place ghosts: rank remaining cells by proximity to top center (1, mid_c)
    available_for_ghosts = [p for p in walkable if p != pacman_spawn]
    if len(available_for_ghosts) < num_ghosts:
        raise ValueError(
            f"Not enough walkable cells ({len(available_for_ghosts)}) for {num_ghosts} ghosts"
        )

    ghost_candidates = sorted(
        available_for_ghosts,
        key=lambda p: (abs(p.row - 1) + abs(p.col - mid_c), p.row, p.col),
    )

    ghost_spawns: dict[int, Position] = {}
    chosen_ghost_positions: set[Position] = set()

    if symmetric:
        gid = 0
        for p in ghost_candidates:
            if gid >= num_ghosts:
                break
            if p in chosen_ghost_positions:
                continue

            mp = Position(p.row, cols - 1 - p.col)
            # If symmetric partner is distinct, walkable, not pacman, and we have room for 2
            if mp != p and mp in available_for_ghosts and (num_ghosts - gid) >= 2:
                ghost_spawns[gid] = p
                chosen_ghost_positions.add(p)
                gid += 1
                ghost_spawns[gid] = mp
                chosen_ghost_positions.add(mp)
                gid += 1
            elif mp == p:  # On the center column
                ghost_spawns[gid] = p
                chosen_ghost_positions.add(p)
                gid += 1
            else:
                # If cannot pair, assign single
                ghost_spawns[gid] = p
                chosen_ghost_positions.add(p)
                gid += 1
    else:
        for gid in range(num_ghosts):
            pos = ghost_candidates[gid]
            ghost_spawns[gid] = pos
            chosen_ghost_positions.add(pos)

    # Remaining available cells for pellets
    available_for_pellets = [
        p for p in walkable if p != pacman_spawn and p not in chosen_ghost_positions
    ]

    if not available_for_pellets:
        raise ValueError("Not enough walkable cells left for pellets")

    pellets = np.zeros((rows, cols), dtype=bool)
    power_pellets = np.zeros((rows, cols), dtype=bool)
    doors = np.zeros((rows, cols), dtype=bool)

    # Place power pellets: prefer corners
    corners = [(1, 1), (1, cols - 2), (rows - 2, 1), (rows - 2, cols - 2)]
    power_candidates = sorted(
        available_for_pellets,
        key=lambda p: min(abs(p.row - cr) + abs(p.col - cc) for cr, cc in corners),
    )

    # Reserve at least 1 cell for regular pellets
    max_power = max(0, min(num_power_pellets, len(available_for_pellets) - 1))
    placed_power = 0

    if symmetric:
        for p in power_candidates:
            if placed_power >= max_power:
                break
            if power_pellets[p.row, p.col]:
                continue
            mp = Position(p.row, cols - 1 - p.col)
            if mp != p and mp in available_for_pellets and (max_power - placed_power) >= 2:
                power_pellets[p.row, p.col] = True
                power_pellets[mp.row, mp.col] = True
                placed_power += 2
            elif mp == p:
                power_pellets[p.row, p.col] = True
                placed_power += 1
    else:
        for i in range(max_power):
            p = power_candidates[i]
            power_pellets[p.row, p.col] = True

    # Fill all remaining available cells with regular pellets
    for p in available_for_pellets:
        if not power_pellets[p.row, p.col]:
            pellets[p.row, p.col] = True

    grid_map = GridMap(
        rows=rows,
        cols=cols,
        walls=walls,
        doors=doors,
        pellets=pellets,
        power_pellets=power_pellets,
        pacman_spawn=pacman_spawn,
        ghost_spawns=ghost_spawns,
        name=f"random_{rows}x{cols}",
        description="Procedurally generated Pacman maze",
    )
    grid_map.validate()
    return grid_map
