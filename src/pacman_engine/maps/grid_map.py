"""Immutable static grid map representation, ASCII parser, and validator."""

from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from pacman_engine.types import MapValidationError, Position


class GridMap:
    """Immutable representation of a static Pacman map layout.

    Attributes:
        rows: Height of the grid.
        cols: Width of the grid.
        walls: 2D boolean array where True indicates impassable wall.
        doors: 2D boolean array where True indicates ghost-only door.
        pellets: 2D boolean array where True indicates standard pellet.
        power_pellets: 2D boolean array where True indicates energizer/power pellet.
        pacman_spawn: Starting Position for Pacman.
        ghost_spawns: Mapping from ghost ID (0-9) to starting Position.
        wrap_horizontal: Whether opposite horizontal edges are connected.
        wrap_vertical: Whether opposite vertical edges are connected.
        name: Optional map identifier.
        description: Optional human-readable description.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        walls: np.ndarray,
        doors: np.ndarray,
        pellets: np.ndarray,
        power_pellets: np.ndarray,
        pacman_spawn: Position,
        ghost_spawns: dict[int, Position],
        wrap_horizontal: bool = False,
        wrap_vertical: bool = False,
        name: str = "",
        description: str = "",
        empty_char: str = " ",
        raw_headers: list[str] | None = None,
    ) -> None:
        self.rows = int(rows)
        self.cols = int(cols)

        # Make numpy arrays read-only copies to guarantee immutability
        self.walls = np.array(walls, dtype=bool, copy=True)
        self.walls.flags.writeable = False

        self.doors = np.array(doors, dtype=bool, copy=True)
        self.doors.flags.writeable = False

        self.pellets = np.array(pellets, dtype=bool, copy=True)
        self.pellets.flags.writeable = False

        self.power_pellets = np.array(power_pellets, dtype=bool, copy=True)
        self.power_pellets.flags.writeable = False

        self.pacman_spawn = pacman_spawn
        self.ghost_spawns = dict(sorted(ghost_spawns.items()))
        self.wrap_horizontal = bool(wrap_horizontal)
        self.wrap_vertical = bool(wrap_vertical)
        self.name = str(name)
        self.description = str(description)
        self.empty_char = empty_char if empty_char in (" ", "_") else " "
        self._raw_headers = list(raw_headers) if raw_headers is not None else None

    @property
    def total_pellets(self) -> int:
        """Total count of regular and power pellets on the map at initialization."""
        return int(self.pellets.sum() + self.power_pellets.sum())

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, GridMap):
            return False
        return (
            self.rows == other.rows
            and self.cols == other.cols
            and np.array_equal(self.walls, other.walls)
            and np.array_equal(self.doors, other.doors)
            and np.array_equal(self.pellets, other.pellets)
            and np.array_equal(self.power_pellets, other.power_pellets)
            and self.pacman_spawn == other.pacman_spawn
            and self.ghost_spawns == other.ghost_spawns
            and self.wrap_horizontal == other.wrap_horizontal
            and self.wrap_vertical == other.wrap_vertical
            and self.name == other.name
            and self.description == other.description
        )

    @classmethod
    def from_ascii(cls, text: str) -> "GridMap":
        """Parse an ASCII map string with optional metadata headers into a GridMap."""
        lines = text.splitlines()
        headers: list[str] = []
        grid_lines: list[str] = []

        name = ""
        description = ""
        wrap_horizontal = False
        wrap_vertical = False

        reading_headers = True
        for line in lines:
            stripped = line.strip()
            if reading_headers and stripped.startswith(";"):
                headers.append(line)
                content = stripped[1:].strip()
                if ":" in content:
                    k, v = content.split(":", 1)
                    key = k.strip().lower()
                    val = v.strip()
                    if key == "name":
                        name = val
                    elif key == "description":
                        description = val
                    elif key == "wrap_horizontal":
                        wrap_horizontal = val.lower() in ("true", "1", "yes")
                    elif key == "wrap_vertical":
                        wrap_vertical = val.lower() in ("true", "1", "yes")
                continue
            elif stripped == "" and reading_headers and not grid_lines:
                # Blank lines before grid lines are skipped
                continue
            else:
                reading_headers = False
                grid_lines.append(line)

        if not grid_lines:
            raise MapValidationError("Map has no grid content")

        rows = len(grid_lines)
        cols = len(grid_lines[0])

        for r, row_str in enumerate(grid_lines):
            if len(row_str) != cols:
                raise MapValidationError(
                    f"Map is not rectangular: row {r} has length {len(row_str)} but expected {cols}"
                )

        walls = np.zeros((rows, cols), dtype=bool)
        doors = np.zeros((rows, cols), dtype=bool)
        pellets = np.zeros((rows, cols), dtype=bool)
        power_pellets = np.zeros((rows, cols), dtype=bool)
        pacman_spawns: list[Position] = []
        ghost_spawns: dict[int, Position] = {}

        empty_char_used = " "
        has_empty_char = False

        for r, row_str in enumerate(grid_lines):
            for c, char in enumerate(row_str):
                pos = Position(r, c)
                match char:
                    case "#":
                        walls[r, c] = True
                    case "=":
                        doors[r, c] = True
                    case ".":
                        pellets[r, c] = True
                    case "o":
                        power_pellets[r, c] = True
                    case "P":
                        pacman_spawns.append(pos)
                    case " ":
                        if not has_empty_char:
                            empty_char_used = " "
                            has_empty_char = True
                    case "_":
                        if not has_empty_char:
                            empty_char_used = "_"
                            has_empty_char = True
                    case _ if char.isdigit():
                        gid = int(char)
                        if gid in ghost_spawns:
                            existing = ghost_spawns[gid]
                            raise MapValidationError(
                                f"Duplicate ghost spawn id '{gid}' at row {r}, col {c} "
                                f"(previously at row {existing.row}, col {existing.col})"
                            )
                        ghost_spawns[gid] = pos
                    case _:
                        raise MapValidationError(f"Unknown character '{char}' at row {r}, col {c}")

        if len(pacman_spawns) == 0:
            raise MapValidationError("Map must have exactly one Pacman spawn ('P'), found 0")
        if len(pacman_spawns) > 1:
            locs = ", ".join(f"(row={p.row}, col={p.col})" for p in pacman_spawns)
            raise MapValidationError(
                f"Map must have exactly one Pacman spawn ('P'), "
                f"found {len(pacman_spawns)} at {locs}"
            )

        grid_map = cls(
            rows=rows,
            cols=cols,
            walls=walls,
            doors=doors,
            pellets=pellets,
            power_pellets=power_pellets,
            pacman_spawn=pacman_spawns[0],
            ghost_spawns=ghost_spawns,
            wrap_horizontal=wrap_horizontal,
            wrap_vertical=wrap_vertical,
            name=name,
            description=description,
            empty_char=empty_char_used,
            raw_headers=headers if headers else None,
        )
        grid_map.validate()
        return grid_map

    @classmethod
    def from_file(cls, path: str | Path) -> "GridMap":
        """Load and parse a GridMap from a file."""
        content = Path(path).read_text(encoding="utf-8")
        return cls.from_ascii(content)

    def to_ascii(self, empty_char: str | None = None) -> str:
        """Serialize the GridMap into its exact ASCII representation."""
        char_empty = empty_char if empty_char is not None else self.empty_char

        lines: list[str] = []
        if self._raw_headers is not None:
            lines.extend(self._raw_headers)
        else:
            if self.name:
                lines.append(f"; name: {self.name}")
            if self.description:
                lines.append(f"; description: {self.description}")
            if self.wrap_horizontal:
                lines.append("; wrap_horizontal: true")
            if self.wrap_vertical:
                lines.append("; wrap_vertical: true")

        ghost_by_pos = {pos: gid for gid, pos in self.ghost_spawns.items()}

        for r in range(self.rows):
            row_chars: list[str] = []
            for c in range(self.cols):
                pos = Position(r, c)
                if pos == self.pacman_spawn:
                    row_chars.append("P")
                elif pos in ghost_by_pos:
                    row_chars.append(str(ghost_by_pos[pos]))
                elif self.walls[r, c]:
                    row_chars.append("#")
                elif self.doors[r, c]:
                    row_chars.append("=")
                elif self.power_pellets[r, c]:
                    row_chars.append("o")
                elif self.pellets[r, c]:
                    row_chars.append(".")
                else:
                    row_chars.append(char_empty)
            lines.append("".join(row_chars))

        return "\n".join(lines) + "\n"

    def validate(self) -> None:
        """Validate map integrity according to game rules.

        Raises MapValidationError with row/col information on failure.
        """
        # 1. Rectangular and dimension check
        if self.rows <= 0 or self.cols <= 0:
            raise MapValidationError(f"Invalid dimensions: rows={self.rows}, cols={self.cols}")

        for arr_name, arr in (
            ("walls", self.walls),
            ("doors", self.doors),
            ("pellets", self.pellets),
            ("power_pellets", self.power_pellets),
        ):
            if arr.shape != (self.rows, self.cols):
                raise MapValidationError(
                    f"{arr_name} array shape {arr.shape} does not match map dimensions "
                    f"({self.rows}, {self.cols})"
                )

        # 2. Pacman spawn check
        if not (0 <= self.pacman_spawn.row < self.rows and 0 <= self.pacman_spawn.col < self.cols):
            raise MapValidationError(
                f"Pacman spawn (row={self.pacman_spawn.row}, col={self.pacman_spawn.col}) "
                f"is out of map bounds ({self.rows}x{self.cols})"
            )
        if self.walls[self.pacman_spawn.row, self.pacman_spawn.col]:
            raise MapValidationError(
                f"Pacman spawn at (row={self.pacman_spawn.row}, col={self.pacman_spawn.col}) "
                f"is on a wall"
            )
        if self.doors[self.pacman_spawn.row, self.pacman_spawn.col]:
            raise MapValidationError(
                f"Pacman spawn at (row={self.pacman_spawn.row}, col={self.pacman_spawn.col}) "
                f"is on a ghost door"
            )

        # 3. Ghost spawns check
        seen_gids: set[int] = set()
        for gid, pos in self.ghost_spawns.items():
            if not isinstance(gid, int) or not (0 <= gid <= 9):
                raise MapValidationError(f"Ghost id '{gid}' must be an integer between 0 and 9")
            if gid in seen_gids:
                raise MapValidationError(f"Duplicate ghost spawn id '{gid}'")
            seen_gids.add(gid)

            if not (0 <= pos.row < self.rows and 0 <= pos.col < self.cols):
                raise MapValidationError(
                    f"Ghost spawn '{gid}' (row={pos.row}, col={pos.col}) is out of bounds"
                )
            if self.walls[pos.row, pos.col]:
                raise MapValidationError(
                    f"Ghost spawn '{gid}' at (row={pos.row}, col={pos.col}) is on a wall"
                )

        # 4. Spawn cells never hold pellets
        all_spawns = [(self.pacman_spawn, "Pacman")] + [
            (pos, f"Ghost '{gid}'") for gid, pos in self.ghost_spawns.items()
        ]
        for pos, label in all_spawns:
            if self.pellets[pos.row, pos.col]:
                raise MapValidationError(
                    f"{label} spawn cell at (row={pos.row}, col={pos.col}) cannot contain a pellet"
                )
            if self.power_pellets[pos.row, pos.col]:
                raise MapValidationError(
                    f"{label} spawn cell at (row={pos.row}, col={pos.col}) "
                    "cannot contain a power pellet"
                )

        # 5. At least one pellet
        total_pellets = int(self.pellets.sum() + self.power_pellets.sum())
        if total_pellets == 0:
            raise MapValidationError("Map must contain at least one pellet")

        # Helper for BFS reachability
        def get_accessible_neighbors(curr: Position, can_pass_doors: bool) -> list[Position]:
            res: list[Position] = []
            deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for dr, dc in deltas:
                nr = curr.row + dr
                nc = curr.col + dc

                # Horizontal wrap
                if self.wrap_horizontal and dr == 0:
                    if nc < 0:
                        nc = self.cols - 1
                    elif nc >= self.cols:
                        nc = 0
                elif nc < 0 or nc >= self.cols:
                    continue

                # Vertical wrap
                if self.wrap_vertical and dc == 0:
                    if nr < 0:
                        nr = self.rows - 1
                    elif nr >= self.rows:
                        nr = 0
                elif nr < 0 or nr >= self.rows:
                    continue

                if self.walls[nr, nc]:
                    continue
                if self.doors[nr, nc] and not can_pass_doors:
                    continue

                res.append(Position(nr, nc))
            return res

        # 6. Every pellet and power pellet reachable by Pacman (doors blocked)
        pacman_visited: set[Position] = set()
        queue: deque[Position] = deque([self.pacman_spawn])
        pacman_visited.add(self.pacman_spawn)

        while queue:
            curr = queue.popleft()
            for nxt in get_accessible_neighbors(curr, can_pass_doors=False):
                if nxt not in pacman_visited:
                    pacman_visited.add(nxt)
                    queue.append(nxt)

        for r in range(self.rows):
            for c in range(self.cols):
                if self.pellets[r, c] and Position(r, c) not in pacman_visited:
                    raise MapValidationError(
                        f"Pellet at (row={r}, col={c}) is unreachable by Pacman"
                    )
                if self.power_pellets[r, c] and Position(r, c) not in pacman_visited:
                    raise MapValidationError(
                        f"Power pellet at (row={r}, col={c}) is unreachable by Pacman"
                    )

        # 7. Every ghost spawn can reach Pacman's spawn (doors open)
        for gid, gpos in self.ghost_spawns.items():
            ghost_visited: set[Position] = set()
            ghost_queue: deque[Position] = deque([gpos])
            ghost_visited.add(gpos)

            while ghost_queue:
                curr = ghost_queue.popleft()
                if curr == self.pacman_spawn:
                    break
                for nxt in get_accessible_neighbors(curr, can_pass_doors=True):
                    if nxt not in ghost_visited:
                        ghost_visited.add(nxt)
                        ghost_queue.append(nxt)

            if self.pacman_spawn not in ghost_visited:
                raise MapValidationError(
                    f"Ghost spawn '{gid}' at (row={gpos.row}, col={gpos.col}) cannot reach "
                    f"Pacman spawn at (row={self.pacman_spawn.row}, col={self.pacman_spawn.col})"
                )
