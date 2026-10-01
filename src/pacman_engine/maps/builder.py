"""Fluent builder API for constructing GridMap instances in code."""

from typing import Self

import numpy as np

from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import MapValidationError, Position


class MapBuilder:
    """Fluent builder for programmatically creating and validating GridMap instances."""

    def __init__(
        self,
        rows: int,
        cols: int,
        name: str = "",
        wrap_horizontal: bool = False,
        wrap_vertical: bool = False,
    ) -> None:
        if rows <= 0 or cols <= 0:
            raise ValueError(f"Dimensions must be positive, got {rows}x{cols}")
        self.rows = rows
        self.cols = cols
        self.name = name
        self.description = ""
        self.wrap_horizontal = wrap_horizontal
        self.wrap_vertical = wrap_vertical

        self.walls = np.zeros((rows, cols), dtype=bool)
        self.doors = np.zeros((rows, cols), dtype=bool)
        self.pellets = np.zeros((rows, cols), dtype=bool)
        self.power_pellets = np.zeros((rows, cols), dtype=bool)
        self.pacman_spawn: Position | None = None
        self.ghost_spawns: dict[int, Position] = {}
        self.empty_char: str = " "

    def set_name(self, name: str) -> Self:
        """Set map identifier name."""
        self.name = name
        return self

    def set_description(self, description: str) -> Self:
        """Set map descriptive text."""
        self.description = description
        return self

    def set_wrap(self, horizontal: bool = False, vertical: bool = False) -> Self:
        """Configure edge wrap-around flags."""
        self.wrap_horizontal = horizontal
        self.wrap_vertical = vertical
        return self

    def draw_border(self) -> Self:
        """Surround the perimeter of the map with solid walls."""
        self.walls[0, :] = True
        self.walls[-1, :] = True
        self.walls[:, 0] = True
        self.walls[:, -1] = True
        # Clear items on perimeter
        for r in (0, self.rows - 1):
            for c in range(self.cols):
                self._clear_cell_contents(r, c)
        for c in (0, self.cols - 1):
            for r in range(self.rows):
                self._clear_cell_contents(r, c)
        return self

    def _clear_cell_contents(self, row: int, col: int) -> None:
        self.pellets[row, col] = False
        self.power_pellets[row, col] = False
        self.doors[row, col] = False

    def set_wall(self, row: int, col: int) -> Self:
        """Place a solid wall at the specified grid coordinate."""
        self._check_bounds(row, col)
        self.walls[row, col] = True
        self._clear_cell_contents(row, col)
        return self

    def remove_wall(self, row: int, col: int) -> Self:
        """Remove a wall at the specified coordinate."""
        self._check_bounds(row, col)
        self.walls[row, col] = False
        return self

    def add_door(self, row: int, col: int) -> Self:
        """Place a ghost door at the specified coordinate."""
        self._check_bounds(row, col)
        self.walls[row, col] = False
        self.doors[row, col] = True
        self.pellets[row, col] = False
        self.power_pellets[row, col] = False
        return self

    def add_pellet(self, row: int, col: int) -> Self:
        """Place a standard pellet at the specified coordinate."""
        self._check_bounds(row, col)
        self.walls[row, col] = False
        self.doors[row, col] = False
        self.power_pellets[row, col] = False
        self.pellets[row, col] = True
        return self

    def add_power_pellet(self, row: int, col: int) -> Self:
        """Place an energizer/power pellet at the specified coordinate."""
        self._check_bounds(row, col)
        self.walls[row, col] = False
        self.doors[row, col] = False
        self.pellets[row, col] = False
        self.power_pellets[row, col] = True
        return self

    def fill_pellets(
        self,
        exclude_walls: bool = True,
        exclude_doors: bool = True,
        exclude_spawns: bool = True,
    ) -> Self:
        """Fill all walkable floor cells with standard pellets."""
        spawn_positions = set(self.ghost_spawns.values())
        if self.pacman_spawn is not None:
            spawn_positions.add(self.pacman_spawn)

        for r in range(self.rows):
            for c in range(self.cols):
                if exclude_walls and self.walls[r, c]:
                    continue
                if exclude_doors and self.doors[r, c]:
                    continue
                if exclude_spawns and Position(r, c) in spawn_positions:
                    continue
                if self.power_pellets[r, c]:
                    continue
                self.pellets[r, c] = True
        return self

    def set_pacman_spawn(self, row: int, col: int) -> Self:
        """Set Pacman's spawn position."""
        self._check_bounds(row, col)
        self.walls[row, col] = False
        self.doors[row, col] = False
        self.pellets[row, col] = False
        self.power_pellets[row, col] = False
        self.pacman_spawn = Position(row, col)
        return self

    def add_ghost_spawn(self, ghost_id: int, row: int, col: int) -> Self:
        """Set a ghost's spawn position by id."""
        self._check_bounds(row, col)
        if not isinstance(ghost_id, int) or not (0 <= ghost_id <= 9):
            raise ValueError(f"Ghost id must be 0-9, got {ghost_id}")
        self.walls[row, col] = False
        self.doors[row, col] = False
        self.pellets[row, col] = False
        self.power_pellets[row, col] = False
        self.ghost_spawns[ghost_id] = Position(row, col)
        return self

    def build(self) -> GridMap:
        """Construct, validate, and return the immutable GridMap."""
        if self.pacman_spawn is None:
            raise MapValidationError("Map has no Pacman spawn ('P')")

        grid_map = GridMap(
            rows=self.rows,
            cols=self.cols,
            walls=self.walls,
            doors=self.doors,
            pellets=self.pellets,
            power_pellets=self.power_pellets,
            pacman_spawn=self.pacman_spawn,
            ghost_spawns=self.ghost_spawns,
            wrap_horizontal=self.wrap_horizontal,
            wrap_vertical=self.wrap_vertical,
            name=self.name,
            description=self.description,
            empty_char=self.empty_char,
        )
        grid_map.validate()
        return grid_map

    def _check_bounds(self, row: int, col: int) -> None:
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise IndexError(
                f"Coordinates ({row}, {col}) out of bounds for map size ({self.rows}, {self.cols})"
            )
