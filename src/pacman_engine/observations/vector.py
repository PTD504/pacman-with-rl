"""Flat 1-D vector state observation builder."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from pacman_engine.observations.base import (
    ObservationBuilder,
    ObservationSpec,
    register_observation,
)
from pacman_engine.types import Action, GhostMode, Position

if TYPE_CHECKING:
    from pacman_engine.config import RulesConfig
    from pacman_engine.maps.grid_map import GridMap
    from pacman_engine.state import GameView


@register_observation("vector")
class VectorObservationBuilder(ObservationBuilder):
    """Encodes game state into a 1-D float32 vector of entities, modes, and global features.

    Parameters:
        normalize: Whether coordinates, distances, and counts are normalized to [0, 1] (or [-1, 1]).
        coordinates: "absolute" for map grid coordinates, or "relative_to_pacman" for offsets.
        include_directions: Whether to include one-hot facing direction per entity.
        include_pellet_mask: Whether to include flattened pellet presence mask across all map cells.
        include_wall_mask: Whether to include flattened wall mask across all map cells.
        include_distances: Whether to include BFS distances to ghosts and the nearest pellet.
    """

    def __init__(
        self,
        normalize: bool = True,
        coordinates: Literal["absolute", "relative_to_pacman"] = "absolute",
        include_directions: bool = True,
        include_pellet_mask: bool = False,
        include_wall_mask: bool = False,
        include_distances: bool = False,
        initial_lives: int | None = None,
    ) -> None:
        if coordinates not in ("absolute", "relative_to_pacman"):
            raise ValueError(
                f"coordinates must be 'absolute' or 'relative_to_pacman', got {coordinates!r}"
            )
        self.normalize = bool(normalize)
        self.coordinates = coordinates
        self.include_directions = bool(include_directions)
        self.include_pellet_mask = bool(include_pellet_mask)
        self.include_wall_mask = bool(include_wall_mask)
        self.include_distances = bool(include_distances)
        self.initial_lives = int(initial_lives) if initial_lives is not None else None
        self._cached_rules: RulesConfig | None = None
        self._cached_map: GridMap | None = None

    def spec(self, game_map: GridMap, rules: RulesConfig) -> ObservationSpec:
        """Derive the static ObservationSpec and feature names for a given map and rules."""
        self._cached_rules = rules
        self._cached_map = game_map
        feature_names: list[str] = []
        lows: list[float] = []
        highs: list[float] = []

        rows, cols = game_map.rows, game_map.cols
        max_r = float(max(1, rows - 1))
        max_c = float(max(1, cols - 1))
        max_dist = float(rows * cols)

        # 1. Pacman coordinates
        feature_names.extend(["pacman_row", "pacman_col"])
        if self.coordinates == "relative_to_pacman":
            # Relative to pacman is always (0, 0)
            lows.extend([0.0, 0.0])
            highs.extend([0.0, 0.0])
        elif self.normalize:
            lows.extend([0.0, 0.0])
            highs.extend([1.0, 1.0])
        else:
            lows.extend([0.0, 0.0])
            highs.extend([max_r, max_c])

        # Pacman directions (one-hot for UP, DOWN, LEFT, RIGHT; all zero if STAY)
        if self.include_directions:
            feature_names.extend(
                [
                    "pacman_dir_up",
                    "pacman_dir_down",
                    "pacman_dir_left",
                    "pacman_dir_right",
                ]
            )
            lows.extend([0.0] * 4)
            highs.extend([1.0] * 4)

        # 2. Ghost features (sorted by ghost id)
        ghost_ids = sorted(game_map.ghost_spawns.keys())
        for gid in ghost_ids:
            feature_names.extend([f"ghost_{gid}_row", f"ghost_{gid}_col"])
            if self.coordinates == "relative_to_pacman":
                if self.normalize:
                    lows.extend([-1.0, -1.0])
                    highs.extend([1.0, 1.0])
                else:
                    lows.extend([-max_r, -max_c])
                    highs.extend([max_r, max_c])
            elif self.normalize:
                lows.extend([0.0, 0.0])
                highs.extend([1.0, 1.0])
            else:
                lows.extend([0.0, 0.0])
                highs.extend([max_r, max_c])

            if self.include_directions:
                feature_names.extend(
                    [
                        f"ghost_{gid}_dir_up",
                        f"ghost_{gid}_dir_down",
                        f"ghost_{gid}_dir_left",
                        f"ghost_{gid}_dir_right",
                    ]
                )
                lows.extend([0.0] * 4)
                highs.extend([1.0] * 4)

            feature_names.extend(
                [
                    f"ghost_{gid}_is_frightened",
                    f"ghost_{gid}_is_dead",
                ]
            )
            lows.extend([0.0, 0.0])
            highs.extend([1.0, 1.0])

        # 3. BFS Distances
        if self.include_distances:
            for gid in ghost_ids:
                feature_names.append(f"dist_pacman_to_ghost_{gid}")
                lows.append(0.0)
                highs.append(1.0 if self.normalize else max_dist)

            feature_names.append("dist_pacman_to_nearest_pellet")
            lows.append(0.0)
            highs.append(1.0 if self.normalize else max_dist)

        # 4. Globals
        feature_names.extend(
            [
                "remaining_pellet_fraction",
                "frightened_timer_fraction",
                "lives_fraction",
                "tick_fraction",
            ]
        )
        if self.normalize:
            lows.extend([0.0, 0.0, 0.0, 0.0])
            highs.extend([1.0, 1.0, 1.0, 1.0])
        else:
            lows.extend([0.0, 0.0, 0.0, 0.0])
            max_s = float(rules.max_steps) if rules.max_steps is not None else 10000.0
            highs.extend(
                [
                    float(game_map.total_pellets),
                    float(rules.frightened_duration),
                    float(rules.pacman_lives),
                    max_s,
                ]
            )

        # 5. Pellet mask
        if self.include_pellet_mask:
            for r in range(rows):
                for c in range(cols):
                    feature_names.append(f"pellet_r{r}_c{c}")
                    lows.append(0.0)
                    highs.append(1.0)

        # 6. Wall mask
        if self.include_wall_mask:
            for r in range(rows):
                for c in range(cols):
                    feature_names.append(f"wall_r{r}_c{c}")
                    lows.append(0.0)
                    highs.append(1.0)

        low_arr = np.array(lows, dtype=np.float32)
        high_arr = np.array(highs, dtype=np.float32)

        return ObservationSpec(
            shape=(len(feature_names),),
            dtype=np.dtype(np.float32),
            low=low_arr,
            high=high_arr,
            feature_names=feature_names,
        )

    def build(self, view: GameView) -> np.ndarray:
        """Construct the 1-D vector observation from GameView."""
        game_map = view.graph.grid_map
        rows, cols = game_map.rows, game_map.cols
        max_r = float(max(1, rows - 1))
        max_c = float(max(1, cols - 1))
        max_dist = float(rows * cols)

        features: list[float] = []

        # 1. Pacman coordinates
        p_row, p_col = view.pacman_pos.row, view.pacman_pos.col
        if self.coordinates == "relative_to_pacman":
            features.extend([0.0, 0.0])
        elif self.normalize:
            features.extend([p_row / max_r, p_col / max_c])
        else:
            features.extend([float(p_row), float(p_col)])

        if self.include_directions:
            p_dir = view.pacman_direction
            features.extend(
                [
                    1.0 if p_dir == Action.UP else 0.0,
                    1.0 if p_dir == Action.DOWN else 0.0,
                    1.0 if p_dir == Action.LEFT else 0.0,
                    1.0 if p_dir == Action.RIGHT else 0.0,
                ]
            )

        # 2. Ghost features
        ghost_ids = sorted(view.ghost_positions.keys())
        for gid in ghost_ids:
            g_pos = view.ghost_positions[gid]
            if self.coordinates == "relative_to_pacman":
                drow = float(g_pos.row - p_row)
                dcol = float(g_pos.col - p_col)
                if self.normalize:
                    features.extend([drow / max_r, dcol / max_c])
                else:
                    features.extend([drow, dcol])
            elif self.normalize:
                features.extend([g_pos.row / max_r, g_pos.col / max_c])
            else:
                features.extend([float(g_pos.row), float(g_pos.col)])

            if self.include_directions:
                g_dir = view.ghost_directions.get(gid, Action.STAY)
                features.extend(
                    [
                        1.0 if g_dir == Action.UP else 0.0,
                        1.0 if g_dir == Action.DOWN else 0.0,
                        1.0 if g_dir == Action.LEFT else 0.0,
                        1.0 if g_dir == Action.RIGHT else 0.0,
                    ]
                )

            mode = view.ghost_modes.get(gid, GhostMode.SCATTER)
            features.append(1.0 if mode == GhostMode.FRIGHTENED else 0.0)
            features.append(1.0 if mode == GhostMode.DEAD else 0.0)

        # 3. BFS Distances
        if self.include_distances:
            # Populate Pacman BFS distance cache
            view.graph.distance(view.pacman_pos, view.pacman_pos, for_ghost=False)
            dist_cache = view.graph._dist_cache.get((view.pacman_pos, False), {})

            # Distance to each ghost
            for gid in ghost_ids:
                g_pos = view.ghost_positions[gid]
                d = dist_cache.get(g_pos, -1)
                if d < 0:
                    features.append(1.0 if self.normalize else max_dist)
                else:
                    norm_d = min(float(d), max_dist) / max_dist if self.normalize else float(d)
                    features.append(norm_d)

            # Distance to nearest pellet
            active_pellet_positions = [
                Position(r, c)
                for r in range(rows)
                for c in range(cols)
                if (view.pellets[r, c] or view.power_pellets[r, c])
            ]
            pellet_dists = [dist_cache[pos] for pos in active_pellet_positions if pos in dist_cache]
            if pellet_dists:
                min_pellet_dist = min(pellet_dists)
                if self.normalize:
                    features.append(min(float(min_pellet_dist), max_dist) / max_dist)
                else:
                    features.append(float(min_pellet_dist))
            else:
                features.append(1.0 if self.normalize else max_dist)

        # 4. Globals
        total_pellets = max(1, game_map.total_pellets)
        max_frightened = (
            float(self._cached_rules.frightened_duration) if self._cached_rules else 40.0
        )
        max_lives = (
            float(self.initial_lives)
            if self.initial_lives is not None
            else (float(self._cached_rules.pacman_lives) if self._cached_rules else 3.0)
        )

        max_steps = (
            float(self._cached_rules.max_steps)
            if self._cached_rules and self._cached_rules.max_steps is not None
            else 1000.0
        )

        if self.normalize:
            features.append(view.remaining_pellets / total_pellets)
            frightened_norm = (
                min(1.0, view.frightened_timer / max(1.0, max_frightened))
                if view.frightened_timer > 0
                else 0.0
            )
            features.append(frightened_norm)
            features.append(min(1.0, view.lives / max(1.0, max_lives)))
            features.append(min(1.0, view.tick / max(1.0, max_steps)))
        else:
            features.append(float(view.remaining_pellets))
            features.append(float(view.frightened_timer))
            features.append(float(view.lives))
            features.append(float(view.tick))

        # 5. Pellet mask
        if self.include_pellet_mask:
            combined_pellets = view.pellets | view.power_pellets
            for r in range(rows):
                for c in range(cols):
                    features.append(1.0 if combined_pellets[r, c] else 0.0)

        # 6. Wall mask
        if self.include_wall_mask:
            for r in range(rows):
                for c in range(cols):
                    features.append(1.0 if game_map.walls[r, c] else 0.0)

        return np.array(features, dtype=np.float32)
