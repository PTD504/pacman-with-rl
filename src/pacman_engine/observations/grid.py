"""Multi-channel 2D spatial grid observation builder."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np

from pacman_engine.observations.base import (
    ObservationBuilder,
    ObservationSpec,
    register_observation,
)
from pacman_engine.types import Action, GhostMode

if TYPE_CHECKING:
    from pacman_engine.config import RulesConfig
    from pacman_engine.maps.grid_map import GridMap
    from pacman_engine.state import GameView

DEFAULT_GRID_CHANNELS: tuple[str, ...] = (
    "walls",
    "doors",
    "pellets",
    "power_pellets",
    "pacman",
    "ghosts",
    "frightened_ghosts",
)

SUPPORTED_CHANNELS: set[str] = {
    "walls",
    "doors",
    "pellets",
    "power_pellets",
    "pacman",
    "ghosts",
    "frightened_ghosts",
    "dead_ghosts",
    "all_ghosts",
    "pacman_dir_up",
    "pacman_dir_down",
    "pacman_dir_left",
    "pacman_dir_right",
}


@register_observation("grid")
class GridObservationBuilder(ObservationBuilder):
    """Encodes game state into a multi-channel float32 tensor of binary feature planes.

    Supports custom channel subsets, channel ordering, channels-first/last layout,
    and optional egocentric Pacman-centered cropping with wall padding.

    Parameters:
        channels: Sequence of channel names defining the planes to extract and their order.
                  'pacman_direction' expands to 4 one-hot planes:
                  ('pacman_dir_up', 'pacman_dir_down', 'pacman_dir_left', 'pacman_dir_right').
        channels_first: If True, tensor shape is (C, H, W); if False, (H, W, C).
        view_radius: Optional integer radius for an egocentric window centered on Pacman.
                     Window size is (2*R + 1, 2*R + 1). Coordinates outside the map are padded
                     with wall cells. If None, the entire map is returned.
    """

    def __init__(
        self,
        channels: Sequence[str] | None = None,
        channels_first: bool = True,
        view_radius: int | None = None,
    ) -> None:
        raw_channels = list(channels) if channels is not None else list(DEFAULT_GRID_CHANNELS)
        expanded: list[str] = []
        for ch in raw_channels:
            if ch == "pacman_direction":
                expanded.extend(
                    [
                        "pacman_dir_up",
                        "pacman_dir_down",
                        "pacman_dir_left",
                        "pacman_dir_right",
                    ]
                )
            else:
                expanded.append(ch)

        for ch in expanded:
            if ch not in SUPPORTED_CHANNELS:
                raise ValueError(
                    f"Unsupported channel '{ch}'. Supported channels: {sorted(SUPPORTED_CHANNELS)}"
                )

        if not expanded:
            raise ValueError("channels must not be empty")

        if view_radius is not None and view_radius < 1:
            raise ValueError(f"view_radius must be >= 1, got {view_radius}")

        self.channels: tuple[str, ...] = tuple(expanded)
        self.channels_first: bool = bool(channels_first)
        self.view_radius: int | None = view_radius

    def spec(self, game_map: GridMap, rules: RulesConfig) -> ObservationSpec:
        """Derive the static ObservationSpec from map dimensions and configured channels."""
        if self.view_radius is not None:
            h = 2 * self.view_radius + 1
            w = 2 * self.view_radius + 1
        else:
            h = game_map.rows
            w = game_map.cols

        c = len(self.channels)
        shape = (c, h, w) if self.channels_first else (h, w, c)

        return ObservationSpec(
            shape=shape,
            dtype=np.dtype(np.float32),
            low=0.0,
            high=1.0,
            feature_names=list(self.channels),
        )

    def _build_full_planes(self, view: GameView) -> dict[str, np.ndarray]:
        """Construct all supported full-map binary planes for the current GameView."""
        game_map = view.graph.grid_map
        rows, cols = game_map.rows, game_map.cols

        planes: dict[str, np.ndarray] = {
            "walls": game_map.walls.astype(np.float32),
            "doors": game_map.doors.astype(np.float32),
            "pellets": view.pellets.astype(np.float32),
            "power_pellets": view.power_pellets.astype(np.float32),
        }

        # Pacman plane
        pacman_plane = np.zeros((rows, cols), dtype=np.float32)
        pacman_plane[view.pacman_pos.row, view.pacman_pos.col] = 1.0
        planes["pacman"] = pacman_plane

        # Ghost planes
        ghost_plane = np.zeros((rows, cols), dtype=np.float32)
        frightened_plane = np.zeros((rows, cols), dtype=np.float32)
        dead_plane = np.zeros((rows, cols), dtype=np.float32)
        all_ghosts_plane = np.zeros((rows, cols), dtype=np.float32)

        for gid, pos in view.ghost_positions.items():
            all_ghosts_plane[pos.row, pos.col] = 1.0
            mode = view.ghost_modes.get(gid, GhostMode.SCATTER)
            if mode == GhostMode.FRIGHTENED:
                frightened_plane[pos.row, pos.col] = 1.0
            elif mode == GhostMode.DEAD:
                dead_plane[pos.row, pos.col] = 1.0
            else:
                ghost_plane[pos.row, pos.col] = 1.0

        planes["ghosts"] = ghost_plane
        planes["frightened_ghosts"] = frightened_plane
        planes["dead_ghosts"] = dead_plane
        planes["all_ghosts"] = all_ghosts_plane

        # Direction planes (one-hot indicator at Pacman's position)
        p_row, p_col = view.pacman_pos.row, view.pacman_pos.col
        dir_up = np.zeros((rows, cols), dtype=np.float32)
        dir_down = np.zeros((rows, cols), dtype=np.float32)
        dir_left = np.zeros((rows, cols), dtype=np.float32)
        dir_right = np.zeros((rows, cols), dtype=np.float32)

        if view.pacman_direction == Action.UP:
            dir_up[p_row, p_col] = 1.0
        elif view.pacman_direction == Action.DOWN:
            dir_down[p_row, p_col] = 1.0
        elif view.pacman_direction == Action.LEFT:
            dir_left[p_row, p_col] = 1.0
        elif view.pacman_direction == Action.RIGHT:
            dir_right[p_row, p_col] = 1.0

        planes["pacman_dir_up"] = dir_up
        planes["pacman_dir_down"] = dir_down
        planes["pacman_dir_left"] = dir_left
        planes["pacman_dir_right"] = dir_right

        return planes

    def _crop_egocentric(
        self, full_planes: dict[str, np.ndarray], view: GameView
    ) -> list[np.ndarray]:
        """Crop an egocentric window of radius R centered on Pacman, padding with walls."""
        assert self.view_radius is not None
        r_rad = self.view_radius
        window_size = 2 * r_rad + 1
        p_row, p_col = view.pacman_pos.row, view.pacman_pos.col
        game_map = view.graph.grid_map
        rows, cols = game_map.rows, game_map.cols

        # Map coordinate ranges for the crop window
        r_start = p_row - r_rad
        r_end = p_row + r_rad + 1
        c_start = p_col - r_rad
        c_end = p_col + r_rad + 1

        cropped_channels: list[np.ndarray] = []
        for ch_name in self.channels:
            full = full_planes[ch_name]
            # Initialize with 1.0 for walls (out of bounds is wall), 0.0 for others
            if ch_name == "walls":
                window = np.ones((window_size, window_size), dtype=np.float32)
            else:
                window = np.zeros((window_size, window_size), dtype=np.float32)

            # Determine intersection slice between crop window and valid map coordinates
            map_r0 = max(0, r_start)
            map_r1 = min(rows, r_end)
            map_c0 = max(0, c_start)
            map_c1 = min(cols, c_end)

            if map_r0 < map_r1 and map_c0 < map_c1:
                win_r0 = map_r0 - r_start
                win_r1 = win_r0 + (map_r1 - map_r0)
                win_c0 = map_c0 - c_start
                win_c1 = win_c0 + (map_c1 - map_c0)

                window[win_r0:win_r1, win_c0:win_c1] = full[map_r0:map_r1, map_c0:map_c1]

            cropped_channels.append(window)

        return cropped_channels

    def build(self, view: GameView) -> np.ndarray:
        """Construct the multi-channel grid tensor from GameView."""
        full_planes = self._build_full_planes(view)

        if self.view_radius is not None:
            channel_arrays = self._crop_egocentric(full_planes, view)
        else:
            channel_arrays = [full_planes[ch] for ch in self.channels]

        if self.channels_first:
            # (C, H, W)
            return np.stack(channel_arrays, axis=0)
        else:
            # (H, W, C)
            return np.stack(channel_arrays, axis=-1)
