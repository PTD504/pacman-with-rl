"""Headless 2D pixel frame renderer for Pacman."""

from __future__ import annotations

import dataclasses
import os
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from pacman_engine.rendering.theme import RenderTheme
from pacman_engine.types import Action, GhostMode

if TYPE_CHECKING:
    from pacman_engine.state import GameView


def _get_pygame() -> Any:
    """Lazily import and initialize headless pygame."""
    if "SDL_VIDEODRIVER" not in os.environ:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    if not pygame.get_init():
        pygame.init()
    return pygame


class Renderer:
    """Headless 2D frame renderer for Pacman game states.

    Draws the map topology, pellets, Pacman, and ghosts onto a Pygame surface
    and outputs uint8 NumPy arrays, with support for area-resizing and grayscale conversion.

    Parameters:
        theme: Visual RenderTheme (colors, dimensions). If None, default theme is used.
        cell_size: Optional override for cell_size in the theme.
        grayscale: If True, output has 1 channel (H, W, 1). If False, (H, W, 3).
        output_size: Optional (H, W) or int to resize frames using Pillow area interpolation.
    """

    def __init__(
        self,
        theme: RenderTheme | None = None,
        cell_size: int | None = None,
        grayscale: bool = False,
        output_size: tuple[int, int] | int | None = None,
    ) -> None:
        if theme is None:
            t = RenderTheme()
        else:
            t = theme
        if cell_size is not None:
            t = dataclasses.replace(t, cell_size=cell_size)

        self.theme: RenderTheme = t
        self.grayscale: bool = bool(grayscale)

        if output_size is not None:
            if isinstance(output_size, int):
                self.output_size: tuple[int, int] | None = (output_size, output_size)
            elif len(output_size) == 2:
                self.output_size = (int(output_size[0]), int(output_size[1]))
            else:
                raise ValueError(f"output_size must be a 2-tuple or int, got {output_size}")
        else:
            self.output_size = None

    def render(self, view: GameView) -> np.ndarray:
        """Render the given GameView snapshot to a uint8 NumPy array.

        Returns:
            A uint8 NumPy array of shape (H, W, 3) or (H, W, 1) if grayscale.
        """
        pygame = _get_pygame()

        game_map = view.graph.grid_map
        rows, cols = game_map.rows, game_map.cols
        cell_size = self.theme.cell_size
        width = cols * cell_size
        height = rows * cell_size

        surface = pygame.Surface((width, height))
        surface.fill(self.theme.bg_color)

        # 1. Grid lines (optional)
        if self.theme.grid_lines:
            line_color = self.theme.grid_line_color
            for r in range(rows + 1):
                y = r * cell_size
                pygame.draw.line(surface, line_color, (0, y), (width, y))
            for c in range(cols + 1):
                x = c * cell_size
                pygame.draw.line(surface, line_color, (x, 0), (x, height))

        # 2. Walls and Doors
        for r in range(rows):
            for c in range(cols):
                if game_map.walls[r, c]:
                    rect = (c * cell_size, r * cell_size, cell_size, cell_size)
                    surface.fill(self.theme.wall_color, rect)
                elif game_map.doors[r, c]:
                    # Fill cell with door color
                    rect = (c * cell_size, r * cell_size, cell_size, cell_size)
                    surface.fill(self.theme.door_color, rect)

        # 3. Pellets
        pellet_rad = self.theme.pellet_radius or max(1, cell_size // 8)
        pellet_color = self.theme.pellet_color
        for r in range(rows):
            for c in range(cols):
                if view.pellets[r, c]:
                    cx = c * cell_size + cell_size // 2
                    cy = r * cell_size + cell_size // 2
                    pygame.draw.circle(surface, pellet_color, (cx, cy), pellet_rad)

        # 4. Power pellets
        power_rad = self.theme.power_pellet_radius or max(2, cell_size // 4)
        power_color = self.theme.power_pellet_color
        for r in range(rows):
            for c in range(cols):
                if view.power_pellets[r, c]:
                    cx = c * cell_size + cell_size // 2
                    cy = r * cell_size + cell_size // 2
                    pygame.draw.circle(surface, power_color, (cx, cy), power_rad)

        # 5. Ghosts
        for gid in sorted(view.ghost_positions.keys()):
            pos = view.ghost_positions[gid]
            cx = pos.col * cell_size + cell_size // 2
            cy = pos.row * cell_size + cell_size // 2
            ghost_rad = max(2, cell_size // 2 - 1)
            mode = view.ghost_modes.get(gid, GhostMode.SCATTER)

            if mode == GhostMode.DEAD:
                # Dead ghosts drawn distinctly as returning eyes
                eye_color = self.theme.dead_ghost_color
                eye_rad = max(1, ghost_rad // 3)
                pygame.draw.circle(surface, eye_color, (cx - eye_rad - 1, cy - 1), eye_rad)
                pygame.draw.circle(surface, eye_color, (cx + eye_rad + 1, cy - 1), eye_rad)
                # Small pupil
                pupil_color = (0, 0, 180)
                pupil_r = max(1, eye_rad // 2)
                pygame.draw.circle(surface, pupil_color, (cx - eye_rad - 1, cy - 1), pupil_r)
                pygame.draw.circle(surface, pupil_color, (cx + eye_rad + 1, cy - 1), pupil_r)
            else:
                if mode == GhostMode.FRIGHTENED:
                    # Flashing near expiry
                    is_flashing = (
                        view.frightened_timer <= self.theme.flash_threshold_ticks
                        and (view.tick // 2) % 2 == 1
                    )
                    color = (
                        self.theme.frightened_flash_color
                        if is_flashing
                        else self.theme.frightened_color
                    )
                else:
                    color = self.theme.get_ghost_color(gid)

                # Draw ghost body
                pygame.draw.circle(surface, color, (cx, cy), ghost_rad)

        # 6. Pacman
        p_pos = view.pacman_pos
        pcx = p_pos.col * cell_size + cell_size // 2
        pcy = p_pos.row * cell_size + cell_size // 2
        pac_rad = max(2, cell_size // 2 - 1)
        pac_color = self.theme.pacman_color

        # Fill base circular body (center pixel is yellow)
        pygame.draw.circle(surface, pac_color, (pcx, pcy), pac_rad)

        # Draw mouth wedge facing direction without modifying center pixel (pcx, pcy)
        p_dir = view.pacman_direction
        bg_col = self.theme.bg_color
        half_rad = pac_rad // 2
        if p_dir == Action.RIGHT:
            pts = [(pcx + 1, pcy), (pcx + pac_rad, pcy - half_rad), (pcx + pac_rad, pcy + half_rad)]
            pygame.draw.polygon(surface, bg_col, pts)
        elif p_dir == Action.LEFT:
            pts = [(pcx - 1, pcy), (pcx - pac_rad, pcy - half_rad), (pcx - pac_rad, pcy + half_rad)]
            pygame.draw.polygon(surface, bg_col, pts)
        elif p_dir == Action.UP:
            pts = [(pcx, pcy - 1), (pcx - half_rad, pcy - pac_rad), (pcx + half_rad, pcy - pac_rad)]
            pygame.draw.polygon(surface, bg_col, pts)
        elif p_dir == Action.DOWN:
            pts = [(pcx, pcy + 1), (pcx - half_rad, pcy + pac_rad), (pcx + half_rad, pcy + pac_rad)]
            pygame.draw.polygon(surface, bg_col, pts)

        # Convert surface to NumPy array: (width, height, 3) -> (height, width, 3)
        raw_surf = pygame.surfarray.array3d(surface)
        frame: np.ndarray = np.transpose(raw_surf, (1, 0, 2))

        # Pillow post-processing (grayscale and/or resize)
        if self.grayscale or self.output_size is not None:
            pil_img = Image.fromarray(frame)
            if self.grayscale:
                pil_img = pil_img.convert("L")
            if self.output_size is not None:
                target_h, target_w = self.output_size
                pil_img = pil_img.resize((target_w, target_h), resample=Image.Resampling.BOX)

            arr = np.array(pil_img, dtype=np.uint8)
            if self.grayscale:
                return arr[..., np.newaxis]
            return arr

        return frame
