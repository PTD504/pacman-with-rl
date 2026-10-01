"""RGB frame state observation builder wrapping Renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from pacman_engine.observations.base import (
    ObservationBuilder,
    ObservationSpec,
    register_observation,
)
from pacman_engine.rendering.renderer import Renderer
from pacman_engine.rendering.theme import RenderTheme

if TYPE_CHECKING:
    from pacman_engine.config import RulesConfig
    from pacman_engine.maps.grid_map import GridMap
    from pacman_engine.state import GameView


@register_observation("rgb")
class RgbObservationBuilder(ObservationBuilder):
    """Encodes game state into rendered pixel frames (RGB or Grayscale).

    Wraps the headless Renderer to produce visual observations suitable for Vision RL models.

    Parameters:
        theme: Optional RenderTheme instance or dict of theme parameters.
        cell_size: Optional integer cell size in pixels.
        grayscale: If True, outputs single-channel frame (H, W, 1) or (1, H, W).
        output_size: Optional (height, width) or int to resize frames via area interpolation.
        channels_first: If True, returns (C, H, W); if False, (H, W, C). Defaults to False.
    """

    def __init__(
        self,
        theme: RenderTheme | dict[str, Any] | None = None,
        cell_size: int | None = None,
        grayscale: bool = False,
        output_size: tuple[int, int] | int | None = None,
        channels_first: bool = False,
    ) -> None:
        if isinstance(theme, dict):
            render_theme = RenderTheme(**theme)
        else:
            render_theme = theme

        self.renderer = Renderer(
            theme=render_theme,
            cell_size=cell_size,
            grayscale=grayscale,
            output_size=output_size,
        )
        self.channels_first = bool(channels_first)

    def spec(self, game_map: GridMap, rules: RulesConfig) -> ObservationSpec:
        """Derive the static ObservationSpec from map dimensions and rendering configuration."""
        if self.renderer.output_size is not None:
            h, w = self.renderer.output_size
        else:
            h = game_map.rows * self.renderer.theme.cell_size
            w = game_map.cols * self.renderer.theme.cell_size

        c = 1 if self.renderer.grayscale else 3
        shape = (c, h, w) if self.channels_first else (h, w, c)

        return ObservationSpec(
            shape=shape,
            dtype=np.dtype(np.uint8),
            low=0,
            high=255,
            feature_names=None,
        )

    def build(self, view: GameView) -> np.ndarray:
        """Render GameView to a uint8 NumPy image array."""
        frame = self.renderer.render(view)
        if self.channels_first:
            frame = np.transpose(frame, (2, 0, 1))
        return frame
