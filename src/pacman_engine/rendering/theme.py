"""Theme and color configuration for the Pacman headless renderer."""

from __future__ import annotations

from dataclasses import dataclass, field

DEFAULT_GHOST_COLORS: dict[int, tuple[int, int, int]] = {
    0: (255, 0, 0),  # Blinky (Red)
    1: (255, 184, 255),  # Pinky (Pink)
    2: (0, 255, 255),  # Inky (Cyan)
    3: (255, 184, 82),  # Clyde (Orange)
}

FALLBACK_PALETTE: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0),  # Green
    (180, 0, 255),  # Purple
    (255, 255, 128),  # Pale Yellow
    (0, 180, 255),  # Sky Blue
    (255, 128, 0),  # Amber
    (128, 255, 128),  # Mint
)


@dataclass(frozen=True)
class RenderTheme:
    """Visual style, color palette, and geometric dimensions for rendering frames.

    Attributes:
        cell_size: Square cell dimension in pixels (>= 4).
        bg_color: RGB background color.
        wall_color: RGB color for impassable wall tiles.
        door_color: RGB color for ghost spawn door tiles.
        pellet_color: RGB color for regular dots.
        pellet_radius: Pixel radius for regular pellets (defaults to cell_size // 8).
        power_pellet_color: RGB color for power energizers.
        power_pellet_radius: Pixel radius for power pellets (defaults to cell_size // 4).
        pacman_color: RGB color for Pacman.
        ghost_colors: Mapping from ghost ID to specific RGB colors.
        frightened_color: RGB color for ghosts in FRIGHTENED mode.
        frightened_flash_color: Secondary RGB flash color when frightened mode is expiring.
        flash_threshold_ticks: Ticks remaining below which frightened ghosts alternate colors.
        dead_ghost_color: RGB color representing returning ghost eyes / body.
        grid_lines: Whether to draw faint grid division lines.
        grid_line_color: RGB color for grid division lines.
    """

    cell_size: int = 16
    bg_color: tuple[int, int, int] = (0, 0, 0)
    wall_color: tuple[int, int, int] = (33, 33, 222)
    door_color: tuple[int, int, int] = (255, 184, 255)
    pellet_color: tuple[int, int, int] = (255, 184, 151)
    pellet_radius: int | None = None
    power_pellet_color: tuple[int, int, int] = (255, 184, 151)
    power_pellet_radius: int | None = None
    pacman_color: tuple[int, int, int] = (255, 255, 0)
    ghost_colors: dict[int, tuple[int, int, int]] = field(
        default_factory=lambda: dict(DEFAULT_GHOST_COLORS)
    )
    frightened_color: tuple[int, int, int] = (33, 33, 255)
    frightened_flash_color: tuple[int, int, int] = (255, 255, 255)
    flash_threshold_ticks: int = 10
    dead_ghost_color: tuple[int, int, int] = (200, 200, 255)
    grid_lines: bool = False
    grid_line_color: tuple[int, int, int] = (40, 40, 40)

    def __post_init__(self) -> None:
        if self.cell_size < 4:
            raise ValueError(f"cell_size must be >= 4, got {self.cell_size}")
        if self.pellet_radius is None:
            object.__setattr__(self, "pellet_radius", max(1, self.cell_size // 8))
        if self.power_pellet_radius is None:
            object.__setattr__(self, "power_pellet_radius", max(2, self.cell_size // 4))

    def get_ghost_color(self, ghost_id: int) -> tuple[int, int, int]:
        """Return the RGB color for a given ghost ID."""
        if ghost_id in self.ghost_colors:
            return self.ghost_colors[ghost_id]
        return FALLBACK_PALETTE[ghost_id % len(FALLBACK_PALETTE)]
