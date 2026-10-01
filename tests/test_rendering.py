"""Tests for headless rendering, pixel color contracts, themes, and image generation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from PIL import Image

from pacman_engine.config import GameConfig, build_game
from pacman_engine.maps.loader import list_builtin_maps, load_builtin_map
from pacman_engine.rendering import Renderer, RenderTheme
from pacman_engine.types import Action, GhostMode, Position


def test_renderer_headless_environment() -> None:
    # Ensure headless dummy video driver works
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    theme = RenderTheme(cell_size=16)
    renderer = Renderer(theme=theme)
    assert renderer.theme.cell_size == 16


def test_renderer_output_dimensions() -> None:
    game_map = load_builtin_map("small")
    cfg = GameConfig(map="small")
    game = build_game(cfg, seed=1)

    s = 20
    renderer = Renderer(cell_size=s)
    frame = renderer.render(game.view)

    expected_h = game_map.rows * s
    expected_w = game_map.cols * s
    assert frame.shape == (expected_h, expected_w, 3)
    assert frame.dtype == np.uint8


def test_pixel_color_assertions_at_cell_centers() -> None:
    cfg = GameConfig(map="small")
    game = build_game(cfg, seed=10)
    view = game.view
    game_map = view.graph.grid_map

    theme = RenderTheme(cell_size=16)
    renderer = Renderer(theme=theme)
    frame = renderer.render(view)

    s = theme.cell_size

    def center_pixel(r: int, c: int) -> tuple[int, int, int]:
        cy = r * s + s // 2
        cx = c * s + s // 2
        return tuple(int(x) for x in frame[cy, cx])

    # 1. Wall pixel center
    wall_cells = [
        (r, c) for r in range(game_map.rows) for c in range(game_map.cols) if game_map.walls[r, c]
    ]
    assert wall_cells
    wr, wc = wall_cells[0]
    assert center_pixel(wr, wc) == theme.wall_color

    # 2. Pellet pixel center (cell with pellet and no entity)
    occupied = {view.pacman_pos} | set(view.ghost_positions.values())
    pellet_cells = [
        (r, c)
        for r in range(game_map.rows)
        for c in range(game_map.cols)
        if view.pellets[r, c] and Position(r, c) not in occupied
    ]
    assert pellet_cells
    pr, pc = pellet_cells[0]
    assert center_pixel(pr, pc) == theme.pellet_color

    # 3. Pacman pixel center
    p_pos = view.pacman_pos
    assert center_pixel(p_pos.row, p_pos.col) == theme.pacman_color

    # 4. Normal ghost pixel center
    for gid, g_pos in view.ghost_positions.items():
        if view.ghost_modes[gid] in (GhostMode.SCATTER, GhostMode.CHASE):
            expected_ghost_col = theme.get_ghost_color(gid)
            assert center_pixel(g_pos.row, g_pos.col) == expected_ghost_col

    # 5. Frightened ghost pixel center
    # Trigger power pellet eaten or manually set frightened mode on state for assertion
    game._frightened_timer = 20
    for gid in game._ghosts:
        game._ghosts[gid].mode = GhostMode.FRIGHTENED

    frightened_view = game.view
    frightened_frame = renderer.render(frightened_view)
    for gid, g_pos in frightened_view.ghost_positions.items():
        cy = g_pos.row * s + s // 2
        cx = g_pos.col * s + s // 2
        color = tuple(int(x) for x in frightened_frame[cy, cx])
        assert color == theme.frightened_color


def test_pacman_directions_preserve_center_pixel() -> None:
    cfg = GameConfig(map="tiny")
    game = build_game(cfg, seed=1)
    s = 16
    theme = RenderTheme(cell_size=s)
    renderer = Renderer(theme=theme)

    pr, pc = game.view.pacman_pos.row, game.view.pacman_pos.col
    cy = pr * s + s // 2
    cx = pc * s + s // 2

    for action in (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT, Action.STAY):
        game._pacman.direction = action
        frame = renderer.render(game.view)
        pixel = tuple(int(x) for x in frame[cy, cx])
        assert pixel == theme.pacman_color, f"Pacman center pixel altered for direction {action}"


def test_dead_ghost_rendering() -> None:
    cfg = GameConfig(map="tiny")
    game = build_game(cfg, seed=1)
    s = 16
    theme = RenderTheme(cell_size=s)
    renderer = Renderer(theme=theme)

    for gid in game._ghosts:
        game._ghosts[gid].mode = GhostMode.DEAD

    frame = renderer.render(game.view)
    assert frame.shape == (game.map.rows * s, game.map.cols * s, 3)


def test_grayscale_and_resize_options() -> None:
    cfg = GameConfig(map="small")
    game = build_game(cfg, seed=1)
    view = game.view

    # Grayscale
    gray_renderer = Renderer(grayscale=True, cell_size=16)
    gray_frame = gray_renderer.render(view)
    assert gray_frame.shape == (game.map.rows * 16, game.map.cols * 16, 1)

    # Resize
    resize_renderer = Renderer(output_size=(64, 48), cell_size=16)
    resized_frame = resize_renderer.render(view)
    assert resized_frame.shape == (64, 48, 3)

    # Grayscale + Resize
    combo_renderer = Renderer(grayscale=True, output_size=(84, 84))
    combo_frame = combo_renderer.render(view)
    assert combo_frame.shape == (84, 84, 1)


def test_render_all_builtin_maps_to_png() -> None:
    images_dir = Path("docs/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    builtin_maps = list_builtin_maps()
    renderer = Renderer(cell_size=20)

    for map_name in builtin_maps:
        cfg = GameConfig(map=map_name)
        game = build_game(cfg, seed=42)
        frame = renderer.render(game.view)

        png_path = images_dir / f"{map_name}.png"
        img = Image.fromarray(frame)
        img.save(png_path)

        assert png_path.is_file()
        assert png_path.stat().st_size > 0

        # Reload with Pillow to verify valid PNG image
        loaded = Image.open(png_path)
        assert loaded.size == (game.map.cols * 20, game.map.rows * 20)
