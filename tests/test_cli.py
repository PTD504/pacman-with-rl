"""Tests for CLI tools: pacman-play, pacman-record, and pacman-map."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from PIL import Image

from pacman_engine.config import GameConfig, build_game
from pacman_engine.maps import cli as map_cli
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.play import play
from pacman_engine.recording import main as record_main
from pacman_engine.recording import record_episode


def test_pacman_map_list(capsys: pytest.CaptureFixture[str]) -> None:
    code = map_cli.main(["list"])
    assert code is None or code == 0
    captured = capsys.readouterr()
    assert "classic" in captured.out
    assert "small" in captured.out


def test_pacman_map_preview(tmp_path: Path) -> None:
    out_file = tmp_path / "small_preview.png"
    code = map_cli.main(["preview", "small", "--out", str(out_file), "--cell-size", "16"])
    assert code is None or code == 0
    assert out_file.is_file()

    img = Image.open(out_file)
    assert img.size[0] > 0
    assert img.size[1] > 0


def test_pacman_map_generate(tmp_path: Path) -> None:
    out_file = tmp_path / "generated.map"
    code = map_cli.main(
        [
            "generate",
            "--rows",
            "15",
            "--cols",
            "15",
            "--ghosts",
            "2",
            "--seed",
            "42",
            "--out",
            str(out_file),
        ]
    )
    assert code is None or code == 0
    assert out_file.is_file()

    grid_map = GridMap.from_file(out_file)
    assert grid_map.rows == 15
    assert grid_map.cols == 15
    assert len(grid_map.ghost_spawns) == 2


def test_pacman_map_validate(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # 1. Valid built-in map
    code = map_cli.main(["validate", "small"])
    assert code is None or code == 0
    out = capsys.readouterr().out
    assert "VALID" in out

    # 2. Invalid map file
    invalid_file = tmp_path / "invalid.map"
    invalid_file.write_text("###\n#P#\n###", encoding="utf-8")  # No pellets
    with pytest.raises(SystemExit) as exc_info:
        map_cli.main(["validate", str(invalid_file)])
    assert exc_info.value.code == 1


def test_pacman_record_gif(tmp_path: Path) -> None:
    out_gif = tmp_path / "test_run.gif"
    cfg = GameConfig(map="small")
    game = build_game(cfg, seed=1)

    stats = record_episode(game, out_gif, fps=10, cell_size=16, max_steps=8)
    assert out_gif.is_file()
    assert stats["total_steps"] == 8
    # 1 initial frame + 8 steps = 9 frames
    assert stats["total_frames"] == 9

    with Image.open(out_gif) as img:
        assert getattr(img, "n_frames", 1) == 9


def test_pacman_record_png_sequence(tmp_path: Path) -> None:
    out_dir = tmp_path / "frames"
    cfg = GameConfig(map="small")
    game = build_game(cfg, seed=1)

    stats = record_episode(game, out_dir, fps=10, cell_size=16, max_steps=5)
    assert stats["total_steps"] == 5
    assert stats["total_frames"] == 6

    saved_frames = sorted(out_dir.glob("*.png"))
    assert len(saved_frames) == 6


def test_pacman_record_cli(tmp_path: Path) -> None:
    out_gif = tmp_path / "cli_run.gif"
    record_main(
        [
            "--map",
            "small",
            "--controller",
            "pacman_random",
            "--max-steps",
            "5",
            "--out",
            str(out_gif),
        ]
    )
    assert out_gif.is_file()
    with Image.open(out_gif) as img:
        assert getattr(img, "n_frames", 1) == 6


def test_pacman_play_smoke_with_dummy_video() -> None:
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame

    if not pygame.get_init():
        pygame.init()

    # Post synthetic events into pygame queue
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RIGHT))
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))  # pause
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))  # unpause
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))  # reset
    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_q))  # quit

    # Run play with debug=True for 8 frames
    play(
        map_name="small",
        max_frames=8,
        debug=True,
        fps=60,
    )
