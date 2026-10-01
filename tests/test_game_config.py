"""Tests for GameConfig, serialization round-trips, validation, and error messages."""

from pathlib import Path

import pytest

from pacman_engine.config import GameConfig, GhostConfig, GhostEntry, RulesConfig, build_game
from pacman_engine.controllers.registry import ControllerSpec
from pacman_engine.types import Position


def test_game_config_from_dict_and_to_dict_equality() -> None:
    cfg = GameConfig(
        map="classic",
        rules=RulesConfig(max_steps=500, pacman_lives=3),
        pacman=ControllerSpec("greedy_pellet", {"avoid_ghost_radius": 2}),
        ghosts={
            0: GhostEntry(
                config=GhostConfig(ghost_id=0, move_period=1, scatter_target=Position(0, 27)),
                controller=ControllerSpec("chase"),
            ),
            1: GhostEntry(
                config=GhostConfig(ghost_id=1, move_period=2),
                controller=ControllerSpec("ambush", {"lookahead": 3}),
            ),
        },
    )

    d = cfg.to_dict()
    cfg2 = GameConfig.from_dict(d)
    assert cfg == cfg2


@pytest.mark.parametrize(
    "filename",
    ["classic.yaml", "aggressive_speeds.yaml", "patrol_custom.yaml"],
)
def test_example_configs_roundtrip(filename: str) -> None:
    filepath = Path("configs/examples") / filename
    assert filepath.is_file()

    cfg1 = GameConfig.from_yaml(filepath)
    d = cfg1.to_dict()
    cfg2 = GameConfig.from_dict(d)
    assert cfg1 == cfg2

    # to_yaml roundtrip
    yaml_text = cfg1.to_yaml()
    cfg3 = GameConfig.from_yaml(yaml_text)
    assert cfg1 == cfg3


def test_invalid_config_missing_map() -> None:
    with pytest.raises(ValueError, match="missing required 'map' key"):
        GameConfig.from_dict({"rules": {}})


def test_invalid_config_empty_map() -> None:
    with pytest.raises(ValueError, match="map must be a non-empty string"):
        GameConfig(map="")


def test_invalid_config_nonexistent_map_name() -> None:
    with pytest.raises(FileNotFoundError, match="not a valid file path or built-in map"):
        GameConfig(map="completely_made_up_map_name")


def test_invalid_config_broken_inline_ascii() -> None:
    broken_ascii = "###\n#P\n###"  # Non-rectangular
    with pytest.raises(ValueError, match="Invalid inline ASCII map"):
        GameConfig(map=broken_ascii)


def test_invalid_config_unknown_pacman_controller() -> None:
    with pytest.raises(ValueError, match="Unknown Pacman controller 'super_pacman'"):
        GameConfig(map="classic", pacman=ControllerSpec("super_pacman"))


def test_invalid_config_unknown_ghost_controller() -> None:
    with pytest.raises(ValueError, match="Unknown ghost controller 'terminator'"):
        GameConfig(
            map="classic",
            ghosts={0: GhostEntry(GhostConfig(0), ControllerSpec("terminator"))},
        )


def test_invalid_config_out_of_range_ghost_id() -> None:
    with pytest.raises(ValueError, match="ghost_id must be between 0 and 9"):
        GhostConfig(ghost_id=12)


def test_build_game_ghost_id_not_on_map() -> None:
    # 'tiny' built-in map only has ghost 0
    cfg = GameConfig(
        map="tiny",
        ghosts={
            0: GhostEntry(GhostConfig(0), ControllerSpec("chase")),
            5: GhostEntry(GhostConfig(5), ControllerSpec("chase")),
        },
    )
    with pytest.raises(
        ValueError, match="Ghost ID 5 configured in GameConfig does not exist on map"
    ):
        build_game(cfg)
