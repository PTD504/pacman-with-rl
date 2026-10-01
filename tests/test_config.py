"""Tests for game configuration and rules dataclasses."""

import pytest

from pacman_engine.config import GhostConfig, RulesConfig
from pacman_engine.types import GhostMode, Position


def test_ghost_config_defaults_and_validation() -> None:
    cfg = GhostConfig(ghost_id=0)
    assert cfg.ghost_id == 0
    assert cfg.speed == 1.0
    assert cfg.move_period is None
    assert cfg.start_delay == 0
    assert cfg.scatter_target is None

    # Invalid ghost_id
    with pytest.raises(ValueError, match="ghost_id"):
        GhostConfig(ghost_id=-1)
    with pytest.raises(ValueError, match="ghost_id"):
        GhostConfig(ghost_id=10)

    # Invalid move_period
    with pytest.raises(ValueError, match="move_period"):
        GhostConfig(ghost_id=0, move_period=0)

    # Mutual exclusivity of move_period and speed
    with pytest.raises(ValueError, match="both 'move_period' and 'speed'"):
        GhostConfig(ghost_id=0, move_period=2, speed=0.8)

    # Invalid speed
    with pytest.raises(ValueError, match="speed"):
        GhostConfig(ghost_id=0, speed=0.0)
    with pytest.raises(ValueError, match="speed"):
        GhostConfig(ghost_id=0, speed=1.5)

    # Invalid start_delay
    with pytest.raises(ValueError, match="start_delay"):
        GhostConfig(ghost_id=0, start_delay=-1)

    # Convert tuple scatter_target to Position
    cfg2 = GhostConfig(ghost_id=1, scatter_target=(5, 10))  # type: ignore[arg-type]
    assert cfg2.scatter_target == Position(5, 10)


def test_ghost_config_from_dict() -> None:
    data = {
        "ghost_id": 2,
        "move_period": 2,
        "start_delay": 5,
        "scatter_target": (0, 0),
    }
    cfg = GhostConfig.from_dict(data)
    assert cfg.ghost_id == 2
    assert cfg.move_period == 2
    assert cfg.speed is None
    assert cfg.start_delay == 5
    assert cfg.scatter_target == Position(0, 0)


def test_rules_config_defaults_and_validation() -> None:
    rules = RulesConfig()
    assert rules.max_steps == 1000
    assert rules.pacman_lives == 3
    assert rules.pellet_score == 10
    assert rules.power_pellet_score == 50
    assert rules.ghost_score_sequence == (200, 400, 800, 1600)
    assert rules.frightened_duration == 40
    assert rules.frightened_speed_factor == 1.0
    assert rules.ghost_respawn_delay == 10
    assert rules.mode_schedule == ((GhostMode.SCATTER, 20), (GhostMode.CHASE, 60))
    assert not rules.random_pacman_start
    assert rules.invalid_action_policy == "stay"

    # Validation errors
    with pytest.raises(ValueError, match="max_steps"):
        RulesConfig(max_steps=0)
    with pytest.raises(ValueError, match="frightened_speed_factor"):
        RulesConfig(frightened_speed_factor=0.0)
    with pytest.raises(ValueError, match="frightened_speed_factor"):
        RulesConfig(frightened_speed_factor=1.2)

    with pytest.raises(ValueError, match="pacman_lives"):
        RulesConfig(pacman_lives=0)
    with pytest.raises(ValueError, match="pellet_score"):
        RulesConfig(pellet_score=-1)
    with pytest.raises(ValueError, match="power_pellet_score"):
        RulesConfig(power_pellet_score=-1)
    with pytest.raises(ValueError, match="ghost_score_sequence"):
        RulesConfig(ghost_score_sequence=())
    with pytest.raises(ValueError, match="frightened_duration"):
        RulesConfig(frightened_duration=-1)
    with pytest.raises(ValueError, match="ghost_respawn_delay"):
        RulesConfig(ghost_respawn_delay=-1)
    with pytest.raises(ValueError, match="mode_schedule"):
        RulesConfig(mode_schedule=())
    with pytest.raises(ValueError, match="invalid_action_policy"):
        RulesConfig(invalid_action_policy="ignore")  # type: ignore[arg-type]


def test_rules_config_from_dict() -> None:
    data = {
        "max_steps": 500,
        "pacman_lives": 3,
        "pellet_score": 15,
        "power_pellet_score": 100,
        "ghost_score_sequence": [100, 200, 400],
        "frightened_duration": 30,
        "ghost_respawn_delay": 5,
        "mode_schedule": [("scatter", 10), ("chase", 50)],
        "random_pacman_start": True,
        "invalid_action_policy": "raise",
    }
    rules = RulesConfig.from_dict(data)
    assert rules.max_steps == 500
    assert rules.pacman_lives == 3
    assert rules.pellet_score == 15
    assert rules.power_pellet_score == 100
    assert rules.ghost_score_sequence == (100, 200, 400)
    assert rules.frightened_duration == 30
    assert rules.ghost_respawn_delay == 5
    assert rules.mode_schedule == ((GhostMode.SCATTER, 10), (GhostMode.CHASE, 50))
    assert rules.random_pacman_start is True
    assert rules.invalid_action_policy == "raise"
