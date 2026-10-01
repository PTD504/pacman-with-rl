"""Tests for ControllerSpec and the controller registry."""

from dataclasses import FrozenInstanceError

import pytest

from pacman_engine.config import GameConfig, build_game
from pacman_engine.controllers.base import BaseController
from pacman_engine.controllers.registry import (
    ControllerSpec,
    create_controller,
    list_controllers,
    register_controller,
)
from pacman_engine.types import Action, Position


def test_list_controllers() -> None:
    controllers = list_controllers()
    assert isinstance(controllers, list)
    # Check built-in controllers are registered
    expected = [
        "ambush",
        "chase",
        "external",
        "flank",
        "greedy_pellet",
        "manual",
        "patrol",
        "random",
        "scripted",
    ]
    for exp in expected:
        assert exp in controllers
    assert controllers == sorted(controllers)


def test_controller_spec_frozen_and_validation() -> None:
    spec = ControllerSpec(name="chase", params={"epsilon": 0.1})
    assert spec.name == "chase"
    assert spec.params == {"epsilon": 0.1}

    # Frozen: modifying attributes raises FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        spec.name = "ambush"  # type: ignore[misc]

    # Empty name raises ValueError
    with pytest.raises(ValueError, match="non-empty string"):
        ControllerSpec(name="")

    # Non-dict params raises ValueError
    with pytest.raises(ValueError, match="params must be a dict"):
        ControllerSpec(name="chase", params="invalid")  # type: ignore[arg-type]

    # Serialization round-trip
    d = spec.to_dict()
    spec2 = ControllerSpec.from_dict(d)
    assert spec == spec2

    # from_dict with string
    spec3 = ControllerSpec.from_dict("random")
    assert spec3.name == "random"
    assert spec3.params == {}


def test_create_controller_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown controller 'non_existent'"):
        create_controller("non_existent")


def test_create_controller_bad_params() -> None:
    with pytest.raises(TypeError, match="Unexpected parameter"):
        create_controller("chase", non_existent_arg=123)


def test_user_defined_controller_registered_and_used_in_yaml() -> None:
    """Test registering a custom controller inside a test and running it through YAML."""

    # 1. Define custom controller class inside the test
    @register_controller("test_always_up")
    class AlwaysUpController(BaseController):
        def __init__(self, repeat_count: int = 1) -> None:
            self.repeat_count = repeat_count

        def act(self, view, entity_id, rng) -> Action:
            return Action.UP

    assert "test_always_up" in list_controllers()

    # 2. Instantiate through create_controller
    instance = create_controller("test_always_up", repeat_count=3)
    assert isinstance(instance, AlwaysUpController)
    assert instance.repeat_count == 3

    # 3. Use in YAML configuration
    yaml_text = """
map: |
  #######
  #.....#
  #.P.0.#
  #.....#
  #######
pacman:
  controller:
    name: test_always_up
    params:
      repeat_count: 2
ghosts:
  - ghost_id: 0
    controller: chase
"""
    config = GameConfig.from_yaml(yaml_text)
    assert config.pacman.name == "test_always_up"
    assert config.pacman.params == {"repeat_count": 2}

    # 4. Build game and verify Pacman executes Action.UP from custom controller
    game = build_game(config, seed=42)
    res = game.step()  # Step with pacman_action=None queries pacman_controller
    assert game._pacman.pos == Position(1, 2)  # Moved UP from (2, 2) to (1, 2)
    assert not res.terminated
