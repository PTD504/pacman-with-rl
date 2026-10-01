"""Tests verifying the fluent MapBuilder API and equivalence to ASCII representations."""

import pytest

from pacman_engine.maps.builder import MapBuilder
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import MapValidationError


def test_builder_basic_construction_and_ascii_equivalence() -> None:
    builder = (
        MapBuilder(rows=5, cols=5, name="mini")
        .draw_border()
        .set_pacman_spawn(1, 1)
        .add_ghost_spawn(0, 3, 3)
        .add_pellet(1, 2)
        .add_pellet(1, 3)
        .add_power_pellet(3, 1)
        .add_door(2, 2)
    )
    gm = builder.build()

    expected_ascii = "; name: mini\n#####\n#P..#\n#_=_#\n#o_0#\n#####\n"
    assert gm.to_ascii(empty_char="_") == expected_ascii
    assert gm == GridMap.from_ascii(expected_ascii)


def test_builder_fill_pellets() -> None:
    builder = (
        MapBuilder(rows=5, cols=5)
        .draw_border()
        .set_pacman_spawn(1, 1)
        .add_ghost_spawn(0, 3, 3)
        .add_power_pellet(1, 3)
        .add_door(2, 2)
        .fill_pellets()
    )
    gm = builder.build()

    # Spawns, walls, doors, and power pellet must not have standard pellets
    assert not gm.pellets[1, 1]  # Pacman
    assert not gm.pellets[3, 3]  # Ghost
    assert not gm.pellets[1, 3]  # Power pellet
    assert not gm.pellets[2, 2]  # Door
    assert not gm.pellets[0, 0]  # Wall
    # Open floor should have pellets
    assert gm.pellets[1, 2]
    assert gm.pellets[2, 1]
    assert gm.pellets[3, 1]


def test_builder_missing_pacman_spawn() -> None:
    builder = MapBuilder(5, 5).draw_border().add_ghost_spawn(0, 2, 2).add_pellet(1, 1)
    with pytest.raises(MapValidationError, match="Map has no Pacman spawn"):
        builder.build()


def test_builder_bounds_checks() -> None:
    builder = MapBuilder(5, 5)
    with pytest.raises(IndexError):
        builder.set_wall(5, 0)
    with pytest.raises(IndexError):
        builder.set_pacman_spawn(-1, 0)
    with pytest.raises(IndexError):
        builder.add_ghost_spawn(0, 0, 5)


def test_builder_ghost_id_validation() -> None:
    builder = MapBuilder(5, 5)
    with pytest.raises(ValueError, match="Ghost id must be 0-9"):
        builder.add_ghost_spawn(10, 1, 1)


def test_builder_dimensions_validation() -> None:
    with pytest.raises(ValueError, match="Dimensions must be positive"):
        MapBuilder(0, 5)
