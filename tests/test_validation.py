"""Tests verifying that each map validation rule fails with an informative error message."""

import pytest

from pacman_engine.maps.builder import MapBuilder
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import MapValidationError


def test_empty_map_error() -> None:
    with pytest.raises(MapValidationError, match="Map has no grid content"):
        GridMap.from_ascii("; name: empty\n")


def test_non_rectangular_grid() -> None:
    ascii_text = "#####\n#P..#\n#...\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "rectangular" in msg.lower()
    assert "row 2" in msg


def test_missing_pacman_spawn() -> None:
    ascii_text = "#####\n#...#\n#0..#\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "exactly one Pacman spawn" in msg
    assert "found 0" in msg


def test_multiple_pacman_spawns() -> None:
    ascii_text = "#####\n#P..#\n#..P#\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "exactly one Pacman spawn" in msg
    assert "found 2" in msg
    assert "row=1, col=1" in msg
    assert "row=2, col=3" in msg


def test_duplicate_ghost_spawns() -> None:
    ascii_text = "#####\n#P..#\n#0.0#\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "Duplicate ghost spawn id '0'" in msg
    assert "row 2" in msg


def test_no_pellets() -> None:
    ascii_text = "#####\n#P__#\n#0__#\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    assert "at least one pellet" in str(exc_info.value)


def test_unreachable_pellet_due_to_wall() -> None:
    ascii_text = "#######\n#P..#.#\n#...#.#\n#0..#.#\n#...#.#\n#...#o#\n#######\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "Pellet at (row=1, col=5) is unreachable by Pacman" in msg


def test_pellet_behind_ghost_door_unreachable_by_pacman() -> None:
    ascii_text = "#####\n#P=.#\n#0###\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "Pellet at (row=1, col=3) is unreachable by Pacman" in msg


def test_ghost_cannot_reach_pacman() -> None:
    ascii_text = "#######\n#P....#\n#######\n#0____#\n#######\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "Ghost spawn '0' at (row=3, col=1) cannot reach Pacman spawn at (row=1, col=1)" in msg


def test_unknown_character() -> None:
    ascii_text = "#####\n#P.X#\n#0..#\n#####\n"
    with pytest.raises(MapValidationError) as exc_info:
        GridMap.from_ascii(ascii_text)
    msg = str(exc_info.value)
    assert "Unknown character 'X'" in msg
    assert "row 1, col 3" in msg


def test_spawn_cell_holds_pellet_error() -> None:
    builder = MapBuilder(5, 5)
    builder.draw_border()
    builder.set_pacman_spawn(1, 1)
    builder.add_ghost_spawn(0, 3, 3)
    # Force pellet onto Pacman spawn
    builder.pellets[1, 1] = True
    with pytest.raises(MapValidationError) as exc_info:
        builder.build()
    msg = str(exc_info.value)
    assert "Pacman spawn cell at (row=1, col=1) cannot contain a pellet" in msg


def test_spawn_cell_holds_power_pellet_error() -> None:
    builder = MapBuilder(5, 5)
    builder.draw_border()
    builder.set_pacman_spawn(1, 1)
    builder.add_ghost_spawn(0, 3, 3)
    # Force power pellet onto ghost spawn
    builder.power_pellets[3, 3] = True
    with pytest.raises(MapValidationError) as exc_info:
        builder.build()
    msg = str(exc_info.value)
    assert "Ghost '0' spawn cell at (row=3, col=3) cannot contain a power pellet" in msg


def test_pacman_spawn_on_wall_error() -> None:
    builder = MapBuilder(5, 5)
    builder.draw_border()
    builder.set_pacman_spawn(1, 1)
    builder.add_ghost_spawn(0, 3, 3)
    builder.add_pellet(1, 2)
    # Force wall at Pacman spawn
    builder.walls[1, 1] = True
    with pytest.raises(MapValidationError) as exc_info:
        builder.build()
    assert "Pacman spawn at (row=1, col=1) is on a wall" in str(exc_info.value)


def test_ghost_spawn_on_wall_error() -> None:
    builder = MapBuilder(5, 5)
    builder.draw_border()
    builder.set_pacman_spawn(1, 1)
    builder.add_ghost_spawn(0, 3, 3)
    builder.add_pellet(1, 2)
    # Force wall at Ghost spawn
    builder.walls[3, 3] = True
    with pytest.raises(MapValidationError) as exc_info:
        builder.build()
    assert "Ghost spawn '0' at (row=3, col=3) is on a wall" in str(exc_info.value)
