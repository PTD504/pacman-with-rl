"""Tests verifying that all built-in maps load, validate, and conform to specifications."""

import pytest

from pacman_engine.maps.loader import list_builtin_maps, load_builtin_map


def test_list_builtin_maps() -> None:
    expected_maps = ["classic", "corridor", "open_arena", "small", "tiny"]
    available = list_builtin_maps()
    for exp in expected_maps:
        assert exp in available


def test_every_builtin_map_loads_and_validates() -> None:
    for map_name in list_builtin_maps():
        gm = load_builtin_map(map_name)
        gm.validate()
        assert gm.rows > 0
        assert gm.cols > 0
        assert gm.pacman_spawn is not None
        assert (gm.pellets.sum() + gm.power_pellets.sum()) >= 1


def test_load_builtin_map_with_extension() -> None:
    gm1 = load_builtin_map("tiny")
    gm2 = load_builtin_map("tiny.map")
    assert gm1 == gm2


def test_load_nonexistent_builtin_map() -> None:
    with pytest.raises(FileNotFoundError, match="not found"):
        load_builtin_map("non_existent_map_name")
