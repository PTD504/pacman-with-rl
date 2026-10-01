"""Tests verifying exact round-trip serialization and deserialization of ASCII maps."""

from pacman_engine.maps.grid_map import GridMap
from pacman_engine.maps.loader import list_builtin_maps, load_builtin_map


def test_builtin_maps_ascii_roundtrip() -> None:
    """All built-in maps must round-trip exactly to their original ASCII text."""
    for map_name in list_builtin_maps():
        gm = load_builtin_map(map_name)
        ascii_repr = gm.to_ascii()
        gm_roundtrip = GridMap.from_ascii(ascii_repr)

        assert gm == gm_roundtrip
        assert gm_roundtrip.to_ascii() == ascii_repr


def test_ascii_roundtrip_with_headers() -> None:
    """Map with all supported headers round-trips exactly."""
    ascii_text = (
        "; name: custom_maze\n"
        "; description: A custom testing maze\n"
        "; wrap_horizontal: true\n"
        "; wrap_vertical: true\n"
        "#####\n"
        "#P.o#\n"
        "#0###\n"
        "#...#\n"
        "#####\n"
    )
    gm = GridMap.from_ascii(ascii_text)
    assert gm.name == "custom_maze"
    assert gm.description == "A custom testing maze"
    assert gm.wrap_horizontal is True
    assert gm.wrap_vertical is True
    assert gm.to_ascii() == ascii_text


def test_ascii_roundtrip_with_underscore_empty_cells() -> None:
    """Map using underscore '_' for empty cells round-trips preserving '_'."""
    ascii_text = "#######\n#P___o#\n#.###.#\n#_0=1_#\n#.###.#\n#o...o#\n#######\n"
    gm = GridMap.from_ascii(ascii_text)
    assert gm.empty_char == "_"
    assert gm.to_ascii() == ascii_text


def test_ascii_roundtrip_with_space_empty_cells() -> None:
    """Map using space ' ' for empty cells round-trips preserving ' '."""
    ascii_text = "#######\n#P   o#\n#.###.#\n# 0=1 #\n#.###.#\n#o...o#\n#######\n"
    gm = GridMap.from_ascii(ascii_text)
    assert gm.empty_char == " "
    assert gm.to_ascii() == ascii_text


def test_double_roundtrip_idempotency() -> None:
    """Repeated serialization and parsing is strictly idempotent."""
    ascii_text = "; name: tiny\n#######\n#o.P.o#\n#.###.#\n#.0=1.#\n#.###.#\n#o...o#\n#######\n"
    gm1 = GridMap.from_ascii(ascii_text)
    ascii1 = gm1.to_ascii()
    gm2 = GridMap.from_ascii(ascii1)
    ascii2 = gm2.to_ascii()
    assert ascii1 == ascii2
    assert gm1 == gm2
