"""Functions to discover and load package-shipped built-in maps via importlib.resources."""

from __future__ import annotations

import importlib.resources

from pacman_engine.maps.grid_map import GridMap


def list_builtin_maps() -> list[str]:
    """Return a sorted list of names for all built-in maps shipped with the package."""
    data_pkg = importlib.resources.files("pacman_engine.maps.data")
    names: list[str] = []
    for item in data_pkg.iterdir():
        if item.name.endswith(".map") and item.is_file():
            # Strip .map suffix
            names.append(item.name[:-4])
    return sorted(names)


def load_builtin_map(name: str) -> GridMap:
    """Load, parse, and validate a built-in map by name.

    Args:
        name: Name of the built-in map (e.g. 'classic' or 'classic.map').

    Returns:
        The validated immutable GridMap.

    Raises:
        FileNotFoundError: If the map does not exist in the built-in map package.
    """
    map_name = name[:-4] if name.endswith(".map") else name
    filename = f"{map_name}.map"

    data_pkg = importlib.resources.files("pacman_engine.maps.data")
    map_ref = data_pkg.joinpath(filename)

    if not map_ref.is_file():
        available = ", ".join(list_builtin_maps())
        raise FileNotFoundError(f"Built-in map '{name}' not found. Available maps: [{available}]")

    content = map_ref.read_text(encoding="utf-8")
    return GridMap.from_ascii(content)
