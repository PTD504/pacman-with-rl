"""Command-line interface for Pacman map management: list, preview, generate, validate."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image

from pacman_engine.engine import PacmanGame
from pacman_engine.maps.generator import generate_random_map
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.maps.loader import list_builtin_maps, load_builtin_map
from pacman_engine.rendering.renderer import Renderer
from pacman_engine.rendering.theme import RenderTheme
from pacman_engine.types import MapValidationError


def _load_map_target(target: str | Path) -> GridMap:
    """Load a GridMap from a file path or built-in map name."""
    p = Path(target)
    if p.is_file():
        return GridMap.from_file(p)
    return load_builtin_map(str(target))


def cmd_list(args: argparse.Namespace) -> int:
    """List all available built-in maps."""
    maps = list_builtin_maps()
    print("Available built-in maps:")
    for m in maps:
        print(f"  - {m}")
    return 0


def cmd_preview(args: argparse.Namespace) -> int:
    """Render a static image preview of a map layout."""
    try:
        grid_map = _load_map_target(args.map)
    except Exception as e:
        print(f"Error loading map '{args.map}': {e}", file=sys.stderr)
        return 1

    theme = RenderTheme(cell_size=args.cell_size)
    renderer = Renderer(theme=theme, cell_size=args.cell_size)
    game = PacmanGame(game_map=grid_map)
    frame = renderer.render(game.view)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(frame)
    img.save(out_path)
    print(f"Map preview ({grid_map.rows}x{grid_map.cols}) saved to {out_path}")
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    """Procedurally generate a new valid Pacman map file."""
    try:
        grid_map = generate_random_map(
            rows=args.rows,
            cols=args.cols,
            num_ghosts=args.ghosts,
            num_power_pellets=args.power_pellets,
            loop_fraction=args.loop_fraction,
            symmetric=args.symmetric,
            seed=args.seed,
        )
    except Exception as e:
        print(f"Error generating map: {e}", file=sys.stderr)
        return 1

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(grid_map.to_ascii(), encoding="utf-8")
    print(
        f"Generated map ({grid_map.rows}x{grid_map.cols}, {len(grid_map.ghost_spawns)} ghosts, "
        f"{grid_map.total_pellets} pellets) saved to {out_path}"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Validate a map against all design rules and topological invariants."""
    try:
        grid_map = _load_map_target(args.path)
        grid_map.validate()
        print(
            f"VALID: Map '{args.path}' passed all validation checks "
            f"({grid_map.rows}x{grid_map.cols}, {len(grid_map.ghost_spawns)} ghosts, "
            f"{grid_map.total_pellets} pellets)."
        )
        return 0
    except MapValidationError as e:
        print(f"INVALID: Map validation failed for '{args.path}':\n  {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"ERROR: Could not load map '{args.path}': {e}", file=sys.stderr)
        return 1


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for pacman-map CLI."""
    parser = argparse.ArgumentParser(
        prog="pacman-map",
        description="Inspect, preview, generate, and validate Pacman maps.",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # list
    p_list = subparsers.add_parser("list", help="List all built-in maps.")
    p_list.set_defaults(func=cmd_list)

    # preview
    p_prev = subparsers.add_parser("preview", help="Render a static image preview of a map.")
    p_prev.add_argument("map", type=str, help="Name of built-in map or path to .map file.")
    p_prev.add_argument(
        "--out",
        type=str,
        default="map_preview.png",
        help="Target output image path (default: map_preview.png).",
    )
    p_prev.add_argument(
        "--cell-size",
        type=int,
        default=20,
        help="Cell size in pixels (default: 20).",
    )
    p_prev.set_defaults(func=cmd_preview)

    # generate
    p_gen = subparsers.add_parser("generate", help="Procedurally generate a random valid map.")
    p_gen.add_argument("--rows", type=int, required=True, help="Height of map (odd integer >= 7).")
    p_gen.add_argument("--cols", type=int, required=True, help="Width of map (odd integer >= 7).")
    p_gen.add_argument("--seed", type=int, default=None, help="Random seed for generation.")
    p_gen.add_argument("--ghosts", type=int, default=4, help="Number of ghost spawns (default: 4).")
    p_gen.add_argument(
        "--power-pellets", type=int, default=4, help="Number of energizers/power pellets."
    )
    p_gen.add_argument(
        "--loop-fraction",
        type=float,
        default=0.2,
        help="Fraction of walls to remove for cycles (0.0 to 1.0).",
    )
    p_gen.add_argument(
        "--symmetric",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enforce horizontal mirror symmetry (default: True).",
    )
    p_gen.add_argument("--out", type=str, required=True, help="Target file path to save ASCII map.")
    p_gen.set_defaults(func=cmd_generate)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate an ASCII map file.")
    p_val.add_argument("path", type=str, help="Path to .map file or name of built-in map.")
    p_val.set_defaults(func=cmd_validate)

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for pacman-map."""
    parser = build_parser()
    args = parser.parse_args(argv)
    code = args.func(args)
    if code != 0:
        sys.exit(code)


if __name__ == "__main__":
    main(sys.argv[1:])
