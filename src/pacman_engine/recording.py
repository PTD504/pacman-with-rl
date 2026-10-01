"""Episode recording and trajectory export to animated GIF or PNG sequence."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image

from pacman_engine.config import GameConfig, build_game
from pacman_engine.controllers.base import Controller
from pacman_engine.controllers.pacman import ExternalController, GreedyPelletController
from pacman_engine.controllers.registry import ControllerSpec, create_controller
from pacman_engine.rendering.renderer import Renderer
from pacman_engine.rendering.theme import RenderTheme
from pacman_engine.types import Outcome

if TYPE_CHECKING:
    from pacman_engine.engine import PacmanGame


def record_episode(
    game: PacmanGame,
    path: str | Path,
    fps: int = 15,
    cell_size: int = 20,
    max_steps: int | None = None,
    controller: Controller | str | None = None,
    theme: RenderTheme | None = None,
    renderer: Renderer | None = None,
) -> dict[str, Any]:
    """Execute an episode on the provided game and save rendered frames.

    Args:
        game: Configured PacmanGame instance.
        path: Target file path (ending in .gif) or directory for PNG sequence.
        fps: Frames per second for the recording animation.
        cell_size: Pixel dimension for each grid cell if renderer is not provided.
        max_steps: Optional hard cutoff on the number of steps.
        controller: Optional Pacman controller instance or registered name.
        theme: Optional RenderTheme for custom colors/dimensions.
        renderer: Optional custom Renderer instance.

    Returns:
        A dictionary with episode metrics: total_frames, total_steps, score, outcome, path.
    """
    if controller is not None:
        if isinstance(controller, str):
            game.pacman_controller = create_controller(controller)
        else:
            game.pacman_controller = controller
    elif game.pacman_controller is None or isinstance(game.pacman_controller, ExternalController):
        game.pacman_controller = GreedyPelletController()

    if renderer is None:
        t = theme if theme is not None else RenderTheme(cell_size=cell_size)
        renderer = Renderer(theme=t, cell_size=cell_size)

    out_path = Path(path)
    frames: list[np.ndarray] = []

    # Initial frame at tick 0
    frames.append(renderer.render(game.view))

    step_count = 0
    while game._outcome == Outcome.RUNNING:
        if max_steps is not None and step_count >= max_steps:
            break
        game.step()
        frames.append(renderer.render(game.view))
        step_count += 1

    pil_images = [Image.fromarray(f) for f in frames]
    is_gif = out_path.suffix.lower() == ".gif"

    if is_gif:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Duration per frame in milliseconds
        frame_duration = max(20, int(1000 / fps))
        if pil_images:
            pil_images[0].save(
                out_path,
                save_all=True,
                append_images=pil_images[1:],
                duration=frame_duration,
                loop=0,
            )
    else:
        # PNG sequence
        if out_path.suffix.lower() == ".png" and ("{" in out_path.name or "%" in out_path.name):
            out_dir = out_path.parent
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(pil_images):
                if "{" in out_path.name:
                    fname = out_path.name.format(idx)
                else:
                    fname = out_path.name % idx
                img.save(out_dir / fname)
        else:
            # Treat as directory or single base name
            if out_path.suffix.lower() == ".png":
                out_dir = out_path.parent / out_path.stem
            else:
                out_dir = out_path
            out_dir.mkdir(parents=True, exist_ok=True)
            for idx, img in enumerate(pil_images):
                img.save(out_dir / f"frame_{idx:05d}.png")

    return {
        "total_frames": len(frames),
        "total_steps": step_count,
        "score": game.view.score,
        "outcome": game._outcome.name,
        "path": str(out_path),
    }


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for pacman-record."""
    parser = argparse.ArgumentParser(
        prog="pacman-record",
        description="Run a Pacman episode and record video to GIF or PNG sequence.",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="classic",
        help="Name of built-in map or path to .map file (default: classic).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML game configuration file.",
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="greedy_pellet",
        help="Pacman controller to evaluate (default: greedy_pellet).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master random seed for reproducibility.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Playback speed in frames per second (default: 15).",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=20,
        help="Pixel cell size for rendering (default: 20).",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum simulation steps before terminating.",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="episode.gif",
        help=(
            "Output path for recording: .gif file or directory for PNG sequence "
            "(default: episode.gif)."
        ),
    )

    args = parser.parse_args(argv)

    if args.config is not None:
        cfg = GameConfig.from_yaml(args.config)
        if args.map != "classic":
            cfg = GameConfig(
                map=args.map,
                rules=cfg.rules,
                pacman=cfg.pacman,
                ghosts=cfg.ghosts,
                observation=cfg.observation,
            )
        if args.controller != "greedy_pellet" or cfg.pacman.name == "manual":
            cfg = GameConfig(
                map=cfg.map,
                rules=cfg.rules,
                pacman=ControllerSpec(name=args.controller),
                ghosts=cfg.ghosts,
                observation=cfg.observation,
            )
    else:
        cfg = GameConfig(
            map=args.map,
            pacman=ControllerSpec(name=args.controller),
        )

    game = build_game(cfg, seed=args.seed)
    print(f"Recording episode on map '{cfg.map}' with controller '{cfg.pacman.name}'...")
    stats = record_episode(
        game=game,
        path=args.out,
        fps=args.fps,
        cell_size=args.cell_size,
        max_steps=args.max_steps,
    )
    print(
        f"Episode recorded: {stats['total_steps']} steps, outcome: {stats['outcome']}, "
        f"score: {stats['score']}, saved to: {stats['path']}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
