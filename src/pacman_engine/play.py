"""Interactive game runner and desktop viewer using pygame-ce with decoupled simulation."""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pacman_engine.config import GameConfig, GhostConfig, GhostEntry, RulesConfig, build_game
from pacman_engine.controllers.pacman import ManualController
from pacman_engine.controllers.registry import ControllerSpec
from pacman_engine.rendering.renderer import Renderer
from pacman_engine.rendering.theme import RenderTheme
from pacman_engine.types import Action, Outcome

if TYPE_CHECKING:
    from pacman_engine.engine import PacmanGame


def _init_pygame() -> Any:
    """Lazily import and initialize pygame."""
    import pygame

    if not pygame.get_init():
        pygame.init()
    if not pygame.font.get_init():
        pygame.font.init()
    return pygame


class Clock:
    """Abstract clock interface for decoupling frame timing."""

    def tick(self, fps: float) -> float:
        """Wait for the next frame and return elapsed seconds since last call."""
        raise NotImplementedError

    def get_time(self) -> float:
        """Return current monotonic time in seconds."""
        raise NotImplementedError


class PygameClock(Clock):
    """Pygame-based real-time clock implementation."""

    def __init__(self, pygame: Any) -> None:
        self._clock = pygame.time.Clock()
        self._pygame = pygame

    def tick(self, fps: float) -> float:
        ms = self._clock.tick(int(fps))
        return ms / 1000.0

    def get_time(self) -> float:
        return self._pygame.time.get_ticks() / 1000.0


class FakeClock(Clock):
    """Test clock for deterministic and mock testing."""

    def __init__(self, step_seconds: float = 1.0 / 60.0) -> None:
        self.step_seconds = step_seconds
        self.current_time = 0.0

    def tick(self, fps: float) -> float:
        dt = 1.0 / fps if fps > 0 else self.step_seconds
        self.current_time += dt
        return dt

    def get_time(self) -> float:
        return self.current_time


def play(
    game: PacmanGame | None = None,
    map_name: str = "classic",
    config_path: str | Path | None = None,
    seed: int | None = None,
    fps: int = 60,
    tps: int = 8,
    cell_size: int = 24,
    lives: int | None = None,
    autoplay: str | None = None,
    debug: bool = False,
    max_frames: int | None = None,
    max_ticks: int | None = None,
    clock: Clock | None = None,
) -> None:
    """Run an interactive or autoplay game session with decoupled simulation and render loops.

    Args:
        game: Optional existing PacmanGame instance. If None, built from map/config.
        map_name: Name or file path of the map to load if game is None.
        config_path: Optional path to a YAML configuration file.
        seed: Optional RNG seed.
        fps: Target render frame rate (default: 60).
        tps: Target simulation ticks per second (default: 8).
        cell_size: Pixel dimension for each grid cell.
        lives: Optional initial lives override (>= 1).
        autoplay: Optional name of an autonomous Pacman controller to watch.
        debug: If True, overlay coordinates, TPS, tick, score, lives, and ghost targets.
        max_frames: Optional maximum render frames to execute before exiting.
        max_ticks: Optional maximum simulation ticks to execute before exiting.
        clock: Optional Clock instance (defaults to Pygame real-time clock).
    """
    pygame = _init_pygame()

    # 1. Build or configure game
    if game is None:
        if config_path is not None:
            cfg = GameConfig.from_yaml(config_path)
            if map_name != "classic":
                cfg = dataclasses.replace(cfg, map=map_name)
        else:
            default_yaml = Path("configs/examples/classic_feel.yaml")
            if default_yaml.is_file():
                cfg = GameConfig.from_yaml(default_yaml)
                if map_name != "classic":
                    cfg = dataclasses.replace(cfg, map=map_name)
            else:
                pacman_spec = (
                    ControllerSpec(name=autoplay)
                    if autoplay is not None
                    else ControllerSpec(name="manual")
                )
                cfg = GameConfig(
                    map=map_name,
                    rules=RulesConfig(pacman_lives=3, frightened_speed_factor=0.5),
                    pacman=pacman_spec,
                )

        if lives is not None:
            new_rules = dataclasses.replace(cfg.rules, pacman_lives=lives)
            cfg = dataclasses.replace(cfg, rules=new_rules)

        if autoplay is not None:
            cfg = dataclasses.replace(cfg, pacman=ControllerSpec(name=autoplay))
        elif cfg.pacman.name != "manual":
            cfg = dataclasses.replace(cfg, pacman=ControllerSpec(name="manual"))

        # When using default config, adapt ghost entries to match the map's ghost spawns
        if config_path is None:
            from pacman_engine.maps.grid_map import GridMap
            from pacman_engine.maps.loader import load_builtin_map

            if "\n" in cfg.map:
                map_obj = GridMap.from_ascii(cfg.map)
            elif Path(cfg.map).is_file():
                map_obj = GridMap.from_file(cfg.map)
            else:
                map_obj = load_builtin_map(cfg.map)

            map_ghost_ids = set(map_obj.ghost_spawns.keys())
            if set(cfg.ghosts.keys()) != map_ghost_ids:
                filtered_ghosts = {
                    gid: entry for gid, entry in cfg.ghosts.items() if gid in map_ghost_ids
                }
                for gid in map_ghost_ids:
                    if gid not in filtered_ghosts:
                        filtered_ghosts[gid] = GhostEntry(
                            config=GhostConfig(ghost_id=gid, speed=0.9),
                            controller=ControllerSpec("chase"),
                        )
                cfg = dataclasses.replace(cfg, ghosts=filtered_ghosts)

        game = build_game(cfg, seed=seed)

    manual_controller: ManualController | None = None
    if isinstance(game.pacman_controller, ManualController):
        manual_controller = game.pacman_controller
    elif autoplay is None:
        manual_controller = ManualController()
        game.pacman_controller = manual_controller

    theme = RenderTheme(cell_size=cell_size)
    renderer = Renderer(theme=theme)

    cols = game.game_map.cols
    rows = game.game_map.rows
    width = cols * cell_size
    hud_height = cell_size + 8
    height = rows * cell_size + hud_height

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Pacman Engine")

    try:
        font = pygame.font.SysFont("monospace", max(12, cell_size // 2), bold=True)
        big_font = pygame.font.SysFont("monospace", max(18, cell_size), bold=True)
        hud_font = pygame.font.SysFont("monospace", max(14, cell_size // 2 + 2), bold=True)
    except Exception:
        font = None
        big_font = None
        hud_font = None

    if clock is None:
        clock = PygameClock(pygame)

    current_tps = max(2, min(30, int(tps)))
    dt_sim = 1.0 / current_tps
    accumulator = 0.0

    running = True
    paused = False
    ready_timer = 1.5  # Initial 1.5s READY pause
    frames_elapsed = 0
    ticks_elapsed = 0

    while running:
        if max_frames is not None and frames_elapsed >= max_frames:
            break
        if max_ticks is not None and ticks_elapsed >= max_ticks:
            break

        dt = clock.tick(fps)
        dt = min(dt, 0.25)  # Cap dt to prevent accumulator explosion

        # -----------------------------------------------------------------
        # 1. Input & Event Handling
        # -----------------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                break
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    running = False
                    break
                elif event.key == pygame.K_p:
                    paused = not paused
                elif event.key == pygame.K_r:
                    game.reset(seed)
                    if manual_controller is not None:
                        manual_controller.reset()
                    ready_timer = 1.5
                    accumulator = 0.0
                elif event.key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
                    current_tps = min(30, current_tps + 1)
                    dt_sim = 1.0 / current_tps
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    current_tps = max(2, current_tps - 1)
                    dt_sim = 1.0 / current_tps
                elif manual_controller is not None:
                    # Directional input latched into ManualController
                    if event.key in (pygame.K_UP, pygame.K_w):
                        manual_controller.set_action(Action.UP)
                    elif event.key in (pygame.K_DOWN, pygame.K_s):
                        manual_controller.set_action(Action.DOWN)
                    elif event.key in (pygame.K_LEFT, pygame.K_a):
                        manual_controller.set_action(Action.LEFT)
                    elif event.key in (pygame.K_RIGHT, pygame.K_d):
                        manual_controller.set_action(Action.RIGHT)

        if not running:
            break

        # -----------------------------------------------------------------
        # 2. Simulation Step (Fixed-Timestep Accumulator)
        # -----------------------------------------------------------------
        if not paused and game.outcome == Outcome.RUNNING:
            if ready_timer > 0.0:
                ready_timer -= dt
                if ready_timer <= 0.0:
                    accumulator += -ready_timer
                    ready_timer = 0.0
            else:
                accumulator += dt
            while accumulator >= (dt_sim - 1e-9) and game.outcome == Outcome.RUNNING:
                if max_ticks is not None and ticks_elapsed >= max_ticks:
                    break

                prev_lives = game.view.lives
                game.step()
                ticks_elapsed += 1
                accumulator = max(0.0, accumulator - dt_sim)

                # On life loss with lives remaining, enter READY pause
                if game.view.lives < prev_lives and game.outcome == Outcome.RUNNING:
                    ready_timer = 1.5
                    accumulator = 0.0
                    break

        # -----------------------------------------------------------------
        # 3. Rendering
        # -----------------------------------------------------------------
        screen.fill((0, 0, 0))

        # Render maze field
        frame = renderer.render(game.view)
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Render HUD at bottom
        y_hud = rows * cell_size
        pygame.draw.rect(screen, (20, 20, 20), (0, y_hud, width, hud_height))
        pygame.draw.line(screen, (50, 50, 50), (0, y_hud), (width, y_hud), 1)

        if hud_font is not None:
            score_txt = hud_font.render(f"SCORE: {game.view.score}", True, (255, 255, 255))
            screen.blit(score_txt, (10, y_hud + (hud_height - score_txt.get_height()) // 2))

        # Draw remaining life icons
        life_r = max(4, min(8, cell_size // 3))
        for i in range(max(0, game.view.lives)):
            lx = width - 15 - i * (life_r * 2 + 8)
            ly = y_hud + hud_height // 2
            pygame.draw.circle(screen, theme.pacman_color, (lx, ly), life_r)
            # Pacman mouth wedge facing left
            wedge_pts = [(lx, ly), (lx - life_r, ly - life_r // 2), (lx - life_r, ly + life_r // 2)]
            pygame.draw.polygon(screen, (20, 20, 20), wedge_pts)

        # -----------------------------------------------------------------
        # 4. Debug Overlay
        # -----------------------------------------------------------------
        if debug:
            for gid, ctrl in sorted(game.ghost_controllers.items()):
                last_target = getattr(ctrl, "last_target", None)
                if last_target is not None:
                    tr = last_target.row
                    tc = last_target.col
                    gcolor = theme.get_ghost_color(gid)
                    target_rect = pygame.Rect(
                        tc * cell_size + 1, tr * cell_size + 1, cell_size - 2, cell_size - 2
                    )
                    pygame.draw.rect(screen, gcolor, target_rect, width=2)
                    cx = tc * cell_size + cell_size // 2
                    cy = tr * cell_size + cell_size // 2
                    pygame.draw.line(screen, gcolor, (cx - 3, cy), (cx + 3, cy), 1)
                    pygame.draw.line(screen, gcolor, (cx, cy - 3), (cx, cy + 3), 1)

            if font is not None:
                lines = [
                    f"TPS: {current_tps}  FPS: {fps}  Tick: {game.view.tick}",
                    f"Score: {game.view.score}  Lives: {game.view.lives}",
                    f"Pacman: ({game.view.pacman_pos.row}, {game.view.pacman_pos.col})",
                ]
                ghost_coords = ", ".join(
                    f"{gid}:({pos.row},{pos.col})"
                    for gid, pos in sorted(game.view.ghost_positions.items())
                )
                if ghost_coords:
                    lines.append(f"Ghosts: {ghost_coords}")

                y_offset = 4
                for line in lines:
                    txt_surface = font.render(line, True, (255, 255, 255), (0, 0, 0))
                    screen.blit(txt_surface, (6, y_offset))
                    y_offset += txt_surface.get_height() + 2

        # -----------------------------------------------------------------
        # 5. Overlays (READY, PAUSED, GAME OVER, YOU WIN)
        # -----------------------------------------------------------------
        if big_font is not None:
            maze_center_y = (rows * cell_size) // 2
            maze_center_x = width // 2

            if paused:
                p_text = big_font.render("PAUSED", True, (255, 255, 0))
                rect = p_text.get_rect(center=(maze_center_x, maze_center_y))
                bg_rect = rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                pygame.draw.rect(screen, (255, 255, 0), bg_rect, 2)
                screen.blit(p_text, rect)
            elif ready_timer > 0.0 and game.outcome == Outcome.RUNNING:
                r_text = big_font.render("READY!", True, (255, 255, 0))
                rect = r_text.get_rect(center=(maze_center_x, maze_center_y))
                bg_rect = rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                pygame.draw.rect(screen, (255, 255, 0), bg_rect, 2)
                screen.blit(r_text, rect)
            elif game.outcome != Outcome.RUNNING:
                msg = (
                    "YOU WIN!" if game.outcome == Outcome.WIN else f"GAME OVER: {game.outcome.name}"
                )
                color = (0, 255, 0) if game.outcome == Outcome.WIN else (255, 50, 50)
                go_text = big_font.render(msg, True, color)
                rect = go_text.get_rect(center=(maze_center_x, maze_center_y - 12))
                bg_rect = rect.inflate(30, 24)
                pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                pygame.draw.rect(screen, color, bg_rect, 2)
                screen.blit(go_text, rect)

                if font is not None:
                    restart_text = font.render("Press 'R' to Restart", True, (220, 220, 220))
                    r_rect = restart_text.get_rect(center=(maze_center_x, maze_center_y + 14))
                    screen.blit(restart_text, r_rect)

        pygame.display.flip()
        frames_elapsed += 1

    pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint for pacman-play."""
    parser = argparse.ArgumentParser(
        prog="pacman-play",
        description="Interactive Pacman player and autonomous controller viewer.",
    )
    parser.add_argument(
        "--map",
        type=str,
        default="classic",
        help="Name of built-in map (e.g. 'classic', 'small') or path to .map file.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML game configuration file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Master random seed for reproducible episodes.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Target render frame rate (default: 60).",
    )
    parser.add_argument(
        "--tps",
        type=int,
        default=8,
        help="Simulation speed in ticks per second (default: 8). In-game +/- adjusts live.",
    )
    parser.add_argument(
        "--lives",
        type=int,
        default=None,
        help="Number of initial Pacman lives (overrides config if specified).",
    )
    parser.add_argument(
        "--cell-size",
        type=int,
        default=24,
        help="Pixel dimension of each square grid cell (default: 24).",
    )
    parser.add_argument(
        "--autoplay",
        type=str,
        default=None,
        help="Watch an autonomous Pacman controller (e.g. 'greedy_pellet', 'pacman_random').",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Display debug overlay with coordinates, TPS, tick, score, lives, and ghost targets.",
    )

    args = parser.parse_args(argv)
    play(
        map_name=args.map,
        config_path=args.config,
        seed=args.seed,
        fps=args.fps,
        tps=args.tps,
        cell_size=args.cell_size,
        lives=args.lives,
        autoplay=args.autoplay,
        debug=args.debug,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
