"""Gymnasium environment wrapper for pacman-engine."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from pacman_engine.config import (
    GameConfig,
    ObservationConfig,
    build_game_and_observation,
)
from pacman_engine.engine import PacmanGame, StepResult
from pacman_engine.rendering.renderer import Renderer
from pacman_engine.rendering.theme import RenderTheme
from pacman_engine.state import GameView
from pacman_engine.types import Action


def default_reward_func(step_result: StepResult, view: GameView) -> float:
    """Default reward function for reinforcement learning agents.

    - Pellet eaten: +1.0
    - Power pellet eaten: +5.0
    - Ghost eaten: +10.0
    - Pacman died: -10.0
    - Level cleared / Win: +50.0
    - Time step penalty: -0.01 (encourages efficiency)
    """
    reward = -0.01
    for event in step_result.events:
        name = type(event).__name__
        if name == "PelletEaten":
            reward += 1.0
        elif name == "PowerPelletEaten":
            reward += 5.0
        elif name == "GhostEaten":
            reward += 10.0
        elif name == "PacmanDied":
            reward -= 10.0
        elif name in ("LevelCleared", "Win"):
            reward += 50.0
    return reward


class PacmanEnv(gym.Env):
    """Standard Gymnasium environment interface for Pacman.

    Attributes:
        action_space: spaces.Discrete(5) corresponding to Action enum:
            0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY.
        observation_space: spaces.Box matching the configured ObservationBuilder.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        config: GameConfig | dict[str, Any] | str | Path | None = None,
        map_name: str = "classic",
        obs_type: str = "vector",
        obs_params: dict[str, Any] | None = None,
        reward_func: Callable[[StepResult, GameView], float] | None = None,
        render_mode: str | None = None,
        cell_size: int = 20,
        lives: int | None = None,
        max_steps: int | None = 1000,
    ) -> None:
        """Initialize PacmanEnv.

        Args:
            config: Optional GameConfig instance, dict, or path to YAML config.
            map_name: Built-in map name or path to map file (used if config is None).
            obs_type: Observation format ('vector', 'grid', 'rgb').
            obs_params: Extra parameters passed to the ObservationConfig.
            reward_func: Callable (step_result, view) -> float.
                If None, default_reward_func is used.
            render_mode: Optional render mode ('rgb_array' or 'human').
            cell_size: Pixel size for renderer if rendering is enabled.
            lives: Initial lives override for Pacman (>= 1).
            max_steps: Maximum step limit for episode cutoff.
        """
        super().__init__()
        self.render_mode = render_mode
        self.reward_func = reward_func or default_reward_func
        self._cell_size = cell_size
        self._renderer: Renderer | None = None
        self._window: Any = None
        self._clock: Any = None

        # Build GameConfig
        if config is None:
            game_cfg = GameConfig(
                map=map_name,
                observation=ObservationConfig(name=obs_type, params=obs_params or {}),
            )
        elif isinstance(config, GameConfig):
            game_cfg = config
        elif isinstance(config, dict):
            game_cfg = GameConfig.from_dict(config)
        elif isinstance(config, (str, Path)):
            game_cfg = GameConfig.from_yaml(config)
        else:
            raise TypeError(f"Invalid config type: {type(config)}")

        # Apply overrides if specified
        if lives is not None:
            game_cfg = dataclasses.replace(
                game_cfg, rules=dataclasses.replace(game_cfg.rules, pacman_lives=lives)
            )
        if max_steps is not None:
            game_cfg = dataclasses.replace(
                game_cfg, rules=dataclasses.replace(game_cfg.rules, max_steps=max_steps)
            )

        self._base_config = game_cfg
        self.game: PacmanGame
        self.obs_builder: Any

        # Build initial instance to construct spaces
        self.game, self.obs_builder = build_game_and_observation(self._base_config, seed=0)

        # Action space: 5 discrete movements
        self.action_space = spaces.Discrete(5)

        # Observation space from builder spec
        spec = self.obs_builder.spec(self.game.game_map, self._base_config.rules)
        self.observation_space = spaces.Box(
            low=spec.low,
            high=spec.high,
            shape=spec.shape,
            dtype=spec.dtype,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment to start a new episode."""
        super().reset(seed=seed)

        self.game, self.obs_builder = build_game_and_observation(self._base_config, seed=seed)
        view = self.game.reset(seed=seed)
        obs = self.obs_builder.build(view)

        info = {
            "score": view.score,
            "lives": view.lives,
            "tick": view.tick,
            "remaining_pellets": view.remaining_pellets,
        }
        return obs, info

    def step(
        self,
        action: int | Action,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Execute one simulation step."""
        act = Action(int(action))
        step_result = self.game.step(act)
        obs = self.obs_builder.build(self.game.view)

        reward = float(self.reward_func(step_result, self.game.view))
        terminated = step_result.terminated
        truncated = step_result.truncated

        info = {
            "score": self.game.view.score,
            "lives": self.game.view.lives,
            "tick": self.game.view.tick,
            "remaining_pellets": self.game.view.remaining_pellets,
            "outcome": self.game.outcome.name,
            "events": step_result.events,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Render the environment according to render_mode."""
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = Renderer(theme=RenderTheme(cell_size=self._cell_size))

        frame = self._renderer.render(self.game.view)

        if self.render_mode == "rgb_array":
            return frame

        if self.render_mode == "human":
            import pygame

            if not pygame.get_init():
                pygame.init()
            if self._window is None:
                h, w, _ = frame.shape
                self._window = pygame.display.set_mode((w, h))
                pygame.display.set_caption("Pacman Gym Environment")
                self._clock = pygame.time.Clock()

            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            self._window.blit(surf, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            if self._clock is not None:
                self._clock.tick(self.metadata["render_fps"])
            return None

        raise ValueError(f"Unsupported render mode: {self.render_mode}")

    def close(self) -> None:
        """Clean up rendering resources."""
        if self._window is not None:
            import pygame

            pygame.display.quit()
            self._window = None
            self._clock = None
