"""Gymnasium environment wrapper for pacman-engine."""

from __future__ import annotations

import dataclasses
from collections import deque
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
        frame_skip: int = 1,
        frame_stack: int = 1,
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
            frame_skip: Number of simulation ticks to repeat action (>= 1).
            frame_stack: Number of consecutive observations to stack together (>= 1).
        """
        super().__init__()
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, got {frame_skip}")
        if frame_stack < 1:
            raise ValueError(f"frame_stack must be >= 1, got {frame_stack}")

        self.frame_skip = int(frame_skip)
        self.frame_stack = int(frame_stack)
        self._frames: deque[np.ndarray] = deque(maxlen=self.frame_stack)

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
        base_shape = spec.shape
        dtype = spec.dtype
        low = spec.low
        high = spec.high

        if self.frame_stack == 1:
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                shape=base_shape,
                dtype=dtype,
            )
        else:
            channels_first = getattr(self.obs_builder, "channels_first", False)
            if len(base_shape) == 1:
                dim = base_shape[0] * self.frame_stack
                low_arr = np.tile(low, self.frame_stack) if not np.isscalar(low) else low
                high_arr = np.tile(high, self.frame_stack) if not np.isscalar(high) else high
                self.observation_space = spaces.Box(
                    low=low_arr,
                    high=high_arr,
                    shape=(dim,),
                    dtype=dtype,
                )
            elif len(base_shape) == 2:
                self.observation_space = spaces.Box(
                    low=low,
                    high=high,
                    shape=(self.frame_stack, *base_shape),
                    dtype=dtype,
                )
            elif len(base_shape) == 3:
                if channels_first:
                    c, h, w = base_shape
                    self.observation_space = spaces.Box(
                        low=low,
                        high=high,
                        shape=(c * self.frame_stack, h, w),
                        dtype=dtype,
                    )
                else:
                    h, w, c = base_shape
                    self.observation_space = spaces.Box(
                        low=low,
                        high=high,
                        shape=(h, w, c * self.frame_stack),
                        dtype=dtype,
                    )
            else:
                raise ValueError(f"Unsupported observation shape: {base_shape}")

    def _get_stacked_obs(self) -> np.ndarray:
        """Combine frames currently stored in deque according to frame_stack rules."""
        if self.frame_stack == 1:
            return self._frames[-1]

        frames = list(self._frames)
        spec_shape = self.obs_builder.spec(self.game.game_map, self._base_config.rules).shape
        if len(spec_shape) == 1:
            return np.concatenate(frames, axis=0)
        elif len(spec_shape) == 2:
            return np.stack(frames, axis=0)
        elif len(spec_shape) == 3:
            channels_first = getattr(self.obs_builder, "channels_first", False)
            if channels_first:
                return np.concatenate(frames, axis=0)
            else:
                return np.concatenate(frames, axis=-1)
        return frames[-1]

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
        first_obs = self.obs_builder.build(view)

        self._frames.clear()
        for _ in range(self.frame_stack):
            self._frames.append(first_obs)

        obs = self._get_stacked_obs()
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
        """Execute one simulation step with optional frame_skip."""
        act = Action(int(action))
        total_reward = 0.0
        terminated = False
        truncated = False
        combined_events = []

        for _ in range(self.frame_skip):
            step_result = self.game.step(act)
            combined_events.extend(step_result.events)
            total_reward += float(self.reward_func(step_result, self.game.view))
            terminated = step_result.terminated
            truncated = step_result.truncated
            if terminated or truncated:
                break

        latest_obs = self.obs_builder.build(self.game.view)
        self._frames.append(latest_obs)
        obs = self._get_stacked_obs()

        info = {
            "score": self.game.view.score,
            "lives": self.game.view.lives,
            "tick": self.game.view.tick,
            "remaining_pellets": self.game.view.remaining_pellets,
            "outcome": self.game.outcome.name,
            "events": combined_events,
        }
        return obs, total_reward, terminated, truncated, info

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
