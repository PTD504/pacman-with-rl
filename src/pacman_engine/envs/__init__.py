"""Gymnasium environment definitions and registration."""

from __future__ import annotations

import gymnasium as gym

from pacman_engine.envs.gym_env import PacmanEnv, default_reward_func

# Register Pacman environments in Gymnasium's global registry
gym.register(
    id="Pacman-v0",
    entry_point="pacman_engine.envs:PacmanEnv",
    kwargs={"map_name": "classic", "obs_type": "vector"},
)

gym.register(
    id="PacmanSmall-v0",
    entry_point="pacman_engine.envs:PacmanEnv",
    kwargs={"map_name": "small", "obs_type": "vector"},
)

gym.register(
    id="PacmanRGB-v0",
    entry_point="pacman_engine.envs:PacmanEnv",
    kwargs={"map_name": "classic", "obs_type": "rgb"},
)

__all__ = ["PacmanEnv", "default_reward_func"]
