"""Tests for the Gymnasium environment wrapper (PacmanEnv)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium.utils.env_checker import check_env

from pacman_engine.envs import PacmanEnv
from pacman_engine.types import Action


def test_gym_vector_env_checker() -> None:
    """Validate that vector observation environment conforms to Gymnasium standard."""
    env = PacmanEnv(map_name="small", obs_type="vector")
    check_env(env.unwrapped)
    env.close()


def test_gym_grid_env_checker() -> None:
    """Validate that spatial grid environment conforms to Gymnasium standard."""
    env = PacmanEnv(map_name="small", obs_type="grid", obs_params={"channels_first": True})
    check_env(env.unwrapped)
    env.close()


def test_gym_rgb_env_checker() -> None:
    """Validate that visual rgb environment conforms to Gymnasium standard."""
    env = PacmanEnv(
        map_name="small",
        obs_type="rgb",
        obs_params={"output_size": (40, 40), "grayscale": False, "channels_first": False},
    )
    check_env(env.unwrapped)
    env.close()


def test_gym_make_registered_envs() -> None:
    """Test loading environments via standard gym.make."""
    env = gym.make("PacmanSmall-v0")
    obs, info = env.reset(seed=123)
    assert isinstance(obs, np.ndarray)
    assert "score" in info

    next_obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    env.close()


def test_custom_reward_func() -> None:
    """Verify custom reward function gets invoked and reflected."""
    called = []

    def my_reward(step_result, view) -> float:
        called.append(True)
        return 42.0

    env = PacmanEnv(map_name="small", reward_func=my_reward)
    env.reset()
    _, reward, _, _, _ = env.step(Action.RIGHT)
    assert len(called) == 1
    assert reward == 42.0
    env.close()


def test_gym_render_rgb_array() -> None:
    """Test render() returning valid rgb_array."""
    env = PacmanEnv(map_name="small", render_mode="rgb_array", cell_size=16)
    env.reset()
    frame = env.render()
    assert frame is not None
    assert isinstance(frame, np.ndarray)
    assert frame.ndim == 3
    assert frame.shape[2] == 3
    env.close()


def test_gym_deterministic_seed() -> None:
    """Verify that setting the same seed produces identical reset states."""
    env1 = PacmanEnv(map_name="small", obs_type="vector")
    env2 = PacmanEnv(map_name="small", obs_type="vector")

    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)

    np.testing.assert_array_equal(obs1, obs2)
    env1.close()
    env2.close()
