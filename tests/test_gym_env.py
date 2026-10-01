"""Tests for the Gymnasium environment wrapper (PacmanEnv)."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
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


def test_gym_frame_skip() -> None:
    """Verify frame_skip executes multiple ticks and accumulates rewards."""
    env = PacmanEnv(map_name="small", frame_skip=4)
    env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step(Action.STAY)
    # Exactly 4 ticks elapsed and 4 step penalties (-0.01 each) accumulated
    assert info["tick"] == 4
    assert reward == pytest.approx(-0.04)

    # Step RIGHT to eat pellet
    _, reward2, _, _, info2 = env.step(Action.RIGHT)
    assert info2["tick"] == 8
    # Reward includes pellet eaten (+1.0) and step penalties
    assert reward2 > 0.0
    assert any(type(e).__name__ == "PelletEaten" for e in info2["events"])
    env.close()


def test_gym_frame_stack_rgb() -> None:
    """Verify frame_stack stacks consecutive channels for visual observations."""
    env = PacmanEnv(
        map_name="small",
        obs_type="rgb",
        obs_params={"grayscale": True, "channels_first": True, "output_size": (40, 40)},
        frame_skip=2,
        frame_stack=4,
    )
    # Base grayscale shape: (1, 40, 40) -> Stacked: (4, 40, 40)
    assert env.observation_space.shape == (4, 40, 40)

    obs, info = env.reset(seed=0)
    assert obs.shape == (4, 40, 40)

    next_obs, reward, terminated, truncated, info = env.step(Action.RIGHT)
    assert next_obs.shape == (4, 40, 40)
    assert info["tick"] == 2  # frame_skip=2

    check_env(env.unwrapped)
    env.close()


def test_gym_frame_stack_vector() -> None:
    """Verify frame_stack concatenates 1D vector observations."""
    base_env = PacmanEnv(map_name="small", obs_type="vector", frame_stack=1)
    base_dim = base_env.observation_space.shape[0]
    base_env.close()

    stacked_env = PacmanEnv(map_name="small", obs_type="vector", frame_stack=3)
    assert stacked_env.observation_space.shape == (base_dim * 3,)

    obs, _ = stacked_env.reset(seed=123)
    assert obs.shape == (base_dim * 3,)

    next_obs, _, _, _, _ = stacked_env.step(Action.UP)
    assert next_obs.shape == (base_dim * 3,)
    stacked_env.close()


def test_gym_invalid_frame_skip_or_stack() -> None:
    """Verify validation error when frame_skip or frame_stack is less than 1."""
    with pytest.raises(ValueError, match="frame_skip must be >= 1"):
        PacmanEnv(frame_skip=0)

    with pytest.raises(ValueError, match="frame_stack must be >= 1"):
        PacmanEnv(frame_stack=-1)
