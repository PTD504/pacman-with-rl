from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from stable_baselines3 import DQN

from pacman_engine.envs.callbacks import MetricTrackingCallback, plot_learning_curves
from pacman_engine.envs.gym_env import PacmanEnv


def test_metric_tracking_callback_with_eval_env(tmp_path: Path) -> None:
    train_env = PacmanEnv(map_name="classic", obs_type="vector")
    eval_env = PacmanEnv(map_name="classic", obs_type="vector")

    callback = MetricTrackingCallback(
        eval_env=eval_env,
        eval_freq=10,
        n_eval_episodes=2,
        log_dir=tmp_path / "logs",
        verbose=0,
    )

    model = DQN("MlpPolicy", train_env, learning_starts=0, buffer_size=500, batch_size=16)
    model.learn(total_timesteps=25, callback=callback)

    assert len(callback.history) >= 2
    df = callback.to_dataframe()
    assert not df.empty
    assert "timestep" in df.columns
    assert "mean_reward" in df.columns
    assert "mean_score" in df.columns
    assert "mean_pellets" in df.columns
    assert "mean_ticks" in df.columns

    # Verify CSV file export
    csv_file = tmp_path / "logs" / "eval_metrics.csv"
    assert csv_file.exists()

    train_env.close()
    eval_env.close()


def test_metric_tracking_callback_auto_clone(tmp_path: Path) -> None:
    train_env = PacmanEnv(map_name="classic", obs_type="vector", frame_skip=2, frame_stack=1)

    callback = MetricTrackingCallback(
        eval_env=None,  # Should auto-clone from train_env
        eval_freq=10,
        n_eval_episodes=1,
        verbose=0,
    )

    model = DQN("MlpPolicy", train_env, learning_starts=0, buffer_size=500, batch_size=16)
    model.learn(total_timesteps=15, callback=callback)

    assert callback.eval_env is not None
    assert callback.eval_env.frame_skip == 2
    assert len(callback.history) >= 1

    train_env.close()
    if callback.eval_env is not None:
        callback.eval_env.close()


def test_plot_learning_curves(tmp_path: Path) -> None:
    cb1 = MetricTrackingCallback(eval_freq=10, n_eval_episodes=1, verbose=0)
    cb1.history = [
        {
            "timestep": 1000,
            "mean_reward": -5.0,
            "std_reward": 1.0,
            "mean_score": 100,
            "std_score": 10,
            "mean_pellets": 5,
            "std_pellets": 1,
            "mean_ticks": 20,
            "std_ticks": 2,
        },
        {
            "timestep": 2000,
            "mean_reward": 10.0,
            "std_reward": 1.5,
            "mean_score": 300,
            "std_score": 20,
            "mean_pellets": 15,
            "std_pellets": 2,
            "mean_ticks": 50,
            "std_ticks": 5,
        },
    ]

    cb2 = MetricTrackingCallback(eval_freq=10, n_eval_episodes=1, verbose=0)
    cb2.history = [
        {
            "timestep": 1000,
            "mean_reward": -8.0,
            "std_reward": 2.0,
            "mean_score": 50,
            "std_score": 5,
            "mean_pellets": 2,
            "std_pellets": 1,
            "mean_ticks": 10,
            "std_ticks": 1,
        },
        {
            "timestep": 2000,
            "mean_reward": 2.0,
            "std_reward": 1.0,
            "mean_score": 150,
            "std_score": 15,
            "mean_pellets": 8,
            "std_pellets": 1,
            "mean_ticks": 30,
            "std_ticks": 3,
        },
    ]

    save_img = tmp_path / "learning_curves.png"
    fig = plot_learning_curves(
        {"Model A": cb1, "Model B": [cb1, cb2]},
        title="Test Curves",
        save_path=save_img,
    )

    assert isinstance(fig, plt.Figure)
    assert save_img.exists()
    plt.close(fig)
