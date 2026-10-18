"""Stable-Baselines3 callbacks for tracking and plotting Pacman metrics over timesteps."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

from pacman_engine.envs.gym_env import PacmanEnv, default_reward_func


class MetricTrackingCallback(BaseCallback):
    """Callback to periodically evaluate and record comprehensive Pacman metrics
    over global timesteps.

    Evaluates the current policy on a dedicated evaluation environment and records:
    - mean / std reward
    - mean / std game score
    - mean / std pellets eaten
    - mean / std survival ticks
    - mean ghosts eaten

    Attributes:
        eval_env: Dedicated PacmanEnv instance used for deterministic evaluation.
        eval_freq: Timestep frequency for evaluation runs.
        n_eval_episodes: Number of evaluation episodes per checkpoint.
        history: List of recorded metric dictionaries over global timesteps.
    """

    def __init__(
        self,
        eval_env: PacmanEnv | None = None,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        log_dir: str | Path | None = None,
        verbose: int = 1,
    ) -> None:
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.log_dir = Path(log_dir) if log_dir is not None else None
        self.history: list[dict[str, Any]] = []

    def _init_callback(self) -> None:
        if self.eval_env is None:
            # Automatically clone a dedicated evaluation environment from training env config
            train_env = self.training_env.envs[0].unwrapped
            base_config = getattr(train_env, "_base_config", None)
            reward_fn = getattr(train_env, "reward_func", default_reward_func)
            if base_config is not None:
                self.eval_env = PacmanEnv(
                    config=base_config,
                    reward_func=reward_fn,
                    frame_skip=getattr(train_env, "frame_skip", 1),
                    frame_stack=getattr(train_env, "frame_stack", 1),
                )
            else:
                raise ValueError("eval_env must be provided if training env lacks _base_config")

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate_and_record()
        return True

    def _evaluate_and_record(self) -> None:
        assert self.eval_env is not None, "Evaluation environment is not initialized"
        scores: list[float] = []
        rewards: list[float] = []
        pellets: list[float] = []
        ticks: list[float] = []
        ghosts_eaten: list[float] = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            done = False
            ep_reward = 0.0
            init_pellets = self.eval_env.game.view.remaining_pellets
            ep_ghosts = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, r, term, trunc, info = self.eval_env.step(action)
                ep_reward += r

                for ev in info.get("events", []):
                    if type(ev).__name__ == "GhostEaten":
                        ep_ghosts += 1

                done = term or trunc

            final_pellets = info.get("remaining_pellets", 0)
            pellets.append(float(init_pellets - final_pellets))
            scores.append(float(info.get("score", 0)))
            rewards.append(float(ep_reward))
            ticks.append(float(info.get("tick", 0)))
            ghosts_eaten.append(float(ep_ghosts))

        snapshot = {
            "timestep": self.num_timesteps,
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "mean_score": float(np.mean(scores)),
            "std_score": float(np.std(scores)),
            "mean_pellets": float(np.mean(pellets)),
            "std_pellets": float(np.std(pellets)),
            "mean_ticks": float(np.mean(ticks)),
            "std_ticks": float(np.std(ticks)),
            "mean_ghosts_eaten": float(np.mean(ghosts_eaten)),
        }
        self.history.append(snapshot)

        # Log to TensorBoard if logger is active
        if self.logger is not None:
            self.logger.record("eval/mean_reward", snapshot["mean_reward"])
            self.logger.record("eval/mean_score", snapshot["mean_score"])
            self.logger.record("eval/mean_pellets", snapshot["mean_pellets"])
            self.logger.record("eval/mean_ticks", snapshot["mean_ticks"])
            self.logger.record("eval/mean_ghosts_eaten", snapshot["mean_ghosts_eaten"])

        if self.verbose > 0:
            print(
                f"[{self.num_timesteps:7,d} steps] "
                f"Reward: {snapshot['mean_reward']:6.2f} ± {snapshot['std_reward']:4.2f} | "
                f"Score: {snapshot['mean_score']:6.1f} | "
                f"Pellets: {snapshot['mean_pellets']:4.1f} | "
                f"Ghosts: {snapshot['mean_ghosts_eaten']:3.1f}"
            )

        if self.log_dir is not None:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.save_csv(self.log_dir / "eval_metrics.csv")

    def to_dataframe(self) -> pd.DataFrame:
        """Convert history of recorded metrics to a pandas DataFrame."""
        return pd.DataFrame(self.history)

    def save_csv(self, filepath: str | Path) -> None:
        """Export history to a CSV file."""
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)


def plot_learning_curves(
    callbacks: dict[str, MetricTrackingCallback | Sequence[MetricTrackingCallback]],
    title: str = "Comparative Learning Curves across Global Timesteps",
    metrics: Sequence[str] = ("mean_reward", "mean_score", "mean_pellets", "mean_ticks"),
    metric_labels: Sequence[str] = (
        "Mean Reward",
        "Mean Game Score",
        "Pellets Cleared",
        "Survival Ticks",
    ),
    colors: Sequence[str] | None = None,
    figsize: tuple[int, int] = (16, 9),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot comparative multi-metric learning curves over global training timesteps.

    Supports both single MetricTrackingCallback instances and lists of callbacks
    from multiple seeds (in which case it plots the mean and cross-seed std band).

    Args:
        callbacks: Mapping of representation/model names to MetricTrackingCallback
            or sequence of MetricTrackingCallback instances.
        title: Overall super-title for the figure.
        metrics: Keys to plot from callback history dictionaries.
        metric_labels: Human-readable titles for each subplot axis.
        colors: Optional list of hex colors for each model series.
        figsize: Figure dimension tuple (width, height).
        save_path: Optional path to save the generated figure.

    Returns:
        The generated matplotlib Figure object.
    """
    default_colors = ["#3498db", "#2ecc71", "#e74c3c", "#9b59b6", "#f39c12"]
    series_colors = list(colors) if colors is not None else default_colors

    n_metrics = len(metrics)
    rows = (n_metrics + 1) // 2
    cols = 2 if n_metrics > 1 else 1

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes_flat = np.array(axes).flatten()

    for idx, (metric_key, metric_label) in enumerate(zip(metrics, metric_labels, strict=False)):
        ax = axes_flat[idx]

        for s_idx, (name, cb_entry) in enumerate(callbacks.items()):
            color = series_colors[s_idx % len(series_colors)]

            if isinstance(cb_entry, (list, tuple)):
                dfs = [c.to_dataframe() for c in cb_entry if not c.to_dataframe().empty]
                if not dfs:
                    continue
                combined_df = pd.concat(dfs, ignore_index=True)
                if metric_key not in combined_df.columns:
                    continue
                grouped = combined_df.groupby("timestep")[metric_key]
                timesteps = np.array(grouped.mean().index)
                values = grouped.mean().to_numpy()
                std_vals = grouped.std().fillna(0.0).to_numpy()
            else:
                df = cb_entry.to_dataframe()
                if df.empty or metric_key not in df.columns:
                    continue
                timesteps = df["timestep"].to_numpy()
                values = df[metric_key].to_numpy()
                std_col = metric_key.replace("mean_", "std_")
                std_vals = (
                    df[std_col].to_numpy() if std_col in df.columns else np.zeros_like(values)
                )

            ax.plot(
                timesteps,
                values,
                label=name,
                color=color,
                linewidth=2.2,
                marker="o",
                markersize=4,
            )
            ax.fill_between(
                timesteps,
                values - std_vals,
                values + std_vals,
                color=color,
                alpha=0.15,
            )

        ax.set_title(metric_label, fontweight="bold", fontsize=12)
        ax.set_xlabel("Global Timesteps", fontsize=10)
        ax.set_ylabel(metric_label, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        ax.legend(loc="best", framealpha=0.9)

    plt.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")

    return fig
