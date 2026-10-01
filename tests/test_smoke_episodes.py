"""Full-episode smoke tests and statistical performance benchmarks across multiple seeds."""

from pathlib import Path

import pytest

from pacman_engine.config import GameConfig, GhostConfig, GhostEntry, RulesConfig, build_game
from pacman_engine.controllers.registry import ControllerSpec
from pacman_engine.types import Outcome


@pytest.mark.parametrize(
    "config_name",
    ["classic.yaml", "aggressive_speeds.yaml", "patrol_custom.yaml"],
)
def test_full_episodes_smoke_and_determinism(config_name: str) -> None:
    """Run 20 full episodes per example config to completion, verifying determinism."""
    config_path = Path("configs/examples") / config_name
    config = GameConfig.from_yaml(config_path)

    for seed in range(20):
        # Run 1
        game1 = build_game(config, seed=seed)
        step_count1 = 0
        while not (game1._terminated or game1._truncated):
            game1.step()
            step_count1 += 1

        assert game1._outcome != Outcome.RUNNING
        assert step_count1 > 0
        if config.rules.max_steps is not None:
            assert step_count1 <= config.rules.max_steps

        # Run 2 (Determinism check with identical seed)
        game2 = build_game(config, seed=seed)
        step_count2 = 0
        while not (game2._terminated or game2._truncated):
            game2.step()
            step_count2 += 1

        assert step_count1 == step_count2
        assert game1._score == game2._score
        assert game1._outcome == game2._outcome
        assert game1._lives == game2._lives
        assert game1._remaining_pellets == game2._remaining_pellets


def test_statistical_greedy_pellet_vs_random_ghosts() -> None:
    """Statistical assertion with empirically measured thresholds on fixed seeds.

    Empirically measured over seeds 0-19 on the 'small' map:
    - Clear/Win Rate: 100% (20/20)
    - Mean Score: 1060.0 (Min: 690, Max: 1690)

    Asserting generous margins:
    - Win rate >= 80% (threshold 0.8)
    - Mean score >= 800 (threshold 800.0)
    """
    cfg = GameConfig(
        map="small",
        rules=RulesConfig(max_steps=500),
        pacman=ControllerSpec("greedy_pellet", {"avoid_ghost_radius": 1}),
        ghosts={
            0: GhostEntry(GhostConfig(0), ControllerSpec("ghost_random")),
            1: GhostEntry(GhostConfig(1), ControllerSpec("ghost_random")),
        },
    )

    seeds = list(range(20))
    scores: list[int] = []
    wins = 0

    for s in seeds:
        game = build_game(cfg, seed=s)
        while not (game._terminated or game._truncated):
            game.step()
        scores.append(game._score)
        if game._outcome == Outcome.WIN:
            wins += 1

    win_rate = wins / len(seeds)
    mean_score = sum(scores) / len(scores)

    # Generous margin assertions
    assert win_rate >= 0.80, f"Win rate {win_rate} was below threshold 0.80"
    assert mean_score >= 800.0, f"Mean score {mean_score} was below threshold 800.0"
