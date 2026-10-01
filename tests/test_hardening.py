"""Hardening property-style tests: 200 randomized episodes verifying invariants and determinism."""

from __future__ import annotations

import numpy as np

from pacman_engine.config import GameConfig, GhostConfig, GhostEntry, RulesConfig, build_game
from pacman_engine.controllers.registry import ControllerSpec
from pacman_engine.maps.generator import generate_random_map
from pacman_engine.types import Outcome, Position


def test_200_random_episodes_invariants_and_determinism() -> None:
    """Run 200 property-style episodes over random valid maps, rules, and controllers.

    Validates:
    - Nothing crashes.
    - All game invariants hold during and at the end of every episode.
    - Strict determinism: identical seed produces identical trajectory and outcome.
    """
    master_rng = np.random.default_rng(2026)
    pacman_controller_choices = ["greedy_pellet", "pacman_random"]
    ghost_controller_choices = ["chase", "ambush", "flank", "patrol", "ghost_random"]

    for ep_idx in range(200):
        # 1. Procedural map generation
        rows = int(master_rng.choice([7, 9, 11, 13]))
        cols = int(master_rng.choice([7, 9, 11, 13]))
        num_ghosts = int(master_rng.integers(1, 3))
        num_power_pellets = int(master_rng.integers(0, 3))
        loop_fraction = float(master_rng.uniform(0.0, 0.3))
        symmetric = bool(master_rng.choice([True, False]))

        gmap = generate_random_map(
            rows=rows,
            cols=cols,
            num_ghosts=num_ghosts,
            num_power_pellets=num_power_pellets,
            loop_fraction=loop_fraction,
            symmetric=symmetric,
            seed=ep_idx * 17 + 1,
        )

        # 2. Random valid rules
        lives = int(master_rng.integers(1, 4))
        frightened_dur = int(master_rng.integers(5, 20))
        max_steps = int(master_rng.integers(25, 60))
        random_start = bool(master_rng.choice([True, False]))
        rules = RulesConfig(
            pacman_lives=lives,
            frightened_duration=frightened_dur,
            max_steps=max_steps,
            random_pacman_start=random_start,
        )

        # 3. Random controller combinations
        pacman_ctrl = str(master_rng.choice(pacman_controller_choices))
        ghosts_dict: dict[int, GhostEntry] = {}
        for gid in sorted(gmap.ghost_spawns.keys()):
            gctrl = str(master_rng.choice(ghost_controller_choices))
            move_period = int(master_rng.choice([1, 2]))
            ctrl_params = (
                {"waypoints": [gmap.ghost_spawns[gid], Position(1, 1)]} if gctrl == "patrol" else {}
            )
            ghosts_dict[gid] = GhostEntry(
                GhostConfig(ghost_id=gid, move_period=move_period),
                ControllerSpec(gctrl, params=ctrl_params),
            )

        cfg = GameConfig(
            map=gmap.to_ascii(),
            rules=rules,
            pacman=ControllerSpec(pacman_ctrl),
            ghosts=ghosts_dict,
        )

        seed = ep_idx * 31 + 7

        # --- Run 1 ---
        game1 = build_game(cfg, seed=seed)
        step_count1 = 0

        # Invariant checks during run
        while game1.outcome == Outcome.RUNNING:
            v = game1.view
            # Position invariants
            pr, pc = v.pacman_pos.row, v.pacman_pos.col
            assert 0 <= pr < rows and 0 <= pc < cols
            assert not gmap.walls[pr, pc]
            assert not gmap.doors[pr, pc]

            for gid, gpos in v.ghost_positions.items():
                gr, gc = gpos.row, gpos.col
                assert 0 <= gr < rows and 0 <= gc < cols
                assert not gmap.walls[gr, gc]

            assert v.lives >= 0
            assert v.remaining_pellets == int(v.pellets.sum() + v.power_pellets.sum())
            assert v.remaining_pellets >= 0

            game1.step()
            step_count1 += 1

        # Post-episode invariants
        assert game1.outcome != Outcome.RUNNING
        assert step_count1 > 0
        if game1.outcome == Outcome.WIN:
            assert game1.view.remaining_pellets == 0
        elif game1.outcome == Outcome.LOSS:
            assert game1.view.lives == 0
        elif game1.outcome == Outcome.TIMEOUT:
            assert game1.view.tick >= max_steps

        # --- Run 2 (Determinism & Reproducibility) ---
        game2 = build_game(cfg, seed=seed)
        step_count2 = 0
        while game2.outcome == Outcome.RUNNING:
            game2.step()
            step_count2 += 1

        # Exact match assertions between Run 1 and Run 2
        assert step_count1 == step_count2
        assert game1.outcome == game2.outcome
        assert game1.view.score == game2.view.score
        assert game1.view.lives == game2.view.lives
        assert game1.view.tick == game2.view.tick
        assert game1.view.remaining_pellets == game2.view.remaining_pellets
        assert game1.view.pacman_pos == game2.view.pacman_pos
        assert game1.view.ghost_positions == game2.view.ghost_positions
