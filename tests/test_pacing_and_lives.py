"""Acceptance tests for pacing, input buffering, fractional speeds,
lives lifecycle, and play loop.
"""

from __future__ import annotations

import os

import numpy as np

from pacman_engine.config import GameConfig, GhostConfig, GhostEntry, RulesConfig, build_game
from pacman_engine.controllers.pacman import ManualController
from pacman_engine.controllers.registry import ControllerSpec
from pacman_engine.maps.loader import load_builtin_map
from pacman_engine.observations.vector import VectorObservationBuilder
from pacman_engine.play import FakeClock, play
from pacman_engine.types import Action, Outcome, Position

# =========================================================================
# 1. Lives Lifecycle & Respawn Tests
# =========================================================================


def test_lives_default_and_configured_values() -> None:
    # Default is 3
    rules_default = RulesConfig()
    assert rules_default.pacman_lives == 3

    # Configured values 1, 2, 5
    for count in (1, 2, 5):
        cfg = GameConfig(map="small", rules=RulesConfig(pacman_lives=count))
        game = build_game(cfg, seed=0)
        assert game.view.lives == count


def test_full_life_loss_reset_checklist() -> None:
    """Verify exact lifecycle on death with lives remaining.

    Checklist:
    - Pacman returns to map spawn (not random cell)
    - Ghosts return to spawns
    - Ghost start_delay restarts
    - Frightened state and combo reset
    - Pellets and score kept
    - Only last life death gives Outcome.LOSS
    - PacmanDied event has correct lives_left
    """
    map_str = """
#####
#P0.#
#...#
#####
""".strip()
    # Start delayed ghost with scatter_target towards Pacman
    cfg = GameConfig(
        map=map_str,
        rules=RulesConfig(pacman_lives=2, ghost_respawn_delay=5),
        ghosts={
            0: GhostEntry(
                GhostConfig(ghost_id=0, start_delay=2, scatter_target=Position(1, 1)),
                ControllerSpec("chase"),
            )
        },
    )
    game = build_game(cfg, seed=42)

    # Tick 1: Pacman stays at (1, 1), ghost start_delay decrements from 2 to 1
    game.step(Action.STAY)
    pellets_before_death = game.view.remaining_pellets
    assert game._ghosts[0].start_delay_remaining == 1

    # Tick 2: Pacman stays at (1, 1), ghost start_delay decrements from 1 to 0
    game.step(Action.STAY)
    assert game._ghosts[0].start_delay_remaining == 0

    # Tick 3: Ghost moves to (1, 1) and collides with Pacman
    res3 = game.step(Action.STAY)

    # Pacman died on collision (lives went from 2 to 1)
    death_events = [e for e in res3.events if type(e).__name__ == "PacmanDied"]
    assert len(death_events) == 1
    assert death_events[0].lives_left == 1
    assert game.view.lives == 1
    assert game.outcome == Outcome.RUNNING  # Not LOSS yet!

    # 1. Pacman returned to map spawn
    assert game.view.pacman_pos == game.game_map.pacman_spawn

    # 2. Ghost returned to spawn
    assert game.view.ghost_positions[0] == game.game_map.ghost_spawns[0]

    # 3. Ghost start_delay restarted to 2
    assert game._ghosts[0].start_delay_remaining == 2

    # 4. Frightened state & combo reset
    assert game.view.frightened_timer == 0
    assert game._combo_index == 0

    # 5. Pellets and score kept
    assert game.view.remaining_pellets == pellets_before_death

    # Wait for ghost delay to elapse (ticks 4 and 5)
    game.step(Action.STAY)
    game.step(Action.STAY)
    # Ghost now active again at (1, 2) and steps into Pacman at (1, 1) on tick 6
    res6 = game.step(Action.STAY)

    assert game.view.lives == 0
    assert game.outcome == Outcome.LOSS
    assert any(type(e).__name__ == "PacmanDied" for e in res6.events)


def test_vector_observation_lives_normalization() -> None:
    gmap = load_builtin_map("small")
    for init_lives in (1, 3, 5):
        rules = RulesConfig(pacman_lives=init_lives)
        builder = VectorObservationBuilder(initial_lives=init_lives)
        builder.spec(gmap, rules)

        cfg = GameConfig(map="small", rules=rules)
        game = build_game(cfg, seed=0)

        obs = builder.build(game.view)
        # Find index of lives_fraction
        spec = builder.spec(gmap, rules)
        lives_idx = spec.feature_names.index("lives_fraction")
        # Full lives initially -> exactly 1.0
        assert np.isclose(obs[lives_idx], 1.0)


# =========================================================================
# 2. Speed Accumulator Tests
# =========================================================================


def test_speed_accumulator_fractions_and_pattern() -> None:
    """Verify deterministic accumulator moves."""
    # Speed 0.5: exactly N/2 moves over N ticks
    map_str = """
#######
#P...0#
#######
""".strip()
    # Ghost has corridor to move LEFT
    cfg_half = GameConfig(
        map=map_str,
        ghosts={0: GhostEntry(GhostConfig(ghost_id=0, speed=0.5), ControllerSpec("chase"))},
    )
    game_half = build_game(cfg_half, seed=0)

    # Over 20 ticks, Pacman stays
    ghost_moves = 0
    last_pos = game_half.view.ghost_positions[0]
    for _ in range(20):
        game_half.step(Action.STAY)
        curr_pos = game_half.view.ghost_positions[0]
        if curr_pos != last_pos:
            ghost_moves += 1
            last_pos = curr_pos

    assert ghost_moves == 10  # Exactly 10 moves over 20 ticks (0.5)

    # Speed 1.0 matches old move_period 1 behavior (moves every tick)
    cfg_one = GameConfig(
        map=map_str,
        ghosts={0: GhostEntry(GhostConfig(ghost_id=0, speed=1.0), ControllerSpec("chase"))},
    )
    game_one = build_game(cfg_one, seed=0)
    last_pos = game_one.view.ghost_positions[0]
    moves_one = 0
    for _ in range(4):
        game_one.step(Action.STAY)
        curr_pos = game_one.view.ghost_positions[0]
        if curr_pos != last_pos:
            moves_one += 1
            last_pos = curr_pos
    assert moves_one == 4


def test_frightened_speed_factor() -> None:
    map_str = """
#################
#P..o.........0.#
#################
""".strip()
    # Ghost speed 1.0, frightened_speed_factor 0.5
    cfg = GameConfig(
        map=map_str,
        rules=RulesConfig(frightened_speed_factor=0.5, frightened_duration=20),
        ghosts={
            0: GhostEntry(
                GhostConfig(ghost_id=0, speed=1.0),
                ControllerSpec("chase", params={"frightened_behavior": "flee"}),
            )
        },
    )
    game = build_game(cfg, seed=0)

    # Pacman moves RIGHT to eat power pellet
    while game.view.frightened_timer == 0:
        game.step(Action.RIGHT)

    assert game.view.frightened_timer > 0

    # Measure moves while frightened
    frightened_moves = 0
    last_pos = game.view.ghost_positions[0]
    ticks = 6
    for _ in range(ticks):
        game.step(Action.STAY)
        curr = game.view.ghost_positions[0]
        if curr != last_pos:
            frightened_moves += 1
            last_pos = curr

    # With frightened_speed_factor=0.5, moves once every 2 ticks
    assert frightened_moves == ticks // 2


# =========================================================================
# 3. Manual Controller Classic Input Handling Tests
# =========================================================================


def test_manual_controller_heading_continuation_and_reverse() -> None:
    map_str = """
#######
#P....#
#######
""".strip()
    cfg = GameConfig(map=map_str, pacman=ControllerSpec("manual"))
    game = build_game(cfg, seed=0)
    manual_ctrl: ManualController = game.pacman_controller  # type: ignore[assignment]

    # Initial action
    manual_ctrl.set_action(Action.RIGHT)
    game.step()
    assert game.view.pacman_pos == Position(1, 2)

    # Next tick: no key input provided, should CONTINUE heading RIGHT
    game.step()
    assert game.view.pacman_pos == Position(1, 3)

    # Immediate reverse
    manual_ctrl.set_action(Action.LEFT)
    game.step()
    assert game.view.pacman_pos == Position(1, 2)


def test_manual_controller_junction_queueing() -> None:
    # T-junction: Pacman starts at (2, 1), moves RIGHT along row 2.
    # At (2, 2), UP is an opening to (1, 2).
    map_str = """
#####
##.##
#P..#
#####
""".strip()
    cfg = GameConfig(map=map_str, pacman=ControllerSpec("manual"))
    game = build_game(cfg, seed=0)
    manual_ctrl: ManualController = game.pacman_controller  # type: ignore[assignment]

    # Pacman is at (2, 1). Set heading RIGHT, but immediately queue UP early!
    manual_ctrl.set_action(Action.RIGHT)
    game.step()  # moves to (2, 2)
    assert game.view.pacman_pos == Position(2, 2)

    # While at (2, 2), queue UP
    manual_ctrl.set_action(Action.UP)
    game.step()
    # Turns UP into (1, 2)
    assert game.view.pacman_pos == Position(1, 2)


def test_manual_controller_multiple_keys_latched() -> None:
    ctrl = ManualController()
    # Press UP then RIGHT before tick
    ctrl.set_action(Action.UP)
    ctrl.set_action(Action.RIGHT)
    assert ctrl._latched_action == Action.RIGHT  # Last one wins


# =========================================================================
# 4. Decoupled Pacing & Play Loop Tests
# =========================================================================


def test_play_fixed_timestep_pacing_with_fake_clock() -> None:
    """Verify that with fake clock, simulation advances exactly tps ticks per second."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    fake_clock = FakeClock()

    cfg = GameConfig(map="small", rules=RulesConfig(pacman_lives=3))
    game = build_game(cfg, seed=0)

    # tps=10, fps=60. Run for 1.5s READY pause + 1.0s simulation = 2.5s total = 150 frames
    tps = 10
    fps = 60
    total_frames = int(2.5 * fps)

    play(
        game=game,
        fps=fps,
        tps=tps,
        max_frames=total_frames,
        clock=fake_clock,
    )

    # In 1.0s simulation at tps=10, exactly 10 ticks must have elapsed!
    assert game.view.tick == 10


def test_play_smoke_ready_pause_and_death_dummy() -> None:
    """Smoke test running play with READY pause and death under dummy driver."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    fake_clock = FakeClock()

    cfg = GameConfig(
        map="small",
        rules=RulesConfig(pacman_lives=2),
    )
    game = build_game(cfg, seed=0)

    # At 30 TPS, after 1.5s READY pause (90 frames), ghost collides
    # with Pacman at tick 29 (frame 148)
    play(
        game=game,
        fps=60,
        tps=30,
        max_frames=160,
        debug=True,
        clock=fake_clock,
    )
    # Death occurred with 1 life remaining, outcome is still RUNNING
    assert game.view.lives == 1
    assert game.outcome == Outcome.RUNNING


def test_play_tps_keys_and_clamping() -> None:
    """Verify in-game '+' / '-' keys change ticks per second and are clamped to [2, 30]."""
    import pygame

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    map_str = """
#####
#P..#
#####
""".strip()

    # Post '+' key repeatedly to clamp to 30 TPS
    for _ in range(50):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_PLUS))

    fake_clock = FakeClock()
    game1 = build_game(GameConfig(map=map_str), seed=0)
    # 1.5s READY (90 frames) + 1.0s simulation (60 frames) = 150 frames at 60 fps
    play(game=game1, fps=60, tps=15, max_frames=150, clock=fake_clock)
    assert game1.view.tick == 30  # Clamped to max 30 TPS

    # Test '-' key repeatedly to clamp to 2 TPS
    pygame.init()
    for _ in range(50):
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_MINUS))

    fake_clock2 = FakeClock()
    game2 = build_game(GameConfig(map=map_str), seed=0)
    play(game=game2, fps=60, tps=15, max_frames=150, clock=fake_clock2)
    assert game2.view.tick == 2  # Clamped to min 2 TPS
