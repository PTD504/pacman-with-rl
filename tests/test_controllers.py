"""Tests for Pacman and Ghost pluggable behaviors/controllers."""

import numpy as np
import pytest

from pacman_engine.config import RulesConfig
from pacman_engine.controllers.ghosts import (
    AmbushGhostController,
    ChaseGhostController,
    FlankGhostController,
    PatrolGhostController,
)
from pacman_engine.controllers.pacman import (
    ExternalController,
    GreedyPelletController,
    ManualController,
    RandomPacmanController,
    ScriptedController,
)
from pacman_engine.engine import PacmanGame
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.types import Action, GhostMode, Position

# ---------------------------------------------------------------------------
# Hand-built Maps for Precise Unit Testing
# ---------------------------------------------------------------------------
# Simple 1D corridor:
# #######
# #P...0#
# #######
CORRIDOR_MAP = """
#######
#P...0#
#######
"""

# 2D Grid map:
# #######
# #.....#
# #.P.0.#
# #.....#
# #.1...#
# #.....#
# #######
GRID_MAP = """
#######
#.....#
#.P.0.#
#.....#
#.1...#
#.....#
#######
"""

# T-junction and dead-end map for testing allow_reverse:
# #######
# #..P..#
# ###.###
#   #0#
#   ###
DEAD_END_MAP = """
#######
#..P..#
###.###
###0###
#######
"""


# ---------------------------------------------------------------------------
# Pacman Controller Tests
# ---------------------------------------------------------------------------
def test_external_controller_marker() -> None:
    ctrl = ExternalController()
    gm = GridMap.from_ascii(CORRIDOR_MAP)
    game = PacmanGame(gm, pacman_controller=ctrl, seed=42)

    # Calling act() directly on ExternalController must raise RuntimeError
    view = game.reset()
    with pytest.raises(
        RuntimeError, match="ExternalController does not generate autonomous actions"
    ):
        ctrl.act(view, None, np.random.default_rng(0))

    # Passing action directly to step() works normally
    res = game.step(Action.RIGHT)
    assert game._pacman.pos == Position(1, 2)
    assert not res.terminated


def test_manual_controller() -> None:
    ctrl = ManualController(initial_action=Action.STAY)
    gm = GridMap.from_ascii(CORRIDOR_MAP)
    game = PacmanGame(gm, pacman_controller=ctrl, seed=42)

    # Initially STAY
    game.step()
    assert game._pacman.pos == Position(1, 1)

    # Update manual action
    ctrl.set_action(Action.RIGHT)
    game.step()
    assert game._pacman.pos == Position(1, 2)

    # String action parsing
    ctrl.set_action("left")
    game.step()
    assert game._pacman.pos == Position(1, 1)

    # Reset
    ctrl.reset()
    assert ctrl.act(game.reset(), None, np.random.default_rng(0)) == Action.STAY


def test_scripted_controller() -> None:
    # Test without loop
    ctrl = ScriptedController(actions=[Action.RIGHT, "stay"], loop=False)
    gm = GridMap.from_ascii(CORRIDOR_MAP)
    game = PacmanGame(gm, pacman_controller=ctrl, seed=42)

    game.step()  # RIGHT -> (1, 2)
    assert game._pacman.pos == Position(1, 2)
    game.step()  # STAY -> (1, 2)
    assert game._pacman.pos == Position(1, 2)
    game.step()  # Sequence exhausted -> defaults to STAY
    assert game._pacman.pos == Position(1, 2)

    # Test with loop
    ctrl_loop = ScriptedController(actions=[Action.RIGHT, Action.LEFT], loop=True)
    game_loop = PacmanGame(gm, pacman_controller=ctrl_loop, seed=42)
    game_loop.step()  # RIGHT -> (1, 2)
    assert game_loop._pacman.pos == Position(1, 2)
    game_loop.step()  # LEFT -> (1, 1)
    assert game_loop._pacman.pos == Position(1, 1)
    game_loop.step()  # Looped RIGHT -> (1, 2)
    assert game_loop._pacman.pos == Position(1, 2)


def test_random_pacman_controller_never_hits_walls() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    ctrl = RandomPacmanController()
    game = PacmanGame(
        gm,
        pacman_controller=ctrl,
        rules=RulesConfig(invalid_action_policy="raise", max_steps=100),
        seed=123,
    )
    # Should run 50 steps without raising InvalidActionError
    for _ in range(50):
        if game._terminated or game._truncated:
            break
        game.step()


def test_greedy_pellet_navigation() -> None:
    # Map where pellet is to the right
    gm = GridMap.from_ascii(CORRIDOR_MAP)
    ctrl = GreedyPelletController()
    game = PacmanGame(gm, pacman_controller=ctrl, seed=42)
    # First step should move toward the pellet at (1, 2)
    game.step()
    assert game._pacman.pos == Position(1, 2)


def test_greedy_pellet_avoids_ghost() -> None:
    # In GRID_MAP:
    # P is at (2, 2), Ghost 0 is at (2, 4)
    # If avoid_ghost_radius is 2, Pacman should move away from Ghost 0
    gm = GridMap.from_ascii(GRID_MAP)
    ctrl = GreedyPelletController(avoid_ghost_radius=2)
    game = PacmanGame(gm, pacman_controller=ctrl, seed=42)
    view = game.reset()
    action = ctrl.act(view, None, np.random.default_rng(0))
    assert action != Action.RIGHT  # Must not step into ghost radius


def test_greedy_pellet_hunt_frightened() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    ctrl = GreedyPelletController(hunt_frightened=True)
    game = PacmanGame(gm, pacman_controller=ctrl, seed=42)
    game.reset()
    # Force ghost 0 into FRIGHTENED mode
    game._ghosts[0].mode = GhostMode.FRIGHTENED
    game._frightened_timer = 20
    view_frightened = game._get_view()

    # Pacman at (2, 2), Ghost 0 at (2, 4). Hunting frightened moves RIGHT toward Ghost 0!
    action = ctrl.act(view_frightened, None, np.random.default_rng(0))
    assert action == Action.RIGHT


# ---------------------------------------------------------------------------
# Ghost Controller Tests
# ---------------------------------------------------------------------------
def test_chase_ghost_reduces_distance_to_pacman() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    chase_ctrl = ChaseGhostController()
    game = PacmanGame(
        gm,
        ghost_controllers={0: chase_ctrl},
        rules=RulesConfig(mode_schedule=((GhostMode.CHASE, 100),)),
        seed=42,
    )
    view = game.reset()
    initial_dist = view.graph.distance(view.ghost_positions[0], view.pacman_pos, for_ghost=True)

    # Step Pacman STAY so ghost moves
    game.step(Action.STAY)
    new_view = game._get_view()
    new_dist = new_view.graph.distance(
        new_view.ghost_positions[0], new_view.pacman_pos, for_ghost=True
    )

    assert new_dist < initial_dist
    assert chase_ctrl.last_target == view.pacman_pos


def test_ambush_ghost_targets_expected_cell() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    ambush_ctrl = AmbushGhostController(lookahead=2)
    game = PacmanGame(
        gm,
        ghost_controllers={0: ambush_ctrl},
        rules=RulesConfig(mode_schedule=((GhostMode.CHASE, 100),), pacman_lives=1),
        seed=42,
    )
    # Step Pacman to the right, so Pacman faces RIGHT
    game.step(Action.RIGHT)
    # Pacman is now at (2, 3) facing Action.RIGHT
    # With lookahead=2, ambush target should be (2, 3 + 2) = (2, 5)
    view = game._get_view()
    target = ambush_ctrl.chase_target(view)
    assert target == Position(2, 5)


def test_flank_ghost_targets_mirrored_cell() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    # Pacman at (2, 2). Partner ghost 0 at (2, 4).
    # Mirrored target for ghost 1 through partner 0:
    # 2 * Pacman - Partner = 2 * (2, 2) - (2, 4) = (2, 0) -> clamped to (2, 1) walkable
    flank_ctrl = FlankGhostController(partner_id=0)
    game = PacmanGame(gm, ghost_controllers={1: flank_ctrl}, seed=42)
    view = game.reset()
    target = flank_ctrl.chase_target(view)
    assert target == Position(2, 1)


def test_patrol_ghost_visits_waypoints_in_order() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    waypoints = [Position(2, 4), Position(1, 4), Position(1, 5)]
    patrol_ctrl = PatrolGhostController(waypoints=waypoints)
    game = PacmanGame(
        gm,
        ghost_controllers={0: patrol_ctrl},
        rules=RulesConfig(mode_schedule=((GhostMode.CHASE, 100),)),
        seed=42,
    )
    view = game.reset()
    # Ghost 0 starts at (2, 4), which is waypoint 0!
    # When ghost is at waypoint 0, target advances to waypoint 1
    _ = patrol_ctrl.act(view, 0, np.random.default_rng(0))
    assert patrol_ctrl.last_target == Position(1, 4)


def test_flee_ghost_increases_distance_from_pacman() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    # In GRID_MAP: Pacman at (2, 2), Ghost 0 at (2, 4).
    # Fleeing from Pacman should pick RIGHT (moving away from (2, 2) toward (2, 5))
    ghost_ctrl = ChaseGhostController(frightened_behavior="flee")
    game = PacmanGame(gm, ghost_controllers={0: ghost_ctrl}, seed=42)
    game.reset()
    game._ghosts[0].mode = GhostMode.FRIGHTENED
    game._frightened_timer = 20
    view = game._get_view()

    init_dist = view.graph.distance(view.ghost_positions[0], view.pacman_pos, for_ghost=True)
    action = ghost_ctrl.act(view, 0, np.random.default_rng(0))
    # Action should move to (2, 5), increasing distance
    gpos = view.ghost_positions[0]
    nbr = Position(gpos.row + action.delta[0], gpos.col + action.delta[1])
    new_dist = view.graph.distance(nbr, view.pacman_pos, for_ghost=True)
    assert new_dist > init_dist


def test_scatter_ghost_goes_to_corner() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    # Ghost 0 default corner is Top-Right: (0, cols - 1) = (0, 6)
    ghost_ctrl = ChaseGhostController()
    game = PacmanGame(
        gm,
        ghost_controllers={0: ghost_ctrl},
        rules=RulesConfig(mode_schedule=((GhostMode.SCATTER, 100),)),
        seed=42,
    )
    view = game.reset()
    assert view.ghost_modes[0] == GhostMode.SCATTER

    # Ghost is at (2, 4). Moving towards (0, 6) should go UP or RIGHT, not DOWN or LEFT
    action = ghost_ctrl.act(view, 0, np.random.default_rng(0))
    assert action in (Action.UP, Action.RIGHT)
    assert ghost_ctrl.last_target == Position(0, 6)


def test_epsilon_randomness_vs_determinism() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    # Deterministic controller (epsilon=0.0)
    det_ctrl = ChaseGhostController(epsilon=0.0)
    game = PacmanGame(gm, ghost_controllers={0: det_ctrl}, seed=42)
    view = game.reset()

    actions_det = [det_ctrl.act(view, 0, np.random.default_rng(i)) for i in range(10)]
    # All 10 must be identical
    assert len(set(actions_det)) == 1

    # Stochastic controller (epsilon=1.0)
    rand_ctrl = ChaseGhostController(epsilon=1.0)
    actions_rand = [rand_ctrl.act(view, 0, np.random.default_rng(i)) for i in range(50)]
    # Must produce multiple distinct legal actions
    assert len(set(actions_rand)) > 1


def test_allow_reverse_respected() -> None:
    gm = GridMap.from_ascii(GRID_MAP)
    # Ghost is moving RIGHT. Reversing would be LEFT.
    # With allow_reverse=False, LEFT must be excluded when other options (UP, DOWN, RIGHT) exist.
    ctrl_no_rev = ChaseGhostController(allow_reverse=False)
    game = PacmanGame(gm, ghost_controllers={0: ctrl_no_rev}, seed=42)
    game.reset()
    game._ghosts[0].direction = Action.RIGHT

    # Move target to the left of the ghost so optimal would be LEFT if allowed
    ctrl_no_rev.scatter_target = Position(2, 1)
    game._ghosts[0].mode = GhostMode.SCATTER

    action = ctrl_no_rev.act(game._get_view(), 0, np.random.default_rng(0))
    assert action != Action.LEFT  # Reverse is disallowed!

    # In a dead end, reverse is allowed
    gm_dead = GridMap.from_ascii(DEAD_END_MAP)
    ctrl_dead = ChaseGhostController(allow_reverse=False)
    game_dead = PacmanGame(gm_dead, ghost_controllers={0: ctrl_dead}, seed=42)
    game_dead.reset()
    game_dead._ghosts[0].direction = Action.DOWN  # Facing dead-end wall
    view_dead = game_dead._get_view()
    # Only legal move is UP (reverse)
    action_dead = ctrl_dead.act(view_dead, 0, np.random.default_rng(0))
    assert action_dead == Action.UP
