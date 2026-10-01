"""Unit tests for PacmanGame mechanics, rules, and collision handling."""

import numpy as np
import pytest

from pacman_engine import (
    Action,
    FrightenedEnded,
    FrightenedStarted,
    GameOverError,
    GhostConfig,
    GhostEaten,
    GhostMode,
    GhostRespawned,
    GridMap,
    InvalidActionError,
    LevelCleared,
    Outcome,
    PacmanDied,
    PacmanGame,
    PelletEaten,
    Position,
    PowerPelletEaten,
    RulesConfig,
    TimeLimitReached,
)
from pacman_engine.controllers import BaseController
from pacman_engine.state import GameView


class ScriptedController(BaseController):
    """Test controller executing a deterministic sequence of actions."""

    def __init__(self, actions: list[Action]) -> None:
        self.actions = list(actions)
        self.idx = 0

    def act(self, view: GameView, entity_id: int | None, rng: np.random.Generator) -> Action:
        if self.idx < len(self.actions):
            act = self.actions[self.idx]
            self.idx += 1
            return act
        return Action.STAY

    def reset(self) -> None:
        self.idx = 0


def create_simple_map(
    layout: str,
    name: str = "test_map",
    wrap_horizontal: bool = False,
    wrap_vertical: bool = False,
) -> GridMap:
    headers = [f"; name: {name}"]
    if wrap_horizontal:
        headers.append("; wrap_horizontal: true")
    if wrap_vertical:
        headers.append("; wrap_vertical: true")
    ascii_str = "\n".join(headers) + "\n" + layout.strip() + "\n"
    return GridMap.from_ascii(ascii_str)


def test_wall_and_door_blocking() -> None:
    # 3x6 map with pellet at (1, 1), Pacman at (1, 2), door at (1, 3)
    layout = """
######
#.P= #
######
"""
    m = create_simple_map(layout)
    game = PacmanGame(m, rules=RulesConfig(invalid_action_policy="stay"))

    # Moving UP into wall (#) stays in place
    res = game.step(Action.UP)
    assert game._pacman.pos == Position(1, 2)
    assert not res.terminated

    # Moving RIGHT into door (=) stays in place for Pacman
    res = game.step(Action.RIGHT)
    assert game._pacman.pos == Position(1, 2)

    # With invalid_action_policy="raise"
    game_raise = PacmanGame(m, rules=RulesConfig(invalid_action_policy="raise"))
    with pytest.raises(InvalidActionError):
        game_raise.step(Action.UP)
    with pytest.raises(InvalidActionError):
        game_raise.step(Action.RIGHT)


def test_door_permeable_to_ghosts() -> None:
    # Pellet at (1, 1), Pacman at (1, 2), door at (1, 3), ghost 0 at (1, 4)
    layout = """
######
#.P=0#
######
"""
    m = create_simple_map(layout)
    ghost_ctrl = ScriptedController([Action.LEFT, Action.LEFT])
    game = PacmanGame(
        m,
        rules=RulesConfig(max_steps=10),
        ghost_controllers={0: ghost_ctrl},
    )

    # Ghost moves LEFT into door cell (1, 3)
    game.step(Action.STAY)
    assert game._ghosts[0].pos == Position(1, 3)


def test_pellet_and_power_pellet_scoring() -> None:
    # Pacman at (1, 1), pellet at (1, 2), power pellet at (1, 3)
    layout = """
#######
#P.o .#
#######
"""
    m = create_simple_map(layout)
    rules = RulesConfig(pellet_score=10, power_pellet_score=50, frightened_duration=5)
    game = PacmanGame(m, rules=rules)

    # Step RIGHT onto standard pellet
    res1 = game.step(Action.RIGHT)
    assert res1.score_delta == 10
    assert any(isinstance(e, PelletEaten) and e.pos == Position(1, 2) for e in res1.events)
    assert game._pacman.pos == Position(1, 2)
    assert game._score == 10

    # Step RIGHT onto power pellet
    res2 = game.step(Action.RIGHT)
    assert res2.score_delta == 50
    assert any(isinstance(e, PowerPelletEaten) and e.pos == Position(1, 3) for e in res2.events)
    assert any(isinstance(e, FrightenedStarted) for e in res2.events)
    assert game._frightened_timer == 4  # 5 minus 1 at phase 4
    assert game._score == 60


def test_combo_ghost_scoring() -> None:
    # Pacman at (1, 1), power pellet at (1, 2), ghosts 0 and 1 at (1, 3) and (1, 4)
    layout = """
########
#Po01 .#
########
"""
    m = create_simple_map(layout)
    rules = RulesConfig(
        pellet_score=10,
        power_pellet_score=50,
        ghost_score_sequence=(200, 400, 800, 1600),
        frightened_duration=10,
        ghost_respawn_delay=5,
    )
    # Ghosts stay in place
    game = PacmanGame(m, rules=rules)

    # Step 1: eat power pellet at (1, 2) -> ghosts 0 and 1 become FRIGHTENED
    res1 = game.step(Action.RIGHT)
    assert res1.score_delta == 50
    assert game._ghosts[0].mode == GhostMode.FRIGHTENED
    assert game._ghosts[1].mode == GhostMode.FRIGHTENED

    # Step 2: move to (1, 3) and eat ghost 0 -> 200 pts (combo 0)
    res2 = game.step(Action.RIGHT)
    assert res2.score_delta == 200
    eaten_0 = [e for e in res2.events if isinstance(e, GhostEaten)]
    assert len(eaten_0) == 1
    assert eaten_0[0].ghost_id == 0
    assert eaten_0[0].score == 200
    assert eaten_0[0].combo_index == 0
    # Ghost 0 returns to spawn and becomes DEAD
    assert game._ghosts[0].pos == Position(1, 3)  # spawn was (1, 3)
    assert game._ghosts[0].mode == GhostMode.DEAD

    # Step 3: move to (1, 4) and eat ghost 1 -> 400 pts (combo 1)
    res3 = game.step(Action.RIGHT)
    assert res3.score_delta == 400
    eaten_1 = [e for e in res3.events if isinstance(e, GhostEaten)]
    assert len(eaten_1) == 1
    assert eaten_1[0].ghost_id == 1
    assert eaten_1[0].score == 400
    assert eaten_1[0].combo_index == 1
    assert game._ghosts[1].pos == Position(1, 4)
    assert game._ghosts[1].mode == GhostMode.DEAD


def test_swap_crossover_collision() -> None:
    # Pacman at (1, 1), ghost 0 at (1, 2)
    layout = """
######
#P0 .#
######
"""
    m = create_simple_map(layout)
    # Pacman moves RIGHT while Ghost moves LEFT -> crossover swap
    ghost_ctrl = ScriptedController([Action.LEFT])
    game = PacmanGame(
        m,
        rules=RulesConfig(pacman_lives=1),
        ghost_controllers={0: ghost_ctrl},
    )

    res = game.step(Action.RIGHT)
    assert res.terminated
    assert res.outcome == Outcome.LOSS
    assert any(isinstance(e, PacmanDied) and e.ghost_id == 0 for e in res.events)


def test_ghost_death_respawn_timing_and_start_delay() -> None:
    # Pacman at (1, 1), power pellet at (1, 2), ghost 0 at (1, 3)
    layout = """
#######
#Po0 .#
#######
"""
    m = create_simple_map(layout)
    rules = RulesConfig(frightened_duration=10, ghost_respawn_delay=2)
    ghost_cfg = {0: GhostConfig(ghost_id=0, start_delay=2)}
    game = PacmanGame(m, rules=rules, ghost_configs=ghost_cfg)

    # Start delay check: for 2 ticks ghost stays at spawn
    assert game._ghosts[0].start_delay_remaining == 2
    game.step(Action.STAY)
    assert game._ghosts[0].start_delay_remaining == 1
    game.step(Action.STAY)
    assert game._ghosts[0].start_delay_remaining == 0

    # Pacman moves to power pellet (1, 2)
    game.step(Action.RIGHT)
    assert game._ghosts[0].mode == GhostMode.FRIGHTENED

    # Pacman eats ghost 0 at (1, 3)
    game.step(Action.RIGHT)
    assert game._ghosts[0].mode == GhostMode.DEAD
    # ghost_respawn_delay is 2: after this step's phase 4, timer decrements from 2 to 1
    assert game._ghosts[0].respawn_timer == 1

    # Next step: timer decrements from 1 to 0 -> GhostRespawned event emitted!
    res_respawn = game.step(Action.STAY)
    assert any(isinstance(e, GhostRespawned) and e.ghost_id == 0 for e in res_respawn.events)
    # Still during frightened window, ghost returns to FRIGHTENED
    assert game._ghosts[0].mode == GhostMode.FRIGHTENED


def test_frightened_timer_expiry() -> None:
    # Pacman at (1, 1), power pellet at (1, 2), ghost 0 at (1, 4)
    layout = """
########
#Po 0 .#
########
"""
    m = create_simple_map(layout)
    rules = RulesConfig(frightened_duration=2)
    game = PacmanGame(m, rules=rules)

    # Eat power pellet: frightened timer set to 2, decrements to 1 at phase 4
    game.step(Action.RIGHT)
    assert game._frightened_timer == 1
    assert game._ghosts[0].mode == GhostMode.FRIGHTENED

    # Next step: timer decrements from 1 to 0 -> FrightenedEnded emitted
    res = game.step(Action.STAY)
    assert game._frightened_timer == 0
    assert any(isinstance(e, FrightenedEnded) for e in res.events)
    assert game._ghosts[0].mode == GhostMode.SCATTER


def test_multi_life_reset() -> None:
    # Pacman at (1, 1), ghost 0 at (1, 2), pellet at (1, 3)
    layout = """
#######
#P0 . #
#######
"""
    m = create_simple_map(layout)
    rules = RulesConfig(pacman_lives=2)
    ghost_ctrl = ScriptedController([Action.LEFT, Action.STAY])
    game = PacmanGame(m, rules=rules, ghost_controllers={0: ghost_ctrl})

    # First lethal collision (lives goes from 2 to 1)
    res1 = game.step(Action.STAY)
    assert not res1.terminated
    assert game._lives == 1
    assert any(isinstance(e, PacmanDied) and e.lives_left == 1 for e in res1.events)
    # Pacman and ghost return to spawns
    assert game._pacman.pos == Position(1, 1)
    assert game._ghosts[0].pos == Position(1, 2)
    # Pellets are kept!
    assert game._pellets[1, 4]

    # Second lethal collision (lives goes from 1 to 0 -> LOSS)
    ghost_ctrl.actions = [Action.LEFT]
    ghost_ctrl.idx = 0
    res2 = game.step(Action.STAY)
    assert res2.terminated
    assert res2.outcome == Outcome.LOSS
    assert any(isinstance(e, PacmanDied) and e.lives_left == 0 for e in res2.events)


def test_win_condition() -> None:
    # 1 pellet at (1, 2)
    layout = """
#####
#P. #
#####
"""
    m = create_simple_map(layout)
    game = PacmanGame(m)

    res = game.step(Action.RIGHT)
    assert res.terminated
    assert res.outcome == Outcome.WIN
    assert any(isinstance(e, LevelCleared) for e in res.events)


def test_timeout_condition() -> None:
    # 2 pellets at (1, 2) and (1, 3), max_steps = 1
    layout = """
######
#P.. #
######
"""
    m = create_simple_map(layout)
    rules = RulesConfig(max_steps=1)
    game = PacmanGame(m, rules=rules)

    res = game.step(Action.RIGHT)
    assert res.truncated
    assert res.outcome == Outcome.TIMEOUT
    assert any(isinstance(e, TimeLimitReached) for e in res.events)


def test_game_over_error() -> None:
    layout = """
#####
#P. #
#####
"""
    m = create_simple_map(layout)
    game = PacmanGame(m)
    game.step(Action.RIGHT)  # Wins immediately
    assert game._outcome == Outcome.WIN

    with pytest.raises(GameOverError, match="Cannot step"):
        game.step(Action.STAY)


def test_random_pacman_start_reproducibility() -> None:
    layout = """
#######
#P... #
#######
"""
    m = create_simple_map(layout)
    rules = RulesConfig(random_pacman_start=True)

    game1 = PacmanGame(m, rules=rules, seed=42)
    pos1 = game1._pacman.pos

    game2 = PacmanGame(m, rules=rules, seed=42)
    pos2 = game2._pacman.pos
    assert pos1 == pos2

    game3 = PacmanGame(m, rules=rules, seed=9999)
    # Different seed may yield different pos
    assert isinstance(game3._pacman.pos, Position)


def test_render_ascii_legend() -> None:
    layout = """
#######
#Po0. #
#######
"""
    m = create_simple_map(layout)
    game = PacmanGame(m)
    ascii_view = game.render_ascii()
    assert "P" in ascii_view
    assert "0" in ascii_view
    assert "o" in ascii_view
    assert "." in ascii_view
    assert "#" in ascii_view
