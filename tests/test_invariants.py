"""Invariant fuzzing tests across all built-in maps over 1000+ episodes."""

import numpy as np

from pacman_engine import (
    Action,
    BaseController,
    Outcome,
    PacmanGame,
    RulesConfig,
    list_builtin_maps,
    load_builtin_map,
)
from pacman_engine.state import GameView


class RandomController(BaseController):
    """Uniformly random action selector using the provided seeded PRNG."""

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        return Action(int(rng.integers(0, 5)))


def test_simulation_invariants_1000_episodes() -> None:
    builtin_names = list_builtin_maps()
    episodes_per_map = max(200, 1000 // len(builtin_names) + 1)
    total_episodes = 0

    rules = RulesConfig(max_steps=100, random_pacman_start=True)

    for map_name in builtin_names:
        grid_map = load_builtin_map(map_name)

        ghost_controllers = {gid: RandomController() for gid in grid_map.ghost_spawns.keys()}
        pacman_controller = RandomController()

        for ep in range(episodes_per_map):
            seed = ep * 1000 + hash(map_name) % 100000
            game = PacmanGame(
                game_map=grid_map,
                rules=rules,
                ghost_controllers=ghost_controllers,
                pacman_controller=pacman_controller,
                seed=seed,
            )

            prev_pellets = game._remaining_pellets

            # Invariant checks at start of episode
            assert not grid_map.walls[game._pacman.pos.row, game._pacman.pos.col]
            assert not grid_map.doors[game._pacman.pos.row, game._pacman.pos.col]
            for gid, ghost in game._ghosts.items():
                assert not grid_map.walls[ghost.pos.row, ghost.pos.col]

            # Run episode
            steps = 0
            while game._outcome == Outcome.RUNNING:
                res = game.step()
                steps += 1

                # Invariant 1: Pacman never in wall or door
                pr, pc = game._pacman.pos.row, game._pacman.pos.col
                assert not grid_map.walls[pr, pc], f"Pacman in wall at ({pr}, {pc})"
                assert not grid_map.doors[pr, pc], f"Pacman in door at ({pr}, {pc})"

                # Invariant 2: Ghosts never in wall
                for gid, ghost in game._ghosts.items():
                    gr, gc = ghost.pos.row, ghost.pos.col
                    assert not grid_map.walls[gr, gc], f"Ghost {gid} in wall at ({gr}, {gc})"

                # Invariant 3: Pellet count never increases
                assert game._remaining_pellets <= prev_pellets, (
                    f"Pellets increased from {prev_pellets} to {game._remaining_pellets}"
                )
                prev_pellets = game._remaining_pellets

            # Invariant 4: Episode always terminates with valid terminal outcome
            assert game._outcome in (Outcome.WIN, Outcome.LOSS, Outcome.TIMEOUT)
            assert res.terminated or res.truncated
            total_episodes += 1

    assert total_episodes >= 1000
