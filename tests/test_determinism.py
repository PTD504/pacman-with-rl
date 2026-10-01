"""Determinism, reproducibility, snapshot, restore, and clone tests."""

import hashlib

from pacman_engine import Action, PacmanGame, RulesConfig, load_builtin_map


def compute_state_hash(game: PacmanGame) -> str:
    """Compute a cryptographic hash of all game state variables and arrays."""
    h = hashlib.sha256()
    h.update(str(game._tick).encode())
    h.update(str(game._score).encode())
    h.update(str(game._lives).encode())
    h.update(str(game._remaining_pellets).encode())
    h.update(str(game._pacman.pos).encode())
    h.update(str(game._pacman.direction).encode())
    for gid in sorted(game._ghosts.keys()):
        g = game._ghosts[gid]
        h.update(
            f"{gid}:{g.pos}:{g.direction}:{g.mode}:{g.respawn_timer}:{g.start_delay_remaining}".encode()
        )
    h.update(game._pellets.tobytes())
    h.update(game._power_pellets.tobytes())
    h.update(str(game._frightened_timer).encode())
    h.update(str(game._combo_index).encode())
    h.update(str(game._schedule_index).encode())
    h.update(str(game._schedule_timer).encode())
    h.update(str(game._global_mode).encode())
    h.update(str(game._outcome).encode())
    return h.hexdigest()


def test_determinism_same_seed_and_actions() -> None:
    m = load_builtin_map("small")
    actions = [Action.UP, Action.RIGHT, Action.DOWN, Action.LEFT, Action.RIGHT, Action.UP] * 10

    # Run 1 with seed=12345
    game1 = PacmanGame(m, rules=RulesConfig(max_steps=100), seed=12345)
    hashes1 = []
    for act in actions:
        if game1._outcome != 0:
            break
        game1.step(act)
        hashes1.append(compute_state_hash(game1))

    # Run 2 with seed=12345
    game2 = PacmanGame(m, rules=RulesConfig(max_steps=100), seed=12345)
    hashes2 = []
    for act in actions:
        if game2._outcome != 0:
            break
        game2.step(act)
        hashes2.append(compute_state_hash(game2))

    assert hashes1 == hashes2
    assert len(hashes1) > 0


def test_random_divergence_with_different_seeds() -> None:
    # Using random_pacman_start or random actions
    m = load_builtin_map("small")
    rules = RulesConfig(random_pacman_start=True, max_steps=100)

    # Starts should diverge on a multi-walkable cell map
    spawns = [PacmanGame(m, rules=rules, seed=s)._pacman.pos for s in range(20)]
    unique_spawns = set(spawns)
    assert len(unique_spawns) > 1


def test_snapshot_and_restore() -> None:
    m = load_builtin_map("small")
    game = PacmanGame(m, rules=RulesConfig(max_steps=50), seed=42)

    # Step a few times
    for act in [Action.RIGHT, Action.DOWN, Action.LEFT]:
        if game._outcome == 0:
            game.step(act)

    snap = game.snapshot()
    hash_at_snap = compute_state_hash(game)

    # Continue stepping original game
    for act in [Action.UP, Action.RIGHT, Action.DOWN]:
        if game._outcome == 0:
            game.step(act)

    assert compute_state_hash(game) != hash_at_snap

    # Restore
    game.restore(snap)
    assert compute_state_hash(game) == hash_at_snap


def test_clone_equivalence() -> None:
    m = load_builtin_map("small")
    game = PacmanGame(m, rules=RulesConfig(max_steps=50), seed=777)

    for act in [Action.RIGHT, Action.UP]:
        if game._outcome == 0:
            game.step(act)

    # Clone
    cloned = game.clone()
    assert compute_state_hash(game) == compute_state_hash(cloned)

    # Step both with identical actions
    actions = [Action.DOWN, Action.LEFT, Action.RIGHT, Action.UP, Action.STAY]
    for act in actions:
        if game._outcome == 0:
            r1 = game.step(act)
            r2 = cloned.step(act)
            assert r1.score_delta == r2.score_delta
            assert r1.terminated == r2.terminated
            assert r1.truncated == r2.truncated
            assert r1.outcome == r2.outcome
            assert compute_state_hash(game) == compute_state_hash(cloned)
