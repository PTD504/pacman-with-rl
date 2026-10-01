"""Tests for procedural map generator verifying validity over 200 seeds and determinism."""

import numpy as np
import pytest

from pacman_engine.maps.generator import generate_random_map


def test_generator_valid_across_200_seeds() -> None:
    """Generate maps across 200 distinct seeds with varying parameters and verify validity."""
    configs = [
        (7, 7, 2, 2, 0.1, True),
        (9, 9, 3, 2, 0.2, True),
        (11, 11, 4, 4, 0.3, True),
        (13, 11, 4, 4, 0.2, False),
        (9, 15, 3, 3, 0.4, False),
    ]

    for seed in range(200):
        rows, cols, num_ghosts, num_power, loop_frac, symmetric = configs[seed % len(configs)]
        gm = generate_random_map(
            rows=rows,
            cols=cols,
            num_ghosts=num_ghosts,
            num_power_pellets=num_power,
            loop_fraction=loop_frac,
            symmetric=symmetric,
            seed=seed,
        )
        # Validation is executed in generate_random_map, but explicit call double-checks
        gm.validate()
        assert gm.rows == rows
        assert gm.cols == cols
        assert len(gm.ghost_spawns) == num_ghosts
        assert (gm.pellets.sum() + gm.power_pellets.sum()) >= 1


def test_generator_determinism() -> None:
    """Same seed and parameters produce strictly identical maps."""
    for seed in (0, 42, 12345, 999999):
        gm1 = generate_random_map(rows=11, cols=11, seed=seed)
        gm2 = generate_random_map(rows=11, cols=11, seed=seed)

        assert gm1 == gm2
        assert gm1.to_ascii() == gm2.to_ascii()


def test_generator_different_seeds_produce_different_maps() -> None:
    """Different seeds result in distinct layouts."""
    gm1 = generate_random_map(rows=11, cols=11, seed=1)
    gm2 = generate_random_map(rows=11, cols=11, seed=2)
    assert not np.array_equal(gm1.walls, gm2.walls)


def test_generator_symmetry() -> None:
    """Maps generated with symmetric=True exhibit horizontal reflection symmetry."""
    for seed in (10, 20, 30):
        gm = generate_random_map(rows=11, cols=11, symmetric=True, seed=seed)
        # Check walls symmetry
        assert np.array_equal(gm.walls, np.fliplr(gm.walls))


def test_generator_invalid_dimensions() -> None:
    """Even or sub-7 dimensions must raise ValueError."""
    with pytest.raises(ValueError, match="Dimensions must be odd integers >= 7"):
        generate_random_map(rows=6, cols=7)
    with pytest.raises(ValueError, match="Dimensions must be odd integers >= 7"):
        generate_random_map(rows=7, cols=8)
    with pytest.raises(ValueError, match="Dimensions must be odd integers >= 7"):
        generate_random_map(rows=5, cols=7)


def test_generator_invalid_arguments() -> None:
    with pytest.raises(ValueError, match="num_ghosts"):
        generate_random_map(rows=7, cols=7, num_ghosts=11)
    with pytest.raises(ValueError, match="loop_fraction"):
        generate_random_map(rows=7, cols=7, loop_fraction=1.5)
