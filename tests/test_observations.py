"""Tests for observation builders: vector, grid, RGB, specs, and GameConfig integration."""

from __future__ import annotations

import numpy as np
import pytest

from pacman_engine.config import GameConfig, build_game, build_game_and_observation
from pacman_engine.maps.loader import list_builtin_maps, load_builtin_map
from pacman_engine.observations import (
    GridObservationBuilder,
    ObservationConfig,
    ObservationSpec,
    RgbObservationBuilder,
    VectorObservationBuilder,
    create_observation,
    list_observations,
    register_observation,
)


def test_observation_spec_properties() -> None:
    spec = ObservationSpec(
        shape=(10,),
        dtype=np.float32,
        low=0.0,
        high=1.0,
        feature_names=["f" + str(i) for i in range(10)],
    )
    assert spec.shape == (10,)
    assert spec.dtype == np.dtype(np.float32)
    assert spec.low == 0.0
    assert spec.high == 1.0
    assert len(spec.feature_names) == 10


def test_registry_list_and_create() -> None:
    obs_list = list_observations()
    assert "vector" in obs_list
    assert "grid" in obs_list
    assert "rgb" in obs_list

    vec_builder = create_observation("vector")
    assert isinstance(vec_builder, VectorObservationBuilder)

    grid_builder = create_observation("grid", channels_first=False)
    assert isinstance(grid_builder, GridObservationBuilder)
    assert not grid_builder.channels_first

    rgb_builder = create_observation("rgb", cell_size=8)
    assert isinstance(rgb_builder, RgbObservationBuilder)
    assert rgb_builder.renderer.theme.cell_size == 8

    with pytest.raises(ValueError, match="Unknown observation builder"):
        create_observation("nonexistent_builder")


def test_register_custom_observation() -> None:
    @register_observation("dummy_custom")
    class DummyObservation:
        def spec(self, game_map, rules):
            return ObservationSpec((1,), np.float32, 0.0, 1.0)

        def build(self, view):
            return np.zeros((1,), dtype=np.float32)

    assert "dummy_custom" in list_observations()
    builder = create_observation("dummy_custom")
    assert isinstance(builder, DummyObservation)


def test_game_view_to_dict() -> None:
    cfg = GameConfig(map="tiny")
    game = build_game(cfg, seed=42)
    view = game.view
    d = view.to_dict()

    assert "pacman" in d
    assert "row" in d["pacman"]
    assert "col" in d["pacman"]
    assert "direction" in d["pacman"]
    assert "ghosts" in d
    assert "remaining_pellets" in d
    assert "tick" in d
    assert d["remaining_pellets"] == view.remaining_pellets
    assert d["pacman"]["row"] == view.pacman_pos.row
    assert d["pacman"]["col"] == view.pacman_pos.col


@pytest.mark.parametrize("map_name", list_builtin_maps())
@pytest.mark.parametrize(
    "builder_name,builder_kwargs",
    [
        ("vector", {}),
        ("vector", {"normalize": False}),
        ("vector", {"coordinates": "relative_to_pacman"}),
        (
            "vector",
            {
                "include_distances": True,
                "include_pellet_mask": True,
                "include_wall_mask": True,
            },
        ),
        ("grid", {"channels_first": True}),
        ("grid", {"channels_first": False}),
        ("grid", {"view_radius": 2}),
        ("rgb", {}),
        ("rgb", {"grayscale": True, "output_size": (32, 32)}),
    ],
)
def test_all_maps_and_builders_contract(
    map_name: str, builder_name: str, builder_kwargs: dict
) -> None:
    cfg = GameConfig(map=map_name)
    game = build_game(cfg, seed=123)
    game_map = game.map
    rules = cfg.rules

    builder = create_observation(builder_name, **builder_kwargs)
    spec = builder.spec(game_map, rules)

    obs1 = builder.build(game.view)
    obs2 = builder.build(game.view)

    # 1. Output is identical for identical views (determinism)
    np.testing.assert_array_equal(obs1, obs2)

    # 2. Shape and dtype match spec
    assert obs1.shape == spec.shape
    assert obs1.dtype == spec.dtype

    # 3. Values lie within [low, high]
    low = spec.low
    high = spec.high
    if isinstance(low, np.ndarray):
        assert np.all(obs1 >= low - 1e-5), f"Min value violation: {obs1[obs1 < low]}"
    else:
        assert np.all(obs1 >= low - 1e-5)

    if isinstance(high, np.ndarray):
        assert np.all(obs1 <= high + 1e-5), f"Max value violation: {obs1[obs1 > high]}"
    else:
        assert np.all(obs1 <= high + 1e-5)


def test_vector_cross_checks() -> None:
    game_map = load_builtin_map("small")
    cfg = GameConfig(map="small")
    game = build_game(cfg, seed=99)
    view = game.view
    rules = cfg.rules

    # Test absolute coordinates de-normalization
    abs_builder = VectorObservationBuilder(normalize=True, coordinates="absolute")
    abs_spec = abs_builder.spec(game_map, rules)
    abs_vec = abs_builder.build(view)

    p_row_idx = abs_spec.feature_names.index("pacman_row")
    p_col_idx = abs_spec.feature_names.index("pacman_col")

    max_r = float(max(1, game_map.rows - 1))
    max_c = float(max(1, game_map.cols - 1))

    denorm_p_row = round(abs_vec[p_row_idx] * max_r)
    denorm_p_col = round(abs_vec[p_col_idx] * max_c)
    assert denorm_p_row == view.pacman_pos.row
    assert denorm_p_col == view.pacman_pos.col

    for gid, gpos in view.ghost_positions.items():
        g_row_idx = abs_spec.feature_names.index(f"ghost_{gid}_row")
        g_col_idx = abs_spec.feature_names.index(f"ghost_{gid}_col")
        denorm_g_row = round(abs_vec[g_row_idx] * max_r)
        denorm_g_col = round(abs_vec[g_col_idx] * max_c)
        assert denorm_g_row == gpos.row
        assert denorm_g_col == gpos.col

    # Test relative coordinates consistency
    rel_builder = VectorObservationBuilder(normalize=True, coordinates="relative_to_pacman")
    rel_spec = rel_builder.spec(game_map, rules)
    rel_vec = rel_builder.build(view)

    for gid, gpos in view.ghost_positions.items():
        rel_row_idx = rel_spec.feature_names.index(f"ghost_{gid}_row")
        rel_col_idx = rel_spec.feature_names.index(f"ghost_{gid}_col")
        denorm_rel_row = round(rel_vec[rel_row_idx] * max_r)
        denorm_rel_col = round(rel_vec[rel_col_idx] * max_c)
        assert view.pacman_pos.row + denorm_rel_row == gpos.row
        assert view.pacman_pos.col + denorm_rel_col == gpos.col


def test_grid_cross_checks() -> None:
    game_map = load_builtin_map("classic")
    cfg = GameConfig(map="classic")
    game = build_game(cfg, seed=1)
    view = game.view

    builder = GridObservationBuilder(channels_first=True)
    spec = builder.spec(game_map, cfg.rules)
    tensor = builder.build(view)

    # 1. Pellet plane sum equals remaining pellets
    pellet_idx = spec.feature_names.index("pellets")
    power_idx = spec.feature_names.index("power_pellets")
    pellet_sum = int(np.sum(tensor[pellet_idx]))
    power_sum = int(np.sum(tensor[power_idx]))
    assert pellet_sum + power_sum == view.remaining_pellets

    # 2. Pacman plane has exactly one active cell
    pacman_idx = spec.feature_names.index("pacman")
    assert np.sum(tensor[pacman_idx]) == 1.0
    p_row, p_col = np.where(tensor[pacman_idx] == 1.0)
    assert p_row[0] == view.pacman_pos.row
    assert p_col[0] == view.pacman_pos.col


def test_egocentric_window_padding_and_centering() -> None:
    # Use tiny map and test with view_radius=3
    game_map = load_builtin_map("tiny")
    cfg = GameConfig(map="tiny")
    game = build_game(cfg, seed=1)
    view = game.view

    r = 3
    builder = GridObservationBuilder(channels_first=True, view_radius=r)
    spec = builder.spec(game_map, cfg.rules)
    assert spec.shape == (len(builder.channels), 2 * r + 1, 2 * r + 1)

    tensor = builder.build(view)
    pacman_ch = spec.feature_names.index("pacman")
    walls_ch = spec.feature_names.index("walls")

    # Center cell (r, r) must be Pacman
    assert tensor[pacman_ch, r, r] == 1.0
    assert np.sum(tensor[pacman_ch]) == 1.0

    # If Pacman is at (pr, pc), out of bound cells must be walls
    pr, pc = view.pacman_pos.row, view.pacman_pos.col
    for win_r in range(2 * r + 1):
        for win_c in range(2 * r + 1):
            map_r = pr - r + win_r
            map_c = pc - r + win_c
            if not (0 <= map_r < game_map.rows and 0 <= map_c < game_map.cols):
                # Outside map: must be wall and nothing else
                assert tensor[walls_ch, win_r, win_c] == 1.0
                assert tensor[pacman_ch, win_r, win_c] == 0.0


def test_game_config_with_observation_roundtrip() -> None:
    obs_cfg = ObservationConfig(
        name="grid",
        params={"channels_first": False, "view_radius": 4},
    )
    cfg = GameConfig(map="tiny", observation=obs_cfg)
    d = cfg.to_dict()
    assert d["observation"]["name"] == "grid"
    assert d["observation"]["params"]["view_radius"] == 4

    cfg2 = GameConfig.from_dict(d)
    assert cfg == cfg2

    yaml_text = cfg.to_yaml()
    cfg3 = GameConfig.from_yaml(yaml_text)
    assert cfg == cfg3


def test_build_game_and_observation_helper() -> None:
    cfg = GameConfig(
        map="tiny",
        observation=ObservationConfig(name="grid", params={"view_radius": 2}),
    )
    game, obs_builder = build_game_and_observation(cfg, seed=42)
    assert isinstance(obs_builder, GridObservationBuilder)
    assert obs_builder.view_radius == 2

    obs = obs_builder.build(game.view)
    assert obs.shape == (len(obs_builder.channels), 5, 5)
