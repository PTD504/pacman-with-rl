"""Game configuration, rules, ghost parameters, and YAML session definitions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

from pacman_engine.controllers.registry import ControllerSpec, list_controllers
from pacman_engine.observations.base import ObservationConfig
from pacman_engine.types import GhostMode, Position

if TYPE_CHECKING:
    from pacman_engine.engine import PacmanGame
    from pacman_engine.observations.base import ObservationBuilder


@dataclass(frozen=True)
class GhostConfig:
    """Configuration for an individual ghost.

    Attributes:
        ghost_id: Unique ghost identifier (0-9).
        move_period: Optional number of ticks per movement step (>= 1).
        speed: Optional fractional movement speed in (0, 1] (default 1.0).
        start_delay: Number of ticks the ghost waits at spawn before starting.
        scatter_target: Optional target coordinate during SCATTER mode.
    """

    ghost_id: int
    move_period: int | None = None
    speed: float | None = None
    start_delay: int = 0
    scatter_target: Position | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.ghost_id <= 9):
            raise ValueError(f"ghost_id must be between 0 and 9, got {self.ghost_id}")
        if self.move_period is not None and self.speed is not None:
            raise ValueError("Cannot specify both 'move_period' and 'speed' in GhostConfig.")
        if self.move_period is None and self.speed is None:
            object.__setattr__(self, "speed", 1.0)
        if self.move_period is not None:
            if self.move_period < 1:
                raise ValueError(f"move_period must be >= 1, got {self.move_period}")
        if self.speed is not None:
            if not (0.0 < self.speed <= 1.0):
                raise ValueError(f"speed must be in (0, 1], got {self.speed}")
        if self.start_delay < 0:
            raise ValueError(f"start_delay must be >= 0, got {self.start_delay}")
        if self.scatter_target is not None and not isinstance(self.scatter_target, Position):
            if isinstance(self.scatter_target, (tuple, list)) and len(self.scatter_target) == 2:
                object.__setattr__(
                    self,
                    "scatter_target",
                    Position(int(self.scatter_target[0]), int(self.scatter_target[1])),
                )
            else:
                raise ValueError(f"Invalid scatter_target: {self.scatter_target}")

    def to_dict(self) -> dict[str, Any]:
        """Convert GhostConfig to a dictionary."""
        d: dict[str, Any] = {
            "ghost_id": self.ghost_id,
            "start_delay": self.start_delay,
            "scatter_target": (
                [self.scatter_target.row, self.scatter_target.col]
                if self.scatter_target is not None
                else None
            ),
        }
        if self.move_period is not None:
            d["move_period"] = self.move_period
        if self.speed is not None:
            d["speed"] = self.speed
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GhostConfig:
        """Create a GhostConfig instance from a dictionary."""
        d = dict(data)
        if "id" in d and "ghost_id" not in d:
            d["ghost_id"] = d.pop("id")
        if "scatter_target" in d and d["scatter_target"] is not None:
            target = d["scatter_target"]
            if not isinstance(target, Position):
                d["scatter_target"] = Position(int(target[0]), int(target[1]))
        if "move_period" in d and d["move_period"] is not None:
            d["move_period"] = int(d["move_period"])
        if "speed" in d and d["speed"] is not None:
            d["speed"] = float(d["speed"])
        return cls(**d)


@dataclass(frozen=True)
class RulesConfig:
    """Core simulation and gameplay rules configuration.

    Attributes:
        max_steps: Maximum ticks before timeout (None for unlimited).
        pacman_lives: Initial Pacman life count (>= 1, default 3).
        pellet_score: Points awarded for standard pellets.
        power_pellet_score: Points awarded for energizers.
        ghost_score_sequence: Multiplier sequence for consecutive ghosts eaten.
        frightened_duration: Duration in ticks for frightened mode.
        frightened_speed_factor: Speed scaling factor in (0, 1] while ghosts are FRIGHTENED.
        ghost_respawn_delay: Ticks before an eaten ghost respawns at home.
        mode_schedule: Cycle of (GhostMode, duration) phases.
        random_pacman_start: Whether Pacman spawns at a random walkable non-door cell.
        invalid_action_policy: Behavior when moving into walls ('stay' or 'raise').
    """

    max_steps: int | None = 1000
    pacman_lives: int = 3
    pellet_score: int = 10
    power_pellet_score: int = 50
    ghost_score_sequence: tuple[int, ...] = (200, 400, 800, 1600)
    frightened_duration: int = 40
    frightened_speed_factor: float = 1.0
    ghost_respawn_delay: int = 10
    mode_schedule: tuple[tuple[GhostMode, int], ...] = (
        (GhostMode.SCATTER, 20),
        (GhostMode.CHASE, 60),
    )
    random_pacman_start: bool = False
    invalid_action_policy: Literal["stay", "raise"] = "stay"

    def __post_init__(self) -> None:
        if self.max_steps is not None and self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0 or None, got {self.max_steps}")
        if self.pacman_lives < 1:
            raise ValueError(f"pacman_lives must be >= 1, got {self.pacman_lives}")
        if self.pellet_score < 0:
            raise ValueError(f"pellet_score must be >= 0, got {self.pellet_score}")
        if self.power_pellet_score < 0:
            raise ValueError(f"power_pellet_score must be >= 0, got {self.power_pellet_score}")
        if not self.ghost_score_sequence or any(s < 0 for s in self.ghost_score_sequence):
            raise ValueError(
                "ghost_score_sequence must be a non-empty sequence of non-negative ints"
            )
        if self.frightened_duration < 0:
            raise ValueError(f"frightened_duration must be >= 0, got {self.frightened_duration}")
        if not (0.0 < self.frightened_speed_factor <= 1.0):
            raise ValueError(
                f"frightened_speed_factor must be in (0, 1], got {self.frightened_speed_factor}"
            )
        if self.ghost_respawn_delay < 0:
            raise ValueError(f"ghost_respawn_delay must be >= 0, got {self.ghost_respawn_delay}")
        if not self.mode_schedule:
            raise ValueError("mode_schedule cannot be empty")
        for mode, duration in self.mode_schedule:
            if mode not in (GhostMode.SCATTER, GhostMode.CHASE):
                raise ValueError(f"mode_schedule entries must be SCATTER or CHASE, got {mode}")
            if duration <= 0:
                raise ValueError(f"mode_schedule durations must be > 0, got {duration}")
        if self.invalid_action_policy not in ("stay", "raise"):
            raise ValueError(
                f"invalid_action_policy must be 'stay' or 'raise', got {self.invalid_action_policy}"
            )

        # Ensure tuple types
        if not isinstance(self.ghost_score_sequence, tuple):
            object.__setattr__(self, "ghost_score_sequence", tuple(self.ghost_score_sequence))
        if not isinstance(self.mode_schedule, tuple):
            object.__setattr__(
                self,
                "mode_schedule",
                tuple((m, d) for m, d in self.mode_schedule),
            )

    def to_dict(self) -> dict[str, Any]:
        """Convert RulesConfig to a dictionary."""
        return {
            "max_steps": self.max_steps,
            "pacman_lives": self.pacman_lives,
            "pellet_score": self.pellet_score,
            "power_pellet_score": self.power_pellet_score,
            "ghost_score_sequence": list(self.ghost_score_sequence),
            "frightened_duration": self.frightened_duration,
            "frightened_speed_factor": self.frightened_speed_factor,
            "ghost_respawn_delay": self.ghost_respawn_delay,
            "mode_schedule": [[m.name, d] for m, d in self.mode_schedule],
            "random_pacman_start": self.random_pacman_start,
            "invalid_action_policy": self.invalid_action_policy,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RulesConfig:
        """Create a RulesConfig instance from a dictionary."""
        d = dict(data)
        if "ghost_score_sequence" in d:
            d["ghost_score_sequence"] = tuple(d["ghost_score_sequence"])
        if "mode_schedule" in d:
            parsed_schedule = []
            for item in d["mode_schedule"]:
                mode_val, duration = item
                if isinstance(mode_val, str):
                    mode_val = GhostMode[mode_val.upper()]
                elif isinstance(mode_val, int) and not isinstance(mode_val, GhostMode):
                    mode_val = GhostMode(mode_val)
                parsed_schedule.append((mode_val, int(duration)))
            d["mode_schedule"] = tuple(parsed_schedule)
        return cls(**d)


@dataclass(frozen=True)
class GhostEntry:
    """Pairing of ghost simulation parameters with its decision controller specification.

    Attributes:
        config: GhostConfig holding ID, speeds, delays, and scatter targets.
        controller: ControllerSpec defining the controller name and parameters.
    """

    config: GhostConfig
    controller: ControllerSpec = field(default_factory=lambda: ControllerSpec("chase"))

    def to_dict(self) -> dict[str, Any]:
        """Convert GhostEntry to a dictionary."""
        d = self.config.to_dict()
        d["controller"] = self.controller.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any], ghost_id: int | None = None) -> GhostEntry:
        """Create GhostEntry from dictionary."""
        d = dict(data)
        gid = ghost_id if ghost_id is not None else d.get("ghost_id", d.get("id"))
        if gid is None:
            raise ValueError("Ghost entry must specify 'ghost_id' or 'id'")
        gid = int(gid)

        cfg_data: dict[str, Any] = {"ghost_id": gid}
        if "move_period" in d:
            cfg_data["move_period"] = int(d["move_period"])
        if "speed" in d:
            cfg_data["speed"] = float(d["speed"])
        if "start_delay" in d:
            cfg_data["start_delay"] = int(d["start_delay"])
        if "scatter_target" in d and d["scatter_target"] is not None:
            cfg_data["scatter_target"] = d["scatter_target"]

        cfg = GhostConfig.from_dict(cfg_data)

        ctrl_data = d.get("controller", "chase")
        ctrl = ControllerSpec.from_dict(ctrl_data)
        return cls(config=cfg, controller=ctrl)


@dataclass(frozen=True)
class GameConfig:
    """Comprehensive specification of a game session including map, rules, and controllers.

    Attributes:
        map: Built-in map name, file path, or inline ASCII map string.
        rules: RulesConfig governing gameplay rules.
        pacman: ControllerSpec for Pacman.
        ghosts: Mapping of ghost ID to GhostEntry.
    """

    map: str
    rules: RulesConfig = field(default_factory=RulesConfig)
    pacman: ControllerSpec = field(default_factory=lambda: ControllerSpec("external"))
    ghosts: dict[int, GhostEntry] = field(default_factory=dict)
    observation: ObservationConfig = field(default_factory=lambda: ObservationConfig("vector"))

    def __post_init__(self) -> None:
        if not isinstance(self.map, str) or not self.map.strip():
            raise ValueError("GameConfig map must be a non-empty string")

        if isinstance(self.rules, dict):
            object.__setattr__(self, "rules", RulesConfig.from_dict(self.rules))
        elif not isinstance(self.rules, RulesConfig):
            raise ValueError(
                f"GameConfig rules must be RulesConfig, got {type(self.rules).__name__}"
            )

        if isinstance(self.pacman, (str, dict)):
            object.__setattr__(self, "pacman", ControllerSpec.from_dict(self.pacman))
        elif not isinstance(self.pacman, ControllerSpec):
            raise ValueError(
                f"GameConfig pacman must be ControllerSpec, got {type(self.pacman).__name__}"
            )

        if isinstance(self.observation, (str, dict)):
            object.__setattr__(self, "observation", ObservationConfig.from_dict(self.observation))
        elif not isinstance(self.observation, ObservationConfig):
            raise ValueError(
                "GameConfig observation must be ObservationConfig, "
                f"got {type(self.observation).__name__}"
            )

        if isinstance(self.ghosts, list):
            ghosts_dict = {
                entry.config.ghost_id: entry
                for entry in (
                    g if isinstance(g, GhostEntry) else GhostEntry.from_dict(g) for g in self.ghosts
                )
            }
            object.__setattr__(self, "ghosts", ghosts_dict)
        elif isinstance(self.ghosts, dict):
            ghosts_dict = {}
            for k, v in self.ghosts.items():
                gid = int(k)
                if isinstance(v, GhostEntry):
                    ghosts_dict[gid] = v
                elif isinstance(v, dict):
                    ghosts_dict[gid] = GhostEntry.from_dict(v, ghost_id=gid)
                else:
                    raise ValueError(f"Invalid ghost entry for ghost {k}: {v!r}")
            object.__setattr__(self, "ghosts", ghosts_dict)
        else:
            raise ValueError(
                f"GameConfig ghosts must be dict or list, got {type(self.ghosts).__name__}"
            )

        self.validate()

    def validate(self) -> None:
        """Validate map, controller names, observation builder, and parameters.

        Raises:
            ValueError: If map or controllers or observation builder are invalid.
            FileNotFoundError: If a referenced map file is not found.
        """
        from pacman_engine.maps.grid_map import GridMap
        from pacman_engine.maps.loader import list_builtin_maps
        from pacman_engine.observations.base import list_observations

        # 1. Validate map source
        if "\n" in self.map:
            # Inline ASCII
            try:
                GridMap.from_ascii(self.map)
            except Exception as e:
                raise ValueError(f"Invalid inline ASCII map in GameConfig: {e}") from e
        elif Path(self.map).is_file():
            pass
        else:
            # Must be a built-in map
            available = list_builtin_maps()
            clean_name = self.map[:-4] if self.map.endswith(".map") else self.map
            if clean_name not in available:
                raise FileNotFoundError(
                    f"Map '{self.map}' is not a valid file path or built-in map. "
                    f"Available built-in maps: [{', '.join(available)}]"
                )

        # 2. Validate Pacman controller registration
        registered = list_controllers()
        if self.pacman.name not in registered:
            raise ValueError(
                f"Unknown Pacman controller '{self.pacman.name}'. "
                f"Registered controllers: [{', '.join(registered)}]"
            )

        # 3. Validate Ghost controller registrations
        for gid, entry in self.ghosts.items():
            if not (0 <= gid <= 9):
                raise ValueError(f"Invalid ghost ID {gid} in GameConfig. Must be 0-9.")
            if entry.controller.name not in registered:
                raise ValueError(
                    f"Unknown ghost controller '{entry.controller.name}' for ghost {gid}. "
                    f"Registered controllers: [{', '.join(registered)}]"
                )

        # 4. Validate Observation builder registration
        registered_obs = list_observations()
        if self.observation.name not in registered_obs:
            raise ValueError(
                f"Unknown observation builder '{self.observation.name}'. "
                f"Registered builders: [{', '.join(registered_obs)}]"
            )

    def to_dict(self) -> dict[str, Any]:
        """Serialize GameConfig to a dictionary."""
        return {
            "map": self.map,
            "rules": self.rules.to_dict(),
            "pacman": {
                "controller": self.pacman.to_dict(),
            },
            "ghosts": [entry.to_dict() for _, entry in sorted(self.ghosts.items())],
            "observation": self.observation.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameConfig:
        """Deserialize GameConfig from a dictionary."""
        if not isinstance(data, dict):
            raise ValueError(f"GameConfig.from_dict expected dict, got {type(data).__name__}")

        if "map" not in data:
            raise ValueError("GameConfig dictionary missing required 'map' key")

        map_val = data["map"]
        rules_val = RulesConfig.from_dict(data["rules"]) if "rules" in data else RulesConfig()

        pacman_raw = data.get("pacman", "external")
        if isinstance(pacman_raw, dict) and "controller" in pacman_raw:
            pacman_val = ControllerSpec.from_dict(pacman_raw["controller"])
        else:
            pacman_val = ControllerSpec.from_dict(pacman_raw)

        ghosts_raw = data.get("ghosts", {})
        if isinstance(ghosts_raw, list):
            ghosts_dict = {
                entry.config.ghost_id: entry
                for entry in (GhostEntry.from_dict(item) for item in ghosts_raw)
            }
        elif isinstance(ghosts_raw, dict):
            ghosts_dict = {
                int(k): GhostEntry.from_dict(v, ghost_id=int(k)) for k, v in ghosts_raw.items()
            }
        else:
            raise ValueError(f"Expected list or dict for ghosts, got {type(ghosts_raw).__name__}")

        obs_raw = data.get("observation", "vector")
        obs_val = ObservationConfig.from_dict(obs_raw)

        return cls(
            map=map_val,
            rules=rules_val,
            pacman=pacman_val,
            ghosts=ghosts_dict,
            observation=obs_val,
        )

    def to_yaml(self) -> str:
        """Serialize GameConfig to a YAML formatted string."""
        return yaml.safe_dump(self.to_dict(), sort_keys=False)

    @classmethod
    def from_yaml(cls, yaml_content: str | Path) -> GameConfig:
        """Parse GameConfig from a YAML string or file path."""
        if isinstance(yaml_content, Path) or (
            isinstance(yaml_content, str)
            and "\n" not in yaml_content
            and Path(yaml_content).is_file()
        ):
            text = Path(yaml_content).read_text(encoding="utf-8")
        else:
            text = str(yaml_content)

        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid YAML content, expected mapping but got {type(data).__name__}"
            )
        return cls.from_dict(data)


def build_game(config: GameConfig, seed: int | None = None) -> PacmanGame:
    """Build a fully initialized PacmanGame instance from a GameConfig.

    Args:
        config: Fully specified GameConfig session description.
        seed: Optional master RNG seed.

    Returns:
        Configured PacmanGame ready for simulation.

    Raises:
        ValueError: If ghost IDs in configuration do not match map spawns.
    """
    from pacman_engine.controllers.pacman import ExternalController
    from pacman_engine.controllers.registry import create_controller
    from pacman_engine.engine import PacmanGame
    from pacman_engine.maps.grid_map import GridMap
    from pacman_engine.maps.loader import load_builtin_map

    # 1. Resolve map
    if "\n" in config.map:
        game_map = GridMap.from_ascii(config.map)
    elif Path(config.map).is_file():
        game_map = GridMap.from_file(config.map)
    else:
        game_map = load_builtin_map(config.map)

    # 2. Validate configured ghosts exist on map
    map_ghost_ids = set(game_map.ghost_spawns.keys())
    for gid in config.ghosts:
        if gid not in map_ghost_ids:
            raise ValueError(
                f"Ghost ID {gid} configured in GameConfig does not exist on map "
                f"(available ghost spawns: {sorted(map_ghost_ids)})"
            )

    # 3. Create Pacman controller
    if config.pacman.name == "external":
        pacman_ctrl = ExternalController()
    else:
        pacman_ctrl = create_controller(config.pacman.name, **config.pacman.params)

    # 4. Create ghost configs and controllers
    ghost_configs: dict[int, GhostConfig] = {}
    ghost_controllers: dict[int, Any] = {}

    for gid in sorted(map_ghost_ids):
        if gid in config.ghosts:
            entry = config.ghosts[gid]
            ghost_configs[gid] = entry.config
            params = dict(entry.controller.params)
            # If scatter_target configured on GhostConfig and not explicitly passed to controller
            if "scatter_target" not in params and entry.config.scatter_target is not None:
                params["scatter_target"] = entry.config.scatter_target
            ghost_controllers[gid] = create_controller(entry.controller.name, **params)
        else:
            ghost_configs[gid] = GhostConfig(ghost_id=gid)
            ghost_controllers[gid] = create_controller("chase")

    return PacmanGame(
        game_map=game_map,
        rules=config.rules,
        ghost_configs=ghost_configs,
        ghost_controllers=ghost_controllers,
        pacman_controller=pacman_ctrl,
        seed=seed,
    )


def build_game_and_observation(
    config: GameConfig, seed: int | None = None
) -> tuple[PacmanGame, ObservationBuilder]:
    """Build a PacmanGame simulation and its configured ObservationBuilder.

    Args:
        config: Fully specified GameConfig session description.
        seed: Optional master RNG seed.

    Returns:
        A tuple of (PacmanGame, ObservationBuilder).
    """
    from pacman_engine.observations import create_observation

    game = build_game(config, seed=seed)
    builder = create_observation(config.observation.name, **config.observation.params)
    return game, builder
