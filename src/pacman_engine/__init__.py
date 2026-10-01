"""Customizable, headless-first Pacman game engine for reinforcement learning."""

from pacman_engine.config import (
    GameConfig,
    GhostConfig,
    GhostEntry,
    RulesConfig,
    build_game,
    build_game_and_observation,
)
from pacman_engine.controllers import (
    BaseController,
    Controller,
    ControllerSpec,
    create_controller,
    list_controllers,
    register_controller,
)
from pacman_engine.engine import PacmanGame, StepResult
from pacman_engine.envs import PacmanEnv
from pacman_engine.events import (
    Event,
    FrightenedEnded,
    FrightenedStarted,
    GameEvent,
    GhostEaten,
    GhostRespawned,
    LevelCleared,
    PacmanDied,
    PelletEaten,
    PowerPelletEaten,
    TimeLimitReached,
)
from pacman_engine.maps import (
    GridMap,
    MapBuilder,
    MapGraph,
    generate_random_map,
    list_builtin_maps,
    load_builtin_map,
)
from pacman_engine.observations import (
    ObservationBuilder,
    ObservationConfig,
    ObservationSpec,
    create_observation,
    list_observations,
    register_observation,
)
from pacman_engine.play import play
from pacman_engine.recording import record_episode
from pacman_engine.rendering import Renderer, RenderTheme
from pacman_engine.state import GameView, GhostState, PacmanState
from pacman_engine.types import (
    Action,
    GameOverError,
    GhostMode,
    InvalidActionError,
    MapValidationError,
    Outcome,
    Position,
)

__version__ = "0.1.0"

__all__ = [
    "Action",
    "BaseController",
    "Controller",
    "ControllerSpec",
    "Event",
    "FrightenedEnded",
    "FrightenedStarted",
    "GameConfig",
    "GameEvent",
    "GameOverError",
    "GameView",
    "GhostConfig",
    "GhostEaten",
    "GhostEntry",
    "GhostMode",
    "GhostRespawned",
    "GhostState",
    "GridMap",
    "InvalidActionError",
    "LevelCleared",
    "MapBuilder",
    "MapGraph",
    "MapValidationError",
    "ObservationBuilder",
    "ObservationConfig",
    "ObservationSpec",
    "Outcome",
    "PacmanDied",
    "PacmanEnv",
    "PacmanGame",
    "PacmanState",
    "PelletEaten",
    "Position",
    "PowerPelletEaten",
    "RenderTheme",
    "Renderer",
    "RulesConfig",
    "StepResult",
    "TimeLimitReached",
    "build_game",
    "build_game_and_observation",
    "create_controller",
    "create_observation",
    "generate_random_map",
    "list_builtin_maps",
    "list_controllers",
    "list_observations",
    "load_builtin_map",
    "play",
    "record_episode",
    "register_controller",
    "register_observation",
]
