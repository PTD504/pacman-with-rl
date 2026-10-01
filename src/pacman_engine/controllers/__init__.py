"""Entity controllers module and registry."""

from pacman_engine.controllers.base import BaseController, Controller
from pacman_engine.controllers.ghosts import (
    AmbushGhostController,
    ChaseGhostController,
    FlankGhostController,
    GhostController,
    PatrolGhostController,
    RandomController,
    RandomGhostController,
)
from pacman_engine.controllers.pacman import (
    ExternalController,
    GreedyPelletController,
    ManualController,
    RandomPacmanController,
    ScriptedController,
)
from pacman_engine.controllers.registry import (
    ControllerSpec,
    create_controller,
    list_controllers,
    register_controller,
)

__all__ = [
    "AmbushGhostController",
    "BaseController",
    "ChaseGhostController",
    "Controller",
    "ControllerSpec",
    "ExternalController",
    "FlankGhostController",
    "GhostController",
    "GreedyPelletController",
    "ManualController",
    "PatrolGhostController",
    "RandomController",
    "RandomGhostController",
    "RandomPacmanController",
    "ScriptedController",
    "create_controller",
    "list_controllers",
    "register_controller",
]
