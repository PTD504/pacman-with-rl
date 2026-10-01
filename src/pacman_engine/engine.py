"""Core game engine simulation loop, state management, and tick resolution."""

from __future__ import annotations

import copy
from typing import Any, NamedTuple

import numpy as np

from pacman_engine.config import GhostConfig, RulesConfig
from pacman_engine.controllers.base import Controller
from pacman_engine.events import (
    Event,
    FrightenedEnded,
    FrightenedStarted,
    GhostEaten,
    GhostRespawned,
    LevelCleared,
    PacmanDied,
    PelletEaten,
    PowerPelletEaten,
    TimeLimitReached,
)
from pacman_engine.maps.graph import MapGraph
from pacman_engine.maps.grid_map import GridMap
from pacman_engine.state import GameView, GhostState, PacmanState
from pacman_engine.types import (
    Action,
    GameOverError,
    GhostMode,
    InvalidActionError,
    Outcome,
    Position,
)


class StepResult(NamedTuple):
    """Immutable result emitted at the end of each simulation step."""

    events: tuple[Event, ...]
    score_delta: int
    terminated: bool
    truncated: bool
    outcome: Outcome


class PacmanGame:
    """Deterministic, headless Pacman game simulation engine.

    Attributes:
        game_map: Immutable static GridMap layout.
        rules: Active game rules configuration.
        ghost_configs: Mapping from ghost ID to GhostConfig.
        ghost_controllers: Mapping from ghost ID to Controller.
        pacman_controller: Optional default controller for Pacman.
        graph: Precomputed MapGraph topology.
    """

    def __init__(
        self,
        game_map: GridMap,
        rules: RulesConfig | None = None,
        ghost_configs: dict[int, GhostConfig] | None = None,
        ghost_controllers: dict[int, Controller] | None = None,
        pacman_controller: Controller | None = None,
        seed: int | None = None,
    ) -> None:
        self.game_map = game_map
        self.rules = rules if rules is not None else RulesConfig()
        self.graph = MapGraph(game_map)

        # Default ghost configs if not provided
        self.ghost_configs: dict[int, GhostConfig] = {}
        for gid in sorted(self.game_map.ghost_spawns.keys()):
            if ghost_configs is not None and gid in ghost_configs:
                self.ghost_configs[gid] = ghost_configs[gid]
            else:
                self.ghost_configs[gid] = GhostConfig(ghost_id=gid)

        self.ghost_controllers: dict[int, Controller] = (
            dict(ghost_controllers) if ghost_controllers is not None else {}
        )
        self.pacman_controller = pacman_controller

        # Internal state variables
        self._pellets: np.ndarray = np.empty((0, 0), dtype=bool)
        self._power_pellets: np.ndarray = np.empty((0, 0), dtype=bool)
        self._remaining_pellets: int = 0
        self._pacman: PacmanState = PacmanState(pos=self.game_map.pacman_spawn)
        self._ghosts: dict[int, GhostState] = {}
        self._pacman_spawn: Position = self.game_map.pacman_spawn

        self._tick: int = 0
        self._score: int = 0
        self._lives: int = self.rules.pacman_lives
        self._frightened_timer: int = 0
        self._combo_index: int = 0
        self._schedule_index: int = 0
        self._schedule_timer: int = 0
        self._global_mode: GhostMode = self.rules.mode_schedule[0][0]

        self._terminated: bool = False
        self._truncated: bool = False
        self._outcome: Outcome = Outcome.RUNNING

        # PRNG management
        self._seed = seed
        self._init_prngs(seed)

        # Initialize game state
        self.reset(seed)

    def _init_prngs(self, seed: int | None) -> None:
        ss = np.random.SeedSequence(seed)
        num_ghosts = len(self.game_map.ghost_spawns)
        # Partition seeds: 1 for engine, 1 for pacman, N for ghosts
        child_seeds = ss.spawn(2 + num_ghosts)
        self._engine_rng = np.random.default_rng(child_seeds[0])
        self._pacman_rng = np.random.default_rng(child_seeds[1])
        self._ghost_rngs: dict[int, np.random.Generator] = {}
        for i, gid in enumerate(sorted(self.game_map.ghost_spawns.keys())):
            self._ghost_rngs[gid] = np.random.default_rng(child_seeds[2 + i])

    def reset(self, seed: int | None = None) -> GameView:
        """Reset the game state to the beginning of an episode.

        Args:
            seed: Optional master seed to reinitialize all entity PRNGs.

        Returns:
            Read-only GameView at tick 0.
        """
        if seed is not None:
            self._seed = seed
            self._init_prngs(seed)

        # Reset controllers if they support it
        if self.pacman_controller is not None and hasattr(self.pacman_controller, "reset"):
            self.pacman_controller.reset()
        for controller in self.ghost_controllers.values():
            if hasattr(controller, "reset"):
                controller.reset()

        # Reset state counters
        self._tick = 0
        self._score = 0
        self._lives = self.rules.pacman_lives
        self._frightened_timer = 0
        self._combo_index = 0
        self._schedule_index = 0
        self._schedule_timer = 0
        self._global_mode = self.rules.mode_schedule[0][0]

        self._terminated = False
        self._truncated = False
        self._outcome = Outcome.RUNNING

        # Copy mutable pellet arrays
        self._pellets = self.game_map.pellets.copy()
        self._power_pellets = self.game_map.power_pellets.copy()
        self._remaining_pellets = int(self._pellets.sum() + self._power_pellets.sum())

        # Determine Pacman spawn
        if self.rules.random_pacman_start:
            walkable_cells: list[Position] = []
            for r in range(self.game_map.rows):
                for c in range(self.game_map.cols):
                    if not self.game_map.walls[r, c] and not self.game_map.doors[r, c]:
                        walkable_cells.append(Position(r, c))
            idx = int(self._engine_rng.integers(0, len(walkable_cells)))
            self._pacman_spawn = walkable_cells[idx]
            # Spawn rule: spawn cells never contain pellets
            if self._pellets[self._pacman_spawn.row, self._pacman_spawn.col]:
                self._pellets[self._pacman_spawn.row, self._pacman_spawn.col] = False
                self._remaining_pellets -= 1
            if self._power_pellets[self._pacman_spawn.row, self._pacman_spawn.col]:
                self._power_pellets[self._pacman_spawn.row, self._pacman_spawn.col] = False
                self._remaining_pellets -= 1
        else:
            self._pacman_spawn = self.game_map.pacman_spawn

        self._pacman = PacmanState(pos=self._pacman_spawn, direction=Action.STAY)

        # Initialize ghosts
        self._ghosts = {}
        for gid, spawn_pos in self.game_map.ghost_spawns.items():
            cfg = self.ghost_configs[gid]
            self._ghosts[gid] = GhostState(
                id=gid,
                pos=spawn_pos,
                direction=Action.STAY,
                mode=self._global_mode,
                respawn_timer=0,
                start_delay_remaining=cfg.start_delay,
                move_tick_counter=0,
                speed_accumulator=0.0,
            )

        return self._get_view()

    def _get_view(self) -> GameView:
        ghost_pos = {gid: g.pos for gid, g in self._ghosts.items()}
        ghost_dir = {gid: g.direction for gid, g in self._ghosts.items()}
        ghost_mode = {gid: g.mode for gid, g in self._ghosts.items()}
        ghost_timers = {
            gid: (g.respawn_timer if g.mode == GhostMode.DEAD else self._frightened_timer)
            for gid, g in self._ghosts.items()
        }

        return GameView(
            pacman_pos=self._pacman.pos,
            ghost_positions=ghost_pos,
            pacman_direction=self._pacman.direction,
            ghost_directions=ghost_dir,
            ghost_modes=ghost_mode,
            ghost_timers=ghost_timers,
            frightened_timer=self._frightened_timer,
            pellets=self._pellets,
            power_pellets=self._power_pellets,
            lives=self._lives,
            score=self._score,
            tick=self._tick,
            remaining_pellets=self._remaining_pellets,
            graph=self.graph,
        )

    @property
    def view(self) -> GameView:
        """Return the current read-only snapshot GameView."""
        return self._get_view()

    @property
    def map(self) -> GridMap:
        """Return the underlying static GridMap."""
        return self.game_map

    @property
    def outcome(self) -> Outcome:
        """Return the current outcome of the game."""
        return self._outcome

    @property
    def terminated(self) -> bool:
        """Return whether the game terminated due to terminal state (win or pacman death)."""
        return self._terminated

    @property
    def truncated(self) -> bool:
        """Return whether the game truncated due to timeout (max_steps exceeded)."""
        return self._truncated

    def _move_target(self, pos: Position, action: Action) -> Position | None:
        """Compute the destination position after an action, handling wrapping."""
        if action == Action.STAY:
            return pos

        dr, dc = action.delta
        nr = pos.row + dr
        nc = pos.col + dc

        # Horizontal wrap
        if self.game_map.wrap_horizontal and dr == 0:
            if nc < 0:
                nc = self.game_map.cols - 1
            elif nc >= self.game_map.cols:
                nc = 0
        elif nc < 0 or nc >= self.game_map.cols:
            return None

        # Vertical wrap
        if self.game_map.wrap_vertical and dc == 0:
            if nr < 0:
                nr = self.game_map.rows - 1
            elif nr >= self.game_map.rows:
                nr = 0
        elif nr < 0 or nr >= self.game_map.rows:
            return None

        return Position(nr, nc)

    def step(self, pacman_action: Action | None = None) -> StepResult:
        """Advance the simulation by exactly one tick.

        Args:
            pacman_action: Action for Pacman. If None, queries pacman_controller.

        Returns:
            StepResult containing emitted events, score_delta, termination flags, outcome.

        Raises:
            GameOverError: If step() is called on an already finished game.
            InvalidActionError: If invalid_action_policy is 'raise' and Pacman hits a wall/door.
        """
        if self._outcome != Outcome.RUNNING:
            raise GameOverError(f"Cannot step game with outcome {self._outcome.name}")

        view = self._get_view()

        # Determine Pacman action
        if pacman_action is None:
            if self.pacman_controller is not None:
                pacman_action = self.pacman_controller.act(view, None, self._pacman_rng)
            else:
                raise ValueError("pacman_action must be provided when pacman_controller is None")

        # Determine ghost actions
        ghost_actions: dict[int, Action] = {}
        for gid, ghost in self._ghosts.items():
            cfg = self.ghost_configs[gid]
            if ghost.mode == GhostMode.DEAD:
                ghost_actions[gid] = Action.STAY
            elif ghost.start_delay_remaining > 0:
                ghost_actions[gid] = Action.STAY
            else:
                should_move = False
                if cfg.move_period is not None:
                    if (ghost.move_tick_counter % cfg.move_period) == 0:
                        should_move = True
                else:
                    base_speed = cfg.speed if cfg.speed is not None else 1.0
                    effective_speed = (
                        base_speed * self.rules.frightened_speed_factor
                        if ghost.mode == GhostMode.FRIGHTENED
                        else base_speed
                    )
                    ghost.speed_accumulator = round(ghost.speed_accumulator + effective_speed, 9)
                    if ghost.speed_accumulator >= 1.0:
                        should_move = True
                        ghost.speed_accumulator = round(ghost.speed_accumulator - 1.0, 9)

                if should_move:
                    controller = self.ghost_controllers.get(gid)
                    if controller is not None:
                        ghost_actions[gid] = controller.act(view, gid, self._ghost_rngs[gid])
                    else:
                        ghost_actions[gid] = Action.STAY
                else:
                    ghost_actions[gid] = Action.STAY

        events: list[Event] = []
        score_delta: int = 0

        # Snapshot start-of-tick positions for crossover swap detection
        prev_pacman_pos = self._pacman.pos
        prev_ghost_pos = {gid: g.pos for gid, g in self._ghosts.items()}

        # -------------------------------------------------------------
        # Phase 1: Move Phase
        # -------------------------------------------------------------
        # Move Pacman
        p_target = self._move_target(self._pacman.pos, pacman_action)
        is_pacman_blocked = False
        if p_target is None:
            is_pacman_blocked = True
        elif self.game_map.walls[p_target.row, p_target.col]:
            is_pacman_blocked = True
        elif self.game_map.doors[p_target.row, p_target.col]:
            is_pacman_blocked = True

        if is_pacman_blocked:
            if self.rules.invalid_action_policy == "raise":
                raise InvalidActionError(
                    f"Action {pacman_action.name} from {self._pacman.pos} is blocked"
                )
            # Stay in place
            if pacman_action == Action.STAY:
                self._pacman.direction = Action.STAY
        else:
            assert p_target is not None
            self._pacman.pos = p_target
            self._pacman.direction = pacman_action

        # Move Ghosts
        for gid, ghost in self._ghosts.items():
            g_action = ghost_actions[gid]
            # Advance cadence counter only if not waiting on start_delay and not DEAD
            if ghost.start_delay_remaining == 0 and ghost.mode != GhostMode.DEAD:
                ghost.move_tick_counter += 1

            if g_action == Action.STAY:
                ghost.direction = Action.STAY
                continue

            g_target = self._move_target(ghost.pos, g_action)
            # Ghosts cannot enter walls or bounds, but CAN enter doors and pass through each other
            if g_target is None or self.game_map.walls[g_target.row, g_target.col]:
                # Ghost blocked by wall, stays in place
                ghost.direction = Action.STAY
            else:
                ghost.pos = g_target
                ghost.direction = g_action

        # -------------------------------------------------------------
        # Phase 2: Pellet Pickup Phase
        # -------------------------------------------------------------
        pr, pc = self._pacman.pos.row, self._pacman.pos.col
        if self._power_pellets[pr, pc]:
            self._power_pellets[pr, pc] = False
            self._remaining_pellets -= 1
            score_delta += self.rules.power_pellet_score
            events.append(PowerPelletEaten(tick=self._tick, pos=self._pacman.pos))

            # Frighten all live ghosts and reset timer/combo
            self._frightened_timer = self.rules.frightened_duration
            self._combo_index = 0
            for ghost in self._ghosts.values():
                if ghost.mode != GhostMode.DEAD:
                    ghost.mode = GhostMode.FRIGHTENED
            events.append(FrightenedStarted(tick=self._tick))

        elif self._pellets[pr, pc]:
            self._pellets[pr, pc] = False
            self._remaining_pellets -= 1
            score_delta += self.rules.pellet_score
            events.append(PelletEaten(tick=self._tick, pos=self._pacman.pos))

        # -------------------------------------------------------------
        # Phase 3: Collision Resolution Phase
        # -------------------------------------------------------------
        # Identify colliding ghosts (same cell or crossover swap)
        colliding_ghost_ids: list[int] = []
        for gid, ghost in self._ghosts.items():
            if ghost.mode == GhostMode.DEAD:
                continue

            same_cell = ghost.pos == self._pacman.pos
            crossover = (
                ghost.pos == prev_pacman_pos
                and prev_ghost_pos[gid] == self._pacman.pos
                and self._pacman.pos != ghost.pos
            )

            if same_cell or crossover:
                colliding_ghost_ids.append(gid)

        # First, process frightened ghosts eaten
        frightened_eaten: list[int] = []
        for gid in colliding_ghost_ids:
            ghost = self._ghosts[gid]
            if ghost.mode == GhostMode.FRIGHTENED:
                # Eaten!
                seq = self.rules.ghost_score_sequence
                idx = min(self._combo_index, len(seq) - 1)
                eaten_score = seq[idx]
                score_delta += eaten_score
                events.append(
                    GhostEaten(
                        tick=self._tick,
                        ghost_id=gid,
                        pos=self._pacman.pos,
                        score=eaten_score,
                        combo_index=self._combo_index,
                    )
                )
                self._combo_index += 1

                # Ghost returns to spawn and becomes DEAD
                ghost.pos = self.game_map.ghost_spawns[gid]
                ghost.mode = GhostMode.DEAD
                ghost.respawn_timer = self.rules.ghost_respawn_delay
                ghost.direction = Action.STAY
                frightened_eaten.append(gid)

        # Next, check if any remaining colliding ghost causes Pacman death
        pacman_died = False
        killer_gid: int | None = None
        for gid in colliding_ghost_ids:
            if gid not in frightened_eaten:
                ghost = self._ghosts[gid]
                if ghost.mode in (GhostMode.SCATTER, GhostMode.CHASE):
                    pacman_died = True
                    killer_gid = gid
                    break

        if pacman_died:
            assert killer_gid is not None
            self._lives -= 1
            events.append(
                PacmanDied(
                    tick=self._tick,
                    ghost_id=killer_gid,
                    pos=self._pacman.pos,
                    lives_left=self._lives,
                )
            )

            if self._lives > 0:
                # Multi-life reset: Pacman and ghosts return to spawns (pellets kept)
                self._pacman.pos = self.game_map.pacman_spawn
                self._pacman.direction = Action.STAY
                self._frightened_timer = 0
                self._combo_index = 0
                for gid, ghost in self._ghosts.items():
                    cfg = self.ghost_configs[gid]
                    ghost.pos = self.game_map.ghost_spawns[gid]
                    ghost.direction = Action.STAY
                    ghost.mode = self._global_mode
                    ghost.respawn_timer = 0
                    ghost.start_delay_remaining = cfg.start_delay
                    ghost.move_tick_counter = 0
                    ghost.speed_accumulator = 0.0

        # -------------------------------------------------------------
        # Phase 4: Timers & Modes Phase
        # -------------------------------------------------------------
        # Frightened timer decrement
        if self._frightened_timer > 0:
            self._frightened_timer -= 1
            if self._frightened_timer == 0:
                events.append(FrightenedEnded(tick=self._tick))
                for ghost in self._ghosts.values():
                    if ghost.mode == GhostMode.FRIGHTENED:
                        ghost.mode = self._global_mode

        # Dead ghosts respawn timer decrement
        for gid, ghost in self._ghosts.items():
            if ghost.mode == GhostMode.DEAD:
                if ghost.respawn_timer > 0:
                    ghost.respawn_timer -= 1
                if ghost.respawn_timer == 0:
                    ghost.mode = (
                        GhostMode.FRIGHTENED if self._frightened_timer > 0 else self._global_mode
                    )
                    events.append(GhostRespawned(tick=self._tick, ghost_id=gid, pos=ghost.pos))

        # Ghosts start_delay decrement (only if not reset by death on this tick)
        if not (pacman_died and self._lives > 0):
            for ghost in self._ghosts.values():
                if ghost.start_delay_remaining > 0:
                    ghost.start_delay_remaining -= 1

        # Global mode schedule timer (paused during frightened mode)
        if self._frightened_timer == 0:
            self._schedule_timer += 1
            cur_duration = self.rules.mode_schedule[self._schedule_index][1]
            if self._schedule_timer >= cur_duration:
                self._schedule_timer = 0
                self._schedule_index = (self._schedule_index + 1) % len(self.rules.mode_schedule)
                self._global_mode = self.rules.mode_schedule[self._schedule_index][0]
                for ghost in self._ghosts.values():
                    if ghost.mode in (GhostMode.SCATTER, GhostMode.CHASE):
                        ghost.mode = self._global_mode

        # -------------------------------------------------------------
        # Phase 5: Win / Loss / Timeout Evaluation
        # -------------------------------------------------------------
        self._score += score_delta

        if pacman_died and self._lives <= 0:
            self._terminated = True
            self._outcome = Outcome.LOSS
        elif self._remaining_pellets == 0:
            self._terminated = True
            self._outcome = Outcome.WIN
            events.append(LevelCleared(tick=self._tick))
        elif self.rules.max_steps is not None and (self._tick + 1) >= self.rules.max_steps:
            self._truncated = True
            self._outcome = Outcome.TIMEOUT
            events.append(TimeLimitReached(tick=self._tick))

        self._tick += 1

        return StepResult(
            events=tuple(events),
            score_delta=score_delta,
            terminated=self._terminated,
            truncated=self._truncated,
            outcome=self._outcome,
        )

    def snapshot(self) -> dict[str, Any]:
        """Create a deep serializable snapshot of the simulation state and PRNGs."""
        ghosts_state = {}
        for gid, g in self._ghosts.items():
            ghosts_state[gid] = {
                "pos": g.pos,
                "direction": g.direction,
                "mode": g.mode,
                "respawn_timer": g.respawn_timer,
                "start_delay_remaining": g.start_delay_remaining,
                "move_tick_counter": g.move_tick_counter,
                "speed_accumulator": g.speed_accumulator,
            }

        ghost_rng_states = {gid: rng.bit_generator.state for gid, rng in self._ghost_rngs.items()}

        return {
            "tick": self._tick,
            "score": self._score,
            "lives": self._lives,
            "remaining_pellets": self._remaining_pellets,
            "pacman_pos": self._pacman.pos,
            "pacman_direction": self._pacman.direction,
            "pacman_spawn": self._pacman_spawn,
            "ghosts": ghosts_state,
            "pellets": self._pellets.copy(),
            "power_pellets": self._power_pellets.copy(),
            "frightened_timer": self._frightened_timer,
            "combo_index": self._combo_index,
            "schedule_index": self._schedule_index,
            "schedule_timer": self._schedule_timer,
            "global_mode": self._global_mode,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "outcome": self._outcome,
            "engine_rng_state": copy.deepcopy(self._engine_rng.bit_generator.state),
            "pacman_rng_state": copy.deepcopy(self._pacman_rng.bit_generator.state),
            "ghost_rng_states": copy.deepcopy(ghost_rng_states),
        }

    def restore(self, snap: dict[str, Any]) -> None:
        """Restore game state and PRNGs exactly from a snapshot."""
        self._tick = snap["tick"]
        self._score = snap["score"]
        self._lives = snap["lives"]
        self._remaining_pellets = snap["remaining_pellets"]
        self._pacman_spawn = snap["pacman_spawn"]
        self._pacman = PacmanState(pos=snap["pacman_pos"], direction=snap["pacman_direction"])

        self._ghosts = {}
        for gid, gd in snap["ghosts"].items():
            self._ghosts[gid] = GhostState(
                id=gid,
                pos=gd["pos"],
                direction=gd["direction"],
                mode=gd["mode"],
                respawn_timer=gd["respawn_timer"],
                start_delay_remaining=gd["start_delay_remaining"],
                move_tick_counter=gd["move_tick_counter"],
                speed_accumulator=gd.get("speed_accumulator", 0.0),
            )

        self._pellets = snap["pellets"].copy()
        self._power_pellets = snap["power_pellets"].copy()
        self._frightened_timer = snap["frightened_timer"]
        self._combo_index = snap["combo_index"]
        self._schedule_index = snap["schedule_index"]
        self._schedule_timer = snap["schedule_timer"]
        self._global_mode = snap["global_mode"]
        self._terminated = snap["terminated"]
        self._truncated = snap["truncated"]
        self._outcome = snap["outcome"]

        self._engine_rng.bit_generator.state = copy.deepcopy(snap["engine_rng_state"])
        self._pacman_rng.bit_generator.state = copy.deepcopy(snap["pacman_rng_state"])
        for gid, state in snap["ghost_rng_states"].items():
            if gid in self._ghost_rngs:
                self._ghost_rngs[gid].bit_generator.state = copy.deepcopy(state)

    def clone(self) -> PacmanGame:
        """Create a deep clone of the game with identical state and RNGs."""
        cloned = PacmanGame(
            game_map=self.game_map,
            rules=self.rules,
            ghost_configs=self.ghost_configs,
            ghost_controllers=self.ghost_controllers,
            pacman_controller=self.pacman_controller,
            seed=0,
        )
        cloned.restore(self.snapshot())
        return cloned

    def render_ascii(self) -> str:
        """Overlay entities on the map layout for debugging.

        Legend:
            'P': Pacman
            '0'-'9': Ghost in normal mode (SCATTER or CHASE)
            'F': Ghost in FRIGHTENED mode
            'X': Ghost in DEAD mode
            '!': Pacman and ghost occupying the same cell (collision)
            '#': Wall
            '=': Ghost door
            'o': Power pellet
            '.': Standard pellet
            ' ': Walkable empty floor
        """
        rows, cols = self.game_map.rows, self.game_map.cols
        grid: list[list[str]] = []

        for r in range(rows):
            row_chars: list[str] = []
            for c in range(cols):
                if self.game_map.walls[r, c]:
                    row_chars.append("#")
                elif self.game_map.doors[r, c]:
                    row_chars.append("=")
                elif self._power_pellets[r, c]:
                    row_chars.append("o")
                elif self._pellets[r, c]:
                    row_chars.append(".")
                else:
                    row_chars.append(" ")
            grid.append(row_chars)

        # Overlay ghosts
        for gid in sorted(self._ghosts.keys()):
            ghost = self._ghosts[gid]
            gr, gc = ghost.pos.row, ghost.pos.col
            if ghost.mode == GhostMode.DEAD:
                char = "X"
            elif ghost.mode == GhostMode.FRIGHTENED:
                char = "F"
            else:
                char = str(gid)
            grid[gr][gc] = char

        # Overlay Pacman
        pr, pc = self._pacman.pos.row, self._pacman.pos.col
        existing = grid[pr][pc]
        if existing in ("0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "F", "X"):
            grid[pr][pc] = "!"
        else:
            grid[pr][pc] = "P"

        return "\n".join("".join(row) for row in grid) + "\n"
