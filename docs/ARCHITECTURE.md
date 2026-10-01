# Pacman Game Engine — Architecture & Design Contract

This document is the authoritative source of truth for design decisions and contracts across all modules of the Pacman engine.

---

## 1. Coordinates and Actions

- **Coordinate System**: `(row, col)`, with `(0, 0)` at the top-left corner.
  - *Rationale*: Direct mapping to 2D matrix indexing `grid[row, col]` in NumPy without transposition overhead.
- **Position Type**: `Position(NamedTuple)` with integer fields `row: int` and `col: int`.
  - *Rationale*: Immutable, lightweight, hashable (ideal for dictionary/set keys and graph lookups), and unpackable with zero runtime overhead.
- **Actions**: `Action(IntEnum)`:
  - `UP = 0` (delta: `(-1, 0)`)
  - `DOWN = 1` (delta: `(1, 0)`)
  - `LEFT = 2` (delta: `(0, -1)`)
  - `RIGHT = 3` (delta: `(0, 1)`)
  - `STAY = 4` (delta: `(0, 0)`)
  - *Rationale*: Compact integer representation simplifies discrete action spaces for reinforcement learning and zero-copy encoding.

---

## 2. Step and Tick Execution Model

- **Step Concept**: One call to `step(pacman_action)` corresponds to exactly one game tick.
- **Simultaneous Action Selection**: At the start of a tick, all entities (Pacman and ghosts) observe the same frozen start-of-tick state and determine their chosen actions.
- **Simultaneous Movement**: All active entities move simultaneously, at most 1 grid cell per tick.
- **Ghost Speed Control**: Ghost speed can be defined in two mutually exclusive ways:
  - Integer cadence via `move_period`: An integer $N \ge 1$ where the ghost moves once every $N$ ticks (on non-moving ticks, the ghost stays in place).
  - Fractional speed via `speed`: A float in $(0, 1]$ (default 1.0) implemented via a deterministic accumulator. Each tick, `speed_accumulator += speed` (scaled by `RulesConfig.frightened_speed_factor` when in FRIGHTENED mode). When `speed_accumulator >= 1.0`, the ghost executes a 1-cell move and accumulator is decremented by 1.0 (rounded to 9 decimals to eliminate floating-point drift). Specifying both `speed` and `move_period` raises a ValueError.
  - *Rationale*: Integer tick-based cadence and deterministic accumulators avoid sub-grid position errors, guarantee at most 1 cell per tick, and preserve strict discrete state transitions and bitwise determinism.

---

## 3. Tick Resolution Order

Each tick is resolved strictly in the following deterministic sequence:
1. **Move Phase**: All entities move simultaneously according to their chosen actions and environmental boundaries/doors.
2. **Pellet Pickup Phase**: If Pacman arrives at a cell containing a pellet or power pellet, the item is consumed and removed from the active state.
3. **Collision Resolution Phase**: Collisions between Pacman and ghosts are evaluated:
   - *Same cell*: Pacman and a ghost occupy the same cell.
   - *Swapped positions (crossover collision)*: Pacman moved from cell $A$ to $B$ while a ghost moved from cell $B$ to $A$ during the same tick.
   - If ghosts are frightened, ghosts are consumed and enter DEAD mode.
   - If a lethal collision occurs:
     - `PacmanDied` event is emitted with `lives_left = lives - 1`.
     - `lives` decrements by 1.
     - **If lives remain (`lives_left > 0`)**:
       - Pacman resets to the map's designated spawn position `P` (not a random cell).
       - All ghosts reset to their designated map spawn positions.
       - Ghost `start_delay` countdown timers restart to their configured values (delay countdown begins on the subsequent tick).
       - Ghost speed accumulators reset to 0.0.
       - Global frightened mode, frightened timer, and ghost eating combo multipliers reset.
       - Remaining pellets, power pellets, and accumulated score are preserved.
       - Game outcome remains `Outcome.RUNNING`.
     - **If no lives remain (`lives_left == 0`)**:
       - Game outcome becomes `Outcome.LOSS`.
4. **Timers & Modes Phase**: Frightened timers, scatter/chase mode cycles, and entity respawn/cooldown timers decrement.
5. **Win / Loss / Timeout Evaluation**:
   - Pacman death takes precedence over win conditions (e.g. if Pacman eats the last pellet on the same tick as a lethal collision, Pacman dies).
   - Win condition triggers when all pellets and power pellets are cleared.
   - Loss condition triggers only when Pacman dies on its final life.
   - Timeout triggers when `max_steps` is exceeded.

---

## 4. Event-Driven Architecture and Rewards

- **Typed Events**: The engine emits typed event objects each step (e.g. `PelletEaten`, `PowerPelletEaten`, `GhostEaten`, `PacmanDied`, `GhostFrightened`, `Win`, `Timeout`).
- **Score vs. Rewards**:
  - Classic game score is tracked as an informative convenience counter.
  - RL reward signals are intentionally decoupled from the engine score: external environments and wrappers compute domain-specific rewards by inspecting the step's typed events and state transitions.

---

## 5. Determinism and Randomness

- **Master Seed**: The environment is initialized with a single master integer seed.
- **SeedSequence Partitioning**: Independent PRNGs for each entity and stochastic subsystem are derived using `numpy.random.SeedSequence(seed).spawn(...)`.
- **Reproducibility Guarantee**: Given the identical master seed and identical sequence of Pacman actions, the engine guarantees bitwise identical trajectories across all execution environments.

---

## 6. Planned Module Architecture

The engine is structured into modular layers with clear dependency boundaries:
- `pacman_engine/types.py`: Core types, coordinates, actions, shared value objects, and exceptions.
- `pacman_engine/maps/`: Static map representation (`GridMap`), validation, graph analysis (`MapGraph`), procedural generator, and built-in maps.
- `pacman_engine/config.py`: Game rules, speeds, ghost personalities, scoring constants, and timeouts.
- `pacman_engine/state.py`: Mutable game state (entity positions, directions, active pellets, timers, mode flags).
- `pacman_engine/engine.py`: Core simulation loop, tick resolution, and event dispatch.
- `pacman_engine/controllers/`: Ghost AI controllers (classic Chase/Scatter/Frightened, random, heuristic) and human keyboard input.
- `pacman_engine/observations/`: State encodings (flat vector, feature tensors, spatial grid/CNN representations) for RL.
- `pacman_engine/rendering/`: Headless frame renderer (Pillow / RGB array) and optional display surface.
- `pacman_engine/play.py`: Interactive desktop runner utilizing `pygame-ce` (lazy import).
- `pacman_engine/recording.py`: Trajectory recorder and replay serializer.

*Dependency Rule*: Engine core packages (`types`, `maps`, `config`, `state`, `engine`, `controllers`, `observations`) MUST NEVER import `pygame` or any GUI library.

---

## 7. ASCII Map Specification & Legend

### Legend Characters
- `#`: Wall (impassable for all entities).
- `.`: Regular pellet.
- `o`: Power pellet.
- `_` or ` ` (space): Empty walkable floor.
- `P`: Pacman spawn position (must have exactly one).
- `0`-`9`: Ghost spawn position labeled with that ghost ID (0 to 9; each ID unique).
- `=`: Ghost door (ghosts may pass; Pacman may not pass).

*Spawn Rule*: Spawn cells (`P`, `0`-`9`) never contain pellets.

### Optional Header Metadata
Map files may include leading header lines prefixed with `;` in `key: value` format:
- `; name: <string>`
- `; description: <string>`
- `; wrap_horizontal: <bool>` (default: `false`)
- `; wrap_vertical: <bool>` (default: `false`)

### Wrap-Around Mechanics
When `wrap_horizontal` is true, row cells at opposite edges `(r, 0)` and `(r, cols - 1)` are adjacent if both edge cells are walkable (not walls). The same rule applies vertically when `wrap_vertical` is true for columns `(0, c)` and `(rows - 1, c)`.

### Runtime ASCII Overlay Legend (`render_ascii`)
When inspecting or debugging active game state via `PacmanGame.render_ascii()`:
- `P`: Pacman.
- `0`-`9`: Ghost in normal mode (SCATTER or CHASE) labeled with its ID.
- `F`: Ghost in FRIGHTENED mode.
- `X`: Ghost in DEAD mode (respawning).
- `!`: Pacman and a ghost occupying the same cell (collision).
- `#`: Impassable wall.
- `=`: Ghost door.
- `o`: Active power pellet.
- `.`: Active standard pellet.
- ` ` (space): Empty walkable corridor.

---

## 8. GameView Specification

The `GameView` object provides an immutable, read-only state snapshot passed to controllers (`act()`) and observation builders at the beginning of each tick:

- `pacman_pos: Position`: Current (row, col) coordinates of Pacman.
- `ghost_positions: dict[int, Position]`: Mapping from ghost ID (0-9) to current coordinates.
- `positions: dict[int | None, Position]`: Unified mapping of entity IDs to positions (`None` for Pacman, `int` for ghosts).
- `pacman_direction: Action`: Current facing direction of Pacman.
- `ghost_directions: dict[int, Action]`: Mapping from ghost ID to current movement direction.
- `directions: dict[int | None, Action]`: Unified mapping of entity IDs to directions.
- `ghost_modes: dict[int, GhostMode]`: Mapping from ghost ID to active `GhostMode` (SCATTER, CHASE, FRIGHTENED, DEAD).
- `ghost_timers: dict[int, int]`: Remaining countdown timer per ghost (respawn delay if DEAD, or global frightened duration if FRIGHTENED).
- `frightened_timer: int`: Global ticks remaining in frightened mode (0 when inactive).
- `pellets: np.ndarray`: Read-only 2D boolean array of remaining standard pellets (`flags.writeable = False`).
- `power_pellets: np.ndarray`: Read-only 2D boolean array of remaining power pellets (`flags.writeable = False`).
- `lives: int`: Remaining Pacman lives.
- `score: int`: Cumulative classic score.
- `tick: int`: Simulation tick index (0-indexed).
- `remaining_pellets: int`: Total count of active pellets and power pellets.
- `graph: MapGraph`: Reference to the cached map topology graph for shortest paths and neighbor queries.

---

## 9. Writing Your Own Controller

The engine provides a pluggable controller architecture allowing custom Pacman and ghost agents without modifying engine internals.

### Custom Pacman Controller

To create a custom controller, inherit from `BaseController` (or adhere to the `Controller` protocol) and decorate it with `@register_controller("name")`.

```python
import numpy as np
from pacman_engine.controllers import BaseController, register_controller
from pacman_engine.state import GameView
from pacman_engine.types import Action


@register_controller("safe_random")
class SafeRandomController(BaseController):
    """Picks a random legal move while strictly avoiding entering cells with ghosts."""

    def act(
        self,
        view: GameView,
        entity_id: int | None,
        rng: np.random.Generator,
    ) -> Action:
        curr = view.pacman_pos
        neighbors = view.graph.neighbors(curr, for_ghost=False)
        if not neighbors:
            return Action.STAY

        # Exclude neighbors currently occupied by any ghost
        ghost_cells = set(view.ghost_positions.values())
        safe_neighbors = [n for n in neighbors if n not in ghost_cells]
        candidates = safe_neighbors if safe_neighbors else neighbors

        actions = [view.graph.action_between(curr, n) for n in candidates]
        idx = int(rng.integers(0, len(actions)))
        return actions[idx]
```

### Custom Ghost Controller

Ghost controllers can inherit from `GhostController` which automatically manages `SCATTER`, `FRIGHTENED`, `allow_reverse`, and `epsilon`. Subclasses only need to implement `chase_target(view)`:

```python
from pacman_engine.controllers import GhostController, register_controller
from pacman_engine.state import GameView
from pacman_engine.types import Position


@register_controller("corner_camper")
class CornerCamperGhost(GhostController):
    """A ghost that camps at a specific coordinate during chase mode."""

    def __init__(self, target_row: int = 1, target_col: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.camp_target = Position(target_row, target_col)

    def chase_target(self, view: GameView) -> Position:
        return self.camp_target
```

### Using in YAML Configurations

Custom controllers registered via `@register_controller` can be referenced directly by name in YAML configurations:

```yaml
map: classic
pacman:
  controller:
    name: safe_random
ghosts:
  - ghost_id: 0
    controller:
      name: corner_camper
      params:
        target_row: 1
        target_col: 1
```

---

## 10. Observation Builders & RL State Representations

Observation builders provide structured state encodings for reinforcement learning agents by consuming strictly immutable `GameView` snapshots. No direct Gymnasium dependency exists in the engine core; observation shapes and boundaries are communicated via plain `ObservationSpec` objects:

```python
@dataclass(frozen=True)
class ObservationSpec:
    shape: tuple[int, ...]
    dtype: np.dtype
    low: float | np.ndarray
    high: float | np.ndarray
    feature_names: list[str] | None = None
```

### Observation Formats

#### 1. Flat Vector Builder (`"vector"`)
Encodes state into a 1-D `float32` vector:
- **Entity Coordinates**: Pacman `(row, col)` followed by each ghost `(row, col)` sorted by ghost ID (0-9). Supports `coordinates="absolute"` or `coordinates="relative_to_pacman"`.
- **Directions** (when `include_directions=True`): 4-element one-hot encoding per entity `[UP, DOWN, LEFT, RIGHT]` (all zeros if `STAY`).
- **Ghost Flags**: `is_frightened` and `is_dead` binary flags per ghost.
- **BFS Distances** (when `include_distances=True`): Shortest topological graph distance from Pacman to each ghost and to the nearest active pellet.
- **Global Features**: Remaining pellet fraction, frightened timer fraction, lives fraction, and step tick fraction.
- **Spatial Masks** (optional): `include_pellet_mask` and `include_wall_mask` flattened grid arrays.
- **Feature Names**: Every index is documented in `spec.feature_names`.

#### 2. Spatial Grid Builder (`"grid"`)
Encodes state into a multi-channel `float32` binary tensor of one-hot planes.

- **Default Channel Ordering**:
  1. `walls`: Map impassable wall cells.
  2. `doors`: Ghost spawn door cells.
  3. `pellets`: Active standard dot pellets.
  4. `power_pellets`: Active energizer power pellets.
  5. `pacman`: Pacman's current coordinate.
  6. `ghosts`: Active ghosts in normal mode (SCATTER or CHASE).
  7. `frightened_ghosts`: Ghosts currently in FRIGHTENED mode.

- **Optional Channels**:
  - `dead_ghosts`: Ghosts in DEAD mode returning to the spawn box.
  - `all_ghosts`: Ghosts in any mode.
  - `pacman_direction`: Expands into 4 directional planes (`pacman_dir_up`, `pacman_dir_down`, `pacman_dir_left`, `pacman_dir_right`).

- **Layout and Cropping**:
  - `channels_first`: `(C, H, W)` when `True` (default), or `(H, W, C)` when `False`.
  - `view_radius`: Optional integer radius $R$ producing an egocentric $(2R+1, 2R+1)$ window centered on Pacman. Coordinates outside the map boundary are padded with wall cells.

#### 3. Visual RGB Frame Builder (`"rgb"`)
Wraps the headless renderer to output rendered image arrays (`dtype=uint8`, values in `[0, 255]`):
- `output_size`: Optional `(height, width)` resize target using Pillow area interpolation (`Resampling.BOX`).
- `grayscale`: Single-channel output `(H, W, 1)` or `(1, H, W)`.
- `channels_first`: `(C, H, W)` if True, `(H, W, C)` if False (default).

---

## 11. Headless Frame Renderer & Theme Contract

The `Renderer` in `pacman_engine.rendering` produces pixel-accurate `(H, W, 3)` frames from a `GameView`:
- **Theme Configuration** (`RenderTheme`): Defines `cell_size`, background, wall, door, pellet, energizer, Pacman, per-ghost colors, frightened flashing near expiry, and optional grid lines.
- **Center Pixel Contract**: For cell $(r, c)$ at pixel center $(cx, cy)$, the pixel strictly matches the cell's dominant element color (`wall_color`, `pellet_color`, `pacman_color`, or ghost color).
- **Pacman Mouth Wedge**: Drawn facing Pacman's current direction while preserving the exact center pixel color.
- **Headless Isolation**: Lazily initializes Pygame with `SDL_VIDEODRIVER=dummy` so rendering runs seamlessly in headless servers, CI runners, and Docker containers without an X display.

---

## 12. Interactive Play, Recording, and CLI Tools

### Interactive Desktop Viewer (`pacman-play`)
- Located in `pacman_engine.play`.
- Utilizes `pygame-ce` lazily to run interactive or autonomous game sessions.
- Features keyboard control (Arrow keys, WASD), pause (`P`), reset (`R`), and quit (`Q`, `Esc`).
- Supports debug overlay displaying live simulation metrics (tick, score, lives, entity coordinates) and ghost targeting indicators (`last_target`).
- Supports `--autoplay <controller>` mode to watch any registered Pacman agent in real-time.

### Episode Recording (`pacman-record`)
- Located in `pacman_engine.recording`.
- Exposes `record_episode(game, path, fps, cell_size, max_steps, ...)` for programmatic evaluation.
- Exports complete episode trajectories to animated GIFs (via Pillow) or discrete indexed PNG sequences.

### Map Management (`pacman-map`)
- Located in `pacman_engine.maps.cli`.
- Provides subcommands:
  - `list`: Enumerate all built-in maps shipped in the package.
  - `preview <name|path> --out <file.png>`: Render a static image preview of any map layout.
  - `generate --rows --cols --seed --ghosts --out <file>`: Procedurally generate valid symmetric maps.
  - `validate <path>`: Run structural, connectivity, and invariant checks on map files.

---

## 13. Deviations from the Original Contract

None. All implementations strictly adhere to the contracts, coordinate standards, tick resolution order, and modular boundary rules defined in this document.


