# Pacman Engine

A customizable, headless-first, deterministic Pacman game simulation engine designed for Reinforcement Learning (RL) research, benchmarking, and interactive play.

Built with Python 3.12, NumPy, Pillow, PyYAML, and lazy Pygame-ce integration.

---

## Features

- **Headless-First & High Performance**: Core engine runs purely with integer arithmetic and bitwise-exact determinism without requiring any display or GUI dependencies.
- **Deterministic Trajectories**: SeedSequence-based independent PRNG partitioning guarantees bitwise reproducibility across runs.
- **Flexible Observation Formats**:
  - `vector`: 1D normalized features, BFS topological distances, and spatial masks.
  - `grid`: Multi-channel binary spatial planes (walls, doors, pellets, Pacman, ghosts) with optional egocentric cropping.
  - `rgb`: Direct visual frame rendering with custom color themes and resolution scaling.
- **Pluggable Controller Architecture**: Decoupled Pacman and ghost agents with built-in classic behaviors (Blinky Chase, Pinky Ambush, Inky Flank, Clyde Patrol, Frightened fleeing, Greedy Pellet heuristic).
- **Procedural Map Generator & ASCII Legend**: Generate topologically valid, symmetric maps with configurable loops or parse standard ASCII layout files.
- **CLI Tools**: Interactive desktop play (`pacman-play`), GIF/PNG episode recorder (`pacman-record`), and map toolkit (`pacman-map`).

---

## Installation

Install using [`uv`](https://github.com/astral-sh/uv):

```bash
# Clone the repository
git clone https://github.com/PTD504/pacman-with-rl.git
cd pacman-with-rl

# Install dependencies and sync virtual environment
uv sync
```

To run commands or tests using the virtual environment:
```bash
uv run pytest -q
```

---

## Quick Start Snippets

### 1. Run a Headless Episode

```python
from pacman_engine import GameConfig, build_game

# Configure a game session on the classic map with a heuristic Pacman controller
config = GameConfig.from_dict(
    {
        "map": "classic",
        "pacman": {"controller": {"name": "greedy_pellet", "params": {"avoid_ghost_radius": 3}}},
    }
)

game = build_game(config, seed=42)

step_count = 0
while game.outcome.name == "RUNNING":
    step_result = game.step()
    step_count += 1

print(
    f"Episode completed in {step_count} ticks with outcome: {game.outcome.name}, Score: {game.view.score}"
)
```

### 2. Load and Edit a Custom Map

```python
from pacman_engine import GridMap, MapBuilder, Position

# Load an existing map from an ASCII string or file
ascii_map = """
#####
#P.0#
#o. #
#####
""".strip()

grid_map = GridMap.from_ascii(ascii_map)

# Edit map using MapBuilder
builder = MapBuilder.from_grid_map(grid_map)
builder.set_pellet(Position(2, 3), True)
builder.set_wall(Position(1, 2), False)
edited_map = builder.build()

print(edited_map.to_ascii())
```

### 3. Write a Custom Ghost Controller

```python
from pacman_engine import GameView, Position, register_controller
from pacman_engine.controllers import GhostController


@register_controller("patrol_line")
class LinePatrolGhost(GhostController):
    """Ghost that patrols back and forth between two row coordinates."""

    def __init__(self, start_row: int = 1, end_row: int = 5, target_col: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.start_row = start_row
        self.end_row = end_row
        self.target_col = target_col
        self._moving_down = True

    def chase_target(self, view: GameView) -> Position:
        curr_row = view.ghost_positions[self._current_entity_id].row
        if curr_row >= self.end_row:
            self._moving_down = False
        elif curr_row <= self.start_row:
            self._moving_down = True

        next_row = self.end_row if self._moving_down else self.start_row
        return Position(next_row, self.target_col)
```

### 4. Get RGB, Grid, and Vector Observations

```python
from pacman_engine import GameConfig, build_game_and_observation

# 1. Flat Vector Observation
cfg_vec = GameConfig.from_dict({"map": "small", "observation": "vector"})
game, vec_builder = build_game_and_observation(cfg_vec, seed=0)
vec_obs = vec_builder.build(game.view)
print("Vector observation shape:", vec_obs.shape, "Dtype:", vec_obs.dtype)

# 2. Multi-channel Spatial Grid Observation
cfg_grid = GameConfig.from_dict(
    {
        "map": "small",
        "observation": {
            "name": "grid",
            "params": {"channels_first": True, "view_radius": 5},
        },
    }
)
game, grid_builder = build_game_and_observation(cfg_grid, seed=0)
grid_obs = grid_builder.build(game.view)
print("Grid observation shape:", grid_obs.shape)

# 3. Headless Visual RGB Array
cfg_rgb = GameConfig.from_dict(
    {
        "map": "small",
        "observation": {
            "name": "rgb",
            "params": {"output_size": (84, 84), "grayscale": False},
        },
    }
)
game, rgb_builder = build_game_and_observation(cfg_rgb, seed=0)
rgb_obs = rgb_builder.build(game.view)
print("RGB observation shape:", rgb_obs.shape)
```

### 5. Record an Episode to an Animated GIF

```python
from pacman_engine import GameConfig, build_game, record_episode

config = GameConfig.from_dict(
    {
        "map": "small",
        "pacman": {"controller": {"name": "greedy_pellet"}},
    }
)

game = build_game(config, seed=123)
stats = record_episode(game, "gameplay.gif", fps=15, cell_size=24, max_steps=200)
print(f"Saved {stats['total_frames']} frames to {stats['path']}")
```

---

## Command-Line Interface Reference

### `pacman-play`
Interactive player and autonomous controller viewer. Simulation timing is decoupled from rendering via a fixed-timestep loop.

```bash
# Play interactively with classic arcade feel (default: classic_feel.yaml, 8 TPS, 60 FPS, 3 lives)
uv run pacman-play --map classic

# Custom simulation pace and rendering framerate
uv run pacman-play --map classic --tps 8 --fps 60

# Override initial lives
uv run pacman-play --map classic --lives 5

# Watch an autonomous controller play in real-time
uv run pacman-play --map classic --autoplay greedy_pellet --tps 12

# Run with debug overlay (coordinates, ghost targets, scores, lives, TPS, tick)
uv run pacman-play --map small --debug

# Load from a custom YAML configuration
uv run pacman-play --config configs/examples/aggressive_speeds.yaml
```

**Options**:
- `--tps <int>`: Target simulation ticks per second (default: `8` for comfortable human play).
- `--fps <int>`: Target display redraw frame rate (default: `60`). *Note on meaning change*: `--fps` now strictly dictates graphical rendering rate; game simulation speed is independently controlled by `--tps`.
- `--lives <int>`: Override initial lives count (>= 1).
- `--map <name|path>`: Built-in map name or path to ASCII map file.
- `--config <path>`: Path to YAML configuration file (defaults to `configs/examples/classic_feel.yaml` when omitted).
- `--autoplay <name>`: Watch an autonomous Pacman controller (e.g. `greedy_pellet`, `pacman_random`).
- `--debug`: Enable on-screen diagnostic HUD (TPS, tick, coordinates, ghost chase targets).
- `--cell-size <int>`: Pixel dimension per grid tile (default: `20`).

**Controls**:
- `Arrow Keys` or `W`, `A`, `S`, `D`: Steer Pacman (classic input buffering: persistent heading, early junction turn queuing, immediate reverse)
- `+` / `-`: Increase or decrease simulation speed live (clamped to 2–30 TPS)
- `P`: Toggle pause
- `R`: Reset episode
- `Q` or `Esc`: Quit

---

### `pacman-record`
Simulate an episode and export to GIF or PNG image sequence.

```bash
# Record an episode to a GIF
uv run pacman-record --map classic --controller greedy_pellet --out run.gif

# Record with step cutoff and specific FPS
uv run pacman-record --map small --controller pacman_random --max-steps 100 --fps 20 --out small.gif

# Export discrete PNG frames into a directory
uv run pacman-record --map classic --out frames/
```

---

### `pacman-map`
Tool for map discovery, previewing, procedural generation, and validation.

```bash
# List all built-in maps shipped with the package
uv run pacman-map list

# Render a PNG image preview of any map
uv run pacman-map preview classic --out classic_preview.png

# Procedurally generate a new valid, symmetric map
uv run pacman-map generate --rows 21 --cols 21 --ghosts 4 --seed 42 --out custom.map

# Validate a map file against all topological invariants
uv run pacman-map validate custom.map
```

---

## Configuration Reference Tables

### `RulesConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `pacman_lives` | `int` | `3` | Initial lives for Pacman (>= 1). |
| `frightened_speed_factor` | `float` | `1.0` | Ghost speed multiplier while in FRIGHTENED mode in `(0, 1]`. |
| `frightened_duration` | `int` | `30` | Ticks ghosts remain in FRIGHTENED mode when a power pellet is eaten. |
| `ghost_respawn_delay` | `int` | `10` | Ticks a consumed ghost spends returning/cooldown before respawning. |
| `max_steps` | `int \| None` | `1000` | Step limit before game ends with `Outcome.TIMEOUT`. |
| `pellet_score` | `int` | `10` | Points awarded for eating a standard pellet. |
| `power_pellet_score` | `int` | `50` | Points awarded for eating an energizer power pellet. |
| `ghost_score_base` | `int` | `200` | Points for first ghost eaten during frightened mode (doubles with combo). |
| `mode_schedule` | `list[tuple[GhostMode, int]]` | 7s/20s cycles | Alternating sequence of `(GhostMode, duration_ticks)` for global mode. |
| `allow_reversal_on_mode_switch` | `bool` | `True` | Whether ghosts reverse direction when global mode switches. |
| `random_pacman_start` | `bool` | `False` | If True, Pacman spawns at a random walkable floor cell. |
| `invalid_action_policy` | `'ignore' \| 'raise'` | `'ignore'` | Engine behavior if Pacman tries to walk into a wall/door. |

### `GhostConfig`

| Field | Type | Default | Description |
|---|---|---|---|
| `ghost_id` | `int` | Required (0-9) | Unique ID of the ghost. |
| `speed` | `float \| None` | `None` (1.0) | Fractional speed in `(0, 1]` via deterministic accumulator (mutually exclusive with `move_period`). |
| `move_period` | `int \| None` | `1` | Ghost moves once every $N$ ticks (mutually exclusive with `speed`). |
| `start_delay` | `int` | `0` | Delay ticks at spawn before the ghost starts navigating. |
| `scatter_target` | `Position \| None` | Default corner | Custom fixed coordinate targeted during SCATTER mode. |

### Controllers and Parameters

#### Pacman Controllers

| Name | Class | Parameters | Description |
|---|---|---|---|
| `external` | `ExternalController` | None | Actions are passed directly to `game.step(action)`. |
| `manual` | `ManualController` | `initial_action: Action \| str` | Action updated dynamically via `set_action()` (keyboard). |
| `pacman_random` | `RandomPacmanController` | None | Uniformly selects among legal non-wall neighbors. |
| `scripted` | `ScriptedController` | `actions: list`, `loop: bool` | Replays a fixed sequence of actions. |
| `greedy_pellet` | `GreedyPelletController` | `avoid_ghost_radius: int`, `hunt_frightened: bool` | BFS shortest-path navigation to pellets with ghost evasion. |

#### Ghost Controllers

| Name | Class | Parameters | Description |
|---|---|---|---|
| `chase` | `ChaseGhostController` | `scatter_target`, `frightened_behavior`, `allow_reverse`, `epsilon` | Blinky-style direct chase targeting Pacman's cell. |
| `ambush` | `AmbushGhostController` | `ahead_cells: int` (def: 4), base params | Pinky-style ambush targeting cells ahead of Pacman. |
| `flank` | `FlankGhostController` | `pivot_ghost_id: int` (def: 0), base params | Inky-style vector pivot flanking Pacman. |
| `patrol` | `PatrolGhostController` | `waypoints: list[Position | tuple]`, base params | Sequentially visits looping coordinate waypoints. |
| `ghost_random` | `RandomGhostController` | `allow_reverse: bool` | Selects uniform random legal moves without reversing. |


### Observation Builder Parameters

| Name | Parameters | Description |
|---|---|---|
| `vector` | `coordinates: 'absolute' \| 'relative_to_pacman'`, `include_directions: bool`, `include_distances: bool`, `include_pellet_mask: bool`, `include_wall_mask: bool` | 1D float32 tensor of normalized positions, flags, distances, and global timers. |
| `grid` | `channels_first: bool` (def: True), `view_radius: int \| None` (def: None), `include_direction_channels: bool`, `include_dead_ghosts_channel: bool`, `include_all_ghosts_channel: bool` | Multi-channel binary spatial planes for convolutional/spatial networks. |
| `rgb` | `cell_size: int` (def: 20), `output_size: tuple[int, int] \| None`, `grayscale: bool`, `channels_first: bool` (def: False), `theme: RenderTheme \| None` | Rendered pixel arrays from the headless renderer. |

---

## Gymnasium RL Environment (`PacmanEnv`)

The engine includes a full standard `gymnasium.Env` wrapper, ready to plug into **Stable-Baselines3**, **CleanRL**, or any custom training loop:

### 1. Quick Usage via `gym.make`
```python
import gymnasium as gym
import pacman_engine  # Registers Pacman environments

# Built-in registered environments
env = gym.make("Pacman-v0")  # Classic map, vector observation
# env = gym.make("PacmanSmall-v0")  # Small map, vector observation
# env = gym.make("PacmanRGB-v0")    # Classic map, 84x84 RGB observation

obs, info = env.reset(seed=42)
action = env.action_space.sample()  # 0: UP, 1: DOWN, 2: LEFT, 3: RIGHT, 4: STAY
next_obs, reward, terminated, truncated, info = env.step(action)
env.close()
```

### 2. Custom Environment Instantiation & Reward Customization
```python
from pacman_engine import PacmanEnv


# Define your custom reward function based on events and GameView
def custom_reward(step_result, view):
    reward = -0.01  # Step penalty
    for event in step_result.events:
        name = type(event).__name__
        if name == "PelletEaten":
            reward += 1.0
        elif name == "PowerPelletEaten":
            reward += 5.0
        elif name == "GhostEaten":
            reward += 20.0
        elif name == "PacmanDied":
            reward -= 20.0
    return reward


env = PacmanEnv(
    map_name="classic",
    obs_type="vector",  # 'vector', 'grid', or 'rgb'
    obs_params={"coordinates": "relative_to_pacman"},
    reward_func=custom_reward,
    lives=1,  # Single life for RL training episodes
    max_steps=1000,
    frame_skip=4,  # Repeat action for 4 simulation ticks
    frame_stack=4,  # Stack 4 consecutive observations
    render_mode="rgb_array",  # or "human"
)
```

### 3. Visual Atari-Style Setup (Grayscale 84x84 + Frame Stack & Skip)
```python
from pacman_engine import PacmanEnv

# Produces observation tensor of shape (4, 84, 84) with 4-tick action repeat
env = PacmanEnv(
    map_name="classic",
    obs_type="rgb",
    obs_params={
        "grayscale": True,
        "channels_first": True,
        "output_size": (84, 84),
    },
    frame_skip=4,
    frame_stack=4,
    lives=1,
)
```

### 4. Stable-Baselines3 One-Liner Integration
```python
from stable_baselines3 import PPO
from pacman_engine import PacmanEnv

env = PacmanEnv(map_name="small", obs_type="vector", lives=1)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100_000)
```
