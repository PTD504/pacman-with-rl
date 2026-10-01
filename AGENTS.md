# Pacman Game Engine — Agent Guidelines & Conventions

## Overview
This repository contains the foundation of a customizable, headless-first Pacman game engine designed for reinforcement learning.
Speed, determinism, and a clean API take precedence over visual polish.

## Project Rules & Constraints
1. **Python Version & Package Management**:
   - Python 3.12 managed with `uv`.
   - All code, comments, docstrings, identifiers, test names, and documentation MUST be written in English.
2. **GUI / Rendering Isolation**:
   - The engine core MUST NOT import `pygame` or any GUI library.
   - Only rendering and interactive play modules (in later sessions) may import `pygame`, and they must do so lazily.
3. **RL Isolation**:
   - No Gymnasium dependency at this stage; a gymnasium wrapper will be introduced in a future phase.
4. **Source of Truth**:
   - Architecture and design decisions are specified in `docs/ARCHITECTURE.md`. Any future modifications must remain consistent with this contract.

## Development Commands
- **Lint**: `uv run ruff check` (fix: `uv run ruff check --fix`)
- **Format check**: `uv run ruff format --check` (format: `uv run ruff format`)
- **Run tests**: `uv run pytest -q`
