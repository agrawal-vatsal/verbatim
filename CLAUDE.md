# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

`verbatim` is a Python 3.13+ project managed with [uv](https://github.com/astral-sh/uv).

## Common Commands

```bash
# Install dependencies
uv sync

# Add a dependency
uv add <package>

# Run a script
uv run <script>

# Run tests (once a test framework is added)
uv run pytest

# Run a single test
uv run pytest path/to/test_file.py::test_name
```
