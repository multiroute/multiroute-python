# Agent Guidelines - multiroute-python

This document provides essential information for autonomous agents working on the `multiroute` project.

## Project Overview

`multiroute` is a library that provides high-availability wrappers for major LLM providers (OpenAI, Anthropic, Google). It automatically routes requests through a proxy (`api.multiroute.ai`) when a `MULTIROUTE_API_KEY` is present, falling back to the original provider if the proxy fails.

## Build and Test Commands

The project uses `uv` for dependency management and `pytest` for testing.

- **Install dependencies:** `uv sync`
- **Run all tests:** `uv run pytest`
- **Run a specific test file:** `uv run pytest tests/providers/test_openai.py`
- **Run a specific test case:** `uv run pytest tests/providers/test_openai.py -k test_name`
- **Check linting/types:** The project currently relies on standard Python practices. Use `ruff check .` if available, or follow existing patterns.

## Code Style Guidelines

### 1. Imports

- Standard library imports first.
- Third-party library imports second.
- Local project imports last.
- Use absolute imports (e.g., `from multiroute.openai.client import ...`).

### 2. Formatting & Structure

- Follow PEP 8 conventions.
- Use 4 spaces for indentation.
- Keep lines under 88-100 characters where possible.
- Class-based architecture for provider wrappers, inheriting from original SDK classes.

### 3. Typing

- Use type hints for all function signatures and class members.
- Import `Any`, `Optional`, `Union`, etc., from `typing`.
- Return types should be explicitly defined (e.g., `-> Any` if the SDK return type is complex).

### 4. Naming Conventions

- Classes: `PascalCase` (e.g., `MultirouteChatCompletions`).
- Functions/Methods: `snake_case` (e.g., `_is_multiroute_error`).
- Variables/Constants: `snake_case` or `UPPER_SNAKE_CASE` for constants (e.g., `MULTIROUTE_BASE_URL`).

### 5. Error Handling

- Use provider-specific error types for fallback logic.
- Implement helper functions like `_is_multiroute_error` to identify retryable/fallback conditions (e.g., 5xx, timeouts, connection errors).
- Always ensure state (like `base_url` or `api_key`) is restored in `finally` blocks when monkey-patching or wrapping clients.

### 6. Fallback Logic Pattern

Most wrappers follow this pattern:

1. Save original client configuration (`base_url`, `api_key`).
2. Try request via Multiroute proxy.
3. Catch specific errors using `_is_multiroute_error`.
4. Restore original configuration and retry with the native provider on failure.
5. Restore configuration in a `finally` block to prevent leakage.

## Development Environment

- Python Version: `>= 3.9`
- Core dependency: `openai` (used for the OpenAI client and for talking to the multiroute.ai proxy). Optional extras: `anthropic`, `google-genai` (install with `multiroute[anthropic]`, `multiroute[google]`, or `multiroute[all]`). Dev and tests use `multiroute[all]`.
- Proxy URL: `https://api.multiroute.ai/openai/v1`

## Testing Guidelines

- Use `respx` for mocking HTTP requests in tests.
- Ensure both synchronous and asynchronous versions of clients are tested.
- Verify fallback behavior by simulating proxy failures (e.g., 500 Internal Server Error).
- Examples of usage are located in the `examples/` directory.
