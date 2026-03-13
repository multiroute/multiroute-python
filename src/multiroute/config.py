"""Centralised configuration for the Multiroute SDK.

All provider clients import their shared constants from here rather than
defining them locally.  Settings are loaded once at import time from
environment variables (and an optional ``.env`` file) via pydantic-settings.

Environment variables
---------------------
MULTIROUTE_API_KEY
    Your Multiroute API key.  When set, requests are first attempted through
    the Multiroute proxy and fall back to the native provider on failure.
MULTIROUTE_BASE_URL
    Override the default proxy base URL
    (``https://api.multiroute.ai/openai/v1``).
"""

from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class MultirouteSettings(BaseSettings):
    """Pydantic-settings model for Multiroute SDK configuration.

    Fields are populated from environment variables (case-insensitive).
    An ``.env`` file in the working directory is also read if present.

    Note: ``api_key`` is intentionally excluded here because it must be
    read from ``os.environ`` at call time (not at import time) to stay
    compatible with runtime env-var changes (e.g. monkeypatching in tests).
    Use :func:`get_api_key` to retrieve it.
    """

    model_config = SettingsConfigDict(
        env_prefix="MULTIROUTE_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    base_url: str = "https://api.openai.com/v1"
    """Proxy base URL used by all OpenAI-compatible clients
    (env: ``MULTIROUTE_BASE_URL``)."""


# Module-level singleton — imported by all provider clients.
# Only holds static config (URLs); API key is read dynamically.
settings = MultirouteSettings()


def get_api_key() -> Optional[str]:
    """Return the Multiroute API key, read from ``os.environ`` at call time.

    Reading at call time (rather than at import time) ensures that runtime
    changes to the environment — including test monkeypatching — are always
    reflected.
    """
    return os.environ.get("MULTIROUTE_API_KEY") or None
