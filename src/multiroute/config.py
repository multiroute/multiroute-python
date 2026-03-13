from __future__ import annotations

import os
from typing import Optional

from pydantic_settings import BaseSettings


class MultirouteSettings(BaseSettings):
    """Pydantic-settings model for Multiroute SDK configuration."""

    base_url: str = "https://api.multiroute.ai/openai/v1"


settings = MultirouteSettings()


def get_api_key() -> Optional[str]:
    """Return the Multiroute API key, read from ``os.environ`` at call time."""
    return os.environ.get("MULTIROUTE_API_KEY") or None
