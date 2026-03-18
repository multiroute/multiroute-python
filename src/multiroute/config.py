from __future__ import annotations

import os
from typing import Optional

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/openai/v1"


def get_multiroute_base_url() -> str:
    """Return the Multiroute base URL, read from ``os.environ`` at call time."""
    return os.environ.get("MULTIROUTE_BASE_URL") or MULTIROUTE_BASE_URL


def get_multiroute_api_key() -> Optional[str]:
    """Return the Multiroute API key, read from ``os.environ`` at call time."""
    return os.environ.get("MULTIROUTE_API_KEY") or None
