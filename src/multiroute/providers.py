"""
Model resolver: maps a bare model name to a provider-prefixed name
(e.g. "gpt-4o" -> "openai/gpt-4o") by inspecting the client's base_url
against the bundled models.yaml URL registry.

If the model already contains a "/" the name is returned unchanged.
If base_url is not provided, or no URL pattern matches, the model name is
returned unchanged so existing behaviour is preserved.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

_REGISTRY_PATH = Path(__file__).parent / "providers.yaml"


@lru_cache(maxsize=1)
def _load_url_registry() -> List[Tuple[str, str]]:
    """Load the YAML registry and return a list of (url_substring, provider) pairs.

    Substrings are lower-cased.  The list is sorted longest-first so more
    specific entries win (e.g. a longer hostname beats a shorter one).
    """
    with open(_REGISTRY_PATH, encoding="utf-8") as fh:
        data: Dict = yaml.safe_load(fh)

    providers: Dict[str, str] = (data or {}).get("providers", {})

    pairs: List[Tuple[str, str]] = [
        (pattern.lower(), provider) for pattern, provider in providers.items()
    ]

    # Longer patterns should be tested first (more specific wins)
    pairs.sort(key=lambda t: len(t[0]), reverse=True)
    return pairs


def resolve_model(model: str, base_url: Optional[str] = None) -> str:
    """Return a provider-prefixed model name based on the client's base_url.

    Parameters
    ----------
    model:
        The bare or already-prefixed model name (e.g. ``"gpt-4o"`` or
        ``"openai/gpt-4o"``).
    base_url:
        The base URL of the originating client (e.g.
        ``"https://api.openai.com/v1/"``).  When provided, the URL is matched
        against the registry to determine the provider prefix.

    Examples
    --------
    >>> resolve_model("gpt-4o", "https://api.openai.com/v1/")
    'openai/gpt-4o'
    >>> resolve_model("gpt-4o", "https://api.groq.com/openai/v1")
    'groq/gpt-4o'
    >>> resolve_model("openai/gpt-4o", "https://api.groq.com/openai/v1")
    'openai/gpt-4o'
    >>> resolve_model("gpt-4o")                    # no base_url → unchanged
    'gpt-4o'
    >>> resolve_model("gpt-4o", "https://unknown-proxy.example.com")  # no match → unchanged
    'gpt-4o'
    """
    if not model or "/" in model:
        return model

    if not base_url:
        return model

    lower_url = base_url.lower()
    for pattern, provider in _load_url_registry():
        if pattern in lower_url:
            return f"{provider}/{model}"

    return model
