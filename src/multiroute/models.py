"""
Model resolver: maps a bare model name to a provider-prefixed name
(e.g. "gpt-4o" -> "openai/gpt-4o") using the bundled models.yaml registry.

If the model already contains a "/" the name is returned unchanged.
If no match is found in the registry the name is also returned unchanged so
existing behaviour is preserved.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

_REGISTRY_PATH = Path(__file__).parent / "models.yaml"


@lru_cache(maxsize=1)
def _load_registry() -> List[Tuple[str, str]]:
    """Load the YAML registry and return a flat list of (pattern, provider) pairs.

    Patterns are lower-cased model name prefixes so that versioned variants like
    "gpt-4o-2024-05-13" still match the "gpt-4o" entry.
    The list is sorted longest-pattern-first so more specific entries win.
    """
    with open(_REGISTRY_PATH, encoding="utf-8") as fh:
        data: Dict[str, List[str]] = yaml.safe_load(fh)

    pairs: List[Tuple[str, str]] = []
    for provider, models in (data or {}).items():
        for model in models or []:
            pairs.append((model.lower(), provider))

    # Longer patterns should be tested first (more specific wins)
    pairs.sort(key=lambda t: len(t[0]), reverse=True)
    return pairs


def resolve_model(model: str) -> str:
    """Return a provider-prefixed model name.

    Examples
    --------
    >>> resolve_model("gpt-4o")
    'openai/gpt-4o'
    >>> resolve_model("openai/gpt-4o")   # already prefixed → unchanged
    'openai/gpt-4o'
    >>> resolve_model("my-custom-model")  # unknown → unchanged
    'my-custom-model'
    """
    if not model or "/" in model:
        return model

    lower = model.lower()
    for pattern, provider in _load_registry():
        # Match if the model name starts with the pattern (handles dated variants)
        if (
            lower == pattern
            or lower.startswith(pattern + "-")
            or lower.startswith(pattern + ".")
        ):
            return f"{provider}/{model}"

    return model
