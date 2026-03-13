"""
Model resolver: maps a bare model name to a provider-prefixed name
(e.g. "gpt-4o" -> "openai/gpt-4o") by extracting the hostname from the
client's base_url and looking it up in the bundled providers.yaml registry.

If the model already contains a "/" the name is returned unchanged.
If base_url is not provided, or the hostname doesn't match any known provider,
the model name is returned unchanged.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional

import yaml

_REGISTRY_PATH = Path(__file__).parent / "providers.yaml"

# Extracts the hostname from a URL (scheme://hostname/path?query).
_HOSTNAME_RE = re.compile(r"https?://([^/:?#]+)", re.IGNORECASE)


@lru_cache(maxsize=1)
def _load_providers() -> Dict[str, str]:
    """Load providers.yaml and return a {hostname: provider} dict."""
    with open(_REGISTRY_PATH, encoding="utf-8") as fh:
        data: Dict = yaml.safe_load(fh)

    return {k.lower(): v for k, v in ((data or {}).get("providers", {})).items()}


def _extract_hostname(base_url: str) -> Optional[str]:
    """Return the lower-cased hostname from a URL, or None if unparseable."""
    m = _HOSTNAME_RE.match(base_url.strip())
    return m.group(1).lower() if m else None


def resolve_model(model: str, base_url: Optional[str] = None) -> str:
    """Return a provider-prefixed model name based on the client's base_url.

    Parameters
    ----------
    model:
        The bare or already-prefixed model name (e.g. ``"gpt-4o"`` or
        ``"openai/gpt-4o"``).
    base_url:
        The base URL of the originating client (e.g.
        ``"https://api.openai.com/v1/"``).  The hostname is extracted and
        looked up in the registry to determine the provider prefix.

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

    hostname = _extract_hostname(base_url)
    if not hostname:
        return model

    providers = _load_providers()

    # Exact match first, then suffix match to handle wildcard subdomains
    # (e.g. "my-resource.openai.azure.com" matches the "openai.azure.com" entry).
    provider = providers.get(hostname)
    if provider is None:
        for domain, p in providers.items():
            if hostname.endswith("." + domain):
                provider = p
                break

    if provider:
        return f"{provider}/{model}"

    return model
