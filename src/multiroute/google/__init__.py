try:
    from .client import Client
except ImportError as e:
    raise ImportError(
        "Google support requires the google extra. "
        "Install with: pip install multiroute[google]",
    ) from e

__all__ = ["Client"]
