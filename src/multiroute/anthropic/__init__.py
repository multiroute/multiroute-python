try:
    from .client import Anthropic, AsyncAnthropic
except ImportError as e:
    raise ImportError(
        "Anthropic support requires the anthropic extra. "
        "Install with: pip install multiroute[anthropic]",
    ) from e

__all__ = ["Anthropic", "AsyncAnthropic"]
