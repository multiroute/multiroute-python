import os
import copy

try:
    import litellm
except ImportError:
    litellm = None

MULTIROUTE_BASE_URL = "https://api.multiroute.ai/v1"


def _is_multiroute_error(e: Exception) -> bool:
    if litellm is None:
        return False

    from litellm.exceptions import (
        APIConnectionError,
        APIError,
        InternalServerError,
        NotFoundError,
        ServiceUnavailableError,
        Timeout,
    )

    if isinstance(
        e,
        (
            APIConnectionError,
            Timeout,
            ServiceUnavailableError,
            InternalServerError,
            NotFoundError,
        ),
    ):
        return True

    if isinstance(e, APIError):
        status_code = getattr(e, "status_code", None)
        if status_code and (status_code >= 500 or status_code == 404):
            return True

    # Also catch httpx errors just in case litellm leaks them
    import httpx

    if isinstance(e, httpx.RequestError):
        return True
    if isinstance(e, httpx.HTTPStatusError):
        if e.response.status_code >= 500 or e.response.status_code == 404:
            return True

    return False


def completion(**kwargs):
    if litellm is None:
        raise ImportError(
            "litellm is not installed. Please install it with `pip install litellm`."
        )

    mr_api_key = os.environ.get("MULTIROUTE_API_KEY")
    if not mr_api_key:
        return litellm.completion(**kwargs)

    mr_kwargs = copy.copy(kwargs)
    mr_kwargs["api_base"] = MULTIROUTE_BASE_URL
    mr_kwargs["api_key"] = mr_api_key
    mr_kwargs["custom_llm_provider"] = "openai"

    try:
        return litellm.completion(**mr_kwargs)
    except Exception as e:
        if _is_multiroute_error(e):
            return litellm.completion(**kwargs)
        raise


async def acompletion(**kwargs):
    if litellm is None:
        raise ImportError(
            "litellm is not installed. Please install it with `pip install litellm`."
        )

    mr_api_key = os.environ.get("MULTIROUTE_API_KEY")
    if not mr_api_key:
        return await litellm.acompletion(**kwargs)

    mr_kwargs = copy.copy(kwargs)
    mr_kwargs["api_base"] = MULTIROUTE_BASE_URL
    mr_kwargs["api_key"] = mr_api_key
    mr_kwargs["custom_llm_provider"] = "openai"

    try:
        return await litellm.acompletion(**mr_kwargs)
    except Exception as e:
        if _is_multiroute_error(e):
            return await litellm.acompletion(**kwargs)
        raise
