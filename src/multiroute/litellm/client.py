import copy
import logging

import httpx
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    InternalServerError,
    NotFoundError,
    ServiceUnavailableError,
    Timeout,
)

from multiroute.config import get_api_key, settings

try:
    import litellm
except ImportError:
    litellm = None


_MISSING_KEY_MESSAGE = (
    "MULTIROUTE_API_KEY is not set. Requests will go directly to the provider "
    "without Multiroute high-availability routing."
)


def _is_multiroute_error(e: Exception) -> bool:
    if litellm is None:
        return False

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

    if isinstance(e, httpx.RequestError):
        return True

    return bool(
        isinstance(e, httpx.HTTPStatusError)
        and (e.response.status_code >= 500 or e.response.status_code == 404),
    )


def completion(**kwargs):
    if litellm is None:
        raise ImportError(
            "litellm is not installed. Please install it with `pip install litellm`.",
        )

    mr_api_key = kwargs.pop("multiroute_api_key", None) or get_api_key()
    if not mr_api_key:
        logging.error(_MISSING_KEY_MESSAGE)
        return litellm.completion(**kwargs)

    mr_kwargs = copy.copy(kwargs)
    mr_kwargs["api_base"] = settings.base_url
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
            "litellm is not installed. Please install it with `pip install litellm`.",
        )

    mr_api_key = kwargs.pop("multiroute_api_key", None) or get_api_key()
    if not mr_api_key:
        logging.error(_MISSING_KEY_MESSAGE)
        return await litellm.acompletion(**kwargs)

    mr_kwargs = copy.copy(kwargs)
    mr_kwargs["api_base"] = settings.base_url
    mr_kwargs["api_key"] = mr_api_key
    mr_kwargs["custom_llm_provider"] = "openai"

    try:
        return await litellm.acompletion(**mr_kwargs)
    except Exception as e:
        if _is_multiroute_error(e):
            return await litellm.acompletion(**kwargs)
        raise
