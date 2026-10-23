from __future__ import annotations

from typing import Any

import httpx
import openai
from openai import OpenAI
from tenacity import retry_if_exception, stop_after_attempt, wait_exponential

from ..config import get_settings


def make_client() -> OpenAI:
    s = get_settings()
    return OpenAI(api_key=s.OPENAI_API_KEY, base_url=s.OPENAI_BASE_URL or None, timeout=300, max_retries=0)


def _is_transient(e: BaseException) -> bool:
    if isinstance(e, (openai.APIConnectionError, openai.APITimeoutError, openai.RateLimitError,
                      openai.InternalServerError, httpx.TransportError)):
        return True
    if isinstance(e, openai.APIStatusError):
        return e.status_code == 429 or e.status_code >= 500
    if isinstance(e, httpx.HTTPStatusError):
        return e.response.status_code == 429 or e.response.status_code >= 500
    return False


RETRY_KW: dict[str, Any] = {
    "retry": retry_if_exception(_is_transient),
    "stop": stop_after_attempt(3),
    "wait": wait_exponential(multiplier=2, min=2, max=20),
    "reraise": True,
}


def g(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)
