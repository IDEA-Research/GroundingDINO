from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from typing import Any


class LLMConfigurationError(RuntimeError):
    pass


class LLMJsonParseError(ValueError):
    pass


class LLMOutputValidationError(ValueError):
    pass


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise LLMJsonParseError(f"LLM output is not valid JSON: {exc.msg}") from exc

    if not isinstance(parsed, dict):
        raise LLMJsonParseError("LLM output must be a JSON object")
    return parsed


class BaseLLMClient(ABC):
    @abstractmethod
    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Return one JSON object parsed from the provider response."""

