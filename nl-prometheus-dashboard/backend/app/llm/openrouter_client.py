from __future__ import annotations

import os
import json
from typing import Any

import httpx

from .base import BaseLLMClient, LLMConfigurationError, parse_json_object


class OpenRouterLLMClient(BaseLLMClient):
    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 60.0,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

        if not self.api_key:
            raise LLMConfigurationError("OPENROUTER_API_KEY is required for OpenRouterLLMClient")

    async def generate_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context_block = "" if context is None else f"\n\nContext JSON:\n{json.dumps(context, ensure_ascii=False)}"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"{user_prompt}{context_block}"},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return parse_json_object(content)
