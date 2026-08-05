from __future__ import annotations

import os
import json
from typing import Any

import httpx

from .base import BaseLLMClient, LLMConfigurationError, parse_json_object


class LocalLLMClient(BaseLLMClient):
    """OpenAI-compatible local LLM adapter.

    TODO: Add provider-specific adapters for Ollama, llama.cpp server, and vLLM
    when their deployment contract is known.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.base_url = (base_url or os.getenv("LOCAL_LLM_BASE_URL") or "").rstrip("/")
        self.model = model or os.getenv("LOCAL_LLM_MODEL", "local-model")
        self.timeout_seconds = timeout_seconds

        if not self.base_url:
            raise LLMConfigurationError("LOCAL_LLM_BASE_URL is required for LocalLLMClient")

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
        }
        async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
            response = await client.post(f"{self.base_url}/chat/completions", json=payload)
            response.raise_for_status()
            data = response.json()

        content = data["choices"][0]["message"]["content"]
        return parse_json_object(content)
