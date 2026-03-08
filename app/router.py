import httpx
import time
import json
from typing import List, Dict, Optional
from cachetools import TTLCache
from app.config import get_settings
from app.models import ModelUsage

# Simple response cache for identical prompts (TTL: 5 minutes)
_response_cache: TTLCache = TTLCache(maxsize=100, ttl=300)


class ModelRouter:
    def __init__(self):
        self.settings = get_settings()
        self.logs: List[ModelUsage] = []
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.settings.openrouter_base_url,
                headers={
                    "Authorization": f"Bearer {self.settings.openrouter_api_key}",
                    "Content-Type": "application/json",
                },
                timeout=120.0,
            )
        return self._client

    def select_model(
        self, preference: Optional[str] = None, complexity: str = "normal"
    ) -> str:
        if preference and preference in (
            self.settings.primary_model,
            self.settings.secondary_model,
        ):
            return preference

        if complexity == "simple":
            return self.settings.secondary_model
        return self.settings.primary_model

    async def complete(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        complexity: str = "normal",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        use_cache: bool = True,
    ) -> Dict:
        selected_model = model or self.select_model(complexity=complexity)
        temp = temperature if temperature is not None else self.settings.temperature
        max_tok = max_tokens or self.settings.max_tokens

        # Check response cache
        if use_cache:
            cache_key = json.dumps(
                {"model": selected_model, "messages": messages, "temp": temp},
                sort_keys=True,
            )
            cache_hash = hash(cache_key)
            if cache_hash in _response_cache:
                cached = _response_cache[cache_hash]
                usage = cached["model_usage"]
                cached_usage = ModelUsage(
                    model=usage.model,
                    tokens_input=usage.tokens_input,
                    tokens_output=usage.tokens_output,
                    latency_ms=0.0,
                    cached_tokens=usage.tokens_input,
                )
                return {"content": cached["content"], "model_usage": cached_usage}

        # Apply Anthropic prompt caching for Claude models
        formatted_messages = self._apply_prompt_caching(messages, selected_model)

        payload = {
            "model": selected_model,
            "messages": formatted_messages,
            "temperature": temp,
            "max_tokens": max_tok,
        }

        start_time = time.time()

        response = await self.client.post("/chat/completions", json=payload)
        response.raise_for_status()
        data = response.json()

        latency_ms = round((time.time() - start_time) * 1000, 2)

        usage = data.get("usage", {})
        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        cached_tokens = (
            usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        )

        model_usage = ModelUsage(
            model=selected_model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            cached_tokens=cached_tokens,
        )
        self.logs.append(model_usage)

        content = data["choices"][0]["message"]["content"]

        result = {"content": content, "model_usage": model_usage}

        # Cache the response
        if use_cache:
            _response_cache[cache_hash] = result

        return result

    def _apply_prompt_caching(
        self, messages: List[Dict], model: str
    ) -> List[Dict]:
        """Apply Anthropic prompt caching for Claude models.

        Static system prompts are marked with cache_control so Claude
        reuses the cached prefill on subsequent turns — cache hits cost
        only 10% of standard input pricing.
        """
        if "anthropic" not in model:
            return messages

        formatted = []
        for msg in messages:
            new_msg = dict(msg)
            if msg["role"] == "system":
                new_msg["content"] = [
                    {
                        "type": "text",
                        "text": msg["content"],
                        "cache_control": {"type": "ephemeral"},
                    }
                ]
            formatted.append(new_msg)

        return formatted

    def get_logs(self, limit: int = 50) -> List[ModelUsage]:
        return self.logs[-limit:]

    def get_available_models(self) -> List[str]:
        return [self.settings.primary_model, self.settings.secondary_model]

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
