import time
import json
from typing import List, Dict, Optional
from cachetools import TTLCache
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from app.config import get_settings
from app.models import ModelUsage

# Simple response cache for identical prompts (TTL: 5 minutes)
_response_cache: TTLCache = TTLCache(maxsize=100, ttl=300)


class ModelRouter:
    def __init__(self):
        self.settings = get_settings()
        self.logs: List[ModelUsage] = []
        self._llms: dict = {}

    def _get_llm(self, model: str, temperature: float, max_tokens: int) -> ChatOpenAI:
        """Get or create a ChatOpenAI instance for the given configuration."""
        key = (model, temperature, max_tokens)
        if key not in self._llms:
            self._llms[key] = ChatOpenAI(
                model=model,
                api_key=self.settings.openrouter_api_key,
                base_url=self.settings.openrouter_base_url,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        return self._llms[key]

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
        cache_hash = None
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

        # Convert dict messages to LangChain message objects
        lc_messages = self._to_langchain_messages(messages)

        llm = self._get_llm(selected_model, temp, max_tok)

        start_time = time.time()
        response = await llm.ainvoke(lc_messages)
        latency_ms = round((time.time() - start_time) * 1000, 2)

        # Extract token usage from response metadata
        usage_meta = getattr(response, "usage_metadata", None) or {}
        tokens_in = usage_meta.get("input_tokens", 0)
        tokens_out = usage_meta.get("output_tokens", 0)

        # Fallback to response_metadata if usage_metadata is empty
        if not tokens_in and not tokens_out:
            resp_meta = getattr(response, "response_metadata", {})
            token_usage = resp_meta.get("token_usage", {})
            tokens_in = token_usage.get("prompt_tokens", 0)
            tokens_out = token_usage.get("completion_tokens", 0)

        model_usage = ModelUsage(
            model=selected_model,
            tokens_input=tokens_in,
            tokens_output=tokens_out,
            latency_ms=latency_ms,
            cached_tokens=0,
        )
        self.logs.append(model_usage)

        content = response.content

        result = {"content": content, "model_usage": model_usage}

        # Cache the response
        if use_cache and cache_hash is not None:
            _response_cache[cache_hash] = result

        return result

    @staticmethod
    def _to_langchain_messages(messages: List[Dict]) -> list:
        """Convert dict messages to LangChain message objects."""
        lc_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                lc_msgs.append(SystemMessage(content=content))
            elif role == "user":
                lc_msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                lc_msgs.append(AIMessage(content=content))
        return lc_msgs

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
        """Clean up resources."""
        self._llms.clear()
