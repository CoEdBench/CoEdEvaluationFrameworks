import os
import json
import logging
import requests
from typing import List, Dict, Any, Optional
from src.domain.interfaces import ILLMClient
from src.domain.types import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class OpenAIClient(ILLMClient):
    """
    OpenAI-compatible API client (supports GPT-4, GPT-3.5, etc.)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4o", base_url: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        # OpenAI official API does not require base_url, but compatible interfaces (e.g., Ollama) do
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
        self.model = model

    def generate_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 0.0,
            max_tokens: int = 1024,
            json_mode: bool = True
    ) -> LLMResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            logger.debug(f"Sending request to OpenAI: {self.model}")

            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=60
            )
            response.raise_for_status()

            data = response.json()
            content = data["choices"][0]["message"]["content"]

            usage_data = data.get("usage", {})
            token_usage = TokenUsage(
                prompt_tokens=usage_data.get("prompt_tokens", 0),
                completion_tokens=usage_data.get("completion_tokens", 0),
                total_tokens=usage_data.get("total_tokens", 0),
                model_name=self.model
            )

            return LLMResponse(content=content, usage=token_usage)

        except Exception as e:
            logger.error(f"OpenAI API Request failed: {str(e)}")
            raise RuntimeError(f"LLM invocation failed: {str(e)}")
