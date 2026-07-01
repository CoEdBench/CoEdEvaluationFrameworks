import os
import logging
import time
from typing import List, Dict, Optional
from openai import OpenAI, OpenAIError  # Import OpenAI official library

from src.domain.interfaces import ILLMClient
from src.domain.types import LLMResponse, TokenUsage

logger = logging.getLogger(__name__)


class DeepSeekClient(ILLMClient):
    """
    DeepSeek API client implementation (based on OpenAI official SDK)
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "",
                 temperature = 0.8, max_tokens = 7192, base_url="",
                 api_type: str = "chat", max_retries: int = 3):
        self.api_key = api_key
        if not self.api_key:
            raise ValueError("DeepSeek API Key is required.")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model = model
        self.base_url = base_url
        self.api_type = api_type  # "chat" or "completions"
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=300
        )

    def generate_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 1.0,
            max_tokens: int = 8192,
            json_mode: bool = True
    ) -> LLMResponse:

        last_exception: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(f"🤖 LLM call [{attempt}/{self.max_retries}]: "
                            f"model={self.model}, api_type={self.api_type}")

                if self.api_type == "completions":
                    # Build prompt
                    prompt_parts = []
                    for m in messages:
                        role = m.get("role", "user")
                        c = m.get("content", "")
                        prompt_parts.append(f"{role}: {c}")
                    prompt = "\n".join(prompt_parts)

                    raw = self.client.completions.with_raw_response.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        stream=False,
                    )
                    logger.info(f"Completions [{raw.status_code}] body[:500]: {raw.text[:500]}")
                    response = raw.parse()
                    if response.choices and len(response.choices) > 0:
                        content = response.choices[0].text
                    else:
                        logger.error(f"Completions no choices. Full response: {response}")
                        content = ""
                    usage_raw = response.usage

                elif self.api_type == "responses":
                    # Responses API (/v1/responses) — GPT-5 series models only
                    instructions = None
                    input_messages = []
                    for m in messages:
                        role = m.get("role", "user")
                        c = m.get("content", "")
                        if role == "system":
                            instructions = c
                        else:
                            input_messages.append({"role": role, "content": c})

                    params = dict(
                        model=self.model,
                        input=input_messages,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    )
                    if instructions:
                        params["instructions"] = instructions

                    response = self.client.responses.create(**params)

                    # Parse Responses API output
                    content = ""
                    if response.output and len(response.output) > 0:
                        for output_item in response.output:
                            if hasattr(output_item, "content") and output_item.content:
                                for content_block in output_item.content:
                                    if hasattr(content_block, "text") and content_block.text:
                                        content += content_block.text
                    usage_raw = response.usage

                else:
                    # default chat endpoint
                    params = {
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": self.max_tokens,
                        "stream": False,
                    }
                    response = self.client.chat.completions.create(**params)
                    content = response.choices[0].message.content
                    usage_raw = response.usage

                # Get token usage (compatible with chat/completions/responses usage formats)
                usage = usage_raw
                cached = 0
                if usage:
                    prompt_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0) or 0
                    completion_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0) or 0
                    total_tokens = getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens)

                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
                        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                    elif hasattr(usage, "input_tokens_details") and usage.input_tokens_details:
                        cached = getattr(usage.input_tokens_details, "cached_tokens", 0) or 0
                else:
                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0

                token_usage = TokenUsage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    prompt_cached_tokens=cached,
                    model_name=self.model
                )

                return LLMResponse(content=content, usage=token_usage)

            except OpenAIError as e:
                last_exception = e
                logger.warning(f"⚠️ LLM call [{attempt}/{self.max_retries}] failed: {e}")
                if attempt < self.max_retries:
                    wait = 2 ** attempt
                    logger.info(f"⏳ Retrying in {wait}s...")
                    time.sleep(wait)

        raise RuntimeError(f"LLM invocation failed after {self.max_retries} retries: {last_exception}")
