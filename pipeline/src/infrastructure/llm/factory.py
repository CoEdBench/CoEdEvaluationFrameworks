import os
from src.domain.interfaces import ILLMClient
from src.infrastructure.llm.deepseek_client import DeepSeekClient
from src.infrastructure.llm.openai_client import OpenAIClient


def create_llm_client(provider: str, **kwargs) -> ILLMClient:
    """
    LLM client factory method
    :param provider: "deepseek" | "openai"
    :param kwargs: arguments passed to the specific client (e.g., api_key, model)
    """
    provider = provider.lower()

    if provider == "deepseek":
        return DeepSeekClient(**kwargs)

    elif provider == "openai":
        return OpenAIClient(**kwargs)

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
