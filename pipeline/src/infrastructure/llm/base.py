from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.domain.types import TokenUsage

class BaseLLMClient(ABC):
    @abstractmethod
    def chat_completion(self, messages: List[Dict[str, str]], temperature: float = 0.0) -> (str, TokenUsage):
        """
        Execute chat completion
        :return: (content_string, usage_object)
        """
        pass
