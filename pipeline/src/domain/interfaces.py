import json
import logging
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any
from src.domain.types import LLMResponse

from json_repair import repair_json

logger = logging.getLogger(__name__)


class ILLMClient(ABC):
    """
    LLM Client abstract base class
    """

    @abstractmethod
    def generate_completion(
            self,
            messages: List[Dict[str, str]],
            temperature: float = 1.0,
            max_tokens: int = 32768,
            json_mode: bool = True
    ) -> LLMResponse:
        """
        Send chat request and get response content along with token consumption
        """
        pass

    def parse_json_response(
            self,
            response_content: str,
            raise_on_error: bool = False
    ) -> Dict[str, Any]:
        """
        [Generic method] Clean and parse JSON string returned by LLM.

        Handles the following common LLM output formats:
          1. Plain JSON string
          2. ```json ... ``` wrapped JSON
          3. ``` ... ``` wrapped JSON
          4. JSON with extra descriptive text before/after
          5. Truncated JSON due to token cutoff (requires json_repair)

        Args:
            response_content: Raw string returned by LLM
            raise_on_error: If True, raises exception on parse failure; if False, returns empty dict (default)

        Returns:
            Parsed dictionary, or {} on failure (raise_on_error=False)
        """
        if not response_content or not response_content.strip():
            logger.warning("parse_json_response: received empty content")
            return {}

        cleaned = response_content.strip()
        cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
        # ── Step 1: Prefer extracting ```json block, then regular code block ──────────
        # First try exact ```json match
        code_block_match = re.search(r"```json\s*(.*?)```", cleaned, re.DOTALL)
        if not code_block_match:
            # Then try regular ``` block, but require content starting with { or [ (ensuring it is JSON)
            code_block_match = re.search(r"```\s*(\{.*?\}|\[.*?\])\s*```", cleaned, re.DOTALL)

        if code_block_match:
            cleaned = code_block_match.group(1).strip()
        else:
            # Step 2: No code block, extract first complete JSON object or array
            json_match = re.search(r"(\{.*\}|\[.*\])", cleaned, re.DOTALL)
            if json_match:
                cleaned = json_match.group(1).strip()

        # ── Step 3: Try direct parsing ─────────────────────────────────────────────
        try:
            parsed = json.loads(cleaned)
            # Handle case where LLM returns list instead of dict (e.g. [{...}])
            if isinstance(parsed, list):
                if len(parsed) == 1 and isinstance(parsed[0], dict):
                    logger.warning("LLM returned list instead of dict, auto-extracting first element")
                    return parsed[0]
                logger.warning(f"LLM returned list (length {len(parsed)}), cannot auto-handle")
                return {}
            return parsed
        except json.JSONDecodeError as e:
            logger.warning(f"Standard JSON parsing failed, attempting fault-tolerant repair. Error: {e}")

        # ── Step 4: Try json_repair for fault-tolerant repair (handles truncated JSON)──
        try:
            repaired = repair_json(cleaned)
            result = json.loads(repaired)
            logger.warning("json_repair succeeded, recommend checking LLM output quality")
            return result
        except ImportError:
            logger.debug("json_repair not installed, skipping fault-tolerant repair step")
        except (json.JSONDecodeError, Exception) as e:
            logger.error(f"json_repair still failed after repair: {e}")

        # ── Step 5: All attempts failed, handle per strategy ────────────────────────
        logger.error(
            f"JSON parsing completely failed.\n"
            f"Original content:\n{response_content}\n"
            f"Cleaned content:\n{cleaned}"
        )
        if raise_on_error:
            raise ValueError(
                f"Unable to parse JSON from LLM response.\nOriginal content:\n{response_content}"
            )
        return {}
