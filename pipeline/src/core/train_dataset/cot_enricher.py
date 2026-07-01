"""
cot_enricher.py
===============
Invokes external LLM (DeepSeek / GPT) to generate CoT reasoning chains for training samples.

Ground truth output format:
    <think>
      ...model reasoning process...
    </think>
    <answer>
      { ...original JSON... }
    </answer>
"""

import asyncio
import json
import logging
import os
import re
from typing import List, Optional

import httpx

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# System Prompt
# ══════════════════════════════════════════════════════════════════════════

COT_SYSTEM_PROMPT = """\
You are an expert software engineer performing code change impact analysis.

## Output Format (MANDATORY)
You MUST output EXACTLY this structure — no exceptions:

<think>
[Your step-by-step reasoning here. REQUIRED. Cannot be empty or omitted.]
</think>
<answer>
[JSON object here]
</answer>

Skipping <think> or outputting only <answer> is INVALID.

## How to reason (inside <think>)
1. Read the root change diff carefully.
2. Identify what changed (variables, function signatures, logic, types, etc.).
3. Scan the provided code contexts for all references to the changed elements.
4. Derive which lines/files are impacted and explain why each one needs updating.
Do NOT speculate — all relevant code is already provided in the prompt above.
Reason FORWARD: analyze the change first, derive impacts, then verify against
the ground truth. Do NOT read the ground truth first and work backwards.

## How to fill <answer>
  - Stage1: "impacted_locations[].file", "impacted_locations[].lines"
  - Stage2: "next_version" (preserve null or non-null as-is)

REWRITE fields (must reflect your own analysis from <think>, do NOT copy from ground truth):
  - Stage1: "reasoning", "impacted_locations[].reason"
  - Stage2: "change_summary"
"""


# ══════════════════════════════════════════════════════════════════════════
# CoTEnricher
# ══════════════════════════════════════════════════════════════════════════

class CoTEnricher:
    """
    Batch-generates ground truth in <think>...</think><answer>...</answer> format for training samples.

    Supported providers:
      - "deepseek" → deepseek-reasoner (R1, native reasoning_content)
      - "openai"   → o3-mini
      - "custom"   → custom base_url (OpenAI-compatible protocol)
    """

    PROVIDER_DEFAULTS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com/v1",
            "model":    "deepseek-reasoner",
        },
        "openai": {
            "base_url": "https://api.openai.com/v1",
            "model":    "o3-mini",
        },
    }

    def __init__(
            self,
            provider:    str   = "deepseek",
            api_key:     str   = "",
            model:       str   = "",
            base_url:    str   = "",
            max_retries: int   = 3,
            retry_delay: float = 2.0,
            timeout:     float = 120.0,
            temperature: float = 1,
            max_tokens:  int   = 10240,
            concurrency: int   = 4,
    ):
        defaults         = self.PROVIDER_DEFAULTS.get(provider, {})
        self.base_url    = base_url or defaults.get("base_url", "")
        self.model       = model    or defaults.get("model", "")
        self.api_key     = api_key  or os.environ.get("API_KEY", "")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.timeout     = timeout
        self.temperature = temperature
        self.max_tokens  = max_tokens
        self._semaphore  = asyncio.Semaphore(concurrency)

        if not self.api_key:
            logger.warning("⚠️  CoTEnricher: no API key, will use fallback mode.")
        if not self.base_url:
            raise ValueError(f"Unknown provider '{provider}' and no base_url given.")

    # ──────────────────────────────────────────────────────────────
    # Public interface
    # ──────────────────────────────────────────────────────────────

    def enrich_batch(self, samples: List[dict]) -> List[dict]:
        """Synchronous entry: batch CoT enrichment."""
        return asyncio.run(self._enrich_batch_async(samples))

    # ──────────────────────────────────────────────────────────────
    # Async core
    # ──────────────────────────────────────────────────────────────

    async def _enrich_batch_async(self, samples: List[dict]) -> List[dict]:
        tasks = [self._enrich_one(s) for s in samples]
        return list(await asyncio.gather(*tasks))

    async def _enrich_one(self, sample: dict) -> dict:
        async with self._semaphore:
            try:
                cot_gt = await self._generate_cot(sample["messages"], sample["ground_truth"])
                return {
                    **sample,
                    "ground_truth":     cot_gt,
                    "ground_truth_raw": sample["ground_truth"],
                    "cot_enriched":     True,
                }
            except Exception as e:
                logger.error(f"❌ CoT failed for {sample.get('id')}: {e}. Fallback.")
                return {
                    **sample,
                    "ground_truth": (
                        f"<think>\n(reasoning not available)\n</think>\n"
                        f"<answer>\n{sample['ground_truth']}\n</answer>"
                    ),
                    "ground_truth_raw": sample["ground_truth"],
                    "cot_enriched":     False,
                }

    # ──────────────────────────────────────────────────────────────
    # LLM call
    # ──────────────────────────────────────────────────────────────

    async def _generate_cot(self, messages: List[dict], gt_json: str) -> str:
        """Construct meta-prompt, call LLM, return complete CoT string."""
        user_content = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            "(no user message)",
        )
        cot_messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {"role": "user",   "content": (
                f"## Inference Prompt (what the model sees at runtime)\n\n"
                f"{user_content}\n\n"
                f"---\n\n"
                f"## Ground Truth\n\n"
                f"```json\n{gt_json}\n```\n\n"
                f"## Your Task\n\n"
                f"1. Inside `<think>`: reason step-by-step about the change and its impact.\n"
                f"   - Analyze the root change FIRST, then derive impacts.\n"
                f"   - Do NOT work backwards from the Ground Truth.\n"
                f"2. Inside `<answer>`: output the complete JSON.\n"
                f"   - Copy Ground Truth into the correct positions.\n"
                f"   - Write your own values for all REWRITE fields:\n"
                f"     * Stage1: `reasoning`, each `impacted_locations[].reason`\n"
                f"     * Stage2: `change_summary`\n"
                f'''## Output Format (MANDATORY)\n\nYou MUST output EXACTLY this structure — no exceptions:\n<think>\n[Your step-by-step reasoning here. REQUIRED. Cannot be empty or omitted.]\n</think>\n <answer>\n```json\n[JSON object here]\n```\n</answer>
                '''
            )},
        ]
        for attempt in range(1, self.max_retries + 1):
            try:
                raw = await self._call_api(cot_messages)
                return self._parse_and_validate(raw, gt_json)
            except (ValueError, httpx.HTTPError) as e:
                logger.warning(f"⚠️  Attempt {attempt}/{self.max_retries}: {e}")
                if attempt < self.max_retries:
                    await asyncio.sleep(self.retry_delay * attempt)
                else:
                    raise

    async def _call_api(self, messages: List[dict]) -> str:
        """Call OpenAI-compatible API, return raw response text."""
        url     = f"{self.base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.model,
            "messages":    messages,
            # "temperature": self.temperature,
            # "max_tokens":  self.max_tokens,
        }

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            if not resp.is_success:
                # Print full error body for debugging
                logger.error(
                    f"API error {resp.status_code}: {resp.text[:500]}"
                )
            resp.raise_for_status()

        message   = resp.json()["choices"][0]["message"]
        reasoning = message.get("reasoning_content", "")  # DeepSeek-R1 native field
        content   = message.get("content", "").strip()

        # R1 native CoT: reasoning_content contains the thinking process, content is the final answer
        # Separate fields, need to assemble into unified format
        if reasoning:
            return (
                f"<think>\n{reasoning.strip()}\n</think>\n"
                f"<answer>\n{content}\n</answer>"
            )

        # Regular chat models (deepseek-chat / gpt etc.):
        # Content should contain the full <think>...</think><answer>...</answer>
        # Return directly, _parse_and_validate handles parsing validation
        return content

    # ──────────────────────────────────────────────────────────────
    # Parsing & consistency check
    # ──────────────────────────────────────────────────────────────

    def _parse_and_validate(self, text: str, original_gt: str) -> str:
        think  = re.search(r"<think>(.*?)</think>",   text, re.DOTALL)
        answer = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)

        if not think or not answer:
            raise ValueError(f"Missing <think> or <answer> tags: {text[:300]!r}")

        think_content = think.group(1).strip()
        if not think_content or think_content == "(reasoning not available)":
            raise ValueError("Empty or placeholder <think> content — retrying.")

        # Clean any markdown code blocks from answer
        ans = re.sub(r"^```(?:json)?\s*", "", answer.group(1).strip())
        ans = re.sub(r"\s*```$", "", ans).strip()

        try:
            ans_obj = json.loads(ans)
        except json.JSONDecodeError as e:
            raise ValueError(f"<answer> is not valid JSON: {e}")

        self._check_consistency(json.loads(original_gt), ans_obj)

        return (
            f"<think>\n{think_content}\n</think>\n"
            f"<answer>\n{json.dumps(ans_obj, ensure_ascii=False, indent=2)}\n</answer>"
        )

    @staticmethod
    def _check_consistency(original: dict, generated: dict) -> None:
        """Verify LOCKED fields were not modified by LLM."""
        # Stage1: impacted file list must match exactly
        if "impacted_locations" in original:
            orig_files = {loc["file"] for loc in original["impacted_locations"]}
            gen_files = {loc["file"] for loc in generated.get("impacted_locations", [])}
            if orig_files != gen_files:
                raise ValueError(
                    f"impacted_locations file mismatch: "
                    f"expected {orig_files}, got {gen_files}"
                )

            # Fix: same file may be split into multiple entries by model
            #   Aggregate all lines by file before comparing, not entry by entry
            def aggregate(locations: list) -> dict:
                result: dict = {}
                for loc in locations:
                    result.setdefault(loc["file"], set()).update(loc.get("lines", []))
                return {f: sorted(lines) for f, lines in result.items()}

            orig_lines_map = aggregate(original["impacted_locations"])
            gen_lines_map = aggregate(generated.get("impacted_locations", []))

            if orig_lines_map != gen_lines_map:
                raise ValueError(
                    f"impacted_locations lines mismatch: "
                    f"expected {orig_lines_map}, got {gen_lines_map}"
                )

        # Stage2: next_version null/non-null must match, and content must match when non-null
        if "next_version" in original:
            orig_nv = original["next_version"]
            gen_nv = generated.get("next_version")
            if (orig_nv is None) != (gen_nv is None):
                raise ValueError(
                    f"next_version null/non-null mismatch: "
                    f"expected null={orig_nv is None}, got null={gen_nv is None}"
                )
            if orig_nv is not None and orig_nv != gen_nv:
                raise ValueError("next_version content was modified by LLM.")
