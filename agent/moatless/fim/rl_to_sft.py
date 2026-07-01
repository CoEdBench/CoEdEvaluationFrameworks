"""
fim/rl_to_sft.py
Convert FIM trajectory traces to LlamaFactory ShareGPT format (JSONL).

Output format per line::

    {
      "system": "<system prompt>",
      "tools": "<JSON-encoded OpenAI tool schemas>",
      "conversations": [
        {"from": "human", "value": "<task prompt>"},
        {"from": "function_call", "value": '{"name": "ReadFile", "arguments": {...}}'},
        {"from": "observation", "value": "<tool output>"},
        ...
        {"from": "function_call", "value": '{"name": "SubmitCompletion", "arguments": {...}}'},
        {"from": "gpt", "value": "<final response>"}
      ]
    }

Usage:
    python -m moatless.fim.rl_to_sft <trace_root> --output <path> [options]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Static OpenAI-compatible tool schemas for the 4 FIM pipeline actions.
# These mirror what ActionArguments.tool_schema() generates for each class.
_FIM_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "ReadFile",
            "description": "Read specific lines from a file. This action allows you to read the contents of a file, either in its entirety or a specific range of lines. It's useful for examining code, configuration files, or any text file in the repository. The action will return at most 200 lines of content at a time. If more lines are requested, the content will be truncated and a note will be added indicating the truncation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "The path to the file you want to read, relative to the repository root. For example: 'src/main.py'",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "The first line number to include in the output (1-based indexing). If not specified, reading starts from the beginning of the file.",
                        "default": None,
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "The last line number to include in the output (inclusive). If not specified, reading continues until the end of the file or until reaching the 100-line limit.",
                        "default": None,
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "GlobTool",
            "description": "Fast file pattern matching tool that works with any codebase size. Supports glob patterns like '**/*.js' or 'src/**/*.ts'. Use this tool when you need to find files by name patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern to match files (e.g. '**/*.js', 'src/**/*.ts').",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to return",
                        "default": 10,
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ViewCode",
            "description": "View the code in a file or a specific code span.",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file_path": {
                                    "type": "string",
                                    "description": "The file path where the relevant code is found.",
                                },
                                "start_line": {
                                    "type": "integer",
                                    "description": "The start line of the code to add to context.",
                                    "default": None,
                                },
                                "end_line": {
                                    "type": "integer",
                                    "description": "The end line of the code to add to context.",
                                    "default": None,
                                },
                                "span_ids": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Span IDs identifying the relevant code spans.",
                                    "default": [],
                                },
                            },
                            "required": ["file_path"],
                        },
                        "description": "The code that should be provided in the file context.",
                    },
                },
                "required": ["files"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SubmitCompletion",
            "description": "Submit the final code completion.",
            "parameters": {
                "type": "object",
                "properties": {
                    "completion": {
                        "type": "string",
                        "description": "Code to fill into the requested line range. Return code only, no markdown fences.",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score for the completion.",
                        "default": 1.0,
                    },
                    "reasoning": {
                        "type": "string",
                        "description": "Reasoning for the completion.",
                        "default": "",
                    },
                },
                "required": ["completion"],
            },
        },
    },
]

# Mapping from action_args_class suffix to tool name
_ARGS_TO_TOOL_NAME = {
    "ReadFileArgs": "ReadFile",
    "GlobArgs": "GlobTool",
    "ViewCodeArgs": "ViewCode",
    "SubmitCompletionArgs": "SubmitCompletion",
}


def _build_tool_schemas() -> str:
    return json.dumps(_FIM_TOOL_SCHEMAS)


def _get_tool_name(action: dict) -> str:
    """Derive tool name from an action step's data."""
    args_class = action.get("action_args_class", "")
    suffix = args_class.rsplit(".", 1)[-1]
    return _ARGS_TO_TOOL_NAME.get(suffix, suffix.replace("Args", ""))


def _strip_metadata(action: dict) -> dict:
    """Return action dict without internal metadata keys."""
    return {k: v for k, v in action.items() if k not in ("action_args_class",)}


def _build_system_prompt(max_iterations: int = 8) -> str:
    """Build a minimal FIM system prompt (mirrors build_system_prompt logic)."""
    return (
        f"You are an AI coding assistant. You have access to tools to explore "
        f"a code repository. Your task is to complete missing code ranges. "
        f"You have a maximum of {max_iterations} iterations to explore, read "
        f"files, and understand the context before submitting your completion. "
        f"Read the relevant file sections to understand the code structure, "
        f"then call SubmitCompletion with the completed code."
    )


def trace_to_sharegpt(
    trace_dir: str,
    system_prompt: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    """
    Convert a single trace directory to ShareGPT format.

    Returns a dict with keys ``system``, ``tools``, ``conversations``,
    or ``None`` if the trace cannot be parsed.

    The ``conversations`` array follows ShareGPT tool-call conventions::

        human -> function_call -> observation -> ... -> function_call -> gpt

    - Intermediate assistant reasoning (``assistant_message``) from non-terminal
      nodes is discarded; only the structured tool call is kept.
    - The final ``gpt`` message contains the terminal node's
      ``assistant_message`` (the model's reasoning before submission).
    - SubmitCompletion observation is omitted so the conversation ends on
      ``gpt``.
    """
    trace_path = Path(trace_dir)

    # -- result.json -------------------------------------------------------
    result_path = trace_path / "result.json"
    if not result_path.exists():
        logger.warning("No result.json in %s, skipping", trace_dir)
        return None

    with open(result_path, encoding="utf-8") as f:
        result_data = json.load(f)

    # -- trajectory.json ---------------------------------------------------
    trajectory_path = trace_path / "trajectory.json"
    if not trajectory_path.exists():
        logger.warning("No trajectory.json in %s, skipping", trace_dir)
        return None

    with open(trajectory_path, encoding="utf-8") as f:
        trajectory_data = json.load(f)

    nodes = trajectory_data.get("nodes", [])
    if not nodes:
        logger.warning("Empty nodes list in %s, skipping", trace_dir)
        return None

    metadata = result_data.get("metadata", {})

    # ---------- system prompt ----------
    if system_prompt is None:
        max_iterations = int(metadata.get("task_max_iterations", 8))
        system_prompt = _build_system_prompt(max_iterations)

    # ---------- task prompt from root node ----------
    root = nodes[0]
    task_prompt = root.get("user_message")
    if not task_prompt:
        task_prompt = _fallback_task_prompt(result_data, metadata)

    tools = _build_tool_schemas()
    conversations: list[dict[str, str]] = [
        {"from": "human", "value": task_prompt},
    ]

    submitted = False

    # ---------- walk non-root nodes ----------
    for node in nodes[1:]:
        action_steps = node.get("action_steps", [])
        if not action_steps:
            continue

        for step in action_steps:
            action = step.get("action", {})
            action_args_class = action.get("action_args_class", "")
            is_submit = "SubmitCompletionArgs" in action_args_class

            tool_name = _get_tool_name(action)
            args = _strip_metadata(action)

            func_call_value = json.dumps(
                {"name": tool_name, "arguments": args}, ensure_ascii=False
            )

            if is_submit:
                conversations.append(
                    {"from": "function_call", "value": func_call_value}
                )
                submitted = True
                continue

            # Regular tool call
            conversations.append(
                {"from": "function_call", "value": func_call_value}
            )

            # Observation
            obs = step.get("observation") or {}
            if isinstance(obs, dict):
                obs_msg = obs.get("message") or obs.get("output")
            else:
                obs_msg = str(obs) if obs else None

            if obs_msg:
                conversations.append(
                    {"from": "observation", "value": str(obs_msg).strip()}
                )

    # ---------- final gpt message ----------
    if not submitted:
        return None

    if conversations[-1]["from"] == "function_call":
        terminal_node = nodes[-1]
        assistant_msg = terminal_node.get("assistant_message", "")
        gpt_content = assistant_msg.strip() or "Done."
        conversations.append({"from": "gpt", "value": gpt_content})
    elif conversations[-1]["from"] != "gpt":
        return None

    return {
        "system": system_prompt,
        "tools": tools,
        "conversations": conversations,
    }


def _fallback_task_prompt(result_data: dict, metadata: dict) -> str:
    """Minimal prompt when the root node has no ``user_message``."""
    file_path = result_data.get("file_path", "")
    start_line = int(result_data.get("start_line", 1))
    end_line = int(result_data.get("end_line", 1))
    repo_path = metadata.get("resolved_repo_path", metadata.get("repo_path", ""))

    return (
        f"Repository: {repo_path}\n"
        f"Target file: {file_path}\n"
        f"Missing range: lines {start_line}-{end_line}\n\n"
        "Use tools if needed, then call SubmitCompletion."
    )


def batch_convert(
    trace_root: str,
    output_path: str,
    *,
    min_reward: float = -float("inf"),
    max_samples: Optional[int] = None,
    task_ids: Optional[set[str]] = None,
    multi_hunk_only: bool = False,
    system_prompt: Optional[str] = None,
) -> int:
    """
    Scan *trace_root*, filter qualifying traces, and write ShareGPT-formatted
    JSONL to *output_path*.

    Returns the number of samples written.
    """
    root = Path(trace_root)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    samples: list[dict] = []
    counters: dict[str, int] = {"parse": 0, "reward": 0, "multi": 0, "task": 0}

    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        if max_samples is not None and len(samples) >= max_samples:
            break

        rp = entry / "result.json"
        if rp.exists():
            with open(rp, encoding="utf-8") as f:
                rd = json.load(f)

            if rd.get("reward", 0.0) < min_reward:
                counters["reward"] += 1
                continue

            if multi_hunk_only and not (rd.get("metadata") or {}).get(
                "multi_hunk_mode", False
            ):
                counters["multi"] += 1
                continue

            tid = rd.get("task_id")
            if task_ids is not None and tid not in task_ids:
                counters["task"] += 1
                continue

        sample = trace_to_sharegpt(str(entry), system_prompt=system_prompt)
        if sample is None:
            counters["parse"] += 1
            continue

        samples.append(sample)

    with open(output_path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    logger.info(
        "Exported %d SFT samples to %s "
        "(skipped: parse=%d reward=%d multi=%d task=%d)",
        len(samples),
        output_path,
        counters["parse"],
        counters["reward"],
        counters["multi"],
        counters["task"],
    )
    return len(samples)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convert FIM trajectory traces to ShareGPT SFT format (JSONL)."
    )
    p.add_argument(
        "trace_root",
        help="Root directory containing trace subdirectories.",
    )
    p.add_argument(
        "--output",
        "-o",
        default="moatless/results/sft_training_data.jsonl",
        help="Output JSONL path.",
    )
    p.add_argument(
        "--min-reward",
        type=float,
        default=-float("inf"),
        help="Minimum reward threshold.",
    )
    p.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to export.",
    )
    p.add_argument(
        "--multi-hunk-only",
        action="store_true",
        default=False,
        help="Only export multi-hunk traces.",
    )
    p.add_argument(
        "--task-ids",
        nargs="+",
        default=None,
        help="Space-separated list of task_ids to include.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> None:
    args = _build_parser().parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    trace_root = Path(args.trace_root)
    if not trace_root.exists():
        raise FileNotFoundError(f"trace_root not found: {trace_root}")

    count = batch_convert(
        trace_root=str(trace_root),
        output_path=args.output,
        min_reward=args.min_reward,
        max_samples=args.max_samples,
        task_ids=set(args.task_ids) if args.task_ids else None,
        multi_hunk_only=args.multi_hunk_only,
    )

    print(f"Exported {count} SFT samples to {args.output}")


if __name__ == "__main__":
    main()
