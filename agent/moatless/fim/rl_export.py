#!/usr/bin/env python3
"""
fim/rl_export.py
Export FIM trajectory traces as RL training samples (JSONL).

Usage:
    python -m moatless.fim.rl_export <trace_root> --output <path> [options]
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Optional

from moatless.fim.dataset import load_requests_from_jsonl
from moatless.fim.prompt import build_system_prompt, build_user_prompt
from moatless.fim.rl_types import RLStep, RLTrainingSample

logger = logging.getLogger(__name__)


def convert_trace_to_rl_sample(
    trace_dir: str,
    task_request: Any = None,
) -> Optional[RLTrainingSample]:
    """
    Load a single trace directory and convert it to an RLTrainingSample.

    Args:
        trace_dir: Path to the trace directory (containing result.json, trajectory.json).
        task_request: Optional FillRequest to reconstruct the exact prompt context.
                      If None, context is reconstructed from metadata in result.json.

    Returns:
        RLTrainingSample or None if the trace cannot be parsed.
    """
    trace_path = Path(trace_dir)

    # 1. Load result.json
    result_path = trace_path / "result.json"
    if not result_path.exists():
        logger.warning("No result.json in %s, skipping", trace_dir)
        return None

    with open(result_path, encoding="utf-8") as f:
        result_data = json.load(f)

    # 2. Load trajectory.json
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

    # 3. Extract metadata
    reward = result_data.get("reward", 0.0)
    per_hunk_eval_raw = result_data.get("per_hunk_eval", [])
    metadata = result_data.get("metadata", {})
    is_multi_hunk = bool(metadata.get("multi_hunk_mode", False))

    # 4. Build RLStep list from nodes
    steps: list[RLStep] = []
    submit_completion: Optional[str] = None

    for node in nodes:
        node_id = node.get("node_id", 0)
        node_assistant_message = node.get("assistant_message")
        for step_data in node.get("action_steps", []):
            action = step_data.get("action", {})
            action_name = action.get("action_args_class", "").split(".")[-1]

            if action_name == "SubmitCompletionArgs":
                submit_completion = action.get("completion", "")

            # Extract logprobs from step completion
            completion = step_data.get("completion", {})
            logprob_sum: Optional[float] = None
            token_logprobs: Optional[list[dict]] = None
            prompt_tokens = 0
            completion_tokens = 0

            if completion:
                for attempt in completion.get("attempts", []):
                    if attempt.get("logprob_sum") is not None:
                        logprob_sum = attempt["logprob_sum"]
                    if attempt.get("logprobs"):
                        token_logprobs = attempt["logprobs"]
                    usage = attempt.get("usage", {})
                    prompt_tokens += usage.get("prompt_tokens", 0)
                    completion_tokens += usage.get("completion_tokens", 0)

            observation = step_data.get("observation", {})
            if isinstance(observation, dict):
                observation = observation.get("message") or observation.get("output")

            steps.append(RLStep(
                step_index=node_id,
                action_name=action_name,
                action_args=action,
                logprob_sum=logprob_sum,
                token_logprobs=token_logprobs,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                observation=str(observation) if observation else None,
                assistant_message=node_assistant_message,
                timestamp=node.get("timestamp"),
            ))

    # 5. Reconstruct prompt context
    context = _reconstruct_context(result_data, metadata, task_request)

    # 6. Normalize per_hunk_eval to dict list
    per_hunk_eval_dicts: list[dict[str, Any]] = []
    for hunk in per_hunk_eval_raw:
        if isinstance(hunk, dict):
            per_hunk_eval_dicts.append(hunk)
        elif hasattr(hunk, "__dict__"):
            per_hunk_eval_dicts.append(hunk.__dict__)
        else:
            per_hunk_eval_dicts.append({"index": 0, "_error": f"unparseable: {type(hunk)}"})

    return RLTrainingSample(
        task_id=result_data.get("task_id", "unknown"),
        model_name=result_data.get("model_name", "unknown"),
        file_path=result_data.get("file_path", ""),
        context=context,
        completion=submit_completion or "",
        reward=reward,
        steps=steps,
        is_multi_hunk=is_multi_hunk,
        per_hunk_eval=per_hunk_eval_dicts,
        metadata={"trace_dir": trace_dir, **metadata},
    )


def _reconstruct_context(
    result_data: dict[str, Any],
    metadata: dict[str, Any],
    task_request: Any = None,
) -> str:
    """
    Reconstruct the full prompt context for RL training.
    """
    if task_request is not None:
        system = build_system_prompt(task_request.max_iterations)
        user = build_user_prompt(task_request)
        return f"{system}\n\n{user}"

    # Fallback: reconstruct from metadata
    max_iterations = int(metadata.get("task_max_iterations", 8))
    system = build_system_prompt(max_iterations)

    repo_path = metadata.get("resolved_repo_path", metadata.get("repo_path", ""))
    file_path = result_data.get("file_path", "")
    start_line = int(result_data.get("start_line", 1))
    end_line = int(result_data.get("end_line", 1))
    context_lines = int(metadata.get("context_lines", 30))
    language = metadata.get("language", "python")

    # Try to read file context from disk
    try:
        from moatless.fim.prompt import build_context_snippet
        context = build_context_snippet(
            repo_path=repo_path,
            file_path=file_path,
            start_line=start_line,
            end_line=end_line,
            context_lines=context_lines,
            use_fill_marker=True,
        )
    except Exception:
        context = "(file context unavailable)"

    user = (
        f"Repository: {repo_path}\n"
        f"Target file: {file_path}\n"
        f"Missing range: lines {start_line}-{end_line}\n\n"
        f"Context:\n```{language}\n{context}\n```\n\n"
        "Use tools if needed, then call SubmitCompletion."
    )
    return f"{system}\n\n{user}"


def load_task_requests(task_jsonl: str) -> dict[str, Any]:
    """Load FillRequest objects from a tasks JSONL, return task_id -> FillRequest dict."""
    requests = load_requests_from_jsonl(task_jsonl)
    return {r.task_id: r for r in requests}


def export_rl_traces(
    trace_root: str,
    output_path: str,
    min_reward: float = -float("inf"),
    max_samples: Optional[int] = None,
    task_ids: Optional[set[str]] = None,
    multi_hunk_only: bool = False,
    request_lookup: Optional[dict[str, Any]] = None,
) -> int:
    """
    Scan trace_root, convert qualifying traces to RLTrainingSamples, write JSONL.

    Returns number of samples written.
    """
    trace_root_path = Path(trace_root)
    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)

    samples: list[RLTrainingSample] = []
    skipped = {"parse": 0, "reward": 0, "multi": 0, "task": 0}

    for entry in sorted(trace_root_path.iterdir()):
        if not entry.is_dir():
            continue

        if max_samples is not None and len(samples) >= max_samples:
            break

        # Infer task_id from directory name (format: task_id_timestamp)
        dir_name = entry.name
        dir_task_id = dir_name.rsplit("_", 1)[0]

        # Try to find matching request
        request = None
        if request_lookup:
            request = request_lookup.get(dir_task_id) or request_lookup.get(dir_name)
            if request is None:
                for known_id, known_req in request_lookup.items():
                    if dir_name.startswith(known_id):
                        request = known_req
                        break

        sample = convert_trace_to_rl_sample(str(entry), task_request=request)
        if sample is None:
            skipped["parse"] += 1
            continue

        if multi_hunk_only and not sample.is_multi_hunk:
            skipped["multi"] += 1
            continue

        if sample.reward < min_reward:
            skipped["reward"] += 1
            continue

        if task_ids is not None and sample.task_id not in task_ids:
            skipped["task"] += 1
            continue

        samples.append(sample)

    # Write JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample.to_jsonl_dict(), ensure_ascii=False, default=str) + "\n")

    logger.info(
        "Exported %d samples to %s (skipped: parse=%d reward=%d multi=%d task=%d)",
        len(samples), output_path,
        skipped["parse"], skipped["reward"], skipped["multi"], skipped["task"],
    )
    return len(samples)


# ── CLI ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export FIM trajectory traces as RL training samples (JSONL)."
    )
    parser.add_argument("trace_root", help="Root directory containing trace subdirectories.")
    parser.add_argument(
        "--output", "-o",
        default="moatless/results/rl_training_data.jsonl",
        help="Output JSONL path (default: moatless/results/rl_training_data.jsonl).",
    )
    parser.add_argument(
        "--min-reward", type=float, default=-float("inf"),
        help="Minimum reward threshold (traces below are skipped).",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Maximum number of samples to export.",
    )
    parser.add_argument(
        "--multi-hunk-only", action="store_true", default=False,
        help="Only export multi-hunk traces.",
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=None,
        help="Space-separated list of task_ids to include.",
    )
    parser.add_argument(
        "--task-jsonl", default=None,
        help="Path to the tasks JSONL for exact prompt reconstruction.",
    )
    parser.add_argument(
        "--summary", "-s", default=None,
        help="Path to write a JSON summary of the export.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level.",
    )
    return parser


def _write_summary(summary_path: str, trace_root: str, args: argparse.Namespace, count: int) -> None:
    summary = {
        "trace_root": trace_root,
        "output_path": args.output,
        "export_count": count,
        "min_reward": args.min_reward,
        "max_samples": args.max_samples,
        "multi_hunk_only": args.multi_hunk_only,
        "task_ids": args.task_ids,
        "task_jsonl": args.task_jsonl,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary written to %s", summary_path)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    trace_root = Path(args.trace_root)
    if not trace_root.exists():
        raise FileNotFoundError(f"trace_root not found: {trace_root}")

    output_path = str(Path(args.output))
    task_ids = set(args.task_ids) if args.task_ids else None

    # Load task requests for prompt reconstruction
    request_lookup = None
    if args.task_jsonl:
        task_jsonl_path = Path(args.task_jsonl)
        if task_jsonl_path.exists():
            request_lookup = load_task_requests(str(task_jsonl_path))
            logger.info("Loaded %d task requests from %s", len(request_lookup), task_jsonl_path)
        else:
            logger.warning("Task JSONL not found: %s", task_jsonl_path)

    count = export_rl_traces(
        trace_root=str(trace_root),
        output_path=output_path,
        min_reward=args.min_reward,
        max_samples=args.max_samples,
        task_ids=task_ids,
        multi_hunk_only=args.multi_hunk_only,
        request_lookup=request_lookup,
    )

    print(f"Exported {count} RL training samples to {output_path}")

    if args.summary:
        _write_summary(args.summary, str(trace_root), args, count)


if __name__ == "__main__":
    main()
