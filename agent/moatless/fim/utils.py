"""
fim/utils.py
Generic utility functions: line replacement, trace persistence, submission extraction
"""
import json
import logging
from pathlib import Path
from typing import Optional

from moatless.fim.schema import FillResult, SubmitCompletionArgs, SubmitCompletion

logger = logging.getLogger(__name__)


def replace_line_range(original: str, start_line: int, end_line: int, replacement: str) -> str:
    """Replace the range [start_line, end_line] in original with replacement."""
    lines = original.splitlines(keepends=True)
    replacement_lines = replacement.splitlines(keepends=True)
    if replacement and not replacement.endswith(("\n", "\r")):
        replacement_lines.append("\n")
    new_lines = lines[: start_line - 1] + replacement_lines + lines[end_line:]
    return "".join(new_lines)


def extract_submission(flow) -> Optional[SubmitCompletionArgs]:
    """Find the first SubmitCompletionArgs from all flow nodes in reverse order."""
    for node in reversed(flow.root.get_all_nodes()):
        for step in reversed(node.action_steps or []):
            if isinstance(step.action, SubmitCompletionArgs):
                return step.action
            if isinstance(step.action, SubmitCompletion):
                for attr in ("args", "action_args", "arguments", "input"):
                    val = getattr(step.action, attr, None)
                    if isinstance(val, SubmitCompletionArgs):
                        return val
            for attr in ("action_arguments", "args", "arguments"):
                val = getattr(step, attr, None)
                if isinstance(val, SubmitCompletionArgs):
                    return val
    return None


def persist_trace(flow, output_dir: Path, result: FillResult) -> None:
    """Persist flow settings, trajectory, and results to output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "flow.json", "w", encoding="utf-8") as f:
        json.dump(flow.get_flow_settings(), f, ensure_ascii=False, indent=2, default=str)
    with open(output_dir / "trajectory.json", "w", encoding="utf-8") as f:
        json.dump(flow.get_trajectory_data(), f, ensure_ascii=False, indent=2, default=str)
    with open(output_dir / "result.json", "w", encoding="utf-8") as f:
        payload = result.to_output_dict() if hasattr(result, "to_output_dict") else result.__dict__
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)
    logger.debug("Trace saved to %s", output_dir)
