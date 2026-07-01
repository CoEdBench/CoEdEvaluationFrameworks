"""
fim/schema.py
Data structures: FillRequest, FillResult, SubmitCompletion Action
"""
from dataclasses import dataclass, field
from typing import Any, Optional

from pydantic import ConfigDict, Field

from moatless.actions.action import Action
from moatless.actions.schema import ActionArguments

RESULT_SCHEMA_VERSION = "fim_result.v1"


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    if metadata is None:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)
    # Keep unexpected values observable without breaking output schema.
    return {"_raw_metadata": metadata}


# ── Input ──────────────────────────────────────────────────────────────────────

@dataclass
class FillRequest:
    task_id: str
    model_name: str
    repo_path: str
    file_path: str
    start_line: int
    end_line: int
    language: str = "python"
    context_lines: int = 30
    max_iterations: int = 30

    ground_truth: Optional[str] = None
    metadata: dict = field(default_factory=dict)


# ── Output ─────────────────────────────────────────────────────────────────────

@dataclass
class FillResult:
    task_id: str
    model_name: str
    file_path: str
    trace_dir: str
    start_line: int
    end_line: int
    completion: str
    updated_file_content: str
    success: bool
    git_patch: str = ""
    changed_files: list[str] = field(default_factory=list)
    updated_files_content: dict[str, str] = field(default_factory=dict)
    updated_files_content_before: dict[str, str] = field(default_factory=dict)
    finish_reason: Optional[str] = None
    error: Optional[str] = None
    confidence: float = 0.0
    reasoning: str = ""
    action_steps: int = 0
    # Evaluation fields (populated by evaluator)
    eval_exact_match: Optional[bool] = None
    eval_edit_similarity: Optional[float] = None
    eval_bleu: Optional[float] = None
    eval_execution_pass: Optional[bool] = None   # Docker execution result
    # Reserved fields
    eval_codebleu: Optional[float] = None
    eval_partial_match: Optional[float] = None
    eval_syntax_pass: Optional[bool] = None
    task_runtime_seconds: Optional[float] = None
    tool_call_distribution: Optional[dict[str, int]] = None
    trajectory_metrics: Optional[dict[str, Any]] = None
    metadata: dict = field(default_factory=dict)

    def to_output_dict(self) -> dict[str, Any]:
        """
        Stable JSON schema for result output files.
        """
        output = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "task_id": self.task_id,
            "model_name": self.model_name,
            "file_path": self.file_path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "completion": self.completion,
            "updated_file_content": self.updated_file_content,
            "git_patch": self.git_patch,
            "changed_files": self.changed_files,
            "updated_files_content": self.updated_files_content,
            "updated_files_content_before": self.updated_files_content_before,
            "success": self.success,
            "finish_reason": self.finish_reason,
            "error": self.error,
            "trace_dir": self.trace_dir,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "action_steps": self.action_steps,
            "eval_exact_match": self.eval_exact_match,
            "eval_edit_similarity": self.eval_edit_similarity,
            "eval_bleu": self.eval_bleu,
            "eval_execution_pass": self.eval_execution_pass,
            "eval_codebleu": self.eval_codebleu,
            "eval_partial_match": self.eval_partial_match,
            "eval_syntax_pass": self.eval_syntax_pass,
            "task_runtime_seconds": self.task_runtime_seconds,
            "tool_call_distribution": self.tool_call_distribution,
            "trajectory_metrics": self.trajectory_metrics,
            "metadata": _normalize_metadata(self.metadata),
        }

        # Keep compatibility with any extra attributes attached dynamically.
        extra_fields = {
            k: v for k, v in self.__dict__.items()
            if k not in output and k != "metadata"
        }
        if extra_fields:
            output["extra_fields"] = extra_fields

        return output


# ── Evaluation Data Structures ─────────────────────────────────────────────────


@dataclass
class HunkEvalResult:
    """Per-hunk evaluation result for multi-hunk completion."""

    index: int
    file_path: str
    start_line: int
    end_line: int
    predicted: str
    ground_truth: str
    exact_match: bool
    edit_similarity: float
    is_missing: bool = False
    is_extra: bool = False
    error_category: str = "correct"  # "correct" | "wrong_file" | "right_file_wrong_lines" | "missing" | "root_hunk_leak" | "partial_overlap"


# ── SubmitCompletion Action ───────────────────────────────────────────────────

class SubmitCompletionArgs(ActionArguments):
    completion: str = Field(
        description="Code to fill into the requested line range. Return code only, no markdown fences."
    )
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str = Field(default="")

    model_config = ConfigDict(title="SubmitCompletion")


class SubmitCompletion(Action):
    args_schema = SubmitCompletionArgs
    is_terminal: bool = True

    async def _execute(self, args: SubmitCompletionArgs, file_context=None) -> str:
        return (
            f"Submission accepted.\n"
            f"confidence={args.confidence:.3f}\n"
            f"reasoning={args.reasoning}\n"
            f"completion:\n{args.completion}"
        )
