"""
fim/__init__.py
Unified exports: from moatless.fim import ...
"""
from moatless.fim.schema import FillRequest, FillResult, HunkEvalResult, SubmitCompletion, SubmitCompletionArgs
from moatless.fim.pipeline import run_fill_task, run_fill_batch
from moatless.fim.dataset import load_requests_from_jsonl, save_results_to_jsonl, DockerSandbox
from moatless.fim.evaluator import evaluate_batch, summarize, EvalSummary

__all__ = [
    "FillRequest",
    "FillResult",
    "SubmitCompletion",
    "SubmitCompletionArgs",
    "run_fill_task",
    "run_fill_batch",
    "load_requests_from_jsonl",
    "save_results_to_jsonl",
    "DockerSandbox",
    "evaluate_batch",
    "summarize",
    "EvalSummary",
]