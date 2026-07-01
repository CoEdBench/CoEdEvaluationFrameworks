"""
fim/rl_types.py
RL training data structures for GRPO fine-tuning from multi-hunk FIM trajectories.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional


@dataclass
class RLStep:
    """One step in the agent trajectory (one node's action)."""
    step_index: int
    action_name: str
    action_args: dict[str, Any]
    logprob_sum: Optional[float] = None
    token_logprobs: Optional[list[dict]] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    observation: Optional[str] = None
    assistant_message: Optional[str] = None
    timestamp: Optional[str] = None


@dataclass
class RLTrainingSample:
    """
    One complete trajectory from a FIM task, shaped for RL training.

    Fields:
        task_id, model_name   — traceability
        context               — reconstructed system + user prompt
        completion            — final SubmitCompletion payload
        reward                — scalar reward (from reward.py)
        steps                 — per-node trajectory with logprobs
        is_multi_hunk         — whether this is a multi-hunk task
        per_hunk_eval         — per-hunk metrics (list of HunkEvalResult dicts)
        metadata              — extra info (trace_dir, etc.)
    """
    task_id: str
    model_name: str
    file_path: str
    context: str
    completion: str
    reward: float
    steps: list[RLStep] = field(default_factory=list)
    is_multi_hunk: bool = False
    per_hunk_eval: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_jsonl_dict(self) -> dict[str, Any]:
        """Serialize to a flat dict for JSONL output."""
        return {
            "task_id": self.task_id,
            "model_name": self.model_name,
            "file_path": self.file_path,
            "context": self.context,
            "completion": self.completion,
            "reward": self.reward,
            "is_multi_hunk": self.is_multi_hunk,
            "steps": [asdict(s) for s in self.steps],
            "per_hunk_eval": self.per_hunk_eval,
            "metadata": self.metadata,
        }
