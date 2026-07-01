import time
from typing import List, Optional, Any, Dict
from enum import Enum
from pydantic import BaseModel, Field
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════
# Timing & Stats
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class LLMCallStat:
    """Single LLM call statistics"""
    stage: str
    duration_s: float
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class TimingStats:
    """Complete timing statistics"""
    graph_build_s: float = 0.0
    llm_calls: List[LLMCallStat] = field(default_factory=list)
    total_s: float = 0.0

    @property
    def llm_call_count(self) -> int:
        return len(self.llm_calls)

    @property
    def llm_total_s(self) -> float:
        return sum(c.duration_s for c in self.llm_calls)

    @property
    def llm_avg_s(self) -> float:
        return self.llm_total_s / self.llm_call_count if self.llm_calls else 0.0

    @property
    def other_s(self) -> float:
        return self.total_s - self.graph_build_s - self.llm_total_s

    def summary(self) -> dict:
        return {
            "total_s": round(self.total_s, 3),
            "graph_build_s": round(self.graph_build_s, 3),
            "llm_call_count": self.llm_call_count,
            "llm_total_s": round(self.llm_total_s, 3),
            "llm_avg_s": round(self.llm_avg_s, 3),
            "other_s": round(self.other_s, 3),
            "llm_calls_detail": [
                {
                    "stage": c.stage,
                    "duration_s": round(c.duration_s, 3),
                    "input_tokens": c.input_tokens,
                    "output_tokens": c.output_tokens,
                }
                for c in self.llm_calls
            ],
        }


# ══════════════════════════════════════════════════════════════════════════
# Enums (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class ContextRole(str, Enum):
    INTERNAL_CALLER   = "INTERNAL_CALLER"
    INTERNAL_USAGE    = "INTERNAL_USAGE"
    INTERNAL_METHOD   = "INTERNAL_METHOD"
    EXTERNAL_CALLER   = "EXTERNAL_CALLER"
    EXTERNAL_IMPORTER = "EXTERNAL_IMPORTER"
    EXTERNAL_SUBCLASS = "EXTERNAL_SUBCLASS"
    EXTERNAL_USER     = "EXTERNAL_USER"
    CALLEE            = "CALLEE"
    UNKNOWN           = "UNKNOWN"
    ORACLE_GT         = "ORACLE_GT"


# ══════════════════════════════════════════════════════════════════════════
# Dataset Core Structures
# ══════════════════════════════════════════════════════════════════════════

class Hunk(BaseModel):
    """
    Smallest code modification unit, corresponding to a hunk in git diff.
    Used by both ordered_hunks and test_hunks.
    """
    id: str                        # Unique identifier, format: "file_path:start_line"
    file_path: str                 # Relative path (original, with platform separators)
    content: str                   # Diff content fragment
    old_start_line: int            # Old start line
    old_len: int                   # Old line count (0 = pure addition)
    new_start_line: int            # New start line
    new_len: int                   # New line count
    start_line: int                # Change range start line (for context reading)
    end_line: int                  # Change range end line
    order_index: int               # Causal order index; -1 indicates test hunk


class CausalAnalysis(BaseModel):
    """
    Causal analysis result, generated during dataset preprocessing.
    Describes causal relationships among hunks in this commit.
    """
    root_hunk_id: str              # Root hunk id
    confidence: float              # Causal analysis confidence
    reasoning: str                 # Causal reasoning description (JSON string)
    change_pattern: str            # Change pattern, e.g. "Error Handling"
    is_single_requirement: bool    # Whether it belongs to a single requirement
    requirement_summary: str       # Requirement summary (replaces old commit_msg)
    hunk_order: List[int]          # Causal execution order of ordered_hunks (index list)


class DataItem(BaseModel):
    """
    A complete record in the dataset, corresponding to a commit.
    """
    hash: str                                  # commit hash
    repo: str                                  # Repo name
    msg: str                                   # commit message
    source_diff: str                           # Full git diff text
    issue_description: Optional[str] = None   # Issue description (can be empty)
    ordered_hunks: List[Hunk]                  # All non-test hunks (including root)
    test_hunks: List[Hunk]                     # Test file hunks (order_index=-1)
    causal_analysis:  Optional[CausalAnalysis] = None            # Causal analysis result
    dependency_label: str = ""                 # Dependency graph label, e.g. "NO_GRAPH"


class ParsedItem(BaseModel):
    """
    Runtime data structure after parsing and path correction.
    Generated by DataItemParser, passed directly to Pipeline.
    """
    data_item: DataItem            # Original data (kept for reference)
    repo_root: str                 # Sandbox repo root path (absolute)

    root_hunk: Hunk                # Root hunk (path corrected to relative)
    target_hunks: List[Hunk]       # Non-root hunks to predict (ordered by causality)

    # Before/after code reconstructed from sandbox files (for Prompt use)
    root_before_code: str          # Root hunk code before modification
    root_after_code: str           # Root hunk code after modification (read after apply)
    root_context_snippet: str = ""
    requirement:str
# ══════════════════════════════════════════════════════════════════════════
# LLM Interaction Structures (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class TokenUsage(BaseModel):
    """Record token consumption of LLM calls"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cached_tokens: int = 0   # Prompt cached token count
    model_name: str = "unknown"

    def add(self, other: "TokenUsage"):
        self.prompt_tokens       += other.prompt_tokens
        self.completion_tokens    += other.completion_tokens
        self.total_tokens        += other.total_tokens
        self.prompt_cached_tokens += other.prompt_cached_tokens


class LLMResponse(BaseModel):
    content: str
    usage: TokenUsage


# ══════════════════════════════════════════════════════════════════════════
# Pipeline Output Structures (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class PredictedEdit(BaseModel):
    file_path: str
    predicted_line_nums: List[int]
    start_line: int
    end_line: int
    next_version: Optional[str]
    current_version: Optional[str] = None
    change_summary: Optional[str] = None
    predicted_order: Optional[int] = None


# ══════════════════════════════════════════════════════════════════════════
# Evaluation Structures (unchanged)
# ══════════════════════════════════════════════════════════════════════════

class GTCoverageDetail(BaseModel):
    gt_rel_file_path: str
    gt_start_line: int
    gt_end_line: int
    file_matched: bool
    code_matched: bool
    is_covered: bool
    matched_context_role: str = ""
    matched_node_id: str = ""
    miss_reason: str = ""


class ContextCoverageResult(BaseModel):
    total_gt: int
    covered_count: int
    recall: float
    is_framework_issue: bool
    details: List[GTCoverageDetail] = []

    def summary(self) -> dict:
        return {
            "total_gt": self.total_gt,
            "covered_count": self.covered_count,
            "recall": round(self.recall, 4),
            "is_framework_issue": self.is_framework_issue,
            "uncovered": [
                {
                    "file": d.gt_rel_file_path,
                    "lines": f"{d.gt_start_line}-{d.gt_end_line}",
                    "miss_reason": d.miss_reason,
                }
                for d in self.details if not d.is_covered
            ],
        }


# ══════════════════════════════════════════════════════════════════════════
# Run Record (upgraded)
# ══════════════════════════════════════════════════════════════════════════

class RunRecord(BaseModel):
    """Complete record of a single run, for storage and subsequent evaluation"""
    run_id: str
    timestamp: float = Field(default_factory=time.time)
    timing: dict = {}


    commit_hash: str
    repo_name: str

    # Core input: root hunk (may be None if checkout failed)
    root_hunk: Optional[Hunk] = None

    # Output
    predictions: List[PredictedEdit]
    final_diff: Optional[str] = None
    token_usage: TokenUsage

    # Annotations (for evaluation)
    target_hunks: List[Hunk] = []

    # Metadata
    llm_model_config: Dict[str, Any] = {}
    error: Optional[str] = None
    context_coverage: Optional[ContextCoverageResult] = None

    # Stage 1 raw output (LLM-identified impacted locations)
    stage1_result: Optional[Dict] = None

    # File snapshots (taken at Pipeline runtime, for evaluator use)
    # {rel_path: file_content_at_commit^_after_root_hunk_applied}
    before_files: Dict[str, str] = {}
    # {rel_path: file_content_after_apply_target_hunks} — GT final state
    gt_after_files: Dict[str, str] = {}
