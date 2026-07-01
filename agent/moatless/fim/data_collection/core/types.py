from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple


class FileContent(BaseModel):
    file_path: str
    content_before: Optional[str]   # Full content before modification (with line numbers)
    content_after:  Optional[str]   # Full content after modification (with line numbers)

class Symbol(BaseModel):
    name: str
    kind: str  # 'function', 'class', 'variable', 'import', 'attribute'

    def __hash__(self):
        return hash((self.name, self.kind))

class Relation(BaseModel):
    source_hunk_id: str
    target_hunk_id: str
    reason: str  # e.g., "Defines variable 'x'" or "Text similarity 0.8"


class FileChange(BaseModel):
    old_path: Optional[str]
    new_path: Optional[str]
    change_type: str  # ADD, MODIFY, DELETE, RENAME
    diff: str
    source_code: Optional[str]
    is_test: bool = False  # Flag indicating whether it is a test file


class CommitCandidate(BaseModel):
    """
    The core object that flows through the pipeline.
    Phase 1 populates basic information.
    Phase 2 will populate dependency_graph.
    Phase 3 will populate scenarios.
    """
    repo_name:str
    hash: str
    msg: str
    author_date: str
    repo_url: str
    is_merge: bool

    # Statistics
    # files_count: int
    # lines_added: int
    # lines_removed: int
    issue_ids: List[str] = Field(default_factory=list)
    # Separate storage for convenient downstream processing
    source_changes: List[FileChange] = []  # Core data: participates in collaborative modification analysis
    test_changes: List[FileChange] = []  # Validation data: only used for testing

    # Statistics
    source_files_count: int
    test_files_count: int

    metadata: Dict[str, Any] = Field(default_factory=dict)

class Hunk(BaseModel):
    id: str  # Unique identifier: "filepath:start_line"
    file_path: str
    content: str  # The actual code text of the Hunk

    # === New/Explicit line number fields ===
    old_start_line: int  # Old version start line
    old_len: int  # Old version length
    new_start_line: int  # New version start line
    new_len: int  # New version length

    # Compatibility fields (usually point to new version, used for sorting)
    start_line: int
    end_line: int


    # Sort result
    order_index: int = -1

class LLMAnalysisResult(BaseModel):
    root_hunk_id: str = Field(..., description="The Hunk ID determined to be the source")
    confidence: float = Field(..., description="Confidence 0.0-1.0")
    reasoning: str = Field(..., description="Reasoning for the determination")
    change_pattern: str = Field(..., description="Change pattern: Refactor/Feature/Fix")

class DependencyChain(BaseModel):
    """
    Describes the dependency path between two Hunks
    """
    source: str
    target: str
    path: List[str]
    raw_path: Optional[List[str]] = None


class AnalyzedCommit(BaseModel):
    hash: str
    repo: str
    msg: str

    # Added: associated Issue description (if available)
    issue_description: Optional[str] = None
    # Sorted source Hunk sequence (training target)
    ordered_hunks: List[Hunk]

    # Test Hunk (validation context)
    test_hunks: List[Hunk]

    # Dependency relationships (for visualization or debugging)
    # Format: [(hunk_id_from, hunk_id_to, reason), ...]
    # dependencies: List[Tuple[str, str, str]] = []
    dependencies: List[Dict[str, Any]]
    # --- New addition ---
    # Store edges in the graph: [(source_id, target_id, type), ...]
    dependency_edges: List[Tuple[str, str, str]] = Field(default_factory=list)

    # Store complete dependency chains
    dependency_chains: List[DependencyChain] = Field(default_factory=list)

    # Old version (Parent) dependency information
    old_dependencies: List[Dict[str, Any]]
    # old_dependencies: List[Tuple[str, str, str]] = Field(default_factory=list)
    old_dependency_chains: List[DependencyChain] = Field(default_factory=list)

    # Dependency change label: "BOTH", "NEW_ONLY", "OLD_ONLY", "NONE"
    dependency_label: str = "NONE"

    # Statistics / metrics
    old_metrics: Dict[str, Any] = Field(default_factory=dict)
    new_metrics: Dict[str, Any] = Field(default_factory=dict)

    causal_analysis: Optional[LLMAnalysisResult] = None