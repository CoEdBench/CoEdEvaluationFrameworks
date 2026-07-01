import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src.core.parser import DataItemParser
from src.core.pipeline import MultiPointPipeline
from src.domain.interfaces import ILLMClient
from src.domain.types import (
    Hunk, RunRecord, TokenUsage, PredictedEdit,
)
from src.utils.sandbox_utils import prepare_sandbox_repo, cleanup_sandbox_repo

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Runner Configuration
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RunnerConfig:
    data_path: str                          # Input JSONL path
    repos_base: str                         # Repository root directory (with repo subdirectories)
    output_path: str                        # Output JSONL path
    cache_dir: str = ".cache"               # Dependency graph cache directory
    log_dir: str = "llm_logs"              # LLM interaction log directory
    context_mode: str = "auto"             # "auto" | "oracle"
    dry_run: bool = False                  # Skip LLM when True
    limit: Optional[int] = None            # Only process first N items (None = all)
    resume: bool = True                    # Skip already-processed hashes
    repo_filter: Optional[str] = None      # Only process specific repo name
    # FIX: Optional[List] defaults to None directly, no need for field(default=None)
    hash_filter: Optional[List[str]] = None  # Only process specific hash list
    git_timeout: int = 300                  # Git command timeout (seconds)


# ══════════════════════════════════════════════════════════════════════════
# Run Statistics
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class RunStats:
    total: int = 0
    succeeded: int = 0          # Pipeline succeeded (including no_predictions)
    no_predictions: int = 0     # Pipeline succeeded but no predictions
    skipped_resume: int = 0
    skipped_filter: int = 0
    failed_checkout: int = 0
    failed_parse: int = 0
    failed_pipeline: int = 0

    @property
    def processed(self) -> int:
        """Actually processed count (excluding skip)"""
        return (
            self.succeeded
            + self.failed_checkout
            + self.failed_parse
            + self.failed_pipeline
        )

    @property
    def total_failed(self) -> int:
        return self.failed_checkout + self.failed_parse + self.failed_pipeline

    def summary(self) -> dict:
        return {
            "total_seen":       self.total,
            "processed":        self.processed,
            "succeeded":        self.succeeded,
            "no_predictions":   self.no_predictions,
            "skipped_resume":   self.skipped_resume,
            "skipped_filter":   self.skipped_filter,
            "failed_checkout":  self.failed_checkout,
            "failed_parse":     self.failed_parse,
            "failed_pipeline":  self.failed_pipeline,
        }


# ══════════════════════════════════════════════════════════════════════════
# Git Utilities
# ══════════════════════════════════════════════════════════════════════════

def _git(args: List[str], cwd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    """Run git command, return CompletedProcess. Raise RuntimeError on failure."""
    cmd = ["git"] + args
    result = subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed (rc={result.returncode}): "
            f"{result.stderr.strip()}"
        )
    return result


def _checkout_parent(repo_path: str, commit_hash: str, timeout: int = 30) -> None:
    """
    Checkout repo to commit^ (parent of target commit),
    i.e., the state before the code change occurred.

    Use `git checkout --force` to discard working directory changes,
    preventing residual apply from the previous item affecting this one.
    """
    parent_ref = f"{commit_hash}^"
    _git(["checkout", "--force", parent_ref], cwd=repo_path, timeout=timeout)
    logger.info(f"✅ Checked out {parent_ref} in {repo_path}")


def _get_current_head(repo_path: str, timeout: int = 30) -> str:
    """Get current HEAD commit hash (for logging)"""
    try:
        result = _git(["rev-parse", "HEAD"], cwd=repo_path, timeout=timeout)
        return result.stdout.strip()
    except Exception:
        return "unknown"


# ══════════════════════════════════════════════════════════════════════════
# Serialization Utilities
# ══════════════════════════════════════════════════════════════════════════

def _serialize_record(record: RunRecord) -> str:
    """Serialize RunRecord to single-line JSON string."""
    data = record.model_dump()
    return json.dumps(data, ensure_ascii=False)


def _load_done_hashes(output_path: str) -> Set[str]:
    """
    Read all already-processed commit_hashes from existing output file for resume mode.
    Returns empty set if file does not exist.
    """
    done: Set[str] = set()
    if not os.path.exists(output_path):
        return done
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    h = obj.get("commit_hash", "")
                    if h:
                        done.add(h)
                except json.JSONDecodeError:
                    continue
        logger.info(f"📋 Resume: found {len(done)} already-processed hashes.")
    except OSError as e:
        logger.warning(f"⚠️ Could not read existing output for resume: {e}")
    return done


def _match_hash_filter(commit_hash: str, hash_filter_set: Set[str]) -> bool:
    """
    FIX: Support prefix matching (consistent with main.py comments).
    Each element in hash_filter_set can be a full hash or a prefix.
    """
    for h in hash_filter_set:
        if commit_hash.startswith(h) or h.startswith(commit_hash):
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════

class Runner:
    """
    Driver for batch-running MultiPointPipeline.

    Flow (per DataItem):
      1. Filter (repo_filter / hash_filter / resume)
      2. git checkout <hash>^  (pre-change state)
      3. DataItemParser.parse() -> ParsedItem
      4. MultiPointPipeline.run() -> predictions
      5. Build RunRecord, append to output JSONL
    """

    def __init__(self, config: RunnerConfig, llm_client: ILLMClient):
        self.config     = config
        self.llm_client = llm_client
        self.parser     = DataItemParser()
        self.stats      = RunStats()

        # FIX: Use abspath to prevent dirname("file.jsonl") == "" causing makedirs errors
        os.makedirs(os.path.dirname(os.path.abspath(config.output_path)), exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)
        os.makedirs(config.cache_dir, exist_ok=True)

        # Scan repos_base/{lang}/{repo} structure, build repo_name -> absolute path mapping
        self._repo_path_map = self._scan_repos(config.repos_base)

    @staticmethod
    def _scan_repos(repos_base: str) -> Dict[str, str]:
        """
        Scan repos_base for {language}/{repo} structure, return {repo_name: abs_path}.
        Supports repos placed directly under repos_base (compatible with both structures).
        """
        repo_map: Dict[str, str] = {}
        if not os.path.isdir(repos_base):
            logger.warning(f"⚠️ repos_base not found: {repos_base}")
            return repo_map

        entries = os.listdir(repos_base)
        for entry in entries:
            sub_path = os.path.join(repos_base, entry)
            if not os.path.isdir(sub_path):
                continue
            # Check if sub_path itself is a git repo -> directly one level
            if os.path.isdir(os.path.join(sub_path, ".git")):
                repo_map[entry] = sub_path
            else:
                # Otherwise try {language}/{repo}
                for repo_name in os.listdir(sub_path):
                    repo_path = os.path.join(sub_path, repo_name)
                    if os.path.isdir(os.path.join(repo_path, ".git")):
                        repo_map[repo_name] = repo_path

        logger.info(f"📂 Scanned {len(repo_map)} repos in {repos_base}")
        return repo_map

    # ──────────────────────────────────────────────────────────────
    # Main Entry
    # ──────────────────────────────────────────────────────────────

    def run_all(self) -> RunStats:
        """
        Iterate through the JSONL dataset, process each item.
        Return final statistics.
        """
        cfg = self.config

        # Resume: load already-processed hashes
        done_hashes: Set[str] = set()
        if cfg.resume:
            done_hashes = _load_done_hashes(cfg.output_path)

        # Convert hash_filter to set for faster lookup
        hash_filter_set: Optional[Set[str]] = None
        if cfg.hash_filter:
            hash_filter_set = set(cfg.hash_filter)
            logger.info(f"🔍 Hash filter: {len(hash_filter_set)} hashes")

        wall_start = time.perf_counter()

        # FIX: Open files separately to ensure fout exception does not affect fin closing
        # FIX: Ensure trailing newline before appending to prevent concatenation
        with open(cfg.data_path, "r", encoding="utf-8") as fin:
            # Check if existing file has trailing newline, add one if not
            fout = open(cfg.output_path, "a", encoding="utf-8")
            try:
                if os.path.getsize(cfg.output_path) > 0:
                    with open(cfg.output_path, "rb") as _f:
                        _f.seek(-1, os.SEEK_END)
                        if _f.read(1) != b"\n":
                            fout.write("\n")

                for raw_line in fin:
                    raw_line = raw_line.strip()
                    if not raw_line:
                        continue

                    # ── Parse raw JSON ─────────────────────────────
                    try:
                        raw = json.loads(raw_line)
                    except json.JSONDecodeError as e:
                        logger.warning(f"⚠️ Skipping malformed JSON line: {e}")
                        continue

                    commit_hash = raw.get("hash", "")
                    repo_name   = raw.get("repo", "")
                    self.stats.total += 1

                    tag = f"[{repo_name}/{commit_hash[:10]}]"

                    # ── Filter: repo_filter ─────────────────────────
                    if cfg.repo_filter and repo_name != cfg.repo_filter:
                        self.stats.skipped_filter += 1
                        continue

                    # ── Filter: hash_filter (supports prefix match)─────────
                    if hash_filter_set and not _match_hash_filter(commit_hash, hash_filter_set):
                        self.stats.skipped_filter += 1
                        continue

                    # ── Filter: resume ──────────────────────────────
                    if cfg.resume and commit_hash in done_hashes:
                        self.stats.skipped_resume += 1
                        logger.debug(f"⏭️  {tag} Already done, skipping.")
                        continue

                    # FIX: Limit based on actual processed count, not succeeded count
                    if cfg.limit is not None and self.stats.processed >= cfg.limit:
                        logger.info(f"🏁 Reached limit={cfg.limit}, stopping.")
                        break

                    logger.info(f"\n{'─'*60}\n▶ {tag} Processing...")

                    record = self._process_one(raw, repo_name, commit_hash)
                    fout.write(_serialize_record(record) + "\n")
                    fout.flush()

                    if record.error:
                        logger.warning(f"⚠️ {tag} Written with error: {record.error}")
                    else:
                        # FIX: succeeded incremented here, no_predictions counted separately
                        self.stats.succeeded += 1
                        if record.predictions:
                            logger.info(f"✅ {tag} {len(record.predictions)} prediction(s)")
                        else:
                            self.stats.no_predictions += 1
                            logger.info(f"✅ {tag} No predictions (no-op or dry-run)")

            finally:
                fout.close()

        wall_elapsed = time.perf_counter() - wall_start
        logger.info(
            f"\n{'═'*60}\n"
            f"🏁 Run complete in {wall_elapsed:.1f}s\n"
            f"{json.dumps(self.stats.summary(), indent=2)}\n"
            f"{'═'*60}"
        )
        return self.stats

    # ──────────────────────────────────────────────────────────────
    # Single Item Processing
    # ──────────────────────────────────────────────────────────────

    def _process_one(
        self,
        raw: dict,
        repo_name: str,
        commit_hash: str,
    ) -> RunRecord:
        """
        Process a single DataItem, return RunRecord (whether success or failure).
        All exceptions are caught within this method and not propagated upward.
        """
        cfg       = self.config
        repo_path = self._repo_path_map.get(repo_name) or os.path.join(cfg.repos_base, repo_name)
        sandbox_repo_path = os.path.join(cfg.cache_dir,commit_hash)
        run_id    = str(uuid.uuid4())
        tag       = f"[{repo_name}/{commit_hash[:10]}]"

        def _make_error_record(err: str, root_hunk: Optional[Hunk] = None) -> RunRecord:
            """Factory function for failed RunRecord to avoid duplicate code."""
            return RunRecord(
                run_id=run_id,
                commit_hash=commit_hash,
                repo_name=repo_name,
                root_hunk=root_hunk,
                predictions=[],
                token_usage=TokenUsage(),
                error=err,
            )

        # ── Step 1: Validate repo directory ──────────────────────────────
        if not os.path.isdir(repo_path):
            err = f"Repo directory not found: {repo_path}"
            logger.error(f"❌ {tag} {err}")
            self.stats.failed_checkout += 1
            return _make_error_record(err)

        # ── Step 2: Git checkout commit^ ──────────────────────
        try:
            prepare_sandbox_repo(repo_path,sandbox_repo_path,commit_hash)
        except Exception as e:
            err = f"git checkout failed: {e}"
            logger.error(f"❌ {tag} {err}")
            self.stats.failed_checkout += 1
            cleanup_sandbox_repo(sandbox_repo_path)
            return _make_error_record(err)

        # ── Step 3: Parse DataItem -> ParsedItem ───────────────
        try:
            parsed_item = self.parser.parse(raw, repo_root=sandbox_repo_path)
        except Exception as e:
            err = f"Parser failed: {e}"
            logger.error(f"❌ {tag} {err}")
            self.stats.failed_parse += 1
            cleanup_sandbox_repo(sandbox_repo_path)
            return _make_error_record(err)

        # ── Step 4: Build Pipeline ─────────────────────────────
        pipeline = MultiPointPipeline(
            run_id = run_id,
            repo_path=sandbox_repo_path,
            cache_dir=os.path.join(cfg.cache_dir, repo_name),
            llm_client=self.llm_client,
            log_dir=cfg.log_dir,
            commit_hash=commit_hash,
        )

        # ── Step 5: Run Pipeline ─────────────────────────────
        try:
            predictions: List[PredictedEdit] = pipeline.run(
                parsed_item=parsed_item,
                dry_run=cfg.dry_run,
                context_mode=cfg.context_mode,
            )
        except Exception as e:
            err = f"Pipeline failed: {e}"
            # FIX: Use logger.exception for full stack trace to aid debugging
            logger.exception(f"❌ {tag} {err}")
            self.stats.failed_pipeline += 1
            cleanup_sandbox_repo(sandbox_repo_path)
            return RunRecord(
                run_id=run_id,
                commit_hash=commit_hash,
                repo_name=repo_name,
                root_hunk=parsed_item.root_hunk,
                target_hunks=parsed_item.target_hunks,
                predictions=[],
                token_usage=pipeline.total_usage,
                timing=pipeline.timing.summary(),
                llm_model_config=self._get_model_config(),
                error=err,
            )

        # ── Step 6: Build complete RunRecord ────────────────────────
        cleanup_sandbox_repo(sandbox_repo_path)
        return RunRecord(
            run_id=run_id,
            commit_hash=commit_hash,
            repo_name=repo_name,
            root_hunk=parsed_item.root_hunk,
            target_hunks=parsed_item.target_hunks,
            predictions=predictions,
            token_usage=pipeline.total_usage,
            timing=pipeline.timing.summary(),
            llm_model_config=self._get_model_config(),
            context_coverage=pipeline.context_coverage,
            stage1_result=pipeline.stage1_result,
            before_files=pipeline.before_files,
            gt_after_files=pipeline.gt_after_files,
            error=None,
        )

    # ──────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────

    def _get_model_config(self) -> dict:
        """Extract model config info from llm_client (for RunRecord metadata)"""
        return {
            "model":       getattr(self.llm_client, "model", "unknown"),
            "temperature": getattr(self.llm_client, "temperature", None),
            "max_tokens":  getattr(self.llm_client, "max_tokens", None),
        }
