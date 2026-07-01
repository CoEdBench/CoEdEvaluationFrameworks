"""
train_runner.py
================
Batch-drives TrainingSetBuilder to convert ParsedItem into training samples and write to JSONL.

Flow (per DataItem):
  1. Filter (repo_filter / hash_filter / resume)
  2. prepare_sandbox_repo → checkout commit^
  3. DataItemParser.parse() -> ParsedItem
  4. TrainingSetBuilder.build() -> List[dict] training samples
  5. Append to output JSONL
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from src.core.parser import DataItemParser
from src.core.train_make_builder import TrainingSetBuilder
from src.core.train_dataset.cot_enricher import CoTEnricher
from src.utils.sandbox_utils import prepare_sandbox_repo, cleanup_sandbox_repo

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Configuration
# ══════════════════════════════════════════════════════════════════════════
load_dotenv()
@dataclass
class TrainRunnerConfig:
    data_path:      str
    repos_base:     str
    output_path:    str
    cache_dir:      str = ".cache"
    log_dir:        str = "train_logs"
    limit:          Optional[int]  = None
    resume:         bool           = True
    repo_filter:    Optional[str]  = None
    hash_filter:    Optional[List[str]] = None
    context_window: int = 30
    snippet_window: int = 10
    # CoT enrichment (optional)
    cot_provider:   Optional[str]  = None
    cot_api_key:    str            = os.environ.get("API_KEY")
    cot_model:      str            = os.environ.get("LLM_MODEL")
    cot_concurrency: int           = 1
    use_cot:        Optional[bool] = None



# ══════════════════════════════════════════════════════════════════════════
# Statistics
# ══════════════════════════════════════════════════════════════════════════

@dataclass
class TrainRunStats:
    total:           int = 0
    skipped_resume:  int = 0
    skipped_filter:  int = 0
    failed_checkout: int = 0
    failed_parse:    int = 0
    failed_build:    int = 0
    succeeded:       int = 0
    samples_written: int = 0

    @property
    def processed(self) -> int:
        return self.succeeded + self.failed_checkout + self.failed_parse + self.failed_build

    def summary(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"processed": self.processed}


# ══════════════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════════════

def _load_done_hashes(output_path: str) -> Set[str]:
    """Read all already-processed commit_hashes from existing output file, for resume."""
    done: Set[str] = set()
    if not os.path.exists(output_path):
        return done
    try:
        with open(output_path, encoding="utf-8") as f:
            for line in f:
                try:
                    h = json.loads(line).get("meta", {}).get("commit_hash", "")
                    if h:
                        done.add(h)
                except json.JSONDecodeError:
                    continue
        logger.info(f"Resume: {len(done)} hashes already processed.")
    except OSError as e:
        logger.warning(f"Could not read output for resume: {e}")
    return done


def _hash_matches(commit_hash: str, filter_set: Set[str]) -> bool:
    """Hash filter supporting prefix matching."""
    return any(
        commit_hash.startswith(h) or h.startswith(commit_hash)
        for h in filter_set
    )


def _ensure_newline(path: str) -> None:
    """Add trailing newline if missing, ensuring correct JSONL format on append."""
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


# ══════════════════════════════════════════════════════════════════════════
# TrainRunner
# ══════════════════════════════════════════════════════════════════════════

class TrainRunner:

    def __init__(self, config: TrainRunnerConfig):
        self.cfg    = config
        self.stats  = TrainRunStats()
        self.parser = DataItemParser()

        for d in (os.path.dirname(os.path.abspath(config.output_path)),
                  config.log_dir, config.cache_dir):
            os.makedirs(d, exist_ok=True)

        # Scan repos_base/{lang}/{repo} structure
        self._repo_path_map = self._scan_repos(config.repos_base)

        # CoT Enricher (optional)
        self.cot_enricher: Optional[CoTEnricher] = None
        if config.use_cot is True:
            self.cot_enricher = CoTEnricher(
                provider    = config.cot_provider,
                api_key     = config.cot_api_key,
                model       = config.cot_model,
                concurrency = config.cot_concurrency,
            )
            logger.info("CoT enricher enabled: %s/%s", config.cot_provider, config.cot_model)

    @staticmethod
    def _scan_repos(repos_base: str) -> Dict[str, str]:
        repo_map: Dict[str, str] = {}
        if not os.path.isdir(repos_base):
            logger.warning("repos_base not found: %s", repos_base)
            return repo_map
        for entry in os.listdir(repos_base):
            sub_path = os.path.join(repos_base, entry)
            if not os.path.isdir(sub_path):
                continue
            if os.path.isdir(os.path.join(sub_path, ".git")):
                repo_map[entry] = sub_path
            else:
                for repo_name in os.listdir(sub_path):
                    repo_path = os.path.join(sub_path, repo_name)
                    if os.path.isdir(os.path.join(repo_path, ".git")):
                        repo_map[repo_name] = repo_path
        logger.info("Scanned %d repos in %s", len(repo_map), repos_base)
        return repo_map

    # ──────────────────────────────────────────────────────────────
    # Main Entry
    # ──────────────────────────────────────────────────────────────

    def run_all(self) -> TrainRunStats:
        cfg = self.cfg

        done_hashes    = _load_done_hashes(cfg.output_path) if cfg.resume else set()
        hash_filter_set = set(cfg.hash_filter) if cfg.hash_filter else None

        _ensure_newline(cfg.output_path)
        t0 = time.perf_counter()

        with open(cfg.data_path, encoding="utf-8") as fin, \
             open(cfg.output_path, "a", encoding="utf-8") as fout:

            for raw_line in fin:
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                try:
                    raw = json.loads(raw_line)
                except json.JSONDecodeError as e:
                    logger.warning(f"Malformed JSON, skipping: {e}")
                    continue

                commit_hash = raw.get("hash", "")
                repo_name   = raw.get("repo", "")
                self.stats.total += 1
                tag = f"[{repo_name}/{commit_hash[:10]}]"

                # ── filter ──────────────────────────────────────────
                if cfg.repo_filter and repo_name != cfg.repo_filter:
                    self.stats.skipped_filter += 1
                    continue

                if hash_filter_set and not _hash_matches(commit_hash, hash_filter_set):
                    self.stats.skipped_filter += 1
                    continue

                if cfg.resume and commit_hash in done_hashes:
                    self.stats.skipped_resume += 1
                    logger.debug(f"Skip {tag} Already done.")
                    continue

                if cfg.limit is not None and self.stats.processed >= cfg.limit:
                    logger.info(f"Reached limit={cfg.limit}, stopping.")
                    break

                logger.info(f"\n{'─'*60}\n{tag} Building training samples...")

                # ── Process ─────────────────────────────────────────
                samples, err = self._process_one(raw, repo_name, commit_hash)

                if err:
                    logger.warning(f"{tag} Skipped: {err}")
                    continue

                for sample in samples:
                    fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
                fout.flush()

                self.stats.succeeded       += 1
                self.stats.samples_written += len(samples)
                logger.info(f"{tag} +{len(samples)} samples (total={self.stats.samples_written})")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"\n{'═'*60}\n"
            f"Done in {elapsed:.1f}s\n"
            f"{json.dumps(self.stats.summary(), indent=2)}\n"
            f"{'═'*60}"
        )
        return self.stats

    # ──────────────────────────────────────────────────────────────
    # Single Item Processing
    # ──────────────────────────────────────────────────────────────

    def _process_one(
            self,
            raw:         dict,
            repo_name:   str,
            commit_hash: str,
    ) -> tuple[List[dict], Optional[str]]:
        """
        Returns (samples, error_msg).
        On failure: samples=[], error_msg is non-null; all exceptions caught here, not propagated upward.
        """
        cfg          = self.cfg
        repo_path    = self._repo_path_map.get(repo_name) or os.path.join(cfg.repos_base, repo_name)
        sandbox_path = os.path.join(cfg.cache_dir, commit_hash)
        tag          = f"[{repo_name}/{commit_hash[:10]}]"

        # Step 1: Verify repository
        if not os.path.isdir(repo_path):
            self.stats.failed_checkout += 1
            return [], f"Repo not found: {repo_path}"

        # Step 2: Prepare sandbox
        try:
            prepare_sandbox_repo(repo_path, sandbox_path, commit_hash)
        except Exception as e:
            self.stats.failed_checkout += 1
            cleanup_sandbox_repo(sandbox_path)
            return [], f"prepare_sandbox_repo: {e}"

        # Step 3 & 4: parse -> build (shared finally block for sandbox cleanup)
        try:
            parsed_item = self._parse(raw, sandbox_path, tag)
            samples     = self._build(parsed_item, sandbox_path, commit_hash, tag)
            return samples, None

        except _StepError as e:
            return [], str(e)
        finally:
            cleanup_sandbox_repo(sandbox_path)

    def _parse(self, raw: dict, sandbox_path: str, tag: str):
        try:
            return self.parser.parse(raw, repo_root=sandbox_path)
        except Exception as e:
            self.stats.failed_parse += 1
            raise _StepError(f"Parser: {e}") from e

    def _build(self, parsed_item, sandbox_path: str, commit_hash: str, tag: str) -> List[dict]:
        try:
            builder = TrainingSetBuilder(
                repo_path=sandbox_path,
                commit_hash=commit_hash,
                context_window=self.cfg.context_window,
                snippet_window=self.cfg.snippet_window,
                cot_enricher=self.cot_enricher,
            )
            return builder.build(parsed_item, use_cot=self.cfg.use_cot)  # ← pass through use_cot
        except Exception as e:
            self.stats.failed_build += 1
            raise _StepError(f"Builder: {e}") from e


# ══════════════════════════════════════════════════════════════════════════
# Internal exception (for _process_one flow control only)
# ══════════════════════════════════════════════════════════════════════════

class _StepError(Exception):
    """Internal signal for parse/build step failure, carrying error description."""