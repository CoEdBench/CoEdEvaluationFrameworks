import json
import os
import logging
import re
import time
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

from src.domain.interfaces import ILLMClient
from src.domain.types import (
    ParsedItem, Hunk,
    PredictedEdit, TokenUsage, TimingStats, LLMCallStat,
)
from src.infrastructure.locagent.local_analyzer import LocalCodeAnalyzer
from src.core.assembler import PromptAssembler
from src.core.parser import _parse_diff_content
from src.evaluation.apply_patch import apply_diff_to_content

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# Window merging utility (used in oracle mode)
# ══════════════════════════════════════════════════════════════════════════

def _merge_windows(windows: list) -> list:
    if not windows:
        return []

    merged = []
    cur = {
        "win_start":   windows[0]["win_start"],
        "win_end":     windows[0]["win_end"],
        "hunk_ranges": [(windows[0]["hunk_start"], windows[0]["hunk_end"])],
        "node_ids":    [windows[0]["node_id"]],
        "usage_lines": list(windows[0]["usage_lines"]),
    }

    for w in windows[1:]:
        if w["win_start"] <= cur["win_end"] + 1:
            cur["win_end"] = max(cur["win_end"], w["win_end"])
            cur["hunk_ranges"].append((w["hunk_start"], w["hunk_end"]))
            cur["node_ids"].append(w["node_id"])
            cur["usage_lines"].extend(list(w["usage_lines"]))
        else:
            merged.append(cur)
            cur = {
                "win_start":   w["win_start"],
                "win_end":     w["win_end"],
                "hunk_ranges": [(w["hunk_start"], w["hunk_end"])],
                "node_ids":    [w["node_id"]],
                "usage_lines": list(w["usage_lines"]),
            }

    merged.append(cur)
    return merged


def _format_lines_with_lineno(
        file_lines: list,
        win_start: int,
        win_end: int,
) -> str:
    """
    Extract [win_start, win_end] window from file lines with line number prefix.
    Format: '   163: code...'
    win_start / win_end are 1-based.
    """
    formatted = []
    for i in range(win_start - 1, win_end):
        if i >= len(file_lines):
            break
        abs_line = i + 1
        content = file_lines[i].rstrip("\n")
        formatted.append(f"   {abs_line}: {content}")
    return "\n".join(formatted)


# ══════════════════════════════════════════════════════════════════════════
# MultiPointPipeline
# ══════════════════════════════════════════════════════════════════════════

def _find_reason_for_file(analysis_result: dict, target_file: str) -> str:
    for ctx in analysis_result.get("related_contexts", []):
        if ctx.get("file_path", "") == target_file:
            return ctx.get("reason", "")
    return ""


class MultiPointPipeline:

    def __init__(
            self,
            run_id: str,
            repo_path: str,
            cache_dir: str,
            llm_client: ILLMClient,
            log_dir: str = "llm_logs",
            commit_hash: str = "",
    ):
        self.repo_path   = repo_path
        self.cache_dir   = cache_dir
        self.assembler   = PromptAssembler()
        self.llm_client  = llm_client
        self.commit_hash = commit_hash

        self.total_usage      = TokenUsage()
        self.timing           = TimingStats()
        self.context_coverage = None
        self.stage1_result: Optional[Dict] = None

        self._analyzer: Optional[LocalCodeAnalyzer] = None
        self._graph_build_s: float = 0.0

        # File snapshots (for evaluation)
        self.before_files: Dict[str, str] = {}      # {rel_path: content}
        self.gt_after_files: Dict[str, str] = {}    # {rel_path: content_after_gt}

        self.run_id  = run_id
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.llm_log_file = os.path.join(
            self.log_dir,
            f"{self.run_id}_{self.commit_hash}_llm_interaction_{timestamp}.log",
        )
        logger.info(f"📝 LLM log: {self.llm_log_file}")

    # ──────────────────────────────────────────────────────────────
    # Lazy-load Analyzer
    # ──────────────────────────────────────────────────────────────

    @property
    def analyzer(self) -> LocalCodeAnalyzer:
        if self._analyzer is None:
            logger.info("🔄 Initializing LocalCodeAnalyzer (lazy)...")
            t0 = time.perf_counter()
            self._analyzer = LocalCodeAnalyzer(
                self.repo_path, cache_dir=self.cache_dir
            )
            self._graph_build_s = time.perf_counter() - t0
            logger.info(f"⏱️  Graph build: {self._graph_build_s:.2f}s")
        return self._analyzer

    # ──────────────────────────────────────────────────────────────
    # LLM call wrapper
    # ──────────────────────────────────────────────────────────────

    def _timed_llm_call(self, stage: str, messages: list):
        t0 = time.perf_counter()
        response_obj = self.llm_client.generate_completion(
            messages
        )
        elapsed = time.perf_counter() - t0

        stat = LLMCallStat(
            stage=stage,
            duration_s=elapsed,
            input_tokens=getattr(response_obj.usage, "prompt_tokens", 0),
            output_tokens=getattr(response_obj.usage, "completion_tokens", 0),
        )
        self.timing.llm_calls.append(stat)
        logger.info(
            f"⏱️  LLM [{stage}] {elapsed:.2f}s | "
            f"in={stat.input_tokens} out={stat.output_tokens}"
        )
        return response_obj

    # ──────────────────────────────────────────────────────────────
    # LLM interaction log
    # ──────────────────────────────────────────────────────────────

    def _log_llm_interaction(
            self,
            stage: str,
            messages: List[Dict[str, Any]],
            response_content: str,
            usage: Any = None,
    ) -> None:
        sep  = "=" * 60
        thin = "-" * 40
        ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = ["", sep, f"[{ts}]  Stage: {stage}", sep,
                 ">>> INPUT (messages)", thin]
        for i, msg in enumerate(messages):
            lines.append(f"[Message {i}]  role={msg.get('role', '').upper()}")
            lines.append(msg.get("content", ""))
            lines.append(thin)
        lines += ["<<< OUTPUT (raw response)", thin, response_content, thin]
        if usage is not None:
            lines.append(f"Token Usage: {usage}")
        lines += [sep, ""]

        try:
            with open(self.llm_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            logger.error(f"❌ Failed to write LLM log: {e}")

    def _log_coverage_to_file(self, coverage) -> None:
        sep = "=" * 60
        ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "", sep,
            f"[{ts}]  Stage: Context-Coverage-Check", sep,
            f"Recall: {coverage.recall:.2%}  "
            f"({coverage.covered_count}/{coverage.total_gt} GT covered)",
            f"Is Framework Issue: {coverage.is_framework_issue}",
            "",
            "Detail:",
            json.dumps(coverage.summary(), indent=2, ensure_ascii=False),
            sep, "",
        ]
        try:
            with open(self.llm_log_file, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except OSError as e:
            logger.error(f"❌ Failed to write coverage log: {e}")

    # ──────────────────────────────────────────────────────────────
    # File reading utilities
    # ──────────────────────────────────────────────────────────────

    def _read_file_lines(self, rel_path: str) -> List[str]:
        """Read all lines of file (preserve newlines), return empty list on error"""
        abs_path = os.path.join(self.repo_path, rel_path)
        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                return f.readlines()
        except OSError as e:
            logger.error(f"Failed to read {abs_path}: {e}")
            return []

    def _read_file_snippets(
            self,
            rel_path: str,
            line_nums: List[int],
            context_window: int = 10,
    ) -> List[Tuple[str, int, int]]:
        """
        For each impacted line number, generate independent windows, merge adjacent/overlapping windows,
        return multiple (snippet, start_line, end_line) tuples.
        """
        lines = self._read_file_lines(rel_path)
        if not lines:
            return []

        total = len(lines)

        if not line_nums:
            logger.warning(f"No line_nums for {rel_path}, reading full file.")
            snippet = "".join(lines)
            return [(snippet, 1, total)]

        # ── Step 1: Generate raw windows for each line number ──────────────────────────
        raw_windows: List[Tuple[int, int]] = []
        for ln in sorted(line_nums):
            win_start = max(1, ln - context_window)
            win_end = min(total, ln + context_window)
            raw_windows.append((win_start, win_end))

        # ── Step 2: Merge overlapping/adjacent windows ───────────────────────────────
        merged_windows: List[Tuple[int, int]] = []
        cur_start, cur_end = raw_windows[0]
        for win_start, win_end in raw_windows[1:]:
            if win_start <= cur_end:  # Adjacent or overlapping, merge
                cur_end = max(cur_end, win_end)
            else:  # Not adjacent, save current, start new window
                merged_windows.append((cur_start, cur_end))
                cur_start, cur_end = win_start, win_end
        merged_windows.append((cur_start, cur_end))

        # ── Step 3: Extract code snippet for each window ──────────────────────────
        results: List[Tuple[str, int, int]] = []
        for win_start, win_end in merged_windows:
            snippet = "".join(lines[win_start - 1: win_end])
            logger.info(
                f"📖 Snippet [{rel_path}] lines {win_start}~{win_end} "
                f"(impacted={line_nums}, window=±{context_window})"
            )
            results.append((snippet, win_start, win_end))

        return results

    # ──────────────────────────────────────────────────────────────
    # Apply Root Hunk to Disk
    # ──────────────────────────────────────────────────────────────
    def _apply_root_hunk_to_disk(self, parsed_item: ParsedItem) -> None:
        """
        Apply root hunk changes to sandbox files.

        Only exact string matching replacement. Raises RuntimeError on failure,
        caught upstream as failed_pipeline.
        """
        root_hunk = parsed_item.root_hunk
        before_code = parsed_item.root_before_code or ""
        after_code = parsed_item.root_after_code or ""
        abs_path = os.path.join(self.repo_path, root_hunk.file_path)

        if not os.path.exists(abs_path):
            raise RuntimeError(
                f"Root hunk file not found: {abs_path}"
            )

        try:
            with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                original = f.read()
        except OSError as e:
            raise RuntimeError(
                f"Failed to read root hunk file {abs_path}: {e}"
            )

        def _norm(s: str) -> str:
            return s.replace("\r\n", "\n").replace("\r", "\n")

        original_norm = _norm(original)
        before_norm = _norm(before_code)
        after_norm = _norm(after_code)

        if before_norm and before_norm in original_norm:
            result_content = original_norm.replace(before_norm, after_norm, 1)
            try:
                with open(abs_path, "w", encoding="utf-8") as f:
                    f.write(result_content)
            except OSError as e:
                raise RuntimeError(
                    f"Failed to write root hunk file {abs_path}: {e}"
                )
            logger.info(
                f"⚡ Applied root hunk [exact]: {root_hunk.file_path} "
                f"[old_start={root_hunk.old_start_line}, "
                f"old_len={root_hunk.old_len} → new_len={root_hunk.new_len}]"
            )
        else:
            raise RuntimeError(
                f"Root hunk exact match failed for {root_hunk.file_path}: "
                f"before_code not found in file content."
            )

    # ──────────────────────────────────────────────────────────────
    # Oracle Context Building
    # ──────────────────────────────────────────────────────────────

    def _build_oracle_context(
            self,
            parsed_item: ParsedItem,
            context_window: int = 30,
    ) -> dict:
        root_hunk = parsed_item.root_hunk

        context_report = {
            "focus_node": {
                "id":         root_hunk.id,
                "type":       "hunk",
                "file":       root_hunk.file_path,
                "start_line": root_hunk.start_line,
                "end_line":   root_hunk.end_line,
                "code":       parsed_item.root_before_code,
            },
            "related_contexts": [],
        }

        file_hunk_map: Dict[str, List[Hunk]] = defaultdict(list)
        for hunk in parsed_item.target_hunks:
            file_hunk_map[hunk.file_path].append(hunk)

        for rel_file_path, hunks in file_hunk_map.items():
            file_lines = self._read_file_lines(rel_file_path)
            if not file_lines:
                logger.warning(f"⚠️ [Oracle] Cannot read: {rel_file_path}")
                continue

            total_lines = len(file_lines)
            windows = []
            for hunk in hunks:
                win_start = max(1, hunk.start_line - context_window)
                win_end   = min(total_lines, hunk.end_line + context_window)
                windows.append({
                    "win_start":   win_start,
                    "win_end":     win_end,
                    "hunk_start":  hunk.start_line,
                    "hunk_end":    hunk.end_line,
                    "node_id":     hunk.id,
                    "usage_lines": list(range(hunk.start_line, hunk.end_line + 1)),
                })

            windows.sort(key=lambda w: w["win_start"])
            merged = _merge_windows(windows)

            for m in merged:
                snippet = _format_lines_with_lineno(
                    file_lines=file_lines,
                    win_start=m["win_start"],
                    win_end=m["win_end"],
                )
                context_report["related_contexts"].append({
                    "role":           "ORACLE_GT",
                    "reason":         f"Target hunk(s) at lines {m['hunk_ranges']}",
                    "distance":       1,
                    "node_ids":       m["node_ids"],
                    "node_id":        m["node_ids"][0],
                    "file_path":      rel_file_path,
                    "rel_file_path":  rel_file_path,
                    "relevant_code":  snippet,
                    "usage_line_nums": m["usage_lines"],
                })
                logger.info(
                    f"📌 [Oracle] {rel_file_path}: "
                    f"window [{m['win_start']}~{m['win_end']}] "
                    f"covering {m['hunk_ranges']}"
                )

        logger.info(
            f"🔮 [Oracle] Built {len(context_report['related_contexts'])} context(s)"
        )
        return context_report

    # ──────────────────────────────────────────────────────────────
    # LLM Output Parsing
    # ──────────────────────────────────────────────────────────────
    def _parse_next_version(
            self,
            llm_output: str,
            current_version:str,
            predicted_line_nums: List[int],
            file_path: str,
            start_line: int,
            end_line: int,
            order: int,
    ) -> Optional[PredictedEdit]:
        """
        Parse Stage 2 JSON output.

        LLM only needs to output:
          { "next_version": "...", "change_summary": "..." }

        start_line / end_line computed by pipeline from Stage 1 line numbers ±10,
        not dependent on LLM output, avoiding line number hallucination.
        """
        raw = llm_output.strip()

        data = self.llm_client.parse_json_response(raw)
        if not isinstance(data, dict):
            logger.warning(f"⚠️ [Stage2] LLM returned non-dict type: {type(data).__name__}, Skipping {file_path}")
            return None
        try:
            next_version = data.get("next_version")  # null -> None -> no change needed
            change_summary = data.get("change_summary", "")
        except (KeyError, TypeError) as e:
            logger.warning(f"⚠️ [Stage2] Invalid JSON fields for {file_path}: {e}")
            return None

        if next_version is None:
            logger.info(f"ℹ️  [Stage2] No change needed for {file_path}: {change_summary}")
            return None

        return PredictedEdit(
            file_path=file_path,
            predicted_line_nums=predicted_line_nums,
            start_line=start_line,  # from pipeline, actual line numbers
            end_line=end_line,  # from pipeline, actual line numbers
            next_version=next_version,
            current_version = current_version,
            change_summary=change_summary,
            predicted_order=order,
        )

    @staticmethod
    def _try_parse_json(text: str) -> Optional[dict]:
        """Attempt JSON parsing, return None on failure"""
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except json.JSONDecodeError:
            return None

    # ──────────────────────────────────────────────────────────────
    # File snapshots (for evaluation)
    # ──────────────────────────────────────────────────────────────

    def _snapshot_files(self, target_hunks: List[Hunk], predictions: List[PredictedEdit]) -> None:
        """Read all involved files, build before/gt_after snapshots."""
        # Collect involved files
        all_files: set = set()
        for h in target_hunks:
            all_files.add(h.file_path)
        for p in predictions:
            all_files.add(p.file_path)

        self.before_files = {}
        self.gt_after_files = {}

        for rel_path in all_files:
            abs_path = os.path.join(self.repo_path, rel_path)
            try:
                with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read()
            except OSError as e:
                logger.warning(f"⚠️ Cannot snapshot {rel_path}: {e}")
                continue

            self.before_files[rel_path] = content

            # Build gt_after: apply target_hunks in reverse line number order
            gt_content = content
            related_hunks = sorted(
                [h for h in target_hunks if h.file_path == rel_path],
                key=lambda h: -h.start_line,
            )
            for hunk in related_hunks:
                before, after = _parse_diff_content(hunk.content)
                result = apply_diff_to_content(
                    content=gt_content,
                    before_code=before,
                    after_code=after,
                    old_start_line=hunk.start_line,
                    old_len=hunk.end_line - hunk.start_line + 1,
                )
                if result is not None:
                    gt_content = result
                else:
                    logger.warning(f"⚠️ Failed to apply GT hunk {hunk.id} to {rel_path}")

            self.gt_after_files[rel_path] = gt_content

        logger.info(
            f"📸 Snapshot {len(self.before_files)} files for evaluation"
        )

    # ──────────────────────────────────────────────────────────────
    # Main Pipeline
    # ──────────────────────────────────────────────────────────────

    def run(
            self,
            parsed_item: ParsedItem,
            dry_run: bool = False,
            context_mode: str = "auto",
    ) -> List[PredictedEdit]:
        """
        Main pipeline flow:
          1. Apply root_hunk to disk (simulate change already applied)
          2. Build context (oracle / auto)
          3. [auto] Recall validation, early return on failure
          4. [dry_run] Stop here, return []
          5. Stage 1: LLM identifies impacted files and line numbers
          6. Stage 2: Per-file context extraction by line number → LLM generates next_version
        """
        self.total_usage      = TokenUsage()
        self.timing           = TimingStats()
        self.timing.graph_build_s = self._graph_build_s
        self.context_coverage = None
        pipeline_start = time.perf_counter()

        root_hunk    = parsed_item.root_hunk
        target_hunks = parsed_item.target_hunks
        mode_tag     = f"[{'DRY-RUN ' if dry_run else ''}{context_mode.upper()}]"

        logger.info(
            f"🚀 {mode_tag} Pipeline start | "
            f"root={root_hunk.id} | targets={len(target_hunks)}"
        )

        # ── Main body wrapped in try/finally to always set total_s ─
        try:
            # ── Step 1: Apply root hunk to disk ─────────────────────
            self._apply_root_hunk_to_disk(parsed_item)

            # ── Step 2: Build context ────────────────────────────────
            if context_mode == "oracle":
                if not target_hunks:
                    logger.warning("⚠️ Oracle mode: no target hunks, nothing to predict.")
                    return []
                analysis_result = self._build_oracle_context(parsed_item)

            else:  # "auto"
                raise RuntimeError(f"Context from graph not supported: {context_mode}")
                try:
                    analysis_result = self.analyzer.get_co_edit_context(
                        root_hunk.file_path, root_hunk.start_line
                    )
                except Exception as e:
                    logger.exception(f"❌ LocalCodeAnalyzer failed: {e}")
                    raise RuntimeError(f"Context analysis failed: {e}") from e

                if target_hunks:
                    coverage = self.analyzer.check_context_coverage(
                        analysis_result, target_hunks
                    )
                    self.context_coverage = coverage
                    self._log_coverage_to_file(coverage)
                    if coverage.is_framework_issue:
                        logger.warning(
                            f"🚨 Framework recall issue. "
                            f"Recall={coverage.recall:.2%}. Skipping LLM."
                        )
                        return []

            # ── Step 3: Early exit if no context available ───────────────────
            if not analysis_result.get("related_contexts"):
                logger.info("✅ No related contexts found.")
                return []

            # ── Step 4: Dry-run stop here ──────────────────────────
            if dry_run:
                logger.info(f"✅ {mode_tag} Dry-run done. Skipping LLM.")
                self._snapshot_files(target_hunks, [])
                return []

            # ── Step 5: Stage 1 ──────────────────────────────────
            stage1_messages = self.assembler.assemble(
                parsed_item=parsed_item,
                context_report=analysis_result,
            )
            logger.info(f"🤖 {mode_tag} [Stage 1] Identifying impacted lines...")

            stage1_response = self._timed_llm_call(
                stage="Stage1-identify-lines",
                messages=stage1_messages,
            )
            self.total_usage.add(stage1_response.usage)
            self._log_llm_interaction(
                stage="Stage1-identify-lines",
                messages=stage1_messages,
                response_content=stage1_response.content,
                usage=stage1_response.usage,
            )
            response_json = self.llm_client.parse_json_response(stage1_response.content)
            self.stage1_result = response_json if isinstance(response_json, dict) else {"raw": stage1_response.content}
            if not isinstance(response_json, dict):
                logger.warning(f"⚠️ [Stage1] LLM returned non-dict type: {type(response_json).__name__}, Skipping")
                return []
            raw_locations = response_json.get("impacted_locations", [])
            stage1_reasoning = response_json.get("reasoning", "")

            impacted_map: Dict[str, List[int]] = {}
            file_to_reason: Dict[str, str] = {}
            for loc in raw_locations:
                if not isinstance(loc, dict):
                    logger.warning(f"⚠️ Invalid location entry: {loc!r}, skipping.")
                    continue
                file_key = loc.get("file", "").strip()
                if not file_key:
                    logger.warning(f"⚠️ Missing 'file' key: {loc!r}, skipping.")
                    continue
                parsed_lines: List[int] = []
                for ln in loc.get("lines", []):
                    try:
                        parsed_lines.append(int(ln))
                    except (TypeError, ValueError):
                        logger.warning(f"⚠️ Invalid line number {ln!r} for {file_key}.")
                if parsed_lines:
                    existing = impacted_map.get(file_key, [])
                    impacted_map[file_key] = sorted(set(existing + parsed_lines))
                loc_reason = loc.get("reason", "").strip()
                if loc_reason and file_key not in file_to_reason:
                    file_to_reason[file_key] = loc_reason

            logger.info(f"🎯 {mode_tag} [Stage 1] Impacted: {impacted_map}")

            if not impacted_map:
                logger.warning(f"⚠️ {mode_tag} [Stage 1] Empty impacted locations — LLM found no impacted locations, skipping Stage 2")
                return []

            # ── Step 6: Stage 2 ──────────────────────────────────
            predictions: List[PredictedEdit] = []

            for order, (target_file, line_nums) in enumerate(impacted_map.items()):
                logger.info(f"🔧 [Stage2] {target_file} | lines={line_nums}")
                predicted_line_nums = line_nums.copy()
                snippets = self._read_file_snippets(
                    target_file, line_nums, context_window=10
                )

                if not snippets:
                    logger.warning(f"⚠️ Empty snippets for {target_file}, skipping.")
                    continue

                for snippet_idx, (current_version, start_line, end_line) in enumerate(snippets):
                    logger.info(
                        f"🔧 [Stage2] {target_file} | "
                        f"window[{snippet_idx}] lines {start_line}~{end_line}"
                    )

                    window_lines = [ln for ln in line_nums if start_line <= ln <= end_line]
                    file_reason = file_to_reason.get(target_file, "")
                    fix_messages = self.assembler.assemble_fix(
                        parsed_item=parsed_item,
                        file_path=target_file,
                        current_version=current_version,
                        reasoning=stage1_reasoning,
                        impacted_lines=window_lines,
                        location_reason=file_reason,
                        snippet_start_line=start_line,
                    )

                    fix_response = self._timed_llm_call(
                        stage=f"Stage2-fix [{target_file}][{start_line}-{end_line}]",
                        messages=fix_messages
                    )
                    self.total_usage.add(fix_response.usage)
                    self._log_llm_interaction(
                        stage=f"Stage2-fix [{target_file}][{start_line}-{end_line}]",
                        messages=fix_messages,
                        response_content=fix_response.content,
                        usage=fix_response.usage,
                    )

                    edit = self._parse_next_version(
                        llm_output=fix_response.content,
                        current_version=current_version,
                        predicted_line_nums=predicted_line_nums,
                        file_path=target_file,
                        start_line=start_line,
                        end_line=end_line,
                        order=order,
                    )
                    if edit is not None:
                        predictions.append(edit)
                        logger.info(
                            f"✅ [Stage2] {target_file} | "
                            f"window[{snippet_idx}] lines {edit.start_line}–{edit.end_line} | "
                            f"{edit.change_summary}"
                        )
                    else:
                        logger.warning(
                            f"⚠️ [Stage2] No valid edit for {target_file} "
                            f"window[{snippet_idx}] lines {start_line}~{end_line}"
                        )

            # ── Step 7: Snapshot file content for evaluation ────────────────
            self._snapshot_files(target_hunks, predictions)

            return predictions

        finally:
            self.timing.total_s = time.perf_counter() - pipeline_start