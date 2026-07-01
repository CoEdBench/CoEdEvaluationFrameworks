"""
training_set_builder.py
=======================
Build LLM training set from ParsedItem, output JSONL.

Dependencies:
  - prompt_assembler.py  -> PromptAssembler
  - cot_enricher.py      -> CoTEnricher (optional)

Each ParsedItem produces:
  - 1 Stage1 sample (identify impacted locations)
  - N Stage2 samples (one per target hunk window)

Output format (JSONL):
  {
    "id":             str,
    "stage":          "stage1" | "stage2",
    "messages":       [...],
    "ground_truth":   str,   # raw JSON or <think>...</think><answer>...</answer>
    "ground_truth_raw": str, # only exists in CoT mode, preserves raw JSON
    "cot_enriched":   bool,  # only exists in CoT mode
    "meta":           {...}
  }
"""

import json
import logging
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.domain.types import Hunk, ParsedItem
from src.core.train_dataset.cot_enricher import CoTEnricher
from src.core.train_dataset.prompt_assembler import PromptAssembler

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════
# File Utilities
# ══════════════════════════════════════════════════════════════════════════

def _read_lines(abs_path: str) -> List[str]:
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except OSError as e:
        logger.error(f"Failed to read {abs_path}: {e}")
        return []


def _format_lines_with_lineno(lines: List[str], start: int, end: int) -> str:
    return "\n".join(
        f"   {i + 1}: {lines[i].rstrip()}"
        for i in range(start - 1, end)
        if i < len(lines)
    )


def _merge_line_windows(
        line_nums: List[int],
        total: int,
        window: int,
) -> List[Tuple[int, int]]:
    """Extend discrete line numbers into windows and merge overlapping intervals."""
    if not line_nums:
        return [(1, total)]
    raw = [(max(1, ln - window), min(total, ln + window)) for ln in sorted(line_nums)]
    merged, cur_s, cur_e = [], raw[0][0], raw[0][1]
    for s, e in raw[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _merge_hunk_windows(windows: List[dict]) -> List[dict]:
    """Merge adjacent/overlapping hunk windows, preserving hunk_ranges / node_ids / usage_lines."""
    if not windows:
        return []
    windows = sorted(windows, key=lambda w: w["win_start"])
    merged, cur = [], dict(windows[0])
    cur["hunk_ranges"] = [(windows[0]["hunk_start"], windows[0]["hunk_end"])]
    cur["node_ids"]    = [windows[0]["node_id"]]
    cur["usage_lines"] = list(windows[0]["usage_lines"])

    for w in windows[1:]:
        if w["win_start"] <= cur["win_end"] + 1:
            cur["win_end"] = max(cur["win_end"], w["win_end"])
            cur["hunk_ranges"].append((w["hunk_start"], w["hunk_end"]))
            cur["node_ids"].append(w["node_id"])
            cur["usage_lines"].extend(w["usage_lines"])
        else:
            merged.append(cur)
            cur = {**w,
                   "hunk_ranges": [(w["hunk_start"], w["hunk_end"])],
                   "node_ids":    [w["node_id"]],
                   "usage_lines": list(w["usage_lines"])}
    merged.append(cur)
    return merged


# ══════════════════════════════════════════════════════════════════════════
# TrainingSetBuilder
# ══════════════════════════════════════════════════════════════════════════

class TrainingSetBuilder:
    """
    Build training samples from ParsedItem with no LLM dependency (CoT enrichment optional).

    Args:
      repo_path      - Repository root directory
      commit_hash    - Current commit hash, used for sample IDs
      context_window - Oracle context window (lines)
      snippet_window - Stage2 code snippet window (lines)
      cot_enricher   - CoTEnricher instance (None -> raw JSON mode)
    """

    def __init__(
            self,
            repo_path:       str,
            commit_hash:     str                    = "",
            context_window:  int                    = 30,
            snippet_window:  int                    = 10,
            cot_enricher:    Optional[CoTEnricher]  = None,
    ):
        self.repo_path      = repo_path
        self.commit_hash    = commit_hash
        self.context_window = context_window
        self.snippet_window = snippet_window
        self.cot_enricher   = cot_enricher
        self.assembler      = PromptAssembler()

    # ──────────────────────────────────────────────────────────────
    # File reading
    # ──────────────────────────────────────────────────────────────

    def _abs(self, rel: str) -> str:
        return os.path.join(self.repo_path, rel)

    def _file_lines(self, rel: str) -> List[str]:
        return _read_lines(self._abs(rel))

    def _file_snippets(
            self,
            rel: str,
            line_nums: List[int],
            window: int,
    ) -> List[Tuple[str, int, int]]:
        """Returns list of (snippet_text, start_line, end_line)."""
        lines = self._file_lines(rel)
        if not lines:
            return []
        merged = _merge_line_windows(line_nums, len(lines), window)
        return [("".join(lines[s - 1: e]), s, e) for s, e in merged]

    # ──────────────────────────────────────────────────────────────
    # Apply Root Hunk to Disk
    # ──────────────────────────────────────────────────────────────

    def _apply_root_hunk(self, parsed_item: ParsedItem) -> bool:
        """Write root hunk before -> after to disk, using three-level matching strategy."""
        hunk        = parsed_item.root_hunk
        abs_path    = self._abs(hunk.file_path)
        before      = (parsed_item.root_before_code or "").replace("\r\n", "\n").replace("\r", "\n")
        after       = (parsed_item.root_after_code  or "").replace("\r\n", "\n").replace("\r", "\n")

        if not os.path.exists(abs_path):
            logger.error(f"File not found: {abs_path}")
            return False

        try:
            original = open(abs_path, encoding="utf-8", errors="replace").read()
        except OSError as e:
            logger.error(f"Read failed: {e}")
            return False

        # original = original.replace("\r\n", "\n").replace("\r", "\n")
        result   = self._patch(original, before, after, hunk)

        if result is None:
            logger.error(f"All patch strategies failed for {hunk.file_path}")
            return False

        try:
            open(abs_path, "w", encoding="utf-8").write(result)
            logger.info(f"Applied root hunk: {hunk.file_path}")
            return True
        except OSError as e:
            logger.error(f"Write failed: {e}")
            return False

    @staticmethod
    def _patch(original: str, before: str, after: str, hunk) -> Optional[str]:
        """Three-level patch strategy: exact -> strip fuzzy -> line-range fallback."""
        # L1: Exact match
        if before and before in original:
            return original.replace(before, after, 1)

        # L2: Strip fuzzy match
        if before:
            orig_lines   = original.splitlines(keepends=True)
            before_lines = before.splitlines(keepends=True)
            after_lines  = after.splitlines(keepends=True)
            b_stripped   = [l.strip() for l in before_lines]
            n            = len(b_stripped)
            for i in range(len(orig_lines) - n + 1):
                if [l.strip() for l in orig_lines[i:i + n]] == b_stripped:
                    return "".join(orig_lines[:i] + after_lines + orig_lines[i + n:])

        # L3: Line-range fallback
        logger.warning(f"Falling back to line-range patch for {hunk.file_path}")
        orig_lines  = original.splitlines(keepends=True)
        after_lines = after.splitlines(keepends=True)
        s           = max(0, hunk.old_start_line - 1)
        e           = min(s + max(hunk.old_len, 1), len(orig_lines))
        if s >= len(orig_lines):
            return None
        if after_lines and not after_lines[-1].endswith("\n") and e < len(orig_lines):
            after_lines = list(after_lines)
            after_lines[-1] += "\n"
        return "".join(orig_lines[:s] + after_lines + orig_lines[e:])

    # ──────────────────────────────────────────────────────────────
    # Oracle context
    # ──────────────────────────────────────────────────────────────

    def _oracle_context(self, parsed_item: ParsedItem) -> dict:
        root_hunk      = parsed_item.root_hunk
        related        = []
        file_hunk_map: Dict[str, List[Hunk]] = defaultdict(list)

        for hunk in parsed_item.target_hunks:
            file_hunk_map[hunk.file_path].append(hunk)

        for rel_path, hunks in file_hunk_map.items():
            file_lines = self._file_lines(rel_path)
            if not file_lines:
                logger.warning(f"[Oracle] Cannot read: {rel_path}")
                continue
            total = len(file_lines)
            is_root_file = (rel_path == root_hunk.file_path)

            raw_windows = [
                {
                    "win_start":   max(1, (h.start_line if is_root_file else h.old_start_line) - self.context_window),
                    "win_end":     min(total, (h.end_line if is_root_file else h.old_start_line + max(h.old_len, 1) - 1) + self.context_window),
                    "hunk_start":  h.start_line if is_root_file else h.old_start_line,
                    "hunk_end":    h.end_line if is_root_file else h.old_start_line + max(h.old_len, 1) - 1,
                    "node_id":     h.id,
                    "usage_lines": list(range(h.start_line if is_root_file else h.old_start_line,
                                              (h.end_line if is_root_file else h.old_start_line + max(h.old_len, 1) - 1) + 1)),
                }
                for h in hunks
            ]

            for m in _merge_hunk_windows(raw_windows):
                related.append({
                    "role":            "ORACLE_GT",
                    "reason":          f"Target hunk(s) at lines {m['hunk_ranges']}",
                    "distance":        1,
                    "node_ids":        m["node_ids"],
                    "node_id":         m["node_ids"][0],
                    "file_path":       rel_path,
                    "rel_file_path":   rel_path,
                    "relevant_code":   _format_lines_with_lineno(
                        file_lines, m["win_start"], m["win_end"]
                    ),
                    "usage_line_nums": m["usage_lines"],
                })

        return {
            "focus_node": {
                "id":         root_hunk.id,
                "type":       "hunk",
                "file":       root_hunk.file_path,
                "start_line": root_hunk.start_line,
                "end_line":   root_hunk.end_line,
                "code":       parsed_item.root_before_code,
            },
            "related_contexts": related,
        }

    # ──────────────────────────────────────────────────────────────
    # Ground Truth build
    # ──────────────────────────────────────────────────────────────

    def _stage1_gt(self, parsed_item: ParsedItem, file_reason_map: Dict[str, str]) -> str:
        """impacted_locations line numbers must match snippet line numbers in the prompt."""
        file_lines_map: Dict[str, List[int]] = defaultdict(list)
        root_path = parsed_item.root_hunk.file_path

        for hunk_ in parsed_item.target_hunks:
            if hunk_.file_path == root_path:
                lines = range(hunk_.start_line, hunk_.end_line + 1)
            else:
                lines = range(hunk_.old_start_line, hunk_.old_start_line + max(hunk_.old_len, 1))
            file_lines_map[hunk_.file_path].extend(lines)

        locations = [
            {"file": fp, "lines": sorted(set(lns)), "reason": file_reason_map.get(fp, "")}
            for fp, lns in file_lines_map.items()
        ]
        return json.dumps({
            "reasoning": self._get_causal_reasoning(parsed_item),
            "impacted_locations": locations,
        }, ensure_ascii=False, indent=2)

    @staticmethod
    def _derive_hunk_reason(hunk: Hunk, use_old: bool = False) -> str:
        """Derive change reason from +/- direction of hunk content, replacing non-existent hunk.description.

        use_old=True means the hunk's file is the parent version, using old_start_line for positioning.
        """
        content = hunk.content or ""
        has_add = any(l.startswith("+") for l in content.splitlines())
        has_del = any(l.startswith("-") for l in content.splitlines())

        if use_old:
            s = hunk.old_start_line
            e = hunk.old_start_line + max(hunk.old_len, 1) - 1
        else:
            s = hunk.start_line
            e = hunk.end_line

        if has_add and has_del:
            return f"Modify lines {s}-{e} — replace changed lines"
        elif has_add:
            return f"Insert new code at lines {s}-{e}"
        elif has_del:
            return f"Delete code at lines {s}-{e}"
        else:
            return f"Update lines {s}-{e}"

    @staticmethod
    def _get_causal_reasoning(parsed_item: ParsedItem) -> str:
        """
        Construct Stage 1 reasoning.

        Do NOT use causal_analysis.reasoning — it references changes by internal indices like "Hunk 0/1/2",
        which the model cannot see in the prompt and would be confusing.

        Instead, use structured fields change_pattern + requirement_summary,
        combined with actual root change information.
        """
        ca = getattr(parsed_item.data_item, "causal_analysis", None)
        root_hunk = parsed_item.root_hunk
        n_files = len(set(h.file_path for h in parsed_item.target_hunks))
        n_hunks = len(parsed_item.target_hunks)

        if ca and ca.change_pattern and ca.requirement_summary:
            # Infer change type from root diff
            root_has_add = any(l.startswith("+") for l in (root_hunk.content or "").splitlines())
            root_has_del = any(l.startswith("-") for l in (root_hunk.content or "").splitlines())
            if root_has_add and root_has_del:
                action = "modifies"
            elif root_has_add:
                action = "adds to"
            elif root_has_del:
                action = "removes from"
            else:
                action = "changes"

            return (
                f"Root change ({ca.change_pattern}) {action} `{root_hunk.file_path}`: "
                f"{ca.requirement_summary} "
                f"This impacts {n_hunks} location(s) across {n_files} file(s) "
                f"that reference or depend on the changed code."
            )

        # fallback
        return (
            f"Root change in {parsed_item.root_hunk.file_path} "
            f"requires updates in {n_files} file(s)."
        )

    @staticmethod
    def _apply_hunk_to_snippet(hunk: Hunk, snippet: str, snippet_start_line: int = 1) -> Optional[str]:
        """
        Parse hunk.content (unified diff +/-) and apply to snippet.
        L1: Exact string match -> replace
        L3: Line-range fallback (compute offset relative to snippet_start_line)

        L2 strip fuzzy matching is no longer used (it would change indentation).
        """
        content = hunk.content
        before_lines: List[str] = []
        after_lines: List[str] = []

        for line in content.splitlines(keepends=True):
            if line.startswith("-"):
                before_lines.append(line[1:])
            elif line.startswith("+"):
                after_lines.append(line[1:])
            elif line.startswith("\\"):
                pass
            else:
                ctx = line[1:] if line.startswith(" ") else line
                before_lines.append(ctx)
                after_lines.append(ctx)

        before_str = "".join(before_lines)
        after_str = "".join(after_lines)

        if not before_str.strip():
            # Pure addition: no corresponding content in snippet, insert new lines
            if after_str.strip():
                snip_lines = snippet.splitlines(keepends=True)
                insert_at = hunk.old_start_line - snippet_start_line  # 0-based
                if insert_at < 0:
                    insert_at = len(snip_lines)
                return "".join(snip_lines[:insert_at] + after_lines + snip_lines[insert_at:])
            return snippet

        # L1: Exact match
        if before_str in snippet:
            return snippet.replace(before_str, after_str, 1)

        # L3: Line-range fallback
        snip_lines = snippet.splitlines(keepends=True)
        rel_start = hunk.old_start_line - snippet_start_line  # 0-based
        if rel_start < 0:
            rel_start = 0
        replace_len = max(hunk.old_len, 1)
        end_idx = min(rel_start + replace_len, len(snip_lines))

        if rel_start >= len(snip_lines):
            return None

        # Ensure consistent trailing newline
        if after_lines and not after_lines[-1].endswith("\n") and end_idx < len(snip_lines):
            after_lines = list(after_lines)
            after_lines[-1] += "\n"

        return "".join(snip_lines[:rel_start] + after_lines + snip_lines[end_idx:])

    def _stage2_gt(self, hunks: List[Hunk], current_version: str, snippet_start_line: int = 1,
                    use_old: bool = False) -> str:
        """
        Apply all overlapping hunks in the window sequentially to current_version, generating next_version.
        current_version has line number prefix ("   42: some code\n"), strip before processing.
        """
        # Strip line number prefixes
        raw_lines: List[str] = []
        for line in current_version.splitlines(keepends=True):
            m = re.match(r"^\s*\d+: ?(.*)$", line.rstrip("\n"))
            raw_lines.append((m.group(1) if m else line.rstrip("\n")) + "\n")
        raw_snippet = "".join(raw_lines)

        current = raw_snippet
        applied_summaries: List[str] = []
        any_failed = False

        for hunk in hunks:
            content = getattr(hunk, "content", None) or ""
            if not content.strip():
                continue

            next_raw = self._apply_hunk_to_snippet(
                hunk, current, snippet_start_line
            )

            if next_raw is None:
                logger.warning(
                    f"_stage2_gt: hunk {hunk.id} could not be applied "
                    f"(exact match + line-range fallback both failed)"
                )
                any_failed = True
            else:
                current = next_raw
                applied_summaries.append(self._derive_hunk_reason(hunk, use_old=use_old))

        if not applied_summaries:
            return json.dumps(
                {"next_version": None, "change_summary": "no change needed"},
                ensure_ascii=False, indent=2,
            )

        if current.strip() == raw_snippet.strip():
            return json.dumps(
                {"next_version": None, "change_summary": "no change needed"},
                ensure_ascii=False, indent=2,
            )

        return json.dumps({
            "next_version": current,
            "change_summary": "; ".join(applied_summaries),
        }, ensure_ascii=False, indent=2)

    def _apply_cot_if_needed(
            self,
            samples: List[dict],
            root_hunk_id: str,
            use_cot: Optional[bool],
    ) -> List[dict]:
        """
        Determine whether to execute CoT enrichment based on use_cot argument and instance-level cot_enricher.

        use_cot=None  -> Follow instance default (execute if cot_enricher exists)
        use_cot=True  -> Force execute (raise ValueError if cot_enricher is None)
        use_cot=False -> Force skip (even if cot_enricher exists)
        """
        if use_cot is True and self.cot_enricher is None:
            raise ValueError(
                "use_cot=True but cot_enricher is None. "
                "Please pass a CoTEnricher instance to the constructor."
            )

        should_cot = use_cot if use_cot is not None else bool(self.cot_enricher)

        if should_cot and self.cot_enricher and samples:
            logger.info(f"CoT enriching {len(samples)} samples for root={root_hunk_id}")
            samples = self.cot_enricher.enrich_batch(samples)
            enriched = sum(1 for s in samples if s.get("cot_enriched"))
            logger.info(f"CoT done: {enriched}/{len(samples)} for root={root_hunk_id}")
        else:
            logger.debug(f"CoT skipped for root={root_hunk_id} (use_cot={use_cot})")

        return samples

    # ──────────────────────────────────────────────────────────────
    # Main Entry
    # ──────────────────────────────────────────────────────────────

    def build(self, parsed_item: ParsedItem,
              use_cot: Optional[bool] = None, ) -> List[dict]:
        """Build all training samples for a single ParsedItem (with optional CoT enrichment)."""
        root_hunk    = parsed_item.root_hunk
        target_hunks = parsed_item.target_hunks

        if not target_hunks:
            logger.warning(f"No target hunks for root={root_hunk.id}")
            return []
        if not self._apply_root_hunk(parsed_item):
            logger.error(f"Skipping root={root_hunk.id}")
            return []

        oracle_ctx = self._oracle_context(parsed_item)
        if not oracle_ctx.get("related_contexts"):
            logger.warning(f"No oracle contexts for root={root_hunk.id}")
            return []

        samples = []

        # ── Build file->reason map (shared by _stage1_gt and stage2) ────────
        file_reason_map: Dict[str, str] = {}
        file_hunk_map: Dict[str, List[Hunk]] = defaultdict(list)
        for h in target_hunks:
            file_hunk_map[h.file_path].append(h)
            use_old = (h.file_path != root_hunk.file_path)
            file_reason_map.setdefault(h.file_path, self._derive_hunk_reason(h, use_old=use_old))

        # ── Stage1 ────────────────────────────────────────────────
        s1_gt  = self._stage1_gt(parsed_item, file_reason_map)
        s1_obj = json.loads(s1_gt)
        samples.append({
            "id":           f"{self.commit_hash}_{root_hunk.id}_stage1",
            "stage":        "stage1",
            "messages":     self.assembler.stage1(parsed_item, oracle_ctx),
            "ground_truth": s1_gt,
            "meta": {
                "commit_hash":     self.commit_hash,
                "root_hunk_id":    root_hunk.id,
                "root_file":       root_hunk.file_path,
                "target_hunk_ids": [h.id for h in target_hunks],
            },
        })

        # ── Stage2 ────────────────────────────────────────────────
        reasoning = s1_obj.get("reasoning", "")

        for order, (target_file, hunks) in enumerate(file_hunk_map.items()):
            # Target file comes from commit^ (old version), use old_start_line for positioning.
            # Root file has been updated to new version by _apply_root_hunk, use start_line.
            is_root_file = (target_file == root_hunk.file_path)
            if is_root_file:
                all_lines = sorted({
                    ln for h in hunks
                    for ln in range(h.start_line, h.end_line + 1)
                })
            else:
                all_lines = sorted({
                    ln for h in hunks
                    for ln in range(h.old_start_line, h.old_start_line + max(h.old_len, 1))
                })
            location_reason = file_reason_map.get(target_file, "")
            snippets = self._file_snippets(target_file, all_lines, self.snippet_window)
            if not snippets:
                logger.warning(f"Empty snippets for {target_file}")
                continue

            for idx, (cur_ver, s_line, e_line) in enumerate(snippets):
                if is_root_file:
                    overlap = [h for h in hunks if h.start_line <= e_line and h.end_line >= s_line]
                else:
                    overlap = [h for h in hunks if h.old_start_line <= e_line
                               and (h.old_start_line + max(h.old_len, 1) - 1) >= s_line]
                if not overlap:
                    continue
                primary = overlap[0]
                s2_gt   = self._stage2_gt(overlap, cur_ver, s_line, use_old=not is_root_file)
                samples.append({
                    "id": f"{self.commit_hash}_{root_hunk.id}_{primary.id}_w{idx}",
                    "stage":        "stage2",
                    "messages":     self.assembler.stage2(
                        parsed_item=parsed_item,
                        file_path=target_file,
                        current_version=cur_ver,
                        reasoning=reasoning,
                        impacted_lines=all_lines,
                        location_reason=location_reason,
                        snippet_start_line=s_line,
                    ),
                    "ground_truth": s2_gt,
                    "meta": {
                        "commit_hash":    self.commit_hash,
                        "root_hunk_id":   root_hunk.id,
                        "root_file":      root_hunk.file_path,
                        "target_hunk_id": primary.id,
                        "target_hunk_ids": [h.id for h in overlap],
                        "target_file":    target_file,
                        "window_start":   s_line,
                        "window_end":     e_line,
                        "snippet_idx":    idx,
                        "order":          order,
                    },
                })

        # ── CoT enrichment (optional) ─────────────────────────────────
        samples = self._apply_cot_if_needed(samples, root_hunk.id, use_cot)
        logger.info(f"Built {len(samples)} samples for root={root_hunk.id}")
        return samples
