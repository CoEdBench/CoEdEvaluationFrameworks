import os
import logging
from typing import List, Tuple

from src.domain.types import (
    DataItem, ParsedItem, Hunk, CausalAnalysis,
)

logger = logging.getLogger(__name__)

# Line count margin when reading context around root hunk
_CONTEXT_MARGIN = 10


# ══════════════════════════════════════════════════════════════════════════
# Path Utilities
# ══════════════════════════════════════════════════════════════════════════

def _normalize_path(raw_path: str) -> str:
    """
    Normalize dataset paths (may contain \\ or /) to consistent format.
    Strip leading separators to ensure relative paths.
    """
    # normalized = raw_path.replace("\\", "/").replace("/", os.sep)
    normalized = raw_path.replace("\\", "/")
    return normalized.lstrip(os.sep)


def _normalize_hunk_paths(hunk: Hunk) -> Hunk:
    """Return a new Hunk with corrected file_path"""
    return hunk.model_copy(update={
        "file_path": _normalize_path(hunk.file_path)
    })


# ══════════════════════════════════════════════════════════════════════════
# Diff Content Parsing
# ══════════════════════════════════════════════════════════════════════════

def _parse_diff_content(content: str) -> Tuple[str, str]:
    """
    Parse from hunk content field (diff fragment with +/- prefixes):
      - before_code: Code before modification (keeps context lines and - lines, removes + lines)
      - after_code:  Code after modification (keeps context lines and + lines, removes - lines)

    Rules:
      - Lines starting with '-': belong to before, not after (strip '-' prefix)
      - Lines starting with '+': belong to after, not before (strip '+' prefix)
      - Other lines (context): kept in both before and after
    """
    before_lines: List[str] = []
    after_lines: List[str] = []

    for line in content.split("\n"):
        if line.startswith("-"):
            before_lines.append(line[1:])
        elif line.startswith("+"):
            after_lines.append(line[1:])
        elif line.startswith("\\"):
            pass  # "\ No newline at end of file"
        else:
            # context line (starts with space or no prefix)
            ctx = line[1:] if line.startswith(" ") else line
            before_lines.append(ctx)
            after_lines.append(ctx)

    return "\n".join(before_lines), "\n".join(after_lines)


# ══════════════════════════════════════════════════════════════════════════
# Sandbox File Reading
# ══════════════════════════════════════════════════════════════════════════

def _read_file_lines(abs_path: str) -> List[str]:
    """Read all lines from file, return empty list on error."""
    try:
        with open(abs_path, "r", encoding="utf-8", errors="replace") as f:
            return f.readlines()
    except Exception as e:
        logger.error(f"❌ Failed to read file {abs_path}: {e}")
        return []


def _read_context_snippet(
    repo_root: str,
    rel_path: str,
    start_line: int,
    end_line: int,
    margin: int = _CONTEXT_MARGIN,
) -> str:
    """
    Read code from sandbox file in range [start_line - margin, end_line + margin],
    with line number prefix, format:
        "   163: code..."
    For Prompt display use.
    """
    abs_path = os.path.join(repo_root, rel_path)
    lines = _read_file_lines(abs_path)
    if not lines:
        return ""

    total = len(lines)
    # Convert to 0-based index
    slice_start = max(0, start_line - 1 - margin)
    slice_end   = min(total, end_line + margin)

    formatted = []
    for i in range(slice_start, slice_end):
        abs_line_no = i + 1
        content     = lines[i].rstrip("\n")
        formatted.append(f"   {abs_line_no}: {content}")

    logger.debug(
        f"📖 Snippet [{rel_path}] lines {slice_start + 1}~{slice_end} "
        f"(hunk: {start_line}~{end_line}, margin={margin})"
    )
    return "\n".join(formatted)


# ══════════════════════════════════════════════════════════════════════════
# DataItemParser
# ══════════════════════════════════════════════════════════════════════════

class DataItemParser:
    """
    Parse raw dict from JSONL dataset into ParsedItem.

    Responsibilities:
      1. Deserialize into DataItem (Pydantic validation)
      2. Normalize all Hunk file_paths (unify path separators)
      3. Extract root_hunk (determined by causal_analysis.root_hunk_id)
      4. Extract target_hunks (non-root, ordered by hunk_order causality)
      5. Parse root_before_code / root_after_code from hunk.content
      6. Read root_hunk context snippet from sandbox file (with line numbers, for Prompt use)
    """

    def parse(self, raw: dict, repo_root: str) -> ParsedItem:
        """
        :param raw:       Raw dict from json.loads on JSONL file
        :param repo_root: Sandbox repo root directory (absolute path, checked out to commit^)
        :return:          ParsedItem, ready for Pipeline
        :raises ValueError: when root_hunk_id not found in ordered_hunks
        """
        # ── Step 1: Deserialize ──────────────────────────────────────
        item = DataItem.model_validate(raw)
        for h in item.ordered_hunks:
            h.file_path = h.file_path.replace("\\","/")
            h.id = h.id.replace("\\","/")
        requirement = ""
        if item.causal_analysis is not None:
            requirement = item.causal_analysis.requirement_summary
        # ── Step 2: Normalize all Hunk paths ─────────────────────────
        item = item.model_copy(update={
            "ordered_hunks": [_normalize_hunk_paths(h) for h in item.ordered_hunks],
            "test_hunks":    [_normalize_hunk_paths(h) for h in item.test_hunks],
        })

        # ── Step 3: Extract root_hunk ────────────────────────────────
        root_hunk, root_idx = self._resolve_root_hunk(item)

        # ── Step 4: Extract target_hunks (ordered by hunk_order causality)───
        target_hunks = self._resolve_target_hunks(item, root_idx)

        # ── Step 5: Parse before/after code from content ────────────
        root_before_code, root_after_code = _parse_diff_content(root_hunk.content)

        # ── Step 6: Read context snippet with line numbers from sandbox ─────────────
        # Note: sandbox is at commit^ state (before modification), so reads before state
        root_context_snippet = _read_context_snippet(
            repo_root=repo_root,
            rel_path=root_hunk.file_path,
            start_line=root_hunk.start_line,
            end_line=root_hunk.end_line,
            margin=_CONTEXT_MARGIN,
        )

        if not root_context_snippet:
            logger.warning(
                f"⚠️ Could not read context snippet for root_hunk: "
                f"{root_hunk.file_path}:{root_hunk.start_line}"
            )

        logger.info(
            f"✅ Parsed item [{item.hash[:12]}] | "
            f"root={root_hunk.id} | "
            f"targets={len(target_hunks)}"
        )

        return ParsedItem(
            data_item=item,
            repo_root=repo_root,
            root_hunk=root_hunk,
            target_hunks=target_hunks,
            root_before_code=root_before_code,
            root_after_code=root_after_code,
            requirement = requirement,
        )

    # ──────────────────────────────────────────────────────────────
    # Private Methods
    # ──────────────────────────────────────────────────────────────

    def _resolve_root_hunk(self, item: DataItem) -> Tuple[Hunk, int]:
        """
        Find the Hunk corresponding to root_hunk_id in ordered_hunks.

        Matching strategy (priority high to low):
          1. Exact match on hunk.id == root_hunk_id
          2. Match after path normalization (compatible with \\ and / mixing)

        :returns: (root_hunk, root_idx) where root_idx is the index in ordered_hunks
        :raises ValueError: when not found
        """
        root_id = item.causal_analysis.root_hunk_id

        # Exact match
        for idx, hunk in enumerate(item.ordered_hunks):
            if hunk.id == root_id:
                return hunk, idx

        # Match after normalization (compatible with path separator differences)
        normalized_root_id = _normalize_path(root_id)
        for idx, hunk in enumerate(item.ordered_hunks):
            normalized_hunk_id = _normalize_path(hunk.id)
            if normalized_hunk_id == normalized_root_id:
                logger.debug(
                    f"🔍 root_hunk matched via normalized path: "
                    f"{hunk.id!r} ≈ {root_id!r}"
                )
                return hunk, idx

        raise ValueError(
            f"root_hunk_id {root_id!r} not found in ordered_hunks "
            f"(commit={item.hash[:12]}). "
            f"Available ids: {[h.id for h in item.ordered_hunks]}"
        )

    def _resolve_target_hunks(self, item: DataItem, root_idx: int) -> List[Hunk]:
        """
        Extract non-root ordered_hunks, sorted by causal_analysis.hunk_order.

        hunk_order is an index list of ordered_hunks, representing causal execution order.
        e.g. hunk_order=[1, 2, 0], root_idx=0
          -> ordered_target_indices = [1, 2]
          -> target_hunks = [ordered_hunks[1], ordered_hunks[2]]

        Fallback: if hunk_order is empty or indices out of bounds,
        sort by order_index field ascending.
        """
        hunk_order = item.causal_analysis.hunk_order
        n = len(item.ordered_hunks)

        # Filter out root from hunk_order, keep valid indices
        ordered_target_indices = [
            i for i in hunk_order
            if i != root_idx and 0 <= i < n
        ]

        if ordered_target_indices:
            target_hunks = [item.ordered_hunks[i] for i in ordered_target_indices]
        else:
            # Fallback: hunk_order missing or invalid, sort by order_index
            logger.warning(
                f"⚠️ [{item.hash[:12]}] hunk_order is empty or invalid, "
                f"falling back to order_index sort."
            )
            target_hunks = sorted(
                [h for i, h in enumerate(item.ordered_hunks) if i != root_idx],
                key=lambda h: h.order_index,
            )

        if not target_hunks:
            logger.info(f"ℹ️ [{item.hash[:12]}] No target hunks (single-hunk commit).")

        return target_hunks
