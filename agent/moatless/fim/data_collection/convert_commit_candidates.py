"""
Convert Phase-1 CommitCandidate JSONL to Phase-2 FillRequest JSONL.

This script covers:
- Task #1: schema conversion
- Task #2: extract start_line/end_line + ground_truth from unified diff
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
import re
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
HUNK_HEADER_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class ParsedHunk:
    old_start: int
    old_count: int
    new_start: int
    new_count: int
    lines: list[str]


@dataclass
class FillSpan:
    start_line: int
    end_line: int
    ground_truth_lines: list[str]
    hunk_index: int


def _parse_repo_map(values: list[str] | None) -> dict[str, str]:
    """
    Parse repeated --repo-map entries in the form:
      repo_name=/absolute/local/repo/path
    """
    mapping: dict[str, str] = {}
    if not values:
        return mapping

    for raw in values:
        if "=" not in raw:
            raise ValueError(f"Invalid --repo-map entry: {raw!r}. Expected format repo_name=/abs/path")
        repo_name, repo_path = raw.split("=", 1)
        repo_name = repo_name.strip()
        repo_path = repo_path.strip()
        if not repo_name or not repo_path:
            raise ValueError(f"Invalid --repo-map entry: {raw!r}. Empty repo_name or repo_path.")
        mapping[repo_name] = repo_path
    return mapping


def _parse_repo_map_file(repo_map_file: str | None) -> dict[str, str]:
    """
    Parse JSON repo map file.

    Supported formats:
    1) Object:
       {"scikit-learn": "/abs/path/scikit-learn"}
    2) List of objects:
       [{"repo_name": "scikit-learn", "repo_path": "/abs/path/scikit-learn"}]
    """
    if not repo_map_file:
        return {}
    with open(repo_map_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    mapping: dict[str, str] = {}
    if isinstance(data, dict):
        for repo_name, repo_path in data.items():
            mapping[str(repo_name)] = str(repo_path)
        return mapping

    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                raise ValueError("Invalid repo map list item: must be object with repo_name/repo_path")
            repo_name = str(item.get("repo_name", "")).strip()
            repo_path = str(item.get("repo_path", "")).strip()
            if not repo_name or not repo_path:
                raise ValueError("Invalid repo map list item: missing repo_name or repo_path")
            mapping[repo_name] = repo_path
        return mapping

    raise ValueError("Invalid repo map file format: expected JSON object or list of objects")


def _lookup_repo_map(repo_map: dict[str, str], repo_name: str) -> str | None:
    if repo_name in repo_map:
        return repo_map[repo_name]
    repo_name_lower = repo_name.lower()
    for key, value in repo_map.items():
        if key.lower() == repo_name_lower:
            return value
    return None


def _normalize_repo_path(path_value: str | None) -> str:
    """
    Normalize repo path to an absolute path string with forward slashes.
    """
    if not path_value:
        return ""
    normalized = path_value.strip().replace("\\", "/")
    p = Path(normalized).expanduser()
    if p.is_absolute():
        p = p.resolve()
    else:
        p = (Path.cwd() / p).resolve()
    return p.as_posix()


def _is_accessible_directory(path_value: str) -> bool:
    return os.path.isdir(path_value) and os.access(path_value, os.R_OK | os.X_OK)


def _normalize_rel_path(path_value: str | None) -> str:
    if not path_value:
        return ""
    return path_value.replace("\\", "/")


def _guess_language(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".java":
        return "java"
    if suffix in {".ts", ".tsx"}:
        return "typescript"
    if suffix in {".js", ".jsx"}:
        return "javascript"
    return "text"


def _parse_hunks(diff_text: str) -> list[ParsedHunk]:
    hunks: list[ParsedHunk] = []
    current: ParsedHunk | None = None

    for line in diff_text.splitlines():
        m = HUNK_HEADER_RE.match(line)
        if m:
            if current is not None:
                hunks.append(current)
            current = ParsedHunk(
                old_start=int(m.group(1)),
                old_count=int(m.group(2) or 1),
                new_start=int(m.group(3)),
                new_count=int(m.group(4) or 1),
                lines=[],
            )
            continue
        if current is not None:
            current.lines.append(line)

    if current is not None:
        hunks.append(current)
    return hunks


def _extract_fill_spans(hunks: list[ParsedHunk]) -> list[FillSpan]:
    spans: list[FillSpan] = []
    for hunk_index, hunk in enumerate(hunks):
        old_line = hunk.old_start
        new_line = hunk.new_start
        current_start: int | None = None
        current_end: int | None = None
        current_gt: list[str] = []

        def close_segment() -> None:
            nonlocal current_start, current_end, current_gt
            if current_start is None:
                return
            if current_gt:
                end_line = current_end if current_end is not None else current_start
                spans.append(
                    FillSpan(
                        start_line=current_start,
                        end_line=end_line,
                        ground_truth_lines=current_gt.copy(),
                        hunk_index=hunk_index,
                    )
                )
            current_start = None
            current_end = None
            current_gt = []

        for line in hunk.lines:
            if line.startswith("\\ No newline at end of file"):
                continue
            if line.startswith(" "):
                close_segment()
                old_line += 1
                new_line += 1
                continue
            if line.startswith("-") and not line.startswith("---"):
                if current_start is None:
                    current_start = new_line
                old_line += 1
                continue
            if line.startswith("+") and not line.startswith("+++"):
                if current_start is None:
                    current_start = new_line
                current_gt.append(line[1:])
                current_end = new_line
                new_line += 1
                continue
            close_segment()

        close_segment()

    return spans


def _choose_fill_span(spans: list[FillSpan], multi_hunk_policy: str, commit_hash: str) -> FillSpan | None:
    if not spans:
        return None
    if len(spans) == 1:
        return spans[0]
    if multi_hunk_policy == "first":
        logger.warning(
            "commit=%s has %d candidate spans; selecting first span due to --multi-hunk-policy=first",
            commit_hash[:12],
            len(spans),
        )
        return spans[0]
    if multi_hunk_policy == "skip":
        logger.warning("commit=%s has %d candidate spans; skip", commit_hash[:12], len(spans))
        return None
    raise ValueError(
        f"commit={commit_hash[:12]} has {len(spans)} candidate spans; "
        "set --multi-hunk-policy=first|skip to continue"
    )


def _build_ground_truth(lines: list[str]) -> str:
    if not lines:
        return ""
    return "\n".join(lines) + "\n"


def _extract_commit_message(candidate: dict[str, Any], commit_message_field: str) -> str | None:
    if commit_message_field == "none":
        return None
    value = candidate.get(commit_message_field)
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _pick_source_change(candidate: dict[str, Any]) -> dict[str, Any] | None:
    source_changes = candidate.get("source_changes") or []
    if not source_changes:
        return None

    # Single-point datasets should already have one source change.
    # We still guard for robustness.
    for change in source_changes:
        if not change.get("is_test", False):
            return change
    return source_changes[0]


def _resolve_repo_path(
    repo_name: str,
    candidate_repo_url: str | None,
    repo_map: dict[str, str],
    default_repo_path: str | None,
) -> tuple[str, str]:
    mapped = _lookup_repo_map(repo_map, repo_name)
    if mapped:
        return mapped, "repo_map"
    if default_repo_path:
        return default_repo_path, "default_repo_path"
    # fallback (may be a Windows path in raw data; downstream can remap later)
    return candidate_repo_url or "", "candidate_repo_url"


def _convert_one(
    candidate: dict[str, Any],
    idx: int,
    model_name: str,
    repo_map: dict[str, str],
    default_repo_path: str | None,
    repo_path_policy: str,
    context_lines: int,
    max_iterations: int,
    multi_hunk_policy: str,
    commit_message_field: str,
) -> dict[str, Any] | None:
    source_change = _pick_source_change(candidate)
    if not source_change:
        return None

    repo_name = str(candidate.get("repo_name", "")).strip()
    repo_url = candidate.get("repo_url")
    raw_repo_path, repo_path_source = _resolve_repo_path(repo_name, repo_url, repo_map, default_repo_path)
    repo_path = _normalize_repo_path(raw_repo_path)
    repo_path_accessible = _is_accessible_directory(repo_path)
    if not repo_path_accessible:
        msg = (
            f"repo={repo_name} resolved repo_path is not accessible: {repo_path!r} "
            f"(source={repo_path_source})"
        )
        if repo_path_policy == "skip":
            logger.warning("%s; skip", msg)
            return None
        raise ValueError(msg)

    raw_file_path = source_change.get("new_path") or source_change.get("old_path") or ""
    file_path = _normalize_rel_path(raw_file_path)
    diff_text = source_change.get("diff", "") or ""
    hunks = _parse_hunks(diff_text)
    hunk_count = len(hunks)
    spans = _extract_fill_spans(hunks)
    commit_hash = str(candidate.get("hash", "unknown"))
    chosen = _choose_fill_span(spans=spans, multi_hunk_policy=multi_hunk_policy, commit_hash=commit_hash)
    if chosen is None:
        return None
    ground_truth = _build_ground_truth(chosen.ground_truth_lines)
    commit_message = _extract_commit_message(candidate, commit_message_field)

    task_id = f"{repo_name or 'repo'}_{commit_hash[:12]}_{idx:05d}"

    metadata: dict[str, Any] = {
        "conversion_stage": "phase1_schema_and_line_extraction",
        "line_range_status": "derived_from_diff",
        "ground_truth_status": "derived_from_diff_added_lines",
        "repo_name": repo_name,
        "commit_hash": commit_hash,
        "commit_author_date": candidate.get("author_date"),
        "issue_ids": candidate.get("issue_ids", []),
        "source_diff": diff_text,
        "source_hunk_count": hunk_count,
        "source_span_count": len(spans),
        "selected_span_hunk_index": chosen.hunk_index,
        "multi_hunk_policy": multi_hunk_policy,
        "source_old_path": _normalize_rel_path(source_change.get("old_path")),
        "source_new_path": _normalize_rel_path(source_change.get("new_path")),
        "source_change_type": source_change.get("change_type"),
        "source_files_count": candidate.get("source_files_count"),
        "test_files_count": candidate.get("test_files_count"),
        "repo_path_source": repo_path_source,
        "repo_path_accessible": repo_path_accessible,
        "commit_message_source_field": commit_message_field,
        "commit_message_present": commit_message is not None,
    }
    if commit_message is not None:
        metadata["commit_message"] = commit_message

    return {
        "task_id": task_id,
        "model_name": model_name,
        "repo_path": repo_path,
        "file_path": file_path,
        "start_line": chosen.start_line,
        "end_line": chosen.end_line,
        "language": _guess_language(file_path),
        "context_lines": context_lines,
        "max_iterations": max_iterations,
        "ground_truth": ground_truth,
        "metadata": metadata,
    }


def convert_commit_candidates(
    input_path: str,
    output_path: str,
    model_name: str,
    repo_map: dict[str, str],
    default_repo_path: str | None,
    repo_path_policy: str,
    context_lines: int,
    max_iterations: int,
    multi_hunk_policy: str,
    commit_message_field: str,
) -> tuple[int, int]:
    converted = 0
    skipped = 0

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(input_path, "r", encoding="utf-8") as fin, open(output_path, "w", encoding="utf-8") as fout:
        for idx, raw in enumerate(fin):
            line = raw.strip()
            if not line:
                continue
            try:
                candidate = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                logger.warning("Skip line %d: invalid JSON", idx + 1)
                continue

            task = _convert_one(
                candidate=candidate,
                idx=idx,
                model_name=model_name,
                repo_map=repo_map,
                default_repo_path=default_repo_path,
                repo_path_policy=repo_path_policy,
                context_lines=context_lines,
                max_iterations=max_iterations,
                multi_hunk_policy=multi_hunk_policy,
                commit_message_field=commit_message_field,
            )
            if not task:
                skipped += 1
                continue

            fout.write(json.dumps(task, ensure_ascii=False) + "\n")
            converted += 1

            if task["start_line"] > task["end_line"] or not task["ground_truth"]:
                logger.warning(
                    "task=%s has suspicious range/ground_truth: start=%s end=%s gt_len=%d",
                    task["task_id"],
                    task["start_line"],
                    task["end_line"],
                    len(task["ground_truth"]),
                )

    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert CommitCandidate JSONL to FillRequest JSONL with line-range extraction."
    )
    parser.add_argument("--input", required=True, help="Input CommitCandidate JSONL path.")
    parser.add_argument("--output", required=True, help="Output FillRequest JSONL path.")
    parser.add_argument(
        "--model-name",
        default="placeholder-model",
        help="Model name to write into FillRequest.model_name.",
    )
    parser.add_argument(
        "--repo-map",
        action="append",
        default=[],
        help="Map repo_name to local repo path. Example: --repo-map scikit-learn=/data/repos/scikit-learn",
    )
    parser.add_argument(
        "--repo-map-file",
        default=None,
        help="Path to JSON repo map file. CLI --repo-map entries override file entries.",
    )
    parser.add_argument(
        "--default-repo-path",
        default=None,
        help="Fallback repo path when repo_name is not in --repo-map.",
    )
    parser.add_argument(
        "--repo-path-policy",
        default="error",
        choices=["error", "skip"],
        help=(
            "What to do when resolved repo_path is not a readable local directory. "
            "'error' stops conversion, 'skip' skips that record."
        ),
    )
    parser.add_argument("--context-lines", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=8)
    parser.add_argument(
        "--multi-hunk-policy",
        default="skip",
        choices=["skip", "first", "error"],
        help=(
            "Policy when multiple candidate fill spans are found in one commit. "
            "'skip' drops the record, 'first' keeps the first span, 'error' raises."
        ),
    )
    parser.add_argument(
        "--commit-message-field",
        default="msg",
        choices=["msg", "re_msg", "none"],
        help=(
            "Which input field to copy into metadata.commit_message. "
            "'re_msg' uses rewritten message; 'none' omits commit message."
        ),
    )
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"Input file not found: {args.input}")

    repo_map_from_file = _parse_repo_map_file(args.repo_map_file)
    repo_map_from_cli = _parse_repo_map(args.repo_map)
    repo_map = {**repo_map_from_file, **repo_map_from_cli}
    if not repo_map and not args.default_repo_path:
        logger.warning(
            "No repo map/default repo path provided. "
            "Will fall back to candidate repo_url (often non-local paths)."
        )

    converted, skipped = convert_commit_candidates(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        repo_map=repo_map,
        default_repo_path=args.default_repo_path,
        repo_path_policy=args.repo_path_policy,
        context_lines=args.context_lines,
        max_iterations=args.max_iterations,
        multi_hunk_policy=args.multi_hunk_policy,
        commit_message_field=args.commit_message_field,
    )
    logger.info("Done. converted=%d skipped=%d output=%s", converted, skipped, args.output)


if __name__ == "__main__":
    main()
