"""
Convert phase3 ordered_hunks JSONL to standard FillRequest JSONL.

Conversion rule (current experiment setting):
- Input = ordered_hunks[0] + requirement_summary (fallback issue_description).
- Ground truth = one-shot multi-hunk targets from ordered_hunks[1:].
- Other row fields are preserved in metadata as ground-truth side data.

Note:
- Some datasets keep requirement_summary under causal_analysis.requirement_summary.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
MULTI_HUNK_BLOCK_START = "<<<TARGET_HUNK>>>"
MULTI_HUNK_BLOCK_END = "<<<END_TARGET_HUNK>>>"


def _parse_repo_map(values: list[str] | None) -> dict[str, str]:
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
    if not path_value:
        return ""
    normalized = path_value.strip().replace("\\", "/")
    p = Path(normalized).expanduser()
    if p.is_absolute():
        p = p.resolve()
    else:
        p = (Path.cwd() / p).resolve()
    return p.as_posix()


def _normalize_rel_path(path_value: str | None) -> str:
    if not path_value:
        return ""
    return path_value.replace("\\", "/")


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _extract_requirement_summary(row: dict[str, Any]) -> str:
    # 1) top-level requirement_summary
    direct = _safe_text(row.get("requirement_summary"))
    if direct:
        return direct

    # 2) nested causal_analysis.requirement_summary
    causal_analysis = row.get("causal_analysis")
    if isinstance(causal_analysis, dict):
        nested = _safe_text(causal_analysis.get("requirement_summary"))
        if nested:
            return nested

    # 3) fallback fields
    issue = _safe_text(row.get("issue_description"))
    if issue:
        return issue
    return _safe_text(row.get("msg"))


def _is_accessible_directory(path_value: str) -> bool:
    return os.path.isdir(path_value) and os.access(path_value, os.R_OK | os.X_OK)


def _resolve_repo_path(
    row: dict[str, Any],
    repo_map: dict[str, str],
    default_repo_path: str | None,
) -> tuple[str, str]:
    repo_name = str(row.get("repo", "")).strip()
    mapped = _lookup_repo_map(repo_map, repo_name)
    if mapped:
        return mapped, "repo_map"

    explicit_repo_path = str(row.get("repo_path", "")).strip()
    if explicit_repo_path:
        return explicit_repo_path, "row_repo_path"

    if default_repo_path:
        return default_repo_path, "default_repo_path"

    return "", "unresolved"


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
    if suffix in {".go"}:
        return "go"
    if suffix in {".rs"}:
        return "rust"
    return "text"


def _extract_added_lines(content: str) -> list[str]:
    added_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
    return added_lines


def _build_ground_truth(added_lines: list[str]) -> str:
    if not added_lines:
        return ""
    text = "".join(added_lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _normalize_hunk_for_metadata(hunk: dict[str, Any], hunk_index: int) -> dict[str, Any]:
    file_path = _normalize_rel_path(str(hunk.get("file_path", "")).strip())
    start_line = hunk.get("start_line")
    end_line = hunk.get("end_line")
    content = hunk.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    return {
        "hunk_index": hunk_index,
        "id": hunk.get("id"),
        "order_index": hunk.get("order_index"),
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "content": content,
        "old_start_line": hunk.get("old_start_line"),
        "old_len": hunk.get("old_len"),
        "new_start_line": hunk.get("new_start_line"),
        "new_len": hunk.get("new_len"),
    }


def _build_multi_hunk_targets(
    ordered_hunks: list[Any],
    idx: int,
    empty_hunk_policy: str,
) -> list[dict[str, Any]]:
    if len(ordered_hunks) <= 1:
        if empty_hunk_policy == "skip":
            logger.warning("line %d: ordered_hunks has no prediction target (len<=1); skip", idx + 1)
            return []
        raise ValueError(f"line {idx + 1}: ordered_hunks has no prediction target (len<=1)")

    targets: list[dict[str, Any]] = []
    for target_index in range(1, len(ordered_hunks)):
        raw_hunk = ordered_hunks[target_index]
        if not isinstance(raw_hunk, dict):
            if empty_hunk_policy == "skip":
                logger.warning("line %d: ordered_hunks[%d] is not object; skip row", idx + 1, target_index)
                return []
            raise ValueError(f"line {idx + 1}: ordered_hunks[{target_index}] is not object")

        file_path = _normalize_rel_path(str(raw_hunk.get("file_path", "")).strip())
        if not file_path:
            if empty_hunk_policy == "skip":
                logger.warning(
                    "line %d: ordered_hunks[%d].file_path empty; skip row",
                    idx + 1,
                    target_index,
                )
                return []
            raise ValueError(f"line {idx + 1}: ordered_hunks[{target_index}].file_path empty")

        start_line = _safe_int(
            raw_hunk.get("start_line"),
            f"ordered_hunks[{target_index}].start_line",
            idx,
        )
        end_line = _safe_int(
            raw_hunk.get("end_line"),
            f"ordered_hunks[{target_index}].end_line",
            idx,
        )
        if start_line < 1 or end_line < start_line:
            if empty_hunk_policy == "skip":
                logger.warning(
                    "line %d: ordered_hunks[%d] invalid line range %s-%s; skip row",
                    idx + 1,
                    target_index,
                    start_line,
                    end_line,
                )
                return []
            raise ValueError(
                f"line {idx + 1}: ordered_hunks[{target_index}] invalid line range {start_line}-{end_line}"
            )

        content = raw_hunk.get("content") or ""
        if not isinstance(content, str):
            if empty_hunk_policy == "skip":
                logger.warning("line %d: ordered_hunks[%d].content is not string; skip row", idx + 1, target_index)
                return []
            raise ValueError(f"line {idx + 1}: ordered_hunks[{target_index}].content is not string")

        completion = _build_ground_truth(_extract_added_lines(content))
        targets.append(
            {
                "target_index": target_index,
                "id": raw_hunk.get("id"),
                "order_index": raw_hunk.get("order_index"),
                "file_path": file_path,
                "start_line": start_line,
                "end_line": end_line,
                "completion": completion,
            }
        )

    return targets


def _build_multi_hunk_ground_truth(targets: list[dict[str, Any]]) -> str:
    chunks: list[str] = []
    for target in targets:
        header = (
            f"{MULTI_HUNK_BLOCK_START} "
            f"index={target['target_index']} "
            f"file_path={target['file_path']} "
            f"start_line={target['start_line']} "
            f"end_line={target['end_line']}\n"
        )
        body = target["completion"]
        if body and not body.endswith("\n"):
            body += "\n"
        footer = f"{MULTI_HUNK_BLOCK_END}\n"
        chunks.append(f"{header}{body}{footer}")
    return "".join(chunks)


def _safe_int(value: Any, field_name: str, idx: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ValueError(f"line {idx + 1}: {field_name} is not int: {value!r}")


def _convert_one(
    row: dict[str, Any],
    idx: int,
    model_name: str,
    repo_map: dict[str, str],
    default_repo_path: str | None,
    repo_path_policy: str,
    context_lines: int,
    max_iterations: int,
    empty_hunk_policy: str,
) -> dict[str, Any] | None:
    ordered_hunks = row.get("ordered_hunks")
    if not isinstance(ordered_hunks, list) or not ordered_hunks:
        if empty_hunk_policy == "skip":
            logger.warning("line %d: missing/empty ordered_hunks; skip", idx + 1)
            return None
        raise ValueError(f"line {idx + 1}: missing/empty ordered_hunks")

    first_hunk = ordered_hunks[0]
    if not isinstance(first_hunk, dict):
        if empty_hunk_policy == "skip":
            logger.warning("line %d: ordered_hunks[0] is not object; skip", idx + 1)
            return None
        raise ValueError(f"line {idx + 1}: ordered_hunks[0] is not object")

    file_path = _normalize_rel_path(str(first_hunk.get("file_path", "")).strip())
    if not file_path:
        if empty_hunk_policy == "skip":
            logger.warning("line %d: ordered_hunks[0].file_path empty; skip", idx + 1)
            return None
        raise ValueError(f"line {idx + 1}: ordered_hunks[0].file_path empty")

    start_line = _safe_int(first_hunk.get("start_line"), "ordered_hunks[0].start_line", idx)
    end_line = _safe_int(first_hunk.get("end_line"), "ordered_hunks[0].end_line", idx)
    if start_line < 1 or end_line < start_line:
        if empty_hunk_policy == "skip":
            logger.warning("line %d: invalid line range %s-%s; skip", idx + 1, start_line, end_line)
            return None
        raise ValueError(f"line {idx + 1}: invalid line range {start_line}-{end_line}")

    content = first_hunk.get("content") or ""
    if not isinstance(content, str):
        if empty_hunk_policy == "skip":
            logger.warning("line %d: ordered_hunks[0].content is not string; skip", idx + 1)
            return None
        raise ValueError(f"line {idx + 1}: ordered_hunks[0].content is not string")

    known_input_hunk_added_lines = _extract_added_lines(content)
    known_input_hunk_added_text = _build_ground_truth(known_input_hunk_added_lines)
    target_hunks = _build_multi_hunk_targets(
        ordered_hunks=ordered_hunks,
        idx=idx,
        empty_hunk_policy=empty_hunk_policy,
    )
    if not target_hunks:
        return None

    repo_name = str(row.get("repo", "")).strip()
    commit_hash = str(row.get("hash", "")).strip()
    task_id = str(row.get("task_id", "")).strip() or f"{repo_name or 'repo'}_{commit_hash[:12] or 'nohash'}_{idx:05d}"

    raw_repo_path, repo_path_source = _resolve_repo_path(
        row=row,
        repo_map=repo_map,
        default_repo_path=default_repo_path,
    )
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

    issue_description = _safe_text(row.get("issue_description"))
    msg_text = _safe_text(row.get("msg"))
    requirement_summary = _extract_requirement_summary(row)
    source_diff = row.get("source_diff")
    test_hunks = row.get("test_hunks")

    metadata: dict[str, Any] = {
        "conversion_stage": "phase3_ordered_hunks_to_fill_request_multi_hunk",
        "task_mode": "phase3_multi_hunk_one_shot",
        "output_format": "multi_hunk_blocks_v1",
        "input_format": "phase3_ordered_hunks",
        "line_range_status": "from_ordered_hunks_first_as_input_anchor",
        "ground_truth_status": "from_ordered_hunks_remaining_added_lines",
        "repo": repo_name,
        "repo_name": repo_name,
        "hash": commit_hash,
        "msg": msg_text,
        "requirement_summary": requirement_summary,
        "issue_description": issue_description,
        "ordered_hunk_count": len(ordered_hunks),
        "ordered_hunk_input_index": 0,
        "ordered_hunk_id": first_hunk.get("id"),
        "ordered_hunk_order_index": first_hunk.get("order_index"),
        "known_input_hunk": _normalize_hunk_for_metadata(first_hunk, hunk_index=0),
        "known_input_hunk_added_lines": known_input_hunk_added_lines,
        "known_input_hunk_added_text": known_input_hunk_added_text,
        "target_hunk_count": len(target_hunks),
        "target_hunks": target_hunks,
        "source_diff": source_diff,
        "test_hunks": test_hunks,
        "ground_truth_all_ordered_hunks": ordered_hunks,
        "ground_truth_remaining_ordered_hunks": ordered_hunks[1:],
        "repo_path_source": repo_path_source,
        "repo_path_accessible": repo_path_accessible,
        # Prompt control: phase3 runs should use requirement summary context only.
        "hide_commit_context": True,
        # This ground-truth sequence is cross-hunk and won't match the masked anchor span.
        "span_check_policy": "off",
    }

    return {
        "task_id": task_id,
        "model_name": model_name,
        "repo_path": repo_path,
        "file_path": file_path,
        "start_line": start_line,
        "end_line": end_line,
        "language": _guess_language(file_path),
        "context_lines": context_lines,
        "max_iterations": max_iterations,
        "ground_truth": _build_multi_hunk_ground_truth(target_hunks),
        "metadata": metadata,
    }


def convert_phase3_ordered_hunks(
    input_path: str,
    output_path: str,
    model_name: str,
    repo_map: dict[str, str],
    default_repo_path: str | None,
    repo_path_policy: str,
    context_lines: int,
    max_iterations: int,
    empty_hunk_policy: str,
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
                row = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                logger.warning("Skip line %d: invalid JSON", idx + 1)
                continue

            if not isinstance(row, dict):
                skipped += 1
                logger.warning("Skip line %d: row is not JSON object", idx + 1)
                continue

            task = _convert_one(
                row=row,
                idx=idx,
                model_name=model_name,
                repo_map=repo_map,
                default_repo_path=default_repo_path,
                repo_path_policy=repo_path_policy,
                context_lines=context_lines,
                max_iterations=max_iterations,
                empty_hunk_policy=empty_hunk_policy,
            )
            if not task:
                skipped += 1
                continue

            fout.write(json.dumps(task, ensure_ascii=False) + "\n")
            converted += 1

    return converted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert phase3 ordered_hunks JSONL to FillRequest JSONL."
    )
    parser.add_argument("--input", required=True, help="Input phase3 JSONL path.")
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
        help="Map repo_name to local repo path. Example: --repo-map fastapi=/data/repos/fastapi",
    )
    parser.add_argument(
        "--repo-map-file",
        default=None,
        help="Path to JSON repo map file. CLI --repo-map entries override file entries.",
    )
    parser.add_argument(
        "--default-repo-path",
        default=None,
        help="Fallback repo path when repo is not in --repo-map.",
    )
    parser.add_argument(
        "--repo-path-policy",
        default="error",
        choices=["error", "skip"],
        help="When resolved repo_path is not accessible: error or skip.",
    )
    parser.add_argument(
        "--empty-hunk-policy",
        default="skip",
        choices=["skip", "error"],
        help="When input/target hunks are invalid: skip row or raise error.",
    )
    parser.add_argument("--context-lines", type=int, default=30)
    parser.add_argument("--max-iterations", type=int, default=8)
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
            "No repo map/default repo path provided. Conversion may fail if rows do not carry valid local repo_path."
        )

    converted, skipped = convert_phase3_ordered_hunks(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model_name,
        repo_map=repo_map,
        default_repo_path=args.default_repo_path,
        repo_path_policy=args.repo_path_policy,
        context_lines=args.context_lines,
        max_iterations=args.max_iterations,
        empty_hunk_policy=args.empty_hunk_policy,
    )
    logger.info("Done. converted=%d skipped=%d output=%s", converted, skipped, args.output)


if __name__ == "__main__":
    main()
