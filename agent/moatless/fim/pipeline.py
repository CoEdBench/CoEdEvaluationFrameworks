"""
fim/pipeline.py
Core execution logic: run_fill_task / run_fill_batch
"""
import asyncio
import copy
import logging
import os.path
import re
import shutil
import subprocess
import tempfile
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from moatless.message_history.message_history import MessageHistoryGenerator

from moatless.actions.finish import Finish, FinishArgs
from moatless.actions.glob import GlobTool
from moatless.actions.grep_tool import GrepTool
from moatless.actions.list_files import ListFiles
from moatless.actions.read_file import ReadFile
from moatless.actions.string_replace import StringReplace
from moatless.actions.think import Think
from moatless.actions.view_code import ViewCode
from moatless.agent.agent import ActionAgent
from moatless.completion.base import BaseCompletionModel
from moatless.environment.local import LocalBashEnvironment
from moatless.flow.loop import AgenticLoop
from moatless.repository.file import FileRepository
from moatless.workspace import Workspace

from moatless.fim.schema import FillRequest, FillResult
from moatless.fim.prompt import build_system_prompt, build_user_prompt
from moatless.fim.utils import replace_line_range, persist_trace

logger = logging.getLogger(__name__)
MULTI_HUNK_BLOCK_START = "<<<TARGET_HUNK>>>"
MULTI_HUNK_BLOCK_END = "<<<END_TARGET_HUNK>>>"
MULTI_HUNK_HEADER_RE = re.compile(
    r"^index=(\d+)\s+file_path=([^\s]+)\s+start_line=(\d+)\s+end_line=(\d+)$"
)
FORCED_FINISH_PROMPT_SNIPPET = "MUST now call Finish immediately"
FORCED_FINISH_USER_MESSAGE = (
    "You have used all available exploration steps. "
    "Based on everything you have read, you MUST now call Finish immediately. "
    "Call the Finish tool with a finish_reason explaining what you did."
)


def _get_forced_finish_max_attempts() -> int:
    raw = str(os.getenv("FIM_FORCED_FINISH_MAX_ATTEMPTS", "2")).strip()
    try:
        value = int(raw)
    except ValueError:
        return 2
    return max(1, min(value, 3))


def _build_forced_finish_user_message(
    *,
    task: FillRequest,
    attempt: int,
    retry_reason: Optional[str],
) -> str:
    if attempt <= 1:
        return FORCED_FINISH_USER_MESSAGE
    retry_note = (
        "Previous attempt failed because Finish was not produced. "
        if retry_reason == "no_finish_action"
        else "Previous attempt failed due to an error. "
    )
    return f"{FORCED_FINISH_USER_MESSAGE}\n\n{retry_note}You MUST call the Finish tool now — no other action is accepted."


def _get_forced_submit_max_attempts() -> int:
    raw = str(os.getenv("FIM_FORCED_SUBMIT_MAX_ATTEMPTS", "2")).strip()
    try:
        value = int(raw)
    except ValueError:
        return 2
    return max(1, min(value, 3))



RESULT_METADATA_TEMPLATE: dict[str, Any] = {
    "repo_checkout_mode": None,
    "repo_checkout_commit": None,
    "repo_checkout_status": None,
    "resolved_repo_path": None,
    "repo_checkout_temp_dir": None,
    "preflight_status": None,
    "preflight_stage": None,
    "mask_status": None,
    "mask_range": None,
    "mask_marker": None,
    "mask_line_count": None,
    "span_consistency": None,
    "span_check_policy": None,
    "total_prompt_tokens": 0,
    "total_completion_tokens": 0,
    "total_reasoning_tokens": 0,
    "total_cache_read_tokens": 0,
    "total_cost_usd": 0.0,
    "task_max_iterations": None,
    "flow_max_iterations": None,
    "submission_mode": None,
    "first_submit_node_id": None,
    "forced_submit_node_id": None,
    "error_type": None,
    # Reserved for future observability fields (P2-9).
    "inference_duration_sec": None,
    "tool_call_distribution": None,
    "tool_call_count": None,
}


def _build_result_metadata(
    base_metadata: Optional[dict[str, Any]] = None,
    **updates: Any,
) -> dict[str, Any]:
    metadata = dict(RESULT_METADATA_TEMPLATE)
    if isinstance(base_metadata, dict):
        metadata.update(base_metadata)
    metadata.update(updates)
    return metadata


def _extract_commit_hash(task: FillRequest) -> Optional[str]:
    metadata = task.metadata or {}
    for key in ("commit_hash", "hash", "target_commit"):
        value = metadata.get(key)
        if value:
            return str(value)
    return None


def _is_forced_finish_prompt(user_message: Optional[str]) -> bool:
    if not user_message:
        return False
    return FORCED_FINISH_PROMPT_SNIPPET in user_message


def _collect_finish_and_forced_node_ids(all_nodes: list[Any]) -> tuple[list[int], list[int]]:
    finish_node_ids: list[int] = []
    forced_prompt_node_ids: list[int] = []

    for node in all_nodes:
        node_id = getattr(node, "node_id", None)
        if not isinstance(node_id, int):
            continue
        if _is_forced_finish_prompt(getattr(node, "user_message", None)):
            forced_prompt_node_ids.append(node_id)

        for step in getattr(node, "action_steps", []) or []:
            action = getattr(step, "action", None)
            if isinstance(action, FinishArgs):
                finish_node_ids.append(node_id)

    return finish_node_ids, forced_prompt_node_ids


def _classify_finish_mode(
    *,
    has_finish: bool,
    has_error: bool,
    first_finish_node_id: Optional[int],
    forced_finish_node_id: Optional[int],
    task_max_iterations: int,
) -> str:
    if not has_finish:
        return "error_no_finish" if has_error else "no_finish"

    if forced_finish_node_id is not None:
        return "forced_finish"

    if first_finish_node_id is None:
        return "natural_finish"

    if first_finish_node_id < task_max_iterations:
        return "natural_early_finish"

    if first_finish_node_id == task_max_iterations:
        return "natural_deadline_finish"

    return "natural_late_finish"


def _classify_error_type(error: Optional[str]) -> Optional[str]:
    if not error:
        return None

    text = error.strip()
    if not text:
        return None

    if "No tool calls found in response" in text:
        return "protocol_no_tool_call"

    if "Completion validation error" in text:
        return "completion_validation_error"

    if "failed to checkout commit" in text:
        return "repo_checkout_error"

    if "mask injection" in text:
        return "mask_injection_error"

    if "known-input hunk seed" in text:
        return "known_input_seed_error"

    return text.splitlines()[0][:200]


def _extract_finish_args(flow) -> Optional[FinishArgs]:
    for node in reversed(flow.root.get_all_nodes()):
        for step in reversed(node.action_steps or []):
            if isinstance(step.action, FinishArgs):
                return step.action
            action_obj = getattr(step, "action", None)
            if isinstance(action_obj, Finish):
                for attr in ("args", "action_args", "arguments", "input"):
                    val = getattr(action_obj, attr, None)
                    if isinstance(val, FinishArgs):
                        return val
    return None


def _extract_line_range_content(file_content: str, start_line: int, end_line: int) -> str:
    lines = file_content.splitlines(keepends=True)
    if start_line < 1 or end_line < start_line:
        return ""
    start_idx = start_line - 1
    end_idx = min(end_line, len(lines))
    if start_idx >= len(lines):
        return ""
    return "".join(lines[start_idx:end_idx])


def _run_git(
    repo_path: str,
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", repo_path, *args],
        check=check,
        capture_output=True,
        text=True,
    )


def _capture_baseline_tree(repo_path: str) -> Optional[str]:
    try:
        _run_git(repo_path, ["add", "-A"])
        tree = _run_git(repo_path, ["write-tree"]).stdout.strip()
        _run_git(repo_path, ["reset", "--mixed", "-q"])
        return tree or None
    except Exception as exc:
        logger.warning("Failed to capture baseline tree for %s: %s", repo_path, exc)
        return None


def _extract_patch_against_baseline(repo_path: str, baseline_tree: Optional[str]) -> tuple[str, list[str]]:
    if not baseline_tree:
        try:
            patch = _run_git(repo_path, ["diff", "--no-color"]).stdout
            changed_files = [
                _normalize_rel_file_path(line)
                for line in _run_git(repo_path, ["diff", "--name-only"]).stdout.splitlines()
                if line.strip()
            ]
            return patch, changed_files
        except Exception as exc:
            logger.warning("Failed to extract default git diff for %s: %s", repo_path, exc)
            return "", []

    try:
        patch = _run_git(
            repo_path,
            ["diff", "--no-color", "--find-renames", "--unified=3", baseline_tree],
        ).stdout
        changed_files = [
            _normalize_rel_file_path(line)
            for line in _run_git(repo_path, ["diff", "--name-only", baseline_tree]).stdout.splitlines()
            if line.strip()
        ]
        return patch, changed_files
    except Exception as exc:
        logger.warning("Failed to extract git diff against baseline tree for %s: %s", repo_path, exc)
        return "", []


def _read_file_text_if_exists(repo_root: Path, rel_path: str) -> str:
    full_path = repo_root / rel_path
    if not full_path.exists():
        return ""
    try:
        return full_path.read_text(encoding="utf-8")
    except Exception:
        return ""


def _read_tree_file_content(repo_path: str, tree_hash: Optional[str], rel_path: str) -> str:
    if not tree_hash:
        return ""
    spec = f"{tree_hash}:{rel_path}"
    try:
        return _run_git(repo_path, ["show", spec]).stdout
    except Exception:
        return ""


def _collect_changed_file_snapshots(
    *,
    repo_path: str,
    changed_files: list[str],
    baseline_tree: Optional[str],
) -> tuple[dict[str, str], dict[str, str]]:
    repo_root = Path(repo_path)
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    for rel_path in changed_files:
        rel_norm = _normalize_rel_file_path(rel_path)
        before[rel_norm] = _read_tree_file_content(repo_path, baseline_tree, rel_norm)
        after[rel_norm] = _read_file_text_if_exists(repo_root, rel_norm)
    return before, after


def _safe_task_name(task_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]", "_", task_id)


def _prepare_isolated_repo(repo_path: str, commit_hash: str, task_id: str, temp_root: Optional[str] = None) -> str:
    """
    Create an isolated temporary clone and checkout to commit_hash (or any git revspec).
    """
    source_repo = str(Path(repo_path).resolve())
    if not os.path.isdir(source_repo):
        raise ValueError(f"repo_path does not exist or is not a directory: {source_repo}")

    try:
        subprocess.run(
            ["git", "-C", source_repo, "rev-parse", "--is-inside-work-tree"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise ValueError(f"repo_path is not a git repository: {source_repo}") from exc

    temp_root_dir = temp_root or os.getenv("FIM_TEMP_REPO_ROOT")
    if temp_root_dir:
        Path(temp_root_dir).mkdir(parents=True, exist_ok=True)

    temp_repo_dir = tempfile.mkdtemp(prefix=f"fim_{_safe_task_name(task_id)}_", dir=temp_root_dir)
    try:
        subprocess.run(
            ["git", "clone", "--quiet", "--no-hardlinks", source_repo, temp_repo_dir],
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "-C", temp_repo_dir, "checkout", "--quiet", commit_hash],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        stderr = (exc.stderr or "").strip()
        stdout = (exc.stdout or "").strip()
        details = stderr or stdout or str(exc)
        raise ValueError(
            f"failed to checkout commit {commit_hash!r} in isolated repo for task {task_id}: {details}"
        ) from exc

    return temp_repo_dir


def _build_preflight_error_result(
    task: FillRequest,
    trace_root: str,
    finish_reason: str,
    error: str,
    commit_hash: Optional[str],
    extra_metadata: Optional[dict[str, Any]] = None,
) -> FillResult:
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    trace_dir = os.path.join(trace_root, f"{task.task_id}_{timestamp}")
    error_type = _classify_error_type(error)
    metadata = _build_result_metadata(
        base_metadata=task.metadata if isinstance(task.metadata, dict) else None,
        repo_checkout_mode="temp_clone_checkout" if commit_hash else "direct_repo_path",
        repo_checkout_commit=commit_hash,
        repo_checkout_status="failed" if finish_reason == "repo_checkout_error" else "unknown",
        resolved_repo_path=task.repo_path,
        preflight_status="failed",
        preflight_stage=finish_reason,
        submission_mode="error_no_finish",
        error_type=error_type,
    )
    if extra_metadata:
        metadata.update(extra_metadata)
    return FillResult(
        task_id=task.task_id,
        model_name=task.model_name,
        file_path=task.file_path,
        trace_dir=trace_dir,
        start_line=task.start_line,
        end_line=task.end_line,
        completion="",
        updated_file_content="",
        success=False,
        finish_reason=finish_reason,
        error=error,
        confidence=0.0,
        reasoning="",
        action_steps=0,
        metadata=metadata,
    )


def _normalize_code_for_check(code: str) -> str:
    return "\n".join(line.rstrip() for line in code.strip().splitlines())


def _comment_prefix_for_file(file_path: str) -> str:
    suffix = Path(file_path).suffix.lower()
    if suffix in {".py", ".rb", ".sh", ".yaml", ".yml", ".toml", ".ini", ".cfg"}:
        return "#"
    if suffix in {".sql"}:
        return "--"
    return "//"


def _normalize_rel_file_path(file_path: str) -> str:
    return str(file_path or "").strip().replace("\\", "/")


def _is_multi_hunk_one_shot_task(task: FillRequest) -> bool:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    task_mode = str(metadata.get("task_mode") or "").strip().lower()
    return task_mode == "phase3_multi_hunk_one_shot"


def _extract_target_hunks(task: FillRequest) -> list[dict[str, Any]]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    raw = metadata.get("target_hunks")
    if not isinstance(raw, list):
        return []
    targets: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            targets.append(item)
    return targets


def _extract_known_input_hunk(task: FillRequest) -> dict[str, Any]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    hunk = metadata.get("known_input_hunk")
    if isinstance(hunk, dict):
        return hunk
    return {}


def _extract_known_input_hunk_added_lines(task: FillRequest, known_hunk: dict[str, Any]) -> list[str]:
    metadata = task.metadata if isinstance(task.metadata, dict) else {}
    raw_lines = metadata.get("known_input_hunk_added_lines")
    if isinstance(raw_lines, list):
        return [str(line) for line in raw_lines]

    added_text = metadata.get("known_input_hunk_added_text")
    if isinstance(added_text, str) and added_text:
        return added_text.splitlines(keepends=True)

    content = known_hunk.get("content")
    if not isinstance(content, str) or not content:
        return []

    lines: list[str] = []
    for line in content.splitlines(keepends=True):
        if line.startswith("+++") or line.startswith("@@"):
            continue
        if line.startswith("+"):
            lines.append(line[1:])
    return lines


def _resolve_checkout_ref(task: FillRequest, commit_hash: Optional[str]) -> Optional[str]:
    if not commit_hash:
        return None
    if _is_multi_hunk_one_shot_task(task):
        return f"{commit_hash}^"
    return commit_hash


def _seed_known_input_hunk(task: FillRequest) -> dict[str, Any]:
    known_hunk = _extract_known_input_hunk(task)
    seed_file_path = _normalize_rel_file_path(str(known_hunk.get("file_path") or task.file_path))
    if not seed_file_path:
        raise ValueError("known-input hunk seed failed: missing file_path")

    raw_old_start_line = known_hunk.get("old_start_line")
    raw_old_len = known_hunk.get("old_len")
    try:
        old_start_line = int(raw_old_start_line)
        old_len = int(raw_old_len)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"known-input hunk seed failed: invalid old range ({raw_old_start_line!r}, {raw_old_len!r})"
        ) from exc

    if old_start_line < 1:
        raise ValueError(f"known-input hunk seed failed: old_start_line must be >=1, got {old_start_line}")
    if old_len < 0:
        raise ValueError(f"known-input hunk seed failed: old_len must be >=0, got {old_len}")

    replacement_lines = _extract_known_input_hunk_added_lines(task, known_hunk)

    full_path = Path(task.repo_path) / seed_file_path
    if not full_path.exists():
        raise ValueError(f"known-input hunk seed failed: target file not found: {full_path}")

    original_content = full_path.read_text(encoding="utf-8")
    lines = original_content.splitlines(keepends=True)
    start_idx = old_start_line - 1
    end_idx = start_idx + old_len
    if start_idx > len(lines):
        raise ValueError(
            f"known-input hunk seed failed: start index out of range old_start_line={old_start_line}, "
            f"file_lines={len(lines)} file={full_path}"
        )
    if end_idx > len(lines):
        raise ValueError(
            f"known-input hunk seed failed: old range exceeds file old_start_line={old_start_line}, "
            f"old_len={old_len}, file_lines={len(lines)} file={full_path}"
        )

    seeded_lines = lines[:start_idx] + replacement_lines + lines[end_idx:]
    full_path.write_text("".join(seeded_lines), encoding="utf-8")

    return {
        "file_full_path": str(full_path),
        "original_file_content": original_content,
        "seed_file_path": seed_file_path,
        "seed_old_start_line": old_start_line,
        "seed_old_len": old_len,
        "seed_new_line_count": len(replacement_lines),
        "seed_status": "applied",
    }


def _parse_multi_hunk_completion_blocks(completion: str) -> list[dict[str, Any]]:
    text = completion or ""
    blocks: list[dict[str, Any]] = []
    cursor = 0

    while True:
        start = text.find(MULTI_HUNK_BLOCK_START, cursor)
        if start == -1:
            break

        header_end = text.find("\n", start)
        if header_end == -1:
            break

        header_text = text[start + len(MULTI_HUNK_BLOCK_START) : header_end].strip()
        end = text.find(MULTI_HUNK_BLOCK_END, header_end + 1)
        if end == -1:
            body = text[header_end + 1 :]
            cursor = len(text)
        else:
            body = text[header_end + 1 : end]
            cursor = end + len(MULTI_HUNK_BLOCK_END)
            if cursor < len(text) and text[cursor] == "\n":
                cursor += 1

        block: dict[str, Any] = {
            "raw_header": header_text,
            "body": body,
            "header_valid": False,
        }
        m = MULTI_HUNK_HEADER_RE.match(header_text)
        if m:
            block.update(
                {
                    "header_valid": True,
                    "index": int(m.group(1)),
                    "file_path": _normalize_rel_file_path(m.group(2)),
                    "start_line": int(m.group(3)),
                    "end_line": int(m.group(4)),
                }
            )
        blocks.append(block)

    return blocks


def _has_valid_multi_hunk_blocks(completion: str) -> bool:
    parsed_blocks = _parse_multi_hunk_completion_blocks(completion)
    return any(bool(block.get("header_valid")) for block in parsed_blocks)


def _apply_multi_hunk_completion(
    *,
    task: FillRequest,
    completion: str,
    mask_info: Optional[dict[str, Any]],
    masked_anchor_content: str,
) -> tuple[str, dict[str, Any]]:
    repo_root = Path(task.repo_path)
    anchor_file_path = _normalize_rel_file_path(task.file_path)
    target_hunks = _extract_target_hunks(task)
    parsed_blocks = _parse_multi_hunk_completion_blocks(completion)
    valid_blocks = [b for b in parsed_blocks if b.get("header_valid")]

    # Deduplicate by predicted index and by predicted location.
    blocks_by_index: dict[int, dict[str, Any]] = {}
    blocks_by_key: dict[tuple[str, int, int], dict[str, Any]] = {}
    duplicate_indices: list[int] = []
    duplicate_keys: list[tuple[str, int, int]] = []
    predicted_keys_by_index: dict[int, list[tuple[str, int, int]]] = {}
    for block in valid_blocks:
        idx = int(block["index"])
        key = (
            _normalize_rel_file_path(str(block.get("file_path") or "")),
            int(block.get("start_line", -1)),
            int(block.get("end_line", -1)),
        )
        if idx in blocks_by_index:
            duplicate_indices.append(idx)
        else:
            blocks_by_index[idx] = block
        if key in blocks_by_key:
            duplicate_keys.append(key)
        else:
            blocks_by_key[key] = block
        predicted_keys_by_index.setdefault(idx, []).append(key)

    target_indices: set[int] = set()
    target_keys: set[tuple[str, int, int]] = set()
    grouped_targets: dict[str, list[dict[str, Any]]] = {}
    invalid_target_indices: list[Any] = []
    for target in target_hunks:
        try:
            idx = int(target.get("target_index"))
            start_line = int(target.get("start_line"))
            end_line = int(target.get("end_line"))
        except (TypeError, ValueError):
            invalid_target_indices.append(target.get("target_index"))
            continue

        target_path = _normalize_rel_file_path(str(target.get("file_path") or ""))
        if not target_path:
            invalid_target_indices.append(target.get("target_index"))
            continue

        target_indices.add(idx)
        target_key = (target_path, start_line, end_line)
        target_keys.add(target_key)
        grouped_targets.setdefault(target_path, []).append(
            {
                "target_index": idx,
                "file_path": target_path,
                "start_line": start_line,
                "end_line": end_line,
                "target_key": target_key,
            }
        )

    extra_predicted_indices = sorted(
        {
            int(block["index"])
            for block in valid_blocks
            if (
                _normalize_rel_file_path(str(block.get("file_path") or "")),
                int(block.get("start_line", -1)),
                int(block.get("end_line", -1)),
            )
            not in target_keys
        }
    )
    missing_indices: list[int] = []
    header_mismatch_indices: list[int] = []
    applied_indices: list[int] = []
    unreadable_target_files: list[str] = []

    # Build baseline file content map. For anchor file, use pre-mask original content if available.
    file_contents: dict[str, str] = {}
    for target_path in grouped_targets.keys():
        if (
            target_path == anchor_file_path
            and mask_info
            and isinstance(mask_info.get("original_file_content"), str)
        ):
            file_contents[target_path] = str(mask_info["original_file_content"])
            continue

        full_path = repo_root / target_path
        if not full_path.exists():
            unreadable_target_files.append(target_path)
            continue
        try:
            file_contents[target_path] = full_path.read_text(encoding="utf-8")
        except Exception:
            unreadable_target_files.append(target_path)

    # Apply replacements per file from bottom to top to avoid line-shift issues.
    for target_path, targets in grouped_targets.items():
        if target_path not in file_contents:
            for target in targets:
                missing_indices.append(int(target["target_index"]))
            continue

        content = file_contents[target_path]
        sorted_targets = sorted(targets, key=lambda x: (int(x["start_line"]), int(x["end_line"])), reverse=True)
        for target in sorted_targets:
            idx = int(target["target_index"])
            target_key = (
                str(target["file_path"]),
                int(target["start_line"]),
                int(target["end_line"]),
            )
            block = blocks_by_key.get(target_key)
            if not block:
                missing_indices.append(idx)
                predicted_for_same_index = predicted_keys_by_index.get(idx, [])
                if predicted_for_same_index and target_key not in predicted_for_same_index:
                    # Index exists but points to different location than this GT target.
                    header_mismatch_indices.append(idx)
                continue

            content = replace_line_range(
                content,
                int(target["start_line"]),
                int(target["end_line"]),
                str(block.get("body") or ""),
            )
            applied_indices.append(idx)

        file_contents[target_path] = content

    updated_anchor_content = file_contents.get(anchor_file_path)
    if updated_anchor_content is None:
        # Fallback to previous single-anchor behavior when anchor content cannot be reconstructed.
        updated_anchor_content = replace_line_range(
            masked_anchor_content,
            task.start_line,
            task.end_line,
            completion,
        )

    target_count = len(target_indices)
    if target_count == 0:
        apply_status = "no_targets"
    elif len(applied_indices) == target_count:
        apply_status = "applied_all"
    elif applied_indices:
        apply_status = "applied_partial"
    elif valid_blocks:
        apply_status = "parsed_but_not_applied"
    else:
        apply_status = "no_valid_blocks"

    metadata_update: dict[str, Any] = {
        "multi_hunk_mode": True,
        "multi_hunk_apply_status": apply_status,
        "multi_hunk_target_count": target_count,
        "multi_hunk_prediction_block_count": len(parsed_blocks),
        "multi_hunk_valid_block_count": len(valid_blocks),
        "multi_hunk_applied_count": len(applied_indices),
        "multi_hunk_applied_indices": sorted(applied_indices),
        "multi_hunk_missing_indices": sorted(set(missing_indices)),
        "multi_hunk_header_mismatch_indices": sorted(set(header_mismatch_indices)),
        "multi_hunk_extra_predicted_indices": extra_predicted_indices,
        "multi_hunk_duplicate_predicted_indices": sorted(set(duplicate_indices)),
        "multi_hunk_duplicate_predicted_keys": sorted({f"{k[0]}:{k[1]}-{k[2]}" for k in duplicate_keys}),
        "multi_hunk_invalid_target_indices": invalid_target_indices,
        "multi_hunk_unreadable_target_files": sorted(set(unreadable_target_files)),
        "multi_hunk_updated_files": sorted(file_contents.keys()),
    }
    return updated_anchor_content, metadata_update


def _apply_fim_mask(task: FillRequest) -> dict[str, Any]:
    """
    Inject FIM mask into the task file while preserving line count.
    """
    full_path = Path(task.repo_path) / task.file_path
    if not full_path.exists():
        raise ValueError(f"target file not found for mask injection: {full_path}")

    original_content = full_path.read_text(encoding="utf-8")
    lines = original_content.splitlines(keepends=True)
    total_lines = len(lines)
    if total_lines == 0:
        raise ValueError(f"target file is empty, cannot mask: {full_path}")

    if task.start_line < 1 or task.end_line < task.start_line or task.end_line > total_lines:
        raise ValueError(
            f"invalid mask range {task.start_line}-{task.end_line} for file with {total_lines} lines: {full_path}"
        )

    start_idx = task.start_line - 1
    end_idx = task.end_line
    target_lines = lines[start_idx:end_idx]
    target_span_text = "".join(target_lines)

    span_check_policy = str((task.metadata or {}).get("span_check_policy", "warn")).lower()
    span_consistency = "not_checked"
    if task.ground_truth and span_check_policy != "off":
        is_match = _normalize_code_for_check(target_span_text) == _normalize_code_for_check(task.ground_truth)
        span_consistency = "matched" if is_match else "mismatch"
        if not is_match:
            msg = (
                f"task={task.task_id} ground_truth mismatch before mask: "
                f"range={task.start_line}-{task.end_line} file={task.file_path}"
            )
            if span_check_policy == "error":
                raise ValueError(msg)
            if span_check_policy == "warn":
                logger.warning(msg)

    prefix = _comment_prefix_for_file(task.file_path)
    indent_match = re.match(r"^\s*", target_lines[0]) if target_lines else None
    indent = indent_match.group(0) if indent_match else ""
    # marker_line = f"{indent}{prefix} <FILL_IN> lines {task.start_line}-{task.end_line}\n"
    marker_line = "<FILL_IN>"
    replacement_lines = [marker_line] + ["\n"]
    masked_lines = lines[:start_idx] + replacement_lines + lines[end_idx:]
    masked_content = "".join(masked_lines)
    full_path.write_text(masked_content, encoding="utf-8")

    return {
        "file_full_path": str(full_path),
        "original_file_content": original_content,
        "mask_marker": marker_line.rstrip("\n"),
        "mask_range": f"{task.start_line}-{task.end_line}",
        "mask_line_count": len(target_lines),
        "span_consistency": span_consistency,
        "span_check_policy": span_check_policy,
        "target_span_preview": target_span_text[:500],
    }


async def _run_forced_finish_only(
    flow: AgenticLoop,
    last_node,
    task: FillRequest,
    task_id: str,
    max_attempts: int,
) -> bool:
    """
    Run forced round(s) that only allow executing Finish.
    Non-finish actions returned by the model are dropped and never executed.
    """
    current_last_node = last_node
    retry_reason: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        forced_node = flow._create_next_node(current_last_node)
        forced_node.user_message = _build_forced_finish_user_message(
            task=task,
            attempt=attempt,
            retry_reason=retry_reason,
        )

        await flow.agent._generate_actions(forced_node)

        finish_steps = [
            step for step in (forced_node.action_steps or [])
            if isinstance(step.action, FinishArgs)
        ]

        if not finish_steps:
            forced_node.action_steps = []
            forced_node.terminal = True
            forced_node.error = (
                "Forced finish round produced no Finish action. "
                "No non-finish actions were executed."
            )
            retry_reason = "no_finish_action"
            current_last_node = forced_node
            logger.warning(
                "task=%s forced finish attempt=%d/%d produced no Finish; retrying.",
                task_id,
                attempt,
                max_attempts,
            )
            continue

        original_count = len(forced_node.action_steps or [])
        dropped_count = max(0, original_count - 1)
        if dropped_count > 0:
            logger.warning(
                "task=%s forced finish attempt=%d/%d dropped %d actions; executing Finish only.",
                task_id,
                attempt,
                max_attempts,
                dropped_count,
            )

        # Execute only one Finish in forced mode.
        forced_node.action_steps = [finish_steps[0]]
        await flow.agent.run(forced_node)
        return _extract_finish_args(flow) is not None

    logger.warning(
        "task=%s forced finish exhausted %d attempts without valid finish action.",
        task_id,
        max_attempts,
    )
    return False


async def run_fill_task(
    task: FillRequest,
    completion_model: BaseCompletionModel,
    trace_root: str = "moatless/results/fim_traces",
) -> FillResult:
    """Run a single FIM task and return FillResult."""
    temp_repo_dir: Optional[str] = None
    mask_info: Optional[dict[str, Any]] = None
    seed_info: Optional[dict[str, Any]] = None
    restore_original_file = False
    keep_temp_repo = str(os.getenv("FIM_KEEP_TEMP_REPO", "0")).lower() in {"1", "true", "yes"}
    target_commit_hash = _extract_commit_hash(task)
    checkout_ref = _resolve_checkout_ref(task, target_commit_hash)
    run_repo_path = task.repo_path

    if checkout_ref:
        try:
            temp_repo_dir = _prepare_isolated_repo(
                repo_path=task.repo_path,
                commit_hash=checkout_ref,
                task_id=task.task_id,
            )
            run_repo_path = temp_repo_dir
            logger.info(
                "task=%s checked out ref=%s (target_commit=%s) in isolated repo=%s",
                task.task_id,
                checkout_ref,
                target_commit_hash,
                run_repo_path,
            )
        except Exception as exc:
            logger.error("task=%s checkout failed: %s", task.task_id, exc)
            return _build_preflight_error_result(
                task=task,
                trace_root=trace_root,
                finish_reason="repo_checkout_error",
                error=str(exc),
                commit_hash=checkout_ref,
                extra_metadata={"repo_target_commit": target_commit_hash},
            )

    task_for_run = replace(task, repo_path=run_repo_path)
    try:
        if _is_multi_hunk_one_shot_task(task_for_run):
            seed_info = _seed_known_input_hunk(task_for_run)
            if temp_repo_dir is None:
                restore_original_file = True
            logger.info(
                "task=%s known-input hunk seeded on %s (old_start_line=%s old_len=%s new_lines=%s)",
                task.task_id,
                seed_info.get("seed_file_path"),
                seed_info.get("seed_old_start_line"),
                seed_info.get("seed_old_len"),
                seed_info.get("seed_new_line_count"),
            )
        else:
            mask_info = _apply_fim_mask(task_for_run)
            # If not using an isolated temp repo, restore original file after run.
            if temp_repo_dir is None:
                restore_original_file = True
            logger.info(
                "task=%s mask injected on %s (range=%s, lines=%s)",
                task.task_id,
                task_for_run.file_path,
                mask_info["mask_range"],
                mask_info["mask_line_count"],
            )
    except Exception as exc:
        preflight_stage = "known_input_seed_error" if _is_multi_hunk_one_shot_task(task_for_run) else "mask_injection_error"
        logger.error("task=%s pre-agent preparation failed (%s): %s", task.task_id, preflight_stage, exc)
        return _build_preflight_error_result(
            task=task,
            trace_root=trace_root,
            finish_reason=preflight_stage,
            error=str(exc),
            commit_hash=checkout_ref,
            extra_metadata={
                "repo_target_commit": target_commit_hash,
                "resolved_repo_path": run_repo_path,
                "repo_checkout_temp_dir": temp_repo_dir,
            },
        )

    try:
        repo = FileRepository(repo_path=task_for_run.repo_path)
        environment = LocalBashEnvironment(cwd=task_for_run.repo_path)
        workspace = Workspace(repository=repo, environment=environment)
        model = copy.deepcopy(completion_model)
        # AgenticLoop counts nodes (including root) against max_iterations.
        # Reserve one extra loop slot so a normal final Finish turn can happen
        # after exploration, instead of being structurally forced into fallback.
        loop_max_iterations = max(task_for_run.max_iterations + 1, 2)
        baseline_tree = _capture_baseline_tree(task_for_run.repo_path)

        actions = [GrepTool(), ListFiles(), ViewCode(), ReadFile(), Think(), StringReplace(), Finish()]
        agent = ActionAgent(
            completion_model=model,
            system_prompt=build_system_prompt(task_for_run.max_iterations),
            actions=actions,
            shadow_mode=False,
            memory=MessageHistoryGenerator(
                max_tokens=16000,
                max_tokens_per_observation=2000,
            ),
        )
        flow = AgenticLoop.create(
            message=build_user_prompt(task_for_run),
            agent=agent,
            project_id="fim_eval",
            trajectory_id=task_for_run.task_id,
            max_iterations=loop_max_iterations,
        )

        error: Optional[str] = None
        try:
            await flow.run(workspace=workspace)
        except Exception as exc:
            logger.warning("flow.run raised: %s", exc)
            error = str(exc)

        # ── Collect state ──────────────────────────────────────────
        all_nodes = flow.root.get_all_nodes()
        last_node = flow.get_last_node()
        finish_reason = flow.is_finished()

        logger.debug(
            "task=%s nodes=%d last_node=%d finish_reason=%s",
            task.task_id, len(all_nodes), last_node.node_id, finish_reason,
        )

        # ── Fallback: Force Finish at max_iterations ────────────────
        if finish_reason == "max_iterations":
            existing_finish = _extract_finish_args(flow)
            if existing_finish is not None:
                # Finish already exists on the boundary step; avoid duplicate forced finish attempts.
                finish_reason = "terminal"
                logger.info(
                    "task=%s hit max_iterations but Finish already exists; skip forced finish.",
                    task.task_id,
                )
            else:
                logger.info("task=%s hit max_iterations, forcing Finish...", task.task_id)
                try:
                    forced_finish_max_attempts = _get_forced_finish_max_attempts()
                    flow.max_iterations += max(1, forced_finish_max_attempts)
                    try:
                        forced_finished = await _run_forced_finish_only(
                            flow=flow,
                            last_node=last_node,
                            task=task_for_run,
                            task_id=task.task_id,
                            max_attempts=forced_finish_max_attempts,
                        )
                    finally:
                        flow.max_iterations -= max(1, forced_finish_max_attempts)

                    all_nodes = flow.root.get_all_nodes()
                    last_node = flow.get_last_node()
                    finish_reason = "forced_terminal" if forced_finished else "max_iterations"
                    logger.info("task=%s forced finish result: finish_reason=%s", task.task_id, finish_reason)
                except Exception as e:
                    logger.warning("task=%s forced finish failed: %s", task.task_id, e)
                    error = error or str(e)

        # ── Extract result & build result ──────────────────────────
        finish_args = _extract_finish_args(flow)
        total_steps = sum(len(n.action_steps) for n in all_nodes if n.action_steps)
        usage = flow.total_usage()
        result_error = error or getattr(last_node, "error", None)

        finish_node_ids, forced_prompt_node_ids = _collect_finish_and_forced_node_ids(all_nodes)
        first_finish_node_id = min(finish_node_ids) if finish_node_ids else None
        forced_finish_node_candidates = sorted(set(finish_node_ids) & set(forced_prompt_node_ids))
        forced_finish_node_id = forced_finish_node_candidates[0] if forced_finish_node_candidates else None
        finish_mode = _classify_finish_mode(
            has_finish=finish_args is not None,
            has_error=result_error is not None,
            first_finish_node_id=first_finish_node_id,
            forced_finish_node_id=forced_finish_node_id,
            task_max_iterations=task_for_run.max_iterations,
        )
        error_type = _classify_error_type(result_error)

        full_path = Path(task_for_run.repo_path) / task_for_run.file_path
        with open(full_path, encoding="utf-8") as f:
            updated = f.read()

        completion = _extract_line_range_content(updated, task_for_run.start_line, task_for_run.end_line)
        git_patch, changed_files = _extract_patch_against_baseline(task_for_run.repo_path, baseline_tree)
        updated_files_content_before, updated_files_content = _collect_changed_file_snapshots(
            repo_path=task_for_run.repo_path,
            changed_files=changed_files,
            baseline_tree=baseline_tree,
        )

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        trace_dir = Path(trace_root) / f"{task_for_run.task_id}_{timestamp}"
        mask_status = "applied" if mask_info else ("skipped_known_input_seeded" if seed_info else "not_applied")
        result_metadata = _build_result_metadata(
            base_metadata=task.metadata if isinstance(task.metadata, dict) else None,
            repo_checkout_mode="temp_clone_checkout" if checkout_ref else "direct_repo_path",
            repo_checkout_commit=checkout_ref,
            repo_checkout_status="ok",
            repo_target_commit=target_commit_hash,
            resolved_repo_path=run_repo_path,
            repo_checkout_temp_dir=temp_repo_dir,
            preflight_status="passed",
            mask_status=mask_status,
            total_prompt_tokens=usage.prompt_tokens,
            total_completion_tokens=usage.completion_tokens,
            total_reasoning_tokens=usage.reasoning_tokens,
            total_cache_read_tokens=usage.cache_read_tokens,
            total_cost_usd=usage.completion_cost,
            task_max_iterations=task_for_run.max_iterations,
            flow_max_iterations=loop_max_iterations,
            submission_mode=finish_mode,
            finish_mode=finish_mode,
            first_submit_node_id=first_finish_node_id,
            forced_submit_node_id=forced_finish_node_id,
            first_finish_node_id=first_finish_node_id,
            forced_finish_node_id=forced_finish_node_id,
            error_type=error_type,
            git_patch_size=len(git_patch),
            changed_files_count=len(changed_files),
            changed_files=changed_files,
        )
        if mask_info:
            result_metadata.update(
                {
                    "mask_range": mask_info.get("mask_range"),
                    "mask_marker": mask_info.get("mask_marker"),
                    "mask_line_count": mask_info.get("mask_line_count"),
                    "span_consistency": mask_info.get("span_consistency"),
                    "span_check_policy": mask_info.get("span_check_policy"),
                }
            )
        if seed_info:
            result_metadata.update(
                {
                    "known_input_seed_status": seed_info.get("seed_status"),
                    "known_input_seed_file_path": seed_info.get("seed_file_path"),
                    "known_input_seed_old_start_line": seed_info.get("seed_old_start_line"),
                    "known_input_seed_old_len": seed_info.get("seed_old_len"),
                    "known_input_seed_new_line_count": seed_info.get("seed_new_line_count"),
                }
            )
        if _is_multi_hunk_one_shot_task(task_for_run):
            result_metadata.update(
                {
                    "multi_hunk_mode": True,
                    "multi_hunk_target_count": len(_extract_target_hunks(task_for_run)),
                    "multi_hunk_changed_file_count": len(changed_files),
                    "multi_hunk_changed_files": changed_files,
                }
            )

        success = finish_args is not None and result_error is None
        result = FillResult(
            task_id=task_for_run.task_id,
            model_name=task_for_run.model_name,
            file_path=task_for_run.file_path,
            start_line=task_for_run.start_line,
            end_line=task_for_run.end_line,
            completion=completion,
            updated_file_content=updated,
            git_patch=git_patch,
            changed_files=changed_files,
            updated_files_content=updated_files_content,
            updated_files_content_before=updated_files_content_before,
            trace_dir=str(trace_dir),
            success=success,
            finish_reason=finish_reason or "unknown",
            error=result_error,
            confidence=1.0 if finish_args else 0.0,
            reasoning=finish_args.finish_reason if finish_args else "",
            action_steps=total_steps,
            metadata=result_metadata,
        )

        persist_trace(flow, trace_dir, result)
        return result
    finally:
        if restore_original_file:
            restore_payload = mask_info or seed_info
            if restore_payload:
                file_full_path = restore_payload.get("file_full_path")
                original_file_content = restore_payload.get("original_file_content")
                if file_full_path and original_file_content is not None:
                    try:
                        Path(file_full_path).write_text(original_file_content, encoding="utf-8")
                    except Exception as exc:
                        logger.warning(
                            "task=%s failed to restore original file after pre-agent preparation: %s",
                            task.task_id,
                            exc,
                        )

        if temp_repo_dir and not keep_temp_repo:
            shutil.rmtree(temp_repo_dir, ignore_errors=True)


def _has_existing_trace(trace_root: Path, task_id: str) -> bool:
    """Check if task_id already has a trace directory (regardless of success/failure).

    Matches directories named ``{task_id}_*`` (timestamp suffix).
    """
    if not trace_root.is_dir():
        return False

    for d in trace_root.iterdir():
        if d.is_dir() and d.name.startswith(f"{task_id}_"):
            return True
    return False


async def run_fill_batch(
    tasks: list[FillRequest],
    completion_model: BaseCompletionModel,
    trace_root: str = "moatless/results/fim_traces",
    max_concurrency: int = 4,
    resume: bool = False,
) -> list[FillResult]:
    """Run multiple FIM tasks concurrently. If resume=True, skip tasks with existing traces."""
    trace_root_path = Path(trace_root)

    # ── Filter out tasks with existing traces ──
    pending_tasks: list[FillRequest] = []
    skipped = 0

    for t in tasks:
        if resume and _has_existing_trace(trace_root_path, t.task_id):
            skipped += 1
        else:
            pending_tasks.append(t)

    if skipped:
        logger.info("Resume: skipping %d task(s) with existing traces", skipped)

    # ── Run only unfinished tasks ──
    sem = asyncio.Semaphore(max_concurrency)

    async def _run_one(t: FillRequest) -> FillResult:
        async with sem:
            return await run_fill_task(t, completion_model, trace_root=trace_root)

    if not pending_tasks:
        logger.info("All tasks have existing traces, nothing to run.")
        return []

    return await asyncio.gather(*[_run_one(t) for t in pending_tasks])
