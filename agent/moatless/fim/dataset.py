"""
fim/dataset.py
JSONL dataset read/write + Docker interaction interface (reserved for extension)

Example JSONL format (one JSON per line):
{
  "task_id": "fim_001",
  "model_name": "Qwen/Qwen3.5-35B-A3B",
  "repo_path": "/workspace/scikit-learn",   // Docker path
  "file_path": "sklearn/neighbors/_nca.py",
  "start_line": 367,
  "end_line": 369,
  "language": "python",
  "ground_truth": "        check_is_fitted(self)\n        X = self._validate_data(...)\n",
  "metadata": {}
}
"""
import json
import logging
import subprocess
from pathlib import Path
from typing import Any, Mapping, Optional

from moatless.fim.schema import FillRequest, FillResult

logger = logging.getLogger(__name__)
MULTI_HUNK_BLOCK_START = "<<<TARGET_HUNK>>>"
MULTI_HUNK_BLOCK_END = "<<<END_TARGET_HUNK>>>"


# ── JSONL Reading ────────────────────────────────────────────────────────────
REQUIRED_KEYS = ("file_path", "start_line", "end_line")


def _ensure_dict(data: Any, line_no: int) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError(f"line {line_no}: expected JSON object, got {type(data).__name__}")
    return data


def _validate_required_keys(data: dict[str, Any], line_no: int, repo_path_override: Optional[str]) -> None:
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if repo_path_override is None and "repo_path" not in data:
        missing.append("repo_path")
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(f"line {line_no}: missing required key(s): {missing_keys}")


def _parse_int_field(
    data: dict[str, Any], key: str, line_no: int, default: Optional[int] = None, min_value: Optional[int] = None
) -> int:
    if key not in data:
        if default is None:
            raise ValueError(f"line {line_no}: missing required key: {key}")
        value = default
    else:
        raw = data[key]
        try:
            value = int(raw)
        except (TypeError, ValueError):
            raise ValueError(f"line {line_no}: key '{key}' expects int, got {raw!r}")

    if min_value is not None and value < min_value:
        raise ValueError(f"line {line_no}: key '{key}' must be >= {min_value}, got {value}")
    return value


def _parse_model_name(data: dict[str, Any], default_model_name: Optional[str], line_no: int) -> str:
    model_name = data.get("model_name") or default_model_name
    if not model_name:
        raise ValueError(
            f"line {line_no}: missing key 'model_name' and no default_model_name provided."
        )
    return str(model_name)


def _parse_repo_path(data: dict[str, Any], repo_path_override: Optional[str], line_no: int) -> str:
    repo_path = repo_path_override or data.get("repo_path")
    if not repo_path:
        raise ValueError(f"line {line_no}: resolved empty repo_path")
    return str(repo_path)


def _parse_metadata(data: dict[str, Any], line_no: int) -> dict[str, Any]:
    metadata = data.get("metadata", {})
    if metadata is None:
        return {}
    if not isinstance(metadata, dict):
        raise ValueError(f"line {line_no}: key 'metadata' expects object, got {type(metadata).__name__}")
    return metadata


def _normalize_file_path(path_value: Any, line_no: int, field_name: str = "file_path") -> str:
    file_path = str(path_value or "").strip()
    if not file_path:
        raise ValueError(f"line {line_no}: key '{field_name}' is empty")
    # Phase3 data paths may use Windows separators; normalize to POSIX style.
    return file_path.replace("\\", "/")


def _extract_added_lines_from_hunk_content(content: str) -> list[str]:
    added_lines: list[str] = []
    for line in content.splitlines(keepends=True):
        # Ignore hunk/file header lines.
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added_lines.append(line[1:])
    return added_lines


def _build_completion_from_added_lines(added_lines: list[str]) -> str:
    if not added_lines:
        return ""
    text = "".join(added_lines)
    if not text.endswith("\n"):
        text += "\n"
    return text


def _extract_requirement_summary_from_phase3(data: Mapping[str, Any]) -> str:
    direct = str(data.get("requirement_summary") or "").strip()
    if direct:
        return direct

    causal_analysis = data.get("causal_analysis")
    if isinstance(causal_analysis, Mapping):
        nested = str(causal_analysis.get("requirement_summary") or "").strip()
        if nested:
            return nested

    issue = str(data.get("issue_description") or "").strip()
    if issue:
        return issue
    return str(data.get("msg") or "").strip()


def _normalize_hunk_for_metadata(hunk: Mapping[str, Any], hunk_index: int) -> dict[str, Any]:
    content = hunk.get("content")
    if not isinstance(content, str):
        content = "" if content is None else str(content)
    file_path = str(hunk.get("file_path") or "").strip().replace("\\", "/")
    return {
        "hunk_index": hunk_index,
        "id": hunk.get("id"),
        "order_index": hunk.get("order_index"),
        "file_path": file_path,
        "start_line": hunk.get("start_line"),
        "end_line": hunk.get("end_line"),
        "content": content,
        "old_start_line": hunk.get("old_start_line"),
        "old_len": hunk.get("old_len"),
        "new_start_line": hunk.get("new_start_line"),
        "new_len": hunk.get("new_len"),
    }


def _build_multi_hunk_targets_from_phase3(
    ordered_hunks: list[Any],
    line_no: int,
) -> list[dict[str, Any]]:
    if len(ordered_hunks) <= 1:
        raise ValueError(f"line {line_no}: ordered_hunks has no prediction target (len<=1)")

    targets: list[dict[str, Any]] = []
    for target_index in range(1, len(ordered_hunks)):
        hunk = _ensure_dict(ordered_hunks[target_index], line_no)
        start_line = _parse_int_field(
            hunk,
            "start_line",
            line_no,
            min_value=1,
        )
        end_line = _parse_int_field(
            hunk,
            "end_line",
            line_no,
            min_value=1,
        )
        if start_line > end_line:
            raise ValueError(
                f"line {line_no}: ordered_hunks[{target_index}] start_line({start_line}) > end_line({end_line})"
            )

        file_path = _normalize_file_path(
            hunk.get("file_path"),
            line_no,
            field_name=f"ordered_hunks[{target_index}].file_path",
        )
        content = hunk.get("content") or ""
        if not isinstance(content, str):
            raise ValueError(
                f"line {line_no}: key 'ordered_hunks[{target_index}].content' expects string, "
                f"got {type(content).__name__}"
            )

        completion = _build_completion_from_added_lines(_extract_added_lines_from_hunk_content(content))
        targets.append(
            {
                "target_index": target_index,
                "id": hunk.get("id"),
                "order_index": hunk.get("order_index"),
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


def _normalize_repo_map(repo_map: Optional[Mapping[str, str]]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    if not repo_map:
        return normalized
    for raw_repo_name, raw_repo_path in repo_map.items():
        repo_name = str(raw_repo_name or "").strip()
        repo_path = str(raw_repo_path or "").strip()
        if repo_name and repo_path:
            normalized[repo_name] = repo_path
    return normalized


def _build_fill_request(
    data: dict[str, Any],
    line_no: int,
    default_model_name: Optional[str],
    repo_path_override: Optional[str],
) -> FillRequest:
    _validate_required_keys(data, line_no, repo_path_override)

    start_line = _parse_int_field(data, "start_line", line_no, min_value=1)
    end_line = _parse_int_field(data, "end_line", line_no, min_value=1)
    if start_line > end_line:
        raise ValueError(f"line {line_no}: start_line({start_line}) > end_line({end_line})")

    context_lines = _parse_int_field(data, "context_lines", line_no, default=30, min_value=0)
    max_iterations = _parse_int_field(data, "max_iterations", line_no, default=30, min_value=1)

    file_path = str(data["file_path"])
    if not file_path:
        raise ValueError(f"line {line_no}: key 'file_path' is empty")

    return FillRequest(
        task_id=str(data.get("task_id", f"task_{line_no:04d}")),
        model_name=_parse_model_name(data, default_model_name, line_no),
        repo_path=_parse_repo_path(data, repo_path_override, line_no),
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=str(data.get("language", "python")),
        context_lines=context_lines,
        max_iterations=max_iterations,
        ground_truth=data.get("ground_truth"),
        metadata=_parse_metadata(data, line_no),
    )


def _resolve_repo_path_for_phase3(
    data: dict[str, Any],
    line_no: int,
    repo_path_override: Optional[str],
    repo_map: dict[str, str],
    default_repo_path: Optional[str],
) -> str:
    if repo_path_override:
        return str(repo_path_override)

    explicit_repo_path = str(data.get("repo_path") or "").strip()
    if explicit_repo_path:
        return explicit_repo_path

    repo_name = str(data.get("repo") or "").strip()
    if repo_name and repo_name in repo_map:
        return repo_map[repo_name]

    if default_repo_path:
        return str(default_repo_path)

    raise ValueError(
        f"line {line_no}: cannot resolve repo_path for phase3 row "
        f"(repo={repo_name!r}). Provide --repo-path-override, --repo-map, or --default-repo-path."
    )


def _build_fill_request_from_phase3_ordered_hunks(
    data: dict[str, Any],
    line_no: int,
    default_model_name: Optional[str],
    repo_path_override: Optional[str],
    repo_map: dict[str, str],
    default_repo_path: Optional[str],
) -> FillRequest:
    ordered_hunks = data.get("ordered_hunks")
    if not isinstance(ordered_hunks, list) or not ordered_hunks:
        raise ValueError(f"line {line_no}: key 'ordered_hunks' expects non-empty list")

    first_hunk = _ensure_dict(ordered_hunks[0], line_no)
    start_line = _parse_int_field(first_hunk, "start_line", line_no, min_value=1)
    end_line = _parse_int_field(first_hunk, "end_line", line_no, min_value=1)
    if start_line > end_line:
        raise ValueError(f"line {line_no}: ordered_hunks[0] start_line({start_line}) > end_line({end_line})")

    file_path = _normalize_file_path(
        first_hunk.get("file_path"),
        line_no,
        field_name="ordered_hunks[0].file_path",
    )
    hunk_content = first_hunk.get("content")
    if hunk_content is None:
        hunk_content = ""
    if not isinstance(hunk_content, str):
        raise ValueError(
            f"line {line_no}: key 'ordered_hunks[0].content' expects string, got {type(hunk_content).__name__}"
        )
    known_input_hunk_added_lines = _extract_added_lines_from_hunk_content(hunk_content)
    known_input_hunk_added_text = _build_completion_from_added_lines(known_input_hunk_added_lines)
    target_hunks = _build_multi_hunk_targets_from_phase3(ordered_hunks, line_no)
    multi_hunk_ground_truth = _build_multi_hunk_ground_truth(target_hunks)

    row_repo_name = str(data.get("repo") or "").strip()
    row_hash = str(data.get("hash") or "").strip()
    row_task_id = str(data.get("task_id") or "").strip()
    if not row_task_id:
        task_prefix = row_repo_name or "phase3"
        hash_part = row_hash[:12] if row_hash else "nohash"
        row_task_id = f"{task_prefix}_{hash_part}_{line_no:05d}"

    issue_description = str(data.get("issue_description") or "").strip()
    requirement_summary = _extract_requirement_summary_from_phase3(data)

    metadata = _parse_metadata(data, line_no)
    metadata.update(
        {
            "input_format": "phase3_ordered_hunks",
            "task_mode": "phase3_multi_hunk_one_shot",
            "output_format": "multi_hunk_blocks_v1",
            "repo": row_repo_name,
            # Keep hash key for commit checkout in pipeline._extract_commit_hash().
            "hash": row_hash,
            "msg": str(data.get("msg") or "").strip(),
            # Phase3 only uses requirement summary as prompt context.
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
            "ground_truth_source": "ordered_hunks[1:].content_added_lines",
            # Keep all multi-point labels for downstream evaluation.
            "ground_truth_all_ordered_hunks": ordered_hunks,
            "ground_truth_remaining_ordered_hunks": ordered_hunks[1:],
            "ground_truth_test_hunks": data.get("test_hunks"),
            "ground_truth_source_diff": data.get("source_diff"),
            # Keep checkout metadata but suppress commit block in prompt.
            "hide_commit_context": True,
            # This ground-truth sequence does not match start/end span by design.
            "span_check_policy": "off",
        }
    )

    return FillRequest(
        task_id=row_task_id,
        model_name=_parse_model_name(data, default_model_name, line_no),
        repo_path=_resolve_repo_path_for_phase3(
            data=data,
            line_no=line_no,
            repo_path_override=repo_path_override,
            repo_map=repo_map,
            default_repo_path=default_repo_path,
        ),
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=str(data.get("language", "python")),
        context_lines=_parse_int_field(data, "context_lines", line_no, default=30, min_value=0),
        max_iterations=_parse_int_field(data, "max_iterations", line_no, default=30, min_value=1),
        ground_truth=multi_hunk_ground_truth,
        metadata=metadata,
    )


def load_requests_from_jsonl(
    jsonl_path: str,
    repo_path_override: Optional[str] = None,
    max_items: Optional[int] = None,
    default_model_name: Optional[str] = "placeholder-model",
    strict: bool = True,
    input_format: str = "fill_request",
    repo_map: Optional[Mapping[str, str]] = None,
    default_repo_path: Optional[str] = None,
) -> list[FillRequest]:
    """
    Load FillRequest list from a JSONL file.

    Args:
        jsonl_path:          Path to the JSONL file
        repo_path_override:  If set, overrides repo_path for all tasks (for local debugging)
        max_items:           Maximum number of items to load (for debugging, counted by successful loads)
        default_model_name:  Default model name used when a record lacks model_name
        strict:              True=raise on format errors immediately; False=skip bad lines
        input_format:        Input format: fill_request / phase3_ordered_hunks
        repo_map:            Mapping from repo name to local path (used by phase3)
        default_repo_path:   Default repo path for phase3 when repo_map does not match
    """
    normalized_input_format = str(input_format or "fill_request").strip().lower()
    if normalized_input_format not in {"fill_request", "phase3_ordered_hunks"}:
        raise ValueError(
            f"Unsupported input_format={input_format!r}. "
            "Expected one of: fill_request, phase3_ordered_hunks"
        )

    normalized_repo_map = _normalize_repo_map(repo_map)
    requests = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if max_items is not None and len(requests) >= max_items:
                break
            line = raw.strip()
            if not line:
                continue
            try:
                data = _ensure_dict(json.loads(line), line_no)
                if normalized_input_format == "fill_request":
                    request = _build_fill_request(
                        data=data,
                        line_no=line_no,
                        default_model_name=default_model_name,
                        repo_path_override=repo_path_override,
                    )
                else:
                    request = _build_fill_request_from_phase3_ordered_hunks(
                        data=data,
                        line_no=line_no,
                        default_model_name=default_model_name,
                        repo_path_override=repo_path_override,
                        repo_map=normalized_repo_map,
                        default_repo_path=default_repo_path,
                    )
                requests.append(request)
            except json.JSONDecodeError as e:
                msg = f"line {line_no}: invalid JSON ({e})"
                if strict:
                    raise ValueError(msg) from e
                logger.warning("Skip %s", msg)
            except ValueError as e:
                if strict:
                    raise
                logger.warning("Skip %s", e)
    logger.info("Loaded %d tasks from %s", len(requests), jsonl_path)
    return requests


# ── JSONL Writing ────────────────────────────────────────────────────────────

def save_results_to_jsonl(results: list[FillResult], output_path: str) -> None:
    """Write a list of FillResult objects to a JSONL file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            payload = r.to_output_dict() if hasattr(r, "to_output_dict") else r.__dict__
            f.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    logger.info("Saved %d results to %s", len(results), output_path)


# ── Docker Interaction (Reserved)────────────────────────────────────────────────────

class DockerSandbox:
    """
    Docker sandbox interface for:
      1. Writing modified files into a container
      2. Running tests inside the container, returning pass/fail
    Methods can be filled in during later implementation; the interface is stable.
    """

    def __init__(self, container_name: str, workspace_dir: str = "/workspace"):
        self.container_name = container_name
        self.workspace_dir = workspace_dir

    def write_file(self, file_path: str, content: str) -> None:
        """Write content to file_path inside the container."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py",
                                         delete=False, encoding="utf-8") as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            dest = f"{self.container_name}:{self.workspace_dir}/{file_path}"
            subprocess.run(["docker", "cp", tmp_path, dest], check=True)
            logger.debug("Copied %s -> %s", tmp_path, dest)
        finally:
            os.unlink(tmp_path)

    def run_tests(
        self,
        test_cmd: str = "pytest --tb=no -q",
        timeout: int = 60,
    ) -> tuple[bool, str]:
        """
        Run test commands inside the container.
        Returns:
            (passed: bool, output: str)
        """
        result = subprocess.run(
            ["docker", "exec", self.container_name, "bash", "-c", test_cmd],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        passed = result.returncode == 0
        output = result.stdout + result.stderr
        logger.info("Docker test: passed=%s cmd=%s", passed, test_cmd)
        return passed, output

    def run_fill_and_test(
        self,
        fill_result: FillResult,
        test_cmd: str = "pytest --tb=no -q",
        timeout: int = 60,
    ) -> tuple[bool, str]:
        """
        Write fill_result changes into the container, run tests, and write results back.
        """
        self.write_file(fill_result.file_path, fill_result.updated_file_content)
        passed, output = self.run_tests(test_cmd=test_cmd, timeout=timeout)
        fill_result.eval_execution_pass = passed
        return passed, output
