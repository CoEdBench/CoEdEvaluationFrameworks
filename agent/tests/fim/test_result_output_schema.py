import json
from pathlib import Path

from moatless.fim.dataset import save_results_to_jsonl
from moatless.fim.schema import FillResult, RESULT_SCHEMA_VERSION
from moatless.fim.utils import persist_trace


class _DummyFlow:
    def get_flow_settings(self):
        return {"kind": "dummy"}

    def get_trajectory_data(self):
        return {"nodes": []}


def _build_result() -> FillResult:
    return FillResult(
        task_id="task_001",
        model_name="mock-model",
        file_path="pkg/main.py",
        trace_dir="moatless/results/fim_traces/task_001_20260407_000000",
        start_line=10,
        end_line=11,
        completion="return 1",
        updated_file_content="def f():\n    return 1\n",
        success=True,
        finish_reason="terminal",
        error=None,
        metadata={"commit_hash": "abc123"},
    )


def test_fill_result_to_output_dict_has_stable_schema():
    result = _build_result()
    row = result.to_output_dict()

    assert row["schema_version"] == RESULT_SCHEMA_VERSION
    assert row["task_id"] == "task_001"
    assert row["completion"] == "return 1"
    assert row["finish_reason"] == "terminal"
    assert row["error"] is None
    assert row["trace_dir"].endswith("task_001_20260407_000000")
    assert row["metadata"]["commit_hash"] == "abc123"

    # Reserved placeholder fields from P2 schema stabilization.
    assert row["eval_codebleu"] is None
    assert row["eval_partial_match"] is None
    assert row["eval_syntax_pass"] is None
    assert row["task_runtime_seconds"] is None
    assert row["tool_call_distribution"] is None


def test_save_results_to_jsonl_uses_stable_schema(tmp_path: Path):
    output_path = tmp_path / "fim_results.jsonl"
    save_results_to_jsonl([_build_result()], str(output_path))

    lines = output_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])

    assert row["schema_version"] == RESULT_SCHEMA_VERSION
    assert row["task_id"] == "task_001"
    assert "metadata" in row
    assert "finish_reason" in row
    assert "trace_dir" in row


def test_persist_trace_writes_result_with_schema_version(tmp_path: Path):
    trace_dir = tmp_path / "trace"
    persist_trace(_DummyFlow(), trace_dir, _build_result())

    result_json = json.loads((trace_dir / "result.json").read_text(encoding="utf-8"))
    assert result_json["schema_version"] == RESULT_SCHEMA_VERSION
    assert result_json["task_id"] == "task_001"
